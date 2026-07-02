"""
SemEval 2026 Task 4 Subtask 1 - v7
Self-consistency + Few-shot examples for maximum accuracy.

Uses Gemini 2.5 Flash with:
- 5 samples per example (temperature=0.7)
- Majority voting
- 3 few-shot examples
"""

import os
import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from tqdm.asyncio import tqdm
from collections import Counter

# Configuration
API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v7_results.csv"
CONCURRENCY_LIMIT = 100
NUM_SAMPLES = 5  # Self-consistency samples
TEMPERATURE = 0.7

client = genai.Client(api_key=API_KEY)

# Few-shot examples (hand-picked diverse examples)
FEW_SHOT_EXAMPLES = """
Example 1:
Anchor: A young orphan discovers they have magical powers and is sent to a special school where they learn to control their abilities, make friends, and ultimately face a dark villain threatening their world.
Story A: A teenager inherits a fortune and attends an elite boarding school where they navigate social hierarchies and romance.
Story B: A child from humble origins learns they are destined for greatness, trains under mentors, bonds with loyal companions, and must confront an evil force that killed their parents.
Answer: B (same hero's journey arc: discovery of destiny → training → friendship → confronting evil connected to past)

Example 2:
Anchor: Two people from feuding families fall in love, marry in secret, but miscommunication leads to both their deaths, which finally reconciles their families.
Story A: A woman poisons her husband after discovering his affair, then remarries his business partner.
Story B: Lovers from rival gangs try to escape together but are caught in crossfire; their deaths end the gang war.
Answer: B (same arc: forbidden love → secret union → tragic death → reconciliation of enemies)

Example 3:
Anchor: A detective investigates a series of murders and discovers the killer is someone close to them, forcing a moral choice between justice and loyalty.
Story A: A journalist uncovers corporate fraud and must decide whether to publish a story that will destroy their mentor's career.
Story B: A cop tracks a serial killer through clues at crime scenes, eventually catching them after a car chase.
Answer: A (same arc: investigation → discovery that it's someone close → moral dilemma between duty and personal relationship)

"""

# Simple schema - decision only for faster inference
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(
            type=types.Type.STRING, 
            enum=["A", "B"]
        )
    },
    required=["decision"]
)


async def predict_single(prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Make a single prediction."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=TEMPERATURE
                    )
                )
                data = json.loads(response.text)
                return data.get("decision", "A")
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return "A"  # Default fallback


async def predict_with_self_consistency(row: dict, semaphore: asyncio.Semaphore) -> tuple[str, list[str]]:
    """Run multiple samples and return majority vote + all votes."""
    prompt = f"""{FEW_SHOT_EXAMPLES}
Now determine which story is more narratively similar to the Anchor.

Focus on NARRATIVE SIMILARITY:
1. Abstract Theme - core problems, central ideas, motifs (NOT concrete setting)
2. Course of Action - sequence of events, turning points, how conflicts develop
3. Outcomes - how the story ends, character fates, lessons

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Answer (A or B):"""

    # Run NUM_SAMPLES predictions in parallel
    tasks = [predict_single(prompt, semaphore) for _ in range(NUM_SAMPLES)]
    votes = await asyncio.gather(*tasks)
    
    # Majority vote
    vote_counts = Counter(votes)
    majority = vote_counts.most_common(1)[0][0]
    
    return majority, list(votes)


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v7")
    print(f"Model: {MODEL}")
    print(f"Strategy: Self-consistency (k={NUM_SAMPLES}) + Few-shot")
    print(f"Temperature: {TEMPERATURE}")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    print(f"   Total API calls: {len(df) * NUM_SAMPLES}")
    
    # Run predictions
    print(f"\n🧠 Running predictions with self-consistency...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [predict_with_self_consistency(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Unpack results
    predictions = [r[0] for r in results]
    all_votes = [r[1] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["votes"] = [",".join(v) for v in all_votes]
    df["vote_agreement"] = [max(Counter(v).values()) / len(v) for v in all_votes]
    
    # Calculate accuracy
    correct = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).sum()
    accuracy = correct / len(df)
    
    # Analyze by vote agreement
    unanimous = df[df["vote_agreement"] == 1.0]
    split = df[df["vote_agreement"] < 1.0]
    
    unanimous_correct = (unanimous["predicted_text_a_is_closer"] == unanimous["text_a_is_closer"]).sum() if len(unanimous) > 0 else 0
    split_correct = (split["predicted_text_a_is_closer"] == split["text_a_is_closer"]).sum() if len(split) > 0 else 0
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(df)})")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*60}")
    print(f"Unanimous votes ({len(unanimous)}): {unanimous_correct}/{len(unanimous)} = {unanimous_correct/len(unanimous):.1%}" if len(unanimous) > 0 else "No unanimous votes")
    print(f"Split votes ({len(split)}): {split_correct}/{len(split)} = {split_correct/len(split):.1%}" if len(split) > 0 else "No split votes")
    print(f"{'═'*60}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

