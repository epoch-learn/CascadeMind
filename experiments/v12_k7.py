"""
SemEval 2026 Task 4 Subtask 1 - v12 k=7 variant
Simple self-consistency with k=7 samples.

Key insight from v12 analysis:
- Bidirectional didn't improve accuracy on agreement cases
- The gain came from using combined/forward majority on conflicts
- Higher k should reduce split rate and improve accuracy

Strategy:
1. Run k=7 samples with Gemini 2.5 Flash
2. Take majority vote
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
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

MODEL_FAST = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v12_k7_results.csv"
CONCURRENCY_LIMIT = 20
NUM_SAMPLES = 7  # Increased from 5
TEMPERATURE = 0.7

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Few-shot examples
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

# Gemini response schema
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])
    },
    required=["decision"]
)


def build_prompt(row: dict) -> str:
    """Build the comparison prompt."""
    return f"""{FEW_SHOT_EXAMPLES}
Now determine which story is more narratively similar to the Anchor.

Focus on NARRATIVE SIMILARITY:
1. Abstract Theme - core problems, central ideas, motifs (NOT concrete setting)
2. Course of Action - sequence of events, turning points, how conflicts develop
3. Outcomes - how the story ends, character fates, lessons

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Answer (A or B):"""


async def predict_single_gemini(prompt: str, semaphore: asyncio.Semaphore) -> str | None:
    """Make a single prediction with Gemini API. Returns None on error."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=MODEL_FAST,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=TEMPERATURE
                    )
                )
                data = json.loads(response.text)
                decision = data.get("decision")
                if decision in ["A", "B"]:
                    return decision
                return None
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return None


async def predict_majority(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """Simple k=7 majority vote prediction."""
    prompt = build_prompt(row)
    
    # Run k samples
    tasks = [predict_single_gemini(prompt, semaphore) for _ in range(NUM_SAMPLES)]
    results = await asyncio.gather(*tasks)
    
    # Filter out errors
    valid_votes = [r for r in results if r is not None]
    errors = sum(1 for r in results if r is None)
    
    vote_counts = Counter(valid_votes)
    
    if valid_votes:
        decision = vote_counts.most_common(1)[0][0]
        margin = abs(vote_counts.get('A', 0) - vote_counts.get('B', 0))
        unanimous = len(vote_counts) == 1
    else:
        decision = "ERROR"
        margin = 0
        unanimous = False
    
    return {
        "decision": decision,
        "votes": valid_votes,
        "errors": errors,
        "margin": margin,
        "unanimous": unanimous
    }


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v12 (k=7)")
    print(f"Model: {MODEL_FAST} (native Gemini API)")
    print(f"Strategy: Simple majority vote with k={NUM_SAMPLES}")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    tasks = [predict_majority(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    predictions = [r["decision"] for r in results]
    all_votes = [r["votes"] for r in results]
    margins = [r["margin"] for r in results]
    unanimous = [r["unanimous"] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["votes"] = [",".join(v) if v else "none" for v in all_votes]
    df["margin"] = margins
    df["unanimous"] = unanimous
    
    # Calculate accuracy
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) > 0 else 0
    total_errors = len(df) - len(valid_df)
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS - v12 (k={NUM_SAMPLES})")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'═'*60}")
    
    # Analyze by margin
    for margin_thresh in [7, 5, 3, 1]:
        margin_df = valid_df[valid_df["margin"] >= margin_thresh]
        if len(margin_df) > 0:
            margin_correct = (margin_df["predicted_text_a_is_closer"] == margin_df["text_a_is_closer"]).sum()
            print(f"Margin >= {margin_thresh}: {len(margin_df)} cases → {margin_correct}/{len(margin_df)} = {margin_correct/len(margin_df):.1%}")
    
    print(f"{'═'*60}")
    
    # Vote distribution
    all_valid_votes = [v for votes in all_votes for v in votes]
    if all_valid_votes:
        vote_a = all_valid_votes.count('A')
        vote_b = all_valid_votes.count('B')
        print(f"\n📊 Vote distribution: A={vote_a} ({vote_a/len(all_valid_votes):.1%}) B={vote_b} ({vote_b/len(all_valid_votes):.1%})")
    
    # Unanimous stats
    unan_df = valid_df[valid_df["unanimous"]]
    if len(unan_df) > 0:
        unan_correct = (unan_df["predicted_text_a_is_closer"] == unan_df["text_a_is_closer"]).sum()
        print(f"📊 Unanimous (7-0): {len(unan_df)} cases → {unan_correct}/{len(unan_df)} = {unan_correct/len(unan_df):.1%}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

