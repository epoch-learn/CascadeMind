"""
SemEval 2026 Task 4 Subtask 1 - GPT-5-mini Only
Test GPT-5-mini (Azure) on all cases directly.
"""

import os
import pandas as pd
import json
import asyncio
import time
import httpx
from tqdm.asyncio import tqdm

# Configuration
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT = "gpt-5-mini"
AZURE_API_VERSION = "2024-06-01"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/gpt5_only_results.csv"
CONCURRENCY_LIMIT = 5  # Very conservative for Azure rate limits

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

Respond with JSON: {{"decision": "A" or "B"}}"""


async def predict_single(row: dict, http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> dict:
    """Make a single prediction with GPT-5-mini."""
    prompt = build_prompt(row)
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    
    last_error = None
    async with semaphore:
        for attempt in range(3):
            try:
                response = await http_client.post(
                    url,
                    headers={
                        "api-key": AZURE_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 1.0,  # GPT-5-mini only supports temp=1.0
                        "response_format": {"type": "json_object"}
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                data = json.loads(content)
                decision = data.get("decision")
                if decision in ["A", "B"]:
                    return {"decision": decision, "error": None}
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        
        return {"decision": "ERROR", "error": last_error}


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 - GPT-5-mini Only")
    print(f"Model: {AZURE_DEPLOYMENT} (Azure)")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples")
    
    # Run predictions
    print(f"\n🧠 Running predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with httpx.AsyncClient() as http_client:
        tasks = [predict_single(row, http_client, semaphore) for _, row in df.iterrows()]
        results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    predictions = [r["decision"] for r in results]
    errors = [r["error"] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["error"] = errors
    
    # Calculate accuracy (excluding errors)
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) > 0 else 0
    total_errors = len(df) - len(valid_df)
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS - GPT-5-mini Only")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'═'*60}")
    
    # Vote distribution
    a_count = predictions.count('A')
    b_count = predictions.count('B')
    print(f"\n📊 Decision distribution: A={a_count} ({a_count/len(df):.1%}) B={b_count} ({b_count/len(df):.1%})")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

