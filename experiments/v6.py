"""
SemEval 2026 Task 4 Subtask 1 - v6
Original 76% baseline restored: Gemini 2.5 Flash with simple prompt.
"""

import os
import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from tqdm.asyncio import tqdm

# Configuration
API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v6_results.csv"
CONCURRENCY_LIMIT = 50

client = genai.Client(api_key=API_KEY)

# Simple schema - decision only
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


async def predict(row: dict, semaphore: asyncio.Semaphore) -> str:
    """Make a single prediction for which text is more similar to the anchor."""
    prompt = f"""Which story (A or B) is more similar to the Anchor story?

Focus on NARRATIVE SIMILARITY based on:
1. Abstract themes (core problems, central ideas) - IGNORE concrete settings
2. Course of action (sequence of events)
3. Outcomes (how the story ends)

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Decision (A or B):"""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                )
                
                data = json.loads(response.text)
                return data.get("decision", "A")
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return "A"  # Default fallback


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v6")
    print(f"Model: {MODEL}")
    print(f"Strategy: Original simple baseline")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [predict(row, semaphore) for _, row in df.iterrows()]
    predictions = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    
    # Calculate accuracy
    correct = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).sum()
    accuracy = correct / len(df)
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS")
    print(f"{'─'*60}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(df)})")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'═'*60}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

