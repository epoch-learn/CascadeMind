"""
SemEval 2026 Task 4 Subtask 1 Baseline
Narrative Similarity: Given an anchor text, predict which of two candidate texts 
(A or B) is more similar to the anchor.

Uses Gemini 3 Flash Preview with reasoning in schema for predictions.
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
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY is required. Copy .env.example to .env, fill a rotated key, and load it into your shell.")
MODEL = "gemini-3-flash-preview"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/baseline_results.csv"
CONCURRENCY_LIMIT = 50

client = genai.Client(api_key=API_KEY)

# Schema for structured JSON output with reasoning
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "reasoning": types.Schema(
            type=types.Type.STRING,
            description="Your step-by-step analysis comparing the stories based on abstract themes, course of action, and outcomes. Do NOT focus on surface features like setting, era, or genre."
        ),
        "decision": types.Schema(
            type=types.Type.STRING, 
            enum=["A", "B"],
            description="Your final choice: A if Story A is more similar to Anchor, B if Story B is more similar."
        )
    },
    required=["reasoning", "decision"]
)


async def predict(row: dict, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """Make a single prediction for which text is more similar to the anchor.
    
    Returns:
        tuple: (decision, reasoning)
    """
    prompt = f"""Which story (A or B) is more similar to the Anchor story?

Focus on NARRATIVE SIMILARITY based on:
1. Abstract themes (core problems, central ideas) - IGNORE concrete settings
2. Course of action (sequence of events)
3. Outcomes (how the story ends)

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Provide your reasoning and decision."""

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
                return data.get("decision", "A"), data.get("reasoning", "")
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return "A", ""  # Default fallback


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - Baseline")
    print(f"Model: {MODEL} (with reasoning in schema)")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"data file not found: {DATA_PATH}. See data/README.md for download and placement notes.")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running predictions with reasoning...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [predict(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Unpack predictions and reasoning
    predictions = [r[0] for r in results]
    reasoning = [r[1] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["reasoning"] = reasoning
    
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
    
    # Show a sample reasoning
    print(f"\n💭 Sample reasoning (first example):")
    for i, r in enumerate(reasoning):
        if r:
            print(f"{'─'*60}")
            print(f"Example {i+1}: Predicted {predictions[i]}, Actual {'A' if df.iloc[i]['text_a_is_closer'] else 'B'}")
            print(f"Reasoning: {r[:800]}..." if len(r) > 800 else f"Reasoning: {r}")
            break
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
