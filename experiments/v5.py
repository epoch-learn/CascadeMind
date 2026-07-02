"""
SemEval 2026 Task 4 Subtask 1 - v5
Prompt-based red herring avoidance.

Uses explicit warnings about common model errors instead of preprocessing.
Uses Gemini 2.5 Flash.
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
OUTPUT_PATH = "data/v5_results.csv"
CONCURRENCY_LIMIT = 100

client = genai.Client(api_key=API_KEY)

# Enhanced prompt with red herring warnings
PROMPT_TEMPLATE = """Which story (A or B) is more narratively similar to the Anchor?

RED HERRINGS TO AVOID:
- Matching on surface labels (e.g., "both involve deception") without checking if the NARRATIVE ARC is similar
- Concrete settings (time period, location, historical events) - these do NOT define narrative similarity
- Format/style ("both are films", "both are sci-fi") - irrelevant
- Similar character types without similar story arcs
- Shared genres or tropes that don't share the same story structure

WHAT ACTUALLY MATTERS:
1. Abstract Theme - What is this story fundamentally ABOUT? (not the setting, the core problem/idea)
2. Course of Action - The SEQUENCE of events, turning points, how conflicts develop and resolve
3. Outcomes - How it ENDS, character fates, lessons learned

Anchor: {anchor}

Story A: {text_a}

Story B: {text_b}

Provide your reasoning and decision."""

# Schema for structured JSON output with reasoning
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "reasoning": types.Schema(
            type=types.Type.STRING,
            description="Your analysis comparing the NARRATIVE ARCS (not surface features) on abstract theme, course of action, and outcomes."
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
    prompt = PROMPT_TEMPLATE.format(
        anchor=row['anchor_text'],
        text_a=row['text_a'],
        text_b=row['text_b']
    )

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
    print(f"SemEval 2026 Task 4 Subtask 1 - v5")
    print(f"Model: {MODEL}")
    print(f"Strategy: Red herring avoidance via prompt")
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
            print(f"Reasoning: {r[:600]}..." if len(r) > 600 else f"Reasoning: {r}")
            break
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

