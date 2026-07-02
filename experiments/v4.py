"""
SemEval 2026 Task 4 Subtask 1 - v4 Maximum Accuracy
Strip forbidden factors before comparison (per official guidelines).

Step 1: Strip stories (remove names, locations, dates, concrete settings)
Step 2: Compare stripped versions on narrative similarity

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
OUTPUT_PATH = "data/v4_results.csv"
CONCURRENCY_LIMIT = 100  # Heavy parallelization

client = genai.Client(api_key=API_KEY)

# Strip prompt - removes factors that do NOT contribute to narrative similarity
STRIP_PROMPT = """Rewrite this story, removing factors that do NOT contribute to narrative similarity:

REMOVE (per official guidelines):
- Character names → replace with roles (protagonist, antagonist, friend, etc.)
- Location names → replace with generic settings (a city, a village, abroad)
- Time periods/dates → remove or say "in the past" / "during a conflict"
- Concrete historical settings → abstract (e.g., "World War I" → "a major war")
- Film/movie format markers → remove ("The film follows..." → "The story follows...")

KEEP (these define narrative similarity):
- Abstract theme (what the story is fundamentally about)
- Course of action (sequence of events, turning points)
- Outcomes (how it ends, character fates, lessons)

Story:
{story}

Provide what you stripped and the rewritten version."""

# Schema for strip output
strip_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "stripped_elements": types.Schema(
            type=types.Type.STRING,
            description="Brief list of what was removed (names, locations, dates, etc.)"
        ),
        "rewritten": types.Schema(
            type=types.Type.STRING,
            description="The story rewritten with forbidden factors removed"
        )
    },
    required=["stripped_elements", "rewritten"]
)

# Compare prompt
COMPARE_PROMPT = """Which story (A or B) is more narratively similar to the Anchor?

Consider:
1. Abstract Theme - the defining problems, central ideas, core motifs
2. Course of Action - sequence of events, turning points, how conflicts develop
3. Outcomes - how the story ends, character fates, lessons

Anchor: {anchor}

Story A: {text_a}

Story B: {text_b}

Provide your reasoning and decision."""

# Schema for decision with reasoning
decision_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "reasoning": types.Schema(
            type=types.Type.STRING,
            description="Your analysis comparing the stories on abstract theme, course of action, and outcomes"
        ),
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])
    },
    required=["reasoning", "decision"]
)


async def strip_story(story: str, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """Strip forbidden factors from a story. Returns (rewritten, stripped_elements)."""
    prompt = STRIP_PROMPT.format(story=story)
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=strip_schema
                    )
                )
                data = json.loads(response.text)
                return data.get("rewritten", story), data.get("stripped_elements", "")
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return story, ""  # Fallback to original if stripping fails


async def compare_stories(anchor: str, text_a: str, text_b: str, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """Compare stripped stories and return (decision, reasoning)."""
    prompt = COMPARE_PROMPT.format(anchor=anchor, text_a=text_a, text_b=text_b)
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=decision_schema
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
    print(f"SemEval 2026 Task 4 Subtask 1 - v4")
    print(f"Model: {MODEL}")
    print(f"Strategy: Strip forbidden factors, then compare")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # STEP 1: Strip ALL stories in parallel (600 calls)
    print(f"\n🧹 Step 1: Stripping all stories ({len(df) * 3} calls)...")
    start_time = time.time()
    
    all_strip_tasks = []
    for _, row in df.iterrows():
        all_strip_tasks.append(strip_story(row['anchor_text'], semaphore))
        all_strip_tasks.append(strip_story(row['text_a'], semaphore))
        all_strip_tasks.append(strip_story(row['text_b'], semaphore))
    
    all_stripped = await tqdm.gather(*all_strip_tasks)
    
    # Reorganize results (every 3 items = anchor, a, b) - each is (rewritten, stripped_elements)
    stripped_anchors = [all_stripped[i*3][0] for i in range(len(df))]
    stripped_anchor_info = [all_stripped[i*3][1] for i in range(len(df))]
    stripped_as = [all_stripped[i*3 + 1][0] for i in range(len(df))]
    stripped_a_info = [all_stripped[i*3 + 1][1] for i in range(len(df))]
    stripped_bs = [all_stripped[i*3 + 2][0] for i in range(len(df))]
    stripped_b_info = [all_stripped[i*3 + 2][1] for i in range(len(df))]
    
    strip_time = time.time() - start_time
    print(f"   Stripping done in {strip_time:.1f}s")
    
    # STEP 2: Compare ALL stripped stories in parallel (200 calls)
    print(f"\n🔍 Step 2: Comparing stripped stories ({len(df)} calls)...")
    compare_start = time.time()
    
    compare_tasks = [
        compare_stories(stripped_anchors[i], stripped_as[i], stripped_bs[i], semaphore)
        for i in range(len(df))
    ]
    compare_results = await tqdm.gather(*compare_tasks)
    
    # Unpack decisions and reasoning
    predictions = [r[0] for r in compare_results]
    reasoning = [r[1] for r in compare_results]
    
    compare_time = time.time() - compare_start
    print(f"   Comparing done in {compare_time:.1f}s")
    
    elapsed = time.time() - start_time
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["reasoning"] = reasoning
    df["stripped_anchor"] = stripped_anchors
    df["stripped_anchor_info"] = stripped_anchor_info
    df["stripped_a"] = stripped_as
    df["stripped_a_info"] = stripped_a_info
    df["stripped_b"] = stripped_bs
    df["stripped_b_info"] = stripped_b_info
    
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
    
    # Show a sample of stripping and reasoning
    print(f"\n📝 Sample (first example):")
    print(f"{'─'*60}")
    print(f"ORIGINAL ANCHOR (truncated):")
    print(f"{df.iloc[0]['anchor_text'][:200]}...")
    print(f"\nSTRIPPED (what was removed):")
    print(f"{stripped_anchor_info[0]}")
    print(f"\nSTRIPPED ANCHOR:")
    print(f"{stripped_anchors[0][:200]}...")
    print(f"\n💭 REASONING:")
    print(f"{reasoning[0][:400]}..." if len(reasoning[0]) > 400 else reasoning[0])
    print(f"\nPredicted: {predictions[0]}, Actual: {'A' if df.iloc[0]['text_a_is_closer'] else 'B'}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

