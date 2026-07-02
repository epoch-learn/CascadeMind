"""
SemEval 2026 Task 4 Subtask 1 - Synopsis Cascade Strategy
Phase 1: 3 votes on ORIGINAL text
Phase 2 (If split): Generate Synopses -> 5 votes on SHORTENED text

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
from collections import Counter

# --- Configuration ---
API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/mainv2_results.csv"
CONCURRENCY_LIMIT = 30

client = genai.Client(api_key=API_KEY)

# Schema for structured JSON output
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])
    },
    required=["decision"]
)


async def get_synopsis(text: str, target_word_count: int, semaphore: asyncio.Semaphore) -> str:
    """Summarizes text to a specific word count, removing fluff/adjectives."""
    prompt = f"""Rewrite the following story/text to be approximately {target_word_count} words long.
   
    Instructions:
    1. STRICTLY limit the length to around {target_word_count} words.
    2. Remove unnecessary adjectives, descriptive details, and fluff.
    3. Keep only the core events and essential meaning.
   
    Text to rewrite:
    {text}
    """
   
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return text  # Fallback: return original if summary fails


async def single_prediction(
    semaphore: asyncio.Semaphore,
    anchor: str,
    text_a: str,
    text_b: str
) -> str:
    """Makes a prediction for which text is more similar to the anchor."""
    prompt = f"""Which story (A or B) is more similar to the Anchor story?

Anchor: {anchor}

Story A: {text_a}

Story B: {text_b}

Return your decision as JSON with a "decision" field set to either "A" or "B"."""

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
        return "A"


async def predict_with_cascade(row: dict, semaphore: asyncio.Semaphore):
    """
    Phase 1: 3 votes on ORIGINAL text.
    Phase 2 (If split): Generate Synopses -> 5 votes on SHORTENED text.
    """
    anchor = row['anchor_text']
    text_a = row['text_a']
    text_b = row['text_b']
    
    # --- Round 1: Standard Prediction on Original Text ---
    votes = await asyncio.gather(
        single_prediction(semaphore, anchor, text_a, text_b),
        single_prediction(semaphore, anchor, text_a, text_b),
        single_prediction(semaphore, anchor, text_a, text_b)
    )

    counter = Counter(votes)
   
    # --- Check for Unanimous Agreement (3-0 or 0-3) ---
    if counter.most_common(1)[0][1] == 3:
        majority = counter.most_common(1)[0][0]
        return (majority == "A", votes, "unanimous_3", 3)

    # --- ESCALATION: Split vote detected (2-1) ---
    # 1. Calculate word counts
    len_anchor = len(str(anchor).split())
    len_a = len(str(text_a).split())
    len_b = len(str(text_b).split())
   
    # 2. Find target length (shortest of the 3)
    target_len = min(len_anchor, len_a, len_b)
   
    # 3. Generate Synopses (Parallelized)
    syn_results = await asyncio.gather(
        get_synopsis(anchor, target_len, semaphore),
        get_synopsis(text_a, target_len, semaphore),
        get_synopsis(text_b, target_len, semaphore)
    )
   
    syn_anchor, syn_a, syn_b = syn_results

    # 4. Round 2: 5 Extra Votes on the NEW Synopses
    extra_votes = await asyncio.gather(
        single_prediction(semaphore, syn_anchor, syn_a, syn_b),
        single_prediction(semaphore, syn_anchor, syn_a, syn_b),
        single_prediction(semaphore, syn_anchor, syn_a, syn_b),
        single_prediction(semaphore, syn_anchor, syn_a, syn_b),
        single_prediction(semaphore, syn_anchor, syn_a, syn_b)
    )

    # 5. Combine Votes (3 Original + 5 Synopsis)
    all_votes = list(votes) + list(extra_votes)
    counter = Counter(all_votes)
    majority = counter.most_common(1)[0][0]
   
    # Total calls = 3 (initial) + 3 (summaries) + 5 (extra) = 11 calls for escalated rows
    return (majority == "A", all_votes, "escalated_synopsis_8", 11)


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - Synopsis Cascade")
    print(f"Model: {MODEL}")
    print(f"Strategy: Unanimous@3 → Summarize → Majority@8")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True)
    # Evaluating on 200 rows
    test_df = df.head(200).copy()
    print(f"   Evaluating on {len(test_df)} examples")

    print(f"\n🧠 Running predictions...")
    start_time = time.time()

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
   
    tasks = [predict_with_cascade(row, semaphore) for _, row in test_df.iterrows()]
    results = await tqdm.gather(*tasks)

    # Unpack results
    test_df["predicted_text_a_is_closer"] = [r[0] for r in results]
    test_df["votes"] = [r[1] for r in results]
    test_df["strategy"] = [r[2] for r in results]
    test_df["num_calls"] = [r[3] for r in results]

    accuracy = (test_df["predicted_text_a_is_closer"] == test_df["text_a_is_closer"]).mean()
    elapsed = time.time() - start_time

    # Stats
    unanimous = test_df[test_df["strategy"] == "unanimous_3"]
    escalated = test_df[test_df["strategy"] == "escalated_synopsis_8"]
   
    u_acc = (unanimous['predicted_text_a_is_closer'] == unanimous['text_a_is_closer']).mean() if len(unanimous) > 0 else 0
    e_acc = (escalated['predicted_text_a_is_closer'] == escalated['text_a_is_closer']).mean() if len(escalated) > 0 else 0
    total_calls = test_df["num_calls"].sum()

    print(f"\n{'═'*60}")
    print(f"📊 SYNOPSIS CASCADE RESULTS")
    print(f"{'─'*60}")
    print(f"Unanimous@3 (Original): {len(unanimous)} cases → {u_acc:.1%} accuracy")
    print(f"Escalated@8 (Synopsis): {len(escalated)} cases → {e_acc:.1%} accuracy")
    print(f"{'─'*60}")
    print(f"📊 OVERALL ACCURACY: {accuracy:.1%} ({int(accuracy * len(test_df))}/{len(test_df)})")
    print(f"📊 Total API calls: {total_calls} (~{total_calls/len(test_df):.1f} avg per row)")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"{'═'*60}")

    test_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
