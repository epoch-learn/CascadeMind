import os
import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from tqdm.asyncio import tqdm
from collections import Counter

API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])
    },
    required=["decision"]
)

async def single_prediction(row, semaphore):
    prompt = f"""Which story (A or B) is more similar to the Anchor story?

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Return your decision as JSON with a "decision" field set to either "A" or "B"."""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                )
                data = json.loads(response.text)
                return data.get('decision', 'A')
            except:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return 'A'

async def predict_with_cascade(row, semaphore):
    """Cascade: Maj@3 first, if split -> +5 more votes for Maj@8."""
    # First round: 3 votes
    votes = await asyncio.gather(
        single_prediction(row, semaphore),
        single_prediction(row, semaphore),
        single_prediction(row, semaphore)
    )
    
    counter = Counter(votes)
    if counter.most_common(1)[0][1] == 3:  # All 3 agree
        majority = counter.most_common(1)[0][0]
        return (majority == "A", votes, "unanimous_3", 3)
    
    # Split vote: run 5 more times
    extra_votes = await asyncio.gather(
        single_prediction(row, semaphore),
        single_prediction(row, semaphore),
        single_prediction(row, semaphore),
        single_prediction(row, semaphore),
        single_prediction(row, semaphore)
    )
    
    all_votes = list(votes) + list(extra_votes)  # 8 total votes
    counter = Counter(all_votes)
    majority = counter.most_common(1)[0][0]
    return (majority == "A", all_votes, "escalated_8", 8)

async def main():
    print("🚀 Loading Dataset...")
    df = pd.read_json("data/dev_track_a.jsonl", lines=True)
    test_df = df.head(100).copy()

    print(f"🧠 Running CASCADE (Maj@3 → Maj@8) on {len(test_df)} rows...")
    print(f"   Phase 1: Majority@3 for all")
    print(f"   Phase 2: +5 votes for split cases → Majority@8")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(50)
    tasks = [predict_with_cascade(row, semaphore) for _, row in test_df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    test_df["predicted_text_a_is_closer"] = [r[0] for r in results]
    test_df["votes"] = [r[1] for r in results]
    test_df["strategy"] = [r[2] for r in results]
    test_df["num_votes"] = [r[3] for r in results]
    
    accuracy = (test_df["predicted_text_a_is_closer"] == test_df["text_a_is_closer"]).mean()
    elapsed = time.time() - start_time
    
    # Breakdown by strategy
    unanimous = test_df[test_df["strategy"] == "unanimous_3"]
    escalated = test_df[test_df["strategy"] == "escalated_8"]
    
    unanimous_acc = (unanimous['predicted_text_a_is_closer'] == unanimous['text_a_is_closer']).mean() if len(unanimous) > 0 else 0
    escalated_acc = (escalated['predicted_text_a_is_closer'] == escalated['text_a_is_closer']).mean() if len(escalated) > 0 else 0
    
    total_calls = unanimous["num_votes"].sum() + escalated["num_votes"].sum()
    
    print(f"\n{'═'*60}")
    print(f"📊 CASCADE RESULTS (gemini-2.5-flash only):")
    print(f"{'─'*60}")
    print(f"Unanimous@3: {len(unanimous)} cases → {unanimous_acc:.1%} accuracy")
    print(f"Escalated@8: {len(escalated)} cases → {escalated_acc:.1%} accuracy")
    print(f"{'─'*60}")
    print(f"📊 OVERALL ACCURACY: {accuracy:.1%} ({int(accuracy * len(test_df))}/{len(test_df)})")
    print(f"📊 Total API calls: {total_calls}")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"{'═'*60}")
    
    test_df.to_csv("data/cascade_maj_results.csv", index=False)
    print(f"📂 Results saved to: data/cascade_maj_results.csv")

if __name__ == "__main__":
    asyncio.run(main())
