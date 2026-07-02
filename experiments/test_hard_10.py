"""
Test Gemini 3 Flash on 10 hardest cases from previous runs.
These are cases that required escalation (32 votes) and were still wrong.
"""

import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from datetime import datetime
from collections import Counter
import nest_asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
import os

# Load sentence transformer model for semantic similarity
print("Loading sentence-transformers model...")
_embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optimized ensemble weights
ENSEMBLE_WEIGHTS = np.array([0.0784, 0.4880, 0.4025, 0.0152, 0.0159])

nest_asyncio.apply()

# Token tracking
total_input_tokens = 0
total_output_tokens = 0
total_api_calls = 0
token_lock = asyncio.Lock()
api_call_log = []
log_lock = asyncio.Lock()

API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3-flash-preview"

K_INITIAL = 8
K_ESCALATE = 8

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"run_hard10_{RUN_ID}"

# 10 hardest cases (escalated AND wrong in previous runs)
HARD_INDICES = [3, 5, 9, 11, 12, 18, 19, 20, 23, 28]


def build_prompt(anc, txt_a, txt_b):
    """Build the 3-shot narrative similarity prompt."""
    return f"""You are a literary analyst comparing narrative structures. Analyze which story shares more structural similarity with the Anchor.

NARRATIVE SIMILARITY has three aspects:
1. ABSTRACT THEME: Core problems, central ideas, motifs (NOT concrete setting/time period)
2. COURSE OF ACTION: Sequence of events and their ORDER
3. OUTCOMES: Final results only (conflict resolution, character fates, lessons learned)

IGNORE these factors completely:
- Writing style, prose quality
- Concrete setting, time period, genre
- Character/location names
- Text length, level of detail

Example 1:
Anchor: Simon's expedition into the rainforest encounters difficulties: an animal attacks, they can't sleep, members develop fever. Finally they reach an ancient pyramid hidden in the forest.
Story A: Ed's team traverses planet Xylox. Robotic drones attack, they can't rest, members develop fever from radiation. Finally they reach an ancient monolith hidden beneath the sands.
Story B: Mary's expedition into the rainforest suffers injuries. They reach their destination but find no pyramid—only jungle. Weakened, they succumb to fever; the rainforest swallows them.
Answer: A
Why: Same structure (obstacles → fever → reach hidden goal) despite different settings. B has same setting but opposite outcome.

Example 2:
Anchor: Andrew buys food and drinks, prepares everything at home, then impresses arriving family with homemade cookies and fancy drinks.
Story A: Zoie buys ammunition and guns, prepares traps and firing positions at home, then deals destruction when zombies rush her doors. She ultimately loses.
Story B: Erica attends a paper plane competition with little preparation and wins.
Answer: A
Why: Same course of action (purchase → prepare → use preparations) despite completely different themes and outcomes.

Example 3:
Anchor: Neo loses eyesight in accident. He learns to handle new challenges. He makes a new friend Anna who helps him, finding connections he never thought possible.
Story A: Adam has many friends. He loses hearing in accident. He struggles with challenges, alienates friends, becomes very lonely.
Story B: Brian leads a lonely life. He encounters a fellow artist. They make a deep connection over shared interests and remain friends.
Answer: B
Why: A has similar theme (disability) but OPPOSITE trajectory (connected→isolated vs isolated→connected). B matches the arc: lonely → encounter → meaningful connection.

Now analyze:

Anchor: {anc}

Story A: {txt_a}

Story B: {txt_b}

Respond with JSON containing:
- "reasoning": Your analysis of narrative similarity (2-3 sentences)
- "confidence": Your confidence level 0.00-1.00 (1.00=certain, 0.00=guessing)
- "decision": "A" or "B"

Example response: {{"reasoning": "Both anchor and A follow a redemption arc...", "confidence": 0.85, "decision": "A"}}"""


async def get_single_vote(prompt, semaphore, row_idx, vote_idx, round_name):
    """Get a single vote with full logging."""
    global total_input_tokens, total_output_tokens, total_api_calls, api_call_log

    call_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    timestamp = datetime.now().isoformat()

    log_entry = {
        "call_id": call_id,
        "run_id": RUN_ID,
        "timestamp": timestamp,
        "row_idx": row_idx,
        "vote_idx": vote_idx,
        "round": round_name,
        "model": MODEL_NAME,
        "status": "pending",
        "decision": None,
        "confidence": None,
        "reasoning": None,
        "raw_response": None,
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": None,
        "error": None,
        "error_type": None,
        "attempt": 0,
        "needs_patch": False
    }

    async with semaphore:
        for attempt in range(3):
            log_entry["attempt"] = attempt + 1
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=1.0,
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        ]
                    )
                )

                latency_ms = (time.time() - start_time) * 1000
                log_entry["latency_ms"] = round(latency_ms, 2)

                input_tokens = 0
                output_tokens = 0
                if response.usage_metadata:
                    input_tokens = response.usage_metadata.prompt_token_count or 0
                    output_tokens = response.usage_metadata.candidates_token_count or 0

                log_entry["input_tokens"] = input_tokens
                log_entry["output_tokens"] = output_tokens

                async with token_lock:
                    total_api_calls += 1
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                if not response.candidates:
                    raise ValueError("No candidates returned")

                raw_text = response.candidates[0].content.parts[0].text
                log_entry["raw_response"] = raw_text

                data = json.loads(raw_text)
                decision = data.get('decision')
                confidence = data.get('confidence')
                reasoning = data.get('reasoning')

                log_entry["decision"] = decision
                log_entry["confidence"] = confidence
                log_entry["reasoning"] = reasoning

                # Normalize confidence to 0.0-1.0
                if confidence is not None:
                    if isinstance(confidence, (int, float)):
                        if confidence > 1.0:
                            confidence = confidence / 10.0
                        confidence = max(0.0, min(1.0, float(confidence)))
                    else:
                        confidence = None

                if decision in ['A', 'B']:
                    log_entry["status"] = "success"
                    async with log_lock:
                        api_call_log.append(log_entry)
                    return {
                        "decision": decision,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "call_id": call_id,
                        "status": "success",
                        "error": None
                    }

                raise ValueError(f"Invalid decision: {decision}")

            except Exception as e:
                error_str = str(e)
                log_entry["error"] = error_str

                if "429" in error_str:
                    log_entry["error_type"] = "rate_limit"
                    await asyncio.sleep(5 * (attempt + 1))
                elif "SAFETY" in error_str.upper() or "BLOCKED" in error_str.upper():
                    log_entry["error_type"] = "safety_filter"
                elif "timeout" in error_str.lower():
                    log_entry["error_type"] = "timeout"
                elif "json" in error_str.lower():
                    log_entry["error_type"] = "json_parse"
                else:
                    log_entry["error_type"] = "unknown"

                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        log_entry["status"] = "error"
        log_entry["needs_patch"] = True
        log_entry["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        async with log_lock:
            api_call_log.append(log_entry)

        return {
            "decision": "ERROR",
            "confidence": None,
            "reasoning": None,
            "call_id": call_id,
            "status": "error",
            "error": log_entry["error"],
            "error_type": log_entry["error_type"],
            "needs_patch": True
        }


async def get_votes(row, row_idx, semaphore, num_votes=8, round_name="initial"):
    """Get multiple votes via parallel individual API calls."""
    anc = row['anchor_text']
    txt_a = row['text_a']
    txt_b = row['text_b']

    prompt = build_prompt(anc, txt_a, txt_b)

    tasks = [
        get_single_vote(prompt, semaphore, row_idx, vote_idx, round_name)
        for vote_idx in range(num_votes)
    ]
    results = await asyncio.gather(*tasks)

    return results


async def predict_with_cascade(row, row_idx, semaphore):
    """Full cascade with supermajority detection and escalation."""
    vote_results = await get_votes(row, row_idx, semaphore, K_INITIAL, "round1")

    valid_votes = [v for v in vote_results if v["decision"] in ['A', 'B']]
    if not valid_votes:
        return {
            "prediction": "ERROR",
            "votes": vote_results,
            "strategy": "all_errors",
            "num_rounds": 1
        }

    decisions = [v["decision"] for v in valid_votes]
    counter = Counter(decisions)

    top_count = counter.most_common(1)[0][1]
    if top_count >= 7:
        majority = counter.most_common(1)[0][0]
        return {
            "prediction": majority == "A",
            "votes": vote_results,
            "strategy": f"supermajority_{top_count}/{len(valid_votes)}",
            "num_rounds": 1
        }

    # Escalation
    all_vote_results = list(vote_results)

    for batch_num in range(3):
        extra_results = await get_votes(row, row_idx, semaphore, K_ESCALATE, f"escalation_{batch_num+1}")
        all_vote_results.extend(extra_results)

    valid_all = [v for v in all_vote_results if v["decision"] in ['A', 'B']]

    if not valid_all:
        return {
            "prediction": "ERROR",
            "votes": all_vote_results,
            "strategy": "all_errors",
            "num_rounds": 4
        }

    all_decisions = [v["decision"] for v in valid_all]
    counter = Counter(all_decisions)

    majority = counter.most_common(1)[0][0]

    return {
        "prediction": majority == "A",
        "votes": all_vote_results,
        "strategy": f"escalated_{K_INITIAL}+{K_ESCALATE}x3",
        "num_rounds": 4
    }


async def main():
    global api_call_log

    os.makedirs(RUN_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f" Testing Gemini 2.5 Flash on 10 HARDEST cases")
    print(f" (Cases that required escalation AND were wrong previously)")
    print(f" Run ID: {RUN_ID}")
    print(f" Output: {RUN_DIR}/")
    print(f"{'='*70}")

    print("\n Loading Dataset...")
    df = pd.read_json("data/dev_track_a.jsonl", lines=True)

    # Select only the hard cases
    test_df = df.iloc[HARD_INDICES].copy()
    test_df = test_df.reset_index(drop=True)
    test_df['original_idx'] = HARD_INDICES

    print(f" Selected {len(test_df)} hardest cases: {HARD_INDICES}")
    print(f" Model: {MODEL_NAME}")
    print(f" K_INITIAL={K_INITIAL}, K_ESCALATE={K_ESCALATE}")

    start_time = time.time()

    semaphore = asyncio.Semaphore(20)

    results = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        orig_idx = HARD_INDICES[i]
        print(f"\n Processing case {i+1}/10 (original idx={orig_idx})...")
        result = await predict_with_cascade(row, orig_idx, semaphore)
        results.append(result)

        # Show immediate result
        pred = "A" if result["prediction"] == True else ("B" if result["prediction"] == False else "ERROR")
        truth = "A" if row["text_a_is_closer"] else "B"
        correct = "✓" if pred == truth else "✗"
        votes = [v["decision"] for v in result["votes"] if v["decision"] in ['A', 'B']]
        vote_summary = f"A:{votes.count('A')} B:{votes.count('B')}"
        print(f"   Result: pred={pred}, truth={truth} {correct} | votes: {vote_summary} | strategy: {result['strategy']}")

    elapsed = time.time() - start_time

    # Compile results
    test_df["predicted_text_a_is_closer"] = [r["prediction"] for r in results]
    test_df["strategy"] = [r["strategy"] for r in results]
    test_df["num_rounds"] = [r["num_rounds"] for r in results]
    test_df["votes_summary"] = [
        ",".join([v["decision"] for v in r["votes"] if v["decision"] in ['A', 'B']])
        for r in results
    ]

    # Calculate accuracy
    valid_df = test_df[test_df["predicted_text_a_is_closer"] != "ERROR"]
    accuracy = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).mean() if len(valid_df) > 0 else 0

    print(f"\n{'='*70}")
    print(f" RESULTS - 10 Hardest Cases")
    print(f"{'-'*70}")
    print(f" ACCURACY: {accuracy:.0%} ({int(accuracy * len(valid_df))}/{len(valid_df)})")
    print(f" (Previous system: 0% on these same cases)")
    print(f"{'-'*70}")
    print(f" Total API Calls: {total_api_calls}")
    print(f" Time: {elapsed:.1f}s")
    print(f" Tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"{'='*70}")

    # Save results
    results_csv = os.path.join(RUN_DIR, "results.csv")
    test_df.to_csv(results_csv, index=False)
    print(f"\n Saved to: {results_csv}")

    # Save detailed votes
    votes_file = os.path.join(RUN_DIR, "votes_detailed.json")
    votes_data = []
    for i, r in enumerate(results):
        pred = r["prediction"]
        truth = bool(test_df.iloc[i]["text_a_is_closer"])
        # Convert numpy bool to Python bool for JSON serialization
        if pred != "ERROR":
            pred_bool = bool(pred) if isinstance(pred, (bool, np.bool_)) else pred
            correct = bool(pred_bool == truth)
        else:
            correct = None
        votes_data.append({
            "test_idx": i,
            "original_idx": HARD_INDICES[i],
            "prediction": pred_bool if pred != "ERROR" else "ERROR",
            "ground_truth": truth,
            "correct": correct,
            "strategy": r["strategy"],
            "votes": r["votes"]
        })

    with open(votes_file, 'w') as f:
        json.dump(votes_data, f, indent=2)
    print(f" Detailed votes: {votes_file}")

    # Save API call log
    log_file = os.path.join(RUN_DIR, "api_calls.jsonl")
    with open(log_file, 'w') as f:
        for entry in api_call_log:
            f.write(json.dumps(entry) + '\n')
    print(f" API log: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())
