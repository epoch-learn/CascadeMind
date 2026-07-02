"""
SemEval 2026 Task 4 Subtask 1 - v14 OPTIMAL
Optimized based on empirical analysis of v12 and v13 results.

Key insights from analysis:
1. Unanimous forward votes achieve 83-84% accuracy
2. Bidirectional agreement also achieves 83-84% accuracy  
3. Pro/tiebreaker models only achieve ~50% on conflict cases (ambiguous by nature)

OPTIMAL STRATEGY:
1. Forward pass k=7 (higher k = more unanimous cases)
2. If unanimous → use immediately (fastest path, 83%+ accuracy)
3. If strong majority (≥6/7) → use that (very reliable)
4. Otherwise reverse pass k=7
5. If bidirectional agreement → use (83%+ accuracy)
6. If conflict → use COMBINED vote count (no complex tiebreaker)

This maximizes the high-accuracy paths and avoids wasting API calls on unreliable tiebreakers.
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

MODEL = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v14_optimal_results.csv"
CONCURRENCY_LIMIT = 30
NUM_SAMPLES = 7  # Higher k for more unanimous cases
TEMPERATURE = 0.7

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY)

# Enhanced few-shot examples with clear reasoning
FEW_SHOT_EXAMPLES = """Analyze NARRATIVE SIMILARITY using the official 3-dimension framework.

=== EXAMPLE 1 ===
Anchor: A young orphan discovers magical powers, goes to special school, makes friends, faces dark villain.
Story A: Teenager inherits fortune, attends elite boarding school, navigates social hierarchies and romance.
Story B: Child from humble origins learns destiny, trains with mentors, bonds with companions, confronts evil force that killed parents.

Analysis:
- THEME: Anchor/B share "chosen one discovering destiny, fighting evil connected to past" - A is about wealth/social climbing
- ACTION: Anchor/B follow discovery→training→bonding→confrontation - A follows inheritance→social navigation→romance
- OUTCOME: Anchor/B end with hero facing villain - A resolves romantic/social conflicts

Answer: B

=== EXAMPLE 2 ===
Anchor: Two from feuding families fall in love, marry secretly, miscommunication causes both deaths, reconciles families.
Story A: Woman poisons husband after affair discovery, remarries business partner.
Story B: Lovers from rival gangs try escaping, caught in crossfire, deaths end gang war.

Analysis:
- THEME: Anchor/B share "forbidden love, tragic death brings peace" - A is about murder for personal gain
- ACTION: Anchor/B: meet across divide→fall in love→try to unite→die→peace achieved - A: betrayal→murder→remarriage
- OUTCOME: Anchor/B have tragic deaths achieving reconciliation - A has calculated murder and remarriage

Answer: B

=== EXAMPLE 3 ===
Anchor: Detective investigates murders, discovers killer is someone close, faces moral choice between justice and loyalty.
Story A: Journalist uncovers corporate fraud, must decide whether to publish story destroying mentor's career.
Story B: Cop tracks serial killer through crime scene clues, catches them after car chase.

Analysis:
- THEME: Anchor/A share "investigation reveals someone close is guilty, duty vs loyalty dilemma" - B is straightforward crime-solving
- ACTION: Anchor/A: investigate→discover close person guilty→face dilemma - B: investigate→chase→arrest
- OUTCOME: Anchor/A end with impossible choice - B ends with simple arrest

Answer: A

=== NOW ANALYZE ===
"""

# Schema
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])
    },
    required=["decision"]
)


def build_prompt(row: dict, reverse: bool = False) -> str:
    """Build comparison prompt. reverse=True swaps A/B presentation."""
    if reverse:
        story_a, story_b = row['text_b'], row['text_a']
    else:
        story_a, story_b = row['text_a'], row['text_b']
    
    return f"""{FEW_SHOT_EXAMPLES}
Anchor: {row['anchor_text']}

Story A: {story_a}

Story B: {story_b}

Focus on NARRATIVE SIMILARITY (not surface features like setting, names, genre):
1. ABSTRACT THEME - Core problems, central ideas, motifs
2. COURSE OF ACTION - Event sequences, turning points, conflict development  
3. OUTCOMES - Endings, character fates, resolutions

Which story (A or B) is more narratively similar to the Anchor?"""


async def predict_single(prompt: str, semaphore: asyncio.Semaphore) -> str | None:
    """Make a single prediction. Returns None on error."""
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
                decision = data.get("decision")
                if decision in ["A", "B"]:
                    return decision
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1.5 ** attempt)
        return None


def map_reverse_vote(vote: str) -> str:
    """Map reverse vote back to original labels. In reverse: A→B, B→A."""
    return "B" if vote == "A" else "A"


async def predict_optimal(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Optimal prediction pipeline:
    1. Forward pass k=7
    2. If unanimous (all same) or strong majority (≥6/7) → use immediately
    3. Otherwise reverse pass k=7
    4. If bidirectional agreement → use
    5. If conflict → use combined vote count
    """
    
    # Stage 1: Forward pass
    forward_prompt = build_prompt(row, reverse=False)
    forward_tasks = [predict_single(forward_prompt, semaphore) for _ in range(NUM_SAMPLES)]
    forward_results = await asyncio.gather(*forward_tasks)
    
    forward_votes = [r for r in forward_results if r is not None]
    forward_errors = sum(1 for r in forward_results if r is None)
    forward_counts = Counter(forward_votes)
    
    # Check for unanimous or strong majority in forward pass
    if len(forward_votes) >= 5:  # Need at least 5 valid votes
        max_votes = max(forward_counts.values()) if forward_counts else 0
        if max_votes >= 6:  # 6/7 or 7/7 is very strong signal
            decision = forward_counts.most_common(1)[0][0]
            return {
                "decision": decision,
                "forward_votes": forward_votes,
                "reverse_votes": [],
                "forward_errors": forward_errors,
                "reverse_errors": 0,
                "method": "strong_forward",
                "confidence": max_votes / len(forward_votes)
            }
    
    # Stage 2: Reverse pass (for weaker forward signals)
    reverse_prompt = build_prompt(row, reverse=True)
    reverse_tasks = [predict_single(reverse_prompt, semaphore) for _ in range(NUM_SAMPLES)]
    reverse_results = await asyncio.gather(*reverse_tasks)
    
    # Map reverse votes back to original labels
    reverse_votes_raw = [r for r in reverse_results if r is not None]
    reverse_votes = [map_reverse_vote(v) for v in reverse_votes_raw]
    reverse_errors = sum(1 for r in reverse_results if r is None)
    reverse_counts = Counter(reverse_votes)
    
    # Get majorities
    forward_majority = forward_counts.most_common(1)[0][0] if forward_votes else None
    reverse_majority = reverse_counts.most_common(1)[0][0] if reverse_votes else None
    
    # Stage 3: Check bidirectional agreement
    if forward_majority and reverse_majority:
        if forward_majority == reverse_majority:
            # Strong signal: both directions agree
            combined = forward_votes + reverse_votes
            combined_counts = Counter(combined)
            confidence = combined_counts[forward_majority] / len(combined)
            
            return {
                "decision": forward_majority,
                "forward_votes": forward_votes,
                "reverse_votes": reverse_votes,
                "forward_errors": forward_errors,
                "reverse_errors": reverse_errors,
                "method": "bidirectional_agree",
                "confidence": confidence
            }
    
    # Stage 4: Conflict - use combined vote count (simple, effective)
    all_votes = forward_votes + reverse_votes
    if all_votes:
        combined_counts = Counter(all_votes)
        decision = combined_counts.most_common(1)[0][0]
        confidence = combined_counts[decision] / len(all_votes)
        
        return {
            "decision": decision,
            "forward_votes": forward_votes,
            "reverse_votes": reverse_votes,
            "forward_errors": forward_errors,
            "reverse_errors": reverse_errors,
            "method": "combined_vote",
            "confidence": confidence
        }
    
    return {
        "decision": "ERROR",
        "forward_votes": [],
        "reverse_votes": [],
        "forward_errors": forward_errors,
        "reverse_errors": reverse_errors,
        "method": "all_failed",
        "confidence": 0.0
    }


async def main():
    print(f"{'═'*65}")
    print(f"  SemEval 2026 Task 4 Subtask 1 - v14 OPTIMAL")
    print(f"{'═'*65}")
    print(f"  Model: {MODEL}")
    print(f"  Strategy: Bidirectional k={NUM_SAMPLES} with early exit")
    print(f"            Strong forward (≥6/{NUM_SAMPLES}) → immediate")
    print(f"            Bidirectional agree → high confidence")
    print(f"            Combined voting for conflicts")
    print(f"{'═'*65}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples")
    
    # Run predictions
    print(f"\n🧠 Running optimal predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [predict_optimal(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    df["predicted_decision"] = [r["decision"] for r in results]
    df["predicted_text_a_is_closer"] = [r["decision"] == "A" for r in results]
    df["forward_votes"] = [",".join(r["forward_votes"]) if r["forward_votes"] else "none" for r in results]
    df["reverse_votes"] = [",".join(r["reverse_votes"]) if r["reverse_votes"] else "none" for r in results]
    df["method"] = [r["method"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    
    # Calculate accuracy
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) > 0 else 0
    total_errors = len(df) - len(valid_df)
    
    # Print results
    print(f"\n{'═'*65}")
    print(f"📊 RESULTS - v14 OPTIMAL")
    print(f"{'─'*65}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*65}")
    
    # Method breakdown
    for method in ["strong_forward", "bidirectional_agree", "combined_vote"]:
        method_df = valid_df[valid_df["method"] == method]
        if len(method_df) > 0:
            method_correct = (method_df["predicted_text_a_is_closer"] == method_df["text_a_is_closer"]).sum()
            method_acc = method_correct / len(method_df)
            print(f"{method}: {len(method_df)} cases → {method_correct}/{len(method_df)} = {method_acc:.1%}")
    
    print(f"{'═'*65}")
    
    # Vote distribution
    all_forward = [v for r in results for v in r["forward_votes"]]
    all_reverse = [v for r in results for v in r["reverse_votes"]]
    
    if all_forward:
        fwd_a = all_forward.count('A')
        fwd_b = all_forward.count('B')
        print(f"\n📊 Forward votes: A={fwd_a} ({fwd_a/len(all_forward):.1%}) B={fwd_b} ({fwd_b/len(all_forward):.1%})")
    
    if all_reverse:
        rev_a = all_reverse.count('A')
        rev_b = all_reverse.count('B')
        print(f"📊 Reverse votes: A={rev_a} ({rev_a/len(all_reverse):.1%}) B={rev_b} ({rev_b/len(all_reverse):.1%})")
    
    # Confidence analysis
    avg_conf = df["confidence"].mean()
    print(f"📊 Average confidence: {avg_conf:.1%}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")
    
    return accuracy


if __name__ == "__main__":
    asyncio.run(main())

