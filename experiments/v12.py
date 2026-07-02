"""
SemEval 2026 Task 4 Subtask 1 - v12
Bidirectional Self-Consistency Architecture

Strategy:
1. Run k=5 samples with Gemini 2.5 Flash (forward pass)
2. If unanimous → use that answer
3. If split → run k=5 samples with A/B swapped (reverse pass)
4. If forward and reverse majorities agree → use that
5. If conflict → route to Gemini 2.5 Pro for tiebreaker

Key insight: Position bias causes model to vote A 57% vs B 43% (ground truth is 50/50).
Bidirectional evaluation detects and corrects this bias.
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
MODEL_PRO = "gemini-2.5-pro"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v12_results.csv"
CONCURRENCY_LIMIT = 20
NUM_SAMPLES = 5
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

# Pro model response schema with reasoning
pro_response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "reasoning": types.Schema(type=types.Type.STRING),
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])
    },
    required=["reasoning", "decision"]
)


def build_prompt(row: dict) -> str:
    """Build the forward comparison prompt (A first, B second)."""
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


def build_reverse_prompt(row: dict) -> str:
    """Build the reverse comparison prompt (B first, A second).
    
    NOTE: In reverse, we present text_b as "Story A" and text_a as "Story B".
    When the model answers "A", it means text_b. When it answers "B", it means text_a.
    We map these back after getting the response.
    """
    return f"""{FEW_SHOT_EXAMPLES}
Now determine which story is more narratively similar to the Anchor.

Focus on NARRATIVE SIMILARITY:
1. Abstract Theme - core problems, central ideas, motifs (NOT concrete setting)
2. Course of Action - sequence of events, turning points, how conflicts develop
3. Outcomes - how the story ends, character fates, lessons

Anchor: {row['anchor_text']}

Story A: {row['text_b']}

Story B: {row['text_a']}

Answer (A or B):"""


def build_pro_prompt(row: dict, forward_votes: list[str], reverse_votes: list[str]) -> str:
    """Build prompt for Gemini Pro tiebreaker when bidirectional results conflict."""
    forward_counts = Counter(forward_votes)
    reverse_counts = Counter(reverse_votes)
    
    return f"""You are an expert at identifying NARRATIVE SIMILARITY. This is a difficult case where bidirectional analysis produced conflicting results.

Forward pass (Story A presented first): {forward_counts.get('A', 0)} votes for A, {forward_counts.get('B', 0)} votes for B
Reverse pass (Story B presented first): {reverse_counts.get('A', 0)} votes for original A, {reverse_counts.get('B', 0)} votes for original B

The conflicting votes suggest position bias may be affecting the decision. Please analyze carefully.

Focus on NARRATIVE SIMILARITY (not surface features):
1. Abstract Theme - core problems, central ideas, motifs
2. Course of Action - sequence of events, turning points
3. Outcomes - how the story ends, character fates

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Analyze which story shares more STRUCTURAL similarity with the Anchor, then decide A or B."""


async def predict_single_gemini(prompt: str, semaphore: asyncio.Semaphore, model: str = MODEL_FAST) -> str | None:
    """Make a single prediction with Gemini API. Returns None on error."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=model,
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


async def predict_with_pro(row: dict, forward_votes: list[str], reverse_votes: list[str], semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """Get decision from Gemini 2.5 Pro for conflicting bidirectional results."""
    prompt = build_pro_prompt(row, forward_votes, reverse_votes)
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=MODEL_PRO,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=pro_response_schema,
                        temperature=0.0  # Deterministic for tiebreaker
                    )
                )
                data = json.loads(response.text)
                decision = data.get("decision")
                reasoning = data.get("reasoning", "")
                if decision in ["A", "B"]:
                    return decision, reasoning
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback: use combined majority vote
        all_votes = forward_votes + reverse_votes
        if all_votes:
            return Counter(all_votes).most_common(1)[0][0], "Pro failed, using combined majority"
        return "ERROR", "All models failed"


def map_reverse_vote(vote: str) -> str:
    """Map reverse vote back to original labels.
    
    In reverse prompt: "A" means text_b, "B" means text_a
    So we swap: A→B, B→A
    """
    return "B" if vote == "A" else "A"


async def predict_bidirectional(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Bidirectional prediction:
    1. Run k samples forward (A first, B second)
    2. If unanimous → return immediately
    3. Run k samples reverse (B first, A second)
    4. Map reverse votes back to original labels
    5. If majorities agree → use that answer
    6. If conflict → route to Gemini Pro
    """
    # Stage 1: Forward pass
    forward_prompt = build_prompt(row)
    forward_tasks = [predict_single_gemini(forward_prompt, semaphore) for _ in range(NUM_SAMPLES)]
    forward_results = await asyncio.gather(*forward_tasks)
    
    forward_votes = [r for r in forward_results if r is not None]
    forward_errors = sum(1 for r in forward_results if r is None)
    forward_counts = Counter(forward_votes)
    
    # Check for unanimous forward
    if len(forward_votes) >= 3 and len(forward_counts) == 1:
        return {
            "decision": forward_votes[0],
            "forward_votes": forward_votes,
            "reverse_votes": [],
            "forward_errors": forward_errors,
            "reverse_errors": 0,
            "method": "unanimous_forward",
            "pro_reasoning": ""
        }
    
    # Stage 2: Reverse pass (for non-unanimous)
    reverse_prompt = build_reverse_prompt(row)
    reverse_tasks = [predict_single_gemini(reverse_prompt, semaphore) for _ in range(NUM_SAMPLES)]
    reverse_results = await asyncio.gather(*reverse_tasks)
    
    # Map reverse votes back to original labels
    reverse_votes_raw = [r for r in reverse_results if r is not None]
    reverse_votes = [map_reverse_vote(v) for v in reverse_votes_raw]
    reverse_errors = sum(1 for r in reverse_results if r is None)
    reverse_counts = Counter(reverse_votes)
    
    # Get majorities
    forward_majority = forward_counts.most_common(1)[0][0] if forward_votes else None
    reverse_majority = reverse_counts.most_common(1)[0][0] if reverse_votes else None
    
    # Stage 3: Compare majorities
    if forward_majority and reverse_majority:
        if forward_majority == reverse_majority:
            # Both directions agree → strong signal
            return {
                "decision": forward_majority,
                "forward_votes": forward_votes,
                "reverse_votes": reverse_votes,
                "forward_errors": forward_errors,
                "reverse_errors": reverse_errors,
                "method": "bidirectional_agree",
                "pro_reasoning": ""
            }
        else:
            # Conflict detected → use Gemini Pro
            pro_decision, pro_reasoning = await predict_with_pro(row, forward_votes, reverse_votes, semaphore)
            return {
                "decision": pro_decision,
                "forward_votes": forward_votes,
                "reverse_votes": reverse_votes,
                "forward_errors": forward_errors,
                "reverse_errors": reverse_errors,
                "method": "pro_tiebreaker",
                "pro_reasoning": pro_reasoning
            }
    
    # Fallback: combined majority
    all_votes = forward_votes + reverse_votes
    if all_votes:
        decision = Counter(all_votes).most_common(1)[0][0]
        return {
            "decision": decision,
            "forward_votes": forward_votes,
            "reverse_votes": reverse_votes,
            "forward_errors": forward_errors,
            "reverse_errors": reverse_errors,
            "method": "combined_majority",
            "pro_reasoning": ""
        }
    
    return {
        "decision": "ERROR",
        "forward_votes": [],
        "reverse_votes": [],
        "forward_errors": forward_errors,
        "reverse_errors": reverse_errors,
        "method": "all_failed",
        "pro_reasoning": ""
    }


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v12 (BIDIRECTIONAL)")
    print(f"Fast Model: {MODEL_FAST} (native Gemini API)")
    print(f"Pro Model: {MODEL_PRO} (for conflicts)")
    print(f"Strategy: Bidirectional self-consistency (k={NUM_SAMPLES} each)")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running bidirectional predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    tasks = [predict_bidirectional(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    predictions = [r["decision"] for r in results]
    forward_votes = [r["forward_votes"] for r in results]
    reverse_votes = [r["reverse_votes"] for r in results]
    methods = [r["method"] for r in results]
    pro_reasonings = [r["pro_reasoning"] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["forward_votes"] = [",".join(v) if v else "none" for v in forward_votes]
    df["reverse_votes"] = [",".join(v) if v else "none" for v in reverse_votes]
    df["method"] = methods
    df["pro_reasoning"] = pro_reasonings
    
    # Calculate accuracy
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) > 0 else 0
    total_errors = len(df) - len(valid_df)
    
    # Analyze by method
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS - v12 (BIDIRECTIONAL)")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*60}")
    
    for method in ["unanimous_forward", "bidirectional_agree", "pro_tiebreaker", "combined_majority"]:
        method_df = valid_df[valid_df["method"] == method]
        if len(method_df) > 0:
            method_correct = (method_df["predicted_text_a_is_closer"] == method_df["text_a_is_closer"]).sum()
            print(f"{method}: {len(method_df)} cases → {method_correct}/{len(method_df)} = {method_correct/len(method_df):.1%}")
    
    print(f"{'═'*60}")
    
    # Analyze vote distribution
    all_forward = [v for votes in forward_votes for v in votes]
    all_reverse = [v for votes in reverse_votes for v in votes]
    
    if all_forward:
        fwd_a = all_forward.count('A')
        fwd_b = all_forward.count('B')
        print(f"\n📊 Forward votes: A={fwd_a} ({fwd_a/len(all_forward):.1%}) B={fwd_b} ({fwd_b/len(all_forward):.1%})")
    
    if all_reverse:
        rev_a = all_reverse.count('A')
        rev_b = all_reverse.count('B')
        print(f"📊 Reverse votes (mapped): A={rev_a} ({rev_a/len(all_reverse):.1%}) B={rev_b} ({rev_b/len(all_reverse):.1%})")
    
    # Show sample Pro reasoning
    pro_cases = valid_df[valid_df["method"] == "pro_tiebreaker"]
    if len(pro_cases) > 0:
        sample_idx = pro_cases.index[0]
        print(f"\n💭 Sample Pro reasoning (example {sample_idx + 1}):")
        print(f"{'─'*60}")
        print(f"Forward: {df.iloc[sample_idx]['forward_votes']}")
        print(f"Reverse: {df.iloc[sample_idx]['reverse_votes']}")
        reasoning = df.iloc[sample_idx]['pro_reasoning']
        print(f"Reasoning: {reasoning[:500]}..." if len(reasoning) > 500 else f"Reasoning: {reasoning}")
        print(f"Decision: {df.iloc[sample_idx]['predicted_decision']}")
        print(f"Correct: {'A' if df.iloc[sample_idx]['text_a_is_closer'] else 'B'}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

