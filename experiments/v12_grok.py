"""
SemEval 2026 Task 4 Subtask 1 - v12 Grok
Bidirectional Self-Consistency using Grok 4.1 Fast for everything

Strategy:
1. Run k=5 samples with Grok 4.1 Fast (forward pass)
2. If unanimous → use that answer
3. If split → run k=5 samples with A/B swapped (reverse pass)
4. If forward and reverse majorities agree → use that
5. If conflict → use combined majority
"""

import os
import pandas as pd
import json
import asyncio
import time
import httpx
from tqdm.asyncio import tqdm
from collections import Counter

# Configuration
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = "x-ai/grok-4.1-fast"  # Grok 4.1 Fast via OpenRouter
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v12_grok_results.csv"
CONCURRENCY_LIMIT = 200  # Maximum parallelization
NUM_SAMPLES = 5
TEMPERATURE = 0.7

def build_prompt(row: dict) -> str:
    """Build the forward comparison prompt (A first, B second)."""
    return f"""Determine which story (A or B) is more NARRATIVELY similar to the Anchor.

NARRATIVE SIMILARITY focuses on THREE aspects:
1. ABSTRACT THEME: The defining problems, central ideas, and core motifs (NOT the concrete setting or time period)
2. COURSE OF ACTION: The sequence of events, conflicts, and turning points - and their ORDER
3. OUTCOMES: The results at the END - conflict resolution, characters' fates, moral lessons

IGNORE these factors (they do NOT count):
- Writing style or prose quality
- Concrete setting or time period (medieval vs modern, war vs peace)
- Character and location names
- Text length or level of detail

Focus on: What is the CORE story about? What HAPPENS? How does it END?

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Which story shares more narrative similarity with the Anchor?
Return JSON: {{"decision": "A"}} or {{"decision": "B"}}"""


def build_reverse_prompt(row: dict) -> str:
    """Build the reverse comparison prompt (B first, A second)."""
    return f"""Determine which story (A or B) is more NARRATIVELY similar to the Anchor.

NARRATIVE SIMILARITY focuses on THREE aspects:
1. ABSTRACT THEME: The defining problems, central ideas, and core motifs (NOT the concrete setting or time period)
2. COURSE OF ACTION: The sequence of events, conflicts, and turning points - and their ORDER
3. OUTCOMES: The results at the END - conflict resolution, characters' fates, moral lessons

IGNORE these factors (they do NOT count):
- Writing style or prose quality
- Concrete setting or time period (medieval vs modern, war vs peace)
- Character and location names
- Text length or level of detail

Focus on: What is the CORE story about? What HAPPENS? How does it END?

Anchor: {row['anchor_text']}

Story A: {row['text_b']}

Story B: {row['text_a']}

Which story shares more narrative similarity with the Anchor?
Return JSON: {{"decision": "A"}} or {{"decision": "B"}}"""


async def predict_single_grok(prompt: str, http_client: httpx.AsyncClient, 
                               semaphore: asyncio.Semaphore) -> str | None:
    """Make a single prediction with Grok via OpenRouter. Returns None on error."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await http_client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=60.0
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2))
                    await asyncio.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                data = json.loads(content)
                decision = data.get("decision")
                
                if decision in ["A", "B"]:
                    return decision
                return None
                
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return None


def map_reverse_vote(vote: str) -> str:
    """Map reverse vote back to original labels."""
    return "B" if vote == "A" else "A"


async def predict_bidirectional(row: dict, http_client: httpx.AsyncClient, 
                                 semaphore: asyncio.Semaphore) -> dict:
    """Bidirectional prediction using Grok 4.1 Fast."""
    
    # Stage 1: Forward pass
    forward_prompt = build_prompt(row)
    forward_tasks = [predict_single_grok(forward_prompt, http_client, semaphore) 
                     for _ in range(NUM_SAMPLES)]
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
        }
    
    # Stage 2: Reverse pass (for non-unanimous)
    reverse_prompt = build_reverse_prompt(row)
    reverse_tasks = [predict_single_grok(reverse_prompt, http_client, semaphore) 
                     for _ in range(NUM_SAMPLES)]
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
            }
        else:
            # Conflict → use combined majority
            all_votes = forward_votes + reverse_votes
            decision = Counter(all_votes).most_common(1)[0][0]
            return {
                "decision": decision,
                "forward_votes": forward_votes,
                "reverse_votes": reverse_votes,
                "forward_errors": forward_errors,
                "reverse_errors": reverse_errors,
                "method": "conflict_combined",
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
        }
    
    return {
        "decision": "ERROR",
        "forward_votes": [],
        "reverse_votes": [],
        "forward_errors": forward_errors,
        "reverse_errors": reverse_errors,
        "method": "all_failed",
    }


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v12 (GROK 4.1 FAST)")
    print(f"Model: {MODEL} (OpenRouter)")
    print(f"Strategy: Bidirectional k={NUM_SAMPLES}, heavy parallelization")
    print(f"Concurrency: {CONCURRENCY_LIMIT}")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running bidirectional predictions with Grok 4.1 Fast...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with httpx.AsyncClient() as http_client:
        tasks = [predict_bidirectional(row, http_client, semaphore) 
                 for _, row in df.iterrows()]
        results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    predictions = [r["decision"] for r in results]
    forward_votes = [r["forward_votes"] for r in results]
    reverse_votes = [r["reverse_votes"] for r in results]
    methods = [r["method"] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["forward_votes"] = [",".join(v) if v else "none" for v in forward_votes]
    df["reverse_votes"] = [",".join(v) if v else "none" for v in reverse_votes]
    df["method"] = methods
    
    # Calculate accuracy
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) > 0 else 0
    total_errors = len(df) - len(valid_df)
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS - v12 (GROK 4.1 FAST)")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*60}")
    
    for method in ["unanimous_forward", "bidirectional_agree", "conflict_combined", "combined_majority"]:
        method_df = valid_df[valid_df["method"] == method]
        if len(method_df) > 0:
            method_correct = (method_df["predicted_text_a_is_closer"] == method_df["text_a_is_closer"]).sum()
            print(f"{method}: {len(method_df)} cases → {method_correct}/{len(method_df)} = {method_correct/len(method_df):.1%}")
    
    print(f"{'═'*60}")
    
    # Vote distribution
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
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
