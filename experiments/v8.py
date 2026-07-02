"""
SemEval 2026 Task 4 Subtask 1 - v8
Self-consistency with Gemini 2.5 Flash (k=5) via OpenRouter, 
route to GPT-4.1 for split votes.

Strategy:
1. Run k=5 samples with Gemini 2.5 Flash (via OpenRouter)
2. If unanimous → use that answer
3. If split → route to GPT-4.1 for final decision
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
MODEL_FAST = "google/gemini-2.5-flash"  # Gemini 2.5 Flash via OpenRouter
MODEL_PRO = "openai/gpt-4.1"  # GPT-4.1 via OpenRouter for splits
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v8_results.csv"
CONCURRENCY_LIMIT = 50
NUM_SAMPLES = 5
TEMPERATURE = 0.7

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


def build_prompt(row: dict) -> str:
    """Build the comparison prompt."""
    return f"""{FEW_SHOT_EXAMPLES}
Now determine which story is more narratively similar to the Anchor.

Focus on NARRATIVE SIMILARITY:
1. Abstract Theme - core problems, central ideas, motifs (NOT concrete setting)
2. Course of Action - sequence of events, turning points, how conflicts develop
3. Outcomes - how the story ends, character fates, lessons

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Respond with JSON: {{"decision": "A" or "B"}}"""


def build_pro_prompt(row: dict, votes: list[str]) -> str:
    """Enhanced prompt for split votes."""
    vote_counts = Counter(votes)
    return f"""This is a difficult case where initial analysis was split ({vote_counts['A']} votes for A, {vote_counts['B']} votes for B).

Please carefully analyze which story is more NARRATIVELY similar to the Anchor.

CRITICAL: Focus ONLY on these three aspects (per official guidelines):
1. ABSTRACT THEME - The defining problems, central ideas, core motifs. IGNORE concrete setting, time period, names.
2. COURSE OF ACTION - The SEQUENCE of events, turning points, how conflicts develop and resolve.
3. OUTCOMES - How the story ENDS, character fates, lessons learned.

DO NOT consider: writing style, genre, setting details, character names, time period, format.

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

Respond with JSON: {{"reasoning": "your detailed analysis", "decision": "A" or "B"}}"""


async def call_openrouter(prompt: str, model: str, temperature: float, http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore, use_schema: bool = True) -> dict | None:
    """Call OpenRouter API. Returns parsed JSON or None on error."""
    
    # JSON schema for structured output
    decision_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "decision_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["A", "B"]
                    }
                },
                "required": ["decision"],
                "additionalProperties": False
            }
        }
    }
    
    reasoning_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "reasoning_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["A", "B"]
                    }
                },
                "required": ["reasoning", "decision"],
                "additionalProperties": False
            }
        }
    }
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await http_client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://semeval-task4.example.com",
                        "X-Title": "SemEval Task 4"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "response_format": decision_schema if use_schema else reasoning_schema
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return None


async def predict_single(prompt: str, http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> str | None:
    """Make a single prediction with Gemini via OpenRouter. Returns None on error."""
    data = await call_openrouter(prompt, MODEL_FAST, TEMPERATURE, http_client, semaphore)
    if data:
        decision = data.get("decision")
        if decision in ["A", "B"]:
            return decision
    return None


async def predict_with_pro(row: dict, votes: list[str], http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """Get decision from GPT-4.1 for split votes."""
    prompt = build_pro_prompt(row, votes)
    data = await call_openrouter(prompt, MODEL_PRO, 0.0, http_client, semaphore, use_schema=False)
    
    if data:
        decision = data.get("decision")
        if decision in ["A", "B"]:
            return decision, data.get("reasoning", "")
    
    # Fallback to majority vote
    if votes:
        return Counter(votes).most_common(1)[0][0], "Pro model failed, using majority vote"
    return "ERROR", "Both models failed"


async def predict_cascade(row: dict, http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> dict:
    """
    Cascade prediction:
    1. Run k samples with Gemini 2.5 Flash
    2. If unanimous → use that answer
    3. If split → route to GPT-4.1
    """
    prompt = build_prompt(row)
    
    # Stage 1: Self-consistency with Gemini
    tasks = [predict_single(prompt, http_client, semaphore) for _ in range(NUM_SAMPLES)]
    results = await asyncio.gather(*tasks)
    
    # Filter out errors
    valid_votes = [r for r in results if r is not None]
    errors = sum(1 for r in results if r is None)
    
    vote_counts = Counter(valid_votes)
    
    # Check for unanimous (all valid votes agree)
    if len(valid_votes) >= 3 and len(vote_counts) == 1:
        # Unanimous
        return {
            "decision": valid_votes[0],
            "votes": valid_votes,
            "errors": errors,
            "routed_to_pro": False,
            "pro_reasoning": ""
        }
    
    # Split votes or not enough votes - route to GPT-4.1
    pro_decision, pro_reasoning = await predict_with_pro(row, valid_votes, http_client, semaphore)
    return {
        "decision": pro_decision,
        "votes": valid_votes,
        "errors": errors,
        "routed_to_pro": True,
        "pro_reasoning": pro_reasoning
    }


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v8")
    print(f"Fast Model: {MODEL_FAST}")
    print(f"Pro Model: {MODEL_PRO}")
    print(f"Strategy: Self-consistency (k={NUM_SAMPLES}) → GPT-4.1 for splits")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running cascade predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with httpx.AsyncClient() as http_client:
        tasks = [predict_cascade(row, http_client, semaphore) for _, row in df.iterrows()]
        results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    predictions = [r["decision"] for r in results]
    all_votes = [r["votes"] for r in results]
    all_errors = [r["errors"] for r in results]
    routed_to_pro = [r["routed_to_pro"] for r in results]
    pro_reasoning = [r["pro_reasoning"] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["votes"] = [",".join(v) if v else "none" for v in all_votes]
    df["errors"] = all_errors
    df["routed_to_pro"] = routed_to_pro
    df["pro_reasoning"] = pro_reasoning
    
    # Calculate accuracy (excluding errors)
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) > 0 else 0
    total_errors = len(df) - len(valid_df)
    
    # Analyze by routing
    fast_df = valid_df[~valid_df["routed_to_pro"]]
    routed_df = valid_df[valid_df["routed_to_pro"]]
    
    fast_correct = (fast_df["predicted_text_a_is_closer"] == fast_df["text_a_is_closer"]).sum() if len(fast_df) > 0 else 0
    routed_correct = (routed_df["predicted_text_a_is_closer"] == routed_df["text_a_is_closer"]).sum() if len(routed_df) > 0 else 0
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*60}")
    print(f"Unanimous (Gemini): {len(fast_df)} cases → {fast_correct}/{len(fast_df)} = {fast_correct/len(fast_df):.1%}" if len(fast_df) > 0 else "No unanimous")
    print(f"Routed to GPT-4.1: {len(routed_df)} cases → {routed_correct}/{len(routed_df)} = {routed_correct/len(routed_df):.1%}" if len(routed_df) > 0 else "No routing needed")
    print(f"{'═'*60}")
    
    # Analyze vote distribution
    all_valid_votes = [v for votes in all_votes for v in votes]
    if all_valid_votes:
        vote_a_count = all_valid_votes.count('A')
        vote_b_count = all_valid_votes.count('B')
        total_votes = len(all_valid_votes)
        print(f"\n📊 Vote distribution: A={vote_a_count} ({vote_a_count/total_votes:.1%}) B={vote_b_count} ({vote_b_count/total_votes:.1%})")
    
    # Show sample pro reasoning
    if len(routed_df) > 0:
        sample_idx = routed_df.index[0]
        print(f"\n💭 Sample GPT-4.1 reasoning (example {sample_idx + 1}):")
        print(f"{'─'*60}")
        print(f"Votes: {df.iloc[sample_idx]['votes']}")
        reasoning = df.iloc[sample_idx]['pro_reasoning']
        print(f"Reasoning: {reasoning[:500]}..." if len(reasoning) > 500 else f"Reasoning: {reasoning}")
        print(f"Decision: {df.iloc[sample_idx]['predicted_decision']}")
        print(f"Correct: {'A' if df.iloc[sample_idx]['text_a_is_closer'] else 'B'}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
