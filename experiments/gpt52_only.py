"""
SemEval 2026 Task 4 Subtask 1 - GPT-5.2 Chat Only
Test GPT-5.2 Chat (Azure) on all cases with rate limiting.
Limits: 150k tokens/min, 1500 requests/min
"""

import os
import pandas as pd
import json
import asyncio
import time
import httpx
from tqdm.asyncio import tqdm

# Configuration
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_MODEL = "gpt-5.2-chat"
AZURE_API_VERSION = "2025-04-01-preview"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/gpt52_only_results.csv"

# Rate limiting - conservative to stay under limits
# 1500 req/min = 25 req/sec, but let's be safe with 10 req/sec
# 150k tokens/min = 2500 tokens/sec
CONCURRENCY_LIMIT = 5  # Max concurrent requests
MIN_DELAY_BETWEEN_REQUESTS = 0.1  # 100ms between requests = 10 req/sec max

# Track token usage
total_tokens_used = 0
request_count = 0
start_time = None

# Few-shot examples (minimal to save tokens)
FEW_SHOT_EXAMPLES = """Example 1:
Anchor: Young orphan discovers magical powers, goes to special school, faces dark villain.
Story A: Teenager inherits fortune, attends elite boarding school, navigates social hierarchies.
Story B: Child from humble origins destined for greatness, trains under mentors, confronts evil force.
Answer: B (hero's journey: discovery → training → confronting evil)

Example 2:
Anchor: Lovers from feuding families marry secretly, miscommunication leads to deaths, families reconcile.
Story A: Woman poisons husband after affair, remarries his partner.
Story B: Lovers from rival gangs escape together, caught in crossfire, deaths end gang war.
Answer: B (forbidden love → tragedy → reconciliation)
"""


def build_prompt(row: dict) -> str:
    """Build a concise prompt to save tokens."""
    return f"""{FEW_SHOT_EXAMPLES}
Which story is more narratively similar to the Anchor?

Focus on: Abstract Theme, Course of Action, Outcomes (NOT setting, names, style)

Anchor: {row['anchor_text'][:1500]}

Story A: {row['text_a'][:1500]}

Story B: {row['text_b'][:1500]}

Respond with JSON: {{"decision": "A" or "B"}}"""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars)"""
    return len(text) // 4


async def predict_single(row: dict, http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore, rate_limiter: asyncio.Lock) -> dict:
    """Make a single prediction with rate limiting using Responses API."""
    global total_tokens_used, request_count
    
    prompt = build_prompt(row)
    url = f"{AZURE_ENDPOINT}/openai/responses?api-version={AZURE_API_VERSION}"
    
    # Estimate input tokens
    input_tokens = estimate_tokens(prompt)
    
    last_error = None
    async with semaphore:
        # Rate limiting
        async with rate_limiter:
            await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)
        
        for attempt in range(3):
            try:
                response = await http_client.post(
                    url,
                    headers={
                        "api-key": AZURE_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": AZURE_MODEL,
                        "input": prompt,
                        "max_output_tokens": 3000,
                        "text": {"format": {"type": "json_object"}}
                    },
                    timeout=60.0
                )
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"\n⚠️ Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Track tokens
                usage = result.get("usage", {})
                tokens_used = usage.get("total_tokens", input_tokens + 20)
                total_tokens_used += tokens_used
                request_count += 1
                
                # Responses API format - find the message in output
                content = ""
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                content = c.get("text", "")
                                break
                        if content:
                            break
                
                if not content:
                    raise ValueError(f"No output_text in response: {result.get('output', [])[:200]}")
                
                data = json.loads(content)
                decision = data.get("decision")
                if decision in ["A", "B"]:
                    return {"decision": decision, "error": None, "tokens": tokens_used}
                    
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        
        return {"decision": "ERROR", "error": last_error, "tokens": 0}


async def main():
    global total_tokens_used, request_count, start_time
    
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 - GPT-5.2 Chat Only")
    print(f"Model: {AZURE_MODEL} (Azure)")
    print(f"Rate limits: 150k tokens/min, 1500 req/min")
    print(f"Concurrency: {CONCURRENCY_LIMIT}, Delay: {MIN_DELAY_BETWEEN_REQUESTS}s")
    print(f"{'═'*60}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples")
    
    # Estimate total tokens
    total_input_tokens = sum(estimate_tokens(build_prompt(row)) for _, row in df.iterrows())
    print(f"   Estimated input tokens: ~{total_input_tokens:,}")
    
    # Run predictions
    print(f"\n🧠 Running predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    rate_limiter = asyncio.Lock()
    
    async with httpx.AsyncClient() as http_client:
        tasks = [predict_single(row, http_client, semaphore, rate_limiter) for _, row in df.iterrows()]
        results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    predictions = [r["decision"] for r in results]
    errors = [r["error"] for r in results]
    tokens = [r["tokens"] for r in results]
    
    # Store results
    df["predicted_decision"] = predictions
    df["predicted_text_a_is_closer"] = [p == "A" for p in predictions]
    df["error"] = errors
    
    # Calculate accuracy (excluding errors)
    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    error_count = len(df) - len(valid_df)
    
    if len(valid_df) > 0:
        correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
        accuracy = correct / len(valid_df)
    else:
        correct = 0
        accuracy = 0
    
    # Results
    print(f"\n{'═'*60}")
    print(f"📊 RESULTS - GPT-5.2 Chat Only")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if error_count > 0:
        print(f"Errors: {error_count}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*60}")
    print(f"Total tokens used: {total_tokens_used:,}")
    print(f"Requests made: {request_count}")
    print(f"Tokens/min: {total_tokens_used/(elapsed/60):,.0f}")
    print(f"Requests/min: {request_count/(elapsed/60):,.0f}")
    print(f"{'═'*60}")
    
    # Vote distribution
    a_count = predictions.count('A')
    b_count = predictions.count('B')
    print(f"\n📊 Decision distribution: A={a_count} ({a_count/len(df):.1%}) B={b_count} ({b_count/len(df):.1%})")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

