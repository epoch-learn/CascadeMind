"""
SemEval 2026 Task 4 Subtask 1 - v11
Self-consistency with native Gemini 2.5 Flash (k=5),
route to GPT-5.2 Chat via Azure for split votes.

V11: STRONGER GUIDELINES to prevent surface-level matching errors.
"""

import os
import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
import httpx
from tqdm.asyncio import tqdm
from collections import Counter

# Configuration
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Azure OpenAI Configuration
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_MODEL = "gpt-5.2-chat"
AZURE_API_VERSION = "2025-04-01-preview"

MODEL_FAST = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v11_results.csv"
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

Answer (A or B):"""


def build_pro_prompt(row: dict, votes: list[str]) -> str:
    """MUCH STRONGER prompt for GPT-5.2 to prevent surface-level matching."""
    vote_counts = Counter(votes)
    return f"""You are an expert at identifying NARRATIVE SIMILARITY - matching stories by their DEEP STRUCTURAL PATTERNS, not surface features.

This is a difficult case where initial analysis was split ({vote_counts['A']} votes for A, {vote_counts['B']} votes for B).

═══════════════════════════════════════════════════════════════════
CRITICAL INSTRUCTIONS - READ CAREFULLY BEFORE ANALYZING
═══════════════════════════════════════════════════════════════════

Narrative similarity is based on THREE things ONLY:

1. **ABSTRACT ACTION PATTERN** (most important!)
   - What SEQUENCE of actions does the protagonist take?
   - Example: "person becomes obsessed → commits wrongdoing to pursue obsession → faces consequences"
   - Example: "person receives call to adventure → trains → faces trials → confronts villain"
   - IGNORE who does it, where, when, or why in concrete terms

2. **CORE CONFLICT TYPE**
   - Is it person vs self? Person vs person? Person vs system?
   - What is the NATURE of the obstacle (temptation, enemy, fate, society)?

3. **OUTCOME PATTERN**
   - How does it END structurally? Triumph? Tragedy? Ironic reversal? Moral lesson?

═══════════════════════════════════════════════════════════════════
⛔ FORBIDDEN SIGNALS - DO NOT USE THESE FOR MATCHING ⛔
═══════════════════════════════════════════════════════════════════

DO NOT match based on:
- "Both involve family conflict" (too vague - what KIND of family conflict?)
- "Both have a patriarch" (surface similarity)
- "Both involve inheritance" (what matters is the PATTERN of events, not the object)
- "Both are set in a similar time/place"
- "Both involve war/romance/crime" (genre is not narrative pattern)
- Surface topic words like "family", "power", "love", "betrayal"

INSTEAD ask: Do the protagonists follow the SAME SEQUENCE of actions?

═══════════════════════════════════════════════════════════════════
EXAMPLE OF CORRECT REASONING
═══════════════════════════════════════════════════════════════════

If Anchor = "Man leaves home, becomes obsessed with woman, commits crimes to support her, falls into disgrace"

✅ CORRECT match: A story where someone becomes romantically obsessed → commits wrongdoing → faces downfall
   (even if it's about a woman, or not about crime specifically)

❌ WRONG match: A story about "family inheritance conflict" just because anchor mentioned family
   (the PATTERN is obsession→crime→downfall, not "family dispute")

═══════════════════════════════════════════════════════════════════
NOW ANALYZE
═══════════════════════════════════════════════════════════════════

Anchor: {row['anchor_text']}

Story A: {row['text_a']}

Story B: {row['text_b']}

STEP 1: What is the Anchor's ACTION SEQUENCE? (in 1-2 sentences, no character names)
STEP 2: What is Story A's ACTION SEQUENCE?
STEP 3: What is Story B's ACTION SEQUENCE?
STEP 4: Which sequence matches the Anchor better?

Respond with JSON: {{"anchor_pattern": "...", "a_pattern": "...", "b_pattern": "...", "reasoning": "which matches better and why", "decision": "A" or "B"}}"""


async def predict_single_gemini(prompt: str, semaphore: asyncio.Semaphore) -> str | None:
    """Make a single prediction with native Gemini API. Returns None on error."""
    async with semaphore:
        for attempt in range(3):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=MODEL_FAST,
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


async def predict_with_azure(row: dict, votes: list[str], http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """Get decision from GPT-5.2 Chat via Azure Responses API for split votes."""
    prompt = build_pro_prompt(row, votes)
    url = f"{AZURE_ENDPOINT}/openai/responses?api-version={AZURE_API_VERSION}"
    
    async with semaphore:
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
                    timeout=120.0
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    print(f"\n⚠️ Azure rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Check if response is complete
                if result.get("status") != "completed":
                    raise ValueError(f"Response incomplete: {result.get('incomplete_details')}")
                
                # Parse Azure Responses API format
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
                    raise ValueError("No output_text in response")
                
                data = json.loads(content)
                decision = data.get("decision")
                reasoning = f"Anchor: {data.get('anchor_pattern', '')}\nA: {data.get('a_pattern', '')}\nB: {data.get('b_pattern', '')}\n{data.get('reasoning', '')}"
                if decision in ["A", "B"]:
                    return decision, reasoning
                    
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        
        # Fallback to majority vote
        if votes:
            return Counter(votes).most_common(1)[0][0], "GPT-5.2 failed, using majority vote"
        return "ERROR", "Both models failed"


async def predict_cascade(row: dict, http_client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> dict:
    """
    Cascade prediction:
    1. Run k samples with Gemini 2.5 Flash
    2. If unanimous → use that answer
    3. If split → route to GPT-5.2 Chat
    """
    prompt = build_prompt(row)
    
    # Stage 1: Self-consistency with Gemini
    tasks = [predict_single_gemini(prompt, semaphore) for _ in range(NUM_SAMPLES)]
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
    
    # Split votes or not enough votes - route to GPT-5.2 Chat
    pro_decision, pro_reasoning = await predict_with_azure(row, valid_votes, http_client, semaphore)
    return {
        "decision": pro_decision,
        "votes": valid_votes,
        "errors": errors,
        "routed_to_pro": True,
        "pro_reasoning": pro_reasoning
    }


async def main():
    print(f"{'═'*60}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v11 (STRONGER GUIDELINES)")
    print(f"Fast Model: {MODEL_FAST} (native Gemini API)")
    print(f"Pro Model: {AZURE_MODEL} (Azure Responses API)")
    print(f"Strategy: Self-consistency (k={NUM_SAMPLES}) → GPT-5.2 for splits")
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
    print(f"📊 RESULTS - v11 (STRONGER GUIDELINES)")
    print(f"{'─'*60}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    if total_errors > 0:
        print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'─'*60}")
    print(f"Unanimous (Gemini): {len(fast_df)} cases → {fast_correct}/{len(fast_df)} = {fast_correct/len(fast_df):.1%}" if len(fast_df) > 0 else "No unanimous")
    print(f"Routed to GPT-5.2: {len(routed_df)} cases → {routed_correct}/{len(routed_df)} = {routed_correct/len(routed_df):.1%}" if len(routed_df) > 0 else "No routing needed")
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
        print(f"\n💭 Sample GPT-5.2 reasoning (example {sample_idx + 1}):")
        print(f"{'─'*60}")
        print(f"Votes: {df.iloc[sample_idx]['votes']}")
        reasoning = df.iloc[sample_idx]['pro_reasoning']
        print(f"Reasoning: {reasoning[:800]}..." if len(reasoning) > 800 else f"Reasoning: {reasoning}")
        print(f"Decision: {df.iloc[sample_idx]['predicted_decision']}")
        print(f"Correct: {'A' if df.iloc[sample_idx]['text_a_is_closer'] else 'B'}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

