"""
SemEval 2026 Task 4 Subtask 1 - v3 Maximum Accuracy
Narrative Similarity with research-backed techniques:
- Official 3-dimension framework (Abstract Theme, Course of Action, Outcomes)
- Bidirectional evaluation for position bias mitigation
- Self-consistency with multiple samples
- Confidence-weighted voting
- Few-shot examples from annotation guidelines
- Optional verification for split votes

Uses Gemini 2.5 Flash for predictions.
"""

import os
import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from tqdm.asyncio import tqdm
from collections import defaultdict

# Configuration
API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"
DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v3_results.csv"
CONCURRENCY_LIMIT = 30
SAMPLES_PER_DIRECTION = 3  # 3 samples forward + 3 backward = 6 total
TEMPERATURE = 0.7

client = genai.Client(api_key=API_KEY)

# Schema for structured JSON output with confidence
response_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "theme_analysis": types.Schema(type=types.Type.STRING),
        "action_analysis": types.Schema(type=types.Type.STRING),
        "outcome_analysis": types.Schema(type=types.Type.STRING),
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"]),
        "confidence": types.Schema(type=types.Type.INTEGER)
    },
    required=["decision", "confidence"]
)

# System prompt with official 3-dimension framework
SYSTEM_PROMPT = """You are an expert at analyzing narrative similarity between stories.

OFFICIAL FRAMEWORK (from SemEval annotation guidelines):
Evaluate similarity on THREE dimensions:

1. ABSTRACT THEME: The defining constellation of problems, central ideas, and core motifs.
   - IGNORE: concrete setting, time period, character names
   - Focus: What is this story fundamentally ABOUT?

2. COURSE OF ACTION: Sequences of events, actions, conflicts, and turning points.
   - Focus on the ORDER in which events happen
   - Key turning points and how conflicts develop

3. OUTCOMES: Results at the END of the story.
   - Conflict resolution
   - Characters' fates
   - Moral lessons (if any)
   - NOT intermediate states that change later

WHAT TO IGNORE (per guidelines):
- Style of writing
- Concrete setting (including time period)
- Names of characters and locations
- Length of text
- Level of detail

Focus on CORE narrative aspects. Weight dimensions by intuitive importance for each specific comparison."""

# Few-shot examples from official annotation guidelines
FEW_SHOT_EXAMPLES = """Here are examples of correct narrative similarity analysis:

EXAMPLE 1:
Anchor: Anna loses her purse. She is terrified because there are important documents in it. She retraces her steps but cannot find it. Dan finds it and helpfully returns it to her.

Story A: Brian lost his backpack. He did not care too much, as only a water bottle was in it. After an hour of searching, he finally found it.

Story B: Alex loses his engagement ring while swimming. He freaks out, and after hours of diving for it, he still cannot find it.

Analysis:
- Abstract Theme: All three involve losing a valued item. A and Anchor both involve retrieval; B does not.
- Course of Action: A and Anchor both end with finding the item; B ends without finding it.
- Outcomes: In A (like Anchor), item is retrieved. In B, item remains lost.

Decision: A
Confidence: 9
Reasoning: A and Anchor share the complete narrative arc of losing and recovering an item. The third party finding it (as in Anchor) vs self-finding (as in A) is less important than B's completely different outcome.

---

EXAMPLE 2:
Anchor: Two travelers in medieval Europe are stopped by robbers. They defend themselves but one is captured. The other surrenders to stay with his friend, and the robbers take their belongings but let them live.

Story A: A group travels west by train in 1800s USA. Zombies attack. They fight off attacks. One member goes missing. They perform a daring rescue and escape together.

Story B: Two travelers make their way to the kingdom's capital. The roads are dangerous with tales of robbers, but sticking to main roads, they arrive safely.

Analysis:
- Abstract Theme: Anchor and A both involve travelers facing danger, one being captured/lost, and loyalty/rescue. B is just a safe journey.
- Course of Action: Anchor and A both have: travel → danger → capture → choice to help → resolution together. B lacks this.
- Outcomes: In Anchor and A, companions stick together through adversity. In B, nothing happens.

Decision: A
Confidence: 10
Reasoning: Despite different settings (medieval vs 1800s, robbers vs zombies), A shares the core narrative of loyalty and rescue. B lacks any conflict or similar story structure.

---

Now analyze the following:"""


def build_prompt(anchor: str, text_a: str, text_b: str, order: str = "AB") -> str:
    """Build the prompt with few-shot examples. Order determines A/B presentation."""
    if order == "AB":
        stories = f"""Anchor: {anchor}

Story A: {text_a}

Story B: {text_b}"""
    else:  # BA order - swap labels to maintain consistent output format
        stories = f"""Anchor: {anchor}

Story A: {text_b}

Story B: {text_a}"""
    
    return f"""{FEW_SHOT_EXAMPLES}

{stories}

Analyze each dimension for both stories compared to the Anchor.
Then decide which shares MORE narrative similarity with Anchor.

Return JSON with:
- theme_analysis: Brief analysis of abstract themes
- action_analysis: Brief analysis of course of action
- outcome_analysis: Brief analysis of outcomes
- decision: "A" or "B"
- confidence: 1-10 (how certain you are)"""


async def single_prediction(
    anchor: str, 
    text_a: str, 
    text_b: str,
    order: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """Make a single prediction with analysis."""
    prompt = build_prompt(anchor, text_a, text_b, order)
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=[
                        types.Content(role="user", parts=[types.Part(text=SYSTEM_PROMPT)]),
                        types.Content(role="model", parts=[types.Part(text="I understand. I will analyze narrative similarity using the official 3-dimension framework: Abstract Theme, Course of Action, and Outcomes. I will ignore surface features like setting, names, and writing style.")]),
                        types.Content(role="user", parts=[types.Part(text=prompt)])
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=TEMPERATURE
                    )
                )
                data = json.loads(response.text)
                decision = data.get("decision", "A")
                confidence = min(max(data.get("confidence", 5), 1), 10)
                
                # If order was BA, we need to flip the decision back
                if order == "BA":
                    decision = "B" if decision == "A" else "A"
                
                return {
                    "decision": decision,
                    "confidence": confidence,
                    "order": order
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        
        return {"decision": "A", "confidence": 1, "order": order}


async def verification_prediction(
    anchor: str,
    text_a: str, 
    text_b: str,
    previous_votes: list,
    semaphore: asyncio.Semaphore
) -> dict:
    """Run a verification pass when votes are split."""
    vote_summary = ", ".join([f"{v['decision']}(conf={v['confidence']})" for v in previous_votes])
    
    verification_prompt = f"""{SYSTEM_PROMPT}

Previous analysis produced split votes: {vote_summary}

Please carefully re-analyze:

Anchor: {anchor}

Story A: {text_a}

Story B: {text_b}

The votes were split. Please verify by carefully checking:
1. Which story shares more ABSTRACT THEMES with the Anchor?
2. Which story has more similar COURSE OF ACTION to the Anchor?
3. Which story has more similar OUTCOMES to the Anchor?

After careful verification, provide your final decision.

Return JSON with:
- theme_analysis: Your verified theme analysis
- action_analysis: Your verified action analysis  
- outcome_analysis: Your verified outcome analysis
- decision: "A" or "B"
- confidence: 1-10"""

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=verification_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.3  # Lower temperature for verification
                    )
                )
                data = json.loads(response.text)
                return {
                    "decision": data.get("decision", "A"),
                    "confidence": min(max(data.get("confidence", 5), 1), 10),
                    "order": "verification"
                }
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        
        return {"decision": "A", "confidence": 1, "order": "verification"}


def aggregate_votes(votes: list) -> tuple:
    """Aggregate votes with confidence weighting. Returns (decision, confidence, is_unanimous)."""
    if not votes:
        return ("A", 0, False)
    
    # Calculate confidence-weighted scores
    score_a = sum(v["confidence"] for v in votes if v["decision"] == "A")
    score_b = sum(v["confidence"] for v in votes if v["decision"] == "B")
    
    count_a = sum(1 for v in votes if v["decision"] == "A")
    count_b = sum(1 for v in votes if v["decision"] == "B")
    
    total_confidence = score_a + score_b
    
    if score_a > score_b:
        decision = "A"
        confidence = score_a / total_confidence if total_confidence > 0 else 0.5
    elif score_b > score_a:
        decision = "B"
        confidence = score_b / total_confidence if total_confidence > 0 else 0.5
    else:
        # Tie - use count
        decision = "A" if count_a >= count_b else "B"
        confidence = 0.5
    
    is_unanimous = (count_a == len(votes)) or (count_b == len(votes))
    is_split = (count_a == count_b) and (count_a > 0)
    
    return (decision, confidence, is_unanimous, is_split, votes)


async def evaluate_with_bias_mitigation(row: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Full evaluation pipeline:
    1. Run N samples with forward order (A|B)
    2. Run N samples with backward order (B|A) 
    3. Aggregate all votes with confidence weighting
    4. If split, run verification
    """
    anchor = row['anchor_text']
    text_a = row['text_a']
    text_b = row['text_b']
    
    # Create all prediction tasks
    forward_tasks = [
        single_prediction(anchor, text_a, text_b, "AB", semaphore)
        for _ in range(SAMPLES_PER_DIRECTION)
    ]
    backward_tasks = [
        single_prediction(anchor, text_a, text_b, "BA", semaphore)
        for _ in range(SAMPLES_PER_DIRECTION)
    ]
    
    # Run all samples in parallel
    all_results = await asyncio.gather(*forward_tasks, *backward_tasks)
    
    # Aggregate votes
    decision, confidence, is_unanimous, is_split, votes = aggregate_votes(all_results)
    
    total_calls = len(all_results)
    strategy = "unanimous" if is_unanimous else ("split" if is_split else "majority")
    
    # If split (3-3), run verification
    if is_split:
        verification_result = await verification_prediction(
            anchor, text_a, text_b, all_results, semaphore
        )
        # Add verification vote with higher weight
        verification_result["confidence"] = verification_result["confidence"] * 1.5
        all_votes = all_results + [verification_result]
        decision, confidence, _, _, votes = aggregate_votes(all_votes)
        total_calls += 1
        strategy = "verified"
    
    return {
        "decision": decision,
        "predicted_text_a_is_closer": decision == "A",
        "confidence": confidence,
        "strategy": strategy,
        "total_calls": total_calls,
        "votes": votes
    }


async def main():
    print(f"{'═'*70}")
    print(f"SemEval 2026 Task 4 Subtask 1 - v3 Maximum Accuracy")
    print(f"{'─'*70}")
    print(f"Model: {MODEL}")
    print(f"Techniques:")
    print(f"  • Official 3-dimension framework (Theme, Action, Outcomes)")
    print(f"  • Bidirectional evaluation (position bias mitigation)")
    print(f"  • Self-consistency ({SAMPLES_PER_DIRECTION}x2 = {SAMPLES_PER_DIRECTION*2} samples)")
    print(f"  • Confidence-weighted voting")
    print(f"  • Few-shot examples from annotation guidelines")
    print(f"  • Verification for split votes")
    print(f"{'═'*70}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"   Evaluating on {len(df)} examples from {DATA_PATH}")
    
    # Run predictions
    print(f"\n🧠 Running predictions...")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [evaluate_with_bias_mitigation(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    # Extract results
    df["predicted_decision"] = [r["decision"] for r in results]
    df["predicted_text_a_is_closer"] = [r["predicted_text_a_is_closer"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["strategy"] = [r["strategy"] for r in results]
    df["total_calls"] = [r["total_calls"] for r in results]
    df["votes"] = [str(r["votes"]) for r in results]
    
    # Calculate accuracy
    correct = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).sum()
    accuracy = correct / len(df)
    
    # Strategy breakdown
    strategies = df["strategy"].value_counts()
    total_calls = df["total_calls"].sum()
    
    # Accuracy by strategy
    print(f"\n{'═'*70}")
    print(f"📊 RESULTS")
    print(f"{'─'*70}")
    
    for strategy in strategies.index:
        subset = df[df["strategy"] == strategy]
        strat_acc = (subset["predicted_text_a_is_closer"] == subset["text_a_is_closer"]).mean()
        print(f"  {strategy.capitalize()}: {len(subset)} cases → {strat_acc:.1%} accuracy")
    
    print(f"{'─'*70}")
    print(f"📊 OVERALL ACCURACY: {accuracy:.1%} ({correct}/{len(df)})")
    print(f"📊 Total API calls: {total_calls} (~{total_calls/len(df):.1f} avg per example)")
    print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print(f"{'═'*70}")
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n📂 Results saved to: {OUTPUT_PATH}")
    
    return accuracy


if __name__ == "__main__":
    asyncio.run(main())

