"""
SemEval 2026 Task 4 Subtask 1 - v15 PRO MAX

Strategy:
1) Extract structured narrative signatures (theme, action sequence, conflict, outcome).
2) Score A vs B with a rubric in both A/B orders to reduce position bias.
3) If scores disagree or are too close, run an arbiter on the raw texts.

Goal: improve accuracy by forcing structure and reducing surface-level matching.
"""

import asyncio
import json
import os
import time
from collections import Counter

import pandas as pd
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

# Configuration
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

MODEL_SIGNATURE = os.getenv("MODEL_SIGNATURE", "gemini-2.5-flash")
MODEL_JUDGE = os.getenv("MODEL_JUDGE", "gemini-2.5-pro")
MODEL_ARBITER = os.getenv("MODEL_ARBITER", "gemini-2.5-pro")

DATA_PATH = "data/dev_track_a.jsonl"
OUTPUT_PATH = "data/v15_pro_max_results.csv"

CONCURRENCY_LIMIT = 20
TEMPERATURE_SIGNATURE = 0.2
TEMPERATURE_JUDGE = 0.0
TEMPERATURE_ARBITER = 0.0

# If average score margin is below this, use arbiter.
MIN_MARGIN = 0.6

DIMENSIONS = ["theme", "action", "conflict", "outcome"]
WEIGHTS = {
    "theme": 0.35,
    "action": 0.35,
    "conflict": 0.2,
    "outcome": 0.1,
}

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

signature_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "theme": types.Schema(type=types.Type.STRING),
        "action_sequence": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
        ),
        "conflict": types.Schema(type=types.Type.STRING),
        "outcome": types.Schema(type=types.Type.STRING),
    },
    required=["theme", "action_sequence", "conflict", "outcome"],
)

score_block_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "theme": types.Schema(type=types.Type.INTEGER),
        "action": types.Schema(type=types.Type.INTEGER),
        "conflict": types.Schema(type=types.Type.INTEGER),
        "outcome": types.Schema(type=types.Type.INTEGER),
    },
    required=["theme", "action", "conflict", "outcome"],
)

judge_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "scores": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "A": score_block_schema,
                "B": score_block_schema,
            },
            required=["A", "B"],
        ),
        "rationale": types.Schema(type=types.Type.STRING),
    },
    required=["scores"],
)

arbiter_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "decision": types.Schema(type=types.Type.STRING, enum=["A", "B"]),
        "confidence": types.Schema(type=types.Type.NUMBER),
        "rationale": types.Schema(type=types.Type.STRING),
    },
    required=["decision"],
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SIGNATURE_PROMPT = """Extract a narrative signature from the text.

Rules:
- Use generic roles (protagonist, rival, authority) instead of names.
- Remove specific places, dates, and surface details.
- Focus on narrative structure only.

Return JSON with:
theme: 1 short sentence on the core idea or motif.
action_sequence: 3-6 short steps describing the key events in order.
conflict: 1 short sentence on the primary obstacle or conflict type.
outcome: 1 short sentence on how it ends.

Text:
{text}
"""

JUDGE_PROMPT = """You are judging narrative similarity using structure, not surface details.

Compare each candidate to the anchor signature and score similarity on each dimension.

Scoring rubric (0-3):
3 = nearly the same narrative pattern
2 = clear overlap in the core pattern
1 = weak/partial overlap
0 = different

Do NOT use names, setting, time period, or genre as signals.
Focus on abstract theme, action sequence, conflict type, and outcome.

Anchor signature:
{anchor_sig}

Story A signature:
{a_sig}

Story B signature:
{b_sig}

Return JSON with scores for A and B for each dimension.
"""

ARBITER_PROMPT = """You are an expert judge of narrative similarity.

Only use these dimensions:
1) Abstract theme
2) Course of action (event sequence)
3) Outcomes

Ignore names, settings, time period, and genre. Decide which story is closer to the anchor.

Anchor:
{anchor}

Story A:
{text_a}

Story B:
{text_b}

Return JSON with decision ("A" or "B"), optional confidence (0 to 1), and a short rationale.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_score(value: object) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num < 0:
        return 0.0
    if num > 3:
        return 3.0
    return num


def _normalize_score_block(block: dict) -> dict:
    return {dim: _normalize_score(block.get(dim)) for dim in DIMENSIONS}


def _total_score(block: dict) -> float:
    return sum(block[dim] * WEIGHTS[dim] for dim in DIMENSIONS)


async def _call_gemini(
    prompt: str,
    schema: types.Schema,
    model: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    ),
                )
                return json.loads(response.text)
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return None


signature_cache: dict[str, dict] = {}


async def get_signature(text: str, semaphore: asyncio.Semaphore) -> dict | None:
    if text in signature_cache:
        return signature_cache[text]
    prompt = SIGNATURE_PROMPT.format(text=text)
    data = await _call_gemini(
        prompt=prompt,
        schema=signature_schema,
        model=MODEL_SIGNATURE,
        temperature=TEMPERATURE_SIGNATURE,
        semaphore=semaphore,
    )
    if data:
        signature_cache[text] = data
    return data


async def judge_scores(
    anchor_sig: dict,
    a_sig: dict,
    b_sig: dict,
    semaphore: asyncio.Semaphore,
    swap: bool = False,
) -> dict | None:
    if swap:
        a_sig, b_sig = b_sig, a_sig
    prompt = JUDGE_PROMPT.format(
        anchor_sig=json.dumps(anchor_sig, ensure_ascii=True),
        a_sig=json.dumps(a_sig, ensure_ascii=True),
        b_sig=json.dumps(b_sig, ensure_ascii=True),
    )
    data = await _call_gemini(
        prompt=prompt,
        schema=judge_schema,
        model=MODEL_JUDGE,
        temperature=TEMPERATURE_JUDGE,
        semaphore=semaphore,
    )
    if not data or "scores" not in data:
        return None
    scores = data["scores"]
    if swap:
        scores = {"A": scores.get("B", {}), "B": scores.get("A", {})}
    return scores


async def arbiter_decide(
    anchor: str,
    text_a: str,
    text_b: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    prompt = ARBITER_PROMPT.format(anchor=anchor, text_a=text_a, text_b=text_b)
    return await _call_gemini(
        prompt=prompt,
        schema=arbiter_schema,
        model=MODEL_ARBITER,
        temperature=TEMPERATURE_ARBITER,
        semaphore=semaphore,
    )


async def predict_row(row: dict, semaphore: asyncio.Semaphore) -> dict:
    anchor_text = row["anchor_text"]
    text_a = row["text_a"]
    text_b = row["text_b"]

    sig_anchor, sig_a, sig_b = await asyncio.gather(
        get_signature(anchor_text, semaphore),
        get_signature(text_a, semaphore),
        get_signature(text_b, semaphore),
    )

    if not sig_anchor or not sig_a or not sig_b:
        arb = await arbiter_decide(anchor_text, text_a, text_b, semaphore)
        decision = arb.get("decision") if arb else "A"
        return {
            "decision": decision,
            "method": "arbiter_only",
            "margin": 0.0,
            "scores_forward": None,
            "scores_reverse": None,
            "scores_avg": None,
            "arbiter_confidence": arb.get("confidence") if arb else None,
        }

    scores_forward, scores_reverse = await asyncio.gather(
        judge_scores(sig_anchor, sig_a, sig_b, semaphore, swap=False),
        judge_scores(sig_anchor, sig_a, sig_b, semaphore, swap=True),
    )

    if not scores_forward and not scores_reverse:
        arb = await arbiter_decide(anchor_text, text_a, text_b, semaphore)
        decision = arb.get("decision") if arb else "A"
        return {
            "decision": decision,
            "method": "arbiter_only",
            "margin": 0.0,
            "scores_forward": None,
            "scores_reverse": None,
            "scores_avg": None,
            "arbiter_confidence": arb.get("confidence") if arb else None,
        }

    def normalize_scores(scores: dict | None) -> dict:
        if not scores:
            return {
                "A": {dim: 0.0 for dim in DIMENSIONS},
                "B": {dim: 0.0 for dim in DIMENSIONS},
            }
        return {
            "A": _normalize_score_block(scores.get("A", {})),
            "B": _normalize_score_block(scores.get("B", {})),
        }

    scores_forward_n = normalize_scores(scores_forward)
    scores_reverse_n = normalize_scores(scores_reverse)

    avg_scores = {
        "A": {
            dim: (scores_forward_n["A"][dim] + scores_reverse_n["A"][dim]) / 2.0
            for dim in DIMENSIONS
        },
        "B": {
            dim: (scores_forward_n["B"][dim] + scores_reverse_n["B"][dim]) / 2.0
            for dim in DIMENSIONS
        },
    }

    total_a = _total_score(avg_scores["A"])
    total_b = _total_score(avg_scores["B"])
    margin = abs(total_a - total_b)

    forward_total_a = _total_score(scores_forward_n["A"])
    forward_total_b = _total_score(scores_forward_n["B"])
    reverse_total_a = _total_score(scores_reverse_n["A"])
    reverse_total_b = _total_score(scores_reverse_n["B"])

    forward_decision = "A" if forward_total_a >= forward_total_b else "B"
    reverse_decision = "A" if reverse_total_a >= reverse_total_b else "B"

    decision = "A" if total_a >= total_b else "B"
    method = "signature_scores"

    if margin < MIN_MARGIN or forward_decision != reverse_decision:
        arb = await arbiter_decide(anchor_text, text_a, text_b, semaphore)
        if arb and arb.get("decision") in ["A", "B"]:
            decision = arb["decision"]
            method = "arbiter"
            arb_conf = arb.get("confidence")
        else:
            arb_conf = None
    else:
        arb_conf = None

    return {
        "decision": decision,
        "method": method,
        "margin": margin,
        "scores_forward": scores_forward_n,
        "scores_reverse": scores_reverse_n,
        "scores_avg": avg_scores,
        "arbiter_confidence": arb_conf,
    }


async def main() -> float:
    print("=" * 68)
    print("SemEval 2026 Task 4 Subtask 1 - v15 PRO MAX")
    print(f"Signature model: {MODEL_SIGNATURE}")
    print(f"Judge model: {MODEL_JUDGE}")
    print(f"Arbiter model: {MODEL_ARBITER}")
    print("=" * 68)

    print("\nLoading dataset...")
    df = pd.read_json(DATA_PATH, lines=True).head(200)
    print(f"Evaluating on {len(df)} examples")

    start_time = time.time()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tasks = [predict_row(row, semaphore) for _, row in df.iterrows()]
    results = await tqdm.gather(*tasks)

    elapsed = time.time() - start_time

    df["predicted_decision"] = [r["decision"] for r in results]
    df["predicted_text_a_is_closer"] = [r["decision"] == "A" for r in results]
    df["method"] = [r["method"] for r in results]
    df["margin"] = [r["margin"] for r in results]
    df["scores_forward"] = [
        json.dumps(r["scores_forward"], ensure_ascii=True) for r in results
    ]
    df["scores_reverse"] = [
        json.dumps(r["scores_reverse"], ensure_ascii=True) for r in results
    ]
    df["scores_avg"] = [json.dumps(r["scores_avg"], ensure_ascii=True) for r in results]
    df["arbiter_confidence"] = [r["arbiter_confidence"] for r in results]

    valid_df = df[df["predicted_decision"].isin(["A", "B"])]
    correct = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).sum()
    accuracy = correct / len(valid_df) if len(valid_df) else 0.0

    print("\n" + "=" * 68)
    print("RESULTS - v15 PRO MAX")
    print("-" * 68)
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(valid_df)})")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(df):.2f}s per example)")
    print("-" * 68)

    method_counts = Counter(df["method"])
    for method, count in method_counts.most_common():
        subset = df[df["method"] == method]
        sub_acc = (subset["predicted_text_a_is_closer"] == subset["text_a_is_closer"]).mean()
        print(f"{method}: {count} cases -> {sub_acc:.1%} accuracy")

    print("=" * 68)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    return accuracy


if __name__ == "__main__":
    asyncio.run(main())
