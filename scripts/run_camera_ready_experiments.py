#!/usr/bin/env python3
"""Manifested Gemini experiment runner for camera-ready CascadeMind analyses."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 1.0
PAPER_SYMBOLIC_WEIGHTS = {
    "Semantic": 0.08,
    "Lexical": 0.49,
    "StoryGrammar": 0.40,
    "EventChain": 0.01,
    "Tension": 0.02,
}
SYMBOLIC_WEIGHT_ORDER = ["Semantic", "Lexical", "StoryGrammar", "EventChain", "Tension"]
SYMBOLIC_ONLY_STRATEGIES = {"symbolic_only", "symbolic_everywhere"}

SUITES = {
    "smoke": ["single_vote", "self_consistency_k8", "cascade"],
    "balanced": [
        "single_vote",
        "self_consistency_k8",
        "majority_3calls",
        "cascade",
        "cascade_no_symbolic",
        "symbolic_only",
        "symbolic_everywhere",
    ],
    "pathway": ["cascade"],
    "symbolic": ["symbolic_only", "symbolic_everywhere", "cascade_no_symbolic"],
}


@dataclass
class ApiStats:
    calls: int = 0
    requested_candidates: int = 0
    parsed_votes: int = 0
    prompt_tokens: int = 0
    candidate_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0


def git_value(args: list[str], default: str = "unknown") -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else default


def is_dirty() -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_lfs_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(f"{path}: Git LFS pointer stub, not restored data")


def load_jsonl(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"data file not found: {path}")
    reject_lfs_pointer(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def row_id(row: dict[str, Any], index: int) -> Any:
    for key in ("id", "example_id", "pair_id", "uid"):
        if key in row:
            return row[key]
    return index


def gold_label(row: dict[str, Any]) -> str | None:
    for key in ("label", "gold", "answer", "correct_answer", "decision"):
        value = row.get(key)
        if isinstance(value, str) and value.upper() in {"A", "B"}:
            return value.upper()
    if "text_a_is_closer" in row:
        return "A" if bool(row["text_a_is_closer"]) else "B"
    if "story_a_is_closer" in row:
        return "A" if bool(row["story_a_is_closer"]) else "B"
    return None


def story_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    anchor = row.get("anchor_text") or row.get("anchor") or row.get("text_anchor")
    text_a = row.get("text_a") or row.get("story_a") or row.get("candidate_a")
    text_b = row.get("text_b") or row.get("story_b") or row.get("candidate_b")
    if not all(isinstance(value, str) and value.strip() for value in (anchor, text_a, text_b)):
        raise ValueError("row is missing anchor/text_a/text_b story fields")
    return anchor, text_a, text_b


def build_prompt(row: dict[str, Any]) -> str:
    anchor, text_a, text_b = story_fields(row)
    return f"""Which story (A or B) is more similar to the Anchor story?

Focus on narrative similarity across:
1. Abstract theme
2. Course of action
3. Outcomes

Anchor: {anchor}

Story A: {text_a}

Story B: {text_b}

Return JSON with a "decision" field set to either "A" or "B"."""


def parse_decision(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        decision = parsed.get("decision")
        if isinstance(decision, str) and decision.upper() in {"A", "B"}:
            return decision.upper()
    except json.JSONDecodeError:
        pass
    upper = stripped.upper()
    if '"DECISION"' in upper:
        if '"A"' in upper:
            return "A"
        if '"B"' in upper:
            return "B"
    if upper in {"A", "B"}:
        return upper
    return None


def response_texts(response: Any) -> list[str]:
    texts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content is not None else []
        candidate_text = "".join(getattr(part, "text", "") or "" for part in parts).strip()
        if candidate_text:
            texts.append(candidate_text)
    fallback = getattr(response, "text", None)
    if fallback and fallback not in texts:
        texts.append(fallback)
    return texts


def update_token_stats(stats: ApiStats, response: Any) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    stats.prompt_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
    stats.candidate_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)
    stats.total_tokens += int(getattr(usage, "total_token_count", 0) or 0)


def make_client(api_key: str) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("google-genai is not installed; run `pip install -r requirements.txt`.") from exc
    return genai.Client(api_key=api_key)


def response_schema() -> Any:
    try:
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("google-genai is not installed; run `pip install -r requirements.txt`.") from exc
    return types.Schema(
        type=types.Type.OBJECT,
        properties={"decision": types.Schema(type=types.Type.STRING, enum=["A", "B"])},
        required=["decision"],
    )


def generate_config(candidate_count: int, temperature: float) -> Any:
    from google.genai import types

    return types.GenerateContentConfig(
        candidate_count=candidate_count,
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=response_schema(),
    )


async def gemini_votes(
    row: dict[str, Any],
    *,
    client: Any,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
    candidate_count: int,
    stats: ApiStats,
    retries: int,
) -> list[str]:
    prompt = build_prompt(row)
    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=generate_config(candidate_count, temperature),
                )
                stats.calls += 1
                stats.requested_candidates += candidate_count
                update_token_stats(stats, response)
                votes = [vote for text in response_texts(response) if (vote := parse_decision(text))]
                stats.parsed_votes += len(votes)
                return votes
            except Exception:
                stats.errors += 1
                if attempt >= retries:
                    return []
                await asyncio.sleep(2**attempt)
    return []


def symbolic_weight_source() -> str:
    weights_path = ROOT / "artifacts" / "models" / "ensemble_weights.json"
    return str(weights_path) if weights_path.exists() else "paper_default_weights"


def load_symbolic_weights() -> list[float]:
    weights_path = ROOT / "artifacts" / "models" / "ensemble_weights.json"
    if weights_path.exists():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        names = payload.get("signal_names", [])
        weights = payload.get("weights", [])
        if names and weights and len(names) == len(weights):
            by_name = {str(name): float(weight) for name, weight in zip(names, weights)}
            return [by_name[name] for name in SYMBOLIC_WEIGHT_ORDER]
    return [PAPER_SYMBOLIC_WEIGHTS[name] for name in SYMBOLIC_WEIGHT_ORDER]


def needs_api(strategy: str) -> bool:
    return strategy not in SYMBOLIC_ONLY_STRATEGIES


def symbolic_decision(row: dict[str, Any]) -> tuple[str, dict[str, float]]:
    try:
        import train_ensemble
        import numpy as np
    except ImportError as exc:
        raise SystemExit("symbolic strategies require the dependencies in requirements.txt") from exc

    anchor, text_a, text_b = story_fields(row)
    signals_a, signals_b = train_ensemble.extract_all_signals(anchor, text_a, text_b)
    weights = np.array(load_symbolic_weights())
    score_a = float(signals_a @ weights)
    score_b = float(signals_b @ weights)
    return ("A" if score_a > score_b else "B"), {"symbolic_score_a": score_a, "symbolic_score_b": score_b}


def majority(votes: list[str]) -> tuple[str | None, Counter[str]]:
    counts = Counter(votes)
    if counts["A"] > counts["B"]:
        return "A", counts
    if counts["B"] > counts["A"]:
        return "B", counts
    return None, counts


async def predict_strategy(
    row: dict[str, Any],
    *,
    strategy: str,
    client: Any | None,
    semaphore: asyncio.Semaphore | None,
    model: str,
    temperature: float,
    stats: ApiStats,
    retries: int,
) -> dict[str, Any]:
    if strategy in {"symbolic_only", "symbolic_everywhere"}:
        prediction, extra = symbolic_decision(row)
        return {"prediction": prediction, "pathway": strategy, "votes": {}, **extra}

    if client is None or semaphore is None:
        raise RuntimeError(f"{strategy} requires GEMINI_API_KEY")

    if strategy == "single_vote":
        votes = await gemini_votes(
            row,
            client=client,
            semaphore=semaphore,
            model=model,
            temperature=temperature,
            candidate_count=1,
            stats=stats,
            retries=retries,
        )
        prediction = votes[0] if votes else "A"
        return {"prediction": prediction, "pathway": "single_vote", "votes": dict(Counter(votes))}

    if strategy == "self_consistency_k8":
        votes = await gemini_votes(
            row,
            client=client,
            semaphore=semaphore,
            model=model,
            temperature=temperature,
            candidate_count=8,
            stats=stats,
            retries=retries,
        )
        prediction, counts = majority(votes)
        if prediction is None:
            prediction, extra = symbolic_decision(row)
            return {"prediction": prediction, "pathway": "k8_tie_symbolic", "votes": dict(counts), **extra}
        return {"prediction": prediction, "pathway": "k8_majority", "votes": dict(counts)}

    if strategy == "majority_3calls":
        vote_batches = []
        for _ in range(3):
            vote_batches.extend(
                await gemini_votes(
                    row,
                    client=client,
                    semaphore=semaphore,
                    model=model,
                    temperature=temperature,
                    candidate_count=8,
                    stats=stats,
                    retries=retries,
                )
            )
        prediction, counts = majority(vote_batches)
        if prediction is None:
            prediction, extra = symbolic_decision(row)
            return {"prediction": prediction, "pathway": "majority3_tie_symbolic", "votes": dict(counts), **extra}
        return {"prediction": prediction, "pathway": "majority3", "votes": dict(counts)}

    if strategy in {"cascade", "cascade_no_symbolic"}:
        first_votes = await gemini_votes(
            row,
            client=client,
            semaphore=semaphore,
            model=model,
            temperature=temperature,
            candidate_count=8,
            stats=stats,
            retries=retries,
        )
        first_counts = Counter(first_votes)
        if first_counts and max(first_counts["A"], first_counts["B"]) >= 7:
            prediction = "A" if first_counts["A"] > first_counts["B"] else "B"
            return {"prediction": prediction, "pathway": "supermajority_8", "votes": dict(first_counts)}

        all_votes = list(first_votes)
        for _ in range(3):
            all_votes.extend(
                await gemini_votes(
                    row,
                    client=client,
                    semaphore=semaphore,
                    model=model,
                    temperature=temperature,
                    candidate_count=8,
                    stats=stats,
                    retries=retries,
                )
            )
        prediction, counts = majority(all_votes)
        if prediction is not None:
            return {"prediction": prediction, "pathway": "escalated_majority_32", "votes": dict(counts)}
        if strategy == "cascade_no_symbolic":
            return {"prediction": "A", "pathway": "tie_default_a_no_symbolic", "votes": dict(counts)}
        prediction, extra = symbolic_decision(row)
        return {"prediction": prediction, "pathway": "symbolic_tie_32", "votes": dict(counts), **extra}

    raise ValueError(f"unknown strategy: {strategy}")


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("gold") in {"A", "B"}]
    pred_counts = Counter(row["prediction"] for row in rows)
    result: dict[str, Any] = {
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "prediction_counts": dict(pred_counts),
        "prediction_rate_a": pred_counts["A"] / len(rows) if rows else 0.0,
        "prediction_rate_b": pred_counts["B"] / len(rows) if rows else 0.0,
    }
    if not labeled:
        return result

    correct = sum(1 for row in labeled if row["prediction"] == row["gold"])
    confusion = {
        "gold_A_pred_A": sum(1 for row in labeled if row["gold"] == "A" and row["prediction"] == "A"),
        "gold_A_pred_B": sum(1 for row in labeled if row["gold"] == "A" and row["prediction"] == "B"),
        "gold_B_pred_A": sum(1 for row in labeled if row["gold"] == "B" and row["prediction"] == "A"),
        "gold_B_pred_B": sum(1 for row in labeled if row["gold"] == "B" and row["prediction"] == "B"),
    }
    recall_a = confusion["gold_A_pred_A"] / max(1, confusion["gold_A_pred_A"] + confusion["gold_A_pred_B"])
    recall_b = confusion["gold_B_pred_B"] / max(1, confusion["gold_B_pred_A"] + confusion["gold_B_pred_B"])
    precision_a = confusion["gold_A_pred_A"] / max(1, confusion["gold_A_pred_A"] + confusion["gold_B_pred_A"])
    precision_b = confusion["gold_B_pred_B"] / max(1, confusion["gold_A_pred_B"] + confusion["gold_B_pred_B"])
    f1_a = 2 * precision_a * recall_a / max(1e-12, precision_a + recall_a)
    f1_b = 2 * precision_b * recall_b / max(1e-12, precision_b + recall_b)
    result.update(
        {
            "accuracy": correct / len(labeled),
            "correct": correct,
            "confusion": confusion,
            "recall_a": recall_a,
            "recall_b": recall_b,
            "precision_a": precision_a,
            "precision_b": precision_b,
            "macro_f1": (f1_a + f1_b) / 2,
            "balanced_accuracy": (recall_a + recall_b) / 2,
        }
    )
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def run_strategy(args: argparse.Namespace, rows: list[dict[str, Any]], strategy: str, out_dir: Path) -> dict[str, Any]:
    strategy_needs_api = needs_api(strategy)
    api_key = os.environ.get("GEMINI_API_KEY")
    if strategy_needs_api and not api_key:
        raise SystemExit("GEMINI_API_KEY is required for Gemini strategies. Use a fresh rotated key in the shell environment.")

    stats = ApiStats()
    client = make_client(api_key) if strategy_needs_api and api_key else None
    semaphore = asyncio.Semaphore(args.concurrency) if strategy_needs_api else None

    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            result = await predict_strategy(
                row,
                strategy=strategy,
                client=client,
                semaphore=semaphore,
                model=args.model,
                temperature=args.temperature,
                stats=stats,
                retries=args.retries,
            )
        except ValueError as exc:
            result = {"prediction": "A", "pathway": "row_error_default_a", "votes": {}, "error": str(exc)}
        gold = gold_label(row)
        record = {
            "index": index,
            "id": row_id(row, index),
            "strategy": strategy,
            "gold": gold,
            "prediction": result.pop("prediction"),
            "correct": None,
            **result,
        }
        if gold in {"A", "B"}:
            record["correct"] = record["prediction"] == gold
        predictions.append(record)

    strategy_dir = out_dir / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    with (strategy_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    strategy_metrics = metrics(predictions)
    write_json(strategy_dir / "metrics.json", strategy_metrics)
    write_json(strategy_dir / "api_stats.json", asdict(stats))
    return {"strategy": strategy, "metrics": strategy_metrics, "api_stats": asdict(stats)}


async def main_async(args: argparse.Namespace) -> int:
    data_path = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    rows = load_jsonl(data_path, args.max_examples)
    out_dir = Path(args.out) if args.out else ROOT / "artifacts" / "runs" / f"camera_ready_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    strategies = list(args.strategy or SUITES[args.suite])
    uses_gemini_api = any(needs_api(strategy) for strategy in strategies)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": is_dirty(),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "row_count": len(rows),
        "suite": args.suite,
        "strategies": strategies,
        "uses_gemini_api": uses_gemini_api,
        "model": args.model if uses_gemini_api else None,
        "temperature": args.temperature if uses_gemini_api else None,
        "concurrency": args.concurrency,
        "max_examples": args.max_examples,
        "symbolic_weight_order": SYMBOLIC_WEIGHT_ORDER,
        "symbolic_weights": load_symbolic_weights(),
        "symbolic_weight_source": symbolic_weight_source(),
        "notes": [
            "Post-hoc reruns do not change official SemEval standing.",
            "Predictions omit raw story text to avoid duplicating task data in artifacts.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    summaries = []
    for strategy in strategies:
        print(f"Running {strategy} on {len(rows)} rows...")
        summaries.append(await run_strategy(args, rows, strategy, out_dir))
    write_json(out_dir / "summary.json", summaries)
    print(f"Wrote camera-ready run artifacts to {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/dev_track_a.jsonl", help="JSONL data file")
    parser.add_argument("--suite", choices=sorted(SUITES), default="smoke", help="experiment suite")
    parser.add_argument("--strategy", action="append", choices=sorted({item for values in SUITES.values() for item in values}), help="run only this strategy; may be repeated")
    parser.add_argument("--max-examples", type=int, default=5, help="limit rows for smoke tests; use 0 for all rows")
    parser.add_argument("--out", default=None, help="output directory, default artifacts/runs/camera_ready_<timestamp>")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model ID")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.max_examples == 0:
        args.max_examples = None
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
