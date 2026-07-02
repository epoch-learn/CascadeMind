"""
SemEval 2026 Task 4 - Narrative Similarity
Gemini 3 Flash with MAJ@8 voting and research-grade logging

Every API call is logged with:
- Timestamp, latency, tokens
- Full reasoning, confidence, decision
- Row index, vote index, round info
"""

import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from datetime import datetime
from tqdm.asyncio import tqdm
from collections import Counter
import nest_asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
import os

# Load sentence transformer model for semantic similarity
print("Loading sentence-transformers model for ensemble tiebreaker...")
_embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Optimized ensemble weights (trained on 1900 synthetic examples)
ENSEMBLE_WEIGHTS = np.array([0.0784, 0.4880, 0.4025, 0.0152, 0.0159])

# --- ENVIRONMENT SETUP ---
nest_asyncio.apply()

# Token tracking
total_input_tokens = 0
total_output_tokens = 0
total_api_calls = 0
token_lock = asyncio.Lock()

# Research-grade logging: stores every API call
api_call_log = []
log_lock = asyncio.Lock()

API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# Model selection - Gemini 2.5 Flash Preview
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# k values for self-consistency (using native candidate_count, max 8)
K_INITIAL = 8   # Initial round votes (max candidate_count)
K_ESCALATE = 8  # Escalation round votes

# Run ID for this experiment
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"run_{RUN_ID}"

# ============================================================================
# MULTI-SCALE NARRATIVE ANALYSIS ENSEMBLE (Novel Tiebreaker)
# ============================================================================

def _cosine_sim(a, b):
    """Cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def _safe_encode(text):
    """Safely encode text with padding error handling"""
    if text is None or not isinstance(text, str) or not text.strip():
        return np.zeros(384)
    text = text[:10000] if len(text) > 10000 else text
    try:
        return _embed_model.encode(text.strip(), show_progress_bar=False)
    except Exception:
        return np.zeros(384)

def _semantic_similarity(anchor, story):
    if not anchor or not story:
        return 0.0
    anchor_emb = _safe_encode(anchor)
    story_emb = _safe_encode(story)
    return _cosine_sim(anchor_emb, story_emb)

def _lexical_similarity(anchor, story):
    try:
        if not anchor or not story:
            return 0.0
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform([anchor, story])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        return 0.0

def _extract_story_grammar(text):
    if not text or not isinstance(text, str):
        return {'setting': '', 'conflict': '', 'rising': '', 'climax': '', 'resolution': ''}
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    n = len(sentences)
    if n == 0:
        return {'setting': '', 'conflict': '', 'rising': '', 'climax': '', 'resolution': ''}
    return {
        'setting': ' '.join(sentences[:max(1, n//5)]),
        'conflict': ' '.join(sentences[n//5:2*n//5]) if n > 1 else '',
        'rising': ' '.join(sentences[2*n//5:3*n//5]) if n > 2 else '',
        'climax': ' '.join(sentences[3*n//5:4*n//5]) if n > 3 else '',
        'resolution': ' '.join(sentences[4*n//5:]) if n > 4 else ''
    }

def _story_grammar_similarity(anchor, story):
    a_grammar = _extract_story_grammar(anchor)
    s_grammar = _extract_story_grammar(story)
    similarities = []
    for segment in ['setting', 'conflict', 'rising', 'climax', 'resolution']:
        a_text = a_grammar[segment]
        s_text = s_grammar[segment]
        if a_text and s_text:
            a_emb = _safe_encode(a_text)
            s_emb = _safe_encode(s_text)
            sim = _cosine_sim(a_emb, s_emb)
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0.0

def _extract_event_chain(text):
    if not text or not isinstance(text, str):
        return []
    action_words = {'discover', 'find', 'learn', 'realize', 'arrive', 'leave', 'escape',
        'fight', 'kill', 'save', 'rescue', 'destroy', 'create', 'build',
        'meet', 'marry', 'die', 'live', 'love', 'hate', 'fear', 'hope',
        'travel', 'return', 'begin', 'end', 'start', 'finish', 'win', 'lose',
        'reveal', 'hide', 'seek', 'search', 'chase', 'capture', 'release',
        'transform', 'change', 'become', 'remain', 'struggle', 'overcome',
        'betray', 'trust', 'deceive', 'forgive', 'revenge', 'sacrifice'}
    words = text.lower().split()
    events = [w for w in words if any(w.startswith(av) for av in action_words)]
    return events[:20]

def _lcs_length(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def _event_chain_similarity(anchor, story):
    a_events = _extract_event_chain(anchor)
    s_events = _extract_event_chain(story)
    if not a_events or not s_events:
        return 0.0
    lcs_len = _lcs_length(a_events, s_events)
    return 2 * lcs_len / (len(a_events) + len(s_events))

def _get_tension_curve(text, n_points=10):
    if not text or not isinstance(text, str):
        return np.zeros(n_points)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return np.zeros(n_points)
    tensions = []
    for sent in sentences:
        try:
            blob = TextBlob(sent)
            tension = abs(blob.sentiment.polarity) + blob.sentiment.subjectivity
            tensions.append(tension)
        except:
            tensions.append(0.0)
    if not tensions:
        return np.zeros(n_points)
    tensions = np.array(tensions)
    return np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(tensions)), tensions)

def _tension_similarity(anchor, story):
    a_tension = _get_tension_curve(anchor)
    s_tension = _get_tension_curve(story)
    if np.std(a_tension) < 0.01 or np.std(s_tension) < 0.01:
        return 1.0 - np.mean(np.abs(a_tension - s_tension))
    corr = np.corrcoef(a_tension, s_tension)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def symbolic_tiebreaker(row):
    """Multi-Scale Narrative Analysis Ensemble (Novel Hybrid Tiebreaker)"""
    anchor = row['anchor_text']
    text_a = row['text_a']
    text_b = row['text_b']

    signals_a = np.array([
        _semantic_similarity(anchor, text_a),
        _lexical_similarity(anchor, text_a),
        _story_grammar_similarity(anchor, text_a),
        _event_chain_similarity(anchor, text_a),
        _tension_similarity(anchor, text_a)
    ])

    signals_b = np.array([
        _semantic_similarity(anchor, text_b),
        _lexical_similarity(anchor, text_b),
        _story_grammar_similarity(anchor, text_b),
        _event_chain_similarity(anchor, text_b),
        _tension_similarity(anchor, text_b)
    ])

    score_a = np.dot(ENSEMBLE_WEIGHTS, signals_a)
    score_b = np.dot(ENSEMBLE_WEIGHTS, signals_b)

    return "A" if score_a > score_b else "B"


def build_prompt(anc, txt_a, txt_b):
    """Build the 3-shot narrative similarity prompt with reasoning and confidence."""
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


async def get_votes(row, row_idx, semaphore, num_candidates=8, round_name="initial"):
    """Get multiple votes in ONE API call using native candidate_count."""
    global total_input_tokens, total_output_tokens, total_api_calls, api_call_log

    anc = row['anchor_text']
    txt_a = row['text_a']
    txt_b = row['text_b']
    prompt = build_prompt(anc, txt_a, txt_b)

    call_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    timestamp = datetime.now().isoformat()

    log_entry = {
        "call_id": call_id,
        "run_id": RUN_ID,
        "timestamp": timestamp,
        "row_idx": row_idx,
        "round": round_name,
        "model": MODEL_NAME,
        "candidate_count": num_candidates,
        "status": "pending",
        "votes": [],
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": None,
        "error": None,
        "attempt": 0
    }

    async with semaphore:
        for attempt in range(3):
            log_entry["attempt"] = attempt + 1
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        candidate_count=num_candidates,  # 1-8 candidates per call
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

                async with token_lock:
                    total_api_calls += 1
                    if response.usage_metadata:
                        input_tokens = response.usage_metadata.prompt_token_count or 0
                        output_tokens = response.usage_metadata.candidates_token_count or 0
                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens
                        log_entry["input_tokens"] = input_tokens
                        log_entry["output_tokens"] = output_tokens

                # Extract votes from all candidates
                vote_results = []
                for idx, candidate in enumerate(response.candidates):
                    try:
                        text = candidate.content.parts[0].text
                        data = json.loads(text)
                        decision = data.get('decision')
                        confidence = data.get('confidence')
                        reasoning = data.get('reasoning')

                        # Normalize confidence
                        if confidence is not None and isinstance(confidence, (int, float)):
                            if confidence > 1.0:
                                confidence = confidence / 10.0
                            confidence = max(0.0, min(1.0, float(confidence)))

                        if decision in ['A', 'B']:
                            vote_results.append({
                                "decision": decision,
                                "confidence": confidence,
                                "reasoning": reasoning,
                                "status": "success"
                            })
                    except Exception as parse_err:
                        vote_results.append({
                            "decision": "ERROR",
                            "confidence": None,
                            "reasoning": None,
                            "status": "parse_error",
                            "error": str(parse_err)
                        })

                if vote_results:
                    log_entry["status"] = "success"
                    log_entry["votes"] = vote_results
                    async with log_lock:
                        api_call_log.append(log_entry)
                    return vote_results

                raise ValueError(f"No valid votes from {len(response.candidates)} candidates")

            except Exception as e:
                error_str = str(e)
                log_entry["error"] = error_str

                if "429" in error_str:
                    await asyncio.sleep(5 * (attempt + 1))
                elif attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        # All retries failed
        log_entry["status"] = "error"
        log_entry["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        async with log_lock:
            api_call_log.append(log_entry)

        return [{"decision": "ERROR", "confidence": None, "reasoning": None, "status": "error"}]


async def predict_with_cascade(row, row_idx, semaphore):
    """
    Phase 1: K_INITIAL votes (1 API call with candidate_count) - need supermajority (7/8) to decide.
    Phase 2 (If no supermajority): Add 3 batches of K_ESCALATE votes (3 more API calls).
    Returns full vote details for logging.
    """
    # --- Round 1: Get K_INITIAL votes in ONE API call ---
    vote_results = await get_votes(row, row_idx, semaphore, K_INITIAL, "round1")

    # Extract decisions
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

    # --- Check for Supermajority (7/8 = 87.5%+) ---
    top_count = counter.most_common(1)[0][1]
    if top_count >= 7:
        majority = counter.most_common(1)[0][0]
        return {
            "prediction": majority == "A",
            "votes": vote_results,
            "strategy": f"supermajority_{top_count}/{len(valid_votes)}",
            "num_rounds": 1
        }

    # --- ESCALATION: Split vote detected ---
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

    # --- Check for Perfect Tie - Use Symbolic Tiebreaker ---
    if len(counter) == 2 and counter['A'] == counter['B']:
        decision = symbolic_tiebreaker(row)
        return {
            "prediction": decision == "A",
            "votes": all_vote_results,
            "strategy": "hybrid_tiebreaker",
            "num_rounds": 4
        }

    majority = counter.most_common(1)[0][0]

    return {
        "prediction": majority == "A",
        "votes": all_vote_results,
        "strategy": f"escalated_{K_INITIAL}+{K_ESCALATE}x3",
        "num_rounds": 4
    }


async def main():
    global api_call_log

    # Create run directory
    os.makedirs(RUN_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f" SemEval 2026 Task 4 - Narrative Similarity")
    print(f" Run ID: {RUN_ID}")
    print(f" Output: {RUN_DIR}/")
    print(f"{'='*70}")

    print("\n Loading Dataset...")
    file_path = "data/test_track_a.jsonl"
    df = pd.read_json(file_path, lines=True)
    test_df = df.copy()  # Run on entire test set (400 examples)

    print(f" Running MAJ@{K_INITIAL} CASCADE on {len(test_df)} rows...")
    print(f" Model: {MODEL_NAME} (native candidate_count={K_INITIAL})")
    print(f" K_INITIAL={K_INITIAL}, K_ESCALATE={K_ESCALATE}")
    print(f" Logging: Every API call with reasoning, confidence, latency")
    start_time = time.time()

    # Concurrency limit for parallel voting
    semaphore = asyncio.Semaphore(16)

    tasks = [
        predict_with_cascade(row, idx, semaphore)
        for idx, row in test_df.iterrows()
    ]
    results = await tqdm.gather(*tasks)

    elapsed = time.time() - start_time

    # Extract results for DataFrame
    test_df["predicted_text_a_is_closer"] = [r["prediction"] for r in results]
    test_df["strategy"] = [r["strategy"] for r in results]
    test_df["num_rounds"] = [r["num_rounds"] for r in results]

    # Store vote summaries (decisions only for CSV)
    test_df["votes_summary"] = [
        ",".join([v["decision"] for v in r["votes"] if v["decision"] in ['A', 'B']])
        for r in results
    ]

    # Store confidence stats
    test_df["avg_confidence"] = [
        np.mean([v["confidence"] for v in r["votes"] if v.get("confidence") is not None])
        if any(v.get("confidence") is not None for v in r["votes"]) else None
        for r in results
    ]

    test_df["min_confidence"] = [
        min([v["confidence"] for v in r["votes"] if v.get("confidence") is not None], default=None)
        for r in results
    ]

    test_df["max_confidence"] = [
        max([v["confidence"] for v in r["votes"] if v.get("confidence") is not None], default=None)
        for r in results
    ]

    # Filter out errors
    valid_df = test_df[test_df["predicted_text_a_is_closer"] != "ERROR"]
    error_df = test_df[test_df["predicted_text_a_is_closer"] == "ERROR"]

    # Stats (no accuracy - test set has no labels)
    supermajority_df = valid_df[valid_df["strategy"].str.startswith("supermajority")]
    escalated = valid_df[valid_df["strategy"].str.startswith("escalated")]
    hybrid_df = valid_df[valid_df["strategy"] == "hybrid_tiebreaker"]

    print(f"\n{'='*70}")
    print(f" RESULTS - Run {RUN_ID} (Official Test Set)")
    print(f"{'-'*70}")
    print(f" Supermajority (Neural):   {len(supermajority_df):3d} cases")
    print(f" Escalated (Neural):       {len(escalated):3d} cases")
    print(f" Hybrid Tiebreaker:        {len(hybrid_df):3d} cases")
    if len(error_df) > 0:
        print(f" ERRORS:                   {len(error_df):3d} cases")
    print(f"{'-'*70}")
    print(f" COMPLETED: {len(valid_df)}/{len(test_df)} examples")
    print(f" Total API Calls: {total_api_calls}")
    print(f" Time: {elapsed:.1f}s ({elapsed/len(test_df):.2f}s per example)")
    print(f"{'-'*70}")
    print(f" TOKENS USED:")
    print(f"   Input:  {total_input_tokens:,}")
    print(f"   Output: {total_output_tokens:,}")
    print(f"   Total:  {total_input_tokens + total_output_tokens:,}")
    print(f"{'='*70}")

    # === SAVE RESULTS ===

    # 1. Main results CSV
    results_csv = os.path.join(RUN_DIR, "results.csv")
    test_df.to_csv(results_csv, index=False)
    print(f"\n Results saved to: {results_csv}")

    # 2. Detailed API call log (JSONL - one line per API call)
    log_file = os.path.join(RUN_DIR, "api_calls.jsonl")
    with open(log_file, 'w') as f:
        for entry in api_call_log:
            f.write(json.dumps(entry) + '\n')
    print(f" API call log saved to: {log_file} ({len(api_call_log)} calls)")

    # 3. Full vote details per row (JSON)
    votes_file = os.path.join(RUN_DIR, "votes_detailed.json")
    votes_data = []
    for idx, r in enumerate(results):
        pred = r["prediction"]
        # Convert numpy bool to Python bool for JSON serialization
        if pred != "ERROR":
            pred_bool = bool(pred) if isinstance(pred, (bool, np.bool_)) else pred
        else:
            pred_bool = "ERROR"
        votes_data.append({
            "row_idx": idx,
            "prediction": pred_bool,
            "strategy": r["strategy"],
            "num_rounds": r["num_rounds"],
            "votes": r["votes"]
        })

    with open(votes_file, 'w') as f:
        json.dump(votes_data, f, indent=2)
    print(f" Detailed votes saved to: {votes_file}")

    # 3b. Submission file (one JSON per line)
    submission_file = os.path.join(RUN_DIR, "submission_track_a.jsonl")
    with open(submission_file, 'w') as f:
        for r in results:
            pred = r["prediction"]
            if pred == "ERROR":
                pred_bool = False  # Default to False for errors
            else:
                pred_bool = bool(pred) if isinstance(pred, (bool, np.bool_)) else pred
            f.write(json.dumps({"text_a_is_closer": pred_bool}) + '\n')
    print(f" Submission file saved to: {submission_file}")

    # 4. Summary statistics
    stats_file = os.path.join(RUN_DIR, "stats.json")
    stats = {
        "run_id": RUN_ID,
        "run_dir": RUN_DIR,
        "model": MODEL_NAME,
        "k_initial": K_INITIAL,
        "k_escalate": K_ESCALATE,
        "total_examples": int(len(test_df)),
        "valid_examples": int(len(valid_df)),
        "error_examples": int(len(error_df)),
        "supermajority_count": int(len(supermajority_df)),
        "escalated_count": int(len(escalated)),
        "hybrid_count": int(len(hybrid_df)),
        "total_api_calls": int(total_api_calls),
        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "elapsed_seconds": float(elapsed),
        "avg_confidence": float(test_df["avg_confidence"].mean()) if test_df["avg_confidence"].notna().any() else None
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f" Statistics saved to: {stats_file}")

    # 5. Copy of the prompt used (for reproducibility)
    prompt_file = os.path.join(RUN_DIR, "prompt_template.txt")
    with open(prompt_file, 'w') as f:
        f.write(build_prompt("{ANCHOR}", "{STORY_A}", "{STORY_B}"))
    print(f" Prompt template saved to: {prompt_file}")

    # 6. Errors file for patching (only rows/calls that need patching)
    errors_to_patch = [entry for entry in api_call_log if entry.get("needs_patch", False)]
    if errors_to_patch:
        errors_file = os.path.join(RUN_DIR, "errors_to_patch.jsonl")
        with open(errors_file, 'w') as f:
            for entry in errors_to_patch:
                f.write(json.dumps(entry) + '\n')
        print(f" Errors to patch saved to: {errors_file} ({len(errors_to_patch)} calls)")

        # Also create a summary of rows needing patches
        error_rows = set(entry["row_idx"] for entry in errors_to_patch)
        rows_file = os.path.join(RUN_DIR, "rows_needing_patch.json")
        with open(rows_file, 'w') as f:
            json.dump({
                "row_indices": sorted(list(error_rows)),
                "total_rows": len(error_rows),
                "total_failed_calls": len(errors_to_patch),
                "error_types": dict(Counter(entry.get("error_type", "unknown") for entry in errors_to_patch))
            }, f, indent=2)
        print(f" Rows needing patch: {rows_file} ({len(error_rows)} rows)")

    print(f"\n All outputs saved to: {RUN_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
