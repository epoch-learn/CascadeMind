import os
import pandas as pd
from google import genai
from google.genai import types
import json
import asyncio
import time
from tqdm.asyncio import tqdm
from collections import Counter
import nest_asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import numpy as np
from sentence_transformers import SentenceTransformer

# Load sentence transformer model for semantic similarity
print("Loading sentence-transformers model for ensemble tiebreaker...")
_embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensemble weights aligned with official guidelines (reduced lexical bias)
ENSEMBLE_WEIGHTS = np.array([0.25, 0.10, 0.35, 0.20, 0.10])

# --- 1. ENVIRONMENT SETUP ---
nest_asyncio.apply()

# Token tracking
total_input_tokens = 0
total_output_tokens = 0
total_api_calls = 0
token_lock = asyncio.Lock()

API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# Model selection - Gemini 3 Flash Preview (no candidate_count support)
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"

# Single shot - no voting, just 1 call per row
K_INITIAL = 1
K_ESCALATE = 0

# ============================================================================
# MULTI-SCALE NARRATIVE ANALYSIS ENSEMBLE (Tiebreaker)
# ============================================================================

def _cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def _safe_encode(text):
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


async def get_single_vote(prompt, semaphore):
    """Get a single vote from one API call."""
    global total_input_tokens, total_output_tokens, total_api_calls
    
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=1.0,  # High temp for diversity
                        safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        ]
                    )
                )
                async with token_lock:
                    total_api_calls += 1
                    if response.usage_metadata:
                        total_input_tokens += response.usage_metadata.prompt_token_count or 0
                        total_output_tokens += response.usage_metadata.candidates_token_count or 0
                
                if not response.candidates:
                    raise ValueError("No candidates")
                
                text = response.candidates[0].content.parts[0].text
                data = json.loads(text)
                decision = data.get('decision')
                if decision in ['A', 'B']:
                    return decision
                raise ValueError(f"Invalid decision: {decision}")
                
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return 'ERROR'


async def get_votes(row, semaphore, num_votes=8, anchor=None, text_a=None, text_b=None):
    """Get multiple votes via parallel individual API calls."""
    anc = anchor if anchor else row['anchor_text']
    txt_a = text_a if text_a else row['text_a']
    txt_b = text_b if text_b else row['text_b']

    prompt = f"""Determine which story (A or B) is more NARRATIVELY similar to the Anchor.

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

Anchor: {anc}

Story A: {txt_a}

Story B: {txt_b}

Which story shares more narrative similarity with the Anchor?
Return JSON: {{"decision": "A"}} or {{"decision": "B"}}"""

    # Make parallel calls for all votes
    tasks = [get_single_vote(prompt, semaphore) for _ in range(num_votes)]
    votes = await asyncio.gather(*tasks)
    
    # Filter out errors
    valid_votes = [v for v in votes if v in ['A', 'B']]
    if not valid_votes:
        return ['ERROR']
    return valid_votes


async def predict_with_cascade(row, semaphore):
    """Single shot prediction - 1 API call per row."""
    votes = await get_votes(row, semaphore, 1)
    
    if 'ERROR' in votes or len(votes) == 0:
        return ("ERROR", votes, "needs_patching", 1)
    
    decision = votes[0]
    return (decision == "A", votes, "single_shot", 1)


async def main():
    print(" Loading Dataset...")
    file_path = "/Users/seb/Desktop/test/data/dev_track_a.jsonl"
    df = pd.read_json(file_path, lines=True)
    test_df = df.copy()

    print(f" Running SINGLE-SHOT on {len(test_df)} rows...")
    print(f" Model: {MODEL_NAME}")
    start_time = time.time()

    # Concurrency for parallel voting
    semaphore = asyncio.Semaphore(20)
    
    tasks = [predict_with_cascade(row, semaphore) for _, row in test_df.iterrows()]
    results = await tqdm.gather(*tasks)

    test_df["predicted_text_a_is_closer"] = [r[0] for r in results]
    test_df["votes"] = [r[1] for r in results]
    test_df["strategy"] = [r[2] for r in results]
    test_df["num_api_calls"] = [r[3] for r in results]

    elapsed = time.time() - start_time

    supermajority_df = test_df[test_df["strategy"].str.startswith("supermajority")]
    escalated = test_df[test_df["strategy"].str.startswith("escalated")]
    hybrid_df = test_df[test_df["strategy"] == "hybrid_tiebreaker"]
    
    has_labels = "text_a_is_closer" in test_df.columns
    
    error_df = test_df[test_df["predicted_text_a_is_closer"] == "ERROR"]
    valid_df = test_df[test_df["predicted_text_a_is_closer"] != "ERROR"]
    
    print(f"\n{'='*60}")
    print(f" CASCADE RESULTS (gemini-3-flash-preview):")
    print(f"{'-'*60}")
    print(f" Supermajority (Neural):   {len(supermajority_df)} cases")
    print(f" Escalated (Neural):       {len(escalated)} cases")
    print(f" Hybrid Tiebreaker:        {len(hybrid_df)} cases")
    
    if len(error_df) > 0:
        print(f" ERRORS (need patching):   {len(error_df)} cases")
        error_indices = error_df.index.tolist()
        print(f"   Row indices: {error_indices}")
    
    print(f"{'-'*60}")
    if has_labels and len(valid_df) > 0:
        accuracy = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).mean()
        print(f" ACCURACY: {accuracy:.1%} ({int(accuracy * len(valid_df))}/{len(valid_df)})")
    else:
        print(f" Completed: {len(valid_df)}/{len(test_df)} rows")
    print(f" API Calls: {total_api_calls}")
    print(f" Time: {elapsed:.1f}s")
    print(f"{'-'*60}")
    print(f" TOKENS USED:")
    print(f"   Input:  {total_input_tokens:,}")
    print(f"   Output: {total_output_tokens:,}")
    print(f"   Total:  {total_input_tokens + total_output_tokens:,}")
    print(f"{'='*60}")

    test_df.to_csv("cascade_3flash_results.csv", index=False)
    
    submission_records = []
    for idx, row in test_df.iterrows():
        submission_records.append({
            "text_a_is_closer": bool(row["predicted_text_a_is_closer"]) if row["predicted_text_a_is_closer"] != "ERROR" else False
        })
    
    with open("track_a.jsonl", "w") as f:
        for record in submission_records:
            f.write(json.dumps(record) + "\n")
    
    print(f"\n Submission file created: track_a.jsonl ({len(submission_records)} predictions)")
    
    import zipfile
    with zipfile.ZipFile("submission_3flash.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("track_a.jsonl", "track_a.jsonl")
    
    print(f" Zip file created: submission_3flash.zip")

if __name__ == "__main__":
    asyncio.run(main())

