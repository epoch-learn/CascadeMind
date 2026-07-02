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

# Optimized ensemble weights (trained on 1900 synthetic examples)
# Achieves 99.5% validation accuracy on synthetic narrative similarity task
ENSEMBLE_WEIGHTS = np.array([0.0784, 0.4880, 0.4025, 0.0152, 0.0159])  # Semantic, Lexical, StoryGrammar, EventChain, Tension

# --- 1. ENVIRONMENT SETUP ---
nest_asyncio.apply()

# Token tracking
total_input_tokens = 0
total_output_tokens = 0
total_api_calls = 0
token_lock = asyncio.Lock()

API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# Model selection
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# k values for self-consistency (using native candidateCount, max 8)
K_INITIAL = 8   # Initial round votes (max candidateCount)
K_ESCALATE = 8  # Escalation round votes

# ============================================================================
# MULTI-SCALE NARRATIVE ANALYSIS ENSEMBLE (Novel Tiebreaker)
# ============================================================================
# Combines 5 signals at different levels of narrative abstraction:
#   Level 4: Story Grammar (narrative structure segments)
#   Level 3: Semantic Embedding (meaning via MiniLM)
#   Level 2: Event Chain (plot action sequences)
#   Level 1: Lexical (TF-IDF word overlap)
#   Level 1: Tension Curve (sentiment dynamics)
# ============================================================================

def _cosine_sim(a, b):
    """Cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def _safe_encode(text):
    """Safely encode text with padding error handling"""
    if text is None or not isinstance(text, str) or not text.strip():
        return np.zeros(384)  # MiniLM embedding dimension
    text = text[:10000] if len(text) > 10000 else text
    try:
        return _embed_model.encode(text.strip(), show_progress_bar=False)
    except Exception:
        return np.zeros(384)

def _semantic_similarity(anchor, story):
    """Level 3: Semantic embedding similarity using MiniLM-L6-v2"""
    if not anchor or not story:
        return 0.0
    anchor_emb = _safe_encode(anchor)
    story_emb = _safe_encode(story)
    return _cosine_sim(anchor_emb, story_emb)

def _lexical_similarity(anchor, story):
    """Level 1: TF-IDF word overlap"""
    try:
        if not anchor or not story:
            return 0.0
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform([anchor, story])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        return 0.0

def _extract_story_grammar(text):
    """
    Extract story elements based on narrative theory (Propp, Campbell):
    Setting, Conflict, Rising Action, Climax, Resolution
    """
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
    """Level 4: Compare stories segment-by-segment (setting vs setting, climax vs climax)"""
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
    """Extract plot action verbs (simplified without spacy)"""
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
    """Longest common subsequence length"""
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
    """Level 2: Plot action sequence alignment"""
    a_events = _extract_event_chain(anchor)
    s_events = _extract_event_chain(story)
    if not a_events or not s_events:
        return 0.0
    lcs_len = _lcs_length(a_events, s_events)
    return 2 * lcs_len / (len(a_events) + len(s_events))

def _get_tension_curve(text, n_points=10):
    """Extract narrative tension curve (sentiment intensity + subjectivity)"""
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
    """Level 4: Compare tension curves using correlation"""
    a_tension = _get_tension_curve(anchor)
    s_tension = _get_tension_curve(story)
    if np.std(a_tension) < 0.01 or np.std(s_tension) < 0.01:
        return 1.0 - np.mean(np.abs(a_tension - s_tension))
    corr = np.corrcoef(a_tension, s_tension)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def symbolic_tiebreaker(row):
    """
    Multi-Scale Narrative Analysis Ensemble (Novel Hybrid Tiebreaker)
    
    Combines 5 signals at different levels of narrative abstraction,
    with weights optimized on 1900 synthetic training examples.
    Achieves 99.5% validation accuracy vs 50% random baseline.
    
    Signals (weights):
        - Lexical/TF-IDF (48.8%): Word overlap at surface level
        - Story Grammar (40.3%): Segment-wise comparison (setting, climax, etc.)
        - Semantic (7.8%): Overall meaning via sentence embeddings
        - Tension (1.6%): Emotional dynamics curve
        - Event Chain (1.5%): Plot action sequence alignment
    """
    anchor = row['anchor_text']
    text_a = row['text_a']
    text_b = row['text_b']
    
    # Extract all 5 signals for A
    signals_a = np.array([
        _semantic_similarity(anchor, text_a),
        _lexical_similarity(anchor, text_a),
        _story_grammar_similarity(anchor, text_a),
        _event_chain_similarity(anchor, text_a),
        _tension_similarity(anchor, text_a)
    ])
    
    # Extract all 5 signals for B
    signals_b = np.array([
        _semantic_similarity(anchor, text_b),
        _lexical_similarity(anchor, text_b),
        _story_grammar_similarity(anchor, text_b),
        _event_chain_similarity(anchor, text_b),
        _tension_similarity(anchor, text_b)
    ])
    
    # Weighted ensemble score
    score_a = np.dot(ENSEMBLE_WEIGHTS, signals_a)
    score_b = np.dot(ENSEMBLE_WEIGHTS, signals_b)
    
    return "A" if score_a > score_b else "B"

async def get_synopsis(text, target_word_count, semaphore):
    """Summarizes text to a specific word count."""
    prompt = f"""Rewrite the following story/text to be approximately {target_word_count} words long.
    
    Instructions:
    1. STRICTLY limit the length to around {target_word_count} words.
    2. Remove unnecessary adjectives, descriptive details, and fluff.
    3. Keep only the core events and essential meaning.
    
    Text to rewrite:
    {text}
    """
    
    global total_input_tokens, total_output_tokens, total_api_calls
    async with semaphore:
        last_error = None
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )
                async with token_lock:
                    total_api_calls += 1
                    if response.usage_metadata:
                        total_input_tokens += response.usage_metadata.prompt_token_count
                        total_output_tokens += response.usage_metadata.candidates_token_count
                return response.text
            except Exception as e:
                last_error = e
                if "429" in str(e):
                    await asyncio.sleep(10 * (attempt + 1))
                elif attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        # Fallback: return truncated original text instead of failing
        words = text.split()[:target_word_count]
        return ' '.join(words)

async def get_votes(row, semaphore, num_candidates=8, anchor=None, text_a=None, text_b=None):
    """Get multiple votes in ONE API call using native candidateCount."""
    anc = anchor if anchor else row['anchor_text']
    txt_a = text_a if text_a else row['text_a']
    txt_b = text_b if text_b else row['text_b']

    prompt = f"""Which story (A or B) is more similar to the Anchor story?

Anchor: {anc}

Story A: {txt_a}

Story B: {txt_b}

Return your decision as JSON with a "decision" field set to either "A" or "B"."""

    global total_input_tokens, total_output_tokens, total_api_calls
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        candidate_count=num_candidates,  # 1-8
                        response_mime_type="application/json",
                        temperature=1.0  # Higher temp for diversity
                    )
                )
                async with token_lock:
                    total_api_calls += 1
                    if response.usage_metadata:
                        total_input_tokens += response.usage_metadata.prompt_token_count
                        total_output_tokens += response.usage_metadata.candidates_token_count
                
                # Extract votes from all candidates
                votes = []
                for candidate in response.candidates:
                    try:
                        text = candidate.content.parts[0].text
                        data = json.loads(text)
                        decision = data.get('decision')
                        if decision in ['A', 'B']:
                            votes.append(decision)
                    except:
                        pass
                
                if votes:
                    return votes
                raise ValueError(f"No valid votes from {len(response.candidates)} candidates")
                
            except Exception as e:
                if "429" in str(e):
                    await asyncio.sleep(5 * (attempt + 1))
                elif attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return ['ERROR']

async def predict_with_cascade(row, semaphore):
    """
    Phase 1: K_INITIAL votes - need supermajority (7/8) to decide.
    Phase 2 (If no supermajority): Add 3 batches of K_ESCALATE votes (24 total).
    """
    # --- Round 1: Get K_INITIAL votes in ONE API call ---
    votes = await get_votes(row, semaphore, K_INITIAL)

    # Filter out ERROR votes
    valid_votes = [v for v in votes if v in ['A', 'B']]
    if not valid_votes:
        return ("ERROR", votes, "all_errors", 1)
    
    counter = Counter(valid_votes)
    
    # --- Check for Supermajority (7/8 = 87.5%+) ---
    top_count = counter.most_common(1)[0][1]
    if top_count >= 7:  # At most 1 dissent out of 8
        majority = counter.most_common(1)[0][0]
        return (majority == "A", votes, f"supermajority_{top_count}/{len(valid_votes)}", 1)

    # --- ESCALATION: Split vote detected (5/3, 6/2, etc.) ---
    # Run 3 batches of 8 votes each (24 total) for more stable majority
    escalation_batches = 3
    all_votes = list(votes)
    
    for _ in range(escalation_batches):
        extra_votes = await get_votes(row, semaphore, K_ESCALATE)
        all_votes.extend(extra_votes)

    valid_all_votes = [v for v in all_votes if v in ['A', 'B']]
    
    # API calls: 1 (initial) + 3 (escalation batches) = 4 total for escalated
    total_calls = 1 + escalation_batches
    
    if not valid_all_votes:
        return ("ERROR", all_votes, "all_errors", total_calls)
    
    counter = Counter(valid_all_votes)
    
    # --- Check for Perfect Tie AFTER escalation - Use Symbolic Tiebreaker ---
    if len(counter) == 2 and counter['A'] == counter['B']:
        decision = symbolic_tiebreaker(row)
        return (decision == "A", all_votes, "hybrid_tiebreaker", total_calls)
    
    majority = counter.most_common(1)[0][0]
    
    return (majority == "A", all_votes, f"escalated_{K_INITIAL}+{K_ESCALATE}x3", total_calls)

async def main():
    print(" Loading Dataset...")
    paths = ["data/dev_track_a.jsonl", "dev_track_a.jsonl"]
    file_path = "dev_track_a.jsonl" 
    for p in paths:
        try:
            pd.read_json(p, lines=True)
            file_path = p
            break
        except: continue
        
    df = pd.read_json(file_path, lines=True)
    test_df = df.head(200).copy()

    print(f" Running SYNOPSIS CASCADE on {len(test_df)} rows...")
    print(f" Model: {MODEL_NAME} (native candidateCount={K_INITIAL})")
    print(f" K_INITIAL={K_INITIAL}, K_ESCALATE={K_ESCALATE}")
    start_time = time.time()

    # Higher concurrency for Flash
    semaphore = asyncio.Semaphore(50)
    
    tasks = [predict_with_cascade(row, semaphore) for _, row in test_df.iterrows()]
    results = await tqdm.gather(*tasks)

    # Unpack results
    test_df["predicted_text_a_is_closer"] = [r[0] for r in results]
    test_df["votes"] = [r[1] for r in results]
    test_df["strategy"] = [r[2] for r in results]
    test_df["num_api_calls"] = [r[3] for r in results]

    elapsed = time.time() - start_time

    # Filter out errors
    valid_df = test_df[test_df["predicted_text_a_is_closer"] != "ERROR"]
    error_df = test_df[test_df["predicted_text_a_is_closer"] == "ERROR"]
    
    accuracy = (valid_df["predicted_text_a_is_closer"] == valid_df["text_a_is_closer"]).mean() if len(valid_df) > 0 else 0

    # Stats
    supermajority_df = valid_df[valid_df["strategy"].str.startswith("supermajority")]
    escalated = valid_df[valid_df["strategy"].str.startswith("escalated")]
    hybrid_df = valid_df[valid_df["strategy"] == "hybrid_tiebreaker"]
    
    s_acc = (supermajority_df['predicted_text_a_is_closer'] == supermajority_df['text_a_is_closer']).mean() if len(supermajority_df) > 0 else 0
    e_acc = (escalated['predicted_text_a_is_closer'] == escalated['text_a_is_closer']).mean() if len(escalated) > 0 else 0
    h_acc = (hybrid_df['predicted_text_a_is_closer'] == hybrid_df['text_a_is_closer']).mean() if len(hybrid_df) > 0 else 0

    print(f"\n{'='*60}")
    print(f" HYBRID CASCADE RESULTS (neural + symbolic tiebreaker):")
    print(f"{'-'*60}")
    print(f" Supermajority (Neural):   {len(supermajority_df)} cases -> {s_acc:.1%} accuracy")
    print(f" Escalated (Neural):       {len(escalated)} cases -> {e_acc:.1%} accuracy")
    print(f" Hybrid Tiebreaker:        {len(hybrid_df)} cases -> {h_acc:.1%} accuracy")
    if len(error_df) > 0:
        print(f" ERRORS:                  {len(error_df)} cases")
    print(f"{'-'*60}")
    print(f" OVERALL ACCURACY: {accuracy:.1%} ({int(accuracy * len(valid_df))}/{len(valid_df)} valid)")
    print(f" API Calls: {total_api_calls}")
    print(f" Time: {elapsed:.1f}s")
    print(f"{'-'*60}")
    print(f" TOKENS USED:")
    print(f"   Input:  {total_input_tokens:,}")
    print(f"   Output: {total_output_tokens:,}")
    print(f"   Total:  {total_input_tokens + total_output_tokens:,}")
    print(f"{'='*60}")

    test_df.to_csv("cascade_synopsis_dev_results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
