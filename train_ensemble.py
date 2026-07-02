"""
Multi-Scale Narrative Analysis Ensemble
Train and optimize weights on synthetic data for use as tiebreaker.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from scipy.optimize import differential_evolution
from tqdm import tqdm
import pickle

# ============================================================================
# SIGNAL EXTRACTORS (5 Multi-Scale Signals)
# ============================================================================

# Load sentence transformer model
print("Loading sentence-transformers model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
WEIGHTS_PATH = Path("artifacts/models/ensemble_weights.json")

def get_embedding(text):
    """Get sentence embedding using MiniLM-L6-v2"""
    # Handle None, empty strings, and whitespace-only strings
    if text is None or not isinstance(text, str) or not text.strip():
        return np.zeros(384)  # MiniLM embedding dimension
    # Truncate very long texts to avoid issues
    text = text[:10000] if len(text) > 10000 else text
    try:
        return embed_model.encode(text.strip(), show_progress_bar=False)
    except Exception:
        return np.zeros(384)

def cosine_sim(a, b):
    """Cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# --- Signal 1: Semantic Embedding Similarity ---
def semantic_similarity(anchor, story):
    """
    Level 3: Semantic Content
    Compare overall meaning using sentence embeddings
    """
    anchor_emb = get_embedding(anchor)
    story_emb = get_embedding(story)
    return cosine_sim(anchor_emb, story_emb)

# --- Signal 2: Lexical (TF-IDF) Similarity ---
def lexical_similarity(anchor, story):
    """
    Level 1: Surface Features
    Compare word overlap using TF-IDF
    """
    try:
        if not anchor or not story:
            return 0.0
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf = vectorizer.fit_transform([anchor, story])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        return 0.0

# --- Signal 3: Story Grammar Similarity (Novel) ---
def extract_story_grammar(text):
    """
    Extract story elements based on narrative theory (Propp, Campbell):
    Setting, Conflict, Rising Action, Climax, Resolution
    Using position-based heuristics
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

def story_grammar_similarity(anchor, story):
    """
    Level 4: Narrative Structure
    Compare stories segment-by-segment (setting vs setting, climax vs climax, etc.)
    """
    if not anchor or not story:
        return 0.0
    a_grammar = extract_story_grammar(anchor)
    s_grammar = extract_story_grammar(story)
    
    similarities = []
    for segment in ['setting', 'conflict', 'rising', 'climax', 'resolution']:
        a_text = a_grammar[segment]
        s_text = s_grammar[segment]
        if a_text and s_text and a_text.strip() and s_text.strip():
            sim = cosine_sim(get_embedding(a_text), get_embedding(s_text))
            similarities.append(sim)
    
    return np.mean(similarities) if similarities else 0.0

# --- Signal 4: Event Chain Similarity (Novel) ---
def extract_event_chain(text):
    """
    Extract sequence of main verbs (plot actions)
    Simplified version without spacy for speed
    """
    if not text or not isinstance(text, str):
        return []
    
    # Common action verbs that indicate plot events
    action_words = set([
        'discover', 'find', 'learn', 'realize', 'arrive', 'leave', 'escape',
        'fight', 'kill', 'save', 'rescue', 'destroy', 'create', 'build',
        'meet', 'marry', 'die', 'live', 'love', 'hate', 'fear', 'hope',
        'travel', 'return', 'begin', 'end', 'start', 'finish', 'win', 'lose',
        'reveal', 'hide', 'seek', 'search', 'chase', 'capture', 'release',
        'transform', 'change', 'become', 'remain', 'struggle', 'overcome',
        'betray', 'trust', 'deceive', 'forgive', 'revenge', 'sacrifice'
    ])
    
    words = text.lower().split()
    events = [w for w in words if any(w.startswith(av) for av in action_words)]
    return events[:20]  # Limit to first 20 events

def lcs_length(a, b):
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

def event_chain_similarity(anchor, story):
    """
    Level 2: Event Sequences
    Compare plot action sequences using LCS
    """
    a_events = extract_event_chain(anchor)
    s_events = extract_event_chain(story)
    
    if not a_events or not s_events:
        return 0.0
    
    lcs_len = lcs_length(a_events, s_events)
    return 2 * lcs_len / (len(a_events) + len(s_events))

# --- Signal 5: Narrative Tension Curve (Novel) ---
def get_tension_curve(text, n_points=10):
    """
    Model narrative tension using sentiment intensity + subjectivity
    High tension at conflict/climax, low at resolution
    """
    if not text or not isinstance(text, str):
        return np.zeros(n_points)
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return np.zeros(n_points)
    
    tensions = []
    for sent in sentences:
        try:
            blob = TextBlob(sent)
            # Tension = |sentiment| + subjectivity (uncertainty proxy)
            tension = abs(blob.sentiment.polarity) + blob.sentiment.subjectivity
            tensions.append(tension)
        except:
            tensions.append(0.0)
    
    if not tensions:
        return np.zeros(n_points)
    
    # Interpolate to fixed length
    tensions = np.array(tensions)
    return np.interp(np.linspace(0, 1, n_points),
                     np.linspace(0, 1, len(tensions)), tensions)

def tension_similarity(anchor, story):
    """
    Level 4: Narrative Structure (temporal dynamics)
    Compare tension curves using correlation
    """
    a_tension = get_tension_curve(anchor)
    s_tension = get_tension_curve(story)
    
    # Handle flat curves
    if np.std(a_tension) < 0.01 or np.std(s_tension) < 0.01:
        return 1.0 - np.mean(np.abs(a_tension - s_tension))
    
    corr = np.corrcoef(a_tension, s_tension)[0, 1]
    return corr if not np.isnan(corr) else 0.0

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_all_signals(anchor, text_a, text_b):
    """Extract all 5 signals for a story triplet"""
    signals_a = [
        semantic_similarity(anchor, text_a),
        lexical_similarity(anchor, text_a),
        story_grammar_similarity(anchor, text_a),
        event_chain_similarity(anchor, text_a),
        tension_similarity(anchor, text_a)
    ]
    
    signals_b = [
        semantic_similarity(anchor, text_b),
        lexical_similarity(anchor, text_b),
        story_grammar_similarity(anchor, text_b),
        event_chain_similarity(anchor, text_b),
        tension_similarity(anchor, text_b)
    ]
    
    return np.array(signals_a), np.array(signals_b)

# ============================================================================
# MAIN: LOAD DATA, EXTRACT FEATURES, OPTIMIZE WEIGHTS
# ============================================================================

if __name__ == "__main__":
    # Load synthetic data
    print("\n📂 Loading synthetic training data...")
    data_path = Path("data/synthetic_data_for_classification.jsonl")
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}. See data/README.md for download and placement notes.")
    data = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"   Loaded {len(data)} examples")
    
    # Split into train/val
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    train_idx = indices[:1700]
    val_idx = indices[1700:]
    
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Extract features
    print("\n🔬 Extracting features from training data...")
    X_a_train, X_b_train, y_train = [], [], []
    
    for row in tqdm(train_data, desc="Training features"):
        # Skip rows with missing data
        if row.get('anchor_text') is None or row.get('text_a') is None or row.get('text_b') is None:
            continue
        signals_a, signals_b = extract_all_signals(
            row['anchor_text'], row['text_a'], row['text_b']
        )
        X_a_train.append(signals_a)
        X_b_train.append(signals_b)
        y_train.append(row['text_a_is_closer'])
    
    X_a_train = np.array(X_a_train)
    X_b_train = np.array(X_b_train)
    y_train = np.array(y_train)
    
    print("\n🔬 Extracting features from validation data...")
    X_a_val, X_b_val, y_val = [], [], []
    
    for row in tqdm(val_data, desc="Validation features"):
        # Skip rows with missing data
        if row.get('anchor_text') is None or row.get('text_a') is None or row.get('text_b') is None:
            continue
        signals_a, signals_b = extract_all_signals(
            row['anchor_text'], row['text_a'], row['text_b']
        )
        X_a_val.append(signals_a)
        X_b_val.append(signals_b)
        y_val.append(row['text_a_is_closer'])
    
    X_a_val = np.array(X_a_val)
    X_b_val = np.array(X_b_val)
    y_val = np.array(y_val)
    
    # Optimize weights using differential evolution
    print("\n⚙️ Optimizing ensemble weights...")
    
    def objective(weights, X_a, X_b, y_true):
        """Negative accuracy (to minimize)"""
        weights = np.abs(weights)  # Ensure positive
        scores_a = X_a @ weights
        scores_b = X_b @ weights
        preds = scores_a > scores_b
        return -np.mean(preds == y_true)
    
    # Global optimization (single thread for pickling compatibility)
    bounds = [(0, 1)] * 5
    result = differential_evolution(
        objective, 
        bounds, 
        args=(X_a_train, X_b_train, y_train),
        seed=42,
        maxiter=100,
        disp=True,
        workers=1,  # Single thread for compatibility
        polish=True
    )
    
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()  # Normalize
    
    print(f"\n📊 Optimal weights:")
    signal_names = ['Semantic', 'Lexical', 'StoryGrammar', 'EventChain', 'Tension']
    for name, weight in zip(signal_names, optimal_weights):
        print(f"   {name}: {weight:.4f}")
    
    # Evaluate on training set
    train_scores_a = X_a_train @ optimal_weights
    train_scores_b = X_b_train @ optimal_weights
    train_preds = train_scores_a > train_scores_b
    train_acc = np.mean(train_preds == y_train)
    print(f"\n📈 Training accuracy: {train_acc:.1%}")
    
    # Evaluate on validation set
    val_scores_a = X_a_val @ optimal_weights
    val_scores_b = X_b_val @ optimal_weights
    val_preds = val_scores_a > val_scores_b
    val_acc = np.mean(val_preds == y_val)
    print(f"📈 Validation accuracy: {val_acc:.1%}")
    
    # Compare to individual signals
    print("\n📊 Individual signal performance (validation):")
    for i, name in enumerate(signal_names):
        single_weight = np.zeros(5)
        single_weight[i] = 1.0
        single_scores_a = X_a_val @ single_weight
        single_scores_b = X_b_val @ single_weight
        single_preds = single_scores_a > single_scores_b
        single_acc = np.mean(single_preds == y_val)
        print(f"   {name}: {single_acc:.1%}")
    
    # Save optimal weights
    weights_dict = {
        'weights': optimal_weights.tolist(),
        'signal_names': signal_names,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WEIGHTS_PATH.open("w") as f:
        json.dump(weights_dict, f, indent=2)
    
    print(f"\n✅ Saved optimal weights to {WEIGHTS_PATH}")
    print(f"\n{'='*60}")
    print(f"ENSEMBLE READY: {val_acc:.1%} accuracy on validation")
    print(f"(vs 50% random baseline)")
    print(f"{'='*60}")
