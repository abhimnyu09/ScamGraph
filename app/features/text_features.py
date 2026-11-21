import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional

from app.features.handcrafted import extract_handcrafted_features, get_feature_names


# --- Paths ---
ARTIFACTS_DIR = Path(__file__).parent.parent / "model" / "artifacts"
CHAR_VECTORIZER_PATH = ARTIFACTS_DIR / "char_vectorizer.pkl"
WORD_VECTORIZER_PATH = ARTIFACTS_DIR / "word_vectorizer.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"


# --- Vectorizers (loaded at startup) ---
_char_vectorizer: Optional[TfidfVectorizer] = None
_word_vectorizer: Optional[TfidfVectorizer] = None


def load_vectorizers():
    """Load TF-IDF vectorizers from disk."""
    global _char_vectorizer, _word_vectorizer
    if CHAR_VECTORIZER_PATH.exists() and WORD_VECTORIZER_PATH.exists():
        _char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
        _word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
    else:
        raise FileNotFoundError("Vectorizers not found. Run train.py first.")


def get_vectorizers() -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    if _char_vectorizer is None or _word_vectorizer is None:
        load_vectorizers()
    return _char_vectorizer, _word_vectorizer


def build_vectorizers() -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Create new TF-IDF vectorizers for training."""
    # Character n-grams: catch typos, obfuscation, OCR noise
    char_vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )

    # Word n-grams: catch phrases, terminology
    word_vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )

    return char_vec, word_vec


def extract_text_features(text: str) -> csr_matrix:
    """Transform text into combined TF-IDF + handcrafted feature matrix."""
    char_vec, word_vec = get_vectorizers()

    # TF-IDF features
    char_features = char_vec.transform([text])
    word_features = word_vec.transform([text])

    # Handcrafted features
    handcrafted = extract_handcrafted_features(text)
    handcrafted_sparse = csr_matrix(handcrafted)

    # Combine all
    combined = hstack([char_features, word_features, handcrafted_sparse], format='csr')
    return combined


def extract_text_features_batch(texts: list) -> csr_matrix:
    """Batch transform for training."""
    char_vec, word_vec = get_vectorizers()

    char_features = char_vec.transform(texts)
    word_features = word_vec.transform(texts)

    handcrafted_list = [extract_handcrafted_features(t).flatten() for t in texts]
    handcrafted_sparse = csr_matrix(np.vstack(handcrafted_list))

    combined = hstack([char_features, word_features, handcrafted_sparse], format='csr')
    return combined


def save_vectorizers(char_vec: TfidfVectorizer, word_vec: TfidfVectorizer):
    """Save fitted vectorizers to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(char_vec, CHAR_VECTORIZER_PATH)
    joblib.dump(word_vec, WORD_VECTORIZER_PATH)

    # Save feature names for interpretability
    import json
    char_names = [f"char_{n}" for n in char_vec.get_feature_names_out()]
    word_names = [f"word_{n}" for n in word_vec.get_feature_names_out()]
    handcrafted_names = get_feature_names()
    all_names = char_names + word_names + handcrafted_names
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(all_names, f)