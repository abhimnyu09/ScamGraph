#!/usr/bin/env python3
"""
Training pipeline for ScamGraph fake job detector.
Downloads Kaggle FJDC, adds India-specific samples, trains XGBoost model.
Run: python app/model/train.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
CHAR_VEC_PATH = ARTIFACTS_DIR / "char_vectorizer.pkl"
WORD_VEC_PATH = ARTIFACTS_DIR / "word_vectorizer.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# --- India-specific synthetic samples (fresher scams, training fees, etc.) ---
INDIA_SCAM_SAMPLES = [
    # Training fee scams
    {"title": "Python Developer Trainee", "company": "TechSolutions India", "description": "We are hiring freshers for Python developer role. 3 months training program with placement guarantee. Training fee ₹15,000 refundable after placement. No experience required. Work from home option available. Contact HR on WhatsApp 9876543210 for immediate joining.", "label": 1},
    {"title": "Data Entry Operator", "company": "Global Data Services", "description": "Urgent hiring for data entry operators. Simple copy paste work from home. Daily payment ₹500-1000. Registration fee ₹2,000 required for software access. No experience needed. Students and housewives welcome. Call now 8765432109.", "label": 1},
    {"title": "Business Development Executive", "company": "Growth Partners", "description": "BDE needed for ed-tech startup. 6 months training with stipend ₹5,000/month. Training cost ₹25,000 payable in installments. 100% placement guarantee in top MNCs. Fresher friendly. Email resume to careers@growthpartners.xyz", "label": 1},
    {"title": "Software Testing Trainee", "company": "QA Masters", "description": "Manual and automation testing training program. Fee ₹30,000 with job guarantee. Learn Selenium, API testing, mobile testing. Weekend batches for working professionals. Limited seats. Register at qamasters.tk", "label": 1},
    {"title": "Digital Marketing Intern", "company": "SocialBuzz Agency", "description": "3-month digital marketing internship. Training fee ₹8,000 includes tools access. Certificate provided. Work from home. Stipend after training ₹3,000. Freshers preferred. WhatsApp 7654321098", "label": 1},

    # Legitimate Indian job postings
    {"title": "Senior Software Engineer", "company": "TCS", "description": "We are looking for a Senior Software Engineer with 5+ years of experience in Java, Spring Boot, Microservices. Responsibilities include designing scalable systems, code reviews, mentoring junior developers. Qualifications: B.Tech/M.Tech in Computer Science. Benefits: Health insurance, PF, gratuity, variable pay, learning budget. Location: Mumbai/Pune/Bangalore. Notice period: 60 days.", "label": 0},
    {"title": "Product Manager", "company": "Flipkart", "description": "Product Manager role for consumer marketplace team. 3+ years in product management at scale. Define product strategy, work with engineering, design, analytics. Experience with A/B testing, SQL, user research. Competitive salary, ESOPs, health benefits, flexible work. Location: Bangalore.", "label": 0},
    {"title": "Data Scientist", "company": "Razorpay", "description": "Data Scientist for fraud detection team. 2+ years experience with Python, ML, SQL. Build fraud detection models, analyze transaction patterns. Strong statistics background required. Benefits: Stock options, health insurance, learning allowance, remote-friendly. Location: Bangalore.", "label": 0},
    {"title": "Frontend Developer", "company": "Zomato", "description": "Frontend Developer (React/TypeScript) for consumer app. 2+ years experience. Build performant UI components, optimize bundle size, improve Core Web Vitals. Experience with Redux, testing libraries. Competitive package, ESOPs, food credits, health insurance. Location: Gurugram.", "label": 0},
    {"title": "DevOps Engineer", "company": "Swiggy", "description": "DevOps Engineer for platform team. 3+ years with AWS, Kubernetes, Terraform, CI/CD. Manage infrastructure at scale, improve deployment pipelines. On-call rotation. Great compensation, stock options, health benefits, flexible hours. Location: Bangalore.", "label": 0},
]


def download_fjdc() -> pd.DataFrame:
    """Download Fake Job Posting Detection dataset from Kaggle via kagglehub."""
    try:
        import kagglehub
        print("Downloading FJDC dataset from Kaggle...")
        path = kagglehub.dataset_download("shivamb/real-or-fake-fake-jobposting-prediction")
        csv_path = Path(path) / "fake_job_postings.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from Kaggle FJDC")
            return df
    except Exception as e:
        print(f"Kaggle download failed: {e}")

    # Fallback: check local
    local_path = RAW_DIR / "fake_job_postings.csv"
    if local_path.exists():
        print("Loading local FJDC CSV...")
        return pd.read_csv(local_path)

    raise FileNotFoundError("FJDC dataset not found. Place fake_job_postings.csv in data/raw/ or install kagglehub.")


def prepare_fjdc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare FJDC dataframe."""
    # Combine text fields
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ''

    df['combined_text'] = (
        df['title'].fillna('') + ' ' +
        df['company_profile'].fillna('') + ' ' +
        df['description'].fillna('') + ' ' +
        df['requirements'].fillna('') + ' ' +
        df['benefits'].fillna('')
    )

    # Keep only needed columns
    df = df[['combined_text', 'fraudulent']].copy()
    df.columns = ['text', 'label']

    # Remove duplicates and empty
    df = df.drop_duplicates(subset=['text'])
    df = df[df['text'].str.len() > 50]

    print(f"FJDC prepared: {len(df)} rows, {df['label'].sum()} fraud ({df['label'].mean()*100:.1f}%)")
    return df


def add_india_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Add India-specific scam and legitimate samples."""
    india_df = pd.DataFrame(INDIA_SCAM_SAMPLES)
    india_df['text'] = (
        india_df['title'] + ' ' +
        india_df['company'] + ' ' +
        india_df['description']
    )
    india_df = india_df[['text', 'label']]

    combined = pd.concat([df, india_df], ignore_index=True)
    print(f"After India samples: {len(combined)} rows, {combined['label'].sum()} fraud ({combined['label'].mean()*100:.1f}%)")
    return combined


def train_model():
    """Main training pipeline."""
    print("=" * 60)
    print("ScamGraph Model Training Pipeline")
    print("=" * 60)

    # 1. Load data
    fjdc_df = download_fjdc()
    df = prepare_fjdc(fjdc_df)
    df = add_india_samples(df)

    # 2. Split
    X_text = df['text'].tolist()
    y = df['label'].values
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build vectorizers
    from app.features.text_features import build_vectorizers, extract_text_features_batch, save_vectorizers
    char_vec, word_vec = build_vectorizers()

    print("Fitting vectorizers...")
    char_vec.fit(X_train_text)
    word_vec.fit(X_train_text)

    # Save vectorizers first so extract_text_features_batch can load them
    save_vectorizers(char_vec, word_vec)

    # 4. Extract features
    print("Extracting features...")
    X_train = extract_text_features_batch(X_train_text)
    X_test = extract_text_features_batch(X_test_text)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 5. Train XGBoost
    print("Training XGBoost...")
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        tree_method='hist'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # 6. Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {auc:.4f}")

    # 7. Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(model, MODEL_PATH)
    save_vectorizers(char_vec, word_vec)

    # Save metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        'accuracy': float((y_pred == y_test).mean()),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(auc),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'fraud_ratio_train': float(y_train.mean()),
        'fraud_ratio_test': float(y_test.mean()),
        'scale_pos_weight': float(scale_pos_weight),
        'model_version': '1.0.0'
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizers saved to {ARTIFACTS_DIR}")
    print(f"Metrics saved to {METRICS_PATH}")
    print("\n✅ Training complete!")

    # Feature importance (top 20)
    try:
        from app.features.text_features import get_vectorizers
        char_vec, word_vec = get_vectorizers()
        char_names = [f"char_{n}" for n in char_vec.get_feature_names_out()]
        word_names = [f"word_{n}" for n in word_vec.get_feature_names_out()]
        from app.features.handcrafted import get_feature_names
        handcrafted_names = get_feature_names()
        all_names = char_names + word_names + handcrafted_names

        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        print("\nTop 20 Feature Importances:")
        for idx in top_idx:
            if idx < len(all_names):
                print(f"  {all_names[idx]}: {importances[idx]:.4f}")
    except Exception as e:
        print(f"Could not compute feature importance: {e}")


if __name__ == "__main__":
    train_model()