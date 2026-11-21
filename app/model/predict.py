import joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.sparse import csr_matrix, hstack

from app.features.text_features import extract_text_features, load_vectorizers
from app.features.handcrafted import extract_handcrafted_features, get_feature_names
from app.schemas import CheckResult, AnalysisResult


# --- Paths ---
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


# --- Loaded at startup ---
_model: Optional[xgb.XGBClassifier] = None
_metrics: Dict[str, Any] = {}


def load_model():
    """Load model and metrics from disk."""
    global _model, _metrics
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

    if METRICS_PATH.exists():
        import json
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)


def get_model() -> xgb.XGBClassifier:
    if _model is None:
        load_model()
    return _model


def get_metrics() -> Dict[str, Any]:
    return _metrics


def predict_risk(
    job_title: str,
    company_name: str,
    job_description: str,
    job_url: str = "",
    recruiter_email: str = "",
    recruiter_phone: str = "",
    location: str = "",
    salary_text: str = "",
    employment_type: str = "",
    raw_text: str = ""
) -> AnalysisResult:
    """
    Main prediction function. Returns AnalysisResult with score, tier, and evidence checks.
    """
    model = get_model()

    # Combine all text for TF-IDF
    combined_text = " ".join(filter(None, [
        job_title, company_name, job_description,
        location, salary_text, employment_type, raw_text
    ]))

    # Extract features
    X = extract_text_features(combined_text)

    # Predict probability
    fraud_proba = model.predict_proba(X)[0, 1]
    risk_score = int(round(fraud_proba * 100))

    # Determine tier
    if risk_score >= 70:
        risk_tier = "risky"
    elif risk_score >= 40:
        risk_tier = "caution"
    else:
        risk_tier = "safe"

    # Build evidence checks
    checks = build_evidence_checks(
        job_title, company_name, job_description,
        job_url, recruiter_email, recruiter_phone,
        location, salary_text, combined_text, fraud_proba
    )

    # Summary & recommendation
    summary = generate_summary(risk_tier, risk_score, checks)
    recommendation = generate_recommendation(risk_tier, checks)

    return AnalysisResult(
        risk_score=risk_score,
        risk_tier=risk_tier,
        checks=checks,
        summary=summary,
        recommendation=recommendation,
        model_version=_metrics.get('model_version', '1.0.0')
    )


def build_evidence_checks(
    job_title: str,
    company_name: str,
    job_description: str,
    job_url: str,
    recruiter_email: str,
    recruiter_phone: str,
    location: str,
    salary_text: str,
    combined_text: str,
    fraud_proba: float
) -> List[CheckResult]:
    """Build list of evidence checks with pass/fail/warning status."""
    checks = []
    text_lower = combined_text.lower()

    # 1. Urgency language
    urgency_words = ['urgent', 'immediate', 'asap', 'today', 'limited', 'expire', 'hurry', 'quick', 'instant', '24 hour', '48 hour']
    urgency_count = sum(1 for w in urgency_words if w in text_lower)
    checks.append(CheckResult(
        name="Urgency Pressure",
        check_status="fail" if urgency_count >= 2 else "warning" if urgency_count == 1 else "pass",
        message=f"Found {urgency_count} urgency indicators" if urgency_count else "No urgency pressure detected",
        details="Scammers create false urgency to prevent research" if urgency_count else "Legitimate postings rarely use extreme urgency",
        weight=2.0 if urgency_count >= 2 else 1.0
    ))

    # 2. Money/fee requests
    fee_patterns = ['fee', 'deposit', 'payment', 'charge', 'cost', 'investment', 'registration', 'training fee', 'processing fee', 'security deposit']
    fee_count = sum(1 for p in fee_patterns if p in text_lower)
    checks.append(CheckResult(
        name="Upfront Payment Request",
        check_status="fail" if fee_count >= 1 else "pass",
        message=f"Detected {fee_count} payment-related terms" if fee_count else "No upfront payment requests found",
        details="Legitimate employers never ask candidates to pay fees" if fee_count else "",
        weight=3.0
    ))

    # 3. Personal info requests
    pii_patterns = ['ssn', 'social security', 'bank account', 'passport', 'aadhaar', 'pan card', 'id proof', 'bank statement']
    pii_count = sum(1 for p in pii_patterns if p in text_lower)
    checks.append(CheckResult(
        name="Sensitive Personal Information Request",
        check_status="fail" if pii_count >= 1 else "pass",
        message=f"Requests {pii_count} types of sensitive info" if pii_count else "No sensitive personal info requested",
        details="Real employers collect this only AFTER formal offer, not in job posting" if pii_count else "",
        weight=3.0
    ))

    # 4. Free email domain
    free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'icloud.com', 'protonmail.com']
    is_free_email = 0.0
    if recruiter_email:
        domain = recruiter_email.split('@')[-1].lower() if '@' in recruiter_email else ''
        is_free_email = 1.0 if domain in free_domains else 0.0
    checks.append(CheckResult(
        name="Recruiter Email Domain",
        check_status="fail" if is_free_email else "pass" if recruiter_email else "info",
        message=f"Recruiter uses free email ({recruiter_email})" if is_free_email else f"Recruiter uses corporate email ({recruiter_email})" if recruiter_email else "No recruiter email provided",
        details="Legitimate companies use corporate domains, not Gmail/Yahoo" if is_free_email else "",
        weight=2.5
    ))

    # 5. Suspicious contact methods
    contact_patterns = ['whatsapp', 'telegram', 'signal', 'viber']
    contact_count = sum(1 for p in contact_patterns if p in text_lower)
    checks.append(CheckResult(
        name="Communication Channel",
        check_status="warning" if contact_count >= 1 else "pass",
        message=f"Suggests {contact_count} informal chat apps" if contact_count else "Standard communication channels",
        details="Scammers prefer untraceable chat apps over official email/phone" if contact_count else "",
        weight=1.5
    ))

    # 6. Scam keywords
    scam_kws = ['task', 'data entry', 'copy paste', 'typing', 'form filling', 'captcha', 'survey', 'click', 'like', 'subscribe', 'review', 'rating', 'app testing']
    scam_count = sum(1 for kw in scam_kws if kw in text_lower)
    checks.append(CheckResult(
        name="Task Scam Keywords",
        check_status="fail" if scam_count >= 3 else "warning" if scam_count >= 1 else "pass",
        message=f"Found {scam_count} task-scam keywords" if scam_count else "No task-scam patterns detected",
        details="High density of 'task', 'copy-paste', 'click' indicates task scam" if scam_count >= 1 else "",
        weight=2.0
    ))

    # 7. Legitimate job structure
    legit_kws = ['experience', 'qualification', 'degree', 'responsibilities', 'requirements', 'benefits', 'insurance', 'provident fund', 'gratuity', 'notice period', 'probation']
    legit_count = sum(1 for kw in legit_kws if kw in text_lower)
    checks.append(CheckResult(
        name="Professional Job Structure",
        check_status="pass" if legit_count >= 3 else "warning" if legit_count >= 1 else "fail",
        message=f"Contains {legit_count} professional job indicators" if legit_count else "Lacks standard job posting structure",
        details="Legitimate postings detail requirements, benefits, responsibilities" if legit_count >= 1 else "Minimal job details is a red flag",
        weight=1.5
    ))

    # 8. Salary anomaly (basic check)
    salary_suspicious = 0
    if salary_text:
        # Look for unrealistic amounts
        import re
        amounts = re.findall(r'[\d,]+', salary_text.replace('₹', '').replace('Rs', '').replace('LPA', '').replace('lpa', ''))
        for amt in amounts:
            try:
                val = int(amt.replace(',', ''))
                if val > 5000000:  # >50 LPA for entry level suspicious
                    salary_suspicious = 1
            except:
                pass
    checks.append(CheckResult(
        name="Salary Realism",
        check_status="warning" if salary_suspicious else "pass" if salary_text else "info",
        message="Salary appears unusually high for role" if salary_suspicious else "Salary range seems realistic" if salary_text else "No salary information provided",
        details="Offers 50%+ above market rate are often bait" if salary_suspicious else "",
        weight=1.5
    ))

    # 9. Job board / ATS domain
    job_boards = ['linkedin.com', 'naukri.com', 'indeed.com', 'glassdoor.com', 'foundit.in', 'shine.com', 'timesjobs.com']
    ats_domains = ['greenhouse.io', 'lever.co', 'workday.com', 'smartrecruiters.com']
    on_job_board = any(jb in job_url.lower() for jb in job_boards) if job_url else False
    on_ats = any(ats in job_url.lower() for ats in ats_domains) if job_url else False
    checks.append(CheckResult(
        name="Posting Platform",
        check_status="pass" if on_ats else "warning" if on_job_board else "info" if job_url else "info",
        message="Posted on verified ATS (Greenhouse/Lever/Workday)" if on_ats else f"Posted on job board ({job_url})" if on_job_board else "Direct link or unknown platform" if job_url else "No URL provided",
        details="ATS postings are verified by platform; job boards have moderation" if on_ats or on_job_board else "Direct links bypass platform verification",
        weight=2.0
    ))

    # 10. Model confidence
    checks.append(CheckResult(
        name="ML Model Confidence",
        check_status="fail" if fraud_proba > 0.7 else "warning" if fraud_proba > 0.4 else "pass",
        message=f"Model predicts {fraud_proba*100:.0f}% fraud probability",
        details=f"Ensemble of TF-IDF + XGBoost trained on {_metrics.get('train_samples', 'N/A')} samples",
        weight=3.0
    ))

    return checks


def generate_summary(tier: str, score: int, checks: List[CheckResult]) -> str:
    """Generate human-readable summary."""
    fail_count = sum(1 for c in checks if c.check_status == "fail")
    warn_count = sum(1 for c in checks if c.check_status == "warning")
    pass_count = sum(1 for c in checks if c.check_status == "pass")

    if tier == "risky":
        return f"HIGH RISK ({score}/100) — {fail_count} critical red flags detected. This posting exhibits multiple characteristics of known job scams including upfront payment requests, urgency pressure, and suspicious communication channels."
    elif tier == "caution":
        return f"MODERATE RISK ({score}/100) — {warn_count} warning signs found. While not definitively fraudulent, this posting has unusual elements. Verify independently before sharing personal information."
    else:
        return f"LOW RISK ({score}/100) — {pass_count} positive indicators. This posting appears legitimate with standard job structure, professional details, and no major red flags."


def generate_recommendation(tier: str, checks: List[CheckResult]) -> str:
    """Generate actionable next steps."""
    if tier == "risky":
        return "❌ DO NOT APPLY. Do not share any personal information. Report this posting to the platform. If you already shared info, monitor accounts and consider identity theft protection."
    elif tier == "caution":
        return "⚠️ VERIFY FIRST. Check company's official careers page for this role. Search company + 'scam' online. Contact company via official website (not links in posting). Never pay fees to apply."
    else:
        return "✅ PROCEED WITH NORMAL CAUTION. Still verify company on official website. Use standard job application safety practices. Never pay to apply for a job."