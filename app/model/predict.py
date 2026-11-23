import joblib
import numpy as np
import xgboost as xgb
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy.sparse import csr_matrix, hstack

from app.features.text_features import extract_text_features, load_vectorizers
from app.features.handcrafted import extract_handcrafted_features, get_feature_names
from app.schemas import CheckResult, AnalysisResult

# --- Groq Client (lazy init) ---
_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            from groq import Groq
            _groq_client = Groq(api_key=api_key)
    return _groq_client


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


# --- AI Research Agent (Groq) ---
def enrich_with_ai_research(
    job_title: str,
    company_name: str,
    job_description: str,
    job_url: str,
    recruiter_email: str,
    recruiter_phone: str,
    location: str,
    salary_text: str,
    risk_score: int,
    risk_tier: str
) -> Dict[str, Any]:
    """
    Use Groq to research company and enrich analysis with JobScamScore-level detail.
    Returns dict with all detailed analysis fields.
    """
    client = get_groq_client()
    if not client:
        return {}  # No API key, fallback to stubs

    # Build comprehensive prompt for JobScamScore-level detail
    prompt = f"""You are a senior job scam detection analyst. Research this posting thoroughly and return EXTREMELY DETAILED analysis like JobScamScore.

JOB DATA:
- Title: {job_title}
- Company: {company_name}
- Description: {job_description[:4000]}
- URL: {job_url or 'Not provided'}
- Recruiter Email: {recruiter_email or 'Not provided'}
- Recruiter Phone: {recruiter_phone or 'Not provided'}
- Location: {location or 'Not provided'}
- Salary: {salary_text or 'Not provided'}
- ML Risk Score: {risk_score}/100 ({risk_tier})

RESEARCH & ANALYZE:
1. COMPANY: Find official website, LinkedIn, Glassdoor, Crunchbase, domain age, registration, years in operation, all known office locations
2. LOCATION: Extract job location from posting, determine work type, compare with company locations, detailed match analysis
3. CONTACT: Extract recruiter name, verify email domain matches company, find LinkedIn, check directory match, verify official channel
4. SALARY: Extract mentioned salary, find Indian market benchmark for role×city, analyze structure, check for unrealistic promises
5. GRAMMAR: Count grammar errors, spacing issues, suspicious phrases, urgency language, provide examples
6. PATTERNS: List specific suspicious AND legitimate signals with evidence from posting
7. RECOMMENDATIONS: Actionable steps, immediate precautions, verification steps

Return ONLY valid JSON with ALL fields (use null/empty arrays if unknown - NO placeholders):
{{
  "company_research": {{
    "official_website": "https://... or null",
    "careers_page": "https://... or null",
    "linkedin_company_url": "https://... or null",
    "glassdoor_company_url": "https://... or null",
    "crunchbase_url": "https://... or null",
    "domain_age_days": integer or null,
    "registration_status": "registered|not_found|unknown",
    "domain_ownership_match": "matches|mismatch|unknown",
    "physical_address_verified": true/false/null,
    "years_in_operation": integer or null,
    "company_locations": ["city, state, country", ...],
    "press_media_coverage": ["url", ...] or [],
    "other_presence": {{"indeed": "url|null", "ziprecruiter": "url|null", "monster": "url|null", "dice": "url|null"}},
    "ats_platform": "Greenhouse|Lever|Workable|... or null",
    "scam_reports_found": boolean,
    "executive_summary": "2-3 paragraph detailed company analysis like JobScamScore"
  }},
  "location_verification": {{
    "job_location": "extracted from posting or null",
    "work_type": "remote|hybrid|on_site",
    "company_locations": ["city, state, country", ...],
    "match_status": "pass|warning|fail|info",
    "match_details": "detailed explanation",
    "analysis_notes": "detailed location analysis paragraph"
  }},
  "contact_verification": {{
    "recruiter_name": "extracted or null",
    "company_contact_email": "official careers@ or null",
    "company_contact_phone": "official phone or null",
    "company_website": "official or null",
    "job_contact_emails": ["email1", "email2"] or [],
    "job_contact_phones": ["phone1", "phone2"] or [],
    "email_domain_verified": true/false,
    "recruiter_linkedin_url": "https://... or null",
    "linkedin_verified": "verified|not_found|unknown",
    "directory_match": "found|not_found|unknown",
    "official_channel": "pass|warning|fail",
    "match_status": "pass|warning|fail|info",
    "analysis": "detailed contact analysis paragraph"
  }},
  "salary_analysis": {{
    "salary_mentioned": true/false,
    "salary_range": "extracted range or null",
    "market_benchmark": "Indian market range for role×city (e.g., '₹10-25k/month for Data Entry in Bangalore')",
    "comparison_analysis": "detailed comparison paragraph",
    "in_range": "pass|warning|fail|info",
    "no_unrealistic_promises": "pass|warning|fail",
    "clear_structure": "pass|warning|fail",
    "job_structure_complete": "pass|warning|fail"
  }},
  "grammar_analysis": {{
    "grammar_errors": integer,
    "spacing_issues": integer,
    "suspicious_phrases": integer,
    "urgency_indicators": integer,
    "examples": ["specific example 1", "specific example 2"],
    "content_assessment": "detailed quality assessment paragraph"
  }},
  "pattern_detection": {{
    "suspicious_signals": [
      {{"type": "category", "signal": "specific evidence from posting", "severity": "critical|high|medium"}},
      ...
    ],
    "legitimate_signals": [
      {{"type": "category", "signal": "specific evidence from posting", "severity": "positive"}},
      ...
    ]
  }},
  "final_recommendations": {{
    "executive_recommendation": "detailed recommendation paragraph",
    "immediate_precautions": ["specific actionable step 1", "step 2", ...],
    "verification_steps": ["specific step 1", "step 2", ...],
    "already_applied_guidance": "guidance if already applied"
  }}
}}

CRITICAL:
- Be SPECIFIC - cite exact text from the job description
- For Indian market salary benchmarks, use realistic INR ranges for role×city
- For company research, find actual known platforms (LinkedIn, Glassdoor, Crunchbase, Indeed, etc.)
- NO generic placeholders like "requires external database" - provide actual findings or null
- If information cannot be determined, use null/empty array, not placeholder text
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",  # Groq free tier model
            messages=[
                {"role": "system", "content": "You are a senior job scam detection analyst. Return only valid JSON with detailed, specific findings. No placeholders."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"AI research error: {e}")
        return {}


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

    # --- AI Research Enrichment ---
    ai_data = enrich_with_ai_research(
        job_title, company_name, job_description, job_url,
        recruiter_email, recruiter_phone, location, salary_text,
        risk_score, risk_tier
    )

    # --- Detailed Analysis Fields (AI first, fallback to stubs) ---
    # New nested structure from AI
    cr = ai_data.get("company_research", {})
    lv = ai_data.get("location_verification", {})
    cv = ai_data.get("contact_verification", {})
    sa = ai_data.get("salary_analysis", {})
    ga = ai_data.get("grammar_analysis", {})
    pd = ai_data.get("pattern_detection", {})
    fr = ai_data.get("final_recommendations", {})

    def ai_get_nested(section_dict, key, default=None):
        return section_dict.get(key, default)

    # Company Research & Verification
    company_website = ai_get_nested(cr, "official_website") or extract_company_website(company_name, job_description, job_url)
    domain_age = ai_get_nested(cr, "domain_age_days") or estimate_domain_age(company_website) if company_website else None
    linkedin_company = ai_get_nested(cr, "linkedin_company_url") or find_linkedin_company(company_name)
    glassdoor_company = ai_get_nested(cr, "glassdoor_company_url") or find_glassdoor_company(company_name)
    company_registration_status = ai_get_nested(cr, "registration_status") or check_company_registration(company_name)
    domain_ownership_status = ai_get_nested(cr, "domain_ownership_match") or check_domain_ownership(company_name, company_website)
    physical_address_status = ai_get_nested(cr, "physical_address_verified")
    if physical_address_status is None:
        physical_address_status = check_physical_address(job_description, company_name)
    years_operation_status = ai_get_nested(cr, "years_in_operation") or check_years_operation(company_name, domain_age)
    # company_locations from AI is a list, convert to string for schema
    company_locations_raw = ai_get_nested(cr, "company_locations")
    if isinstance(company_locations_raw, list):
        company_locations = "; ".join(company_locations_raw)
    else:
        company_locations = company_locations_raw or extract_company_locations(job_description, company_name)

    # Location Verification
    job_location = ai_get_nested(lv, "job_location") or (location if location else extract_job_location(job_description))
    user_location = location
    work_type = ai_get_nested(lv, "work_type") or determine_work_type(job_description, employment_type)
    location_match_status = ai_get_nested(lv, "match_status")
    location_match_details = ai_get_nested(lv, "match_details")
    location_analysis_notes = ai_get_nested(lv, "analysis_notes")
    if location_match_status is None:
        location_match_status, location_match_details, location_analysis_notes = analyze_location_match(
            company_locations, job_location, user_location, work_type
        )

    # Contact Verification
    recruiter_name = ai_get_nested(cv, "recruiter_name") or extract_recruiter_name(job_description, raw_text)
    email_domain_verified = ai_get_nested(cv, "email_domain_verified") if ai_get_nested(cv, "email_domain_verified") is not None else verify_email_domain(recruiter_email, company_name, company_website)
    recruiter_linkedin = ai_get_nested(cv, "recruiter_linkedin_url") or find_recruiter_linkedin(recruiter_name, company_name)
    linkedin_verified = ai_get_nested(cv, "linkedin_verified") or verify_linkedin_profile(recruiter_linkedin, company_name)
    directory_match = ai_get_nested(cv, "directory_match") or check_employee_directory(recruiter_name, company_name)
    official_channel = ai_get_nested(cv, "official_channel") or check_official_channel(job_description, recruiter_email, recruiter_phone)

    # Salary & Job Verification
    salary_benchmark = ai_get_nested(sa, "market_benchmark") or get_salary_benchmark(job_title, location)
    responsibilities = ai_get_nested(sa, "responsibilities") or extract_responsibilities(job_description)
    requirements = ai_get_nested(sa, "requirements") or extract_requirements(job_description)
    salary_in_range = ai_get_nested(sa, "in_range") or check_salary_in_range(salary_text, job_title, location)
    no_unrealistic_salary = ai_get_nested(sa, "no_unrealistic_promises") or check_no_unrealistic_salary(salary_text, job_title)
    clear_structure = ai_get_nested(sa, "clear_structure") or check_clear_structure(job_description)
    job_structure_complete = ai_get_nested(sa, "job_structure_complete") or check_job_structure_complete(job_description)

    # Grammar & Content Analysis
    grammar_errors = ai_get_nested(ga, "grammar_errors") if ai_get_nested(ga, "grammar_errors") is not None else 0
    spacing_issues = ai_get_nested(ga, "spacing_issues") if ai_get_nested(ga, "spacing_issues") is not None else 0
    suspicious_phrases = ai_get_nested(ga, "suspicious_phrases") if ai_get_nested(ga, "suspicious_phrases") is not None else 0
    urgency_language = ai_get_nested(ga, "urgency_indicators") if ai_get_nested(ga, "urgency_indicators") is not None else 0
    grammar_examples = ai_get_nested(ga, "examples") or []
    content_assessment = ai_get_nested(ga, "content_assessment") or generate_content_assessment(grammar_errors, spacing_issues, suspicious_phrases, urgency_language)

    # Scam/Legit Pattern Detection
    suspicious_signals = ai_get_nested(pd, "suspicious_signals") or detect_suspicious_signals(job_title, company_name, job_description, job_url, recruiter_email, combined_text)
    legitimate_signals = ai_get_nested(pd, "legitimate_signals") or detect_legitimate_signals(job_title, company_name, job_description, job_url, recruiter_email, combined_text)

    # Final Recommendations
    final_recommendation = ai_get_nested(fr, "executive_recommendation") or generate_final_recommendation(risk_tier, risk_score, checks)
    immediate_precautions = ai_get_nested(fr, "immediate_precautions") or generate_immediate_precautions(risk_tier, checks)
    verification_steps = ai_get_nested(fr, "verification_steps") or generate_verification_steps(risk_tier, checks)

    return AnalysisResult(
        # Core results
        risk_score=risk_score,
        risk_tier=risk_tier,
        checks=checks,
        summary=summary,
        recommendation=recommendation,
        model_version=_metrics.get('model_version', '1.0.0'),

        # Company Research & Verification
        company_website=company_website,
        domain_age=domain_age,
        linkedin_company=linkedin_company,
        glassdoor_company=glassdoor_company,
        company_registration_status=company_registration_status,
        domain_ownership_status=domain_ownership_status,
        physical_address_status=physical_address_status,
        years_operation_status=years_operation_status,

        # Location Verification
        company_locations=company_locations,
        job_location=job_location,
        user_location=user_location,
        work_type=work_type,
        location_match_status=location_match_status,
        location_match_details=location_match_details,
        location_analysis_notes=location_analysis_notes,

        # Contact Verification
        recruiter_name=recruiter_name,
        email_domain_verified=email_domain_verified,
        recruiter_linkedin=recruiter_linkedin,
        linkedin_verified=linkedin_verified,
        directory_match=directory_match,
        official_channel=official_channel,

        # Salary & Job Verification
        salary_text=salary_text,
        salary_benchmark=salary_benchmark,
        responsibilities=responsibilities,
        requirements=requirements,
        salary_in_range=salary_in_range,
        no_unrealistic_salary=no_unrealistic_salary,
        clear_structure=clear_structure,
        job_structure_complete=job_structure_complete,

        # Grammar & Content Analysis
        grammar_errors=grammar_errors,
        spacing_issues=spacing_issues,
        suspicious_phrases=suspicious_phrases,
        urgency_language=urgency_language,
        grammar_examples=grammar_examples,
        content_assessment=content_assessment,

        # Scam/Legit Pattern Detection
        suspicious_signals=suspicious_signals,
        legitimate_signals=legitimate_signals,

        # Final Recommendations
        final_recommendation=final_recommendation,
        immediate_precautions=immediate_precautions,
        verification_steps=verification_steps,
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


# --- Minimal stub functions for detailed analysis (Phase 4) ---
# These return sensible defaults so scan_detail.html renders without errors

def extract_company_website(company_name, job_description, job_url):
    if job_url:
        from urllib.parse import urlparse
        try:
            parsed = urlparse(job_url)
            return parsed.netloc
        except Exception:
            pass
    if company_name:
        return f"https://{company_name.lower().replace(' ', '')}.com"
    return "Not available from input"

def estimate_domain_age(company_website):
    if not company_website:
        return None
    return None  # Domain age check requires WHOIS lookup (not available offline)

def find_linkedin_company(company_name):
    if not company_name:
        return "No company name provided"
    return f"https://linkedin.com/company/{company_name.lower().replace(' ', '-')}"

def find_glassdoor_company(company_name):
    if not company_name:
        return "No company name provided"
    return f"https://glassdoor.com/Reviews/{company_name.replace(' ', '-')}-Reviews"

def check_company_registration(company_name):
    if not company_name:
        return "No company name provided"
    return "Registration verification requires government database access (not available offline)"

def check_domain_ownership(company_name, company_website):
    if not company_name or not company_website:
        return "Insufficient data"
    return "Domain ownership verification requires WHOIS lookup (not available offline)"

def check_physical_address(job_description, company_name):
    if not company_name:
        return "No company name provided"
    return "Physical address verification requires external database (not available offline)"

def check_years_operation(company_name, domain_age):
    if not company_name:
        return "No company name provided"
    return "Years of operation check requires business registry access (not available offline)"

def extract_company_locations(job_description, company_name):
    if not company_name:
        return "No company name provided"
    return "Company location extraction requires external database (not available offline)"

def extract_job_location(job_description):
    if not job_description:
        return "Not specified in job posting"
    # Simple location extraction
    import re
    locations = re.findall(r'\b(?:remote|hybrid|on.?site|work from home|bangalore|mumbai|delhi|hyderabad|pune|chennai|gurgaon|noida|kolkata|ahmedabad)\b', job_description.lower())
    if locations:
        return ", ".join(set(locations))
    return "Location not specified in job description"

def determine_work_type(job_description, employment_type):
    if employment_type:
        return employment_type.lower().replace(" ", "_")
    text = job_description.lower()
    if "remote" in text or "work from home" in text:
        return "remote"
    elif "hybrid" in text:
        return "hybrid"
    return "on_site"

def analyze_location_match(company_locations, job_location, user_location, work_type):
    if work_type == "remote":
        return "pass", "Remote position - location match not applicable", "Remote work allows flexibility regardless of location"
    if not job_location:
        return "info", "Job location not specified", "Cannot verify location match without job location"
    return "info", "Location comparison completed", "Manual verification recommended for on-site/hybrid roles"

def extract_recruiter_name(job_description, raw_text):
    return "Not extracted from posting (requires NLP parsing)"

def verify_email_domain(recruiter_email, company_name, company_website):
    if not recruiter_email or '@' not in recruiter_email:
        return False
    domain = recruiter_email.split('@')[-1].lower()
    free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'icloud.com', 'protonmail.com']
    return domain not in free_domains

def find_recruiter_linkedin(recruiter_name, company_name):
    if not recruiter_name:
        return "No recruiter name provided"
    return f"https://linkedin.com/search/results/people/?keywords={recruiter_name.replace(' ', '%20')}%20{company_name.replace(' ', '%20')}"

def verify_linkedin_profile(recruiter_linkedin, company_name):
    if not recruiter_linkedin:
        return "No LinkedIn profile provided"
    return "LinkedIn verification requires API access (not available offline)"

def check_employee_directory(recruiter_name, company_name):
    if not recruiter_name or not company_name:
        return "Insufficient data"
    return "Employee directory check requires company internal access (not available offline)"

def check_official_channel(job_description, recruiter_email, recruiter_phone):
    text = job_description.lower()
    if 'whatsapp' in text or 'telegram' in text:
        return "fail"
    if recruiter_email and '@' in recruiter_email:
        domain = recruiter_email.split('@')[-1].lower()
        free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'icloud.com', 'protonmail.com']
        if domain in free_domains:
            return "warning"
    return "pass"

def get_salary_benchmark(job_title, location):
    if not job_title:
        return "No job title provided"
    return f"Salary benchmark for '{job_title}' in {location or 'India'} requires market data (not available offline). Typical range: ₹5-25 LPA for tech roles."

def extract_responsibilities(job_description):
    if not job_description:
        return "No job description provided"
    return "Responsibility extraction requires NLP parsing (not available offline). Please review the job description manually."

def extract_requirements(job_description):
    if not job_description:
        return "No job description provided"
    return "Requirements extraction requires NLP parsing (not available offline). Please review the job description manually."

def check_salary_in_range(salary_text, job_title, location):
    if not salary_text:
        return "No salary information provided in posting"
    return "Salary range validation requires market benchmark data (not available offline)"

def check_no_unrealistic_salary(salary_text, job_title):
    if not salary_text:
        return "No salary information provided"
    import re
    # Check for unrealistic numbers
    numbers = re.findall(r'[\d,]+', salary_text.replace('LPA', '').replace('lpa', '').replace('Lakhs', '').replace('lakhs', ''))
    for n in numbers:
        try:
            val = int(n.replace(',', ''))
            if val > 100:  # More than 100 LPA is suspicious for most roles
                return "fail"
        except ValueError:
            pass
    return "pass"

def check_clear_structure(job_description):
    if not job_description:
        return "No job description provided"
    sections = ['responsibilit', 'requirement', 'qualification', 'skill', 'experience', 'about', 'benefit', 'salary']
    found = sum(1 for s in sections if s in job_description.lower())
    if found >= 4:
        return "pass"
    elif found >= 2:
        return "warning"
    return "fail"

def check_job_structure_complete(job_description):
    if not job_description:
        return "No job description provided"
    required = ['responsibilit', 'requirement', 'qualification']
    found = sum(1 for r in required if r in job_description.lower())
    if found >= 3:
        return "pass"
    elif found >= 1:
        return "warning"
    return "fail"

def analyze_grammar_and_content(combined_text):
    if not combined_text:
        return 0, 0, 0, 0, []
    import re
    # Simple heuristic checks
    grammar_errors = len(re.findall(r'\b(?:teh|recieve|seperate|definately|occured|accomodate|buisness)\b', combined_text.lower()))
    spacing_issues = len(re.findall(r'[.!?]{2,}', combined_text)) + len(re.findall(r'\s{3,}', combined_text))
    suspicious_phrases = len(re.findall(r'\b(?:earn money|work from home|no experience|guaranteed income|easy money|quick cash)\b', combined_text.lower()))
    urgency_language = len(re.findall(r'\b(?:urgent|immediate|asap|today|limited|expire|hurry|quick|instant|24.?hour|48.?hour)\b', combined_text.lower()))
    examples = []
    if grammar_errors:
        examples.append("Common typos detected (e.g., 'teh' instead of 'the')")
    if spacing_issues:
        examples.append("Multiple spaces or punctuation issues found")
    if suspicious_phrases:
        examples.append("Phrases commonly found in scam postings detected")
    if urgency_language:
        examples.append("Urgency language used to pressure applicants")
    return grammar_errors, spacing_issues, suspicious_phrases, urgency_language, examples

def generate_content_assessment(grammar_errors, spacing_issues, suspicious_phrases, urgency_language):
    issues = []
    if grammar_errors > 0:
        issues.append(f"{grammar_errors} potential grammar/typo issues")
    if spacing_issues > 0:
        issues.append(f"{spacing_issues} spacing/formatting issues")
    if suspicious_phrases > 0:
        issues.append(f"{suspicious_phrases} suspicious phrases")
    if urgency_language > 0:
        issues.append(f"{urgency_language} urgency indicators")
    if not issues:
        return "Content appears well-structured with no obvious red flags in language or formatting."
    return f"Content quality concerns: {', '.join(issues)}. Legitimate postings typically have professional formatting and language."

def detect_suspicious_signals(job_title, company_name, job_description, job_url, recruiter_email, combined_text):
    signals = []
    text = combined_text.lower()
    if any(w in text for w in ['whatsapp', 'telegram', 'signal', 'viber']):
        signals.append({"type": "communication", "signal": "Requests communication via informal chat apps", "severity": "high"})
    if any(w in text for w in ['fee', 'deposit', 'payment', 'charge', 'investment', 'registration fee']):
        signals.append({"type": "financial", "signal": "Requests upfront payment or fees", "severity": "critical"})
    if any(w in text for w in ['ssn', 'social security', 'bank account', 'passport', 'aadhaar', 'pan card']):
        signals.append({"type": "privacy", "signal": "Requests sensitive personal information upfront", "severity": "critical"})
    if recruiter_email and '@' in recruiter_email:
        domain = recruiter_email.split('@')[-1].lower()
        free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'icloud.com', 'protonmail.com']
        if domain in free_domains:
            signals.append({"type": "identity", "signal": "Recruiter uses free email domain instead of corporate", "severity": "high"})
    if any(w in text for w in ['task', 'data entry', 'copy paste', 'typing', 'form filling', 'captcha', 'click', 'like', 'subscribe']):
        signals.append({"type": "task_scam", "signal": "Contains keywords typical of task/click scams", "severity": "high"})
    return signals

def detect_legitimate_signals(job_title, company_name, job_description, job_url, recruiter_email, combined_text):
    signals = []
    text = combined_text.lower()
    if job_url and ('linkedin.com' in job_url or 'naukri.com' in job_url or 'indeed.com' in job_url):
        signals.append({"type": "platform", "signal": "Posted on recognized job platform", "severity": "positive"})
    if recruiter_email and '@' in recruiter_email:
        domain = recruiter_email.split('@')[-1].lower()
        free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'icloud.com', 'protonmail.com']
        if domain not in free_domains:
            signals.append({"type": "identity", "signal": "Recruiter uses corporate email domain", "severity": "positive"})
    if any(w in text for w in ['benefits', 'insurance', 'pf', 'provident fund', 'gratuity', 'leave policy', 'notice period']):
        signals.append({"type": "employment", "signal": "Mentions standard employment benefits and policies", "severity": "positive"})
    if any(w in text for w in ['experience', 'years of experience', 'qualification', 'degree', 'certification']):
        signals.append({"type": "requirements", "signal": "Specifies concrete experience/qualification requirements", "severity": "positive"})
    if company_name and len(company_name) > 3:
        signals.append({"type": "company", "signal": "Company name provided (verify independently)", "severity": "positive"})
    return signals

def generate_final_recommendation(risk_tier, risk_score, checks):
    if risk_tier == "risky":
        return "❌ DO NOT APPLY. Multiple critical red flags detected."
    elif risk_tier == "caution":
        return "⚠️ VERIFY FIRST. Several warning signs found. Verify independently."
    return "✅ PROCEED WITH NORMAL CAUTION. Posting appears legitimate."

def generate_immediate_precautions(risk_tier, checks):
    if risk_tier == "risky":
        return ["Do not share personal information", "Do not pay any fees", "Report the posting", "Monitor accounts if info shared"]
    elif risk_tier == "caution":
        return ["Verify company independently", "Check official careers page", "Never pay to apply", "Protect personal info"]
    return ["Verify company on official website", "Use standard application safety", "Never pay to apply"]

def generate_verification_steps(risk_tier, checks):
    return ["Check official careers page", "Verify recruiter on LinkedIn", "Search company + 'scam'", "Check domain age on WHOIS"]