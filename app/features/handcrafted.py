import re
import numpy as np
from urllib.parse import urlparse
from typing import List, Dict, Any


# --- Heuristic feature extraction ---

URGENCY_PATTERNS = [
    r"urgent", r"immediate", r"asap", r"today only", r"limited time",
    r"expire", r"deadline", r"hurry", r"quick", r"fast", r"instant",
    r"24\s*hour", r"48\s*hour", r"this week", r"right now"
]

MONEY_REQUEST_PATTERNS = [
    r"registration fee", r"training fee", r"processing fee", r"admin fee",
    r"security deposit", r"upfront", r"pay.*first", r"deposit.*required",
    r"fee.*refund", r"charge.*fee", r"payment.*required", r"cost.*involved",
    r"investment", r"pay.*to.*work", r"buy.*kit", r"material.*cost"
]

PERSONAL_INFO_PATTERNS = [
    r"\bssn\b", r"social security", r"bank account", r"routing number",
    r"credit card", r"passport", r"aadhaar", r"pan card", r"id proof",
    r"address proof", r"bank statement", r"salary slip"
]

CONTACT_PATTERNS = [
    r"whatsapp", r"telegram", r"signal", r"viber", r"skype",
    r"gmail\.com", r"yahoo\.com", r"outlook\.com", r"hotmail\.com",
    r"@.*\.(com|net|org|info|xyz|top|club)"
]

SCAM_KEYWORDS = [
    "task", "data entry", "copy paste", "typing", "form filling",
    "captcha", "survey", "click", "like", "subscribe", "review",
    "rating", "app testing", "website testing", "product review",
    "work from home", "wfh", "part time", "flexible hours",
    "no experience", "fresher", "student", "housewife",
    "daily payment", "weekly payment", "instant payment"
]

LEGIT_KEYWORDS = [
    "experience required", "qualification", "degree", "certification",
    "responsibilities", "requirements", "benefits", "insurance",
    "provident fund", "gratuity", "leave policy", "notice period",
    "probation", "appraisal", "career growth", "learning"
]


def extract_handcrafted_features(text: str, url: str = "", email: str = "") -> np.ndarray:
    """Extract 20+ heuristic features from job posting text."""
    text_lower = text.lower()
    features = []

    # 1. Text length
    features.append(len(text))

    # 2. Word count
    features.append(len(text.split()))

    # 3. Urgency score (0-1)
    urgency_count = sum(1 for p in URGENCY_PATTERNS if re.search(p, text_lower))
    features.append(min(urgency_count / 5.0, 1.0))

    # 4. Money request score (0-1)
    money_count = sum(1 for p in MONEY_REQUEST_PATTERNS if re.search(p, text_lower))
    features.append(min(money_count / 3.0, 1.0))

    # 5. Personal info request score (0-1)
    pii_count = sum(1 for p in PERSONAL_INFO_PATTERNS if re.search(p, text_lower))
    features.append(min(pii_count / 3.0, 1.0))

    # 6. Suspicious contact methods (0-1)
    contact_count = sum(1 for p in CONTACT_PATTERNS if re.search(p, text_lower))
    features.append(min(contact_count / 3.0, 1.0))

    # 7. Scam keyword density
    scam_count = sum(1 for kw in SCAM_KEYWORDS if kw in text_lower)
    features.append(min(scam_count / 10.0, 1.0))

    # 8. Legit keyword density
    legit_count = sum(1 for kw in LEGIT_KEYWORDS if kw in text_lower)
    features.append(min(legit_count / 10.0, 1.0))

    # 9. All-caps ratio
    words = text.split()
    if words:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
    else:
        caps_ratio = 0
    features.append(caps_ratio)

    # 10. Exclamation mark count
    features.append(min(text.count('!') / 10.0, 1.0))

    # 11. Dollar/Rupee symbol count
    features.append(min((text.count('$') + text.count('₹') + text.count('rs') + text.count('INR')) / 5.0, 1.0))

    # 12. Number density (salary amounts, etc.)
    numbers = re.findall(r'\d+', text)
    features.append(min(len(numbers) / 20.0, 1.0))

    # 13. URL presence in text
    features.append(1.0 if re.search(r'https?://', text) else 0.0)

    # 14. Email presence in text
    features.append(1.0 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0.0)

    # 15. Phone number presence
    features.append(1.0 if re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text) else 0.0)

    # 16. Free email domain in recruiter email
    free_email = 0.0
    if email:
        free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com', 'icloud.com', 'protonmail.com']
        domain = email.split('@')[-1].lower() if '@' in email else ''
        free_email = 1.0 if domain in free_domains else 0.0
    features.append(free_email)

    # 17. Domain age proxy (from URL) - placeholder, real check in url_features
    features.append(0.0)

    # 18. LinkedIn/Naukri/Indeed domain in URL
    job_board = 0.0
    if url:
        job_boards = ['linkedin.com', 'naukri.com', 'indeed.com', 'glassdoor.com', 'foundit.in', 'shine.com', 'timesjobs.com']
        job_board = 1.0 if any(jb in url.lower() for jb in job_boards) else 0.0
    features.append(job_board)

    # 19. Company name in text (self-reference)
    # Will be filled dynamically in predict.py
    features.append(0.0)

    # 20. Bullet point structure (legit jobs often have structured bullets)
    bullet_count = text.count('•') + text.count('●') + text.count('▪') + text.count('- ') + text.count('* ')
    features.append(min(bullet_count / 10.0, 1.0))

    return np.array(features, dtype=np.float32).reshape(1, -1)


def get_feature_names() -> List[str]:
    return [
        'text_length', 'word_count', 'urgency_score', 'money_request_score',
        'pii_request_score', 'suspicious_contact_score', 'scam_keyword_density',
        'legit_keyword_density', 'caps_ratio', 'exclamation_count',
        'currency_symbol_count', 'number_density', 'url_in_text',
        'email_in_text', 'phone_in_text', 'free_email_domain',
        'domain_age_proxy', 'job_board_domain', 'company_self_ref',
        'bullet_structure'
    ]