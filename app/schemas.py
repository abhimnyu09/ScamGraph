from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List


class CheckResult(BaseModel):
    name: str
    check_status: str
    message: str
    details: str = ""
    weight: float = 1.0


class AnalysisResult(BaseModel):
    risk_score: int
    risk_tier: str
    checks: List[CheckResult]
    summary: str
    recommendation: str
    model_version: str = "1.0.0"


class TelegramMessage(BaseModel):
    chat_id: int
    text: str
    parse_mode: str = "HTML"


# Plain dict-based input for API (avoids pydantic v1 Optional issues)
def validate_job_input(data: dict) -> dict:
    """Manually validate job input data."""
    errors = []

    # Required fields
    job_title = data.get('job_title', '').strip()
    if not job_title:
        errors.append('job_title is required')
    elif len(job_title) > 200:
        errors.append('job_title too long (max 200 chars)')

    company_name = data.get('company_name', '').strip()
    if not company_name:
        errors.append('company_name is required')
    elif len(company_name) > 200:
        errors.append('company_name too long (max 200 chars)')

    job_description = data.get('job_description', '').strip()
    if not job_description:
        errors.append('job_description is required')
    elif len(job_description) < 50:
        errors.append('job_description too short (min 50 chars)')
    elif len(job_description) > 20000:
        errors.append('job_description too long (max 20000 chars)')

    # Optional fields
    job_url = data.get('job_url', '').strip()
    if job_url and not (job_url.startswith('http://') or job_url.startswith('https://')):
        errors.append('job_url must start with http:// or https://')

    recruiter_email = data.get('recruiter_email', '').strip()
    if recruiter_email and '@' not in recruiter_email:
        errors.append('recruiter_email must be a valid email')

    if errors:
        raise ValueError('; '.join(errors))

    return {
        'job_title': job_title,
        'company_name': company_name,
        'job_description': job_description,
        'job_url': job_url,
        'recruiter_name': data.get('recruiter_name', '').strip(),
        'recruiter_email': recruiter_email,
        'recruiter_phone': data.get('recruiter_phone', '').strip(),
        'location': data.get('location', '').strip(),
        'salary_text': data.get('salary_text', '').strip(),
        'employment_type': data.get('employment_type', '').strip(),
        'raw_text': data.get('raw_text', '').strip(),
    }