import re
import httpx
import tldextract
import whois
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import asyncio


# --- URL/Domain feature extraction ---

TRUSTED_JOB_BOARDS = {
    'linkedin.com', 'naukri.com', 'indeed.com', 'glassdoor.com',
    'foundit.in', 'shine.com', 'timesjobs.com', 'iimjobs.com',
    'hirist.com', 'cutshort.io', 'wellfound.com', 'angel.co'
}

TRUSTED_ATS_DOMAINS = {
    'greenhouse.io', 'lever.co', 'workday.com', 'smartrecruiters.com',
    'bamboohr.com', 'icims.com', 'jobvite.com', 'taleo.net',
    'successfactors.com', 'oraclecloud.com', 'sap.com'
}

FREE_EMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
    'live.com', 'icloud.com', 'protonmail.com', 'yandex.com',
    'zoho.com', 'mail.com', 'gmx.com', 'aol.com', 'rediffmail.com'
}


async def extract_url_features(url: str) -> Dict[str, Any]:
    """Extract domain/URL features for a job posting link."""
    features = {
        'is_job_board': 0.0,
        'is_ats_domain': 0.0,
        'domain_age_days': -1,
        'ssl_valid': 0.0,
        'has_www': 0.0,
        'subdomain_count': 0,
        'url_length': len(url),
        'has_suspicious_tld': 0.0,
        'redirect_count': 0,
        'final_url_same_domain': 1.0,
    }

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace('www.', '')
        extracted = tldextract.extract(url)
        registered_domain = f"{extracted.domain}.{extracted.suffix}"

        # Check if known job board
        features['is_job_board'] = 1.0 if any(jb in domain for jb in TRUSTED_JOB_BOARDS) else 0.0

        # Check if known ATS
        features['is_ats_domain'] = 1.0 if any(ats in domain for ats in TRUSTED_ATS_DOMAINS) else 0.0

        # WWW subdomain
        features['has_www'] = 1.0 if parsed.netloc.startswith('www.') else 0.0

        # Subdomain count
        features['subdomain_count'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0

        # Suspicious TLDs
        suspicious_tlds = {'xyz', 'top', 'club', 'online', 'site', 'space', 'website', 'info', 'biz', 'tk', 'ml', 'ga', 'cf'}
        features['has_suspicious_tld'] = 1.0 if extracted.suffix in suspicious_tlds else 0.0

        # Domain age via WHOIS (slow, cache in production)
        try:
            w = whois.whois(registered_domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date:
                age = (datetime.now() - creation_date).days
                features['domain_age_days'] = max(age, 0)
        except Exception:
            pass  # WHOIS failed, keep -1

        # SSL check via HEAD request (fast)
        try:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                resp = await client.head(url)
                features['ssl_valid'] = 1.0 if resp.url.scheme == 'https' else 0.0
                features['redirect_count'] = len(resp.history)
                features['final_url_same_domain'] = 1.0 if tldextract.extract(str(resp.url)).registered_domain == registered_domain else 0.0
        except Exception:
            pass

    except Exception:
        pass

    return features


async def extract_email_features(email: str) -> Dict[str, Any]:
    """Extract features from recruiter email."""
    features = {
        'is_free_email': 0.0,
        'domain_age_days': -1,
        'domain_matches_company': 0.0,
    }

    try:
        if '@' in email:
            domain = email.split('@')[-1].lower()
            features['is_free_email'] = 1.0 if domain in FREE_EMAIL_DOMAINS else 0.0

            # Domain age
            try:
                w = whois.whois(domain)
                creation_date = w.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                if creation_date:
                    features['domain_age_days'] = max((datetime.now() - creation_date).days, 0)
            except Exception:
                pass
    except Exception:
        pass

    return features


async def check_careers_page_match(company_name: str, job_title: str) -> Dict[str, Any]:
    """
    Check if job exists on company's official careers page.
    This is the 'single strongest signal' per JobScamScore.
    Simplified version: search company careers page for job title.
    """
    result = {
        'careers_page_found': 0.0,
        'job_listed_on_careers': 0.0,
        'careers_url': '',
        'match_confidence': 0.0,
    }

    # Common careers page patterns
    company_slug = company_name.lower().replace(' ', '').replace('.', '').replace(',', '')
    careers_patterns = [
        f"https://{company_slug}.com/careers",
        f"https://{company_slug}.com/jobs",
        f"https://careers.{company_slug}.com",
        f"https://jobs.{company_slug}.com",
        f"https://{company_slug}.greenhouse.io",
        f"https://{company_slug}.lever.co",
    ]

    # Try to find a working careers page
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for url in careers_patterns:
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and len(resp.text) > 5000:
                    result['careers_page_found'] = 1.0
                    result['careers_url'] = url

                    # Check if job title appears on page
                    if job_title.lower() in resp.text.lower():
                        result['job_listed_on_careers'] = 1.0
                        result['match_confidence'] = 0.9
                    else:
                        result['match_confidence'] = 0.3
                    break
            except Exception:
                continue

    return result


def normalize_url_features(features: Dict[str, Any]) -> list:
    """Convert URL features dict to normalized list for model."""
    return [
        features.get('is_job_board', 0.0),
        features.get('is_ats_domain', 0.0),
        min(features.get('domain_age_days', -1) / 3650.0, 1.0) if features.get('domain_age_days', -1) > 0 else 0.0,
        features.get('ssl_valid', 0.0),
        features.get('has_www', 0.0),
        min(features.get('subdomain_count', 0) / 5.0, 1.0),
        min(features.get('url_length', 0) / 200.0, 1.0),
        features.get('has_suspicious_tld', 0.0),
        min(features.get('redirect_count', 0) / 5.0, 1.0),
        features.get('final_url_same_domain', 1.0),
    ]


def get_url_feature_names() -> list:
    return [
        'url_is_job_board', 'url_is_ats', 'url_domain_age_years',
        'url_ssl_valid', 'url_has_www', 'url_subdomain_count',
        'url_length_norm', 'url_suspicious_tld', 'url_redirect_count',
        'url_final_same_domain'
    ]