import re
import json
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any


async def extract_linkedin_job(url: str) -> Dict[str, Any]:
    """Extract job details from LinkedIn job posting page."""
    result = {
        'job_title': '',
        'company_name': '',
        'job_description': '',
        'location': '',
        'employment_type': '',
        'seniority_level': '',
        'job_function': '',
        'industries': '',
        'raw_html': '',
        'success': False,
        'error': None
    }

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            result['raw_html'] = html[:50000]

        soup = BeautifulSoup(html, 'lxml')

        # Try JSON-LD first (most reliable)
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, list):
                    data = data[0]
                if data.get('@type') == 'JobPosting':
                    result['job_title'] = data.get('title', '')
                    result['company_name'] = data.get('hiringOrganization', {}).get('name', '')
                    result['job_description'] = data.get('description', '')
                    location = data.get('jobLocation', {})
                    if isinstance(location, dict):
                        address = location.get('address', {})
                        if isinstance(address, dict):
                            result['location'] = address.get('addressLocality', '')
                    result['employment_type'] = data.get('employmentType', '')
                    result['success'] = True
                    return result
            except Exception:
                pass

        # Fallback: Meta tags
        title_meta = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'title'})
        if title_meta and title_meta.get('content'):
            result['job_title'] = title_meta['content']

        desc_meta = soup.find('meta', property='og:description') or soup.find('meta', attrs={'name': 'description'})
        if desc_meta and desc_meta.get('content'):
            result['job_description'] = desc_meta['content']

        # Try to find company name in various places
        company_selectors = [
            'a[data-tracking-control-name="public_jobs_topcard-org-name"]',
            '.topcard__org-name-link',
            '.job-details-jobs-unified-top-card__company-name',
            'span.job-details-jobs-unified-top-card__company-name',
        ]
        for sel in company_selectors:
            el = soup.select_one(sel)
            if el:
                result['company_name'] = el.get_text(strip=True)
                break

        # Try to get full description from job description section
        desc_selectors = [
            '.jobs-description__content',
            '.job-details-jobs-unified-top-card__job-description',
            '#job-details',
            '.description__text',
        ]
        for sel in desc_selectors:
            el = soup.select_one(sel)
            if el:
                result['job_description'] = el.get_text(strip=True, separator=' ')
                break

        # Location
        loc_selectors = [
            '.job-details-jobs-unified-top-card__bullet',
            '.topcard__flavor--bullet',
        ]
        for sel in loc_selectors:
            els = soup.select(sel)
            for el in els:
                text = el.get_text(strip=True)
                if text and not any(kw in text.lower() for kw in ['full-time', 'part-time', 'contract', 'internship', 'remote', 'hybrid']):
                    result['location'] = text
                    break

        result['success'] = bool(result['job_title'] or result['job_description'])

    except httpx.HTTPStatusError as e:
        result['error'] = f"HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        result['error'] = "Timeout"
    except Exception as e:
        result['error'] = str(e)

    return result


def extract_linkedin_job_sync(url: str) -> Dict[str, Any]:
    """Synchronous wrapper for non-async contexts."""
    import asyncio
    return asyncio.run(extract_linkedin_job(url))