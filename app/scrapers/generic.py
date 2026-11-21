import json
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any


async def extract_generic_job(url: str) -> Dict[str, Any]:
    """Generic job extraction for company career pages (Greenhouse, Lever, Workday, custom)."""
    result = {
        'job_title': '',
        'company_name': '',
        'job_description': '',
        'location': '',
        'employment_type': '',
        'raw_html': '',
        'success': False,
        'error': None,
        'ats_type': 'unknown'
    }

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            result['raw_html'] = html[:50000]

        soup = BeautifulSoup(html, 'lxml')

        # Detect ATS type from URL or page content
        url_lower = url.lower()
        if 'greenhouse.io' in url_lower:
            result['ats_type'] = 'greenhouse'
        elif 'lever.co' in url_lower:
            result['ats_type'] = 'lever'
        elif 'workday' in url_lower or 'myworkdayjobs' in url_lower:
            result['ats_type'] = 'workday'
        elif 'smartrecruiters' in url_lower:
            result['ats_type'] = 'smartrecruiters'
        elif 'bamboohr' in url_lower:
            result['ats_type'] = 'bamboohr'
        elif 'icims' in url_lower:
            result['ats_type'] = 'icims'
        elif 'jobvite' in url_lower:
            result['ats_type'] = 'jobvite'

        # Try JSON-LD (works for most ATS)
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

        # ATS-specific selectors
        if result['ats_type'] == 'greenhouse':
            result.update(_extract_greenhouse(soup))
        elif result['ats_type'] == 'lever':
            result.update(_extract_lever(soup))
        elif result['ats_type'] == 'workday':
            result.update(_extract_workday(soup))
        else:
            # Generic fallback
            result.update(_extract_generic_fallback(soup))

        result['success'] = bool(result['job_title'] or result['job_description'])

    except httpx.HTTPStatusError as e:
        result['error'] = f"HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        result['error'] = "Timeout"
    except Exception as e:
        result['error'] = str(e)

    return result


def _extract_greenhouse(soup: BeautifulSoup) -> Dict[str, Any]:
    result = {}
    title = soup.select_one('.app-title, h1#job-title, .job-title')
    if title:
        result['job_title'] = title.get_text(strip=True)
    company = soup.select_one('.company-name, .app-company')
    if company:
        result['company_name'] = company.get_text(strip=True)
    desc = soup.select_one('#job_description, .job-description, .content')
    if desc:
        result['job_description'] = desc.get_text(strip=True, separator=' ')
    loc = soup.select_one('.location, .job-location')
    if loc:
        result['location'] = loc.get_text(strip=True)
    return result


def _extract_lever(soup: BeautifulSoup) -> Dict[str, Any]:
    result = {}
    title = soup.select_one('.posting-headline h2, h1.posting-title')
    if title:
        result['job_title'] = title.get_text(strip=True)
    company = soup.select_one('.posting-company, .company-name')
    if company:
        result['company_name'] = company.get_text(strip=True)
    desc = soup.select_one('.posting-description, .description, .content')
    if desc:
        result['job_description'] = desc.get_text(strip=True, separator=' ')
    loc = soup.select_one('.posting-categories .location, .location')
    if loc:
        result['location'] = loc.get_text(strip=True)
    return result


def _extract_workday(soup: BeautifulSoup) -> Dict[str, Any]:
    result = {}
    # Workday is heavily JS-rendered; static HTML often lacks content
    # Try meta tags
    title_meta = soup.find('meta', property='og:title')
    if title_meta and title_meta.get('content'):
        result['job_title'] = title_meta['content']
    desc_meta = soup.find('meta', property='og:description')
    if desc_meta and desc_meta.get('content'):
        result['job_description'] = desc_meta['content']
    return result


def _extract_generic_fallback(soup: BeautifulSoup) -> Dict[str, Any]:
    result = {}
    # Common patterns across career sites
    title_selectors = [
        'h1.job-title', 'h1.title', '.job-title', '.position-title',
        '[data-testid="job-title"]', '.posting-title'
    ]
    for sel in title_selectors:
        el = soup.select_one(sel)
        if el:
            result['job_title'] = el.get_text(strip=True)
            break

    company_selectors = [
        '.company-name', '.employer-name', '.organization-name',
        '[data-testid="company-name"]'
    ]
    for sel in company_selectors:
        el = soup.select_one(sel)
        if el:
            result['company_name'] = el.get_text(strip=True)
            break

    desc_selectors = [
        '.job-description', '.description', '.job-details',
        '[data-testid="job-description"]', '.content', 'main'
    ]
    for sel in desc_selectors:
        el = soup.select_one(sel)
        if el:
            result['job_description'] = el.get_text(strip=True, separator=' ')
            break

    return result


def extract_generic_job_sync(url: str) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(extract_generic_job(url))