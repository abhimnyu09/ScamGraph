import json
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any


async def extract_naukri_job(url: str) -> Dict[str, Any]:
    """Extract job details from Naukri job posting page."""
    result = {
        'job_title': '',
        'company_name': '',
        'job_description': '',
        'location': '',
        'experience': '',
        'salary': '',
        'skills': '',
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

        # Try JSON-LD
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
                    result['success'] = True
                    return result
            except Exception:
                pass

        # Meta tags
        title_meta = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'title'})
        if title_meta and title_meta.get('content'):
            result['job_title'] = title_meta['content']

        desc_meta = soup.find('meta', property='og:description') or soup.find('meta', attrs={'name': 'description'})
        if desc_meta and desc_meta.get('content'):
            result['job_description'] = desc_meta['content']

        # Naukri-specific selectors
        if not result['job_title']:
            title_el = soup.select_one('.jd-header-title, .job-title, h1.styles_jdcTitle__3hWCC')
            if title_el:
                result['job_title'] = title_el.get_text(strip=True)

        if not result['company_name']:
            company_el = soup.select_one('.jd-header-comp-name, .company-name, a.styles_jdcCompanyName__1rJCS')
            if company_el:
                result['company_name'] = company_el.get_text(strip=True)

        if not result['job_description']:
            desc_el = soup.select_one('.job-desc, .JDdescription, div.styles_JDC__description__2S_L_')
            if desc_el:
                result['job_description'] = desc_el.get_text(strip=True, separator=' ')

        # Location, experience, salary from info chips
        info_chips = soup.select('.jd-header-info .info-chip, .styles_jdcInfoChip__1rJCS')
        for chip in info_chips:
            text = chip.get_text(strip=True)
            if 'year' in text.lower() or 'exp' in text.lower():
                result['experience'] = text
            elif '₹' in text or 'lac' in text.lower() or 'lpa' in text.lower():
                result['salary'] = text
            elif any(city in text.lower() for city in ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'pune', 'kolkata', 'gurgaon', 'noida', 'remote']):
                result['location'] = text

        # Skills
        skills_els = soup.select('.key-skill, .chip, .styles_chip__3aHxV')
        skills = [s.get_text(strip=True) for s in skills_els if s.get_text(strip=True)]
        result['skills'] = ', '.join(skills[:20])

        result['success'] = bool(result['job_title'] or result['job_description'])

    except httpx.HTTPStatusError as e:
        result['error'] = f"HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        result['error'] = "Timeout"
    except Exception as e:
        result['error'] = str(e)

    return result


def extract_naukri_job_sync(url: str) -> Dict[str, Any]:
    """Synchronous wrapper."""
    import asyncio
    return asyncio.run(extract_naukri_job(url))