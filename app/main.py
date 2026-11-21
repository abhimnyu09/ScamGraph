import os
import logging
import json
import base64
import io
import secrets
import hashlib
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Dict, List

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from starlette.exceptions import HTTPException
from starlette.middleware.sessions import SessionMiddleware

from app.model.predict import predict_risk, load_model
from app.scrapers.linkedin import extract_linkedin_job
from app.scrapers.naukri import extract_naukri_job
from app.scrapers.generic import extract_generic_job
from app.telegram_bot import create_application, process_webhook_update, setup_webhook
from app.schemas import AnalysisResult, validate_job_input

# --- Config ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g., https://scamgraph.onrender.com
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- In-Memory Storage (replace with DB in production) ---
_users_db: Dict[str, dict] = {}  # email -> user data
_scan_history: Dict[str, List[dict]] = {}  # user_id -> list of scans
_session_store: Dict[str, dict] = {}  # session_id -> session data

# --- Auth Utilities ---
def hash_password(password: str) -> str:
    """Hash password with SHA-256 (use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

def get_user_from_session(request: Request) -> Optional[dict]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    for user in _users_db.values():
        if user["id"] == user_id:
            return user
    return None

def require_auth(request: Request) -> dict:
    user = get_user_from_session(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

def set_user_session(request: Request, user_id: str):
    """Set user ID in session."""
    request.session["user_id"] = user_id

def clear_user_session(request: Request):
    """Clear user session."""
    request.session.clear()

def save_scan_to_history(user_id: str, scan_data: dict):
    if user_id not in _scan_history:
        _scan_history[user_id] = []
    scan_record = {
        "id": secrets.token_urlsafe(16),
        "timestamp": datetime.utcnow().isoformat(),
        **scan_data
    }
    _scan_history[user_id].insert(0, scan_record)  # Most recent first
    # Keep only last 100 scans
    _scan_history[user_id] = _scan_history[user_id][:100]

def get_scan_history(user_id: str, limit: int = 50, offset: int = 0, filter_tier: str = None) -> List[dict]:
    scans = _scan_history.get(user_id, [])
    if filter_tier:
        scans = [s for s in scans if s.get("risk_tier") == filter_tier]
    return scans[offset:offset+limit]

# --- Telegram Bot Application (singleton) ---
_telegram_app = None


async def get_telegram_app():
    global _telegram_app
    if _telegram_app is None:
        _telegram_app = create_application()
        await _telegram_app.initialize()
        if WEBHOOK_URL:
            await setup_webhook(_telegram_app, WEBHOOK_URL)
    return _telegram_app


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: Starlette):
    # Startup
    load_model()
    logger.info("Model loaded")

    # Initialize Telegram bot if token present
    if TELEGRAM_BOT_TOKEN:
        try:
            await get_telegram_app()
            logger.info("Telegram bot initialized")
        except Exception as e:
            logger.warning(f"Telegram bot init failed: {e}")

    yield

    # Shutdown
    if _telegram_app:
        await _telegram_app.shutdown()
        logger.info("Telegram bot shutdown")


# --- Templates ---
templates = Jinja2Templates(directory="app/templates")


# --- Auth Routes ---
async def login_page(request: Request):
    """Login page."""
    user = get_user_from_session(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("auth/login.html", {"request": request})


async def register_page(request: Request):
    """Registration page."""
    user = get_user_from_session(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("auth/register.html", {"request": request})


async def login_post(request: Request):
    """Handle login form submission."""
    form = await request.form()
    email = form.get("email", "").lower().strip()
    password = form.get("password", "")
    remember = form.get("remember") == "on"

    user = _users_db.get(email)
    if not user or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse("auth/login.html", {
            "request": request,
            "error": "Invalid email or password",
            "email": email
        })

    set_user_session(request, user["id"])
    if remember:
        request.session.setdefault("permanent", True)

    return RedirectResponse(url="/dashboard", status_code=302)


async def register_post(request: Request):
    """Handle registration form submission."""
    form = await request.form()
    name = form.get("name", "").strip()
    email = form.get("email", "").lower().strip()
    password = form.get("password", "")
    confirm_password = form.get("confirm_password", "")

    if not name or not email or not password:
        return templates.TemplateResponse("auth/register.html", {
            "request": request,
            "error": "All fields are required",
            "name": name,
            "email": email
        })

    if password != confirm_password:
        return templates.TemplateResponse("auth/register.html", {
            "request": request,
            "error": "Passwords do not match",
            "name": name,
            "email": email
        })

    if len(password) < 8:
        return templates.TemplateResponse("auth/register.html", {
            "request": request,
            "error": "Password must be at least 8 characters",
            "name": name,
            "email": email
        })

    if email in _users_db:
        return templates.TemplateResponse("auth/register.html", {
            "request": request,
            "error": "Email already registered",
            "name": name,
            "email": email
        })

    user_id = secrets.token_urlsafe(16)
    _users_db[email] = {
        "id": user_id,
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.utcnow().isoformat(),
        "onboarded": False,
        "preferences": {
            "email_notifications": True,
            "scan_alerts": True
        }
    }

    set_user_session(request, user_id)

    return RedirectResponse(url="/onboarding", status_code=302)


async def logout(request: Request):
    """Handle logout."""
    clear_user_session(request)
    return RedirectResponse(url="/", status_code=302)


async def onboarding_page(request: Request):
    """Onboarding flow page."""
    user = get_user_from_session(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    if user.get("onboarded"):
        return RedirectResponse(url="/dashboard", status_code=302)

    step = int(request.query_params.get("step", "1"))
    return templates.TemplateResponse("auth/onboarding.html", {
        "request": request,
        "user": user,
        "step": step
    })


async def onboarding_post(request: Request):
    """Handle onboarding form submission."""
    user = get_user_from_session(request)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    form = await request.form()
    step = int(form.get("step", "1"))

    if step == 1:
        # Step 1: Role selection
        role = form.get("role", "")
        if role not in ["job_seeker", "recruiter", "student", "other"]:
            return templates.TemplateResponse("auth/onboarding.html", {
                "request": request,
                "user": user,
                "step": 1,
                "error": "Please select a role"
            })
        user["role"] = role
    elif step == 2:
        # Step 2: Preferences
        user["preferences"]["email_notifications"] = form.get("email_notifications") == "on"
        user["preferences"]["scan_alerts"] = form.get("scan_alerts") == "on"
        user["preferences"]["weekly_digest"] = form.get("weekly_digest") == "on"
    elif step == 3:
        # Step 3: Complete
        user["onboarded"] = True
        return RedirectResponse(url="/dashboard", status_code=302)

    _users_db[user["email"]] = user
    return RedirectResponse(url=f"/onboarding?step={step + 1}", status_code=302)


# --- Routes ---
async def health(request: Request):
    return JSONResponse({"status": "ok", "service": "scamgraph"})


async def partial_tab_text(request: Request):
    return templates.TemplateResponse("partials/tab_text.html", {"request": request})


async def partial_tab_url(request: Request):
    return templates.TemplateResponse("partials/tab_url.html", {"request": request})


async def index(request: Request):
    """Main page: job input form."""
    return templates.TemplateResponse("index.html", {"request": request})


async def analyze_form(request: Request):
    """Analyze job posting from form submission (HTMX)."""
    form = await request.form()

    job_title = form.get("job_title", "")
    company_name = form.get("company_name", "")
    job_description = form.get("job_description", "")
    job_url = form.get("job_url", "")
    recruiter_name = form.get("recruiter_name", "")
    recruiter_email = form.get("recruiter_email", "")
    recruiter_phone = form.get("recruiter_phone", "")
    location = form.get("location", "")
    salary_text = form.get("salary_text", "")
    employment_type = form.get("employment_type", "")
    raw_text = form.get("raw_text", "")

    # Use raw_text if provided and other fields empty
    if raw_text and not job_description.strip():
        job_description = raw_text

    result = predict_risk(
        job_title=job_title,
        company_name=company_name,
        job_description=job_description,
        job_url=job_url or "",
        recruiter_email=recruiter_email or "",
        recruiter_phone=recruiter_phone or "",
        location=location or "",
        salary_text=salary_text or "",
        employment_type=employment_type or "",
        raw_text=raw_text or ""
    )

    # Save to history if user is authenticated
    user = get_user_from_session(request)
    if user:
        scan_data = {
            "job_title": job_title,
            "company_name": company_name,
            "risk_score": result.risk_score,
            "risk_tier": result.risk_tier,
            "summary": result.summary,
            "job_url": job_url or "",
            "recruiter_email": recruiter_email or "",
            "checks": [
                {"name": c.name, "status": c.check_status, "message": c.message, "weight": c.weight}
                for c in result.checks
            ]
        }
        save_scan_to_history(user["id"], scan_data)

    return templates.TemplateResponse("partials/result_card.html", {"request": request, "result": result})


async def analyze_json(request: Request):
    """Analyze job posting via JSON API."""
    data = await request.json()

    # Validate manually
    validated = validate_job_input(data)

    result = predict_risk(
        job_title=validated['job_title'],
        company_name=validated['company_name'],
        job_description=validated['job_description'],
        job_url=validated['job_url'] or "",
        recruiter_email=validated['recruiter_email'] or "",
        recruiter_phone=validated['recruiter_phone'] or "",
        location=validated['location'] or "",
        salary_text=validated['salary_text'] or "",
        employment_type=validated['employment_type'] or "",
        raw_text=validated['raw_text'] or ""
    )

    # Convert to dict for JSON response
    return JSONResponse({
        "risk_score": result.risk_score,
        "risk_tier": result.risk_tier,
        "checks": [
            {
                "name": c.name,
                "status": c.check_status,
                "message": c.message,
                "details": c.details,
                "weight": c.weight
            }
            for c in result.checks
        ],
        "summary": result.summary,
        "recommendation": result.recommendation,
        "model_version": result.model_version
    })


async def extract_url(request: Request):
    """Extract job details from URL and prefill form (HTMX)."""
    form = await request.form()
    url = form.get("url", "")
    url_lower = url.lower()

    if 'linkedin.com' in url_lower:
        job_data = await extract_linkedin_job(url)
    elif 'naukri.com' in url_lower:
        job_data = await extract_naukri_job(url)
    else:
        job_data = await extract_generic_job(url)

    if not job_data.get('success'):
        return templates.TemplateResponse("partials/url_error.html", {
            "request": request,
            "error": job_data.get('error', 'Failed to extract job details'),
            "url": url
        })

    return templates.TemplateResponse("partials/prefilled_form.html", {
        "request": request,
        "job_data": job_data,
        "url": url
    })


async def scanner_page(request: Request):
    """Scanner page with 3 input modes."""
    user = get_user_from_session(request)
    return templates.TemplateResponse("scanner.html", {"request": request, "user": user})


async def dashboard_page(request: Request):
    """User dashboard with scan history summary."""
    user = require_auth(request)

    scans = _scan_history.get(user["id"], [])
    total_scans = len(scans)
    safe_count = sum(1 for s in scans if s.get("risk_tier") == "safe")
    caution_count = sum(1 for s in scans if s.get("risk_tier") == "caution")
    risky_count = sum(1 for s in scans if s.get("risk_tier") == "risky")

    recent_scans = scans[:5]

    return templates.TemplateResponse("dashboard/index.html", {
        "request": request,
        "user": user,
        "stats": {
            "total": total_scans,
            "safe": safe_count,
            "caution": caution_count,
            "risky": risky_count
        },
        "recent_scans": recent_scans
    })


async def scan_history_page(request: Request):
    """Scan history page with filtering and pagination."""
    user = require_auth(request)

    page = int(request.query_params.get("page", "1"))
    filter_tier = request.query_params.get("tier", "")
    per_page = 10
    offset = (page - 1) * per_page

    scans = get_scan_history(user["id"], limit=per_page, offset=offset, filter_tier=filter_tier if filter_tier else None)
    total_scans = len(_scan_history.get(user["id"], []))
    if filter_tier:
        total_scans = sum(1 for s in _scan_history.get(user["id"], []) if s.get("risk_tier") == filter_tier)

    total_pages = (total_scans + per_page - 1) // per_page

    return templates.TemplateResponse("dashboard/history.html", {
        "request": request,
        "user": user,
        "scans": scans,
        "page": page,
        "total_pages": total_pages,
        "filter_tier": filter_tier,
        "total_scans": total_scans
    })


async def scan_detail_page(request: Request):
    """Detailed analysis result page."""
    user = require_auth(request)
    scan_id = request.path_params.get("scan_id")

    scans = _scan_history.get(user["id"], [])
    scan = next((s for s in scans if s.get("id") == scan_id), None)

    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    return templates.TemplateResponse("dashboard/scan_detail.html", {
        "request": request,
        "user": user,
        "scan": scan
    })


async def report_scam_page(request: Request):
    """Report a scam page."""
    user = get_user_from_session(request)
    return templates.TemplateResponse("dashboard/report.html", {
        "request": request,
        "user": user
    })


async def report_scam_post(request: Request):
    """Handle scam report submission."""
    user = get_user_from_session(request)
    form = await request.form()

    # In production, save to database and notify moderators
    report = {
        "id": secrets.token_urlsafe(16),
        "reporter_id": user["id"] if user else "anonymous",
        "job_title": form.get("job_title", ""),
        "company_name": form.get("company_name", ""),
        "job_url": form.get("job_url", ""),
        "description": form.get("description", ""),
        "scam_type": form.get("scam_type", ""),
        "contact_method": form.get("contact_method", ""),
        "amount_lost": form.get("amount_lost", ""),
        "submitted_at": datetime.utcnow().isoformat()
    }

    # For now just return success
    return templates.TemplateResponse("dashboard/report.html", {
        "request": request,
        "user": user,
        "success": "Thank you for reporting this scam. Our team will review it."
    })


async def profile_page(request: Request):
    """User profile/settings page."""
    user = require_auth(request)
    scan_history = _scan_history.get(user["id"], [])
    return templates.TemplateResponse("dashboard/profile.html", {
        "request": request,
        "user": user,
        "scan_history": scan_history
    })


async def profile_post(request: Request):
    """Handle profile update."""
    user = require_auth(request)
    form = await request.form()

    user["name"] = form.get("name", user["name"])
    user["preferences"]["email_notifications"] = form.get("email_notifications") == "on"
    user["preferences"]["scan_alerts"] = form.get("scan_alerts") == "on"
    user["preferences"]["weekly_digest"] = form.get("weekly_digest") == "on"

    _users_db[user["email"]] = user

    return templates.TemplateResponse("dashboard/profile.html", {
        "request": request,
        "user": user,
        "success": "Profile updated successfully"
    })


async def analyze_image(request: Request):
    """Analyze job posting from uploaded image (OCR + analysis)."""
    form = await request.form()
    image_file = form.get("image")

    if not image_file:
        return templates.TemplateResponse("partials/result_card.html", {
            "request": request,
            "result": None,
            "error": "No image uploaded"
        })

    # Read image bytes
    image_bytes = await image_file.read()

    # Simple OCR using pytesseract if available, otherwise fallback
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return templates.TemplateResponse("partials/result_card.html", {
                "request": request,
                "result": None,
                "error": "No text could be extracted from the image. Please ensure the image is clear and contains readable text."
            })

        # Use the extracted text for analysis
        # Try to parse title, company from first lines
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        job_title = lines[0] if len(lines) > 0 else "Extracted from Image"
        company_name = lines[1] if len(lines) > 1 else "Unknown Company"
        job_description = extracted_text

        result = predict_risk(
            job_title=job_title[:200],
            company_name=company_name[:200],
            job_description=job_description,
            raw_text=extracted_text
        )

        # Add note about OCR extraction
        result.summary = f"[OCR Extracted] {result.summary}"

    except ImportError:
        # OCR not available, return error
        return templates.TemplateResponse("partials/result_card.html", {
            "request": request,
            "result": None,
            "error": "OCR functionality not available. Please install pytesseract and tesseract-ocr, or use text/URL input instead."
        })
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return templates.TemplateResponse("partials/result_card.html", {
            "request": request,
            "result": None,
            "error": f"Failed to analyze image: {str(e)}"
        })

    return templates.TemplateResponse("partials/result_card.html", {"request": request, "result": result})


# --- Telegram Webhook ---
async def telegram_webhook(request: Request):
    """Receive Telegram webhook updates."""
    if not TELEGRAM_BOT_TOKEN:
        return JSONResponse({"error": "Bot not configured"}, status_code=503)

    try:
        update_data = await request.json()
        telegram_app = await get_telegram_app()
        await process_webhook_update(telegram_app, update_data)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse({"error": "Webhook processing failed"}, status_code=500)


# --- Starlette App ---
app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/partials/tab-text", partial_tab_text, methods=["GET"]),
        Route("/partials/tab-url", partial_tab_url, methods=["GET"]),
        Route("/", index, methods=["GET"]),
        Route("/scanner", scanner_page, methods=["GET"]),
        Route("/analyze", analyze_form, methods=["POST"]),
        Route("/analyze/json", analyze_json, methods=["POST"]),
        Route("/analyze/image", analyze_image, methods=["POST"]),
        Route("/extract-url", extract_url, methods=["POST"]),
        Route("/telegram/webhook", telegram_webhook, methods=["POST"]),
        # Auth routes
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login_post, methods=["POST"]),
        Route("/register", register_page, methods=["GET"]),
        Route("/register", register_post, methods=["POST"]),
        Route("/logout", logout, methods=["GET"]),
        Route("/onboarding", onboarding_page, methods=["GET"]),
        Route("/onboarding", onboarding_post, methods=["POST"]),
        # Dashboard routes (protected)
        Route("/dashboard", dashboard_page, methods=["GET"]),
        Route("/dashboard/history", scan_history_page, methods=["GET"]),
        Route("/dashboard/scan/{scan_id}", scan_detail_page, methods=["GET"]),
        Route("/dashboard/report", report_scam_page, methods=["GET"]),
        Route("/dashboard/report", report_scam_post, methods=["POST"]),
        Route("/dashboard/profile", profile_page, methods=["GET"]),
        Route("/dashboard/profile", profile_post, methods=["POST"]),
    ],
    lifespan=lifespan
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=30*24*60*60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)