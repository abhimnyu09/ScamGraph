import os
import logging
import json
from contextlib import asynccontextmanager
from typing import Optional

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from starlette.exceptions import HTTPException

from app.model.predict import predict_risk, load_model
from app.scrapers.linkedin import extract_linkedin_job
from app.scrapers.naukri import extract_naukri_job
from app.scrapers.generic import extract_generic_job
from app.telegram_bot import create_application, process_webhook_update, setup_webhook
from app.schemas import AnalysisResult, validate_job_input

# --- Config ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g., https://scamgraph.onrender.com
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        Route("/analyze", analyze_form, methods=["POST"]),
        Route("/analyze/json", analyze_json, methods=["POST"]),
        Route("/extract-url", extract_url, methods=["POST"]),
        Route("/telegram/webhook", telegram_webhook, methods=["POST"]),
    ],
    lifespan=lifespan
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)