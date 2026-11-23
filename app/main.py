import os
import logging
import json
import base64
import io
import secrets
import hashlib
import sqlite3
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
from urllib.parse import urlencode

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import httpx
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
from app.schemas import AnalysisResult, validate_job_input

# --- Config ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g., https://scamgraph.onrender.com
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# Google OAuth Config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- In-Memory Storage (users only, scans use SQLite) ---
_users_db: Dict[str, dict] = {}  # email -> user data
_session_store: Dict[str, dict] = {}  # session_id -> session data

# --- SQLite Database for Persistent Scan History ---
DB_PATH = os.getenv("DATABASE_PATH", "scamgraph.db")

def init_db():
    """Initialize SQLite database with scan history table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            job_title TEXT,
            company_name TEXT,
            risk_score INTEGER,
            risk_tier TEXT,
            summary TEXT,
            job_url TEXT,
            recruiter_email TEXT,
            checks TEXT,  -- JSON array
            raw_text TEXT,
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            -- Detailed analysis fields (Phase 4)
            company_website TEXT,
            company_executive_summary TEXT,
            domain_age INTEGER,
            linkedin_company TEXT,
            glassdoor_company TEXT,
            company_registration_status TEXT,
            domain_ownership_status TEXT,
            physical_address_status TEXT,
            years_operation_status TEXT,
            company_locations TEXT,
            job_location TEXT,
            user_location TEXT,
            work_type TEXT,
            location_match_status TEXT,
            location_match_details TEXT,
            location_analysis_notes TEXT,
            recruiter_name TEXT,
            email_domain_verified INTEGER DEFAULT 0,
            recruiter_linkedin TEXT,
            linkedin_verified TEXT,
            directory_match TEXT,
            official_channel TEXT,
            salary_text TEXT,
            salary_benchmark TEXT,
            responsibilities TEXT,
            requirements TEXT,
            salary_in_range TEXT,
            no_unrealistic_salary TEXT,
            clear_structure TEXT,
            job_structure_complete TEXT,
            grammar_errors INTEGER DEFAULT 0,
            spacing_issues INTEGER DEFAULT 0,
            suspicious_phrases INTEGER DEFAULT 0,
            urgency_language INTEGER DEFAULT 0,
            grammar_examples TEXT,
            content_assessment TEXT,
            suspicious_signals TEXT,
            legitimate_signals TEXT,
            final_recommendation TEXT,
            immediate_precautions TEXT,
            verification_steps TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scan_history_user_id
        ON scan_history(user_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp
        ON scan_history(timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_scan_history_risk_tier
        ON scan_history(risk_tier)
    """)

    # User preferences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            email_notifications INTEGER DEFAULT 1,
            scan_alerts INTEGER DEFAULT 1,
            weekly_digest INTEGER DEFAULT 0,
            theme TEXT DEFAULT 'dark',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Shared scans table (for shareable links)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shared_scans (
            id TEXT PRIMARY KEY,
            scan_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            share_token TEXT UNIQUE NOT NULL,
            expires_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scan_id) REFERENCES scan_history(id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_shared_scans_token
        ON shared_scans(share_token)
    """)

    conn.commit()

    # Migration: Add company_executive_summary column if not exists
    cursor.execute("PRAGMA table_info(scan_history)")
    columns = [row[1] for row in cursor.fetchall()]
    if "company_executive_summary" not in columns:
        cursor.execute("ALTER TABLE scan_history ADD COLUMN company_executive_summary TEXT")
        conn.commit()

    conn.close()

# Initialize database on startup
init_db()

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
    """Save scan to persistent SQLite database."""
    scan_id = secrets.token_urlsafe(16)
    timestamp = datetime.utcnow().isoformat()

    # Prepare JSON fields
    checks_json = json.dumps(scan_data.get("checks", []))
    grammar_examples_json = json.dumps(scan_data.get("grammar_examples", []))
    suspicious_signals_json = json.dumps(scan_data.get("suspicious_signals", []))
    legitimate_signals_json = json.dumps(scan_data.get("legitimate_signals", []))
    immediate_precautions_json = json.dumps(scan_data.get("immediate_precautions", []))
    verification_steps_json = json.dumps(scan_data.get("verification_steps", []))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scan_history
        (id, user_id, job_title, company_name, risk_score, risk_tier, summary,
         job_url, recruiter_email, checks, raw_text, timestamp,
         created_at,
         company_website, company_executive_summary, domain_age, linkedin_company, glassdoor_company,
         company_registration_status, domain_ownership_status, physical_address_status,
         years_operation_status, company_locations, job_location, user_location,
         work_type, location_match_status, location_match_details, location_analysis_notes,
         recruiter_name, email_domain_verified, recruiter_linkedin, linkedin_verified,
         directory_match, official_channel, salary_text, salary_benchmark,
         responsibilities, requirements, salary_in_range, no_unrealistic_salary,
         clear_structure, job_structure_complete, grammar_errors, spacing_issues,
         suspicious_phrases, urgency_language, grammar_examples, content_assessment,
         suspicious_signals, legitimate_signals, final_recommendation,
         immediate_precautions, verification_steps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_id,
        user_id,
        scan_data.get("job_title", ""),
        scan_data.get("company_name", ""),
        scan_data.get("risk_score", 0),
        scan_data.get("risk_tier", "info"),
        scan_data.get("summary", ""),
        scan_data.get("job_url", ""),
        scan_data.get("recruiter_email", ""),
        checks_json,
        scan_data.get("raw_text", ""),
        timestamp,
        timestamp,  # created_at
        # Company Research
        scan_data.get("company_website"),
        scan_data.get("company_executive_summary"),
        scan_data.get("domain_age"),
        scan_data.get("linkedin_company"),
        scan_data.get("glassdoor_company"),
        scan_data.get("company_registration_status"),
        scan_data.get("domain_ownership_status"),
        scan_data.get("physical_address_status"),
        scan_data.get("years_operation_status"),
        # Location Verification
        scan_data.get("company_locations"),
        scan_data.get("job_location"),
        scan_data.get("user_location"),
        scan_data.get("work_type"),
        scan_data.get("location_match_status"),
        scan_data.get("location_match_details"),
        scan_data.get("location_analysis_notes"),
        # Contact Verification
        scan_data.get("recruiter_name"),
        1 if scan_data.get("email_domain_verified") else 0,
        scan_data.get("recruiter_linkedin"),
        scan_data.get("linkedin_verified"),
        scan_data.get("directory_match"),
        scan_data.get("official_channel"),
        # Salary & Job Verification
        scan_data.get("salary_text"),
        scan_data.get("salary_benchmark"),
        scan_data.get("responsibilities"),
        scan_data.get("requirements"),
        scan_data.get("salary_in_range"),
        scan_data.get("no_unrealistic_salary"),
        scan_data.get("clear_structure"),
        scan_data.get("job_structure_complete"),
        # Grammar & Content
        scan_data.get("grammar_errors", 0),
        scan_data.get("spacing_issues", 0),
        scan_data.get("suspicious_phrases", 0),
        scan_data.get("urgency_language", 0),
        grammar_examples_json,
        scan_data.get("content_assessment"),
        suspicious_signals_json,
        legitimate_signals_json,
        # Final Recommendations
        scan_data.get("final_recommendation"),
        immediate_precautions_json,
        verification_steps_json,
    ))
    conn.commit()
    conn.close()

    # Increment user's scan usage counter
    increment_scan_usage(user_id)

    # Return the saved record
    return {
        "id": scan_id,
        "timestamp": timestamp,
        **scan_data
    }


def increment_scan_usage(user_id: str):
    """Increment user's scan usage counter in memory."""
    for user in _users_db.values():
        if user["id"] == user_id:
            user["scans_used"] = user.get("scans_used", 0) + 1
            break

def get_scan_history(user_id: str, limit: int = 50, offset: int = 0, filter_tier: str = None) -> List[dict]:
    """Get scan history from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT id, user_id, job_title, company_name, risk_score, risk_tier,
               summary, job_url, recruiter_email, checks, raw_text, timestamp
        FROM scan_history
        WHERE user_id = ?
    """
    params = [user_id]

    if filter_tier:
        query += " AND risk_tier = ?"
        params.append(filter_tier)

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    scans = []
    for row in rows:
        scan = dict(row)
        # Parse checks JSON
        try:
            scan["checks"] = json.loads(scan.get("checks", "[]"))
        except:
            scan["checks"] = []
        scans.append(scan)

    return scans

def get_scan_history_count(user_id: str, filter_tier: str = None) -> int:
    """Get total count of scans for user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM scan_history WHERE user_id = ?"
    params = [user_id]

    if filter_tier:
        query += " AND risk_tier = ?"
        params.append(filter_tier)

    cursor.execute(query, params)
    count = cursor.fetchone()[0]
    conn.close()
    return count

def get_scan_by_id(user_id: str, scan_id: str) -> Optional[dict]:
    """Get a specific scan by ID with all detailed fields."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM scan_history
        WHERE user_id = ? AND id = ?
    """, (user_id, scan_id))

    row = cursor.fetchone()
    conn.close()

    if row:
        scan = dict(row)
        # Parse JSON fields
        for field in ["checks", "grammar_examples", "suspicious_signals", "legitimate_signals", "immediate_precautions", "verification_steps"]:
            try:
                scan[field] = json.loads(scan.get(field, "[]"))
            except:
                scan[field] = []
        # Convert email_domain_verified back to bool
        scan["email_domain_verified"] = bool(scan.get("email_domain_verified", 0))
        return scan
    return None

def delete_scan_from_history(user_id: str, scan_id: str) -> bool:
    """Delete a scan from history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history WHERE user_id = ? AND id = ?", (user_id, scan_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# --- User Preferences ---
def get_user_preferences(user_id: str) -> dict:
    """Get user preferences from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "email_notifications": 1,
        "scan_alerts": 1,
        "weekly_digest": 0,
        "theme": "dark"
    }

def save_user_preferences(user_id: str, prefs: dict):
    """Save user preferences to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO user_preferences
        (user_id, email_notifications, scan_alerts, weekly_digest, theme, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        1 if prefs.get("email_notifications") else 0,
        1 if prefs.get("scan_alerts") else 0,
        1 if prefs.get("weekly_digest") else 0,
        prefs.get("theme", "dark"),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

# --- Scan Sharing ---
def create_shared_scan(user_id: str, scan_id: str, expires_days: int = 30) -> str:
    """Create a shareable link for a scan. Returns the share token."""
    share_token = secrets.token_urlsafe(16)
    share_id = secrets.token_urlsafe(16)
    expires_at = (datetime.utcnow() + timedelta(days=expires_days)).isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO shared_scans (id, scan_id, user_id, share_token, expires_at)
        VALUES (?, ?, ?, ?, ?)
    """, (share_id, scan_id, user_id, share_token, expires_at))
    conn.commit()
    conn.close()
    return share_token

def get_shared_scan(share_token: str) -> Optional[dict]:
    """Get a shared scan by token (no auth required)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ss.*, sh.*
        FROM shared_scans ss
        JOIN scan_history sh ON ss.scan_id = sh.id
        WHERE ss.share_token = ?
    """, (share_token,))
    row = cursor.fetchone()
    conn.close()

    if row:
        scan = dict(row)
        # Parse JSON fields
        for field in ["checks", "grammar_examples", "suspicious_signals", "legitimate_signals", "immediate_precautions", "verification_steps"]:
            try:
                scan[field] = json.loads(scan.get(field, "[]"))
            except:
                scan[field] = []
        scan["email_domain_verified"] = bool(scan.get("email_domain_verified", 0))
        return scan
    return None

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: Starlette):
    # Startup
    load_model()
    logger.info("Model loaded")

    yield

    # Shutdown
    logger.info("Application shutdown")


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
        "scans_used": 0,
        "scans_limit": 100,
        "plan_name": "FREE PLAN",
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


# --- Google OAuth ---
async def google_login(request: Request):
    """Initiate Google OAuth flow."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return templates.TemplateResponse("auth/login.html", {
            "request": request,
            "error": "Google Sign-In not configured. Please contact administrator."
        })

    # Store the intended destination
    request.session["oauth_next"] = request.query_params.get("next", "/dashboard")

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


async def google_callback(request: Request):
    """Handle Google OAuth callback."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return templates.TemplateResponse("auth/login.html", {
            "request": request,
            "error": "Google Sign-In not configured."
        })

    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error or not code:
        return templates.TemplateResponse("auth/login.html", {
            "request": request,
            "error": f"Google Sign-In failed: {error or 'No code received'}"
        })

    try:
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            token_data = token_response.json()

            if "error" in token_data:
                raise Exception(token_data["error_description"])

            access_token = token_data["access_token"]

            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            user_info = user_response.json()

        email = user_info.get("email", "").lower().strip()
        name = user_info.get("name", "").strip()
        google_id = user_info.get("id", "")

        if not email:
            return templates.TemplateResponse("auth/login.html", {
                "request": request,
                "error": "Could not retrieve email from Google."
            })

        # Check if user exists
        user = _users_db.get(email)
        if not user:
            # Create new user
            user_id = secrets.token_urlsafe(16)
            user = {
                "id": user_id,
                "name": name,
                "email": email,
                "password_hash": "",  # No password for OAuth users
                "google_id": google_id,
                "created_at": datetime.utcnow().isoformat(),
                "onboarded": False,
                "scans_used": 0,
                "scans_limit": 100,
                "plan_name": "FREE PLAN",
                "preferences": {
                    "email_notifications": True,
                    "scan_alerts": True
                }
            }
            _users_db[email] = user
        elif not user.get("google_id"):
            # Link Google account to existing user
            user["google_id"] = google_id
            _users_db[email] = user

        set_user_session(request, user["id"])

        # Redirect to intended destination or dashboard
        next_url = request.session.pop("oauth_next", "/dashboard")
        return RedirectResponse(url=next_url, status_code=302)

    except Exception as e:
        logger.error(f"Google OAuth error: {e}")
        return templates.TemplateResponse("auth/login.html", {
            "request": request,
            "error": f"Google Sign-In failed: {str(e)}"
        })


# --- Routes ---
async def health(request: Request):
    return JSONResponse({"status": "ok", "service": "scamgraph"})


async def partial_tab_text(request: Request):
    return templates.TemplateResponse("partials/tab_text.html", {"request": request})


async def partial_tab_url(request: Request):
    return templates.TemplateResponse("partials/tab_url.html", {"request": request})


async def partial_tab_image(request: Request):
    return templates.TemplateResponse("partials/tab_image.html", {"request": request})


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
            "recommendation": result.recommendation,
            "job_url": job_url or "",
            "recruiter_email": recruiter_email or "",
            "recruiter_phone": recruiter_phone or "",
            "recruiter_name": result.recruiter_name,
            "checks": [
                {"name": c.name, "status": c.check_status, "message": c.message, "weight": c.weight}
                for c in result.checks
            ],
            "raw_text": raw_text or "",
            # Company Research & Verification
            "company_website": result.company_website,
            "company_executive_summary": result.company_executive_summary,
            "domain_age": result.domain_age,
            "linkedin_company": result.linkedin_company,
            "glassdoor_company": result.glassdoor_company,
            "company_registration_status": result.company_registration_status,
            "domain_ownership_status": result.domain_ownership_status,
            "physical_address_status": result.physical_address_status,
            "years_operation_status": result.years_operation_status,
            # Location Verification
            "company_locations": result.company_locations,
            "job_location": result.job_location,
            "user_location": location or "",
            "work_type": result.work_type,
            "location_match_status": result.location_match_status,
            "location_match_details": result.location_match_details,
            "location_analysis_notes": result.location_analysis_notes,
            # Contact Verification
            "email_domain_verified": result.email_domain_verified,
            "recruiter_linkedin": result.recruiter_linkedin,
            "linkedin_verified": result.linkedin_verified,
            "directory_match": result.directory_match,
            "official_channel": result.official_channel,
            # Salary & Job Verification
            "salary_text": salary_text or "",
            "salary_benchmark": result.salary_benchmark,
            "responsibilities": result.responsibilities,
            "requirements": result.requirements,
            "salary_in_range": result.salary_in_range,
            "no_unrealistic_salary": result.no_unrealistic_salary,
            "clear_structure": result.clear_structure,
            "job_structure_complete": result.job_structure_complete,
            # Grammar & Content Analysis
            "grammar_errors": result.grammar_errors,
            "spacing_issues": result.spacing_issues,
            "suspicious_phrases": result.suspicious_phrases,
            "urgency_language": result.urgency_language,
            "grammar_examples": result.grammar_examples,
            "content_assessment": result.content_assessment,
            # Scam/Legit Pattern Detection
            "suspicious_signals": result.suspicious_signals,
            "legitimate_signals": result.legitimate_signals,
            # Final Recommendations
            "final_recommendation": result.final_recommendation,
            "immediate_precautions": result.immediate_precautions,
            "verification_steps": result.verification_steps,
        }
        save_scan_to_history(user["id"], scan_data)

    return templates.TemplateResponse("partials/result_card.html", {"request": request, "result": result, "user": user})


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
    return templates.TemplateResponse("scanner.html", {"request": request, "user": user, "current_page": "scanner"})


def get_dashboard_stats(user_id: str) -> dict:
    """Get dashboard statistics from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ?", (user_id,))
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND risk_tier = 'safe'", (user_id,))
    safe = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND risk_tier = 'caution'", (user_id,))
    caution = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM scan_history WHERE user_id = ? AND risk_tier = 'risky'", (user_id,))
    risky = cursor.fetchone()[0]

    # Get recent 5 scans
    cursor.execute("""
        SELECT id, job_title, company_name, risk_score, risk_tier, summary, timestamp
        FROM scan_history
        WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT 5
    """, (user_id,))
    recent_rows = cursor.fetchall()
    conn.close()

    recent_scans = []
    for row in recent_rows:
        recent_scans.append({
            "id": row[0],
            "job_title": row[1],
            "company_name": row[2],
            "risk_score": row[3],
            "risk_tier": row[4],
            "summary": row[5],
            "timestamp": row[6]
        })

    return {
        "total": total,
        "safe": safe,
        "caution": caution,
        "risky": risky,
        "recent_scans": recent_scans
    }


async def dashboard_page(request: Request):
    """User dashboard with scan history summary."""
    user = require_auth(request)

    stats = get_dashboard_stats(user["id"])

    return templates.TemplateResponse("dashboard/index.html", {
        "request": request,
        "user": user,
        "current_page": "dashboard",
        "stats": {
            "total": stats["total"],
            "safe": stats["safe"],
            "caution": stats["caution"],
            "risky": stats["risky"]
        },
        "recent_scans": stats["recent_scans"]
    })


async def scan_history_page(request: Request):
    """Scan history page with filtering and pagination."""
    user = require_auth(request)

    page = int(request.query_params.get("page", "1"))
    filter_tier = request.query_params.get("tier", "")
    per_page = 10
    offset = (page - 1) * per_page

    scans = get_scan_history(user["id"], limit=per_page, offset=offset, filter_tier=filter_tier if filter_tier else None)
    total_scans = get_scan_history_count(user["id"], filter_tier if filter_tier else None)

    total_pages = (total_scans + per_page - 1) // per_page

    return templates.TemplateResponse("dashboard/history.html", {
        "request": request,
        "user": user,
        "current_page": "history",
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

    scan = get_scan_by_id(user["id"], scan_id)

    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    return templates.TemplateResponse("dashboard/scan_detail.html", {
        "request": request,
        "user": user,
        "current_page": "history",
        "scan": scan
    })


async def delete_scan(request: Request):
    """Delete a scan from history (HTMX/DELETE)."""
    user = require_auth(request)
    scan_id = request.path_params.get("scan_id")

    deleted = delete_scan_from_history(user["id"], scan_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Scan not found")

    return JSONResponse({"success": True, "message": "Scan deleted"})


async def report_scam_page(request: Request):
    """Report a scam page."""
    user = get_user_from_session(request)
    return templates.TemplateResponse("dashboard/report.html", {
        "request": request,
        "user": user,
        "current_page": "report"
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
    scan_history = get_scan_history(user["id"], limit=100)
    return templates.TemplateResponse("dashboard/profile.html", {
        "request": request,
        "user": user,
        "current_page": "profile",
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

    # Save to database
    save_user_preferences(user["id"], user["preferences"])
    _users_db[user["email"]] = user

    # Get basic scan history for template (only basic fields to avoid JSON serialization issues)
    scan_history = get_scan_history(user["id"], limit=100)

    return templates.TemplateResponse("dashboard/profile.html", {
        "request": request,
        "user": user,
        "scan_history": scan_history,
        "success": "Profile updated successfully"
    })


async def share_scan(request: Request):
    """Create a shareable link for a scan."""
    user = require_auth(request)
    scan_id = request.path_params.get("scan_id")

    # Verify scan belongs to user
    scan = get_scan_by_id(user["id"], scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    share_token = create_shared_scan(user["id"], scan_id)
    share_url = f"{request.url.scheme}://{request.url.netloc}/shared/{share_token}"

    return JSONResponse({
        "success": True,
        "share_url": share_url,
        "share_token": share_token
    })


async def view_shared_scan(request: Request):
    """View a shared scan (no auth required)."""
    share_token = request.path_params.get("share_token")

    scan = get_shared_scan(share_token)
    if not scan:
        raise HTTPException(status_code=404, detail="Shared scan not found or expired")

    return templates.TemplateResponse("dashboard/scan_detail.html", {
        "request": request,
        "user": None,
        "current_page": "history",
        "scan": scan,
        "is_shared": True
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


# --- Starlette App ---
app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/partials/tab-text", partial_tab_text, methods=["GET"]),
        Route("/partials/tab-url", partial_tab_url, methods=["GET"]),
        Route("/partials/tab-image", partial_tab_image, methods=["GET"]),
        Route("/", index, methods=["GET"]),
        Route("/scanner", scanner_page, methods=["GET"]),
        Route("/analyze", analyze_form, methods=["POST"]),
        Route("/analyze/json", analyze_json, methods=["POST"]),
        Route("/analyze/image", analyze_image, methods=["POST"]),
        Route("/extract-url", extract_url, methods=["POST"]),
        # Auth routes
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login_post, methods=["POST"]),
        Route("/register", register_page, methods=["GET"]),
        Route("/register", register_post, methods=["POST"]),
        Route("/logout", logout, methods=["GET"]),
        Route("/onboarding", onboarding_page, methods=["GET"]),
        Route("/onboarding", onboarding_post, methods=["POST"]),
        # Google OAuth routes
        Route("/auth/google", google_login, methods=["GET"]),
        Route("/auth/google/callback", google_callback, methods=["GET"]),
        # Dashboard routes (protected)
        Route("/dashboard", dashboard_page, methods=["GET"]),
        Route("/dashboard/history", scan_history_page, methods=["GET"]),
        Route("/dashboard/scan/{scan_id}", scan_detail_page, methods=["GET"]),
        Route("/dashboard/scan/{scan_id}", delete_scan, methods=["DELETE"]),
        Route("/dashboard/report", report_scam_page, methods=["GET"]),
        Route("/dashboard/report", report_scam_post, methods=["POST"]),
        Route("/dashboard/profile", profile_page, methods=["GET"]),
        Route("/dashboard/profile", profile_post, methods=["POST"]),
        # Scan sharing routes
        Route("/dashboard/scan/{scan_id}/share", share_scan, methods=["POST"]),
        Route("/shared/{share_token}", view_shared_scan, methods=["GET"]),
    ],
    lifespan=lifespan
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=30*24*60*60)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)