import os
import logging
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode

from app.model.predict import predict_risk, load_model
from app.scrapers.linkedin import extract_linkedin_job
from app.scrapers.naukri import extract_naukri_job
from app.scrapers.generic import extract_generic_job

# --- Config ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("WARNING: TELEGRAM_BOT_TOKEN not set. Bot will not start.")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Helpers ---

def format_result(result) -> str:
    """Format analysis result for Telegram."""
    tier_emoji = {
        "safe": "✅",
        "caution": "⚠️",
        "risky": "🔴"
    }
    tier_label = {
        "safe": "SAFE",
        "caution": "CAUTION",
        "risky": "RISKY"
    }

    risk_tier = result.risk_tier.lower() if isinstance(result.risk_tier, str) else str(result.risk_tier).lower()
    emoji = tier_emoji.get(risk_tier, "❓")
    label = tier_label.get(risk_tier, "UNKNOWN")

    lines = [
        f"{emoji} <b>ScamGraph Analysis Result</b>",
        f"",
        f"<b>Risk Score:</b> {result.risk_score}/100",
        f"<b>Verdict:</b> {label}",
        f"",
        f"<b>Summary:</b> {result.summary}",
        f"",
        f"<b>Recommendation:</b> {result.recommendation}",
        f"",
        f"<b>Key Checks:</b>"
    ]

    # Top 5 checks by weight
    sorted_checks = sorted(result.checks, key=lambda c: c.weight, reverse=True)
    for check in sorted_checks[:5]:
        status_emoji = {
            "pass": "✅",
            "fail": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(check.check_status, "❓")
        lines.append(f"  {status_emoji} {check.name}: {check.message}")

    lines.append("")
    lines.append("<i>ScamGraph — Detect fake jobs before you apply</i>")
    lines.append("<i>This is an AI assessment, not legal advice. Always verify independently.</i>")

    return "\n".join(lines)


def extract_urls(text: str) -> list:
    """Extract URLs from text."""
    import re
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


async def analyze_job_text(text: str) -> str:
    """Analyze raw job text and return formatted result."""
    # Try to extract structured info from text
    lines = text.split('\n')
    job_title = ""
    company_name = ""
    job_description = text

    # Heuristic: first line often title, second company
    if len(lines) >= 1:
        job_title = lines[0][:200]
    if len(lines) >= 2:
        company_name = lines[1][:200]

    result = predict_risk(
        job_title=job_title or "Unknown Position",
        company_name=company_name or "Unknown Company",
        job_description=job_description,
        raw_text=text
    )
    return format_result(result)


async def analyze_url(url: str) -> str:
    """Extract job from URL and analyze."""
    url_lower = url.lower()

    # Try appropriate scraper
    if 'linkedin.com' in url_lower:
        job_data = await extract_linkedin_job(url)
    elif 'naukri.com' in url_lower:
        job_data = await extract_naukri_job(url)
    else:
        job_data = await extract_generic_job(url)

    if not job_data.get('success'):
        return f"❌ Could not extract job details from URL: {job_data.get('error', 'Unknown error')}"

    result = predict_risk(
        job_title=job_data.get('job_title', 'Unknown Position'),
        company_name=job_data.get('company_name', 'Unknown Company'),
        job_description=job_data.get('job_description', ''),
        job_url=url,
        location=job_data.get('location', ''),
        salary_text=job_data.get('salary', '') or job_data.get('salary_text', ''),
        employment_type=job_data.get('employment_type', ''),
    )
    return format_result(result)


# --- Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome = (
        "👋 <b>Welcome to ScamGraph Bot!</b>\n\n"
        "Forward me any job posting (text or link) and I'll analyze it for scam risk.\n\n"
        "<b>What I check:</b>\n"
        "• Upfront payment requests (training fees, registration fees)\n"
        "• Urgency pressure tactics\n"
        "• Requests for sensitive personal info (Aadhaar, bank details)\n"
        "• Free email domains (Gmail, Yahoo) for recruiters\n"
        "• Task scam patterns (data entry, copy-paste, captcha)\n"
        "• Suspicious communication channels (WhatsApp/Telegram only)\n"
        "• Salary anomalies\n\n"
        "<b>How to use:</b>\n"
        "1. Forward a job message from LinkedIn, Telegram, WhatsApp, Email\n"
        "2. Or paste a job description\n"
        "3. Or send a job link (LinkedIn, Naukri, Indeed, company careers)\n\n"
        "I'll reply with a risk score (0-100), verdict, and detailed evidence."
    )
    await update.message.reply_text(welcome, parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "📖 <b>ScamGraph Bot Help</b>\n\n"
        "<b>Commands:</b>\n"
        "/start — Welcome message\n"
        "/help — This help\n"
        "/about — About ScamGraph\n\n"
        "<b>Input formats:</b>\n"
        "• <b>Text:</b> Paste job description\n"
        "• <b>Link:</b> Send LinkedIn/Naukri/Indeed/company careers URL\n"
        "• <b>Forward:</b> Forward any message from Telegram/WhatsApp/Email\n\n"
        "<b>Output:</b>\n"
        "• Risk score 0-100\n"
        "• Verdict: SAFE / CAUTION / RISKY\n"
        "• Evidence breakdown\n"
        "• Actionable recommendation\n\n"
        "⚠️ <i>This is an AI assessment. Always verify independently on company's official website.</i>"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command."""
    about_text = (
        "ℹ️ <b>About ScamGraph</b>\n\n"
        "<b>Mission:</b> Detect fake job postings before you apply.\n\n"
        "<b>Built for:</b> HS 510 — Essentials of Entrepreneurship\n"
        "<b>Team:</b> Abhimanyu\n\n"
        "<b>Tech:</b>\n"
        "• ML: TF-IDF + XGBoost (trained on 18K+ labeled postings)\n"
        "• Data: Kaggle FJDC + India-specific scam patterns\n"
        "• API: FastAPI + HTMX\n"
        "• Deploy: Render.com (free tier)\n\n"
        "<b>Web App:</b> https://scamgraph.onrender.com\n\n"
        "<b>Disclaimer:</b> AI predictions can be wrong. "
        "This tool is for awareness only. "
        "Never pay fees to apply for jobs. "
        "Verify on company's official careers page."
    )
    await update.message.reply_text(about_text, parse_mode=ParseMode.HTML)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages (text, forwards, links)."""
    message = update.message
    if not message:
        return

    # Show typing indicator
    await context.bot.send_chat_action(chat_id=message.chat_id, action="typing")

    text = message.text or message.caption or ""
    if not text.strip():
        await message.reply_text("Please send a job description, link, or forward a message.")
        return

    # Check for URLs
    urls = extract_urls(text)
    if urls:
        # Analyze first URL
        result_text = await analyze_url(urls[0])
        if len(urls) > 1:
            result_text += f"\n\n<i>Note: {len(urls)} links found. Analyzed first one.</i>"
        await message.reply_text(result_text, parse_mode=ParseMode.HTML)
        return

    # Analyze as raw text
    result_text = await analyze_job_text(text)
    await message.reply_text(result_text, parse_mode=ParseMode.HTML)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors."""
    logger.error(f"Update {update} caused error: {context.error}")


# --- Application Factory ---

def create_application() -> Application:
    """Create and configure the Telegram bot application."""
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

    application = Application.builder().token(BOT_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))

    # Messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Errors
    application.add_error_handler(error_handler)

    return application


# --- Webhook Mode (for Render) ---

async def setup_webhook(application: Application, webhook_url: str):
    """Set up webhook for production."""
    await application.bot.set_webhook(
        url=f"{webhook_url}/telegram/webhook",
        allowed_updates=["message", "callback_query"]
    )
    logger.info(f"Webhook set to {webhook_url}/telegram/webhook")


async def process_webhook_update(application: Application, update_data: dict):
    """Process incoming webhook update."""
    update = Update.de_json(update_data, application.bot)
    await application.process_update(update)


# --- Standalone polling mode (for local testing) ---

async def run_polling():
    """Run bot in polling mode (local development)."""
    load_model()
    application = create_application()
    print("Starting bot in polling mode...")
    await application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_polling())