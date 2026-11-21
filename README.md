# ScamGraph — Fake Job Posting Detector

> **ML-powered scam detection for job seekers** — Paste a job posting, get a 0–100 risk score with cited evidence before you apply.

## Features

- **Text Analysis** — Paste job title, company, description → instant risk score
- **URL Extraction** — Drop a LinkedIn/Naukri/Indeed link → we fetch & analyze
- **Explainable Results** — Safe/Caution/Risky verdict + green/red flags with reasons
- **Telegram Bot** — Forward any job message → get verdict in chat
- **India-First** — Trained on global + Indian scam patterns (fresher fees, training scams, etc.)

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI (async, type-safe) |
| Frontend | HTMX + Tailwind CSS (CDN, no build step) |
| ML | TF-IDF (char + word n-grams) + XGBoost |
| Deploy | Render.com (free tier, auto-deploy from GitHub) |
| Bot | python-telegram-bot (webhook mode) |

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/abhimnyu09/ScamGraph.git
cd ScamGraph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Train model (downloads Kaggle FJDC, trains XGBoost, saves artifacts)
python app/model/train.py

# 3. Run locally
uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Project Structure

```
scamgraph/
├── app/
│   ├── main.py              # FastAPI routes + HTMX endpoints
│   ├── model/
│   │   ├── train.py         # Training pipeline (run once)
│   │   ├── predict.py       # Inference wrapper
│   │   └── artifacts/       # model.pkl, vectorizer.pkl (gitignored)
│   ├── scrapers/
│   │   ├── linkedin.py      # LinkedIn job page extraction
│   │   ├── naukri.py        # Naukri job page extraction
│   │   └── generic.py       # Fallback for careers pages
│   ├── telegram_bot.py      # Bot handler (webhook)
│   ├── features/
│   │   ├── text_features.py # TF-IDF vectorizers
│   │   ├── url_features.py  # Domain age, SSL, WHOIS
│   │   └── handcrafted.py   # 20+ heuristic features
│   ├── schemas.py           # Pydantic models
│   └── templates/
│       ├── base.html        # Layout + Tailwind + HTMX
│       ├── index.html       # Input form (tabs + accordion)
│       ├── result.html      # Score card + evidence accordion
│       └── partials/        # HTMX fragments
├── data/                    # Raw CSVs (gitignored)
├── requirements.txt
├── render.yaml              # Render.com config
└── README.md
```

## Model Training

```bash
python app/model/train.py
```

- Downloads **Fake Job Posting Detection (FJDC)** from Kaggle (17,880 rows)
- Adds **handcrafted India-specific samples** (fresher scams, training fees, etc.)
- Features: TF-IDF (char 3-5grams + word 1-2grams) + 20 heuristic features
- Model: **XGBoost** with `scale_pos_weight` for class imbalance
- Output: `model.pkl`, `vectorizer.pkl`, `feature_names.json` in `app/model/artifacts/`

## Telegram Bot Setup

1. Create bot via [@BotFather](https://t.me/BotFather) → get token
2. Set webhook: `https://YOUR_RENDER_URL/telegram/webhook`
3. Add `TELEGRAM_BOT_TOKEN` to Render Environment Variables
4. Forward any job message to bot → get instant verdict

## Deploy to Render (Free)

1. Push to GitHub
2. Connect repo at [dashboard.render.com](https://dashboard.render.com)
3. Render reads `render.yaml` → auto-deploys
4. Add `TELEGRAM_BOT_TOKEN` in Environment Variables
5. Live at `https://scamgraph.onrender.com`

## License

MIT — Built for HS 510 Essentials of Entrepreneurship