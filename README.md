# DarGlobal & Wasalt Property Chatbot

An AI chatbot that answers questions about **DarGlobal** (luxury developer — darglobal.co.uk) projects and **Wasalt** (Saudi property portal — wasalt.sa) listings, using data scraped from their public websites and a **free LLM via OpenRouter**.

**Live demo:** _<Vercel URL>_

## How it works

```
scraper/ ──► data/*.jsonl ──► app/retrieval.py (BM25 + structured filters) ──► OpenRouter free model ──► streamed answer + sources
```

| Part | What it does |
|---|---|
| `scraper/darglobal.py` | Reads `darglobal.co.uk/sitemap.xml`, respects `robots.txt`, renders each page with headless Chromium (the site is a JS app), strips nav/footer boilerplate → `data/darglobal.jsonl` (projects, partners, insights, press, blog). |
| `scraper/wasalt.py` | wasalt.sa HTML is behind a Cloudflare bot challenge, which this project does **not** try to bypass. It uses Wasalt's public XML sitemaps (listed in its `robots.txt`) instead: ~62k sale/rent/daily-rental listings (type, purpose, bedrooms, size, facing, street width, ID, URL) and ~13.7k city/district category pages → `data/wasalt_*.jsonl`. |
| `app/retrieval.py` | Pure-Python BM25 over DarGlobal page chunks + Wasalt per-city summaries, plus a structured listing filter (sale/rent/daily × property type × bedrooms) that returns exact counts and example listings. No numpy or torch, so it fits in a serverless function. |
| `app/main.py` | FastAPI: `GET /` chat UI, `GET /api/health`, `POST /api/chat` (NDJSON streaming). Calls OpenRouter with a fallback list of free models. |
| `app/static/index.html` | Chat UI with streaming answers, markdown, and source links. |

**Model:** `google/gemma-4-31b-it:free` by default. OpenRouter falls back automatically to `nvidia/nemotron-3-ultra-550b-a55b:free` and then `inclusionai/ling-3.2-flash:free`. Set `OPENROUTER_MODELS` to change the list.

## Run locally with Docker

```bash
cp .env.example .env          # add your OPENROUTER_API_KEY
docker compose up --build     # → http://localhost:8000
```

Or without Compose: `docker build -t prop-chatbot . && docker run -p 8000:8000 --env-file .env prop-chatbot`

## Deploy to Vercel

1. Import this GitHub repo in Vercel (Framework preset: **Other**). `vercel.json` routes every request to the FastAPI app in `api/index.py`.
2. Add the environment variable `OPENROUTER_API_KEY` under **Project → Settings → Environment Variables**.
3. Deploy, then open the URL. `/api/health` shows the indexed data counts.

## Refresh the data

```bash
docker build -f scraper/Dockerfile -t prop-scraper .
docker run --rm -v "$PWD/data:/work/data" prop-scraper
```

## Known limitations
- Wasalt listing URLs don't include a city, and their size figures sometimes lose the decimal point (`15264-sqm` is likely 152.64 m²). City-level questions are answered from Wasalt's category pages. Prices are only available as category price bands.
- The data is a snapshot taken at scrape time (Nov 2026).
