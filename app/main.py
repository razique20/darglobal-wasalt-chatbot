"""FastAPI chatbot over scraped DarGlobal + Wasalt data, answered by a free OpenRouter model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .retrieval import KnowledgeBase

ROOT = Path(__file__).resolve().parent


def _load_dotenv():
    p = ROOT.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Primary free model + fallbacks (OpenRouter tries them in order if one is busy/rate-limited)
MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "google/gemma-4-31b-it:free,nvidia/nemotron-3-ultra-550b-a55b:free,inclusionai/ling-3.2-flash:free",
).split(",") if m.strip()]

SYSTEM = """You are a property assistant for two real-estate sources:
- DarGlobal (darglobal.co.uk): an international luxury developer (UAE, Oman, Saudi Arabia, Qatar, Spain, UK, ...).
- Wasalt (wasalt.sa): a Saudi Arabian property listings portal.

Answer ONLY using the CONTEXT below, which was scraped from those public websites. Rules:
- If the context does not contain the answer, say you don't have that information and suggest the relevant site.
- Be concise and specific (names, locations, sizes, statuses, counts). Use short bullet lists where helpful.
- Cite sources inline as markdown links to the URLs given in the context.
- Never invent prices, availability or contact details not present in the context.
- Reply in the same language as the user (English or Arabic)."""

app = FastAPI(title="DarGlobal & Wasalt Property Chatbot")
kb = KnowledgeBase()


class Msg(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Msg]


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "darglobal_pages": kb.n_dg_pages, "wasalt_listings": len(kb.listings),
            "indexed_chunks": kb.n_docs, "models": MODELS, "key_configured": bool(os.getenv("OPENROUTER_API_KEY"))}


@app.post("/api/chat")
async def chat(body: ChatIn):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return JSONResponse({"error": "OPENROUTER_API_KEY is not configured on the server."}, status_code=500)
    msgs = [m for m in body.messages if m.role in ("user", "assistant")][-8:]
    if not msgs or msgs[-1].role != "user":
        return JSONResponse({"error": "last message must be from the user"}, status_code=400)
    question = msgs[-1].content[:2000]
    prev_user = " ".join(m.content for m in msgs[:-1] if m.role == "user")[-300:]
    ctx, sources = kb.context(question, prev_user)

    payload = {
        "models": MODELS,
        "stream": True,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": f"{SYSTEM}\n\nCONTEXT:\n{ctx}"}]
                    + [{"role": m.role, "content": m.content[:4000]} for m in msgs],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "Property Chatbot"}

    async def gen():
        model = None
        yield json.dumps({"sources": sources}) + "\n"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
                async with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as r:
                    if r.status_code != 200:
                        err = (await r.aread()).decode()[:300]
                        yield json.dumps({"error": f"OpenRouter {r.status_code}: {err}"}) + "\n"
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if "error" in obj:
                            yield json.dumps({"error": str(obj["error"])[:300]}) + "\n"
                            return
                        ch = obj.get("choices") or [{}]
                        delta = ch[0].get("delta", {}).get("content")
                        if delta:
                            yield json.dumps({"delta": delta}) + "\n"
                        if obj.get("model"):
                            model = obj["model"]
                    yield json.dumps({"done": True, "model": model}) + "\n"
        except httpx.HTTPError as e:
            yield json.dumps({"error": f"Upstream error: {e.__class__.__name__}"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
