"""Scrape public pages from darglobal.co.uk.

The site is a JS-rendered app, so pages are loaded with a headless browser.
URLs come from the public sitemap.xml and are filtered against robots.txt.
Output: data/darglobal.jsonl  (one {url, title, section, text} per line)
"""
from __future__ import annotations
import asyncio
import json
import re
import sys
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

BASE = "https://darglobal.co.uk"
OUT = Path(__file__).resolve().parent.parent / "data" / "darglobal.jsonl"
CONCURRENCY = 3
DELAY_S = 1.5  # polite delay per worker between requests

# Pages with no useful content for a property chatbot
SKIP = re.compile(
    r"/(thank-you|success|pay-online|bookYourUnit|internship-application|landingpage/|new-home-page|tokenization$)",
    re.I,
)

# Boilerplate lines (nav / footer / cookie banners) dropped from page text
BOILER = {
    "REGISTER INTEREST", "English", "|", "العربية", "DOWNLOAD BROCHURE", "BOOK UNIT",
    "VIEW ALL IMAGES", "Accept", "Reject", "ENQUIRE NOW", "SUBMIT", "Read More",
}


def clean(text: str) -> str:
    lines, seen = [], set()
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln in BOILER or len(ln) < 2:
            continue
        if ln.startswith(("We value our privacy", "We use cookies", "+971 4562 9616")):
            continue
        if ln in seen and len(ln) < 80:  # repeated menu items
            continue
        seen.add(ln)
        lines.append(ln)
    return "\n".join(lines)


async def sitemap_urls(page) -> list[str]:
    await page.goto(f"{BASE}/sitemap.xml", wait_until="domcontentloaded", timeout=60000)
    xml = await page.content()
    urls = sorted(set(re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)))
    if not urls:  # browser may render XML as text
        urls = sorted(set(re.findall(r"https://darglobal\.co\.uk[^\s<\"]*", await page.inner_text("body"))))
    return urls


async def worker(ctx, queue, results, done, total):
    page = await ctx.new_page()
    while True:
        try:
            url = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        for attempt in range(2):
            try:
                resp = await page.goto(url, wait_until="networkidle", timeout=60000)
                await page.wait_for_timeout(1000)
                title = await page.title()
                text = clean(await page.inner_text("body"))
                if resp and resp.status == 200 and len(text) > 200 and "Application error" not in text[:200]:
                    path = urlparse(url).path.strip("/")
                    results.append({
                        "source": "darglobal",
                        "url": url,
                        "title": title,
                        "section": path.split("/")[0] if "/" in path else "page",
                        "text": text,
                    })
                break
            except Exception as e:  # noqa: BLE001
                if attempt:
                    print(f"  fail {url}: {e.__class__.__name__}", file=sys.stderr)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]}/{total} pages, {len(results)} kept", flush=True)
        await asyncio.sleep(DELAY_S)
    await page.close()


async def main(limit: int | None = None):
    rp = urllib.robotparser.RobotFileParser(f"{BASE}/robots.txt")
    rp.read()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(locale="en-GB")
        first = await ctx.new_page()
        urls = await sitemap_urls(first)
        await first.close()
        urls = [u for u in urls if rp.can_fetch("*", u) and not SKIP.search(u)]
        if limit:
            urls = urls[:limit]
        print(f"DarGlobal: crawling {len(urls)} URLs", flush=True)
        q: asyncio.Queue = asyncio.Queue()
        for u in urls:
            q.put_nowait(u)
        results: list = []
        done = [0]
        await asyncio.gather(*(worker(ctx, q, results, done, len(urls)) for _ in range(CONCURRENCY)))
        await browser.close()
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in sorted(results, key=lambda r: r["url"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DarGlobal: wrote {len(results)} pages -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
