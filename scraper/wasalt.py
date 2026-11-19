"""Collect public listing data from Wasalt (wasalt.sa).

wasalt.sa HTML pages sit behind a Cloudflare bot challenge, which we do not try
to bypass. Instead we use the public XML sitemaps that Wasalt advertises in
its robots.txt for crawlers (hosted on cdn.wasalt.sa). Each listing URL encodes
purpose (sale/rent), property type, size, bedrooms, facing, street width and ID;
category URLs encode city / district / type / bedroom / price-band coverage.

Output: data/wasalt_listings.jsonl, data/wasalt_categories.jsonl
"""
from __future__ import annotations

import gzip
import json
import re
import urllib.request
from pathlib import Path

CDN = "https://cdn.wasalt.sa/sitemap/"
DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; property-research-bot/1.0)"}


def fetch_sitemap(name: str) -> list[tuple[str, str]]:
    req = urllib.request.Request(CDN + name, headers=UA)
    raw = urllib.request.urlopen(req, timeout=60).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    xml = raw.decode("utf-8", "replace")
    out = []
    for block in re.findall(r"<url>(.*?)</url>", xml, re.S):
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block)
        mod = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block)
        if loc:
            out.append((loc.group(1), mod.group(1) if mod else ""))
    return out


LISTING = re.compile(r"/property/(?P<purpose>sale|rent)/(?P<slug>.+)-(?P<id>\d+)$")
DAILY = re.compile(r"/dailyrental/(?P<slug>.+)-(?P<id>\d+)$")


def parse_listing(url: str, lastmod: str) -> dict | None:
    m = LISTING.search(url)
    if m:
        slug, purpose, lid = m["slug"], m["purpose"], m["id"]
    else:
        m = DAILY.search(url)
        if not m:
            return None
        slug, purpose, lid = m["slug"], "daily-rental", m["id"]
    ptype = slug.split("-")[0]
    rec = {"id": lid, "url": url, "purpose": purpose, "type": ptype, "lastmod": lastmod}
    if (s := re.search(r"-(\d+)-sqm", slug)):
        rec["size_raw"] = int(s[1])
    if (b := re.search(r"with-(\d+)-bedroom", slug)):
        rec["bedrooms"] = int(b[1])
    if (f := re.search(r"facing-([a-z-]+?)(?:-on-|$)", slug)):
        rec["facing"] = f[1].replace("-", " ")
    if (w := re.search(r"on-(\d+)m-width-street", slug)):
        rec["street_width_m"] = int(w[1])
    return rec



CAT = re.compile(r"/en/(?P<slug>[^/]+)$")


def parse_category(url: str) -> dict | None:
    m = CAT.search(url)
    if not m or "-for-" not in m["slug"]:
        return None
    slug = m["slug"]
    rec = {"source": "wasalt", "kind": "category", "url": url, "slug": slug}
    head, _, tail = slug.partition("-for-")
    rec["what"] = head.replace("-", " ")
    pm = re.match(r"(sale|rent|daily-rental)(?:-in-(.+))?", tail)
    if pm:
        rec["purpose"] = pm[1]
        loc = pm[2] or "saudi-arabia"
        if (pr := re.search(r"-for-sar-(\d+)$", loc)):
            rec["max_price_sar"] = int(pr[1])
            loc = loc[: pr.start()]
        rec["location"] = loc
    return rec


def main():
    DATA.mkdir(exist_ok=True)
    listings = []
    for name in ["product_sitemap_en_sa.xml.gz", "rental_pdp_sitemap_en_sa.xml.gz"]:
        for url, mod in fetch_sitemap(name):
            if (r := parse_listing(url, mod)):
                listings.append(r)
    cats = []
    for name in ["category_sitemap_en_sa.xml.gz", "rental_srp_sitemap_en_sa.xml.gz"]:
        for url, _ in fetch_sitemap(name):
            if (c := parse_category(url)):
                cats.append(c)
    with (DATA / "wasalt_listings.jsonl").open("w") as f:
        for r in listings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DATA / "wasalt_categories.jsonl").open("w") as f:
        for c in cats:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wasalt: {len(listings)} listings, {len(cats)} category pages")


if __name__ == "__main__":
    main()
