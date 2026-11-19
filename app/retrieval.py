"""Knowledge base + retrieval (pure Python, no heavy deps so it fits serverless).

- DarGlobal pages are chunked and indexed with BM25.
- Wasalt category pages and per-city summaries are indexed with BM25.
- Wasalt listings are queried with a structured filter (purpose/type/bedrooms),
  returning exact counts plus example listings.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
TOKEN = re.compile(r"[a-z0-9؀-ۿ]+")
STOP = set("""a an the and or of to in on for with is are was be by at as it this that from what which
who how do does can i you me my we our your about any there their tell show give list please""".split())


def tokenize(text: str) -> list[str]:
    toks = [t for t in TOKEN.findall(text.lower()) if t not in STOP]
    # crude plural stemming so "villas" matches "villa"
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(tokenize(" ".join([d["title"], d["title"], d.get("keys", ""), d.get("keys", ""), d["text"]])))
                   for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings[t].append(i)

    def search(self, query: str, k: int = 6) -> list[tuple[float, dict]]:
        q = set(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(s, self.docs[i]) for i, s in top]


def chunk(text: str, size: int = 1400, overlap: int = 200) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        cut = text.rfind("\n", start + size // 2, end)
        end = cut if cut > 0 and end < len(text) else end
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return out


def _read(name: str) -> list[dict]:
    p = DATA / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


TYPE_WORDS = {
    "apartment": "apartment", "flat": "apartment", "villa": "villa", "land": "land", "plot": "land",
    "floor": "floor", "building": "building", "chalet": "chalet", "farm": "farm", "studio": "studio",
    "showroom": "showroom", "office": "office", "shop": "shop", "warehouse": "warehouse",
    "rest house": "rest", "resthouse": "rest", "duplex": "duplex", "townhouse": "townhouse",
}


def describe(r: dict) -> str:
    purpose = {"sale": "for sale", "rent": "for rent", "daily-rental": "for daily rental / booking"}[r["purpose"]]
    parts = [f"Wasalt listing #{r['id']}: {r['type']} {purpose} in Saudi Arabia"]
    if "bedrooms" in r:
        parts.append(f"{r['bedrooms']} bedroom(s)")
    if "size_raw" in r:
        parts.append(f"area figure {r['size_raw']} sqm (from URL; decimal point may be missing)")
    if "facing" in r:
        parts.append(f"facing {r['facing']}")
    if "street_width_m" in r:
        parts.append(f"on a {r['street_width_m']}m wide street")
    return ", ".join(parts) + f". Updated {r['lastmod']}. {r['url']}"


class KnowledgeBase:
    def __init__(self):
        dg = _read("darglobal.jsonl")
        self.listings = _read("wasalt_listings.jsonl")
        cats = _read("wasalt_categories.jsonl")

        docs = []
        for p in dg:
            for i, c in enumerate(chunk(p["text"])):
                path = p["url"].split("co.uk", 1)[-1]
                boost = "darglobal project " if p["section"] in ("page", "projects", "one-of-one", "rayana", "amaya", "trump-plaza") else ""
                docs.append({"source": "DarGlobal", "url": p["url"], "text": c,
                             "title": p["title"] or path.rsplit("/", 1)[-1].replace("-", " ").title(),
                             "keys": boost + re.sub(r"[/_-]+", " ", path)})

        # Wasalt: one doc per city/area summarising which category pages exist
        by_loc: dict[str, list[dict]] = defaultdict(list)
        for c in cats:
            by_loc[c.get("location", "saudi-arabia")].append(c)
        for loc, items in by_loc.items():
            name = loc.replace("-", " ").title()
            lines = sorted({f"{c['what']} for {c.get('purpose', '').replace('-', ' ')}"
                            + (f" up to SAR {c['max_price_sar']:,}" if c.get("max_price_sar") else "")
                            for c in items})
            main_url = next((c["url"] for c in items if c["what"] == "properties"), items[0]["url"])
            docs.append({
                "source": "Wasalt", "title": f"Wasalt property categories in {name}", "url": main_url,
                "text": f"Wasalt (wasalt.sa) lists the following property categories in {name}, Saudi Arabia: "
                        + "; ".join(lines) + ". Browse: " + ", ".join(c["url"] for c in items[:6]),
            })

        self.stats_text = self._stats()
        docs.append({"source": "Wasalt", "title": "Wasalt listing statistics", "url": "https://wasalt.sa/en/",
                     "text": self.stats_text})
        self.index = BM25(docs)
        self.n_dg_pages, self.n_docs = len(dg), len(docs)
        self.cities = sorted(by_loc)

    def _stats(self) -> str:
        c = Counter((l["purpose"], l["type"]) for l in self.listings)
        beds = Counter((l["type"], l["bedrooms"]) for l in self.listings if "bedrooms" in l)
        lines = [f"Wasalt currently publishes {len(self.listings):,} listings in its sitemaps "
                 f"(sale: {sum(v for (p, _), v in c.items() if p == 'sale'):,}, "
                 f"rent: {sum(v for (p, _), v in c.items() if p == 'rent'):,}, "
                 f"daily rental: {sum(v for (p, _), v in c.items() if p == 'daily-rental'):,})."]
        lines.append("By purpose and type: " + "; ".join(f"{t} for {p}: {n:,}" for (p, t), n in c.most_common(30)))
        lines.append("Bedrooms: " + "; ".join(f"{t} {b}BR: {n:,}" for (t, b), n in beds.most_common(25)))
        return "\n".join(lines)

    def listing_query(self, q: str, k: int = 8) -> str | None:
        ql = q.lower()
        if not re.search(r"wasalt|listing|saudi|ksa|riyadh|jeddah|dammam|for (sale|rent)|rent|buy|bedroom|villa|apartment|land|chalet", ql):
            return None
        purpose = None
        if re.search(r"daily|booking|holiday|night", ql):
            purpose = "daily-rental"
        elif re.search(r"\brent|lease|rental", ql):
            purpose = "rent"
        elif re.search(r"sale|buy|purchase", ql):
            purpose = "sale"
        ptype = next((v for w, v in TYPE_WORDS.items() if re.search(rf"\b{w}s?\b", ql)), None)
        bm = re.search(r"(\d+)\s*-?\s*(?:bed|br\b|bhk|bedroom)", ql)
        beds = int(bm[1]) if bm else None
        if not (purpose or ptype or beds):
            return None
        hits = [l for l in self.listings
                if (not purpose or l["purpose"] == purpose)
                and (not ptype or l["type"] == ptype)
                and (beds is None or l.get("bedrooms") == beds)]
        hits.sort(key=lambda l: l["lastmod"], reverse=True)
        f = ", ".join(x for x in [purpose and f"purpose={purpose}", ptype and f"type={ptype}",
                                  beds is not None and f"bedrooms={beds}"] if x)
        out = [f"Structured search of Wasalt listings ({f}): {len(hits):,} matching listings "
               f"(listing URLs do not include city; sizes are taken from URLs and may be missing a decimal point)."]
        out += [f"- {describe(h)}" for h in hits[:k]]
        return "\n".join(out)

    def context(self, q: str, history: str = "") -> tuple[str, list[dict]]:
        raw = self.index.search(q + " " + history, k=30)
        per_url: Counter = Counter()
        results = []
        ql = q.lower()
        want = None
        if "wasalt" in ql and "darglobal" not in ql and "dar global" not in ql:
            want = "Wasalt"
        elif ("darglobal" in ql or "dar global" in ql) and "wasalt" not in ql:
            want = "DarGlobal"
        if want:
            raw = self.index.search(q + " " + history, k=80)
        for sc, d in raw:
            if want and d["source"] != want:
                continue
            if per_url[d["url"]] < 2:
                per_url[d["url"]] += 1
                results.append((sc, d))
            if len(results) == 8:
                break
        blocks, sources = [], []
        lq = self.listing_query(q)
        if lq:
            blocks.append(f"[W] Wasalt listings\n{lq}")
        for i, (_, d) in enumerate(results, 1):
            blocks.append(f"[{i}] {d['source']} — {d['title']} ({d['url']})\n{d['text']}")
            if d["url"] not in {s["url"] for s in sources}:
                sources.append({"title": d["title"], "url": d["url"], "source": d["source"]})
        return "\n\n".join(blocks), sources
