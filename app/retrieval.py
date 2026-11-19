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
        parts.append(f"size digits in URL '{r['size_raw']}' (decimal point stripped, so NOT a reliable sqm figure; exact area is on the listing page)")
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
                "source": "Wasalt", "title": f"{name} — Wasalt categories", "url": main_url,
                "text": f"Wasalt (wasalt.sa) lists the following property categories in {name}, Saudi Arabia: "
                        + "; ".join(lines) + ". Browse: " + ", ".join(c["url"] for c in items[:6]),
            })

        self.cat_urls = {c["slug"]: c["url"] for c in cats}
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

    def browse_url(self, purpose, ptype, beds) -> str:
        """Best matching public Wasalt category page for a listing filter."""
        plural = {"rest": "rest-houses", "studio": "studio-apartments", "land": "lands"}.get(ptype or "", f"{ptype}s" if ptype else "properties")
        purpose_slug = purpose or "sale"
        cands = []
        if beds and ptype:
            cands.append(f"{beds}-bedroom-{plural}-for-{purpose_slug}-in-saudi-arabia")
        cands += [f"{plural}-for-{purpose_slug}-in-saudi-arabia", f"properties-for-{purpose_slug}-in-saudi-arabia"]
        for c in cands:
            if c in self.cat_urls:
                return self.cat_urls[c]
        return "https://wasalt.sa/en/"

    @staticmethod
    def _filters(text: str):
        t = text.lower()
        purpose = None
        if re.search(r"daily|booking|holiday|night", t):
            purpose = "daily-rental"
        elif re.search(r"\brent|lease|rental", t):
            purpose = "rent"
        elif re.search(r"sale|buy|purchase", t):
            purpose = "sale"
        ptype = next((v for w, v in TYPE_WORDS.items() if re.search(rf"\b{w}s?\b", t)), None)
        bm = re.search(r"(\d+)\s*-?\s*(?:bed|br\b|bhk|bedroom)", t)
        return purpose, ptype, (int(bm[1]) if bm else None)

    def listing_query(self, q: str, history: str = "", k: int = 8) -> tuple[str, str] | None:
        ql = q.lower()
        if not re.search(r"wasalt|listing|saudi|ksa|riyadh|jeddah|dammam|for (sale|rent)|rent|buy|bedroom|villa|apartment|land|chalet|floor|building|studio", ql):
            return None
        purpose, ptype, beds = self._filters(q)
        if history:  # follow-up: current question wins, history fills gaps
            hp, ht, hb = self._filters(history)
            purpose = purpose or hp
            if not ptype:
                ptype, beds = ht, beds if beds is not None else hb
        if not (purpose or ptype or beds):
            return None
        hits = [l for l in self.listings
                if (not purpose or l["purpose"] == purpose)
                and (not ptype or l["type"] == ptype)
                and (beds is None or l.get("bedrooms") == beds)]
        hits.sort(key=lambda l: l["lastmod"], reverse=True)
        f = ", ".join(x for x in [purpose and f"purpose={purpose}", ptype and f"type={ptype}",
                                  beds is not None and f"bedrooms={beds}"] if x)
        browse = self.browse_url(purpose, ptype, beds)
        out = [f"Browse on Wasalt: {browse}",
               f"Structured search of Wasalt listings ({f}): {len(hits):,} matching listings "
               f"(listing URLs do not include city; do not present URL size digits as square metres)."]
        out += [f"- {describe(h)}" for h in hits[:k]]
        return "\n".join(out), browse

    SMALLTALK = re.compile(r"^\W*(hi+|hello|hey|salam|salaam|marhaba|good (morning|afternoon|evening)|thanks?( you)?|thank u|ok(ay)?|cool|great|bye|who are you|what can you do|help)\W*$", re.I)
    FOLLOWUP = re.compile(r"\b(it|its|that|this|those|these|they|them|there|more|also|same|what about|and)\b", re.I)

    def context(self, q: str, history: str = "") -> tuple[str, list[dict]]:
        if self.SMALLTALK.match(q.strip()):
            return "", []
        # only blend in the previous question for short follow-ups ("what about villas?")
        use_hist = history and len(tokenize(q)) <= 5 and self.FOLLOWUP.search(q)
        query = q + (" " + history if use_hist else "")
        ql = q.lower()
        want = None
        if "wasalt" in ql and "darglobal" not in ql and "dar global" not in ql:
            want = "Wasalt"
        elif ("darglobal" in ql or "dar global" in ql) and "wasalt" not in ql:
            want = "DarGlobal"
        raw = self.index.search(query, k=80 if want else 30)
        raw = [(sc, d) for sc, d in raw if not want or d["source"] == want]
        top = raw[0][0] if raw else 0
        per_url: Counter = Counter()
        results = []
        for sc, d in raw:
            if sc < 0.45 * top:  # drop weak, loosely related matches
                break
            if per_url[d["url"]] < 2:
                per_url[d["url"]] += 1
                results.append(d)
            if len(results) == 8:
                break
        blocks, sources = [], []
        lq = self.listing_query(q, history if use_hist else "")
        if lq:
            text, browse = lq
            blocks.append(f"SOURCE: Wasalt listings database\nURL: {browse}\n{text}")
            sources.append({"title": "Matching Wasalt listings", "url": browse, "source": "Wasalt"})
            results = results[:4]  # the structured result is the main evidence
        for d in results:
            blocks.append(f"SOURCE: {d['source']} — {d['title']}\nURL: {d['url']}\n{d['text']}")
            if d["url"] not in {s["url"] for s in sources}:
                sources.append({"title": d["title"], "url": d["url"], "source": d["source"]})
        return "\n\n".join(blocks), sources
