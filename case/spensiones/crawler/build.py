import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import csv, re, collections, datetime
from pathlib import Path
from common import ROOT, IDX, json
from parse import parse_date, extract_number, classify

def load(name):
    p = IDX / name
    return [json.loads(l) for l in open(p)] if p.exists() else []

def classify_area_sec(sp):
    s = sp.lower()
    if "cesant" in s or "protección al empleo" in s or "proteccion al empleo" in s or "afc" in s.split():
        return "cesantia"
    if s.startswith("quiénes somos") or s.startswith("quienes somos"):
        return "institucional"
    if re.search(r"pension|afp|afili|apv|ahorro|fondos|pilar|invalid|sis\b|scomp|compare|rentab|comision|comisión|cotiz|inversi|traspaso|bono|reforma|jubil|vejez|sobreviv|extranjero", s):
        return "pensiones"
    if re.match(r"(informes|fiscalizados|fiscalizaci|estad[ií]sticas|atenci|educaci)", s):
        return "pensiones"
    return "general"

def combine_area(secs, base=None):
    areas = {classify_area_sec(s) for s in secs} if secs else set()
    if base:
        areas = {base}
    if len(areas) > 1:
        areas.discard("general")
    return "+".join(sorted(areas)) if areas else "general"

stats = collections.Counter()
docs = []

# ---------------- normativa ----------------
nrows = load("_rows_normativa.jsonl")
by_url = {}
for r in nrows:
    k = r["download_url"]
    if k in by_url:
        e = by_url[k]
        stats["dup_normativa_crosslist"] += 1
        for s in r["section"]:
            if s not in e["section"]:
                e["section"].append(s)
        e["_statuses"].add(r["status"])
        continue
    r["_statuses"] = {r["status"]}
    by_url[k] = r
num_files = collections.defaultdict(set)
for r in by_url.values():
    if r["doc_type"] == "circular":
        num_files[r["number"]].add(r["download_url"])
for r in by_url.values():
    st = r.pop("_statuses")
    r["status"] = "periodica" if "periodica" in st else ("historica" if "historica" in st else None)
    r["status_note"] = "periodica/historica = pestana del listado (Circulares periodicas vs historicas); vigencia legal no informada"
    if r["doc_type"] == "circular":
        r["id"] = f"circular-{r['number']}" if len(num_files[r["number"]]) == 1 else f"circular-{r['subtype'].lower()}-{r['number']}"
    r["date_source"] = "listing"
    docs.append(r)

# ---------------- cms ----------------
cms = load("_rows_cms_raw.jsonl")
extra = load("_cms_detail_extra.jsonl")
n_page_occ = sum(len(r["sections"]) for r in cms)
stats["cms_raw_rows"] = len(cms)
stats["cms_page_occurrences"] = n_page_occ
have = {(r["aid"], r["res"]) for r in cms if r["aid"]}
html_extra = collections.defaultdict(list)
for e in extra:
    html_extra[e["aid"]].append(e)
by_aid = collections.defaultdict(list)
for r in cms:
    if r["aid"]:
        by_aid[r["aid"]].append(r)
# html_only rows that gained files from detail page
to_add = []
dropped_html = set()
for aid, es in html_extra.items():
    rs = by_aid[aid]
    if any(r.get("html_only") for r in rs) and not any(not r.get("html_only") for r in rs):
        base = next(r for r in rs if r.get("html_only"))
        dropped_html.add(aid)
        for e in es:
            if e["kind"] != "html_only":
                continue
            to_add.append(dict(aid=aid, res=e["res"], ext=e["ext"], ts=e["ts"], download_url=e["url"], detail_url=base["detail_url"],
                               title=base["title"], full_title=e["atitle"], abstract=base["abstract"], sections=list(base["sections"]),
                               listing_urls=list(base["listing_urls"]), groups=list(base["groups"]), from_detail=True, att_title=e["atitle"]))
    else:
        base = rs[0]
        for e in es:
            to_add.append(dict(aid=aid, res=e["res"], ext=e["ext"], ts=e["ts"], download_url=e["url"], detail_url=base["detail_url"],
                               title=base["title"], full_title=e["atitle"], abstract=base["abstract"], sections=list(base["sections"]),
                               listing_urls=list(base["listing_urls"]), groups=list(base["groups"]), from_detail=True, att_title=e["atitle"]))
cms2 = [r for r in cms if not (r.get("html_only") and r["aid"] in dropped_html)] + to_add
stats["cms_html_only_resolved_to_files"] = len(dropped_html)
stats["cms_detail_extra_files"] = len(to_add)
norm_by_url = {d["download_url"]: d for d in docs}
res_per_aid = collections.Counter(r["aid"] for r in cms2 if r["aid"])
seen_ids = set()
IMG = {"png", "jpg", "jpeg", "gif"}
PPT = {"ppt", "pptx", "pps"}
for r in cms2:
    if r["download_url"] in norm_by_url:
        e = norm_by_url[r["download_url"]]
        stats["dup_cms_vs_normativa"] += 1
        for s in r["sections"]:
            if s not in e["section"]:
                e["section"].append(s)
        continue
    if r["aid"]:
        bare = r["res"] in ("recurso_1", "html") or res_per_aid[r["aid"]] == 1
        rid = f"cms-aid-{r['aid']}" if bare else f"cms-aid-{r['aid']}-{re.sub(r'[^a-z0-9_]+', '-', r['res'].lower())}"
    else:
        import hashlib
        rid = "cms-url-" + hashlib.sha1(r["download_url"].encode()).hexdigest()[:10]
    if rid in seen_ids:
        rid += "-x"
    seen_ids.add(rid)
    text = " ".join(x for x in (r["full_title"], r["title"], r.get("abstract")) if x)
    d, prec = parse_date(r["title"] + " " + (r.get("abstract") or ""))
    dsrc = "title" if d else None
    if not d:
        d, prec = parse_date(r["full_title"] or "")
        dsrc = "title" if d else None
    if (not d or prec == "year") and r.get("groups"):
        g = next((x for x in r["groups"] if re.fullmatch(r"(19|20)\d{2}", x)), None)
        if g and not d:
            d, prec, dsrc = g, "year", "group"
    ext = (r["ext"] or "").lower() or None
    secpaths = r["sections"]
    if r.get("html_only"):
        dt = "articulo_html"
    elif ext in IMG:
        dt = "imagen"
    elif ext in PPT:
        dt = "presentacion"
    else:
        dt = classify(r["title"] + " " + (r.get("abstract") or ""), " | ".join(secpaths))
    area = combine_area(secpaths)
    ts_iso = datetime.datetime.fromtimestamp(int(r["ts"]), datetime.timezone.utc).strftime("%Y-%m-%d") if r.get("ts") else None
    r["title"] = r["title"] or r.get("abstract") or r.get("full_title") or (r["res"] if r.get("res") not in (None, "html") else None) or f"article {r['aid']}"
    title = r["title"] if not r.get("att_title") or r["att_title"] == r["title"] else f"{r['title']} - {r['att_title']}"
    docs.append(dict(id=rid, source="cms", section=secpaths, area=area, doc_type=dt, subtype=None, number=extract_number(r["title"]),
                     date=d, date_precision=prec, date_source=dsrc, title=title, description=r.get("abstract"), status=None,
                     detail_url=r["detail_url"], download_url=r["download_url"], file_ext=ext if not r.get("html_only") else "html",
                     listing_url=r["listing_urls"][0], listing_urls=r["listing_urls"], aid=r["aid"], last_modified_hint=ts_iso,
                     html_only=bool(r.get("html_only")), from_detail_page=bool(r.get("from_detail"))))

# ---------------- compendio ----------------
for r in load("_rows_compendio.jsonl"):
    r["date_source"] = None
    docs.append(r)

# ---------------- estadisticas ----------------
for r in load("_rows_series.jsonl"):
    r["date_source"] = None
    docs.append(r)
cuad = load("stats_cuadros.jsonl")
for c in cuad:
    m = re.match(r"\s*(\d+[A-Za-z]?)\.", c["title"])
    docs.append(dict(id=c["id"], source="estadisticas", section=[c["section"]], area=c["area"], doc_type="cuadro", subtype=c["access"],
                     number=m[1] if m else None, date=c["last_period"].replace("/", "-") if c["last_period"] else None, date_precision="month",
                     date_source="last_period", title=c["title"], status="vigente", detail_url=c["listing_url"],
                     download_url=c["latest_html_url"], file_ext="html" if c["latest_html_url"] else None, listing_url=c["listing_url"],
                     n_periods=c["n_periods"], first_period=c["first_period"], last_period=c["last_period"],
                     url_pattern=c["url_pattern_html"]))

# final id uniqueness check
ids = collections.Counter(d["id"] for d in docs)
dups = [k for k, v in ids.items() if v > 1]
assert not dups, dups[:10]

FIELDS = ["id", "source", "section", "area", "doc_type", "subtype", "number", "date", "date_precision", "date_source", "title", "description",
          "status", "detail_url", "download_url", "file_ext", "listing_url", "listing_urls", "aid", "last_modified_hint", "html_only",
          "from_detail_page", "n_periods", "first_period", "last_period", "url_pattern", "cms_id", "version", "status_note"]
with open(IDX / "documents.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps({k: d.get(k) for k in FIELDS}, ensure_ascii=False) + "\n")
with open(IDX / "documents.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for d in docs:
        row = {}
        for k in FIELDS:
            v = d.get(k)
            row[k] = " | ".join(v) if isinstance(v, list) else v
        w.writerow(row)
json.dump(dict(stats), open(IDX / "_build_stats.json", "w"), indent=1)
print(len(docs), dict(stats))
print(collections.Counter(d["source"] for d in docs))
print(collections.Counter(d["doc_type"] for d in docs).most_common())
