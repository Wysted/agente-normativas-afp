import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import re, random, collections
from common import *
from phase_cms import FILE_RE, clean
from bs4 import BeautifulSoup
from urllib.parse import urljoin

SAMPLE_PDF = 60
rows = [json.loads(l) for l in open(IDX / "_rows_cms_raw.jsonl")]
by_key = {(r["aid"], r["res"]): r for r in rows if r["aid"]}
random.seed(7)
html_only = [r for r in rows if r.get("html_only")]
pdf_aids = sorted({r["aid"] for r in rows if r["aid"] and not r.get("html_only")})
sample = random.sample(pdf_aids, SAMPLE_PDF)
targets = [(r["aid"], r["detail_url"], "html_only") for r in html_only] + [(a, by_key_url, "sample") for a in sample for by_key_url in [next(r["detail_url"] for r in rows if r["aid"] == a)]]
extra_found, report = [], collections.Counter()
detail_info = {}
for aid, durl, kind in targets:
    h = get_html(durl)
    if not h:
        report[f"{kind}_fetch_error"] += 1
        continue
    if re.search(r'http-equiv=["\']refresh', h, re.I) and len(h) < 3000:
        report[f"{kind}_redirect_to_listing"] += 1
        continue
    s = BeautifulSoup(h, "lxml")
    main = s.find(id="main")
    if main is None:
        report[f"{kind}_nomain"] += 1
        continue
    files = {}
    for a in main.find_all("a", href=True):
        m = FILE_RE.search(a["href"])
        if m and m[1] == aid:
            files[m[2]] = dict(res=m[2], ext=m[3].lower(), ts=m[4], url=urljoin(durl, a["href"].split("?")[0]),
                               atitle=clean(re.sub(r"^Ir a\s*", "", a.get("title", ""))) or clean(a.get_text(" ")))
    known = {k[1] for k in by_key if k[0] == aid}
    new = [f for f in files.values() if f["res"] not in known]
    report[f"{kind}_pages"] += 1
    report[f"{kind}_files_on_detail"] += len(files)
    report[f"{kind}_new_files"] += len(new)
    if kind == "sample" and new:
        report["sample_pages_with_extra"] += 1
    if kind == "html_only":
        report["html_only_with_files" if files else "html_only_pure_html"] += 1
    h1 = main.find("h1")
    for f in new:
        extra_found.append(dict(aid=aid, **f, detail_title=clean(h1.get_text(" ")) if h1 else None, kind=kind))
with open(IDX / "_cms_detail_extra.jsonl", "w") as f:
    for e in extra_found:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
json.dump(report, open(IDX / "_cms_detail_report.json", "w"), indent=1)
print(dict(report), len(extra_found))
save_stats("cms_detail")
