import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import re, collections
from common import *
from parse import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin

CH = f"{BASE}/portal/institucional/594"
FILE_RE = re.compile(r"articles-(\d+)_([\w\-\.]+?)\.(\w+)(?:\?ts=(\d+))?$", re.I)
DOC_EXT = re.compile(r"\.(pdf|xlsx?|docx?|pptx?|pps|zip|rar|csv|txt)(\?|$)", re.I)
IMG_EXT = {"png", "jpg", "jpeg", "gif"}
nodes = json.loads((ROOT / "nodes.json").read_text())


def clean(t):
    return re.sub(r"\s+", " ", (t or "").replace("\xa0", " ")).strip()


def page_url(kind, i):
    return f"{CH}/w3-{kind}-{i}.html"


def parse_page(h, url, pid):
    """Returns (info, files, articles, directlinks, discovered_links)."""
    s = BeautifulSoup(h, "lxml")
    main = s.find(id="main")
    if main is None:
        return None
    bc = main.select_one("p.breadcrumb")
    crumbs = [clean(x) for x in (bc.get_text("/", strip=True).split("/") if bc else []) if clean(x) and clean(x) != "Inicio"]
    h1 = main.find("h1")
    name = clean(h1.get_text(" ")) if h1 else (crumbs[-1] if crumbs else str(pid))
    if not crumbs or crumbs[-1] != name:
        crumbs.append(name)
    section_path = " > ".join(crumbs)
    # article metadata
    titles, abstracts = {}, {}
    for h3 in main.select("h3.titulo"):
        a = h3.find("a", href=True)
        if a:
            m = re.search(r"w3-article-(\d+)\.html", a["href"])
            if m:
                titles[m[1]] = (clean(a.get_text(" ")), urljoin(url, a["href"]))
    for p in main.select("p.abstract"):
        m = re.search(r"\baid-(\d+)\b", " ".join(p.get("class", [])))
        if m:
            abstracts[m[1]] = clean(p.get_text(" "))
    files, direct, art_links = [], [], {}
    for a in main.find_all("a", href=True):
        href = a["href"].strip()
        m = re.search(r"w3-article-(\d+)\.html", href)
        if m and m[1] not in art_links:
            art_links[m[1]] = (clean(a.get_text(" ")), urljoin(url, href))
        fm = FILE_RE.search(href)
        if fm:
            aid, res, ext, ts = fm[1], fm[2], fm[3].lower(), fm[4]
            grp = a.find_previous("h3", class_="titulo-grupo")
            files.append(dict(aid=aid, res=res, ext=ext, ts=ts, url=urljoin(url, href.split("?")[0]),
                              atitle=clean(re.sub(r"^Ir a\s*", "", a.get("title", ""))) or clean(a.get_text(" ")),
                              group=clean(grp.get_text(" ")) if grp else None))
        elif (DOC_EXT.search(href) or "GetFile.php" in href or "fo-propertyvalue" in href) and "spensiones.cl" in urljoin(url, href):
            direct.append(dict(url=urljoin(url, href), atitle=clean(a.get("title", "")) or clean(a.get_text(" "))))
    disc = set()
    for a in main.find_all("a", href=True):
        full = urljoin(url, a["href"])
        m = re.match(r"https?://www\.spensiones\.cl/portal/institucional/594/w3-(propertyvalue|propertyname)-(\d+)\.html", full)
        if m:
            disc.add((m[1], m[2]))
    return dict(name=name, crumbs=crumbs, section_path=section_path), files, titles, abstracts, art_links, direct, disc


def run():
    ids = [i for i in (ROOT / "ids.txt").read_text().split()]
    queue = collections.deque(("propertyvalue", i) for i in ids if not nodes.get(i, {}).get("redirect"))
    seen = {("propertyvalue", i) for i in ids}
    pages, rows_by_key, pend = [], {}, []
    discovered_new = []
    while queue:
        kind, i = queue.popleft()
        url = page_url(kind, i)
        h = get_html(url)
        rec = dict(node=i, kind=kind, url=url)
        if h is None:
            rec["status"] = "error"
            pages.append(rec)
            continue
        if re.search(r'http-equiv=["\']refresh', h, re.I) and len(h) < 3000:
            rec["status"] = "redirect"
            rec["target"] = re.search(r'url=([^"\'>]+)', h, re.I).group(1) if re.search(r'url=([^"\'>]+)', h, re.I) else None
            pages.append(rec)
            continue
        res = parse_page(h, url, i)
        if res is None:
            rec["status"] = "empty" if len(h) < 1000 else "nomain"
            rec["bytes"] = len(h)
            pages.append(rec)
            continue
        info, files, titles, abstracts, art_links, direct, disc = res
        rec.update(status="ok", name=info["name"], section_path=info["section_path"], n_files=len(files),
                   n_articles=len(art_links), n_direct=len(direct))
        pages.append(rec)
        for d in disc:
            if d not in seen:
                seen.add(d)
                queue.append(d)
                discovered_new.append(d)
        file_aids = set()
        for f in files:
            file_aids.add(f["aid"])
            t, durl = titles.get(f["aid"], (None, None))
            key = (f["aid"], f["res"])
            r = rows_by_key.setdefault(key, dict(
                aid=f["aid"], res=f["res"], ext=f["ext"], ts=f["ts"], download_url=f["url"], detail_url=durl or f"{CH}/w3-article-{f['aid']}.html",
                title=t or f["atitle"], full_title=f["atitle"], abstract=abstracts.get(f["aid"]), sections=[], listing_urls=[], groups=[]))
            if info["section_path"] not in r["sections"]:
                r["sections"].append(info["section_path"])
            if url not in r["listing_urls"]:
                r["listing_urls"].append(url)
            if f["group"] and f["group"] not in r["groups"]:
                r["groups"].append(f["group"])
        for aid, (t, durl) in {**art_links, **{k: v for k, v in titles.items()}}.items():
            if aid in file_aids:
                continue
            key = (aid, "html")
            r = rows_by_key.setdefault(key, dict(aid=aid, res="html", ext="html", ts=None, download_url=None, detail_url=durl,
                                                 title=titles.get(aid, (t,))[0] or t, full_title=t, abstract=abstracts.get(aid),
                                                 sections=[], listing_urls=[], groups=[], html_only=True))
            if info["section_path"] not in r["sections"]:
                r["sections"].append(info["section_path"])
            if url not in r["listing_urls"]:
                r["listing_urls"].append(url)
        for d in direct:
            key = ("url", d["url"])
            ext = (re.search(r"\.(\w+)(?:\?|$)", d["url"].split("namefile=")[-1]) or [None, None])[1]
            r = rows_by_key.setdefault(key, dict(aid=None, res=None, ext=(ext or "").lower() or None, ts=None, download_url=d["url"], detail_url=None,
                                                 title=d["atitle"], full_title=d["atitle"], abstract=None, sections=[], listing_urls=[], groups=[]))
            if info["section_path"] not in r["sections"]:
                r["sections"].append(info["section_path"])
            if url not in r["listing_urls"]:
                r["listing_urls"].append(url)
    return pages, rows_by_key, discovered_new


if __name__ == "__main__":
    pages, rows, new = run()
    with open(IDX / "_pages_cms.jsonl", "w") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(IDX / "_rows_cms_raw.jsonl", "w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("pages", len(pages), collections.Counter(p["status"] for p in pages), "discovered", len(new))
    print("rows", len(rows), "html_only", sum(1 for r in rows.values() if r.get("html_only")))
    save_stats("cms")
