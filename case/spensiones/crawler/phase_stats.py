import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import re, collections
from common import *
from parse import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs

nodes = json.loads((ROOT / "nodes.json").read_text())


def clean(t):
    return re.sub(r"\s+", " ", (t or "").replace("\xa0", " ")).strip()


def norm(u, base=BASE + "/safpstats/stats/"):
    u = clean(u)
    if u.startswith(".sc.php"):
        u = "/safpstats/stats/" + u
    return urljoin(base, u)


def discover():
    seeds = [0, 41, 47]
    seen, queue, cuadros, others = set(), collections.deque(f"/safpstats/stats/.sc.php?_cid={c}" for c in seeds), {}, set()
    while queue:
        u = norm(queue.popleft())
        if u in seen:
            continue
        seen.add(u)
        h = get_html(u)
        if not h:
            continue
        s = BeautifulSoup(h, "lxml")
        for a in s.find_all("a", href=True):
            v = norm(a["href"])
            if "paginaCuadrosCCEE" in v:
                cuadros.setdefault(v, u)
            elif ".sc.php?_cid=" in v and v not in seen:
                queue.append(v)
            elif "getPerInfo" in v or "paginaMenuCCEE" in v:
                others.add(v)
    for n in nodes.values():
        r = n.get("redirect")
        if r and "paginaCuadrosCCEE" in r:
            cuadros.setdefault(norm(r), "nodes.json")
    return seen, cuadros, others


def parse_cuadro_page(h, url):
    s = BeautifulSoup(h, "lxml")
    main = s.find(id="main")
    h1 = main.find("h1") if main else None
    bc = main.select_one("p.breadcrumb") if main else None
    crumbs = [clean(x) for x in bc.get_text("/", strip=True).split("/") if clean(x)] if bc else []
    page_title = clean(h1.get_text(" ")) if h1 else None
    q = parse_qs(urlparse(url).query)
    menu, n1, n2 = (q.get(k, [""])[0].strip() for k in ("menu", "menuN1", "menuN2"))
    series, cuadros = [], []
    for tr in (main or s).find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if not tds:
            continue
        title = clean(tds[0].get_text(" "))
        sel = tr.find("select")
        raw = str(tr)
        xls = re.findall(r"location\.href='([^']+\.xlsx?)'", raw)
        if sel is not None:
            opts = [(o.get("value", "").strip(), clean(o.get_text())) for o in sel.find_all("option")]
            onc = [i.get("onclick", "") for i in tr.find_all("input")] + [a.get("href", "") for a in tr.find_all("a")]
            handlers = [re.search(r"(\w+)\(", x)[1] for x in onc if re.search(r"(\w+)\(", x)]
            cuadros.append(dict(title=title, name=sel.get("name"), options=opts, handlers=handlers, onclicks=onc))
        elif xls:
            for x in xls:
                series.append(dict(title=title, href=urljoin(BASE, x)))
    return dict(menu=menu, menuN1=n1, menuN2=n2, page_title=page_title, crumbs=crumbs), series, cuadros


def run():
    seen, cuadro_pages, others = discover()
    series_rows, cuadro_rows, period_rows, pend = [], [], [], []
    page_log = []
    for url in cuadro_pages:
        h = get_html(url)
        if not h:
            pend.append(dict(section=url, reason="fetch failed"))
            continue
        info, series, cuadros = parse_cuadro_page(h, url)
        page_log.append(dict(url=url, series=len(series), cuadros=len(cuadros), **info))
        sec = " > ".join(["Estadisticas"] + [c for c in info["crumbs"] if c != "Inicio"][-2:] + ([info["page_title"]] if info["page_title"] and (not info["crumbs"] or info["crumbs"][-1] != info["page_title"]) else []))
        area = "cesantia" if info["menu"] in ("sces", "ess") or "cesant" in sec.lower() else "pensiones"
        for k, sr in enumerate(series):
            fn = sr["href"].rsplit("/", 1)[-1]
            sid = "serie-" + re.sub(r"[^a-z0-9]+", "-", sr["href"].split("/inf_estadistica/")[-1].rsplit(".", 1)[0].lower()).strip("-")
            series_rows.append(dict(id=sid, source="estadisticas", section=[sec], area=area, doc_type="serie_xls", number=None, date=None,
                                    date_precision=None, title=sr["title"], status="vigente", detail_url=url, download_url=sr["href"],
                                    file_ext=sr['href'].rsplit('.',1)[-1].lower(), listing_url=url))
        for k, c in enumerate(cuadros):
            opts = c["options"]
            texts = [t for _, t in opts]
            vals = [v for v, _ in opts]
            keys = sorted(texts)
            cid = f"cuadro-{info['menu']}-{info['menuN1']}-{info['menuN2']}-{k + 1}"
            v0 = vals[0] if vals else ""
            hs = sorted(set(c["handlers"]))
            static = bool(v0) and not v0.startswith(("20", "19")) and "#" not in v0 and hs and all(h in ("openFileConPHP", "loadFile", "loadEstadFinancieras") for h in hs)
            def html_of(v):
                v = v.strip().lstrip("/")
                if not v.startswith("inf_estadistica/"):
                    v = "inf_estadistica/" + v
                return f"{BASE}/{v.replace('//', '/')}.html"
            latest = html_of(v0) if static else None
            pattern = None
            if static:
                m = re.match(r"^(.*?)/(\d{4})/(\d{1,2})/(.+)$", html_of(v0)[:-5])
                m2 = re.match(r"^(.*?)/(\d{4})/(.+)$", html_of(v0)[:-5])
                pattern = (f"{m[1]}/{{YYYY}}/{{MM}}/{m[4]}.html" if m else (f"{m2[1]}/{{YYYY}}/{m2[3]}.html" if m2 else None))
            segs = html_of(v0).split("/") if static else []
            per = next((x for x in segs if x in ("mensual", "trimestral", "semestral", "anual")), None)
            tema = segs[4] if static and len(segs) > 4 else None
            if per is None and texts:
                mons = {t[5:7] for t in texts if re.fullmatch(r"\d{4}/\d{2}", t)}
                if all(re.fullmatch(r"\d{4}", t) for t in texts):
                    per = "anual?"
                elif mons and mons <= {"03", "06", "09", "12"}:
                    per = "trimestral?"
                elif mons:
                    per = "mensual?"
            access = "static_html" if static else ("dynamic_app" if hs else "unknown")
            row = dict(id=cid, menu=info["menu"], menuN1=info["menuN1"], menuN2=info["menuN2"], orden=k, title=c["title"],
                       page_title=info["page_title"], section=sec, area=area, tema=tema,
                       periodicidad=per, n_periods=len(opts), first_period=keys[0] if keys else None,
                       last_period=keys[-1] if keys else None, access=access, url_pattern_html=pattern,
                       latest_html_url=latest, sample_option_value=v0,
                       handlers=hs, listing_url=url)
            cuadro_rows.append(row)
            for v, t in opts:
                period_rows.append(dict(cuadro_id=cid, period=t, value=v, html_url=html_of(v) if static else None))
    return series_rows, cuadro_rows, period_rows, pend, page_log, seen, others


if __name__ == "__main__":
    s, c, p, pend, log, seen, others = run()
    for name, rows in (("_rows_series", s), ("stats_cuadros", c), ("stats_periods", p)):
        with open(IDX / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(pending=pend, pages=log, cid_pages=sorted(seen), others=sorted(others)), open(IDX / "_stats_log.json", "w"), ensure_ascii=False, indent=1)
    print("series", len(s), "cuadros", len(c), "periods", len(p), "pages", len(log), "cid pages", len(seen), "pending", pend)
    save_stats("stats")
