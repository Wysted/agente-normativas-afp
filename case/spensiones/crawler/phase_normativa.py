import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import re, html
from common import *
from parse import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin

PAGES = {
    "spens": dict(kind="circular", area="pensiones", label="Circulares Sistema de Pensiones",
                  tabs={"CAFPPER": ("Circulares periodicas (Pensiones)", "periodica"),
                        "CAFPHIST": ("Circulares historicas (Pensiones)", "historica")}),
    "sces": dict(kind="circular", area="cesantia", label="Circulares Seguro de Cesantia",
                 tabs={"CAFCPER": ("Circulares periodicas (Cesantia)", "periodica"),
                       "CAFCHIST": ("Circulares historicas (Cesantia)", "historica")}),
    "ncgsp": dict(kind="ncg", area="pensiones", label="NCG Pensiones",
                  tabs={"NGSP": ("NCG Sistema de AFP y Pilar Solidario", None),
                        "NGIPSSA": ("NCG Sistema de Reparto (IPS)", None)}),
    "ncgsc": dict(kind="ncg", area="cesantia", label="NCG Seguro de Cesantia", tabs={}),
}


def run():
    rows = []
    pend = []
    for key, cfg in PAGES.items():
        url = f"{BASE}/apps/normativaSP/getNormativa.php?id={key}"
        h = get_html(url)
        if not h:
            pend.append({"section": cfg["label"], "url": url, "reason": "fetch failed"})
            continue
        s = BeautifulSoup(h, "lxml")
        tables = s.select("table[id^=documento]")
        for tb in tables:
            pane = tb.find_parent("div", class_="tab-pane")
            pid = pane.get("id") if pane else None
            sec, status = cfg["tabs"].get(pid, (cfg["label"], None))
            for tr in tb.select("tbody tr"):
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 3:
                    continue
                a = tds[0].find("a", href=True)
                if not a:
                    continue
                href = a["href"]
                m = re.search(r"namefile=([^&]+)", href)
                namefile = m[1] if m else href
                num_txt = a.get_text(" ", strip=True).replace("\xa0", " ").split()[0]
                num = str(int(num_txt)) if num_txt.isdigit() else num_txt
                dt = tds[1].get_text(strip=True)
                c = BeautifulSoup(str(tds[2]), "lxml")
                for x in c.select("a, div"):
                    x.decompose()
                title = re.sub(r"\s+", " ", c.get_text(" ", strip=True))
                prefix = re.match(r"(?:NCG-[A-Z\-]+/)?([A-Z]+?)0*\d+\.\w+$", namefile)
                sub = prefix[1] if prefix else None
                ext = namefile.rsplit(".", 1)[-1].lower() if "." in namefile else None
                if key in ("spens", "sces"):
                    rid = f"circular-{key}-{num}"
                elif key == "ncgsp":
                    rid = f"ncg-{'sp' if pid == 'NGSP' else 'ips-sa'}-{num}"
                else:
                    rid = f"ncg-sc-{num}"
                rows.append(dict(
                    id=rid, source="normativa", section=[sec], area=cfg["area"], doc_type=cfg["kind"],
                    subtype=sub, number=num, date=dt if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dt) else (parse_date(dt)[0]),
                    date_precision="day" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", dt) else parse_date(dt)[1],
                    title=title, status=status, detail_url=None,
                    download_url=urljoin(BASE, href), file_ext=ext, listing_url=url, ts=None))
    return rows, pend


if __name__ == "__main__":
    rows, pend = run()
    with open(IDX / "_rows_normativa.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(pend, open(IDX / "_pending_normativa.json", "w"), indent=1)
    import collections
    print(len(rows), collections.Counter((r["id"].rsplit("-", 1)[0]) for r in rows))
    ids = collections.Counter(r["id"] for r in rows)
    print("dup ids", [(k, v) for k, v in ids.items() if v > 1][:10])
    save_stats("normativa")
