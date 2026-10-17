import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import re
from common import *
from parse import *
from bs4 import BeautifulSoup
from urllib.parse import urljoin

CP = f"{BASE}/portal/compendio/596"


def clean(t):
    return re.sub(r"\s+", " ", (t or "").replace("\xa0", " ")).strip()


def pensiones():
    url = f"{CP}/w3-channel.html"
    h = get_html(url)
    s = BeautifulSoup(h, "lxml")
    menu = s.select_one("#menu_lateral_compendio")
    pdfs = {}
    for a in s.find_all("a", href=True):
        m = re.search(r"fo-propertyvalue-(\d+)\.pdf", a["href"])
        if m:
            pdfs[m[1]] = urljoin(url, a["href"])
    rows = []
    sec = "Compendio de Normas del Sistema de Pensiones"
    for li in menu.find("ul").find_all("li", recursive=False):
        a = li.find("a", recursive=False)
        lid = re.search(r"propertyvalue-(\d+)", a["href"])[1]
        lt = clean(a.get_text(" "))
        n = re.match(r"Libro\s+([IVX]+)", lt)
        rows.append(dict(id=f"compendio-sp-libro-{n[1].lower()}", source="compendio", section=[sec], area="pensiones",
                         doc_type="compendio_libro", number=n[1], date=None, date_precision=None, title=lt, status="vigente",
                         detail_url=urljoin(url, a["href"]), download_url=pdfs.get(lid), file_ext="pdf" if pdfs.get(lid) else None,
                         listing_url=url, cms_id=lid))
        for sub in li.find_all("a", href=True):
            if sub is a:
                continue
            tid = re.search(r"propertyvalue-(\d+)", sub["href"])[1]
            tt = clean(sub.get_text(" "))
            tn = re.match(r"T[ií]tulo\s+([IVXL]+)", tt)
            rows.append(dict(id=f"compendio-sp-libro-{n[1].lower()}-titulo-{(tn[1] if tn else tid).lower()}", source="compendio",
                             section=[sec], area="pensiones", doc_type="compendio_titulo", number=tn[1] if tn else None, date=None,
                             date_precision=None, title=f"{lt.split(' ')[0]} {n[1]} / {tt}", status="vigente",
                             detail_url=urljoin(url, sub["href"]), download_url=None, file_ext=None, listing_url=url, cms_id=tid))
    return rows


def cesantia():
    url = f"{BASE}/compendio-ace/index.html"
    p = RAW / "compace_libros.json"
    if not p.exists():
        r = request("GET", f"{BASE}/compendio-ace/data/json/compendio-libros.json")
        p.write_text(r.text)
    d = json.loads(p.read_text())
    ver = d.get("id_version")
    sec = "Compendio de Normas del Seguro de Cesantia"
    rows = []
    for i, lib in enumerate(d["elementos"]):
        n = lib["nombre"].split()[-1]
        rows.append(dict(id=f"compendio-sc-libro-{n.lower()}", source="compendio", section=[sec], area="cesantia", doc_type="compendio_libro",
                         number=n, date=None, date_precision=None, title=f"{lib['nombre']}{lib['descripcion']}".replace("  ", " "), status="vigente",
                         detail_url=f"{BASE}/compendio-ace/index.html#cmp_{i}", download_url=None, file_ext=None, listing_url=url,
                         cms_id=str(lib["id_elemento_compendio"]), version=ver))
        for t in lib["elementos"]:
            tn = t["nombre"].split()[-1]
            rows.append(dict(id=f"compendio-sc-libro-{n.lower()}-titulo-{tn.lower()}", source="compendio", section=[sec], area="cesantia",
                             doc_type="compendio_titulo", number=tn, date=None, date_precision=None,
                             title=f"{lib['nombre']} / {t['nombre']}{t['descripcion']}".replace("  ", " "), status="vigente",
                             detail_url=f"{BASE}/compendio-ace/index.html#cmp_{i}", download_url=None, file_ext=None, listing_url=url,
                             cms_id=str(t["id_elemento_compendio"]), version=ver))
    return rows


if __name__ == "__main__":
    rows = pensiones() + cesantia()
    # discovered via JS (compendio-publico.js): data/pdf/<LibroX>.pdf and a full-compendium PDF; verify each with HEAD
    for r in rows:
        if r["id"].startswith("compendio-sc-libro-") and r["doc_type"] == "compendio_libro":
            u = f"{BASE}/compendio-ace/data/pdf/Libro{r['number']}.pdf"
            resp = request("HEAD", u)
            if resp is not None and resp.status_code == 200 and "pdf" in resp.headers.get("content-type", ""):
                r["download_url"], r["file_ext"] = u, "pdf"
    u = f"{BASE}/compendio-afc/data/pdf/compendio_normas_seguro_cesantia.pdf"
    resp = request("HEAD", u)
    if resp is not None and resp.status_code == 200 and "pdf" in resp.headers.get("content-type", ""):
        rows.append(dict(id="compendio-sc-completo", source="compendio", section=["Compendio de Normas del Seguro de Cesantia"], area="cesantia",
                         doc_type="compendio_completo", number=None, date=None, date_precision=None,
                         title="Compendio de Normas del Seguro de Cesantia (PDF completo)", status="vigente",
                         detail_url=f"{BASE}/compendio-ace/index.html", download_url=u, file_ext="pdf",
                         listing_url=f"{BASE}/compendio-ace/index.html", cms_id=None))
    with open(IDX / "_rows_compendio.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    print(len(rows), collections.Counter((r["id"].split("-")[1], r["doc_type"]) for r in rows))
    save_stats("compendio")
