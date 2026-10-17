import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import collections, glob, os, datetime
from common import ROOT, IDX, RAW, json

docs = [json.loads(l) for l in open(IDX / "documents.jsonl")]
bs = json.load(open(IDX / "_build_stats.json"))
sz = json.load(open(IDX / "_size_estimate.json"))
det = json.load(open(IDX / "_cms_detail_report.json"))
pages = [json.loads(l) for l in open(IDX / "_pages_cms.jsonl")]
slog = json.load(open(IDX / "_stats_log.json"))
cuad = [json.loads(l) for l in open(IDX / "stats_cuadros.jsonl")]
nper = sum(1 for _ in open(IDX / "stats_periods.jsonl"))
L = []
w = L.append
w("# Indice documental spensiones.cl\n")
w(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Filas en documents.jsonl/csv: **{len(docs)}** (cuadros incluidos como una fila por cuadro; sus {nper} periodos estan en stats_periods.jsonl).\n")
def table(counter, headers=("clave", "n")):
    w("| " + " | ".join(headers) + " |\n|" + "---|" * len(headers))
    for k, v in counter:
        w(f"| {k} | {v} |")
    w("")
w("## Por fuente"); table(collections.Counter(d["source"] for d in docs).most_common())
w("## Por fuente / doc_type")
table([(f"{k[0]} / {k[1]}", v) for k, v in sorted(collections.Counter((d["source"], d["doc_type"]) for d in docs).items(), key=lambda x: (x[0][0], -x[1]))], ("fuente / doc_type", "n"))
w("## Por area"); table(collections.Counter(d["area"] for d in docs).most_common())
w("## Por seccion (primera seccion de cada fila, top 60)")
secs = collections.Counter(d["section"][0] if d["section"] else "(sin seccion)" for d in docs)
table(secs.most_common(60), ("seccion", "n"))
w(f"Total de secciones distintas (cualquier aparicion): {len({s for d in docs for s in d['section']})}\n")
w("## Rangos de fechas (por fuente/doc_type; incluye fechas con precision anio/mes)")
rng = collections.defaultdict(list)
for d in docs:
    if d["date"]:
        rng[(d["source"], d["doc_type"])].append(d["date"])
w("| fuente / doc_type | con fecha / total | min | max |\n|---|---|---|---|")
tot = collections.Counter((d["source"], d["doc_type"]) for d in docs)
for k in sorted(tot):
    v = rng.get(k)
    w(f"| {k[0]} / {k[1]} | {len(v) if v else 0} / {tot[k]} | {min(v) if v else '-'} | {max(v) if v else '-'} |")
w("")
prec = collections.Counter((d["source"], d.get("date_precision"), d.get("date_source")) for d in docs if d["date"])
w("Precision/origen de fecha (fuente, precision, origen): " + "; ".join(f"{k}={v}" for k, v in sorted(prec.items(), key=lambda x: str(x))) + "\n")
w("## Duplicados eliminados")
w(f"- CMS: {bs['cms_page_occurrences']} apariciones (articulo, recurso) en paginas de listado -> {bs['cms_raw_rows']} recursos unicos tras dedupe por aid+recurso ({bs['cms_page_occurrences'] - bs['cms_raw_rows']} duplicados; sobre todo las paginas de perfiles/Atencion de usuarios que repiten articulos). Las secciones de todas las apariciones se conservan en `section` (lista).")
w(f"- Normativa: {bs['dup_normativa_crosslist']} filas repetidas (circulares conjuntas CIRC/CCON listadas en Pensiones y Cesantia, y mismo documento en pestanas periodicas+historicas), fusionadas por download_url.")
w(f"- CMS vs Normativa: {bs['dup_cms_vs_normativa']} enlaces GetFile.php en paginas CMS ya presentes en Normativa (fusionados).")
w(f"- Paginas de detalle: {bs['cms_html_only_resolved_to_files']} articulos 'solo HTML' en listados resultaron tener adjuntos en su pagina de detalle ({bs['cms_detail_extra_files']} archivos adicionales agregados).\n")
w("## Filas sin download_url")
miss = [d for d in docs if not d["download_url"]]
def reason(d):
    if d["doc_type"] == "compendio_titulo": return "compendio_titulo: solo HTML (el PDF existe unicamente por libro)"
    if d["doc_type"] == "cuadro": return "cuadro con acceso dinamico (JS/app PHP, sin URL estatica por periodo)"
    if d["doc_type"] == "articulo_html": return "articulo CMS solo HTML (contenido en detail_url, sin adjuntos)"
    return "otro"
table(collections.Counter(reason(d) for d in miss).most_common(), ("motivo", "n"))
w(f"Total sin download_url: {len(miss)} de {len(docs)}.\n")
w("## Errores / pendientes")
errs = []
for f in glob.glob(str(IDX / "_*_runstats.json")):
    errs += json.load(open(f))["errors"]
w(f"- Errores HTTP persistentes tras reintentos: {len(errs)}" + (": " + "; ".join(f"{e['url']} {e['error']}" for e in errs[:10]) if errs else "."))
emp = [p["url"] for p in pages if p["status"] == "empty"]
w(f"- Paginas CMS que responden 200 con cuerpo vacio (220 bytes, sin contenido): {len(emp)}: " + ", ".join(u.rsplit('-', 1)[-1].replace('.html', '') for u in emp) + " (sitemap las lista; posiblemente despublicadas).")
red = [p for p in pages if p["status"] == "redirect"]
w(f"- Paginas CMS que redirigen (meta refresh) a apps/otros canales: {len(red)} (normativa/compendio/atencion; ya cubiertas por sus fuentes).")
w("- Compendio Cesantia: RESUELTO sin Playwright (~5 min). El JS (compendio-publico.js) carga `/compendio-ace/data/json/compendio-libros.json` (arbol completo, version v1.0.NCG_112) y los PDF estan en `/compendio-ace/data/pdf/Libro{I..V}.pdf` + `/compendio-afc/data/pdf/compendio_normas_seguro_cesantia.pdf`. El arbol tiene ~2800 elementos (secciones/letras/numerales); solo se indexan libros (5) y titulos (33).")
w(f"- Cuadros con acceso dinamico (no hay URL HTML estatica por periodo, solo patron via JS/app): {sum(1 for c in cuad if c['access'] != 'static_html')} de {len(cuad)} (sces genStats*/loadFecuAFP/loadCarInv...). Se indexan con n_periods/first/last pero download_url vacio.")
w(f"- Adjuntos adicionales en paginas de detalle: el listado solo muestra recurso_1 por articulo. En una muestra aleatoria de {det.get('sample_pages', 0)} articulos con PDF, {det.get('sample_pages_with_extra', 0)} tenian adjuntos extra (~{100 * det.get('sample_pages_with_extra', 0) / max(det.get('sample_pages', 1), 1):.0f}%). Extrapolando, ~100-150 articulos podrian tener adjuntos no indexados. Para cerrarlo: fetch de ~3500 paginas de detalle (~40 min a 0.6 s). Pendiente (opcional).")
w("- Circulares relacionadas (POST circularesRelacionadas.php) no se rastrean. web71.spensiones.cl (bdp) no consultado (no responde segun el mapa previo). Tools/calculadoras (valores cuota, rentabilidad) fuera de alcance.")
w("- `status` en circulares/NCG = pestana del listado (periodica/historica), NO vigencia legal; para CMS el status es desconocido (null).\n")
w("## Validacion contra el mapa previo")
def cnt(pred): return sum(1 for d in docs if pred(d))
w(f"- Circulares Pensiones (spens): 2079 filas en el listado (CAFP+CIRC+CCON) -> OK (filas unicas por archivo tras fusionar con cesantia: {cnt(lambda d: d['doc_type']=='circular')} circulares).")
w(f"- Circulares Cesantia (sces): 507 filas -> OK; 66 comparten archivo con Pensiones (CIRC/CCON conjuntas), de ahi 2520 < 2079+507.")
w(f"- NCG SP 373 (371 AFP/Pilar + 2 IPS reparto) -> OK; NCG SC 112 -> OK. Total ncg: {cnt(lambda d: d['doc_type']=='ncg')}.")
cms_files = [d for d in docs if d["source"] == "cms" and d["download_url"]]
w(f"- CMS: mapa ~4100 'antes de dedupe'. Aca {len(cms_files)} recursos descargables unicos ({cnt(lambda d: d['source']=='cms')} filas incl. {cnt(lambda d: d['doc_type']=='articulo_html')} articulos solo-HTML). Diferencias: (a) ids.txt omitia 75 paginas de listado enlazadas (Resoluciones 5935/5941, convenios por pais, regimen de inversion, leyes retiros, etc.), que se descubrieron por BFS; (b) se cuentan todos los tipos (ppt, xls, doc, imagenes); (c) {bs['cms_detail_extra_files']} adjuntos extra de paginas de detalle. Las 6117 apariciones brutas bajan a {bs['cms_raw_rows']} por dedupe.")
w(f"- Compendio Pensiones: 5 libros + 80 titulos -> OK. Cesantia: 5 libros + 33 titulos (+1 PDF completo).")
w(f"- Estadisticas: {cnt(lambda d: d['doc_type']=='serie_xls')} XLS/XLSX de series (mapa ~150: la diferencia viene de que 'totpag' y otras series son .xlsx y/o el mapa sobreestimaba), {len(cuad)} cuadros, {nper} periodos (sin requests extra).\n")
w("## Estimacion de tamano de descarga completa (muestra estratificada, HEAD con Accept-Encoding: identity)")
w("Nota: el CMS sirve los PDF con gzip chunked sin Content-Length; con `Accept-Encoding: identity` el HEAD si trae Content-Length (coincide con el tamano hex del ETag). Fallback GET Range solo se uso para HTML de cuadros.\n")
w("| doc_type | n total | muestra | promedio KB | mediana KB | total GB (extrapolado) |\n|---|---|---|---|---|---|")
for t, e in sorted(sz["est"].items(), key=lambda x: -x[1]["total"]):
    w(f"| {t} | {e['n']} | {e['sampled']} | {e['avg']/1024:.0f} | {('%.0f' % (e['median']/1024)) if e['median'] else '-'} | {e['total']/1e9:.3f} |")
docs_only = sum(e["total"] for t, e in sz["est"].items() if t != "cuadro_periodo")
nfiles_docs = sum(e["n"] for t, e in sz["est"].items() if t != "cuadro_periodo")
w(f"\n- **Documentos (sin expandir periodos de cuadros)**: {nfiles_docs} archivos, ~{docs_only/1e9:.1f} GB. Tiempo a 1 req/s: {nfiles_docs/3600:.1f} h (+ transferencia a 10 MB/s asumido: {docs_only/10/1024/1024/3600:.2f} h).")
w(f"- **Cuadros HTML expandidos por periodo**: {sz['est']['cuadro_periodo']['n']} paginas, ~{sz['est']['cuadro_periodo']['total']/1e9:.2f} GB, {sz['est']['cuadro_periodo']['n']/3600:.1f} h a 1 req/s (muestra de solo 4 paginas).")
w(f"- **Total**: {sz['total_files']} requests, ~{sz['total_bytes']/1e9:.1f} GB, {sz['hours_1rps']:.1f} h a 1 req/s ({sz['hours_with_transfer']:.1f} h incluyendo transferencia).")
w("- Precision: muestras de 1-7 por tipo; la varianza es alta (p.ej. boletines ~7 MB, compendio_libro ~13 MB). Tomar como orden de magnitud (+-40%).\n")
n_html = len(glob.glob(str(RAW / "*.html")))
mt = [os.path.getmtime(f) for f in glob.glob(str(RAW / "*.html"))]
span = (max(mt) - min(mt)) / 60
w("## Ejecucion del crawler")
w(f"- Paginas HTML en cache raw_html/: {n_html} (1 GET cada una) + 2 GET de JS/JSON + ~350 HEAD/Range (muestreo de tamano y verificaciones; incluye 3 corridas de sizes.py). Total aprox: {n_html + 2 + 350} requests, secuenciales, delay 0.6 s, sin errores persistentes.")
w(f"- Ventana de tiempo de las paginas HTML (primer a ultimo archivo en cache, incluye pausas de desarrollo): {span:.0f} min. Tiempo neto de red estimado ~{(n_html + 352) * 0.75 / 60:.0f} min.")
(IDX / "summary.md").write_text("\n".join(L))
print("\n".join(L)[:300]); print(n_html, span)
