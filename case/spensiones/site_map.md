# spensiones.cl - mapa del catalogo documental

Base: https://www.spensiones.cl (Newtenberg CMS + apps PHP). robots.txt: `Allow: /`, sin sitemap.xml (404). Sitemap HTML: /portal/institucional/594/w3-contents.html (256 nodos, ids en ids.txt). Sin WAF/captcha; curl con UA de navegador devuelve 200. Sin API/CSV; unica descarga masiva: series XLS.

| Seccion | Entry | Mecanismo | Items aprox |
|---|---|---|---|
| Circulares Pensiones (AFP/CCON/CIRC) | /apps/normativaSP/getNormativa.php?id=spens | 1 pagina HTML, 2 tabs, tabla N/Fecha/Materia, PDF directo /apps/GetFile.php?id=001&namefile=CAFP2425.pdf | ~2079 |
| Circulares Cesantia | ...?id=sces | idem | ~507 |
| NCG Pensiones (+IPS) | ...?id=ncgsp | idem, id=003 NCG-SP/NP0000372.pdf | ~373 |
| NCG Cesantia | ...?id=ncgsc | idem NCG-AFC/NC0000112.pdf | ~112 |
| Compendio Pensiones | /portal/compendio/596/w3-channel.html | 5 libros / 80 titulos HTML; PDF solo por libro: fo-propertyvalue-{libro}.pdf | 5 PDF + 80 HTML |
| Compendio Cesantia | /compendio-ace/index.html | app JS (php wrappers, cmp_N) | 5 libros |
| Resoluciones Pensiones / Cesantia | w3-propertyvalue-5935 / 5941 | listado estatico unico (sin paginar) -> articles-{aid}_recurso_1.pdf | 140 / ~75 |
| Normativa institucional (oficios ordinarios, RE) | w3-propertyvalue-6115 | estatico unico | 145 |
| Leyes/reglamentos, regimen de inversion, leyes retiros | 6287, 5939, 10345, 5942, 5940, 10917, 10397.. | estatico unico | ~32/9/33/15/10/1 |
| Normativa en consulta | 5974, 10810 | estatico, pocos items (nt_N_proy.pdf, minuta) | <10 |
| Fichas, informes mensuales, Compare Fondos, comisiones, inversiones, SIS, invalidez, cesantia | ver site_map.json (ids 10089-10262, 10398-10480) | estatico unico, patron w3-article-{aid}.html + articles-{aid}_recurso_1.pdf | 2.500+ PDFs total |
| Estudios (doc. trabajo, notas, panorama, boletin 1981-2009) | 10183, 10944, 10528, 10185, 10947 | idem | ~330 |
| Cuenta publica, BGI, CTI minutas | 10000, 10001, 6440 | idem | 16/20/186 |
| Centro de Estadisticas (cuadros) | /apps/centroEstadisticas/paginaCuadrosCCEE.php?menu=..&menuN1=..&menuN2=.. | HTML con select de periodos; /inf_estadistica/<tema>/<periodicidad>/YYYY/MM/<cod>.html | cientos de miles de combinaciones |
| Series estadisticas XLS | /safpstats/stats/.sc.php?_cid=41 (AFP), _cid=47 (cesantia) | tabla con XLS directo /inf_estadistica/series_afp/<tema>/<f>.xls | ~150 |

## Notas
- Detalle por item: w3-article-{aid}.html (solo titulo+abstract+PDF, sin fecha estructurada; la fecha esta en el titulo). Las paginas de perfiles (9767-9777, 9896) repiten articulos: deduplicar por aid.
- No existen secciones propias de Dictamenes, Jurisprudencia, Sanciones, Rezagos ni Traspasos. Oficios estan en Normativa institucional (6115). Traspasos: Compendio Libro I Titulo II/X y estadisticas; rezagos/deuda: 10096 y Compendio Libro II Titulo IX (cobranza). Comisiones: 10093/10094 + Compendio Libro IV Titulo IV. Inversiones: Compendio Libro IV + 5942 + 10090. APV: Compendio Libro II Titulos II-IV + 10092. PGU/APS/bono por hijo: Compendio Libro III Titulos V, VII, XVII; estadisticas bph/pilar. SIS: Libro V Titulo VII + 10187. Contabilidad/EEFF AFP: Compendio Libro IV Titulo VII + stats estfinafp/estfinfp.
- Bloqueos: Compendio Cesantia es JS-only; XLS dinamicos de cuadros via siSP.php/_sipc.php sin verificar; "Circulares relacionadas" (POST circularesRelacionadas.php) devolvio vacio; web71.spensiones.cl (bdp) no responde; Atencion ciudadana (/portal/institucional/597) y tramites son redirects a miportal.spensiones.gob.cl, sin documentos.

## Estrategia recomendada
1. Normativa (circulares, NCG): 4 GET a getNormativa.php, parsear tablas con selectolax/lxml (no ejecutar JS); URL PDF = GetFile.php. Re-ejecutar semanal (diff por numero).
2. Resto de listados CMS: iterar ids.txt (256 GET), por cada pagina extraer pares `articles-{aid}_*.pdf` + titulo (h3.titulo o atributo title); deduplicar por aid; delay 0.6s. Detectar redirects meta-refresh (ya mapeados en nodes.json).
3. Compendio Pensiones: bajar 5 PDFs de libros (o los 80 titulos HTML) para indexar; Cesantia con playwright.
4. Estadisticas: bajar series XLS (~150) y, si se necesita historico, enumerar options de los selects de cada cuadro.
