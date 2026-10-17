import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import random, statistics, collections, re
from common import *
import common

docs = [json.loads(l) for l in open(IDX / "documents.jsonl")]
periods = [json.loads(l) for l in open(IDX / "stats_periods.jsonl")]
random.seed(42)
with_url = [d for d in docs if d["download_url"] and d["doc_type"] != "cuadro"]
by_type = collections.defaultdict(list)
for d in with_url:
    by_type[d["doc_type"]].append(d)
PLAN = {"circular": 6, "ncg": 4, "informe": 7, "oficio": 5, "ficha": 4, "presentacion": 4, "resolucion": 4, "minuta": 3, "convenio": 3,
        "estudio": 3, "boletin": 3, "memoria": 3, "guia_faq": 2, "serie_xls": 4, "compendio_libro": 4, "compendio_completo": 1,
        "ley": 2, "decreto": 1, "imagen": 1, "otro": 2, "normativa_consulta": 1, "norma": 1, "licitacion": 1, "reglamento": 1}
sample = []
for t, n in PLAN.items():
    pool = by_type.get(t, [])
    sample += [(t, d["download_url"]) for d in random.sample(pool, min(n, len(pool)))]
static_periods = [p for p in periods if p["html_url"]]
sample += [("cuadro_periodo", p["html_url"]) for p in random.sample(static_periods, 4)]


def size_of(url):
    r = request("HEAD", url, headers={"Accept-Encoding": "identity"})
    if r is not None and r.status_code == 200 and r.headers.get("content-length"):
        return int(r.headers["content-length"]), "HEAD"
    # fallbacks: ranged GET (identity, then default encoding); never read the body
    for hdrs in ({"Range": "bytes=0-0", "Accept-Encoding": "identity"}, {"Range": "bytes=0-0"}):
        common._throttle()
        common.STATS["requests"] += 1
        try:
            with common._client.stream("GET", url, headers=hdrs) as resp:
                cr = resp.headers.get("content-range")
                if cr and "/" in cr and cr.split("/")[-1].isdigit():
                    return int(cr.split("/")[-1]), "RANGE"
                if resp.status_code == 200 and resp.headers.get("content-length") and "identity" in hdrs.get("Accept-Encoding", ""):
                    return int(resp.headers["content-length"]), "GET-headers"
        except Exception as e:
            last = f"error {e!r}"
    return None, "unknown"


results = []
for t, u in sample:
    sz, how = size_of(u)
    results.append(dict(doc_type=t, url=u, size=sz, method=how))
json.dump(results, open(IDX / "_size_sample.json", "w"), indent=1)

per = collections.defaultdict(list)
for r in results:
    if r["size"] is not None:
        per[r["doc_type"]].append(r["size"])
overall = statistics.mean([s for v in per.values() for s in v])
counts = {t: len(v) for t, v in by_type.items()}
counts["cuadro_periodo"] = len(static_periods)
est = {}
for t, c in counts.items():
    v = per.get(t)
    if v:
        est[t] = dict(n=c, sampled=len(v), avg=statistics.mean(v), median=statistics.median(v), total=statistics.mean(v) * c, extrapolated_from="own sample")
    else:
        est[t] = dict(n=c, sampled=0, avg=overall, median=None, total=overall * c, extrapolated_from="overall mean (no sample)")
total_bytes = sum(e["total"] for e in est.values())
total_files = sum(e["n"] for e in est.values())
BW = 10 * 1024 * 1024
out = dict(est=est, total_bytes=total_bytes, total_files=total_files, failed=[r for r in results if r["size"] is None],
           hours_1rps=total_files / 3600, hours_with_transfer=(total_files + total_bytes / BW) / 3600, assumed_bw_mb_s=10)
json.dump(out, open(IDX / "_size_estimate.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "est"}, indent=1)[:1500])
for t, e in sorted(est.items(), key=lambda x: -x[1]["total"]):
    print(f"{t:22s} n={e['n']:5d} sampled={e['sampled']:2d} avg={e['avg']/1024:9.1f}KB total={e['total']/1e9:7.3f}GB")
save_stats("sizes")
