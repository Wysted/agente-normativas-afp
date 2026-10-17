import hashlib, time, json, sys
from pathlib import Path
import httpx

BASE = "https://www.spensiones.cl"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_html"
IDX = ROOT / "index"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
DELAY = 0.6
STATS = {"requests": 0, "cache_hits": 0, "errors": []}
_last = [0.0]
_client = httpx.Client(headers={"User-Agent": UA, "Accept-Language": "es-CL,es;q=0.9"},
                       follow_redirects=True, timeout=httpx.Timeout(60.0), verify=True)


def _throttle():
    wait = DELAY - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def request(method, url, retries=4, **kw):
    for attempt in range(retries):
        _throttle()
        STATS["requests"] += 1
        try:
            r = _client.request(method, url, **kw)
            if r.status_code >= 500 or r.status_code == 429:
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            return r
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            if attempt == retries - 1:
                STATS["errors"].append({"url": url, "error": repr(e)})
                return None
            time.sleep(2 ** (attempt + 1))


def get_html(url):
    """Cached GET returning decoded text (or None)."""
    if url.startswith("/"):
        url = BASE + url
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    p = RAW / f"{key}.html"
    if p.exists():
        STATS["cache_hits"] += 1
        return p.read_text(encoding="utf-8", errors="replace")
    r = request("GET", url)
    if r is None or r.status_code != 200:
        if r is not None:
            STATS["errors"].append({"url": url, "error": f"HTTP {r.status_code}"})
        return None
    txt = r.text
    p.write_text(txt, encoding="utf-8")
    (RAW / f"{key}.url").write_text(url)
    return txt


def save_stats(name):
    (IDX / f"_{name}_runstats.json").write_text(json.dumps(STATS, indent=1))
