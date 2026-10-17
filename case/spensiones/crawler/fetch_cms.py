import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import time
from common import *
t=time.time()
ids=[l.strip() for l in (ROOT/"ids.txt").read_text().split() if l.strip()]
nodes=json.loads((ROOT/"nodes.json").read_text())
for i in ids:
    if nodes.get(i,{}).get("redirect"): continue
    get_html(f"{BASE}/portal/institucional/594/w3-propertyvalue-{i}.html")
print("done",STATS["requests"],STATS["cache_hits"],len(STATS["errors"]),time.time()-t)
save_stats("fetch_cms")
