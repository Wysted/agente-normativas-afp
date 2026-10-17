import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import sys, re
from common import *
for u in sys.argv[1:]:
    h = get_html(u)
    print(u, len(h) if h else None)
