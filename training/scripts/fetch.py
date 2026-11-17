"""Download the held-out training corpus with the repo fetcher's own helper."""
import json, os, sys, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.fetch_corpus import ARXIV_PDF_URL, WIKI_PDF_URL, _download, _wiki_filename
spec = json.load(open(os.path.join(os.path.dirname(__file__), "heldout.json")))
root = os.path.join(os.path.dirname(__file__), "..", "docs")
for lang, s in spec.items():
    jobs = [(ARXIV_PDF_URL.format(id=i), f) for i, f in s.get("arxiv", [])]
    jobs += [(WIKI_PDF_URL.format(lang=lang, title=urllib.parse.quote(t.replace(" ", "_"), safe="")), _wiki_filename(t)) for t in s.get("wikipedia", [])]
    for url, fn in jobs:
        dest = os.path.join(root, lang, fn)
        if os.path.exists(dest):
            print("have", lang, fn); continue
        ok, detail = _download(url, dest)
        print("ok " if ok else "FAIL", lang, fn, detail, flush=True)
