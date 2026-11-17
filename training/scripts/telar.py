"""Telar: a live, interactive view of an SFT run, served locally.

Reads ``runs/<RUN>/live.jsonl`` (written by d_train.py) and serves the page
``telar.html`` beside this file. Standard library only, bound to 127.0.0.1.

    python scripts/telar.py              # newest run, http://127.0.0.1:8765
    python scripts/telar.py --port 9000

Works while a run is training (the page polls) and after it ends (replay).
"""
import argparse, http.server, json, os, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")


def runs():
    found = [d for d in os.listdir(RUNS) if os.path.isfile(os.path.join(RUNS, d, "live.jsonl"))]
    return sorted(found, key=lambda d: os.path.getmtime(os.path.join(RUNS, d, "live.jsonl")), reverse=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        if url.path == "/":
            self.send(open(os.path.join(HERE, "telar.html"), "rb").read(), "text/html; charset=utf-8")
        elif url.path == "/runs":
            self.send(json.dumps(runs()).encode(), "application/json")
        elif url.path == "/live":
            name = q.get("run", [""])[0]
            if name not in runs():
                return self.send(b"[]", "application/json", 404)
            # Byte offset lets the page fetch only what was appended since its last poll.
            start = int(q.get("from", ["0"])[0])
            with open(os.path.join(RUNS, name, "live.jsonl"), "rb") as f:
                f.seek(start)
                chunk = f.read()
            end = chunk.rfind(b"\n") + 1  # never hand over a half-written line
            body = json.dumps({"next": start + end, "lines": chunk[:end].decode("utf-8").splitlines()})
            self.send(body.encode(), "application/json")
        else:
            self.send(b"not found", "text/plain", 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    print(f"Telar en http://127.0.0.1:{port}  (Ctrl+C para parar)", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
