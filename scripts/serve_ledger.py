"""Tiny HTTP server that exposes the latest production ledger to a browser.

Usage:
    python scripts/serve_ledger.py
    # Then open http://localhost:8765 in a browser

Endpoints:
    GET /              -> viewer/index.html
    GET /ledger?latest=1  -> newest *_ledger.json from the output folder
    GET /ledger?path=...  -> specific ledger by absolute path
    GET /list          -> list of available ledgers (newest first)

No external deps. stdlib http.server. Designed to be left running in a
separate cmd window while you queue ComfyUI runs and watch the ledger
populate live in the browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER_HTML = REPO_ROOT / "viewer" / "index.html"
DEFAULT_LEDGER_DIR = Path(os.path.expanduser(
    "~/Documents/ComfyUI/output/old_time_radio"))
DEFAULT_PORT = 8765


def _list_ledgers(ledger_dir: Path):
    """Return ledger files in `ledger_dir` sorted newest-first by mtime."""
    if not ledger_dir.exists():
        return []
    files = []
    for p in ledger_dir.glob("*_ledger.json"):
        try:
            files.append((p.stat().st_mtime, p))
        except OSError:
            continue
    files.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in files]


def _read_ledger_json(path: Path) -> dict:
    """Read + parse a ledger file. Adds a `_meta.path` field so the viewer
    can show which file is currently rendered."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):
        data.setdefault("_meta", {})
        data["_meta"]["path"] = str(path)
        data["_meta"]["mtime"] = path.stat().st_mtime
    return data


class LedgerHandler(BaseHTTPRequestHandler):
    server_version = "OTRLedgerView/0.1"

    def __init__(self, *args, ledger_dir: Path = DEFAULT_LEDGER_DIR, **kw):
        self.ledger_dir = ledger_dir
        super().__init__(*args, **kw)

    # quiet-ish access log
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(),
                                        fmt % args))

    def _json(self, status: int, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _text(self, status: int, body: str, ctype: str = "text/plain"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        try:
            url = urlparse(self.path)
            qs = parse_qs(url.query or "")
            route = (url.path or "/").rstrip("/") or "/"

            if route in ("/", "/index.html"):
                self._serve_index()
                return

            if route == "/ledger":
                if "path" in qs:
                    p = Path(qs["path"][0])
                else:
                    files = _list_ledgers(self.ledger_dir)
                    if not files:
                        self._json(404, {"error": "no ledgers in "
                                                  + str(self.ledger_dir)})
                        return
                    p = files[0]
                if not p.exists():
                    self._json(404, {"error": "not found", "path": str(p)})
                    return
                try:
                    data = _read_ledger_json(p)
                except Exception as e:  # noqa: BLE001
                    self._json(500, {"error": "parse failed: " + str(e),
                                     "path": str(p)})
                    return
                self._json(200, data)
                return

            if route == "/list":
                files = _list_ledgers(self.ledger_dir)
                payload = [{
                    "name": p.name,
                    "path": str(p),
                    "mtime": p.stat().st_mtime,
                    "size": p.stat().st_size,
                } for p in files]
                self._json(200, payload)
                return

            self._text(404, "not found: " + route)
        except Exception as e:  # noqa: BLE001
            self._text(500, "server error: " + str(e))

    def _serve_index(self):
        if not VIEWER_HTML.exists():
            self._text(500, "viewer/index.html missing at " + str(VIEWER_HTML))
            return
        body = VIEWER_HTML.read_text(encoding="utf-8")
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def make_handler(ledger_dir: Path):
    """Bind ledger_dir into a request handler factory."""
    class Bound(LedgerHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, ledger_dir=ledger_dir, **kw)
    return Bound


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"port to listen on (default {DEFAULT_PORT})")
    p.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR),
                   help="directory containing *_ledger.json files")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1)")
    args = p.parse_args(argv)

    ledger_dir = Path(args.ledger_dir).resolve()
    handler = make_handler(ledger_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"OTR ledger viewer: serving {ledger_dir}")
    print(f"  -> open {url} in a browser")
    print(f"  -> ledger endpoint: {url}ledger?latest=1")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
