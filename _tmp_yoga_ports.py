import json
import urllib.error
import urllib.request


def peek(port):
    url = f"http://127.0.0.1:{port}/queue"
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            q = json.loads(resp.read().decode("utf-8"))
        print(f":{port} up running={len(q.get('queue_running') or [])} pending={len(q.get('queue_pending') or [])}")
    except Exception as exc:
        print(f":{port} {type(exc).__name__}: {exc}")


peek(8000)
peek(8188)
