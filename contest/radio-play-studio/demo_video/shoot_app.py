"""Full-size screenshots of the real Radio Play Studio page (headless Chrome,
DevTools protocol): the two story cards, the cast cards after Load the example,
and (when a run is live) the progress bar. Run: python shoot_app.py [progress]"""
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websocket

OUT = Path(__file__).resolve().parent / "shots"
OUT.mkdir(exist_ok=True)
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
URL = "http://127.0.0.1:8767/"
PORT = 9333

proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                         "--window-size=1920,1080", "--hide-scrollbars", "--force-device-scale-factor=1",
                         f"--user-data-dir={OUT.parent / 'chrome-profile'}", "about:blank"])
try:
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
            page = next(t for t in tabs if t["type"] == "page")
            break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
    n = 0

    def call(method, **params):
        global n
        n += 1
        ws.send(json.dumps({"id": n, "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == n:
                return msg.get("result", {})

    def js(expr):
        return call("Runtime.evaluate", expression=expr, awaitPromise=True,
                    returnByValue=True).get("result", {}).get("value")

    def shot(name):
        data = call("Page.captureScreenshot", format="png")["data"]
        (OUT / f"{name}.png").write_bytes(base64.b64decode(data))
        print("saved", name)

    call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
    call("Page.enable")
    call("Page.navigate", url=URL)
    for _ in range(120):                      # the form appears once ComfyUI answers
        if js("!!document.getElementById('mode_own')"):
            break
        time.sleep(0.5)
    time.sleep(1)
    js("document.body.style.zoom='1.25'; window.scrollTo(0,0)")
    time.sleep(0.5)
    shot("01_cards")
    js("document.getElementById('mode_own').click()")
    time.sleep(0.5)
    js("document.getElementById('load_example').click()")
    time.sleep(2)                             # thumbnails load
    # The cards keep the example's photos: My Story runs the open reference
    # twin, which paints each character's stills from them.
    js("document.querySelector('.cast, [class*=cast]') && document.querySelector('.cast, [class*=cast]').scrollIntoView({block:'start'})")
    time.sleep(0.5)
    shot("02_cast_cards")
    if "progress" in sys.argv[1:]:
        js("document.getElementById('run').scrollIntoView({block:'center'})")
        time.sleep(1)
        shot("03_progress")
    ws.close()
finally:
    proc.terminate()
