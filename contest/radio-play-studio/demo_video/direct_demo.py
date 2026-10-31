"""A 2D demo of the Studio's /direct page: the flat tile picker, then its VR ring.

Everything on screen is the real page, running locally against a Studio that
reads its choices from ComfyUI. The flat half is driven with real key presses
and one sweep of the page's own MIDI handler. The VR half calls the page's own
WebXR drawing code (initGL, xrFrame, xrInputs, ringTiles) once per video frame,
with a simulated headset: a head pose that turns, a thumbstick, and the
trigger and grip handlers the page binds. Nothing presses MAKE, so no episode
is ever rendered.

Run the Studio first (any port; pass it with --url), then:
  python direct_demo.py narrate    one wav per narration beat (Kokoro, CPU)
  python direct_demo.py cards      the DIRECT, IN VR and end cards (Blender)
  python direct_demo.py shoot      the page, flat then VR (headless Chrome)
  python direct_demo.py cut        captions, key badges, voice, and the two mp4s
  python direct_demo.py all        the four in order
"""
import argparse
import base64
import json
import math
import subprocess
import time
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\demo_cuts")
WORK = OUT_DIR / "direct_vr"
FINAL = OUT_DIR / "radio_play_studio_direct_vr.mp4"
PHONE = OUT_DIR / "radio_play_studio_direct_vr_720p.mp4"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
KOKORO = Path(r"C:\ComfyUI-Models\TTS\KokoroTTS")
VOICE = "am_michael"
FONTS = Path(r"C:\Windows\Fonts")
DEBUG_PORT = 9334
FPS = 30
W, H = 1920, 1080
CSS_W, CSS_H = 1440, 810          # rendered at 4/3 so the tiles read on a phone
LEAD_S, TAIL_S = 0.25, 0.45       # air before and after each narration beat
FOVY = math.radians(60)           # about 92 degrees across at 16:9, near a headset's
RING_STEP = 0.42                  # the page's RING.step, the angle between tiles

YELLOW = (255, 217, 31)
NIGHT = (26, 13, 46)
INK = (244, 237, 251)

# (key, what Kokoro says, what the caption shows)
BEATS = [
    ("f1", "This is Direct, the quick way to pick an episode in Radio Play Studio.", None),
    ("f2", "It shows one choice at a time, in big tiles. The arrow keys move, and Enter picks.", None),
    ("f3", "First the machine, then the way of filming, which decides the workflow.", None),
    ("f4", "Then the four dials: language, story bank, visual style, and acts.", None),
    ("f5", "A MIDI knob scrubs through the choices, and a gamepad works too.", None),
    ("f6", "Backspace steps back, so you can change your mind.", None),
    ("f7", "When everything is picked, one tile is left: Make. It sends your picks to your own G.P.U.",
     "When everything is picked, one tile is left: Make. It sends your picks to your own GPU."),
    ("v1", "Direct also runs in a V.R. headset. The same tiles wrap around you in a ring.",
     "Direct also runs in a VR headset. The same tiles wrap around you in a ring."),
    ("v2", "Look at a tile to light it, and pull the trigger to pick it.", None),
    ("v3", "The thumbstick turns the ring, and a squeeze on the grip goes back.", None),
    ("v4", "Story bank, visual style, and acts, without putting the controllers down.", None),
    ("v5", "When it's all picked, pull the trigger on Make, and your own G.P.U. films the episode.",
     "When it's all picked, pull the trigger on Make, and your own GPU films the episode."),
    ("end", "Radio Play Studio. Open source, on your own G.P.U.", None),
]
SPOKEN = {k: s for k, s, _ in BEATS}
CAPTION = {k: (c or s) for k, s, c in BEATS}

# The flat walk. ("to", value) presses the arrow key until that tile is lit;
# ("knob", value) sweeps the page's MIDI handler across every choice and back.
FLAT = [
    ("f1", [("hold",)]),
    ("f2", [("key", "ArrowRight"), ("key", "ArrowRight"), ("key", "ArrowLeft"), ("key", "ArrowLeft"),
            ("key", "Enter")]),
    ("f3", [("key", "ArrowRight"), ("key", "ArrowLeft"), ("key", "Enter")]),
    ("f4", [("to", "Japanese"), ("key", "Enter"), ("to", "shakespeare"), ("key", "Enter")]),
    ("f5", [("knob", "storybook_engraving"), ("key", "Enter"), ("to", 2), ("key", "Enter")]),
    ("f6", [("key", "Backspace"), ("key", "ArrowRight"), ("key", "Enter")]),
    ("f7", [("hold",)]),
]

# The VR walk, one list per beat. ("look", value, weight) turns the head to that
# tile; ("look", None, weight) faces straight ahead; ("gaze", k, weight) turns to
# the tile k steps from the middle; ("hold", weight) keeps still; ("trigger",)
# and ("squeeze",) fire the handlers the page binds to selectstart and
# squeezestart; ("stick", +1 or -1) pushes the thumbstick for one step.
VR = [
    ("v1", [("hold", 1), ("gaze", -2, 2), ("gaze", 2, 3), ("look", None, 2)]),
    ("v2", [("hold", 0.6), ("trigger",), ("gaze", 1, 1.2), ("hold", 0.6), ("look", "foley", 1.2),
            ("hold", 0.5), ("trigger",), ("hold", 0.4)]),
    ("v3", [("look", None, 0.6), ("stick", 1), ("hold", 0.4), ("stick", 1), ("hold", 0.4),
            ("look", "Spanish", 0.5), ("trigger",), ("hold", 0.8), ("squeeze",), ("look", "Spanish", 0.6),
            ("hold", 0.3), ("trigger",), ("hold", 0.3)]),
    ("v4", [("look", "original", 1), ("hold", 0.4), ("trigger",), ("look", "cartoon", 1), ("hold", 0.4),
            ("trigger",), ("look", 2, 1), ("hold", 0.4), ("trigger",), ("hold", 0.3)]),
    ("v5", [("look", None, 1), ("hold", 3)]),
]
STICK_S = 0.2                      # one push: held long enough for exactly one step
BADGE_S = 0.6                      # how long a key or button badge stays up


# Beats whose moves need more room than their line takes to say.
EXTRA_S = {"f5": 1.0, "v2": 1.5, "v3": 2.5, "v4": 1.5}


def beat_seconds(lengths, key):
    return round((LEAD_S + lengths[key] + TAIL_S + EXTRA_S.get(key, 0)) * FPS) / FPS


# --------------------------------------------------------------------------
# narrate
# --------------------------------------------------------------------------
def narrate():
    import numpy as np
    from kokoro_onnx import Kokoro

    out = WORK / "narration"
    out.mkdir(parents=True, exist_ok=True)
    voices = sorted(KOKORO.joinpath("voices").glob("_onnx_voices.*.npz"))[0]
    tts = Kokoro(str(KOKORO / "onnx" / "model.onnx"), str(voices))
    lengths = {}
    for key, text in SPOKEN.items():
        samples, rate = tts.create(text, voice=VOICE, speed=1.0, lang="en-us")
        pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
        with wave.open(str(out / f"{key}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        lengths[key] = round(len(pcm) / rate, 3)
        print(f"{key}: {lengths[key]:.2f} s")
    (out / "lengths.json").write_text(json.dumps(lengths, indent=1), encoding="utf-8")
    print("total voice:", round(sum(lengths.values()), 1), "s")


def load_lengths():
    return json.loads((WORK / "narration" / "lengths.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# cards
# --------------------------------------------------------------------------
def cards():
    jobs = [("direct", ["section", "DIRECT"]), ("in_vr", ["section", "IN VR"]), ("end", ["end"])]
    for name, args in jobs:
        out = WORK / "cards" / name
        for old in out.glob("*.png"):
            old.unlink()
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(HERE / "reel_cards.py"), "--",
                        *args, str(out)], check=True, stdout=subprocess.DEVNULL)
        print(name, len(list(out.glob("*.png"))), "frames")


# --------------------------------------------------------------------------
# shoot
# --------------------------------------------------------------------------
class Page:
    """One headless Chrome tab, spoken to over the DevTools protocol."""

    def __init__(self, url):
        import websocket

        self.url = url
        profile = WORK / "chrome-profile"
        self.proc = subprocess.Popen([
            CHROME, "--headless=new", f"--remote-debugging-port={DEBUG_PORT}", f"--window-size={W},{H}",
            "--hide-scrollbars", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--autoplay-policy=no-user-gesture-required", f"--user-data-dir={profile}", "about:blank"])
        tab = None
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json"))
                tab = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Chrome is still starting
                time.sleep(0.2)
        if tab is None:
            raise RuntimeError("headless Chrome did not open a tab")
        self.ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=120, suppress_origin=True)
        self.n = 0
        self.call("Emulation.setDeviceMetricsOverride", width=CSS_W, height=CSS_H,
                  deviceScaleFactor=W / CSS_W, mobile=False)
        self.call("Page.enable")

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        result = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in result:
            raise RuntimeError(f"page error: {result['exceptionDetails'].get('exception', {}).get('description')}")
        return result.get("result", {}).get("value")

    def open(self):
        self.call("Page.navigate", url=self.url)
        for _ in range(240):
            time.sleep(0.25)
            try:
                if self.js("typeof S !== 'undefined' && S.picks.length > 0 && !S.busy && !S.making"):
                    break
            except RuntimeError:
                pass
        else:
            raise RuntimeError("the /direct page never finished reading its choices")
        time.sleep(0.6)

    def settle(self):
        for _ in range(200):
            if self.js("!S.busy"):
                break
            time.sleep(0.1)
        time.sleep(0.25)

    def key(self, name):
        codes = {"ArrowRight": 39, "ArrowLeft": 37, "Enter": 13, "Backspace": 8}
        for kind in ("keyDown", "keyUp"):
            self.call("Input.dispatchKeyEvent", type=kind, key=name, code=name,
                      windowsVirtualKeyCode=codes[name], nativeVirtualKeyCode=codes[name])
        self.settle()

    def shot(self, path):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()


def live_state(page):
    return page.js("({step: S.step, cursor: S.cursor, base: XR.base, atMake: atMake(), "
                   "label: atMake() ? 'MAKE' : live().label, "
                   "values: atMake() ? [] : live().choices.map((c) => String(c.value))})")


def index_of(page, value):
    state = live_state(page)
    want = str(value)
    if want not in state["values"]:
        raise RuntimeError(f"{want!r} is not a choice on the {state['label']} step: {state['values']}")
    return state["values"].index(want), state


def shoot_flat(page, lengths):
    out = WORK / "flat"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    page.open()
    stills = []                  # {file, beat, badge, fixed}

    def snap(beat, badge=None, fixed=None):
        path = out / f"{len(stills):04d}.png"
        page.shot(path)
        stills.append({"file": path.name, "beat": beat, "badge": badge, "fixed": fixed})

    for beat, actions in FLAT:
        snap(beat)
        for action in actions:
            kind = action[0]
            if kind == "hold":
                continue
            if kind == "key":
                page.key(action[1])
                snap(beat, badge={"ArrowRight": "\u2192", "ArrowLeft": "\u2190"}.get(action[1], action[1].upper()))
            elif kind == "to":
                target, state = index_of(page, action[1])
                n = len(state["values"])
                ahead = (target - state["cursor"]) % n
                name, steps = ("ArrowRight", ahead) if ahead <= n - ahead else ("ArrowLeft", n - ahead)
                for _ in range(steps):
                    page.key(name)
                    snap(beat, badge="\u2192" if name == "ArrowRight" else "\u2190")
            elif kind == "knob":
                target, state = index_of(page, action[1])
                n = len(state["values"])
                # The page maps a CC value to a choice: round(v / 127 * (n - 1)).
                path = list(range(state["cursor"], n)) + list(range(n - 2, target - 1, -1))
                for i in path[1:]:
                    page.js(f"onMIDI({{data: [0xb0, 1, {round(i * 127 / (n - 1))}]}})")
                    time.sleep(0.08)
                    snap(beat, badge="MIDI KNOB (SIMULATED)", fixed=0.2)
            else:
                raise ValueError(kind)
    # Each beat's stills share its length: knob steps are quick, the rest even.
    for beat, _ in FLAT:
        mine = [s for s in stills if s["beat"] == beat]
        total = beat_seconds(lengths, beat)
        fixed = sum(s["fixed"] or 0 for s in mine)
        loose = [s for s in mine if not s["fixed"]]
        frames_left = round((total - fixed) * FPS)
        for i, s in enumerate(loose):
            share = frames_left // len(loose) + (1 if i < frames_left % len(loose) else 0)
            s["frames"] = share
        for s in mine:
            if s["fixed"]:
                s["frames"] = round(s["fixed"] * FPS)
    (out / "timeline.json").write_text(json.dumps(stills, indent=1), encoding="utf-8")
    print("flat:", len(stills), "stills,", round(sum(s["frames"] for s in stills) / FPS, 2), "s")


VR_SETUP = """
(() => {
  const W = %(w)d, H = %(h)d, FOVY = %(fovy)r;
  const canvas = document.getElementById("xr-canvas");
  canvas.style.display = "block";
  canvas.width = W; canvas.height = H;
  const gl = canvas.getContext("webgl", {alpha: false, preserveDrawingBuffer: true});
  if (!gl) throw new Error("WebGL is not available in this Chrome");
  const viewport = {x: 0, y: 0, width: W, height: H};
  XR.gl = gl;
  XR.refSpace = {};
  XR.session = {requestAnimationFrame: () => 0, inputSources: [],
                renderState: {baseLayer: {framebuffer: null, getViewport: () => viewport}}};
  initGL(gl);
  XR.base = S.cursor;
  const f = 1 / Math.tan(FOVY / 2), aspect = W / H, near = 0.05, far = 50;
  const P = new Float32Array([f / aspect, 0, 0, 0,  0, f, 0, 0,
                              0, 0, (far + near) / (near - far), -1,  0, 0, 2 * far * near / (near - far), 0]);
  const rotX = (a) => { const o = ident(); const c = Math.cos(a), s = Math.sin(a);
                        o[5] = c; o[6] = s; o[9] = -s; o[10] = c; return o; };
  window.__demoFrame = (t, yaw, pitch, stick) => {
    XR.session.inputSources = stick ? [{gamepad: {axes: [0, 0, stick, 0], buttons: []}}] : [];
    const m = mul(rotY(-yaw), rotX(pitch));
    const inv = mul(rotX(-pitch), rotY(yaw));
    xrFrame(t, {getViewerPose: () => ({views: [{projectionMatrix: P,
                                                transform: {matrix: m, inverse: {matrix: inv}}}]})});
    return S.cursor;
  };
  return true;
})()
"""


def ease(x):
    return x * x * (3 - 2 * x)


def shoot_vr(page, lengths):
    out = WORK / "vr"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    page.open()
    page.js(VR_SETUP % {"w": W, "h": H, "fovy": FOVY})
    frames = []                       # {beat, badge}
    clock_ms = 1000.0                 # the page's stick rest is measured from its last push
    head = {"yaw": 0.0}

    def frame(beat, yaw, stick=0.0, badge=None):
        nonlocal clock_ms
        t = clock_ms / 1000
        # A held head is never quite still: a slow sway well inside the page's
        # gaze tolerance, so the lit tile does not flicker.
        sway = 0.012 * math.sin(t * 1.3) + 0.006 * math.sin(t * 3.1)
        pitch = -0.03 + 0.01 * math.sin(t * 0.9)
        page.js(f"__demoFrame({clock_ms:.1f}, {yaw + sway:.5f}, {pitch:.5f}, {stick})")
        path = out / f"{len(frames):04d}.png"
        page.shot(path)
        frames.append({"beat": beat, "badge": badge})
        clock_ms += 1000 / FPS

    def gaze_yaw(value):
        if value is None:
            return 0.0
        target, state = index_of(page, value)
        n = len(state["values"])
        span = min(n, 7)
        k = (target - state["base"]) % n
        if k > (span - 1) // 2:
            k -= n
        if not -(span // 2) <= k <= (span - 1) // 2:
            raise RuntimeError(f"{value!r} is off the ring on {state['label']}; push the stick first")
        return k * RING_STEP

    for beat, moves in VR:
        total = round(beat_seconds(lengths, beat) * FPS)
        fixed = sum(round(STICK_S * FPS) for m in moves if m[0] == "stick")
        weights = sum(m[-1] for m in moves if m[0] in ("look", "gaze", "hold"))
        unit = (total - fixed) / weights
        planned = [round(m[-1] * unit) if m[0] in ("look", "gaze", "hold") else 0 for m in moves]
        planned[-1] += total - fixed - sum(planned)           # land the beat on its exact length
        start = len(frames)
        badge = None
        for move, count in zip(moves, planned):
            kind = move[0]
            if kind in ("trigger", "squeeze"):
                page.js("pick()" if kind == "trigger" else "back()")
                badge = ("TRIGGER" if kind == "trigger" else "GRIP SQUEEZE", len(frames))
                continue
            if kind == "stick":
                badge = ("STICK \u2192" if move[1] > 0 else "STICK \u2190", len(frames))
                for i in range(round(STICK_S * FPS)):
                    frame(beat, head["yaw"], stick=0.9 * move[1] if i < 3 else 0.0, badge=badge[0])
                continue
            if kind == "hold":
                goal = head["yaw"]
            elif kind == "gaze":
                goal = move[1] * RING_STEP
            else:
                goal = gaze_yaw(move[1])
            begin = head["yaw"]
            for i in range(count):
                x = ease((i + 1) / count)
                head["yaw"] = begin + (goal - begin) * x
                shown = badge[0] if badge and len(frames) - badge[1] < BADGE_S * FPS else None
                frame(beat, head["yaw"], badge=shown)
        assert len(frames) - start == total, (beat, len(frames) - start, total)
    (out / "timeline.json").write_text(json.dumps(frames, indent=1), encoding="utf-8")
    print("vr:", len(frames), "frames,", round(len(frames) / FPS, 2), "s")


def shoot(url):
    lengths = load_lengths()
    page = Page(url)
    try:
        shoot_flat(page, lengths)
        shoot_vr(page, lengths)
    finally:
        page.close()


# --------------------------------------------------------------------------
# cut
# --------------------------------------------------------------------------
def font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONTS / name), size)


def wrap(draw, text, face, width):
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=face) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def overlay(tag, caption, badge):
    """The transparent layer drawn over a frame: the tag, the caption, a badge."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if tag:
        face = font("impact.ttf", 30)
        w = d.textlength(tag, font=face)
        # Bottom-left, over the page's "the full page" link: the header and the
        # pick trail along the top stay readable.
        d.rounded_rectangle((16, H - 90, 54 + w + 18, H - 24), radius=10, fill=(*NIGHT, 255))
        d.text((54, H - 76), tag, font=face, fill=YELLOW)
    if caption:
        face = font("seguisb.ttf", 44)
        lines = wrap(d, caption, face, 1500)
        line_h = 58
        top = H - 150 - line_h * len(lines)
        widest = max(d.textlength(line, font=face) for line in lines)
        d.rounded_rectangle(((W - widest) / 2 - 30, top - 18, (W + widest) / 2 + 30, top + line_h * len(lines) + 10),
                            radius=14, fill=(12, 6, 30, 200))
        for i, line in enumerate(lines):
            d.text((W / 2, top + i * line_h), line, font=face, fill=INK, anchor="ma")
    if badge:
        face = font("segoeuib.ttf", 46)
        w = d.textlength(badge, font=face)
        x1, y0 = W - 60, 650           # right of the hint line, under the tiles
        d.rounded_rectangle((x1 - w - 56, y0, x1, y0 + 78), radius=14, fill=(36, 24, 76, 235),
                            outline=(80, 248, 152), width=4)
        d.text((x1 - w / 2 - 28, y0 + 39), badge, font=face, fill=INK, anchor="mm")
    return img


def cut():
    from PIL import Image
    import numpy as np

    lengths = load_lengths()
    comp = WORK / "comp"
    comp.mkdir(parents=True, exist_ok=True)
    for old in comp.glob("*.png"):
        old.unlink()
    entries = []                      # (file, frames)
    voice_at = {}                     # beat -> frame its segment starts on
    cursor = 0
    cache = {}

    def layer(tag, caption, badge):
        key = (tag, caption, badge)
        if key not in cache:
            cache[key] = overlay(tag, caption, badge)
        return cache[key]

    def add(src, frames, tag=None, caption=None, badge=None):
        nonlocal cursor
        base = Image.open(src).convert("RGBA")
        if base.size != (W, H):
            base = base.resize((W, H), Image.LANCZOS)
        out = comp / f"{len(entries):05d}.png"
        Image.alpha_composite(base, layer(tag, caption, badge)).convert("RGB").save(out)
        entries.append((out, frames))
        cursor += frames

    def card(name):
        for path in sorted((WORK / "cards" / name).glob("*.png")):
            add(path, 1)

    card("direct")
    flat = json.loads((WORK / "flat" / "timeline.json").read_text(encoding="utf-8"))
    tag = "THE REAL /DIRECT PAGE, RUNNING LOCALLY"
    for s in flat:
        voice_at.setdefault(s["beat"], cursor)
        badge_frames = min(s["frames"], round(BADGE_S * FPS)) if s["badge"] else 0
        if badge_frames:
            add(WORK / "flat" / s["file"], badge_frames, tag, CAPTION[s["beat"]], s["badge"])
        if s["frames"] - badge_frames:
            add(WORK / "flat" / s["file"], s["frames"] - badge_frames, tag, CAPTION[s["beat"]])
    card("in_vr")
    vr = json.loads((WORK / "vr" / "timeline.json").read_text(encoding="utf-8"))
    tag = "THE PAGE'S OWN VR VIEW, ONE EYE, SIMULATED HEADSET"
    for i, f in enumerate(vr):
        voice_at.setdefault(f["beat"], cursor)
        add(WORK / "vr" / f"{i:04d}.png", 1, tag, CAPTION[f["beat"]], f["badge"])
    voice_at["end"] = cursor
    card("end")
    end_frames = len(list((WORK / "cards" / "end").glob("*.png")))
    need = round(beat_seconds(lengths, "end") * FPS)
    if need > end_frames:             # hold the last end frame under the line
        last = entries[-1][0]
        entries.append((last, need - end_frames))
        cursor += need - end_frames
    total_s = cursor / FPS

    # One voice track: every beat's wav laid on its segment's start, after the lead.
    rate = 24000
    track = np.zeros(int(total_s * rate) + rate, dtype=np.float32)
    for beat, start in voice_at.items():
        with wave.open(str(WORK / "narration" / f"{beat}.wav")) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767
        at = int((start / FPS + LEAD_S) * rate)
        track[at:at + len(pcm)] += pcm[: max(0, len(track) - at)]
    voice = WORK / "voice.wav"
    with wave.open(str(voice), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(track[: int(total_s * rate)], -1, 1) * 32767).astype(np.int16).tobytes())

    # The concat list holds every composited image with its length in frames.
    listing = WORK / "frames.txt"
    lines = []
    for path, frames in entries:
        lines.append(f"file '{path.as_posix()}'\nduration {frames / FPS:.6f}\n")
    lines.append(f"file '{entries[-1][0].as_posix()}'\n")
    listing.write_text("".join(lines), encoding="utf-8")
    run = ["ffmpeg", "-v", "error", "-y"]
    subprocess.run([*run, "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(voice),
                    "-filter_complex", "[0:v]fps=30,format=yuv420p[v];"
                    "[1:a]loudnorm=I=-16:TP=-1.5,aresample=48000,aformat=channel_layouts=stereo[a]",
                    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{total_s:.3f}", "-movflags", "+faststart",
                    str(FINAL)], check=True)
    subprocess.run([*run, "-i", str(FINAL), "-vf", "scale=1280:720", "-c:v", "libx264", "-crf", "23",
                    "-preset", "slow", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(PHONE)],
                   check=True)
    print(json.dumps({"out": str(FINAL), "phone": str(PHONE), "seconds": round(total_s, 2),
                      "voice_starts": {k: round(v / FPS, 2) for k, v in voice_at.items()}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["narrate", "cards", "shoot", "cut", "all"])
    ap.add_argument("--url", default="http://127.0.0.1:8766/direct")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage in ("narrate", "all"):
        narrate()
    if args.stage in ("cards", "all"):
        cards()
    if args.stage in ("shoot", "all"):
        shoot(args.url)
    if args.stage in ("cut", "all"):
        cut()


if __name__ == "__main__":
    main()
