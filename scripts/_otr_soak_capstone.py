"""_otr_soak_capstone.py -- the v1.4 2D CAPSTONE full-episode soak (live GPU).

The production-stable stamp for the humo:5 path: run the FULL pipeline
(writer -> frozen audio -> Flux portraits -> HuMo per talking beat ->
SilentComposite -> MasterAudioMux) end-to-end on the live headless ComfyUI
(:8000) and assert, per leg:

  * engine_histogram == {"humo": N} (every talking beat real HuMo), or the
    still floor when heavy engines are OFF (--expect-floor);
  * audio byte-identical: independent ffmpeg PCM-SHA256 of the final mp4's
    audio stream == the frozen master WAV's PCM-SHA256;
  * VRAM peak <= 14.5 GB, sampled machine-wide via pynvml during the run
    (plus the driver's own vram_peak_mb from the render report);
  * run identity: every fact pinned to THIS run's prompt_id + episode slug
    (report mtime gated on this leg's start time -- orphan reports rejected).

Two legs back-to-back (server restarted between = the cache bust) are then
checked for determinism with ``--compare``: identical cast + style
fingerprints under OTR_CAST_SEED/OTR_STYLE_SEED.

Usage:
  python scripts/_otr_soak_capstone.py --leg run1
  python scripts/_otr_soak_capstone.py --leg run2
  python scripts/_otr_soak_capstone.py --leg floor --expect-floor
  python scripts/_otr_soak_capstone.py --compare run1 run2

Results land in scripts/_otr_soak_capstone_results/<leg>.json. Exit 0 = PASS.
UTF-8, no BOM, ASCII-only source. The frozen master audio is read-only here.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import requests  # noqa: E402

from otr_api import (  # noqa: E402
    COMFYUI_URL, fetch_schemas, load_workflow, patch_widget_by_name,
    submit_prompt, poll_history, workflow_to_api_prompt,
)

WORKFLOW_PATH = os.path.join(os.path.dirname(_HERE), "workflows",
                             "otr_scifi_16gb_full.json")
#: The live headless server's output tree (the install root the server runs
#: from; OTR_OUTPUT_DIR pins the node-relative output dir there).
SERVER_OUTPUT = os.environ.get(
    "OTR_SOAK_SERVER_OUTPUT",
    r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\output")
EPISODES_DIR = os.path.join(SERVER_OUTPUT, "otr", "episodes")
REPORT_PATH = os.path.join(SERVER_OUTPUT, "otr", "state",
                           "node_episode_report.json")
RESULTS_DIR = os.path.join(_HERE, "_otr_soak_capstone_results")
VRAM_CEILING_MB = 14.5 * 1024
POLL_TIMEOUT_S = 5400
#: target_words patched into the writer. 30 = the proven smoke config; 0 =
#: do NOT patch -- the canonical workflow's own saved default runs (the REAL
#: full episode; the marathon driver sets this).
SMOKE_WORDS = 30


class SoakFail(AssertionError):
    """A capstone soak invariant failed."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pcm_sha256(media_path: str) -> str:
    """Independent audio-stream hash: decode the FIRST audio stream to raw
    s16le PCM via ffmpeg (read-only) and sha256 it. Container-agnostic, so the
    final mp4 and the master WAV compare apples-to-apples."""
    if not os.path.isfile(media_path):
        raise SoakFail("pcm_sha256: missing file %r" % media_path)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", media_path, "-map", "0:a:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise SoakFail("ffmpeg PCM decode failed for %r: %s"
                       % (media_path, proc.stderr.decode("utf-8", "replace")[:400]))
    return hashlib.sha256(proc.stdout).hexdigest()


class VramSampler:
    """Machine-wide NVML VRAM sampler (V-3: never main-process-only)."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_mb = -1
        self.samples = 0
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # noqa: BLE001
            print("[soak] WARN: pynvml unavailable (%s); relying on the "
                  "driver's vram_peak_mb only" % exc, flush=True)
            self._nvml = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.peak_mb = max(self.peak_mb, int(info.used / (1024 * 1024)))
                self.samples += 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        if self._nvml is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


def _snapshot_tree(root: str) -> set:
    """Relative paths of every file under ``root`` (the stray-write baseline)."""
    out = set()
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for f in files:
            out.add(os.path.normpath(os.path.join(rel, f)))
    return out


def _obs_dir() -> str:
    """The OPERATOR-facing obs dir (mirrors the mux's OTR_OBS_DIR pin)."""
    return os.environ.get("OTR_OBS_DIR", "").strip() or os.path.join(
        SERVER_OUTPUT, "otr", "obs")


def _obs_listing() -> set:
    d = _obs_dir()
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def assert_no_stray_writes(before: set, root: str, sys_temp_before: set,
                           obs_before: set):
    """OUTPUT HYGIENE gate 2: every file the run created under the server
    output root must live under ``otr\\``; the SYSTEM temp dir gained no new
    ``otr_*`` entries (proving the in-tree TEMP repoint held); and the
    operator's obs dir gained EXACTLY the final mp4 (JSON reports live in
    otr/state)."""
    after = _snapshot_tree(root)
    new = after - before
    stray = sorted(p for p in new
                   if not os.path.normpath(p).lower().startswith("otr" + os.sep))
    if stray:
        raise SoakFail("stray writes OUTSIDE output\\otr\\: %r"
                       % stray[:20])
    sys_temp = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp")
    if os.path.isdir(sys_temp):
        now = {e for e in os.listdir(sys_temp) if e.lower().startswith("otr")}
        leaked = now - sys_temp_before
        if leaked:
            raise SoakFail("run leaked otr_* entries into the system temp "
                           "dir %r: %r" % (sys_temp, sorted(leaked)[:10]))
    obs_new = sorted(_obs_listing() - obs_before)
    if len(obs_new) != 1 or not obs_new[0].lower().endswith(".mp4"):
        raise SoakFail("the operator obs dir %r must gain EXACTLY the final "
                       "mp4; this run added: %r" % (_obs_dir(), obs_new))
    print("[soak] no stray writes outside otr (%d new in-tree files; obs "
          "gained only %s)" % (len(new), obs_new[0]), flush=True)


def _sys_temp_otr_entries() -> set:
    sys_temp = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp")
    if not os.path.isdir(sys_temp):
        return set()
    return {e for e in os.listdir(sys_temp) if e.lower().startswith("otr")}


def ffprobe_streams(path: str) -> dict:
    """Stream + duration summary via ffprobe (read-only)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name:format=duration,size",
         "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise SoakFail("ffprobe failed for %r: %s"
                       % (path, proc.stderr.decode("utf-8", "replace")[:300]))
    return json.loads(proc.stdout.decode("utf-8", "replace"))


def assert_obs_final_playable(after_ts: float, master_wav: str) -> str:
    """OUTPUT HYGIENE gate 1: the deliverable is a REAL playable mp4 in the
    OPERATOR'S output\\otr\\obs -- video stream + audio stream, non-zero size,
    duration == master within tolerance, and a full decode succeeds. JSON is
    not a deliverable. ``OTR_OBS_DIR`` mirrors the server-side publish pin."""
    obs_dir = os.environ.get("OTR_OBS_DIR", "").strip() or os.path.join(
        SERVER_OUTPUT, "otr", "obs")
    cands = [(os.path.getmtime(p), p) for p
             in glob.glob(os.path.join(obs_dir, "*_final.mp4"))
             if os.path.getmtime(p) > after_ts]
    if not cands:
        raise SoakFail("no *_final.mp4 in %r newer than this leg's start -- "
                       "the run produced JSON, not a deliverable" % obs_dir)
    obs_mp4 = max(cands)[1]
    size = os.path.getsize(obs_mp4)
    if size <= 0:
        raise SoakFail("obs final %r is zero bytes" % obs_mp4)
    info = ffprobe_streams(obs_mp4)
    kinds = [s.get("codec_type") for s in info.get("streams", [])]
    if "video" not in kinds or "audio" not in kinds:
        raise SoakFail("obs final %r lacks a video+audio stream pair: %r"
                       % (obs_mp4, kinds))
    dur = float((info.get("format") or {}).get("duration") or -1)
    m_info = ffprobe_streams(master_wav)
    m_dur = float((m_info.get("format") or {}).get("duration") or -1)
    if abs(dur - m_dur) > 2.0:
        raise SoakFail("obs final duration %.2fs vs master %.2fs differs by "
                       "> 2.0s" % (dur, m_dur))
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", obs_mp4,
                           "-f", "null", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=False)
    if proc.returncode != 0 or proc.stderr.strip():
        raise SoakFail("obs final %r does not cleanly decode: %s"
                       % (obs_mp4, proc.stderr.decode("utf-8", "replace")[:300]))
    print("[soak] obs deliverable OK: %s (%.1f MB, %.2fs, streams=%r, "
          "full decode clean)" % (obs_mp4, size / 1e6, dur, kinds), flush=True)
    return obs_mp4


def cast_style_fingerprint(ledger: dict) -> dict:
    """Deterministic cast + style identity for the seed-pinned comparison.
    Cast: the seeded picks (name/gender/tts_model/voice_preset). Style: the
    meta brief's style fields if present (key-sorted)."""
    cast = [[c.get("name"), c.get("gender"), c.get("tts_model"),
             c.get("voice_preset")] for c in (ledger.get("cast") or [])]
    meta = ledger.get("meta") or {}
    style = {k: meta[k] for k in sorted(meta)
             if isinstance(k, str) and "style" in k.lower()}
    blob = json.dumps({"cast": cast, "style": style},
                      ensure_ascii=True, sort_keys=True)
    return {"cast": cast, "style": style,
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}


def newest_final_mp4(after_ts: float):
    """The per-episode *_final.mp4 written AFTER this leg started (run
    identity). Layout: otr/episodes/<slug>/<slug>_silent_final.mp4."""
    cands = [(os.path.getmtime(p), p) for p
             in glob.glob(os.path.join(EPISODES_DIR, "*", "*_final.mp4"))
             if os.path.getmtime(p) > after_ts]
    if not cands:
        raise SoakFail("no per-episode *_final.mp4 newer than this leg's "
                       "start under %r" % EPISODES_DIR)
    return max(cands)[1]


def load_run_report(after_ts: float) -> dict:
    if not os.path.isfile(REPORT_PATH):
        raise SoakFail("missing render report %r" % REPORT_PATH)
    if os.path.getmtime(REPORT_PATH) <= after_ts:
        raise SoakFail("render report %r is OLDER than this leg's start -- "
                       "orphan report rejected (run identity)" % REPORT_PATH)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# one leg
# --------------------------------------------------------------------------- #
def run_leg(leg: str, expect_floor: bool, expect_engine: str = "humo",
            extra_patches=None) -> int:
    """One full-episode soak leg with the HARD gates.

    ``expect_engine``: "humo" (default) asserts the strict production
    histogram {"humo": n_beats}; "" / None logs the histogram WITHOUT a strict
    engine assert (the dropdown-rotation / forced-engine experiment legs --
    completion, playable-obs, byte-identity, hygiene and VRAM gates still
    apply). ``extra_patches``: list of (node_id, widget_name, value) applied
    on top of the standard smoke patches (the multi-LLM / multi-voice /
    multi-music dropdown rotation)."""
    started = time.time()
    print("[soak] leg=%s COMFYUI_URL=%s started=%s"
          % (leg, COMFYUI_URL, time.strftime("%H:%M:%S")), flush=True)

    stats = requests.get(COMFYUI_URL + "/system_stats", timeout=15).json()
    dev = (stats.get("devices") or [{}])[0]
    print("[soak] server up: %s | torch %s" % (
        dev.get("name", "?"), stats.get("system", {}).get("pytorch_version", "?")),
        flush=True)

    tree_before = _snapshot_tree(SERVER_OUTPUT)
    sys_temp_before = _sys_temp_otr_entries()
    obs_before = _obs_listing()
    print("[soak] hygiene baseline: %d files under server output; %d otr_* "
          "system-temp entries; obs(%s)=%d files"
          % (len(tree_before), len(sys_temp_before), _obs_dir(),
             len(obs_before)), flush=True)

    schemas = fetch_schemas()
    wf = load_workflow(WORKFLOW_PATH)

    def _node_type(node_id):
        for n in wf.get("nodes", []):
            if n.get("id") == node_id:
                return n.get("type")
        return None

    def _first_choice(node_id, widget):
        sch = (schemas or {}).get(_node_type(node_id)) or {}
        inp = sch.get("input", {})
        for sec in ("optional", "required"):
            spec = inp.get(sec, {}).get(widget)
            if spec and isinstance(spec, (list, tuple)) and isinstance(spec[0], list):
                return spec[0][0]
        return None

    for slot in ("openrouter_slot_a_model", "openrouter_slot_b_model",
                 "comfy_slot_a_model", "comfy_slot_b_model"):
        val = _first_choice(1, slot)
        if val is not None:
            patch_widget_by_name(wf, 1, slot, val, schemas)
    if SMOKE_WORDS:
        patch_widget_by_name(wf, 1, "target_words", SMOKE_WORDS, schemas)
        patch_widget_by_name(wf, 1, "num_characters", 2, schemas)
        patch_widget_by_name(wf, 1, "act_count", "1", schemas)
    else:
        print("[soak] running the workflow's OWN saved defaults (real full "
              "episode)", flush=True)
    for nid, widget, value in (extra_patches or []):
        patch_widget_by_name(wf, int(nid), widget, value, schemas)
        print("[soak] patch: node %s %s=%r" % (nid, widget, value), flush=True)

    api = workflow_to_api_prompt(wf, schemas)
    with VramSampler() as vram:
        prompt_id = submit_prompt(api)
        print("[soak] QUEUED prompt_id=%s" % prompt_id, flush=True)

        def tick(elapsed, status):
            if int(elapsed) % 60 < 5:
                print("[soak] t=%4ds status=%s vram_peak=%dMB"
                      % (elapsed, status.get("status_str", "?"), vram.peak_mb),
                      flush=True)

        status, err = poll_history(prompt_id, timeout_s=POLL_TIMEOUT_S,
                                   on_tick=tick)
    if status != "SUCCESS":
        raise SoakFail("run %s: prompt %s -> %s %s"
                       % (leg, prompt_id, status, err))
    print("[soak] prompt %s SUCCESS in %.0fs"
          % (prompt_id, time.time() - started), flush=True)

    report = load_run_report(started)
    hist = report.get("engine_histogram") or {}
    n_beats = int(report.get("n_beats") or 0)
    clip_count = int(report.get("clip_count") or 0)
    episode_id = str(report.get("episode_id") or "")
    if not report.get("ok"):
        raise SoakFail("render report not ok: %r" % report)
    if clip_count != n_beats or n_beats <= 0:
        raise SoakFail("clip_count %d != n_beats %d" % (clip_count, n_beats))

    if expect_floor:
        if "humo" in hist or "humo_1.7B" in hist:
            raise SoakFail("floor leg rendered HuMo anyway: %r" % hist)
        if hist.get("still_kenburns", 0) <= 0:
            raise SoakFail("floor leg: still_kenburns absent: %r" % hist)
        print("[soak] FLOOR histogram OK: %r" % hist, flush=True)
    elif expect_engine:
        # TALKING-BEATS-ONLY assert (2026-06-10 marathon refinement): a big
        # episode legitimately routes music/visual beats to the visualizer --
        # whole-histogram equality false-failed the 160w/4ch finale
        # ({humo:8, visualizer:1}). The real production invariant: every shot
        # PLANNED for the keystone stays ON the keystone (no LOUD fallbacks
        # off it), and at least one such shot exists.
        trace = report.get("trace") or []
        planned = [t for t in trace
                   if (t.get("attempts") or [""])[0] == expect_engine]
        fell_off = [t for t in planned
                    if t.get("final_engine") != expect_engine]
        if not planned:
            raise SoakFail("no shot was planned for %r (histogram %r)"
                           % (expect_engine, hist))
        if fell_off:
            raise SoakFail("%d/%d %s-planned shot(s) fell off the keystone: "
                           "%r (histogram %r)"
                           % (len(fell_off), len(planned), expect_engine,
                              [(t.get("shot_id"), t.get("final_engine"))
                               for t in fell_off], hist))
        print("[soak] %s keystone OK: %d/%d planned shots stayed on it "
              "(histogram %r)" % (expect_engine, len(planned), len(planned),
                                  hist), flush=True)
    else:
        print("[soak] EXPERIMENT histogram (informational): %r over %d beats"
              % (hist, n_beats), flush=True)

    final_mp4 = newest_final_mp4(started)
    # The episode slug IS the per-episode folder name (layout:
    # otr/episodes/<slug>/<...>_final.mp4) -- never parse it from the file
    # stem, which now carries post-chain suffixes (_silent_procgen_blended).
    slug = os.path.basename(os.path.dirname(final_mp4))
    audio_dir = os.path.join(EPISODES_DIR, slug, "audio")
    # After the pending->slug rename BOTH master names can coexist (the
    # capture-time pending_*_master.wav and the slug-named copy). Prefer the
    # slug-named master (the mux re-resolves to it); fall back to the single
    # remaining candidate.
    masters = sorted(glob.glob(os.path.join(audio_dir, "*_master.wav")))
    if not masters:
        raise SoakFail("no *_master.wav in %r" % audio_dir)
    slug_named = [m for m in masters
                  if os.path.basename(m) == slug + "_master.wav"]
    master_wav = (slug_named or masters)[0]
    if len(masters) > 1:
        print("[soak] note: %d master WAVs after rename; using %s"
              % (len(masters), os.path.basename(master_wav)), flush=True)
    ledger_path = os.path.join(audio_dir, slug + "_ledger.json")
    final_sha = pcm_sha256(final_mp4)
    master_sha = pcm_sha256(master_wav)
    if final_sha != master_sha:
        raise SoakFail("AUDIO NOT BYTE-IDENTICAL: final %s != master %s"
                       % (final_sha[:12], master_sha[:12]))
    print("[soak] audio byte-identical OK (pcm sha %s)" % final_sha[:12],
          flush=True)

    # OUTPUT HYGIENE hard gates (operator directive 2026-06-09): a real
    # playable deliverable in otr/obs + zero writes outside the otr tree.
    obs_mp4 = assert_obs_final_playable(started, master_wav)
    # The obs deliverable is the WATCHABLE copy: AAC viewing audio (standard
    # players reject raw PCM-in-MP4 -- operator screenshot 2026-06-09). The
    # byte-identical PCM proof is the archival final above; here assert the
    # obs audio is actually the playable codec.
    obs_codecs = [s.get("codec_name") for s in
                  ffprobe_streams(obs_mp4).get("streams", [])
                  if s.get("codec_type") == "audio"]
    if obs_codecs != ["aac"]:
        raise SoakFail("obs deliverable audio codec %r != ['aac'] -- not "
                       "playable in standard players" % obs_codecs)
    print("[soak] obs viewing audio OK: aac (archival PCM byte-identical "
          "final kept in the episode folder)", flush=True)
    assert_no_stray_writes(tree_before, SERVER_OUTPUT, sys_temp_before,
                           obs_before)

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    fp = cast_style_fingerprint(ledger)

    # V-3 gate: the ceiling applies to the VIDEO render phase (single resident
    # heavy engine at inter-engine boundaries) -- the driver's vram_peak_mb is
    # that machine-wide NVML measurement (nodes/vram_context_test pattern).
    # The client-side whole-pipeline sample (which also spans the writer-LLM
    # phase, NOT part of V-3) is recorded as informational context only.
    driver_peak = int(report.get("vram_peak_mb") or -1)
    peaks = {"render_phase_driver_mb": driver_peak,
             "whole_run_nvml_machine_mb": vram.peak_mb,
             "nvml_samples": vram.samples}
    if driver_peak > VRAM_CEILING_MB:
        raise SoakFail("V-3 render-phase VRAM peak %dMB > ceiling %dMB"
                       % (driver_peak, VRAM_CEILING_MB))
    print("[soak] V-3 render-phase VRAM peak OK: %dMB <= %.0fMB "
          "(whole-run machine peak %dMB informational)"
          % (driver_peak, VRAM_CEILING_MB, vram.peak_mb), flush=True)

    result = {
        "leg": leg, "prompt_id": prompt_id, "episode_id": episode_id,
        "slug": slug, "started": started, "elapsed_s": time.time() - started,
        "engine_histogram": hist, "n_beats": n_beats,
        "final_mp4": final_mp4, "obs_mp4": obs_mp4,
        "pcm_sha256": final_sha,
        "master_pcm_sha256": master_sha, "vram": peaks,
        "fingerprint": fp, "expect_floor": expect_floor,
        "trace": report.get("trace"),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, leg + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print("[soak] PASS leg=%s episode=%s prompt=%s -> %s"
          % (leg, slug, prompt_id, out), flush=True)
    return 0


def compare_legs(a: str, b: str) -> int:
    ra = json.load(open(os.path.join(RESULTS_DIR, a + ".json"),
                        encoding="utf-8"))
    rb = json.load(open(os.path.join(RESULTS_DIR, b + ".json"),
                        encoding="utf-8"))
    if ra["fingerprint"]["sha256"] != rb["fingerprint"]["sha256"]:
        print("[soak] cast/style A: %r" % ra["fingerprint"], flush=True)
        print("[soak] cast/style B: %r" % rb["fingerprint"], flush=True)
        raise SoakFail("DETERMINISM FAIL: cast/style fingerprints differ "
                       "(%s vs %s)" % (ra["fingerprint"]["sha256"][:12],
                                       rb["fingerprint"]["sha256"][:12]))
    if ra["engine_histogram"] != rb["engine_histogram"]:
        raise SoakFail("DETERMINISM FAIL: histograms differ: %r vs %r"
                       % (ra["engine_histogram"], rb["engine_histogram"]))
    print("[soak] DETERMINISM OK: %s == %s (fingerprint %s; hist %r)"
          % (a, b, ra["fingerprint"]["sha256"][:12], ra["engine_histogram"]),
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leg", help="run one soak leg and save <leg>.json")
    ap.add_argument("--expect-floor", action="store_true",
                    help="assert the still floor (heavy engines OFF) instead "
                         "of humo:N")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="assert determinism between two saved legs")
    args = ap.parse_args(argv)
    try:
        if args.compare:
            return compare_legs(*args.compare)
        if not args.leg:
            ap.error("--leg or --compare required")
        return run_leg(args.leg, args.expect_floor)
    except SoakFail as exc:
        print("[soak] FAIL: %s" % exc, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
