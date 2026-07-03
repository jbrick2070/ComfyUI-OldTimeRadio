"""OTR_CreditsRoll -- the ONE late viewer-credits surface (credits enrichment
2026-07-03, GO_FORWARD_CREDITS.md v4).

SCAFFOLD STATUS: this file is UNWIRED until the S3+S1 atomic slice lands
(93 -> OTR_CreditsRoll -> 85 in workflows/otr_scifi_16gb_full.json, link 263
preserved). Unwired code is inert; the spec tests in
tests/test_credits_roll_spec.py are this node's contract.

Why this node exists: node 12 (OTR_SignalLostVideo) renders BEFORE the image
dispatcher (91) and the video render batch (92), so at node-12 time the
engine/image/delivered-voice receipts do not exist -- today's blank voices and
"(not recorded)". This node renders the COMPLETE viewer roll LATE, from
post-render truth: the DURABLE production-ledger singleton (S2 stamps) + the
clip manifest (node 92 slot 1).

Hard contracts (operator directives):
- NO FALLBACKS. Missing manifest / ledger receipt / render / concat failure
  RAISES before mux. No source-copy, no quiet "(not recorded)" placeholder.
  (Node 93 was rejected as a host precisely because of its source-copy-on-
  ffmpeg-failure fallback.)
- SILENT-TAIL audio model: the master audio stays body-length; the credits
  clip is appended as SILENT video; the video ENDS at credit-end. This node
  DECLARES its tail duration to the mux (node 85) so the 45s guard becomes
  credits-AWARE (guard permits v_dur <= a_dur + declared_tail + tol) -- the
  guard is never blind-widened.
- LOOK CONTRACT (operator 2026-06-17, BUG-410): the backdrop under the roll
  is the LAST drama clip LOOPED -- never credits-over-black.
- COLOR DECISION (stated per plan S3): captions/bars burn at node 93 BEFORE
  the append, so this clip escapes the green-only channel crush and MAY be
  full-color. We KEEP the green CRT credits look BY CHOICE -- aesthetic
  continuity with the episode body -- not by pipeline force.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

log = logging.getLogger("OldTimeRadio")

# --------------------------------------------------------------------------- #
# Roll geometry / look constants. S0 owns the +50% type pass: it sets
# _CREDITS_FONT_SCALE to 1.5 and rescales _CREDITS_SCROLL_PPS so the taller
# roll stays inside the (credits-aware) duration budget.
# --------------------------------------------------------------------------- #
_CREDITS_FONT_SCALE = 1.0
_CREDITS_SCROLL_PPS = 65.0          # px/s the roll climbs (S0 rescales)
_CREDITS_MARGIN_X = 72
_CREDITS_HEADER_PT = 34             # pre-scale; multiplied by the scale
_CREDITS_BODY_PT = 24
_CREDITS_LINE_GAP = 10
_CREDITS_SECTION_GAP = 46
# Green-CRT-by-choice palette (full RGB is available post-93; see COLOR
# DECISION above). Hierarchy by green intensity survives even if a future
# rewire pushes this through the green-only blend.
_COL_HEADER = (255, 255, 255)
_COL_BODY = (110, 255, 140)
_COL_DIM = (70, 200, 105)


class CreditsDataError(RuntimeError):
    """A required credits receipt is missing from the durable ledger / the
    clip manifest, or the roll could not be rendered/appended. LOUD by
    design -- the no-fallback contract forbids a quiet placeholder."""


# --------------------------------------------------------------------------- #
# Declared source paths (codex OPTIONAL #2: date / GPU / git SHA must each
# have a declared source -- no ad-hoc reads sprinkled through the renderer).
# --------------------------------------------------------------------------- #
def _git_short_sha() -> str:
    """Repo-file read (.git/HEAD -> ref), no subprocess. Raises
    CreditsDataError when unresolvable -- the SEED/COMMIT line is a receipt."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head_path = os.path.join(repo, ".git", "HEAD")
    try:
        with open(head_path, "r", encoding="utf-8") as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1].strip()
            ref_path = os.path.join(repo, ".git", *ref.split("/"))
            if os.path.exists(ref_path):
                with open(ref_path, "r", encoding="utf-8") as f:
                    return f.read().strip()[:8]
            packed = os.path.join(repo, ".git", "packed-refs")
            with open(packed, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(ref) and not line.startswith("#"):
                        return line.split()[0][:8]
            raise FileNotFoundError(ref)
        return head[:8]
    except Exception as exc:  # noqa: BLE001
        raise CreditsDataError(
            f"git short SHA unresolvable from {head_path!r}: {exc}") from exc


def _gpu_name() -> str:
    """Declared source: the shared system-spec probe (same source the
    treatment uses)."""
    from ._otr_sys_specs import collect_system_specs
    return str((collect_system_specs() or {}).get("gpu") or "").strip() \
        or "(unknown GPU)"


# --------------------------------------------------------------------------- #
# Receipts -> roll sections (the relocated dossier spec lives here now)
# --------------------------------------------------------------------------- #
def _require(container, key, source):
    v = (container or {}).get(key)
    if v is None or v == "" or v == {} or v == []:
        raise CreditsDataError(
            f"required credits receipt missing: {source}.{key} -- the S2 "
            f"durable stamp for it never landed (no-fallback: refusing to "
            f"print a placeholder)")
    return v


def _engine_hist_lines(by_role: dict) -> list:
    lines = []
    for role in sorted(by_role or {}):
        engs = by_role[role] or {}
        lines.append("%-18s %s" % (role, ", ".join(
            "%s x%d" % (e, n) for e, n in sorted(engs.items()))))
    return lines


def build_credits_sections(led: dict) -> list:
    """The viewer roll data, from the DURABLE ledger (S2 stamps). Returns
    ``[{"header": str, "lines": [str, ...]}, ...]`` in roll order.

    NO-FALLBACK: every receipt-bearing block RAISES CreditsDataError when its
    source is missing. meta.image_engines is the ONLY image source (the S2
    durable stamp) -- the legacy ledger['images'] row-scan fallback from the
    node-12 dossier is deliberately NOT reproduced (cleanbreak). Story facts
    (news) render only when present; they are frozen-ledger facts, not
    receipts."""
    if not isinstance(led, dict) or not led:
        raise CreditsDataError("durable ledger missing/empty at credits time")
    meta = led.get("meta")
    if not isinstance(meta, dict) or not meta:
        raise CreditsDataError("durable ledger has no meta at credits time")

    sections: list = []

    # WRITTEN BY
    gp = meta.get("gen_params_initial") or {}
    creative = gp.get("creative_writing_model") or meta.get(
        "creative_writing_model")
    technical = gp.get("technical_model") or meta.get("technical_model")
    if not creative or not technical:
        raise CreditsDataError(
            "required credits receipt missing: writer models "
            "(meta.gen_params_initial.creative_writing_model / "
            "technical_model)")
    sections.append({"header": "WRITTEN BY", "lines": [
        f"{creative}", f"(technical: {technical})"]})

    # CAST & VOICES -- from CastLock's DELIVERED stamp (voice_ref_id /
    # voice_engine per entry; bark-preset engines stamp voice_preset). The
    # PLANNED meta.voice_cast_decision is NOT credited from (CastLock can
    # fall closed past it).
    cast = _require(led, "cast", "ledger")
    voice_slots = (meta.get("cast_voice_slots")
                   if isinstance(meta.get("cast_voice_slots"), dict) else {})
    cl = []
    for entry in cast:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("char_id") or "").strip()
        ref = entry.get("voice_ref_id") or entry.get("voice_preset")
        eng = entry.get("voice_engine") or (
            "bark" if entry.get("voice_preset") else "")
        if not ref or not eng:
            raise CreditsDataError(
                f"cast entry {name or '?'} has no delivered voice stamp "
                f"(voice_ref_id/voice_preset + voice_engine) -- CastLock's "
                f"S2 durable stamp is the required source")
        line = f"{name} .... {eng} · {ref}"
        sig = str(((voice_slots.get(str(entry.get("char_id") or "")) or {})
                   ).get("speech_signature") or "").strip()
        if sig:
            line += f'  "{sig}"'
        cl.append(line)
    if not cl:
        raise CreditsDataError("cast is empty at credits time")
    sections.append({"header": "CAST & VOICES", "lines": cl})

    # IMAGES -- meta.image_engines (S2 durable stamp) is the ONLY source.
    img = _require(meta, "image_engines", "meta")
    sections.append({"header": "IMAGES",
                     "lines": _engine_hist_lines(img.get("by_role") or {})
                     or ["(no stills dispatched this episode)"]})

    # MOTION -- meta.render_engines (singleton-native stamp, now LOUD).
    ren = _require(meta, "render_engines", "meta")
    sections.append({"header": "MOTION",
                     "lines": _engine_hist_lines(ren.get("by_role") or {})
                     or _engine_hist_lines(
                         {"(all)": ren.get("histogram") or {}})})

    # MUSIC -- meta.music_engine (S2; the node-83 done wire is unlinked).
    music = _require(meta, "music_engine", "meta")
    sections.append({"header": "MUSIC",
                     "lines": [f"{music} · closing cue looped"]})

    # NEWS SEED -- frozen-ledger story fact (optional, not a receipt).
    news = meta.get("news") if isinstance(meta.get("news"), dict) else {}
    brief = str(news.get("script_brief") or "").strip()
    if brief:
        sections.append({"header": "NEWS SEED", "lines": [brief]})

    # SEED / COMMIT -- declared sources only.
    seed = (meta.get("cast_contract") or {}).get("cast_seed",
                                                 meta.get("episode_seed"))
    if seed is None:
        raise CreditsDataError(
            "required credits receipt missing: cast seed "
            "(meta.cast_contract.cast_seed / meta.episode_seed)")
    sections.append({"header": "SEED / COMMIT",
                     "lines": [f"cast seed {seed} · {_git_short_sha()}"]})

    return sections


# --------------------------------------------------------------------------- #
# Duration (the declared credits tail) + backdrop plan
# --------------------------------------------------------------------------- #
def compute_roll_duration_s(roll_px: int, viewport_h: int,
                            scroll_pps: float = _CREDITS_SCROLL_PPS) -> float:
    """Content-driven roll duration: the tall canvas enters from the bottom
    and fully exits the top. THIS number is what the node declares to the
    mux guard (credits-aware: v_dur <= a_dur + declared_tail + tol)."""
    if scroll_pps <= 0:
        raise CreditsDataError(f"scroll_pps must be > 0 (got {scroll_pps})")
    return (float(roll_px) + float(viewport_h)) / float(scroll_pps)


def plan_backdrop(clip_manifest: dict) -> dict:
    """LOOK CONTRACT (operator 2026-06-17 / BUG-410, relocated from node 84's
    tail-fill planner): the roll rides over the LAST existing drama clip,
    LOOPED -- never over black. Returns the manifest row. RAISES when the
    manifest has no usable clip (credits-over-black would bounce at eyeball;
    no-fallback means we stop the render instead)."""
    rows = [r for r in ((clip_manifest or {}).get("clips") or [])
            if isinstance(r, dict) and r.get("exists") and r.get("path")]
    if not rows:
        raise CreditsDataError(
            "clip manifest has no existing clip to loop under the credits "
            "roll (look contract: credits-over-black is a bounce)")
    if any(r.get("start_s") is not None for r in rows):
        return max(rows, key=lambda r: float(r.get("start_s") or 0.0))
    return rows[-1]


# --------------------------------------------------------------------------- #
# Rendering + append (ffmpeg; every failure RAISES)
# --------------------------------------------------------------------------- #
def _ffmpeg_bin() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise CreditsDataError("ffmpeg not on PATH -- cannot render credits")
    return p


def _ffprobe_bin() -> str:
    p = shutil.which("ffprobe")
    if not p:
        raise CreditsDataError("ffprobe not on PATH -- cannot render credits")
    return p


def _probe_video(path: str) -> dict:
    out = subprocess.run(
        [_ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise CreditsDataError(f"ffprobe failed on {path!r}: {out.stderr}")
    st = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate") or "25/1").partition("/")
    fps = float(num) / float(den or 1)
    return {"w": int(st.get("width") or 0), "h": int(st.get("height") or 0),
            "fps": fps}


def _load_font(pt: int):
    from PIL import ImageFont
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, pt)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def render_roll_canvas(sections: list, width: int,
                       font_scale: float = _CREDITS_FONT_SCALE):
    """Pre-render the whole roll as ONE tall RGBA canvas (transparent bg;
    ffmpeg overlays + scrolls it). Returns the PIL image. Pure -- no I/O."""
    from PIL import Image, ImageDraw

    h_pt = max(8, int(round(_CREDITS_HEADER_PT * font_scale)))
    b_pt = max(8, int(round(_CREDITS_BODY_PT * font_scale)))
    f_head = _load_font(h_pt)
    f_body = _load_font(b_pt)
    gap = int(round(_CREDITS_LINE_GAP * font_scale))
    sgap = int(round(_CREDITS_SECTION_GAP * font_scale))

    # measure
    total = sgap
    for sec in sections:
        total += h_pt + gap
        total += len(sec["lines"]) * (b_pt + gap)
        total += sgap
    total += sgap

    img = Image.new("RGBA", (width, max(total, 8)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = sgap
    for sec in sections:
        d.text((_CREDITS_MARGIN_X, y), sec["header"],
               fill=_COL_HEADER + (255,), font=f_head)
        y += h_pt + gap
        for line in sec["lines"]:
            d.text((_CREDITS_MARGIN_X + 24, y), line,
                   fill=_COL_BODY + (255,), font=f_body)
            y += b_pt + gap
        y += sgap
    return img


def render_credits_clip(sections: list, backdrop_path: str, out_path: str,
                        *, w: int, h: int, fps: float,
                        scroll_pps: float = _CREDITS_SCROLL_PPS,
                        font_scale: float = _CREDITS_FONT_SCALE) -> float:
    """Render the SILENT credits clip: the tall roll canvas scrolling over the
    LOOPED backdrop clip (look contract), encoded to match the body (h264
    yuv420p, same w/h/fps, no audio stream). Returns the clip duration in
    seconds (== the declared credits tail). RAISES on any failure."""
    canvas = render_roll_canvas(sections, w, font_scale)
    dur = compute_roll_duration_s(canvas.height, h, scroll_pps)

    png = out_path + ".roll.png"
    canvas.save(png)
    try:
        # backdrop loops for the whole roll; darken slightly for legibility;
        # the roll overlays climbing at scroll_pps.
        cmd = [
            _ffmpeg_bin(), "-y",
            "-stream_loop", "-1", "-i", backdrop_path,
            "-i", png,
            "-filter_complex",
            (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
             f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
             f"eq=brightness=-0.25,fps={fps}[bg];"
             f"[bg][1:v]overlay=x=0:y='{h}-t*{scroll_pps}':shortest=0[v]"),
            "-map", "[v]", "-an",
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", "19", out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_path) \
                or os.path.getsize(out_path) == 0:
            raise CreditsDataError(
                f"credits clip render failed (rc={r.returncode}): "
                f"{(r.stderr or '')[-800:]}")
    finally:
        try:
            os.remove(png)
        except OSError:
            pass
    return dur


def append_credits(body_path: str, credits_path: str, out_path: str) -> str:
    """Concat body + credits (both silent video, matching params) via the
    concat demuxer, stream-copy. NO source-copy fallback: any failure RAISES
    (this is exactly the node-93 fallback we refused to inherit)."""
    if not os.path.exists(body_path):
        raise CreditsDataError(f"body video missing: {body_path!r}")
    if not os.path.exists(credits_path):
        raise CreditsDataError(f"credits clip missing: {credits_path!r}")
    lst = out_path + ".concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in (body_path, credits_path):
            f.write("file '%s'\n" % p.replace("\\", "/").replace("'", r"'\''"))
    try:
        r = subprocess.run(
            [_ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", lst,
             "-c", "copy", out_path],
            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_path) \
                or os.path.getsize(out_path) == 0:
            raise CreditsDataError(
                f"credits append (concat) failed (rc={r.returncode}): "
                f"{(r.stderr or '')[-800:]}")
    finally:
        try:
            os.remove(lst)
        except OSError:
            pass
    return out_path


# --------------------------------------------------------------------------- #
# The terminal node (wired at S3: 93 -> OTR_CreditsRoll -> 85)
# --------------------------------------------------------------------------- #
class OTRCreditsRoll:
    """Registered as ``OTR_CreditsRoll`` (registration lands with the S3
    wiring slice). Appends the late viewer roll to node 93's output and hands
    the mux (85) the video path + the DECLARED credits-tail duration."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "roll"
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("video_with_credits_path", "declared_credits_tail_s",
                    "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {
                "forceInput": True,
                "tooltip": "Node 93 output (captions/bars already burned)."}),
            "clip_manifest_json": ("STRING", {
                "forceInput": True,
                "tooltip": "Node 92 slot 1 (third fan-out consumer)."}),
        }}

    def roll(self, video_path, clip_manifest_json):
        from .production_ledger import get_ledger

        if not video_path or not os.path.exists(video_path):
            raise CreditsDataError(
                f"credits input video missing: {video_path!r}")
        try:
            manifest = json.loads(clip_manifest_json or "")
        except (TypeError, ValueError) as exc:
            raise CreditsDataError(
                f"clip_manifest_json unparseable: {exc}") from exc

        led = get_ledger()
        sections = build_credits_sections(led.data)
        backdrop = plan_backdrop(manifest)

        vp = _probe_video(video_path)
        base, ext = os.path.splitext(video_path)
        credits_clip = base + "_credits" + (ext or ".mp4")
        out = base + "_with_credits" + (ext or ".mp4")

        tail_s = render_credits_clip(
            sections, str(backdrop["path"]), credits_clip,
            w=vp["w"], h=vp["h"], fps=vp["fps"])
        append_credits(video_path, credits_clip, out)

        report = json.dumps({
            "ok": True, "declared_credits_tail_s": round(tail_s, 3),
            "sections": [s["header"] for s in sections],
            "backdrop_shot": backdrop.get("shot_id"),
            "output": out}, ensure_ascii=True)
        log.info("[OTR_CreditsRoll] appended %.1fs roll (backdrop=%s) -> %s",
                 tail_s, backdrop.get("shot_id"), out)
        return (out, float(tail_s), report)


__all__ = [
    "OTRCreditsRoll", "CreditsDataError", "build_credits_sections",
    "compute_roll_duration_s", "plan_backdrop", "render_roll_canvas",
    "render_credits_clip", "append_credits",
]
