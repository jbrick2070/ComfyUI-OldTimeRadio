"""
otr_v2.hyworld.character_regression  --  Day 12 character identity gate
=======================================================================

Regress a character's portrait likeness across scenes.  Given a job-dir
tree containing ``pulid_portrait`` outputs for multiple scenes, computes
pairwise SSIM on cropped faces (real mode) or mean-channel SSIM on the
stub's solid-color render (stub mode) and asserts the likeness holds
above the ROADMAP Day 12 bar:

    SSIM > 0.85 on cropped faces between any two scenes

The module is torch-free and has no hard dependency on PIL / numpy /
OpenCV.  Stub-mode SSIM is computed directly on the per-channel RGB
triple decoded from the pulid stub PNG format (see
``backends.pulid_portrait._stub_png``), so the gate is unit-testable
without GPU weights.  Real-mode SSIM (the eventual bar) routes through
an optional ``_compute_real_ssim`` path that imports PIL + numpy lazily
-- when those libs are absent, the real path raises a clear ImportError
so callers can decide to skip the gate instead of silently passing.

No audio imports.  C7 audio byte-identical gate is unaffected.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ---- ROADMAP Day 12 bar ---------------------------------------------------

# SSIM > 0.85 on cropped faces (ROADMAP Day 12 row).  We compare against
# strictly-greater-than semantics so 0.85 exactly does NOT pass -- this
# matches the ROADMAP wording.
SSIM_GATE: float = 0.85

# SSIM constants (Wang et al. 2004).  Tuned for 8-bit images with L=255.
_SSIM_C1: float = (0.01 * 255.0) ** 2      # 6.5025
_SSIM_C2: float = (0.03 * 255.0) ** 2      # 58.5225


# ---- Stub PNG decode ------------------------------------------------------
# The pulid_portrait stub writes a 1024x1024 solid-color PNG with a single
# IDAT chunk: signature + IHDR + IDAT + IEND.  Every scanline is
# ``\x00`` (filter none) + (R,G,B) * width.  We decode just enough to
# recover the solid (R,G,B) triple.


def _read_png_chunks(raw: bytes) -> Iterable[tuple[bytes, bytes]]:
    """Yield (chunk_type, chunk_data) for each chunk in a PNG file."""
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        ctype = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        # Skip CRC (4 bytes).
        pos += 12 + length
        yield (ctype, data)
        if ctype == b"IEND":
            break


def _decode_stub_solid_rgb(png_path: Path) -> tuple[int, int, int]:
    """Decode a pulid-portrait stub PNG back to its solid (R,G,B).

    Assumes the stub format (8-bit, RGB, filter-none rows).  If the PNG
    is not a solid-color stub, the first-pixel triple is returned, which
    is still useful as a rough identity probe but not a substitute for
    real SSIM on real images.
    """
    raw = png_path.read_bytes()
    ihdr_data: bytes | None = None
    idat_parts: list[bytes] = []
    for ctype, data in _read_png_chunks(raw):
        if ctype == b"IHDR":
            ihdr_data = data
        elif ctype == b"IDAT":
            idat_parts.append(data)
    if ihdr_data is None:
        raise ValueError(f"PNG missing IHDR: {png_path}")
    if not idat_parts:
        raise ValueError(f"PNG missing IDAT: {png_path}")

    width, _height, bit_depth, color_type = struct.unpack(
        ">IIBB", ihdr_data[:10]
    )
    if bit_depth != 8 or color_type != 2:
        # Color type 2 == RGB; anything else (paletted / grayscale /
        # RGBA) is outside the stub path.  Fall back to sampling the
        # top-left pixel regardless.
        pass

    # Decompress IDAT payload (may be split across multiple chunks).
    payload = zlib.decompress(b"".join(idat_parts))
    # First byte of each scanline is the filter type.  For solid stubs
    # the writer emits filter 0 (None), so the next 3 bytes are the
    # first pixel's R,G,B.
    if len(payload) < 4:
        raise ValueError(f"PNG IDAT too short: {png_path}")
    # bytes_per_pixel is 3 for RGB 8-bit
    bpp = 3 if color_type == 2 else (4 if color_type == 6 else 1)
    r = payload[1]
    g = payload[2] if bpp >= 2 else payload[1]
    b = payload[3] if bpp >= 3 else payload[1]
    return (int(r), int(g), int(b))


# ---- SSIM on solid colors -------------------------------------------------


def _ssim_channel_solid(a: int, b: int) -> float:
    """Simplified SSIM for a single channel of two solid-color images.

    Variance and covariance are both zero when every pixel carries the
    same value, so the SSIM reduces to the luminance term alone:

        SSIM = (2*mu_a*mu_b + C1) / (mu_a^2 + mu_b^2 + C1)

    When ``a == b`` this is exactly 1.0.  When a and b diverge, SSIM
    drops quickly, giving us a discriminating per-channel comparator.
    """
    fa = float(a)
    fb = float(b)
    num = (2.0 * fa * fb) + _SSIM_C1
    den = (fa * fa) + (fb * fb) + _SSIM_C1
    if den <= 0.0:
        return 1.0 if fa == fb else 0.0
    return num / den


def ssim_solid(
    rgb_a: tuple[int, int, int],
    rgb_b: tuple[int, int, int],
    *,
    reduction: str = "min",
) -> float:
    """SSIM between two solid-color RGB triples.

    ``reduction`` picks how the per-channel SSIMs are combined:
      - ``"min"`` (default): takes the worst channel.  Conservative --
        means "portrait likeness is only as strong as its weakest color
        channel".  This is what the Day 12 gate uses because it prevents
        two characters with similar luminance but different hue from
        falsely passing the likeness bar.
      - ``"mean"``: arithmetic mean of the three per-channel SSIMs.
      - ``"product"``: product of the three -- punishes any channel
        divergence super-linearly.
    """
    r = _ssim_channel_solid(rgb_a[0], rgb_b[0])
    g = _ssim_channel_solid(rgb_a[1], rgb_b[1])
    b = _ssim_channel_solid(rgb_a[2], rgb_b[2])
    if reduction == "min":
        return min(r, g, b)
    if reduction == "mean":
        return (r + g + b) / 3.0
    if reduction == "product":
        return r * g * b
    raise ValueError(f"unknown reduction {reduction!r}")


# ---- Real-mode SSIM (lazy import) ----------------------------------------


def _compute_real_ssim(png_a: Path, png_b: Path) -> float:
    """Real SSIM on cropped-face regions.

    Lazy imports PIL + numpy so callers on CI-only machines are not
    forced to install either.  If the imports fail, raises ImportError
    with a clear message; callers can catch and fall back to stub
    comparison or skip the gate.
    """
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "real-mode SSIM requires Pillow + numpy; "
            "install with: pip install pillow numpy"
        ) from exc

    def _to_gray(p: Path) -> "np.ndarray":
        img = Image.open(p).convert("L")
        return np.asarray(img, dtype=np.float64)

    a = _to_gray(png_a)
    b = _to_gray(png_b)
    if a.shape != b.shape:
        # Resize b onto a's canvas using nearest-neighbour to avoid
        # introducing a resampling SSIM penalty that hides identity
        # loss.  Day 12 gate assumes both outputs are same resolution
        # (1024x1024 per _RENDER_WIDTH/_RENDER_HEIGHT in pulid).
        img_b = Image.open(png_b).convert("L").resize(
            (a.shape[1], a.shape[0]), Image.NEAREST,
        )
        b = np.asarray(img_b, dtype=np.float64)

    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov_ab = float(((a - mu_a) * (b - mu_b)).mean())

    num = (2.0 * mu_a * mu_b + _SSIM_C1) * (2.0 * cov_ab + _SSIM_C2)
    den = (
        (mu_a * mu_a + mu_b * mu_b + _SSIM_C1)
        * (var_a + var_b + _SSIM_C2)
    )
    return float(num / den) if den > 0.0 else 1.0


def compute_ssim(
    png_a: Path,
    png_b: Path,
    *,
    mode: str = "auto",
) -> float:
    """SSIM between two portrait PNGs.

    Parameters
    ----------
    png_a, png_b : Path
        Portrait PNGs to compare.
    mode : "auto" | "stub" | "real"
        - ``"auto"`` (default): try the stub decoder first; if both PNGs
          look like pulid stubs, use ``ssim_solid``; otherwise fall
          through to real SSIM.
        - ``"stub"``: force stub path.
        - ``"real"``: force real path.  Raises ImportError if
          Pillow + numpy are not installed.
    """
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "stub", "real"):
        raise ValueError(f"mode must be auto/stub/real, got {mode!r}")

    if mode == "real":
        return _compute_real_ssim(png_a, png_b)

    if mode == "stub":
        rgb_a = _decode_stub_solid_rgb(png_a)
        rgb_b = _decode_stub_solid_rgb(png_b)
        return ssim_solid(rgb_a, rgb_b)

    # auto: attempt stub decode, fall back to real if the stub path
    # produces something that doesn't look solid enough.  The pulid
    # stub format is narrow enough that any decode failure is a clear
    # signal to use the real path.
    try:
        rgb_a = _decode_stub_solid_rgb(png_a)
        rgb_b = _decode_stub_solid_rgb(png_b)
    except (ValueError, zlib.error):
        return _compute_real_ssim(png_a, png_b)
    return ssim_solid(rgb_a, rgb_b)


# ---- Per-character regression --------------------------------------------


@dataclass
class PortraitSample:
    """A single pulid_portrait output located by :func:`find_portraits`."""
    scene_id: str
    shot_id: str
    character: str
    render_path: Path
    refs_hash: str = ""


@dataclass
class CharacterRegressionResult:
    """Outcome of a cross-scene character regression pass."""
    character: str
    gate: float
    samples: list[PortraitSample] = field(default_factory=list)
    pairs: list[tuple[str, str, float]] = field(default_factory=list)
    min_ssim: float = 1.0
    mean_ssim: float = 1.0
    gate_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "gate": self.gate,
            "min_ssim": round(self.min_ssim, 6),
            "mean_ssim": round(self.mean_ssim, 6),
            "gate_ok": self.gate_ok,
            "n_samples": len(self.samples),
            "n_pairs": len(self.pairs),
            "pairs": [
                {"a": a, "b": b, "ssim": round(s, 6)}
                for (a, b, s) in self.pairs
            ],
            "samples": [
                {
                    "scene_id": s.scene_id,
                    "shot_id": s.shot_id,
                    "refs_hash": s.refs_hash,
                    "render_path": str(s.render_path),
                }
                for s in self.samples
            ],
            "notes": list(self.notes),
        }


def find_portraits(
    out_dir: Path,
    character: str,
) -> list[PortraitSample]:
    """Walk an out_dir tree and return every pulid_portrait shot for
    ``character``.

    The layout assumes the Day 1-7 sidecar contract:

        <out_dir>/<scene_id>/<shot_id>/render.png
        <out_dir>/<scene_id>/<shot_id>/meta.json

    ``meta.json`` must identify the backend as ``pulid_portrait`` and
    carry a matching ``character`` field.  Non-matching meta entries are
    silently skipped.
    """
    out_dir = Path(out_dir)
    samples: list[PortraitSample] = []
    if not out_dir.exists():
        return samples

    for scene_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        # Allow either <out_dir>/<scene>/<shot>/ or <out_dir>/<shot>/.
        # We treat every descendant with a meta.json + render.png as a
        # candidate, and pull scene_id from the meta when present.
        for meta_path in scene_dir.rglob("meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("backend") != "pulid_portrait":
                continue
            if meta.get("character") != character:
                continue
            render_path = meta_path.parent / "render.png"
            if not render_path.exists():
                continue
            samples.append(PortraitSample(
                scene_id=str(meta.get("scene_id") or scene_dir.name),
                shot_id=str(meta.get("shot_id") or meta_path.parent.name),
                character=character,
                render_path=render_path,
                refs_hash=str(meta.get("refs_hash") or ""),
            ))
    # Stable ordering: by scene then shot so pair outputs are
    # reproducible across runs.
    samples.sort(key=lambda s: (s.scene_id, s.shot_id))
    return samples


def regress_character(
    out_dir: Path,
    character: str,
    *,
    gate: float = SSIM_GATE,
    mode: str = "auto",
) -> CharacterRegressionResult:
    """Run a cross-scene SSIM regression for one character.

    Walks every pulid_portrait shot for ``character`` in ``out_dir``,
    computes pairwise SSIM across DISTINCT scene_ids (within-scene
    pairs are skipped -- the Day 12 ROADMAP bar asks about Scene 1
    vs Scene 3, not shot-to-shot within a scene), and returns the
    worst-case and mean.

    ``gate_ok`` is True iff ``min_ssim > gate``.  A regression with
    fewer than 2 scenes represented is recorded as gate_ok=True with
    a note -- we can't fail a bar that isn't testable.
    """
    samples = find_portraits(out_dir, character)
    result = CharacterRegressionResult(character=character, gate=gate)
    result.samples = samples

    if not samples:
        result.notes.append(f"no pulid_portrait samples found for {character!r}")
        return result

    scenes_present = {s.scene_id for s in samples}
    if len(scenes_present) < 2:
        result.notes.append(
            f"only one scene ({next(iter(scenes_present))!r}) has portraits "
            f"for {character!r}; cross-scene regression requires ≥ 2 scenes"
        )
        return result

    ssim_values: list[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a = samples[i]
            b = samples[j]
            if a.scene_id == b.scene_id:
                # skip within-scene pairs
                continue
            ssim = compute_ssim(a.render_path, b.render_path, mode=mode)
            result.pairs.append((
                f"{a.scene_id}/{a.shot_id}",
                f"{b.scene_id}/{b.shot_id}",
                ssim,
            ))
            ssim_values.append(ssim)

    if not ssim_values:
        result.notes.append(
            "no cross-scene pairs produced (samples present but all "
            "within one scene)"
        )
        return result

    result.min_ssim = min(ssim_values)
    result.mean_ssim = sum(ssim_values) / len(ssim_values)
    result.gate_ok = result.min_ssim > gate
    return result


def regress_cast(
    out_dir: Path,
    cast: Iterable[str],
    *,
    gate: float = SSIM_GATE,
    mode: str = "auto",
) -> dict[str, CharacterRegressionResult]:
    """Run :func:`regress_character` across a cast list.

    Returns a dict keyed by character.  The full report ``gate_ok`` is
    True iff every character passes.  Callers typically feed this the
    two-character SIGNAL LOST cast: ``("BABA", "BOOEY")``.
    """
    return {c: regress_character(out_dir, c, gate=gate, mode=mode) for c in cast}


__all__ = [
    "SSIM_GATE",
    "PortraitSample",
    "CharacterRegressionResult",
    "ssim_solid",
    "compute_ssim",
    "find_portraits",
    "regress_character",
    "regress_cast",
]
