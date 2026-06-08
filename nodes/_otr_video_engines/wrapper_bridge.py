"""Shared ComfyUI wrapper-node bridge for the in-process video motion engines.

The in-process motion engines (humo / ltx_video / wan_i2v) render by driving the
installed ComfyUI wrapper NODE CLASSES directly (no GraphBuilder, no HTTP server,
V invariant): resolve the node class out of ComfyUI's ``NODE_CLASS_MAPPINGS``,
execute a small DECLARATIVE node graph in dependency order, pull the decoded
IMAGE batch, and encode it to the platform's ALWAYS-SILENT bt709 / yuv420p MP4
(V-1: only OTR_MasterAudioMux ever adds audio; the terminal mux is -c:a copy).
This module factors those mechanics so each adapter file stays a thin declarative
spec and every mechanic is unit-tested ONCE on the CPU box.

CPU-testable here (tests/test_video_wrapper_bridge.py):
  * resolve_node_class / resolve_graph_classes -- pure NODE_CLASS_MAPPINGS lookup
    by ordered candidate names; NAMED fail-closed (WrapperNodeMissing) when absent;
  * run_graph -- a generic declarative executor (topo-order instantiate + call
    FUNCTION + wire tuple outputs by slot), driven with fake node classes;
  * quantize_frames_4n1 -- the Wan 2.1 VAE 4n+1 latent rule (legacy-proven);
  * images_to_uint8 -- a ComfyUI IMAGE batch (B,H,W,C float 0..1) -> uint8 frames;
  * ffmpeg_silent_mp4_cmd / ffmpeg_still_motion_cmd / ffmpeg_lavfi_floor_cmd --
    the exact ffmpeg arg lists for the silent bt709 / yuv420p clip contract.

The GPU / ffmpeg leaves (encode_frames_to_silent_mp4, run_ffmpeg, run_graph over
real wrapper nodes) need ffmpeg / torch on the box; the arg + topo mechanics
around them are proven on CPU. Cold-import clean (V-12): module scope imports only
the stdlib; numpy / torch / the ComfyUI ``nodes`` registry / folder_paths are
imported LAZILY inside the functions that need them. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

import os
import subprocess

#: Machine-wide VRAM ceiling for the single resident heavy engine (A invariant).
VRAM_CEILING_MB = 14500

#: The CanonicalClip pixel/colour contract the engines' canonicalize() emits.
PIX_FMT = "yuv420p"
COLOR_PRIMARIES = "bt709"


class WrapperNodeMissing(RuntimeError):
    """A required ComfyUI wrapper node class is not installed / registered.

    Raised fail-closed (NAMED) so a render degrades via the engine fallback chain
    with a LOUD restamp instead of crashing -- never a silent skip."""


class GraphExecutionError(RuntimeError):
    """A declarative node graph could not be executed (bad spec, a missing class,
    a cycle, or a node raising). Always NAMED; never a silent failure."""


# --------------------------------------------------------------------------- #
# Node-class resolution (pure; ComfyUI NODE_CLASS_MAPPINGS lookup)
# --------------------------------------------------------------------------- #
def node_class_mappings(mapping=None):
    """Return the ComfyUI NODE_CLASS_MAPPINGS (lazy import) or an injected map.

    Importing ``nodes`` reaches ComfyUI's node registry, not a heavy lib; it is
    imported LAZILY (never at module scope) so the cold-import invariant (V-12)
    holds. A missing registry (the headless CPU box / pytest) returns an empty
    dict, so resolution fails closed with a NAMED WrapperNodeMissing rather than
    an ImportError leaking out."""
    if mapping is not None:
        return mapping
    try:
        import nodes as _comfy_nodes  # ComfyUI's node registry module (lazy)
        return getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    except Exception:  # noqa: BLE001 -- absent registry -> fail closed downstream
        return {}


def resolve_node_class(candidates, mapping=None):
    """Return the first installed node class among ``candidates`` (ordered names).

    ``candidates`` is a str or an ordered iterable of registered class names; the
    first present in NODE_CLASS_MAPPINGS wins (an engine can prefer a wrapper node
    but accept a core fallback). Raises WrapperNodeMissing (NAMED; lists what it
    looked for) when none is installed -- the fail-closed path the engine fallback
    chain consumes."""
    if isinstance(candidates, str):
        candidates = (candidates,)
    names = [c for c in candidates if c]
    m = node_class_mappings(mapping)
    for name in names:
        if name in m:
            return m[name]
    raise WrapperNodeMissing(
        "none of the ComfyUI node classes %r are installed (install the wrapper "
        "+ restart ComfyUI); %d node classes registered" % (names, len(m)))


def resolve_graph_classes(specs, mapping=None):
    """Resolve a dict ``{node_id: candidates}`` -> ``{node_id: class}``.

    Aggregates EVERY missing node into ONE NAMED error so the operator sees the
    full install list at once, not one-at-a-time."""
    m = node_class_mappings(mapping)
    out, missing = {}, []
    for node_id, candidates in specs.items():
        if isinstance(candidates, str):
            candidates = (candidates,)
        hit = next((c for c in candidates if c and c in m), None)
        if hit is None:
            missing.append("%s=%r" % (node_id, [c for c in candidates if c]))
        else:
            out[node_id] = m[hit]
    if missing:
        raise WrapperNodeMissing(
            "missing ComfyUI node classes for: %s (install the wrapper + restart "
            "ComfyUI)" % "; ".join(sorted(missing)))
    return out


# --------------------------------------------------------------------------- #
# Generic declarative node-graph executor (ComfyUI's execution model, in-process)
# --------------------------------------------------------------------------- #
class Wire(tuple):
    """An edge in a declarative graph: the ``slot``-th output of node ``src``.

    A 2-tuple subclass so a graph spec stays plain data and a Wire is trivially
    distinguishable from a literal input value (it is checked BEFORE a generic
    tuple everywhere below)."""

    __slots__ = ()

    def __new__(cls, src, slot=0):
        return super().__new__(cls, (src, int(slot)))

    @property
    def src(self):
        return self[0]

    @property
    def slot(self):
        return self[1]


def _iter_wires(val):
    """Yield every Wire reachable inside an input value (recursing list/tuple/
    dict containers). A Wire is matched BEFORE the generic tuple branch."""
    if isinstance(val, Wire):
        yield val
    elif isinstance(val, dict):
        for v in val.values():
            for w in _iter_wires(v):
                yield w
    elif isinstance(val, (list, tuple)):
        for v in val:
            for w in _iter_wires(v):
                yield w


def _resolve_value(val, results):
    """Resolve Wires in an input value to concrete node outputs (recursing
    containers); literals pass through. Wire is matched BEFORE generic tuple."""
    if isinstance(val, Wire):
        out = results[val.src]
        try:
            return out[val.slot]
        except (TypeError, IndexError, KeyError):
            raise GraphExecutionError(
                "node %r output slot %d unavailable (output was %r)"
                % (val.src, val.slot, type(out).__name__))
    if isinstance(val, dict):
        return {k: _resolve_value(v, results) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return type(val)(_resolve_value(v, results) for v in val)
    return val


def _topo_order(graph):
    """Deterministic Kahn topo-sort of node ids by their Wire dependencies.

    Raises GraphExecutionError on a dangling Wire source or a cycle. Ties break
    on the sorted node id, so execution order is reproducible (determinism)."""
    deps = {nid: set() for nid in graph}
    for nid, node in graph.items():
        for val in (node.get("inputs") or {}).values():
            for w in _iter_wires(val):
                if w.src not in graph:
                    raise GraphExecutionError(
                        "node %r wires from unknown source %r" % (nid, w.src))
                deps[nid].add(w.src)
    order, satisfied, remaining = [], set(), set(graph)
    while remaining:
        ready = sorted(n for n in remaining if deps[n] <= satisfied)
        if not ready:
            raise GraphExecutionError(
                "graph has a cycle among %r" % sorted(remaining))
        for n in ready:
            order.append(n)
            satisfied.add(n)
            remaining.discard(n)
    return order


def run_graph(graph, classes=None, *, terminal=None):
    """Execute a declarative node graph in dependency order; return the results.

    ``graph`` maps ``node_id -> {"class": <class|name>, "inputs": {name: literal |
    Wire(src, slot)}, "function": optional}``. A class may be a resolved class
    object OR a name to look up in ``classes`` (the dict from resolve_graph_classes).
    Each node is instantiated, its ``FUNCTION`` method called with the resolved
    inputs, and its return normalised to a tuple. Returns ``{node_id: out_tuple}``;
    when ``terminal`` is given returns that node's tuple directly. Fail-closed: a
    missing class / cycle / a node raising becomes a NAMED GraphExecutionError
    (never a silent partial render)."""
    classes = classes or {}
    results = {}
    for nid in _topo_order(graph):
        node = graph[nid]
        cls = node.get("class")
        if isinstance(cls, str):
            if cls not in classes:
                raise GraphExecutionError(
                    "node %r class %r unresolved" % (nid, cls))
            cls = classes[cls]
        if cls is None:
            raise GraphExecutionError("node %r has no class" % nid)
        fn_name = node.get("function") or getattr(cls, "FUNCTION", None)
        if not fn_name:
            raise GraphExecutionError(
                "node %r class %r has no FUNCTION" % (nid, getattr(cls, "__name__", cls)))
        inst = cls() if isinstance(cls, type) else cls
        fn = getattr(inst, fn_name, None)
        if not callable(fn):
            raise GraphExecutionError(
                "node %r function %r not callable" % (nid, fn_name))
        kwargs = {k: _resolve_value(v, results)
                  for k, v in (node.get("inputs") or {}).items()}
        try:
            out = fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 -- surfaced NAMED, never silent
            raise GraphExecutionError(
                "node %r (%s) raised %s: %s"
                % (nid, fn_name, type(exc).__name__, exc))
        results[nid] = out if isinstance(out, tuple) else (out,)
    if terminal is not None:
        if terminal not in results:
            raise GraphExecutionError("terminal node %r not in graph" % terminal)
        return results[terminal]
    return results


# --------------------------------------------------------------------------- #
# Frame-count / dimension quantization (pure)
# --------------------------------------------------------------------------- #
def quantize_frames_4n1(target, min_frames=1, max_frames=None):
    """Smallest valid Wan-2.1-VAE length (4n+1) >= ``target``, clamped.

    The Wan 2.1 VAE compresses 4 frames into 1 latent, so a HuMo / Wan length
    widget must satisfy ``(length - 1) % 4 == 0`` (legacy render_humo_batch). This
    returns the smallest ``4n+1 >= max(1, target)``, floored at ``min_frames`` and
    (if given) capped at ``max_frames`` snapped DOWN to a valid 4n+1."""
    t = max(1, int(target))
    frames = 4 * ((t - 1 + 3) // 4) + 1            # smallest 4n+1 >= t
    mn = int(min_frames)
    if frames < mn:
        frames = 4 * ((mn - 1 + 3) // 4) + 1       # snap min UP to a valid 4n+1
    if max_frames is not None and frames > int(max_frames):
        frames = 4 * ((int(max_frames) - 1) // 4) + 1   # snap DOWN to a valid 4n+1
    return frames


def even_dim(x):
    """Round to the nearest even int >= 2 (model-stride / yuv420p mod-2 safe)."""
    n = int(round(float(x)))
    n -= n % 2
    return max(2, n)


# --------------------------------------------------------------------------- #
# IMAGE-batch -> uint8 frames (numpy lazy; CPU-testable with a numpy array)
# --------------------------------------------------------------------------- #
def images_to_uint8(images):
    """A ComfyUI IMAGE batch -> a contiguous uint8 numpy array shaped (B,H,W,C).

    ComfyUI IMAGE tensors are float32 in [0,1], shape (B,H,W,C) with C=3 (RGB).
    Accepts a torch tensor (duck-typed via .detach().cpu().numpy()) OR a numpy
    array, so the conversion math is CPU-testable without torch. numpy is imported
    lazily (kept off the cold-import path)."""
    import numpy as np
    arr = images
    for attr in ("detach", "cpu"):
        m = getattr(arr, attr, None)
        if callable(m):
            arr = m()
    npy = getattr(arr, "numpy", None)
    if callable(npy):
        arr = npy()
    arr = np.asarray(arr)
    if arr.ndim == 3:                              # a single frame -> a batch of 1
        arr = arr[None, ...]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(arr)


# --------------------------------------------------------------------------- #
# ffmpeg command builders (pure) + runners (the ffmpeg leaf)
# --------------------------------------------------------------------------- #
def _bt709_encode_args(crf):
    """The shared silent-clip encode tail: H.264, yuv420p, bt709, NO audio
    (V-1). One definition so every builder emits the identical CanonicalClip
    colour contract."""
    return [
        "-an",                                     # V-1: only the mux adds audio
        "-c:v", "libx264", "-crf", str(int(crf)), "-pix_fmt", PIX_FMT,
        "-color_primaries", COLOR_PRIMARIES, "-color_trc", COLOR_PRIMARIES,
        "-colorspace", COLOR_PRIMARIES, "-movflags", "+faststart",
    ]


def ffmpeg_silent_mp4_cmd(out_path, width, height, fps, *, ffmpeg="ffmpeg", crf=18):
    """ffmpeg arg list: raw rgb24 frames on stdin -> a SILENT bt709 / yuv420p
    H.264 MP4. Frames are piped as width*height*3 byte rgb24 images at ``fps``."""
    return [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "%dx%d" % (even_dim(width), even_dim(height)),
        "-r", str(fps), "-i", "pipe:0",
    ] + _bt709_encode_args(crf) + [out_path]


def ffmpeg_still_motion_cmd(still_path, out_path, width, height, fps, frame_count,
                            *, ffmpeg="ffmpeg", zoom_to=1.08, crf=18):
    """Ken Burns: a slow zoom over ONE still -> a silent bt709 clip of exactly
    ``frame_count`` frames. The still is scaled to COVER the canvas with one
    uniform scale (no stretch) and centre-cropped, then zoompan eases the zoom."""
    n = max(1, int(frame_count))
    w, h = even_dim(width), even_dim(height)
    step = max(0.0, (float(zoom_to) - 1.0)) / n
    vf = (
        "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,"
        "zoompan=z='min(zoom+%.6f,%.4f)':d=%d:s=%dx%d:fps=%d"
        % (w, h, w, h, step, float(zoom_to), n, w, h, int(fps))
    )
    return [
        ffmpeg, "-y", "-loop", "1", "-i", still_path,
        "-frames:v", str(n), "-vf", vf,
    ] + _bt709_encode_args(crf) + [out_path]


def ffmpeg_lavfi_floor_cmd(out_path, width, height, fps, frame_count,
                           *, source=None, ffmpeg="ffmpeg", crf=20):
    """Synthesize a silent bt709 floor clip of exactly ``frame_count`` frames from
    a libavfilter ``source`` (default a dark slate field). No input file required,
    so the radio floor ALWAYS renders (the fallback-chain terminus)."""
    n = max(1, int(frame_count))
    w, h = even_dim(width), even_dim(height)
    src = source or ("color=c=0x0A0E14:s=%dx%d:r=%d" % (w, h, int(fps)))
    return [
        ffmpeg, "-y", "-f", "lavfi", "-i", src, "-frames:v", str(n),
    ] + _bt709_encode_args(crf) + [out_path]


def run_ffmpeg(cmd):
    """Run an ffmpeg command (no stdin); raise a NAMED GraphExecutionError on a
    non-zero exit or a missing ffmpeg. Returns ``cmd`` on success."""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise GraphExecutionError("ffmpeg not found: %s" % exc)
    if proc.returncode != 0:
        raise GraphExecutionError(
            "ffmpeg failed rc=%s: %s"
            % (proc.returncode,
               (proc.stderr or b"").decode("utf-8", "replace")[-400:]))
    return cmd


def encode_frames_to_silent_mp4(frames, out_path, fps, *, ffmpeg="ffmpeg", crf=18):
    """Encode a uint8 (B,H,W,3) frame batch to a silent bt709 MP4 via ffmpeg.

    Returns ``(out_path, frame_count)``. The arg build is tested via
    ffmpeg_silent_mp4_cmd; this runs the encoder (the ffmpeg leaf) and raises a
    NAMED GraphExecutionError on failure."""
    import numpy as np
    frames = np.ascontiguousarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise GraphExecutionError(
            "expected (B,H,W,3) uint8 frames, got shape %r" % (frames.shape,))
    b, h, w, _ = frames.shape
    cmd = ffmpeg_silent_mp4_cmd(out_path, w, h, fps, ffmpeg=ffmpeg, crf=crf)
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, err = proc.communicate(frames.tobytes())
    except FileNotFoundError as exc:
        raise GraphExecutionError("ffmpeg not found: %s" % exc)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise GraphExecutionError(
            "ffmpeg frame encode failed rc=%s: %s"
            % (proc.returncode, (err or b"").decode("utf-8", "replace")[-400:]))
    return out_path, int(b)


# --------------------------------------------------------------------------- #
# Staging inputs into ComfyUI's input dir (the proven legacy LoadImage pattern)
# --------------------------------------------------------------------------- #
def comfy_input_dir():
    """ComfyUI's input directory (where LoadImage / LoadAudio resolve filenames).
    Lazy folder_paths import; falls back to ``<ComfyUI>/input`` headless."""
    try:
        import folder_paths
        return folder_paths.get_input_directory()
    except Exception:  # noqa: BLE001 -- headless fallback
        here = os.path.abspath(__file__)
        comfy = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(here)))))     # ...\ComfyUI
        return os.path.join(comfy, "input")


def stage_into_comfy_input(src_path, dst_name=None):
    """Copy ``src_path`` into ComfyUI's input dir and return the staged BASENAME.

    The in-process motion engines feed a portrait / audio into the wrapper graph
    by filename (the proven legacy render_humo_batch pattern). Raises a NAMED
    GraphExecutionError on a missing source."""
    import shutil
    if not src_path or not os.path.exists(src_path):
        raise GraphExecutionError("input file missing: %r" % src_path)
    dst_dir = comfy_input_dir()
    os.makedirs(dst_dir, exist_ok=True)
    name = dst_name or os.path.basename(src_path)
    shutil.copy2(src_path, os.path.join(dst_dir, name))
    return name


__all__ = [
    "VRAM_CEILING_MB", "PIX_FMT", "COLOR_PRIMARIES",
    "WrapperNodeMissing", "GraphExecutionError",
    "node_class_mappings", "resolve_node_class", "resolve_graph_classes",
    "Wire", "run_graph",
    "quantize_frames_4n1", "even_dim", "images_to_uint8",
    "ffmpeg_silent_mp4_cmd", "ffmpeg_still_motion_cmd", "ffmpeg_lavfi_floor_cmd",
    "run_ffmpeg", "encode_frames_to_silent_mp4",
    "comfy_input_dir", "stage_into_comfy_input",
]
