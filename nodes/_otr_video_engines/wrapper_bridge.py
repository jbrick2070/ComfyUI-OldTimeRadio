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

import logging
import os
import subprocess

_LOG = logging.getLogger("OTR.video.wrapper_bridge")

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


def _run_coroutine(coro):
    """Run a coroutine to completion on a private event loop. ComfyUI's V3 node
    API exposes an async node's FUNCTION as ``EXECUTE_NORMALIZED_ASYNC`` (a
    coroutine); the in-process render driver runs in a worker thread with no
    running loop, so a fresh loop drives it to completion."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def normalize_node_output(out):
    """Normalize a node FUNCTION's return into a plain output tuple.

    ComfyUI's V3 node API (``comfy_api``) makes a node's ``FUNCTION`` the
    classmethod ``EXECUTE_NORMALIZED`` / ``EXECUTE_NORMALIZED_ASYNC``, which
    returns a ``NodeOutput`` wrapper (its positional outputs live on ``.args``)
    -- or, for an async ``execute``, a coroutine of one. The legacy V1 API
    returns a plain tuple. This unwraps both to the raw output tuple
    :func:`run_graph` wires by slot (duck-typed -- never imports comfy)."""
    import inspect
    if inspect.iscoroutine(out):
        out = _run_coroutine(out)
    if (out.__class__.__name__ == "NodeOutput"
            and hasattr(out, "args") and hasattr(out, "block_execution")):
        return tuple(out.args)
    return out


def _soft_free():
    """Lightweight reclaim of dropped intermediates: GC + ComfyUI's soft cache
    empty. Deliberately does NOT force-unload resident models: the proven HuMo
    path (BUG-265 low_vram_default) keeps the 1.7B stack FULLY RESIDENT with zero
    offload, and an aggressive ``free_memory`` here only fragments the 16 GB
    allocator into the OOM/thrash it was meant to avoid. A no-op off the ComfyUI
    box (comfy import guarded)."""
    import gc
    gc.collect()
    try:
        import comfy.model_management as _mm
        _mm.soft_empty_cache()
    except Exception:  # noqa: BLE001 -- not inside ComfyUI / no torch -> skip
        pass


def reclaim_idle_models(reason=""):
    """Evict every currently-loaded ComfyUI model's weights off the GPU, the
    proven HuMo VRAM discipline the in-process refactor dropped.

    This is the BUG-291 reclaim ported from the legacy
    ``_otr_vram_levers.free_otr_pipeline_residue`` step 3a: walk
    ``comfy.model_management.current_loaded_models`` and ``detach(unpatch_all=
    True)`` each patcher -- which moves its weights to the offload (CPU) device
    even though a wrapper ref survives -- so the umt5 CLIP + whisper encoders'
    VRAM is reclaimed after a render and the resident heavy stack drops back
    under the single-resident-engine ceiling. It deliberately does NOT call
    ``unload_all_models`` (forbidden by V-4/V-5); the detach is the actual VRAM
    move, the registry clear is not needed. LOUD by contract. Returns the number
    detached. A no-op off the ComfyUI box (comfy import guarded)."""
    try:
        import comfy.model_management as _mm
    except Exception:  # noqa: BLE001 -- not inside ComfyUI -> nothing to reclaim
        return 0
    try:
        loaded = list(getattr(_mm, "current_loaded_models", None) or [])
    except Exception:  # noqa: BLE001
        loaded = []
    detached = 0
    for lm in loaded:
        patcher = getattr(lm, "model", lm)        # LoadedModel.model is the patcher
        try:
            det = getattr(patcher, "detach", None)
            if callable(det):
                try:
                    det(unpatch_all=True)
                except TypeError:
                    det()                         # older signature: no kwargs
                detached += 1
                continue
        except Exception:  # noqa: BLE001
            pass
        try:                                      # fallback: move weights to CPU
            inner = getattr(patcher, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to("cpu")
                detached += 1
        except Exception:  # noqa: BLE001
            pass
    try:
        import gc
        gc.collect()
        _mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass
    _LOG.warning(
        "[OTR video] LOUD VRAM reclaim%s: detached %d resident model(s) "
        "(encoder eviction; no unload_all_models)",
        (" (%s)" % reason) if reason else "", detached)
    return detached


def run_graph(graph, classes=None, *, terminal=None, free_after_use=False,
              keep=None):
    """Execute a declarative node graph in dependency order; return the results.

    ``graph`` maps ``node_id -> {"class": <class|name>, "inputs": {name: literal |
    Wire(src, slot)}, "function": optional}``. A class may be a resolved class
    object OR a name to look up in ``classes`` (the dict from resolve_graph_classes).
    Each node is instantiated, its ``FUNCTION`` method called with the resolved
    inputs, and its return normalised to a tuple. Returns ``{node_id: out_tuple}``;
    when ``terminal`` is given returns that node's tuple directly. Fail-closed: a
    missing class / cycle / a node raising becomes a NAMED GraphExecutionError
    (never a silent partial render).

    ``free_after_use`` mirrors ComfyUI's executor memory model: a node's outputs
    are dropped (and a soft VRAM reclaim run) once every later consumer has run,
    so a big intermediate (e.g. the umt5 CLIP / whisper encoder) is freed before
    the sampler instead of pinning the whole graph resident (the A-S7.5
    single-resident-engine VRAM ceiling). ``keep`` is the set of node ids never
    freed (the terminal + the MODEL nodes an engine retains for V-4 teardown)."""
    classes = classes or {}
    keep = set(keep or ())
    if terminal is not None:
        keep.add(terminal)
    # Remaining-consumer count per source node (for free_after_use).
    node_srcs = {}
    remaining = {nid: 0 for nid in graph}
    for nid in graph:
        srcs = set()
        for val in (graph[nid].get("inputs") or {}).values():
            for w in _iter_wires(val):
                srcs.add(w.src)
        node_srcs[nid] = srcs
        for s in srcs:
            remaining[s] = remaining.get(s, 0) + 1
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
            out = normalize_node_output(fn(**kwargs))
        except Exception as exc:  # noqa: BLE001 -- surfaced NAMED, never silent
            raise GraphExecutionError(
                "node %r (%s) raised %s: %s"
                % (nid, fn_name, type(exc).__name__, exc))
        results[nid] = out if isinstance(out, tuple) else (out,)
        if free_after_use:
            did_free = False
            for s in node_srcs.get(nid, ()):
                remaining[s] -= 1
                if remaining[s] <= 0 and s not in keep and s in results:
                    del results[s]
                    did_free = True
            if did_free:
                _soft_free()
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


def extend_frames_to_target(frames, target_frame_count):
    """PING-PONG / mirror-extend a short frame batch up to ``target_frame_count``.

    The clip-fill companion to the dynamic-VRAM budget (GO_FORWARD 2026-06-18): an
    engine that can only afford a SHORT render (e.g. wan_ti2v's VRAM-predicted
    length) extends here so the beat is FILLED with continuous motion instead of
    the composite holding the last frame (the 0.68s-then-freeze bug). Generalizes
    LTX's one-shot boomerang into a tiled mirror cycle
    (``[0,1,..,N-1,N-2,..,1]``, period ``2N-2``) tiled and trimmed to the target,
    so the loop has NO hard seam (the cycle joins frame 1 -> 0 -> 1, symmetric).

    Returns ``frames`` UNCHANGED when it already has >= target frames (or target
    <= 0). A single frame cannot mirror into motion, so it is repeated (the caller
    surfaces that LOUD). ``frames`` is the uint8 ``[N,H,W,C]`` array from
    :func:`images_to_uint8`; numpy is imported lazily (cold-import V-12). Pure;
    CPU-testable."""
    import numpy as np
    arr = np.asarray(frames)
    n = int(arr.shape[0]) if arr.ndim else 0
    target = int(target_frame_count or 0)
    if n == 0 or target <= n:
        return arr
    if n < 2:                                   # no motion to mirror -> repeat
        reps = (target + n - 1) // n
        return np.ascontiguousarray(np.concatenate([arr] * reps, axis=0)[:target])
    cycle = np.concatenate([arr, arr[-2:0:-1]], axis=0)   # period 2N-2, seamless
    reps = (target + len(cycle) - 1) // len(cycle)
    tiled = np.concatenate([cycle] * reps, axis=0)
    return np.ascontiguousarray(tiled[:target])


class MirrorExtensionForbidden(ValueError):
    """A short render on an AUDIO-SYNCED lane cannot be mirror-extended.

    Raised by :func:`fit_frames_to_target` when ``allow_mirror=False`` and the
    render is shorter than the audio target. Carries the numbers so the caller
    can name the tier and the remedy.
    """

    def __init__(self, rendered, target):
        self.rendered = int(rendered)
        self.target = int(target)
        super().__init__(
            "audio-synced render is %d frame(s) but the beat's audio needs %d; "
            "mirror-extension is FORBIDDEN on a lip-synced lane (it would play "
            "the mouth backwards)" % (self.rendered, self.target))


def fit_frames_to_target(frames, target_frame_count, *, allow_mirror=True):
    """EXACT-FIT ``frames`` to ``target_frame_count``: TRIM when over,
    mirror-extend (:func:`extend_frames_to_target`) when short, identity on a
    match. The capped render tiers (e.g. HuMo-14B's VRAM frame cap) need this so
    the encoded clip's frame_count EQUALS the audio-derived target -- a short clip
    would otherwise make the composite hold the last frame, and a long one would
    drift the manifest timing. ``frames`` is the uint8 ``[N,H,W,C]`` array from
    :func:`images_to_uint8`; numpy is imported lazily (cold-import V-12). Pure;
    CPU-testable.

    ``allow_mirror=False`` (operator directive 2026-07-25: "lip sync needs
    continuity, no render backwards, that doesn't work") makes a SHORT render
    TERMINAL instead of mirrored. :func:`extend_frames_to_target` builds the
    cycle ``[0,1,..,N-1,N-2,..,1]`` -- the back half of every cycle is the
    render played in REVERSE. On a scene lane that is decorative motion and
    the loop is seamless by design. On an AUDIO-SYNCED lane (an
    ``audio_driven_face`` talking head, anything lip-synced) it plays the
    mouth backwards against forward speech, which is a sync defect that ships
    as a finished episode. TRIMMING stays legal under this flag -- trimming
    never reverses time.
    """
    import numpy as np
    arr = np.asarray(frames)
    n = int(arr.shape[0]) if arr.ndim else 0
    target = int(target_frame_count or 0)
    if n == 0 or target <= 0 or n == target:
        return arr
    if n > target:
        return np.ascontiguousarray(arr[:target])
    if not allow_mirror:
        raise MirrorExtensionForbidden(n, target)
    return extend_frames_to_target(arr, target)


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
    """Slow pan: a slow zoom over ONE still -> a silent bt709 clip of exactly
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


def ffmpeg_still_static_cmd(still_path, out_path, width, height, fps, frame_count,
                            *, ffmpeg="ffmpeg", crf=18, pad_color="black"):
    """Flat hold: ONE still shown STATIC (NO pan/zoom) for exactly ``frame_count``
    frames -> a silent bt709 clip. Unlike the slow-pan ``ffmpeg_still_motion_cmd``
    (which scales to COVER + centre-CROPs -> can cut a face), this scales to FIT
    (``force_original_aspect_ratio=decrease``) and PADs (letterboxes) so the WHOLE
    image is always visible and NOTHING is cropped -- the flat-still contract for
    'I just want stills, no motion'. Deterministic; no RNG."""
    n = max(1, int(frame_count))
    w, h = even_dim(width), even_dim(height)
    vf = (
        "scale=%d:%d:force_original_aspect_ratio=decrease,"
        "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=%s,setsar=1,fps=%d"
        % (w, h, w, h, pad_color, int(fps))
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


def ffmpeg_concat_segments_cmd(segments, out_path, *, ffmpeg="ffmpeg", crf=18):
    """ffmpeg arg list: several rendered SEGMENTS -> ONE silent bt709 beat clip.

    Multi-clip coverage chunk 6d (2026-07-26). ``segments`` is a sequence of
    ``(path, drop_head, keep_frames)``: the assembler drops ``drop_head`` frames
    from the front of that segment and keeps exactly ``keep_frames`` after it.

    FRAME-EXACT BY SELECTION, not by time. Every trim is expressed as
    ``select='between(n, first, last)'`` over the frame INDEX -- a
    seconds-based trim rounds, and rounding a beat's length is the drift the
    exact-sum partitioner exists to refuse. ``setpts`` restamps each kept run
    so concat sees contiguous timestamps.

    The output goes through the SAME ``_bt709_encode_args`` tail as every other
    OTR clip, so an assembled beat is a CanonicalClip like any other -- not a
    special case the mux has to know about.
    """
    rows = [(str(p), int(d), int(k)) for p, d, k in segments]
    if not rows:
        raise GraphExecutionError(
            "concat was handed NO segments -- an assembled beat with nothing "
            "in it is not a beat")
    # REFUSE, do not clamp (2026-07-26 QA panel, both seats). An earlier draft
    # wrote ``max(1, int(k))``, which turned a segment that keeps zero or fewer
    # frames -- a plan the partitioner would never emit -- into a segment that
    # keeps exactly one, and then assembled a beat of the wrong length without
    # a word. This build refuses clamps everywhere else; it refuses here too.
    for i, (path, drop, keep) in enumerate(rows):
        if drop < 0 or keep < 1:
            raise GraphExecutionError(
                "concat segment %d (%s) keeps %d frame(s) after dropping %d -- "
                "a segment must keep at least one frame and cannot drop a "
                "negative number. NO CLAMP." % (i, path, keep, drop))
    args = [ffmpeg, "-y"]
    for path, _drop, _keep in rows:
        args += ["-i", path]
    chains = []
    for i, (_path, drop, keep) in enumerate(rows):
        chains.append(
            "[%d:v]select='between(n\\,%d\\,%d)',setpts=N/FRAME_RATE/TB[v%d]"
            % (i, drop, drop + keep - 1, i))
    joined = "".join("[v%d]" % i for i in range(len(rows)))
    graph = ";".join(chains) + ";" + joined + "concat=n=%d:v=1:a=0[out]" % len(rows)
    args += ["-filter_complex", graph, "-map", "[out]"]
    return args + _bt709_encode_args(crf) + [out_path]


def ffmpeg_terminal_frame_cmd(clip_path, out_path, *, ffmpeg="ffmpeg"):
    """ffmpeg arg list: a rendered clip -> ONE image file holding its LAST frame.

    Multi-clip coverage chunk 6c (2026-07-26). A CHAIN successor begins exactly
    on its predecessor's terminal frame, and segment N+1 needs that frame
    SYNCHRONOUSLY -- inside the render loop, not from a post-episode pass. This
    is the seam that produces it.

    DECODES THE WHOLE CLIP ON PURPOSE, rather than seeking with ``-sseof``.
    ``-update 1`` rewrites the same output file for every decoded frame, so the
    file that survives IS the last frame -- exactly, for any clip, including a
    9-frame segment where a one-second tail seek has nothing to land on and a
    non-seekable container where it lands somewhere else entirely. A segment is
    tens of frames; a wrong "last" frame is a visible jump at every cut.
    """
    return [
        ffmpeg, "-y", "-i", clip_path,
        "-an",                                # V-1: never carry audio around
        "-update", "1", "-f", "image2", out_path,
    ]


def extract_terminal_frame(clip_path, out_path, *, ffmpeg="ffmpeg"):
    """Write ``clip_path``'s LAST frame to ``out_path`` and PROVE it landed.

    Returns ``out_path``. Raises a NAMED :class:`GraphExecutionError` if ffmpeg
    fails, or if it exits 0 having written nothing usable -- ffmpeg reports
    success for an input it decoded zero frames from, and a 0-byte file handed
    on as the next segment's init image is a black frame at the cut with a
    clean exit code in front of it. Asset existence proves completion, never
    the return code.
    """
    run_ffmpeg(ffmpeg_terminal_frame_cmd(clip_path, out_path, ffmpeg=ffmpeg))
    if not os.path.exists(out_path) or os.path.getsize(out_path) <= 0:
        raise GraphExecutionError(
            "terminal-frame extraction produced no usable image for %r (ffmpeg "
            "exited 0 but wrote %s). The next segment would have begun on "
            "nothing. NO FALLBACK."
            % (clip_path,
               "no file" if not os.path.exists(out_path) else "0 bytes"))
    return out_path


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
    "PIX_FMT", "COLOR_PRIMARIES",
    "WrapperNodeMissing", "GraphExecutionError",
    "node_class_mappings", "resolve_node_class", "resolve_graph_classes",
    "Wire", "run_graph",
    "reclaim_idle_models",
    "quantize_frames_4n1", "even_dim", "images_to_uint8",
    "extend_frames_to_target",
    "ffmpeg_silent_mp4_cmd", "ffmpeg_still_motion_cmd", "ffmpeg_lavfi_floor_cmd",
    "ffmpeg_terminal_frame_cmd", "extract_terminal_frame",
    "run_ffmpeg", "encode_frames_to_silent_mp4",
    "comfy_input_dir", "stage_into_comfy_input",
]
