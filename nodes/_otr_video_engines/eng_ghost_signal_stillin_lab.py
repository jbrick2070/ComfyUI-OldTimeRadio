"""``animatediff15_v3_stillin_lab_video`` -- the haunted v3 lane started from a STILL.

THE STILL-IN LAB PEER (campaign item 2, 2026-09-02; design arc in
``docs/2026-09-02-animatediff-ledger-experiments/still-in-peer/driver_anchor.md``,
sections 10 + 12 are the contract this file implements). A NEW engine id beside
the shipping ``animatediff15_v3_haunted_video``, subclassing it and touching
nothing in it. Same recipe (v3 motion module + adapter, 20 / 8.0 / euler /
normal, 512x288 hold-2, the static 16/4 pyramid, the live negative), one change:
the sampler starts from an IN-FAMILY PLATE instead of an empty latent.

THE PLATE. One 512x288 frame minted in the same graph from the same SD1.5
checkpoint with the adapter at 0.0 (the plain MODEL handle ``prepare`` already
owns), prompted with the style pack's FULL language plus the ledger's setting,
palette and lighting -- composed by ``render_driver`` onto the declared request
field ``plate_prompt`` (this engine has no ledger and never reads one). The plate
carries WORLD and MEDIUM only: no character, no name, no camera word. The
figure keeps coming from motif + leaf + law through the video prompt exactly as
the shipping lane does, so the beat's planned ``ghost_prompt`` is untouched.
Then the plate's LATENT is repeated in Python to the sampler batch (no cap,
no new node class) and the video sampler runs at ``denoise < 1.0`` (E1).

WHY THIS IS A PROBE, NOT A BUILD. The 2026-08-30 arc on this same lane
(``docs/SPEC_haunted_image_to_video.md``) found that a repeated init latent
plausibly suppresses motion by construction -- identical cross-frame keys in
the temporal attention -- so the lane exists to run the denoise grid
0.35 / 0.50 / 0.65 / 0.80 through the canonical graph, published to
``otr/obs/``, and be judged by eye against its own A/A null. It is a lab id:
never a default, never in a shipping profile, ``status: draft``.

RECEIPT LAW. ``sampler_inputs_for`` carries INPUTS only (the plate prompt, seed,
cells, the resolved denoise, the repeat count); the plate's rendered SHA-256 is
an OUTPUT and rides BESIDE the causal hash (``qc`` on the clip, the non-causal
block on the receipt). Two A/A replays therefore agree by construction, and a
GPU that is not bit-stable shows up as a plate-hash disagreement the verifier
names rather than as an unexplained video difference.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re

from . import eng_ghost_signal as _gs
from .eng_ghost_signal_official import GhostSignalV3HauntedEngine
from .registry import register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.video.ghost_signal_stillin_lab")

#: The denoise dial for the probe grid. Read INSIDE a method through a strict
#: parser: a malformed value is a NAMED refusal at ``assert_usable``, never a
#: silent default -- a silently-defaulted denoise would print a receipt that
#: lies about what conditioned the pixels.
STILLIN_LAB_DENOISE_ENV = "OTR_STILLIN_LAB_DENOISE"
STILLIN_LAB_DENOISE_DEFAULT = 0.65

# Node ids of the plate branch. The CLASSES are the parent's seven (no new
# candidate name anywhere); only INSTANCES are added.
NODE_PLATE_POSITIVE = "plate_positive"
NODE_PLATE_LATENT = "plate_latent"
NODE_PLATE_SAMPLER = "plate_sampler"
NODE_PLATE_DECODE = "plate_decode"
#: The external id the video sampler reads its init latent from. The parent's
#: ``latent`` node is DELETED from the sampler graph, never shadowed: the
#: executor refuses an external id that is also a graph node.
EXTERNAL_PLATE_INIT = "plate_init"

PLATE_SOURCE_MINTED = "minted"
PLATE_DIRNAME = "ghost_plates"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitise_for_filename(text: str) -> str:
    out = _SAFE_NAME.sub("_", str(text or "")).strip("._")
    return (out or "shot")[:80]


def repeat_latent(latent, amount: int):
    """The live ``RepeatLatentBatch`` rule, in Python, with NO 64 cap.

    Copies the LATENT dict, repeats ``samples`` along batch, repeats
    ``noise_mask`` when present, and extends ``batch_index`` by offset blocks --
    the body of ``ComfyUI/nodes.py::RepeatLatentBatch.repeat`` copied at build
    (2026-09-02), not a second rule. Pure over the dict; the caller owns the
    batch-1 original for the plate decode.
    """
    if not isinstance(latent, dict) or "samples" not in latent:
        raise RuntimeError(
            "stillin-lab: the plate sampler returned no LATENT dict with "
            "'samples' (got %s)" % type(latent).__name__)
    amount = int(amount)
    if amount < 1:
        raise RuntimeError("stillin-lab: repeat amount must be >= 1, got %d" % amount)
    samples = latent["samples"]
    batch = int(getattr(samples, "shape", (0,))[0] or 0)
    if batch != 1:
        raise RuntimeError(
            "stillin-lab: the plate latent must be batch 1, got %d" % batch)
    out = dict(latent)
    out["samples"] = samples.repeat((amount,) + ((1,) * (samples.ndim - 1)))
    mask = latent.get("noise_mask")
    if mask is not None and int(getattr(mask, "shape", (0,))[0] or 0) > 1:
        if mask.shape[0] < samples.shape[0]:
            mask = mask.repeat((math.ceil(samples.shape[0] / mask.shape[0]),)
                               + ((1,) * (mask.ndim - 1)))[:samples.shape[0]]
        out["noise_mask"] = mask.repeat((amount,) + ((1,) * (mask.ndim - 1)))
    if "batch_index" in latent:
        bi = list(latent["batch_index"])
        offset = max(bi) - min(bi) + 1
        out["batch_index"] = bi + [x + (i * offset)
                                   for i in range(1, amount) for x in bi]
    return out


@register
class GhostSignalV3StillInLabEngine(GhostSignalV3HauntedEngine):
    """``animatediff15_v3_stillin_lab_video`` -- v3 haunted, started from a plate."""

    name = "animatediff15_v3_stillin_lab_video"
    recipe_receipt_id = "animatediff_sd15_v3_haunted_stillin_e1_512x288_lab_v1"

    #: The capability ``render_driver`` branches on to compose ``plate_prompt``
    #: and ``plate_path`` onto the request. A capability, never an id compare.
    wants_plate_prompt = True

    #: The plate's own cells: the lane's, at the pristine image domain.
    PLATE_STEPS = _gs.GHOST_STEPS
    PLATE_CFG = _gs.GHOST_CFG
    PLATE_SAMPLER = _gs.GHOST_SAMPLER_NAME
    PLATE_SCHEDULER = _gs.GHOST_SCHEDULER
    PLATE_ADAPTER_STRENGTH = 0.0

    # ---- the dial ------------------------------------------------------ #
    def resolve_denoise(self) -> float:
        """The video sampler's denoise for THIS process, or a NAMED refusal."""
        raw = os.environ.get(STILLIN_LAB_DENOISE_ENV)
        if raw is None or not str(raw).strip():
            return float(STILLIN_LAB_DENOISE_DEFAULT)
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise _gs.EngineUnusable(
                self.name, self.family,
                _gs.EngineUsabilityReason.MALFORMED_CONFIG,
                "%s: %s=%r is not a finite number in [0, 1]. The dial is never "
                "defaulted silently -- a receipt must name the denoise that "
                "made the picture." % (self.name, STILLIN_LAB_DENOISE_ENV, raw),
                kind="video")
        return value

    def assert_usable(self, host_caps, profile, request_template=None):
        super().assert_usable(host_caps, profile, request_template)
        self.resolve_denoise()
        return self.name

    # ---- the plate identity ---------------------------------------------- #
    @staticmethod
    def _plate_prompt_of(request) -> str:
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        return str(get("plate_prompt") or "").strip()

    @staticmethod
    def _plate_dir_of(request) -> str:
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        return str(get("plate_path") or "").strip()

    def plate_identity(self, request, *, plan=None, plate_prompt=None):
        """``(dict, sha256)`` of every INPUT that decides the plate.

        The checkpoint enters as ``(name, size, mtime_ns)`` -- the same
        ``_file_receipt`` the lane's session identity uses -- never as a
        per-shot content digest of a 2 GB file; the content digest is already on
        every receipt once per process through ``model_artifacts``.
        """
        plan = plan or self._build_render_request(request)
        prompt = (plate_prompt if plate_prompt is not None
                  else self._plate_prompt_of(request))
        receipt = self._file_receipt(self._ckpt_path() or "")
        identity = {
            "checkpoint": {"name": _gs.GHOST_CHECKPOINT_NAME,
                           "receipt": list(receipt) if receipt else []},
            "plate_positive": prompt,
            "plate_negative": plan["negative_prompt"],
            "seed": int(plan["seed"]),
            "steps": int(self.PLATE_STEPS),
            "cfg": float(self.PLATE_CFG),
            "sampler": str(self.PLATE_SAMPLER),
            "scheduler": str(self.PLATE_SCHEDULER),
            "canvas": "%dx%d" % (_gs.GHOST_CANVAS_W, _gs.GHOST_CANVAS_H),
            "plate_adapter_strength": float(self.PLATE_ADAPTER_STRENGTH),
        }
        digest = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("utf-8")).hexdigest()
        return identity, digest

    # ---- receipts ---------------------------------------------------------- #
    def sampler_inputs_for(self, request):
        """The parent's cells plus the plate's INPUTS. Never an output hash."""
        plan = self._build_render_request(request)
        out = super().sampler_inputs_for(request)
        prompt = self._plate_prompt_of(request)
        out.update({
            "latent": "ghost_plate_init",
            "init_image": None,
            "denoise": float(self.resolve_denoise()),
            "plate_prompt": prompt,
            "plate_negative": plan["negative_prompt"],
            "plate_seed": int(plan["seed"]),
            "plate_steps": int(self.PLATE_STEPS),
            "plate_cfg": float(self.PLATE_CFG),
            "plate_sampler": str(self.PLATE_SAMPLER),
            "plate_scheduler": str(self.PLATE_SCHEDULER),
            "plate_canvas": "%dx%d" % (_gs.GHOST_CANVAS_W, _gs.GHOST_CANVAS_H),
            "plate_adapter_strength": float(self.PLATE_ADAPTER_STRENGTH),
            "init_repeat_method": "torch_repeat",
            "init_repeat_count": int(plan["source_request"]),
        })
        return out

    def shot_cache_identity(self, request):
        _identity, digest = self.plate_identity(request)
        return super().shot_cache_identity(request) + (
            digest[:16], "denoise=%.4f" % float(self.resolve_denoise()))

    # ---- render ------------------------------------------------------------ #
    def render_clip(self, request, prepared):
        """One beat, from a plate: encode (3), plate (2), sample (4), decode (2).

        Eleven render-time node instances on the parent's seven classes. The
        cross-stage release law is the parent's: explicit owner clearing plus
        the public reclaim seam after every bounded executor call; a ``finally``
        drops every plate-side reference on every exit.
        """
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4

        plan = self._build_render_request(request)
        self._assert_required_inputs(plan)
        plate_prompt = self._plate_prompt_of(request)
        if not plate_prompt:
            raise RuntimeError(
                "%s requires a composed plate_prompt on the request and it "
                "carries none -- the plate is this lane's whole point, and an "
                "empty one would render the shipping lane under a lab receipt"
                % self.name)
        denoise = float(self.resolve_denoise())
        _identity, identity_sha = self.plate_identity(
            request, plan=plan, plate_prompt=plate_prompt)
        classes = self._classes or _wb.resolve_graph_classes(
            self._node_candidates())
        owners = prepared if isinstance(prepared, dict) else {}
        repeat_count = int(plan["source_request"])

        positive_cond = negative_cond = plate_cond = None
        plate_latent = None
        init = None
        plate_frames = None
        exec_plate: list = []          # fresh per audited call (executor law)
        exec_plate_decode: list = []
        try:
            # ---- STAGE 2: ENCODE (positive, negative, plate) ------------- #
            encode_graph = {
                _gs.NODE_POSITIVE: {
                    "class": classes["text_encode"],
                    "inputs": {"text": plan["text_prompt"],
                               "clip": _wb.Wire("clip", 0)}},
                _gs.NODE_NEGATIVE: {
                    "class": classes["text_encode"],
                    "inputs": {"text": plan["negative_prompt"],
                               "clip": _wb.Wire("clip", 0)}},
                NODE_PLATE_POSITIVE: {
                    "class": classes["text_encode"],
                    "inputs": {"text": plate_prompt,
                               "clip": _wb.Wire("clip", 0)}},
            }
            encode_results = _wb.run_graph(
                encode_graph, external_results={"clip": owners["clip"]})
            positive_cond = (encode_results[_gs.NODE_POSITIVE][0],)
            negative_cond = (encode_results[_gs.NODE_NEGATIVE][0],)
            plate_cond = (encode_results[NODE_PLATE_POSITIVE][0],)
            encode_results.clear()
            del encode_graph
            owners.pop("clip", None)
            _wb.reclaim_idle_models("stillin-lab post-encode")

            # ---- STAGE 3a: THE PLATE, on the plain checkpoint MODEL ------ #
            plate_graph = {
                NODE_PLATE_LATENT: {
                    "class": classes["latent"],
                    "inputs": {"width": _gs.GHOST_CANVAS_W,
                               "height": _gs.GHOST_CANVAS_H,
                               "batch_size": 1}},
                NODE_PLATE_SAMPLER: {
                    "class": classes["sampler"],
                    "inputs": {
                        "model": _wb.Wire("base_model", 0),
                        "seed": plan["seed"],
                        "steps": int(self.PLATE_STEPS),
                        "cfg": float(self.PLATE_CFG),
                        "sampler_name": str(self.PLATE_SAMPLER),
                        "scheduler": str(self.PLATE_SCHEDULER),
                        "positive": _wb.Wire("plate_cond", 0),
                        "negative": _wb.Wire("negative_cond", 0),
                        "latent_image": _wb.Wire(NODE_PLATE_LATENT, 0),
                        "denoise": 1.0,
                    }},
            }
            plate_out = _wb.run_graph(
                plate_graph,
                external_results={"base_model": owners["base_model"],
                                  "plate_cond": plate_cond,
                                  "negative_cond": negative_cond},
                terminal=NODE_PLATE_SAMPLER,
                audit_node_ids={NODE_PLATE_SAMPLER},
                execution_records=exec_plate)
            plate_latent = plate_out[0]          # batch 1, KEPT for the decode
            init = repeat_latent(plate_latent, repeat_count)
            plate_cond = None
            del plate_graph, plate_out
            _wb.reclaim_idle_models("stillin-lab post-plate")

            # ---- STAGE 3b: SAMPLE, from the repeated plate --------------- #
            sample_graph = {
                _gs.NODE_CONTEXT: {
                    "class": classes["context"],
                    "inputs": {
                        "context_length": _gs.GHOST_CONTEXT_LENGTH,
                        "context_overlap": _gs.GHOST_CONTEXT_OVERLAP,
                        "fuse_method": _gs.GHOST_CONTEXT_FUSE_METHOD,
                        "use_on_equal_length": _gs.GHOST_CONTEXT_USE_ON_EQUAL_LENGTH,
                        "start_percent": _gs.GHOST_CONTEXT_START_PERCENT,
                        "guarantee_steps": _gs.GHOST_CONTEXT_GUARANTEE_STEPS,
                    }},
                _gs.NODE_ADE: {
                    "class": classes["ade"],
                    "inputs": {
                        "model": _wb.Wire("base_model", 0),
                        "model_name": self.motion_module_name,
                        "beta_schedule": _gs.GHOST_BETA_SCHEDULE,
                        "context_options": _wb.Wire(_gs.NODE_CONTEXT, 0),
                    }},
                # NO `latent` node: the init latent is the external below.
                _gs.NODE_SAMPLER: {
                    "class": classes["sampler"],
                    "inputs": {
                        "model": _wb.Wire(_gs.NODE_ADE, 0),
                        "seed": plan["seed"],
                        "steps": _gs.GHOST_STEPS,
                        "cfg": _gs.GHOST_CFG,
                        "sampler_name": _gs.GHOST_SAMPLER_NAME,
                        "scheduler": _gs.GHOST_SCHEDULER,
                        "positive": _wb.Wire("positive_cond", 0),
                        "negative": _wb.Wire("negative_cond", 0),
                        "latent_image": _wb.Wire(EXTERNAL_PLATE_INIT, 0),
                        "denoise": denoise,
                    }},
            }
            if self.lora_name:
                sample_graph[_gs.NODE_LORA] = {
                    "class": classes["lora"],
                    "inputs": {
                        "model": _wb.Wire("base_model", 0),
                        "lora_name": self.lora_name,
                        "strength_model": float(self.lora_strength),
                    }}
                sample_graph[_gs.NODE_ADE]["inputs"]["model"] = _wb.Wire(
                    _gs.NODE_LORA, 0)

            def _register_patched(node_id, out_tuple):
                if node_id == _gs.NODE_LORA:
                    owner_key = "lora_model"
                elif node_id == _gs.NODE_ADE:
                    owner_key = "ade_model"
                else:
                    return
                if not out_tuple:
                    raise RuntimeError(
                        "%s: the %s loader returned no MODEL" % (self.name, node_id))
                patched = out_tuple[0]
                if not callable(getattr(patched, "detach", None)):
                    raise RuntimeError(
                        "%s: the patched %s MODEL has no callable detach() -- "
                        "this adapter cannot take ownership of a patcher it "
                        "cannot release" % (self.name, node_id))
                owners[owner_key] = patched
                self._patchers.append(patched)

            sampled = _wb.run_graph(
                sample_graph,
                external_results={"base_model": owners["base_model"],
                                  "positive_cond": positive_cond,
                                  "negative_cond": negative_cond,
                                  EXTERNAL_PLATE_INIT: (init,)},
                terminal=_gs.NODE_SAMPLER, on_result=_register_patched)
            sampled_latent = (sampled[0],)
            self._release_sampling_patchers_before_decode(owners)
            del sample_graph, sampled
            positive_cond = None
            negative_cond = None
            init = None
            _wb.reclaim_idle_models("stillin-lab post-sample")

            # ---- STAGE 4: DECODE the beat, then the plate ---------------- #
            images = _wb.run_graph(
                {_gs.NODE_DECODE: {
                    "class": classes["decode"],
                    "inputs": {"samples": _wb.Wire("sampled_latent", 0),
                               "vae": _wb.Wire("vae", 0)}}},
                external_results={"sampled_latent": sampled_latent,
                                  "vae": owners["vae"]},
                terminal=_gs.NODE_DECODE)[0]
            frames = _wb.images_to_uint8(images)
            decoded = int(frames.shape[0])
            if decoded != plan["source_request"]:
                raise RuntimeError(
                    "%s decoded %d frame(s) but asked for exactly %d -- a count "
                    "mismatch is a graph fault, and this lane pads nothing"
                    % (self.name, decoded, plan["source_request"]))
            del images, sampled_latent

            plate_images = _wb.run_graph(
                {NODE_PLATE_DECODE: {
                    "class": classes["decode"],
                    "inputs": {"samples": _wb.Wire("plate_kept", 0),
                               "vae": _wb.Wire("vae", 0)}}},
                external_results={"plate_kept": (plate_latent,),
                                  "vae": owners["vae"]},
                terminal=NODE_PLATE_DECODE,
                audit_node_ids={NODE_PLATE_DECODE},
                execution_records=exec_plate_decode)[0]
            plate_frames = _wb.images_to_uint8(plate_images)
            if int(plate_frames.shape[0]) < 1:
                raise RuntimeError("%s: the plate decoded to no frame" % self.name)
            plate_latent = None
            del plate_images
            _wb.reclaim_idle_models("stillin-lab post-plate-decode")

            # ---- CADENCE + DELIVERY (the parent's, verbatim) -------------- #
            selector = _gs.ghost_hold_selector(plan["target_frame_count"],
                                               self.hold_factor)
            unique = plan["unique_source_count"]
            delivered = frames[[min(i, unique - 1) for i in selector]]
            out_path = otr_engine_tmp_mp4("ghost_signal_stillin_lab")
            out_path, proven = _wb.encode_frames_to_silent_mp4(
                delivered, out_path, int(self.target_fps))
            if int(proven) != int(plan["target_frame_count"]):
                raise RuntimeError(
                    "%s encoded %s frame(s) where the beat's delivered target "
                    "is %d" % (self.name, proven, plan["target_frame_count"]))

            # ---- THE PLATE ON DISK, for the record and the eye ------------ #
            plate_name, plate_sha, plate_file = self._write_plate_png(
                plate_frames[0], request, plan, identity_sha,
                fallback_dir=os.path.dirname(out_path))

            raw = {
                "out_path": out_path,
                "frame_count": int(proven),
                "recipe": self._recipe_receipt(),
                "domain_adapter": self.lora_name,
                "domain_adapter_strength": (
                    float(self.lora_strength) if self.lora_name else None),
                "render_canvas": "%dx%d" % (_gs.GHOST_CANVAS_W, _gs.GHOST_CANVAS_H),
                "vram_peak_mb": None,
                # THE PLATE RECORD -- outputs, beside the causal hash.
                "plate_sha256": plate_sha,
                "plate_name": plate_name,
                "plate_path": plate_file,
                "plate_source": PLATE_SOURCE_MINTED,
                "plate_identity_sha256": identity_sha,
                "plate_denoise": denoise,
                "graph_exec": list(exec_plate) + list(exec_plate_decode),
            }
            raw.update(_gs._ghost_cadence_receipts_for(
                self, plan["target_frame_count"], plan["source_request"]))
            _LOG.info(
                "[OTR video] %s beat %s: T=%d U=%d requested=%d decoded=%d "
                "denoise=%.3f plate=%s tail_trim=%d @ %dx%d",
                self.name, plan["shot_id"], plan["target_frame_count"], unique,
                plan["source_request"], decoded, denoise, plate_sha[:12],
                raw["cadence_tail_trim"], _gs.GHOST_CANVAS_W, _gs.GHOST_CANVAS_H)
            if self.lora_name:
                applied = float(self.lora_strength)
                _LOG.info("%s: domain adapter %s at strength %.4f",
                          self.name, self.lora_name, applied)
                if applied == 0.0:
                    _LOG.warning(
                        "%s: strength is 0.0, so ComfyUI returns the base model "
                        "UNPATCHED -- this beat is the clean picture under a "
                        "haunted lab receipt; if that was not meant, %s was not "
                        "read.", self.name, "OTR_GHOST_HAUNTED_LORA_STRENGTH")
            return raw
        finally:
            # Every plate-side reference goes on every exit, so a failed write
            # or a raising sampler cannot retain a beat-sized CUDA tensor.
            plate_cond = None
            plate_latent = None
            init = None
            plate_frames = None

    def _write_plate_png(self, frame, request, plan, identity_sha, *, fallback_dir):
        """Write the plate PNG via a temp sibling + ``os.replace``; return
        ``(name, sha256, path)``. The directory is the request's ``plate_path``
        (filled by the driver) or, for a request without one, the lane's tmp dir
        beside the clip -- the file is always written, because the record is the
        point."""
        from PIL import Image  # lazy: cold-import law

        target_dir = self._plate_dir_of(request) or fallback_dir
        os.makedirs(target_dir, exist_ok=True)
        name = "%s_%s.png" % (_sanitise_for_filename(plan["shot_id"]),
                              identity_sha[:16])
        path = os.path.join(target_dir, name)
        tmp = path + ".tmp"
        Image.fromarray(frame).save(tmp, format="PNG")
        os.replace(tmp, path)
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        return name, sha.hexdigest(), path

    # ---- canonicalize ------------------------------------------------------ #
    def canonicalize(self, raw, request, profile):
        """The parent's proof, restated here on purpose: the module that CALLS
        the encoder must itself call ``validate_silent_clip_contract`` (the
        terminal-frame gate) and the module that DEFINES ``canonicalize`` must
        name it (G5.1). ONE probe, both checks, the exact-512x288 refusal that
        earns the composite's clean full-frame Lanczos."""
        raw = raw or {}
        path = raw.get("out_path", "")
        if path:
            fields = ffprobe_clip_fields(path)
            validate_silent_clip_contract(fields, self.target_fps)
            width = int(fields.get("width") or 0)
            height = int(fields.get("height") or 0)
            if width != _gs.GHOST_CANVAS_W or height != _gs.GHOST_CANVAS_H:
                raise RuntimeError(
                    "%s produced a %dx%d clip; this lane delivers by clean "
                    "full-frame Lanczos with no pad or crop, so only an exact "
                    "%dx%d source can be enlarged without distorting it"
                    % (self.name, width, height, _gs.GHOST_CANVAS_W,
                       _gs.GHOST_CANVAS_H))
        clip = self._clip_from_raw(raw, request)
        for key in ("model_frame_count", "cadence_mode",
                    "cadence_source_frame_count",
                    "cadence_delivered_frame_count", "cadence_tail_trim",
                    "delivery_scale_mode"):
            if key in raw:
                clip[key] = raw[key]
        return clip

    def _clip_from_raw(self, raw, request):
        clip = super()._clip_from_raw(raw, request)
        raw = raw or {}
        clip["qc"] = {
            "plate_sha256": raw.get("plate_sha256"),
            "plate_name": raw.get("plate_name"),
            "plate_path": raw.get("plate_path"),
            "plate_source": raw.get("plate_source"),
            "plate_identity_sha256": raw.get("plate_identity_sha256"),
            "plate_denoise": raw.get("plate_denoise"),
            "graph_exec": list(raw.get("graph_exec") or []),
        }
        return clip
