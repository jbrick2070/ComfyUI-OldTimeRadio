"""
run_video_arm_bakeoff.py
========================

CLAMPED MULTI-ARM VIDEO BENCH. Answers exactly one question -- "which video
engine can render a shippable clip at a canvas we would actually ship, at every
length an 8 GB tier needs, under an 8 GiB loader clamp, on a licence we can
ship" -- and, explicitly, nothing else.

Spec: ``docs/2026-07-31-four-arm-clamped-video-bench-SPEC.md``.
Judgment of record: ``docs/2026-07-31-wan-8gb-adversarial-review/report.md``.

BENCH != QUALIFICATION (spec s2). This measures the VIDEO STAGE IN ISOLATION on
a 16 GB card under a loader clamp. It cannot reproduce an 8 GB address space, a
real WDDM/display baseline, fragmentation, PCIe pressure or another GPU
architecture; it does not prove teardown; it may not lower the admission guard,
refit ``FRAME_COST_MODEL`` or promote 14B. Every result carries the mandatory
label ``16 GB card, 8 GiB-reserve direct-node prequalification`` -- NEVER
"8 GB qualified".

Why a fork of ``run_wan_ti2v_bakeoff.py`` and not a second harness: that runner
already IS a clamped multi-leg bench (boot-per-leg through the proven launcher,
selective-CIM reset, background NVML sampler, incremental results). Its arm axis
was a quant FILENAME; here it becomes an ``ArmSpec``, because a Wan->LTX arm is
node SUBSTITUTION, which the old ``mutate()`` docstring forbids.

Each arm carries its OWN pinned graph under ``scripts/bench_graphs/`` rather than
sharing one mutated graph. Submitting a stock-node API graph is admitted by the
CLAUDE.md s0A isolated-bench carve-out (operator decision O6, 2026-07-31) -- a
narrow, measurement-only exemption from "every API run must load the real JSON".

THE ESTIMATOR IS NOT IN THIS PATH. Driving stock nodes directly bypasses the OTR
engine adapters, so ``motion_common``'s admission estimator never runs and the
``OTR_VIDEO_COST_*`` / ``OTR_VIDEO_BUDGET_MARGIN`` seams cannot refuse a cell.
They are recorded in every receipt and flagged INERT so no reader mistakes a
bypassed predictor for a working one. Refitting it is out of scope (spec s9.1).

  python run_video_arm_bakeoff.py --dry-validate   # no-GPU preflight, renders nothing
  python run_video_arm_bakeoff.py --arms A,B-partial

Launch DETACHED and poll the log (the ~60s MCP ceiling). Reset before EVERY run
per CLAUDE.md s4 -- this harness asserts it rather than assuming it.

UTF-8, no BOM. ASCII-only. SFW.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------
# paths + campaign constants
# --------------------------------------------------------------------------
COMFY_ROOT = r"C:\Users\jeffr\Documents\ComfyUI"
SERVER_INSTALL = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI"
SRC_INPUT = os.path.join(COMFY_ROOT, "input")
SERVER_INPUT = os.path.join(SERVER_INSTALL, "input")
GRAPH_DIR = os.path.join(REPO, "scripts", "bench_graphs")
SOAK_LAUNCH_CMD = os.path.join(REPO, "scripts", "_otr_soak_server_launch.cmd")

#: The vendored measurement package (spec G3). ``custom_nodes\otr_bakeoff_helper``
#: is in NO git repository, so a pushed harness may not depend on it in place:
#: this tracked copy is the source of truth and is INSTALLED (byte-identically)
#: before boot.
HELPER_SRC = os.path.join(REPO, "scripts", "bench_helper", "otr_bakeoff_helper")
HELPER_DST = os.path.join(COMFY_ROOT, "custom_nodes", "otr_bakeoff_helper")

FIXED_ASSETS = ("c02_466a19906ccb.png",)

#: ONE base URL owns the port (spec G4/D14). The launcher honours
#: OTR_HEADLESS_PORT while the old harness hard-coded 8000; they are now the
#: same number by construction.
BASE_URL = (os.environ.get("COMFYUI_URL") or "http://127.0.0.1:8000").rstrip("/")
PORT = urlsplit(BASE_URL).port or 8000

#: ONE absolute output root, propagated to launcher, submit and validation.
OUTPUT_BASE = os.path.abspath(
    os.environ.get("COMFYUI_OUTPUT") or os.path.join(COMFY_ROOT, "output"))
BENCH_DIR = os.path.join(OUTPUT_BASE, "otr", "episodes", "_bench_4arm")

#: 8 GiB is the SOLE campaign clamp (spec 7.3). ``--reserve-vram`` REPLACES the
#: Windows default rather than adding to it, and it allocates nothing: it moves
#: ComfyUI's ``maximum_vram_for_weights``, which is the loader behaviour we are
#: trying to observe. It is NOT a physical partition and is NOT visible to OTR's
#: admission check.
CAMPAIGN_CLAMP_GIB = 8

#: The greenlight bar (spec s12). ``peak_delta_mib + display_allowance <= 8192``.
#: The 1024 MiB allowance is a DECLARED CONSERVATIVE ASSUMPTION, not a
#: measurement (operator decision O5) -- if it is wrong every verdict shifts by
#: the difference.
DISPLAY_ALLOWANCE_MIB = 1024
GREENLIGHT_PEAK_DELTA_MIB = 8192 - DISPLAY_ALLOWANCE_MIB   # 7168

#: The shared frame ladder. Legal on BOTH frame contracts (Wan: min 17 quantum 4;
#: LTX 8GB: min 9 quantum 8), so no arm is special-cased. Judge on the SPREAD:
#: an arm that only passes at 17 is a demo, not a tier.
CAMPAIGN_LENGTHS = (17, 49, 81)

#: Recorded in every receipt and flagged INERT -- see the module docstring.
ESTIMATOR_SEAMS = ("OTR_VIDEO_COST_OVERHEAD_MB", "OTR_VIDEO_COST_PER_FRAME_MB",
                   "OTR_VIDEO_BUDGET_MARGIN")

#: The values the server is booted with. ``per_frame = 0`` makes
#: ``motion_common.compute_real_frame_budget`` skip its refusal branch entirely
#: (``if per_frame_at_res > 0``), so nothing anywhere can refuse a leg on the
#: coefficients we already know are poisoned.
NEUTRALIZED_ESTIMATOR = {
    "OTR_VIDEO_COST_OVERHEAD_MB": "0",
    "OTR_VIDEO_COST_PER_FRAME_MB": "0",
    "OTR_VIDEO_BUDGET_MARGIN": "1.0",
}

RESULT_LABEL = "16 GB card, 8 GiB-reserve direct-node prequalification"

#: NVML must settle to within this of the recorded desktop baseline before a
#: boot. Above it the box is not quiescent and the cell FAILS -- booting onto
#: residue measures garbage (CLAUDE.md s4 is a hard gate).
QUIESCENCE_TOLERANCE_MIB = 512


# --------------------------------------------------------------------------
# arm specification -- the axis this bench exists for
# --------------------------------------------------------------------------
class BenchError(RuntimeError):
    """A named bench failure. THE LAW: anything that cannot honour a request
    RAISES -- no fallback, no silent degrade, no substituted model."""


#: Frame contracts, mirrored from the engines so an ArmSpec validates its own
#: ladder instead of discovering an illegal rung at submit time.
#: ``eng_wan_ti2v.py:259-266`` / ``eng_ltx_8gb.py:534-541``.
FRAME_CONTRACTS = {
    "wan_ti2v": {"min": 17, "quantum": 4, "max": 177},
    "ltx_8gb": {"min": 9, "quantum": 8, "max": 161},
}


class ModelPin(object):
    """One loader node's model, and the dropdown it must be visible in.

    Lifted from ``otr_wan_smoke.assert_model_visible``: a cell FAILS before
    ``/prompt`` if the file is not in the LIVE ``/object_info`` enum. A missing
    enum is a hard campaign-incomplete condition -- never a silent substitution
    and never a shrunk denominator."""

    __slots__ = ("node_id", "loader_class", "input_name", "filename", "role")

    def __init__(self, node_id, loader_class, input_name, filename, role):
        self.node_id = str(node_id)
        self.loader_class = loader_class
        self.input_name = input_name
        self.filename = filename
        self.role = role

    def as_dict(self):
        return {"node_id": self.node_id, "loader_class": self.loader_class,
                "input_name": self.input_name, "filename": self.filename,
                "role": self.role}


class ArmSpec(object):
    """One bench arm. Validated at construction -- an invalid arm may not reach
    a GPU."""

    def __init__(self, label, engine_id, graph, graph_sha256, steps, canvas,
                 lengths, required_classes, models, length_node, save_node,
                 notes, gated_reason=None):
        self.label = label
        self.engine_id = engine_id
        self.graph = graph
        self.graph_sha256 = (graph_sha256 or "").lower()
        self.steps = steps
        self.canvas = tuple(canvas)
        self.lengths = tuple(lengths)
        self.required_classes = tuple(required_classes)
        self.models = tuple(models)
        self.length_node = str(length_node)
        self.save_node = str(save_node)
        self.notes = notes
        self.gated_reason = gated_reason
        self.validate()

    # -- validation ------------------------------------------------------
    def validate(self):
        if not self.label or re.search(r"[^A-Za-z0-9_.-]", self.label):
            raise BenchError("arm label %r must be non-empty and filesystem-safe"
                             % (self.label,))
        if self.engine_id not in FRAME_CONTRACTS:
            raise BenchError("arm %s: unknown engine_id %r (known: %s)"
                             % (self.label, self.engine_id,
                                ", ".join(sorted(FRAME_CONTRACTS))))
        if not re.fullmatch(r"[0-9a-f]{64}", self.graph_sha256 or ""):
            raise BenchError("arm %s: graph_sha256 must be 64 hex chars, got %r"
                             % (self.label, self.graph_sha256))
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise BenchError("arm %s: steps must be a positive int, got %r"
                             % (self.label, self.steps))
        w, h = self.canvas
        if w % 32 or h % 32:
            raise BenchError("arm %s: canvas %dx%d violates OTR's both-axes/32 "
                             "rule" % (self.label, w, h))
        if not self.lengths:
            raise BenchError("arm %s: no lengths" % self.label)
        fc = FRAME_CONTRACTS[self.engine_id]
        for n in self.lengths:
            if n < fc["min"] or n > fc["max"] or (n - fc["min"]) % fc["quantum"]:
                raise BenchError(
                    "arm %s: length %d is illegal for %s (min %d, max %d, "
                    "quantum %d)" % (self.label, n, self.engine_id, fc["min"],
                                     fc["max"], fc["quantum"]))
        if not self.required_classes:
            raise BenchError("arm %s: no required node classes" % self.label)
        if not self.models:
            raise BenchError("arm %s: no model pins" % self.label)
        seen = set()
        for m in self.models:
            if m.role in seen:
                raise BenchError("arm %s: duplicate model role %r"
                                 % (self.label, m.role))
            seen.add(m.role)

    # -- helpers ---------------------------------------------------------
    @property
    def graph_path(self):
        return os.path.join(GRAPH_DIR, self.graph)

    def cell_id(self, length, repeat=0):
        """Unique per (arm, length, repeat). Repeats get their OWN id so a
        winner's three cold measurements can never overwrite each other."""
        base = "%s__%df" % (self.label, int(length))
        return base if repeat == 0 else "%s__r%d" % (base, int(repeat))

    def as_dict(self):
        return {"label": self.label, "engine_id": self.engine_id,
                "graph": self.graph, "graph_sha256": self.graph_sha256,
                "steps": self.steps, "canvas": list(self.canvas),
                "lengths": list(self.lengths),
                "required_classes": list(self.required_classes),
                "models": [m.as_dict() for m in self.models],
                "length_node": self.length_node, "save_node": self.save_node,
                "notes": self.notes, "gated_reason": self.gated_reason}


_WAN_CLASSES = ("ModelSamplingSD3", "CLIPTextEncode", "VAELoader", "LoadImage",
                "Wan22ImageToVideoLatent", "KSampler", "VAEDecodeTiled",
                "CreateVideo", "SaveVideo", "OTR_BakeoffVramReset",
                "OTR_BakeoffVramProbe")

#: THE ARMS. Arm C (FastWan 5B) is CUT and arm C has no entry BY DESIGN: a
#: blocked arm must not add dead branching. Its licence PASSES (apache-2.0 at
#: both levels) but no ComfyUI base graph exists at any priority -- the weights
#: are Diffusers layout and the 3-step DMD schedule (timesteps 1000,757,522) is
#: not ordinary ``KSampler steps=3``. Re-open it as a separate research probe
#: behind a conversion/loader receipt (spec 6.4, operator decision O2).
ARMS = (
    ArmSpec(
        label="A", engine_id="wan_ti2v",
        graph="arm_a_wan_ti2v_gguf.json",
        graph_sha256="254319962e922ca87bde40a45fdf2f103bdba0541b95566c32991be415c2bea7",
        steps=30, canvas=(832, 480), lengths=CAMPAIGN_LENGTHS,
        required_classes=("UnetLoaderGGUF", "CLIPLoaderGGUF") + _WAN_CLASSES,
        models=(ModelPin("1", "UnetLoaderGGUF", "unet_name",
                         "Wan2.2-TI2V-5B-Q5_K_M.gguf", "unet"),
                ModelPin("3", "CLIPLoaderGGUF", "clip_name",
                         "umt5-xxl-encoder-Q5_K_M.gguf", "clip"),
                ModelPin("6", "VAELoader", "vae_name",
                         "wan2.2_vae.safetensors", "vae")),
        length_node="8", save_node="12",
        notes="INCUMBENT BASELINE. Q5_K_M GGUF UNet + umt5 GGUF encoder. Stock "
              "ComfyUI-GGUF forces both patchers back to legacy ModelPatcher, "
              "so neither gets AIMDO/VBAR ModelPatcherDynamic -- that is the "
              "mechanism under test, not a verdict."),
    ArmSpec(
        label="B-partial", engine_id="wan_ti2v",
        graph="arm_b_partial_wan_ti2v_gguf_unet_st_clip.json",
        graph_sha256="3f86eb466bbb397c2758bf3a9e6473f489423e3eab4a5722ecfe7ad4a920d0b1",
        steps=30, canvas=(832, 480), lengths=CAMPAIGN_LENGTHS,
        required_classes=("UnetLoaderGGUF", "CLIPLoader") + _WAN_CLASSES,
        models=(ModelPin("1", "UnetLoaderGGUF", "unet_name",
                         "Wan2.2-TI2V-5B-Q5_K_M.gguf", "unet"),
                ModelPin("3", "CLIPLoader", "clip_name",
                         "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "clip"),
                ModelPin("6", "VAELoader", "vae_name",
                         "wan2.2_vae.safetensors", "vae")),
        length_node="8", save_node="12",
        notes="PARTIAL, AND LABELLED SO. Swaps ONLY the encoder, because the "
              "native FP16 TI2V-5B UNet is not on disk. It is the ONLY cell "
              "that isolates the encoder bundle, which is why it is mandatory "
              "rather than a fallback. Production can reach this exact "
              "configuration through eng_wan_ti2v._clip_loader_mode "
              "(OTR_WAN_TI2V_CLIP_LOADER=safetensors), so a win is actionable."),
    ArmSpec(
        label="B", engine_id="wan_ti2v",
        graph="arm_b_wan_ti2v_native.json",
        graph_sha256="f2bc210cf0310e8914b68bfdb782eafb8228fab03f116aa20a308d611dfc205a",
        steps=30, canvas=(832, 480), lengths=CAMPAIGN_LENGTHS,
        required_classes=("UNETLoader", "CLIPLoader") + _WAN_CLASSES,
        models=(ModelPin("1", "UNETLoader", "unet_name",
                         "wan2.2_ti2v_5B_fp16.safetensors", "unet"),
                ModelPin("3", "CLIPLoader", "clip_name",
                         "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "clip"),
                ModelPin("6", "VAELoader", "vae_name",
                         "wan2.2_vae.safetensors", "vae")),
        length_node="8", save_node="12",
        notes="THE PROTECTED ARM -- ComfyUI's own stated 8 GB path (FP16 UNet + "
              "scaled-FP8 encoder, NOT fp8 throughout). A/B against B-partial "
              "isolates the UNet format with the encoder held constant.",
        gated_reason="wan2.2_ti2v_5B_fp16.safetensors is not on disk (spec G2 / "
                     "operator decision O3): acquire from "
                     "Comfy-Org/Wan_2.2_ComfyUI_Repackaged with an immutable "
                     "revision, licence, SHA-256, byte size and live UNETLoader "
                     "dropdown visibility before this arm may render."),
    ArmSpec(
        label="D", engine_id="ltx_8gb",
        graph="arm_d_ltx_8gb.json",
        graph_sha256="55cd4a60c45c9966a13d36a123d2ceddaeca65c3d750b9bc8d29cb0bdb1f2db0",
        steps=8, canvas=(512, 288), lengths=CAMPAIGN_LENGTHS,
        required_classes=("CheckpointLoaderSimple", "CLIPLoader",
                          "CLIPTextEncode", "LoadImage", "ModelSamplingLTXV",
                          "LTXVImgToVideo", "LTXVConditioning", "LTXVScheduler",
                          "KSamplerSelect", "SamplerCustom", "VAEDecodeTiled",
                          "CreateVideo", "SaveVideo", "OTR_BakeoffVramReset",
                          "OTR_BakeoffVramProbe"),
        models=(ModelPin("1", "CheckpointLoaderSimple", "ckpt_name",
                         "ltxv-2b-0.9.8-distilled.safetensors", "ckpt"),
                ModelPin("3", "CLIPLoader", "clip_name",
                         "t5xxl_fp16.safetensors", "clip")),
        length_node="8", save_node="12",
        notes="RUNS AT ITS OWN SHIPPED CANVAS, 512x288 (eng_ltx_8gb:518), on "
              "its own shipped recipe LTX8_RECIPE_V2 (8 steps, cfg 1.0, "
              "t5_device=cpu, tiled VAE 512/64/16/8). This is deliberately NOT "
              "settings-matched to the Wan arms and is not comparable to them: "
              "different canvas, different sampler, different step count, and a "
              "T5 placement lever Wan has no knob for. The question each arm "
              "answers is only 'does this engine run on a small GPU, and how "
              "much VRAM does it eat' -- read the arms as three separate "
              "answers, never as a ranking. Licence: LTX-Video Open Weights "
              "0.X, revenue-capped -- gates SHIPPING, not measuring (O4)."),
)

ARMS_BY_LABEL = {a.label: a for a in ARMS}


# --------------------------------------------------------------------------
# graph load, pin verification, and node substitution
# --------------------------------------------------------------------------
def log(msg):
    print("[video-arm-bench] %s" % msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_graph(arm, verify_pin=True):
    """Load an arm's API-prompt graph and verify its SHA-256 pin.

    A graph whose digest does not match its pin FAILS the cell -- cross-arm
    comparison is void if the graph moved under us (CLAUDE.md s0A)."""
    path = arm.graph_path
    if not os.path.exists(path):
        raise BenchError("arm %s: graph missing: %s" % (arm.label, path))
    if verify_pin:
        actual = sha256_file(path)
        if actual != arm.graph_sha256:
            raise BenchError(
                "arm %s: graph SHA-256 mismatch -- pinned %s, on disk %s (%s). "
                "Re-pin deliberately or restore the graph; do NOT run a cell "
                "against an unpinned graph."
                % (arm.label, arm.graph_sha256, actual, arm.graph))
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise BenchError("arm %s: graph is not an object" % arm.label)
    if isinstance(raw.get("nodes"), list):
        raise BenchError(
            "arm %s: %s is LiteGraph UI format (top-level 'nodes'), not an API "
            "prompt. Bench graphs are API format on purpose -- LiteGraph carries "
            "positional widgets_values drift (BUG-LOCAL-097)."
            % (arm.label, arm.graph))
    prompt = {k: v for k, v in raw.items()
              if not k.startswith("_") and isinstance(v, dict)
              and "class_type" in v}
    if not prompt:
        raise BenchError("arm %s: graph has no executable nodes" % arm.label)
    return prompt


class Substitution(object):
    """A node-class substitution: change ``class_type`` AND reconcile inputs.

    The old harness could only swap ``unet_name`` on a hardwired
    ``UnetLoaderGGUF`` -- a safetensors UNet returned ``None`` and raised
    ``TypeError``, and there was no CLIP hook at all. Arms differ by node CLASS,
    so a substitution must ADD the new class's required inputs and REMOVE the
    old class's stale ones. Concretely: ``UNETLoader`` needs ``weight_dtype``
    and ``UnetLoaderGGUF`` must not carry it; core ``CLIPLoader`` takes
    ``device`` and ``CLIPLoaderGGUF`` takes none (verified vs /object_info,
    mirrored at eng_wan_ti2v.py:841-847)."""

    __slots__ = ("node_id", "class_type", "set_inputs", "drop_inputs")

    def __init__(self, node_id, class_type, set_inputs=None, drop_inputs=()):
        self.node_id = str(node_id)
        self.class_type = class_type
        self.set_inputs = dict(set_inputs or {})
        self.drop_inputs = tuple(drop_inputs)


def apply_substitution(prompt, sub):
    """Return a COPY of ``prompt`` with ``sub`` applied. Never mutates in place."""
    p = copy.deepcopy(prompt)
    node = p.get(sub.node_id)
    if node is None:
        raise BenchError("substitution target node %r not in graph (have: %s)"
                         % (sub.node_id, ", ".join(sorted(p))))
    node["class_type"] = sub.class_type
    inputs = node.setdefault("inputs", {})
    for stale in sub.drop_inputs:
        inputs.pop(stale, None)
    inputs.update(sub.set_inputs)
    return p


#: The declared shape difference between arms, used to PROVE the tracked graph
#: files are consistent rather than three files that silently drifted apart.
#: ``arm A + CLIP_TO_SAFETENSORS == arm B-partial`` and
#: ``arm B-partial + UNET_TO_NATIVE == arm B``, exactly.
CLIP_TO_SAFETENSORS = Substitution(
    "3", "CLIPLoader",
    set_inputs={"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan", "device": "default"},
    drop_inputs=())
UNET_TO_NATIVE = Substitution(
    "1", "UNETLoader",
    set_inputs={"unet_name": "wan2.2_ti2v_5B_fp16.safetensors",
                "weight_dtype": "default"},
    drop_inputs=())
CLIP_TO_GGUF = Substitution(
    "3", "CLIPLoaderGGUF",
    set_inputs={"clip_name": "umt5-xxl-encoder-Q5_K_M.gguf", "type": "wan"},
    drop_inputs=("device",))
UNET_TO_GGUF = Substitution(
    "1", "UnetLoaderGGUF",
    set_inputs={"unet_name": "Wan2.2-TI2V-5B-Q5_K_M.gguf"},
    drop_inputs=("weight_dtype",))


def graph_seed(prompt):
    """The seed the graph actually submits. Wan's KSampler calls it ``seed``,
    LTX's SamplerCustom calls it ``noise_seed``; both are recorded so a leg can
    be replayed."""
    seeds = []
    for nid in sorted(prompt, key=lambda k: (len(k), k)):
        inputs = prompt[nid].get("inputs", {})
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), (int, float)):
                seeds.append(int(inputs[key]))
    if not seeds:
        raise BenchError("graph carries no seed / noise_seed -- a leg that "
                         "cannot be replayed is not a measurement")
    if len(set(seeds)) != 1:
        raise BenchError("graph carries conflicting seeds: %s" % sorted(set(seeds)))
    return seeds[0]


def graph_shape(prompt):
    """(class_type, inputs) per node -- the executable shape, with the human
    ``_meta`` titles dropped. Two graphs with the same shape submit identically."""
    return {nid: (nd.get("class_type"), nd.get("inputs", {}))
            for nid, nd in prompt.items()}


def assert_arm_graphs_consistent():
    """The tracked graph files must differ EXACTLY by the declared substitution.

    Three hand-authored files can drift apart silently; this makes the
    substitution contract and the files check each other. Raises on drift."""
    a = load_graph(ARMS_BY_LABEL["A"])
    bp = load_graph(ARMS_BY_LABEL["B-partial"])
    b = load_graph(ARMS_BY_LABEL["B"])
    derived_bp = apply_substitution(a, CLIP_TO_SAFETENSORS)
    if graph_shape(derived_bp) != graph_shape(bp):
        raise BenchError(
            "arm B-partial's graph is not arm A's graph plus the declared "
            "encoder substitution -- the files have drifted apart")
    derived_b = apply_substitution(bp, UNET_TO_NATIVE)
    if graph_shape(derived_b) != graph_shape(b):
        raise BenchError(
            "arm B's graph is not arm B-partial's graph plus the declared UNet "
            "substitution -- the files have drifted apart")
    # and the inverse pair must round-trip, which is what proves the stale-input
    # removal actually removes (a substitution that only ADDS would pass above)
    back = apply_substitution(apply_substitution(b, UNET_TO_GGUF), CLIP_TO_GGUF)
    if graph_shape(back) != graph_shape(a):
        raise BenchError(
            "arm B does not invert to arm A -- a class substitution is leaving "
            "a stale class-specific input behind (weight_dtype or device)")
    return True


# --------------------------------------------------------------------------
# vendored helper install (spec G3)
# --------------------------------------------------------------------------
def install_bench_helper():
    """Install the TRACKED helper package into custom_nodes, byte-identically.

    ``custom_nodes\\otr_bakeoff_helper`` is in no git repository, so the tracked
    copy under ``scripts/bench_helper/`` is the source of truth. Verifies the
    installed digest afterwards and RAISES on mismatch -- a bench whose probes
    are a different build than the one in the commit is measuring an unknown."""
    src = os.path.join(HELPER_SRC, "__init__.py")
    if not os.path.exists(src):
        raise BenchError("vendored helper missing: %s" % src)
    want = sha256_file(src)
    dst = os.path.join(HELPER_DST, "__init__.py")
    have = sha256_file(dst) if os.path.exists(dst) else None
    if have != want:
        os.makedirs(HELPER_DST, exist_ok=True)
        shutil.copyfile(src, dst)
        log("installed bench helper -> %s (was %s)"
            % (dst, "absent" if have is None else "stale " + have[:12]))
    got = sha256_file(dst)
    if got != want:
        raise BenchError("bench helper install verify failed: want %s, got %s"
                         % (want, got))
    return want


# --------------------------------------------------------------------------
# NVML + the clamp
# --------------------------------------------------------------------------
def nvml_mib():
    """(total, used) board memory in MiB. RAISES rather than returning the old
    fictional 16.0 fallback -- an invented total silently rescales the clamp and
    every peak_delta derived from it (spec D6)."""
    try:
        import pynvml
    except Exception as e:  # noqa: BLE001
        raise BenchError("pynvml unavailable, so board memory cannot be read; "
                         "the bench refuses to guess it (%r)" % (e,))
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.total / (1024.0 * 1024.0), info.used / (1024.0 * 1024.0)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def clamp_triple(reserve_gib):
    """ComfyUI's own weight budget under ``--reserve-vram <reserve_gib>``.

    ``maximum_vram_for_weights = total * 0.88 - (0.8 GiB + EXTRA_RESERVED_VRAM)``
    and ``--reserve-vram`` REPLACES the Windows default rather than adding to it
    (model_management.py:807-814,1051-1052). On this 16303 MiB board a reserve of
    8 allows ~5.21 GiB of weights; a real 8 GB card allows ~5.65 GiB, so exact
    emulation would be R = 7.56 GiB. Keeping 8 is a justified CONSERVATIVE
    choice, not an exact emulation, and the receipt says so."""
    total_mib, _ = nvml_mib()
    total_gib = total_mib / 1024.0
    minimum_inference_gib = total_gib * 0.88
    maximum_vram_for_weights_gib = minimum_inference_gib - (0.8 + float(reserve_gib))
    return {
        "extra_reserved_vram_gib": round(float(reserve_gib), 4),
        "minimum_inference_gib": round(minimum_inference_gib, 4),
        "maximum_vram_for_weights_gib": round(maximum_vram_for_weights_gib, 4),
        "total_board_mib": round(total_mib, 1),
        "note": "computed from the documented formula; the receipt records the "
                "IN-PROCESS values reported by the running server as the "
                "authority (spec s15).",
    }


# --------------------------------------------------------------------------
# box reset -- ASSERT, do not print (CLAUDE.md s4 is a hard gate)
# --------------------------------------------------------------------------
_RESET_PS = r"""
$ErrorActionPreference='SilentlyContinue'
# kill ONLY a ComfyUI server (cmdline names main.py) -- NEVER a blanket
# Stop-Process -Name python: that also kills the Claude MCP extension pythons
# and severs the tools mid-run.
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'ComfyUI\\main\.py' -or
                 ($_.CommandLine -match 'main\.py' -and $_.CommandLine -match '--port\s+__PORT__') } |
  ForEach-Object { Write-Output ("kill server pid " + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }
Get-NetTCPConnection -LocalPort __PORT__ -State Listen |
  ForEach-Object { Write-Output ("kill port-owner pid " + $_.OwningProcess); Stop-Process -Id $_.OwningProcess -Force }
Start-Sleep -Seconds 3
$residual = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -match 'main\.py' }).Count
$listen = @(Get-NetTCPConnection -LocalPort __PORT__ -State Listen).Count
Write-Output ("RESIDUAL=" + $residual)
Write-Output ("LISTEN=" + $listen)
"""


def reset_box(port=None, settle_to_mib=None, tolerance_mib=QUIESCENCE_TOLERANCE_MIB,
              settle_timeout_s=90):
    """Selective reset, then ASSERT quiescence. Returns the settled NVML used MiB.

    The old harness printed the residual count and never checked ``nvidia-smi``.
    Booting onto residue measures garbage, and the soak harness leaves a server
    RESIDENT holding ~60% VRAM, so 'a previous run cleaned up' is never assumed."""
    port = int(port or PORT)
    ps = _RESET_PS.replace("__PORT__", str(port))
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=120)
    out = r.stdout or ""
    residual = listen = None
    for ln in out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("RESIDUAL="):
            residual = int(ln.split("=", 1)[1])
        elif ln.startswith("LISTEN="):
            listen = int(ln.split("=", 1)[1])
        else:
            log("reset: " + ln)
    if residual is None or listen is None:
        raise BenchError("reset could not read residual/listen counts; stderr=%r"
                         % (r.stderr or "")[:400])
    if residual != 0:
        raise BenchError("reset FAILED: %d ComfyUI server process(es) still "
                         "running -- refusing to boot onto residue" % residual)
    if listen != 0:
        raise BenchError("reset FAILED: port %d still has a listener -- "
                         "refusing to boot onto residue" % port)
    deadline = time.time() + settle_timeout_s
    used = None
    while True:
        _, used = nvml_mib()
        if settle_to_mib is None or used <= settle_to_mib + tolerance_mib:
            break
        if time.time() >= deadline:
            raise BenchError(
                "reset FAILED: NVML did not settle -- %.0f MiB used vs a %.0f "
                "MiB desktop baseline (tolerance %d MiB) after %ds"
                % (used, settle_to_mib, tolerance_mib, settle_timeout_s))
        time.sleep(5)
    log("reset OK: 0 servers, port %d free, NVML used %.0f MiB" % (port, used))
    return used


# --------------------------------------------------------------------------
# server lifecycle + live-schema gates
# --------------------------------------------------------------------------
def boot_server(server_log_path, reserve_gib):
    """Boot via the PROVEN launcher passed as -FilePath (CLAUDE.md s5).

    NEVER ``cmd.exe /c "<cmd>" "<log>"`` -- the /c two-quoted-token rule eats the
    outer quotes and the log comes back empty. PYTHONUTF8/PYTHONIOENCODING are
    mandatory: a detached cmd inherits cp1252 and prestartup_script.py dies on
    the first emoji (~13s boot, exit 1, 'SERVER DID NOT COME UP')."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)     # MUST run on the GPU
    env.pop("OTR_TEST_MODE", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OTR_HEADLESS_PORT"] = str(PORT)
    env["COMFYUI_OUTPUT"] = OUTPUT_BASE
    env["OTR_HEADLESS_RESERVE_VRAM_GB"] = str(reserve_gib)
    # Disarm the admission predictor. It is not in this path at all (stock nodes
    # bypass the adapters), but setting it means no code path anywhere can refuse
    # a leg for a reason we already know is wrong. per_frame 0 makes
    # compute_real_frame_budget skip its refusal branch outright.
    env.update(NEUTRALIZED_ESTIMATOR)
    log("booting ComfyUI :%d --reserve-vram %s -> %s"
        % (PORT, reserve_gib, server_log_path))
    proc = subprocess.Popen([SOAK_LAUNCH_CMD, server_log_path, "FLOOR"],
                            cwd=os.path.dirname(SOAK_LAUNCH_CMD), env=env)
    return proc


def wait_ready(timeout_s=180):
    """Boot is ~20s, not minutes. If it 'hangs', the log already says it died."""
    import requests
    start = time.time()
    last = None
    while time.time() - start < timeout_s:
        try:
            r = requests.get(BASE_URL + "/object_info", timeout=10)
            if r.status_code == 200:
                log("server ready in %.0fs" % (time.time() - start))
                return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(3)
    raise BenchError("ComfyUI did not become ready within %ds (last error %r)"
                     % (timeout_s, last))


def assert_classes_live(schemas, arm):
    """Every required node class must be live. Lifted from the old
    ``REQUIRED_CLASSES`` currency gate and made per-arm."""
    missing = [c for c in arm.required_classes if c not in schemas]
    if missing:
        raise BenchError(
            "arm %s: node classes absent from live /object_info: %s. A missing "
            "class is a hard campaign-incomplete condition, never a silent "
            "substitution." % (arm.label, ", ".join(missing)))
    return True


def resolve_enum(schemas, loader_class, input_name, filename):
    """Match a model filename to the LIVE dropdown string, handling the
    subfolder separator ComfyUI actually lists. Returns None if absent."""
    info = schemas.get(loader_class, {})
    opts = (info.get("input", {}).get("required", {}).get(input_name)
            or [None])[0]
    if not isinstance(opts, list):
        return None
    base = os.path.basename(filename)
    for o in opts:
        if o == filename:
            return o
    for o in opts:
        if os.path.basename(str(o).replace("\\", "/")) == base:
            return o
    norm = filename.replace("\\", "/").lower()
    for o in opts:
        if str(o).replace("\\", "/").lower() == norm:
            return o
    return None


def assert_models_visible(schemas, arm):
    """Every pinned model must be in the live dropdown BEFORE /prompt.

    The pattern is lifted from ``otr_wan_smoke.assert_model_visible`` (:52-65):
    fail the cell here rather than discover it as a mid-render node error.
    Returns {role: resolved_enum}."""
    resolved, missing = {}, []
    for pin in arm.models:
        enum = resolve_enum(schemas, pin.loader_class, pin.input_name,
                            pin.filename)
        if enum is None:
            missing.append("%s=%s (dropdown %s.%s)"
                           % (pin.role, pin.filename, pin.loader_class,
                              pin.input_name))
        else:
            resolved[pin.role] = enum
    if missing:
        raise BenchError(
            "arm %s: model(s) not visible in the live loader dropdown: %s. The "
            "cell FAILS -- it may never shrink the denominator and may never "
            "substitute another file." % (arm.label, "; ".join(missing)))
    return resolved


# --------------------------------------------------------------------------
# telemetry -- fail closed. A sampler that failed may never produce a PASS.
# --------------------------------------------------------------------------
#: Max tolerated gap between consecutive samples. A long gap means the sampler
#: was starved and a short peak could have been missed entirely.
MAX_SAMPLE_GAP_S = 2.0
MIN_SAMPLES = 8


class PeakSampler(object):
    """Background NVML + psutil peak sampler with a fail-closed verdict.

    The first draft of the greenlight bar could be passed by a TELEMETRY FAILURE:
    the old sampler wrote ``-1`` on NVML error and ``-1 <= 7168`` is True. Here a
    failed sampler sets ``valid=False`` and no cell can pass without it."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_vram_mib = 0.0
        self.peak_sysram_mib = 0.0
        self.peak_swap_mib = 0.0
        self.baseline_sysram_mib = None
        self.baseline_swap_mib = None
        self.samples = 0
        self.errors = []
        self.max_gap_s = 0.0
        self.first_ts = None
        self.last_ts = None
        self._stop = threading.Event()
        self._t = None

    @property
    def valid(self):
        return (not self.errors and self.samples >= MIN_SAMPLES
                and self.peak_vram_mib > 0 and self.peak_sysram_mib > 0
                and self.max_gap_s <= MAX_SAMPLE_GAP_S)

    def invalid_reasons(self):
        why = []
        if self.errors:
            why.append("sampler errors: " + "; ".join(self.errors[:4]))
        if self.samples < MIN_SAMPLES:
            why.append("only %d samples (need >= %d)" % (self.samples, MIN_SAMPLES))
        if self.peak_vram_mib <= 0:
            why.append("no NVML peak recorded")
        if self.peak_sysram_mib <= 0:
            why.append("no system-RAM peak recorded")
        if self.max_gap_s > MAX_SAMPLE_GAP_S:
            why.append("max inter-sample gap %.2fs exceeds %.2fs"
                       % (self.max_gap_s, MAX_SAMPLE_GAP_S))
        return why

    def _run(self):
        try:
            import pynvml
            import psutil
        except Exception as e:  # noqa: BLE001
            self.errors.append("import failed: %r" % (e,))
            return
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001
            self.errors.append("nvmlInit failed: %r" % (e,))
            return
        mib = 1024.0 * 1024.0
        self.baseline_sysram_mib = psutil.virtual_memory().used / mib
        self.baseline_swap_mib = psutil.swap_memory().used / mib
        prev = None
        try:
            while not self._stop.is_set():
                now = time.time()
                try:
                    used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / mib
                    sys_used = psutil.virtual_memory().used / mib
                    swap_used = psutil.swap_memory().used / mib
                except Exception as e:  # noqa: BLE001
                    self.errors.append("sample failed: %r" % (e,))
                    return
                self.peak_vram_mib = max(self.peak_vram_mib, used)
                self.peak_sysram_mib = max(self.peak_sysram_mib, sys_used)
                self.peak_swap_mib = max(self.peak_swap_mib, swap_used)
                self.samples += 1
                if self.first_ts is None:
                    self.first_ts = now
                self.last_ts = now
                if prev is not None:
                    self.max_gap_s = max(self.max_gap_s, now - prev)
                prev = now
                time.sleep(self.interval)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
            if self._t.is_alive():
                self.errors.append("sampler thread did not join")

    def as_dict(self):
        return {
            "peak_vram_mib": round(self.peak_vram_mib, 1),
            "peak_sysram_mib": round(self.peak_sysram_mib, 1),
            "baseline_sysram_mib": (round(self.baseline_sysram_mib, 1)
                                    if self.baseline_sysram_mib else None),
            "peak_swap_mib": round(self.peak_swap_mib, 1),
            "baseline_swap_mib": (round(self.baseline_swap_mib, 1)
                                  if self.baseline_swap_mib is not None else None),
            "samples": self.samples,
            "max_sample_gap_s": round(self.max_gap_s, 3),
            "window_s": (round(self.last_ts - self.first_ts, 2)
                         if self.first_ts and self.last_ts else None),
            "errors": list(self.errors),
            "valid": self.valid,
            "invalid_reasons": self.invalid_reasons(),
        }


# --------------------------------------------------------------------------
# server-log parsing
# --------------------------------------------------------------------------
#: NAMED spill signatures. The old harness matched a bare ``offload``, which is
#: ordinary loader chatter -- the hint fired on healthy runs and meant nothing.
#: Each of these is a real degradation, not routine placement.
SPILL_SIGNATURES = (
    ("cuda_oom", r"(?i)torch\.(?:cuda\.)?OutOfMemoryError|CUDA out of memory"),
    ("alloc_retry", r"(?i)tried to allocate .* free|allocation on device .* failed"),
    ("sysmem_fallback", r"(?i)(shared|system) memory fallback|falling back to (cpu|system)"),
    ("model_evicted_midrun", r"(?i)unloading model .* to free|forced unload"),
)

_PROBE_RE = re.compile(
    r"\[OTR_BakeoffVramProbe\] marker=(?P<marker>[0-9a-f]+) "
    r"max_allocated_mb=(?P<alloc>[0-9.]+) max_reserved_mb=(?P<resv>[0-9.]+)")
_RESET_RE = re.compile(
    r"\[OTR_BakeoffVramReset\] marker=(?P<marker>[0-9a-f]+) "
    r"reset_peak_memory_stats OK")


def parse_timing(log_slice):
    """s/it and the executed-in line, read from the log alone.

    Returns ``(None, "unavailable", ...)`` rather than inventing a number: the
    wall/steps fallback belongs to the CALLER, which owns the arm's step count.
    A module-global ``STEPS = 30`` corrupts s/it for the 8-step LTX arm."""
    sit = None
    for m in re.finditer(r"(\d+\.?\d*)\s*s/it", log_slice):
        sit = float(m.group(1))              # last wins (the sampler bar)
    source = "log"
    if sit is None:
        for m in re.finditer(r"(\d+\.?\d*)\s*it/s", log_slice):
            its = float(m.group(1))
            sit = (1.0 / its) if its else None
    if sit is None:
        sit, source = None, "unavailable"
    prompt_exec = None
    m = re.search(r"Prompt executed in ([0-9:.]+)", log_slice)
    if m:
        prompt_exec = m.group(1)
    return sit, source, prompt_exec


def parse_spill(log_slice):
    """Named degradation signatures present in this cell's log slice."""
    return [name for name, pat in SPILL_SIGNATURES
            if re.search(pat, log_slice)]


def parse_stage_probe(log_slice):
    """The order-safe sample+decode torch segment (spec 9.3).

    ``OTR_BakeoffVramReset`` opens it on the latent edge immediately before the
    sampler; ``OTR_BakeoffVramProbe`` closes it on the image edge immediately
    after decode.

    COUNT DISTINCT MARKERS, NOT LINES. Each probe emits its line TWICE by design
    -- once through ``print`` and once through ``_LOG.warning`` -- so the server
    log carries two copies of one execution. The marker is a fresh uuid per call,
    so distinct markers is the honest count of executions. Still fail-closed:
    zero markers means the probes did not run (a cached node, a graph that lost
    them), and two distinct markers means the slice spans cells and the numbers
    are not this cell's.

    ADVISORY ONLY. Per-stage measurement was declined (operator ruling O7:
    whole-window peak is enough), so this does NOT gate a cell -- the gating
    telemetry is the whole-window NVML sampler."""
    resets = set(_RESET_RE.findall(log_slice))
    by_marker = {}
    for m in _PROBE_RE.finditer(log_slice):
        by_marker[m.group("marker")] = {
            "marker": m.group("marker"),
            "torch_max_allocated_mib": float(m.group("alloc")),
            "torch_max_reserved_mib": float(m.group("resv"))}
    found = list(by_marker.values())
    if len(resets) != 1 or len(found) != 1:
        return {"valid": False, "resets": len(resets), "probes": len(found),
                "advisory": True,
                "reason": "expected exactly one distinct reset marker and one "
                          "distinct probe marker in this cell's log slice, got "
                          "%d and %d" % (len(resets), len(found))}
    out = dict(found[0])
    out["advisory"] = True
    out["valid"] = True
    out["segment"] = "sample+decode (order-safe half only; the text-encode / "
    out["segment"] += "image-encode boundaries need forced ordering, which "
    out["segment"] += "changes the memory schedule under test -- operator "
    out["segment"] += "decision O7, deferred to a diagnostic campaign)"
    return out


# --------------------------------------------------------------------------
# per-cell graph mutation
# --------------------------------------------------------------------------
def build_cell_prompt(arm, base_prompt, length, resolved_models, filename_prefix):
    """Apply the per-cell deltas: length, resolved model enums, save prefix.

    Everything else -- canvas, seed, prompts, recipe, decode geometry -- is baked
    into the arm's pinned graph and is NOT touched here, so no cell can quietly
    differ from the graph its receipt names."""
    p = copy.deepcopy(base_prompt)
    node = p.get(arm.length_node)
    if node is None or "length" not in node.get("inputs", {}):
        raise BenchError("arm %s: length node %r has no 'length' input"
                         % (arm.label, arm.length_node))
    fc = FRAME_CONTRACTS[arm.engine_id]
    if length < fc["min"] or length > fc["max"] or (length - fc["min"]) % fc["quantum"]:
        raise BenchError("arm %s: length %d is illegal for %s"
                         % (arm.label, length, arm.engine_id))
    node["inputs"]["length"] = int(length)
    # CANVAS IS ARM-OWNED. Each arm renders at ITS OWN shipped canvas -- this is
    # not a controlled comparison, it is "does this engine run on a small GPU and
    # how much does it eat". The ArmSpec is the single authority; the graph file
    # agrees with it, and offline_preflight proves they agree.
    w, h = arm.canvas
    for axis, value in (("width", w), ("height", h)):
        if axis not in node["inputs"]:
            raise BenchError("arm %s: length node %r has no %r input"
                             % (arm.label, arm.length_node, axis))
        node["inputs"][axis] = int(value)
    for pin in arm.models:
        enum = resolved_models.get(pin.role)
        if enum is None:
            raise BenchError("arm %s: no resolved enum for model role %r"
                             % (arm.label, pin.role))
        target = p.get(pin.node_id)
        if target is None:
            raise BenchError("arm %s: model node %r not in graph"
                             % (arm.label, pin.node_id))
        if target.get("class_type") != pin.loader_class:
            raise BenchError(
                "arm %s: node %s is %r in the graph but the spec pins it as %r "
                "-- the ArmSpec and its graph disagree"
                % (arm.label, pin.node_id, target.get("class_type"),
                   pin.loader_class))
        target["inputs"][pin.input_name] = enum
    save = p.get(arm.save_node)
    if save is None or "filename_prefix" not in save.get("inputs", {}):
        raise BenchError("arm %s: save node %r has no 'filename_prefix'"
                         % (arm.label, arm.save_node))
    save["inputs"]["filename_prefix"] = filename_prefix
    return p


# --------------------------------------------------------------------------
# asset validation -- from /history, never a glob
# --------------------------------------------------------------------------
def _history_outputs(prompt_id, save_node):
    import requests
    r = requests.get("%s/history/%s" % (BASE_URL, prompt_id), timeout=30)
    r.raise_for_status()
    hist = r.json() or {}
    entry = hist.get(str(prompt_id)) or {}
    outputs = (entry.get("outputs") or {}).get(str(save_node)) or {}
    files = []
    for value in outputs.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("filename"):
                    files.append(item)
    return files


def validate_asset(prompt_id, arm, expected_frames, window_start, window_end):
    """Resolve the EXACT asset this cell produced from /history, then ffprobe it.

    A glob over the output dir can select a stale mp4 from an earlier cell and
    report success for a render that never wrote anything. History names the file
    the server actually created."""
    files = _history_outputs(prompt_id, arm.save_node)
    if len(files) != 1:
        raise BenchError("arm %s: /history reports %d output files for save node "
                         "%s; expected exactly 1" % (arm.label, len(files),
                                                     arm.save_node))
    meta = files[0]
    path = os.path.abspath(os.path.join(OUTPUT_BASE, meta.get("subfolder") or "",
                                        meta["filename"]))
    root = os.path.abspath(OUTPUT_BASE) + os.sep
    if not path.startswith(root):
        raise BenchError("arm %s: history asset %s escapes the configured output "
                         "root %s" % (arm.label, path, OUTPUT_BASE))
    if not os.path.exists(path):
        raise BenchError("arm %s: history names %s but it is not on disk"
                         % (arm.label, path))
    mtime = os.path.getmtime(path)
    if not (window_start - 5 <= mtime <= window_end + 60):
        raise BenchError(
            "arm %s: %s was not created inside this cell's window (mtime %.0f, "
            "window %.0f..%.0f) -- it is a stale asset"
            % (arm.label, path, mtime, window_start, window_end))
    size = os.path.getsize(path)
    if size <= 0:
        raise BenchError("arm %s: %s is 0 bytes" % (arm.label, path))
    probe = ffprobe_video(path)
    w, h = arm.canvas
    if (probe["width"], probe["height"]) != (w, h):
        raise BenchError("arm %s: %s is %dx%d, expected %dx%d"
                         % (arm.label, path, probe["width"], probe["height"], w, h))
    if probe["frames"] != int(expected_frames):
        raise BenchError("arm %s: %s has %s frames, expected %d"
                         % (arm.label, path, probe["frames"], expected_frames))
    if probe["duration_s"] <= 0:
        raise BenchError("arm %s: %s has non-positive duration" % (arm.label, path))
    probe["path"] = path
    probe["bytes"] = size
    return probe


def _ffprobe(args):
    exe = os.environ.get("OTR_FFPROBE") or "ffprobe"
    r = subprocess.run([exe] + args, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise BenchError("ffprobe failed (%d): %s"
                         % (r.returncode, (r.stderr or "").strip()[:400]))
    return json.loads(r.stdout or "{}")


def ffprobe_video(path):
    """width / height / frames / fps / codec / duration. Frame count falls back
    to a decoded count when the container does not carry ``nb_frames``."""
    data = _ffprobe(["-v", "error", "-select_streams", "v:0", "-of", "json",
                     "-show_entries",
                     "stream=width,height,nb_frames,codec_name,avg_frame_rate:"
                     "format=duration", path])
    streams = data.get("streams") or []
    if not streams:
        raise BenchError("ffprobe found no video stream in %s" % path)
    s = streams[0]
    frames = s.get("nb_frames")
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        frames = None
    if not frames:
        counted = _ffprobe(["-v", "error", "-select_streams", "v:0", "-of",
                            "json", "-count_frames", "-show_entries",
                            "stream=nb_read_frames", path])
        cs = (counted.get("streams") or [{}])[0]
        try:
            frames = int(cs.get("nb_read_frames"))
        except (TypeError, ValueError):
            raise BenchError("ffprobe could not determine a frame count for %s"
                             % path)
    rate = s.get("avg_frame_rate") or "0/1"
    try:
        num, den = rate.split("/")
        fps = (float(num) / float(den)) if float(den) else 0.0
    except Exception:  # noqa: BLE001
        fps = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {"width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
            "frames": frames, "fps": round(fps, 4),
            "codec": s.get("codec_name"), "duration_s": round(duration, 4)}


# --------------------------------------------------------------------------
# staging + campaign manifest
# --------------------------------------------------------------------------
def stage_assets():
    """The headless server reads LoadImage from the INSTALL-root input dir, not
    Documents\\ComfyUI\\input. Fail loud if a source is missing."""
    os.makedirs(SERVER_INPUT, exist_ok=True)
    for name in FIXED_ASSETS:
        src = os.path.join(SRC_INPUT, name)
        if not os.path.exists(src):
            raise BenchError("fixed asset missing: %s" % src)
        shutil.copy2(src, os.path.join(SERVER_INPUT, name))
        log("staged asset -> %s" % os.path.join(SERVER_INPUT, name))


def build_manifest(arms, helper_sha, git_head=None):
    """Written BEFORE cell 1. Any mismatch fails a cell -- cross-arm comparison
    is void if the stack moved mid-campaign (spec s8/D13)."""
    total_mib, _ = nvml_mib()
    return {
        "campaign": "four-arm clamped video bench",
        "label": RESULT_LABEL,
        "spec": "docs/2026-07-31-four-arm-clamped-video-bench-SPEC.md",
        "judgment_of_record": "docs/2026-07-31-wan-8gb-adversarial-review/report.md",
        "authority": "CLAUDE.md s0A isolated-bench carve-out (operator O6)",
        "git_head": git_head,
        "python": sys.version.split()[0],
        "board_total_mib": round(total_mib, 1),
        "clamp_reserve_gib": CAMPAIGN_CLAMP_GIB,
        "clamp": clamp_triple(CAMPAIGN_CLAMP_GIB),
        "greenlight_peak_delta_mib": GREENLIGHT_PEAK_DELTA_MIB,
        "display_allowance_mib": DISPLAY_ALLOWANCE_MIB,
        "display_allowance_note":
            "DECLARED CONSERVATIVE ASSUMPTION, not a measurement (operator O5). "
            "If it is wrong every verdict shifts by the difference.",
        "lengths": list(CAMPAIGN_LENGTHS),
        "bench_helper_sha256": helper_sha,
        "base_url": BASE_URL,
        "port": PORT,
        "output_root": OUTPUT_BASE,
        "estimator_seams": {k: os.environ.get(k) for k in ESTIMATOR_SEAMS},
        "estimator_status":
            "INERT ON THIS PATH. Driving stock nodes directly bypasses the OTR "
            "engine adapters, so motion_common's admission estimator never runs "
            "and these seams cannot refuse a cell. They are recorded for the "
            "record only. Do not read a bypassed predictor as a working one.",
        "arms": [a.as_dict() for a in arms],
        "not_a_qualification":
            "This bench does not qualify the 8 GB tier, does not license "
            "lowering FRAME_COST_MODEL, does not promote 14B, does not refit the "
            "estimator, and does not prove teardown or whole-pipeline phase "
            "behaviour. Results are never worded as '8 GB qualified'.",
    }


# --------------------------------------------------------------------------
# watchdog (spec D12)
# --------------------------------------------------------------------------
WATCHDOG_PS1 = os.path.join(REPO, "scripts", "otr_render_watchdog.ps1")


class Watchdog(object):
    """Wraps ``otr_render_watchdog.ps1``: exit 2 = DEAD (5-min heartbeat stall or
    a down queue endpoint), exit 0 = the leg finished. It REPORTS only; this
    harness owns the reset."""

    def __init__(self, leg_log, enabled=True):
        self.leg_log = leg_log
        self.enabled = enabled and os.path.exists(WATCHDOG_PS1)
        self.proc = None
        self.returncode = None

    def __enter__(self):
        if self.enabled:
            self.proc = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", WATCHDOG_PS1, "-LegLog", self.leg_log,
                 "-QueueUrl", BASE_URL + "/queue"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *a):
        if self.proc is not None:
            self.returncode = self.proc.poll()
            if self.returncode is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=15)
                except Exception:  # noqa: BLE001
                    self.proc.kill()

    @staticmethod
    def verdict(returncode):
        """Map an exit code to a cell verdict. Exit 2 is the ONLY failing code;
        None means the watchdog was still running when the render finished, which
        is the healthy case."""
        if returncode is None:
            return {"ran": True, "dead": False, "note": "still running at cell end"}
        if returncode == 2:
            return {"ran": True, "dead": True,
                    "note": "watchdog declared the leg DEAD (heartbeat stall or "
                            "down queue endpoint)"}
        return {"ran": True, "dead": False, "note": "exit %d" % returncode}


# --------------------------------------------------------------------------
# one cell = (arm x length x repeat): own reset + boot + render + teardown
# --------------------------------------------------------------------------
def run_cell(arm, length, repeat, manifest, campaign_baseline_mib,
             use_watchdog=True):
    cell_id = arm.cell_id(length, repeat)
    arm_dir = os.path.join(BENCH_DIR, arm.label)
    os.makedirs(arm_dir, exist_ok=True)
    server_log = os.path.join(arm_dir, "comfy_server_%s.log" % cell_id)
    receipt = {
        "cell_id": cell_id, "arm": arm.label, "engine_id": arm.engine_id,
        "length": int(length), "repeat": int(repeat),
        "canvas": list(arm.canvas), "steps": arm.steps,
        "graph": arm.graph, "graph_sha256": arm.graph_sha256,
        "clamp_reserve_gib": CAMPAIGN_CLAMP_GIB,
        "label": RESULT_LABEL,
        "arm_notes": arm.notes,
        "estimator_status": manifest["estimator_status"],
        "estimator_seams": manifest["estimator_seams"],
        "status": "error",
    }
    if arm.gated_reason:
        receipt["status"] = "gated"
        receipt["error"] = arm.gated_reason
        log("CELL %s GATED: %s" % (cell_id, arm.gated_reason))
        return receipt
    try:
        baseline = reset_box(PORT, settle_to_mib=campaign_baseline_mib)
        receipt["desktop_baseline_nvml_mib"] = round(baseline, 1)
        boot_server(server_log, CAMPAIGN_CLAMP_GIB)
        schemas = wait_ready()
        assert_classes_live(schemas, arm)
        resolved = assert_models_visible(schemas, arm)
        receipt["resolved_models"] = resolved
        _, pre_submit = nvml_mib()
        receipt["pre_submit_nvml_mib"] = round(pre_submit, 1)
        base_prompt = load_graph(arm)
        prefix = "otr/episodes/_bench_4arm/%s/%s" % (arm.label, cell_id)
        prompt = build_cell_prompt(arm, base_prompt, length, resolved, prefix)
        receipt["seed"] = graph_seed(prompt)
        log("CELL %s | %s | %dx%d | %d frames | %d steps | seed %d | reserve %d GiB"
            % (cell_id, arm.label, arm.canvas[0], arm.canvas[1], length,
               arm.steps, receipt["seed"], CAMPAIGN_CLAMP_GIB))

        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            slog_offset = f.tell()

        import _run_baseline as RB
        t0 = time.time()
        with Watchdog(server_log, enabled=use_watchdog) as wd:
            with PeakSampler() as ps:
                prompt_id = RB._submit_prompt(prompt, BASE_URL)
                RB._poll_for_completion(prompt_id, BASE_URL, timeout_s=2400.0)
        t1 = time.time()
        receipt["wall_s"] = round(t1 - t0, 1)
        receipt["prompt_id"] = prompt_id
        receipt["telemetry"] = ps.as_dict()
        receipt["watchdog"] = Watchdog.verdict(wd.returncode)
        receipt["peak_vram_mib"] = round(ps.peak_vram_mib, 1)
        receipt["peak_delta_mib"] = round(ps.peak_vram_mib - baseline, 1)
        receipt["sysram_delta_mib"] = (
            round(ps.peak_sysram_mib - ps.baseline_sysram_mib, 1)
            if ps.baseline_sysram_mib else None)
        receipt["pagefile_commit_delta_mib"] = (
            round(ps.peak_swap_mib - ps.baseline_swap_mib, 1)
            if ps.baseline_swap_mib is not None else None)

        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(slog_offset)
            slice_ = f.read()
        sit, source, pexec = parse_timing(slice_)
        receipt["s_per_it"] = (sit if sit is not None
                               else round((t1 - t0) / arm.steps, 3))
        receipt["s_per_it_source"] = (source if sit is not None else "wall/steps")
        receipt["prompt_executed_in"] = pexec
        receipt["spill_signatures"] = parse_spill(slice_)
        receipt["stage_probe"] = parse_stage_probe(slice_)

        receipt["asset"] = validate_asset(prompt_id, arm, length, t0, t1)
        receipt["status"] = "ok"
        log("CELL %s OK | peak %.0f MiB | delta %.0f MiB | %ss/it | %s"
            % (cell_id, receipt["peak_vram_mib"], receipt["peak_delta_mib"],
               receipt["s_per_it"], receipt["asset"]["path"]))
    except Exception as e:  # noqa: BLE001
        receipt["error"] = "%s: %s" % (type(e).__name__, e)
        log("CELL %s FAIL: %s" % (cell_id, receipt["error"]))
    finally:
        try:
            reset_box(PORT)
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)
    return receipt


# --------------------------------------------------------------------------
# greenlight -- an arm passes only if ALL of spec s12 holds
# --------------------------------------------------------------------------
#: Declared numeric rejection thresholds. Avoiding CUDA OOM through uncontrolled
#: system-memory thrash is a FAILURE, not a pass.
MAX_SYSRAM_DELTA_MIB = 24576
MAX_PAGEFILE_COMMIT_DELTA_MIB = 8192

#: Licence receipt per arm (greenlight condition 6). A measurement may run on a
#: licence we could not ship; a PASS may not.
ARM_LICENCE = {
    "A": {"shippable": True, "licence": "apache-2.0",
          "source": "Wan-AI/Wan2.2-TI2V-5B (checked at both levels)"},
    "B-partial": {"shippable": True, "licence": "apache-2.0",
                  "source": "Wan-AI/Wan2.2-TI2V-5B + Comfy-Org repackaged encoder"},
    "B": {"shippable": True, "licence": "apache-2.0",
          "source": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"},
    "D": {"shippable": None, "licence": "LTX-Video Open Weights License 0.X",
          "source": "Lightricks/LTX-Video (HF license: other)",
          "note": "UNRESOLVED (operator decision O4): the >= $10,000,000 annual "
                  "revenue clause admits two readings. Gates SHIPPING, not "
                  "measuring. An arm-D PASS is provisional until O4 is settled."},
}


def cell_blocks_pass(receipt):
    """Reasons this cell blocks its arm's PASS. Empty list = clean.

    A missing, skipped, telemetry-invalid, OOM, watchdog-failed or
    decode-invalid cell blocks PASS and NEVER shrinks the denominator."""
    why = []
    if receipt.get("status") != "ok":
        why.append("status=%s (%s)" % (receipt.get("status"),
                                       receipt.get("error", "")))
        return why
    tel = receipt.get("telemetry") or {}
    if not tel.get("valid"):
        why.append("telemetry invalid: " + "; ".join(tel.get("invalid_reasons")
                                                     or ["unknown"]))
    # The per-stage torch probe is ADVISORY, never gating: per-stage measurement
    # was declined (operator ruling O7 -- whole-window peak is enough). The
    # gating telemetry is the whole-window NVML sampler checked above.
    if receipt.get("watchdog", {}).get("dead"):
        why.append("watchdog declared the leg dead")
    spills = receipt.get("spill_signatures") or []
    if spills:
        why.append("named spill signature(s): " + ", ".join(spills))
    delta = receipt.get("peak_delta_mib")
    if delta is None:
        why.append("no peak_delta_mib recorded")
    elif delta > GREENLIGHT_PEAK_DELTA_MIB:
        why.append("peak_delta %.0f MiB exceeds the %d MiB bar"
                   % (delta, GREENLIGHT_PEAK_DELTA_MIB))
    sysd = receipt.get("sysram_delta_mib")
    if sysd is not None and sysd > MAX_SYSRAM_DELTA_MIB:
        why.append("system-RAM delta %.0f MiB exceeds %d MiB -- avoiding CUDA "
                   "OOM by host thrash is a failure" % (sysd, MAX_SYSRAM_DELTA_MIB))
    pfd = receipt.get("pagefile_commit_delta_mib")
    if pfd is not None and pfd > MAX_PAGEFILE_COMMIT_DELTA_MIB:
        why.append("pagefile commit delta %.0f MiB exceeds %d MiB"
                   % (pfd, MAX_PAGEFILE_COMMIT_DELTA_MIB))
    if not (receipt.get("asset") or {}).get("path"):
        why.append("no validated asset")
    return why


def grade_arm(arm, receipts):
    """PASS / FAIL / floor-only for one arm over the MANDATORY denominator."""
    by_length = {}
    for r in receipts:
        if r.get("arm") == arm.label and int(r.get("repeat", 0)) == 0:
            by_length.setdefault(int(r["length"]), []).append(r)
    blockers, clean_lengths = [], []
    for length in arm.lengths:
        cells = by_length.get(length, [])
        if len(cells) != 1:
            blockers.append("length %d: expected exactly 1 required cell, got %d"
                            % (length, len(cells)))
            continue
        why = cell_blocks_pass(cells[0])
        if why:
            blockers.append("length %d: %s" % (length, "; ".join(why)))
        else:
            clean_lengths.append(length)
    licence = ARM_LICENCE.get(arm.label, {})
    if licence.get("shippable") is False:
        blockers.append("licence is not shippable: %s" % licence.get("licence"))
    verdict = "PASS" if not blockers else "FAIL"
    if verdict == "PASS" and clean_lengths == [min(arm.lengths)]:
        verdict = "floor-only"
    return {
        "arm": arm.label, "verdict": verdict, "label": RESULT_LABEL,
        "clean_lengths": clean_lengths, "required_lengths": list(arm.lengths),
        "blockers": blockers, "licence": licence,
        "meaning": "A PASS earns permission to proceed to qualification and "
                   "NOTHING else: it does not lower the admission guard, does "
                   "not refit the estimator, and does not qualify the 8 GB "
                   "tier. 'floor-only' means it passed at the shortest length "
                   "alone -- a demo, not a tier candidate.",
    }


# --------------------------------------------------------------------------
# results -- written ATOMICALLY (an interrupted multi-hour run must not leave a
# truncated result file)
# --------------------------------------------------------------------------
def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def results_table(receipts):
    """THE deliverable: one row per cell, and one line per failure. Nothing else.

    Deliberately carries no verdict, no ranking and no tier language. Each arm
    ran at ITS OWN shipped canvas and recipe, so the rows are three separate
    answers to 'does this engine run on a small GPU and how much does it eat' --
    they are not a comparison and must not be read as one."""
    lines = [
        "| arm | engine | canvas | frames | PASS/FAIL | peak VRAM MB | seconds "
        "| asset path |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    failures = []
    for r in receipts:
        ok = r.get("status") == "ok" and not cell_blocks_pass(r)
        canvas = "x".join(str(v) for v in (r.get("canvas") or ["?", "?"]))
        asset = (r.get("asset") or {}).get("path") or "-"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("arm", "?"), r.get("engine_id", "?"), canvas,
            r.get("length", "-"), "PASS" if ok else "FAIL",
            r.get("peak_vram_mib", "-"), r.get("wall_s", "-"), asset))
        if not ok:
            why = r.get("error") or "; ".join(cell_blocks_pass(r)) or "unknown"
            failures.append("%s @ %s %sf: %s"
                            % (r.get("arm", "?"), canvas, r.get("length", "?"),
                               why))
    return "\n".join(lines), failures


def write_results(receipts, manifest, arms):
    os.makedirs(BENCH_DIR, exist_ok=True)
    grades = [grade_arm(a, receipts) for a in arms]
    payload = {"manifest": manifest, "grades": grades, "cells": receipts}
    write_atomic(os.path.join(BENCH_DIR, "video_arm_bench_results.json"),
                 json.dumps(payload, indent=2) + "\n")
    table, failures = results_table(receipts)
    out = ["# Video arm bench -- results", "",
           "%s. Seed %s. Clamp --reserve-vram %d. Each arm at ITS OWN shipped "
           "canvas and recipe; the rows are separate answers, not a comparison."
           % (RESULT_LABEL,
              (receipts[0].get("seed") if receipts else "?"),
              CAMPAIGN_CLAMP_GIB),
           "", table, ""]
    if failures:
        out += ["## Failures", ""] + ["- " + f for f in failures] + [""]
    write_atomic(os.path.join(BENCH_DIR, "video_arm_bench_table.md"),
                 "\n".join(out) + "\n")


# --------------------------------------------------------------------------
# preflights
# --------------------------------------------------------------------------
def offline_preflight(arms):
    """Everything checkable with NO server and NO GPU. Raises on the first
    failure; this is what the unit tests drive."""
    assert_arm_graphs_consistent()
    for arm in arms:
        prompt = load_graph(arm)
        for pin in arm.models:
            node = prompt.get(pin.node_id)
            if node is None:
                raise BenchError("arm %s: model node %r missing from %s"
                                 % (arm.label, pin.node_id, arm.graph))
            if node.get("class_type") != pin.loader_class:
                raise BenchError(
                    "arm %s: node %s is %r in %s but the ArmSpec pins %r"
                    % (arm.label, pin.node_id, node.get("class_type"),
                       arm.graph, pin.loader_class))
            if pin.input_name not in node.get("inputs", {}):
                raise BenchError("arm %s: node %s has no input %r"
                                 % (arm.label, pin.node_id, pin.input_name))
        # the graph file's canvas must agree with the ArmSpec, so a reader of
        # either one sees the canvas the arm actually renders at
        latent = prompt.get(arm.length_node, {}).get("inputs", {})
        declared = (latent.get("width"), latent.get("height"))
        if declared != arm.canvas:
            raise BenchError(
                "arm %s: %s declares canvas %sx%s but the ArmSpec says %dx%d"
                % (arm.label, arm.graph, declared[0], declared[1],
                   arm.canvas[0], arm.canvas[1]))
        graph_seed(prompt)
        classes = {nd.get("class_type") for nd in prompt.values()}
        undeclared = classes - set(arm.required_classes)
        if undeclared:
            raise BenchError(
                "arm %s: graph uses classes the ArmSpec does not declare "
                "required: %s" % (arm.label, ", ".join(sorted(undeclared))))
        # every rung must build, so an illegal ladder cannot reach a GPU
        resolved = {p.role: p.filename for p in arm.models}
        for length in arm.lengths:
            build_cell_prompt(arm, prompt, length, resolved,
                              "otr/episodes/_bench_4arm/%s/probe" % arm.label)
        ids = [arm.cell_id(n, r) for n in arm.lengths for r in (0, 1, 2)]
        if len(set(ids)) != len(ids):
            raise BenchError("arm %s: cell ids collide across repeats" % arm.label)
    return True


def dry_validate(arms, boot=True):
    """Offline checks, then (optionally) boot ONCE and confirm node-class
    currency + loader enums per arm. Renders NOTHING."""
    offline_preflight(arms)
    log("offline preflight PASS (graphs pinned, consistent, every rung builds)")
    helper_sha = install_bench_helper()
    log("bench helper sha256 %s" % helper_sha)
    if not boot:
        return 0
    os.makedirs(BENCH_DIR, exist_ok=True)
    server_log = os.path.join(BENCH_DIR, "comfy_server_dryvalidate.log")
    ok = True
    try:
        baseline = reset_box(PORT)
        log("desktop baseline NVML %.0f MiB" % baseline)
        boot_server(server_log, CAMPAIGN_CLAMP_GIB)
        schemas = wait_ready()
        for arm in arms:
            try:
                assert_classes_live(schemas, arm)
                resolved = assert_models_visible(schemas, arm)
                log("arm %-10s classes OK | models %s"
                    % (arm.label, json.dumps(resolved)))
            except BenchError as e:
                ok = False
                log("arm %-10s PREFLIGHT FAIL: %s" % (arm.label, e))
        log("DRY-VALIDATE %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        try:
            reset_box(PORT)
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)


def regrade(arms):
    """Re-derive the table from receipts already on disk. NO GPU, NO re-render.

    Each cell booted its own server into its own log, so that whole log IS the
    cell's slice -- the advisory stage probe is re-parsed from it rather than
    left stale. Measured values (peak, wall, asset) are never recomputed: they
    are what the run observed and this must not be able to launder them."""
    path = os.path.join(BENCH_DIR, "video_arm_bench_results.json")
    if not os.path.exists(path):
        raise BenchError("nothing to regrade: %s does not exist" % path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    receipts = payload.get("cells") or []
    if not receipts:
        raise BenchError("nothing to regrade: %s has no cells" % path)
    for r in receipts:
        arm_dir = os.path.join(BENCH_DIR, str(r.get("arm")))
        log_path = os.path.join(arm_dir,
                                "comfy_server_%s.log" % r.get("cell_id"))
        if not os.path.exists(log_path):
            r.setdefault("stage_probe", {})["regrade_note"] = (
                "server log absent; advisory probe left as recorded")
            continue
        with open(log_path, encoding="utf-8", errors="replace") as f:
            r["stage_probe"] = parse_stage_probe(f.read())
    write_results(receipts, payload.get("manifest") or {}, arms)
    log("regraded %d cells from receipts on disk (no GPU)" % len(receipts))
    return receipts


def git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=30)
        return (r.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Clamped multi-arm video bench. BENCH != QUALIFICATION.")
    ap.add_argument("--dry-validate", action="store_true",
                    help="offline checks + one boot to confirm node classes and "
                         "loader enums per arm; renders NOTHING")
    ap.add_argument("--offline-only", action="store_true",
                    help="offline checks only -- no server, no GPU")
    ap.add_argument("--regrade", action="store_true",
                    help="re-derive the table from receipts already on disk; "
                         "no server, no GPU, measured values never recomputed")
    ap.add_argument("--arms", default="",
                    help="comma list of arm labels (default: every arm that is "
                         "not gated). Known: %s" % ", ".join(a.label for a in ARMS))
    ap.add_argument("--lengths", default="",
                    help="comma list of frame lengths (default %s)"
                         % ",".join(str(n) for n in CAMPAIGN_LENGTHS))
    ap.add_argument("--include-gated", action="store_true",
                    help="run gated arms too; they will record status=gated with "
                         "the named acquisition blocker")
    ap.add_argument("--no-watchdog", action="store_true",
                    help="do not attach otr_render_watchdog.ps1")
    args = ap.parse_args(argv)

    sel = set(s.strip() for s in args.arms.split(",") if s.strip())
    unknown = sel - set(ARMS_BY_LABEL)
    if unknown:
        raise BenchError("unknown arm label(s): %s (known: %s)"
                         % (", ".join(sorted(unknown)),
                            ", ".join(a.label for a in ARMS)))
    arms = [a for a in ARMS if not sel or a.label in sel]
    if not args.include_gated and not sel:
        arms = [a for a in arms if not a.gated_reason]
    # ascending, always: rungs ascend within an arm so a terminal OOM stops that
    # arm's ladder at the first length it cannot carry
    lengths = sorted([int(x) for x in args.lengths.split(",") if x.strip()]
                     or list(CAMPAIGN_LENGTHS))

    if args.offline_only:
        offline_preflight(ARMS)
        log("offline preflight PASS for all %d arms" % len(ARMS))
        return 0
    if args.regrade:
        regrade(arms)
        return 0
    if args.dry_validate:
        return dry_validate(arms)

    stage_assets()
    helper_sha = install_bench_helper()
    offline_preflight(arms)
    os.makedirs(BENCH_DIR, exist_ok=True)
    manifest = build_manifest(arms, helper_sha, git_head())
    write_atomic(os.path.join(BENCH_DIR, "campaign_manifest.json"),
                 json.dumps(manifest, indent=2) + "\n")
    log("manifest written; %s" % RESULT_LABEL)

    campaign_baseline = reset_box(PORT)
    log("campaign desktop baseline: %.0f MiB" % campaign_baseline)

    receipts = []
    for arm in arms:
        for length in lengths:
            if length not in arm.lengths:
                log("arm %s: length %d is not in its ladder -- skipped"
                    % (arm.label, length))
                continue
            receipts.append(run_cell(arm, length, 0, manifest,
                                     campaign_baseline,
                                     use_watchdog=not args.no_watchdog))
            write_results(receipts, manifest, arms)     # incremental
            if receipts[-1].get("status") != "ok":
                log("arm %s: rung %d did not pass -- stopping this arm's ladder"
                    % (arm.label, length))
                break
    write_results(receipts, manifest, arms)
    for g in [grade_arm(a, receipts) for a in arms]:
        log("ARM %-10s %s (%s)" % (g["arm"], g["verdict"], RESULT_LABEL))
    log("DONE. %d cells. Results under %s" % (len(receipts), BENCH_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
