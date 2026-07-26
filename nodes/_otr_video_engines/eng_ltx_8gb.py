"""LTX-Video 0.9.8 distilled 2B image->video motion adapter (8GB tier) -- in-process.

The 8GB-tier LTX sibling. It animates a still into motion on the OFFICIAL LTX-Video
**0.9.8 distilled 2B** all-in-one checkpoint (`ltxv-2b-0.9.8-distilled.safetensors`),
NOT the LTX-2.3 22B stack (`eng_ltx_video` / `eng_ltx_av`) and NOT the forbidden
original `ltx-video-2b-v0.9.safetensors`. It is its own adapter/recipe: the 0.9.8
graph was captured from a LIVE `/object_info` + a functional in-process smoke on the
5080 (2026-07-20; see docs/2026-07-20-OTR-video-tiers/ltx_8gb_discovery.md).

The 0.9.8 all-in-one checkpoint carries MODEL + the video VAE **embedded** (no
separate VAE fetch); it has **no text encoder**, so the T5 is the shared
`t5xxl_fp16.safetensors` loaded through a separate `CLIPLoader` (type `ltxv`), which
for the 8GB tier defaults to `device='cpu'` (encode first on CPU, diffuse on GPU).
The graph (discovery-verified):

  CheckpointLoaderSimple(0.9.8) -> MODEL(+embedded VAE)
  CLIPLoader(t5xxl_fp16, type=ltxv, device=cpu) -> CLIPTextEncode x2 (pos/neg)
  ModelSamplingLTXV(max_shift, base_shift) -> MODEL
  LTXVImgToVideo(pos,neg,vae,image,w,h,length,strength) -> pos,neg,latent
  LTXVConditioning(frame_rate) -> pos,neg
  LTXVScheduler(steps,shift,stretch,terminal,latent) -> SIGMAS
  KSamplerSelect(euler) -> SAMPLER
  SamplerCustom(model,cfg,pos,neg,sampler,sigmas,latent) -> LATENT
  VAEDecode / VAEDecodeTiled -> IMAGE

Single-pass, NO upscaler (the 8GB routes stay single-pass; the internal x2 latent
upscaler is the 16GB ltx_video route only). SILENT output (V-1); OTR master audio is
muxed later by OTR_MasterAudioMux. Length is LTX's 8n+1 rule (min 9); a short render
is ping-pong-extended (CLIP-FILL) to the beat window -- never a freeze-hold.

Registered as a NORMAL selectable row: ``requires_flag=None``, empty ``default_roles``
(selectable, not a default). ORDINARY preflight ONLY -- checkpoint + T5 present +
a checkpoint sanity floor + node classes resolve at load. NO VRAM/NVML/vendor gate,
NO auto-fallback (operator directive 2026-07-20). Cold-import clean (V-12): module
scope imports only the stdlib + the dep-free shared helpers; torch / PIL / ffmpeg /
the ComfyUI node registry are imported LAZILY inside load / render_clip. UTF-8, no BOM.

Config (env; all optional): ``OTR_LTX_8GB_CKPT`` explicit checkpoint path;
``OTR_LTX_8GB_CKPT_NAME`` / ``OTR_LTX_8GB_T5_NAME`` loader basenames;
``OTR_LTX_8GB_CKPT_DIR`` / ``OTR_LTX_8GB_T5_DIR`` dir overrides; ``OTR_LTX_8GB_STEPS``
(default 8) / ``OTR_LTX_8GB_CFG`` (1.0) / ``OTR_LTX_8GB_SAMPLER`` (euler) /
``OTR_LTX_8GB_MAX_SHIFT`` (2.05) / ``OTR_LTX_8GB_BASE_SHIFT`` (0.95) /
``OTR_LTX_8GB_TERMINAL`` (0.1) / ``OTR_LTX_8GB_MAX_FRAMES`` (cap; 8n+1) /
``OTR_LTX_8GB_T5_DEVICE`` (default cpu) / ``OTR_LTX_8GB_TILED_VAE`` (default off) /
``OTR_LTX_8GB_NEGATIVE``.
"""
from __future__ import annotations

import logging
import os
from collections import namedtuple

from . import motion_common as _MC
from . import wan_shared as _WS
from .._otr_shared.role_compat import ROLES
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.video.ltx_8gb")

#: Default 8GB-tier checkpoint (0.9.8 distilled 2B all-in-one; VAE embedded, no TE).
_LTX8_DEFAULT_CKPT = "ltxv-2b-0.9.8-distilled.safetensors"
#: Shared T5 text encoder (already on disk; the 0.9.8 checkpoint carries no TE).
_LTX8_DEFAULT_T5 = "t5xxl_fp16.safetensors"
#: The checkpoint sanity floor (bytes): the real 0.9.8 distilled 2B is ~6.34 GiB;
#: anything under 4 GiB resolved under this basename is the WRONG file -- fail closed.
_LTX8_CKPT_MIN_BYTES = 4 * 1024 * 1024 * 1024

#: LTX temporal length rule: 8n+1, floor 9 (the min the smoke decoded). Cap is a
#: STATIC config (env OTR_LTX_8GB_MAX_FRAMES), never a live-VRAM adaptive resize
#: (S4 platform-portability: the render never resizes itself; CLIP-FILL loops a
#: short render up to the beat window instead).
_LTX8_MIN_FRAMES = 9
_LTX8_MAX_FRAMES_DEFAULT = 161            # 8*20+1; env-overridable per hardware
_LTX8_DEFAULT_W = 832
_LTX8_DEFAULT_H = 480
_LTX8_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, distorted, watermark, text, static")

#: The defined recipe-receipt string threaded into the manifest (S-B/E5).
RECIPE_LTX8_I2V = "ltx098_distilled_2b_i2v_single_pass"

#: The FROZEN per-beat resolution (B2). Everything a beat's HANDLES depend on,
#: read exactly ONCE and passed by value to identity, prepare, assert_usable and
#: graph construction. It exists because three accessors used to re-read
#: ``os.environ`` independently, and because ``BeatSession`` asks for the session
#: identity BEFORE ``prepare()`` installs the active profile -- so an identity
#: derived from live state would describe different things before and after the
#: load, which is exactly the drift the identity is supposed to detect.
LtxSessionConfig = namedtuple("LtxSessionConfig", (
    "engine", "recipe",
    "ckpt_token", "ckpt_path", "ckpt_receipt",
    "t5_token", "t5_path", "t5_receipt",
    "t5_device", "tiled_vae",
    "steps", "cfg", "sampler", "max_shift", "base_shift", "terminal",
    "max_frames",
))


class FileReceiptUnavailable(RuntimeError):
    """A resolved model file could not be stat'ed. Internal to this adapter --
    ``resolve_session_config`` converts it into a NAMED ``EngineUnusable``."""


def _same_file(a, b):
    """True when two path SPELLINGS name the same file on disk.

    ``os.path.abspath`` normalises separators and resolves relative paths, but
    it does NOT fold case and does NOT resolve junctions or symlinks. On Windows
    both matter: NTFS is case-insensitive, and this build reaches its own repo
    through a JUNCTION, with `extra_model_paths.yaml` the standard way a shared
    model store gets aliased. Comparing ``abspath`` strings therefore REFUSES
    correct configurations -- ``C:\\Models\\x`` vs ``c:\\models\\x`` is the same
    file and compares unequal.

    ``os.path.samefile`` is the real test (it compares what the OS resolves to,
    reparse points included) but it RAISES when either side is missing, so it
    cannot stand alone. Fall back to ``normcase(realpath(...))``, which folds
    case and resolves links without requiring the file to exist."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return (os.path.normcase(os.path.realpath(a))
                == os.path.normcase(os.path.realpath(b)))


def _file_receipt(path):
    """A BOUNDED, stable receipt for a model file: (basename, size, mtime_ns).

    Deliberately NOT a content hash: the identity is re-read before every
    segment, and hashing a 6.34 GiB checkpoint per segment would cost more than
    the render. Size + mtime catches a swapped or rebuilt weight, which is the
    drift this is guarding against.

    Raises ``FileReceiptUnavailable`` rather than a raw ``OSError`` when the file
    vanishes between resolution and stat (a concurrent re-download, a cleanup, an
    AV quarantine) -- the caller turns it into the same NAMED refusal every other
    failure in this path produces."""
    try:
        st = os.stat(path)
    except OSError as exc:
        raise FileReceiptUnavailable(
            "%s could not be stat'ed after it resolved (%s: %s) -- it moved or "
            "was removed between resolution and receipt"
            % (path, type(exc).__name__, exc))
    return (os.path.basename(path), int(st.st_size), int(st.st_mtime_ns))


def _ltx8_frame_length(target_frame_count, cap):
    """The FINAL LTX 0.9.8 graph length for a shot's frame ask: floor to
    ``_LTX8_MIN_FRAMES``, cap at ``cap`` (a static per-hardware ceiling), then snap
    to LTX's 8n+1 rule. Pure; CPU-tested. NO decode-floor-raise (0.9.8 core VAEDecode
    decodes the 9-frame minimum -- smoke-proven -- unlike the 2.3 tiled band)."""
    length = max(_LTX8_MIN_FRAMES, int(target_frame_count or _LTX8_MIN_FRAMES))
    cap = max(_LTX8_MIN_FRAMES, int(cap))
    if length > cap:
        length = cap
    return ((length - 1) // 8) * 8 + 1


#: S1 (2026-07-25) per-model still plan for ltx_8gb (spec section 3,
#: Shape A -- scene spine). FILE-LOCAL, fully declared. scene_open +
#: scene_beat + scene_character all WIDE + required=always; portrait per
#: subject at inherit_engine + not required.
_LTX_8GB_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry=("in-character cinematic medium shot, head "
                                   "and shoulders, face clearly visible, "
                                   "subject centred with natural headroom "
                                   "above the head (never crop the top of the "
                                   "head)"),
                 style_tail_policy="full"),
)


@register
class Ltx8gbEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """The ltx_8gb LTX-Video 0.9.8 distilled 2B image->video adapter (8GB tier)."""

    name = "ltx_8gb"
    family = "image_to_video"
    #: 16:9 (matches the menu suffix ``ltx_8gb (16:9)``): the director mints a WIDE
    #: init still (non-HuMo, non-mesh-portrait).
    render_aspect = "wide"
    #: S1 per-model still plan (see ``_LTX_8GB_STILL_PLAN`` above).
    still_plan = _LTX_8GB_STILL_PLAN
    # FLEXIBLE: eligible for every role -- role_compat is the real gate (it admits
    # ltx_8gb only where the role supplies the required init_image). Opening `roles`
    # lets the operator pick ltx_8gb for any still-bearing beat; required_inputs
    # still prevents a truly broken pick.
    roles = ROLES
    default_roles = ()
    required_inputs = ("init_image",)
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). 8n+1: 9 .. 161 (8*20+1).
    #: max_frames is the LITERAL 161, not ``_LTX8_MAX_FRAMES_DEFAULT`` read
    #: through ``OTR_LTX_8GB_MAX_FRAMES`` -- a contract that moves with the
    #: environment is not a contract. When the env disagrees with this number
    #: the engine must REFUSE, not quietly re-plan (enforced in chunk 7b).
    #: CONTINUITY: LoadImage -> LTXVImgToVideo(image, strength=1.0) hard-pins
    #: the still into the latent's first frame.
    frame_contract = FrameContract(
        min_frames=9,
        max_frames=161,
        quantum=8,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )
    # LTX-Video 0.9.x Open Weights License (Lightricks; HF license:other) -- commercial
    # use permitted below the revenue threshold (same revenue-capped community model
    # already treated as clean elsewhere; same LTX family as ltx_video/ltx_av).
    # NOTE: commercial_clean is NOT a selection gate -- it drives only the release-gate
    # non-blocking warning + the release filename tag. Operator confirms at license review.
    commercial_clean = True
    requires_flag = None                  # registry IS the menu; no flag gate
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    #: Portable sampler floor (core, cross-platform). assert_usable fails closed on
    #: anything else; the default is IN the set so an unset env self-passes.
    _PORTABLE_SAMPLERS = frozenset({"euler", "euler_ancestral", "dpmpp_2m"})
    _DEFAULT_SAMPLER = "euler"

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "decode"

    # ---- config resolution (env override -> box default) ----
    def _ckpt_name(self):
        return os.environ.get("OTR_LTX_8GB_CKPT_NAME") or _LTX8_DEFAULT_CKPT

    def _t5_name(self):
        return os.environ.get("OTR_LTX_8GB_T5_NAME") or _LTX8_DEFAULT_T5

    def _ckpt_path(self):
        """Resolved checkpoint path: explicit ``OTR_LTX_8GB_CKPT`` wins, else
        folder_paths / the checkpoints category (honours extra_model_paths). ``None``
        when absent everywhere (the offline invariant -- no runtime fetch)."""
        explicit = os.environ.get("OTR_LTX_8GB_CKPT")
        if explicit:
            return explicit if os.path.exists(explicit) else None
        return self._resolve_model_file(
            ("checkpoints",), self._ckpt_name(), "OTR_LTX_8GB_CKPT_DIR")

    def _t5_path(self):
        return self._resolve_model_file(
            ("text_encoders", "clip"), self._t5_name(), "OTR_LTX_8GB_T5_DIR")

    def _installed(self):
        return self._ckpt_path() is not None

    # ---- the FROZEN session config (B2) ----
    def _loader_token_path(self, categories, token, env_dir, env_explicit=None):
        """Where the LOADER NODE will actually find ``token``.

        ``_build_graph`` hands ``CheckpointLoaderSimple`` / ``CLIPLoader`` a BARE
        BASENAME, which ComfyUI resolves through ``folder_paths``. So the file the
        graph loads is whatever that TOKEN resolves to -- never an absolute path
        an env var happens to name. ``_ckpt_path()`` short-circuits on the explicit
        ``OTR_LTX_8GB_CKPT``, which means a receipt taken from it can describe a
        DIFFERENT file than the one that loads: the preflight passes, the identity
        is a lie, and the beat renders from a weight nobody recorded. Resolve by
        token here, and make a disagreeing override terminal rather than silent."""
        by_token = self._resolve_model_file(categories, token, env_dir)
        explicit = os.environ.get(env_explicit) if env_explicit else None
        if explicit:
            if by_token is None or not _same_file(explicit, by_token):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s=%r names a file the loader will NOT load: the graph passes "
                    "the basename %r, which resolves to %r. An explicit path can "
                    "only be honoured when it IS the token's resolution -- "
                    "otherwise the receipt describes one weight and the render "
                    "uses another. Point %s at the registered file, or drop it and "
                    "use %s."
                    % (env_explicit, explicit, token, by_token, env_explicit,
                       env_dir), kind="video")
        return by_token

    def resolve_session_config(self, profile=None):
        """Resolve, ONCE, every input a beat's handles depend on. Fail CLOSED.

        Called BEFORE the first ``session_identity()`` check and reused for
        ``prepare``, per-segment usability and graph construction, so no two
        readers can disagree. ``profile`` is accepted now and consulted for the
        levers once they have a profile channel (B6); today it is recorded so the
        signature does not have to change under callers later."""
        cfg = self._resolve_render_config()          # range-checked, fail-closed
        ckpt = self._loader_token_path(
            ("checkpoints",), self._ckpt_name(), "OTR_LTX_8GB_CKPT_DIR",
            env_explicit="OTR_LTX_8GB_CKPT")
        if ckpt is None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb checkpoint %r not found; fetch it via "
                "scripts/download_ltx_0_9_8.ps1 (or drop it in models/checkpoints)"
                % self._ckpt_name(), kind="video")
        t5 = self._loader_token_path(
            ("text_encoders", "clip"), self._t5_name(), "OTR_LTX_8GB_T5_DIR")
        if t5 is None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb text encoder %r absent -- the 0.9.8 checkpoint carries no "
                "text encoder, so the shared t5xxl_fp16 must be on disk (offline "
                "invariant, no runtime fetch); fix OTR_LTX_8GB_T5_NAME / "
                "OTR_LTX_8GB_T5_DIR" % self._t5_name(), kind="video")
        try:
            ckpt_receipt = _file_receipt(ckpt)
            t5_receipt = _file_receipt(t5)
        except FileReceiptUnavailable as exc:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb %s" % (exc,), kind="video")
        return LtxSessionConfig(
            engine=self.name, recipe=RECIPE_LTX8_I2V,
            ckpt_token=self._ckpt_name(), ckpt_path=ckpt,
            ckpt_receipt=ckpt_receipt,
            t5_token=self._t5_name(), t5_path=t5,
            t5_receipt=t5_receipt,
            t5_device=self._t5_device(), tiled_vae=self._tiled_vae(),
            steps=cfg["steps"], cfg=cfg["cfg"], sampler=cfg["sampler"],
            max_shift=cfg["max_shift"], base_shift=cfg["base_shift"],
            terminal=cfg["terminal"], max_frames=cfg["max_frames"])

    def session_identity(self):
        """What this adapter's HANDLES are -- engine, recipe, weights.

        ``BeatSession`` REFUSES a multi-segment beat from an engine that cannot
        answer this (``SessionIdentityUnavailable``, no fallback), because
        nothing could then prove segment N renders with the model segment 1
        loaded. Until this existed, every multi-segment beat was refused for
        every engine in the roster.

        Deliberately PRE-LOAD STABLE: it is read once before the weights land,
        again after ``prepare()``, and before every segment, so it may only
        describe things that do not change across the load. It carries the
        model-sampling shifts because those are baked into the hoisted patcher,
        and it EXCLUDES per-segment state -- prompt, seed, frame count, canvas --
        along with ``tiled_vae``, which selects the decode NODE CLASS rather than
        anything about the handles.

        Not cached, by design: the whole job is to notice a weight that MOVED,
        so the receipts are re-stat'ed on every ask (two stats, not a hash)."""
        cfg = self.resolve_session_config()
        return (cfg.engine, cfg.recipe,
                cfg.ckpt_token, repr(cfg.ckpt_receipt),
                cfg.t5_token, repr(cfg.t5_receipt),
                cfg.t5_device,
                "max_shift=%s" % (cfg.max_shift,),
                "base_shift=%s" % (cfg.base_shift,))

    def _tiled_vae(self):
        """Whether to decode through ``VAEDecodeTiled`` (default OFF: 0.9.8 core
        VAEDecode handles the 8GB peak at the smoke canvas; C3 tuning may flip this
        via env if a larger canvas needs it). Truthy {1,true,yes,on} enables it."""
        return (os.environ.get("OTR_LTX_8GB_TILED_VAE", "0").strip().lower()
                in ("1", "true", "yes", "on"))

    def _t5_device(self):
        """T5 CLIPLoader device: default ``cpu`` for the 8GB tier (t5xxl_fp16 alone
        is ~9 GB, so it encodes on CPU first, then diffusion runs on the GPU). The
        offload-on-vs-off VRAM measurement (C3) may flip this via OTR_LTX_8GB_T5_DEVICE."""
        dev = (os.environ.get("OTR_LTX_8GB_T5_DEVICE") or "cpu").strip().lower()
        return dev if dev in ("cpu", "default") else "cpu"

    def _resolve_render_config(self):
        """Parse + RANGE-CHECK the render knobs ONCE (shared by assert_usable and
        _build_graph). A bad env value fails CLOSED here with a named MALFORMED_CONFIG,
        never a raw int()/float() crash mid-render. The sampler is validated against
        the portable floor whitelist."""
        def _num(env, dflt, lo, hi, cast):
            raw = os.environ.get(env)
            if raw is None or raw == "":
                return cast(dflt)
            try:
                val = cast(raw)
            except (TypeError, ValueError):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s=%r is not a valid number" % (env, raw), kind="video")
            if not (lo <= val <= hi):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s=%s out of range [%s, %s]" % (env, val, lo, hi), kind="video")
            return val

        sampler = (os.environ.get("OTR_LTX_8GB_SAMPLER")
                   or self._DEFAULT_SAMPLER).strip()
        if sampler not in self._PORTABLE_SAMPLERS:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "OTR_LTX_8GB_SAMPLER=%r is not in the portable floor whitelist %s"
                % (sampler, sorted(self._PORTABLE_SAMPLERS)), kind="video")
        return {
            "steps": _num("OTR_LTX_8GB_STEPS", 8, 1, 100, int),
            "cfg": _num("OTR_LTX_8GB_CFG", 1.0, 0.0, 30.0, float),
            "max_shift": _num("OTR_LTX_8GB_MAX_SHIFT", 2.05, 0.0, 100.0, float),
            "base_shift": _num("OTR_LTX_8GB_BASE_SHIFT", 0.95, 0.0, 100.0, float),
            "terminal": _num("OTR_LTX_8GB_TERMINAL", 0.1, 0.0, 0.99, float),
            "max_frames": _num("OTR_LTX_8GB_MAX_FRAMES", _LTX8_MAX_FRAMES_DEFAULT,
                               _LTX8_MIN_FRAMES, 16384, int),
            "sampler": sampler,
        }

    # ---- usability (fail-closed BEFORE any forward; ordinary preflight ONLY) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordinary asset preflight -- NO VRAM/NVML/vendor gate, NO fallback
        (operator directive 2026-07-20). Fail CLOSED on a bad render knob, then a
        missing checkpoint, a wrong-size checkpoint, or a missing T5."""
        self._resolve_render_config()                 # range-checked, fail-closed
        ckpt = self._ckpt_path()
        if ckpt is None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb checkpoint %r not found; fetch it via "
                "scripts/download_ltx_0_9_8.ps1 (or set OTR_LTX_8GB_CKPT / drop it "
                "in models/checkpoints)" % self._ckpt_name(), kind="video")
        try:
            size = os.path.getsize(ckpt)
        except OSError:
            size = 0
        if size < _LTX8_CKPT_MIN_BYTES:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb checkpoint %r is only %d bytes (< %d floor) -- this is not "
                "the 0.9.8 distilled 2B all-in-one weight; re-fetch via "
                "scripts/download_ltx_0_9_8.ps1"
                % (ckpt, size, _LTX8_CKPT_MIN_BYTES), kind="video")
        if self._t5_path() is None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb text encoder %r absent -- the 0.9.8 checkpoint carries no "
                "text encoder, so the shared t5xxl_fp16 must be on disk (offline "
                "invariant, no runtime fetch); fix OTR_LTX_8GB_T5_NAME / "
                "OTR_LTX_8GB_T5_DIR" % self._t5_name(), kind="video")
        return self.name

    # ---- in-process graph spec (0.9.8 distilled; discovery + smoke verified) ----
    def _node_candidates(self):
        decode_cls = (("VAEDecodeTiled",) if self._tiled_vae() else ("VAEDecode",))
        return {
            "ckpt": ("CheckpointLoaderSimple",),
            "clip": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "loadimage": ("LoadImage",),
            "modelsampling": ("ModelSamplingLTXV",),
            "img2vid": ("LTXVImgToVideo",),
            "cond": ("LTXVConditioning",),
            "sched": ("LTXVScheduler",),
            "sampler": ("KSamplerSelect",),
            "sample": ("SamplerCustom",),
            "decode": decode_cls,
        }

    def _decode_inputs(self, W):
        """VAEDecode inputs; the tiled path adds the schema-verified tile/temporal
        knobs (env-overridable). All grounded vs the discovery /object_info."""
        base = {"samples": W("sample", 0), "vae": W("ckpt", 2)}
        if not self._tiled_vae():
            return base
        def _i(env, dflt):
            try:
                return int(os.environ.get(env, str(dflt)))
            except (TypeError, ValueError):
                return dflt
        base.update({
            "tile_size": _i("OTR_LTX_8GB_VAE_TILE", 512),
            "overlap": _i("OTR_LTX_8GB_VAE_OVERLAP", 64),
            "temporal_size": _i("OTR_LTX_8GB_VAE_TEMPORAL", 16),
            "temporal_overlap": _i("OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP", 8),
        })
        return base

    def _build_graph(self, request, image_name, plan, length, width, height):
        """The declarative LTX 0.9.8 distilled I2V graph (wrapper_bridge.run_graph
        format). The 0.9.8 all-in-one gives MODEL(0)+embedded VAE(2); the T5 is the
        separate CLIPLoader (device cpu on the 8GB tier)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        cfg = self._resolve_render_config()
        positive = get("text_prompt") or "subtle natural motion, cinematic light"
        negative = (get("negative_prompt")
                    or os.environ.get("OTR_LTX_8GB_NEGATIVE", _LTX8_DEFAULT_NEGATIVE))
        return {
            "ckpt": {"class": "ckpt", "inputs": {"ckpt_name": self._ckpt_name()}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": self._t5_name(), "type": "ltxv",
                                "device": self._t5_device()}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "modelsampling": {"class": "modelsampling",
                              "inputs": {"model": W("ckpt", 0),
                                         "max_shift": cfg["max_shift"],
                                         "base_shift": cfg["base_shift"]}},
            "img2vid": {"class": "img2vid",
                        "inputs": {"positive": W("pos", 0), "negative": W("neg", 0),
                                   "vae": W("ckpt", 2), "image": W("loadimage", 0),
                                   "width": int(width), "height": int(height),
                                   "length": int(length), "batch_size": 1,
                                   "strength": 1.0}},
            "cond": {"class": "cond",
                     "inputs": {"positive": W("img2vid", 0),
                                "negative": W("img2vid", 1),
                                "frame_rate": float(self.target_fps)}},
            "sched": {"class": "sched",
                      "inputs": {"steps": cfg["steps"], "max_shift": cfg["max_shift"],
                                 "base_shift": cfg["base_shift"], "stretch": True,
                                 "terminal": cfg["terminal"],
                                 "latent": W("img2vid", 2)}},
            "sampler": {"class": "sampler",
                        "inputs": {"sampler_name": cfg["sampler"]}},
            "sample": {"class": "sample",
                       "inputs": {"model": W("modelsampling", 0), "add_noise": True,
                                  "noise_seed": int(plan.get("seed", 0)),
                                  "cfg": cfg["cfg"], "positive": W("cond", 0),
                                  "negative": W("cond", 1), "sampler": W("sampler", 0),
                                  "sigmas": W("sched", 0),
                                  "latent_image": W("img2vid", 2)}},
            "decode": {"class": "decode", "inputs": self._decode_inputs(W)},
        }

    # ---- residency ----
    def load(self):
        """Fail CLOSED until installed, then resolve the installed ComfyUI node
        classes (0.9.8 core LTX nodes). Weights load when the loader nodes execute
        in render_clip."""
        if not self._installed():
            raise RuntimeError(
                "ltx_8gb not installed: checkpoint %r missing -- fetch it via "
                "scripts/download_ltx_0_9_8.ps1" % self._ckpt_name())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE image->video clip via the in-process LTX 0.9.8 graph: stage the
        init image (no silent stretch, N9), execute the graph, encode the decoded
        IMAGE batch to a SILENT bt709 clip (V-1), CLIP-FILL to the beat window,
        retain the MODEL patcher for V-4 teardown, and thread the measured VRAM peak
        into the recipe receipt. M7 ffprobe-proves the silent-clip contract."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "ltx_8gb requires init_image (got %r)" % plan["init_image"])
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        cap = self._resolve_render_config()["max_frames"]
        length = _ltx8_frame_length(plan["target_frame_count"], cap)
        graph = self._build_graph(request, image_name, plan, length, width, height)
        # free_after_use: the T5 text-encode frees before the sampler; the checkpoint
        # (MODEL + embedded VAE) + the model-sampling patch + the terminal are kept.
        # The NVML peak probe spans the whole render window (telemetry only -- no
        # ceiling enforcement; the operator's tier JSON owns the OOM budget).
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(
                graph, classes, free_after_use=True,
                keep={"ckpt", "modelsampling", self._TERMINAL})
            images = results[self._TERMINAL][0]               # VAEDecode IMAGE batch
        finally:
            render_peak = probe.stop()
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("modelsampling", "ckpt"):
            out = results.get(nid)
            if not out:
                continue
            model = out[0]
            if id(model) not in seen and callable(getattr(model, "detach", None)):
                bucket.append(model)
                seen.add(id(model))
        frames = _wb.images_to_uint8(images)
        # CLIP-FILL: ping-pong-extend the (possibly short) render up to the beat's
        # audio-derived target so the composite fills the beat with motion instead of
        # holding the last frame. A no-op when the native render already meets target.
        target_frames = int(plan.get("target_frame_count") or 0)
        n_native = len(frames)
        if target_frames > n_native:
            frames = _wb.extend_frames_to_target(frames, target_frames)
            _LOG.warning(
                "[OTR video] ltx_8gb CLIP-FILL: rendered %d frame(s) -> ping-pong "
                "extended to %d (beat target %d) @ %dx%d so the beat is FILLED with "
                "motion (no hold-last-frame freeze)",
                n_native, len(frames), target_frames, width, height)
        out_path = otr_engine_tmp_mp4("otr_ltx_8gb_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7: PROVE the silent-clip color/stream contract on the emitted mp4.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] ltx_8gb VRAM render-phase peak %s MB @ %dx%d len=%d",
                      render_peak, width, height, n)
        return {"out_path": path, "frame_count": n,
                "vram_peak_mb": render_peak, "recipe": RECIPE_LTX8_I2V,
                "render_canvas": "%dx%d" % (int(width), int(height))}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)

    def _clip_from_raw(self, raw, request):
        """The self-declared silent CanonicalClip dict canonicalize() returns
        (bt709 / yuv420p; frame_count is the integer timing authority). At parity
        with wan/ltx PLUS the S-B/E5 recipe-receipt keys the render batch reads via
        clip.get(...) -- recipe / render_canvas / vram_peak_mb (render_driver.py:
        2817-2824). NO CanonicalClip schema change (the returned dict is consumed as
        a plain dict, not through the extra='forbid' model)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or ("%s_clip" % self.name),
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            # PASS-PM: the REAL render-window NVML peak (None off the GPU box).
            "vram_peak_mb": raw.get("vram_peak_mb"),
            # S-B/E5 recipe receipt (None for a raw that did not carry it).
            "recipe": raw.get("recipe"),
            "render_canvas": raw.get("render_canvas"),
        }


__all__ = ["Ltx8gbEngine", "RECIPE_LTX8_I2V"]
