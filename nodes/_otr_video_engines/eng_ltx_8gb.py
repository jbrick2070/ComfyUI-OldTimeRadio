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
muxed later by OTR_MasterAudioMux. Length is LTX's 8n+1 rule (min 9): an ask off
that grid renders the next legal rung UP and the surplus is TRIMMED in real
frames, so every frame delivered is a rendered frame in order. An ask this
engine cannot reach -- past its declared 161 or past ``OTR_LTX_8GB_MAX_FRAMES``
-- is REFUSED before anything is staged. There is no ping-pong on this lane
(deleted B4, 2026-07-27): padding a short render back up to the ask let a
render that did not happen pass the plan-vs-output count gate, and on a
``strict_first_frame`` lane the next chained segment would have begun on a
mirrored frame. (The trailing "WAN keeps its extension -- it renders short on
purpose" is HISTORY: WAN's ping-pong was deleted under the operator's no-mirror
ruling and it now refuses such a beat rather than padding it. There is no
extension left on any lane.)

Registered as a NORMAL selectable row: ``requires_flag=None``, empty ``default_roles``
(selectable, not a default). ORDINARY preflight ONLY -- checkpoint + T5 present +
a checkpoint sanity floor + node classes resolve at load. NO VRAM/NVML/vendor gate,
NO auto-fallback (operator directive 2026-07-20). Cold-import clean (V-12): module
scope imports only the stdlib + the dep-free shared helpers; torch / PIL / ffmpeg /
the ComfyUI node registry are imported LAZILY inside load / render_clip. UTF-8, no BOM.

Config (env). READ THE SPLIT -- since B6 (2026-07-27) most of these do NOT bind
on a production leg:

* STILL LIVE, every leg: ``OTR_LTX_8GB_CKPT`` explicit checkpoint path;
  ``OTR_LTX_8GB_CKPT_NAME`` / ``OTR_LTX_8GB_T5_NAME`` loader basenames;
  ``OTR_LTX_8GB_CKPT_DIR`` / ``OTR_LTX_8GB_T5_DIR`` DEPRECATED dir overrides
  (see the paragraph below before reaching for either); and
  ``OTR_LTX_8GB_MAX_FRAMES`` (render-length ceiling; 8n+1) -- a CEILING, not a
  recipe value, so it keeps its channel and its fail-closed range check.
* FROZEN IN CODE as ``LTX8_RECIPE`` (v2, measured), ignored-with-a-warning:
  ``OTR_LTX_8GB_STEPS`` / ``_CFG`` / ``_SAMPLER`` / ``_MAX_SHIFT`` /
  ``_BASE_SHIFT`` / ``_TERMINAL`` / ``_T5_DEVICE`` / ``_TILED_VAE`` /
  ``_NEGATIVE`` / ``_VAE_TILE`` / ``_VAE_OVERLAP`` / ``_VAE_TEMPORAL`` /
  ``_VAE_TEMPORAL_OVERLAP``. Their defaults live in that dict, not here, so
  this list cannot drift from the values.
* THE CONSENT ACT: set ``OTR_LTX_8GB_PREQUALIFICATION=1`` and the frozen knobs
  bind again, range-checked and fail-closed, for a MEASUREMENT run -- whose
  clips stamp a ``+prequalification`` recipe receipt so a sweep artifact is
  never mistaken for a production one.

Why the freeze: a production episode is submitted to an ALREADY-BOOTED server,
so a knob exported at launch cannot bind the work (``PBUG-20260723-02``). Code
binds on every leg; an environment cannot.

``OTR_LTX_8GB_CKPT`` and the two ``*_DIR`` overrides are CHECKED against the
loader's own token resolution and REFUSE with MALFORMED_CONFIG when they
disagree. The graph passes ComfyUI a BARE BASENAME, so a path an env var names
was never what actually rendered -- honouring it silently would make the receipt
describe one weight while the render used another. To point this build at a
model store, register the folder in ``extra_model_paths.yaml``: that is the
channel that reaches the loader.
"""
from __future__ import annotations

import logging
import os
from collections import namedtuple

from . import motion_common as _MC
from . import recipe_departures as _RD
from . import wan_shared as _WS
from .._otr_shared.role_compat import ROLES
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.video.ltx_8gb")

# PROMPT-STYLE OVERLAY: this engine has NO pair of its own, ON PURPOSE. The
# RESEARCH doc (2026-08-17-per-engine-prompt-style-guide-RESEARCH.md, in the docs
# dir -- named WITHOUT a path prefix because `tools/engine_matrix.py` scrapes
# engine sources for cap-evidence citations and a phrasing doc is not frame
# evidence) treats "ltx_video / ltx_8gb" as ONE block, and the shared facts are the
# ones the directive is actually built from: same family, same cfg-1.0 distilled
# default (so the negative is inert on both), same i2v-anchor doctrine (the still
# carries the LOOK, the prompt moves), same tight char budget. The authority is
# `eng_ltx_video.PROMPT_STYLE_DIRECTIVE` / `.PROMPT_STYLE_NOTES`. Read it there.
#
# THE ENCODERS ARE NOT THE SAME, and an earlier version of this comment said they
# were -- a Sonnet QA pass caught it. This tier's 0.9.8 checkpoint carries no text
# encoder and borrows the shared T5-XXL (`_LTX8_DEFAULT_T5`); `eng_ltx_video` runs
# GEMMA-3 through `LTXAVTextEncoderLoader`. Two different encoder architectures.
# It does not change the shared directive, because not one clause of it is
# encoder-specific -- it is all i2v doctrine, cfg and budget. But it IS the
# condition for splitting: the moment a clause turns encoder-specific (anything of
# the "full grammar, not comma-separated tags" kind, which is exactly what
# distinguishes an LLM encoder from a CLIP-lineage one on the stills side), this
# tier needs its own pair rather than this pointer.
#
# DO NOT PASTE A COPY HERE. Two byte-identical strings in two files is the
# duplicate-drift shape D-BIS finding 2 already flags in the negative constants:
# the same 7-term boilerplate exists in four copies and two of them silently
# diverged with no recorded reason. `tests/test_prompt_style_directives.py`
# asserts this module defines neither constant, so a well-meaning copy fails the
# suite instead of drifting quietly. A deliberate DEPARTURE for the 8GB tier is a
# real possibility later -- if it happens, it gets its own pair plus a recorded
# reason in the notes, which is what makes it a departure rather than a drift.

#: Default 8GB-tier checkpoint (0.9.8 distilled 2B all-in-one; VAE embedded, no TE).
_LTX8_DEFAULT_CKPT = "ltxv-2b-0.9.8-distilled.safetensors"
#: Shared T5 text encoder (already on disk; the 0.9.8 checkpoint carries no TE).
_LTX8_DEFAULT_T5 = "t5xxl_fp16.safetensors"
#: The checkpoint sanity floor (bytes): the real 0.9.8 distilled 2B is ~6.34 GiB;
#: anything under 4 GiB resolved under this basename is the WRONG file -- fail closed.
_LTX8_CKPT_MIN_BYTES = 4 * 1024 * 1024 * 1024

#: LTX temporal length rule: 8n+1, floor 9 (the min the smoke decoded). Cap is a
#: STATIC config (env OTR_LTX_8GB_MAX_FRAMES), never a live-VRAM adaptive resize
#: (S4 platform-portability: the render never resizes itself). A render that
#: cannot reach the ask is REFUSED, not padded (B4, 2026-07-27).
#:
#: This number is ALSO declared on ``Ltx8gbEngine.frame_contract`` as
#: ``min_frames``, and the contract is the authority every length decision goes
#: through. This constant survives for ONE job: the lower bound
#: ``_resolve_render_config`` range-checks ``OTR_LTX_8GB_MAX_FRAMES`` against.
#: ``tests/test_ltx_8gb_graph_and_loads.py`` pins the two equal so they cannot
#: drift into a config check that accepts a cap below the real floor.
_LTX8_MIN_FRAMES = 9
_LTX8_MAX_FRAMES_DEFAULT = 161            # 8*20+1; env-overridable per hardware
_LTX8_DEFAULT_W = 832
_LTX8_DEFAULT_H = 480
_LTX8_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, distorted, watermark, text, static")

#: The defined recipe-receipt string threaded into the manifest (S-B/E5).
#:
#: THE VERSION LIVES IN THE STRING (B6, 2026-07-27), not in a separate constant.
#: This value rides `_clip_from_raw` -> the manifest row -> the render-batch
#: receipt -> `stamp_durable(meta.render_engines)`, so it is what a PUBLISHED
#: EPISODE's DURABLE LEDGER can be asked "which recipe rendered you". (It also
#: reaches `otr_credits_roll`, which builds a `video_suffix` from it -- but
#: `_draw_models` does not currently draw that field, so the recipe is in the
#: ledger, not yet on the burned-in card. Tracked in GO_FORWARD open bugs.)
#: A bare `LTX8_RECIPE_VERSION` constant that never reached that string would
#: leave the recipe invisible to every consumer that matters -- an unowned
#: ledger field in all but name. It is also `cfg.recipe` in `session_identity`,
#: so a version bump moves the identity for free.
RECIPE_LTX8_I2V = "ltx098_distilled_2b_i2v_single_pass_v2"

#: THE CONSENT ACT that re-opens the recipe knobs (B6). One explicit env var,
#: the same shape as the client-bank build's `--activate`: an operator who
#: means to sweep says so, and a production box that never sets it gets the
#: frozen recipe no matter how the server booted.
PREQUALIFICATION_ENV = "OTR_LTX_8GB_PREQUALIFICATION"

#: What a MEASUREMENT run stamps onto its clips instead of the frozen name.
#: See ``recipe_receipt`` -- a sweep's artifacts must be distinguishable from
#: production's in the durable ledger, because they are not the same recipe.
#:
#: SOURCED, not spelled (LANE 2): the same mark is stamped by the WAN adapters
#: through their own lane, and two literals with one value is how a ledger grows
#: two dialects. ``recipe_departures`` owns the format for every adapter.
PREQUALIFICATION_RECIPE_SUFFIX = _RD.PREQUALIFICATION_SUFFIX

#: THE FROZEN ltx_8gb RECIPE, v1 (B6, 2026-07-27).
#:
#: WHY IT EXISTS. The profile schema accepts only `device_policy`,
#: `dtype_policy` and `max_render_frames`, so this tier's real levers -- T5
#: device, tiled VAE decode, the sampling knobs -- have NO end-to-end channel.
#: They were read from `os.environ`, and per `PBUG-20260723-02` a production
#: episode is submitted to an ALREADY-BOOTED server, so a profile's
#: `launch.env` can never reach it and `otr_8gb_ltx.json`'s is empty. The
#: tier's recipe was therefore whatever the server happened to boot with.
#: Code binds on every leg regardless -- that is the whole point of that PBUG's
#: Bible rule, and this dict is it.
#:
#: WHAT v1 IS, STATED HONESTLY. These are TODAY'S SHIPPED DEFAULTS, not a
#: measured selection. The judgment orders "build mechanics first, MEASURE
#: second, freeze third", and no measurement has happened -- prequalification
#: is the next step. Freezing them now is behaviour-preserving on any box that
#: did not set the env vars, and it is reversible: prequalification measures
#: and produces v2. Each value already has a recorded reason -- the T5 offloads
#: to CPU because `t5xxl_fp16` alone is ~9 GB (load-bearing, not an
#: optimisation), and tiled VAE is OFF because core `VAEDecode` handles the
#: 8 GB peak at the smoke canvas.
#:
#: `max_frames` IS DELIBERATELY NOT HERE. It is a render-length CEILING, not a
#: recipe knob: B3 gave the tier ceiling its own profile -> ledger channel and
#: B4's pre-render refusal reads the env cap to make a plan-vs-box
#: disagreement terminal. Folding it in would silence that refusal, and
#: `test_an_ask_ABOVE_the_cap_is_refused_before_anything_is_staged` is the
#: trip-wire that catches anyone trying.
LTX8_RECIPE_V1 = {
    "steps": 8,
    "cfg": 1.0,
    "max_shift": 2.05,
    "base_shift": 0.95,
    "terminal": 0.1,
    "sampler": "euler",
    "t5_device": "cpu",
    "tiled_vae": False,
    #: The negative conditioning text. A RENDER INPUT, so it belongs here for
    #: the same reason the samplers do: read from `os.environ` it made two
    #: boxes produce visibly different clips that both stamped the same recipe
    #: receipt. A per-shot `negative_prompt` on the request still wins -- that
    #: is the director's channel, and it travels WITH the work rather than with
    #: the server's boot.
    "negative": _LTX8_DEFAULT_NEGATIVE,
    #: The tiled-decode geometry. Only consumed while `tiled_vae` is True, so
    #: these are inert today -- and that is exactly why they are frozen NOW:
    #: the day a measured v2 flips tiled decode on, four env-driven render
    #: inputs would otherwise come live again with no demotion notice and no
    #: receipt, quietly re-opening the hole this chunk closed.
    "vae_tile": 512,
    "vae_overlap": 64,
    "vae_temporal": 16,
    "vae_temporal_overlap": 8,
}

#: THE ACTIVE ltx_8gb RECIPE, v2 -- MEASURED, 2026-07-27 (prequalification).
#:
#: v1 was today's shipped defaults, stated honestly as such. This is the
#: measured selection that replaces it, from a four-cell sweep at the
#: production canvas 512x288 -- the first time this tier ever rendered live at
#: its own declared canvas. Every cell was a full canonical leg
#: (`RESULT SUCCESS` + `obs_publish OK` + the asset on disk), and the peaks
#: below are the adapter's own render-phase `vram_peak_mb`:
#:
#:   cell  t5_device  tiled_vae   shots   min MB   max MB   SPREAD   wall
#:   A     cpu        False          11     8662    10859     2197    842s
#:   B     default    False          12    11163    16127     4964    744s
#:   C     cpu        True           10     8241     8278       37    824s  <-
#:   D     default    True           11    11062    16086     5024    765s
#:
#: WHAT CHANGED FROM v1, AND WHY -- `tiled_vae` False -> True.
#: v1 kept tiled decode off because core `VAEDecode` "handles the 8 GB peak at
#: the smoke canvas". It does; it just costs 2.6 GB more to do it. The
#: decisive column is not the minimum, it is the SPREAD: with tiled decode ON
#: the peak is flat at 8241-8278 MB across every clip length the sweep
#: produced (17 to 161 frames), while OFF it climbs with length -- 8662 MB at
#: len=33 up to 10859 MB at len=161. An 8 GB tier needs a ceiling a long beat
#: cannot grow through, and only tiled decode gives one. It costs no wall
#: clock (824s vs 842s, inside the noise of a different episode).
#:
#: WHAT DID NOT CHANGE -- `t5_device` stays "cpu", now with a number behind
#: the rationale rather than an argument. On GPU the peak lands at 16.0-16.1 GB
#: on a 16.3 GB card, so an 8 GB box does not render at all. v1 called this
#: load-bearing rather than an optimisation; the sweep agrees.
#:
#: HONEST LIMIT OF THIS MEASUREMENT: `VramPeakProbe` samples MACHINE-WIDE NVML
#: usage, and the sweep ran unclamped, so these absolutes include whatever else
#: was resident and are NOT a proof that the winner fits in 8 GB. What they
#: support is the RANKING, which is what selects a recipe. A clamped
#: confirmation of the winner is still owed -- see GO_FORWARD.
#:
#: v1 IS KEPT, NOT EDITED. Receipts stamped `..._v1` are on disk in this
#: repo's own episode tree; a v1 dict that had been mutated into v2's values
#: would make those receipts uninterpretable and would quietly rewrite what
#: the regression fixtures pin.
LTX8_RECIPE_V2 = {
    "steps": 8,
    "cfg": 1.0,
    "max_shift": 2.05,
    "base_shift": 0.95,
    "terminal": 0.1,
    "sampler": "euler",
    "t5_device": "cpu",
    "tiled_vae": True,
    "negative": _LTX8_DEFAULT_NEGATIVE,
    "vae_tile": 512,
    "vae_overlap": 64,
    "vae_temporal": 16,
    "vae_temporal_overlap": 8,
}

#: THE ONE NAME EVERY CONSUMER READS. Bumping a recipe is repointing this and
#: the version inside `RECIPE_LTX8_I2V` -- never editing a versioned dict in
#: place. Kept as a separate binding so `LTX8_RECIPE_V1` stays readable as
#: history and the diff of a future v3 is one line plus a new dict.
LTX8_RECIPE = LTX8_RECIPE_V2

#: The env var each frozen field was read from, kept so the demotion can NAME
#: what it is ignoring. Presence is all this map is used for outside
#: prequalification -- see `_ignored_override_keys`.
_RECIPE_ENV_KEYS = {
    "steps": "OTR_LTX_8GB_STEPS",
    "cfg": "OTR_LTX_8GB_CFG",
    "max_shift": "OTR_LTX_8GB_MAX_SHIFT",
    "base_shift": "OTR_LTX_8GB_BASE_SHIFT",
    "terminal": "OTR_LTX_8GB_TERMINAL",
    "sampler": "OTR_LTX_8GB_SAMPLER",
    "t5_device": "OTR_LTX_8GB_T5_DEVICE",
    "tiled_vae": "OTR_LTX_8GB_TILED_VAE",
    "negative": "OTR_LTX_8GB_NEGATIVE",
    "vae_tile": "OTR_LTX_8GB_VAE_TILE",
    "vae_overlap": "OTR_LTX_8GB_VAE_OVERLAP",
    "vae_temporal": "OTR_LTX_8GB_VAE_TEMPORAL",
    "vae_temporal_overlap": "OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP",
}

#: Per-knob bounds for the tiled-decode geometry, from the LIVE ``/object_info``
#: capture of 2026-07-20 (docs/2026-07-20-OTR-video-tiers/ltx_8gb_discovery.json):
#: ``VAEDecodeTiled`` declares tile_size min 64, overlap min 0, temporal_size
#: min 8, temporal_overlap min 4. A value under the NODE'S OWN floor is a render
#: that dies inside ComfyUI, so it is refused here like every other knob rather
#: than handed over to fail late.
_VAE_TILE_BOUNDS = {
    "vae_tile": (64, 4096),
    "vae_overlap": (0, 4096),
    "vae_temporal": (8, 4096),
    "vae_temporal_overlap": (4, 4096),
}

_TRUTHY = ("1", "true", "yes", "on")
#: The explicit NO spellings. A consent-act knob must be recognisably yes or
#: recognisably no -- "anything not yes means no" is how a typo becomes a
#: silently different measurement.
_FALSY = ("0", "false", "no", "off")
#: The devices the T5 CLIPLoader may be pointed at. Anything else is a typo,
#: and under the consent act a typo stops the sweep rather than clamping.
_T5_DEVICES = ("cpu", "default")


def _prequalification_active():
    """True when the operator has explicitly opened the recipe knobs.

    Deliberately NOT inferred from the absence of a ledger or from any other
    ambient condition: a signal you can arrive at by accident is one a
    production leg can arrive at by accident."""
    return (os.environ.get(PREQUALIFICATION_ENV, "") or "").strip().lower() \
        in _TRUTHY


def _ignored_override_keys():
    """Which frozen knobs the environment is trying to set, by NAME ONLY.

    PRESENCE, NEVER PARSING, and that is load-bearing rather than tidy. If this
    parsed, a stale malformed `OTR_LTX_8GB_STEPS=not-a-number` left in a
    long-booted server's environment would raise MALFORMED_CONFIG and kill a
    production leg over a knob that has NO EFFECT on that leg -- the precise
    shape `PBUG-20260723-02` says must not happen. Outside prequalification
    these values cannot bind, so they cannot be wrong; they can only be
    ignored, and the operator is told."""
    return sorted(env for env in _RECIPE_ENV_KEYS.values()
                  if os.environ.get(env) not in (None, ""))


def recipe_receipt(departed=None):
    """The recipe string a rendered clip is STAMPED with.

    NOT simply ``RECIPE_LTX8_I2V``, and the difference is a ledger-integrity
    one. Under prequalification the knobs genuinely bind, so the clip on disk
    may share NONE of the frozen values -- while the receipt rides
    ``_clip_from_raw`` -> the manifest row ->
    ``stamp_durable(meta.render_engines)``, which is a DURABLE ledger a
    published episode carries. Stamping the frozen name onto a sweep artifact
    would make a measurement indistinguishable from production in the one
    record that outlives the run.

    So the consent act marks its own output. It also moves ``session_identity``
    (the recipe is element [1]), which is correct: a sweep segment and a
    production segment must never be mistaken for the same session -- and since
    LANE 2, neither may two sweep CELLS.

    ``departed`` is the mapping of knobs this cell actually changed. It is
    OPTIONAL because a caller with no instance in hand (a test, a log line)
    legitimately wants the bare mark, and REQUIRED IN PRACTICE at the two stamp
    sites, which both go through ``Ltx8gbEngine._recipe_receipt`` --
    ``test_EVERY_stamp_goes_through_the_receipt_helper`` is the source-level
    guard that keeps them there. Passing nothing yields exactly the pre-LANE-2
    receipt, so the fallback is the old behaviour rather than a new silence."""
    if _prequalification_active():
        return RECIPE_LTX8_I2V + _RD.format_suffix(departed or {})
    return RECIPE_LTX8_I2V

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


# ``_ltx8_frame_length`` LIVED HERE and is DELETED (B4, 2026-07-27).
#
# It snapped an ask DOWN to the nearest legal 8n+1 length and clamped it to the
# env cap, which is exactly why the CLIP-FILL ping-pong existed: something had
# to put the missing frames back. Removing the pad without removing the
# snap-down would have shipped a short clip instead.
#
# Its two jobs now have explicit owners, which is the whole point of retiring
# it rather than leaving it dead:
#   * the LADDER is owned by ``Ltx8gbEngine.frame_contract`` --
#     ``smallest_legal_at_least`` walks the same 9 + 8k rungs the coverage
#     planner partitions against, so the adapter and the planner cannot
#     disagree about what a legal length is. It snaps UP, and ``render_clip``
#     trims the surplus in REAL frames.
#   * the CAP is owned by an explicit refusal in ``render_clip``, before
#     anything is staged. An ask the engine cannot reach is an error, not a
#     number to quietly shrink.


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
                     ("cinematic three-quarter framing, the subject shown "
                      "whole with clear space around it inside frame, "
                      "balanced composition")),
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
    #: THE RENDER CANVAS, DECLARED STATICALLY (B5, 2026-07-27) -- beside the
    #: frame contract, the aspect and the fps, because it is the same kind of
    #: fact: what this adapter renders, decided per engine and readable without
    #: loading anything.
    #:
    #: NOT a ledger read and NOT an env var, on the O1 canvas judgment's
    #: explicit instruction. That judgment enumerated FIVE channels that name a
    #: render canvas and found this tier's -- the profile's `render.canvas_w/h`
    #: travelling to `video.canonical_canvas` -- to be the one DEAD channel: the
    #: profile asked for 512x288, `build_request_from_shot` handed every
    #: non-face engine 1472x832, and nothing on the render path ever read the
    #: stamp. 8.3x the pixels, on the tier that exists because 8 GB cannot
    #: afford them.
    #:
    #: 512x288 is the decided value (8gb judgment section 1, four independent
    #: sources): with 1024x576 it is one of only two rungs that are BOTH exactly
    #: 16:9 and /32-clean, and it scales 3.75x to the 1920x1080 deliverable with
    #: zero pad area. 832x480 is 26:15 and would pillarbox.
    #:
    #: Reading it from the ledger instead would make an operator who forces this
    #: engine onto another workflow -- the coverage matrix does exactly that
    #: through `role_overrides` -- inherit that workflow's 26:15 canvas and
    #: either pillarbox or be refused. A declaration cannot be displaced by
    #: where it is pointed. `tests/test_ltx_8gb_canonical_canvas.py` pins the
    #: profile's own `render.canvas_w/h` equal to this pair so the dead channel
    #: cannot silently drift away from it.
    render_canvas = (512, 288)
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
        """Resolved checkpoint path, by the LOADER'S token. ``None`` when absent
        everywhere (the offline invariant -- no runtime fetch).

        DELEGATES to :meth:`_loader_token_path`, which is the ONE place the
        explicit-override-vs-token question is decided. It used to short-circuit
        on ``OTR_LTX_8GB_CKPT`` after a bare existence check, which made this the
        SECOND authority on which file is the checkpoint -- and the one
        ``assert_usable`` consulted, so the whole single-clip path (the common
        case) validated one file while ``_build_graph`` handed the loader a bare
        basename that resolves to another. Two resolvers over one fact.

        Consequence worth knowing: unlike its five sibling adapters' ``_ckpt_path``,
        this one can RAISE ``EngineUnusable`` -- because this is the only adapter
        with an explicit-full-path override to contradict. The shared naming
        convention (wan_shared) is preserved; only the AUTHORITY moved."""
        return self._loader_token_path(
            ("checkpoints",), self._ckpt_name(), "OTR_LTX_8GB_CKPT_DIR",
            env_explicit="OTR_LTX_8GB_CKPT")

    def _t5_path(self):
        """Resolved T5 path, by the loader's token. There is no
        explicit-full-path override for the T5, so ``OTR_LTX_8GB_T5_DIR`` is the
        ONLY channel that can contradict the token on this side -- and the
        directory tripwire in ``_loader_token_path`` is what catches it.
        Delegated for symmetry, so a future ``OTR_LTX_8GB_T5`` cannot reopen the
        identity lie here by forgetting the guard."""
        return self._loader_token_path(
            ("text_encoders", "clip"), self._t5_name(), "OTR_LTX_8GB_T5_DIR")

    def _installed(self):
        """Is the checkpoint present? A PREDICATE -- it may not raise.

        Every sibling adapter's ``_installed`` returns a bare bool and callers
        treat it as one (``load`` gates on ``not self._installed()``). Now that
        ``_ckpt_path`` can refuse a contradictory override, that refusal has to
        become False here rather than escaping from a predicate. The operator
        still gets the precise message: ``assert_usable`` runs BEFORE ``load``
        on every real render path and raises the named MALFORMED_CONFIG there."""
        try:
            return self._ckpt_path() is not None
        except EngineUnusable:
            return False

    # ---- the FROZEN session config (B2) ----
    def _loader_token_path(self, categories, token, env_dir, env_explicit=None):
        """Where the LOADER NODE will actually find ``token``.

        ``_build_graph`` hands ``CheckpointLoaderSimple`` / ``CLIPLoader`` a BARE
        BASENAME, which ComfyUI resolves through ``folder_paths``. So the file the
        graph loads is whatever that TOKEN resolves to -- never an absolute path
        an env var happens to name. An explicit override that names a DIFFERENT
        file would make the receipt describe one weight while the render used
        another: preflight green, identity a lie. Resolve by token here, and make
        a disagreeing override terminal rather than silent.

        THIS IS THE ONE AUTHORITY. ``_ckpt_path`` / ``_t5_path`` delegate to it,
        and ``resolve_session_config`` reaches the same answer through them, so
        the single-clip path (via ``assert_usable`` -> ``_ckpt_path``) and the
        multi-segment path (via ``session_identity`` -> ``resolve_session_config``)
        can no longer disagree about which file is the checkpoint.

        BOTH override channels are checked against the loader's own answer: the
        explicit path (``env_explicit``) and the directory (``env_dir``). The
        directory one is a DEPRECATION TRIPWIRE, not a feature -- ``*_DIR`` has
        only ever affected preflight, never which weights load, and nothing in
        the shipped configuration sets it. It refuses loudly and points at
        ``extra_model_paths.yaml``, which is the channel that does reach the
        loader and is already live on this box. If the tripwire never fires,
        delete the variable and this branch.

        SCOPED TO THIS ADAPTER on purpose. ``eng_wan_ti2v`` / ``eng_wan_i2v``
        carry the same lie (their loaders also take bare tokens), but their test
        suites use ``*_DIR`` as the mock seam for a no-ComfyUI box
        (``tests/test_wan_loader_preflight.py`` says so in its own docstring), so
        fixing them means migrating those fixtures first. Separate chunk."""
        by_token = self._resolve_model_file(categories, token, env_dir)
        # A ``*_DIR`` override wins in _resolve_model_file on EXISTENCE ALONE and
        # never consults folder_paths -- but the graph hands the loader the bare
        # token, which ComfyUI resolves through folder_paths. So a directory
        # override can satisfy this preflight while being invisible to the loader
        # that actually loads the weights: the same identity lie as the explicit
        # path, one level up. Ask what the LOADER would find and require the two
        # to agree. Only fires when the operator actually set a *_DIR.
        if os.environ.get(env_dir):
            loader_would_load = self._resolve_model_file_by_token(categories, token)
            if loader_would_load is None or not _same_file(by_token or "",
                                                           loader_would_load):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s=%r does not change which weights load. The graph asks "
                    "ComfyUI for the bare name %r, which resolves to %s -- not to "
                    "your directory. That is this build's bug, not your setup: the "
                    "variable has only ever affected the preflight check, never the "
                    "render. Register your folder in extra_model_paths.yaml (the "
                    "channel that does reach the loader) or move the file under "
                    "models/, then unset %s. Stopping now rather than spending a "
                    "render on weights you did not choose."
                    % (env_dir, os.environ.get(env_dir), token,
                       repr(loader_would_load) if loader_would_load
                       else "nothing at all on this box",
                       env_dir), kind="video")
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
                "invariant, no runtime fetch); drop it in models/text_encoders or "
                "register its folder in extra_model_paths.yaml, and fix "
                "OTR_LTX_8GB_T5_NAME if the basename differs. Do not reach for "
                "OTR_LTX_8GB_T5_DIR -- it never reached the loader and is now "
                "refused" % self._t5_name(), kind="video")
        try:
            ckpt_receipt = _file_receipt(ckpt)
            t5_receipt = _file_receipt(t5)
        except FileReceiptUnavailable as exc:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb %s" % (exc,), kind="video")
        return LtxSessionConfig(
            engine=self.name, recipe=self._recipe_receipt(cfg),
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
        """Whether to decode through ``VAEDecodeTiled`` (v2: ON, measured).

        v1 kept this off on the grounds that core ``VAEDecode`` handles the
        peak at the smoke canvas. The 2026-07-27 sweep showed it does -- for
        2.6 GB more, and with a peak that GROWS with clip length (8662 MB at
        len=33 to 10859 MB at len=161) instead of staying flat at 8241-8278 MB.
        Truthy {1,true,yes,on} enables it under the consent act."""
        if not _prequalification_active():
            return bool(LTX8_RECIPE["tiled_vae"])
        # The prequalification DEFAULT is the frozen value, not a literal: a
        # sweep that opens the knobs but re-exports only some of them must
        # measure the recipe it is validating, not a third configuration.
        dflt = "1" if LTX8_RECIPE["tiled_vae"] else "0"
        # `or dflt`, not `get(name, dflt)`: an exported-but-EMPTY var would
        # otherwise read as "" -- not truthy -- and force the knob OFF against
        # a frozen default of ON. Every other accessor treats empty as unset.
        raw = (os.environ.get("OTR_LTX_8GB_TILED_VAE") or dflt).strip().lower()
        if raw in _TRUTHY:
            return True
        if raw in _FALSY:
            return False
        # FAIL CLOSED under the consent act. Outside it this var is never
        # parsed at all; inside it, it is a MEASUREMENT INPUT, and anything
        # unrecognised used to collapse to False -- so a sweep could mistype
        # the knob it was varying, decode untiled, and stamp a receipt saying
        # it had measured tiled. Same rule as `_config_number`: one bad value
        # stops the sweep instead of quietly becoming a third configuration.
        raise EngineUnusable(
            self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "OTR_LTX_8GB_TILED_VAE=%r is not a yes/no value (yes: %s; no: %s)"
            % (os.environ.get("OTR_LTX_8GB_TILED_VAE"),
               ", ".join(_TRUTHY), ", ".join(_FALSY)), kind="video")

    def _t5_device(self):
        """T5 CLIPLoader device: default ``cpu`` for the 8GB tier (t5xxl_fp16 alone
        is ~9 GB, so it encodes on CPU first, then diffusion runs on the GPU). The
        offload-on-vs-off VRAM measurement (C3) may flip this via OTR_LTX_8GB_T5_DEVICE."""
        frozen = str(LTX8_RECIPE["t5_device"])
        if not _prequalification_active():
            return frozen
        dev = (os.environ.get("OTR_LTX_8GB_T5_DEVICE") or frozen).strip().lower()
        if dev in _T5_DEVICES:
            return dev
        # FAIL CLOSED, not clamp-to-recipe. Clamping kept a bad device string
        # away from ComfyUI, which was right, but it also meant a sweep cell
        # that typed `cuda:7` measured the FROZEN device and stamped a receipt
        # claiming otherwise -- the measurement equivalent of a render that did
        # not happen being counted as one. Refusing by name does both jobs.
        raise EngineUnusable(
            self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "OTR_LTX_8GB_T5_DEVICE=%r is not one of %s"
            % (os.environ.get("OTR_LTX_8GB_T5_DEVICE"),
               ", ".join(_T5_DEVICES)), kind="video")

    def _negative_prompt(self):
        """The FROZEN negative conditioning (B6).

        The per-shot ``negative_prompt`` on the request still wins over this --
        that is the director's channel and it travels with the work. What is
        gone is the ``os.environ`` fallback: it made two boxes render visibly
        different clips from the same episode while both stamped the same
        recipe receipt, which is the whole defect this chunk closes."""
        frozen = str(LTX8_RECIPE["negative"])
        if not _prequalification_active():
            return frozen
        return os.environ.get("OTR_LTX_8GB_NEGATIVE") or frozen

    def _negative_for(self, shot_negative):
        """The negative conditioning for THIS shot, and the one place that
        decides it.

        PRODUCTION: the per-shot value wins, unchanged. The director's channel
        travels WITH the work, and B6 demoted only the server-boot channel.

        UNDER THE CONSENT ACT IT IS TERMINAL, and that is a LANE 2 consequence
        rather than new policy for its own sake. The receipt is SESSION-scoped:
        ``session_identity`` is read before the weights land and again before
        every segment, so it may only describe request-independent things, and
        the recipe string is element [1] of it. That means the receipt can only
        ever report what the RECIPE resolved. A sweep whose measured negative
        was displaced per shot would render one conditioning and stamp a
        receipt naming another -- the exact class of lie LANE 2 exists to
        remove, and worse than the generic mark it replaces, because a specific
        false claim is more credible than a vague true one.

        Today ``render_driver`` never populates ``negative_prompt`` for a video
        shot (grep: zero occurrences), so this cannot fire on the shipped path.
        It is the tripwire for whoever wires that channel."""
        frozen = self._negative_prompt()
        if not shot_negative:
            return frozen
        if _prequalification_active() and shot_negative != frozen:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "a shot supplied its own negative_prompt during a %s=1 "
                "measurement run, which would displace the negative this cell "
                "is measuring -- the clip would render one conditioning and "
                "stamp a receipt naming another. Clear the shot's "
                "negative_prompt for the sweep, or sweep a different knob"
                % PREQUALIFICATION_ENV, kind="video")
        return shot_negative

    def _config_number(self, env, dflt, lo, hi, cast):
        """Parse + RANGE-CHECK one numeric env knob, or fail CLOSED by name.

        THE ONE implementation, deliberately. It used to be a closure inside
        ``_resolve_render_config`` while ``_decode_inputs`` carried a second,
        quieter copy that swallowed a bad value and silently substituted the
        default -- so the tile geometry was the single knob on this adapter
        that failed OPEN. A sweep could then mistype the value it was
        measuring, render at something else, and stamp a receipt saying it had
        measured it. Two implementations of one rule is how that happens; one
        is the fix."""
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

    def _resolve_render_config(self):
        """Parse + RANGE-CHECK the render knobs ONCE (shared by assert_usable and
        _build_graph). A bad env value fails CLOSED here with a named MALFORMED_CONFIG,
        never a raw int()/float() crash mid-render. The sampler is validated against
        the portable floor whitelist."""
        _num = self._config_number

        # THE CEILING IS RESOLVED ON EVERY LEG (B6). ``max_frames`` is NOT part
        # of the frozen recipe -- it is a render-length ceiling B4's pre-render
        # refusal reads to make a plan-vs-box disagreement terminal -- so it
        # keeps its env channel AND its fail-closed range check everywhere.
        max_frames = _num("OTR_LTX_8GB_MAX_FRAMES", _LTX8_MAX_FRAMES_DEFAULT,
                          _LTX8_MIN_FRAMES, 16384, int)

        if not _prequalification_active():
            # THE PRODUCTION LEG: the frozen recipe binds, whatever the server
            # booted with. Anything the environment is trying to say is named
            # and ignored -- named because a knob that silently does nothing is
            # the complaint B3 already collected, and ignored (never parsed)
            # because a stale malformed value must not be able to kill a leg it
            # cannot influence. See ``_ignored_override_keys``.
            ignored = _ignored_override_keys()
            if ignored:
                _LOG.warning(
                    "[OTR video] ltx_8gb recipe %s is FROZEN in code; ignoring "
                    "%s from the environment. These knobs bind only under %s=1 "
                    "during prequalification -- a production leg is submitted "
                    "to an already-booted server, so its recipe may not depend "
                    "on how that server started (PBUG-20260723-02).",
                    RECIPE_LTX8_I2V, ", ".join(ignored), PREQUALIFICATION_ENV)
            # SPELLED OUT rather than `dict(LTX8_RECIPE)` so BOTH legs return
            # the SAME KEY SET. The recipe dict also carries `t5_device` and
            # `tiled_vae`, whose owners are `_t5_device()` / `_tiled_vae()`;
            # passing them through here on the production leg only would give
            # this function a return shape that varies by mode, and the next
            # reader to write `cfg["t5_device"]` would get a KeyError that only
            # reproduces under prequalification.
            return {
                "steps": LTX8_RECIPE["steps"],
                "cfg": LTX8_RECIPE["cfg"],
                "max_shift": LTX8_RECIPE["max_shift"],
                "base_shift": LTX8_RECIPE["base_shift"],
                "terminal": LTX8_RECIPE["terminal"],
                "max_frames": max_frames,
                "sampler": LTX8_RECIPE["sampler"],
            }

        # PREQUALIFICATION: the knobs are open, every value is range-checked as
        # before, and each honoured override is announced so a sweep's log says
        # what it actually measured.
        sampler = (os.environ.get("OTR_LTX_8GB_SAMPLER")
                   or self._DEFAULT_SAMPLER).strip()
        if sampler not in self._PORTABLE_SAMPLERS:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "OTR_LTX_8GB_SAMPLER=%r is not in the portable floor whitelist %s"
                % (sampler, sorted(self._PORTABLE_SAMPLERS)), kind="video")
        honoured = _ignored_override_keys()
        if honoured:
            _LOG.warning(
                "[OTR video] ltx_8gb PREQUALIFICATION (%s is set): honouring "
                "%s from the environment INSTEAD of frozen recipe %s. This is "
                "a measurement run, not a production contract.",
                PREQUALIFICATION_ENV, ", ".join(honoured), RECIPE_LTX8_I2V)
        return {
            "steps": _num("OTR_LTX_8GB_STEPS", LTX8_RECIPE["steps"],
                          1, 100, int),
            "cfg": _num("OTR_LTX_8GB_CFG", LTX8_RECIPE["cfg"],
                        0.0, 30.0, float),
            "max_shift": _num("OTR_LTX_8GB_MAX_SHIFT",
                              LTX8_RECIPE["max_shift"], 0.0, 100.0, float),
            "base_shift": _num("OTR_LTX_8GB_BASE_SHIFT",
                               LTX8_RECIPE["base_shift"], 0.0, 100.0, float),
            "terminal": _num("OTR_LTX_8GB_TERMINAL", LTX8_RECIPE["terminal"],
                             0.0, 0.99, float),
            "max_frames": max_frames,
            "sampler": sampler,
        }

    def _tile_geometry(self, key):
        """One tiled-decode geometry value: frozen, or range-checked under the
        consent act.

        THE one implementation. ``_decode_inputs`` builds the graph from it and
        ``_recipe_departures`` reports it, and two copies of one range check
        with two failure modes is the exact defect this adapter already paid
        for once (B6 finding 4)."""
        dflt = int(LTX8_RECIPE[key])
        if not _prequalification_active():
            return dflt
        lo, hi = _VAE_TILE_BOUNDS[key]
        return self._config_number(_RECIPE_ENV_KEYS[key], dflt, lo, hi, int)

    def _recipe_departures(self, knobs=None):
        """Which frozen knobs THIS CELL actually changed. ``{}`` on production.

        The four cells of the 2026-07-27 sweep all stamped one generic
        ``+prequalification``, so a winning artifact could not prove which knob
        values produced it: the ledger said a sweep ran, not which cell. This is
        what makes the mark specific.

        RESOLVED VALUES, not env presence. An operator who exports
        ``OTR_LTX_8GB_STEPS=8`` when 8 is already frozen has changed nothing,
        and a receipt claiming a departure there would describe a cell that does
        not exist. Only a value that actually differs is named.

        NOTHING IS PARSED ON A PRODUCTION LEG. The early return is the same
        contract ``_ignored_override_keys`` holds: outside the consent act these
        knobs cannot bind, so a stale malformed one must not be able to raise --
        ``PBUG-20260723-02`` wearing the opposite mask. It also means the
        production receipt is byte-identical to what B6 shipped.

        THE TILE GEOMETRY IS ONLY REPORTED WHEN TILED DECODE IS ON, because a
        knob the render never reached is not a departure that describes the
        clip -- and reading it anyway would newly refuse a sweep whose stale
        tile value has no effect on the cell it is measuring."""
        if not _prequalification_active():
            return {}
        knobs = self._resolve_render_config() if knobs is None else knobs
        resolved = {
            "steps": knobs["steps"], "cfg": knobs["cfg"],
            "max_shift": knobs["max_shift"], "base_shift": knobs["base_shift"],
            "terminal": knobs["terminal"], "sampler": knobs["sampler"],
            "t5_device": self._t5_device(),
            "tiled_vae": self._tiled_vae(),
            "negative": self._negative_prompt(),
        }
        if resolved["tiled_vae"]:
            for key in _VAE_TILE_BOUNDS:
                resolved[key] = self._tile_geometry(key)
        return _RD.departures(LTX8_RECIPE, resolved)

    def _recipe_receipt(self, knobs=None):
        """THE stamp. Both stamp sites go through here and nowhere else --
        ``test_EVERY_stamp_goes_through_the_receipt_helper`` pins it at the
        source level, because a second stamp site that reached for the bare
        constant would put an unmarked sweep clip into the durable ledger."""
        return recipe_receipt(self._recipe_departures(knobs))

    # ---- usability (fail-closed BEFORE any forward; ordinary preflight ONLY) ----
    def _assert_checkpoint_integrity(self, ckpt):
        """The 4 GiB checkpoint floor -- the ONE place it lives (B1b).

        It used to live inline in ``assert_usable``, and that was sound only
        while the weights loaded inside ``render_clip``. ``assert_usable`` runs
        PER SEGMENT inside ``render_driver._render_one``, which is AFTER
        ``BeatSession`` has opened the session -- so once B1b moved the real
        ``CheckpointLoaderSimple`` execution into ``prepare()``, the load
        overtook the only size check in the adapter. A truncated or wrong file
        would reach the loader first and the operator would get a deep loader
        traceback instead of the named refusal that tells them to re-fetch.
        ``resolve_session_config`` does not close this: it proves the file
        EXISTS and takes its receipt, never its size.

        Same reason code, same message, same ordering -- it still fires after
        the missing-file verdict and before the T5 verdict. The only change is
        that ``prepare()`` calls it too, before it loads anything."""
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

    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordinary asset preflight -- NO VRAM/NVML/vendor gate, NO fallback
        (operator directive 2026-07-20). Fail CLOSED on SageAttention, then a bad
        render knob, then a missing checkpoint, a wrong-size checkpoint, a
        missing T5, and finally any missing ComfyUI node class.

        S8b-13 (lane 8, 2026-08-11) ADDED THE FIRST AND LAST OF THOSE. This is an
        LTX-Video 0.9.8 engine -- the exact family BUG-070 was written for -- and
        it was the only one of the three LTX lanes with NO Sage gate at all,
        while both siblings call ``assert_sage_not_patched`` (``eng_ltx_video``,
        ``eng_ltx_av``). int8-PV SageAttention process-aborts LTX with no
        traceback, so "no gate" means the failure mode is a dead process rather
        than a named refusal. The node gate was the same shape of hole one level
        down: a missing LTXV class surfaced at ``load()`` -- mid-render, after
        the checkpoint had been paid for -- instead of at preflight."""
        # BUG-070 SageAttention contamination -- FIRST, before any weight is
        # resolved: a refusal that costs nothing beats one that costs a load.
        _MC.assert_sage_not_patched(self.name, self.family)
        self._resolve_render_config()                 # range-checked, fail-closed
        ckpt = self._ckpt_path()
        if ckpt is None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb checkpoint %r not found; fetch it via "
                "scripts/download_ltx_0_9_8.ps1, drop it in models/checkpoints, or "
                "register its folder in extra_model_paths.yaml -- OTR_LTX_8GB_CKPT "
                "only names a file, it cannot make the loader find one"
                % self._ckpt_name(), kind="video")
        self._assert_checkpoint_integrity(ckpt)
        if self._t5_path() is None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_8gb text encoder %r absent -- the 0.9.8 checkpoint carries no "
                "text encoder, so the shared t5xxl_fp16 must be on disk (offline "
                "invariant, no runtime fetch); drop it in models/text_encoders or "
                "register its folder in extra_model_paths.yaml, and fix "
                "OTR_LTX_8GB_T5_NAME if the basename differs. Do not reach for "
                "OTR_LTX_8GB_T5_DIR -- it never reached the loader and is now "
                "refused" % self._t5_name(), kind="video")
        # NODE GATE (S8b-13, lane 8): every required ComfyUI class must resolve
        # HERE, at preflight, on the ACTIVE candidate set -- the tiled-VAE knob
        # swaps VAEDecode for VAEDecodeTiled, so reading `_node_candidates()`
        # rather than a fixed list is what keeps this honest. Collect every
        # missing class before raising: naming one at a time turns a fresh
        # install into a sequence of failed renders.
        from . import wrapper_bridge as _wb
        mapping = _wb.node_class_mappings()
        missing = []
        for _logical, candidates in self._node_candidates().items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 -- collect every missing class
                missing.append("/".join(candidates))
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s (install/update "
                "ComfyUI-LTXVideo)" % (self.name, ", ".join(missing)),
                kind="video")
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
        # The frozen geometry, env-overridable ONLY under the consent act, and
        # range-checked through the SAME helper as every other knob so a
        # mistyped value stops a sweep instead of being quietly replaced by the
        # default it was meant to be measured against.
        #
        # ``_tile_geometry`` is the ONE implementation (LANE 2). It used to be a
        # closure here, and ``_recipe_departures`` would have needed a second
        # copy to report what a sweep cell actually decoded with -- which is how
        # this adapter grew two range checks with opposite failure modes the
        # first time (B6 finding 4).
        _i = self._tile_geometry
        base.update({
            "tile_size": _i("vae_tile"),
            "overlap": _i("vae_overlap"),
            "temporal_size": _i("vae_temporal"),
            "temporal_overlap": _i("vae_temporal_overlap"),
        })
        return base

    def _build_graph(self, request, image_name, plan, length, width, height,
                     external_results=None):
        """The declarative LTX 0.9.8 distilled I2V graph (wrapper_bridge.run_graph
        format). The 0.9.8 all-in-one gives MODEL(0)+embedded VAE(2); the T5 is the
        separate CLIPLoader (device cpu on the 8GB tier).

        ``external_results`` names the ids the CALLER already produced and owns
        for the whole beat (B1b: ``prepare()`` loads the checkpoint once). Those
        node DEFINITIONS are omitted while every wire that reads them stays --
        the executor resolves them from the caller's handles instead. Omitting
        rather than leaving them is not an optimisation: ``run_graph`` REFUSES a
        graph that also defines an id the caller supplied, because then two
        parties claim to produce one output.

        Conditional on purpose. A caller that prepared nothing -- the
        single-clip path, which is what production runs today, and every
        sibling test that hand-builds ``prepared={"patchers": []}`` -- gets the
        loader nodes exactly as before."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        cfg = self._resolve_render_config()
        positive = get("text_prompt") or "subtle natural motion, cinematic light"
        negative = self._negative_for(get("negative_prompt"))
        graph = {
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
        for nid in set(external_results or ()):
            graph.pop(nid, None)
        return graph

    # ---- residency ----
    def load(self):
        """Fail CLOSED until installed, then resolve the installed ComfyUI node
        classes (0.9.8 core LTX nodes). Resolves CLASSES only -- no weights. The
        checkpoint's weights load in :meth:`prepare` (once per beat); the T5's
        load when its ``CLIPLoader`` executes inside each segment's graph."""
        if not self._installed():
            raise RuntimeError(
                "ltx_8gb not installed: checkpoint %r missing -- fetch it via "
                "scripts/download_ltx_0_9_8.ps1" % self._ckpt_name())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def prepare(self, host_caps, profile, session_ctx):
        """Load the CHECKPOINT ONCE PER BEAT instead of once per segment (B1b).

        ``BeatSession`` calls this at the top of a multi-segment beat. Before
        B1b it only resolved node CLASSES, so every segment's graph re-executed
        ``CheckpointLoaderSimple`` and re-read 6.34 GiB from disk -- "one model
        load per beat" was the design and not the behaviour. Now a LOADER-ONLY
        mini-graph runs here and its results become
        ``prepared["external_results"]``, which ``render_clip`` forwards and
        ``_build_graph`` omits the definition for.

        THE CHECKPOINT ONLY. The T5 ``CLIPLoader`` is deliberately left in the
        segment graph: the pos/neg encodes happen per segment either way, so
        hoisting the loader would buy wall-clock while pinning ~9 GB of
        ``t5xxl_fp16`` resident for the whole beat -- a guaranteed OOM on an
        8 GB tier under ``OTR_LTX_8GB_T5_DEVICE=default``. ``ModelSamplingLTXV``
        stays per segment too: it is a cheap clone+patch of an already-resident
        MODEL, and it is correctly per-render.

        ORDER MATTERS, and each step is here because of a specific defect:

        1. Resolve the frozen config and check the checkpoint's SIZE **before**
           ``super().prepare()``. The floor is the only thing that distinguishes
           the real 2B all-in-one from a truncated download, and its old home
           (``assert_usable``) now runs after this. Doing it first also means a
           bad checkpoint is refused BEFORE the cross-process GPU lease is
           taken -- the C-3 lesson: a raise after the lease strands it for the
           life of the server.
        2. ``super().prepare()`` takes the lease and resolves classes.
        3. Execute the mini-graph, registering each detachable handle through
           ``on_result`` as it lands, so a failure mid-hoist still has an owner
           for everything already loaded.
        4. On ANY failure, tear down what we took and re-raise. The teardown
           call is itself wrapped, because ``teardown`` -> ``unload`` is
           engine-overridable and a raising unload must not replace the real
           error with its own.
        """
        from . import wrapper_bridge as _wb
        cfg = self.resolve_session_config(profile)
        self._assert_checkpoint_integrity(cfg.ckpt_path)
        prepared = super().prepare(host_caps, profile, session_ctx)
        try:
            classes = getattr(self, "_classes", None) \
                or _wb.resolve_graph_classes(self._node_candidates())
            bucket = prepared.setdefault("patchers", self._patchers)
            seen = {id(p) for p in bucket}

            def _register(nid, out):
                """Take ownership the moment a handle exists (not after the
                graph returns): if a later node raises, run_graph never returns
                a results dict, and an unregistered patcher is VRAM nothing
                will ever detach.

                SLOT 0 ONLY, and deliberately. ``CheckpointLoaderSimple``
                returns (MODEL, CLIP, VAE); the MODEL is the ``ModelPatcher``
                that ``_detach_patchers`` knows how to unwind. The duck-typed
                ``detach`` check is what keeps that honest rather than a
                positional assumption -- a slot that cannot detach is not
                something teardown can own. The embedded VAE is not tracked
                here, which matches every sibling adapter; see ``teardown``."""
                model = out[0] if out else None
                if (model is not None and id(model) not in seen
                        and callable(getattr(model, "detach", None))):
                    bucket.append(model)
                    seen.add(id(model))

            results = _wb.run_graph(
                {"ckpt": {"class": "ckpt",
                          "inputs": {"ckpt_name": cfg.ckpt_token}}},
                classes, on_result=_register)
            out = results.get("ckpt") or ()
            if len(out) < 3:
                raise _wb.GraphExecutionError(
                    "ltx_8gb hoisted checkpoint %r returned %d output(s); the "
                    "0.9.8 all-in-one must give MODEL(0), CLIP(1) and the "
                    "embedded VAE(2) -- the segment graph wires slot 2"
                    % (cfg.ckpt_token, len(out)))
            prepared["external_results"] = results
        except BaseException:
            try:
                self.teardown(prepared)
            except BaseException:          # noqa: BLE001 -- never mask the cause
                pass
            raise
        return prepared

    def teardown(self, prepared):
        """Drop the beat-scoped handles BEFORE delegating, then the base runs.

        ``prepared["external_results"]`` holds strong references to the hoisted
        MODEL and to the checkpoint's embedded VAE. The base teardown detaches
        the patchers, unloads, releases the lease and then waits for
        machine-wide VRAM to settle -- and every one of those steps would run
        with this dict still pinning the very tensors it is trying to reclaim,
        so the stability wait would be watching its own referent. Clearing
        first costs nothing.

        Scope, so this is not read as more than it is: the DETACH covers the
        MODEL patcher, which is what the harvest registers. The embedded VAE at
        slot 2 has never been handed to ``_detach_patchers`` -- not here and
        not in any sibling adapter -- so clearing this dict drops its last
        reference from the beat rather than reclaiming it explicitly. That gap
        is family-wide and pre-dates the hoist (which, if anything, shrinks it:
        one VAE load per beat instead of one per segment). Separate ticket.

        Never raises out of teardown (it is a finally-path)."""
        if isinstance(prepared, dict):
            prepared.pop("external_results", None)
        return super().teardown(prepared)

    def render_clip(self, request, prepared):
        """Drive ONE image->video clip via the in-process LTX 0.9.8 graph: stage the
        init image (no silent stretch, N9), execute the graph, encode the decoded
        IMAGE batch to a SILENT bt709 clip (V-1), CLIP-FILL to the beat window,
        retain the MODEL patcher for V-4 teardown, and thread the measured VRAM peak
        into the recipe receipt. M7 ffprobe-proves the silent-clip contract.

        ``prepared["external_results"]`` (B1b) carries the beat-scoped handles
        :meth:`prepare` loaded. They are forwarded to the executor and their
        node definitions are omitted from the graph. A caller that prepared
        nothing renders exactly as before -- the single-clip path, which is
        what production runs today, and every sibling test that hand-builds
        ``prepared={"patchers": []}``.

        The per-render patcher harvest below deliberately still names ``ckpt``.
        It is a no-op on the hoisted path (``prepare`` already put that handle
        in the SAME bucket, so the id-dedupe skips it) and it is load-bearing
        on the unprepared one, where the checkpoint is still loaded here and
        nothing else would ever hand it to teardown."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "ltx_8gb requires init_image (got %r)" % plan["init_image"])
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        # THE LENGTH IS DECIDED BEFORE ANYTHING IS STAGED (B4). The refusal
        # below must not come after an init image has been written or a graph
        # built, and it must never come after the GPU work -- a request this
        # adapter cannot render is knowable from pure numbers.
        target_frames = int(plan.get("target_frame_count") or 0)
        # Resolved ONCE and reused by the receipt stamp at the end of this
        # method. Not an optimisation: on a measurement leg every resolve emits
        # the "honouring X from the environment" warning, and a sweep's log is
        # the evidence the sweep exists to produce -- the same line three times
        # per clip is a line the operator stops reading.
        knobs = self._resolve_render_config()
        cap = knobs["max_frames"]
        contract = self.frame_contract
        rung = contract.smallest_legal_at_least(target_frames)
        if rung is None or rung > cap:
            raise _wb.GraphExecutionError(
                "ltx_8gb was asked for %d frame(s); the shortest legal render "
                "that covers it is %s and this engine may render at most %d "
                "(declared max %d, OTR_LTX_8GB_MAX_FRAMES %d). NO FALLBACK -- "
                "a short render padded back up to the ask is not a render of "
                "the ask, and on a chained lane the next segment would begin "
                "on a duplicated frame."
                % (target_frames, rung, cap, int(contract.max_frames), cap))
        width, height = self._dims(request)
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        length = rung
        ext = (prepared or {}).get("external_results") \
            if isinstance(prepared, dict) else None
        graph = self._build_graph(request, image_name, plan, length, width, height,
                                  external_results=ext)
        # free_after_use: the T5 text-encode frees before the sampler; the checkpoint
        # (MODEL + embedded VAE) + the model-sampling patch + the terminal are kept.
        # The NVML peak probe spans the whole render window (telemetry only -- no
        # ceiling enforcement; the operator's tier JSON owns the OOM budget).
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(
                graph, classes, free_after_use=True,
                external_results=ext,
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
        # THE PING-PONG IS GONE (B4, 2026-07-27), and these two invariants are
        # what it was standing in front of.
        #
        # It used to mirror-extend a short render up to the ask. That let a
        # render the engine could not actually deliver PASS the plan-vs-output
        # count gate (``render_driver``'s ``got != segment.render_frames``)
        # wearing the right number: part real motion, part mirrored frames, and
        # on this lane -- which declares ``strict_first_frame`` -- the next
        # chained segment would begin on a MIRRORED tail frame.
        #
        # The old closing clause -- "WAN keeps its extension: it renders short ON
        # PURPOSE and fills the beat with it, the shipped 8GB tier contract
        # (PBUG-20260723-02)" -- EXPIRED on 2026-08-02. WAN's ping-pong was
        # deleted under the no-mirror ruling and ``wan_ti2v`` was added to
        # ``PLANNING_CAP_ENGINES`` so the planner splits those beats into
        # affordable native segments instead. This lane was never the exception;
        # it was simply first.
        #
        # 1. THE PIPELINE INVARIANT. The graph was asked for exactly ``length``
        #    frames, so anything else is an under-delivery that the pad used to
        #    absorb for ANY reason, not just a cap disagreement. Deleting the
        #    pad without this would send a short clip on to the composite,
        #    which loop-fills it with the warning suppressed -- trading a
        #    logged mirror for a silent jump-cut repeat. Found by the pre-code
        #    panel, and it is the reason this block is not simply deleted.
        n_native = len(frames)
        if n_native != length:
            raise _wb.GraphExecutionError(
                "ltx_8gb asked its graph for %d frame(s) and decoded %d. NO "
                "FALLBACK -- padding the difference is how a render that did "
                "not happen gets counted as one." % (length, n_native))
        # 2. THE TAIL TRIM, in REAL frames. When the ask is off the 8n+1 grid
        #    the engine renders the next legal rung UP and drops the excess,
        #    which is what ``allow_tail_trim`` in its own contract declares.
        #    Every frame delivered is a rendered frame in order -- no mirror,
        #    no loop, no held frame. A target below the declared minimum is
        #    left alone: the contract says 9 is the shortest thing this adapter
        #    renders, and cutting below it would be inventing a length the
        #    declaration forbids.
        if length > target_frames >= int(contract.min_frames):
            frames = frames[:target_frames]
            _LOG.info(
                "[OTR video] ltx_8gb tail trim: rendered %d real frame(s) "
                "(nearest legal rung at or above the %d-frame ask) and kept "
                "%d @ %dx%d",
                length, target_frames, len(frames), width, height)
        out_path = otr_engine_tmp_mp4("otr_ltx_8gb_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7: PROVE the silent-clip color/stream contract on the emitted mp4.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] ltx_8gb VRAM render-phase peak %s MB @ %dx%d len=%d",
                      render_peak, width, height, n)
        # `native_frame_count` / `extension_mode` (2026-08-06). This adapter is
        # in ``frame_contract.PLANNING_CAP_ENGINES`` -- it is listed there
        # SPECIFICALLY so its beats are split into real segments -- and it was
        # the one coverage planner that answered neither question, so every
        # multi-segment ltx_8gb beat tripped the W5 grader's "declares no
        # extension_mode" branch. The number was already computed above and
        # already LOGGED at the tail trim; it was simply never stamped, which is
        # the PBUG-20260805-04 shape (a consumer armed against a producer that
        # never fires).
        #
        # EMITTED scope, not decoded scope: the count is ``n``, what this clip
        # actually carries, NOT the pre-trim ``n_native``. The whole point of
        # the field is that ``frame_count - native_frame_count`` is the number
        # of MANUFACTURED frames, and a decoded-scope count would make that gap
        # negative on every off-grid ask. Every frame this adapter delivers is a
        # rendered frame in order (see the tail-trim note above), so the honest
        # answer is that all of them are native and nothing was extended.
        return {"out_path": path, "frame_count": n,
                "vram_peak_mb": render_peak,
                "recipe": self._recipe_receipt(knobs),
                "native_frame_count": n,
                "extension_mode": "none",
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
            # The extension receipts (2026-08-06). This is the SECOND producer
            # seam: a field the engine returns and this method drops is a field
            # the grader never sees, and this adapter does NOT share the WAN
            # pair's ``wan_shared._clip_from_raw`` passthrough -- it has its own,
            # right here, which is exactly why the receipt went missing on this
            # lane and nowhere else.
            "native_frame_count": raw.get("native_frame_count"),
            "extension_mode": raw.get("extension_mode"),
            "render_canvas": raw.get("render_canvas"),
        }


__all__ = ["Ltx8gbEngine", "RECIPE_LTX8_I2V"]
