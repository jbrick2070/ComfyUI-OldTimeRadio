"""
ComfyUI-OldTimeRadio — AI-Powered Sci-Fi Radio Drama Generator
================================================================

Generates full-length sci-fi anthology radio dramas using:
  - LLM local inference (Gemma series, Nemo, etc.) for story writing + director
  - Bark (Suno) TTS with emotional bracket tags [sighs] [whispers] etc.
  - 48kHz stereo audio mastering; optional spatial effects (off by default)

Self-contained: drop into custom_nodes/ and go. No external node deps.

Audio:  LedgerScriptWriter -> FreezeCascade -> BatchBark -> SceneSequencer -> AudioEnhance -> EpisodeAssembler
Video:  EpisodeAssembler -> SignalLostVideo -> .mp4 + _treatment.txt (cast, voices, full script, stats)

BEST PRACTICE (per comfyui-custom-node-survival-guide Section 8):
  Uses isolated per-node loading so a broken dependency in one node
  doesn't prevent the rest from loading.

v1.0  2026-04-04  Jeffrey Brick — initial release
v1.4  2026-04-10  Jeffrey Brick — VRAM Hardening (v1.4 Flagship, 2GB Sovereignty)
"""

import importlib
import logging
import os
import warnings

log = logging.getLogger("OTR")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL LOG / WARNING SUPPRESSION — runs once before any node module loads.
#
# Three separate systems produce noise that we don't want:
#   1. HuggingFace Hub telemetry and ETag network checks → env vars
#   2. transformers' own logging (INFO/WARNING level) → hf_logging verbosity
#   3. Python's warnings system (FutureWarning/UserWarning from transformers
#      internals and Bark's hardcoded max_length=20 kwarg) → filterwarnings
#
# The Bark library module (_otr_bark_lib.py) also has targeted
# filterwarnings calls as a belt-and-suspenders fallback.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SAFETENSORS CONVERSION MOCK — NOW HANDLED IN prestartup_script.py (earlier)
# ─────────────────────────────────────────────────────────────────────────────
# (the nuclear mock runs before this file is even executed)

# THE ENV OWNER. Imported HERE -- above the first write below, and OUTSIDE the
# two swallowing try/excepts further down -- because it is stdlib-only and
# CANNOT fail. If it ever did, the boot must say so loudly rather than skip the
# OTR_OUTPUT_DIR pin behind a debug line.
try:
    from .nodes._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat load
    from nodes._otr_shared import env as otr_env  # type: ignore

# 1. Hub telemetry — disable before any transformers/huggingface_hub import
otr_env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# 2. transformers + huggingface_hub logging — errors only, no INFO/WARNING chatter
#    These are two separate logging systems — both need to be silenced.
#    The HF_TOKEN "unauthenticated requests" warning comes from huggingface_hub,
#    not transformers. No token needed — we run local_files_only=True throughout.
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass  # transformers not installed yet — will be caught at node load time
try:
    import huggingface_hub.utils._logging as hfh_logging
    hfh_logging.set_verbosity_error()
except Exception:
    pass

# 3. Python warnings — broad module-scoped filter for transformers FutureWarnings
#    (deprecation notices for APIs we don't control, e.g. Bark's generate() kwargs)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"transformers\..*")
warnings.filterwarnings("ignore", category=UserWarning,   module=r"transformers\..*")

# 4. HF_TOKEN bake-in — ComfyUI's desktop process does NOT inherit user-scope
#    env vars from HKCU\Environment, so gated models (Gemma, Mistral, FLUX-dev)
#    401 on first download.  Pull the token from the user registry now and
#    export it into os.environ so every downstream loader picks it up.
try:
    from .nodes._otr_shared.hf_token import ensure_hf_token
    ensure_hf_token()
except Exception as _hf_err:
    log.debug("[OldTimeRadio] HF_TOKEN bake-in skipped: %s", _hf_err)

# 5. OTR output base pin -- pin ONE output root so every consumer of
#    _otr_paths.comfy_output_dir() (portraits, ledger, episode/obs dirs)
#    uses the SAME tree. The pin is ComfyUI's live output directory
#    (folder_paths.get_output_directory), which honors --output-directory
#    and Desktop remaps. A walk-up from this file is NOT used: Desktop
#    can load the pack through an Installs junction, and abspath keeps
#    that path, so the episode would miss <output>/otr/obs. Gated on
#    folder_paths being importable so it ONLY fires inside the ComfyUI
#    process; CLI/pytest get no pin (preserves test isolation). Skipped
#    when the operator already set OTR_OUTPUT_DIR.
if not otr_env.get("OTR_OUTPUT_DIR"):
    try:
        from .nodes._otr_paths import pin_output_dir_from_comfy
        _otr_out = pin_output_dir_from_comfy()
        if _otr_out:
            log.info("[OldTimeRadio] OTR_OUTPUT_DIR pinned (ComfyUI output): %s", _otr_out)
    except Exception as _out_err:
        log.debug("[OldTimeRadio] OTR_OUTPUT_DIR pin skipped: %s", _out_err)

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATED PER-NODE LOADING
# If one node fails to import (e.g. missing transformers, parler_tts lib),
# the rest still load and work. This is critical for partial installs.
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# DUPLICATE PACK GUARD (2026-09-14). ComfyUI loads every folder under
# custom_nodes. A git clone plus a Manager install, or a leftover .bak
# folder, used to decorate GET /otr/latest_ledger twice. aiohttp then
# raised "method HEAD is already registered" in add_routes and the
# server never came up.
#
# A registry zip is one folder. First install and later Manager update
# both hit this branch with _otr_dup is None and load exactly as before.
# Fail OPEN if the helper cannot import: a missing guard must not skip
# the nodes or the HTTP route for a normal single-folder install.
_otr_dup = None
try:
    from .nodes._otr_pack_singleton import (
        PACK_GUARD as _OTR_PACK_GUARD,
        claim as _otr_claim_pack,
    )
    _otr_dup = _otr_claim_pack(
        _OTR_PACK_GUARD,
        os.path.abspath(os.path.dirname(__file__)),
    )
except Exception:  # pragma: no cover -- fail open for a single pack
    try:
        from nodes._otr_pack_singleton import (  # type: ignore
            PACK_GUARD as _OTR_PACK_GUARD,
            claim as _otr_claim_pack,
        )
        _otr_dup = _otr_claim_pack(
            _OTR_PACK_GUARD,
            os.path.abspath(os.path.dirname(__file__)),
        )
    except Exception:
        _otr_dup = None
if _otr_dup is not None:
    print(
        "[OldTimeRadio] DUPLICATE PACK skipped.\n"
        "  Already loaded from:\n"
        f"    {_otr_dup}\n"
        "  This extra copy is:\n"
        f"    {os.path.abspath(os.path.dirname(__file__))}\n"
        "  Keep ONE OldTimeRadio folder under custom_nodes (git clone "
        "OR Manager install, not both). Do not leave a .bak next to "
        "the live pack -- ComfyUI only skips names ending in "
        ".disabled. A second copy used to crash boot: GET "
        "/otr/latest_ledger registered twice, aiohttp raised "
        "'method HEAD is already registered', and the server never "
        "came up."
    )

_NODE_MODULES = {
    # key = NODE_CLASS_MAPPINGS key (permanent public ID — never rename)
    # value = (module_path, class_name, display_name)
    "OTR_LedgerScriptWriter": (".nodes.OTR_LedgerScriptWriter", "OTR_LedgerScriptWriter", " LPL Script Writer (v2.0)"),
    # Ledger Freeze Cascade: wires AFTER OTR_LedgerScriptWriter, BEFORE
    # OTR_SceneSequencer. The 3-pass cast-gated reviewer (Phases 1, 2, 9) is
    # wrapped by Phase 0 (gap_audit_pre) at entry and Phase 10 (gap_audit_post +
    # freeze) at exit.
    "OTR_LedgerFreezeCascade": (".nodes.OTR_LedgerFreezeCascade", "OTR_LedgerFreezeCascade", " LFC Ledger Freeze Cascade (v2.0)"),
    "OTR_SceneSequencer":     (".nodes.scene_sequencer",     "SceneSequencer",       " Scene Sequencer"),
    "OTR_EpisodeAssembler":   (".nodes.scene_sequencer",     "EpisodeAssembler",     " Episode Assembler"),
    "OTR_AudioEnhance":       (".nodes.audio_enhance",       "AudioEnhance",         " Spatial Audio Enhance"),
    "OTR_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", " Signal Lost Video"),
    # S26 Sprint 3 (T1.2): opt-in execution-time workflow contract
    # validator. Reads the workflow JSON from disk and runs the same
    # validate_workflow_contract check the S16.6 CI test runs. Place
    # as the first node in a workflow to catch contract drift at queue
    # time.
    "OTR_WorkflowValidator":       (".nodes._otr_workflow_validator", "WorkflowValidator", " Workflow Validator (opt-in, S14.2)"),
    # Plan 0k (2026-09-26): the ONE node that receives the queue's Comfy API
    # key. A V1 node declaring the hidden key writes it into /history when it
    # raises; this one cannot raise. Wired into the validator, the root.
    "OTR_ComfyCredential":         (".nodes.otr_comfy_credential", "OTR_ComfyCredential", "0 - Comfy Credential"),

    # =========================================================================
    # OTR Open Video Model Platform -- A-Seam core (CW-1, 2026-06-06).
    # Model-agnostic, per-role video model selection; NO model is "primary".
    # Additive shell: VideoDirector (per-role A/B/C model + image selectors,
    # Other-Beats clip mode) -> ShotLock (audio-derived clip budget +
    # DAG-validated execution_groups + M4 per-beat creative derivation). The
    # engine adapters and the render path live in nodes/_otr_video_engines/.
    # =========================================================================
    "OTR_VideoDirector":           (".nodes.otr_video_director", "OTRVideoDirector", " VideoDirector (per-role model select)"),
    "OTR_ShotLock":                (".nodes.otr_shot_lock",      "OTRShotLock",      " Shot Lock (video plan authority)"),

    # =========================================================================
    # v2.0 Image platform (Subproject C1) -- model-agnostic image-gen adapters,
    # one level UPSTREAM of video. Per-role image engine + granularity policy ->
    # Meta-Brief prompts -> cache-checked dispatch + ledger write-back + image_done.
    # "Flux" is just gen 1 (nodes/_otr_image_engines/flux_gen1.py); swap it and
    # nothing downstream changes. Cold-import clean; the live render is GPU.
    # =========================================================================
    "OTR_ImageDirector":           (".nodes.otr_image_director",          "OTRImageDirector",           " ImageDirector (per-role image model + granularity)"),
    "OTR_MetaBriefImagePromptGen": (".nodes.otr_meta_brief_image_prompt", "OTRMetaBriefImagePromptGen", " Meta-Brief Image Prompt Gen"),
    "OTR_ImageGenDispatcher":      (".nodes.otr_image_gen_dispatcher",    "OTRImageGenDispatcher",      " Image Gen Dispatcher (cache + ledger + image_done)"),

    # =========================================================================
    # v2.0 Video render path (A-S3 / CW-4) -- M1 first watchable episode.
    # The render output is composited into ONE always-silent canonical video
    # (OTR_SilentComposite), then the FROZEN master audio is muxed on LAST
    # (OTR_MasterAudioMux: -c:a copy, NO -shortest, byte-identical assert). Only
    # MasterAudioMux may add audio (V-1). Cheap radio-floor families register in
    # nodes/_otr_video_engines/cheap_families.py. The live episode render is an
    # interactive ComfyUI smoke.
    # =========================================================================
    "OTR_SilentComposite":         (".nodes.otr_silent_composite",   "OTRSilentComposite",   " SilentComposite (render -> one always-silent video)"),
    "OTR_CaptionBurn":             (".nodes.otr_caption_burn",       "OTRCaptionBurn",       " CaptionBurn (SDH open captions on the silent video; default-OFF)"),
    "OTR_MasterAudioMux":          (".nodes.otr_master_audio_mux",   "OTRMasterAudioMux",    " MasterAudioMux (terminal mux-LAST, -c:a copy)"),

    # =========================================================================
    # v2.0 in-process render driver (A-S7.5) -- the model-agnostic render entry
    # that walks the registry engines (prepare->render_clip->canonicalize->
    # teardown) with the A-S7 retry taxonomy + fallback chain + LOUD restamp.
    # "soak" mode is the A-ship full-episode gate; "single" validates one
    # engine's in-process forward. No model is "primary". See
    # nodes/_otr_video_engines/render_driver.py.
    # =========================================================================
    "OTR_VideoRenderBatch":        (".nodes.otr_video_render_batch", "OTRVideoRenderBatch", " Video Render Batch (A-S7.5 in-process render)"),

    # =========================================================================
    # v2.0 late viewer-credits surface (credits enrichment 2026-07-03). Renders
    # the ONE unified credits roll LATE (after nodes 91/92) from the durable
    # ledger stamps (S2) + the clip manifest, appends it as a SILENT tail to
    # node 86's output (the caption burn), and declares its tail duration to the
    # mux (credits-aware guard). Wired 86 -> OTR_CreditsRoll -> 85; no fallbacks
    # (a missing receipt RAISES). See nodes/otr_credits_roll.py + GO_FORWARD_CREDITS.md.
    # =========================================================================
    "OTR_CreditsRoll":             (".nodes.otr_credits_roll",       "OTRCreditsRoll",       " CreditsRoll (late unified viewer credits, silent tail)"),


}

# ---------------------------------------------------------------------------
# v2 audio/casting nodes (Wave 1a-c, 2a) -- registered from the single source
# of truth in nodes/_otr_class_registry.new_node_modules_table(), and ONLY for
# the nodes whose module file already exists on disk. Merging by table (not as
# literal "OTR_..." keys) keeps the class-registry collision guard meaningful;
# the file-existence check keeps a box-fresh load banner clean -- a node whose
# module has not landed yet (e.g. OTR_CastLock before Wave 2a) is simply
# skipped, so the loader never logs a phantom "Skipped" line (C-6).
try:
    from .nodes._otr_class_registry import (
        new_node_modules_table as _otr_new_audio_table,
    )

    _otr_pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for _otr_key, _otr_value in _otr_new_audio_table().items():
        _otr_rel = _otr_value[0].lstrip(".").replace(".", os.sep) + ".py"
        if os.path.exists(os.path.join(_otr_pkg_dir, _otr_rel)):
            _NODE_MODULES[_otr_key] = _otr_value
except Exception as _otr_audio_reg_exc:  # noqa: BLE001
    log.warning(
        "[OldTimeRadio] v2 audio node table merge skipped: %s",
        _otr_audio_reg_exc,
    )

if _otr_dup is None:
    for node_name, (module_path, class_name, display_name) in _NODE_MODULES.items():
        try:
            mod = importlib.import_module(module_path, package=__name__)
            cls = getattr(mod, class_name)

            # Single canonical registration (OTR_ prefix only); no bare-name
            # alias or parallel legacy-workflow path.
            NODE_CLASS_MAPPINGS[node_name] = cls
            NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name

        except Exception as e:
            log.warning("[OldTimeRadio] Failed to load '%s': %s", node_name, e)
            print(f"[OldTimeRadio] Skipped '{node_name}': {e}")

# ─────────────────────────────────────────────────────────────────────────────
# No rename-alias or back-compat surface: every workflow JSON references
# the current canonical class names directly, and a workflow that names a
# stale class fails loudly.
# ─────────────────────────────────────────────────────────────────────────────

_loaded = sum(1 for k in NODE_CLASS_MAPPINGS if k.startswith("OTR_"))
_total = len(_NODE_MODULES)
if _otr_dup is None:
    if _loaded == _total:
        print(f"[OldTimeRadio] OK - All {_total} nodes loaded successfully")
    else:
        print(f"[OldTimeRadio] Loaded {_loaded}/{_total} nodes ({_total - _loaded} failed)")
# A first-time user has a menu full of nodes and no idea the episode workflow
# exists. Registering
# nodes is not the deliverable -- the workflow is. Point at it from the one place
# they are already looking on first boot.
# ComfyUI's template gallery serves ONE folder per pack: it registers a static
# mount at the same URL for every folder named example_workflows / example /
# examples / workflow / workflows, and the first mount wins. This pack's
# templates therefore all live in workflows/; an example_workflows/ folder
# beside it made otr_canonical list in the gallery and 404 on click (2026-09-01
# ship audit).
#
# NAME ONLY WHAT SHIPS: a first-boot message that names a template that is
# not in the gallery is the exact failure the paragraph above exists to
# prevent. Verify against the PUBLISHED bundle rather than the repo, because
# .comfyignore decides what ships. The gallery lists only the directory level
# (`*/workflows/*.json`), so every per-machine workflow sits in workflows/
# beside the canonical, and the gallery lists every one.
# The banner names the canonical, which runs on any machine, and points at
# apple/MACHINES.md for the per-card pick. Every machine configuration is a
# row in config/workflow_matrix.json. If a template is added or dropped,
# this line changes in the same edit.
# Say what the README says, or say nothing: the canonical resolves the device
# at run time and every dropdown already holds a working value.
# The gallery keys its Extensions entry on this pack's FOLDER (ComfyUI's
# custom_node_manager keys on the directory that holds workflows/), and
# locales/en/main.json maps BOTH folder names -- comfyui-old-time-radio from the
# Manager, ComfyUI-OldTimeRadio from a git clone -- to "Old-Time Radio", which
# is what the frontend shows (seen live on 1.52.7, 2026-09-26).
if _otr_dup is None:
    print("[OldTimeRadio] Load the show:  Workflow > Browse Templates > "
          "Extensions > Old-Time Radio > otr_canonical  (runs on any "
          "machine -- pick it, then press Run; the other entries there are "
          "the same show preset for a specific card, listed in "
          "apple/MACHINES.md) -- or drag workflows/otr_canonical.json onto the "
          "canvas. Nothing needs changing: it resolves your device at run "
          "time. The finished episode lands in <output>/otr/obs/.")
    # AnimateDiff-Evolved's red "No motion models found" at every boot until
    # the first AnimateDiff run is expected; say so (2026-09-27). A note only:
    # any failure here is silent, and nothing is downloaded at boot.
    try:
        import folder_paths as _otr_fp  # type: ignore
        from .nodes._otr_boot_notes import ade_motion_module_note as _otr_ade_note
        _otr_note = _otr_ade_note(_otr_fp.get_folder_paths("custom_nodes"),
                                  _otr_fp.models_dir)
        if _otr_note:
            print(_otr_note)
    except Exception:  # noqa: BLE001 -- a boot note never affects loading
        pass

# =====================================================================
# HTTP route: GET /otr/latest_ledger
# Exposes the in-flight ledger (else the freshest per-episode *_ledger.json)
# as plain JSON over ComfyUI's existing
# HTTP server. Lets the live-run-tail Cowork artifact poll a single URL
# without needing Desktop Commander or any MCP transport.
# Wrapped in try/except so a server import failure cannot break node load.
# A single-folder install (first time or a later update) always enters
# this try: _otr_dup is None. Only a second custom_nodes folder skips.
# =====================================================================
class _OTRDuplicateRoute(Exception):
    """Internal: extra pack folder; do not decorate GET /otr/latest_ledger."""


try:
    if _otr_dup is not None:
        raise _OTRDuplicateRoute()
    import json as _otr_json
    import os as _otr_os
    from server import PromptServer as _otr_PromptServer  # type: ignore
    from aiohttp import web as _otr_web  # type: ignore

    _OTR_CORS_HEADERS = {
        # NO WILDCARD (2026-09-04). This route is registered on EVERY
        # install and needs no authentication, and it answers with the
        # whole ledger plus `fullpath` -- an absolute path that names the
        # operator. `*` is what makes that response READABLE cross-origin,
        # so any site visited while ComfyUI is running could take it.
        # Nothing shipped consumes this endpoint: `viewer/` is excluded by
        # .comfyignore, and it called /ledger and /list, which this pack
        # never registered. Same-origin callers (the ComfyUI front end)
        # are unaffected -- they never needed the header.
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Cache-Control": "no-store",
    }

    import re as _otr_re

    #: An absolute local path, in the spellings a ledger actually holds:
    #: `C:\...`, `C:/...`, a POSIX `/...`, and the UNC `\\host\share`. Matched
    #: on the WHOLE value -- a sentence that merely mentions a path is prose and
    #: is left alone, because rewriting narrative text would corrupt the reader's
    #: view of the episode for no privacy gain.
    _OTR_ABS_PATH = _otr_re.compile(
        r"^(?:[A-Za-z]:[\\/]|\\\\|/)[^\r\n]*$")

    def _otr_scrub_paths(value, _depth=0):
        """Return ``value`` with every absolute-path STRING reduced to its
        basename. Recursive over dicts and lists; never mutates the input.

        Depth-bounded because this walks a document that arrived from disk, and
        an unbounded recursion in an HTTP handler is a denial of service the
        route would hand out for free. A ledger nests ~6 deep; 24 is slack.
        """
        if _depth > 24:
            return value
        if isinstance(value, dict):
            return {k: _otr_scrub_paths(v, _depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [_otr_scrub_paths(v, _depth + 1) for v in value]
        if isinstance(value, str) and _OTR_ABS_PATH.match(value):
            # `basename` on a Windows path under POSIX returns the whole string,
            # so split on both separators rather than trusting os.path here.
            return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or value
        return value

    @_otr_PromptServer.instance.routes.get("/otr/latest_ledger")
    async def _otr_latest_ledger(request):
        try:
            # Delegate to the canonical resolver in_flight_ledger_path() -- it
            # returns the in-flight Ledger singleton's path during a run and
            # falls back to the per-episode mtime walker headless. Same resolver
            # every node uses, so the endpoint cannot desync from the real
            # on-disk layout.
            try:
                from .nodes._otr_ledger import in_flight_ledger_path
            except Exception:  # noqa: BLE001
                import _otr_ledger as _otr_led_mod  # type: ignore
                in_flight_ledger_path = _otr_led_mod.in_flight_ledger_path
            latest_p = in_flight_ledger_path()
            if latest_p is None:
                return _otr_web.json_response({
                    "ok": False,
                    "reason": "no ledger found: no in-flight singleton "
                              "and no ledger under output/otr/episodes/",
                }, headers=_OTR_CORS_HEADERS)
            latest = str(latest_p)
            with open(latest, "r", encoding="utf-8") as f:
                ledger = _otr_json.load(f)
            # NO ABSOLUTE PATH IN THE RESPONSE (2026-09-05). This route is
            # registered on every install and needs no authentication, and it
            # named the operator's own directory tree -- their Windows username
            # included -- to anyone who could reach it.
            #
            # REMOVING THE TOP-LEVEL `fullpath` WAS NOT ENOUGH, and the first
            # pass at this stopped there. The response returns the whole ledger
            # DOCUMENT, and that document is full of absolute paths: the
            # keys `_otr_ledger` writes under `meta.paths` (ledger_path,
            # episode_root, audio_dir, stills/portraits/videos dirs,
            # obs_dir, obs_final), plus every still's `path` and cache
            # `pool_path`, the music-cue WAVs, and the final audio/video/publish
            # targets. One live episode ledger measured 75 of them. So the
            # scrub happens on the SERIALIZED RESPONSE, recursively, by value:
            # any string that looks like an absolute local path becomes its
            # basename. The on-disk ledger is untouched -- this is a projection
            # for one HTTP reader, not a change to the record.
            return _otr_web.json_response({
                "ok": True,
                "filename": _otr_os.path.basename(latest),
                "mtime": _otr_os.path.getmtime(latest),
                "size": _otr_os.path.getsize(latest),
                "ledger": _otr_scrub_paths(ledger),
            }, headers=_OTR_CORS_HEADERS)
        except Exception as exc:
            # The exception TEXT carries paths too (a FileNotFoundError names
            # the file it could not open), so it goes to the server log, where
            # the operator reads it, and never into the HTTP body.
            print(f"[OldTimeRadio] latest_ledger failed: {exc}")
            return _otr_web.json_response(
                {"ok": False, "reason": "ledger could not be read; see the "
                                        "ComfyUI console for the reason"},
                status=500,
                headers=_OTR_CORS_HEADERS,
            )

    @_otr_PromptServer.instance.routes.options("/otr/latest_ledger")
    async def _otr_latest_ledger_options(request):
        return _otr_web.Response(status=204, headers=_OTR_CORS_HEADERS)

    print("[OldTimeRadio] HTTP route registered: GET /otr/latest_ledger (with CORS)")
except _OTRDuplicateRoute:
    print("[OldTimeRadio] HTTP route skipped (duplicate pack folder)")
except Exception as _otr_route_err:
    print(f"[OldTimeRadio] HTTP route registration skipped: {_otr_route_err}")

# =====================================================================
# NO POST ROUTE HERE MAY START WORK FROM A CALLER-SUPPLIED PATH. An
# unauthenticated route with a side effect is the Comfy Registry ban class
# `policy-v0.2: UNAUTHENTICATED_SIDE_EFFECT`, and no reviewer verdict says an
# env gate around a ROUTE REGISTRATION discharges it, so the answer is not
# to ship the construct rather than to argue about it. The real render path
# is the canonical workflow through `/prompt`; a development render harness
# belongs in `scripts/` (excluded from the published zip), not in the module
# every install imports.
# =====================================================================


# =====================================================================
# OH-3 janitor (output-tree contract 2026-06-11): server-boot sweep of
# stale episodes/_shared/tmp entries -- the ONE sanctioned auto-delete --
# plus the _shared README drop. Fully fail-soft: a janitor problem must
# never block node registration.
# =====================================================================
try:
    from .nodes._otr_janitor import run_boot_sweep as _otr_boot_sweep
    _otr_boot_sweep()
except Exception as _otr_janitor_err:  # noqa: BLE001 -- PD1
    print(f"[OldTimeRadio] janitor boot sweep skipped: {_otr_janitor_err}")

# =====================================================================
# THE WRITER'S MODEL FOLDER (2026-09-25): register ComfyUI's `LLM` model
# category, default `<models root>/LLM`. New writer downloads land there as
# real files (no hub symlinks, no Windows Developer Mode warning), and a
# user's extra_model_paths.yaml `LLM:` entry -- loaded by ComfyUI before any
# custom node -- keeps its place in front. Models already in the hub cache
# keep loading from there; nothing is moved. Fail-soft: a registration
# problem must never block node registration.
# =====================================================================
if _otr_dup is None:
    try:
        from .nodes._otr_llm_folder import register_llm_category as _otr_register_llm
        _otr_register_llm()
        from .nodes._otr_llm_folder import llm_roots as _otr_llm_roots
        _otr_llm_first = (_otr_llm_roots() or [None])[0]
        if _otr_llm_first is not None:
            # The FIRST registered path is where downloads go -- a user's
            # extra_model_paths.yaml entry when there is one, else ours.
            log.info("[OldTimeRadio] writer LLM folder: %s", _otr_llm_first)
    except Exception as _otr_llm_err:  # noqa: BLE001 -- PD1
        print(f"[OldTimeRadio] LLM folder registration skipped: {_otr_llm_err}")

# =====================================================================
# WEB_DIRECTORY -- the saved-workflow schema boundary (js/workflow_schema.js).
#
# LiteGraph restores widget values POSITIONALLY, so a node that drops a widget
# shifts every later value up by one on load: no error, no warning, a workflow that
# looks fine and renders something else. That extension reconciles an older
# saved workflow against the schema this build declares, BY NAME, before the loader
# sees it -- or refuses and leaves the open canvas untouched.
#
# It has to own the loader call rather than hook `beforeConfigureGraph`: the
# frontend runs those hooks through `invokeExtensionsAsync`, which catches and
# merely logs whatever they throw, so throwing there cannot stop a stale workflow.
#
# The same directory also serves `js/lane_node_packs.js` (2026-09-27): an
# ADVISORY `beforeConfigureGraph` hook that adds a lane's node pack (today
# AnimateDiff-Evolved) to the frontend's missing-node list when a workflow
# opens. A hint may live in a hook whose throws are swallowed; a refusal may not.
#
# ComfyUI serves this directory automatically when the module exports the name.
# =====================================================================
WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
