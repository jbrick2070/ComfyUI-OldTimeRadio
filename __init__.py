"""
ComfyUI-OldTimeRadio — AI-Powered Sci-Fi Radio Drama Generator
================================================================

Generates full-length sci-fi anthology radio dramas using:
  - LLM local inference (Gemma series, Nemo, etc.) for story writing + director
  - Bark (Suno) TTS with emotional bracket tags [sighs] [whispers] etc.
  - 48kHz stereo spatial audio mastering (Haas effect, mid-side widening)
  - Procedural SFX (theremin, static, room tone)

Self-contained: drop into custom_nodes/ and go. No external node deps.

Audio:  LedgerScriptWriter -> FreezeCascade -> BatchBark -> SceneSequencer -> AudioEnhance -> EpisodeAssembler
Video:  EpisodeAssembler -> SignalLostVideo -> .mp4 + _treatment.txt (cast, voices, full script, stats)
        (legacy "Director" stage was removed in voice-path-cleanbreak S2)

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

# 1. Hub telemetry — disable before any transformers/huggingface_hub import
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

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
    from .visual._hf_token import ensure_hf_token
    ensure_hf_token()
except Exception as _hf_err:
    log.debug("[OldTimeRadio] HF_TOKEN bake-in skipped: %s", _hf_err)

# 5. OTR output base pin (BUG-LOCAL-292) -- pin ONE output root so every OTR
#    consumer of _otr_paths.comfy_output_dir() (portraits, the latest_ledger
#    route, episode/obs dirs) resolves the SAME tree the ledger + video writers
#    use. Those writers land under the NODE-RELATIVE ComfyUI output dir (the
#    ComfyUI folder that contains this custom_nodes pack -- Documents\ComfyUI\
#    output on this box). folder_paths.get_output_directory() can differ (the
#    bundled install base under AppData), which is exactly what split portraits
#    from the ledger. So pin OTR_OUTPUT_DIR (tier 1) to the node-relative output
#    -- portable ("relative output/otr for everyone") and matching where
#    episodes already live: output/otr/{episodes,obs}. Gated on folder_paths
#    being importable so it ONLY fires inside the ComfyUI process; CLI/pytest
#    get no pin (preserves test isolation + monkeypatch). Skipped when set.
if not os.environ.get("OTR_OUTPUT_DIR"):
    try:
        import folder_paths  # presence == running inside the ComfyUI process
        _ = folder_paths.get_output_directory  # touch attr; absence -> skip
        _otr_out = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "output",
        )
        os.environ["OTR_OUTPUT_DIR"] = _otr_out
        log.info("[OldTimeRadio] OTR_OUTPUT_DIR pinned (node-relative): %s", _otr_out)
    except Exception as _out_err:
        log.debug("[OldTimeRadio] OTR_OUTPUT_DIR pin skipped: %s", _out_err)

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATED PER-NODE LOADING
# If one node fails to import (e.g. missing transformers, parler_tts lib),
# the rest still load and work. This is critical for partial installs.
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_NODE_MODULES = {
    # key = NODE_CLASS_MAPPINGS key (permanent public ID — never rename)
    # value = (module_path, class_name, display_name)
    "OTR_LedgerScriptWriter": (".nodes.OTR_LedgerScriptWriter", "OTR_LedgerScriptWriter", " LPL Script Writer (v2.0)"),
    # Ledger Freeze Cascade (LFC sprint, 2026-05-11). Renamed from
    # OTR_LedgerScriptReviewer in commit 2 of 14. Wires AFTER
    # OTR_LedgerScriptWriter, BEFORE OTR_SceneSequencer. Existing
    # 3-pass cast-gated reviewer (Phases 1, 2, 9) is now wrapped by
    # Phase 0 (gap_audit_pre) at entry and Phase 10 (gap_audit_post +
    # freeze) at exit. The legacy OTR_LedgerScriptReviewer name is
    # DEAD as of S29 (2026-05-14) per the no-legacy-back-compat
    # standing directive; workflow JSONs that still reference it
    # fail to load loudly via the workflow validator's
    # DELETED_NODE_TYPES sentinel. They must be re-saved against
    # OTR_LedgerFreezeCascade. See S29 final QA review.
    "OTR_LedgerFreezeCascade": (".nodes.OTR_LedgerFreezeCascade", "OTR_LedgerFreezeCascade", " LFC Ledger Freeze Cascade (v2.0)"),
    # S30 B4 (2026-05-14): standalone OTR_LFCPhase4Scene /
    # OTR_LFCPhase5Voice / OTR_LFCPhase6Arc node classes DELETED.
    # The three classes were orphaned from every shipped workflow JSON
    # (per S30 parent plan section 2a-bis audit). Combined with the
    # deletion-bias policy, the node files + their _otr_lfc phase
    # function backing (_phase_3_per_line_polish,
    # _phase_4_scene_coherence, _phase_4_5_smart_suggestion,
    # _phase_5_voice_drift, _phase_6_episode_arc) go entirely. B7 adds
    # the three class names as forbidden-pattern markers so the symbols
    # cannot reappear.
    # Voice-path-cleanbreak Sprint 2 (2026-05-12): OTR_LLMDirector deleted.
    # The LLMDirector class was the legacy LLM-derived production plan
    # generator. P2 severed the voice side (commit 446ec81); Sprint 2
    # migrated the two remaining video-side consumers (OTR_SignalLostVideo,
    # OTR_VideoPlan) to read meta.visual_plan + meta.voice_assignments +
    # meta.style directly from the L3 ledger stamped by the writer.
    # The deleted Director class + registration + workflow node +
    # workflow links 17 + 38 all delete in lockstep. See:
    #   docs/voice-path-cleanbreak-execution-plan.md Sprint 2
    #   docs/2026-05-12-voice-path-cleanbreak-qa.md §3 + Q3
    #
    # Voice-path-cleanbreak 2026-05-12 (P3): the legacy single-line
    # nodes OTR_BarkTTS / OTR_SFXGenerator / OTR_VoiceRender plus the
    # pre-L3 parser-list reader OTR_BatchKokoroGenerator are deleted.
    # Their registrations are removed in lockstep with the file deletes.
    "OTR_SceneSequencer":     (".nodes.scene_sequencer",     "SceneSequencer",       " Scene Sequencer"),
    "OTR_EpisodeAssembler":   (".nodes.scene_sequencer",     "EpisodeAssembler",     " Episode Assembler"),
    "OTR_AudioEnhance":       (".nodes.audio_enhance",       "AudioEnhance",         " Spatial Audio Enhance"),
    "OTR_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", " Signal Lost Video"),
    # The DMD / Self-Forcing restart transition (kibitz r2 F2, moved 2026-08-01).
    # PRODUCTION machinery: fastwan_8gb's adapter builds it into its render graph,
    # and the four-arm bench's API graphs name this same registered key -- one
    # implementation, two consumers. It used to live in the DIAGNOSTIC bench helper
    # (scripts/bench_helper/otr_bakeoff_helper), which a production engine may not
    # depend on; that copy is deleted rather than kept in sync. The bench's VRAM
    # probes stay OUT of the pack -- they are measurement machinery, and that
    # asymmetry is deliberate.
    "OTR_DMDRestartSamplerSelect": (".nodes._otr_video_engines.dmd_sampler", "OTR_DMDRestartSamplerSelect", " OTR DMD Restart Sampler (predict-x0 + renoise)"),
    "OTR_ProjectStateLoader": (".nodes.project_state",         "ProjectStateLoader",      " Project State Loader"),
    "OTR_VRAMGuardian":       (".nodes.vram_guardian",          "VRAMGuardian",            " VRAM Guardian"),
    "OTR_VRAMContextTest":    (".nodes.vram_context_test",     "VRAMContextTest",         " VRAM Context Test (diagnostics)"),
    # v2.0 Visual Generation Trio
    # Sidecar-isolated visual (stills/portraits/motion) generation from
    # the L3 ledger (replaces the legacy OTR Director output the visual
    # bridge used to read). Audio path NEVER touched. Falls back to
    # OTR_SignalLostVideo on failure.  See docs/OTR_PIPELINE_EXPLAINER.md
    "OTR_VisualBridge":         (".visual.bridge",            "VisualBridge",         " Visual Bridge"),
    "OTR_VisualPoll":           (".visual.poll",              "VisualPoll",           " Visual Poll"),
    "OTR_VisualRenderer":       (".visual.renderer",          "VisualRenderer",       " Visual Renderer"),
    "OTR_VisualPromptCoercion": (".visual.prompt_coercion",   "VisualPromptCoercion", " Visual Prompt Coercion"),
    # S30 B5 (2026-05-14): OTR_VisualLLMSelector node + visual/llm_selector.py
    # DELETED. The visual-polish path now consumes the writer's broadcast
    # `creative_writing_model` socket directly (the selector was a
    # redundant local model picker that duplicated the writer's surface).
    # visual/llm_polish.py's _POLISH_CACHE module-level dict + _load_model()
    # function were collapsed into the modern _otr_model_loader.LLM_CACHE
    # in the same commit (Prime Directive 2: never double-load
    # Mistral-Nemo on the 16 GB card).
    "OTR_VisualExtractFluxPrompt": (".visual.flux_prompt_extractor", "VisualExtractFluxPrompt", " Visual Extract FLUX Prompt"),
    # OTR_FluxBranchGate -- DELETED in the Chunk E cleanbreak completion
    # (2026-06-09). The Sprint H 3.7 FLUX topology gate only sequenced the
    # legacy FLUX batch chain (deleted in the CW cleanbreak); the platform
    # image gate (OTR_ImageGenDispatcher) is ordered by audio_done +
    # script_json edges instead. Tombstoned in DELETED_NODE_TYPES.
    # OTR_DeferredCheckpointLoader + OTR_DeferredLtxTextEncoderLoader --
    # DELETED in the Chunk E cleanbreak completion (2026-06-09). V-5: ALL
    # model loading is adapter-internal (comfy model_management); no
    # deferred-loader shell nodes survive. Their only consumers were the
    # deleted legacy FLUX/LTX batch chains. Tombstoned in
    # DELETED_NODE_TYPES; the Sprint H finding they encoded (executor
    # pre-loads loaders at graph-start regardless of downstream gates)
    # lives on in the adapters' lazy in-execute loading.
    # OTR_VideoPlan -- DELETED in CW-1 (2026-06-06) of the OTR video platform
    # build. Its prompt-generation + planning was absorbed by OTR_ShotLock
    # (M4 per-beat creative derivation). The legacy FLUX/HuMo/LTX render chain
    # it fed was unwired from otr_canonical.json in the same commit; the
    # type is in the workflow validator's DELETED_NODE_TYPES sentinel so a
    # stale workflow referencing it fails loudly (must be re-saved).
    # OTR_FixedShotDurationStub -- registration REMOVED in the Chunk E
    # cleanbreak completion (2026-06-09; the module-level mapping was
    # already cleared 2026-06-08). OTR_ShotLock owns ALL per-episode
    # budget / shot-duration logic. The class + pure helpers stay in
    # nodes/otr_shot_duration_calculator.py for unit tests only; the type
    # is tombstoned in DELETED_NODE_TYPES so stale workflow JSONs fail
    # loudly at validation.
    # OTR_PostAudioVideoPipeline -- DELETED S27 (commit lands in s27-
    # cleanbreak-tail). The class was a subprocess trigger for the
    # pre-2026-04-27 HuMo batch + concat pipeline; it was superseded
    # in-graph by OTR_BatchHumoRender + OTR_VideoComposite. S26 kept
    # the registration "so any old workflow JSON that still references
    # it loads without error" -- exactly the back-compat-for-old-data
    # pattern S27 was authorized to delete. Old workflow JSONs that
    # still name the type now fail to load loudly via the workflow
    # validator's DELETED_NODE_TYPES sentinel; they must be re-saved.
    # S26 Sprint 3 (T1.2): opt-in execution-time workflow contract
    # validator. Reads the workflow JSON from disk and runs the same
    # validate_workflow_contract check the S16.6 CI test runs. Place
    # as the first node in a workflow to catch contract drift at queue
    # time. ADR: docs/2026-05-13-S14_2-active-validation-ADR.md.
    "OTR_WorkflowValidator":       (".nodes._otr_workflow_validator", "WorkflowValidator", " Workflow Validator (opt-in, S14.2)"),
    # OTR_HuMoTierLoader -- DELETED in the CW cleanbreak (2026-06-08): it only
    # fed the now-removed OTR_BatchHumoRender; the in-process HuMo adapter
    # (nodes/_otr_video_engines/eng_humo.py) loads its own stack. Tombstoned in
    # the workflow validator's DELETED_NODE_TYPES.
    # OTR_VideoComposite -- DELETED (CW-4 legacy render-path teardown,
    # 2026-06-07). The legacy episode compositor mixed audio inside the
    # graph (master_mix / per-clip-mux / humo_concat) and used
    # ffmpeg `-shortest`, which the frozen-audio spine forbids. The new
    # render path is SignalLostVideo -> OTR_SilentComposite (always
    # silent, no audio) -> OTR_MasterAudioMux (terminal, -c:a copy, NO
    # -shortest, byte-identical master). The type is tombstoned in the
    # workflow validator's DELETED_NODE_TYPES so any stale workflow JSON
    # naming it fails loudly at validation; such graphs must be re-saved.
    # Queue item 8 (2026-08-08): the NVIDIA-only RTX VSR upscaler
    # (nodes/rtx_upscale.py) was RIPPED and REPLACED by the device-
    # selectable upscale namespace at nodes/_otr_upscale_engines/. The new
    # stage runs INSIDE OTR_SilentComposite (per-clip model dispatch in
    # _encode_segment's sharpen=True branch), so it does NOT ship as a
    # separate ComfyUI node -- the SilentComposite widget dropdown selects
    # the engine, and cross-vendor support (CUDA + CPU today; MPS deferred
    # pending a Mac receipt) comes from the shipped device_backends
    # declaration on each engine row. OTR_RTXUpscale is in
    # nodes/_workflow_validation.py DELETED_NODE_TYPES so any stale saved
    # workflow fails loudly at validation.
    # BUG-LOCAL-028 fix (2026-05-03): per-episode-aware image save sink.
    # Replaces stock SaveImage nodes whose hardcoded filename_prefix
    # couldn't track the in-flight episode_id. Reads the Ledger singleton
    # at runtime and routes images to output/otr/episodes/<ep>/stills/
    # or .../portraits/. Falls back to legacy flat dirs in headless/test.
    "OTR_SaveToEpisodeWorkspace":  (".nodes.otr_save_to_episode_workspace", "SaveToEpisodeWorkspace", " Save to Episode Workspace (FLUX stills/portraits)"),
    # BUG-LOCAL-030 Phase B (2026-05-03 EVENING): post-RTXUpscale procgen
    # visual blend. Overlays 1920x1080 native procgen on the upscaled
    # HuMo + LTX composite at delivery res. Audio passes through with
    # -c:a copy so C7 byte-identity holds end-to-end. Fills the visible
    # HuMo black pillarbox bars from Phase A simple-pillarbox composite
    # with the SIGNAL LOST CRT signature (audio-reactive scanlines +
    # flicker over the otherwise-static black surround).
    "OTR_PostUpscaleProcgenBlend": (".nodes.otr_post_upscale_procgen_blend", "PostUpscaleProcgenBlend", " Post-Upscale Procgen Blend (1080p)"),
    # §4D v2 scene-aware scopes: a late, additive node that draws GREEN-ONLY
    # audio-reactive scopes into the REAL per-beat gutters (read from the clip
    # manifest), screened+lightened over the upscaled video by the blend's 3rd
    # input. The floor keeps its full v1 video; only draw_scopes=False turns its
    # in-frame scopes off so they don't double.
    "OTR_SceneAwareScopes": (".nodes.otr_scene_aware_scopes", "SceneAwareScopes", " Scene-Aware Scopes (v2)"),

    # =========================================================================
    # OTR Open Video Model Platform -- A-Seam core (CW-1, 2026-06-06).
    # Model-agnostic, per-role video model selection; NO model is "primary".
    # Additive shell: VideoProbe (usable engines + host caps) -> VideoDirector
    # (per-role A/B/C model + image selectors, Other-Beats clip mode) ->
    # ShotLock (audio-derived clip budget + DAG-validated execution_groups +
    # M4 per-beat creative derivation; supersedes OTR_VideoPlan). Concrete
    # engine adapters + the render path land in later windows.
    # =========================================================================
    "OTR_VideoProbe":              (".nodes.otr_video_probe",    "OTRVideoProbe",    " Video Probe (usable engines + host caps)"),
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
    # node 93's output, and declares its tail duration to the mux (credits-aware
    # guard). Wired 93 -> OTR_CreditsRoll -> 85; no fallbacks (missing receipt
    # RAISES). See nodes/otr_credits_roll.py + GO_FORWARD_CREDITS.md.
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

for node_name, (module_path, class_name, display_name) in _NODE_MODULES.items():
    try:
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)

        # Single canonical registration (OTR_ prefix only). The legacy
        # bare-name (NodeName) alias mirror loop was deleted in the
        # voice-path-cleanbreak 2026-05-12 sprint per the Standing
        # Directive. Saved workflow JSONs that reference bare-name
        # node types are expected to be rewritten against the OTR_
        # prefix; there is no parallel legacy-workflow path.
        NODE_CLASS_MAPPINGS[node_name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name

    except Exception as e:
        log.warning("[OldTimeRadio] Failed to load '%s': %s", node_name, e)
        print(f"[OldTimeRadio] Skipped '{node_name}': {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Clean-break v2.0-alpha (2026-05-12): _RENAME_ALIASES dict removed. No
# back-compat surface. Every workflow JSON references the current canonical
# class names directly. Legacy class names (OTR_Gemma4Director,
# OTR_LedgerScriptReviewer, OTR_Gemma4ScriptWriter, OTR_LLMScriptWriter) are
# DEAD -- attempting to load a workflow that references one will fail loudly,
# which is the desired behaviour during a greenfield rewrite.
# ─────────────────────────────────────────────────────────────────────────────

_loaded = sum(1 for k in NODE_CLASS_MAPPINGS if k.startswith("OTR_"))
_total = len(_NODE_MODULES)
if _loaded == _total:
    print(f"[OldTimeRadio] OK - All {_total} nodes loaded successfully")
else:
    print(f"[OldTimeRadio] Loaded {_loaded}/{_total} nodes ({_total - _loaded} failed)")
# A first-time user has 34 nodes and no idea the episode graph exists. Registering
# nodes is not the deliverable -- the workflow is. Point at it from the one place
# they are already looking on first boot.
print("[OldTimeRadio] Load the show:  Workflow > Browse Templates > "
      "EXTENSIONS > comfyui-old-time-radio > otr_canonical")

# =====================================================================
# HTTP route: GET /otr/latest_ledger
# Exposes the freshest *_ledger.json from output/otr/audio/ (or the legacy
# output/old_time_radio/ fallback) as plain JSON over ComfyUI's existing
# HTTP server. Lets the live-run-tail Cowork artifact poll a single URL
# without needing Desktop Commander or any MCP transport.
# Wrapped in try/except so a server import failure cannot break node load.
# =====================================================================
try:
    import json as _otr_json
    import os as _otr_os
    from server import PromptServer as _otr_PromptServer  # type: ignore
    from aiohttp import web as _otr_web  # type: ignore

    _OTR_CORS_HEADERS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Cache-Control": "no-store",
    }

    @_otr_PromptServer.instance.routes.get("/otr/latest_ledger")
    async def _otr_latest_ledger(request):
        try:
            # Ledger durability P1 (2026-05-19): the old discovery scanned
            # hardcoded legacy-flat dirs (output/otr/audio +
            # output/old_time_radio) that have been extinct since the S28
            # per-episode workspace reorg, so this endpoint returned stale
            # pre-S28 ledgers and never saw a live run. Delegate to the
            # canonical resolver in_flight_ledger_path() -- it returns the
            # in-flight Ledger singleton's path during a run and falls back
            # to the per-episode mtime walker headless. Same resolver every
            # node uses, so the endpoint can no longer desync from the real
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
            return _otr_web.json_response({
                "ok": True,
                "filename": _otr_os.path.basename(latest),
                "fullpath": latest,
                "mtime": _otr_os.path.getmtime(latest),
                "size": _otr_os.path.getsize(latest),
                "ledger": ledger,
            }, headers=_OTR_CORS_HEADERS)
        except Exception as exc:
            return _otr_web.json_response(
                {"ok": False, "reason": str(exc)},
                status=500,
                headers=_OTR_CORS_HEADERS,
            )

    @_otr_PromptServer.instance.routes.options("/otr/latest_ledger")
    async def _otr_latest_ledger_options(request):
        return _otr_web.Response(status=204, headers=_OTR_CORS_HEADERS)

    print("[OldTimeRadio] HTTP route registered: GET /otr/latest_ledger (with CORS)")
except Exception as _otr_route_err:
    print(f"[OldTimeRadio] HTTP route registration skipped: {_otr_route_err}")

# =====================================================================
# HTTP routes: POST /otr/video_render_single + POST /otr/video_render_soak
# Trigger the in-process A-S7.5 render driver (nodes/_otr_video_engines/
# render_driver.py) in a background thread so the request returns at once;
# the worker writes a JSON report under output/otr/aship/ the operator polls.
# In-process == NODE_CLASS_MAPPINGS is populated, so the heavy engine forwards
# actually run on the GPU (the A-ship gate). Wrapped so a failure cannot break
# node load.
# =====================================================================
try:
    import json as _otr_rj
    import os as _otr_ro
    import threading as _otr_threading
    from server import PromptServer as _otr_PS2  # type: ignore
    from aiohttp import web as _otr_web2  # type: ignore

    def _otr_render_report_dir():
        # OH-1 (output-tree contract 2026-06-11): probe/soak reports go to
        # the per-machine state tier episodes/_shared/state via the ONE
        # path authority (the old top-level otr/aship is RETIRED).
        from .nodes._otr_paths import otr_state_dir as _otr_state_dir
        d = str(_otr_state_dir())
        _otr_ro.makedirs(d, exist_ok=True)
        return d

    def _otr_run_render_async(report_name, fn):
        path = _otr_ro.path.join(_otr_render_report_dir(), report_name)
        with open(path, "w", encoding="utf-8") as f:
            _otr_rj.dump({"ok": None, "status": "running"}, f)

        def _worker():
            try:
                rep = fn()
            except Exception as exc:  # noqa: BLE001 - report honestly
                import traceback as _tb
                rep = {"ok": False, "status": "crashed",
                       "error": "%s: %s" % (type(exc).__name__, exc),
                       "traceback": _tb.format_exc()[-2000:]}
            try:
                with open(path, "w", encoding="utf-8") as f:
                    _otr_rj.dump(rep, f, default=str)
            except Exception:  # noqa: BLE001
                pass

        _otr_threading.Thread(target=_worker, daemon=True).start()
        return path

    async def _otr_body(request):
        try:
            return await request.json()
        except Exception:  # noqa: BLE001
            return {}

    @_otr_PS2.instance.routes.post("/otr/video_render_single")
    async def _otr_video_render_single(request):
        body = await _otr_body(request)
        from .nodes._otr_video_engines import render_driver as _rd
        assets = {"init_image": body.get("portrait", ""),
                  "audio_ref": body.get("audio", "")}
        engine = str(body.get("engine", "humo"))
        fc = int(body.get("frame_count", 33))
        path = _otr_run_render_async(
            "single_%s.json" % engine,
            lambda: _rd.render_single(engine, assets=assets, frame_count=fc))
        return _otr_web2.json_response({"started": True, "report_path": path})

    @_otr_PS2.instance.routes.post("/otr/video_render_soak")
    async def _otr_video_render_soak(request):
        body = await _otr_body(request)
        from .nodes._otr_video_engines import render_driver as _rd
        assets = {"init_image": body.get("portrait", ""),
                  "audio_ref": body.get("audio", "")}
        beats = int(body.get("beats", 40))
        oom = int(body.get("oom_index", 20))
        fc = int(body.get("frame_count", 25))
        path = _otr_run_render_async(
            "soak_report.json",
            lambda: _rd.run_gpu_soak(n_beats=beats, oom_index=oom,
                                     frame_count=fc, assets=assets))
        return _otr_web2.json_response({"started": True, "report_path": path})

    print("[OldTimeRadio] HTTP routes registered: POST /otr/video_render_single,"
          " POST /otr/video_render_soak")
except Exception as _otr_render_route_err:
    print(f"[OldTimeRadio] render route registration skipped: {_otr_render_route_err}")

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

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
