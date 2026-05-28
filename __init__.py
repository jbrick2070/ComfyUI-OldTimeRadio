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
# Individual node files (bark_tts.py, batch_bark_generator.py) also have
# targeted filterwarnings calls as a belt-and-suspenders fallback.
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
    "OTR_BatchBarkGenerator": (".nodes.batch_bark_generator", "BatchBarkGenerator",   " Batch Bark Generator"),
    "OTR_BatchAudioGenGenerator":(".nodes.batch_audiogen_generator", "BatchAudioGenGenerator"," Batch AudioGen (Foley)"),
    "OTR_BatchProceduralSFX": (".nodes.batch_procedural_sfx", "BatchProceduralSFX",   " Batch Procedural SFX (Obsidian)"),
    "OTR_SignalLostVideo":    (".nodes.video_engine",          "SignalLostVideoRenderer", " Signal Lost Video"),
    "OTR_ProjectStateLoader": (".nodes.project_state",         "ProjectStateLoader",      " Project State Loader"),
    "OTR_KokoroAnnouncer":    (".nodes.kokoro_announcer",      "KokoroAnnouncer",         " Kokoro Announcer"),
    "OTR_MusicGenTheme":      (".nodes.musicgen_theme",        "MusicGenTheme",           " MusicGen Theme"),
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
    "OTR_CheckpointLoaderGated":   (".visual.checkpoint_loader_gated", "CheckpointLoaderGated", " Checkpoint Loader (gated)"),
    "OTR_UnloadAll":               (".visual.unload_all",              "UnloadAll",             " Unload All (VRAM release)"),
    # Sprint H §3.7 topology gate (Jeffrey 2026-05-17): forces
    # LTXAVTextEncoderLoader's downstream consumer to wait for
    # OTR_UnloadAll completion. Passthrough, no logic.
    "OTR_LtxBranchGate":           (".visual.ltx_branch_gate",         "LtxBranchGate",         " LTX Branch Gate (topology)"),
    # Sprint H §3.7 follow-up gate (Jeffrey 2026-05-17, mirror of
    # OTR_LtxBranchGate after b5c1441 proved the pattern): forces
    # every FLUX consumer to wait until OTR_LedgerFreezeCascade
    # has emitted its script_json (writer fully done + Gemma
    # evicted via the cascade's finally-block unload_llm).
    "OTR_FluxBranchGate":          (".visual.flux_branch_gate",        "FluxBranchGate",        " FLUX Branch Gate (topology)"),
    # Sprint H §3.7 Path G (Jeffrey 2026-05-18): deferred-loader
    # wrappers. Retest #13 produced the campaign's core finding:
    # ComfyUI's executor pre-loads CheckpointLoaderSimple +
    # LTXAVTextEncoderLoader at graph-start regardless of any
    # downstream gate ([FluxBranchGate] fire reading 22.18 GiB
    # confirmed the worst-case scenario). Gates defer consumer
    # execution; deferred loaders defer the loader's GPU
    # materialization itself by adding gate_signal (STRING,
    # forceInput=True) as a required input. ComfyUI topo-sorts
    # them downstream of OTR_LedgerFreezeCascade.script_json.
    "OTR_DeferredCheckpointLoader": (".nodes._otr_deferred_loaders",   "DeferredCheckpointLoader", " Deferred Checkpoint Loader (gate-bound)"),
    "OTR_DeferredLtxTextEncoderLoader": (".nodes._otr_deferred_loaders", "DeferredLtxTextEncoderLoader", " Deferred LTX Text Encoder Loader (gate-bound)"),
    "OTR_BatchFluxRender":         (".visual.batch_flux_render",       "BatchFluxRender",       " Batch FLUX Render"),
    # BUG-LOCAL-078 fix (2026-05-03 EVENING). Per-cast portrait render.
    # Generates one clean head-and-shoulders FLUX portrait for each
    # cast member, saves to per-episode portraits/<char_id>_portrait.png,
    # stamps cast[i].portrait_path into the ledger so HuMo's tier 1
    # portrait lookup hits instead of falling through to the env-still
    # tier 4 stopgap.
    "OTR_BatchFluxPortraitRender": (".visual.batch_flux_portrait_render", "BatchFluxPortraitRender", " Batch FLUX Portrait Render (per-cast)"),
    # v2.0 MIT-original video-chain nodes (replace VideoHelperSuite-GPL deps)
    "OTR_VideoConcat":             (".nodes.otr_video_concat",         "OTRVideoConcat",        " OTR Video Concat"),
    # v2.0 read-only ledger/script adapter for multi-pass FLUX rendering
    # (legacy "Director" naming retired in voice-path-cleanbreak S23.9).
    "OTR_VideoPlan":               (".nodes.otr_video_plan",           "OTRVideoPlan",          " OTR Video Plan"),
    # v2.0 multi-clip shot expansion (needs shot durations from audio
    # timeline). Sprint E E8 rename: was OTR_ShotDurationCalculator;
    # the new name surfaces the stub-nature of the current implementation
    # (hand-crafted JSON array of durations until Bark audio-timeline
    # wiring lands). Per CLAUDE.md no-back-compat: no alias for the old
    # name.
    "OTR_FixedShotDurationStub":  (".nodes.otr_shot_duration_calculator", "OTRFixedShotDurationStub", " OTR Fixed Shot Duration Stub"),
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
    # HuMo model-tier loader (BUG-LOCAL-265, round-robin 2026-05-24,
    # Option C). One node loads the full HuMo stack (diffusion model +
    # umt5 + wan VAE + Whisper) for one of three tiers:
    # low_vram_default (HuMo-1.7B, the shipped default),
    # high_quality_unsafe_on_16gb (HuMo-17B/14B fp8 + distill LoRA,
    # opt-in), experimental_gguf.
    # Carries the Lever-1 pipeline-residue free + the hard
    # auto-downgrade rule so a 16 GB card never silently hits the HuMo
    # in-pipeline thrash path. Feeds OTR_BatchHumoRender; tiering lives
    # here, upstream, so BatchHumoRender keeps its pre-loaded-inputs
    # surface.
    "OTR_HuMoTierLoader":          (".nodes._otr_humo_tier_loader", "HuMoTierLoader", " HuMo Tier Loader (1.7B default / 17B opt-in)"),
    # v2.0 in-graph batch HuMo lip-sync renderer. Loads HuMo + Lora +
    # CLIP + VAE + Whisper once, loops every ledger line internally,
    # writes per-line clips at output/otr/videos/<ep_id>/<line_id>.mp4.
    # Replaces scripts/render_humo_batch.py for production use; the
    # CLI script stays as an ad-hoc smoke tool.
    "OTR_BatchHumoRender":         (".nodes.batch_humo_render", "BatchHumoRender", " Batch HuMo Render"),
    # v2.0 in-graph batch LTX-2 renderer for non-character ledger lines
    # (announcer / music_open / music_close / music_inter / sfx).
    # Loads LTX-Video 2B + T5 once, loops every non-character ledger
    # line internally, feeds the radio_bookend.png as BOTH start and
    # end keyframes via LTXVAddGuide for seamless loop, writes per-line
    # clips alongside HuMo's output at
    # output/otr/videos/<ep_id>/<line_id>.mp4. Architecture locked
    # 2026-05-01 with Jeffrey after BUG-LOCAL-129 settled that HuMo
    # cannot animate non-face references; LTX is the answer for
    # "the radio is the visual performer for non-dialogue."
    "OTR_BatchLTXRender":          (".nodes.batch_ltx_render", "BatchLTXRender", " Batch LTX Render (radio for music/sfx/announcer)"),
    # v2.0 in-graph episode compositor. Pillarbox HuMo center 624x1080
    # in 1920x1080 canvas + additive-blend SignalLostVideo proc gen at
    # 50% opacity + audio mux from proc gen. Single ffmpeg invocation.
    # Replaces scripts/render_episode_concat.py for production use.
    "OTR_VideoComposite":          (".nodes.video_composite", "VideoComposite", " Video Composite (1080p)"),
    # v2.0 final-stage RTX VSR upscaler. Path-in / path-out wrapper around
    # NVIDIA's RTXVideoSuperResolution that preserves C7 audio identity:
    # decodes video frames in chunks via ffmpeg pipe, runs nvvfx HW-accel
    # upscale, encodes silent libx264 yuv420p, then muxes the original mp4
    # audio with -c:a copy (zero audio re-encode). Bypassable via the
    # `bypass` widget for raw 832x480 deliverables.
    "OTR_RTXUpscale":              (".nodes.rtx_upscale", "RTXUpscale", " RTX VSR Upscale (1080p)"),
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
    # BUG-LOCAL-031 QA tee (2026-05-03 EVENING). Per-stage video copy
    # so smoke workflows can preserve the output of every pipeline
    # node before the next node potentially clobbers the canonical
    # filename. Pure side-effect; passes the upstream path through
    # unchanged so production wiring is not affected.
    "OTR_SaveCopy": (".nodes.otr_save_copy", "SaveCopy", " Save Copy (per-stage QA tee)"),
    # Sprint 10B Wave 1 Agent D (2026-05-27): Director agent. Reads a
    # news seed + locked cast and emits a structured DirectorBrief
    # (design Section 3.1). DORMANT in Wave 1: workflow JSON entry
    # lands with the output socket disconnected; Wave 2 Agent F's
    # Story Room loop wires the brief into the Writer's loop. PD6
    # contract: no model_id widget, technical_model flows over a
    # STRING input socket from the writer's broadcast output.
    "OTR_DirectorBrief": (".nodes.OTR_DirectorBrief", "OTR_DirectorBrief", " OTR Director Brief (dormant)"),
    # BUG-LOCAL-231 Step 6 bisect (2026-05-19): minimal STRING-emit node
    # used to feed `gate_signal` (forceInput=True) on
    # OTR_DeferredCheckpointLoader / OTR_FluxBranchGate variants without
    # bringing the full ledger flow into the bisect graph. TEMPORARY --
    # delete this registration + nodes/_bisect_string_source.py + every
    # workflows/_bisect_*.json after BUG-LOCAL-231 closes (Step 9
    # cleanup of the 9-step plan).
    "OTR_BisectStringSource": (".nodes._bisect_string_source", "BisectStringSource", " Bisect String Source (BUG-LOCAL-231 -- delete at close)"),
    # Sprint 10B Wave 1 Agent E (2026-05-27): Editor agent (DORMANT).
    # Consumes a DirectorBrief (Section 3.1) + Writer draft and emits a
    # typed EditorVerdict (Section 3.2) scoring against the existing
    # rubric axes via constrained decode against EditorVerdictSchema.
    # Wave 1 registers the node; Wave 2 Agent F (Story Room loop) wires
    # it into the Writer's revision cycles. Today's pipeline is
    # unchanged (sockets ship disconnected in the workflow JSON). PD6
    # contract: no model_id widget; technical_model flows over a STRING
    # input socket from the writer's broadcast output.
    "OTR_EditorPass": (".nodes.OTR_EditorPass", "OTR_EditorPass", " OTR Editor Pass (dormant)"),
    # Sprint 10B Wave 2 Agent F (2026-05-27): Story Room loop. Wires the
    # Director (Wave 1 Agent D), the Writer (creative slot, free-form
    # prose), and the Editor (Wave 1 Agent E, technical slot) into a
    # bounded conversation that converges on a publishable Writer draft.
    # Emits a StoryRoomTranscript (design Section 3.3) for Wave 2 Agent G
    # to extract structured artifacts from. Ships behind use_story_room:
    # bool = False so PD1 byte-identity holds; Wave 3 operator A/B flips
    # it on. PD6: both slot ids arrive over STRING forceInput sockets.
    "OTR_StoryRoom": (".nodes.OTR_StoryRoom", "OTR_StoryRoom", " OTR Story Room (dormant)"),
    # Sprint 10B Wave 2 Agent G (2026-05-27): Story Room transcript-to-
    # structured extraction. Consumes the Wave 2 Agent F transcript and
    # emits a typed StoryRoomExtraction (design Section 3.4) whose
    # dict-shaped fields match the keys the announcer / continuity
    # ledger / music+SFX cue builder / bark generator already consume.
    # DORMANT in Wave 2: when the upstream payload is empty / dormant /
    # malformed the node returns a sentinel without an LLM call. Wave 3's
    # bridge wires the structured output into the legacy ledger when the
    # operator A/B confirms use_story_room ships on by default. PD6:
    # technical_model arrives over a STRING forceInput socket.
    "OTR_StoryRoomExtract": (".nodes.OTR_StoryRoomExtract", "OTR_StoryRoomExtract", " OTR Story Room Extract (dormant)"),
    # Wave 3 bridge (2026-05-27): commit the Story Room extraction into
    # the in-flight ledger so the freeze cascade + Bark see Story Room
    # dialogue instead of the legacy compose_line output. Defaults
    # commit=False (no-op pass-through, PD1 byte-identity holds). When
    # commit=True AND extraction.status='ok', walks extraction.dialogue
    # and overwrites ledger.lines[*].text per beat_id. Place between
    # OTR_LedgerScriptWriter.script_json and OTR_LedgerFreezeCascade
    # input. script_json passes through verbatim; the actual ledger
    # write is a side-effect on the in-flight singleton.
    "OTR_StoryRoomCommit": (".nodes.OTR_StoryRoomCommit", "OTR_StoryRoomCommit", " OTR Story Room Commit (Wave 3 bridge, dormant)"),

    # Sprint 4.1 wire-up (2026-05-28): best-of-N beat sheet selector.
    # Pure-Python mechanical scorer; no LLM call. Accepts up to 3
    # candidate Stage1Plan JSON inputs, validates each via the Sprint
    # 2 structural validators, picks the highest-total-score eligible
    # winner with deterministic lowest-index tie break. Raises
    # NoValidBeatSheetError-as-empty-output when every candidate
    # fails; operator wires upstream Stage 1 fan-out manually.
    # PD6: no model_id widget (no LLM call to slot).
    "OTR_BeatSelector": (".nodes.OTR_BeatSelector", "OTR_BeatSelector", " OTR Beat Selector (Sprint 4 best-of-N)"),
}

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

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
