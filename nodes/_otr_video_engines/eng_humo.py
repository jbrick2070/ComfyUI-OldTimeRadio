"""HuMo audio-driven-face motion adapter (A-S6 / CW-7) -- in-process, default-OFF.

HuMo is OTR's heaviest engine: an audio-conditioned image-to-video model that
animates a reference PORTRAIT (init_image) in sync with a speech AUDIO reference
into a talking-character clip. It is the ``audio_driven_face`` family -- the
talking-head path for the announcer + character roles. Like ltx_video / wan_i2v
it runs IN-PROCESS in the main ComfyUI cu130 venv (it loads
MODEL+CLIP+VAE+AUDIO_ENCODER internally via ``comfy.model_management``) and is the
SINGLE resident heavy engine while it holds the AS-3 lease. Native output is
480x832 @ 25 fps; a portrait init is fit to the canvas with ONE uniform scale,
never stretched, and the compositor pillarboxes the portrait clip (pre-mortem N9).

Registered DEFAULT-OFF / dark (empty ``default_roles`` + gated behind
``OTR_ENABLE_HUMO``) so it shows in the static per-role dropdown (V-6) but is
never a default and fails CLOSED until the operator enables it AND the HuMo
wrapper + checkpoints are installed and verified on the GPU box (the A-S6 smoke).
No model is "primary" -- HuMo is one peer adapter among the motion engines.

NO FALLBACKS (operator directive 2026-07-02: NO fallbacks / NO auto-defaults
anywhere): a render-time failure raises a named RenderError LOUD -- there is
no degrade to the 1.7B tier and no still floor (``fallback_engine = None`` on
every tier). The audio that drives HuMo is the FROZEN
master; HuMo emits an ALWAYS-SILENT clip (``has_audio`` False) -- only
``OTR_MasterAudioMux`` ever adds audio (invariant V-1).

Cold-import clean (V-12): module scope imports only stdlib + the dep-free shared
helpers + the registry. torch / the HuMo wrapper are imported LAZILY in ``load``
/ ``render_clip`` (the GPU-smoke render slice), never here. UTF-8, no BOM,
ASCII-only source.

Config (env): ``OTR_ENABLE_HUMO`` opt-in flag; ``OTR_HUMO_CKPT`` the primary
checkpoint path the load probe checks (verify-at-build; the full multi-handle
MODEL+CLIP+VAE+AUDIO_ENCODER load is confirmed on the GPU box).
"""
from __future__ import annotations

import os

from . import motion_common as _MC
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_SOFT_REFERENCE, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

# A-ship in-process forward (the native graph PROVEN in scripts/render_humo_batch.py
# build_humo_prompt, verified on this RTX 5080 at 480x832 fp8). The graph TOPOLOGY +
# widget values are legacy-proven; the exact checkpoint / encoder FILENAMES are
# env-overridable (VERIFY-ON-GPU) so the operator confirms them against the
# installed models without editing code. Native portrait 480x832 @ 25 fps; the Wan
# 2.1 VAE 4n+1 length rule is enforced via wrapper_bridge.quantize_frames_4n1.
_HUMO_MIN_FRAMES = 33          # below this has hung this hardware (legacy floor)
_HUMO_MAX_FRAMES = 177         # last empirically verified ceiling at 480x832 fp8
# Route-A (2026-06-28): the 14B fp8 tier rides ~15.9 GB at 832x480 -- thin
# headroom ACCEPTED by the operator, BOUNDED by a per-beat render-frame cap. 49
# (4n+1) is the bakeoff-proven safe length (docs/2026-06-27-humo-bakeoff: zero
# OOM at 832x480/<=49f, single + cross-engine resident). Beats longer than the
# cap render at the cap then mirror-extend to the audio target (exact-fit);
# raise it ONLY after a higher-frame GPU probe. env: OTR_HUMO_14B_SAFE_FRAMES.
_HUMO_14B_SAFE_RENDER_FRAMES = 49
# (HuMo render dims live in _otr_shared.aspect now -- portrait 480x832 / wide
# 832x480 -- selected by each engine's render_aspect, so the constants moved out.)
# An ASCII negative (CLAUDE.md: ASCII-only source). HuMo's best negative is the
# ByteDance Chinese default; set OTR_HUMO_NEGATIVE to it on the box to match the
# legacy template exactly.
_HUMO_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "extra fingers, bad hands, bad face, static, watermark, text")

# Production keystone tier baked in as the DEFAULT (the FAST pre-refactor config
# that scored 37.5/45): the 14B Kijai fp8 UNET + the lightx2v 480p distill LoRA
# at 6 steps / cfg 1.0 / ModelSamplingSD3 shift 8. The LoRA, steps, cfg and shift
# defaults below already match this tier; only the UNET name differs from the
# legacy 1.7B placeholder, so this is the single value that pins the tier. The
# weight lives in diffusion_models and is resolved via ComfyUI folder_paths, so
# it is found through extra_model_paths.yaml without a hardcoded box path. The
# 1.7B tier (no-LoRA, ~20 steps) is far slower; override OTR_HUMO_UNET_NAME /
# OTR_HUMO_CKPT + OTR_HUMO_LORA_NAME=none + OTR_HUMO_STEPS only to fall back to it.
_HUMO_DEFAULT_UNET = "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors"


#: S1 (2026-07-25) per-model still plan for the PORTRAIT HuMo engines --
#: ``humo`` and ``humo_1.7B``, both shipping ``render_aspect="portrait"``
#: (spec section 3, Shape A -- scene spine, audio-driven-face variant with
#: portrait REQUIRED). FILE-LOCAL, fully declared.
#:
#: S1b SPLIT (2026-07-25): S1 had all FOUR HuMo engines sharing ONE plan
#: object across TWO shipped aspects. That was survivable only while the
#: portrait row's ``framing_geometry`` was a paraphrase. It is not survivable
#: now: the producer picks ``PORTRAIT_GEOMETRY`` for a portrait renderer and
#: ``WIDE_PORTRAIT_GEOMETRY`` for a wide one (``_style_anchor_for_aspect`` --
#: a three-quarter body shot decapitates the subject in a short landscape
#: frame, the 2026-06-17 operator catch), so one static string cannot serve
#: both. The wide siblings get their own plan below. This also honours the
#: spec's own "no inheritance or shared defaults" rule, and
#: ``tests/test_still_plan_layer2_parity.py`` now enforces it so the sharing
#: cannot regrow.
#:
#: The ``aspect="inherit_engine"`` FIELD is unchanged -- it still drives the
#: DIMENSIONS through ``resolve_row_aspect``. Only the authored layer-2 TEXT
#: is pinned per file. S2's HuMo cutover is the FOUR hosts-off bookend
#: role-cells (portrait humo / humo_1.7B x announcer / music, redirected by
#: ``_enforce_radio_is_host`` to the WIDE ``ltx_audio_in`` that renders them);
#: with ``OTR_ENABLE_HUMO_HOSTS=1`` a portrait HuMo keeps its portrait still.
_HUMO_STILL_PLAN = (
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
                 required="always",
                 framing_geometry=(
                     ("in-character cinematic three-quarter portrait, full "
                      "head and face clearly visible with natural headroom "
                      "above the head (never crop the top of the head)")),
                 style_tail_policy="full"),
)

#: S1b (2026-07-25) per-model still plan for the LANDSCAPE HuMo siblings --
#: ``humo_1.7B_169`` and ``humo_14B_169``, both shipping
#: ``render_aspect="wide"``. Structurally IDENTICAL to
#: :data:`_HUMO_STILL_PLAN`; it exists as a SEPARATE plan object because the
#: portrait row's layer-2 text differs by shipped aspect (see the split note
#: above). Fully declared -- no inheritance from, and no shared rows with,
#: the portrait plan, per spec section 6.
#:
#: The ComfyUI dropdown labels this split to the operator as "(portrait)" vs
#: "(16:9)" and that is a VISIBLE PRODUCT CONTRACT: these two siblings were
#: ALREADY wide before this build, nothing about them flips, and a blanket
#: "HuMo -> wide" would be a regression rather than a fix.
_HUMO_169_STILL_PLAN = (
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
                 required="always",
                 framing_geometry=(
                     ("in-character cinematic medium shot, head and "
                      "shoulders, face clearly visible, subject centred with "
                      "natural headroom above the head (never crop the top of "
                      "the head)")),
                 style_tail_policy="full"),
)


@register
class HuMoEngine(_MC.MotionEngineBase):
    """The humo audio-driven-face adapter (in-process, default-OFF / dark)."""

    name = "humo"
    family = "audio_driven_face"
    #: S1 per-model still plan (see ``_HUMO_STILL_PLAN`` above -- audio-
    #: driven-face with portrait REQUIRED).
    still_plan = _HUMO_STILL_PLAN
    # RADIO IS THE HOST (reversal of Route-A 2026-06-28, operator 2026-06-30):
    # HuMo's finetuned weights only animate a FACE. Route-A had added
    # announcer_visual/music_visual here + a radio-face-still workaround
    # (render_driver._radio_bookend_image, since REMOVED) so the operator could
    # pick HuMo for the music/announcer bookends; the eyeballed render showed a
    # generic human host, not a radio -- confirming the ORIGINAL 2026-05-01
    # BUG-LOCAL-129 finding (nodes/_otr_speaker_role.py) still holds. HuMo now
    # serves ONLY character_video (real dialogue lip-sync). This tuple is
    # UI-SORT/self-description metadata only (role_compat.engine_fits_role is
    # capability-only and does not consult it -- see registry.VideoEngineRegistry
    # docstring); the actual hard gate is render_driver._enforce_radio_is_host,
    # which wires the previously-dormant _otr_speaker_role.is_never_humo_role
    # into real dispatch and redirects any announcer_visual/music_visual +
    # audio_driven_face pick to ltx_audio_in (the LTX-2.3 audio-in lane already
    # DEFAULT for those two roles -- see eng_ltx_av.py default_roles).
    roles = ("character_video",)
    default_roles = ()
    required_inputs = ("audio_ref", "init_image")
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). ``_HUMO_MIN_FRAMES`` ..
    #: ``_HUMO_MAX_FRAMES`` -- 33 is the legacy floor (below it has hung this
    #: hardware) and 177 is the last empirically verified ceiling at 480x832
    #: fp8. 4n+1 stride. Inherited by all four HuMo variants.
    #: CONTINUITY: soft_reference. The graph wires LoadImage ->
    #: WanHuMoImageToVideo ``ref_image``, NOT ``start_image``: per the Bug
    #: Bible the reference is an identity hint, not a first-frame lock. These
    #: beats take an honest jump cut rather than a pretended chain.
    frame_contract = FrameContract(
        min_frames=_HUMO_MIN_FRAMES,
        max_frames=_HUMO_MAX_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    #: Per-beat render-frame cap (Route-A). ``None`` = uncapped (use
    #: ``_HUMO_MAX_FRAMES``, the legacy behaviour for humo / humo_1.7B*). A heavy
    #: tier that rides thin VRAM headroom (the 14B) overrides this with a safe
    #: integer; render_clip then renders at the cap and EXACT-FITS the decoded
    #: frames to the audio-derived target (trim/mirror-extend) so frame_count
    #: still matches.
    safe_render_frames = None
    #: Render aspect == this engine's IDENTITY (the operator picks the look with
    #: one dropdown choice): 'portrait' 480x832 (the classic pillarbox) here on
    #: the base; the humo_1.7B_169 variant overrides to 'wide' 832x480. The
    #: character still + composite canvas follow this (see _otr_shared.aspect).
    render_aspect = "portrait"
    fallback_engine = None               # NO FALLBACKS (2026-07-02): fail LOUD

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        # Explicit override wins; else resolve the default 14B UNET via ComfyUI
        # folder_paths (honors extra_model_paths.yaml), else a best-effort join
        # for the cheap existence check (headless / no folder_paths).
        env = os.environ.get("OTR_HUMO_CKPT")
        if env:
            return env
        name = os.environ.get("OTR_HUMO_UNET_NAME") or _HUMO_DEFAULT_UNET
        try:
            import folder_paths  # type: ignore
            p = folder_paths.get_full_path("diffusion_models", name)
            if p:
                return p
        except Exception:  # noqa: BLE001 - headless/CPU: fall back to a join
            pass
        return os.path.join(_COMFY_ROOT, "models", "diffusion_models", name)

    def _installed(self):
        """True iff the primary checkpoint exists on disk (no import -- cheap,
        headless-safe). The full MODEL+CLIP+VAE+AUDIO_ENCODER multi-handle load
        (+ the low/high/gguf tier pick) is the GPU smoke."""
        return os.path.exists(self._ckpt_path())

    # ---- sampler tier (overridable per HuMo tier; the 1.7B downgrade isolates
    # ---- its own steps/cfg so the 14B OTR_HUMO_STEPS/CFG never bleed into it) --
    def _steps(self):
        """KSampler steps for this tier. 14B + lightx2v distill = 6 (fast)."""
        return int(os.environ.get("OTR_HUMO_STEPS", "6"))

    def _cfg(self):
        """KSampler cfg for this tier. 14B + lightx2v distill = 1.0."""
        return float(os.environ.get("OTR_HUMO_CFG", "1.0"))

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward on checkpoint presence
        (verify-at-build). Imports nothing heavy -- runs at
        lock/validate time on the CPU box. HuMo loads in-process; its
        SageAttention tolerance is a GPU-smoke verify item, NOT a hard CPU gate
        (unlike ltx_video's BUG-070 int8-PV abort)."""
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "humo checkpoint not found at %s; install the HuMo wrapper + "
                "ckpt and verify on the GPU box (set OTR_HUMO_CKPT)"
                % self._ckpt_path(), kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- beat session (WIRE-W4a, 2026-07-29) ----
    #: The ComfyUI ``folder_paths`` categories each loader token resolves
    #: through -- the same lookup the loader NODE will do, which is the only
    #: resolution the identity may describe (a ``*_DIR``-style override that
    #: the loader cannot see must not be able to satisfy it).
    _LOADER_CATEGORIES = {
        "unet": ("diffusion_models", "unet"),
        "lora": ("loras",),
        "clip": ("text_encoders", "clip"),
        "vae": ("vae",),
        "whisper": ("audio_encoders",),
    }

    def _humo_file_receipt(self, path):
        """A BOUNDED, stable receipt for a model file: (basename, size,
        mtime_ns). Deliberately NOT a content hash -- ``session_identity()`` is
        re-read before EVERY segment and hashing a 14B UNET per segment would
        cost more than the render it protects. Size + mtime catches a swapped
        or rebuilt weight, which is the drift being guarded against.

        Its own four lines ON PURPOSE. ``wan_shared`` says the same thing for
        the WAN lanes and ``eng_ltx_8gb`` for its own, and that file's docstring
        records why: reaching across lanes for a four-line stat is the coupling
        this build keeps paying for. The MECHANISM is a convention; the DATA is
        per adapter.

        ``None`` when the file cannot be stat'ed -- the CALLER decides whether
        that is fatal, because the caller knows which NAMED refusal to raise."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (os.path.basename(path), int(st.st_size), int(st.st_mtime_ns))

    def _humo_token_path(self, categories, name):
        """Where the LOADER would find ``name``: ``folder_paths`` first (so
        ``extra_model_paths.yaml`` is honoured), then the standard
        ``models/<category>/`` layout. ``None`` when absent everywhere -- the
        offline invariant means a missing file is fail-closed, never fetched."""
        for category in categories:
            try:
                import folder_paths                  # ComfyUI runtime only
                hit = folder_paths.get_full_path(category, name)
                if hit:
                    return hit
            except Exception:            # noqa: BLE001 -- no ComfyUI (tests)
                pass
            cand = os.path.join(_COMFY_ROOT, "models", category, name)
            if os.path.exists(cand):
                return cand
        return None

    def _humo_session_receipts(self):
        """``(label, token, receipt)`` for every file this beat's handles are.

        Built from the adapter's OWN ``_loader_names()`` so a tier that points
        at a different UNET, LoRA or encoder gets it in the identity for free
        -- the three HuMo subclasses override exactly that method and nothing
        here needs to know they exist."""
        names = self._loader_names()
        out = [("unet", names["unet"],
                self._humo_file_receipt(self._ckpt_path() or ""))]
        for label in ("lora", "clip", "vae", "whisper"):
            token = names.get(label) or ""
            if label == "lora" and self._lora_is_skipped(token):
                out.append((label, "", None))       # a tier that runs LoRA-free
                continue
            path = self._humo_token_path(
                self._LOADER_CATEGORIES[label], token) if token else None
            out.append((label, token,
                        self._humo_file_receipt(path) if path else None))
        return tuple(out)

    @staticmethod
    def _lora_is_skipped(lora_name):
        """THE ONE reading of "this tier runs LoRA-free".

        The lightx2v distill LoRA is a 14B-shaped adapter and is INCOMPATIBLE
        with the 1.7B tiers, which switch it off by name. Both the graph and
        the session have to agree about that: a session that hoisted a ``lora``
        node the graph never defines would wire a handle nothing reads, and one
        that skipped a node the graph DOES define would let it reload every
        segment. This used to be spelled out inside ``_build_graph`` alone."""
        return (not lora_name) or str(lora_name).strip().lower() in (
            "none", "skip", "off")

    def _session_node_ids(self):
        """The graph ids ``prepare`` hoists: every pure file-to-handle LOADER.

        WIDER THAN THE WAN LANES, and the difference is a real property of this
        family rather than a preference. WAN renders with ``free_after_use=True``
        precisely so umt5 and the diffusion UNET are never co-resident, so
        hoisting its CLIP would delete that mitigation. HuMo renders FULLY
        RESIDENT by contract (BUG-265: forcing inter-node eviction fragmented
        the allocator into an OOM), so every loader is resident for the whole
        render anyway -- hoisting them changes how many times they are READ,
        not how much is held.

        It is also what makes skipping the between-segment reclaim safe. See
        :meth:`render_clip`: without the hoist, segment 2's graph would build
        NEW umt5 and whisper handles beside the ones segment 1 left resident,
        and dropping the reclaim would accumulate rather than reuse.

        ``modelsampling`` is deliberately NOT here -- it is a cheap clone+patch
        of an already-resident MODEL and is correctly per-render. Neither are
        ``loadaudio``/``audioenc``: those are the one thing that genuinely
        DIFFERS per segment on a lip-synced lane."""
        ids = ["unet", "clip", "vae", "audioenc_loader"]
        if not self._lora_is_skipped(self._loader_names().get("lora")):
            ids.insert(1, "lora")
        return tuple(ids)

    def session_identity(self):
        """What this adapter's HANDLES are. ``BeatSession`` refuses a
        multi-segment beat without it -- which is why all three HuMo lanes came
        back NO_RENDER on the 2026-07-28 campaign.

        It carries the engine name (which is what picks the tier's steps, cfg
        and native dims), every loader file's token AND receipt, and whether
        the LoRA is MERGED -- because it is merged into the hoisted patcher, so
        a beat that turned it off midway would be reusing a model the later
        segments' graph no longer describes.

        It EXCLUDES per-segment state: prompt, seed, frame count, audio slice.
        It also excludes steps/cfg, which are KSampler arguments rather than
        anything baked into a handle.

        Not cached -- the whole job is to notice a weight that MOVED, so the
        receipts are re-stat'ed on every ask."""
        return (self.name,
                "lora_merged=%s" % (
                    not self._lora_is_skipped(self._loader_names().get("lora")),),
                repr(self._humo_session_receipts()))

    def prepare(self, host_caps, profile, session_ctx):
        """Load every heavy handle ONCE PER BEAT instead of once per segment.

        The ``eng_wan_i2v`` sequence, followed deliberately: ``super().prepare()``
        takes the cross-process GPU lease and resolves classes, a LOADER-ONLY
        mini-graph runs with ``on_result`` registering each patcher as it lands
        (if a later node raised, ``run_graph`` never returns a results dict and
        an unregistered patcher is VRAM nothing will ever detach), and ANY
        failure tears down what was taken before re-raising.

        ``session_ctx`` is KEPT, because :meth:`render_clip` has no other way to
        know whether it is one segment of a beat -- and on this lane that
        decides whether the post-decode VRAM reclaim runs now or at teardown.

        NO ``free_after_use``: this graph is executed exactly the way
        ``render_clip`` executes its own, because BUG-265 established that
        forcing inter-node eviction on this stack fragments the allocator into
        an OOM."""
        from . import wrapper_bridge as _wb
        # session_ctx comes back IN `prepared` from the base now (2026-08-01) --
        # the local re-add that used to live here is the base's job.
        prepared = super().prepare(host_caps, profile, session_ctx)
        try:
            classes = getattr(self, "_classes", None) \
                or _wb.resolve_graph_classes(self._node_candidates())
            bucket = prepared.setdefault("patchers", self._patchers)
            seen = {id(p) for p in bucket}

            def _register(nid, out):
                obj = out[0] if out else None
                if (obj is not None and id(obj) not in seen
                        and callable(getattr(obj, "detach", None))):
                    bucket.append(obj)
                    seen.add(id(obj))

            wanted = set(self._session_node_ids())
            # BUILT FROM THE REAL GRAPH, never hand-copied. A second literal
            # spelling of the loader inputs is a second thing to keep in step
            # with `_build_graph`, and the two drifting is exactly how a
            # session hoists a handle the render graph would have built
            # differently.
            full = self._build_graph("_session_.png", "_session_.wav",
                                     {"seed": 0}, _HUMO_MIN_FRAMES,
                                     *self._native_dims())
            loaders = {nid: spec for nid, spec in full.items() if nid in wanted}
            missing = wanted - set(loaders)
            if missing:
                raise _wb.GraphExecutionError(
                    "%s wanted to hoist %s but its render graph defines no such "
                    "node(s) -- the session and the graph disagree about what "
                    "this lane loads. NO FALLBACK."
                    % (self.name, sorted(missing)))
            results = _wb.run_graph(loaders, classes, on_result=_register)
            absent = [nid for nid in wanted if not results.get(nid)]
            if absent:
                raise _wb.GraphExecutionError(
                    "%s hoisted loader(s) %s returned no output; the beat "
                    "session has nothing to reuse and every segment would "
                    "silently reload" % (self.name, sorted(absent)))
            prepared["external_results"] = {
                nid: results[nid] for nid in wanted}
        except BaseException:
            try:
                self.teardown(prepared)
            except BaseException:          # noqa: BLE001 -- never mask the cause
                pass
            raise
        return prepared

    def teardown(self, prepared):
        """Drop the beat-scoped handles, RECLAIM once, then delegate.

        ``prepared["external_results"]`` holds strong references to every
        hoisted handle; the base teardown detaches the tracked patchers,
        unloads, releases the lease and then waits for machine-wide VRAM to
        settle, and every one of those steps would run with this dict still
        pinning the tensors it is reclaiming.

        THE RECLAIM MOVES HERE FOR A MULTI-SEGMENT BEAT (see
        :meth:`render_clip`). It exists to stop CROSS-BEAT accumulation -- "the
        resident stack drops back down before the next soak beat starts" -- and
        between two segments of ONE beat there is nothing to drop back for,
        because the next segment needs the same stack. Running it once here
        keeps the cross-beat promise exactly.

        Never raises out of teardown (it is a finally-path)."""
        from . import wrapper_bridge as _wb
        if isinstance(prepared, dict):
            prepared.pop("external_results", None)
            if (prepared.get("session_ctx") or {}).get("multi_clip"):
                try:
                    _wb.reclaim_idle_models(reason="humo beat teardown")
                except Exception:          # noqa: BLE001 -- best-effort
                    _LOG.warning("[OTR video] %s beat-teardown reclaim failed",
                                 self.name, exc_info=True)
        return super().teardown(prepared)

    # ---- in-process graph spec (proven topology; filenames VERIFY-ON-GPU) ----
    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (all core /
        comfy_extras classes used by the proven legacy HuMo graph)."""
        return {
            "unet": ("UNETLoader",),
            "lora": ("LoraLoaderModelOnly",),
            "modelsampling": ("ModelSamplingSD3",),
            "clip": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadaudio": ("LoadAudio",),
            "audioenc_loader": ("AudioEncoderLoader",),
            "audioenc": ("AudioEncoderEncode",),
            "loadimage": ("LoadImage",),
            "humo": ("WanHuMoImageToVideo",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    def _loader_names(self):
        """Model / encoder FILENAMES the loader nodes consume. Defaults are the
        legacy-proven names; each is env-overridable so the operator points at the
        installed files without editing code (VERIFY-ON-GPU)."""
        return {
            "unet": os.environ.get("OTR_HUMO_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "lora": os.environ.get(
                "OTR_HUMO_LORA_NAME",
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
            "clip": os.environ.get(
                "OTR_HUMO_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "vae": os.environ.get("OTR_HUMO_VAE_NAME", "wan_2.1_vae.safetensors"),
            "whisper": os.environ.get(
                "OTR_HUMO_AUDIO_ENCODER_NAME", "whisper_large_v3_fp16.safetensors"),
        }

    def _native_dims(self):
        """HuMo's render size from this engine's ``render_aspect``: 'portrait'
        480x832 (the classic talking-head pillarbox, humo_1.7B) or 'wide' 832x480
        (the 16:9 character beat, humo_1.7B_169). The aspect is the engine's
        identity, so the operator picks the look with ONE dropdown choice.
        Explicit OTR_HUMO_WIDTH/HEIGHT still win (manual override)."""
        from .._otr_shared.aspect import humo_dims_for_aspect
        _dw, _dh = humo_dims_for_aspect(self.render_aspect)
        w = int(os.environ.get("OTR_HUMO_WIDTH", _dw))
        h = int(os.environ.get("OTR_HUMO_HEIGHT", _dh))
        return w, h

    def _build_graph(self, image_name, audio_name, plan, length, width, height,
                     external_results=None):
        """The declarative HuMo graph (wrapper_bridge.run_graph format). Mirrors
        the proven build_humo_prompt wiring node-for-node:
        UNETLoader->LoRA->ModelSamplingSD3->KSampler (model); CLIPLoader->pos/neg;
        VAELoader; LoadAudio->AudioEncoderEncode(+AudioEncoderLoader);
        LoadImage->ref_image; WanHuMoImageToVideo->KSampler->VAEDecode.

        ``external_results`` names the ids the CALLER already produced and owns
        for the whole beat (WIRE-W4a: :meth:`prepare` loads every loader once).
        Those node DEFINITIONS are omitted while every wire that reads them
        stays -- the executor resolves them from the caller's handles instead.
        Omission is a contract, not an optimisation: ``run_graph`` REFUSES a
        graph that also defines an id the caller supplied, because then two
        parties claim to produce one output.

        Conditional on purpose. A caller that prepared nothing gets the loader
        nodes exactly as before, byte for byte -- which is the single-clip path
        and every sibling test that hand-builds ``prepared={"patchers": []}``."""
        from . import wrapper_bridge as _wb
        names = self._loader_names()
        steps = self._steps()
        cfg = self._cfg()
        positive = plan.get("text_prompt") or "a person speaking, subtle facial motion"
        negative = os.environ.get("OTR_HUMO_NEGATIVE", _HUMO_DEFAULT_NEGATIVE)
        W = _wb.Wire
        # The lightx2v distill LoRA is a 14B-shaped adapter: it is INCOMPATIBLE
        # with the 1.7B tier (shape-mismatch -> not merged + wasted VRAM). Make
        # it optional so the 1.7B tier runs LoRA-free (set OTR_HUMO_LORA_NAME to
        # none/skip and raise OTR_HUMO_STEPS, since the distill shortcut is gone).
        lora_name = names["lora"]
        skip_lora = self._lora_is_skipped(lora_name)
        graph = {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": names["unet"],
                                "weight_dtype": "default"}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": names["clip"], "type": "wan",
                                "device": "default"}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "vae": {"class": "vae", "inputs": {"vae_name": names["vae"]}},
            "loadaudio": {"class": "loadaudio", "inputs": {"audio": audio_name}},
            "audioenc_loader": {"class": "audioenc_loader",
                                "inputs": {"audio_encoder_name": names["whisper"]}},
            "audioenc": {"class": "audioenc",
                         "inputs": {"audio_encoder": W("audioenc_loader", 0),
                                    "audio": W("loadaudio", 0)}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "humo": {"class": "humo",
                     "inputs": {"width": int(width), "height": int(height),
                                "length": int(length), "batch_size": 1,
                                "positive": W("pos", 0), "negative": W("neg", 0),
                                "vae": W("vae", 0),
                                "audio_encoder_output": W("audioenc", 0),
                                "ref_image": W("loadimage", 0)}},
        }
        model_src = "unet"
        if not skip_lora:
            graph["lora"] = {"class": "lora",
                             "inputs": {"lora_name": lora_name,
                                        "strength_model": 1.0,
                                        "model": W("unet", 0)}}
            model_src = "lora"
        graph["modelsampling"] = {
            "class": "modelsampling",
            "inputs": {"shift": 8.0, "model": W(model_src, 0)}}
        graph["ksampler"] = {
            "class": "ksampler",
            "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                       "cfg": cfg, "sampler_name": "uni_pc",
                       "scheduler": "simple", "denoise": 1.0,
                       "model": W("modelsampling", 0),
                       "positive": W("humo", 0), "negative": W("humo", 1),
                       "latent_image": W("humo", 2)}}
        graph["vaedecode"] = {
            "class": "vaedecode",
            "inputs": {"samples": W("ksampler", 0), "vae": W("vae", 0)}}
        for nid in set(external_results or ()):
            graph.pop(nid, None)
        return graph

    def _retain_model_patchers(self, results, prepared):
        """Best-effort V-4: keep the MODEL ModelPatchers the graph produced so
        teardown can detach(unpatch_all=True) them (the lease + VRAM settle-wait in
        MotionEngineBase.teardown is the real residency guard)."""
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("unet", "lora", "modelsampling"):
            out = results.get(nid)
            if not out:
                continue
            obj = out[0]
            if id(obj) not in seen and callable(getattr(obj, "detach", None)):
                bucket.append(obj)
                seen.add(id(obj))

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Fail CLOSED until installed, then RESOLVE the installed ComfyUI wrapper
        node classes (fail-closed NAMED if absent). The heavy weight load happens
        when the loader nodes execute inside render_clip (ComfyUI's own model
        management), so load() stays cheap and the AS-3 lease brackets the real
        residency. The live VRAM<=14.5 GB peak + render-twice pixels are the A-S6
        GPU smoke (operator)."""
        if not self._installed():
            raise RuntimeError(
                "humo not installed: checkpoint missing at %s -- install the HuMo "
                "wrapper + ckpt, set OTR_ENABLE_HUMO=1, and run the A-S6 GPU "
                "smoke" % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE audio-driven-face clip via the in-process HuMo wrapper graph.

        Stages the portrait + speech into ComfyUI's input dir (the proven legacy
        LoadImage / LoadAudio pattern), executes the proven node graph in
        dependency order, encodes the decoded IMAGE batch to a SILENT bt709 clip
        (V-1), retains the MODEL patchers for V-4 teardown, and asserts the
        mid-render NVML ceiling. Returns the raw ``{out_path, frame_count}`` that
        canonicalize() normalises. Fail-closed NAMED if a wrapper node is missing
        or an input is absent."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["audio_path"] or not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "humo requires both audio_ref and init_image (got audio=%r "
                "init_image=%r)" % (plan["audio_path"], plan["init_image"]))
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        audio_name = _wb.stage_into_comfy_input(plan["audio_path"])
        image_name = _wb.stage_into_comfy_input(plan["init_image"])
        target_fc = int(plan["target_frame_count"] or 0)
        # Route-A: a capped tier (the 14B) renders at its VRAM-safe cap, then
        # exact-fits the decoded frames back to the audio target below. Uncapped
        # tiers (humo / humo_1.7B*) keep the legacy 177-frame ceiling unchanged.
        cap = self.safe_render_frames
        if cap is not None:
            cap = int(os.environ.get("OTR_HUMO_14B_SAFE_FRAMES", cap))
        render_max = cap if cap is not None else _HUMO_MAX_FRAMES
        length = _wb.quantize_frames_4n1(
            target_fc or self.target_fps,
            min_frames=_HUMO_MIN_FRAMES, max_frames=render_max)
        width, height = self._native_dims()
        # The beat-scoped handles prepare() hoisted, if any. Their node
        # definitions drop out of the graph and run_graph resolves the wires
        # from these instead -- and it adds them to `keep`, so nothing can evict
        # a handle the NEXT segment still needs.
        session = (prepared or {}) if isinstance(prepared, dict) else {}
        external = session.get("external_results") or {}
        multi_clip = bool((session.get("session_ctx") or {}).get("multi_clip"))
        graph = self._build_graph(image_name, audio_name, plan, length, width,
                                  height, external_results=external)
        # Render FULLY RESIDENT -- the proven BUG-265 low_vram_default path: the
        # HuMo-1.7B stack (3.3 GB + umt5/whisper) stays resident with zero offload
        # on a 16 GB card. (No free_after_use: forcing inter-node model eviction
        # only fragmented the allocator into an OOM -- it is NOT what the working
        # OTR_BatchHumoRender does.)
        results = _wb.run_graph(graph, classes, external_results=external)
        images = results[self._TERMINAL][0]                   # VAEDecode IMAGE batch
        self._retain_model_patchers(results, prepared)
        frames = _wb.images_to_uint8(images)
        # Route-A exact-fit: a capped tier may have rendered FEWER frames than the
        # audio target (or, on a tiny beat quantized up to the floor, more) -- trim
        # so the encoded clip's frame_count EQUALS target_fc (else the manifest
        # timing drifts).
        #
        # NO MIRROR ON A LIP-SYNCED LANE (operator directive 2026-07-25: "lip
        # sync needs continuity, no render backwards, that doesn't work").
        # HuMo is ``audio_driven_face`` -- its whole output is a mouth driven by
        # this beat's speech. ``fit_frames_to_target`` used to mirror-extend a
        # SHORT capped render up to the audio target, and the back half of that
        # mirror cycle is the render in REVERSE, so a capped 14B beat longer
        # than the cap shipped a face that speaks forwards and then unspeaks
        # backwards, in time with forward audio. Trimming is still fine (it
        # never reverses time); a SHORT render is now TERMINAL and LOUD, with
        # the remedy in the message. This can fail a beat that previously
        # "succeeded" -- those episodes had backwards lip sync.
        if cap is not None and target_fc > 0:
            try:
                frames = _wb.fit_frames_to_target(
                    frames, target_fc, allow_mirror=False)
            except _wb.MirrorExtensionForbidden as exc:
                raise _wb.GraphExecutionError(
                    "%s: %s. This tier's VRAM-safe cap is %d frame(s) and the "
                    "beat needs %d. NO FALLBACK -- fix the routing, not the "
                    "frames: raise OTR_HUMO_14B_SAFE_FRAMES if the card can "
                    "afford it, route this beat to an uncapped HuMo tier, or "
                    "give the beat a shorter line. Mirroring it would play the "
                    "mouth backwards." % (
                        self.name, exc, int(cap), int(target_fc))) from exc
        out_path = otr_engine_tmp_mp4("otr_humo_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # Restore the proven HuMo VRAM discipline the refactor dropped: the frames
        # are on disk, so evict the umt5 CLIP + whisper encoders (BUG-291 detach
        # reclaim, the same the legacy free_otr_pipeline_residue used as
        # inter-phase cleanup) so the resident stack drops back down before the
        # next soak beat starts (no cross-beat accumulation). LOUD; no
        # unload_all_models (V-4/V-5).
        #
        # NOT BETWEEN THE SEGMENTS OF ONE BEAT (WIRE-W4a). The sentence above
        # is the whole justification -- "before the next soak beat starts" --
        # and between two segments of the SAME beat there is nothing to drop
        # back for, because the next segment needs the same stack. Reclaiming
        # here would detach(unpatch_all=True) the very handles ``prepare``
        # hoisted, so the "one load per beat" contract would survive as a
        # Python reference while the weights bounced to CPU and back every
        # segment. ``teardown`` runs it once, which keeps the cross-beat
        # promise exactly as it was.
        if not multi_clip:
            _wb.reclaim_idle_models(reason="humo post-decode")
        # M7 (added 2026-07-27): PROVE the silent-clip color/stream contract on
        # the emitted mp4 before canonicalize self-declares it -- ffprobe is the
        # source of truth, the hardcoded dict is not. Four siblings (wan_i2v,
        # wan_ti2v, ltx_8gb, ltx_video) have carried this line; this adapter and
        # eng_ltx_av did not, so their clips reached the ledger entirely
        # self-declared while CanonicalClip.frame_count is documented as "the
        # integer timing authority". The two derivations agree today only
        # because every producer pipes exact bytes -- a property of the current
        # producers, not a contract anything enforced.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        return {"out_path": path, "frame_count": n}

    def canonicalize(self, raw, request, profile):
        """Normalize a rendered clip into the ALWAYS-SILENT bt709 / yuv420p
        CanonicalClip contract (frame_count is the integer timing authority)."""
        return self._clip_from_raw(raw, request)

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of an audio_ref / init_image that may be a
        bare string OR a mapping carrying a ``path`` key (the schema AudioRef
        shape). Returns "" when nothing path-like is present."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _init_image_ref(self, request):
        """The portrait init image path from ``asset_refs{init_image}`` (or "")."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _aspect_plan(self, request):
        """The pad / crop / fit transform mapping the portrait init into the
        canvas with ONE uniform scale (never a stretch, pre-mortem N9). Returns
        ``None`` when the canvas or init dims are absent (the GPU smoke probes the
        real init dims), but still validates the policy token fail-closed."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        dst_w = int(c_get("w", 0) or 0)
        dst_h = int(c_get("h", 0) or 0)
        policy = (c_get("aspect_policy", _MC.DEFAULT_ASPECT_POLICY)
                  or _MC.DEFAULT_ASPECT_POLICY)
        src_w = int(get("init_w", 0) or 0)
        src_h = int(get("init_h", 0) or 0)
        if min(dst_w, dst_h, src_w, src_h) <= 0:
            _MC.assert_aspect_policy(policy)     # validate the token even unsized
            return None
        return _MC.resolve_aspect_transform(src_w, src_h, dst_w, dst_h, policy)

    def _build_render_request(self, request):
        """Pure: the normalized inference request the HuMo wrapper consumes, from
        a VideoRequest-shaped object OR a plain dict. Deterministic (seed + audio
        + init + aspect flow straight through) -- the render-twice determinism
        contract (V-7). The audio_ref is the FROZEN master that DRIVES the face;
        the output clip stays silent (V-1)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return {
            "init_image": self._init_image_ref(request),
            "audio_path": self._ref_path(get("audio_ref")),
            "text_prompt": get("text_prompt") or "",
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
            "aspect_plan": self._aspect_plan(request),
        }

    def _clip_from_raw(self, raw, request):
        """Pure: shape a worker / wrapper result into the silent CanonicalClip
        dict (bt709 / yuv420p; frame_count is the integer timing authority)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "humo_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }


#: 1.7B fallback-tier defaults (the lighter, no-LoRA talking face). Env-isolated
#: from the 14B knobs via the OTR_HUMO_17B_* namespace so a 14B OTR_HUMO_STEPS=6
#: / OTR_HUMO_CFG=1.0 never bleeds into this slower tier.
_HUMO_17B_UNET = "humo_1.7B_fp16.safetensors"


@register
class HuMo17BEngine(HuMoEngine):
    """The 1.7B HuMo downgrade tier. The OOM/VRAM hard auto-downgrade from the
    14B keystone lands HERE before the still floor, so a heavy episode that
    can't fit the 14B keeps a REAL audio-driven talking face. Config is isolated
    from the 14B env: its own UNET, NO LoRA (the lightx2v distill is 14B-shaped
    and shape-mismatches the 1.7B tier), and its own steps/cfg via OTR_HUMO_17B_*
    (defaults: 20 steps -- the distill shortcut is gone so it needs more steps --
    and cfg 1.0, dropped from 5.0 which produced a blue colour cast, fixed
    2026-06-17). Shares roles / required_inputs / requires_flag / the in-process
    graph topology with the 14B base; only the tier config differs. Degrades on
    to the zero-VRAM still floor (humo -> humo_1.7B -> still_motion)."""

    name = "humo_1.7B"
    engine_version = "1"
    fallback_engine = None               # NO FALLBACKS (2026-07-02): fail LOUD
    #: S1 per-model still plan (see ``_HUMO_STILL_PLAN`` -- shared HuMo shape,
    #: portrait REQUIRED for the audio-driven-face lane).
    still_plan = _HUMO_STILL_PLAN

    def _ckpt_path(self):
        env = os.environ.get("OTR_HUMO_17B_CKPT")
        if env:
            return env
        name = os.environ.get("OTR_HUMO_17B_UNET_NAME") or _HUMO_17B_UNET
        try:
            import folder_paths  # type: ignore
            p = folder_paths.get_full_path("diffusion_models", name)
            if p:
                return p
        except Exception:  # noqa: BLE001 - headless/CPU: fall back to a join
            pass
        return os.path.join(_COMFY_ROOT, "models", "diffusion_models", name)

    def _loader_names(self):
        names = super()._loader_names()
        names["unet"] = os.path.basename(self._ckpt_path())
        # The 14B-shaped lightx2v distill LoRA is incompatible with 1.7B: run
        # LoRA-free (more steps below make up for the lost distill shortcut).
        names["lora"] = os.environ.get("OTR_HUMO_17B_LORA_NAME", "none")
        return names

    def _steps(self):
        return int(os.environ.get("OTR_HUMO_17B_STEPS", "20"))

    def _cfg(self):
        # cfg 1.0 (was 5.0): the LoRA-free 1.7B tier at cfg 5.0 produced a strong
        # BLUE colour cast -- red crushed, blue pushed (B-R measured +44.6 vs the
        # source still's +25 on 2026-06-17) -- while cfg 1.0 is colour-balanced
        # (B-R +2.8, red recovered) on the identical beat. HuMo's talking-face
        # motion is driven by the AUDIO conditioning, not cfg, so the lip motion
        # survives the drop. Tune via OTR_HUMO_17B_CFG.
        return float(os.environ.get("OTR_HUMO_17B_CFG", "1.0"))


@register
class HuMo17BLandscapeEngine(HuMo17BEngine):
    """1.7B HuMo in 16:9 LANDSCAPE (832x480) -- the modern character-beat look,
    selectable in the role dropdowns ALONGSIDE the portrait humo_1.7B (the
    operator wants both: portrait for old-time radio, wide for the new standard).
    Same checkpoint / roles / required inputs / in-process graph as humo_1.7B;
    only the render aspect and the cfg default differ. The 16:9 cfg sweet spot
    measured 2026-06-16 (the cfg sweep) is 2.5, so this tier defaults there while
    portrait humo_1.7B now uses cfg 1.0 (de-blued 2026-06-17, was 5.0). A 16:9
    character still (832x480) feeds it --
    meta_brief mints the still to match the selected engine's aspect via the video
    policy, so ONE dropdown pick aligns dims, cfg, still and composite canvas.
    Not in the auto-fallback chain (it is a deliberate operator pick): a failure
    degrades like humo_1.7B (-> still floor), never a silent aspect swap."""

    name = "humo_1.7B_169"
    render_aspect = "wide"
    #: S1b per-model still plan (see ``_HUMO_169_STILL_PLAN`` -- the LANDSCAPE
    #: HuMo plan). ``render_aspect="wide"`` drives ``resolve_row_aspect`` to
    #: size portraits WIDE, and the plan's portrait row carries the WIDE
    #: layer-2 geometry to match. This sibling was ALREADY wide before the
    #: still-plans build; nothing about it flips.
    still_plan = _HUMO_169_STILL_PLAN

    def _cfg(self):
        # 16:9 sweet spot (cfg sweep 2026-06-16); override via OTR_HUMO_17B_169_CFG.
        return float(os.environ.get("OTR_HUMO_17B_169_CFG", "2.5"))


@register
class HuMo14BLandscapeEngine(HuMoEngine):
    """14B HuMo in 16:9 LANDSCAPE (832x480) -- the 2026-06-09 keystone quality
    (the fast 14B Kijai + lightx2v 6-step distill) in the modern WIDE character-beat
    aspect. The operator wants the 14B look in 16:9, NON-blue: the 1.7B downgrade
    tier carries a color cast (red-crushed / blue-pushed, measured 2026-06-17),
    but the 14B shares the same wan_2.1_vae and its latents MATCH it, so the 14B
    output is colour-correct. Same 14B checkpoint / LoRA / steps / cfg / roles /
    in-process graph as the humo base; ONLY the render aspect differs
    (portrait 480x832 -> wide 832x480). Selectable in all three role dropdowns
    alongside the portrait 14B and the 1.7B tiers. A 16:9 character still (832x480)
    feeds it -- meta_brief mints the still to the selected engine's aspect via the
    video policy, so one dropdown pick aligns dims, still and composite canvas.
    Degrades like the 14B base (-> humo_1.7B -> still); a deliberate
    operator pick, never a silent aspect swap."""

    name = "humo_14B_169"
    render_aspect = "wide"
    #: S1b per-model still plan (see ``_HUMO_169_STILL_PLAN`` -- the LANDSCAPE
    #: HuMo plan). ``render_aspect="wide"`` drives portraits WIDE via
    #: ``resolve_row_aspect``, and the plan's portrait row carries the WIDE
    #: layer-2 geometry to match. Already wide before this build.
    still_plan = _HUMO_169_STILL_PLAN
    #: Route-A VRAM frame cap (only this 14B tier; base humo / humo_1.7B* stay
    #: uncapped). render_clip renders at the cap then exact-fits to the audio
    #: target. env override: OTR_HUMO_14B_SAFE_FRAMES (after a higher-frame probe).
    safe_render_frames = _HUMO_14B_SAFE_RENDER_FRAMES

    #: ITS OWN LADDER, because its ceiling is its own (chunk 7a QA panel,
    #: 2026-07-26). This tier inherited HuMoEngine's ``max_frames=177`` for
    #: exactly as long as it took a panel to read ``render_clip``, which does
    #: ``render_max = cap if cap is not None else _HUMO_MAX_FRAMES`` -- and
    #: ``cap`` is ``safe_render_frames`` = 49 HERE and None on all three
    #: siblings. So the inherited contract advertised 3.6x the frames this
    #: engine will actually render, and any beat over 49 frames raises
    #: ``MirrorExtensionForbidden`` today (mirroring an audio-synced lane would
    #: play the mouth backwards, which is correctly forbidden rather than
    #: worked around).
    #:
    #: This is the inheritance trap the roster audit was written to look for and
    #: did not catch: the audit checks each contract against ITSELF, never
    #: against the subclass's runtime behaviour. A subclass that changes what it
    #: renders must change what it declares, and 49 is the number.
    #:
    #: 49 = 4*12+1, so it is on the shared 4n+1 ladder and the quantum is
    #: unchanged. min stays 33: below that has hung this hardware.
    frame_contract = FrameContract(
        min_frames=_HUMO_MIN_FRAMES,
        max_frames=_HUMO_14B_SAFE_RENDER_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )

    def _cfg(self):
        # Inherit the 14B distill cfg (1.0); a 16:9 tuning hook is available via
        # OTR_HUMO_14B_169_CFG without disturbing the portrait 14B's OTR_HUMO_CFG.
        return float(os.environ.get("OTR_HUMO_14B_169_CFG", str(super()._cfg())))


__all__ = [
    "HuMoEngine", "HuMo14BLandscapeEngine",
    "HuMo17BEngine", "HuMo17BLandscapeEngine",
]
