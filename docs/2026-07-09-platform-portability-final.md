# OTR Platform Portability -- Converged Plan (FINAL, 2026-07-09)

Campaign: analysis-only review window. Anchor + 5 file-grounded audits + R1 cloud
roundtable (GPT-5.6 / Gemini 3.1 Pro / DeepSeek v4 Pro / Tencent Hy3, ~$0.15) + R2-R4
local kibitz (Codex + Antigravity, $0). R4 CONVERGED: no new must-fix class. Every claim
below was verified against the real files; round history + judgments live in
docs/2026-07-09-platform-portability/ (gitignored working folder) and
kibitz-runs/2026-07-09-platform-portability/. Grounded at 3060fd3a; loader/profile/
registry/canonical files re-verified untouched by the coder window's parallel commits.

## PAGE ONE -- GO / NO-GO SUMMARY (operator)

**GO -- the architecture is buildable and half of it already exists.** The 2026-06-10
GATE B capability-profile stack (profiles + widget_mapping + apply_profile + per-engine
CAPABILITIES) is the right foundation; this campaign extends it to a platform matrix and
builds the never-shipped S3 snapshot generator. ONE canonical JSON stays the sole graph;
a variant = a stamped, regenerated copy + a generated launch recipe. No auto-detection
anywhere: explicit widgets + profiles, fail-loud validation at profile-load, emit,
queue, and model-load time.

Per-tier verdicts:
- **NVIDIA 16 GB (nv50)**: GO -- identity variant; only tier with soak history.
- **NVIDIA 12 / 8 GB**: GO, draft until smoke -- same stack, lower ceilings; wan_ti2v +
  GGUF quant ladder are the proven levers. 14 GB-class lanes (ltx_av/ltx_video/
  wan_i2v/humo-14B) are OUT on these tiers.
- **AMD ROCm Linux 16 / 8 GB**: CONDITIONAL GO, draft -- no code-level blocker on the
  selected lane set (torch-ROCm presents as "cuda"), but bnb is Preview (LLM quant lane
  off), fp8/fp4 lanes off, sage off, ltx_av excluded by its NVML gate, dia excluded by
  its cu128-only pin. UNVERIFIED on real hardware -- ships draft behind acceptance gates.
- **Apple Silicon MPS**: CONDITIONAL GO, draft -- kokoro TTS, stable_audio_3 music,
  GGUF-Metal writer, viz/stills video; every fp8/fp4 lane excluded (Float8_e4m3fn
  unsupported); wan_ti2v is the only local-motion candidate and stays OFF until a
  real-Mac smoke. MPS is currently invisible to the profile enum + host detection --
  both get extended.
- **CPU-only**: GO, cheapest honest tier -- GGUF-CPU or remote story, kokoro voices,
  musicgen music, viz/still video, cloud image lane, full ffmpeg composite. NO diffusion
  video, NO stable_audio_3 (impractical), NO indextts2/dia/bark. First tier to smoke
  (this box, --cpu).
- **Comfy Cloud hosting: NO-GO** (researched, cited in audit): allowlist-only custom
  nodes, no env vars, 30-60 min execution cap. The cloud tier is therefore
  **otr_cloud_lanes** -- a CPU-baseline host orchestrating the EXISTING cloud lanes
  (OpenRouter/Comfy-Credits/Google LLM slots, elevenlabs/sonilo/google audio,
  google/partner video+image). [OPERATOR: ratify the name.]
- **1080p ceiling holds everywhere** (composite chain is already 1920x1080; render
  canvas 832x480). **upscale_stage**: NAME-ONLY reservation, default off, nothing built
  (panel wanted it cut three times; kept because the brief explicitly reserves it).

**Committed-state defects the plan fixes first (S0)** -- these currently break non-CUDA
tiers outright: cpu_floor.json selects bark whose generation path hardcodes CUDA (+
monkeypatched tensor factories) contradicting its registry row; the transformers loader
crashes on any non-CUDA host (unguarded get_device_capability, _otr_model_loader.py:257);
a registered VRAM-test node is unguarded-CUDA; host detection is MPS-blind; master_hash
is stamped but never verified; the HuMo download script fetches a different filename
than the engine default expects (FRESH-INSTALL breaker); the image dispatcher silently
skips generation on CPU; the default char-voice engine's install script is referenced
but does not exist in the repo.

**Operator decisions required before variant emission** (generator REFUSES to emit
variants whose profile carries an unratified field): (1) nv50 baseline = current
canonical values, with 16gb_full.json regenerated from canonical (committed profile is
stale: humo_14B_169/flux_gen1 vs canonical viz/z_image); (2) cloud tier naming
(otr_cloud_lanes) + OpenRouter model pins; (3) mac_mps VRAM ceiling per target machine;
(4) AMD/Mac verification strategy -- rent/borrow hardware for a one-day smoke vs ship
draft-marked UNVERIFIED vs defer those variants.

Total external campaign spend: ~$0.15.

## 1. Architecture

A platform variant = TWO artifacts generated together by `scripts/build_variants.py`
(offline stdlib-only CLI -- the never-built GATE B "S3 emit_snapshot"; never a node):

1. **Stamped variant JSON** `workflows/variants/otr_<tier>.json` =
   apply_profile(canonical, profile) + stamps written into OTR_WorkflowValidator's
   EXISTING widgets (profile_id, master_hash, generated_by). Self-check: apply -> stamp
   -> compare profile-managed widgets (stamps excluded) -> refuse on mismatch.
   `--check` regenerates committed variants AND recipes, diffs both, loads each variant
   and confirms profile_id/master_hash/recipe agreement, asserts the stale-variant
   soft-skip path is dead, exits nonzero on any drift (CI rule: generated, never
   hand-edited). Refuses emission on ratify_before_emit fields.
2. **Launch recipe** `otr_<tier>.launch.md` from the SAME validated profile object:
   comfy args, sage flag, env (PYTHONUTF8=1 on Windows; models_root overrides -- the
   C:\ComfyUI-Models defaults in _otr_hf_env.py:45 / _otr_gguf_backend.py are
   Windows-only conveniences), key NAMES from preflight.required_keys (values never
   stored; Google key preferred name OTR_GOOGLE_API_KEY with GEMINI_API_KEY /
   GOOGLE_API_KEY documented as accepted aliases per _otr_google_api/client.py:43-54),
   install pointers (torch build, llama-cpp-python wheel flavor CUDA/HIP/Metal/CPU,
   ffmpeg + Mac libx264/aac codec note, libcairo for viz_mandala on minimal Linux).

Switch layers: (a) existing canonical widgets carry most per-variant VALUES; (b) profile
JSONs (schema v2) are the value sets + launch/validation policy; (c) NEW append-only
widgets only for runtime values no widget carries (nodes never read profiles at
runtime). Applier facts: the ONE applier is nodes/_otr_workflow_apply.py; scripts/
otr_api.py IMPORTS apply_profile (:803-804) and keeps a parity-pinned
patch_widget_by_name copy; admit-path logic exists in both copies (:208-261 applier
side). Mapping-schema changes touch applier + otr_api copy + parity tests atomically.

**Drift tripwire:** semantic master_hash = sha256 over node types + links (sorted) +
profile-MANAGED widget set only; excludes pos/size/UI keys, the stamps, and all creative
widgets (title/premise/seeds free to edit). Normalizer in _otr_workflow_apply, shared by
generator + validator. _assert_stamp gains the assert (today the validator only LOGS
master_hash and byte-hashes separately, _otr_workflow_validator.py:271-325) and takes
the parsed dict. Host-reality check: _detect_host + torch.backends.mps.is_available()
+ vendor via torch.version.hip / device name; ANY mismatch raises; NEVER reconfigures.

**Validation order fix:** OTR_WorkflowValidator currently gates only OTR_VideoDirector
(link 269) -- an invalid variant burns LLM/API work before failing. Fix: `gate_in`
forceInput optional STRING on OTR_LedgerScriptWriter (full contract section 4) wired
from node 63 in the SAME commit. Headless lane: the EXISTING otr_api.py preflight
(widget validation / profile apply / stamp normalization before POST /prompt)
additionally asserts stamps + master_hash; a variant whose widget vector mismatches
live INPUT_TYPES is a HARD FAIL ("regenerate variants"), replacing today's soft skip.
Queue-level ComfyUI hook: rejected (no stable API; gate_in suffices).

## 2. Profile schema v2 + registry v2

capability_profiles.py: _TOP_LEVEL_KEYS += gpu_vendor, llm, video, image, audio,
render, preflight (nested sub-validators each); _PLATFORMS += "linux"; _DEVICE_BACKENDS
+= "mps"; launch keeps sage_attention + extra_args, gains env. Fail-closed stays. All
EIGHT committed profiles migrate as fixtures in the SAME commit;
tests/test_capability_profiles.py:114-123 re-parametrized (it currently asserts
mps/linux REJECTED). _DECL_KEYS does NOT move here -- it moves in the S3 registry
commit (staggering them opens a crash window).

Fields (class W = widget-mapped, E = emit/validation-time, L = launch-only; one class
per field): platform E; device_backend E (cuda|cpu|mps; ROCm = "cuda"); gpu_vendor E
(nvidia|amd|apple|none); llm.device W; llm.attn_impl W (sdpa|flash_attention_2|eager);
llm.quant_policy W (bnb_nf4|bnb_8bit|none); llm.vram_ceiling_gb W (0 = disabled, cpu
tier only); llm.gguf_n_ctx W; llm.gguf_quant W (Q8_0|Q6_K|Q4_K_M); llm.creative_model /
llm.technical_model / 6 slot keys W (admit-path-gated); llm.lane_allowlist E (checked
in _otr_workflow_validator, which may import the catalog -- capability_profiles is
stdlib-only by contract -- plus a runtime backstop in request_slot; remote backends
ASSERT the allowlist admits them and IGNORE hardware fields, documented per backend);
video.device_policy W; video.dtype_policy W (fp8_ok|no_fp8|no_fp8_no_fp4);
image.dtype_policy W; audio.voice_device W; render.fps/canvas_w/canvas_h/composite_res/
composite_w/composite_h/frame_budget/beats W; launch.* L; preflight.required_models /
required_keys E (checked fail-loud at boot; aliases allowed for the Google key).

Registry CAPABILITIES v2 -- ONE atomic commit across audio + video + IMAGE registries
(_otr_image_engines/registry.py has rows too) + _DECL_KEYS
(capability_profiles.py:236,255) + cross-validation tests: device_backends list
(supersedes bare cpu_ok), requires_vendor ("nvidia" makes the ltx_av NVML gate
table-visible), needs_fp8_te / needs_fp4_te, practical_without_gpu,
sidecar_conditional. Ruling (R4): bark = device_backends ["cuda"],
practical_without_gpu false -- registry exclusion, NO bark code surgery this campaign.
Tests must prove the old false bark row cannot survive v2 semantics.

## 3. Policy propagation contracts

- **LLM:** frozen LLMRuntimePolicy (device, attn_impl, quant_policy, vram_ceiling_gb,
  gguf_n_ctx, gguf_quant, lane_allowlist) built from the writer's widgets in
  _resolve_inputs (OTR_LedgerScriptWriter.py:1145) -> _SlotScheduler (:471) ->
  request_slot(slot, id, policy=None) -> every backend .load(..., policy). NO
  legacy/auto sentinel: the widget defaults EQUAL today's resolved baseline values, so
  the explicit policy reproduces 16 GB behavior while deleting the FA2 auto-probe and
  tag-based auto-quant; policy=None persists only for unmigrated callers DURING S1,
  which ends with zero. GGUF backend: quant -> filename + expected-size artifact table
  (fail-loud for known quants, absence-only for unknown; GEMMA4_12B_GGUF_PATH env stays
  as escape hatch); the silent n_ctx 4096->2048 downgrade (:290-297) and the "preflight
  failed (proceeding anyway)" tolerance (:307) both become raises. check_vram_fit
  ceiling from policy (pre-download fail-loud validation -- deliberately kept post-rip;
  it never adapts).
- **Video/image:** directors emit policy_version 2 (+device_policy, dtype_policy);
  OTR_ShotLock stamps policy v2 into patched_ledger_json; render_driver fills REAL
  host_caps (cuda_available, mps_available, vendor, total_vram_gb) + profile at what is
  today `assert_usable(host_caps={}, profile={})` / `prepare({},{},{})`
  (render_driver.py:2198-2201, verified) -- the single most important wiring fix: the
  adapter-level enforcement protocol EXISTS (registry.py:93) but currently receives
  empty dicts. Every consumer asserts policy_version==2 (the image dispatcher parses
  with no version gate today, :911-914) and dispatchers call ADAPTER-level assert_usable
  (the image path calls only the registry-level name/role check, :635). gen_fn contract:
  production gen_fn=None for a required target raises ImageRenderError; injected test
  callables (tests/test_image_platform_c1.py:396) remain valid -- kills the silent
  "skipped on CPU" path (:647-649).
- **Audio:** CastLock.voice_device stamped into the ledger like the engine stamps
  (cast_lock.py:625-657 pattern); _otr_voice_node_common.generate threads explicit
  device into every adapter (:230-260, :313-326); theme music reads the SAME field
  (StableAudioTheme consumes the ledger, link 237); comfy-native stable_audio_3
  documents that Comfy core owns its device. Waterfalls removed: kokoro cuda->mps->cpu,
  musicgen + chatterbox cuda->cpu. motion_common.compute_real_frame_budget: static
  widget value only; runtime may raise, never resize.

## 4. Exact widget wiring (verified against the audited canonical baseline)

| Node (id) | Count | New widget -> append index | New count |
|---|---|---|---|
| OTR_LedgerScriptWriter (1) | 28 | llm_device->28, llm_attn_impl->29, llm_quant_policy->30, llm_vram_ceiling_gb->31, gguf_n_ctx->32, gguf_quant->33 | 34 |
| OTR_VideoDirector (87) | 12 | device_policy->12, dtype_policy->13 | 14 |
| OTR_ImageDirector (88) | 7 | dtype_policy->7 | 8 |
| OTR_CastLock (80) | 5 | voice_device->5 | 6 |

All optional; defaults (cuda, sdpa, bnb_nf4, 14.5, 4096, Q8_0, cuda, fp8_ok, fp8_ok,
cuda) preserve current behavior -- the identity variant equals canonical modulo stamps.
No other node needs a device/dtype widget (composite chain is ffmpeg; VideoRenderBatch
+ voice nodes consume ledger policy).

gate_in (ONE commit): forceInput optional STRING on the writer -- consumes NO
widgets_values slot (count stays 34); run() (:2795) + _resolve_inputs (:1145) accept
it; new link [279, 63, 0, 1, <slot>, "STRING"]; node 63 outputs[0].links [269] ->
[269, 279]; last_link_id 278 -> 279; writer added to otr_api parity fixtures.

oom_index (node 92 index 2): POSITIONALLY INERT -- it precedes frame_count (index 3;
otr_video_render_batch.py:153-158); removal would shift positions (BUG-LOCAL-097
class) and is OUT of this campaign.

widget_mapping.json v2: writer leaves exempt_node_types; NEW exempt_widget_names
{"OTR_LedgerScriptWriter": [episode_title, target_words, num_characters,
custom_premise, include_act_breaks, act_count, creativity, perfect_run_spacesaver,
min_p, repetition_penalty, max_new_tokens_cap, lemmy_cameo, use_exchange,
enable_production_stage3_validators, news_briefs_required, refine_target_grade,
story_scaffold, source_bank, visual_style, source_ref]}; never_patch unchanged.
Managed keys: the section-2 W keys 1:1 to new widgets; llm model/slot keys -> the
writer's 8 existing model widgets; render.fps -> [(12,fps),(84,fps),(85,fps),(86,fps),
(87,fps)]; render.canvas_w/h -> node 87; render.composite_res -> [(12,resolution)];
render.composite_w -> [(84,canvas_w),(94,out_w)]; render.composite_h ->
[(84,canvas_h),(94,out_h)]; render.frame_budget -> [(92,frame_count)]; render.beats ->
[(92,beats)]; audio.voice_device -> [(80,voice_device)].

## 5. Sprint breakdown (coder window; regression suite + Bug Bible after every chunk)

S0 defects (land first, no design deps): loader :257 guard + per-backend max_memory
keying; vram_context_test guards (+torch.mps equivalents where available); HuMo
artifact alignment (download script fetches humo_17B_fp8_e4m3fn.safetensors :34-36,
engine defaults to the 14B_KJ filename, eng_humo.py:79 -- fresh-install breaker; align
script + default + registry label); image dispatcher gen_fn contract (section 3);
_detect_host mps; bark registry ruling (section 2). Sidecar installer gap: the
indextts2 install script named in its own error message does not exist -- write it or
re-derive the pin before any non-Windows recipe for the default voice.
S1 LLM policy threading (ends: zero policy=None callers; GGUF artifact table;
downgrade/proceed-anyway -> raises).
S2 profile schema v2 + 8 fixture migrations + _flatten_profile_values
(_otr_workflow_apply.py:429 currently flattens ONLY role/slot/features + 2 seed keys)
+ mapping v2 (exempt_widget_names; applier + otr_api parity copy + tests atomic) +
enum-test flips. NO _DECL_KEYS here.
S3 registry v2 ATOMIC (three registries + _DECL_KEYS + tests).
S4 policy consumers (policy_version 2; ShotLock ledger stamp; render_driver
host_caps/profile fill; adapter-level assert_usable calls; audio device threading;
waterfall + frame-budget removals).
S5 generator + validator (build_variants.py incl. --check + ratify_before_emit;
semantic hash; _assert_stamp; gate_in commit; per-node-group atomic widget commits:
writer / video+image / castlock -- each = INPUT_TYPES + canonical append + validator
fixture + mapping targets + tests).
S6 emit variants + recipes; smoke gates: cpu tier FIRST on this box (--cpu), then nv50
identity re-soak; AMD/MPS stay draft pending hardware.

## 6. Variant table (single values; ratify_before_emit fields marked)

Profile migration: 16gb_full -> otr_nv50_16gb (REGENERATED from canonical
[ratify_before_emit]); 8gb_lite -> otr_nv30_8gb; cpu_floor -> otr_cpu_only (bark ->
kokoro AND voice_bank bark_legacy -> default -- committed profile is broken today);
cloud_all -> otr_cloud_lanes [ratify_before_emit: name + pins]; google_* stay lane
presets; NEW otr_nv40_12gb, otr_amd16_rocm, otr_amd8_rocm, otr_mac_mps
[ratify_before_emit: ceiling]. Emission fails on missing profile ids.

| Key | nv50_16gb | nv40_12gb | nv30_8gb | amd16_rocm | amd8_rocm | mac_mps | cpu_only | cloud_lanes |
|---|---|---|---|---|---|---|---|---|
| status | shipping | draft | draft | draft | draft | draft | draft | draft |
| platform/backend/vendor | any/cuda/nvidia | any/cuda/nvidia | any/cuda/nvidia | linux/cuda/amd | linux/cuda/amd | mac/mps/apple | any/cpu/none | any/cpu/none |
| llm.device / quant_policy | cuda / bnb_nf4 | cuda / bnb_nf4 | cuda / none | cuda / none | cuda / none | mps / none | cpu / none | cpu / none |
| llm.attn_impl | sdpa | sdpa | sdpa | sdpa | sdpa | sdpa | sdpa | sdpa |
| llm.vram_ceiling_gb * | 14.5 | 10.5 | 6.8 | 14.5 | 6.8 | 10.0 (ratify) | 0 | 0 |
| llm.gguf_n_ctx * / gguf_quant | 4096 / Q8_0 | 4096 / Q6_K | 2048 / Q4_K_M | 4096 / Q8_0 | 2048 / Q4_K_M | 4096 / Q6_K | 2048 / Q4_K_M | 2048 / Q4_K_M |
| llm.creative / technical | gemma4-GGUF / Mistral-Nemo(bnb) | gemma4-GGUF / Mistral-Nemo(bnb) | gemma4-GGUF / gemma4-GGUF | gemma4-GGUF / gemma4-GGUF | gemma4-GGUF / gemma4-GGUF | gemma4-GGUF / gemma4-GGUF | gemma4-GGUF / gemma4-GGUF | openrouter:slot-a / slot-b (pins ratify) |
| llm.lane_allowlist | all | all | all-minus-bnb | all-minus-bnb | all-minus-bnb | gguf+remote | gguf+remote | remote-only |
| video.device / dtype | cuda / fp8_ok | cuda / fp8_ok | cuda / fp8_ok | cuda / no_fp8 | cuda / no_fp8 | mps / no_fp8_no_fp4 | cpu / no_fp8_no_fp4 | cpu / no_fp8_no_fp4 |
| image.dtype | fp8_ok | fp8_ok | fp8_ok | no_fp8 | no_fp8 | no_fp8_no_fp4 | no_fp8_no_fp4 | no_fp8_no_fp4 |
| audio.voice_device | cuda | cuda | cuda | cuda | cuda | mps | cpu | cpu |
| char video / render engine | viz_camera | wan_ti2v | wan_ti2v | wan_ti2v | wan_ti2v | still_motion | still_motion | google_veo_video |
| announcer+music video | viz_mxc_cpu + viz_mxc_mandala | same | same | same | same | same | same | same |
| images x3 | z_image_turbo | z_image_turbo | z_image_turbo | z_image_turbo | z_image_turbo | google lane | google lane | google lane |
| char / announcer voice | indextts2 / kokoro | indextts2 / kokoro | kokoro / kokoro | kokoro / kokoro | kokoro / kokoro | kokoro / kokoro | kokoro / kokoro | elevenlabs / google_tts |
| voice_bank / cast_policy | default / auto_registry | same | same | same | same | same | same (NOT bark_legacy) | same |
| music_engine | stable_audio_3 | stable_audio_3 | stable_audio_3 | stable_audio_3 | musicgen | stable_audio_3 | musicgen | sonilo |
| render fps/composite | 25 / 1920x1080 | same | same | same | same | same | same | same |
| canvas / frame_budget * / beats | 832x480 / 25 / 40 | 832x480 / 25 / 40 | 832x480 / 21 / 40 | 832x480 / 25 / 40 | 832x480 / 17 / 40 | 832x480 / 17 / 40 | 832x480 / 25 / 40 | 832x480 / 25 / 40 |
| launch sage / args | true / [] | true / [] | true / [] | false / [] | false / [] | false / [] | false / [--cpu] | false / [--cpu] |
| preflight.required_keys | [] | [] | [] | [] | [] | [OTR_GOOGLE_API_KEY] | [OTR_GOOGLE_API_KEY] | [OPENROUTER_API_KEY, OTR_GOOGLE_API_KEY, OTR_COMFY_API_KEY per lanes] |

\* Values marked with the asterisk on draft tiers are INITIAL SMOKE VALUES (policy
choices), not measured capacities -- each tier's first acceptance run calibrates them.
cloud_lanes = cpu_only host baseline + cloud values; no failover -- a lane outage is a
loud failure. wan_ti2v on mps stays unselected until a real-Mac smoke passes.

Honest tier exclusions: nv40/nv30 -- no 14GB-class lanes; amd -- no ltx_av (NVML gate),
no dia (cu128 pin), no sage, no fp8 defaults, no nvvfx, .ps1-only sidecar installers
need ports; mps -- additionally no bark, no bnb, no indextts2 (no install path); cpu --
no diffusion video, no practical stable_audio_3, no local image gen (cloud lane or
pre-staged stills); cloud_lanes -- local compute is orchestration + ffmpeg only, real
per-episode provider cost.

## 7. Acceptance gates + verify-at-build register

Every non-nv50 variant ships status=draft until, on real target hardware: cold launch,
one full episode end-to-end, per-lane smokes, peak-memory observation, and fail-loud
NEGATIVE tests (device=cuda on the wrong host must raise; banned dtype must raise) all
pass. CPU tier first (this box). Determinism statement: per-variant reproducibility
(same box + backend + seeds); cross-platform bit-identity is impossible and explicitly
disclaimed.

Verify-at-build register: stable_audio_3 on ROCm/MPS (rides Comfy core device layer);
z_image_turbo on ROCm/MPS/8GB (thinnest audit coverage); comfy_credits lane
off-Windows; llama-cpp HIP/Metal wheel install; ROCm RDNA stability over a full
episode; torch.mps.empty_cache availability; gen_fn-None production behavior post-S0;
SA3 device-ownership doc; Mac ffmpeg codec availability; OpenRouter concrete slugs
before cloud_lanes emission; host_caps minimum shape (cuda_available, mps_available,
vendor, total_vram_gb) suffices for every adapter; ShotLock policy stamp vs
freeze-cascade revisions.

## 8. Campaign record

R1 cloud roundtable ~$0.1541 (gpt-5.6-sol, gemini-3.1-pro-preview, deepseek-v4-pro,
hy3:free): 22 accepted / 4 rejected -- headline catches: validator gating order, nv50
identity contradiction, semantic hash, offline generator, per-widget exemption.
R2 kibitz $0: 17 accepted / 2 rejected -- LLM policy threading path, registry
atomicity incl. image registry, oom_index positional guard, HuMo fresh-install
severity, image-dispatcher CPU skip.
R3 kibitz $0: 15 accepted / 2 rejected -- render_driver empty-dict adapter boundary
(the campaign's most consequential wiring catch), gate_in full contract, schema
completion, sprint resequencing, enum-test flips.
R4 kibitz $0 (codex + anchor; antigravity quota-failed twice): 7 accepted / 1 rejected
-- CONVERGED, no new must-fix class. Standing panel disagreement recorded: codex would
cut the upscale_stage name reservation (rejected 3x on operator-brief grounds).
Notable discarded misreads (grounded with citations): hy3 conflating wan_ti2v/wan_i2v
CLIP lanes; antigravity's "google_veo_video duplicate key" (a comment line); GPT's
"widget contract must precede R3" (the campaign's R3 delivered it).

Working artifacts (gitignored by design): docs/2026-07-09-platform-portability/
(anchor, 5 audits, 4 draft deliverables, roundtable pass00-pass04 plans + judgments)
and kibitz-runs/2026-07-09-platform-portability/ (r2-r4 agent reviews + logs).
