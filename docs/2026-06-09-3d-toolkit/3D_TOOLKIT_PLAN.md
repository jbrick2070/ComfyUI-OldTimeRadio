# OTR 3D Toolkit Plan v2 -- the `character_3d` family (NEXT PHASE, canonical)

> **STATUS: v2, ROUNDTABLE-HARDENED 2026-06-09 (three passes; campaign closed at the pass
> cap -- architecture stable since pass 1, pass-3 findings were wiring-contract level only).** 3D stays **PARKED --
> no implementation code, no spikes run, no asset downloads** until the operator green-lights.
> Hardening: 2 passes x 4-model panel (GPT-5.5, Gemini 3.1 Pro, Grok 4.3, DeepSeek V4-Pro) +
> Claude as 5th panelist AND final judge, every critique grounded against the live repo
> (`schemas.py`, `eng_character_3d.py`, `render_driver.py`, `resolver.py`, `_b_harness.py`,
> `otr_image_director.py`). Artifacts:
> `otr-video-roundtable\roundtable_campaign\2026-06-09-3dplan\h_pass01..02\` (+ judgments).

## What changed v1 -> v2 (and why)

- **Reconciled plan vs the SHIPPED dark scaffold** (all panelists): the repo already registers
  `hunyuan3d_talk`/`trellis_talk` and hard-codes them in `SYNTH_FALLBACKS`/`ENGINE_FAMILY`/
  `OOM_ENGINES`/`EXPECTED_OOM_TRAIL` + 3 test files. v2 adds the explicit **W7-pre migration
  slice** (section 7.0) instead of a vague "names follow the probe".
- **Fixed the `assert_usable` deadlock** (Gemini): the live TripoSG adapter must NOT gate on
  `OTR_B_MESH_DIR` (meshes are *generated* at `prepare()`); the mesh-dir check moves to probe_c
  only. New runtime gate order in section 7.1.
- **Pinned the alpha handoff** to ONE contract: PNG/EXR frame **directory** (`CanonicalClip.type
  ="directory"`, `alpha="straight"`) -- the schema already supports it; the vp9/prores mux
  branches are cut from v1 (GPT).
- **Corrected the fallback trail** to the shipped chain (`humo -> humo_1.7B -> latentsync ->
  still_kenburns` after the 3D hop) -- v1's `triposg_talk->humo->still_kenburns` contradicted
  `EXPECTED_OOM_TRAIL` (GPT).
- **Added the v1 LOOK contract** (texture/material QC) -- v1 as written would ship a gray clay
  head (Claude-panelist + GPT + DeepSeek). Section 7.4.
- **Hardened cache keys** (curves bind to master CONTENT hash + line timing; `_slice_master_audio`
  keys by path only today), ledger boundaries (slice paths recorded in `ledger['video']`, never
  `ledger['audio']`), execution-group wiring (`depends_on` takes GROUP IDs, not "flux_portrait"),
  request validation (`build_request()` emits non-`VideoRequest`-valid dicts today -- the 3D
  adapter schema-validates at its boundary), and S-3D-0 now mandates the shipped
  `_b_harness.build_sidecar_env()` sanitizer (GPT).
- **Honest Plan-B for S-3D-0**: NO-GO is an OPERATOR DECISION (HuMo-2D-only v1 vs approve the
  toolkit install), not an automatic toolkit fallback; added the cheap pre-step (CPU
  marching-cubes variants) before declaring the lane dead (GPT + DeepSeek + Claude).
- **Split the definition of done**: "v1-usable" (one engine, one real episode) vs "B-parity
  ship" (second engine). The >=2-engine invariant binds at SHIP, not at first light (GPT).
- **Pass-2 additions** (all code-verified): the slicer is 44.1 kHz today -- the driver gets its
  own 16 kHz extraction without touching HuMo's; slice-vs-curve cache keys SPLIT (don't over-key
  the cheap WAV); `requires_mesh_portrait` needs a real schema field (`AdapterDescriptor`/
  `VideoProfileRow` are `extra="forbid"`); `video_policy_json` keeps `forceInput:True` when
  moved to required (else ComfyUI spawns a widget = V-11 violation); `OTR_SilentComposite` does
  NOT read frame directories yet -- named W7 build task; request builders emit non-schema-valid
  dicts (extras `init_w/init_h`, `dur_s`, `char_id`) -- builder migration specified; portrait
  UV-project bake bounded to a frontal camera cone with auto-stylized fallback (smear risk);
  character_3d rejects the 480x832 builder default canvas; Blender determinism pins; Windows
  atomic-publish caveat; on-disk ledger transaction; soak decision-count/trail-hop consistency
  check at the W7-pre rename.
- **Pass-3 additions** (wiring contracts, all code-verified): fallback chain must REBUILD the
  request per candidate (a character_3d request lacks latentsync's `base_clip_ref`);
  directory-clip semantics ripple into `_clip_summary`/manifests/soak (getsize on a dir);
  `ShotRow` is `extra="forbid"` -- artifact paths via `cache_keys`/qc, not new top-level keys;
  resolver-prune is shipped but UNWIRED in `render_shot` fallback -- wire it into the restamp
  transaction; `assert_soak_ok` carryover string `input_oom_engine`; slicer signature gains the
  master hash; the 16 kHz Rhubarb input is a downsample of the provided 44.1 kHz slice (the
  adapter never sees the master); pose-envelope/UV metrics actually emitted into qc; one
  binding spike sequence stated everywhere.
- **Story-assets addendum (operator-directed, post-campaign):** new section 2.5 binds the 3D
  path to the UPSTREAM story assets -- script text feeds Rhubarb (`--dialogFile`), ledger
  beats + M4 mood drive a deterministic shot grammar, the shipped per-line delivery vectors
  modulate the idle layer, C's scene stills are the plates, bust framing hides the decapitation
  stump, and the only candidate NEW input asset (a "bust card" body underlay) is explicitly
  deferred to v1.1 via the existing C engine. T4 + the DoD now FAIL a clip that ignores these.
- Citations: [v1.4-III], [SUB-B], [DEP], [PS-3D], [3DC], [GOFWD], [ROUTE-01], [B-SHIP],
  [PASS-3D], [FORK-RT], [H-RT] = this hardening campaign (passes 1-3).

---

## 1. Goal + scope

The 3D character path delivers the **8th engine family, `character_3d`** (already present in
`schemas.FAMILIES` with `FAMILY_REQUIRED_INPUTS = ("audio_ref", "init_image")`): a CONSUMER
adapter that turns a ledger portrait + a frozen audio line into a talking 3D character rendered
as an **ALPHA clip**, composited by `OTR_SilentComposite`, master audio muxed LAST and
byte-identical [v1.4-III]. Sidecar-isolated (V-12), **14.0 GB sub-ceiling**
(`_VRAM_CEILING_MB_3D = 14_000`, already in the scaffold) [SUB-B]. Strategic frame: the 3D
playground thesis -- battle-test any model inside a real narrative pipeline [PS-3D].

**Milestone: ONE mesh-gen model rendering a real character into a real episode ("v1-usable");
the second engine is the separate "B-parity ship" milestone** [GOFWD, H-RT]. OUT of scope for
v1: Airlock, `otr new-adapter`, gauntlet, new narrative roles, 3D-as-control-for-2D, bodies,
crowds -- deferred to the playground phase [PS-3D]. A2F-3D details live in the appendix
(section 10); v1 text stays Rhubarb-only [H-RT].

## 2. The pipeline (end to end)

```
flux_portrait (ledger, decoded-pixel hash)            frozen master audio (read-only)
   |                                                     |
   v                                                     v
[Slot 1  MESH-GEN]  portrait -> head mesh (GLB)      [Slot 2a PERF]  audio slice -> 52 ARKit
   |  (triposg_talk first; engine-agnostic)            weight curves (Rhubarb default; CPU;
   v                                                   cached by MASTER CONTENT HASH +
[Slot 1.5 PREP]  decapitation slice -> sanitize        line_id + start_s/dur_s + driver ver
   (orientation/scale normalize, remove loose          + mapping hash + fps + onset policy;
   parts, recalc normals, hole-fill INCLUDING          runs IN PARALLEL with Slot 1 -- needs
   the neck stump) -> optional deterministic           NO mesh) [FORK-RT, H-RT]
   remesh; post-slice quality check fail-closed                 |
   |                                                            |
   v                                                            |
[WRAP -- KEYSTONE]  template-first shrinkwrap onto the          |
   canonical ARKit-rigged template (barycentric/ICP,            |
   bake offsets, copy 52 deltas) -> GLB with EXACTLY            |
   52 ARKit shape keys [PS-3D]                                  |
   |                                                            |
   +----------------------------+-------------------------------+
                                v
[Slot 2b DEFORM/RENDER]  Blender headless subprocess: curves -> wrapped mesh -> ALPHA frames
   at EXACTLY Canvas.w/h/fps and ledger frame_count (no PyTorch3D/nvdiffrast) [PS-3D, H-RT]
                                v
OTR_SilentComposite (alpha over plate; flatten yuv420p) -> MasterAudioMux (mux-LAST, -c:a copy)
```

Mechanics (unchanged from v1.4 where not amended): `prepare()` = mesh ONCE per character,
**scoped to episode/session + char_id (not per shot/group)** [H-RT]; cache under
`character_mesh_cache/<id>/` keyed `portrait_content_hash + mesh_quality + mesh_engine_lock +
rig_tool_lock + driver_rig_format_lock`, **published atomically (write temp dir -> validate
manifest -> os.replace) with a per-key lockfile** [H-RT]; LRU 10 / 5 GB, evict at episode
teardown only. Sidecar lifecycle = spawn-per-execution-group with the `request_type` enum
(`mesh_prepass | render_group`) [v1.4-III]; **the mesh-gen spawn EXITS (NVML floor re-probed)
before Blender launches -- never co-resident** [H-RT, Gemini]. Every spawn (TripoSG, wrap if
GPU-backed, Blender, and any in-process fallback forward) brackets with the shared
`otr_gpu_residency.lockdir` lease + NVML residual-floor (~768 MB) check; per-spawn peaks
measured separately vs 14000 [PASS-3D #3, H-RT]. Driver curves travel as
`conditioning_refs["arkit_curves_json"]` -- no schema change [H-RT].

## 2.5 Upstream story assets -> the visual story (ledger / script / M4 / 2D stills)

The pipeline already produces everything a good-looking, story-driven episode needs; the 3D
path CONSUMES these rather than synthesizing new asset classes. This section is binding for
W7 (it is what makes the output a STORY, not a tech demo):

- **Script text -> phonemes (free lip-sync quality).** The ledger carries every line's exact
  text. Rhubarb accepts a dialog file (`--dialogFile`) and produces markedly cleaner phonemes
  with it than audio-only guessing [PS-3D "script-first variant"]. The Slot-2a extraction
  ALWAYS passes the line text from the ledger script; the curve cache key gains the
  script-line text hash. No new asset -- the script finally feeds the face.
- **Ledger beats + M4 mood -> deterministic SHOT GRAMMAR.** Per-beat camera/framing/light are
  POLICY-derived, not hard-coded: announcer beats = locked center bust; character dialogue =
  alternating 3/4 framings INSIDE the +/-15-deg bake cone; M4 mood/emotional-peak fields =
  slow push-in; scene transitions stay procgen title cards. Selection is seeded per the OTR
  true-randomization contract (OS entropy, `OTR_CAST_SEED`-style env override for
  reproducibility) and recorded in the render manifest. M4 remains the single prompt/style
  authority (V-11: policy fields, no new widgets); the stylized-shader palette derives from
  the episode's M4 style fields so 3D shots match the episode grade.
- **Delivery vectors -> the idle layer.** The shipped per-line delivery vectors (indextts2/
  chatterbox expressive lane) modulate the procedural blink/eye-dart/brow idle layer and the
  camera micro-energy, deterministically (vector in the curve cache key; `OTR_DELIVERY_VECTOR=0`
  kills it, matching the audio lane). The same emotion that shapes the VOICE now shapes the
  face and camera -- zero new models.
- **C's scene stills ARE the background plates.** The alpha head composites over the
  EXISTING Subproject-C scene still (ken-burns motion) or `ltx_video` clip per the shipped
  conditional-background sub-DAG -- stated explicitly so nobody synthesizes new plates for v1.
- **Framing hides the neck stump (the decapitation contract).** Slot-1.5 slices below the
  neck; v1's shot grammar therefore mandates BUST framing -- the frame bottom sits ABOVE the
  hole-filled stump in every grammar variant (period radio-studio close-up language, which fits
  the show). A LOOK-gate check asserts no stump pixels in any delivered frame.
- **The mesh-portrait HANDSHAKE -- the 2D engine learns it is feeding a 3D model.** The
  plumbing is the section-3 capability chain: the 3D adapter declares
  `requires_mesh_portrait=True` -> `OTR_VideoDirector` policy -> `OTR_ImageDirector` lock ->
  **M4, which emits the mesh-optimized framing IN THE TEXT PROMPT** (the technical model owns
  it; no new widget/mode, per [ROUTE-01]). The optimized-asset CONTRACT the M4 prompt encodes
  when the flag is set (derived from the known wrap-killers in the adversarial corpus):
  front-facing (yaw within ~+/-10 deg), neutral-to-mild expression with the **mouth CLOSED**
  (open mouths wreck the mouth-loop topology), full head + hairline + both ears in frame,
  even diffuse lighting (no hard side shadows -- they poison depth), sharp focus / no motion
  blur, NO glasses/hat/microphone occluding the face, plain mid-tone background (clean
  silhouette for mesh extraction), head dominant with shoulders visible (the same asset still
  serves the bake and HuMo). Cached with `purpose="mesh_portrait"` metadata (object_id +
  variant + engine_id + prompt_hash) so 2D and 3D portraits coexist without splitting
  character identity -- and per the character-level rule, once a character is 3D-assigned the
  clean portrait is used GLOBALLY for them.
  **Firing discipline (keeps the [ROUTE-01] verdict):** DEFAULT = stock flux portraits first;
  the handshake fires (a) automatically in the bounded retry -- mesh quality gate fails ->
  ONE `mesh_portrait` variant via the EXISTING C dispatcher -> retry once; and (b) as the
  GLOBAL default for 3D-assigned characters ONLY if the T3/T2b evidence says stock portraits
  wrap poorly. T3 therefore generates the corpus from STOCK portraits, records the
  per-portrait wrap outcome, and T2b's report ends with an explicit
  **stock-vs-mesh-prompt recommendation** -- the keystone data, not a hunch, decides whether
  the 2D engine is always told.
- **The ONLY candidate NEW asset is deferred:** an optional "bust card" (head-and-shoulders
  body underlay so wider shots can show a torso) -- if ever wanted, it is ONE M4-prompted
  variant from the EXISTING C image engine (a policy row, not a new pipeline), v1.1 at the
  earliest. Everything else (portraits, plates, line timing, script text, delivery vectors,
  mood) already exists upstream; the template is fetched (section 5) and the mapping table is
  authored -- no other inputs are synthesized.

## 3. Carry-in: the 3D image-routing MUST-FIXES (non-negotiable; code-verified still OPEN)

Carried from [GOFWD]/[ROUTE-01]; the hardening pass RE-VERIFIED each against the live
`otr_image_director.py` -- all still open. Land WITH the character_3d wiring, as these exact
edits [H-RT]:

- **`video_policy_json` is still OPTIONAL** (in the `optional` dict of `INPUT_TYPES`, default
  `"{}"`; `_parse_json_obj` warns instead of raising). FIX: move it to the **`required`** dict
  (ComfyUI only enforces a wired connection for required inputs) **PRESERVING
  `{"forceInput": True}`** -- without it ComfyUI auto-generates a multiline text widget,
  violating V-11/the static shell [H-RT p2, Gemini]; raise on empty/malformed; wire
  `OTR_VideoDirector -> OTR_ImageDirector` provider-before-consumer; add the downstream
  ShotLock/Dispatcher HALT on (3D engine + per_beat).
- **`enforce_3d_granularity_lock` currently COERCES** non-per_object to per_object with a
  warning. FIX: replace coercion with a raise (fail-closed; matches its own docstring).
- **`_is_3d_engine` still hard-codes** `family == "character_3d"` via the registry. FIX: read a
  capability field `requires_mesh_portrait: True`. **Schema reality: `AdapterDescriptor` and
  `VideoProfileRow` are `extra="forbid"` -- an ad-hoc key is rejected.** Add
  `requires_mesh_portrait: bool = False` as a REAL field to both models (preferred; migrate the
  schema/registry tests in the same commit) [H-RT p2, GPT]. Unknown custom engines selected for
  a 3D role with the capability missing FAIL CLOSED (covers `custom_models_json` adapters)
  [H-RT, DeepSeek].
- 3D-awareness is CHARACTER-level, not slot-level (clean front-facing portrait used GLOBALLY
  for that character).
- No "3D-ready image mode" widget; mesh-friendly framing, if ever needed, is an M4 PROMPT
  change (V-11). **No new widgets anywhere in the 3D path** [H-RT].
- Bounded RETRY (one mesh_portrait variant, once/N), never a DAG loop.
- Config SPLIT: Director holds image-routing only; mesh/wrap/driver/sidecar settings in the 3D
  adapter panel. Mixed 2D/3D in one slot = UNSUPPORTED.
- Cache metadata: object_id + variant/purpose + engine_id + prompt_hash alongside content hash.

## 4. Dependencies + the cu128 question

**Protected main venv (never touched):** py3.13, torch 2.10+cu130, numpy 2.4.4, transformers
5.5, diffusers 0.37, torchao 0.16, sageattention 2.2+cu130 [DEP]. All 3D GPU work runs in
isolated **cu128 sidecars** (Path-B; templates `eng_indextts2.py` / `eng_latentsync.py`).

**The re-frame [FORK-RT]:** the cu128 **VENV** (prebuilt wheels, no compiler) is distinct from
the cu128 **TOOLKIT** (ninja + nvcc 12.8 + VS2022, source builds only). probe_a's NO-GO
[B-SHIP] blocks only the toolkit lane. **v1 = the NO-COMPILE lane, conditional on S-3D-0:**

| Component | v1 pick | Verdict | Toolkit? |
|---|---|---|---|
| Slot 1 | TripoSG (MIT, ~6 GB geometry) | prebuilt cu128 wheels + SDPA -- **PROBE S-3D-0** (mcubes/diso ext risk) [3DC] | No (if probe passes) |
| Slot 1.5 | trimesh + Blender headless | pure python + standalone Blender (never `pip install bpy`) [DEP] | No |
| WRAP | template shrinkwrap (trimesh + scipy sparse) [PASS-3D] | pure python | No |
| Slot 2a | Rhubarb (MIT, CPU) + mapping table | binary, zero GPU | No |
| Slot 2b | Blender headless, alpha frames | standalone | No |
| OPT-IN | A2F-3D (appendix, section 10) | own TRT vendor runtime | Own toolchain |
| DEFERRED | Hunyuan3D, TRELLIS, SuGaR | source builds [DEP] | **Yes** |

Sidecar venv spec: pin sidecar Python to the TripoSG wheel set (expect py3.10/3.11); wheel
acquisition = separate online step with SHA-256 pins; build runs offline `--no-index
--find-links <wheelhouse> --only-binary=:all:`, FAILS on any sdist; self-test asserts SDPA + no
flash_attn import; **env built via the shipped `_b_harness.build_sidecar_env()` sanitizer**
(strips PYTHONPATH/TORCH_HOME/HF caches/PIP_*, sets PYTHONNOUSERSITE + CUDA_HOME +
TORCH_EXTENSIONS_DIR) so the protected venv can never leak in [H-RT, GPT]. Licenses: TripoSG
MIT, Step1X-3D Apache-2.0, Rhubarb MIT, ICT-FaceKit MIT (DATA audit, section 5), Blender GPL
(subprocess = fine), Hunyuan3D NC+territory (operator sign-off before weights) [DEP, 3DC].

## 5. Assets -- fetchable vs operator-supplied

| Asset | Status | How |
|---|---|---|
| **ARKit-52 template** (`OTR_B_ARKIT_TEMPLATE_NPZ`: `verts/faces/mouth_idx + delta_<name> x52`, matching `_b_harness.ARKIT_52`) | **FETCHABLE** (was assumed operator-blocked [B-SHIP]) | Derive from **USC ICT-FaceKit** (MIT; ARKit-named shapes with `_L/_R` splits). Converter merges ONLY canonically-singular shapes (browInnerUp, cheekPuff, ...), PRESERVES the canonically-asymmetric L/R pairs, via a checked-in **`facekit_to_arkit.json`** name map (classes copy / merge_to_unilateral / split_preserved / drop) and emits a **machine-readable conversion report** (source name, target name, class, vertex count, topology hash, delta-norm stats); a MISSING shape is a HARD FAIL -- never a zero-delta placeholder [H-RT, GPT/DeepSeek]. Validates exactly 52 deltas + `mouth_idx` + manifoldness; jawOpen/mouthClose/mouthFunnel deformation smoke (reuse `wrap_topology_check` max_delta_ratio<=0.5 as a property test on all 52 deltas at weight 1.0); SEMANTIC checks beyond name-count: coordinate-system/scale vs the harness convention, left/right delta sign sanity, per-shape delta-norm ranges, golden deformation thumbnails (a 52-name npz can still be semantically wrong) [H-RT p2, GPT]; pins `ARKIT_TEMPLATE_HASH`. License gate: audit the model-DATA terms, pin commit + per-asset SHA-256; fetch = separate acquisition step. REJECTED: MetaHuman extracts (Epic), paid add-ons [FORK-RT]. |
| **~25-mesh keystone corpus** (`OTR_B_MESH_DIR`) | **GENERATED by the pipeline** | TripoSG sidecar from ~25 real Flux portraits incl. adversarial (glasses, hair-over-face, profile, open mouth, low-res) **+ period-style cases (sepia/monochrome, high-contrast studio light, hats, microphone occlusion)** [PS-3D, H-RT]. Corpus is probe_c EVIDENCE only -- never a runtime gate (section 7.1). |
| TripoSG / Step1X-3D weights | Fetchable (HF, ungated) into `C:\ComfyUI-Models` | After S-3D-0; SHA-256 pinned |
| Rhubarb binary + `rhubarb_to_arkit.json` | Fetchable (MIT) / authored in-repo | Mapping reviewable, golden curve plots; driver ALWAYS fed the ledger script line via `--dialogFile` (section 2.5) |
| Background plates | **REUSE C's scene stills / ltx clips -- no new synthesis** | Conditional-background sub-DAG (section 2.5) |
| "Bust card" body underlay | OPTIONAL, v1.1+ -- one M4-prompted variant from the EXISTING C engine | Only if wider-than-bust shots are ever wanted (section 2.5) |
| Blender (standalone) | Fetchable into `C:\ComfyUI-Models\tools\` | Operator INFORMED (not an interactive gate) [H-RT, Gemini] |
| **cu128 TOOLKIT** | OPERATOR-BLOCKED, **deferred out of v1** | Only Hunyuan3D/TRELLIS/SuGaR/flash_attn [B-SHIP] |
| A2F-3D weights + TRT | Operator-gated (NVOML read) | Appendix tier |
| Hunyuan3D weights | Operator-gated (NC + EU/UK/KR recorded) [DEP] | Deferred engine |

## 6. Prerequisite spikes (gate before any adapter code)

Keystone-first, corpus-honest [FORK-RT]; `keystone_gate` in `_b_harness.py` ALREADY encodes the
strict rule (GO only if failure rate **< 20%**; exactly 5/25 = NO-GO) -- docs/tests adopt that
phrasing everywhere, no harness change [H-RT].

| Spike | What | BINDING pass/fail |
|---|---|---|
| **S-3D-0 (gates the lane)** | TripoSG sidecar venv OFFLINE from wheels only (`--only-binary=:all:`, `build_sidecar_env()` sanitizer); self-test: import clean + SDPA + no flash_attn + portrait->GLB + manifold + mesh-spawn VRAM <= 14000 | Any source build -> NO-GO. **Pre-step before declaring dead: swap the extraction stage to a wheel-clean CPU marching-cubes -- candidates in order: `skimage.measure.marching_cubes` (scipy stack, pure wheels), `PyMCubes` (verify a binary wheel exists for the sidecar python), else NO -- and RECORD the exact outcome so the operator decides informed** [H-RT p2]. Hard NO-GO -> **OPERATOR DECISION**: (a) v1 ships HuMo-2D only, 3D stays deferred; or (b) operator approves the machine-level cu128 toolkit install (VS2022 BuildTools + nvcc 12.8 + ninja; hours) and the lane proceeds compiled. Neither is automatic [H-RT, GPT]. ORDERING NOTE: the shipped spike README says "cheap keystone screen first" -- that rule predates the corpus-honest reorder (the binding keystone needs TripoSG meshes, so S-3D-0 must precede it); T2a remains the cheap screen; update the README wording in the W7-pre docs task [H-RT p2, GPT] |
| **T1 template** | Fetch ICT-FaceKit (pinned commit) -> converter + report (section 5) -> pin `ARKIT_TEMPLATE_HASH` | Fail-closed: 52 deltas + mouth_idx + deformation smoke, else phase stays parked |
| **T2a wrap smoke (NON-BINDING)** | Existing `probe_c` synthetic harness + the real template | Labelled NON-BINDING [B-SHIP]; minutes |
| **T3 corpus** | TripoSG generates the 25-mesh corpus from **STOCK flux portraits** (adversarial + period styles), recording the per-portrait wrap outcome -- T3 doubles as the stock-portrait spike that decides the mesh-portrait handshake default; T2b's report ends with the stock-vs-mesh-prompt recommendation (section 2.5) | 25 meshes; probe_b manifold pre-screen each, **plus trimesh/Blender self-intersection + normal-orientation checks (manifold_report alone misses self-intersection)** [H-RT, GPT] |
| **T2b KEYSTONE (BINDING)** | `probe_c` wrap on the TripoSG corpus; automated deformation-transfer, TIMEBOX ~1 week [PASS-3D] | GO = failures <= 4/25. NO-GO -> HuMo-2D stays; character_3d deferred [GOFWD] |
| **T4 driver + alpha + LOOK** | Rhubarb -> curves -> Blender alpha render -> SilentComposite flatten -> mux; all spawn peaks measured | One real talking alpha clip; onset +-1 frame compensated INSIDE the fixed-length clip -- **the delivered silent clip keeps EXACTLY the ledger `target_frame_count` sum; never shortened** (V-1 stays untouchable) [H-RT, GPT]; audio stream hash == master; yuv420p after flatten; checkerboard seam golden (straight-vs-premultiplied documented); LOOK gate per section 7.4 |

## 7. Build contracts + waves

### 7.0 W7-pre: the migration slice (reconciling the shipped scaffold) [H-RT, all panelists]
The repo ALREADY ships `eng_character_3d.py` registering `hunyuan3d_talk` + `trellis_talk`,
`render_driver.SYNTH_FALLBACKS = {"hunyuan3d_talk": "humo"}`, `ENGINE_FAMILY`/`OOM_ENGINES`/
`EXPECTED_OOM_TRAIL` entries, the soak twin in `scripts/otr_video_soak.py`, and expectations in
`test_video_character_3d.py` / `test_video_render_driver.py` / `test_video_soak_fixture.py`.
ONE atomic slice, before any live forward:
- ADD `TripoSGTalkEngine` (`triposg_talk`, `OTR_ENABLE_TRIPOSG_TALK`,
  `OTR_TRIPOSG_SIDECAR_PYTHON`, `requires_mesh_portrait=True`, `fallback_engine="humo"`) as a
  third dark adapter **with its OWN `assert_usable` helper -- do NOT edit the shared
  `_assert_usable_3d` (hunyuan/trellis keep their existing semantics + tests unchanged)**
  [H-RT p2, GPT/DS/Grok]. KEEP `hunyuan3d_talk`/`trellis_talk` registered-dark (future engines,
  not legacy; the probe decides what ever goes live).
- Update `SYNTH_FALLBACKS`, `ENGINE_FAMILY`, `OOM_ENGINES`, `EXPECTED_OOM_TRAIL` (BOTH copies;
  first hop string becomes `"triposg_talk->humo (oom)"`), `_CHAR3D` soak profile, and the three
  test files in the SAME commit, targeted suites green. **Consistency check while renaming:**
  `assert_soak_ok` expects exactly 3 LOUD OOM decisions while `EXPECTED_OOM_TRAIL` lists 4 hops
  (the `humo->humo_1.7B` hop is an intra-engine tier swap, not a restamp decision -- the soak is
  green today, so this is semantics to PRESERVE, not a bug to "fix"; keep the two constants
  consistent under the new names) [H-RT p2, judge ruling on GPT's claim].
- **Builder migration (code-verified gap):** `build_request()`/`build_request_from_shot()` emit
  dicts `VideoRequest` rejects (`extra="forbid"`): extras `init_w/init_h`, `timing["dur_s"]`
  (the schema field is `target_duration_s`), top-level `char_id`, and missing
  `role`/`family_hint`/`profile_id`. Fix the builders to emit schema-valid requests (`char_id`
  rides in `conditioning_refs`); CPU test:
  `VideoRequest.model_validate(build_request_from_shot(...character_3d...))` passes and a
  missing `audio_ref`/`init_image` fails closed [H-RT p2, GPT/DS/Grok].
- **Fallback chain unit test:** `triposg_talk -> humo -> humo_1.7B -> latentsync ->
  still_kenburns` via `make_fallback_of()`, terminating, no cycles; unknown non-floor engines
  degrade to `still_kenburns` [H-RT p2]. **Down-chain request shape (p3, GPT):**
  `render_shot()` passes ONE request unchanged through every candidate, but `lipsync_overlay`
  requires `base_clip_ref` that a `character_3d` request lacks -- REBUILD/validate the request
  per candidate after each restamp; if the next family's required inputs can't be satisfied,
  SKIP it loudly to a compatible floor (never feed a 3D request to latentsync). Also update the
  `assert_soak_ok` carryover check `input_oom_engine != "hunyuan3d_talk"` to the new id (p3,
  Gemini) and leave a comment explaining the 4-hop-trail/3-decision semantics.
- **Resolver-prune wiring (p3, GPT):** `resolver.py` ships the orphaned-background prune, but
  `render_shot()` never calls it -- on any family-changing fallback, run the prune against
  `ledger['video']['execution_groups']` in the SAME ledger transaction as the shot restamp +
  decision append; test: 3D consumer degrades to humo -> background provider group removed.
- ComfyUI RESTART after the .py change (module cache); post-restart GATE: static dropdown shows
  dark `triposg_talk`, cold-import pulls no torch/diffusers/comfy, no widget mutation [H-RT].
  Then re-run the A-S7.5 soak (or the sanctioned lighter B-sidecar smoke).

### 7.1 Runtime `assert_usable` (live triposg_talk) -- the deadlock fix [H-RT, Gemini]
Order: flag -> sidecar venv -> **TripoSG weights** (`local_files_only`) -> ARKit template npz +
`ARKIT_TEMPLATE_HASH` match -> Rhubarb binary -> Blender exe. **Never the generated meshes**
(`OTR_B_MESH_DIR` is probe_c evidence only -- gating on it deadlocks `prepare()`, which is what
GENERATES meshes). Re-check on EVERY attempt (no cached "missing") [GOFWD]. The dark scaffold's
mesh-dir check is REMOVED for triposg_talk in the W7-pre slice.

### 7.2 I/O contracts (code-verified) [H-RT, GPT]
- `VideoRequest` is `extra="forbid"` and requires role/family_hint/profile_id; the builders are
  migrated in W7-pre (section 7.0). The 3D adapter ADDITIONALLY **schema-validates via
  `VideoRequest.model_validate` at its boundary BEFORE any sidecar spawn** (defense in depth).
  **Canvas: character_3d REQUIRES the episode canonical 16:9 canvas from `ledger['video']` and
  REJECTS the builder's 480x832 portrait default** [H-RT p2, GPT].
- Alpha handoff = **ONE contract**: `CanonicalClip(type="directory", pixel_format="rgba",
  alpha="straight", has_audio=False, frame_count=N)` of PNG/EXR frames. **`OTR_SilentComposite`
  does NOT read frame directories today -- implementing the directory input (frames sorted by
  name -> overlay -> flatten yuv420p) is a NAMED W7 build task with a CPU/golden straight-alpha
  test** [H-RT p2, DS/GPT]. A canonicalize-time validator enforces type/pixel_format/alpha/
  has_audio/`frame_count == timing.target_frame_count` == frames on disk -- never trust the
  schema defaults (`container="mp4"`/`codec="h264"` defaults are meaningless for directories).
  **Directory semantics ripple (p3, GPT):** `_clip_summary()` uses `os.path.getsize(path)` and
  `all_clips_real` requires size > 0 -- update `_clip_summary`/`build_clip_manifest`/soak
  assertions to treat `type=="directory"` as "dir exists + exactly N sorted nonzero frames";
  add a CPU directory-clip fixture through manifest -> composite -> mux BEFORE any Blender
  work. The `webm/vp9`/`mov/prores4444` branches are CUT from v1 [H-RT, GPT].
- **Curve-file validation before the Blender spawn**: existence, JSON schema, fps,
  frame_count, channel set, mapping hash -- malformed curves fail closed to fallback, never a
  frozen mouth [H-RT p2, GPT]. Test fixture: a tiny synthetic GLB with 52 stub shape keys
  exercises Blender curve application without TripoSG [H-RT p2].
- `execution_groups.depends_on` takes **GROUP IDs** (resolver validates membership): the
  character_3d consumer group depends on the ShotLock-stamped portrait provider GROUP id --
  never the literal string "flux_portrait". Resolver unit test: portrait provider +
  character_3d consumer + background provider, prune-on-degrade (AS-2 path already shipped in
  `resolver.py`).
- Blender render contract: input = wrapped GLB (exactly 52 keys) + `arkit_curves_json` +
  camera/framing; output frames at EXACTLY `Canvas.w/h/fps` and ledger frame_count; Blender
  writes NO audio; full process exit before any fallback load. **Determinism pins: Blender
  version, render engine, color management, camera/lights, seeds, device selection, output
  frame naming -- all recorded in the render manifest** [H-RT p2, GPT]. The 3D path renders at
  the episode 16:9 canvas natively -- NO pillarbox inheritance from HuMo.

### 7.3 Ledger + caches [H-RT]
- Stamps go to the SINGLE `ledger['video']` via the shipped path (`restamp_shot_row` +
  `append_runtime_fallback_decision`, same `video_revision`); the 3D adapter calls the SAME
  functions on fallback -- no new ledger machinery. `ledger['audio']` is byte-for-byte
  untouched; add a test asserting audio-section equality before/after a character_3d episode
  INCLUDING the fallback path.
- **Sample-rate reality**: the shipped `_slice_master_audio` writes **44.1 kHz** mono
  (`-ar 44100`, render_driver.py:244) for HuMo. The Rhubarb driver gets its OWN 16 kHz
  extraction (a `sample_rate` parameter or a separate `_slice_master_audio_for_driver`) --
  **HuMo's slicer is NOT changed** [H-RT p2, GPT/Gemini].
- **Cache keys SPLIT** (don't over-key the cheap WAV) [H-RT p2, Gemini]: SLICE key = master
  CONTENT hash (`ledger['audio']['master_audio_sha256']`, already available) + start_s + dur_s
  + sample rate + channels + slicer version (the shipped path-only key under-invalidates on a
  new master at the same path); CURVE key = slice key + line_id + fps + driver version +
  mapping-table hash + onset policy (line_id/beat_id threaded from the request builder).
  **Mechanics (p3, Gemini):** the slicer signature gains the hash
  (`_slice_master_audio(..., master_hash)` fed from `ledger['audio']['master_audio_sha256']`);
  and since the 3D adapter only ever receives the already-sliced 44.1 kHz `audio_ref.path`
  (not the master), the 16 kHz Rhubarb input is a DOWNSAMPLE OF THAT SLICE (ffmpeg, MAIN or
  sidecar-local), never a re-slice of the master. **ShotRow is `extra="forbid"` (p3, GPT):**
  artifact PATHS may not be appended as top-level shot-row keys -- stable keys go in the
  existing `cache_keys` dict (or add a real `artifact_refs` field with test migration); paths
  live in the clip manifest / `CanonicalClip.qc`. Nothing in `ledger['audio']` [H-RT, GPT/DS].
- Mesh cache: the KEY AUTHORITY is the decoded-pixel portrait hash + the three `*_lock`
  versions (+ object_id/variant/engine_id/prompt_hash per section 3) -- `char_id` only scopes
  WHEN prepare() runs, never the cache identity (same char_id with a changed portrait hash
  REGENERATES; add that test) [H-RT p2, GPT]. Atomic publish + lockfile -- **Windows caveat:
  `os.replace` is not atomic for non-empty directories; publish a versioned directory then
  atomically flip a small pointer/manifest FILE; stale-lock timeout documented** [H-RT p2,
  GPT].
- **Ledger on-disk transaction**: `run_episode` mutates a deep-copied ledger in memory; any
  on-disk persistence is write-temp -> fsync -> atomic replace, so a sidecar/Blender death
  mid-shot never leaves a partial ledger [H-RT p2, GPT].
- Node-level caching: the adapter does NOT rely on ComfyUI `IS_CHANGED` reuse; any node
  surfacing 3D artifacts keys its change-hash on concrete artifact hashes (template hash, mesh
  cache key, curve key, request hash) -- never mtime [H-RT].

### 7.4 The v1 LOOK contract (end-result quality) [H-RT, Claude-panelist/GPT/DeepSeek]
TripoSG is a geometry foundation -- raw output may be untextured. v1 will NOT ship a gray clay
head:
- **Primary look**: Blender UV-project-from-view bake of the LEDGER PORTRAIT onto the wrapped
  mesh (deterministic; reuses the identity asset we already trust) -- **BOUNDED to a frontal
  camera cone (~+/-15 deg yaw); beyond the safe projection angle the stylized fallback
  auto-triggers** (a single frontal projection smears ears/sides/back -- the "hollow face"
  trap) [H-RT p2, Gemini/DS/GPT]. The render manifest RECORDS the camera yaw/pitch + head-pose
  envelope per shot; ANY delivered frame outside the cone (or `qc.uv_unmapped_ratio` /
  max-UV-stretch over threshold) auto-switches that shot to the stylized material -- the
  metrics are actually EMITTED into `CanonicalClip.qc` by the Blender wrapper, or the gate is
  unenforceable [H-RT p3]. Side-view frames included in the T4 contact sheet.
- **Fallback look**: a deterministic 1940s-radio stylized/sepia shader (limited palette;
  post-composite film grain optional), stamped `qc.texture_mode="stylized_no_albedo"` in
  `CanonicalClip.qc`. Judge ruling: the portrait bake STAYS primary (identity matters; DS's
  "cut the bake" rejected) but inside the bounded cone only.
- **Operator visual sign-off**: T4's report carries a human "looks right" boolean (texture,
  seams, lip-sync readability) -- automated gates alone don't define GOOD [H-RT p2, DS].
- **Material QC gate** (T4 + per-render): nonzero albedo/vertex-color coverage on the face, no
  transparent face, no inverted normals, no missing eye material, no mouth hole through the
  skull; failures restamp via the SAME loud fallback path with a distinct `qc_failed_mesh` /
  `qc_failed_alpha` detail string.
- **Driver scope, stated honestly**: v1 acceptance = a "mouth-readable Rhubarb performance"
  (jaw/lip channels driven; brows/cheeks/tongue are deterministic idle layers or zero) -- NOT a
  full 52-channel facial performance; assert all non-driven channels are idle-layer or zero;
  side-by-side Rhubarb-vs-silence mouth-closure test + max-mouth-open-on-silence guard
  [H-RT p2, GPT].
- **Story integration QC (section 2.5 enforced here)**: T4's clip is rendered from a REAL
  ledger beat -- script line fed to Rhubarb via dialogFile, shot grammar selected from the
  beat's role + M4 mood, delivery vector modulating the idle layer, plate = the episode's C
  still, no stump pixels in frame. A clip that passes the mechanical gates but used none of
  the upstream story assets FAILS T4.
- **Driver QC suite**: golden plots for the viseme->ARKit mapping; jawOpen/mouthClose/
  mouthFunnel/mouthPucker sanity ranges; silence closes the mouth within N frames; coefficients
  clamped to [0,1]; Savitzky-Golay smoothing; the deterministic blink/eye-dart layer NEVER
  conflicts with mapped `eyeBlink*` channels; per-character debug contact sheet (portrait, raw
  mesh, sanitized, wrapped neutral, jawOpen, final alpha frame).

### 7.5 Waves
- **W7/B1 -- `triposg_talk` live**: prepare (mesh-once) + wrap + Rhubarb + Blender alpha +
  caches + the section-3 ImageDirector edits + SilentComposite alpha-directory input. B1
  REOPENS SilentComposite -> A's W2 `widget_vector_exact` golden + audio-green gate = the
  MUST-PASS exit [v1.4-III, PASS-3D]. Heavy in-process FALLBACK forwards (humo) stay on the
  ComfyUI EXECUTOR thread via the `OTR_VideoRenderBatch` /prompt path (the sidecar spawn may
  block there; IPC uses timeout + periodic poll so cancellation still works; `taskkill /F /T`
  on hang) [H-RT, GPT/DeepSeek].
- **Fallback ladder v1 (matches the shipped chain)**: `triposg_talk -> humo -> (humo_1.7B ->
  latentsync) -> still_kenburns`; every hop LOUD; no CPU fallback for character_3d; the
  MediaPipe-extrusion floor stays deferred to the playground [H-RT, FORK-RT].
- **W7.5/W8/B2 -- `step1x3d_talk`** (geometry-only; stylized-look acknowledged) only AFTER
  v1-usable; = the B-parity ship milestone [3DC, H-RT].

**Gates carried**: `test_audio_byte_identical` green at every step; per-spawn VRAM <= 14000
under the lease; determinism v1 = identical `request_hash` across two spawns PLUS a structural
manifest compare (engine id, cache keys, frame counts, alpha mode, fallback decisions --
excluding absolute temp paths) [H-RT]; cold-import (V-12) + static dropdown (V-6); mesh-cache
hit on line 2; non-blank assertion; SageAttention/BUG-070: N/A to the Blender lane, but stays a
gate if Step1X's SDXL texture stage is ever enabled.

## 8. Forks -- RESOLVED (recommendation each)

1. **Mesh-gen: TripoSG first**, Step1X-3D second (post-v1-usable); Hunyuan3D/TRELLIS deferred
   (license/territory + toolkit + probes). [3DC 4/4, FORK-RT, H-RT]
2. **cu128: toolkit DEFERRED; v1 = no-compile lane**, conditional on S-3D-0; NO-GO = operator
   decision (HuMo-2D-only vs approve toolkit), with the CPU-marching-cubes pre-step. [FORK-RT,
   H-RT]
3. **ARKit-52 template: fetchable** (ICT-FaceKit + audited converter + report), not
   operator-supplied. [FORK-RT, H-RT]
4. **Ordering: S-3D-0 -> T1 -> T2a -> T3 -> T2b -> T4** (one binding sequence, stated
   identically everywhere; the spike README's older "cheap keystone screen first" wording is
   updated in W7-pre); Slot-2a decoupled. [FORK-RT, H-RT p3]
5. **Driver: Rhubarb default**; A2F-3D = appendix opt-in tier. [GOFWD supersession, FORK-RT]

**Operator decisions (not blockers):** (a) green-light S-3D-0; (b) the S-3D-0 NO-GO branch
choice if it fires; (c) A2F-3D NVOML acceptability (only when the opt-in tier is wanted);
(d) Hunyuan3D NC sign-off (only if ever wanted); (e) B-parity timing -- is the second engine
required before episodes use 3D, or after?

## 9. Definition of done

**v1-usable**: a live `triposg_talk` clip in a REAL episode -- generated mesh, 52-key wrap,
Rhubarb-synced to frozen audio **with the ledger script line driving the phonemes, the beat's
role + M4 mood driving the shot grammar, the delivery vector driving the idle layer, and the
episode's C still as the plate (section 2.5 -- the ledger and script visibly telling the
story)**, portrait-baked or stylized look (LOOK gate green), alpha composited + flattened,
master audio byte-identical, all spawns <= 14.0 GB, request-hash + manifest deterministic,
mesh-cache hit on line 2, fail-closed LOUD to humo when assets absent, full suite + Bug Bible
green, soak (or sanctioned smoke) re-certified with the real trail.
**B-parity ship**: `step1x3d_talk` selectable and passing the same per-engine gates; the
playground opens after that.

## 10. Appendix -- the A2F-3D opt-in tier (deferred detail)

NVIDIA Open Model License (operator legal read); own TRT vendor-runtime sidecar (NOT the torch
cu128 pattern); `.engine` pre-compiled at setup, never at render [PASS-3D #4]; onset machinery
(`probe_d` / `classify_onset`) applies HERE -- engine-constant onset -> fixed video trim inside
the fixed-length clip, variable onset -> tier NO-GO [v1.4-III]. **probe_d is explicitly NOT a
binding gate for the Rhubarb v1 ship** (Rhubarb curves are timestamp-aligned by construction)
[H-RT p2, Gemini]. UniTalker stays v2 (NC checkpoint); EmbedTalk stays verify-gated [3DC].
