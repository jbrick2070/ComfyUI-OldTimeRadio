# OTR 3D Toolkit Plan -- the `character_3d` family (NEXT PHASE, canonical)

> **STATUS: CONSOLIDATED 2026-06-09 (planning window, docs-only).** The 2D video build is
> SHIPPED (B-ship @ f003978, M0-M5 green, HuMo-2D character path). 3D stays **PARKED -- no
> implementation code, no spikes run, no asset downloads** until the operator green-lights this
> plan. This file supersedes the skeleton at `otr-video-roundtable\3D_TOOLKIT_PLAN.md` and is
> the single canonical 3D plan. Source docs remain in `C:\Users\jeffr\Documents\otr-video-roundtable\`
> (cited inline as [v1.4-III], [SUB-B], [DEP], [PS-3D], [3DC], [GOFWD], [ROUTE-01], [B-SHIP],
> [PASS-3D], [FORK-RT] = `roundtable_campaign/2026-06-09-3dplan/pass01_judgment.md`, this
> consolidation's 3-model roundtable).

---

## 1. Goal + scope

The 3D character path delivers the **8th engine family, `character_3d`**: a CONSUMER adapter
that turns a ledger portrait + a frozen audio line into a talking 3D character rendered as an
**ALPHA clip**, composited over a background by `OTR_SilentComposite`, with the master audio
muxed LAST and byte-identical [v1.4-III]. It plugs into the shipped model-agnostic platform
(registry, Probe/Director/ShotLock/RenderBatch/SilentComposite/Mux, per-role selectors, fallback
resolver) as a peer engine -- sidecar-isolated under V-12, on a **14.0 GB sidecar sub-ceiling**
(stricter than the 14.5 GB machine ceiling) [SUB-B]. The strategic frame is the operator's 3D
playground thesis: the path is a **pluggable slot inside a real end-to-end narrative pipeline**,
so any new 3D model can be battle-tested in production, not admired in a turntable demo [PS-3D].

**Milestone before everything else: 1-2 mesh-gen models rendering a real character into a real
episode** [GOFWD]. Explicitly OUT of scope for v1: the Airlock conformance test, the
`otr new-adapter` scaffold, the gauntlet micro-episode, new narrative roles (2.5D parallax,
Tier-A EXR passes), 3D-as-control-for-2D, the artist's-pass devices, bodies/gestures, crowds --
all DEFERRED to the playground phase after the first path is green [PS-3D, GOFWD]. v1 renders
photoreal-ish texture only as far as the chosen mesh engine provides it; Step1X-3D geometry-only
implies vertex-color/stylized shading in the Blender render, acknowledged up front [FORK-RT].

## 2. The pipeline (end to end)

```
flux_portrait (ledger, decoded-pixel hash)            frozen master audio (read-only)
   |                                                     |
   v                                                     v
[Slot 1  MESH-GEN]  portrait -> head mesh (GLB)      [Slot 2a PERF]  audio -> 52 ARKit
   |  (triposg_talk first; engine-agnostic)            weight curves (Rhubarb default;
   v                                                   CPU; cached by audio-segment hash;
[Slot 1.5 PREP]  decapitation slice -> sanitize        runs IN PARALLEL with Slot 1 --
   (scale/origin normalize, remove loose parts,        it needs NO mesh) [FORK-RT]
   recalc normals, hole-fill INCLUDING the neck                 |
   stump) -> optional deterministic remesh                      |
   |                                                            |
   v                                                            |
[WRAP -- the KEYSTONE]  template-first shrinkwrap onto the      |
   canonical ARKit-rigged template (barycentric/ICP register,   |
   bake offsets, copy the 52 reference shape-key deltas) ->     |
   GLB carrying EXACTLY 52 ARKit shape keys [PS-3D]             |
   |                                                            |
   +----------------------------+-------------------------------+
                                v
[Slot 2b DEFORM/RENDER]  Blender headless subprocess applies the curves to the wrapped
   mesh, renders ALPHA frames (no PyTorch3D / nvdiffrast / custom-CUDA rasterizer) [PS-3D]
                                v
OTR_SilentComposite (alpha over plate; flatten to yuv420p)  ->  MasterAudioMux (mux-LAST,
   -c:a copy, no -shortest, audio byte-identical) [v1.4-III]
```

Registry family `character_3d`; the adapter (`nodes/_otr_video_engines/eng_character_3d.py`
dark scaffold per [GOFWD] Phase 3) encapsulates this whole pipeline behind one engine per
mesh-gen backend; the driver is an output of the driver gate, never baked into the engine id
(`_talk`, not `_rhubarb`/`_a2f`) [v1.4-III]. Engine IDs follow the probe winner -- do NOT
hard-commit `hunyuan3d_talk`/`trellis_talk` names in scaffold/SYNTH_FALLBACKS/soak/tests; the
first registered pair is expected to be `triposg_talk` (+ later `step1x3d_talk`) [GOFWD,
FORK-RT]. Mesh is generated ONCE per character (`prepare()`), cached under
`character_mesh_cache/<id>/` keyed by `portrait_content_hash + mesh_quality + mesh_engine_lock
+ rig_tool_lock + driver_rig_format_lock`, LRU max 10 entries / 5 GB, evict only at episode
teardown [v1.4-III]. Sidecar lifecycle = spawn-per-execution-group, JSON file-pair IPC, atomic
`os.replace`, NVML residual-floor gate (~768 MB) before every spawn; shared
`otr_gpu_residency.lockdir` lease (AS-3) [v1.4-II/III, SUB-B]. Per-spawn VRAM is MEASURED
separately for the mesh-gen spawn, the wrap spawn, and the Blender render spawn against
`sidecar_process_budget_mb=14000` -- never assumed from vendor mesh-gen numbers [PASS-3D #3,
FORK-RT].

## 3. Carry-in: the 3D image-routing MUST-FIXES (non-negotiable)

Carried verbatim from [GOFWD] "3D image-routing -- must-fixes before character_3d ships
(roundtable 2026-06-08)" + [ROUTE-01]. Land these WITH the character_3d wiring:

- **BUG (fail-open -- the real find):** `OTR_ImageDirector.video_policy_json` is OPTIONAL
  (defaults `"{}"`, malformed -> warned). If `OTR_VideoDirector -> OTR_ImageDirector` isn't
  strictly wired, the 3D granularity lock fails SILENTLY -> `per_beat` slips through for a 3D
  character -> mesh-per-beat VRAM thrash. FIX: make it REQUIRED + fail-closed (raise on
  empty/malformed); wire provider-before-consumer; add a downstream HALT (ShotLock/Dispatcher)
  on (3D engine + per_beat granularity).
- 3D-awareness is CHARACTER-level, not slot-level: if ANY role assigns 3D to a character, use
  the clean front-facing portrait for that character GLOBALLY (2D renders fine from it) -- else
  the portrait cache splits + consistency breaks.
- Capability-declared 3D detection: lock on a policy field (`requires_mesh_portrait`), NOT the
  hard-coded `family=="character_3d"` registry lookup, so "+ Add Custom Model" 3D adapters get
  the lock.
- The 3D-aware behavior ALREADY exists as the `three_d_locked_slots` -> `per_object`
  (mesh-once-per-character) lock; do NOT add a "3D-ready image mode" widget. If the
  stock-portrait mesh spike FAILS, the mesh-friendly framing is an M4 PROMPT change (wire
  `video_policy` -> M4), NOT a new mode (V-11).
- The "loop-de-loop" = a BOUNDED RETRY (portrait -> mesh -> quality gate -> on fail, ONE
  mesh_portrait variant + retry once/N), NEVER a DAG loop back to the still nodes.
- Config SPLIT: the Director policy holds image-routing ONLY; mesh/wrap/driver/sidecar settings
  stay in the 3D-adapter panel (spine section J). Mixed 2D/3D in one slot = UNSUPPORTED
  (document; needs per-beat engine selection, out of scope).
- Minor [ROUTE-01]: fix the docstring/behavior mismatch (3D `per_beat` must RAISE fail-closed,
  not silently coerce); cache metadata stores object_id + variant/purpose + engine_id +
  prompt_hash alongside the content hash.

## 4. Dependencies + the cu128 question (re-framed by this consolidation)

**Protected main venv (never touched):** py3.13, torch 2.10+cu130, numpy 2.4.4, transformers
5.5, diffusers 0.37, torchao 0.16, sageattention 2.2+cu130 [DEP]. The ENTIRE 3D stack runs in
isolated **cu128 sidecars** behind the proven Path-B subprocess/JSON pattern (templates:
`eng_indextts2.py`, `eng_latentsync.py`) [DEP, GOFWD].

**The re-frame (roundtable-hardened, [FORK-RT]):** distinguish the cu128 **VENV** (pip from
prebuilt wheels -- needs NO compiler) from the cu128 **TOOLKIT** (ninja + nvcc 12.8 + VS2022 --
needed ONLY for source-built CUDA extensions). probe_a's NO-GO [B-SHIP] blocks only the
toolkit lane. **v1 = the NO-COMPILE lane**, CONDITIONAL on spike S-3D-0 (section 6):

| Component | v1 pick | Dep verdict | Needs toolkit? |
|---|---|---|---|
| Slot 1 mesh-gen | TripoSG (MIT, ~6 GB geometry) | prebuilt cu128 wheels + SDPA -- **PROBE S-3D-0** (mcubes/diso ext risk) [3DC, FORK-RT] | No (if probe passes) |
| Slot 1.5 prep | trimesh + Blender headless subprocess | pure python + standalone Blender (KNOWN-GOOD pattern; never `pip install bpy`) [DEP] | No |
| WRAP keystone | template-first shrinkwrap (trimesh + scipy sparse deformation-transfer; no human sculpting) [PASS-3D] | pure python | No |
| Slot 2a driver | Rhubarb Lip Sync (MIT, CPU) + viseme->ARKit mapping table | binary download, zero GPU | No |
| Slot 2b render | Blender headless subprocess, alpha EXR/PNG out | standalone Blender | No |
| OPT-IN tier | A2F-3D driver (NVIDIA Open Model License -- read clauses) | own TRT/CUDA vendor runtime sidecar; `.engine` pre-compiled at setup, never at render [PASS-3D #4] | Own toolchain (TRT), probe-gated |
| DEFERRED | Hunyuan3D (NC + EU/UK/KR + flash_attn + custom_rasterizer source builds), TRELLIS (diff-gaussian-rasterization LIKELY-FAILS sm_120 + spconv), SuGaR | source builds [DEP] | **Yes -- the toolkit gate lives here** |

Sidecar venv spec [FORK-RT]: pin the sidecar Python to the version the TripoSG wheel set
supports (expect py3.10/3.11, NOT py3.13); wheel acquisition is a SEPARATE online step with
SHA-256 pins; the venv build itself runs offline `--no-index --find-links <wheelhouse>
--only-binary=:all:` and FAILS if any sdist build is attempted. Self-test asserts the SDPA
attention backend and that no `flash_attn` import is attempted. This reconciles [GOFWD]
Phase 4: the "ONE shared TOOLKIT" is deferred with the engines that need it; venvs stay
per-sidecar regardless. Licenses: TripoSG MIT, Step1X-3D Apache-2.0, Rhubarb MIT, ICT-FaceKit
MIT (DATA license audit required, section 5), Blender GPL (subprocess-isolated = fine),
A2F-3D NVOML (operator legal read), Hunyuan3D NC+territory (operator sign-off gate in
`video_profiles.yaml` before any weights download) [DEP, 3DC].

## 5. Assets -- fetchable vs operator-supplied (the honest ledger)

| Asset | Status | How |
|---|---|---|
| **ARKit-52 template** (`OTR_B_ARKIT_TEMPLATE_NPZ`) | **FETCHABLE** (was assumed operator-blocked [B-SHIP]) | Derive from **USC ICT-FaceKit** (MIT; ARKit-named expression shapes with `_L/_R` splits). Conversion script merges ONLY the canonically-singular shapes (browInnerUp, cheekPuff, ...) via an explicit name map (classes: copy / merge_to_unilateral / split_preserved / drop) and PRESERVES the canonically-asymmetric L/R shapes; validates EXACTLY 52 deltas, identical topology, neutral-relative, documented dtype/scale/index-base; pins `ARKIT_TEMPLATE_HASH` (SHA-256) the probe checks [GOFWD Phase 5, FORK-RT]. GATES: audit the model-DATA license (not just repo code license); pin commit + per-asset SHA-256; converter fail-closed on `mouth_idx` derivation + manifoldness; jawOpen/mouthClose/mouthFunnel deformation smoke before acceptance. REJECTED sources: MetaHuman extracts (Epic license), paid add-ons [FORK-RT]. |
| **~25-mesh keystone corpus** (`OTR_B_MESH_DIR`) | **GENERATED by the pipeline itself** (not operator-sourced) | Produced by the TripoSG sidecar from ~25 real Flux portraits (include adversarial cases: glasses, hair-over-face, profile, open mouth, low-res [PS-3D]). This is why T3 precedes the BINDING keystone verdict (section 6). |
| TripoSG / Step1X-3D weights | Fetchable (HF, ungated, MIT/Apache) into `C:\ComfyUI-Models` | After S-3D-0; SHA-256 pinned in `dependency_manifest` |
| Rhubarb binary + mapping table | Fetchable (MIT) / hand-authored in-repo | Mapping table is a reviewable `rhubarb_to_arkit.json` with golden curve plot |
| Blender (standalone) | Fetchable installer | Operator approves the install location; subprocess only |
| **cu128 TOOLKIT** (ninja + nvcc 12.8 + VS2022) | **OPERATOR-BLOCKED** (machine-level install) -- but DEFERRED out of v1 | Only needed for Hunyuan3D/TRELLIS/SuGaR/flash_attn source builds [B-SHIP, FORK-RT] |
| A2F-3D weights + TRT stack | Operator-gated (NVOML legal read + vendor runtime install) | Opt-in tier only |
| Hunyuan3D weights | Operator-gated (NC + EU/UK/KR sign-off recorded) [DEP] | Deferred engine |

## 6. Prerequisite spikes (gate before any adapter code)

Keystone-first order (pre-mortem reorder carried; manifold pre-screen probe_b stays first-in-
harness) with the corpus chicken-and-egg resolved honestly [FORK-RT]: a clean-parametric-head
corpus would FALSE-PASS the keystone (its whole difficulty is generative "marching-cube soup"),
so the binding verdict waits for real Slot-1 meshes.

| Spike | What | BINDING pass/fail |
|---|---|---|
| **S-3D-0 (NEW -- gates the lane)** | TripoSG sidecar venv builds OFFLINE from prebuilt wheels only (`--only-binary=:all:`); self-test = import clean + SDPA asserted + no flash_attn + one portrait->GLB + manifold check + mesh-gen-spawn VRAM <= 14000 | ANY attempted source build or compiled-ext failure -> NO-GO for the no-compile lane -> the cu128 TOOLKIT re-enters as a v1 prerequisite (operator install) before anything proceeds |
| **T1 template** | Fetch ICT-FaceKit (pinned commit), run the converter (section 5 gates), pin `ARKIT_TEMPLATE_HASH` | Converter fail-closed: exactly 52 deltas + valid `mouth_idx` + deformation smoke, else NO template = whole phase stays parked |
| **T2a wrap smoke (NON-BINDING)** | Existing `probe_c_arkit_wrap.py` synthetic harness + the new real template | Labelled NON-BINDING explicitly [B-SHIP precedent]; catches harness/template breakage in minutes |
| **T3 corpus** | TripoSG sidecar generates the 25-mesh corpus from real portraits (incl. adversarial) into `OTR_B_MESH_DIR` | 25 meshes on disk, manifold pre-screen (probe_b) run on each |
| **T2b KEYSTONE (BINDING)** | `probe_c` mesh -> ARKit-52 WRAP on the TripoSG corpus; wrap = automated headless deformation-transfer/shape-key bake, TIMEBOX ~1 week [PASS-3D] | **GO = failures <= 4/25; NO-GO = >= 5/25** (ambiguity in prior docs pinned here [FORK-RT]). NO-GO -> HuMo-2D stays the character path; `character_3d` stays deferred [GOFWD Phase 5] |
| **T4 driver + alpha** | Rhubarb -> mapping table -> curves on the wrapped mesh -> Blender alpha render -> `OTR_SilentComposite` flatten -> mux; alpha-composite end-to-end (old S5) + render-spawn + wrap-spawn VRAM measured | One real talking alpha clip, onset within +-1 frame (video trimmed, NEVER audio -- BUG-102/V-1), audio stream hash == master, yuv420p after flatten, straight-vs-premultiplied alpha documented + golden-tested [PASS-3D] |

Old B-dep (build Hunyuan3D+TRELLIS+A2F-3D venvs first) is RETIRED as the first gate -- those
probes move behind the opt-in/deferred tiers they belong to [FORK-RT cut]. Old S4 (LTX@16GB
background) is unchanged where the composite needs a background engine; `still_kenburns`
remains the background fallback [v1.4-III].

## 7. Build waves + test gates

After ALL spikes green + operator go (no coding before that [SUB-B]):

- **W7 / B1 -- `triposg_talk` adapter FULL** (mesh-once prepare + wrap + Rhubarb driver +
  Blender alpha render + mesh cache), sidecar-isolated, spawn-per-execution-group; extend
  `OTR_SilentComposite` for the 2.5D alpha overlay (alpha codec `webm/vp9|mov/prores4444` iff
  `alpha != none`, FLATTEN to yuv420p at the composite output; alpha never reaches RTXUpscale);
  conditional-background via A's AS-2 resolver-prune (never a new DAG edge); role gating via
  shared `role_compat.py` (AS-1); `character_3d` group `depends_on:[flux_portrait]` (BUG-086).
  **B1 REOPENS SilentComposite -> re-run A's W2 `widget_vector_exact` golden + the audio-green
  migration gate as the MUST-PASS exit** [v1.4-III, SUB-B, PASS-3D].
- **W7.5/W8 / B2 -- engine #2** = `step1x3d_talk` GEOMETRY-ONLY (skip the SDXL texture stage;
  vertex-color/stylized shading acknowledged), only AFTER the first path renders a real episode
  clip [3DC, FORK-RT]. Both engines selectable = the no-single-vendor invariant satisfied at
  ship.
- Fallback ladder v1: `triposg_talk -> humo -> still_kenburns`, every hop LOUD (log swap +
  ledger restamp at same `video_revision`, never silent). No CPU fallback for `character_3d`
  (GPU sidecar only; fail loud -> humo); the MediaPipe-extrusion never-fail mesh floor is
  DEFERRED to the playground phase with an explicit contract amendment if adopted [FORK-RT].
  When the chain goes live, update `SYNTH_FALLBACKS` + `EXPECTED_OOM_TRAIL` in
  `render_driver.py` + `scripts/otr_video_soak.py` to the REAL engine ids and re-run/re-certify
  the A-S7.5 soak (or the sanctioned lighter B-sidecar smoke -- operator's call) [GOFWD Phase 5].

**Output-correctness gates (all carried):** `test_audio_byte_identical` GREEN at every step
(demuxed elementary stream hash, mux-LAST `-c:a copy`, no `-shortest`); per-spawn VRAM <= 14000
under the shared lease, never co-resident; determinism v1 = identical `request_hash` across two
spawns (`binds_seed=False`; pixel determinism = v2); cold-import clean (V-12) + engines visible
in the static dropdown (V-6); mesh-cache hit on line 2; non-blank assertion (>1% non-zero px +
vertex count > N); `assert_usable` fail-closed in explicit order flag -> sidecar-venv ->
template -> mesh assets, re-checked EVERY attempt (no cached "missing") [GOFWD Phase 3/5,
v1.4-III]. Driver contract spec: `rhubarb_to_arkit.json` mapping + Savitzky-Golay smoothing +
silence/onset policy + deterministic procedural blink/eye-dart layer (characters look dead
without it [PS-3D]); curve cache key = audio hash + driver version + mapping-table hash + fps +
onset policy [FORK-RT].

## 8. Open forks -- RESOLVED (recommendation each; [FORK-RT] = this consolidation's 3-model roundtable, GPT-5.5 + Gemini 3.1 Pro + Grok 4.3, $0.23, grounded)

1. **Mesh-gen model: TripoSG first** (then Step1X-3D geometry-only). Hunyuan3D/TRELLIS
   deferred behind license/territory + the toolkit + their build probes. Basis: pass3DC 4/4
   unanimous + [GOFWD] "do not hard-commit the pair" + [FORK-RT]. NOT re-litigated -- the prior
   verdict stands; the only new element is making the first slice single-engine.
2. **cu128 toolchain: DEFER the toolkit; v1 = the no-compile lane**, conditional on S-3D-0.
   This unblocks the phase from the probe_a operator blocker. If S-3D-0 fails, the toolkit
   (operator machine-level install) is honestly back on the critical path. [FORK-RT MUST-FIX 1]
3. **ARKit-52 template: fetchable** (ICT-FaceKit MIT + converter), NOT operator-supplied --
   subject to the data-license audit + fail-closed converter gates. Operator involvement drops
   from "supply a template" to "approve the license read". [FORK-RT MUST-FIX 2/10]
4. **Ordering: keystone-first, corpus-honest**: T1 -> T2a(non-binding) -> T3 -> T2b(BINDING) ->
   T4. Slot-2a curves decoupled (parallel with Slot 1). Slot1 -> 1.5 -> WRAP -> 2b ordering
   confirmed; the only correction was 2a's false dependency. [FORK-RT MUST-FIX 3/6]
5. **Driver: Rhubarb default** (deterministic, CPU, never-OOM); A2F-3D opt-in tier behind its
   own TRT + NVOML probes. ([GOFWD] supersession confirmed; the [SUB-B] "A2F-3D = v1 driver"
   framing is SUPERSEDED.) [FORK-RT MUST-FIX 5]

**Left for the operator (decisions, not blockers):** (a) A2F-3D NVOML acceptability for this
build; (b) Hunyuan3D NC+territory sign-off if ever wanted; (c) approve the Blender install +
the one-time online wheel/asset acquisition step; (d) green-light to start S-3D-0.

## 9. Definition of done (the 3D phase)

A live `triposg_talk` clip composited into a REAL episode: a generated character mesh, wrapped
to 52 ARKit keys, lip-synced by Rhubarb curves to a frozen audio line it did not choose,
rendered as alpha, composited + flattened, master audio byte-identical (demuxed stream hash),
all spawns <= 14.0 GB under the shared lease, render-twice `request_hash`-deterministic,
mesh-cache hit on the second line, fail-closed to `humo` (LOUD, ledger-restamped) when any
asset/venv is absent, full suite + Bug Bible green, and the soak (or sanctioned lighter smoke)
re-certified with the real fallback trail. A second engine (`step1x3d_talk`) selectable =
B-parity; the playground (Airlock, gauntlet, new roles) opens only after that.
