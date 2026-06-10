# OTR 3D Toolkit -- SUBAGENT SPRINT PLAN (dependency-ordered tickets) -- 2026-06-09

> Companion to `3D_TOOLKIT_PLAN.md` (the CONTRACT -- every ticket cites its sections; on any
> conflict the plan wins). PLANNING artifact: no code until the operator green-lights. Status
> of record = `SPRINT_STATUS.json` (same folder) -- **every ticket completion updates it in the
> same commit** (the live tracker artifact reads it; see Status discipline). Handoffs between
> coding sessions use the `otr-3d-handoff` skill.

**Invariants (every ticket):** audio frozen + mux-LAST + byte-identical; single resident heavy
engine (14.5 machine / 14.0 3D sidecar); protected cu130 main venv never touched; V-6 static
dropdowns / V-11 no new widgets / V-12 cold-import; fail-closed, every fallback LOUD; UTF-8 no
BOM; SFW (never the d-word); run Bug Bible + core + dropdown regression after each change;
commit per ticket, do NOT push unprompted (except docs).

## Dependency DAG
```
LANE R (repo, CPU, NO-REGRET -- may start BEFORE the green-light)
  R1 scaffold/migration ──► R3 builders ──► R4 directory-clips ──► R5 caches/ledger
        │  (R2 ImageDirector is independent -- run parallel to R1..R5)
        ▼
LANE T (template, CPU)          LANE G (GPU spikes, operator-gated)
  T1 converter ──► T2a smoke      G0 S-3D-0 ──► G1 weights+corpus(T3)
        └────────────┬───────────────┘
                     ▼
            G2 T2b BINDING KEYSTONE  ──►  G3 T4 driver+alpha+LOOK
                     ▼
LANE W (live build; needs R*+T*+G* green + operator go)
  W1 triposg_talk live ──► W2 story wiring ──► W3 soak + v1-usable DoD
```
- **Join points:** G2 needs T1+G1. W1 needs ALL of R, T, G green.
- **Parallel day one:** R1..R5, R2, T1 -- zero GPU; ~60% of the sprint runs before/without G0.
- **Collision guard:** only ONE lane edits shared modules at a time. R2 owns
  `otr_image_director.py` + `schemas.py` capability field; R1/R3..R5 own
  `eng_character_3d.py`/`render_driver.py`/`otr_video_soak.py`/resolver wiring + tests; T1 owns
  `scripts/_otr_b_spikes/` + the converter; G* runs probes (no production edits); W* opens only
  after R/T/G freeze. Two subagents never hold the same file.

## Status discipline (what makes the tracker autonomous)
Every ticket ends with: (1) tests green for its gate; (2) edit
`docs/2026-06-09-3d-toolkit/SPRINT_STATUS.json` -- set the ticket `status`
(`pending|in_progress|done|blocked|nogo`), `note` (<=140 chars), `updated` (ISO date),
top-level `updated`/`updated_by`, and `next_action`; (3) commit code+status TOGETHER. The live
artifact renders this file -- no human tracker edits needed.

---

## LANE R -- repo hardening (CPU, no-regret; pays off even on an S-3D-0 NO-GO)

### R1 -- W7-pre migration slice  [plan 7.0]
- **Depends:** none. **Blocks:** R3, W1.
- **Goal:** ADD `TripoSGTalkEngine` (`triposg_talk`, `OTR_ENABLE_TRIPOSG_TALK`,
  `OTR_TRIPOSG_SIDECAR_PYTHON`, `requires_mesh_portrait=True`, `fallback_engine="humo"`) as a
  third DARK adapter with its OWN assert helper (do NOT edit `_assert_usable_3d`; no mesh-dir
  gate -- plan 7.1); keep hunyuan/trellis dark. Update `SYNTH_FALLBACKS` / `ENGINE_FAMILY` /
  `OOM_ENGINES` / `EXPECTED_OOM_TRAIL` (BOTH copies; first hop `triposg_talk->humo (oom)`) /
  `_CHAR3D` / `assert_soak_ok` carryover `input_oom_engine` / 3 test files, ONE commit; comment
  the 4-hop-trail-vs-3-decision semantics (preserve, don't "fix").
- **Files:** `nodes/_otr_video_engines/eng_character_3d.py`, `render_driver.py`,
  `scripts/otr_video_soak.py`, `tests/test_video_character_3d.py`,
  `tests/test_video_render_driver.py`, `tests/test_video_soak_fixture.py`.
- **Gate:** targeted suites + fallback-chain unit test (`triposg_talk->humo->humo_1.7B->
  latentsync->still_kenburns`, terminating); cold-import; ComfyUI RESTART then V-6 dropdown
  shows dark `triposg_talk`; soak fixture green. **Do-not-touch:** audio nodes, shared modules.

### R2 -- ImageDirector + capability hardening  [plan section 3; INDEPENDENT of R1]
- **Depends:** none. **Blocks:** W2.
- **Goal:** `video_policy_json` -> `required` dict PRESERVING `{"forceInput": True}`; raise on
  empty/malformed; `enforce_3d_granularity_lock` RAISES on 3D+per_beat (no coercion); add
  `requires_mesh_portrait: bool=False` as a REAL field to `AdapterDescriptor` +
  `VideoProfileRow` (extra="forbid") + migrate schema/registry tests same commit;
  `_is_3d_engine` reads the capability; unknown custom engines fail closed; downstream
  ShotLock/Dispatcher HALT on (3D + per_beat).
- **Files:** `nodes/otr_image_director.py`, `nodes/_otr_video_engines/schemas.py`, their tests.
- **Gate:** new fail-closed tests (empty policy raises; per_beat+3D raises; custom-engine
  missing capability raises); widget_vector/forceInput gates green (the required-move must NOT
  spawn a widget). **Do-not-touch:** the granularity-lock SEMANTICS (per_object stays).

### R3 -- request-builder migration  [plan 7.0/7.2]
- **Depends:** R1. **Blocks:** R4, W1.
- **Goal:** `build_request()`/`build_request_from_shot()` emit `VideoRequest`-valid payloads
  (add role/family_hint/profile_id; drop `init_w/init_h`; `dur_s`->`timing.target_duration_s`;
  `char_id`->`conditioning_refs`); adapter-boundary `model_validate` (defense in depth);
  character_3d REQUIRES the canonical 16:9 canvas (reject 480x832 default); down-chain rebuild:
  re-validate per fallback candidate, loud-skip families whose required inputs can't be met.
- **Gate:** CPU test `VideoRequest.model_validate(build_request_from_shot(...character_3d...))`
  passes; missing audio_ref/init_image fails closed; existing humo/ltx builder tests stay green.

### R4 -- directory-clip plumbing  [plan 7.2]
- **Depends:** R3. **Blocks:** W1.
- **Goal:** `CanonicalClip(type="directory", alpha="straight")` validator (frame_count == files
  on disk == timing.target_frame_count; never trust defaults); `_clip_summary`/
  `build_clip_manifest`/soak assertions treat directory as "dir + exactly N sorted nonzero
  frames"; `OTR_SilentComposite` directory input (sorted frames -> overlay -> flatten yuv420p)
  + CPU/golden straight-alpha test (checkerboard seam); synthetic 52-key GLB fixture; CPU
  directory-clip fixture through manifest -> composite -> mux.
- **Gate:** the CPU golden; SilentComposite OPAQUE path byte-identical (A's W2
  widget_vector_exact + audio-green = MUST-PASS); vp9/prores branches absent.

### R5 -- cache + ledger contracts  [plan 7.3]
- **Depends:** R3 (threading line_id). **Blocks:** W1.
- **Goal:** slice key += master content hash (+ar/ac/slicer ver; signature gains `master_hash`);
  separate 16 kHz driver extraction = DOWNSAMPLE of the provided 44.1 kHz slice (HuMo slicer
  unchanged); curve key = slice key + line_id + fps + driver ver + mapping hash + onset policy;
  artifact paths via `cache_keys`/`CanonicalClip.qc` (ShotRow extra="forbid"); resolver-prune
  WIRED into the fallback restamp transaction + test (3D->humo prunes the background group);
  mesh-cache atomic publish (versioned dir + pointer-file flip, lockfile, stale-timeout);
  on-disk ledger transaction (temp+fsync+replace); audio-ledger byte-equality test incl. every
  fallback variant; portrait-hash-change regenerates mesh test.
- **Gate:** all named tests green; `test_audio_byte_identical` green.

## LANE T -- template (CPU; parallel with everything)

### T1 -- ICT-FaceKit -> ARKit-52 converter  [plan section 5 / spike T1]
- **Depends:** none (one online fetch, pinned commit + SHA-256). **Blocks:** G2.
- **Goal:** `facekit_to_arkit.json` name map (copy/merge_to_unilateral/split_preserved/drop;
  merge ONLY canonically-singular shapes); converter -> `OTR_B_ARKIT_TEMPLATE_NPZ`
  (verts/faces/mouth_idx + delta_<name> x52 matching `_b_harness.ARKIT_52`) + machine-readable
  report; HARD-FAIL on missing shapes (no zero-delta placeholders); semantic checks (coords/
  scale, L/R sign, delta-norm ranges, golden thumbnails); 52-delta property test via
  `wrap_topology_check`; jawOpen/mouthClose/mouthFunnel smoke; pin `ARKIT_TEMPLATE_HASH`;
  model-DATA license audit recorded.
- **Gate:** converter fail-closed proven; then **T2a** = existing `probe_c` synthetic harness
  with the REAL template, labelled NON-BINDING. **Do-not-touch:** `_b_harness.keystone_gate`
  (already strict).

## LANE G -- GPU spikes (operator green-light required to START)

### G0 -- S-3D-0 no-compile probe  [plan section 6; GATES THE LANE]
- **Depends:** operator go. **Blocks:** G1.
- **Goal:** TripoSG cu128 sidecar venv OFFLINE from a pinned wheelhouse
  (`--no-index --find-links --only-binary=:all:`, `build_sidecar_env()` sanitizer; sidecar
  python per the wheel set, expect py3.10/3.11); self-test: import clean + SDPA asserted + no
  flash_attn + portrait->GLB + manifold + mesh-spawn VRAM <= 14000. On ext failure: CPU
  marching-cubes pre-step (`skimage.measure.marching_cubes`, then PyMCubes binary wheel; any
  source patch = NO). RECORD the exact outcome in SPRINT_STATUS + the probe log.
- **Gate:** GO -> G1. **Hard NO-GO -> STOP THE LANE; OPERATOR DECIDES** (HuMo-2D-only v1 vs
  approve the cu128 toolkit install). Update README "cheap keystone first" wording here.

### G1 -- weights + T3 corpus
- **Depends:** G0, (mesh-pack portraits from the C engine -- M4 handshake prompts, plan 2.5).
- **Goal:** fetch TripoSG weights (SHA-256 pinned, `C:\ComfyUI-Models`); generate the 25-mesh
  corpus from MESH-PACK portraits + adversarial + period-style cases, ONE spawn, per-portrait
  outcomes recorded; probe_b manifold pre-screen + self-intersection + normal-orientation per
  mesh. **Blocks:** G2.

### G2 -- T2b BINDING KEYSTONE  [plan section 6]
- **Depends:** T1 + G1. **Blocks:** G3, W1.
- **Goal:** `probe_c` wrap on the corpus; deformation-transfer, TIMEBOX ~1 week.
- **Gate:** GO = failures <= 4/25 (5/25 = NO-GO, `keystone_gate` is already strict). NO-GO ->
  HuMo-2D stays; character_3d deferred; record + stop.

### G3 -- T4 driver + alpha + LOOK  [plan section 6/7.4]
- **Depends:** G2 (reuse a T2b-wrapped mesh -- no fresh mesh-gen). **Blocks:** W1.
- **Goal:** Rhubarb (with --dialogFile script text) -> mapping -> curves -> Blender alpha
  (per-group invocation, determinism pins, canvas-native 16:9) -> SilentComposite flatten ->
  mux; all spawn peaks measured under the lease; LOOK gate (portrait bake in-cone /
  auto-stylize; material QC; qc metrics EMITTED); story-integration QC (real ledger beat);
  operator visual sign-off boolean.
- **Gate:** clip frame-count == ledger sum; audio stream hash == master; T4 report complete.

## LANE W -- live build (after R+T+G green AND operator go)

### W1 -- `triposg_talk` live forward  [plan 7.5]
- **Depends:** R1,R3,R4,R5 + G2,G3. **Goal:** prepare (ONE spawn, ALL 3D characters,
  mesh-once cache) + wrap + curves (episode-batch after audio_done) + Blender render + alpha
  directory handoff + lease bracketing + NVML floor between spawns + executor-thread fallback
  discipline + cancellation polling/taskkill. **Gate:** plan 7.5 gates; mesh-cache hit line 2.
### W2 -- story wiring  [plan 2.5]
- **Depends:** W1 + R2. **Goal:** 3D pack generation in the C image phase (M4 handshake
  prompts, purpose-tagged, default-ON for 3D-assigned characters); shot grammar
  (role+mood->framing, seeded per true-randomization); delivery-vector idle modulation;
  dialogFile wiring; plate selection from C stills. **Gate:** story-integration QC on a real
  episode beat.
### W3 -- soak + v1-usable  [plan section 9]
- **Depends:** W1+W2. **Goal:** re-run A-S7.5 soak (or sanctioned lighter B-sidecar smoke)
  with the real trail; full suite + Bug Bible; the v1-usable DoD checklist, item by item.
  **Exit:** v1-usable DECLARED in SPRINT_STATUS; B-parity (step1x3d_talk) spins out as its own
  later sprint.

## Window mapping (suggested; one gate per session)
CW-3D-1 = R1+R3 | CW-3D-2 = R2 | CW-3D-3 = R4 | CW-3D-4 = R5 | CW-3D-5 = T1+T2a |
CW-3D-6 = G0+G1 (operator present) | CW-3D-7 = G2 | CW-3D-8 = G3 | CW-3D-9 = W1 |
CW-3D-10 = W2 | CW-3D-11 = W3. Each window: start with the otr-3d-handoff kickoff prompt,
end at its gate + commit (code+status together) + regenerate the handoff.
