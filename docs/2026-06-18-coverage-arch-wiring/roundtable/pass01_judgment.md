# Coverage-arch wiring -- judgment (Claude as judge)

NOTE: the live panel stalled with no output (~6.5 min, empty logs) -- the known
roundtable-launcher stall (recorded in GO_FORWARD open tickets + the 3D-input
entry). Rather than grind a retry, Claude judged the A-vs-B fork directly against
the grounded engine inventory (the roundtable's whole point is the grounding, not
the model count). The pass01 coverage-arch synthesis (gemini+grok, earlier today)
already converged on the capability approach; this confirms + picks the wiring.

## Decision: Hybrid-A (capability with a BASE DEFAULT) -- NOT pure convention (B)

The operator wants two things that pull apart: (1) "all video/3D accept whatever
image is selected, one place, no per-model whitelist" and (2) "maybe just always
look at a folder, no gatekeeper". Pure convention (B) was REJECTED on three
grounded cons:
- It breaks the shipped accessible floor "all-procedural episode invokes NO image
  model" (e8a00e9 / b2f07e0): always rendering a flux still per beat forces an
  image model on users who selected none and spends a heavy render nothing uses.
- 3D's `still_kind=mesh_portrait` + the per_beat mesh-rebuild lock CANNOT be pure
  convention -- something must still declare "this lane wants a mesh portrait, no
  per-beat rebuild". A gatekeeper-shaped capability is unavoidable for 3D.
- "Always read a fixed path" reintroduces the fixed-filename overwrite risk (S7)
  and deletes the LOUD-skip signal.

Hybrid-A gives the operator (1) AND the *feel* of (2): the capability lives as a
BASE-CLASS DEFAULT (`accepts_still=True` on MotionEngineBase), so every real video
lane -- present and future -- accepts the selected image automatically. A new
engine inherits the default and "just works": no per-model whitelist, one place.
Only the genuine opt-outs declare `accepts_still=False` (the audio-only music lane,
the pure procedural floors). That preserves the accessible floor + the 3D lock.

## Wiring (the slice that ships now -- small, reversible)
1. `MotionEngineBase` (motion_common.py): add class attrs
   `accepts_still=True`, `still_input_name="init_image"`, `still_kind="init_image"`.
   -> ltx_video, humo*, wan*, ltx_av_talk inherit True (ltx_video starts consuming
   the selected still -> the flux2-on-LTX fix).
2. `LtxAvMusicEngine` (eng_ltx_av.py): `accepts_still=False` (audio-only, no still).
   `VisualizerEngine` (eng_visualizer.py): `accepts_still=False` (explicit floor).
3. Dispatcher (otr_image_gen_dispatcher.py): ONE helper
   `engine_consumes_still(eng)` = explicit `accepts_still` if declared, else legacy
   `"init_image" in required_inputs` (dual-read; cheap floors unchanged).
   `_still_needed_for_role` calls it; the unknown-engine path logs LOUD + returns
   True (fix the bare silent `except: return True`).
4. 3D talkers (eng_character_3d.py): add `accepts_still=True` + `still_kind=
   "mesh_portrait"` (forward-compat; the granularity lock keeps reading
   requires_mesh_portrait for now -- full unification is a follow-up, Decision 5).

## Deferred (follow-up, not needed for the deliverable)
- Decision 3 central `image_engines.registry.usable(name, role)` consolidation.
- Decision 5 delete `requires_mesh_portrait` once `still_kind` is equivalent.
These are additive; the slice above is behavior-correct without them.

## Invariants preserved
No silent fallback (LOUD except); role_compat still the role filter (unchanged);
model-agnostic; single-resident unchanged (metadata only); cold-import clean (plain
attrs); workflow JSON untouched (no node/widget change); UTF-8 no BOM.
