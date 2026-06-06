# Delivery-vector wiring -- inputs/outputs QA (pass00)

A CODE review of a small, finished change. The per-line delivery (emotion) vector
is now wired through dispatch. Full suite is green (subset 100/0 so far). Review
the ACTUAL grounding files for correctness of the INPUTS (where the vector comes
from) and OUTPUTS (how each engine projects it). Be concrete: cite file + failure.

## What changed (in `nodes/_otr_voice_node_common.py::_render_per_line`)
Previously `prep(text, None)` and `adapter.generate_voice(prepared, voice_ref,
None, seed)` -- the delivery vector was hardcoded None, so indextts2 emo-vector
and chatterbox exaggeration were always flat (defaults). Now, per line:
- `_delivery_on = os.getenv("OTR_DELIVERY_VECTOR", "1") != "0"` (kill-switch;
  "0" reproduces the old flat renders).
- vector = stamped `ln["delivery"]["emotion_vector"]` if present, else
  `deterministic_delivery_vector(text, scene_tension)` (pure, no RNG -> C7).
- passed to BOTH `prep(text, delivery_vector)` and
  `adapter.generate_voice(prepared, voice_ref, delivery_vector, seed)`.
- the bark MISSING-REF fallback path still passes None (bark ignores it anyway).

## INPUTS to verify (against the real files)
- Source: `nodes/_otr_delivery_vector.py` -- `deterministic_delivery_vector(text,
  scene_tension)` returns an 8-dim `{emotion: 0..1}` in the canonical `EMOTIONS`
  order. `stamp_delivery_vectors` exists but is NOT called anywhere (so the derive
  path is the live one). Tension is read as `ln["scene_tension"]` ->
  `ln["tension"]` -> 0.0.
- Is the stamped-vs-derived precedence correct + safe (dict shape guarded)?
- Could `delivery_vector` ever be a non-dict that breaks a downstream projection?

## OUTPUTS to verify
- `eng_indextts2.py::emo_list(dv)` -> list in EMOTIONS order -> worker
  `emo_vector` (worker only applies it when any value != 0). Order match?
- `eng_chatterbox.py::_project(dv)` -> single `exaggeration` float. Sane range +
  uses `dv["calm"]`?
- `eng_bark.py` / `eng_kokoro.py` `prepare_text` + `generate_voice` IGNORE
  `delivery_vector` -> byte-identical baseline. CONFIRM they ignore it.
- dia ignores it (transcript/audio_prompt only) -> unaffected.

## Invariants (reject any "fix" that breaks one)
C7 determinism (the derive is pure; engine_seed unchanged); byte-identical
bark + kokoro + dia; PD1 always-renders; `prep` must stay text-only (no audio
direction injected into words); OTR_DELIVERY_VECTOR=0 == exactly the old None path;
C-5 import-time clean (the delivery import is lazy, inside the method).

## Specifically hunt
1. EMOTIONS-order mismatch between `_otr_delivery_vector.EMOTIONS` and what
   indextts2's `emo_list` / the worker `emo_vector` expects.
2. chatterbox `_project` projection sanity (does flat/neutral map to a reasonable
   exaggeration; any divide/keyerror if dv lacks "calm").
3. Does passing dv to `prep` change ANY engine's prepared text? (all prepare_text
   ignore it -- confirm.)
4. Determinism: is the derived vector stable run-to-run; does it perturb the
   engine_seed or RNG scope? (it must not.)
5. The kill-switch path (OTR_DELIVERY_VECTOR=0): is it EXACTLY the prior behavior
   (delivery_vector=None everywhere)?
6. Any exception path in the derive that could break PD1 (it is wrapped, but
   verify).

## Out of scope (do NOT raise)
Wiring `stamp_delivery_vectors` into a freeze node (the derive path covers it);
delivery PROFILES (`_otr_delivery_profiles`) as a separate projection layer;
cross-platform path handling; the chatterbox/Dia GPU verify-at-build items.
