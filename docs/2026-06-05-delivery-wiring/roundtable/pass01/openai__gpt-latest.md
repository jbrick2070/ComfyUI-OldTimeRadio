<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Derived live path is mostly correct, but stamped vectors are accepted without numeric/range validation and can crash or send invalid values to expressive engines.

MUST-FIX BEFORE BUILD:
1. [INPUTS to verify] `_otr_voice_node_common.py::_render_per_line` accepts any non-empty stamped dict as `delivery_vector` without shape/value guarding:
   - Failure: `ln["delivery"]["emotion_vector"] = {"calm": "bad"}` reaches `eng_chatterbox.py::_project`, which does `float(delivery_vector.get("calm", 0.5))` and raises `ValueError`.
   - Failure: `{"happy": "bad"}` reaches `eng_indextts2.py::emo_list`, which does `float(dv.get(e, 0.0))` and raises.
   - Failure: `{"happy": 99, "calm": -4}` reaches `emo_list` unclamped and sends out-of-contract values to the worker.
   - Concrete fix: add a central sanitizer before accepting `_stamped`: require dict, canonical `EMOTIONS` keys only or default missing keys to `0.0`, coerce numeric finite values, clamp to `0.0..1.0`, round, and if coercion fails discard stamped vector and derive instead. Also use the sanitized form for derived vectors defensively.

2. [OUTPUTS to verify / eng_indextts2.py] `IndexTTS2Engine.emo_list()` does not enforce the documented `{emotion: 0..1}` contract. It imports `EMOTIONS` so order matches `_otr_delivery_vector.EMOTIONS`, but values are only `float()` + `round()`.
   - Failure: direct or unsanitized caller can pass negative, >1, NaN-like, or non-numeric values; non-numeric raises, out-of-range is forwarded as `emo_vector`.
   - Concrete fix: clamp each value to `0.0..1.0`, reject/non-finite -> `0.0` or raise a named local validation error before worker protocol write. Prefer sharing the same sanitizer with dispatch.

3. [OUTPUTS to verify / eng_chatterbox.py] `ChatterboxEngine._project()` is not robust to malformed stamped vectors.
   - Failure: truthy non-dict passed directly to adapter, e.g. list, raises `AttributeError` on `.get`; malformed `"calm"` raises `ValueError`.
   - Live `_render_per_line` currently prevents non-dict, but not bad dict values.
   - Concrete fix: in `_project`, require `isinstance(delivery_vector, dict)`, coerce `calm` under `try`, clamp finite `calm` to `0.0..1.0`, else use neutral default.

SHOULD-FIX:
1. [Invariants / kill-switch] `_otr_voice_node_common.py::_render_per_line` imports `deterministic_delivery_vector` unconditionally before checking `OTR_DELIVERY_VECTOR`.
   - Failure mode: `OTR_DELIVERY_VECTOR=0` still depends on the new delivery module importing successfully, so it is not strictly the old path if that module is broken/missing.
   - Concrete fix: move `from ._otr_delivery_vector import deterministic_delivery_vector` inside `if _delivery_on:` before first derive use.

2. [Specifically hunt #1] `eng_indextts2.py::emo_list()` order matches `_otr_delivery_vector.EMOTIONS` because it imports that tuple directly. However the worker-side expected order is not shown.
   - verify: `scripts/_otr_indextts2_worker.py` must pass the list to IndexTTS2 in the same order: `happy, angry, sad, afraid, disgusted, melancholic, surprised, calm`.
   - Concrete fix if mismatch: either reorder in `emo_list()` or make the worker consume a named dict instead of positional list.

3. [OUTPUTS to verify / dia] No `eng_dia.py` grounding excerpt is provided.
   - verify: Dia’s `prepare_text`/`generate_voice` signatures accept the extra `delivery_vector` argument if routed through this per-line path, and Dia does not use it in transcript/audio-prompt construction.
   - Concrete fix if not true: add ignored `delivery_vector=None` parameters or keep Dia off this dispatch path.

4. [OUTPUTS to verify / chatterbox projection sanity] `eng_chatterbox.py::_project()` maps `None` to `0.5`, but a derived neutral/calm vector with `calm=1.0` maps to `0.3`.
   - This is not a crash, but it means enabling delivery changes neutral Chatterbox lines from the old default exaggeration `0.5` to `0.3`.
   - Concrete fix if old-neutral parity is desired: make `calm=1.0` project to `0.5`, then increase exaggeration as calm falls.

OPTIONAL / NICE-TO-HAVE:
1. `_otr_delivery_vector.py::stamp_delivery_vectors()` does `float(tension)` without a try/except. It is not live per the stated scope, but if wired later it can fail on non-numeric ledger tension. Mirror the guarded derive path in `_render_per_line`.
2. Add small tests for malformed stamped vectors: non-dict, missing `calm`, string value, NaN/inf, negative, >1.

CUT THESE (over-engineering):
1. None. The delivery derivation and adapter projections are small; the main issue is missing validation, not excess structure.