<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes -- the core wiring is correct but there is a latent crash risk from non-numeric values in a stamped emotion_vector that could bypass the dict shape guard and break the render.

MUST-FIX BEFORE BUILD:
1. [eng_indextts2.py::emo_list + eng_chatterbox.py::_project] Neither function validates that the values inside `delivery_vector` are numeric. The dict shape guard in `_render_per_line` accepts any dict, but `float(dv.get(e))` will raise `ValueError` (or `TypeError` for `None`) on a non‑numeric value, crashing the render. Both should handle conversion safely (e.g., `try: ... except (ValueError,TypeError): default`) so PD1 is not violated by a malformed stamped vector.
   Concrete fix: wrap the per‑emotion `float()` in a helper that falls back to `0.0`, and in `_project` use a similar safe‑float for `calm`.

SHOULD-FIX:
1. [eng_chatterbox.py::_project] If a stamped vector lacks the key `"calm"`, the default is `0.5`, which maps to an exaggeration of `0.65` – not neutral. While the deterministic derivation always includes `"calm"`, a stamped vector from a future or manual source could be missing it. Consider documenting that the default is `0.5` (neutral) or add a guard to ensure a reasonable default.
2. [eng_indextts2.py::emo_list] The import of `EMOTIONS` from `.._otr_delivery_vector` creates a hard binding to that module’s tuple. If the tuple order were ever changed (unlikely), the worker would silently receive a permuted vector. No action needed now, but note the tight coupling.

OPTIONAL / NICE-TO-HAVE:
- In `_render_per_line`, the tension reading `ln.get("scene_tension", ln.get("tension", 0.0)) or 0.0` could be simplified to `float(ln.get("scene_tension") or ln.get("tension") or 0.0)` once, avoiding float‑conversion inside the try block.

CUT THESE (over-engineering):
- None. The code is lean; the unused `stamp_delivery_vectors` is marked out of scope and harmless.

[ASSUMPTION] The worker process (IndexTTS2 / Chatterbox) expects exactly the 8‑dim list in `EMOTIONS` order, and that order is documented in `_otr_delivery_vector.py` as canonical. I see no mismatch.