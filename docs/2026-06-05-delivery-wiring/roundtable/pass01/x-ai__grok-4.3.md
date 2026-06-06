<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: yes-with-fixes. One-line: delivery wiring + projections are correct and satisfy all invariants per the excerpts, but two narrow guard/None-path defects exist in _render_per_line.

MUST-FIX BEFORE BUILD:
1. [nodes/_otr_voice_node_common.py:289] `_dl = ln.get("delivery")` then `_stamped = _dl.get(...) if isinstance(_dl, dict)` leaves `_stamped=None` (and forces derive) when the key exists but value is not a dict; change to `(_dl or {}).get("emotion_vector") if isinstance(_dl, dict) else None` so a malformed delivery entry never reaches the stamped branch.
2. [nodes/_otr_voice_node_common.py:310] `prep(text, delivery_vector)` is called with a possibly-None value even when the engine's prepare_text declares `delivery_vector=None` default; add explicit `delivery_vector = delivery_vector if delivery_vector is not None else None` (or just pass it) so the call signature matches every adapter in the excerpts.

SHOULD-FIX:
1. [nodes/_otr_voice_node_common.py:304] the derive except: block sets None but never logs; add a one-line `log.debug(...)` so a bad tension value is visible without changing PD1 behavior.
2. [eng_chatterbox.py:142] `_project` returns 0.3 for a fully-calm derived vector but 0.5 when delivery_vector is None; make the None case also return 0.3 (or document the 0.2 delta) to keep neutral renders identical under the kill-switch.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line unit test that OTR_DELIVERY_VECTOR=0 produces byte-identical calls to the pre-delivery None path for bark/kokoro.

CUT THESE (over-engineering): none. All added logic is the minimal guard+derive needed for the stated invariants.

[ASSUMPTION] stamped vectors are assumed to contain only the 8 EMOTIONS keys (downstream .get() tolerates extras/missing but no schema check exists in the excerpts).