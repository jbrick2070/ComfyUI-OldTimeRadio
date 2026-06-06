<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: yes-with-fixes — the round‑2 fixes are correctly implemented, but the announcer-voice fallback block is restricted to `char_voice` only, breaking PD1 for engines that also support `announcer_voice` and require a reference clip.

MUST-FIX BEFORE BUILD:
1. [Section: `_otr_voice_node_common._render_per_line`] The per‑line fallback to a preset engine (`bark`) is guarded by `if self.ROLE == "char_voice" …`. Engines like `chatterbox` declare `roles = ("char_voice", "announcer_voice")` and `requires_voice_ref = True`. When the announcer node (role `announcer_voice`) selects `chatterbox` and no usable reference is available, the condition is never true → the adapter receives `voice_ref=None` → the render fails hard instead of falling back (violates PD1 always‑renders).  
   **Fix**: change the check to  
   `if self.ROLE in ("char_voice", "announcer_voice") and _engine_requires_voice_ref(adapter) and not voice_ref and fb_name:`  
   (or, more robustly, check that the engine’s own `roles` contain the current role, but the minimal change is to include `"announcer_voice"`).  

SHOULD-FIX: none (the rest is correct).

OPTIONAL / NICE-TO-HAVE:
- `_otr_dia_worker._ensure_rate` includes a numpy‑interpolation fallback for resampling; it could be dropped if `torchaudio` is guaranteed present in the Dia worker venv, but it’s harmless.

CUT THESE (over-engineering): none.

[ASSUMPTION] The `ledger_json` parameter is never `None` (ComfyUI passes the default `""`); the code `(ledger_json or "").strip()` would handle `None` anyway, so no action needed.