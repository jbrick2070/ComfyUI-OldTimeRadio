<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Fails on schema mismatches, impossible deterministic checks, and prompt-target confusion.

MUST-FIX BEFORE BUILD:
1. [S3] Prompt-target confusion. The plan injects "require each beat intent -> an ACTION VERB UNDER PRESSURE" into the *line-composer prompt*. The line composer does not write beat intents; it consumes them. The intent is written in `_otr_outline.py` Stage 3.
   Fix: Move the action-verb constraint to `_BEAT_SYSTEM_PROMPT` in `_otr_outline.py`.
2. [C2] Schema gap. The plan derives `central_object