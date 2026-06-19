<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

**VERDICT: yes-with-fixes**
The strategy is sound, but wiring Kokoro for `char_voice` as proposed will cause characters to speak stage directions aloud, and the proposed "mirroring" of Bark's casting is unnecessary parallel work.

**MUST-FIX BEFORE BUILD:**
1. **[kokoro-as-char_voice wiring]** `eng_kokoro.py