<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan conflates engine capabilities with role capabilities, which will break the procedural floor if implemented as proposed.

MUST-FIX BEFORE BUILD:
1. [ Universal init_image lane / Dispatcher gate ] The plan proposes that the dispatcher decide to render a still based purely on the video engine's capability (`accepts_still: True` or similar). This