<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan misidentifies where `flux_still` gets its `init_image` and misses that `_portrait_index` only indexes objects with a `char_id`.

MUST-FIX BEFORE BUILD:
1. [FIX 1] `build_request_from_shot` ignores scene stills for `flux_still`. The