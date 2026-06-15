<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. Option A is the correct path, but the proposed drift assertion in the muxer is over-engineered and impossible without wiring changes.

MUST-FIX BEFORE BUILD:
1. [Candidate fixes - A / otr_master_audio_mux.py] The muxer's duration gate strictly forbids `v_dur > a_dur + tol`, which will crash when the ~20s silent credits tail is introduced. 
   Fix: In `otr_master_audio_mux.py`, delete the block `if v_dur >= 0 and a_dur >= 0 and v_dur > a_dur + tol: raise ValueError(...)`. The composite already strictly enforces the frame budget (`got == total`), so removing this upper bound is safe and allows the silent tail.
2. [Candidate fixes - A / otr_silent_composite.py] The composite explicitly caps the assembled length to the master audio duration (`base_total = max(0, int(round(master_dur * fps)) - 1)`), which cuts off the credits. 
   Fix: In `assemble_silent_timeline`, update the `base_total` calculation to use the floor's full length if it is longer:
   ```python
   if master_dur > 0:
       base_total = max(0, int(round(master_dur * fps)) - 1)
       if floor_ok and floor_frames > base_total:
           base_total = floor_frames
       if base_total > 0 and ...
   ```
3. [Questions for the panel - 4 / §4D blend]