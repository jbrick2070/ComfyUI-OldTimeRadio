VERDICT: no. The runbook has the right invariant, but it is not tied to one executable harness and its env names/gates conflict with current soak scripts.

MUST-FIX BEFORE BUILD:
1. [0, 7] The “first leg must ship OBS final” invariant is not mapped to the actual unattended drivers. `scripts\_otr_soak_capstone.py:305-342` enforces a real `*_final.mp4` in OBS, but `scripts\_otr_nightly_anthology_soak.py:423-440` and `scripts\_otr_overnight_story_soak.py:443-457` only abort on `n_lines <= 0`, so a render/publish failure can still pass the smoke gate conceptually. Fix: name the canonical soak driver for this runbook and require its first-leg gate to use the capstone OBS deliverable check, or explicitly state that story-only soaks are excluded.

2. [1, 3, 7] The env knob story is internally inconsistent and contradicts current code. The runbook names `OTR_SOAK_TARGET_WORDS` / `OTR_SOAK_ACT_COUNT`, but anthology/story drivers use `OTR_SOAK_WORDS` / `OTR_SOAK_ACT` with default `auto` (`scripts\_otr_nightly_anthology_soak.py:73,82`; `scripts\_otr_overnight_story_soak.py:56,63`), combo soak uses `OTR_SOAK_TARGET_WORDS` / `OTR_SOAK_ACT_COUNT` default `3` (`scripts\_otr_combo_soak.py:14-15,57-58`), and marathon patches `act_count` to `auto` (`scripts\_otr_soak_marathon.py:220-223`). Fix: split the runbook by driver or create one canonical wrapper with one env schema.

3. [2, 4] “Successive harness invocations against one booted server” conflicts with “flags are read once at boot.” The launcher applies engine flags before starting ComfyUI (`scripts\_otr_soak_server_launch.cmd:60-85,108-116`), while Section 4 expects changing engine coverage across successive invocations on the same already-booted server. Fix: either require all needed `OTR_ENABLE_*` flags before the single boot, or say each flag-set change requires a server reboot and a fresh Section 0 gate.

4. [2, 4] LTX engine naming is conflated. `ltx_video` is gated by `OTR_ENABLE_LTX_VIDEO` and documented default-on/opt-out (`nodes\_otr_video_engines\eng_ltx_video.py:18,284,367`), while audio-input LTX is `ltx_audio_in` gated by `OTR_ENABLE_LTX_AV` (`nodes\_otr_video_engines\registry.py:232-247`; `nodes\_otr_video_engines\eng_ltx_av.py:273`). The launcher’s `LTX` lane sets `OTR_ENABLE_LTX_VIDEO`, not `OTR_ENABLE_LTX_AV` (`scripts\_otr_soak_server_launch.cmd:63-66`). Fix: list exact engine IDs and exact flags separately: `ltx_video` vs `ltx_audio_in`.

5. [6] The “scheduled morning task” is only a wish, not an executable part of the runbook. I found generated reports/status helpers, but no cited canonical scheduled reporter that reads `otr\obs`, `otr\episodes`, and the ComfyUI log as described. [ASSUMPTION] Fix: add the exact script/task name and output artifact path, or downgrade this to manual morning triage and write “verify: scheduled task exists.”

SHOULD-FIX:
1. [0] The file gate should use a pre-leg snapshot or timestamp, not just “confirm a fresh `*_final.mp4` exists.” The capstone code already does this with `after_ts` and OBS listing delta (`scripts\_otr_soak_capstone.py:312-318,277-280`). Fix: specify “newer than leg start” or “OBS listing delta exactly one final mp4.”

2. [3] The per-role knob names do not match all live drivers. Section 3 lists `OTR_SOAK_ANNOUNCER`, `OTR_SOAK_MUSIC`, `OTR_SOAK_BEATS`, etc.; those exist in older/specific scripts such as `scripts\_otr_overnight_420_soak.py:79-91`, while combo uses `OTR_COMBO_ANNOUNCER/MUSIC/BEATS` (`scripts\_otr_combo_soak.py:10-18,47-58`) and story/anthology use different role controls (`scripts\_otr_overnight_story_soak.py:183-192`; `scripts\_otr_nightly_anthology_soak.py:100-121`). Fix: make the config section a per-driver matrix.

3. [4] The coverage matrix mixes smoke validation, story-quality evaluation, long-form stress, and image-model bakeoff into one “good soak” arc. These are different success criteria: OBS-final shipment, story quality, VRAM endurance, and image coverage. Fix: define one primary soak objective, then move the other campaigns into optional profiles.

4. [5] “Prefer freeing the writer LLM before voice/render” is not actionable. [ASSUMPTION] The runbook assumes an operator knows how to free Ollama/local LLM residency safely. Fix: cite the exact reset/free command or remove the instruction.

OPTIONAL / NICE-TO-HAVE:
- Add a compact “known-good commands” block per canonical driver.
- Add a “what success looks like” sample row for the chosen JSON/CSV summary.

CUT THESE (scope / over-engineering):
1. [4] Cut “Long multi-video” from the first build of this runbook. It is a stress campaign, not required to prevent the 337-error/0-episode failure, and it increases ambiguity around server boot flags.

2. [4] Cut “Image variety” from the mandatory checklist. It serves coverage breadth, not the stated invariant that an unattended soak must first prove it can ship episodes.

3. [6] Cut “scheduled morning task” until it names an actual script/task. As written it adds process obligation without an implementation contract.