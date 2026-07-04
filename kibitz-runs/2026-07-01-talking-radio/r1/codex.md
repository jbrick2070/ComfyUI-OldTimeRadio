VERDICT: no. The core premise says LTX will lip-sync a radio mouth, but the repo’s current LTX path is explicitly ambient I2V motion, not true lip-sync.

MUST-FIX BEFORE BUILD:
1. [The idea (grounded)] Defect: the plan’s “radio TALKS/SINGS” premise contradicts the current implementation notes: `eng_ltx_av.py` says there is “no separate LTX 'lip-sync' parameter” and only conditions on still + audio (`C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py:842-848`); `render_driver.py` says `OTR_LTX_RADIO_FACE` is ambient motion, not lip-sync (`.../nodes/_otr_video_engines/render_driver.py:1074-1079`); the shipped A/B doc says HuMo remains the only true talking/singing host (`.../docs/2026-07-01-brief-driven-radio-host/ADDENDUM_ltx_audio_in_ab.md:40-59`). Concrete fix: reframe the goal as “test whether LTX ambient audio-conditioned motion reads as a talking radio,” or keep HuMo as the only true lip-sync route until a live render proves otherwise.

2. [SUB-PLAN C] Defect: the go/no-go decision is underspecified for the plan’s central risk. “Eyeball: talks vs ambient-drift” is not enough when the codebase already documents LTX as ambient. Concrete fix: define the acceptance evidence before build: same frozen audio/still, side-by-side `OTR_LTX_RADIO_FACE=0/1`, plus a short written criterion for mouth-open/close correlation to speech/music transients. Keep “retire HuMo for bookends” out of scope unless that evidence passes.

3. [SUB-PLAN A] Defect: the two-stage upsampler is not causally tied to the talking-radio question. Sharper output may improve image quality, but it does not address whether LTX can animate a grille-mouth as speech. Current graph is single-stage through `LTXVConcatAVLatent -> sampler -> LTXVSeparateAVLatent -> VAEDecodeTiled` (`.../nodes/_otr_video_engines/eng_ltx_av.py:584-614`) and lacks the proposed upsampler candidates (`Select-String` found no `LTXVLatentUpsampler`, `LatentUpscaleModelLoader`, `LTXVCropGuides`, or `OTR_LTX_AV_UPSCALE` in `eng_ltx_av.py`). Concrete fix: make A a follow-up quality pass after C proves the visual concept, or explicitly state A is optional and not required for the talking-radio proof.

4. [SUB-PLAN B] Defect: “image prompt only” is true mechanically, but the prompt is shared across HuMo host and LTX radio-face consumers. Current HuMo radio-host mint uses `build_radio_host_prompt(... style="console_face")` (`.../nodes/otr_meta_brief_image_prompt.py:1078-1092`), and the LTX A/B stills use the same builder with `console_face`/`radio_head_person` (`.../nodes/otr_meta_brief_image_prompt.py:1095-1118`). A mouth-forward change can improve LTX while harming HuMo’s face-readability. Concrete fix: split consumer intent in the plan: either add separate prompt constants/styles for LTX radio-face vs HuMo radio-host, or require acceptance tests for both consumers before landing B.

5. [THREE DISTINCT SUB-PLANS] Defect: the plan says A and B can run in parallel in separate coder windows, but repo rules explicitly require “One coder window in the code at a time” serialized via `docs/GO_FORWARD_PLAN.md` (`C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/CLAUDE.md:57-58`). Concrete fix: sequence B first, then C proof, then A if still needed; do not advertise parallel edits.

SHOULD-FIX:
1. [SUB-PLAN A] The plan says today’s render is “512x288,” but current engine native fallback is `832x480` (`.../nodes/_otr_video_engines/eng_ltx_av.py:58-59`, `:772-777`) and workflow node 87 canvas is `832,480` on line 1 of `.../workflows/otr_scifi_16gb_full.json`. Concrete fix: correct the baseline dimensions before estimating VRAM or quality delta.

2. [SUB-PLAN A] The plan says “add the new node ids to `_node_candidates`,” but `_node_candidates` is also part of the usability gate (`.../nodes/_otr_video_engines/eng_ltx_av.py:459-494`). If optional upscaler classes are always required, OFF mode can stop being byte-identical on installs missing those nodes. Concrete fix: specify that upscaler candidates are only required/resolved when `OTR_LTX_AV_UPSCALE=1`.

3. [SUB-PLAN C] The plan says no workflow JSON change, but persistent routing changes must land in `workflows/otr_scifi_16gb_full.json` per project rules, and current node 87 saves announcer/music as `viz_green` (`.../workflows/otr_scifi_16gb_full.json:1`; decoded node 87 `widgets_values[0:3] = viz_green`). Concrete fix: state C is override-only; any persistent default switch is a separate same-commit workflow JSON change.

4. [Ask for the panel #4] [ASSUMPTION] The plan asks whether LTX face-detection/crop preprocessing could block non-face radio lip-sync, but does not make that a build gate. Concrete fix: require live `/object_info` capture and a one-beat probe before committing to engine surgery.

OPTIONAL / NICE-TO-HAVE:
- Add a manifest stamp for “ambient vs lip-sync expectation” in C so future readers do not misread a non-talking face as a failed render.
- Keep the official comfy.org workflow comparison as research evidence, but do not treat it as proof of this repo’s runtime behavior without a local probe.

CUT THESE (scope / over-engineering):
1. [SUB-PLAN A] Cut the LTX two-stage latent upsampler from the first build. It is safe to cut because the existing graph already renders `ltx_audio_in`; the unresolved risk is mouth-motion semantics, not sharpness.
2. [SUB-PLAN C] Cut “retire `OTR_ENABLE_HUMO_HOSTS` for bookends” from this plan. It is safe to defer because the current repo explicitly treats HuMo as the true talking-host path and C is only an eyeball decision, not a production routing change.
3. [THREE DISTINCT SUB-PLANS] Cut the “parallel coder windows” operating model. It conflicts with repo rules and does not serve the creative proof.