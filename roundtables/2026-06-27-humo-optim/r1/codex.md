VERDICT: no. The stated goal mixes isolated quality tuning with diagnosing production AV-stack downgrade behavior, and the current code appears to contradict the “auto-downgrade” premise.

MUST-FIX BEFORE BUILD:
1. [Current state] The “humo(14B) -> humo_1.7B -> still_kenburns” downgrade premise is not currently true for the main render path: `eng_humo.py` declares `fallback_engine = "humo_1.7B"` (`nodes/_otr_video_engines/eng_humo.py:106`), but `render_shot()` says fallbacks are disabled and raises on failure (`nodes/_otr_video_engines/render_driver.py:1470-1495`). Concrete fix: rewrite the bakeoff premise around “prove which graph/model actually ran” instead of “silent downgrade,” or explicitly identify a different caller path where fallback still executes. verify: whether any non-`render_driver.render_shot()` production path still honors HuMo fallback.

2. [GOAL] “ISOLATED bakeoff” cannot prove the suspected failure mode “under the AV stack’s VRAM pressure.” The LTX bakeoff template resets and boots per leg (`scripts/run_ltx_av_q_bakeoff.py:20-23`, `:646-679`), which is good for clean quality measurements but removes cross-engine residency pressure. Concrete fix: split into Phase A isolated HuMo quality sweep and Phase B production-pressure sentinel using the real workflow or a minimal LTX/AV -> HuMo sequence, with `final_engine`, UNET filename, LoRA name, peak VRAM, and failure/no-fallback state recorded.

3. [OPEN QUESTION / Deliver] This is not yet a build spec: it asks for “recommended bakeoff leg set,” “metrics,” and “single fixed still+audio choice,” but does not define them. Concrete fix: freeze a staged matrix before coding: baseline 14B portrait, forced 14B no-downgrade identity check, 1.7B control, then one-lever sweeps only after the identity rail passes. Pick one fixed still/audio pair with visible mouth, plosives/sibilants, and a known duration. [ASSUMPTION] Quality comparison needs human eyeball plus objective proxies.

4. [Starter levers: SHIFT] SHIFT is listed as a lever, but current HuMo engine hardcodes `ModelSamplingSD3` shift to `8.0` (`nodes/_otr_video_engines/eng_humo.py:269-271`). Concrete fix: decide whether the bakeoff mutates a direct API graph like the old temp sweep (`scripts/_otr_humo_shift_sweep.py`) or adds a real configurable knob. If this is meant to mirror production, the manifest must assert the resolved shift value.

5. [Starter levers: RESOLUTION] The plan frames resolution as “480x832 native vs others,” but the code already has a 14B wide engine (`HuMo14BLandscapeEngine`) and 1.7B wide engine (`nodes/_otr_video_engines/eng_humo.py:519-563`), with official HuMo dims centralized as portrait `480x832` and wide `832x480` (`nodes/_otr_shared/aspect.py:49-51`). Concrete fix: either scope the bakeoff explicitly to portrait HuMo only, or include 14B portrait vs 14B wide as first-class user-facing variants.

SHOULD-FIX:
1. [Metrics] VRAM and s/it are feasibility metrics, not face/lip-sync quality. Concrete fix: add at least one objective proxy: face detection success, landmark mouth-motion/audio-energy correlation, identity drift against the source still, and color delta/source-still blue cast. [ASSUMPTION] Existing local dependencies may not include a face/landmark stack; verify available libraries before promising these.

2. [Starter levers] The lever list is combinatorial. TIER, STEPS, CFG, LoRA, SHIFT, RESOLUTION, FRAMES, NEGATIVE, and DOWNGRADE PROTECTION cannot all be open-ended without destroying “vary ONE lever per leg.” Concrete fix: gate later sweeps on baseline identity/VRAM success and carry forward winners stage by stage, like `run_ltx_av_q_bakeoff.py` does with manifests and staged picks (`scripts/run_ltx_av_q_bakeoff.py:790-981`).

3. [Current state] The document says env knobs include steps/cfg/UNET/LoRA/checkpoint/negative, but not width/height, while code also supports `OTR_HUMO_WIDTH` and `OTR_HUMO_HEIGHT` (`nodes/_otr_video_engines/eng_humo.py:208-212`). Concrete fix: list all intended bakeoff-controlled knobs and classify each as production env, bakeoff-only API mutation, or not allowed.

4. [DOWNGRADE PROTECTION] “Reserved/evicted so it never silently falls to 1.7B” conflates two mechanisms: current HuMo already reclaims idle models after decode (`nodes/_otr_video_engines/eng_humo.py:361`) and the episode driver performs inter-beat reclaim when engine changes (`nodes/_otr_video_engines/render_driver.py:1551-1569`). Concrete fix: define the actual protection target: prevent OOM, prevent wrong UNET load, or fail loud instead of fallback.

OPTIONAL / NICE-TO-HAVE:
- Generate a side-by-side contact sheet with labels burned outside the frame, but keep raw clips as the primary deliverable.
- Add a “manifest-only dry validate” mode modeled on the LTX bakeoff manifest rail (`scripts/run_ltx_av_q_bakeoff.py:326-432`).

CUT THESE (scope / over-engineering):
1. [Starter levers: NEGATIVE] Cut negative-prompt sweeps from the first build. Negative text is secondary to proving tier identity, LoRA, steps, cfg, shift, and VRAM; keep default vs one operator-provided negative only after baseline passes.

2. [Starter levers: FRAMES] Cut broad frame-length sweeps. Use one short lip-sync clip and one max-length sentinel only. The code already clamps HuMo to 33-177 frames (`nodes/_otr_video_engines/eng_humo.py:51-52`), so broad frame search is unlikely to answer the main quality/downgrade question.

3. [Starter levers: RESOLUTION] Cut arbitrary “others.” Test only code-supported native portrait `480x832` and wide `832x480` first (`nodes/_otr_shared/aspect.py:49-51`). Non-native sizes can wait until a native winner exists.

4. [Starter levers: TIER] Cut optimization of 1.7B in round one. Keep 1.7B as a control/proof of downgrade detection; do not spend early legs tuning it unless 14B fails the 14.5 GB ceiling.