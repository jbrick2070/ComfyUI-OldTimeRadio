VERDICT: no. The plan does not yet define a checkable two-stage boundary/reclaim contract, and one control leg is wired to the wrong aspect tier.

MUST-FIX BEFORE BUILD:
1. [Q1 ISOLATION] The “standalone graph variant” cannot claim two-stage TE eviction until it names the executable boundary. Current production HuMo is single `_wb.run_graph(graph, classes)` then post-decode reclaim, not pre-sampler reclaim: `nodes/_otr_video_engines/eng_humo.py:343-361`. `wrapper_bridge.reclaim_idle_models()` is a Python helper, not shown as a registered LiteGraph/API node: `nodes/_otr_video_engines/wrapper_bridge.py:248-301`. Concrete fix: implement the bakeoff runner so Stage A executes conditioning, calls `_wb.reclaim_idle_models(reason="humo bakeoff pre-sampler")`, then Stage B executes sampler/decode under one continuous `VramPeak` window; or verify and cite a real Comfy node from `/object_info` that performs that reclaim and is dependency-wired before `KSampler`.

2. [Q2 LEGS] The proposed `humo_1.7B control` is not an interface-compatible 16:9 control for `humo_14B_169`. `humo_1.7B` is the portrait tier by inheritance; the wide 16:9 control is `humo_1.7B_169`: `nodes/_otr_video_engines/eng_humo.py:96-99`, `nodes/_otr_video_engines/eng_humo.py:479`, `nodes/_otr_video_engines/eng_humo.py:519-533`. Concrete fix: make the control leg `humo_1.7B_169`, same still/audio/seed/aspect, and record its 16:9 cfg source.

3. [Q3 METRICS] The gating VRAM number is underspecified and will drift to the repo’s 14.5 GB default if the LTX harness is copied. Defaults are 14500 MB in `motion_common`, `wrapper_bridge`, and the LTX bakeoff runner: `nodes/_otr_video_engines/motion_common.py:37-55`, `nodes/_otr_video_engines/wrapper_bridge.py:36-37`, `scripts/run_ltx_av_q_bakeoff.py:103`. Concrete fix: HuMo bakeoff must hard-gate the heavy-engine target at 13500 MB while separately reporting the 14500 MB absolute box ceiling.

4. [Q4 RESET/BOOT] The boot profile depends on the chosen isolation path. `_otr_soak_server_launch.cmd FLOOR` clears `OTR_ENABLE_HUMO`; the default branch enables it: `scripts/_otr_soak_server_launch.cmd:60-80`. Concrete fix: if running raw API core nodes, `FLOOR` is acceptable only after `/object_info` confirms HuMo wrapper classes are registered; if importing/using `HuMoEngine`, boot with HuMo enabled and manifest `OTR_ENABLE_HUMO`, `OTR_HUMO_*`, and resolved class names.

5. [Q4 RESET/BOOT] Clean-boot peak alone is not production-equivalent for the cross-engine residency question. Production does pre-render and inter-beat reclaim, but both are best-effort and no reboot occurs: `nodes/_otr_video_engines/render_driver.py:1540-1548`, `nodes/_otr_video_engines/render_driver.py:1550-1572`. Concrete fix: include a no-reboot sentinel leg that first runs/loads the representative LTX-AV stack, then runs the HuMo two-stage leg after the same reclaim path, under one server session.

6. [Q4 RESET/BOOT] Do not copy the LTX reset verbatim without adding prior-harness cleanup. It kills Comfy `main.py` and port owners only: `scripts/run_ltx_av_q_bakeoff.py:146-154`. The repo rule also requires killing orphan soak/sweep harnesses by command line. Concrete fix: reset must selectively kill prior `run_humo_*bakeoff.py` processes except the current PID, plus Comfy server and port 8000.

SHOULD-FIX:
1. [Q3 METRICS] Manifest checks must assert, not just record, the resolved API prompt values: UNET filename, LoRA present/absent, `ModelSamplingSD3 shift=8`, steps, cfg, seed, dimensions, 4n+1 frame length, terminal, and output path. LTX’s runner already has this fail-loud pattern: `scripts/run_ltx_av_q_bakeoff.py:342-414`.

2. [Q3 METRICS] Keep HuMo output silent and encode consistently with production. `HuMoEngine` emits silent MP4 through `wrapper_bridge.encode_frames_to_silent_mp4`: `nodes/_otr_video_engines/eng_humo.py:352-367`; legacy `render_humo_batch.py` used `CreateVideo/SaveVideo` with audio: `scripts/render_humo_batch.py:670-685`. Concrete fix: use `SaveImage` frames plus production silent encoder, not `CreateVideo/SaveVideo`.

3. [Q3 METRICS] Objective quality proxies need feature detection before the run. Concrete fix: write a preflight manifest section for `cv2`, face detector, landmark stack, and ffmpeg/ffprobe availability; mark each metric skipped instead of failing or silently omitting.

4. [Q1 ISOLATION] If the harness imports `wrapper_bridge` by file path like LTX Q does, keep it cold-import-only and avoid package `nodes` import side effects. LTX Q demonstrates that pattern at `scripts/run_ltx_av_q_bakeoff.py:77-84`.

OPTIONAL / NICE-TO-HAVE:
- Add a dry-validate mode that boots once, converts/builds every leg, writes manifests, and renders nothing, mirroring `scripts/run_ltx_av_q_bakeoff.py:870-901`.

CUT THESE (over-engineering):
1. [Q2 LEGS] Cut optional leg (iv) from the minimal build. The no-LoRA ~25-step 14B reference is not needed to answer “5/21 quality AND <=13.5 GB”; baseline 14B single-graph, 14B two-stage, and 1.7B_169 control are enough.