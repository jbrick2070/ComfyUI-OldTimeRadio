VERDICT: no. The plan's core premise of a silent auto-downgrade contradicts the production render path where all fallbacks are disabled and fail loud.

MUST-FIX BEFORE BUILD:
1. [Current state] The plan states that the 14B model silently downgrades to humo_1.7B on VRAM/OOM issues under the AV stack. However, the production render driver ([render_driver.py:1470-1475](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L1470-L1475)) has fallbacks disabled and raises a hard `RenderError` loud on failure. Silent downgrades are currently impossible in production.
   Fix: Restructure the bakeoff goal to verify engine selection correctness and trace actual UNET files loaded under VRAM constraints rather than testing a nonexistent auto-downgrade loop.
2. [GOAL] The "ISOLATED bakeoff" goal of simulating "AV stack's VRAM pressure" by booting a clean server per leg ([run_ltx_av_q_bakeoff.py:20-25](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/run_ltx_av_q_bakeoff.py#L20-25)) is logically contradictory. Clean boot states eliminate the residual memory fragmentation and model residency of other active nodes.
   Fix: Split the bakeoff into Phase A (clean isolated sweep of hyperparameters) and Phase B (dirty sequence run, loading LTX-AV and Whisper models immediately before the HuMo leg without server reboot to simulate stack pressure).
3. [OPEN QUESTION] The plan fails to define the baseline test assets (still image, driving audio) and the objective quality metrics to evaluate "quality" quantitatively.
   Fix: Specify fixed test filenames (e.g., standardizing on assets like `c02_466a19906ccb.png` and `c02_b002_line.wav` from [run_ltx_av_q_bakeoff.py:92-93](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/run_ltx_av_q_bakeoff.py#L92-93)) and define objective quality proxies (e.g., face detection confidence, landmarks motion range, and SSIM of the lip area).
4. [Starter levers: SHIFT] The plan proposes sweeping the SD3 shift value, but the shift is currently hardcoded to `8.0` in [eng_humo.py:271](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_humo.py#L271).
   Fix: Modify `eng_humo.py` to check an environment variable like `OTR_HUMO_SHIFT` or describe how the bakeoff runner will dynamically rewrite the API graph shift parameter.

SHOULD-FIX:
1. [Starter levers: CFG] The CFG sweep (1.0 vs higher) is proposed without isolating it from LoRA. Distill LoRAs are trained to operate at CFG 1.0; higher CFG values on distilled legs will result in severe blue saturation.
   Fix: Restrict CFG sweeps to legs where the distill LoRA is disabled (`OTR_HUMO_LORA_NAME=none`).
2. [Starter levers: RESOLUTION] The plan proposes a general resolution sweep, ignoring that HuMo's native aspect ratios (480x832 and 832x480) are baked into separate classes in `eng_humo.py` (e.g., `HuMo14BLandscapeEngine` [eng_humo.py:541-558](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_humo.py#L541-L558)). Non-native dimensions risk severe interpolation blur.
   Fix: Limit resolution sweeps to the two registered production aspect ratios (480x832 portrait and 832x480 landscape).
3. [Starter levers: DOWNGRADE PROTECTION] The plan asks how the 14B model can be reserved/evicted, but post-decode model eviction is already handled in [eng_humo.py:361](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_humo.py#L361).
   Fix: Frame the "downgrade protection" lever specifically around testing pre-emptive VRAM reservations (e.g. startup `--reserve-vram` settings).

OPTIONAL / NICE-TO-HAVE:
- Build a manifest dry-run validation mode (`--dry-validate`) in the bakeoff script to verify graph topology before booting the ComfyUI server.

CUT THESE (scope / over-engineering):
1. [Starter levers: NEGATIVE] Cut negative prompt sweeping. It introduces a subjective variable that is secondary to resolving memory and model identity bugs.
2. [Starter levers: FRAMES] Cut frame-length sweeps. Length boundaries (33 to 177 frames) are already enforced by [eng_humo.py:53-54](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_humo.py#L53-L54); checking intermediate frame lengths adds runtime without helping quality.

[ASSUMPTION] We assume the available environment on the GPU host contains basic dependencies (such as OpenCV or face-landmark packages) if objective face metrics are to be compiled, otherwise the bakeoff will rely solely on VRAM/speed metrics and human QA.
