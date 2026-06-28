# HuMo r4 -- CONVERGENCE / residual defects on the standalone bakeoff (final pass before coding)

The plan below is r3-hardened and near build-ready. r4 = LAST pass: hunt ONLY residual must-fix
defects that would break the build or violate an invariant. If none, say so (CONVERGED). Do NOT
re-open settled decisions or add scope. Ground every claim in the real files.

## The near-final plan (r3)
STANDALONE diagnostic bakeoff, ZERO production touch (no edit to eng_humo.py / otr_scifi_16gb_full.json
/ the OTR pack __init__.py). NEW: `custom_nodes/otr_bakeoff_helper/__init__.py` registering ONE node
`OTR_BakeoffReclaim` (calls wrapper_bridge.reclaim_idle_models, passthrough) wired BETWEEN conditioning
and KSampler for mid-graph umt5+whisper eviction in one /prompt; `scripts/build_humo_bakeoff_workflow.py`
(emits per-leg standalone graph JSON, reuses HuMoEngine._build_graph read-only); `scripts/run_humo_bakeoff.py`
(boot-per-leg headless, external nvidia-smi VRAM peak, fail-loud asserting manifest, clips). Cold-import
wrapper_bridge by file path.
LEGS (fixed still c02_466a19906ccb.png + audio c02_b002_line.wav + seed): (i) humo_14B_169 single-graph
6-step distill = 5/21 baseline; (ii) humo_14B_169 TWO-STAGE (OTR_BakeoffReclaim) = candidate;
(iii) humo_1.7B_169 control; (iv) no-reboot SENTINEL: load LTX-AV+Whisper, then leg (ii) same session =
production-true peak. CUT the no-LoRA 25-step leg.
METRICS: external VRAM peak, gate 13500 MB (report 14500 box ceiling); s/it + wall-clock; side-by-side
clips -> otr/episodes/_bakeoff_humo/<leg>.mp4; blue-cast delta in pure PIL+numpy; face/lip metrics
soft-gated (cv2/dlib/mediapipe/ffprobe preflight, skip if absent). Manifest ASSERTS resolved
unet/lora/shift8/steps/cfg/seed/dims/4n+1/terminal/output + the id that ACTUALLY ran.
OUTPUT: SaveImage PNG frames -> production wrapper_bridge.encode_frames_to_silent_mp4 (silent, byte-match).
BOOT/RESET: boot WITHOUT FLOOR (FLOOR clears OTR_ENABLE_HUMO); reset_box selectively CIM-kills (by
cmdline, excl. current PID) prior run_humo_bakeoff.py trees + the SOAK_LAUNCH_CMD cmd.exe + Comfy server
+ port 8000; --dry-validate mode. All production-candidate changes DEFERRED, operator-gated on the clips.

## r4 questions (residual only)
1. Does `OTR_BakeoffReclaim` as a passthrough node actually force the reclaim to run BEFORE KSampler in
   ComfyUI's execution order? (topological dependency: KSampler's model/latent input must route THROUGH
   the node, else the executor may run reclaim after sampling, or cache-skip it -- verify the wiring
   contract + IS_CHANGED so it isn't cached away.)
2. Does reclaim_idle_models, called MID-GRAPH from inside a node, risk evicting a model KSampler still
   needs (the 14B unet / VAE), not just the umt5/whisper encoders? Confirm what it evicts + that the
   sampler model survives.
3. Can the sentinel leg's "representative LTX-AV render" run in the SAME server session as a HuMo graph
   without a class/import conflict, and is its residency representative of production?
4. Any remaining boot/reset/manifest defect that would make a leg silently measure the wrong graph.

Deliver: the residual must-fix list (or CONVERGED), each grounded with a file:line + a concrete fix.
