# 4060 clean-room install: friction log (2026-09-01)

Method: build ComfyUI from the official portable zip in a folder that has never seen OTR,
follow README literally, count every stop. The Desktop install on the same box was left
untouched. Profiles used are headless stand-ins for saved dropdown values (one canonical
JSON, different dropdowns), not new in-app profiles.

Box: MRKT, RTX 4060 Laptop 8 GB, Windows 11, 32 GB RAM, ~44 MB/s to Hugging Face.
Clean room: C:\OTR-CleanRoom, fresh ComfyUI_windows_portable_nvidia v0.34.0
(Python 3.13.14, torch 2.13.0+cu130), fresh HF cache, OTR by git clone at d4e1d5af
(registry install impossible: both registry versions Flagged, latest_version null).
The existing Desktop install on the same box was not touched.

## Friction points, in the order a new user meets them

F1  BLOCKER  `pip install -r requirements.txt` fails on Python 3.13 (the interpreter both
    ComfyUI Desktop and the portable ship today). Cause: `kokoro>=0.7.16`. kokoro 0.7.16
    pins numpy==1.26.4 (no 3.13 wheel); kokoro 0.8+/0.9.x and every misaki>=0.7.5 declare
    Requires-Python <3.13, and misaki[en] pulls spacy/thinc/blis with no 3.13 wheels.
    pip resolves all-or-nothing, so NONE of the 18 requirements install. Verified on the
    box (pip freeze before == after) and against PyPI metadata. Kokoro is the shipped
    default announcer voice and the 8 GB class char voice. --ignore-requires-python does
    not rescue it (numpy pin, then spacy source build). Workaround used for this test:
    install everything except kokoro, run voices on bark.
F2  BOX QUIRK (not OTR) pip on the portable Python raised WinError 448 "untrusted mount
    point" under C:\Users\jeffr\AppData; fixed by pointing TEMP/TMP/PIP_CACHE_DIR at a plain
    folder. Recorded, not counted against the pack.
F3  OK  Stock boot (portable .bat semantics, no PYTHONUTF8): prestartup 23.6 s (fetches 28
    Kokoro voice files at boot even though kokoro cannot be installed), import 4.6 s,
    "All 25 nodes loaded", routes registered. No emoji/cp1252 crash.
F4  DOC  README "git clone" step: fine (94 MiB pack, under a minute). README's Browse
    Templates quick start: example_workflows/ carries only otr_4060_floor.json (audit
    fresh-install-docs-01).
F5  MODELS  No 8 GB profile exists for LTX 2.5, HuMo 1.7B or H3; the only proven 8 GB lane
    is AnimateDiff. The LTX 2.5 and HuMo 1.7B tiers are fetchable only through the pod
    provisioner (scripts/otr_provision.py, which does not ship); the Windows fetcher
    (scripts/otr_fetch_lane_weights.py) has no ltx25 / humo_1.7B lane and defaults its
    models root to C:\ComfyUI-Models rather than the running install's models tree.
    For this test the weights were fetched with the provisioner's own pins (bytes + sha256
    verified): LTX 2.5 set 23.9 GB, HuMo 1.7B set 13.6 GB, z_image int8 set 14.6 GB.
F6  H3 is operator-only (private repacks); not a from-nothing candidate.

## Server launch notes
Detached launches from an SSH session die with the session (Start-Process children were
killed at session end). Task Scheduler (schtasks) is the reliable way to keep a headless
server alive on this box.

F7  GOOD  With weights absent, the pack's own preflight refused the queue BEFORE the writer
    ran and named every missing file ("PREFLIGHT FAIL: profile requires model file(s) the
    running server cannot see: LTX-2.5-Distilled-Q3_K_M.gguf, ..."), with the restart hint.
    One wrinkle: the hint cites scripts/_otr_headless_model_paths.yaml, which registry
    installs do not carry.
F8  BLOCKER (LTX 2.5 only)  The LTX 2.5 lane needs ComfyUI-GGUF cloned at a pinned commit
    AND a patch applied from patches/ (git apply) AND its requirements installed, then a
    restart. README section 2b lists only ComfyUI-AnimateDiff-Evolved; nothing tells a
    Windows user about ComfyUI-GGUF or the patch until the render fails. Every LTX 2.5 core
    class (LTXV*, ManualSigmas, ResizeImageMaskNode, VAEDecodeTiled) is already in stock
    ComfyUI v0.34.0; only UnetLoaderGGUF / CLIPLoaderGGUF are missing.
F9  GOOD  HuMo 1.7B needs NO extra node pack: WanHuMoImageToVideo, AudioEncoderLoader and
    AudioEncoderEncode are all in stock ComfyUI v0.34.0. Its four weights are ungated
    Comfy-Org repackages. On node-pack friction it is the lowest-friction meaty video lane.
F10 BOX QUIRK  pip's console-script PATH scan tripped WinError 448 on a junction in this
    box's PATH (an OpenAI Codex app folder); --no-warn-script-location avoids it.

## Leg results

All legs: one act, `public_domain` bank, the pack's own headless runner
(`scripts/otr_canonical_api_run.py`) against the clean-room server on port 8001, started
through Task Scheduler. Profiles are clean-room copies (`otr_cleanroom_8gb_*`, untracked)
because no shipped 8 GB profile carries LTX 2.5 or HuMo; voices are Bark because kokoro
cannot pip-install on the portable's Python 3.13 (PBUG-20260901-04).

R1  2026-09-01 20:5x  `otr_cleanroom_8gb_ltx25`, STOCK flags (DynamicVRAM on). Writer,
    cast and Bark voices completed. The first `z_image_turbo` still (int8 convrot) killed the
    whole ComfyUI process at sampler step 5/8: `Fatal Python error: Aborted`, stack in
    comfy/ldm/lumina/model.py (PBUG-03 shape). Log: server_run1_zimage_abort.log.
R2  2026-09-01 21:0x  Same profile with the lowvram flag pair
    (`--disable-dynamic-vram --lowvram --disable-pinned-memory`). Queued, sat `pending`
    for ~4 min while the writer ran, then stopped on operator instruction before any
    still. No result.
LEG C (stock)  2026-09-02 00:18-01:38  `otr_cleanroom_8gb_klein_ltx25` (the ruled
    low-VRAM image default: `flux2_klein` for all three image roles, LTX 2.5 video), STOCK
    flags, tree at c0ebe31f. Writer 00:18-00:32 (6 lines / 66 words on gemma-4-E2B),
    Bark voices + musicgen cues + master mix by 00:43, ShotLock passed. Klein DID NOT
    ABORT -- the Z-Image abort is engine-specific, which is what this leg was for -- and
    minted a clean 832x480 still (receipt: `legC/legC_stock_klein_c01_832x480.png`, the
    announcer's set: a 1940s radio, dial detail intact, no artifacts). But ONE still took
    ~42 minutes: DynamicVRAM staged the 7.7 GB bf16 Qwen3-4B text encoder
    (`Model Flux2TEModel_ prepared for dynamic VRAM loading. 7671MB Staged`), then loaded
    the 2.6 GB Klein DiT with `0 models unloaded ... loaded partially; 0.00 MB usable,
    0.00 MB loaded, 2591.65 MB offloaded`, 2 min 13 s of "Model Initializing" and then
    ~120 s per step for 20 steps, GPU pinned at 7.9 GB / 100 %. The second still began the
    same way. Stopped after the first still to run the flag pair instead. Verdict for the
    stock path on 8 GB: WORKS, UNUSABLE (an episode with 8 stills would take ~6 hours
    before any video). Log: server_legC_stock_klein.log.
LEG C2  2026-09-02 01:38-02:3x  Same profile, the lowvram flag pair, bf16 encoder. SAME
    SHAPE: the encoder loaded fully (`loaded completely; 7672.25 MB loaded`), then
    `Requested to load Flux2 / 0 models unloaded / loaded partially; 0.00 MB usable`, and
    the first step took 143 s. The flag pair is NOT the fix; the encoder simply never leaves
    the card before the sampler. Stopped at the first still once the root cause was found.
ROOT CAUSE (read in the engine code, 02:0x)  The three local image engines with a separate
    CLIPLoader node (flux2_klein, z_image_turbo, lumina_image) ran their graphs WITHOUT
    `free_after_use`, unlike every video engine (eng_wan_ti2v, eng_ltx_8gb, eng_minimax_h3
    all pass `free_after_use=True` so the text encoder is dropped before the diffusion
    peak). On 16 GB the encoder and the DiT co-reside, so nothing showed. Fix: the same
    one-line pattern (`free_after_use=True, keep={"unet"}`) in all three, measured on the
    RTX 5080 before/after with a fixed seed: byte-identical stills in all three engines
    (same sha256), peak VRAM 14.9/14.4/14.6 GB -> 7.9/9.0/9.6 GB, times 18.2/7.3/16.6 s
    -> 15.3/7.8/14.4 s (within single-run noise; no slowdown on the 16 GB card).
    Receipts: `legC/5080_probe_before.json`, `legC/5080_probe_after.json`. This is also
    the likeliest root of R1's Z-Image abort (the same 0-MB-usable partial load, one
    engine over): Leg C3 tells.
LEG C3  the fix under STOCK flags: same profile, bf16 encoder, the pushed fix pulled into
    the clean room, `_boot_stock.cmd STOCK` (DynamicVRAM on, nothing special). (result below)
LEG C4  only if C3 is still slow: the fp8 encoder (`qwen_3_4b_fp8_mixed.safetensors`,
    5.6 GB, staged in the clean room; the 4060 drill already used it for Z-Image) through
    the engine's `OTR_FLUX2_KLEIN_TE` knob in the server's launch environment
    (`_boot_klein_fp8.cmd`, task OTRCleanRoomServerFP8).
