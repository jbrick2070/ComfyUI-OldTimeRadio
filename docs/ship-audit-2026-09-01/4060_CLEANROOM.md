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
LEG C3  2026-09-02 02:31-03:08  the fix (9b90189a) under STOCK flags, DynamicVRAM on, bf16
    encoder. SAME SHAPE AGAIN: `Model Flux2TEModel_ prepared for dynamic VRAM loading.
    7671MB Staged` then `Requested to load Flux2 / 0 models unloaded / loaded partially;
    0.00 MB usable`. Under DynamicVRAM the encoder's staged weights are not released by
    dropping the pack's reference. Stopped at the first still.
PROBE (in-process on the clean room's own portable python, 03:08)  Classic path (no
    aimdo): after CLIPLoader + CLIPTextEncode the encoder held 5.6 GB (free 1297 MB);
    `del clip; gc.collect(); soft_empty_cache()` released ALL of it (free 6966 MB,
    current_loaded_models empty). Then the pack's Klein engine WITH the fix: at
    "Requested to load Flux2" free=6966 MB, loaded=[] -> the DiT loaded 2592 MB resident;
    a 2-step render took 17.7 s end to end. So the fix is correct and sufficient on the
    classic path (`--disable-dynamic-vram`), which is also what ComfyUI < 0.3x and every
    non-NVIDIA box run. Logs: `docs/2026-09-02-encoder-eviction/4060_probe_residency.log`
    (classic) and `4060_probe_residency_aimdo.log` (DynamicVRAM).
PROBE, DynamicVRAM (aimdo initialised in-process exactly as main.py does, pinned staging
    on: `Model Flux2TEModel_ prepared for dynamic VRAM loading. 7671MB Staged`, 03:15)
    After encode the encoder held 6.2 GB (free 620 MB). Dropping every reference + gc +
    soft_empty_cache + cleanup_models + free_memory + unload_all_models: NOTHING released
    (free stayed at ~640 MB; the registry was already empty -- the VBAR pages are an orphan
    that only another DYNAMIC model's pressure reclaims, and the GGUF DiT is a classic
    patcher). That drop-everything-first sequence was the FIRST aimdo run (03:11), whose
    log was overwritten by the re-run that added the A1b line; the committed aimdo log
    shows the same calls as no-ops AFTER A1b (free already 6998 MB), which is the same
    fact from the other side.
    `comfy.model_management.unload_model_and_clones(clip.patcher)` WHILE still registered:
    free 620 MB -> 6998 MB. End to end, same probe, with a CRUDE all-registry eviction
    (`free_memory(1e30)`) monkeypatched into the executor's drop step -- not the shipped
    node-scoped form: `Requested to load Flux2` -> `loaded completely; 5578.68 MB usable,
    2591.65 MB loaded`, 2 steps in 9.4 s (the server took ~4 min for the same two steps).
    This is the root cause of Leg C / C2 / C3 and the likely root of R1's Z-Image abort.
    Design arc + receipts: `docs/2026-09-02-encoder-eviction/` (the classic log there
    predates the probe's A1b line and the Phase B patch; the aimdo log is the full run).
    Shipped as `run_graph(..., evict_after_use={"clip"})` in the three image engines
    (both arc seats APPROVE A2; classic patchers keep the reference drop by an
    `is_dynamic()` gate). 5080 proof on that tree: sha256 identical in all three engines on
    BOTH paths; classic peaks identical (7901/9015/9634 MiB) and warm times unchanged
    within run-to-run noise; DynamicVRAM 13.8/4.8/12.1 s. Receipts
    `legC/5080_probe_after2*.json` (`after2_classic` is the cold first-touch run,
    `after2b_classic_warm` the warm re-measure).
LEG C4  2026-09-02 03:57-04:34  ad6a635f under STOCK flags (DynamicVRAM on, bf16 encoder).
    Pass condition was: `Requested to load Flux2` followed by `loaded completely; <nonzero>
    MB usable, <nonzero> MB loaded, full load: True`, seconds per step, more than one still.
    RESULT: FAIL. The eviction FIRED -- server log line 366 `[OTR graph-exec]
    evict_after_use 'clip': unloaded 1 dynamic model patcher(s) through
    comfy.model_management before the drop` -- and the very next lines were `Requested to
    load Flux2 / 0 models unloaded / loaded partially; 0.00 MB usable`, GPU at 7896 MiB
    during sampling, `Model Initializing ...` again. So on the SERVER the unload ran and the
    encoder's pages stayed on the card, while in-process the identical call freed them
    (free 620 MB -> 6998 MB). The one remaining difference between the two, read from the
    two logs: the server runs torch's cudaMallocAsync allocator (`Device: cuda:0 ... :
    cudaMallocAsync`, set by main.py's `cuda_malloc` import before torch loads) and the
    probe ran the native allocator (`Device: cuda:0 ... : native`); both had aimdo hooks,
    NVML pressure and pinned staging. Stopped at the first still (no second still, so the
    multi-still condition was not reached either). Log: server_legC4_ad6a635f_stock_klein.log.
PROBE, DynamicVRAM + cudaMallocAsync (the server's exact allocator state), 04:36  In-process
    the SHIPPED engine code (no monkeypatch) worked here too: after encode free=583 MB;
    `unload_model_and_clones` alone -> free 6998 MB (the extra release calls changed nothing,
    there was nothing left to release); the engine's own `evict_after_use` then loaded the
    DiT with 2592 MB resident and rendered 2 steps in 10.8 s. So the allocator is NOT the
    difference. Whatever keeps the encoder's pages on the card in the SERVER is something the
    in-process probe cannot reproduce; the next leg (C4b) runs the server with an
    INSTRUMENTED bridge on the clone only (registry entries with clone ids, free VRAM and
    VBAR page residency logged before and after the unload). Log:
    `docs/2026-09-02-encoder-eviction/4060_probe_residency_aimdo_async.log`.
LEG C4b  2026-09-02 04:38-05:14  same as C4 with the instrumented bridge (uncommitted, clone
    only; restored with `git checkout` afterwards). THE ANSWER, server log lines 374-375:
    `EVICT PROBE 'clip' before: free=48MB ... loaded=0MB registry=['Flux2TEModel_:...:0MB']
    vbar_pages=255 resident=0 pinned=0` and `after: free=16MB ... registry=[] resident=0`.
    The encoder was NEVER what filled the card inside the server: its VBAR had zero
    resident pages (the encode streamed through pinned host staging) and the whole 8 GB
    was already taken before it loaded. The only heavy thing loaded right before the
    stills is the WRITER LLM (gemma-4-E2B, transformers, out of ComfyUI's registry): it
    composed the still prompts at lines 348-360 and nothing in the general path releases
    it before the image stage -- `_otr_vram_levers.free_otr_pipeline_residue()` (the
    canonical residue freer: writer LLM + Bark + flush) is called only by the LTX 2.5
    engine and the GGUF backend in their load preflight, and the ghost lane has its own
    `_ghost_unload_writer`; the ImageGenDispatcher never calls it. On 16 GB the E2B
    writer (~5 GB) co-resides with the stills unnoticed; on 8 GB it leaves nothing. So the
    encoder-eviction work (9b90189a + ad6a635f) was a real defect but the SECOND one; the
    first is the resident writer. Logs: `docs/2026-09-02-encoder-eviction/
    4060_server_legC4b_instrumented.log`.
LEG C5  the dispatcher fix: `free_otr_pipeline_residue(reason="image engine load
    preflight")` once per dispatch before the first LOCAL still (the same call the video
    preflights make), under STOCK flags on the clean room. Pass condition as C4, plus the
    dispatcher's own line `pipeline residue freed before the first local still: free
    <several> GB after`.
    5080 MEASUREMENT FIRST (the other box, before the push; headless :8000, working tree,
    `otr_soak_still_motion_flux2_klein`, 1 act, 05:22): the same defect was live on the
    16 GB card -- at the first still the 12B writer was still resident:
    `[VRAMLevers] free_otr_pipeline_residue (image engine load preflight (z_image_turbo))
    OK: unload_llm, _unload_bark, gc.collect, soft_empty_cache, cuda.synchronize,
    cuda.empty_cache, cuda.ipc_collect | allocated 7387 -> 6` and `pipeline residue freed
    before the first local still: free 14.4 GB after`. Five Z-Image stills then minted in
    sequence (cfg 1.0, 8 steps) with no errors; the leg continued into video. Until this
    change the 5080 was rendering stills with a 7.4 GB writer co-resident (17 GB of
    models on a 16 GB card, paged by DynamicVRAM, silently slower). That leg finished:
    `RESULT SUCCESS`, 22 LTX 2.5 clips at ~265 s each (the soak profile's own shape),
    `obs_publish OK -> otr/obs/signal_lost_the_weaver_of_dreams_20260902_052849_..._
    final.mp4`, prompt executed in 1:57:56; the server was released afterwards.
    4060 RESULT (05:35-, da2b7a36, STOCK flags, DynamicVRAM on, bf16 encoder): PASS.
    Server log line 360 `pipeline residue freed before the first local still: free 6.9 GB
    after; ran=unload_llm,_unload_bark,gc.collect,soft_empty_cache,cuda.synchronize,
    cuda.empty_cache,cuda.ipc_collect`, then line 373-374 `Requested to load Flux2 /
    loaded completely; 5560.68 MB usable, 2591.65 MB loaded, full load: True`, then the
    sampler at 1.07 s per step (20 steps, ~21 s a still; it was 120-143 s per step, ~42 min
    a still, in Legs C, C2, C3, C4), `minted still 832x480 seed=701221525`, GPU 4150 MiB
    during sampling. The SECOND still loaded the same way (line 395, `5558.68 MB usable,
    2591.65 MB loaded, full load: True`) -- the multi-still condition met in one server
    process. The leg then continued into the LTX 2.5 video stage (its own, separate
    8 GB question; see below). Klein 4B Q4 GGUF on an 8 GB card under stock launch flags
    is MEASURED at ~21 s a still. Log: server_legC5_da2b7a36_stock_klein.log (the running
    server.log at the time of writing).
    Nine stills in all (three portraits at 832x480, six scene beats at 1472x832), every
    one `loaded completely`. Then the LTX 2.5 stage -- Leg A's own question, answered by
    the same leg: the 12B encoder GGUF pinned to CPU by the engine (`text encoder pinned
    to CPU; GPU encode spike avoided`), the Q3_K_M DiT `loaded partially; 5418.31 MB
    usable, 5391.26 MB loaded, 5708.43 MB offloaded`, and the FIRST CLIP PASSED:
    `ltx25_video TWO-STAGE PASS nodes=3 decode=1664x960 render_elapsed_s=1018.258` --
    17 min a clip on the 4060 under stock flags, half the DiT streaming from host RAM.
    The first LTX 2.5 clip ever rendered on an 8 GB card in this project. Clips 2-6 then
    passed at 827-851 s each (the encoder cache warm), every one a two-stage pass at
    1664x960. The beats are multi-segment (`shot_b001 has 4 segments`), so the episode
    is ~20 clips, not 8 -- the 5080 leg of the same shape ran 22 -- which puts the 4060's
    obs publish about five hours after the leg started (~10:30-11:00), at ~14 min a clip.
    The leg is left running; the episode lands in the clean room's own
    `ComfyUI\output\otr\obs\` when it finishes. (episode result below)
LEG C6  only if C5 is still slow: the fp8 encoder (`qwen_3_4b_fp8_mixed.safetensors`,
    staged; `_boot_klein_fp8.cmd`, task OTRCleanRoomServerFP8).
(The fp8-encoder fallback formerly listed here as Leg C5 is Leg C6 above: 5.6 GB
    `qwen_3_4b_fp8_mixed.safetensors`, staged in the clean room, through the engine's
    `OTR_FLUX2_KLEIN_TE` knob in the server's launch environment.)
