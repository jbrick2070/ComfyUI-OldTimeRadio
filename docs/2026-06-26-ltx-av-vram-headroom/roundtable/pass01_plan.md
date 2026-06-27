# LTX-AV VRAM Headroom -- Hardened Plan (pass01, CONVERGED + IMPLEMENTED)

## Decision (grounded root cause)

In `eng_ltx_av._build_graph` the single `videovae` node fed BOTH `i2v`
(pre-sampler) AND `decode` (post-sampler). `wrapper_bridge.run_graph`'s
`free_after_use` frees a source only after its LAST consumer, so the 1.38 GB
VideoVAE was pinned in VRAM through the ENTIRE denoise loop -> no activation
headroom -> the audio-conditioned sampler peak spilled to system RAM (NVIDIA
sysmem fallback) -> 223 s/it on the knife-edge (b003 fit at 6.84 s/it, b004 did
not, at byte-identical reported load; nvidia-smi 211 MiB free during the spill).

**FIX (implemented):** split into `videovae_enc` (-> i2v) and `videovae_dec`
(-> decode), so `free_after_use` reclaims ~1.38 GB BEFORE the sampler runs. No
invented API, no download, no quality loss, self-contained in `eng_ltx_av.py`.
The OTR workflow JSON is unchanged -- the engine is already wired into
`otr_scifi_16gb_full.json`; the split is in the engine's INTERNAL render graph
(built in code per render), not an OTR node/widget, so CLAUDE.md s0 does not
apply. Locked by
`tests/test_ltx_audio_in_engine.py::test_ltx_audio_in_videovae_is_split_enc_dec`.

## Rejected / cut (panel-converged + grounded)

- **reserve-inference-VRAM global / `minimum_memory_required` hint** -- the
  pass00 primary. Undocumented + brittle (all 3 panelists + Claude anchor); the
  GGUF `UnetLoaderGGUF` may ignore it. Unnecessary once the split reclaims the
  headroom natively.
- **Q2_K unet** -- lossy on a 22B audio-conditioned model + a ~6.5 GB download
  not on disk. Manual last-resort runbook only.
- **NVIDIA sysmem-fallback policy (D)** -- a driver/Control-Panel setting, not an
  OTR code fix. Operator-doc only.

## Deferred follow-ups (NOT blocking; add only if the split alone is insufficient)

- **B. `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`** in OTR
  prestartup, as cross-beat fragmentation defense (explains the
  b002=25/b003=6.84/b004=223 non-monotonicity). Effect on the already-running
  Desktop app is UNVERIFIED (must be set before CUDA init). `expandable_segments`
  stays operator opt-in (Windows torch 2.10.0+cu130 support unverified). Add only
  if cross-beat variance persists after the split.
- **Ceiling-assert hardening (GPT MUST-FIX #5):** `render_clip` asserts the
  14.5 GB ceiling AFTER post-decode `reclaim_idle_models`, so it does not catch
  the sampler peak. Add NVML peak sampling around `run_graph` for a real guard.

## Validation

- LTX + video-motion subset: **122/122 PASS** incl. the new split-lock test.
- Full suite: only the **5 pre-existing 267a53e workflow-pin fails** (PROVEN
  pre-existing via a stash-baseline run -- they fail identically without this
  change; orthogonal -- they test workflow JSON + capability profiles, not the
  video engine graph). Bug Bible **16 passed / 7 skipped / 3 xfailed** green.
- **GPU speed validation: PENDING the operator's next `ltx_audio_in` render**
  (expect a steady ~10 s/it with no 223 s/it spikes). The GPU was busy with the
  operator's manual render throughout this work, so a headless re-measure was not
  run; the operator validates on the next render.

## Spend

Roundtable: **~$0.38 total** (one truncated probe pass + one full 12k-token
3-model pass: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro).

## Iteration 2 -- VRAM reserve (GPU-VALIDATED 2026-06-26)

The VideoVAE split (bought ~1 GB: usable 12297 -> 13296) was NOT enough on its
own: on the operator's box the **desktop apps eat ~5 GB VRAM** (Chrome, the Claude
app, Snagit, StreamDeck, Edge webviews, ...), so the 22B AV unet (10.5 GB) still
full-loaded with near-zero activation room and stalled at "Model Initializing".
The fundamental issue is capacity: 10.5 GB unet + audio activations must fit
alongside a 5 GB desktop tax on a 16 GB card.

FIX (the deferred B lever, now load-bearing): **force a PARTIAL unet load by
holding ~3 GB free for activations.** Grounded -- ComfyUI's `EXTRA_RESERVED_VRAM`
global (comfy/model_management.py L789-800; the `--reserve-vram` lever) is real
and settable, and the GGUF unet DOES honor partial load (the `ltx_video` b001 beat
already logged "loaded partially; 5455 loaded, 5081 offloaded"). Two surfaces:

* **Engine-scoped (the durable cross-path fix):** `eng_ltx_av._ltx_av_vram_reserve()`
  bumps `EXTRA_RESERVED_VRAM` to `OTR_LTX_AV_RESERVE_VRAM_GB` (default 3.0) around
  `run_graph`, restores in `finally` (exception-safe; never lowers a higher
  operator reserve). Works in the **GUI** (where the operator renders) AND
  headless. Locked by 4 CPU tests.
* **Headless boot belt-and-suspenders:** `--reserve-vram 3` in
  `_otr_overnight_420_boot.cmd` (proven to apply: usable dropped 14737 -> 10860 =
  ~3.9 GB held back).

RESULT (live 30-word all-`ltx_audio_in`, headless): the AV beats render at a
**steady ~11.5 s/it** (b000 ~11.5, b001 ~11.0 -- consistent step-to-step, no
thrash) = ~92 s/clip, vs the old **223 s/it** spill. **~19x faster and
deterministic** -- the 6.84-or-223 lottery is gone. Tune via
`OTR_LTX_AV_RESERVE_VRAM_GB` (lower = faster but less spill margin under heavy
desktop load). Suite: LTX/motion 126/126; Bug Bible 16/7/3; full suite only the 5
pre-existing 267a53e workflow-pin fails.
