# Driver anchor -- evict the text encoder before the sampler, on DynamicVRAM too (2026-09-02)

Written by the driver (Claude, 5080 window) BEFORE any panel round, from measured facts.
Panel proposes; the driver disposes; every claim below is grounded in a log named here.

## The defect, measured

* 4060 clean room (8 GB, ComfyUI portable v0.34.0, Python 3.13, DynamicVRAM on, pinned
  memory on), Leg C / C2 / C3 (`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`): Klein 4B Q4
  GGUF stills take ~42 min each. Server log, identical every time:
  `Model Flux2TEModel_ prepared for dynamic VRAM loading. 7671MB Staged` ->
  `Requested to load Flux2` -> `0 models unloaded.` ->
  `loaded partially; 0.00 MB usable, 0.00 MB loaded, 2591.65 MB offloaded`, then 120-143 s
  per step. Same under `--disable-dynamic-vram --lowvram --disable-pinned-memory` (C2) and
  under stock flags (C, C3).
* Commit 9b90189a added `free_after_use=True, keep={"unet"}` to flux2_klein, z_image_turbo,
  lumina_image (the video-engine pattern). On the 5080 (16 GB): byte-identical stills, peak
  14.9/14.4/14.6 GB -> 7.9/9.0/9.6 GB. On the 4060 in-process CLASSIC path (aimdo off): the
  encoder is released by the reference drop, the DiT loads 2592 MB resident, 2 steps in
  17.7 s (`4060_probe_residency.log`). On the 4060 SERVER (DynamicVRAM on): unchanged, 0 MB usable
  (Leg C3). So the reference-drop pattern is sufficient on the classic path and NOT on
  DynamicVRAM.
* Why (read in ComfyUI 0.34.2 `comfy/model_management.py`, `comfy/model_patcher.py`,
  `execution.py`): a `ModelPatcherDynamic` keeps its weights in an aimdo VBAR owned by the
  nn.Module (`model.dynamic_vbars`) plus pinned host buffers (`dynamic_pins`). When the pack
  drops its last reference, `LoadedModel` (weakref) is cleaned out of `current_loaded_models`
  but the VBAR pages STAY allocated (an orphan); only aimdo pressure from ANOTHER dynamic
  model reclaims them. The Klein DiT comes from ComfyUI-GGUF's loader, a classic (non-dynamic)
  patcher, so it never applies that pressure: `free_memory` finds an empty registry,
  `get_free_memory` sees ~0.6 GB, and the DiT is loaded with 0 MB usable and streamed from
  host RAM every step. The executor frees pins only under RAM pressure and only ends the
  prompt-model tracker at prompt end; nothing in ComfyUI evicts a dropped dynamic model
  mid-prompt.
* The call that works, measured under the server's exact path (aimdo + `7671MB Staged`),
  `4060_probe_residency_aimdo.log` Phase A: `comfy.model_management.unload_model_and_clones(
  clip.patcher)` WHILE the encoder is still registered -> free 620 MB -> 6998 MB, registry
  empty, nothing else touched. After the reference is gone, `free_memory(64 GiB)`,
  `unload_all_models()`, `cleanup_models_gc()` are all no-ops (registry already empty).
  `unload_model_and_clones` = `free_memory(1e30, device, keep_loaded=<everything but this
  model and its clones>)` (model_management.py:2067-2090); it is the public path ComfyUI's
  own multigpu code uses.
* End to end, same probe, Phase B: the pack's Klein engine (with 9b90189a) plus a crude
  executor-side eviction at the free_after_use drop (`free_memory(1e30)` when the registry
  is non-empty) under aimdo + pinned staging: encoder evicted at the drop, `Requested to
  load Flux2` -> 2592 MB RESIDENT, 2 steps rendered in 9.4 s end to end (the server needed
  ~4 min for the same two steps). The crude form also evicted the DiT after the sampler
  consumed it (harmless there, wrong in general), which is why the shipped form must target
  only the model being dropped. Receipts in this folder: `4060_probe_residency.log` (classic;
  from an earlier revision of the probe -- it predates the A1b line and the Phase B
  monkeypatch, so the committed script does not reproduce it line for line),
  `4060_probe_residency_aimdo.log` (DynamicVRAM; the full run of the committed script),
  `4060_residency_probe.py` (the historical probe, crude Phase B form included).

## The design question (the only one)

WHERE does the explicit unload live, given the executor (`wrapper_bridge.run_graph`) already
knows the exact moment an intermediate's last consumer has run?

A. `run_graph(..., unload_dropped_models=True)` (opt-in): in the `free_after_use` branch,
   before `del results[s]`, if the dropped value is a `ModelPatcher` or carries `.patcher`
   (a `comfy.sd.CLIP`), call `unload_model_and_clones(patcher)`. Precise (only the model
   whose consumers are done), aimdo-safe (partially_unload -> vbar free + pins), leaves
   every other resident model alone (HuMo's fully-resident stack is in `keep` and is never
   dropped). Image engines pass it; video engines unchanged tonight (their 5080 renders are
   not re-measured), with the same latent defect logged for Leg A/B.
B. Engine-level split: run the encode subgraph, call the unload, run the sampler subgraph
   with `external_results`. Explicit, but three copies of the same dance and a second
   `run_graph` call per still.
C. Make `_soft_free` itself call `free_memory(1e30)` when something was dropped. Rejected
   by the driver: it evicts EVERY registered model, including a kept one whose consumers
   are not done (the `_soft_free` docstring already warns this fragments HuMo), and it is
   not opt-in.

Driver's position: A. Flag default False so no existing caller changes behaviour; the three
image engines opt in; the flag is asserted by their render tests next to `free_after_use`.
Proof plan: 5080 same-seed byte-identical probe (three engines) before/after, then the 4060
clean-room Leg C4 under stock flags must show `Requested to load Flux2` followed by a real
resident load and seconds per step, then LTX 2.5 behind it.

## Round 1 outcome (03:30) -- both seats APPROVE A2; conditions adopted

Roster, stated exactly: two Cowork subagents (Opus seat, Sonnet seat), each reading the
anchor, the receipts, the executor and ComfyUI 0.34.2 cold. Codex was mid-edit in this
checkout on the RunPod work and Antigravity is not reachable from this window, so both
kibitz lanes were substituted per the 2026-08-17 standing rule. The driver reviewed the same
files first and independently reached A2 (VAE carries a patcher too, `comfy/sd.py:1083`;
the Z-Image / Lumina `sampling` output is a clone of the DiT, `comfy_extras/
nodes_model_advanced.py:132` + `model_patcher.py:500`).

Corrections the panel made to the anchor, verified by the driver:
* The dynamic VBAR free is reached through `detach(unpatch_weights=True)` ->
  `ModelPatcherDynamic.unpatch_model` (model_patcher.py:2132) -> `partially_unload`, not
  through `model_unload`'s partial branch (skipped because memory_to_free is ~1e30).
* `ModelPatcherDynamic.__del__` calls `detach(unpatch_all=False)`, which never reaches
  `unpatch_model` -- that is exactly why the reference drop leaves an orphan.
* On a CLASSIC patcher the explicit unload adds a whole-encoder device-to-host copy
  (`ModelPatcher.unpatch_model:1155`, `model.to(offload_device)`) that the reference drop
  never paid. Hence the gate below.
* The classic 4060 receipt (`4060_probe_residency.log`) predates the A1b line; the classic
  behaviour of the explicit call is code-reasoned and, with the gate, never executed.

Conditions adopted in the code (A2 as shipped):
1. `run_graph(..., evict_after_use={node ids})`, explicit ids only; validated up front
   (requires free_after_use; ids must exist; ids must not be in keep) -- NAMED errors.
2. The unload is gated on `patcher.is_dynamic()`: classic patchers keep the proven
   reference-drop path, byte for byte.
3. `unload_model_and_clones` is looked up with getattr and wrapped; a missing or raising
   API degrades to the reference drop with a LOUD warning. ComfyUI versions proven: 0.34.0
   (clean room) and 0.34.2 (5080).
4. The three render tests assert the full kwargs contract including `evict_after_use`;
   `tests/test_run_graph_evict_after_use.py` pins the executor semantics with a fake
   `comfy.model_management` (drop-time unload of the named node only, classic skipped,
   the three errors, missing API, no comfy at all).
5. Proof: 5080 same-seed probe reports sha256 + seconds + peak_mib per engine, classic AND
   aimdo-enabled; Leg C4 on the 4060 must show `Requested to load Flux2` followed by a
   resident load (`... MB loaded`, not `0.00 MB usable`) and must render MORE THAN ONE
   still in the same server process (the Sonnet seat's multi-still condition).

## Round 2 outcome (03:50) -- both seats CLEAN on the diff; should-fixes applied

Same two seats, reading the real diff plus ComfyUI 0.34.2 `comfy/sd.py`, `model_patcher.py`,
`model_management.py`. Both CLEAN. Verified by both and by the driver: the duck-type detection
cannot misfire on `comfy.sd.CLIP` (no `detach`, no `model`), on a VAE, on an nn.Module (no
`detach`) or on a tensor (no `model`); `is_dynamic()` exists on both patcher classes
(model_patcher.py:402 / 1791) and is True exactly when the server booted with DynamicVRAM
(main.py rebinds `CoreModelPatcher`); the three validation errors match the tests; the drop
branch cannot fire twice for one node (`node_srcs` is a set, `results[s]` is deleted in the
same iteration); the video engines never reach the helper (no caller passes the keyword).
Applied from round 2: the summary log line fires only when something was unloaded; a
WARNING when a patcher has no `is_dynamic()` (unload anyway, never silently); the run_graph
docstring names the gate; the first executor test now records the sampler in the same trace
as the unload and asserts `["te", "sample-ran"]`, so it pins the DROP-TIME ordering rather
than end-of-graph selectivity. 5080 receipts filed (`docs/ship-audit-2026-09-01/legC/
5080_probe_after2*.json`; `after2_classic` is the cold first-touch run, `after2b_classic_warm`
the warm re-measure): sha256 identical on both paths; classic peaks identical to 9b90189a and
warm times unchanged within run-to-run noise; DynamicVRAM 13.8 / 4.8 / 12.1 s.

## Round 3 outcome (04:05) -- WIRING: both seats REVISE, on the receipts, not the code

Same two seats. Both confirmed the production path end to end (`workflows/otr_canonical.json`
-> `OTR_ImageGenDispatcher` -> `_inprocess_gen_fn` -> `eng.render_image` -> the single
`run_graph` call with `evict_after_use={"clip"}`), that "clip" is the CLIPLoader node in all
three builders, that the z_image reference branch never adds a consumer of "clip" after the
sampler (and is unreachable through `run_graph` in production: `accepts_reference_image =
False`), that no other test stubs `_soft_free` or the bridge for these engines, and that every
number in the commit message matches its JSON or log receipt. What they found, all fixed:
* BLOCKER (Opus seat): `.gitignore:255` `docs/2026-*/` ignores this folder; the four files
  are force-added BY NAME (the rule's own comment sanctions a release-relevant artifact; a
  stray `__pycache__` was removed first so the folder form was never used).
* BLOCKER (Opus seat): two legs labelled C4 in the clean-room log -> the fp8 fallback is C5.
* BLOCKER (Sonnet seat): Leg C4 was cited as a measurement but had not run -> commit message,
  clean-room log and PBUG-20260902-01 all say PENDING, with the pass condition tightened to
  the literal ComfyUI line (`loaded completely; <nonzero> MB usable, <nonzero> MB loaded,
  full load: True`) plus the multi-still condition.
* Receipt wording: "wall time IDENTICAL" -> unchanged within run-to-run noise; the cold
  `after2_classic` run named; the glob `5080_probe_after2*.json`; the clean-room PROBE entry
  says the 9.4 s receipt used the CRUDE all-registry eviction, not the shipped shape; the
  docs probe script header labels itself a historical pre-fix receipt; `4060_` prefixes on
  the log names; the test count re-run on the frozen tree (190); the unreceipted "no
  eviction warnings" clause and the stale file count dropped from the message.

## Round 4 outcome (04:2x) -- CONVERGENCE: no code change; three receipt fixes, then converged

Same two seats, reading the frozen tree. Code, tests and the 5080 numbers: clean on both
seats (the ordering assertion is real, `_soft_free` is unchanged, no leftovers, no fake
module leaks). The Opus seat asked for: this section and the round 3 one (the message cites
a four-round arc, so the anchor must carry all four); the clean-room log's remaining bare log
names (fixed); and the "free 643 MB" figure, which came from the FIRST aimdo run whose log the
A1b re-run overwrote -- now stated as such next to the committed log's after-A1b no-ops. The
staging order was changed to force-add the four files by name after removing `__pycache__`.
Provenance, stated exactly: every round on Opus + Sonnet as Cowork subagents; no Codex or
Antigravity lane in any round; the driver grounded each claim against ComfyUI 0.34.2 and the
real files before adopting it.

## Leg C4 outcome (04:34) -- the shipped shape did NOT release on the server: FAIL

ad6a635f under stock flags on the clean room. The eviction fired (server log: `[OTR graph-exec]
evict_after_use 'clip': unloaded 1 dynamic model patcher(s) ...`) and the DiT still loaded
with `0.00 MB usable`; nvidia-smi read 7896 MiB during sampling, so the encoder's pages never
left the card. The identical call in-process freed them (Phase A1b). The one difference left
between the two runs, read from both logs: the SERVER's torch allocator is cudaMallocAsync
(main.py imports `cuda_malloc` before torch, which sets `PYTORCH_CUDA_ALLOC_CONF=backend:
cudaMallocAsync`); the probe ran the native allocator. Both had the aimdo hooks, NVML
pressure and pinned staging. The probe is being re-run with the async allocator and the
release calls tried one at a time after the unload (`torch.cuda.synchronize`, `empty_cache`,
`soft_empty_cache(force=True)`, `comfy_aimdo.control.analyze()`,
`vbars_reset_watermark_limits()`), then the shipped engine code end to end
(`4060_probe_residency_aimdo_async.log`). Until that lands, ad6a635f is correct in intent,
proven harmless on the 5080 on both paths, and NOT yet the fix on a stock 8 GB server.

## Leg C4b outcome (05:14) -- the card was full of the WRITER, not the encoder

Instrumented bridge on the clone only (registry entries with clone ids, free VRAM, VBAR page
residency before/after the unload; `4060_server_legC4b_instrumented.log` lines 374-375):
`before: free=48MB ... loaded=0MB registry=['Flux2TEModel_:5e976af3:0MB'] vbar_pages=255
resident=0 pinned=0` / `after: free=16MB ... registry=[] resident=0`. The encoder's VBAR held
ZERO resident pages inside the server (the encode streamed through pinned host staging into
a card that was already full), so evicting it could not free what it never held. The in-process
probes all started from an empty card, which is why every one of them "worked". What fills the
card on the server is the WRITER LLM (gemma-4-E2B, transformers, outside ComfyUI's registry):
it composes the still prompts at lines 348-360, and the general path never releases it before
the image stage. The canonical residue freer (`_otr_vram_levers.free_otr_pipeline_residue`:
writer LLM + Bark + surgical detach + flush) is called only by the LTX 2.5 engine and the GGUF
backend in their load preflight; the ghost lane has `_ghost_unload_writer`; the
ImageGenDispatcher has nothing. On 16 GB an E2B writer (~5 GB) co-resides with the stills
unnoticed; on 8 GB it leaves nothing.

Ruling on the two commits: 9b90189a + ad6a635f fixed a real, measured, second-order defect
(encoder co-residency; proven on the 5080 with byte-identical stills and lower peaks) and are
kept. The first-order 8 GB defect is the resident writer, fixed in the dispatcher: one
`free_otr_pipeline_residue(reason="image engine load preflight (<engine>)")` before the first
LOCAL still of a dispatch, the same call the video preflights make. No LLM slot is requested
after the image stage (cast_lock, FreezeCascade, ScriptWriter, ShotLock, bark_lib, llm_policy,
writer_inputs are the only slot users), so there is no reload cost. Wiring conformance to the
existing canonical call: one Sonnet QA on the diff (CLEAN), a 5080 leg with the change, then
Leg C5.

5080 leg (05:22, headless :8000, `otr_soak_still_motion_flux2_klein`, 1 act): the same defect
was live on the 16 GB card. At the first still the 12B writer was resident and the new call
released it -- `free_otr_pipeline_residue (image engine load preflight (z_image_turbo)) OK:
unload_llm, _unload_bark, ... | allocated 7387 -> 6`, `free 14.4 GB after` -- then five
Z-Image stills minted in sequence with no errors and the leg went on into video. Every 5080
still before this change rendered next to a 7.4 GB writer (17 GB of models on 16 GB, paged
by DynamicVRAM). The 8 GB proof is Leg C5 in the clean-room log.

## What is NOT in scope

The recipe (20 steps, guidance 4.0), the encoder choice (bf16 vs fp8), the video engines'
own eviction under DynamicVRAM (same defect class; logged, measured by Leg A/B), and the
`_soft_free` semantics for HuMo.
