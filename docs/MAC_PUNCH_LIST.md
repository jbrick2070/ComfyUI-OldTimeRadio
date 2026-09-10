# Mac rental punch list -- what is left to close out Apple Silicon testing

Written 2026-09-08, closeout added 2026-09-10. The machine is rented and expires
2026-09-14, so this is ordered by what buys the most closure per hour, not by
tidiness. Every status below was checked against the repo and the machine on
2026-09-09, not carried forward.

Where things stand overall is in `MAC_PORTABILITY_GUIDE.md` section 0 and, per
engine, in `MAC_COMPLIANCE_MATRIX.md`. This file lists only what is OPEN, what
closed and how, and what was deliberately left.

## Rental closeout -- 2026-09-10

**The operator accepted the final font replay and closed Mac testing.** The
centered title and credits were explicitly signed off. Canonical replay
published the ninth Mac video in 147.28 seconds with the current font fixes;
the renderer is stopped. Receipt: `2026-09-10-mac-final/README.md`.

The new Lightning attempt failed during VAE decode with a Metal out-of-memory
error, even though its 100 latents were below the measured 136-latent anchor.
It is not counted as another successful Lightning episode. No guard, profile,
recipe, or credits layout was changed to close the font check.

The earlier open items below are retained as the campaign's historical
backlog, not instructions to keep using the rental. CUDA writer-unload proof
and GO_FORWARD row 2.2 need the 5080. Confirm the nine videos and their episode
folders have arrived off the Mac before retiring it; a queued transfer is not
a completed backup. The handoff receipt records that status separately.

## Scoreboard recorded on 2026-09-09

| item | status | evidence |
| --- | --- | --- |
| A1 AnimateDiff to `otr/obs/` | **DONE** 2026-09-09 | `lightning_mac_proof_2_20260909_100958__arch__adlt__none__...mp4` on disk; 23 beats, 2,736 frames, 02:32:34 |
| A2 `flux2_klein` to an episode | **OPEN** | one clean still, no episode; no `fkln` slug in `otr/obs/` |
| B1 `spandrel_esrgan` rejects `"mps"` | **DONE, pending review** (uncommitted at time of writing) | bit-exact receipt, `tests/test_upscale_mps_receipt.py` |
| B2 `_estimate_resident_gb` halves non-GGUF models | **OPEN** | `_otr_model_catalog.py:1898` still `/ 2.0` |
| B3 `stable_audio_music` needs undeclared `stable-audio-tools` | **OPEN** | not in `requirements.txt` or `pyproject.toml` |
| C1 5080 proof of the writer-unload fix | **OPEN -- needs the other machine** | no CUDA receipt in the log |
| E1 panel reasoning copied into `docs/` | DONE | guide section 7.5 |
| E2 six-commit codex review | dropped | -- |

## The most valuable thing this machine produced is not a Mac fix

Read this before the lists, because it changes what the rental was for.

The writer-unload ordering bug, ffmpeg 9's removed `-vsync`, and the `tokenizers`
boot brick are all LATENT ON EVERY PLATFORM. The 5080 has been paying two
needless `from_pretrained` + warmup cycles per episode -- roughly 25-40 s each --
for as long as that ordering has been wrong. Nothing about it is Apple Silicon.

What the Mac has is NO SLACK: no separate VRAM to offload into, and an OOM that
REBOOTS THE MACHINE instead of raising. Shared waste that is invisible on a
discrete card becomes fatal here, and fatal is easy to find. That is a better
argument for keeping a low-slack machine in the loop than any portability result
this rental produced -- and it suggests the next such bug is found the same way,
by running the real pipeline somewhere unforgiving, not by auditing for
portability.

## A. Finish what is in flight

**A1. AnimateDiff to `otr/obs/` -- DONE 2026-09-09.** `animatediff15_lightning_video`
published a complete 23-beat episode through the real OTR adapter path, and its
`device_backends` row earned `mps` on that receipt. Guide section 7 has the
episode, the recipe (hold 3, ft-mse decoder, aligned source counts) and the
setup.

Two things the green tick does not cover, and they stay open by design rather
than as punch-list items:

* **The adaptive-hold guard has never fired under live fire.** Every beat in
  the proof episode fell under the 136-latent ceiling. A synthetic long beat
  cannot be faked -- `validate_coverage_plan` refuses it, correctly -- so the
  only honest test is a real episode whose audio produces a ~20-second beat.
  With hold 3 as the default the guard fires only past roughly T=480-500.
  Closed with that gap stated (PBUG-20260909-01).
* **The 136-latent ceiling is one measured bracket** (136 survived, 160
  rebooted), scaled by RAM and canvas. Not a memory model. A second point in
  either direction would be worth more than any argument about it.

**A2. `flux2_klein` to an episode -- OPEN.** It minted a clean 1472x832 still
and settled the "K_M quants garble on MPS" question in the good direction, but
never completed an episode -- the run was killed when swap ate 14 GB of disk.
The reason it swaps is now known (PBUG-20260908-02: its 7.67 GB text encoder is
not evicted on Metal, so all 10.99 GB is resident at once), and its row is still
`["cuda"]`. ~8 min/still is the known cost; one run with the `still_*` lanes
closes the local-image story on this platform. Budget accordingly or shorten the
episode. Note the compliance matrix already lists it as PROVEN on the strength
of the still; an episode is what would make that verdict match this repo's
usual standard.

## B. Cheap clears -- small changes with known answers

**B1. `spandrel_esrgan` rejects `"mps"` by name -- DONE, pending review.**
`_resolve.resolve_device` accepts `mps` (a named refusal, not a CPU
fall-through, when Metal is absent), both upscale rows declare
`["cuda","cpu","mps"]`, and `tests/test_upscale_mps_receipt.py` re-takes the
receipt on any Mac that has the weights. The receipt, 2026-09-09 on this M4:
RealESRGAN_x2plus via spandrel 0.4.2, 128x128 -> 256x256 on `mps` in 0.51 s,
all outputs finite, **max |mps - cpu| = 0.00000**. The parity number is the
load-bearing half -- a forward that merely returns proves nothing, because a
wrong kernel returns too, and this project has already shipped one silently
wrong Metal path (sub-quadratic attention). The change is in the working tree
uncommitted at the time of writing; no episode has rendered through the upscale
stage on a Mac yet.

**B2. `_estimate_resident_gb` halves every non-GGUF model -- OPEN.**
`nodes/_otr_model_catalog.py:1898` still divides by 2.0 regardless of
`quant_policy`. That is why the Selector logs `vram_fit=WARN@4.3 GB` for a
model that is 8.68 GB on disk and unquantized here -- 87% of the ceiling
reported as 43%. It appears 74-157 times per run. It did not cause any failure
(the gate only refuses at 1.5x the ceiling), but every one of those lines is
wrong by 2x. Same defect family as PBUG-20260829-08, un-applied to transformers
rows.

**B3. `stable_audio_music` needs `stable-audio-tools` -- OPEN.**
`eng_stable_audio.py` imports it; neither `requirements.txt` nor
`pyproject.toml` declares it. Same undeclared-dependency class as the
ffmpeg/ffprobe bug. Either declare it or document the lane as manual-install.
(Its weights are gated as well -- friction, not failure -- and the lane has
never been run on a Mac; the matrix rates it LIKELY.)

## C. The one item that needs the other machine

**C1. 5080 proof for the writer-unload ordering fix -- OPEN.** This is the only
shared-code change from the rental that is unvalidated on CUDA, and its commit
says so in capitals. Three numbers, before and after:

* writer loads per episode (expect 4 -> 2)
* ledger / output identity (must be byte-identical -- same model, same policy,
  same prompts)
* pre-audio allocated memory

The safety argument is that nothing between the removed unloads and the new one
loads a different model. That is what needs confirming; the memory saving is a
bonus, and on CUDA the win is ~25-40 s per avoided reload rather than survival.

## D. Recorded, no action proposed

* **PBUG-20260908-02** -- `free_after_use` is a no-op on MPS. Understood, not
  fixed, and now COMPENSATED FOR rather than solved: the unified-memory guard
  charges every artifact as concurrently resident on Metal because of it. A
  candidate root cause is recorded (ComfyUI's `get_free_memory` on mps returns
  `psutil.virtual_memory().available`, and macOS counts purgeable memory as
  available, so `free_memory` concludes nothing needs freeing and logs "0 models
  unloaded"). That is a pointer for a targeted test, not a verified cause.
* **PBUG-20260908-03** -- writer teardown, fixed, with a correction filed
  against my own first mechanism.
* **PBUG-20260908-04** -- WITHDRAWN. I claimed `--profile` ignored a profile's
  render/video sections; it does not, and the 5080 window caught it.
* **PBUG-20260908-05** -- the attention forcing on MPS costs ~1.15x end to end,
  not the ~14x first claimed. The forcing stays; no scoped fix is worth doing.
* **The engines nobody has run here** -- `humo_1.7B`, `humo_1.7B_169`,
  `lumina_image`, `flux_gen1`, `mesh_stage`, `ltx_video`, the three `ltx25_*`,
  both `minimax_*`, and the rest of the matrix's LIKELY and OOM RISK rows.

  **EVERY "UNSAFE AT 16 GB" JUDGEMENT ON THOSE IS A SIZE ESTIMATE, NOT A
  MEASUREMENT, and it must be read as one.** It comes from summing declared
  artifact sizes against a 16 GB ceiling. That is exactly the shape of reasoning
  behind every `device_backends: ["cuda"]` row this rental found to be untested
  policy rather than a hardware fact -- `ltx_8gb` was "cuda-only" until it
  published two episodes here, `bark` was `["cuda","cpu"]` until it ran on
  `mps`, and AnimateDiff was "blocked" until the day after it was written down.
  An estimate that says "probably will not fit" is not the same claim as "was
  tried and failed", and neither this file nor the matrix should let the two
  blur.

  The reason to leave them anyway is different and still good: an OOM on unified
  memory reboots the machine, so "try it and see" costs a reboot rather than a
  stack trace. That is a decision about the COST OF BEING WRONG, not confidence
  that the estimate is right. If one of these is ever wanted, the honest route
  is to check its concurrently-resident footprint (every artifact, since nothing
  is evicted on Metal) against `motion_common.unified_memory_budget_mb()` first
  and let the guard refuse it, not to launch it and watch.

* **The cloud POST is untested.** The local half of every cloud video lane is
  proven on this Mac with a real mp4 standing in for the provider; the POST
  itself needs a logged-in ComfyUI Desktop session, which a headless `main.py`
  does not have. Not a Metal question, and not worth a paid test from here
  (matrix, "Cloud lanes").

## E. Loose ends with a deadline

**E1. The panel reasoning lives only on this Mac -- DONE.** `kibitz-runs/` is
gitignored, so the r1 synthesis that established why the AnimateDiff clip cannot
be capped would have disappeared when the rental ends. The durable part is guide
section 7.5.

**E2. A six-commit codex review never returned** after two hours. Dropped;
cursor and Sonnet covered that range and their corrections are already folded
in.

## Recommended order

A2 -> B2 -> B3 -> C1 (on the 5080), then stop. A1 and B1 are closed. That would
leave every local lane that can run on 16 GB proven end to end -- Klein
included -- the two lying numbers fixed, and the single shared-code risk
validated on the machine it could affect.
