# Mac rental punch list -- what is left to close out Apple Silicon testing

Written 2026-09-08. The machine is rented and expires 2026-09-14, so this is
ordered by what buys the most closure per hour, not by tidiness.

## Where things actually stand

**PROVEN on this M4 / 16 GB, with episodes in `otr/obs/`:**
the procgen visualizer lanes (`viz_mxc_cpu`, `viz_green`, `viz_camera`);
all four `still_*` lanes; `sd15` local stills; `ltx_8gb` local video diffusion;
kokoro voices; `stable_audio_3` music; bark; musicgen. Six episodes.

**PROVEN BROKEN, with the reason:** `z_image_turbo` (12.3 GB bf16 needs ~20.4 GiB
in the KSampler; the int8 build hits `aten::_int_mm`, unimplemented on MPS);
`wan_ti2v` / `fastwan_8gb` (fits as GGUF, but an open ComfyUI issue reproduces
Wan temporal corruption on this exact macOS/torch generation, so it would render
garbage); `viz_mxc_mandala` (pycairo has no macOS wheel -- reachable after
`brew install cairo pkg-config`, just not a default).

**FIXED TODAY:** the `tokenizers` boot brick; ffmpeg/ffprobe as declared
dependencies; the SA3 NaN; MPS sub-quadratic attention; the mandala registration
regression; `mono_font` (Consolas is Windows-only); a Windows path literal
creating an 8.7 GB junk directory; the LLM teardown's missing Metal branch; and
the writer-unload ordering (4 loads per episode to 2).

## A. Finish what is in flight

**A1. AnimateDiff to `otr/obs/`.** Running at the time of writing, clip 2 of an
episode, ~133 s/step. It is the last local video lane that can fit 16 GB and has
not published. The machine has already survived far past the three points where
it died before the writer fix, which is the result that matters even if this
particular episode does not land.

**A2. `flux2_klein` to an episode.** It minted a clean 1472x832 still and settled
the "K_M quants garble on MPS" question in the good direction, but never
completed an episode -- I killed that run when swap ate 14 GB of disk. One run
with the `still_*` lanes closes the local-image story. ~8 min/still is the known
cost; budget accordingly or shorten the episode.

## B. Cheap clears -- small changes with known answers

**B1. `spandrel_esrgan` rejects `"mps"` by name** at
`nodes/_otr_upscale_engines/_resolve.py:25`. The entire upscale namespace is
untested on this platform because of one string comparison. Both upscale rows
declare `["cuda","cpu"]`, and the registry comment says MPS is deliberately
absent "until proven" -- so this is a measurement, not a policy fight.

**B2. `_estimate_resident_gb` halves every non-GGUF model** regardless of
`quant_policy` (`nodes/_otr_model_catalog.py:1898`). That is why the Selector
logs `vram_fit=WARN@4.3 GB` for a model that is 8.68 GB on disk and unquantized
here -- 87% of the ceiling reported as 43%. It appears 74-157 times per run.
It did not cause any failure (the gate only refuses at 1.5x the ceiling), but
every one of those lines is wrong by 2x. Same defect family as PBUG-20260829-08,
un-applied to transformers rows.

**B3. `stable_audio_music` needs `stable-audio-tools`**, which is declared in
neither `requirements.txt` nor `pyproject.toml`. Same undeclared-dependency class
as the ffmpeg/ffprobe bug fixed this morning. Either declare it or document the
lane as manual-install.

## C. The one item that needs the other machine

**C1. 5080 proof for the writer-unload ordering fix.** This is the only
shared-code change from today that is unvalidated on CUDA, and its commit says so
in capitals. Three numbers, before and after:

* writer loads per episode (expect 4 -> 2)
* ledger / output identity (must be byte-identical -- same model, same policy,
  same prompts)
* pre-audio allocated memory

The safety argument is that nothing between the removed unloads and the new one
loads a different model. That is what needs confirming; the memory saving is a
bonus, and on CUDA the win is ~25-40 s per avoided reload rather than survival.

## D. Recorded, no action proposed

* **PBUG-20260908-02** -- `free_after_use` is a no-op on MPS. Understood, not
  fixed. A candidate root cause is recorded: ComfyUI's `get_free_memory` on mps
  returns `psutil.virtual_memory().available`, and macOS counts purgeable memory
  as available, so `free_memory` concludes nothing needs freeing and logs
  "0 models unloaded". That is a pointer for a targeted test, not a verified
  cause.
* **PBUG-20260908-03** -- writer teardown, fixed, with a correction filed
  against my own first mechanism.
* **PBUG-20260908-04** -- WITHDRAWN. I claimed `--profile` ignored a profile's
  render/video sections; it does not, and the 5080 window caught it.
* `humo_1.7B`, `humo_1.7B_169`, `lumina_image`, `flux_gen1`, `mesh_stage`,
  `ltx_video`, the three `ltx25_*`, both `minimax_*` -- all NEVER TESTED, and all
  flagged unsafe to attempt at 16 GB by the 60-engine inventory. Leave them.
  An OOM here kills the machine, so "try it and see" is not a free move.

## E. Loose ends with a deadline

**E1. The panel reasoning lives only on this Mac.** `kibitz-runs/` is gitignored,
so today's r1 synthesis -- the one that established why the AnimateDiff clip
cannot be capped -- disappears when the rental ends. Copy the durable part into
`docs/`. (Done: guide section 10.7.)

**E2. A six-commit codex review never returned** after two hours. Dropped; cursor
and Sonnet covered that range and their corrections are already folded in.

## Recommended order

A1 -> A2 -> B1 -> C1, then stop. That leaves every local lane that can run on
16 GB proven end to end, the upscale namespace unblocked, and the single
shared-code risk validated on the machine it could affect.
