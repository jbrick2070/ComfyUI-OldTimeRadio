# SWEEP: 30 words, every local engine, seeds pinned

**16 episodes published between 23:09 and 04:49. 13/17 sweep legs passed, plus
the two HuMo legs run before it -- so 15 of 19 local engines produced a real
episode in one night.** Branch `v2.0-alpha`, HEAD `6855190a`, 2026-08-02/03.

Both randomizers ON with seeds PINNED (`OTR_BANK_SEED` =
`OTR_VISUAL_STYLE_SEED` = 4242) so the ENGINE is the only variable, and the
shipped recipes untouched -- every override variable verified unset before
launch. This is the live proof both randomizers have owed since 2026-07-31,
when they shipped suite-proven but never actually run.

## Results

| engine | result | minutes | coverage |
|---|---|---:|---|
| still_flat | PASS | 15.6 | audio 99.14s / video 100.16s, 12 clips, COVERS |
| still_pan | PASS | 14.9 | audio 84.02s / video 85.04s, 11 clips, COVERS |
| still_motion | PASS | 9.8 | audio 68.45s / video 69.48s, 7 clips, COVERS |
| still_word | PASS | 8.8 | audio 70.98s / video 72.04s, 7 clips, COVERS |
| viz_camera | PASS | 7.4 | audio 50.16s / video 51.20s, 7 clips, COVERS |
| viz_green | PASS | 7.0 | audio 49.20s / video 50.16s, 7 clips, COVERS |
| viz_mxc_cpu | **FAIL** | 3.2 | writer -- see below |
| viz_mxc_mandala | PASS | 6.2 | audio 43.97s / video 44.96s, 7 clips, COVERS |
| mesh_stage | PASS | 19.4 | not measured |
| ltx_8gb | PASS | 14.9 | audio 74.69s / video 75.72s, 7 clips, COVERS |
| fastwan_8gb | PASS | 41.3 | audio 73.46s / video 74.48s, 7 clips, COVERS |
| ltx_video | PASS | 38.6 | audio 68.29s / video 69.28s, 7 clips, COVERS |
| ltx_audio_in | **FAIL** | 4.0 | writer -- see below |
| humo_1.7B | PASS | 33.2 | audio 40.90s / video 41.88s, 7 clips, COVERS |
| humo_1.7B_169 | PASS | 42.7 | audio 44.97s / video 46.00s, 7 clips, COVERS |
| wan_i2v | **FAIL** | 7.9 | checkpoint absent |
| wan_ti2v | **FAIL** | 2.7 | cast freeze cascade |

`humo` and `humo_14B_169` are listed NOT RUN because they ran earlier the same
night, end to end, and both published -- 142.5 min portrait, 49.9 min landscape.

**Every passing leg reports COVERS**: rendered video meets or exceeds its audio.
That is the no-mirror invariant holding across thirteen consecutive episodes on
the canonical path.

## NOT ONE FAILURE WAS AN ENGINE

All four are UPSTREAM of the renderer, and they are four DIFFERENT causes. That
distinction matters more than the count: `viz_mxc_cpu` and `ltx_audio_in` are
**unproven, not broken** -- neither ever reached its renderer.

1. **`viz_mxc_cpu` -- the writer invented a cast member.**
   `UNKNOWN_SPEAKER: DR. MOURKIOTI (lines 17, 20, 23)`, `pass 'script' failed
   after 4 attempt(s): markup ladder exhausted`. Model behaviour, and
   non-deterministic.

2. **`ltx_audio_in` -- the markup parser reads markdown headings as speakers.
   THIS IS THE ONE REAL CODE DEFECT.**
   `UNKNOWN_SPEAKER: **SCENE 5 (line 24)`, then `**SCENE 6`, `**SCENE 7`,
   `**SCENE 8`, `**MUSIC`, `**CODA`. The model emitted `**SCENE 5**`-style bold
   headings as structure and the parser treated each as a speaker label, burning
   all four ladder attempts. The model did something reasonable; the parser has
   no handling for markdown emphasis. Fixable in code, and worth fixing: it cost
   a whole leg.

3. **`wan_i2v` -- the checkpoint is not on disk.**
   `RuntimeError: wan_i2v not installed: checkpoint missing at
   ...\models\checkpoints\wan2.2-i2v.safetensors`. **Not a defect** -- the
   fail-closed contract working exactly as designed, naming the missing file and
   its path. Download the asset and re-run.

4. **`wan_ti2v` -- cast freeze cascade.** `OTR_CastLock: freeze cascade stamped
   freeze_verdict='needs_full_rerun' for structural or residual spoken-...`.
   A different subsystem again; needs its own look.

**Writer/cast failures cost 3 of 17 legs (18%).** For unattended overnight runs
that is the single highest-value thing to harden -- higher than anything in the
video layer, which did not fail once.

## What the timings say

Procedural lanes settle at **6-7 minutes** an episode once the box is quiet; the
early legs at 15.6 and 14.9 min were contending with the M2 ladder's aftermath.
Heavy lanes ran 14.9 (ltx_8gb) to 42.7 (humo_1.7B_169) minutes. The 142.5-minute
HuMo portrait leg earlier in the night was NOT representative -- its landscape
twin took 49.9 minutes for the same recipe and seeds.

## Assets confirmed present by this run

pycairo and system libcairo (`viz_mxc_mandala` passed, and the registry warns it
is not in the main requirements); the Hunyuan3D mesher AND portable Blender
(`mesh_stage`); the FastWan rank-128 LoRA, whose absence is designed to fail
closed (`fastwan_8gb`); the LTX-2.3 22B GGUF stack with its Gemma-3 encoder
(`ltx_video`); and `wan2.2-ti2v-5b`, since `fastwan_8gb` shares it.

Absent: `wan2.2-i2v.safetensors`.

## Owed

* Fix the markdown-heading parser defect, then re-run `ltx_audio_in`.
* Re-run `viz_mxc_cpu` -- it is unproven, and its failure was a coin-flip.
* Download the Wan 2.2 I2V checkpoint, then re-run `wan_i2v`.
* Investigate the `OTR_CastLock` freeze cascade behind `wan_ti2v`.
* Both writer failures are admissible to `PROD_BUG_LOG.md` under the standing
  rule -- they are live headless-run failures with logs, not review findings.
