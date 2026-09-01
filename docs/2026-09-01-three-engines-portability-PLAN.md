# Three-engine portability -- controlling answer

**Status: implementation and local qualification complete. No RunPod or GPU
render was started by this work.**

This document supersedes the earlier planning record. Status words are strict:

- **PROVEN** means a published OTR episode on named hardware.
- **LAB-PROVEN** means a named, isolated recipe produced a receipt-bearing
  artifact on named hardware. It is not a full OTR episode claim.
- **CANDIDATE** means the configuration is plausible but unqualified.

## Answer

| engine | clean-machine install path | honest hardware status |
|---|---|---|
| **HuMo 14B** | `python scripts/otr_fetch_lane_weights.py humo` owns the five exact files the engine resolves, including Kijai's `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`. Revisions, destinations, byte counts, and SHA-256 values are pinned; `.part` files never count. The lane shipped in commit `2cf5626`. | **PROVEN on the RTX 5080 Laptop only.** The named run measured 13.06 GiB VRAM and 27.53 GiB host RAM at 832x480x97. Admit the first remote lab with at least 32 GiB host RAM; 8 GB VRAM is not a target. Ampere, Ada desktop, and RunPod remain candidates. |
| **LTX 2.5** | The provisioner pins ComfyUI-GGUF and ComfyUI-LTXVideo, applies the in-repo Gemma-4/BF16 patch, and the engine semantically verifies the registered loader. `MANUAL_TIERS["ltx25"]` in `scripts/otr_provision.py` is the authoritative five-file weight manifest. After the Lightricks terms click, `docs/RUNPOD_PORTABILITY_LAB.md` is the copy/paste procedure; an executable parity test prevents it drifting from the manifest. | **PROVEN on the RTX 5080 Laptop only.** The shipped Foley path is 832x480 source, 1664x960 two-stage output. The prior Ampere pod reached decode and was SIGKILLed at a 57.7 GiB cgroup limit: a negative receipt for that RAM cap, not for Ampere or RunPod generally. Sixteen 5080 reserve/clamp runs stayed near 15.47-15.60 GiB, a strong warning that the working set does not shrink under pressure, but a larger GPU under a reserve cap is not a physical-8GB surrogate. The physical-4060 LTX 2.5 plan is staged but has no completed receipt, so physical 8 GB remains **UNKNOWN/unqualified**. |
| **MiniMax H3** | Current ComfyUI core supplies the node classes. The complete H3-engine-weight command is `python scripts/otr_fetch_lane_weights.py minimax_h3`: five pinned files, 63,440,965,087 bytes (59.084 GiB). Public bundles never auto-select it. The selected profile also needs its image/music dependencies: `otr_provision.py --profile <h3-profile>` routes those automatically, verifies the explicit H3 lane after it lands, and returns a complete receipt only when both sides are present. | **PROVEN on the RTX 5080 Laptop** under the signed operating standard. Legal 124-model/129-canvas-frame cold receipts measured 6,315 MB FL2VA and 6,678 MB REF2VA absolute VRAM; host RAM was not captured. The supplied NVFP4 encoder is not a Blackwell-only runtime according to [Comfy-Org's model card](https://huggingface.co/Comfy-Org/MiniMax-H3), and the exact encoder independently loaded in the physical Ada RTX 4060 lab receipt below. |

### What the physical RTX 4060 actually proved

Memory was right, but it was a VRAM-lab result rather than a full OTR episode:

- `vram-recipe-lab/eightgb_bench` ran a hash-bound physical RTX 4060 Laptop
  sequence at 864x480x90. Cold/warm/warm peaks were 7.21/6.79/6.79 GiB VRAM;
  all three artifacts passed the machine and media gates. The action and motion
  demos were human-approved.
- The 4060 transfer share also retains playable 864x480x124 Ref2VA dialogue
  artifacts with native audio. That corroborates a legal-length isolated H3
  cell, not the canonical OTR graph.
- There is no physical-4060 `otr_4060_h3_nano` receipt with `RESULT SUCCESS`,
  `obs_publish OK`, and a published episode. The profile and launcher therefore
  remain **draft/unqualified**.

So the correct public sentence is: **H3 is LAB-PROVEN for isolated clips on the
physical RTX 4060; the current full OTR H3 episode is not 4060-proven.**

The remembered LTX Foley episode was on the RTX 5080 Laptop: its server log
names that GPU and records repeated `ltx25_foley_plus` two-stage and Foley
passes. The physical-4060 LTX 2.5 work currently stops at an enrolled plan,
pinned-model import, runtime staging, and headless frontend; no terminal
physical receipt or exported LTX artifact exists yet.

## Stranger / RunPod procedure

1. Start with `scripts/otr_pod_provision.sh`, or inspect exact work with
   `python scripts/otr_provision.py --profile <profile> --list`.
2. HuMo 14B is one-command and ungated. LTX 2.5 requires the terms click and
   the complete five-file manual recipe. H3 is deliberately explicit and
   operator-local: run its five-file command, then rerun the selected-profile
   provisioner so it verifies H3 and completes the profile's image/music lanes.
   A dropdown selection does not download weights.
3. HuMo/LTX profiles using IndexTTS2 need two distinct authorized PCM WAVs.
   `scripts/otr_make_portable_voice_bank.py` preserves non-Index voices,
   installs one male and one female reference under content-addressed names,
   and publishes the bank only after both references pass validation. Its exact
   route-id exception waives only the intentionally absent private Lemmy row;
   every other qualification and route remains fail-closed.
4. Admit the first remote HuMo/LTX lab only with at least 100 GiB effective
   cgroup RAM and 150 GiB free model storage. These are conservative lab-entry
   values with diagnostic/cache headroom, not claimed engine minima.
5. Load `workflows/otr_canonical.json`, publish to `otr/obs/`, retain GPU and
   cgroup receipts, and promote only the exact tuple that succeeds.

5090, 4090, 3090, 3080 Ti, 3080, and rental configurations are **CANDIDATES**,
not refusals and not yet promotions. The first RunPod qualification follows
local green. The later physical-4060 pass should rerun the canonical still and
AnimateDiff lanes as regression coverage while preserving the separate H3 lab
claim. ROCm/MPS video remains UNKNOWN; broad AMD still-image reach should reuse
the existing still engines before adding a new lane.

The public `mkhamra/quibble-h3` workflow and the supplied RTX 3080 10 GB / 32
GB Ref2VA report are useful external lab leads. They do not establish OTR's
profile, legal floor, host-RAM peak, or published-episode result.

## Concurrent file ownership

This change deliberately does not edit or stage the six files owned by the
other Claude window:

- `scripts/otr_machine_matrix.py`
- `config/machine_classes.json`
- `docs/MACHINE_MATRIX.md`
- `scripts/otr_pod_provision.sh`
- `config/profiles/otr_runpod_starter.json`
- `README.md`

Their owner must fold in the same status language. In particular, the current
machine-matrix prose still says that the LTX patch is not public/in-repo and
that HuMo lacks a fetch lane; both claims became stale after this work. No file
should promote HuMo or LTX beyond the 5080 or promote H3 beyond the precisely
scoped 4060 lab evidence above.

## Local verification receipt

The definitive Windows suite completed with **12,608 passed, 121 skipped, 1
expected xfail, and zero failures** in 371.47 seconds. The separate Bug Bible
completed with **22 passed, 26 skipped, and 3 expected xfails**. Ordered
installer/resolver reproductions that caught the two CLI-environment leaks are
green, including 89/89 for H3 followed by the Lemmy/gender resolvers and 52/52
for the HuMo/IndexTTS provisioners followed by the gender resolver.

`scripts/build_variants.py --check` reports **90 variants, 0 failures**.
`scripts/validate_canonical_workflow.py` reports **23 nodes and 60 links**. The
canonical workflow was not edited and remains SHA-256
`A5C1894ED2D68274C625519FDB31095BF768C2499FECC1971EF800BE420ED481`.
`git diff --check` is clean. These are local CPU/static gates; they do not
promote any untested GPU tuple or replace the RunPod and physical-card receipts
defined above.
