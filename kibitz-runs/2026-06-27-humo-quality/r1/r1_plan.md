# HuMo quality + VRAM-fit -- r1 hardened idea set (Codex + Claude; Antigravity pending)

Panel this round: Codex (codex.md, grounded) + Claude anchor (claude_anchor_r1.md).
Antigravity did not return within the window (agy still running) -- fold on its land or a
re-run. Every claim below was checked against the real files; misreads discarded.

## Acceptance gates to put on ANY idea (from Codex MUST/SHOULD; all grounded)
- **Frame-count fit matrix, not a single 49f point.** Production HuMo allows up to 177
  frames (`eng_humo.py` `_HUMO_MAX_FRAMES=177`); the bakeoff measured only 49f. A lever
  is "viable" only if it holds <= ~13.5 GB at 49 / a representative beat / max-safe
  frames. (CONFIRMED.)
- **Define a mouth/teeth acceptance rubric + fixed plosive/vowel clips.** The harness
  today measures only blue-cast + a soft Haar face count (`run_humo_bakeoff.py`); it has
  NO mouth-interior / teeth / lip-closure / audio-sync metric. Add an operator rubric +
  side-by-side gate before claiming any mouth win. (CONFIRMED.)
- **"14B-quality" = non-regression vs the bakeoff clips** `i_14B_single.mp4` /
  `ii_14B_twostage.mp4` (face crop, expression, coat edges must not get worse).
- **Promotion path is in-process `wrapper_bridge.run_graph`, NOT the harness HTTP path.**
  `run_humo_bakeoff.py` submits via `RB._submit_prompt` (HTTP /prompt) -- fine for
  measurement, but any winner must be re-expressed through the in-process graph +
  `workflows/otr_scifi_16gb_full.json` + `config/profiles/16gb_full.json` (still pins
  `humo_1.7B`) in the SAME change. Each idea must name its exact workflow/profile edit.
  (CONFIRMED.)
- **Separate "measurement control" from "acceptable fallback":** the operator rejected
  1.7B for FINAL; keep it as a control only, not a quality fallback.
- **Do-not-promote list:** audio stream present; peak > 13.5 GB at any matrix point; face
  crop regresses; mouth worse than the current 14B.

## Ranked ideas (impact x feasibility, for a shippable 14B-quality talking head)

1. **Allocator-cache probe (CHEAPEST; settle true-vs-cached peak FIRST).** Re-run the
   two-stage leg with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and/or a
   `max_split_size_mb`) in the boot env. nvidia-smi "used" includes the cached reserve
   pool; if true demand is < 13.5 GB the fp8 14B may already be promotable. First step:
   set the env in the runner's boot, add an A/B leg. Risk: may not move the peak (then
   #2 is mandatory). Measure: `run_humo_bakeoff.py` leg ii peak, env on vs off, across
   the frame matrix. [Claude anchor; ASSUMPTION the cache is a large fraction -- verify.]

2. **Quantized HuMo-14B GGUF (the real weight-floor fix).** The fp8 14B UNET is ~14 GB =
   the floor; a Q4_K/Q5 (~7-9 GB) is the only lever that lowers it. Feasibility GATE
   before any bakeoff (Codex MF5): (a) a HuMo-14B GGUF actually exists; (b) add an
   `UnetLoaderGGUF` candidate in `eng_humo._node_candidates` behind an env; (c) prove the
   class on `/object_info`; (d) confirm `WanHuMoImageToVideo` audio-cross-attn survives a
   GGUF-loaded model in a ONE-FRAME smoke; THEN the frame-matrix bakeoff. Highest FIT
   impact; medium-high effort; research lane. (Loader gap CONFIRMED.)

3. **Mouth realism -- two measured probes (gate on the new rubric).**
   (a) no-distill-LoRA ceiling probe: a bakeoff leg with `OTR_HUMO_LORA_NAME=none` +
   ~20-25 steps to see if more compute fixes the mouth (expect higher VRAM/blue -- a
   CEILING probe, not a ship config). (b) higher-res / face-forward INPUT still feeding
   HuMo (it animates the ref portrait) -- cheap, reuses the image pipeline. NOTE (Codex
   CUT): neither is a VRAM fix; keep them on the MOUTH track only.

4. **Newer / dedicated lip-sync model (deepest mouth fix) -- behind a dep probe.** A
   mouth-region second pass (LatentSync / MuseTalk) or a full swap (Sonic / Hallo2 /
   EchoMimic). Codex CUT the broad sweep as "model-shopping": gate it behind ONE
   Windows / Blackwell sm_120 / torch 2.10 / offline dependency probe, and require the
   winner to map into the in-process ALWAYS-SILENT wrapper path (or be rejected).
   Highest mouth impact; highest effort/risk.

## Cut / rejected
- Lower native resolution / shorter clip as a PRIMARY fit fix -- diagnostic only; doesn't
  touch the 14 GB weight floor and risks the preferred look (Codex CUT).
- Raising cfg on the 14B distill to "sharpen" the mouth -- distill is cfg-1.0-trained;
  higher cfg = blue + artifacts, not detail (Claude; eng_humo de-blue history).
- Broad lip-sync model sweep before a dep probe (Codex CUT).

## Judgment log
ACCEPTED (grounded): frame-count fit matrix; mouth acceptance rubric + fixed clips;
promotion via run_graph not HTTP; per-idea workflow/profile mutation; control-vs-fallback
split; do-not-promote list; GGUF feasibility gate; allocator-cache probe (Claude add).
MISREAD (discarded): Codex MF1 "RESULTS.md does not exist" -- it exists at
docs/2026-06-27-humo-bakeoff/RESULTS.md (Codex looked under kibitz-runs); the advice
(cite the real artifact) is already met. PENDING: Antigravity review (not returned).
