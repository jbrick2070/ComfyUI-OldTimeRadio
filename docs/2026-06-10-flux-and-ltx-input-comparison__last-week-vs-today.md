# FLUX + LTX input comparison -- "last week" vs today (operator request, 2026-06-10 evening)

The operator asked for the literal comparison: what the image/video models were
FED in the era the visuals looked right, vs what they are fed today. Everything
below is git-anchored; nothing is from memory.

## 1. The LTX scene-open prompt -- four generations

| Gen | When / where | The literal prompt | Result |
|---|---|---|---|
| 1 | pre-round-4 (`build_request` default, still in code at `render_driver.py` ~L193; and the 3f55ef9 period-radio string) | `"a 1940s radio studio, warm tungsten light, on air"` | The LOVED look (ptw / whispers / molten opens). Concrete nouns, bright key light; per-seed variety. Operator's round-4 catch: the SAME string repeated every beat ("same orange studio blob"). |
| 2 | round 4 (`dae597a`) | `"cinematic establishing shot, {setting terms}, {style} period atmosphere, a vintage radio set glowing in the scene, moody dusk light, gentle film grain, slow cinematic camera drift, no on-screen text"` | Brief-grounded but **"moody dusk light"** baked in -> the dark, HUD-buried Symphony opens. |
| 3 | gap-audit (`c51526b`/round-5 first cut) | `"{narrative logline}, {role clause}, {beat clauses}, drift, no-text"` + era tail | The brief LOGLINE led ("After heated debate, a scientist apologizes...") -- LTX paints narrative prose as red-brown murk (ticking_lab / shattered_silencing opens). |
| 4 | NOW (`379dd41`) | `"a 1940s radio station studio, a vintage radio set glowing warmly, lit dials and tubes, warm tungsten light, {brief setting terms}, {atmosphere} mood, {beat clause}, slow cinematic camera drift, no on-screen text"` + era tail | Gen-1's concrete bright subject FIRST; the brief contributes TERMS only (per-episode variety); beat_intent/arc_phase clauses (per-beat variety); no dusk clause. |

Render-length side: gen-1/2 asks were never capped (the 9.5s synthetic open
asked 238f -> mud); now capped AND floored to the wrapper's decode-proven 169f
band (`OTR_LTX_MAX_FRAMES` / `OTR_LTX_MIN_DECODE_FRAMES`, eng_ltx_video.py).

## 2. FLUX portrait inputs

**Last week (pre `e74a3ce`, the CW-1 teardown commit "unwire legacy FLUX/HuMo/
LTX render path"):** portraits + scene stills came from the legacy visual-plan
composer (preserved verbatim at
`docs/2026-06-10-brief-downstream-gaps/legacy_otr_video_plan_e74a3ce.py.txt`):
`compose_shot_prompt = truncated_portrait_prompt + scene.visual_prompt +
shot_hint + era_tail + style_tail` (5 layers; PASS-1 char portraits reused per
character, PASS-2 per-scene env prompts from the writer LLM). Era tail =
atmosphere_line -> palette top-3 -> v1 lighting (BUG-LOCAL-250: brief-derived,
never a style preset). Renderer: the legacy FLUX batch node, unwired at e74a3ce.

**Today (`HEAD`):** portrait prompts come from
`nodes/otr_meta_brief_image_prompt.py` (NEW in the range, +401 lines):
- one writer-LLM instruction per character at **temperature=0**: "the CHARACTER
  THEMSELVES -- a person with a clearly visible face... ground it in the
  appearance and the story setting";
- consistency guard + **person guard** (`_PERSON_WORDS` regex -- the
  "microphone, no person" round-4 catch) -> template fallback, never a
  prop-only portrait;
- finished AFTER the guards with the same era tail + style tail
  (`finish_visual_prompt`, c51526b), hash stamped after finishing;
- `character_description` (the writer's per-character appearance) feeds the
  chain (435ba0a -- the shared-portrait fix).

**Renderer settings today** (`nodes/_otr_image_engines/flux_gen1.py`, the
image-platform adapter, NEW since e74a3ce): steps **20**, cfg **1.0**,
**832x1216**, env-overridable (`OTR_FLUX_STEPS/CFG/WIDTH/HEIGHT`). The legacy
batch node's exact widget values live at `git show e74a3ce~1` if ever needed --
but tonight's portraits (3 distinct, in-character, faces) are NOT the problem
layer; all three acceptance renders produced healthy portraits.

## 3. The talking-head (HuMo) text inputs -- the open finding

The new prompt observability (round 5) exposed that the M4 creative prompts
NEVER reach the HuMo requests in the live graph (trace rows for cast beats
carry no prompt_source) -- cast beats render on `build_request`'s default
`"a 1940s radio studio, warm tungsten light, on air"` + the portrait init.
That default is gen-1's string -- which is WHY older cast beats often looked
fine. The M4->request seam is the next fix candidate (post-eyeball); the
subject-anchor + person-gate work (F3) lands value the moment that seam is
wired.

## 4. Temps (LLM derivation)

- Image-prompt derivation: temperature=0 (otr_meta_brief_image_prompt).
- ShotLock M4 batch derivation: writer-slot LLM, temperature 0.1 (082a229 --
  HF lane rejects 0.0 with do_sample=True).
- Writer story passes: unchanged this round (the story spine is parked).
