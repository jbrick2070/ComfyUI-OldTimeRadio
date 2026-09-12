# Our Stable Audio 3 graph against the publishers' own templates

Driver: Claude Opus 5 (Cowork, 5080), sole judge. Written 2026-09-12 on the
operator's lead: *"sure there are some good proven SA3 template graphs with
prompts"* and *"hopefully that graph diff is gold -- our graph vs true
canonical SA3"*.

Every row below is read from the shipped template JSON on this box
(`.venv/Lib/site-packages/comfyui_workflow_templates_json/templates/`) and from
`nodes/_otr_audio_engines/eng_stable_audio_3.py`. Nothing here is inferred.

## 1. The graphs, side by side

| | OTR today | `audio_stable_audio_3_medium` (distilled) | `..._medium_base` |
|---|---|---|---|
| checkpoint | `stable_audio_3_small_music` | `stable_audio_3_medium` | `stable_audio_3_medium_base` |
| SA text encoder | `t5gemma_b_b_ul2` | `t5gemma_b_b_ul2` | `t5gemma_b_b_ul2` |
| second encoder | none | `qwen3.5_2b_bf16` (for the reprompt) | same |
| prompt written by | our composer | a Qwen `TextGenerate` reprompt | same |
| reprompt category | none | `Music` / `Instrument` / `SFX` / `One-shot` | same |
| negative prompt | a long list | **empty** | **empty** |
| timing conditioning | `ConditioningStableAudio(start, total)` | **no such node** | **no such node** |
| latent | `EmptyLatentAudio(cue seconds)` = 4-12 s | `EmptyLatentAudio(60)` | `EmptyLatentAudio(60)` |
| sampler / scheduler | `dpmpp_3m_sde_gpu` / `exponential` | **`lcm` / `simple`** | **`lcm` / `simple`** |
| steps | 100 | 8 | 50 |
| cfg | 7.0 | 1.0 | 7.0 |
| denoise | 1.0 | 1.0 | 1.0 |

Our sampler/scheduler pair is not from any SA3 graph. It is the recipe in
`audio_stable_audio_example.json`, the **Stable Audio 1.0** template
(`stable-audio-open-1.0.safetensors`, `t5-base`, 50 steps, cfg 4.98, latent
47.6 s). The engine's own comment already recorded the steps/cfg half of this
("ours is the base recipe and then double the steps") and described the sampler
as "Stable Audio's reference", which is true of 1.0 and not of 3.

## 2. What that means, stated carefully

* **cfg 7 on a distilled checkpoint is the classic over-cook.** Comfy-Org pair
  cfg 1 with the distilled member and cfg 7 only with the base. We run the base
  cfg on a member whose name (`small_music`) and shipped recipe are unknown to
  us. A distilled model driven at high guidance is the "deep-fried" failure
  mode.
* **At cfg 1 a negative prompt does nothing** -- there is no unconditional
  branch to steer away from. So the publishers' empty negative is consistent
  with their cfg, and ours is only meaningful because we run cfg 7. If the lcm
  recipe wins, the anti-loop negative added on 2026-09-12 stops being load
  bearing and the loop work has to be re-argued on the prompt alone.
* **No SA3 template uses the timing-conditioning node at all.** Our whole
  `seconds_start` / `seconds_total` window -- BUG-408's fix, and the thing
  widened to a 3x floor on 2026-09-12 -- has no counterpart in either graph.
  That does not make it wrong (it exists in the node set and the model was
  trained with timing conditioning), but it is unvalidated by the publishers.
* **Prompt shape differs in three measurable ways.** Theirs: 140-185
  characters, genre first ("Tropical house track with marimba, steel drums,
  ..."), closing with a vibe clause. Ours: ~390 characters, instruments first,
  genre (`idiom`) in the middle. Their reprompt system prompt says in as many
  words: *"Start with the genre or style"*.
* **They generate long and we generate short.** 60 s latents against our 4-12 s
  cues. A model asked for 12 seconds is being asked for something closer to a
  one-shot than to a track, which is the loop pressure measured on 2026-09-12.

## 3. What is testable tonight, and how

`scripts/otr_music_ab.py` renders through `workflows/otr_canonical.json` and
measures each cue's loopiness, peak, RMS and clipped samples. The sampler,
scheduler, steps and cfg are env-overridable, so three arms need no code:

    python scripts/otr_music_ab.py --banks shakespeare \
      --arms "current:OTR_SA3_SAMPLER=dpmpp_3m_sde_gpu,OTR_SA3_SCHEDULER=exponential,OTR_SA3_STEPS=100,OTR_SA3_CFG=7.0" \
             "lcm_distilled:OTR_SA3_SAMPLER=lcm,OTR_SA3_SCHEDULER=simple,OTR_SA3_STEPS=8,OTR_SA3_CFG=1.0" \
             "lcm_base:OTR_SA3_SAMPLER=lcm,OTR_SA3_SCHEDULER=simple,OTR_SA3_STEPS=50,OTR_SA3_CFG=7.0"

An arm that sets an engine variable boots its own server, because the engine
reads those inside the ComfyUI process and the runner only POSTs to it.

The distilled arm is also **12x cheaper** (8 steps against 100). If it sounds
as good, every episode gets its music back in a fraction of the time, which is
worth knowing on the 8 GB card as well.

## 4. What is NOT testable by env, and is therefore a separate decision

* the genre-first prompt order and the shorter prompt (composer change),
* removing the timing-conditioning node (engine change),
* a longer latent with a slice (engine change, and it costs render time),
* the Qwen reprompt (a new model dependency; our prompt is already detailed,
  so this is the least interesting of the four).

## 5. The bar

The numbers rank; **the operator's ear decides** (`judge it as radio drama`).
A recipe is hard-won here: nothing in this document is a reason to change one
without a listen, and the harness qualifies nothing.
