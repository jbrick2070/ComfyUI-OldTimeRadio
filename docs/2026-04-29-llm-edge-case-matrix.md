# LLM Edge-Case Test Matrix - 2026-04-29

Goal: validate every model in the dropdown plus the BUG-109 retry loop / BUG-100
narration filter / BUG-101 paren strip / BUG-108 dual-ledger fix against the
toughest combinations the dropdown can produce.

## Run order (worst likely failures first)

| # | model_id                                                  | creativity      | target_words | length         | hypothesis                                                                 |
|---|-----------------------------------------------------------|-----------------|--------------|----------------|----------------------------------------------------------------------------|
| 1 | google/gemma-2-2b-it                                      | balanced        | 350          | short (3 acts) | tiny model on standard short - tests format gates on small context         |
| 2 | google/gemma-2-2b-it                                      | maximum chaos   | 350          | short (3 acts) | small model + chaos - strongest stress on BUG-109 retry loop               |
| 3 | inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)              | maximum chaos   | 350          | short (3 acts) | sister RP fine-tune - does it short-output the way Captain-Eris did?       |
| 4 | google/gemma-2-9b-it                                      | balanced        | 700          | medium (5 acts)| mid-tier model on standard episode - should clean-pass                     |
| 5 | google/gemma-4-E2B-it                                     | balanced        | 350          | short (3 acts) | Gemma 4 effective-2B, newest featherweight, edge-targeted                  |
| 6 | google/gemma-4-E4B-it                                     | balanced        | 350          | short (3 acts) | Gemma 4 effective-4B, edge sweet spot for 16 GB GPUs                       |
| 7 | Qwen/Qwen2.5-14B-Instruct [ALPHA]                         | balanced        | 700          | medium (5 acts)| larger model, alpha - validates [ALPHA] suffix-strip + format gates        |
| 8 | mistralai/Mistral-Nemo-Instruct-2407                      | maximum chaos   | 350          | short (3 acts) | baseline - confirms Mistral handles max chaos cleanly                      |

Common settings for every run:
- num_characters: 3 (or 2 if user prefers; tighter cast surfaces gaps faster)
- style_variant: space opera epic
- target_length: per row above
- arc_enhancer: ON
- optimization_profile: Pro (Ultra Quality)
- cleanup_model_id: auto (use story model)
- genre_flavor: any (vary if you want extra coverage)

## Per-run capture

For each row, record:

| field                          | value | note                                                                |
|--------------------------------|-------|---------------------------------------------------------------------|
| episode_id                     |       | from ledger filename                                                |
| schema_version                 |       | should be l3-2026-04-28                                             |
| commit                         |       | should match git HEAD on v2.0-alpha at run time                     |
| total_word_count               |       | target was X, ratio = actual / target                               |
| word_ratio_pct                 |       | flag if < 80                                                        |
| total_dialogue_lines           |       |                                                                     |
| cast members with 0 lines      |       | from BUG-109b log line                                              |
| empty scenes                   |       | from BUG-109b log line                                              |
| total_episode_dur_s            |       |                                                                     |
| BUG-109 retry fired?           |       | console line "WORD_ENFORCEMENT: UNDER THRESHOLD"                    |
| retry attempts taken           |       | 0, 1, 2, or 3                                                       |
| retry final ratio              |       | post-retry final %                                                  |
| BUG-109b empty-cast log fired? |       | yes/no                                                              |
| audio_gates count              |       | should be 4 (post_bark, post_scene_seq, post_audio_enh, post_assem) |
| radio_bookend_path stamped?    |       | yes/no                                                              |
| HuMo clips rendered            |       | clips[] count                                                       |
| any parse-fatal in console?    |       | yes/no - quote line                                                 |
| any OOM / VRAM warning?        |       | yes/no                                                              |
| verdict                        |       | PASS / DEGRADED-BUT-USABLE / FAIL                                   |

## Results

(fill in below as runs land)

### Run 1: gemma-2-2b-it / balanced / 350 / short
- episode_id:
- ...
- verdict:

### Run 2: gemma-2-2b-it / maximum chaos / 350 / short
- ...

### Run 3: Mag-Mell (EXPERIMENTAL) / maximum chaos / 350 / short
- ...

### Run 4: gemma-2-9b-it / balanced / 700 / medium
- ...

### Run 5: gemma-4-E2B-it / balanced / 350 / short
- ...

### Run 6: gemma-4-E4B-it / balanced / 350 / short
- ...

### Run 7: Qwen 2.5 14B [ALPHA] / balanced / 700 / medium
- ...

### Run 8: Mistral-Nemo / maximum chaos / 350 / short
- ...

## Pre-flight references for Gemma 4

- vLLM Gemma 4 deployment recipes (memory + context limits):
  https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md
- ComfyUI-LLM-Session (GGUF runner inside ComfyUI, recently added Gemma 4):
  https://github.com/kantan-kanto/ComfyUI-LLM-Session
- Unsloth Gemma 4 (E2B) benchmark notebook:
  https://github.com/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb

Before wiring Gemma 4 into the pipeline, cross-check VRAM/context targets
against the vLLM recipe so the OTR LLM context_cap (currently 6144 for
Mistral-Nemo) stays consistent with Gemma 4's allocation profile.

## Known prior results (for comparison)

| episode                                                  | model               | creativity     | target | actual | ratio | verdict       |
|----------------------------------------------------------|---------------------|----------------|--------|--------|-------|---------------|
| signal_lost_wormhole_swallowing_phobos_20260429_101909   | Captain-Eris-Violet | maximum chaos  | 350    | 65     | 18.6% | FAIL          |
| signal_lost_echo_chamber_20260429_105002                 | Captain-Eris-Violet | maximum chaos  | 350    | 23     | 6.5%  | FAIL          |

Both Captain-Eris max-chaos runs proved the model refuses to extend on
the BUG-109 retry pass; the no-progress guard breaks out at attempt 1.
This is the failure mode the EXPERIMENTAL tag advertises.
