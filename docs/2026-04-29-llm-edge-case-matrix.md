# LLM Edge-Case Test Matrix - 2026-04-29

Goal: validate every model in the dropdown plus the BUG-109 retry loop / BUG-100
narration filter / BUG-101 paren strip / BUG-108 dual-ledger fix against the
toughest combinations the dropdown can produce.

## 2026-04-29 update: Gemma 2 family REMOVED

`google/gemma-2-2b-it` failed with a CUDA device-side assert during the
OpenClose Spine inference on this hardware (RTX 5080 Laptop, Blackwell sm_120,
torch 2.10.0 + CUDA 13.0, bitsandbytes 4-bit NF4). The same NF4 quantization
path is what gemma-2-9b-it would use; high confidence it would hit the same
assert. Both Gemma 2 entries removed from the dropdown. Gemma 4 E2B / E4B
retained as the sole Gemma path going forward.

Failure log excerpt:

    Loading LLM model: google/gemma-2-2b-it (quantized=True)
    [StoryOrchestrator] Enabling 4-bit quantization (NF4) for Ultra-Low VRAM
    [StoryOrchestrator] Prompt truncated: 9623 -> 8042 tokens to fit context cap 8192
    [StoryOrchestrator] Starting inference (max_new_tokens=150)...
    [OpenClose] SPINE CHARACTER-DRIVEN failed: CUDA error: device-side assert triggered

## Run order (worst likely failures first)

| # | model_id                                                  | creativity      | target_words | length         | hypothesis                                                                 |
|---|-----------------------------------------------------------|-----------------|--------------|----------------|----------------------------------------------------------------------------|
| 1 | inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)              | maximum chaos   | 350          | short (3 acts) | sister RP fine-tune to Captain-Eris - does it short-output the same way?   |
| 2 | google/gemma-4-E2B-it                                     | balanced        | 350          | short (3 acts) | Gemma 4 effective-2B, newest featherweight, edge-targeted                  |
| 3 | google/gemma-4-E4B-it                                     | balanced        | 350          | short (3 acts) | Gemma 4 effective-4B, edge sweet spot for 16 GB GPUs                       |
| 4 | google/gemma-4-E4B-it                                     | maximum chaos   | 350          | short (3 acts) | Gemma 4 sweet spot under stress - validate BUG-109 retry on a base model   |
| 5 | Qwen/Qwen2.5-14B-Instruct [ALPHA]                         | balanced        | 700          | medium (5 acts)| larger model, alpha - validates [ALPHA] suffix-strip + format gates        |
| 6 | mistralai/Mistral-Nemo-Instruct-2407                      | maximum chaos   | 350          | short (3 acts) | baseline - confirms Mistral handles max chaos cleanly                      |

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

### Run 1: Mag-Mell (EXPERIMENTAL) / maximum chaos / 350 / short
- ...

### Run 2: gemma-4-E2B-it / balanced / 350 / short
- ...

### Run 3: gemma-4-E4B-it / balanced / 350 / short
- ...

### Run 4: gemma-4-E4B-it / maximum chaos / 350 / short
- ...

### Run 5: Qwen 2.5 14B [ALPHA] / balanced / 700 / medium
- ...

### Run 6: Mistral-Nemo / maximum chaos / 350 / short
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
| (gemma-2-2b smoke test, episode never completed)         | gemma-2-2b-it       | balanced       | 350    | -      | -     | FAIL (CUDA)   |

Both Captain-Eris max-chaos runs proved the model refuses to extend on
the BUG-109 retry pass; the no-progress guard breaks out at attempt 1.
This is the failure mode the EXPERIMENTAL tag advertises.

The gemma-2-2b CUDA assert proved the bnb 4-bit NF4 path is not viable
on Blackwell sm_120 with torch 2.10.0 + CUDA 13.0. Drove the Gemma 2
family removal documented at the top of this file.
