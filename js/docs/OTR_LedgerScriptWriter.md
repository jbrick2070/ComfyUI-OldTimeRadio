<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# LPL Script Writer (v2.0)

The first node of every episode. It picks or takes the story bank, writes the outline, the cast and every line of dialogue with the writer models, and hands the rest of the workflow a script ledger to perform. The My Story rows on this node are where your own story goes.

## What it does

Authors the radio drama outline, cast roster, and beat-by-beat dialogue using local or remote language models, initializing the production ledger. Can fetch science news feeds or build stories from custom premises, deriving act breaks and late-bound episode titles. Change the custom premise, act count, requested character count, seed, or creative and technical model selections.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**Story bank** (`source_bank`) -- a choice from the dropdown.

Story-path SOURCE BANK (multi-modal story schema). Selects which registered story pack supplies the pack-routed creative prompts and which lane the episode runs. scifi_news_pro = the local default sci-fi bank, an LLM-first multipass lane using the configured model slots. Each lane is an INDEPENDENT bank (own pack + bank metadata). The only non-runnable row is '+ Add Your Own' (custom_source_bank) -- picking it FAILS LOUD before any story work (no fallback), with its guide_ref naming the real path: author a bundle under user_packs/source_banks/, run 'otr_check bank <path> --activate' (scripts/otr_check.py, in a git clone), restart, and your bank joins this list as its own entry (contract: apple/EXTENDING.md, which ships with the pack; the longer apple/EXTENDING_OTR.md is in the git tree at https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/EXTENDING_OTR.md). A bank's own default_story_model picks its story pack -- there is no separate pack widget. ROLL: pick 'roll (any eligible bank)' to let the run choose for you, uniformly, from every runnable bank whose lane can build the requested shape. This is INDEPENDENT of the visual_style roll -- rolling one does not roll the other. The choice is recorded in the ledger at meta.bank_roll (selected id, seed, and the exact pool it drew from); set OTR_BANK_SEED to replay a past roll. A pinned source_ref cannot be combined with the roll -- a pinned source belongs to one bank.

### Optional

**Source to adapt** (`source_ref`) -- text; starts as empty.

Optional. A source id for the banks that adapt a text: Shakespeare and Public Domain read it (a 'source:unit' id from their manifests); my_story refuses one; the RSS and original banks ignore it, and the ledger keeps it only as requested_source_ref. Blank picks the bank's default. With Story bank on the roll a non-blank id is refused: a pinned source belongs to one bank.

**Visual style** (`visual_style`) -- a choice from the dropdown.

NO VISIBLE EFFECT ON THE SHIPPED GRAPH. It styles the still and video prompts, and the canonical's three video lanes are procedural -- they draw from the audio and read no still, so this changes the filename and the receipt and nothing you can see. Switch a video lane to a still or diffusion engine first, then it is one of the strongest controls here. Story content is never touched either way. 'roll (any style)' picks one for you, recorded at meta.style_roll.

**`episode_title`** -- text; starts as empty.

Optional episode title override. Stamped at ledger.meta.episode_title so SignalLostVideo picks it up directly without title-chain fallback. Leave blank to let the outline supply a title.

**`custom_premise`** -- text, as many lines as you like; starts as empty.

For My Story, enter your idea here and optionally add characters, plot, setting and who it is by. Leave every creative field blank and the standing premise that ships with the pack is written instead, so a rolled My Story run never refuses. Other banks keep their own source and premise rules.

**My Story - who is in it** (`story_characters`) -- text, as many lines as you like; starts as empty.

MY STORY ONLY -- read when Story bank is my_story, or when its roll lands on My Story. A roll that lands on another bank ignores these boxes; another bank picked BY HAND with text left here stops the run and says so (clear the boxes, pick my_story, or roll). Nothing here is ever cleared for you.

Character names and notes, in ordinary words: 'Ada (she), the lighthouse keeper. Her brother Tom (he), who does not believe her.'

Names you write here are kept exactly as you type them. To set someone's gender, put (she) or (he) right after the name -- (female), (male), (woman) and (man) work too. That is binding: the voice is cast to match. Anyone you leave unmarked is given male or female to fit their part before a voice is chosen -- a unisex name is fine, but every character is voiced and pictured as a man or a woman.

Someone merely mentioned does not become a speaking part. Leave it empty and the cast is invented for you.

**My Story - what happens** (`story_plot`) -- text, as many lines as you like; starts as empty.

MY STORY ONLY. Plot ideas: events, the conflict, a twist, an ending you want. Rough notes are fine.

What you ask for here outranks anything the model would rather invent. What you leave out is invented to fit.

**My Story - where and when** (`story_setting`) -- text, as many lines as you like; starts as empty.

MY STORY ONLY. Setting: place, era, weather, mood, the world it happens in.

This is the story's setting, not its LOOK -- the visual style dropdown chooses the look, and changing it never rewrites your setting.

**My Story - by** (`story_author`) -- text; starts as empty.

MY STORY ONLY. Who the story is by. Put your own name here, or a pen name, or anyone you want it credited to.

The announcer says it out loud and the closing credits print it. Leave it empty and the episode credits 'one of our listeners' -- no name is ever filled in for you.

**Acts** (`act_count`) -- a choice from the dropdown.

Number of acts, 1-7. This is the only knob that shapes episode length, and your pick is always honoured.

More acts means a story with more turns in it -- each act gets its own beat skeleton and its own pass. The episode ends up as long as the story needs; length is reported afterwards, never requested up front.

  1 -> a single scene (trial length)
  2 -> setup, resolution
  3 -> setup, complication, resolution
  7 -> the full arc, through crisis and climax

**`include_act_breaks`** -- on or off; starts as on.

When ON (default), a music interlude beat is placed between every two acts (one fewer than the act count) so the episode breathes between scenes.

When OFF, the acts run as one continuous flow with no interludes.

The sci-fi news lane builds its own running order and does not read this switch.

**Cast size** (`num_characters`) -- a whole number from 1 to 10; starts as `2`.

REQUESTED number of speaking characters (plus ANNOUNCER bookends). 1 = monologue/diary mode. This is a request, not a cap: a story that genuinely needs another voice may use one. The real ceiling is the voice stock, because two characters never share a voice.

**Lemmy cameo** (`lemmy_cameo`) -- a choice from the dropdown.

LEMMY easter-egg cameo -- the genial Cockney communications officer who occasionally joins the cast.

  roll (~11% chance) -- default; LEMMY may appear at random. The roll uses OS entropy and is NOT tied to the seed (BUG-LOCAL-260), so a fixed seed no longer pins him on or off.
  always include -- force LEMMY into the cast this run.
  never include -- keep LEMMY out this run.

'always' consumes one of the num_characters slots, exactly as a natural roll does; 'never' leaves every slot to the cast.

**`story_scaffold`** -- a choice from the dropdown.

How much the radio-drama SCAFFOLD shapes the story. off = a story drawn straight from the news seed (the base prompt only -- no style catalog, no climax-shape grammar, no grounding gate; the writer's own take). on = the news story shaped by ONE of the ~100 radio-drama styles (varied climax + ending + the premise-grounding body gate). auto (default) = follow the OTR_ENABLE_STYLE_GRAMMAR env / its default (ON). on/off override that env for THIS run.

**`min_p`** -- a number from 0.0 to 0.5; starts as `0.05`.

min_p sampling threshold (HuggingFace transformers).

0.05 (default) cuts the long tail of low-probability tokens that produce the occasional off-key word in an otherwise good line on 7B-14B small local LLMs (Mistral-Nemo, Gemma-2, Qwen2.5). Tier 2 fix #17 (2026-05-11) flipped this from 0.0 — preserving an unvalidated baseline is not preservation. 0.0 = disabled.

Aggressive: 0.10. Pairs with the model's own top_p — when both are active the tail cut is the union.

**`repetition_penalty`** -- a number from 1.0 to 1.2; starts as `1.03`.

Repetition penalty for HuggingFace transformers generate.

1.03 (default) is gentle and helps small local LLMs avoid looping on character names / high-frequency tokens in short outputs. Tier 2 fix #17 (2026-05-11) flipped this from 1.0 — preserving an unvalidated baseline is not preservation. 1.0 = disabled. Values above 1.08 commonly damage short generations on the 7B-14B class.

**`max_new_tokens_cap`** -- a whole number from 40 to 400; starts as `200`.

Per-line max_new_tokens ceiling on the composer hot-path.

Default 200 preserves current behavior. The Attempt-1 uses this cap directly; attempt-2 retry uses the full cap. It is a per-CALL decode budget, not a length target -- it was scaled from target_words until 2026-08-14, which made it a token ceiling derived from a word request.

**Writer - story** (`creative_writing_model`) -- a choice from the dropdown.

LLM for the creative/narrative passes (outline, cast, dialogue composer, style picker invention). To use a remote OpenRouter model, set OPENROUTER_API_KEY and pick OpenRouter A/B (see https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/openrouter-setup.md).

Writers -- download size and where each fits:
- Qwen/Qwen3.5-4B: about 9.3 GB; fits mac16-tight nv8-nf4 nv16 nv24
- Qwen/Qwen3.8-27B: about 55.6 GB; fits nv24-nf4
- unsloth/Llama-3.2-3B-Instruct: about 6.4 GB; fits mac16 nv8 nv16 nv24
- mistralai/Mistral-Nemo-Instruct-2407: about 24.5 GB; fits nv16-nf4 nv24-nf4
- google/gemma-4-E2B-it: about 10.2 GB; fits mac16-tight nv8-nf4 nv16 nv24
- google/gemma-4-E4B-it: about 16.0 GB; fits mac16-tight nv16-nf4 nv24
- google/gemma-4-12b-it: about 23.9 GB; fits nv16-nf4 nv24-nf4
- google/gemma-2-2b-it: about 5.2 GB; gated (needs a Hugging Face licence and HF_TOKEN); fits mac16 nv8 nv16 nv24
- comfy_native:gemma4-e2b-it-int8-convrot: about 5.2 GB; run by ComfyUI itself; fit not rated by these tags
- comfy_native:gemma4-e4b-it-int8-convrot: about 8.1 GB; run by ComfyUI itself; fit not rated by these tags
- comfy_native:gemma4-12b-int8-convrot: about 12.1 GB; run by ComfyUI itself; fit not rated by these tags
nv8 / nv16 / nv24: fits an NVIDIA card of that many GB at full precision; -nf4: only as a 4-bit load; mac16: a 16 GB Apple Silicon Mac; -tight: fits with nothing to spare. Sizes are the one-time download, not the memory it uses while running.

**Writer - structure** (`technical_model`) -- a choice from the dropdown.

LLM for the technical/structured passes (JSON validators, GBNF grammar output, reviewer verdicts, cast contract checks, format normalization, news interpreter). Profile/platform-owned baseline: your hardware profile pins it, and a direct headless -Set override is sanctioned and wins over the profile when supplied. Pick a smaller model here when you want Slot 1 != Slot 2 routing for VRAM headroom. To use a remote OpenRouter model, set OPENROUTER_API_KEY and pick OpenRouter A/B (see https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/openrouter-setup.md).

Writers -- download size and where each fits:
- Qwen/Qwen3.5-4B: about 9.3 GB; fits mac16-tight nv8-nf4 nv16 nv24
- Qwen/Qwen3.8-27B: about 55.6 GB; fits nv24-nf4
- unsloth/Llama-3.2-3B-Instruct: about 6.4 GB; fits mac16 nv8 nv16 nv24
- mistralai/Mistral-Nemo-Instruct-2407: about 24.5 GB; fits nv16-nf4 nv24-nf4
- google/gemma-4-E2B-it: about 10.2 GB; fits mac16-tight nv8-nf4 nv16 nv24
- google/gemma-4-E4B-it: about 16.0 GB; fits mac16-tight nv16-nf4 nv24
- google/gemma-4-12b-it: about 23.9 GB; fits nv16-nf4 nv24-nf4
- google/gemma-2-2b-it: about 5.2 GB; gated (needs a Hugging Face licence and HF_TOKEN); fits mac16 nv8 nv16 nv24
- comfy_native:gemma4-e2b-it-int8-convrot: about 5.2 GB; run by ComfyUI itself; fit not rated by these tags
- comfy_native:gemma4-e4b-it-int8-convrot: about 8.1 GB; run by ComfyUI itself; fit not rated by these tags
- comfy_native:gemma4-12b-int8-convrot: about 12.1 GB; run by ComfyUI itself; fit not rated by these tags
nv8 / nv16 / nv24: fits an NVIDIA card of that many GB at full precision; -nf4: only as a 4-bit load; mac16: a 16 GB Apple Silicon Mac; -tight: fits with nothing to spare. Sizes are the one-time download, not the memory it uses while running.

**Cloud writer - OpenRouter A** (`openrouter_slot_a_model`) -- a choice from the dropdown.

OpenRouter model slug bound to the 'openrouter:slot-a' handle (the creative slot). Passive: only used when creative_writing_model is set to 'openrouter:slot-a'. Choices are the curated '~family-latest' aliases (which resolve upstream, so they never go stale) plus your favourites from the cached catalog; run the refresh script or set OTR_OPENROUTER_FULL_CATALOG=1 to browse every cached slug. Shows '(enable OpenRouter)' until OPENROUTER_API_KEY is set. A saved slug is preserved even if absent from a stale cache. See https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/openrouter-setup.md.

**Cloud writer - OpenRouter B** (`openrouter_slot_b_model`) -- a choice from the dropdown.

OpenRouter model slug bound to the 'openrouter:slot-b' handle (the technical slot). Passive: only used when technical_model is set to 'openrouter:slot-b'. Choices are the curated '~family-latest' aliases plus your favourites from the cached catalog; shows '(enable OpenRouter)' until remote is enabled. OTR_OPENROUTER_SLOT_B_REQUIRE_JSON=1 limits the CACHED CATALOG rows to structured-output models; the curated aliases are policy, not catalog discovery, so they are still offered -- check the family before binding one here. See https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/openrouter-setup.md.

**Cloud writer - Comfy Credits A** (`comfy_slot_a_model`) -- a choice from the dropdown.

Comfy Credits model slug bound to the 'comfy:slot-a' handle (the creative slot). Passive: only used when creative_writing_model is set to 'comfy:slot-a'. Choices are the pinned ComfyUI partner-node catalog (Sonnet 5, Luna and Sol are always listed so a saved cloud graph loads). The enable-sentinel remains choices[0]. Credit-billed at generate() when a Comfy API key is present. See https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/comfy-credits-setup.md.

**Cloud writer - Comfy Credits B** (`comfy_slot_b_model`) -- a choice from the dropdown.

Comfy Credits model slug bound to the 'comfy:slot-b' handle (the technical slot). Passive: only used when technical_model is set to 'comfy:slot-b'. Choices are the pinned ComfyUI partner-node catalog (Sonnet 5, Luna and Sol are always listed so a saved cloud graph loads). The enable-sentinel remains choices[0]. Credit-billed at generate() when a Comfy API key is present. See https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/comfy-credits-setup.md.

**Cloud writer - Google A** (`google_api_slot_a_model`) -- a choice from the dropdown.

Google Gemini API model bound to 'google_api:slot-a' (creative slot). Env-only auth: OTR_GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY. Passive until the main model dropdown selects google_api:slot-a. No local fallback.

**Cloud writer - Google B** (`google_api_slot_b_model`) -- a choice from the dropdown.

Google Gemini API model bound to 'google_api:slot-b' (technical slot). Use a structured-output capable text model for JSON passes. Env-only auth, no local fallback.

**`llm_device`** -- a choice from the dropdown.

LLM device. 'default' asks ComfyUI what this machine has and RECORDS what it chose; an explicit device is never second-guessed and still fails loud if it is not there.

**`llm_attn_impl`** -- a choice from the dropdown.

Attention implementation for the transformers lane (the FA2 auto-probe is gone). sdpa = the proven baseline.

**`llm_quant_policy`** -- a choice from the dropdown.

Quantization for the transformers lane. bnb lanes are OFF on ROCm/MPS/CPU tiers (missing bitsandbytes fails loud).

**`llm_vram_ceiling_gb`** -- a number from 0.0 to 96.0; starts as `14.5`.

Pre-download VRAM-fit ceiling (GB). 0 DISABLES the gate (cpu tier only).

**`use_exchange`** -- on or off; starts as off.

Build 4 grouped-exchange dialogue. OFF (default) keeps the per-beat composer; PD1 byte-identity holds. ON groups 2-3 consecutive voiced beats and renders each as one exchange (compose_exchange) with the Build 3 contracts + Build 2 Tier-A check; one block per slot, one repair-by-group, then legacy fallback. ANNOUNCER/MUSIC + trailing singletons keep their pass. Any failure falls back to legacy per beat -- audio is never blocked. Validate VRAM <= 14.5 GB + zero slot drift on a live N=3 run.

**`enable_production_stage3_validators`** -- on or off; starts as off.

OFF (default): no validator runs. ON: a narrow telemetry check on character lines written by the per-line composer (lines the grouped exchange pass wrote skip it). It flags only an empty spoken line or one decorated with transport/speaker markup -- the old pronoun, length, style and content checks are retired. Findings are saved under meta.stage3_findings_per_beat; NOTHING is regenerated, rerolled or rejected, and it costs no extra LLM call. A diagnostic, not a story control.

**`news_briefs_required`** -- on or off; starts as on.

ON (default): typed non-quality source-interpreter failures stay fail-loud. OFF: the legacy branch may degrade to raw news_seed. Rejected/malformed LLM briefs do not reach this switch: they rotate through fresh technical/creative repair passes and end at a validated bank-specific source floor.

**`replay_from`** -- text; starts as empty.

CANONICAL REPLAY (2026-09-02). Empty = normal authorship. A frozen replay bundle directory (scripts/otr_freeze_replay_bundle.py, in a git clone) = re-render THAT episode's ledger through the whole canonical graph as a new episode: no writer, no TTS, no music, no stills minted; same seeds, same audio, only the video phase runs, on the engine the bundle froze (derive a bundle with --derive-engine to replay on another lane). The A/A null.

**Language** (`episode_language`) -- a choice from the dropdown.

The language the EPISODE is in -- spoken dialogue, the announcer, the title card, the captions and the closing credits.

ComfyUI's own knobs stay English, the music prompts stay English, and the machine readouts in the credits (VRAM, CUDA, model ids, seeds) stay English serial numbers.

Anything you type yourself is kept as you typed it: your premise, your title, your story fields.

Off is not a language -- it is off: nothing is stamped and the episode runs exactly as it did before this control existed. English does the same thing and says so on the ledger.

Every language here speaks through Kokoro, and Google TTS (your own Google API key) voices every one of them too. The Shakespeare and Public Domain banks perform the author's own words: in English as written; in another language, a real translator's public-domain text when the corpus holds that scene (config/source_banks/<bank>/translations -- the manifest there is the list), and otherwise the writer model's own translation of the selected passage, which the ledger says so.

**Space saver** (`asset_cleanup`) -- a choice from the dropdown.

What to delete from this episode's folder once it has been published.

off keeps everything.

partial deletes the sound and the pictures (video, audio, images) and keeps every text file: the ledger, the canon, the treatment, the manifests, the captions and the QA reports.

full deletes the whole episode folder -- except that an episode whose rights receipt did not clear keeps its ledger, which holds that receipt.

All three leave otr/obs alone: the published video there is never touched, and nothing is deleted unless that published copy is on disk. An episode with no published copy is never cleaned.

A partial or full episode can no longer be frozen into a replay bundle (the freeze needs the master WAV and the stills). Freeze first, or run it off.

**`gate_in`** -- a connection from another node (`STRING`).

Ordering/validation signal (wire OTR_WorkflowValidator.validation_report).

**Styles to roll** (`style_roll_pool`) -- text; starts as empty.

Read only when Visual style is 'roll (any style)'. Tick the styles to roll among (or type them, separated by commas); none rolls among every style, one is simply that style. Choose from: anime, archival_documentary, cartoon, paper_origami, recur_frac, sci_fi_radio, shakespeare_stage_realism, storybook_engraving, video_art, visual_storybased. Recorded at meta.style_roll.

**Languages to roll** (`language_roll_pool`) -- text; starts as empty.

Read only when Language is 'roll (any language)'. Tick the languages to roll among (or type them, separated by commas); none rolls among every language, one is simply that language. A language this run's voices cannot speak on this machine is left out when you press Run. Choose from: English, Spanish, Portuguese, Italian, French, Hindi, Japanese, Mandarin. Off is never rolled. Recorded at meta.language_roll.

**Banks to roll** (`bank_roll_pool`) -- text; starts as empty.

Read only when Story bank is 'roll (any eligible bank)'. Tick the banks to roll among (or type them, separated by commas); none rolls among every eligible bank, one is simply that bank. Choose from: media_archive, my_story, original, public_domain, scifi_news_pro, shakespeare. Recorded at meta.bank_roll.

## Outputs

- `script_text` (`STRING`)
- `script_json` (`STRING`)
- `news_used` (`STRING`)
- `estimated_minutes` (`INT`)
- `technical_model` (`STRING`)

## Read more

- [BANKS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/BANKS.md)
- [WRITERS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/WRITERS.md)
- [MULTILINGUAL.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MULTILINGUAL.md)
- [CLOUD.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/CLOUD.md)

Class `OTR_LedgerScriptWriter`, category `OldTimeRadio`.
