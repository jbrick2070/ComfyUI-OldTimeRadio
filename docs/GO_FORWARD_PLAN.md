# OTR Go-Forward Plan

**ONLY UNFINISHED WORK BELONGS HERE.** When work finishes its receipt moves to
[HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence folder and the row leaves this
page. A finished prerequisite earns **one clause inside the row that still needs
it** -- never a receipt, never a measurement write-up, never a struck-through or
"SHIPPED" row. The test is one question: *does a row still in this file stop
making sense without that sentence?* No -> cut it.

**THIS FILE IS ARC AND CODE. TESTING IS NOT IN IT** (operator, 2026-09-12:
*"let's just do the coding and arcs first, let's not even talk testing yet"*).
Decide first, build second -- and when both are empty, section 6 at the bottom
is what you have earned. A deferred arc is deferred coding,
and a gate on evidence a leg produces is not a valid deferral either, because
the legs run last: a row that can only be settled by live evidence is settled
WITHOUT it or cut with the reason written in. **The only things that genuinely
defer** are a row blocked on an operator ruling (section 3) and a row
deliberately cut with its reason. Apply that test whenever a row claims to be
blocked.

Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md) first;
this file does not restate them. **For what has already happened -- commits,
measurements, receipts -- read [HANDOFF_LOG](HANDOFF_LOG.md), newest entry
first.**

## 0. The bar

> **"As long as it doesn't crash when it's not supposed to."** -- operator,
> 2026-09-11. Exactness is not the goal: *"I'm not expecting anything exact."*

**CRASH-CLASS AND DURABILITY-CLASS DEFECTS ARE THE WORK** -- an uncaught
exception, a live asset written where a sweeper can delete it, an identity that
silently resolves outside its episode, and **a machine that silently renders a
configuration we have already proven wrong.** Rows below use the phrase "not
crash-class" against this definition. Aesthetic drift is closed and is not work;
see [ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md).

## 1. ARC -- settle these before writing the code

Every row here has more than one defensible answer, so it gets its round, its
measurement or its ruling BEFORE code. An arc costs a wait, not a budget.

### A1. What the canonical ships for writer + quant + ceiling on an 8 GB card

The device half of system-independence landed 2026-09-12 (`8017a07e`): the
canonical no longer names a vendor, and all four device widgets now read
`default` / `cpu`. **The sizing half did not.** The canonical still carries
`Qwen/Qwen3.5-4B` + `llm_quant_policy "none"` + `vram_ceiling_gb 10.0`, which is
byte-for-byte the `otr_mac_mps` triple -- and 10.0 is unique to that one profile
across all 118. On an 8 GB NVIDIA card `none` means bf16, so 8.68 GB is
downloaded and then moved onto the card in one shot
(`_otr_model_loader.py`, `if quant_config is None and max_memory is None:
model = model.to(device)`), with no offload rescue. The gate says WARN and lets
it through.

**What a contrarian round settled, and it removed two options:**
* The 8 GB stranger already HAS a shipped answer -- `workflows/variants/` holds
  94 generated graphs including `otr_nvidia_8gb_haunted`, and they ship in the
  registry bundle. ComfyUI's template browser globs one directory level, so it
  cannot list them. The README said the folder was empty; that is corrected now,
  and it may be the whole fix.
* An `auto` value on the quant combo is the WORST option, not the obvious one.
  A frozen `LLMRuntimePolicy` feeds `cache_key()`, so a resolving sentinel either
  leaks into cache identity or needs a second resolution layer -- and the
  loader's runtime bitsandbytes probe is deliberate (`01845aad`: "remove the
  policy, test what actually works"). A sentinel puts the guess back one layer up.

**What is still forked:** leave the canonical as a 16 GB-class graph and point
8 GB users at the variant, or retune it to the smallest common denominator.
Needs his call, because it trades a stranger's first run against the writer
quality on the machine that renders the dailies.

### A2. The `nv8` fit tag is computed on a halving the canonical's own setting invalidates

`_otr_model_catalog.py` halves the download size in TWO places -- once in the
gate's estimator and once in `fit_tags_for`, which mints the tag. The canonical's
saved widget string literally reads
`'Qwen/Qwen3.5-4B (8.7 GB, mac16-tight nv8 nv16 nv24)'`: it advertises that it
fits a 7.0 GiB NVIDIA budget, on an assumption of NF4 that its own
`quant_policy "none"` rules out. The label is the only thing a stranger reads
before pressing Queue.

**Why this is an arc and not a fix.** `fit_tags_for` runs at INPUT_TYPES time,
before any widget value exists, so it structurally CANNOT read the quant policy.
Assume one policy, emit both, or drop the tag -- three defensible answers. And
honest tags change the label, which no longer matches the saved
`widgets_values`, so `tests/test_saved_workflow_model_values_resolve.py` goes red
and 94 variants regenerate. Measured: un-halving the GATE alone flips ZERO
profiles' verdict tier (`_FAIL_RATIO` 1.5 is wider than the 1.24 error), so that
half is a truth fix with no safety effect. The tag is where the behaviour is.

### A3. `MODEL_ASSET_INDEX.md` keys rows by filename, not by registered engine id

Consequences measured 2026-09-12: `still_flat` / `still_motion` / `still_pan` /
`still_word` have NO ROW AT ALL (50 profile selections between them) because they
live in `cheap_families.py` and the generator globs `eng_*.py`. Three registered
LTX 2.5 engines collapse into one row flagged "not declared in code -- verify",
a false negative caused by an import style the scanner's regex misses, hiding 19
selections of a GATED 22 GiB family. `bark` carries the same false flag because
`suno` is not in a hardcoded publisher allowlist. The fork is what to render when
one file implements six engines, which is why this is not a glob widening.

## 2. CODE -- the design is settled, build it

### C1. Five shipped profiles cannot load their own configured writer

Measured with the real gate on 2026-09-12: `8gb_lite`, `otr_8gb_wan`,
`otr_8gb_ltx`, `otr_8gb_fastwan` and `otr_4060_12b_gguf_offload` all pair
`creative_model: google/gemma-4-12b-it` with `vram_ceiling_gb: 6.8` and
`quant_policy: none`. `check_vram_fit` returns **FAIL at 11.95 GB against 6.8,
a 1.76x ratio**, and a FAIL verdict is what `request_slot` raises on.

**The cause is a lane that no longer exists, not five bad sizing choices.**
`GGUF_ROWS` in `nodes/_otr_gguf_backend.py` is **EMPTY** -- measured, and
`gguf_row_for_repo` raises for every id -- so the 12B has no quantized route
today and every one of these profiles prices as `provider: local`, i.e. the
bf16 transformers lane. `otr_4060_12b_gguf_offload` is named for the exact
mechanism that is gone. Same root cause explains a contradiction in the
generated writer table, which still marks `google/gemma-4-12b-it` **proven** on
8 GB NVIDIA at a 23.9 GiB download: that receipt was earned through the GGUF
lane before it was emptied. So the choice is to restore a GGUF row for the 12B
or to move these five profiles to a writer that fits unquantized -- and the
receipt in the writer table needs whichever answer is picked recorded against it.

### C2. `machine_classes.json` is missing the `ltx_8gb` receipt that `dropdown_matrix.json` spends

`docs/dropdown_matrix.json` marks `ltx_8gb` **proven** on 8 GB NVIDIA;
`config/machine_classes.json`'s `engine_evidence` carries no such row -- only an
RTX A4500 20 GB and the Mac mini M4. `docs/4060_DRILL_LOG.md` around lines
4579-4834 looks like the real 4060 receipt it was harvested from, so the fix is
probably to add the row rather than to retract the verdict. Two hand-curated
files feed two generated docs and nothing enforces agreement between them.

### C3. Put the SD 1.5 checkpoint in the visual-asset manifest -- it is what the Mac ladder is waiting on

**This is the single highest-leverage item for the three Mac graphs the operator
described**, and the reason is one file. He asked for a procgen JSON, a stills
JSON and an LTX 0.9.8 JSON, "all auto download and non gated". Measured
2026-09-12, only the first is:

| the Mac graph he wants | what it actually costs today |
|---|---|
| procgen (`viz_*`) | nothing. Zero weights, zero downloads. Already true. |
| stills (`still_*`) | one hand-fetched 2 GB `sd15` checkpoint |
| LTX 0.9.8 (`ltx_8gb`) | LTX's own 16.1 GiB self-fetches, but the lane is image-to-video and consumes a still, so it ALSO needs that same 2 GB by hand |

So one file stands between him and two of the three. `sd15` is ungated and
public (`Comfy-Org/stable-diffusion-v1-5-archive`); nothing about it needs to be
manual.

**Why it is a code row and not a one-line manifest entry.**
`ensure_prompt_visual_assets` is SELECTION-DRIVEN -- it plans from the submitted
prompt and intersects with `_COVERED`, so adding a row costs nobody who does not
pick the engine, which is the property that makes this safe. But the function
carries per-engine imports and passes them positionally into `native_requests`
(`zimage=`, `ltx=`, `sa3=`), so a fourth engine touches `MANIFEST`, `_COVERED`,
that import block and that signature. The one real design question is path
resolution: `sd15._resolve_ckpt_name()` supports style-specific checkpoints and
an env override, so the manifest must fetch the DEFAULT file without claiming to
satisfy a pack checkpoint the user chose instead.

`spandrel_esrgan`'s 67 MB upscale model is the same shape and the same fix, and
it is the second of the two hand-fetches in the "Real video diffusion" row of
the README's cheapest-setups table.

### C4. `cpu_floor` has no local writer it is allowed to use

Measured 2026-09-12. It is the ONLY profile whose `lane_allowlist` excludes
`transformers` -- it permits `gguf`, `openrouter`, `comfy_credits`,
`google_api`. Its `creative_model` is `unsloth/Llama-3.2-3B-Instruct`, a
transformers row. And `GGUF_ROWS` is empty, so its one local lane offers nothing.
A CPU-only user therefore has no local route at all: the two that remain are
paid. Either restore a GGUF row, or add `transformers` to that allowlist and let
the 3B run on CPU. Same empty-lane fact is what makes C1's five profiles fail, so
the two rows probably share one answer.

### C5. Small, named, and each takes minutes

* `Comfy-Org/flux2-klein` 307-redirects to `Comfy-Org/vae-text-encorder-for-flux-klein-4b`.
  The pinned SHA still resolves through the redirect, so this is cosmetic --
  fix `docs/RUNPOD_INSTALL.md` and `scripts/otr_provision.py` next time either is open.
* `elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF` is GATED (confirmed live) and flagged
  as such in the provisioner's data, but no prose doc says so. RUNPOD_INSTALL's
  "one terms click" heading undersells a second owner's accept-click.
* The `--machine amd` selector still plans `flux2_klein` while both AMD profiles
  ship `z_image_turbo` (`b1f372a9`). The selector and the profiles disagree about
  the same hardware.
* `tests/test_full_workflow_v2_audio_wiring.py:194` and
  `tests/test_workflow_json_guardrails.py:768` still pin `cuda`; the canonical
  now saves `default`.
* `nodes/_otr_shared/device_options.py` has no tests. It is the module every
  device widget now routes through.
* Multi-GPU silent wrong device: CastLock stamps `cuda:1`, and
  `_voice_device_from_ledger` hands back `cuda`.
* `nodes/_otr_shared/device_options.py::vendor()` still has ZERO callers. Wire it
  or write the row that says what it waits on -- A1 is that row today.

## 3. Blocked on the operator -- each unblocks with one word

**He answered all sixteen on 2026-09-12 and left to do other things, with:**
*"i cant listen until i get back thats ok if thats the one thing waiting, dont
stall for me, keep going and do a bunch of testing, we can fix when i get back."*
So the rows below are what SURVIVED his answers. The twelve rows that closed are
gone from this file by its own rule; the receipt in HANDOFF_LOG carries them.

### Waiting on his ear, and nothing else

* **The IndexTTS2 hang fix is WRITTEN AND HELD, because shipping it demotes
  Lemmy.** Both protocol reads in `nodes/_otr_audio_engines/eng_indextts2.py`
  are bare `proc.stdout.readline()` with no timeout, on the shipped default
  character-voice engine. A stalled worker never returns, so the
  `finally: self._teardown(adapter)` never runs and ComfyUI plus an orphaned
  worker hold VRAM forever with nothing in the log -- indistinguishable from a
  slow render. `eng_dia` and `eng_chatterbox` already route the identical read
  through `_otr_sidecar.read_protocol_line`; this engine simply never imported
  it, so the fix is to do exactly what its two siblings do.
  **Why it is held.** That file is one of three in
  `_otr_voice_route.RUNTIME_FINGERPRINT_SOURCES["indextts2"]`, hashed whole, so
  ANY byte change moves the fingerprint. Measured 2026-09-12: the qualified
  value is `d47779386ce91209` and the fix makes it `c78934682057fc65`, which
  fails `test_the_shipped_lemmy_route_is_selected_again` and un-selects the
  shipped Lemmy route. The demotion is graceful -- the row takes the ordinary
  draw and the episode still publishes -- but the cameo he qualified by ear
  goes away, and the test says plainly: *"re-audition and re-record, do not
  hand-edit the fingerprint"*.
  **What unblocks it: one word from him.** Either "ship it and I will
  re-audition Lemmy", or "hold it". The trade is a certain loss of a cameo he
  likes against protection from a rare hang. The patch is reconstructible in
  minutes from the sibling engines; nothing else is waiting on it.

* **The techno and house cues, re-rendered.** He judged `3_media_archive`
  (jazz) and `4_original` (salsa) RIGHT, and `1_scifi_news` (Detroit techno)
  and `2_public_domain` (Chicago house) WRONG -- *"supposed to be techno whats
  wrong??"* and *"supposed to be house very wrong"*. Three sustained-music
  instructions were reaching a dance cue: a pad closing both electronic
  palettes, the neutral floor word "atmospheric" LEADING the row, and an
  orchestral per-cue arc ("a rising overture", "resolving to a warm held
  chord") that was appended OUTSIDE the guard already written to keep
  orchestral language off a rhythmic palette. All three are fixed and the fix
  is scoped by a new `Palette.groove_arc` flag to exactly the two banks he
  rejected -- **jazz and salsa compose byte-identically to what he approved**,
  because his ear on shipped output outranks a tidier rule (codex refuted the
  wider blast radius and was right). **Unblocks with a listen to the new
  cues in `otr/obs/`.**
* **Whether the causal story is actually right.** Codex's standing objection,
  folded rather than argued with: the operator's verdict is on OUTPUT and the
  diagnosis is a reading of the PROMPT, with no seed-matched A/B between them.
  The untested competitors are the checkpoint (post-trained SA3 ignores cfg and
  negatives) and the sampler. The comments say hypothesis, not fact. **Unblocks
  with the same listen.**
* **SETTLED 2026-09-12 -- the anime checkpoint is IN.** His verdict on
  `the_clanking_chains_20260912_153041__anim__stfl__sd15__...` was one word:
  *"perfect"*. `Counterfeit-V3.0_fp16.safetensors` is fetched, the anime pack
  names it, and the server log records `[sd15] minted still 768x432 ...
  ckpt=Counterfeit-V3.0_fp16.safetensors` with `OTR_SD15_CKPT` unset -- so the
  name came from the pack through the style resolver. **His ear has ruled;
  this row does not get re-asked and the lane is not benched again.**
  The one thing left is bookkeeping, not a question: `config/profiles/
  otr_sd15_stills.json` is still `status: "draft"` because it was written
  minutes before the leg. It wants a couple more episodes on other styles
  before it claims `shipping`, and it is NOT edited while the wave head is
  frozen.

* **HIS EAR CALIBRATED THE METRIC, 2026-09-12, and this is the most useful
  thing in this file.** Two `public_domain` episodes, the SAME bank, a
  BYTE-IDENTICAL composed prompt, opposite verdicts:

  | episode | opening cue | tempo error vs the 122 BPM asked for | his words |
  |---|---|---|---|
  | `the_clanking_chains_153041` | 0.42 / 123.0 BPM | **1.0** | *"perfect music"* |
  | `firelight_skepticism_151426` | 0.29 / 156.6 BPM | **34.6** | *"just house chords no rhythm"* |

  **TEMPO ERROR is the discriminator, and onset periodicity is not.** The two
  differ 35-fold on tempo error and only 0.42 vs 0.29 on periodicity -- a gap
  far too small to have predicted his verdict. Any future guard, bench or
  A/B on groove cues binds to |measured BPM - requested BPM|, never to a
  periodicity floor.

  **It also means the wording is exonerated twice over.** Those two cues asked
  for the same thing in the same words and one was perfect. The variable is the
  render, not the prompt -- so the fix is a recipe or a re-roll, not more
  prompt-craft.

  **A FALSIFIABLE PREDICTION, left here on purpose:**
  `shadows_on_the_catwalk_144526` measures 0.74 / 120.2, i.e. **1.8 BPM off**.
  If tempo error really is the thing his ear tracks, he should like that one
  too. If he does not, this calibration is wrong and the row reopens.

* **CUT 2026-09-12: the anime checkpoint does NOT go to the AnimateDiff lanes,
  and this row is closed, not deferred.** His words: *"lets stop chasing the
  anime things, we can leave it in there if it['s] coded, I don't want to spend
  any more time chasing before release."*

  **What SHIPS is what he already approved:** the pack checkpoint reaches the
  STILL engine only (`nodes/_otr_image_engines/sd15.py`), which he judged
  *"perfect"*. That stays exactly as it is.

  **What is CUT:** extending it to `eng_ghost_signal.py` and the other SD1.5
  motion lanes, where `GHOST_CHECKPOINT_NAME` stays the pinned base checkpoint.
  A design panel on it was STOPPED mid-run on his word. Do not restart it, do
  not "just try" the constant swap, and do not re-raise this before release --
  he noticed the limitation himself and cut it himself, which is the strongest
  form this ruling can take.

  For a future reader who wonders why it looked easy: it was not. That swap is a
  RECIPE BUMP -- `GHOST_RECIPE_RECEIPT` would have to be repointed or every
  receipt already on disk stops being interpretable -- and the lane runs a LIVE
  negative at `GHOST_CFG = 8.0` specifically because the lettering defense needs
  real unconditional conditioning, which is the documented reason an AnimateLCM
  checkpoint was refused once before.

* **KNOWN ISSUE, PARKED FOR AFTER RELEASE: music cues are a lottery, and the
  prompt is not the lever.** He reported two on 2026-09-12 -- a closing cue that
  "is like 2 seconds" and an opening that is "just house chords no rhythm,
  maybe one slight beat". Both are real. A 19-agent panel settled the cause and
  the answer is the same for both: **an under-constrained prompt hands the
  outcome to the seed.** Recorded here so nobody pays for this analysis twice.

  **The driver's first framing was WRONG and the panel corrected it.** It read
  four bad closings, all on the authored `scifi_news_pro` lane, and concluded
  "authored rows are broken". There are SIX authored closings on disk and the
  sample omitted the two that render at 100%. Worse, within the authored group
  the supposed cause runs BACKWARDS: 20 chars -> 100%, 27 -> 100%, 25 -> 48%,
  45 -> 65%, 45 -> 77%, 48 -> 35%. "Soft, contemplative piano" (48%) and "A
  soft, somber piano melody" (100%) are near-identical asks with opposite
  outcomes. And the complained episode's OWN opening is also authored
  ("Detroit techno, 128 BPM", no arc, no tail) and renders at 94% -- a control
  inside the very episode that kills the missing-arc theory.

  **What the panel PROVED cannot be the cause**, each grounded in the file:
  * There is NO length-to-duration coupling anywhere.
    `eng_stable_audio_3.py:209` computes `seconds_total = max(context_s,
    dur * 3.0)` with no prompt term; `:392` builds the latent from `dur` alone;
    sampler, scheduler, steps and denoise are env constants at `:378-385`.
  * A short prompt cannot shorten any tensor. `model_base.py:895-902` pads
    cross-attention to a fixed 256 slots plus the seconds token, all attended.
  * The one prompt-derived path, `_sa3_clip_window`'s "outro"/"opening" text
    fallback, is dead TWICE: `stable_audio_theme.py:331` always passes a real
    placement, and `StableAudio3` never reads `seconds_start` at all -- the
    receipt itself records `seconds_start_read_by_model: false`.
  * Post-processing is clean: `_ceiling_the_cue` is a peak limiter at -1.0
    dBFS, no fade, and the receipts show peak -1.03 / -1.00.

  **The real shape:** composed closings measure 71-100%, mean ~96%, because a
  composed row names instruments, mood, an idiom carrying a BPM, an arc and the
  tail, leaving the sampler almost nothing to choose. Authored closings measure
  35-100%, mean ~71% -- the same distribution with a much worse floor. It is a
  VARIANCE difference, not a deterministic one.

  **THE CHEAP TEST, already designed, ~6 minutes, no episode leg.** Call
  `generate_clip` directly on the exact shipped closing spec at 8 seeds and
  measure audible fraction. A 35-100% spread on one text confirms seed noise
  and closes it; a tight cluster near 35% means the text IS deterministic and
  the hunt moves to arm B (same text plus the arc and tail).

  **NOT RUN, on his instruction:** *"I don't want to spend any more time chasing
  before release."* This row is for after.

* **RULED 2026-09-12: no auto-fetch row for the anime checkpoint.** His words:
  *"I don't think we will set an auto fetch file because it's so specialised."*
  Correct, and it settles the last open question on that feature: a 4.24 GB
  download for one style most users never pick is the wrong default. The
  checkpoint stays OPT-IN, which is exactly what already ships -- present means
  the anime pack uses it, absent means the stock model, and nothing fails
  either way. Documented in the README as opt-in. **This feature is DONE.**

* **THE GROOVE DEFECT IS FOUND, IT IS ONE LINE, AND IT IS THE DRIVER'S OWN BUG
  FROM 2026-09-12.** A 26-agent panel refuted the driver's seed-variance theory
  by measurement and found the real cause. **Awaiting his go/no-go only because
  it touches the render path and the wave head is frozen.**

  **NOT the seed.** Same prompt, different seeds, ruled `small_base` arm: pulse
  spread 0.016 (house) / 0.065 (techno), tempo never missed, 6 of 6 locked.
  A variable with a range of 0.065 cannot produce the shipped range of 0.78.
  Renders are exactly reproducible -- 18 cross-boot pairs agree on every
  recorded digit of seven metrics -- so a retry-on-tempo guard would be
  strictly wasteful: re-rolling the seed at a fixed prompt buys ~0.065 of pulse
  for a whole extra render.

  **The driver's premise was FALSE.** It claimed the two contrasting episodes
  had byte-identical prompts. They have different prompt hashes. What differs is
  the brief-mined MOOD WORDS, and they sit IN FRONT of the tempo instruction:

  | lead words | tempo | pulse | his verdict |
  |---|---|---|---|
  | suspenseful, **driving**, dark | 120.2 | 0.74 | -- |
  | tension, ominous, **frantic** | 123.0 | 0.42 | *"perfect music"* |
  | ominous, suspenseful, eerie | 156.6 | 0.29 | *"no beats, maybe one slight beat"* |

  The two that locked lead with a MOTION word. The one that failed is pure
  texture. `nodes/_otr_music_prompt.py::compose_music_prompt` appends
  `mood_terms` BEFORE `palette.idiom`, so three atmosphere words stand in front
  of "Chicago house at 122 BPM".

  **This is the exact failure the module's own docstring already names** for the
  neutral "atmospheric" default -- and the 2026-09-12 fix only skipped the
  DEFAULT, leaving a real brief's mood words leading on a groove bank. Half a
  fix.

  **THE FIX, one line:** on a `groove_arc` palette, append `palette.idiom`
  FIRST and the mood terms after it, so the genre and its BPM lead. Nothing else
  changes; sustained banks keep today's order exactly.

  **ALSO FOUND, a real bug in a bench tool:** `scripts/music_model_bench.py:295`
  derives the seed from the index into the FILTERED family list while the
  filename records only the family and a k-index, so the same filename means
  different seeds between a smoke run and a full run. That invalidates the
  determinism control cited in
  `docs/2026-09-12-music-model-bench/driver_anchor.md:94-96`. The conclusion
  there survives on other evidence; the cited proof does not.

* **THE SHIPPING SET HE WANTS, stated 2026-09-12, and it is CURATION not
  construction.** His words: *"we will have three NVIDIA eight gigabyte JSONs
  and three to four NVIDIA sixteen gigabyte JSONs and maybe three AMD JSONs...
  but right now we just have to get the original one working."*

  **CANONICAL FIRST. This row is explicitly AFTER that.**

  Measured against `config/profiles/` today -- 45 profiles carry
  `status: shipping`:

  | class | he wants | we have shipping | the actual gap |
  |---|---|---|---|
  | NVIDIA 8 GB (ceiling 6.8) | 3 | **2** | one short: `otr_4060_12b_gguf_offload`, `otr_nvidia_8gb_haunted` |
  | NVIDIA 16 GB (ceiling 14.5) | 3-4 | **42** | ten times too many -- CUT, do not build |
  | AMD | ~3 | **0** | the AMD profiles exist but are all `draft` |
  | Apple | (implied) | 1 | `otr_mac_mps` |

  So the work is: promote one more 8 GB, promote three AMD out of draft once the
  ROCm tester reports, and pick three or four of the 42 sixteen-gigabyte ones to
  be the named set. The other 38 stay available; they just stop being the
  answer to "which one do I use".

  **AND THE GALLERY IS THE DELIVERY MECHANISM, which is why the set matters.**
  Measured against the live server: ComfyUI's template scanner globs exactly one
  level (`*/workflows/*.json`), so `workflows/variants/` is invisible and the
  gallery offers exactly ONE entry today. A JSON in the scanned folder becomes a
  CHOICE and loads nothing until picked, so promoting the curated set costs
  nothing at runtime -- but promoting all 93 would turn a gallery into a
  haystack. That is the reason to curate before promoting, not after.

### Still genuinely open, and not his call

* **`purple_cloud` cannot be vendored from pg11229 and that is now measured,
  not assumed.** The edition carries NO chapter divisions of any kind -- its
  only all-caps headings are the title, "INTRODUCTION" and "THE END.", and the
  body is broken solely by rows of asterisks. So `("chapters", 10, 11)` is
  unsatisfiable and no chapter pair would resolve. It wants either a different
  Gutenberg edition or a new chunk kind that slices on an explicit prose
  landmark. It is left REFUSING on purpose: the near-miss on `ghost_ship` the
  same day (a wrong anchor produced a clean 9,134-word "OK" line that had
  silently swallowed an entirely different story) is the argument against
  anchoring by feel. `ghost_ship` and `beleaguered_city` ARE vendored; the
  manifest went 65 -> 67 and lost nothing.

### Held deliberately, revisit when the thing they wait on lands

* **The 8 GB ship set** stays `draft` until the wave reports its physical 8 GB
  legs. **He ruled hold**, and it could not honestly be anything else yet.
* **`defaults.scene_coherence_check`** stays inert on every bank. **He ruled
  "not yet"**; story quality is closed and the offline corpus measurement was
  never run.
* **An IP-Adapter on the AnimateDiff lanes.** **He ruled hold.** It is a new
  dependency for the registry story and a recipe change on hard-won recipes;
  after the wave, if at all.
* **The ROCm recruitment post.** Written and pushed and waiting for him --
  `ROCM_MISSION_IMPOSSIBLE.md`, the hero still, two drafts in
  `docs/rocm-recruitment-post-draft.md`. **He said he will post it himself.**
  A window must never post it.

## 4. Constraints specific to this plan

Only the ones not already in CLAUDE.md or the standing rulings.

- Full listener source, no RSS. Cast count is flexible and records requested vs
  actual; the house announcer is excluded from dramatic cast.
- **WE DO NOT CHASE ACT COUNT** (operator, 2026-09-11), the same rule as word
  count: the value is a request, and a run delivers the closest performable
  episode.
- Model checking and a fixed attempt budget only -- no separate chunker, no
  recursive loop.
- An exhausted optional correction still yields a usable ledger; no predictive
  word or duration gate.
- Byline and attribution rules differ for My Story, Original and the adaptation
  banks.
- No replay, migration or re-render project: a saved input means fresh
  generation.

## 5. Parked

Parked and tombstoned items are preserved in
[GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md), not here -- by this file's own rule
they are not work. That includes the unqualified installed-family and GGUF opt-in
combinations, the H3 policy receipts, the cfg promotion comparisons, the AMD
scoped pod and platform acceptance, the cloud billing opt-in routing, the
operator-parked casting/adaptation ideas, OTR-Lite after v2, and the release
runway.

## 6. And when all of this is done -- it is time to TEST. Hurrah.

When sections 1 and 2 are empty, the waiting is over and the fun part starts.
Freeze ONE hash, write it into the `WAVE HEAD:` line of
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md), and turn four
machines loose on it at once -- the 5080, the 4060, the Mac and a RunPod box,
each running the real canonical workflow, all reporting home.

Everything they owe is already written and waiting in
[COVERAGE_OWED](2026-09-11-four-machine-test-wave/COVERAGE_OWED.md) beside the
two lane documents. Nothing needs planning when the day comes; it needs starting.

Until then: **do not freeze a head, do not book a leg, and do not settle a row up
there by rendering something.** Two heads were cut early on 2026-09-11 and both
had to be withdrawn. An arc that can only be answered by live evidence is
answered without it, or cut with the reason written in.

**Then the next morning begins in `otr/obs/`, not in the editor** -- count what
landed against what was promised, read the four phone-homes, and triage anything
crash-class first. That is the good problem to have.
