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

**THE DEVICE HALF IS NOW PROVEN, and this row is only about the other half.**
`8017a07e` made the canonical stop naming a vendor; all four device widgets read
`default` / `cpu`. Receipt, 2026-09-12 18:59 on the 5080 after a clean reset and
a fresh boot on the post-fix tree: `otr_canonical_api_run.py --act-count 1` with
NO profile returned RESULT SUCCESS in 407 s and published
`the_handwritten_note_20260912_185540__cart__vcam__none__koko__marc__q354b__sa3_final.mp4`
(86.7 MB) to `otr/obs/`. The ledger records `device: "cuda"` and
`device_policy: "cuda"` -- the widgets said `default`, ComfyUI's own detection
resolved them, and the receipt carries the concrete truth rather than the word
"default". That is the leg Fable said was missing, and it passed.

**The sizing half did not land.** The canonical still carries
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

### A4. The writer's 36-widget order -- SETTLED 2026-09-14, ready to execute

Operator, when asked for his preference: *"ask fable and cursor to agree upon the
most logical order for what we are doing, I don't have preference except think
what Apple would think, what's clean and logical."* So this was decided by three
independent readers, blind to each other, then converged by a fourth pass.

**The roster, stated honestly:** Claude Fable (subagent), GPT-5.6 Sol (via the
Cursor CLI), and DeepSeek v4 Pro (via OpenRouter) each proposed an order
independently. Sol was then given all three plus the corrections below and asked
to converge. The driver made the final call on one row -- see the overrule.

#### THE ORDER

```
 1 source_bank          <- the ONLY `required` entry (see below)
 2 source_ref
 3 visual_style
 4 episode_title
 5 custom_premise
 6 story_characters
 7 story_plot
 8 story_setting
 9 story_author
10 act_count
11 include_act_breaks
12 num_characters
13 lemmy_cameo
14 story_scaffold
15 creativity
16 min_p
17 repetition_penalty
18 max_new_tokens_cap
19 creative_writing_model
20 technical_model
21 openrouter_slot_a_model
22 openrouter_slot_b_model
23 comfy_slot_a_model
24 comfy_slot_b_model
25 google_api_slot_a_model
26 google_api_slot_b_model
27 llm_device
28 llm_attn_impl
29 llm_quant_policy
30 llm_vram_ceiling_gb
31 gguf_n_ctx
32 gguf_quant
33 use_exchange
34 enable_production_stage3_validators
35 news_briefs_required
36 replay_from
```

`gate_in` is a forceInput SOCKET and consumes no `widgets_values` slot; it is not
in this list and its descriptor index moves regardless, so the link-repair-by-
identity step runs either way.

#### THE STRUCTURAL FACT THAT SHAPES EXECUTION

**`required` renders BEFORE `optional`, always.** Verified by Fable in the
INSTALLED frontend
(`.venv/Lib/site-packages/comfyui_frontend_package/static/assets/settingStore-*.js`):
ComfyUI iterates `input.required` then `input.optional` into one ordered map.

Today `required = {episode_title, num_characters}`. **So every proposal that
opens with `source_bank` -- and all three did -- is unrenderable until the
categories move.** Two of the three readers proposed that order without noticing.

So the execution requires: `source_bank` INTO `required`; `episode_title` and
`num_characters` OUT to `optional`. Free at the Python level -- every `run()`
kwarg already carries a default -- but it is a deliberate decision, not a
side effect. `source_bank` is the only genuine routing prerequisite;
`episode_title` calls itself an optional override in its own tooltip, and a
required-looking blank in row 1 tells a first-timer they must fill it.

#### THE FOUR CONTESTED PLACEMENTS, and what won

1. **`visual_style` at row 3**, with the source controls rather than beside
   `creativity`. It is the second independent top-level roll, and it shapes the
   PICTURE, not the writing -- sitting it next to `creativity` implies it changes
   the script.
2. **`episode_title` at row 4.** Once it is `optional` the blank-required-field
   objection dissolves, and the title belongs with the episode's identity rather
   than buried below the My Story block.
3. **Sampling knobs directly after `creativity`** (rows 15-18), so generation
   behaviour reads as one block that then leads into model selection.
4. **Remote pickers directly under the model dropdowns** (rows 21-26). They are
   passive bindings for handles those dropdowns select; separating them hides
   the dependency. Their dead sentinels do not outweigh the causal link.

#### THE ONE OVERRULE, by the driver

The convergence pass moved `use_exchange`, `enable_production_stage3_validators`
and `news_briefs_required` UP to rows 15-17 and did not justify it in any of its
four rulings. Against that: Fable placed them last (its "lab equipment" tier),
DeepSeek placed them at 27-29, and the converging reader's OWN earlier proposal
placed them at 27-29. Three of four readings put them low and the move was
unexplained, so they are restored to 33-35, above `replay_from`.

#### FACTS CORRECTED ALONG THE WAY -- all re-verified against the files

* `custom_premise` is shared by EVERY bank, not a My Story field
  (`nodes/_otr_story_input.py:48-52`). The driver's own brief said otherwise and
  misled all three readers.
* `config/profiles/widget_mapping.json` manages **16** writer widgets, not 14 --
  it also manages `act_count` and `num_characters`. Re-measured.
* `source_ref` is NOT read by `media_archive`:
  `nodes/_otr_media_archive_sources.py:294` does `del bank, technical_model,
  source_ref` and its docstring says RSS feeds ignore it. Only `shakespeare` and
  `public_domain` consume it.
* `story_scaffold` is forced off by `media_archive` and `my_story` as well as by
  `original`.
* `lemmy_cameo` is refused outright on `shakespeare` and `public_domain`
  (`nodes/_otr_casting.py:1178`) and consumes a `num_characters` slot.
* The node has an OUTPUT socket literally named `technical_model`, so any label
  for that widget keeps the word "technical" or it contradicts the socket beside
  it.

#### HOW TO EXECUTE IT

The order is pinned in ONE place already:
`tests/test_openrouter_slot_widgets_s2.py::_EXPECTED_INPUT_ORDER`. Change that
list first, watch the test go red, then move `INPUT_TYPES` to match and migrate
all 17 graphs with `scripts/otr_widget_surgery.py` -- never by hand. The two
guards added 2026-09-13 (`test_widget_schema_order_matches_live_input_types`,
`test_widget_migration_pairs_values_by_name`) exist precisely to catch a reorder
that updates the class and not the graphs, or that lands a value on its
neighbour.

### A5. Foley / mime without GGUF -- official LTX 2.5 safetensors exist; pick the path

**Operator 2026-09-17:** GGUFs do not auto-install, so the public stage is
non-GGUF; he wants a foley/mime substitute that is not the patched
ComfyUI-GGUF `ltx25_*` stack.

**What is true today (measured 2026-09-17, not a guess):**

* Local `ltx25_foley_plus` / `ltx25_mime` / `ltx25_video` still load
  `UnetLoaderGGUF` + `CLIPLoaderGGUF` (`eng_ltx25.py`) and a Gemma-4 12B
  GGUF patch. That is the friction he is leaving.
* **Official non-GGUF weights exist now.** `Lightricks/LTX-2.5` on Hugging
  Face is a split safetensors pack. Comfy's own tutorial
  (docs.comfy.org/tutorials/video/ltx/ltx-2-5) ships native T2V / I2V /
  FLF2V templates. I2V generates picture **and** synced audio -- that is
  the foley/mime job. Distilled Comfy INT8 DiT + Gemma-4 TE + video/audio
  VAEs. **No ComfyUI-GGUF pack.** `ComfyUI-LTXVideo` is already installed
  next to this pack.
* **This 5080 box already has the official INT8 pack on disk** under
  `C:\ComfyUI-Models`:
  `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
  (21.5 GB), `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
  (15.4 GB), video + audio VAEs, spatial upscaler. The GGUF twins are
  still beside them. Official NVFP4 distilled exists on HF; it is **not**
  on this disk today.
* The HF repo is **gated** (license click). That is not "drop a file and
  Queue" for a stranger, but it is a different friction than a third-party
  GGUF loader pack.
* **Measured 2026-09-17: official non-GGUF does not fit 16 GB, even
  with the encoder on CPU.** This card is 16303 MiB (~15.92 GiB);
  OTR's working ceiling is 14.5. On-disk official distilled INT8 DiT
  is **20.027 GiB** (`ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`).
  Official NVFP4 DiT (HF, not on this disk) is **17.433 GiB**
  (`18721548408` bytes). Either file is larger than the card before
  activations. The locked GGUF recipe's sampling peak is
  **9.80 DiT + 3.20 act + 1.48 alloc = 14.48 GiB**, TE and VAEs already
  at zero. Swap only the DiT to official INT8 and keep that same
  CPU-TE / unload-VAE hygiene: **24.71 GiB, ~10.2 over the clamp**.
  Official INT8 Gemma-4 TE is another **14.324 GiB** file (the GGUF Q5
  TE is 8.86 and still spiked encode to ~15.8 GiB until it was pinned
  to CPU). Video VAE 1.371 + audio VAE 0.340 + spatial upscaler 0.927
  = 2.638 GiB more if they ever sit with the DiT. Optional prompt
  enhancer `gemma4_e2b_it_bf16` is 9.573 GiB and is not required for
  foley. There is no official safetensors combo that stays under 16 GB
  including the tensors it needs. The 16 GB local path remains the
  Q3 GGUF DiT.
* **Cloud LTX 2.5 is already wired** as `cloud_ltx25_foley_plus` and
  `cloud_ltx25_audio_in` (partner `ltx/ltx-2-5-t2v` Fast/Pro, Comfy
  Credits, zero local weights). `otr_cloud_deluxe_3act` already ships
  that foley lane. That is a working public substitute today if the spend
  is acceptable.
* **The public GGUF hole is three shipping graphs, not forty-eight.**
  Measured against `SHIPPING_SET` 2026-09-17: only `otr_16gb_video`
  (`ltx25_high_video`), `otr_16gb_foley` (`ltx25_foley_plus`), and
  `otr_16gb_mime` (`ltx25_mime`) still name a GGUF video engine. 8 GB
  haunted / still / `ltx_8gb` / AnimateDiff do not.

**Do not build a native official-safetensors lane for this 16 GB box.**
The files exist; they do not fit. Public foley/mime either stay on the
Q3 GGUF stack (lab / those three shipping graphs) or use the already
wired cloud pair. That is not a VRAM question any more.

## 2. CODE -- the design is settled, build it

### C2. `machine_classes.json` is missing the `ltx_8gb` receipt that `dropdown_matrix.json` spends

`docs/dropdown_matrix.json` marks `ltx_8gb` **proven** on 8 GB NVIDIA;
`config/machine_classes.json`'s `engine_evidence` carries no such row -- only an
RTX A4500 20 GB and the Mac mini M4. `docs/4060_DRILL_LOG.md` around lines
4579-4834 looks like the real 4060 receipt it was harvested from, so the fix is
probably to add the row rather than to retract the verdict. Two hand-curated
files feed two generated docs and nothing enforces agreement between them.

### C6. Full non-GGUF public stage (widened 2026-09-17)

**Operator 2026-09-12:** no public JSON that needs GGUF. **Widened
2026-09-17:** he is running a **full non-GGUF stage** because GGUFs do not
auto-install. Klein is already gone (`259d1faa`). Shipping 8 GB stills are
`sd15`.

**What still names GGUF in the public set (measured 2026-09-17):** only
`otr_16gb_video`, `otr_16gb_foley`, and `otr_16gb_mime`. Those three wait
on A5. Lab / soak recipes that exist to exercise `wan_ti2v`,
`fastwan_8gb`, `ltx_video`, `ltx_audio_in`, or GGUF `ltx25_*` keep the
lane -- do not "repair" them by pretending they are something else.

**Writer half of the same ruling:** shipping 8 GB / Mac / AMD / `otr_cpu_low`
already name `Qwen/Qwen3.5-4B`. Draft `cpu_floor` is the cloud-writer CPU
path (`comfy:slot-a` / `comfy:slot-b`); it does not get a local 4B.

### C7. The widget tier -- verified plan, nothing built

An external UI audit asked for removals, renames, a reorder and two
ownership consolidations. It has been independently QA'd (Cursor, with its own
contrarian) and the answer is **do not ship it as one change**. The plan of
record is [2026-09-13-widget-cleanup-QA-VERDICT.md](2026-09-13-widget-cleanup-QA-VERDICT.md);
where it and the original brief disagree, the verdict wins.

Approved, in order: the two missing guards, then UI labels via `display_name`,
then `perfect_run_spacesaver`, then all FIVE deprecated `ffmpeg` widgets, then
`OTR_VideoDirector`'s two inert seed widgets, then the title consolidation as
separate tested work, then the reorder once A4 lands.

Two no-goes on evidence, not caution. **Voice-engine consolidation** fails on
five blockers, the sharpest being that `char_voice_engine` is legitimately
stamped literal `"auto"` when CastLock resolves nothing and the render node still
needs a concrete engine. **`custom_source_bank`** cannot be fixed as proposed at
all -- the dropdown is fed scalar ids by `list_bank_ids()`, so `banks.json`'s
`label` is not a per-option display label and editing it changes nothing a user
sees.

**Renames are answered without renaming.** Frontend 1.51.10 separates the UI
label from the stable internal key and `node_info` forwards the option, so
`display_name` buys the wording with none of the blast radius.

**The trap that inverts the obvious:** `migrateWidgetsValues` fires at exactly
ONE removal, so dropping a single trailing widget corrupts `replay_from` and the
three My Story fields while dropping the last two corrupts nothing. "Trailing is
free" is false.

`scripts/otr_widget_surgery.py` is the procedure as a module -- both entry points
return `(touched, repairs)`, repair the link table themselves, and refuse a short
or absent `widgets_values`; 14 tests against the real canonical. The execution
brief is [HANDOFF_WIDGET_TIER_EXECUTE.md](HANDOFF_WIDGET_TIER_EXECUTE.md), and
[2026-09-14-CODEX-WIDGET-DRIFT-HARDENING.md](2026-09-14-CODEX-WIDGET-DRIFT-HARDENING.md)
is the brief for attacking the drift model before any of it is executed.

### C0. Kokoro is the default voice in every shipped JSON (operator ruling, 2026-09-12)

Operator: *"for maybe Mac and AMD, well, at least Mac, we'll ship Kokoro because
it's the least friction download. So Kokoro might be our all-arounder for least
friction download and all of the JSONs, but people can change to the other one
if they want, but they'll have to download the models themselves if it won't
auto download."*

**The measurement agrees, and it is not close.** Kokoro is the ONLY voice engine
proven on all three machine columns, and the smallest:

| engine | how you get it | size | 8 GB NV | 16 GB NV | Mac 16 |
|---|---|---|---|---|---|
| `kokoro` | auto (HF cache on first use) | 0.3 GiB | proven | proven | proven |
| `bark` | auto | 4.2 GiB | proven | proven | **OOM -- reboots a Mac** |
| `indextts2` | own Windows-only installer | 11.1 GiB | not offered | proven | not offered |
| `chatterbox` / `dia` | own Windows-only installers | 3.0 / 6.0 GiB | not offered | fits | not offered |

So kokoro is the only choice that is one click on every platform this pack
claims to support, and the ruling makes it the floor rather than an option.

**WHERE IT ALREADY HOLDS:** `otr_canonical.json` ships kokoro on both slots --
confirmed by the 2026-09-12 no-profile receipt, whose filename carries `koko`.
The announcer slot is kokoro in nearly every profile already.

**WHERE IT DOES NOT, measured across all 118 profiles:** 79 pair
`char_voice_engine: indextts2` with a kokoro announcer, and 5 pair `bark` with
one. Most of those are soak/rotation/experimental profiles and are NOT the
shipping surface, so this row is scoped deliberately: **every JSON we SHIP --
the canonical and the per-machine set he is planning -- defaults both voice
slots to kokoro.** A rotation profile whose whole purpose is to exercise
IndexTTS2 keeps IndexTTS2; changing it would delete the test.

The README already needs the other half of his sentence written down: switching
to another voice engine is supported, and on `indextts2` / `chatterbox` / `dia`
it costs a Windows-only installer run that nothing will do for you.

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

**A LIVE RECEIPT LANDED THE SAME HOUR, from the other box, and it is the whole
argument.** `docs/4060_DRILL_LOG.md` CR-20260912-04: a clean-room 4060 followed
the README's 8 GB row by hand, got a valid ledger, six Kokoro clips, a Stable
Audio master and a 78-second 1,950-frame intermediate MP4 -- and then the first
shot died on `FailureKind.DEPENDENCY_MISSING` for exactly
`v1-5-pruned-emaonly-fp16.safetensors`, `v3_sd15_mm.ckpt` and
`v3_sd15_adapter.ckpt`. The drill's own words: *"The engine reports no fallback
and no render-time download."* Nothing reached `otr/obs/`. That is a stranger
losing an entire render to three files the matrix had been calling **auto**.
The label is corrected now; the download is not.

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

**DONE 2026-09-12 for `sd15` itself.** It is row nine of the manifest and the
stills and LTX lanes are one click now.

**THE ANIMATEDIFF FILES ARE DELIBERATELY NOT IN SCOPE.** Operator, same day:
*"animatediff doesn't auto download, it's a niche workflow, we just need to be
sure it works where it's supposed to and how to install it is documented."* So
`v3_sd15_mm.ckpt` and `v3_sd15_adapter.ckpt` stay manual by choice, not by
oversight. What that ruling DID require was closed the same hour: the two repos,
the two destination folders and the copy-paste commands are now in README
section 2b under "The AnimateDiff weights", because until then the filenames
appeared only in `MODEL_ASSET_INDEX.md` with no source -- which is what the 4060
hit, and a filename with no source is not an install instruction.

`spandrel_esrgan`'s 67 MB upscale model is the one remaining hand-fetch in the
README's cheapest-setups table, and it is the same shape as `sd15` was.

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

## FOR THE 4060: STAND DOWN ON NEW LEGS UNTIL THE WRITER SIZING LANDS (2026-09-12 evening)

**Written by the 5080 window because no 4060 session was reachable through
`ListAgents` -- every Remote Control row was offline. The operator relayed it
once; this file is so nobody has to again.**

**STAND DOWN. Do not spend another clean-room leg tonight.** Two of the three
walls your drills hit have moved, one has not, and the one that has not is the
one that would eat your next run.

### What your drills proved, and thank you -- both were real

* **CR-20260912-03 (Plan 1)** died at `llm_device=mps` on a Windows torch.
  **FIXED** in `8017a07e`: every device widget in the canonical now ships
  ComfyUI's own `default` sentinel, resolved per host through
  `comfy.model_management`. Receipt on the 5080 at 18:59 -- `--profile none
  --act-count 1`, RESULT SUCCESS in 407 s, published to `otr/obs/`, ledger
  recording `device: "cuda"` and `device_policy: "cuda"` from widgets that say
  `default`.
* **CR-20260912-04 (Plan 2)** died with `DEPENDENCY_MISSING` on three files.
  **ONE OF THE THREE IS FIXED.** `v1-5-pruned-emaonly-fp16.safetensors` is now
  row nine of `_otr_visual_assets.MANIFEST`, so `OTR_WorkflowValidator` fetches
  it at queue time. `v3_sd15_mm.ckpt` and `v3_sd15_adapter.ckpt` are NOT yet.
  A re-run would get further and stop on the motion module instead.

### Why standing down is right rather than cautious

**Plan 1 will still fail on your card, one stage later and 8.7 GB more
expensive.** The device half of system-independence landed; the SIZING half did
not. The canonical still ships `Qwen/Qwen3.5-4B` with `llm_quant_policy: none`
and `vram_ceiling_gb: 10.0` -- byte-for-byte the `otr_mac_mps` triple, and 10.0
is unique to that one profile across all 118. Unquantized that row wants roughly
8.7 GB and an 8 GB card has about 7 GB to give. The fit gate says WARN and lets
it through, so you would download 8.7 GB and then OOM at
`model.to(device)`. That is GO_FORWARD ARC row A1 and it is waiting on an
operator ruling, not on code.

**And check which install you are testing.** Your CR-2026091 2-0x runs were on
registry **alpha.30**, installed through the Manager. Every fix above is in the
git tree and NOT in that published version, so `git pull` in a checkout changes
nothing about what your ComfyUI loads unless you are running from that checkout.
Say which one you are on in the next drill entry -- it changes how to read every
result.

### What IS worth doing when you resume, in order

1. **Re-run Plan 2 only after the AnimateDiff pair is in the manifest.** That is
   GO_FORWARD CODE row C3 and it is the next thing this window builds. One leg
   then tells us whether the README's 8 GB row survives to a published episode,
   which is the question your drill actually opened.
2. **Do not re-run Plan 1 until A1 is answered.** When it is, the leg is worth a
   lot: it is the only 8 GB receipt for the canonical that exists anywhere.
3. **`docs/4060_DRILL_LOG.md` is yours and the entries are excellent.** Keep
   recording the install route and the exact failing node id; both times that is
   what made the finding actionable from here within the hour.

## 3. Blocked on the operator -- each unblocks with one word

**He answered all sixteen on 2026-09-12 and left to do other things, with:**
*"i cant listen until i get back thats ok if thats the one thing waiting, dont
stall for me, keep going and do a bunch of testing, we can fix when i get back."*
So the rows below are what SURVIVED his answers. The twelve rows that closed are
gone from this file by its own rule; the receipt in HANDOFF_LOG carries them.

### Waiting on his eyeball -- registry and workflow (2026-09-13 night)

* **WATCH FOR 2.1.1 TO GO ACTIVE, then delete the `v2.0-alpha` branch.**
  State: 2.0.0 Active, 2.1.0 and 2.1.1 Pending. Read the WHOLE version list --
  the API sorts by string, so a release sorts below every `-alpha.N` row, and
  the status enum is the only signal there is.
  **Do not delete the branch before 2.1.1 is Active.** 2.0.0 and 2.1.0 have
  `/v2.0-alpha/` frozen into their registry card Icon URL, and the listing
  renders whichever version is current -- deleting early breaks the card art
  on the live page.

* **Fable's answer to "an auto-update agent for version control" is a
  fork-point audit in `scripts/build_variants.py --check` plus a tracked
  `.githooks/pre-push`, not an agent.** Refuse a shipping profile whose
  effective voice pair fails `resolve_casting_plan()`, refuse a pin that equals
  the canonical (hundreds of such pins across the 16 today -- each rots the day the
  canonical moves), print each profile's real diff, and run that check with
  the three sibling `--check`s (dropdown matrix, machine matrix, tier matrix)
  from the hook: seven seconds measured, no CI. Three of tonight's eight
  defects would have been refused before push. Needs his word because a
  hook changes how both windows push (`git config core.hooksPath .githooks`
  per clone). Full write-up: the session scratchpad `fable_vc.md`; ~80 lines
  in build_variants.py, ~10 in otr_asset_index.py, one 25-line hook.

### Waiting on his ear, and nothing else

* **CUT 2026-09-17 -- story and music prompt-craft.** Operator: the story
  is fine as it is; the music is great, it is fixed. Do not reopen writer
  quality, techno/house cue wording, tempo-error A/Bs, or the music-lottery
  seed hunt. The long 2026-09-12 ear write-up moved to
  [GO_FORWARD_ARCHIVE](GO_FORWARD_ARCHIVE.md) under that date. IndexTTS2
  hang (below) is a hang-timeout, not prompt-craft, and stays.

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
  | AMD | ~3 | **0** | CUT FROM v2.0 (operator, 2026-09-13). The profiles exist and stay `draft`. |
  | Apple | (implied) | 1 | `otr_mac_mps` |

  So the work is: promote one more 8 GB, and pick three or four of the 42
  sixteen-gigabyte ones to be the named set. **AMD leaves this table** -- the
  operator cut it from v2.0 on 2026-09-13 rather than claim a platform nobody
  here can put an episode through, and `apple/ROCM.md` now says so in those
  words. The two profiles stay built and `draft`; whoever has the card is
  welcome to them, and nothing in this release waits on that. The other 38 stay available; they just stop being the
  answer to "which one do I use".

  **AND THE GALLERY IS THE DELIVERY MECHANISM, which is why the set matters.**
  Measured against the live server: ComfyUI's template scanner globs exactly one
  level (`*/workflows/*.json`), so `workflows/variants/` is invisible and the
  gallery offers exactly ONE entry today. A JSON in the scanned folder becomes a
  CHOICE and loads nothing until picked, so promoting the curated set costs
  nothing at runtime -- but promoting all 93 would turn a gallery into a
  haystack. That is the reason to curate before promoting, not after.

### Still genuinely open, and not his call

* **ONE HEADLESS LEG WOULD ADMIT A BIBLE RULE THAT IS OTHERWISE HELD.** The
  `custom_premise` source-override hijack (shipped in 2.1.0/2.1.1, fixed in
  2.1.2) is genuinely UNCOVERED as a Bible class -- "a field validation treats
  as optional may be another consumer's override switch; grep every READER,
  not just the validators". It is NOT admissible on the evidence that exists:
  a reviewer ran `_resolve_inputs` headlessly against the real shipped value,
  which this repo's own log has twice ruled is review evidence, not a live
  incident. Nothing was actually mis-rendered -- every episode in the affected
  window ran `my_story`, the one correct bank.
  **The cheap close:** one headless leg on `media_archive`, `public_domain`,
  `shakespeare` or `scifi_news_pro` against a worktree in the pre-fix range
  (`5e60a012`..`bf584e9b`) produces the server-log/ledger evidence
  PBUG-20260819-01 used, and it becomes admissible the same day. The drafted
  Bible entry (`04.15`) and index row are in that session's transcript, ready
  to paste. Bible is 352 entries at `af2557f7`, in sync.

* **CLEAR 14 SUITE DELTAS FROM THE 2026-09-13 NIGHT SESSION.** A full-suite
  diff against `283abaa6` (that session's start) shows 63 failures vs 53 --
  14 tests fail now that passed then, 4 that failed then now pass. They were
  NOT individually cleared before the session ended; that is the work.
  Two were spot-checked and are known:
  * `test_legacy_audit_clean::test_no_unclassified_legacy_references` trips on
    the node TITLES set during the canvas relayout ("Video Director /
    Settings") -- its audit flags Director surfaces. Cosmetic, real.
  * `test_canonical_headless_api::test_visual_style_override_does_not_patch_story_fields`
    PASSES in isolation, so it is order-dependent, not a regression.
  The four `test_scope_render_profile` rows are expected to follow the node
  93/94 removal (`8171e994`) and want confirming, not assuming. The rest are
  unexamined: `test_freeze_cascade_title_rename`,
  `test_gguf_version_pin_is_documented` (x2), `test_google_video_sfx_workflow`,
  `test_model_asset_index_drift`, `test_source_bank_widget_2c`,
  `test_text_metric_ownership`, `test_workflow_director_freedom`.
  Run each in ISOLATION first -- two of the fourteen are already known to
  behave differently alone, so the batch number is not the evidence.

* **WIDGET RENAME/REORDER TIER -- run `docs/HANDOFF_WIDGET_RENAME_REORDER.md`
  in its own window.** Its first deliverable is a GO/NO-GO on the RENAMES, not
  an edit. The blocker to decide: a rename breaks every workflow a USER has
  already saved, which cannot be migrated. Measured costs are in the handoff
  (2,074 saved widget-name refs, 47 mapping targets). If no safe path exists,
  do the REORDER only -- it changes no names.

* **DECIDE: should `llm_quant_policy` and `llm_vram_ceiling_gb` fresh defaults
  match the canonical?** Today they deliberately do not (fresh `bnb_nf4` /
  `14.5`; canonical `none` / `10.0`). Matching would make a dropped node load
  the writer unquantised at a higher ceiling and could OOM an 8 GB card. His
  call. Same question, same answer needed, for `act_count` (fresh 3,
  canonical 1).

* **DECIDE: is `stable_audio_3`'s cuda+mps declaration stale?** A canonical
  `--cpu` leg ran it and published (`the_trembling_silver_signet`; the leg log
  names the checkpoint). If the declaration gains `cpu`, `otr_cpu_low` can move
  to sa3 like the other fifteen. Cost: editing
  `tests/test_capability_profiles.py::test_v2_stable_audio_3_lists_mps_but_not_cpu`,
  whose docstring gives a tier PREFERENCE as its reason, not a capability
  claim.

### Held deliberately, revisit when the thing they wait on lands

* **The 8 GB ship set** stays `draft` until the wave reports its physical 8 GB
  legs. **He ruled hold**, and it could not honestly be anything else yet.
* **`defaults.scene_coherence_check`** stays inert on every bank. **He ruled
  "not yet"**; story quality is closed and the offline corpus measurement was
  never run.
* **An IP-Adapter on the AnimateDiff lanes.** **He ruled hold.** It is a new
  dependency for the registry story and a recipe change on hard-won recipes;
  after the wave, if at all.
* **The ROCm recruitment post.** Posted by him, and it WORKED -- Kate on the
  R9700, two still-tier episodes, issue #2. `apple/ROCM.md`, the hero still,
  and the drafts in `docs/rocm-recruitment-post-draft.md`. **A window must
  never post anything there.** Every public reply is drafted and he sends it,
  except the GitHub #2 thank-you he asked a window to post (2026-09-15).
  **Do not ping her for another test.** Operator, same day, after that
  comment: we told her another run is not needed. That kills the old
  "ping when Manager carries `otr_amd_still`" owe -- it was a third queue.
  Fold her second receipt into `apple/ROCM.md` ourselves; do not ask her.

### Owed to people outside this machine (2026-09-13)

* **`otr_8gb_foley` is RETIRED (operator ruling, 2026-09-13).** He ruled on
  time, not on failure: *"4 hours for 1 act seems too long"*, then *"maybe we
  dump foley on the 8gb lane."* The measurement behind it, from the 4060's own
  server log: `ltx25_foley_plus` averages **673.9 s per clip** over 11 clips
  and `ltx25_mime` **655.1 s** over 20 -- the same render, differing only at
  the mux -- both decoding at 1664x960, which is the SAME resolution the 16 GB
  graph uses. There is no smaller foley lane to fall back to: `ltx25_high_*`
  is the only tier the engine has, so the 8 GB graph was the 16 GB recipe on a
  smaller card. Against that, the whole `otr_8gb_video` episode finished in
  25.4 min on the same box. Foley and mime both remain shipped at 16 GB.

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

**Operator 2026-09-17 confirmed:** *"TEST WAVE AFTER CODING."* Same gate
as 2026-09-12. Do not freeze a wave head to settle a row above.

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
