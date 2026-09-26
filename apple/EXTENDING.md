# Adding to OTR

What you can add: an **engine** (a way of rendering video, images, speech,
music or an upscale), a **source bank** (a place stories come from), a
**writer LLM** (the model that writes the script), an **episode language**
(a registry row that binds writing, voices, captions and credits), and a
**shipped workflow** (a saved workflow for a machine or a Comfy Cloud stack). They are
different jobs.
Engines are a Python file in this repo. A source bank can be a folder of your
own that this repo never sees. A writer LLM is either a snapshot in your
Hugging Face cache, or a curated row in the catalog. A language is data only
when Kokoro already has a real code and voice set for it. A shipped workflow
is a row in a JSON matrix, not a workflow you type.

Read the one page first. Then [PREFLIGHT.md](PREFLIGHT.md) is the checklist that
says whether what you built will actually work. The writer page is
[LLM_PREFLIGHT.md](LLM_PREFLIGHT.md).

---

## The one page

- **The engine namespaces share one recipe**: video, image, audio (speech and
  music) and upscale. One adapter file, an `@register` decorator, one
  `CAPABILITIES` row, and a guarded import in that namespace's `__init__.py`.
  Source banks are the other kind and are a folder, not an adapter.

- **Registering IS joining the dropdown.** There is no allow-list any more —
  `VALIDATED_ENGINES` was removed in 2026-06-29. Whatever registers appears.

- **A broken import is swallowed, and your engine silently vanishes.** Each
  import is guarded so one bad adapter cannot take down the pack. The cost is
  that a typo shows up as an absence, not an error. `audit_engine_roster()` in
  the video registry is what tells you the difference.

- **Declare things out loud, never by silence.** `accepts_still` as a real
  True/False, `engine_version`, `device_backends` from what you actually
  measured rather than what you hope, and for voices `voice_ref_kind`,
  `voice_ref_field` and `sample_rate`.

- **Weights resolve through `folder_paths` or a documented environment pin.** A
  missing weight raises a *named* `EngineUnusable` from `assert_usable` — never
  an `os.path.exists` against a hardcoded default, and never a quiet fallback to
  a different model. **The refusal message is the install instruction**; make
  yours name the exact file and folder.

- **An image engine is inert until a video lane consumes its still.** The shipped
  canonical selects three procedural lanes that mint nothing, so an image engine
  added today renders nothing at all until you switch a video role to a
  still-consuming lane. This confuses everyone once.

- **A new engine id trips fixtures, not just tests.** Regenerate the generated
  documents, add a shortcode, and pick one of the bookend rosters in
  `render_driver.py` -- `ENGINES`, `BOUNDED`, `SELF_COMPOSED`, `NOT_TEXT_DRIVEN`,
  `KNOWN_RED`. They are asserted disjoint, so exactly one is right.

- **Green tests are not a lane.** The proof is one real render through
  `workflows/otr_canonical.json` that lands a file in `otr/obs/`.

- **A language row is not a translation engine.** It asks the writer to author
  natively and filters voices, captions and credits. The full registry and
  add-your-own checklist are in [MULTILINGUAL.md](MULTILINGUAL.md).

- **A shipped workflow is a matrix row, not a JSON you write.** Edit
  `config/workflow_matrix.json`, then `python scripts/build_variants.py --all`.
  Every `workflows/otr_*.json` except the canonical is generated. The canonical
  is the one workflow you may author.

- **Removing is the same job as adding, done atomically** — registry row, module,
  pipeline entries, tests, and a grep that returns exactly the survivors you
  expected.

---

## Adding an engine

### Copy the closest sibling

Adapters live in `nodes/_otr_video_engines/`, `_otr_image_engines/`,
`_otr_audio_engines/` and `_otr_upscale_engines/`. Start from the engine most
like yours rather than from scratch — the class contract is easier to read from a
working example than from prose.

### Declare it

Your class needs a `name` — the internal id, assigned in the class body, because
that is what the tooling reads:

```python
class MyEngine:
    name = "my_engine"
```

Then `@register` it, and add its `CAPABILITIES` row in that namespace's
`registry.py`. The row is where `device_backends`, `model_requirements` and the
rest are declared.

### Import it, guarded

Add it to the namespace's `__init__.py` in the same try/except style as its
siblings. An adapter nothing imports is registered nowhere and simply does not
exist at runtime — which is this repo's most repeated defect, because every test
still passes.

### Fail closed

If your weights are missing, refuse by name before anything loads. Do not
substitute, do not degrade quietly, and do not let the render get hours in before
discovering it. The error text is documentation that reaches the one person who
needs it.

### Run the gates

[PREFLIGHT.md](PREFLIGHT.md), the section for your namespace.

---

## Adding a video lane that is just new weights

This is the common case and the short one. A **lane** is a value in the video
engine dropdown, so adding one gives every workflow a new option -- the canonical
included -- without adding a single JSON file. Somebody with a bigger card
picks it and renders; nobody has to find a special workflow.

If your lane is the same recipe on a different build of the same model, it is
a subclass that changes one thing. The LTX 2.5 tiers are already built this
way: `ltx25_foley_24gb` is the 16 GB foley lane with a different DiT
file and the text encoder left on the GPU.

```python
@register
class MyLaneOnBiggerWeights(TheClosestExistingLane):
    name = "that_lane_bigger"
    _native_dit = "The-Model-You-Actually-Want.safetensors"
```

Then four things, none of them long:

1. **Name it for the person choosing it.** The id is what they read in the
   dropdown. If it needs a 32 GB card, say so in the name rather than in a
   doc they will not open.
2. **Add its `CAPABILITIES` row** in that namespace's `registry.py`, beside
   its sibling. Copy the parent's row and point `model_requirements` at your
   weights. A registered engine with no row is the hole the roster audit
   exists to catch.
3. **Reserve the id before you register it, if it cannot render yet.** Put it
   in the family's reserved tuple -- `LTX25_RESERVED_SIBLING_IDS` for LTX 2.5
   -- which says the name is spoken for while the weights are still being
   fetched. An id that is registered but cannot render is a dropdown entry
   that fails hours into somebody's episode.
4. **Run the gates**, then the whole suite. A new engine id turns fixtures red
   that the gate list does not name: the still-plan parity fixture, the asset
   index, the engine and machine matrices, and two literal roster lists. They
   are regenerated, not hand-edited, and [PREFLIGHT.md](PREFLIGHT.md) has the
   commands.

**Do not reach for a new matrix row and a new workflow JSON to do this.** When
the difference is which weights load, it is a lane, and a lane costs the user
one dropdown instead of one download. When the saved pins themselves should
differ -- a smaller writer on 8 GB, a cloud partner stack -- that is
[a shipped workflow](#adding-or-changing-a-shipped-workflow).

---

## Adding or changing a shipped workflow

A **shipped workflow** is a saved workflow a person can open from Browse Templates:
the canonical, or one of the per-machine workflows listed beside it.
It is not an engine and it is not a lane. A lane is a dropdown value every
workflow already has; the section above is that job. Reach for a new workflow only
when the *saved pins* should differ -- a smaller writer on 8 GB, a cloud
partner stack, a Mac device policy.

**The source of truth is one file: `config/workflow_matrix.json`.** Each row is
one workflow. Edit that file. Do not hand-edit any `workflows/otr_*.json`
other than the canonical -- those JSON files and their launch recipes in
`apple/LAUNCH_RECIPES.md` are generated, and the next rebuild silently undoes you. A new shipping workflow is a
matrix row only -- there is no second place a workflow can be defined.

This wants the git clone. `scripts/` is not in a registry install.

### Two jobs, two files

**Changing the authored workflow** -- a new node, a new socket, a new widget -- is
an edit to `workflows/otr_canonical.json`, in the same change as the node
code. Unwired code is dead. Do not add an `example_workflows/` folder --
ComfyUI mounts both at the same URL and the gallery 404s. Do not drop a
hand-written JSON next to the canonical either: every other `workflows/*.json`
is a generated per-machine workflow, the gallery lists them all, and
`tests/test_workflow_templates_single_folder.py` holds that set to the
canonical plus exactly the matrix's shipping rows. (They lived in
`workflows/variants/` until 2026-09-25, where the gallery never saw them.)

A new optional widget is always **appended** at the end of `widgets_values`.
Inserting in the middle silently shifts every saved value. Removing a widget
that is not last is three edits, not one: drop the value, drop the `inputs`
descriptor, and repair every later link's `dst_slot` (it is an index into that
same array). Trailing widgets are cheap; mid-list ones are not.

If the new widget is something a machine workflow should pin, add it to
`config/widget_map.json` as well, or the matrix cannot reach it.

**Changing what a machine workflow pins** -- writer, lanes, voices, ceiling,
device -- is an edit to that row's `deltas` in the matrix. The canonical is
left alone.

### Adding a row

Copy the closest sibling. Then:

1. **Give it an id** (`otr_8gb_video` style) and a `display_name` a person can
   read. Set `"ships": true` or it will not emit a workflow -- a new row defaults
   to *not* shipping, which is the safe direction.
2. **State every key in `key_indicators`.** The list is at the top of the
   matrix. Writer, ceiling, visual lanes, video engine, voices, act shape,
   device -- even when the value matches the canonical. An unstated indicator
   is inherited, and an 8 GB row that stops naming its writer follows the
   canonical upward onto a model that will not fit.
3. **Omit everything else** unless it actually changes a widget. An incidental
   pin that restates the canonical is a fork: it keeps the old value when the
   canonical moves. If a non-indicator key *is* a decision, add it to
   `key_indicators` instead of leaving it looking accidental.
4. **Metadata that is not a widget** -- `status`, `platform`, `allow_sidecars`,
   `toolchains`, `launch`, `preflight` -- defaults from the matrix `defaults`
   block. State only what differs. `device_backend` is required on every row;
   omitting it does not inherit, it fails validation. State `gpu_vendor` too
   (`nvidia`, `amd`, `apple`, `none`). It is not in `defaults`. Every shipping
   row states it. Omit it and an AMD/cuda row gets the NVIDIA launch recipe,
   and a pinned engine that `requires_vendor` is refused at emit. `allow_sidecars`
   defaults false; set it true only if this workflow should offer engines that
   declare `requires_sidecar`. `preflight.required_keys` is the cloud workflows.

Do not put a JSON in `workflows/` for the new row. Browse Templates stays one
card.

### Rebuild, then prove the rebuild

```bash
python scripts/build_variants.py --all
python scripts/build_variants.py --check
```

`--all` writes the per-machine workflow JSON, the launch recipes
(`apple/LAUNCH_RECIPES.md`, one section per workflow), and the generated docs
(`apple/MACHINES.md`, `apple/DROPDOWN_MATRIX.md`, `apple/MACHINE_MATRIX.md`)
from the same matrix. `--check` diffs the committed workflows against a fresh
regeneration and fails on drift.

A wiring or widget change also wants these four green:

- `scripts/build_variants.py --check`
- `tests/test_widget_value_alignment.py`
- `tests/test_canonical_widget_input_parity.py`
- `tests/test_workflow_link_target_indexes.py`

The first three can all pass while the links are broken. The fourth is the one
that catches a mid-list widget removal.

Other apple pages that name a specific workflow -- [WRITERS.md](WRITERS.md),
[VIDEO_MODELS.md](VIDEO_MODELS.md), [MAC.md](MAC.md) -- are hand-kept. Update
the one that would otherwise lie.

### Stopping one

Set `"ships"` false (or delete the row), **and delete** the matching
`workflows/<id>.json` that `--all` emitted, then re-run `--all` so its section
leaves `apple/LAUNCH_RECIPES.md`. Leaving the files is not enough to fail `--check`: a row that still
exists -- even with `"ships"` false -- is what `load_profile` reads, so the
leftover workflow regenerates cleanly.

### The proof

One real run of that workflow that lands a file in `otr/obs/`. Green `--check`
proves the generator, not the episode.

---

## Adding a writer LLM

This is not an engine. There is no `@register` and no adapter file.

**You can pick anything the dropdown will take.** Cache a CausalLM, choose
Gemma, use a cloud slot. That is not the same as what the pack ships.

**What the pack ships:** Qwen 3.5 as one transformers dropdown row
(NVIDIA NF4, Mac / CPU full).

The full checklist -- on-machine cache path, catalog row, the gates -- is
[LLM_PREFLIGHT.md](LLM_PREFLIGHT.md). Which models already ship, and how to
read the badge, is [WRITERS.md](WRITERS.md).

---

## Adding an episode language

This is not a writer LLM and not a voice engine. If Kokoro already serves the
language, add one complete row to `config/episode_languages.json`, add the
matching language-stamped Kokoro voices to
`config/voice_reference_bank.json`, run the admission and painter tests, then
publish one canonical episode to `otr/obs/`.

The row appears in the writer dropdown automatically; no workflow JSON edit is
needed. Missing translated chrome, duplicate ids/codes, missing tokenizer
extras and empty voice pools all fail closed.

[MULTILINGUAL.md](MULTILINGUAL.md) has the required keys, caption policies,
Python-version boundary, source-bank restrictions and live-proof checklist.

---

## Adding your own source bank

This one does not touch this repository at all. A bank is a self-contained folder:

```
user_packs/source_banks/<your_bank_id>/
    bank.json
    <your_bank_id>.py
    story_packs/
```

### `bank.json`

Every key is required except `defaults`: `source_bank_id`, `label`,
`source_kind`, the fetcher and interpreter entry points,
`default_story_pipeline`, `default_story_model`, `required_seams`, `runnable`,
`guide_ref`, and optionally `defaults`.

Two `defaults` keys are worth knowing because nothing else announces them:

- **`auto_select`** — whether Queue-on-roll may land on your bank. Omit it or
  set it true, and a runnable bank is an **equal** in the roll pool with the
  shipped rows. Set it false only to keep the bank manual-pick while you test.
- **`story_input_mode`** — how your bank takes its input.

### The two functions

Keyword-only, and the writer calls them with exactly these names:

```python
def fetch_source(*, bank, technical_model, source_ref="",
                 load_config=None, policy=None): ...

def interpret_source(*, bank, payload, technical_fn, model_id): ...
```

The writer calls the fetcher with `bank`, `technical_model`, `source_ref`,
`load_config` and `policy`, and the interpreter with `bank`, `payload`,
`technical_fn` and `model_id`. Older notes in `docs/`
show a shorter fetcher; that signature raises `TypeError` on its first
real call. If you are unsure, the binding check below proves it without running
your code.

### Activate it

```bash
python scripts/otr_check.py bank user_packs/source_banks/<your_bank_id> --activate
```

`--activate` is a deliberate consent step: it is the moment your Python is
allowed to be imported. Without it the bundle is inspected, not executed. Restart
ComfyUI afterwards.

Note that `scripts/` is not in a registry install — adding a bank wants the git
clone.

### What your bank must honour

- **Fill the ledger completely.** Downstream steps read fields, not intentions;
  speech, slicing, video direction, captions and credits all consume what you
  wrote. A missing field is a broken render, not a missing nicety.
- **`speaker_role` is one of** `character`, `announcer`,
  `music_open`, `music_close`, `music_inter`. Not free text; anything else
  raises, including the retired `sfx`.
- **Do not filter content.** Adaptation lanes carry the author's own language as
  written. A bank that sanitises its source is producing a fidelity defect.
- **Do not chase a word count.** The target is a request, not a gate.
