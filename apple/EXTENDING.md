# Adding to OTR

Two things you can add: an **engine** (a way of rendering video, images, speech,
music or an upscale) and a **source bank** (a place stories come from). They are
different jobs. Engines are a Python file in this repo; a source bank can be a
folder of your own that this repo never sees.

Read the one page first. Then [PREFLIGHT.md](PREFLIGHT.md) is the checklist that
says whether what you built will actually work.

---

## The one page

- **Four engine namespaces share one recipe**: video, image, audio (speech and
  music) and upscale. One adapter file, an `@register` decorator, one
  `CAPABILITIES` row, and a guarded import in that namespace's `__init__.py`.
  Source banks are the fifth kind and are a folder, not an adapter.

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
  documents, add a shortcode, and pick one of the **five** bookend rosters in
  `render_driver.py` — `ENGINES`, `BOUNDED`, `SELF_COMPOSED`, `NOT_TEXT_DRIVEN`,
  `KNOWN_RED`. They are asserted disjoint, so exactly one is right.

- **Green tests are not a lane.** The proof is one real render through
  `workflows/otr_canonical.json` that lands a file in `otr/obs/`.

- **Removing is the same job as adding, done atomically** — registry row, module,
  pipeline entries, tests, and a grep that returns exactly the survivors you
  expected.

---

## Adding an engine

### 1. Copy the closest sibling

Adapters live in `nodes/_otr_video_engines/`, `_otr_image_engines/`,
`_otr_audio_engines/` and `_otr_upscale_engines/`. Start from the engine most
like yours rather than from scratch — the class contract is easier to read from a
working example than from prose.

### 2. Declare it

Your class needs a `name` — the internal id, assigned in the class body, because
that is what the tooling reads:

```python
class MyEngine:
    name = "my_engine"
```

Then `@register` it, and add its `CAPABILITIES` row in that namespace's
`registry.py`. The row is where `device_backends`, `model_requirements` and the
rest are declared.

### 3. Import it, guarded

Add it to the namespace's `__init__.py` in the same try/except style as its
siblings. An adapter nothing imports is registered nowhere and simply does not
exist at runtime — which is this repo's most repeated defect, because every test
still passes.

### 4. Fail closed

If your weights are missing, refuse by name before anything loads. Do not
substitute, do not degrade quietly, and do not let the render get hours in before
discovering it. The error text is documentation that reaches the one person who
needs it.

### 5. Run the gates

[PREFLIGHT.md](PREFLIGHT.md), the section for your namespace.

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

- **`auto_select`** — whether a blank automatic run may roll onto your bank.
  `runnable: true` alone does **not** put you in the roll pool; you need both.
  Set it false while you are still testing.
- **`story_input_mode`** — how your bank takes its input.

### The two functions

Keyword-only, and the writer calls them with exactly these names:

```python
def fetch_source(*, bank, technical_model, source_ref="",
                 load_config=None, policy=None): ...

def interpret_source(*, bank, payload, technical_fn, model_id): ...
```

**Five keywords on the fetcher, four on the interpreter.** Older notes in `docs/`
show a three-keyword fetcher; that signature raises `TypeError` on its first
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
- **`speaker_role` is one of exactly five values** — `character`, `announcer`,
  `music_open`, `music_close`, `music_inter`. Not free text; anything else
  raises, including the retired `sfx`.
- **Do not filter content.** Adaptation lanes carry the author's own language as
  written. A bank that sanitises its source is producing a fidelity defect.
- **Do not chase a word count.** The target is a request, not a gate.
