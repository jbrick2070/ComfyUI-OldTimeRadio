# OWNERSHIP: the 4060 owns the workflow (operator, 2026-09-05)

**This INVERTS the split in `CLAUDE.md` section 1 for the workflow files.** Read
this before editing anything under `workflows/`.

Operator: *"I'm telling the 4060 it owns the workflow, because it will be too
tough to talk to the 5080 to make changes quickly."*

## Who owns what, starting now

| file | owner |
|---|---|
| `workflows/otr_canonical.json` | **THE 4060** |
| `workflows/variants/**` | **THE 4060** |
| `config/profiles/**` | the 4060 (it regenerates variants) |
| `nodes/**`, `scripts/**`, `tests/**` | the 5080 |
| `pyproject.toml` + anything registry-facing | the 5080 |
| `docs/PROD_BUG_LOG.md` | shared, APPEND-ONLY |

**The 5080 does not edit the workflow JSONs while this stands.** If the 5080
needs a dropdown changed, it asks the 4060 -- `ListAgents`, then `SendMessage` --
it does not reach in. The reason is unchanged from section 1: two windows editing
one JSON is how it gets corrupted, and `merge=union` does NOT protect `.json`.

## What has NOT changed

* **Section 0 still holds.** `workflows/otr_canonical.json` is the source of
  truth, variants are GENERATED (`scripts/build_variants.py --all`, then
  `--check`), and they are never hand-edited.
* **Section 0B still holds.** A change that reaches the other box must PROVE the
  other box is unchanged, measured. Confining a per-machine choice to a variant
  is still the preferred shape.
* The engine-choice guards are gone (operator ruling 2026-09-05) but the
  RUNNABILITY checks stay: whatever is saved must be registered, usable for its
  role, and invocable. `tests/test_saved_dropdowns_are_live_choices.py` still
  asserts canonical's saved values are live menu options.

## State at handoff

Canonical ships the least-friction 8 GB graph, zero third-party node packs:

    video  ltx098_low_video (16:9) x3     image  z_image_turbo x3
    voice  kokoro / kokoro                music  musicgen
    upscale procgen blend  BYPASSED

Published as `2.0.0-alpha.24` (Pending scan at handoff; `alpha.23` is Active, so
Manager's "latest" still resolves to the OLD canonical -- pick alpha.24
explicitly until it promotes).

**Nothing in the canonical model set is Hugging Face gated.** Verified against
the HF API 2026-09-05: `google/gemma-4-12b-it`, `google/gemma-4-E2B-it`,
`unsloth/gemma-4-12b-it-GGUF`, `unsloth/Qwen3-8B-GGUF`,
`facebook/musicgen-small` and `onnx-community/Kokoro-82M-v1.0-ONNX` all report
`gated: false`, public. So a writer failure on the 4060 is a SIZE or runtime
problem, never a licence click -- do not go looking for a token.

## Known open, do not re-debug

* The ten haunted variants carry a bare `animatediff15_v3_haunted_video` where
  the live menu shows `animatediff15_v3_haunted_video (16:9)`. Headless is fine;
  the canvas shows red dropdowns. The one-line fix relabels values that profile
  APPLICATION expects bare and breaks five other tests, so it was reverted --
  the note beside `_director_option_value` records the blast radius. Canonical is
  unaffected.
* `docs/MACHINE_MATRIX.md`'s 8 GB row still recommends the haunted lane, which
  needs `ComfyUI-AnimateDiff-Evolved`; the `extra install` column now says so.
