# OTR Conventions

Living document. Add entries here when a recurring naming or shape
choice gets formalized so future-Claude (and future-Jeffrey) know
what the rule is and why.

## Module naming

### Library-only private modules: `_otr_<name>_lib.py`

When a `nodes/` module is library-only (no `NODE_CLASS_MAPPINGS`
export, exists only to expose helpers to other node modules), use
the prefix-and-suffix shape:

```
nodes/_otr_<name>_lib.py
```

- **leading underscore** (`_`) — Python convention for "private to
  the package; do not import from outside `nodes/`".
- **`otr_` prefix** — scopes the name to this project. Cowork sessions
  often have multiple Python projects mounted; an unprefixed
  `_bark_lib` module name collides visually with anything else named
  `bark_lib` in the user's tree. The prefix kills that ambiguity at
  read-time.
- **`_lib` suffix** — flags the file as library-only. If a reader
  sees `_otr_bark_lib.py` they know it has helpers, not a node class.
  Modules without `_lib` are expected to register a node class.

Current modules following this convention:

| Module                      | Exports                              | Importers                                                                  |
|-----------------------------|--------------------------------------|----------------------------------------------------------------------------|
| `nodes/_otr_bark_lib.py`    | `_load_bark`, `_unload_bark`         | `scene_sequencer.py`, `_otr_model_loader.py`, `_otr_vram_levers.py`        |
| `nodes/_otr_casting.py`     | cast contract helpers                | `OTR_LedgerScriptWriter.py`                                                |
| `nodes/_otr_ledger_consumers.py` | `load_ledger`, `iter_lines`, ... | `scene_sequencer.py`, `cast_lock.py`, `otr_shot_lock.py`, `video_engine.py`, `stable_audio_theme.py`, `otr_credits_roll.py`, `_otr_captions.py`, `_otr_ledger.py`, `_otr_voice_node_common.py`, `_otr_writer_tail.py`, `_otr_content_authorship.py` |
| `nodes/_otr_ledger_freeze.py` | `FreezeCascade`, gate helpers     | `_otr_freeze_cascade.py`, `_otr_captions.py`, `_otr_ledger_cleanup.py`, `_otr_scene_guard.py`, `_otr_cast_coverage_repair.py`, `_otr_cast_voice_coverage.py` |

**Note on the `_lib` suffix:** only the first entry above
(`_otr_bark_lib.py`) carries the strict `_lib` suffix this
section's header documents (`_otr_sfx_lib.py` was removed with
the SFX rip, commit `b56d970f`). The three remaining entries
(`_otr_casting.py`, `_otr_ledger_consumers.py`,
`_otr_ledger_freeze.py`) are private library modules that predate
the suffix convention; they're listed here because they expose
the same kind of helper surface, but they aren't required to
follow the `_lib` test rule (which only fires for
`_otr_*_lib.py`). New private library modules going forward
should adopt the `_otr_<name>_lib.py` shape so the test enforces
the no-node-class invariant; the three legacy entries are
grandfathered.

The S19.2 doc-freshness check
(`tests/test_naming_conventions.py::test_conventions_doc_lists_every_lib_module`)
scans `nodes/_otr_*_lib.py` strictly and asserts every match
appears in this doc. As of 2026-08-28 there is 1 such module and
it is listed. (The check is one-way: it catches a module missing
from this doc, never a doc row whose module was deleted.)

### Test enforcement (S10.3)

Both rules are pinned by `tests/test_naming_conventions.py`:

| Pattern                  | Meaning                                              | Test                                            |
|--------------------------|------------------------------------------------------|-------------------------------------------------|
| `_otr_*.py` (no `_lib`)  | Package-internal, mixed concerns                     | —                                               |
| `_otr_*_lib.py`          | Package-internal, library-only (no node class)       | `test_lib_modules_have_otr_prefix`, `test_lib_modules_have_no_node_class_mappings` |
| `<name>.py` (no prefix)  | Node-bearing module                                  | —                                               |

The two enforced rules:

1. Any module whose name ends `_lib.py` MUST also start with `_otr_`.
2. Any `_otr_*_lib.py` module MUST NOT define `NODE_CLASS_MAPPINGS`
   (AST-walk catches this at any scope).

Rule (3) — non-`_lib` modules registering node classes follow the
`OTR_<Capitalized>` class-name convention — is conventionally honored
but not test-enforced. Add a test if drift surfaces.

### Non-library modules: `<name>.py`

Modules that register a node class (have `NODE_CLASS_MAPPINGS`) use
the public surface name without the underscore prefix or `_lib`
suffix. Examples: `batch_bark_generator.py`, `scene_sequencer.py`,
`OTR_LedgerScriptWriter.py`, `video_engine.py`, `musicgen_theme.py`.

The Python class inside still follows the project convention of
`OTR_<Capitalized>` (e.g. `OTR_LedgerScriptWriter`,
`OTR_BatchBarkGenerator`). The module filename can match
(`OTR_LedgerScriptWriter.py`) or use lowercase
(`batch_bark_generator.py`) -- both are accepted; what matters is
that the class name carries the `OTR_` scope tag.

### Renaming an existing private library

When renaming `nodes/_<name>_lib.py` to `nodes/_otr_<name>_lib.py`:

1. `git mv` the file (preserves history).
2. Update every importer in the same commit -- find them with
   `findstr /SI "_<name>_lib" nodes\*.py tests\*.py scripts\*.py`.
3. Update the in-file rename comment to point at the new name and
   reference this conventions doc.
4. Verify: AST-parse the renamed file + every importer, then run
   the regression. No two-step shipping.

## When NOT to extend this convention

- **External packages** (anything under `comfyui-custom-node-survival-guide/`
  or `comfy/`) keep their own naming. The `_otr_` prefix is a project
  scope tag for files that live inside `ComfyUI-OldTimeRadio/`.
- **One-off scratch scripts** under `scripts/` don't need the prefix
  because they're not imported as modules; they're entry-point
  scripts the user runs directly.
- **Test files** under `tests/` already have the `test_` prefix as
  the scope tag; no `_otr_` needed.
