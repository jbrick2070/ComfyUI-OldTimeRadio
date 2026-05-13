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
| `nodes/_otr_bark_lib.py`    | `_load_bark`, `_unload_bark`         | `batch_bark_generator.py`, `scene_sequencer.py`, `story_orchestrator.py`   |
| `nodes/_otr_sfx_lib.py`     | `SFX_GENERATORS`                     | `batch_procedural_sfx.py`                                                  |
| `nodes/_otr_casting.py`     | cast contract helpers                | `OTR_LedgerScriptWriter.py`                                                |
| `nodes/_otr_ledger_consumers.py` | `load_ledger`, `iter_lines`, ... | `batch_audiogen_generator.py`, `batch_procedural_sfx.py`, `scene_sequencer.py`, `musicgen_theme.py`, `video_engine.py`, `otr_video_plan.py` |
| `nodes/_otr_ledger_freeze.py` | `FreezeCascade`, gate helpers     | `OTR_LedgerScriptWriter.py`, `batch_bark_generator.py`                     |

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
