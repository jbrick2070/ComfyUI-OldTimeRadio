# Dead-symbol inventory -- `nodes/`, 2026-08-22

A CANDIDATE LIST, not a hit list. It exists so the next lean-and-mean pass
starts from evidence instead of a fresh guess, and so nobody re-runs the scan
that produced it.

## How it was produced

`ast` walk over every module-level `def` / `class` / assignment in `nodes/`,
against an identifier set built once from every `.py`, `.json`, `.md`, `.yaml`,
`.cmd` and `.ps1` in the repo (excluding `.git`, `.claude`, `kibitz-runs`,
`otr/cache`, `otr/episodes`, `.venv`). A symbol qualifies when **no file other
than its own defining file mentions it**.

**THE SCAN IS NOT A VERDICT AND MUST NOT BE TREATED AS ONE.** This repo
resolves plenty of names dynamically -- the engine registries decorate classes,
`NODE_CLASS_MAPPINGS` strings the node classes together, adapters use `getattr`
-- so a name can be perfectly live and still show up here. Every removal below
was read in context first, and two candidates were dropped on that reading.

**A private helper used inside its own module is NOT dead.** The scan reports
those too, because "no other file mentions it" is exactly what correct scoping
looks like. Only symbols whose own file mentions them ONCE -- the definition
line and nothing else -- are genuinely unreferenced.

## Removed 2026-08-22 (8 symbols, all verified at zero references)

| file | symbol | why it was safe |
|---|---|---|
| `OTR_LedgerScriptWriter.py` | `_TITLE_PREFIX_RE` | a `None` placeholder whose comment promised a lazy compile that happens in a local instead -- the shape of a plan that changed |
| `_otr_ledger.py` | `_WORD_COUNT_RE` | a PRIVATE back-compat alias with zero callers, which the no-legacy-back-compat directive is aimed squarely at; the compiled expression keeps its one owner in `_otr_text_metrics` |
| `_otr_source_grounding.py` | `_SENTENCE_END_RE` | a compiled regex nothing matches against |
| `_otr_news_wiring.py` | `_word_boundary_pattern` | a helper defined and never called |
| `otr_image_gen_dispatcher.py` | `_IMAGE_EXTENSIONS` | an extension tuple nothing filters on |
| `story_orchestrator.py` | `_TOKEN_RATIO_MIXED`, `_TOKEN_RATIO_OUTLINE`, `_TOKEN_RATIO_ACT_OBSIDIAN` | three unused members of a documented five-entry ratio table; the surrounding comment was trimmed in the same change so it no longer describes constants that do not exist |

Verified after removal: AST parse on all six touched modules, zero remaining
references to any of the eight, canonical workflow validator still 23 nodes /
57 links, full suite green.

## Looked at and DELIBERATELY KEPT

* **`_otr_content_safety.py` -- the whole module.** It looks like textbook dead
  code: no production caller anywhere, ~350 lines of profanity / weapon /
  nudity vocabulary retired by the 2026-08-03 no-guardrails directive. **Do not
  delete it.** `run_ledger_cleanup` never calls it, `meta.ledger_cleanup.safety`
  is stamped `"retired"` so the ledger field keeps an owner, and
  `tests/test_ledger_cleanup_pass.py` is deliberately INVERTED -- it asserts
  the pass is never invoked and the author's line is never edited. That test is
  the tripwire that makes a re-armed content filter fail loudly. Deleting the
  module deletes the tripwire, which is the opposite of what the directive
  wants.
* **`_LTX8_DEFAULT_W` / `_LTX8_DEFAULT_H`.** The scan flagged them; a grep
  found four references. The scan's self-count is a substring count and
  over-reports. Read before cutting.

## Still open -- 28 further candidates

`tmp/dead_symbols.txt` holds the full run. The remainder are NOT cleared for
removal; several are public-looking API (`content_oracle.load_manifest` /
`check_manifest`, `_otr_image_engines/schemas.py`'s `ImageEngineConfig` and
`ImageLedgerSection`, `role_slots.PER_ROLE_VIDEO_SLOTS`) where an unreferenced
name may still be a declared contract someone is meant to call. Each needs the
same read-in-context treatment before anything is cut.

**Cost of the scan, so nobody pays it twice:** the first implementation
searched every symbol against every file with a regex and did not finish in
twenty minutes. The version that works tokenizes each file's identifiers once
into a set and does an O(1) membership test per symbol.
