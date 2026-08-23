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

---

## Second pass, same day -- and the findings matter more than the deletions

### Removed (5 more, one of them a cascade)

| file | symbol | why |
|---|---|---|
| `story_orchestrator.py` | `body_scraper_unavailable()` | its CONSUMER is gone. The docstring says the v4 source-floor failure message reads it so a missing package is never misreported as an exhausted feed pool -- but that message string no longer exists anywhere in the tree. The module variable and the LOUD `log.error` stay; they are the half that still works. |
| `story_orchestrator.py` | `_log_scene_checkpoint()` | SCENE_TRACK telemetry that never fires |
| `story_orchestrator.py` | `_scene_inventory()`, `_RE_SCENE_MARKER`, `_RE_SCENE_TERMINATOR` | the cascade: removing the only caller orphaned the helper, and removing the helper orphaned both regexes. Taken out as one self-contained diagnostic cluster rather than left dangling. |

**Cascades are the reason a dead-symbol pass is iterative.** Cutting one
function turned three more symbols dead in the same file. A single scan run is a
snapshot, not a fixed point.

### THE FINDING WORTH THE OPERATOR'S ATTENTION

**The BUG-020 character-name repair has not run since 2026-05-10.**

`story_orchestrator._cleanup_character_names()` is the fuzzy pass that collapses
LLM name variants -- `NEMEO_SIRIKIT` back to `NEMO SIRIKIT`, a rare misspelling
folded into the dominant spelling. It is **never called**.

Traced, not guessed: `git log -S` puts the call site alive until commit
`eec4718c` ("L3 ledger consumer rewrite sprint -- Phase 3 writer extraction",
2026-05-10), whose diff removes the line

    -        script_text = _cleanup_character_names(script_text, _cast_config_path, pre_rolled_cast)

as part of moving `LegacyLLMScriptWriter` out to `_otr_legacy_writer.py`. No
replacement normalizer exists anywhere in `nodes/`. Its helper
`_extract_all_dialogue()` is orphaned with it.

**It is NOT deleted, and it is NOT rewired, and both of those are deliberate.**
It operated on the LEGACY writer's `script_text`; the modern
`OTR_LedgerScriptWriter` path has its own attribution repair (ShotLock's round-5
F4 backstop warns when a line opens with its own speaker's name). So this may be
correctly obsolete rather than a live gap -- but that is a judgement about what
the current writer actually emits, which the code alone cannot settle.

Deleting a shipped bug fix on a scanner's say-so, or wiring a legacy-shaped pass
into a modern path it was not written for, are both worse than saying this
plainly and letting the operator decide. **Name consistency is one of the
correctness classes he explicitly kept open when story quality closed**, so it is
his call and not a coder's.

### Still not cleared (12 candidates)

`LEDGER_SCHEMA_VERSION_TARGET`, `VOICE_BANK_SCHEMA_VERSION`,
`ImageEngineConfig`, `ImageLedgerSection`, `MAX_PROVENANCE_NOTE_CHARS`,
`P0RepairTrimReceipt`, `content_oracle.check_manifest` / `load_manifest`,
`role_slots.PER_ROLE_VIDEO_SLOTS` / `NEW_ROUTE_A_VIDEO_SLOTS`,
`other_name_policy`, `probe_context_visibility` / `_grade_probe`.

These are schema versions, pydantic models, route maps and public-looking
accessors. An unreferenced name in that set may still be a declared contract
someone is meant to call, and the cost of being wrong is higher than the few
lines saved. Each needs reading in context, the same as everything above.
