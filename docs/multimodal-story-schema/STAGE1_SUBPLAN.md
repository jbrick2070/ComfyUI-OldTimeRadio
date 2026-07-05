# Multi-Modal Story Schema -- STAGE 1 HARDENED SUB-PLAN (v3, post-kibitz r1+r2)

Date: 2026-07-04. Branch: `v2.0-alpha`. Parent: `docs/multimodal-story-schema/BUILD_PLAN.md`.
Status: CONVERGED through kibitz r1+r2 (codex + antigravity, Claude judge; judgments
in `kibitz-runs/2026-07-04-multimodal-stage1/`). Next: Fable structural gate, then code.

## 0. Operator intent (governs every decision)

> "the current prompts need to more or less survive so i can run the new
> workflow when all is done and it works" -- operator, 2026-07-04.

The science/sci-fi lane must render **the same episode** through the new schema.
Stage 1 is a *provably-safe, DORMANT* foundation: it lands the loader + the first
pack + machine-proven byte-identity, and changes NOTHING about how the sci-fi path
runs. The risky part (wiring a live consumer) is quarantined into **Stage 1b**
behind its own kibitz + Fable gate, because the current code uses object-identity
sentinels a naive JSON swap would silently break (kibitz r1/r2, confirmed).

Core law: `JSON owns content+config. Python owns validation/routing/execution.
No fallbacks. No hidden models/engines. Unknown id = hard error.`

## 1. Two safety guarantees (why the sci-fi run cannot move)

1. **Dormant in Stage 1.** No production call site changes; a test proves no
   production file imports/calls the loader. Sci-fi output is byte-for-byte
   identical because no code path changed.
2. **Byte-identity pin via RUNTIME IMPORT.** For every authored seam a test imports
   the real node module and asserts `module.<CONST> == pack[seam]` (char-exact
   `str==str`; robust to implicit concat/`+`-joins, which naive AST would miss).
   Drift becomes a RED test, not a silent episode change.

Stage 1b no-fallback: a MIGRATED seam missing/empty RAISES; the Python constant is
the byte-identity ORACLE (test-time), never a runtime fallback. Unknown triple = raise.

## 2. In-repo shape (NO new package)

- **Behavior (one module):** `nodes/_otr_story_pack.py` -- **stdlib-only** loader +
  validator. Rationale: a ~40-line validator needs no dependency, is trivially
  auditable, and stays agnostic to the pydantic v1/v2 split the repo still carries
  (`news_interpreter.py:66-70` imports pydantic behind a v1 fallback). Zero content
  literals.
- **Content (data):** `nodes/story_packs/science_news/science_news_default.json`
  (the one Stage 1 pack). Module file + data dir under `nodes/`, not a new package.
- No banks/pipelines/visual_styles in Stage 1 (routing = Stage 2, visual = Stage 3).

## 3. Contract + loader (`nodes/_otr_story_pack.py`)

`StoryPack` = a `@dataclass` with an explicit strict validator (no pydantic):

- `REQUIRED_TOP_LEVEL = {source_bank_id, story_model_id, story_pipeline_id,
  schema_version, prompt_stages}` (all `str`, except `prompt_stages: dict[str,str]`).
- Optional/inert fields carry explicit defaults (avoid `StoryPack(**data)` TypeError):
  `label:str=""`, `status:str=""`, `examples:list=field(default_factory=list)`,
  `tone_guardrails`, `forbidden_plot_patterns`, `forbidden_leakage_terms`,
  `source_requirements`, `ledger_validation_notes` (all `default_factory=list`).
  These are TOLERATED INERT in Stage 1 -- kept for forward-compat, not validated
  beyond type, not consumed. No leakage scanner in Stage 1.
- Unknown top-level key -> `StoryPackValidationError` (typos fail loud).
- `schema_version` must be in a hardcoded known set `{"v2.0"}` -> else raise.
- `story_pipeline_id`/ids are OPAQUE validated strings (NO resolution vs banks/
  pipelines -- that resolver is Stage 2).
- `prompt_stages`: every KEY in `PRODUCTION_SEAM_ALLOWLIST` (section 4, == the exact
  authored set); unknown key -> `UnknownSeamError`. Every VALUE a non-empty string
  (reject whitespace-only). Python polices keys/shape, blind to the value text.

Loader:
- `load_pack(path) -> StoryPack`: `Path(path).read_text(encoding="utf-8")` (wrap
  `OSError`/`UnicodeDecodeError` -> `StoryPackNotFoundError`/`StoryPackParseError`);
  `json.loads(..., object_pairs_hook=_reject_dup_keys)` (fires on nested objects too;
  `json.JSONDecodeError` -> `StoryPackParseError`); validate -> `StoryPack`.
  `_PACK_CACHE: dict[str, StoryPack]` loads+parses each path at most once.
- `get_pack_prompt_or_none(pack, seam) -> str|None`: value if present+non-empty,
  else None (NOT-yet-migrated seam; caller keeps its Python constant).
- `get_pack_prompt(pack, seam) -> str`: RAISES if absent/empty (MIGRATED seam --
  no hidden fallback). Seam must be in the allowlist.
- Errors: `StoryPackError` (base) + `StoryPackNotFoundError`, `StoryPackParseError`,
  `StoryPackValidationError`, `UnknownSeamError`.
- No code/paths from JSON, read-only, no RNG/network/VRAM.

## 4. Authored seams (Stage 1 science pack) -- verified live + statically composable

`PRODUCTION_SEAM_ALLOWLIST` = EXACTLY these keys (no reserved future names -- they
would let unpinned seams pass; add them when Stage 2/3 needs them):

```
outline_macro_system   outline_phase_system   outline_beat_system
line_composer_system
coda_system
announcer_intro_system   announcer_intro_safe_system
announcer_outro_system
style_pick_inventor_system   style_pick_inventor_user
style_pick_chooser_system    style_pick_chooser_user
```

Byte source (extraction truth -- pull from HERE; re-grep each to confirm it is a
module-level constant + actually SENT before authoring):

| seam | live constant | note |
|------|---------------|------|
| outline_macro/phase/beat_system | `_otr_outline.py:1102/1115/1130` `_MACRO/_PHASE/_BEAT_SYSTEM_PROMPT` | the REAL sent outline prompts (:1868/:1996/:2101) |
| line_composer_system | `_otr_line_composer.py:1174` `_SYSTEM_PROMPT` | sent directly |
| coda_system | `_otr_line_composer.py:3275`+`:3297` | author PRE-JOINED `_NEWS_CODA_SYSTEM + _NEWS_CODA_SYSTEM_V2_EXAMPLES` (unconditional join at :3407) |
| announcer_intro_system / _safe | `:2905` `_ANNOUNCER_INTRO_SYSTEM` / `:2926` `_..._SAFE` | routed at :3195/:3227 |
| announcer_outro_system | `:2945` `_ANNOUNCER_OUTRO_SYSTEM` | base only |
| style_pick_inventor_system / _user | `_otr_style_picker.py:296/301` | |
| style_pick_chooser_system / _user | `_otr_style_picker.py:329/334` | |

EXPLICITLY NOT AUTHORED (stay Python -- legacy sentinel or conditional/interpolated):
- `_otr_outline._SYSTEM_PROMPT` (:532) -- LEGACY sentinel (comments :1045/:1826),
  used as `resolved is _SYSTEM_PROMPT` (:1847); the model never sees it.
- announcer outro RESOLVED tail (`:3517`) -- inline conditional literal.
- `line_grounding` conditional f-strings (`_build_user_prompt` ~1258-1345); the
  inline source-label/develop-verb branch (`_otr_outline._build_user_prompt` ~570-577).
- `NewsBriefs` interpreter body (Stage 2); visual tails + `_LTX_MOTION_PROMPT_BY_ROLE`
  (Stage 3). Validator/critic prompts = acknowledged deferred debt.

## 5. Chunking

**Stage 1 = Chunk 1 ONLY (dormant foundation, zero behavior change).**
- `nodes/_otr_story_pack.py` (section 3).
- `nodes/story_packs/science_news/science_news_default.json`: the authored seams
  (section 4), byte-for-byte from the live constants (pre-joined where the runtime
  join is unconditional).
- Tests (move WITH the schema, same commit):
  (a) **byte-identity** -- import each source module, assert `CONST == pack[seam]`
      (coda: assert the pre-joined string == pack["coda_system"]);
  (b) **exact-key-set** -- pack's `prompt_stages` keys == the authored set;
  (c) **fail-loud matrix** -- unknown seam key, unknown schema_version, unknown
      top-level key, duplicate key (top + nested), whitespace-only value, malformed
      JSON, missing file each raise the right typed error;
  (d) **extractor semantics** -- `get_pack_prompt_or_none` None for absent + exact
      str for present; `get_pack_prompt` raises on absent;
  (e) **dormancy guard** -- no production file under `nodes/` imports/calls
      `load_pack`/`get_pack_prompt*` (grep-style AST/text assertion).
- Chunk verification (not pytest): `git diff --quiet -- workflows/otr_scifi_16gb_full.json`
  (no workflow change this chunk). NO production call-site change.

**Stage 1b (SEPARATE -- own kibitz + Fable gate) -- first live consumer.**
Pilot seam = `line_composer_system` (sent directly to the LLM, NO identity logic --
codex + antigravity agree; outline is a poor pilot, being the legacy sentinel).
Precondition in the same chunk: retire the object-identity contract at ALL sites --
`test_creative_prompt_router.py:62` AND `:103`, `test_audio_c7_clamp_counter.py:52`,
and `_otr_outline.py:1847` -- migrating `is <const>` to `==`, plus a repo_id ->
pack-coordinates map in `_otr_creative_prompt_router.py`. Wire the seam via
`get_pack_prompt(...)`, proving byte-identity through the existing story +
`test_audio_byte_identical` regressions + a new equivalence test. Not part of Stage 1.

## 6. Security posture

Fail-loud strict validation; stdlib-only (no dep surface); no code/paths from JSON;
duplicate-key + whitespace-only rejection; DORMANT (no consumer -> a broken/missing
pack cannot alter the sci-fi run in Stage 1). No-fallback in Stage 1b. Deterministic,
read-only.

## 7. Gates per chunk (CLAUDE.md)

Full suite (`.venv` python, `PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`) + Bug
Bible (survival-guide repo, relative path) + B7 -- GREEN. UTF-8 no BOM; AST-parse
touched `.py`; commit AND push per green chunk to `v2.0-alpha`; verify HEAD==origin,
no 0-byte. Workflow-JSON (none in Stage 1) same-commit + re-validate if ever touched.
prod/main + tags operator-GATED.

## 8. Acceptance for Stage 1 "done"

- `nodes/_otr_story_pack.py` loads+validates the science pack fail-loud (full matrix
  raises the right typed errors).
- Science pack seams byte-identical to the live runtime constants (runtime-import
  pin green); exact-key-set green.
- Dormancy guard green; `git diff --quiet` on the workflow JSON (untouched).
- Foundation ready for Stage 1b (first consumer) + Stage 2 (routing) without
  reopening Stage 1. Operator note: the NEW workflow surface (source_bank/visual_style
  dropdowns) is Stage 2+; Stage 1 is invisible in the JSON.
