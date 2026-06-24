# Ending-mode design — FINAL (R4-converged, build-ready)

Stop the weak local writer collapsing every premise into the console/kill-switch
climax: at the LINE-COMPOSITION of the FINAL character beat, inject a concrete,
style-driven ENDING that reframes the (still on-stage, still `irreversible_choice`)
climax away from the doomsday button. DARK / default-OFF / byte-identical. 100%
local. Deterministic. (4-round roundtable converged; GPT-5.5 + Gemini-3.1-pro +
DeepSeek-v4-pro + Claude grounded judge.)

## SEQUENCING (the wiring spine)

Grounded order: `generate_outline` (macro premise -> phases -> beat intents)
returns -> writer runs `build_sq_data` (assigns beat_roles incl.
`irreversible_choice` on the last voiced CHARACTER beat + grounds crisis nouns)
-> LINE COMPOSER writes dialogue. So resolve style+ending in the WRITER AFTER
`generate_outline` (premise = `outline.premise`), inject at the LINE COMPOSER's
final-character-beat request. **`OutlineRequest` gets NO new fields.**

## A. Catalog data (`nodes/_otr_style_catalog.py`)

- `ENDING_TAGS` (8): revelation, reversal, unresolved_final_sound, reconciliation,
  bittersweet_parting, ironic_twist, quiet_acceptance, confession.
- `ENDING_TEMPLATES: dict[tag -> concrete final-beat instruction]` — author the 8
  strings (what literally happens / the last sound; "no machinery / no console").
- Each of the 100 entries: ADD `ending_tag` (one of the 8) + `domain` (matching
  l12 `select_domain` outputs). **KEEP the existing `ending_mode` prose key
  unchanged** (`render_style_grammar()` reads it — renaming breaks the baseline).
- `_DEFAULT_ENDING_TAG = "revelation"` (fail-soft, LOUD warn, never raise).
- `validate_catalog()` (test): every entry has a valid `ending_tag` + `domain` +
  `ending_mode`, and a template exists for its tag. No `ENDING_TAG_BY_SLUG` map —
  read via `get_style(slug)["ending_tag"]`.

## B. Deterministic selector

`select_style(premise, meta, cast_seed) -> slug` in `_otr_style_catalog.py`:
l12 `select_domain(meta, premise)` -> domain; candidates = catalog styles with
that `domain`, filtered to `non_emergency_slugs()` unless premise/meta carry
explicit disaster/rescue keywords; deterministic index =
`sha256(f"{cast_seed}:style:{domain}")`. Runs ONLY when the flag is ON.

## C. Writer plumbing (`OTR_LedgerScriptWriter`)

Flag `OTR_ENABLE_STYLE_GRAMMAR` (env). OFF => nothing runs, no telemetry => byte-
identical. ON, after `generate_outline` returns (cast_seed known):
`slug = select_style(outline.premise, meta, cast_seed)`;
`tag = get_style(slug)["ending_tag"]` (default fallback);
`template = ENDING_TEMPLATES[tag]`; stamp telemetry (F); pass
`(template, final_char_beat_id)` to the line composer.
`final_char_beat_id` = the key in `roles_by_beat` whose value ==
`BEAT_ROLE_IRREVERSIBLE_CHOICE` (imported from l12). **Do NOT override
`meta.style`** (risks desyncing the visualizer/LTX).

## D. Line-composer injection (`_otr_line_composer`)

Add `ending_template: str = ""` (+ a final-beat marker) to the composer prompt
path; APPEND the ending instruction to the system/user prompt ONLY for the final
character beat AND only when non-empty (empty => byte-identical line prompt,
golden-asserted). This is the single behavioral injection.

## E. Announcer-outro gating (`_otr_outline._assemble_outline`)

Gate only the announcer close INTENT (do NOT remove it — budget validator #7).
Read the env directly inside `_assemble_outline`
(`os.environ.get("OTR_ENABLE_STYLE_GRAMMAR") == "1"`) so the frozen
`OutlineRequest` is untouched: OFF => exact existing string
("Close on a concrete final image showing what changed...", byte-identical);
ON => "Close the episode without explaining the outcome; identify the program only."

## F. Telemetry

`meta.story_quality`: `style_slug`, `ending_tag`, `domain`, and the crisis-noun
count at the final character beat (reuse `count_ungrounded_crisis`).

## G. Bundle / interaction

Style grammar + L1/L2 crisis-noun grounding (`OTR_STORY_QUALITY_L12`) ship as ONE
"story-grammar" bundle, on together (grammar = climax SHAPE, L1 = trope
VOCABULARY). T4 staging penalty complements §E. T2 critic = orthogonal. Negative
anti-trope ban CUT.

## H. Build order (each chunk: suite + Bug Bible green, byte-identical asserted, push)

1. **Catalog content + contract:** author the 8 `ENDING_TEMPLATES` + assign
   `ending_tag` + `domain` to all 100 entries (keep `ending_mode`) +
   `validate_catalog()`. Pure, no behavior.
2. `select_style` selector (pure, dark).
3. Line-composer `ending_template` injection + final-beat marker (dark).
4. Announcer-outro env gate (dark).
5. Writer flag plumbing + telemetry.
6. Docs: L1/L2 bundle-on. Then the live A/B (GPU, last).

## I. Tests

- Flag OFF => line prompt + announcer intent + writer output byte-identical
  (golden fixture). `validate_catalog()` coverage. selector determinism
  (emergency only on disaster keywords; default pool excludes emergency).
- Flag ON => final-beat line prompt carries the template; announcer intent is the
  non-outcome string.
- C7: full writer run, flag OFF => byte-identical (audio gate holds).

## J. Validation soak (baseline first)

Baseline then lever-ON: crisis-noun density at the final beat (target ~0),
distinct `ending_tag` distribution (>= 80% non-doomsday), critic `arc_verdict`
mix, ~6-episode graded A/B on a couple of seeds. If the weak model still drifts,
make the template more prescriptive (name the final line shape) or use a frontier
writer for the final beat only.

## Hard constraints (carried)

100% local default (frontier opt-in, separate); DARK/default-OFF/byte-identical;
deterministic + C7 seed path holds; env-only gate => NO workflow JSON change;
full suite + Bug Bible green; UTF-8 no BOM; SFW; no new heavy model, no extra
paid call.

## Verify-at-build (the only open items)

- `_otr_line_composer` request shape (not in grounding) — confirm when building.
- The empirical bet: a weak local model honoring a concrete pre-seeded ending —
  settle with §J. This is a measurement, not a design hole.
