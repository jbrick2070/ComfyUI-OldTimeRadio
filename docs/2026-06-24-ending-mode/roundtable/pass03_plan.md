# Ending-mode design — R3-hardened (wiring resolved)

Stop the weak local writer collapsing every premise into the console/kill-switch
climax: at the LINE-COMPOSITION of the FINAL character beat, inject a concrete,
style-driven ENDING that reframes the (still on-stage, still `irreversible_choice`)
climax away from the doomsday button. DARK / default-OFF / byte-identical. 100%
local. Deterministic.

## SEQUENCING (R3 fix — resolves the circular dependency)

Grounded pipeline order: `generate_outline` (macro premise -> phases -> beat
intents) returns; THEN the writer runs `build_sq_data` (L1/L2: assigns beat_roles
incl. `irreversible_choice` on the last voiced CHARACTER beat + grounds crisis
nouns, mutating `beat.intent`); THEN the LINE COMPOSER writes each beat's
dialogue. So:

- Do NOT thread the ending into `generate_outline` / `_build_beat_user_prompt`
  (the role isn't assigned yet there — the original §D was circular).
- Resolve the style + ending in the WRITER, AFTER `generate_outline` returns
  (the premise is now `outline.premise`), BEFORE line composition.
- INJECT the ending template at the LINE COMPOSER's request for the final
  character beat (the `irreversible_choice` beat, now known from roles). This is
  where the dialogue — and the doomsday cliche — is actually written, so it is
  the correct and only injection point needed.

This keeps the lever in the writer + composer (no OutlineRequest premise-timing
problem) and means `OutlineRequest` does NOT need new fields at all.

## A. Catalog data contract (`nodes/_otr_style_catalog.py`)

- `ENDING_TAGS` (8): revelation, reversal, unresolved_final_sound,
  reconciliation, bittersweet_parting, ironic_twist, quiet_acceptance, confession.
- `ENDING_TEMPLATES: dict[tag -> concrete final-beat instruction]` (what literally
  happens / the last sound; "no machinery").
- Each of the 100 entries gains `ending_tag` + `domain`; keep prose as
  `ending_flavor`. `ENDING_TAG_BY_SLUG` external map. `_DEFAULT_ENDING_TAG =
  "revelation"` (fail-soft, LOUD warn, never raise). `validate_catalog()`
  self-check (test): every entry has a valid tag + domain + a template.

## B. Deterministic style selector

`select_style(premise, meta, cast_seed) -> slug` in `_otr_style_catalog.py`:
reuse l12 `select_domain(meta, premise)` -> domain; candidates = catalog styles
with that `domain`, filtered to `non_emergency_slugs()` unless the premise/meta
carry explicit disaster/rescue keywords; deterministic index =
`sha256(f"{cast_seed}:style:{domain}")`. Runs ONLY when the flag is ON.

## C. Writer plumbing (`OTR_LedgerScriptWriter`)

- Flag `OTR_ENABLE_STYLE_GRAMMAR` (env), read in the writer. OFF => nothing runs,
  no telemetry key => byte-identical.
- ON, after `generate_outline` returns and the cast_seed is known:
  `slug = select_style(outline.premise, meta, cast_seed)`;
  `ending_tag = ENDING_TAG_BY_SLUG[slug]` (fallback default);
  `template = ENDING_TEMPLATES[ending_tag]`. Stamp telemetry (D).
- Pass `(ending_tag, template, final_char_beat_id)` into the line composer so the
  final-beat dialogue request appends the ending instruction. final_char_beat_id
  = the `irreversible_choice` beat id from the already-assigned roles
  (`roles_by_beat`), NOT "last in list".
- Optional (flag ON): override `meta.style` with the selected slug so downstream
  visuals (HUD/FLUX/info-card) match — gated; OFF keeps the early picker value.

## D. Line-composer injection (`_otr_line_composer`)

- Add `ending_template: str = ""` (+ a final-beat marker) to the composer's
  LineRequest/prompt path; APPEND the ending instruction to the system/user
  prompt ONLY for the final character beat AND only when non-empty (empty =>
  byte-identical line prompt, golden-asserted). This is the single behavioral
  injection.

## E. Announcer-outro gating (`_otr_outline._assemble_outline`)

Gate only the announcer close INTENT (do NOT remove it — budget validator #7):
flag OFF = exact existing string (byte-identical); flag ON = a non-outcome intent
("Close without explaining the outcome; identify the program only"). The flag
must reach `_assemble_outline` (via the request or a module-level read consistent
with the writer gate). Metric language: "last voiced CHARACTER beat".

## F. Telemetry

`meta.story_quality`: `style_slug`, `ending_tag`, `domain`, and the crisis-noun
count at the final beat (reuse `count_ungrounded_crisis`). Drives the soak metrics.

## G. Interaction / bundle

Style grammar + L1/L2 crisis-noun grounding (`OTR_STORY_QUALITY_L12`) ship as ONE
"story-grammar" bundle, on together (grammar = climax SHAPE, L1 = trope
VOCABULARY). T4 staging penalty complements §E. T2 critic = orthogonal. Negative
anti-trope ban CUT.

## H. Build order (each chunk suite + Bug Bible green, byte-identical asserted, push)

1. Catalog data contract + `validate_catalog()` (pure, no behavior).
2. `select_style` selector (pure, dark).
3. Line-composer `ending_template` injection + final-beat marker (dark).
4. Announcer-outro intent gate (dark).
5. Writer flag plumbing + telemetry + (optional) meta.style override.
6. Docs: L1/L2 bundle-on. Then the live A/B (GPU, last).

## I. Tests

- Flag OFF => line prompt + announcer intent + writer output byte-identical
  (golden over a fixture).
- `validate_catalog()` coverage.
- selector determinism (same premise/meta/cast_seed => same slug; emergency only
  on disaster keywords; default pool excludes emergency).
- Flag ON => final-beat line prompt carries the ending template; announcer intent
  is the non-outcome string.
- C7: full writer run, flag OFF => byte-identical (audio gate holds).

## J. Validation soak (baseline first)

Baseline (current code) then lever-ON: crisis-noun density at the final beat
(target 0), distinct `ending_tag` distribution (>= 80% non-doomsday), critic
`arc_verdict` mix, a ~6-episode graded A/B on a couple of seeds.

## Hard constraints (carried)

100% local default (frontier opt-in, separate); DARK/default-OFF/byte-identical;
deterministic + C7 seed path holds; canonical JSON edited in the same change as
any node/widget change (env-only gate => no JSON change); full suite + Bug Bible
green; UTF-8 no BOM; SFW; no new heavy model, no extra paid call.
