# Ending-mode design — R2-hardened (build-ready coding spec)

Stop the weak local writer collapsing every premise into the console/kill-switch
climax: give the FINAL character beat a concrete, style-driven ENDING that
reframes the (still on-stage, still `irreversible_choice`) climax away from the
doomsday button. DARK / default-OFF / byte-identical. 100% local. Deterministic.

## A. Catalog data contract (`nodes/_otr_style_catalog.py`)

- `ENDING_TAGS: tuple[str,...]` = (`revelation`, `reversal`,
  `unresolved_final_sound`, `reconciliation`, `bittersweet_parting`,
  `ironic_twist`, `quiet_acceptance`, `confession`).
- `ENDING_TEMPLATES: dict[str,str]` — each tag → ONE concrete final-beat
  instruction (what literally happens / the last sound; "no machinery").
- Each of the 100 entries gains `ending_tag` (one of ENDING_TAGS) + a `domain`
  (the l12 domain it best serves) + keep the prose as `ending_flavor`.
- Expose `ENDING_TAG_BY_SLUG: dict` (external map) so the tag is read ONLY by the
  new path — never serialized/hashed by the old picker (byte-identity guard).
- `_DEFAULT_ENDING_TAG = "revelation"` — fail-soft: unknown/missing tag => default
  + LOUD warn, never raise.
- `validate_catalog()` self-check (run in a test): every entry has a valid
  `ending_tag` + `domain` + a template exists. Fail loud on a gap.

## B. Deterministic style selector (replaces the LLM inventor)

- `select_style(premise, meta, cast_seed) -> slug` in `_otr_style_catalog.py`.
  Reuse l12 `select_domain(meta, premise)` for the domain (do NOT duplicate the
  keyword map). Candidate set = catalog styles whose `domain` matches, filtered
  to `non_emergency_slugs()` UNLESS the premise/meta carry explicit
  disaster/rescue keywords (then emergency styles are eligible).
- Deterministic tie-break: `sha256(f"{cast_seed}:style:{domain}")` indexes the
  candidate list. cast_seed is the writer's existing per-episode seed (forwarded
  into the request) — keeps the C7 byte-identity path.
- Runs ONLY when the flag is ON; OFF => the current style picker path is
  untouched (byte-identical).

## C. Flag + OutlineRequest plumbing

- `OutlineRequest` (frozen) gains: `enable_style_grammar: bool = False`,
  `ending_tag: str = ""`, `ending_template: str = ""` (all defaulted => empty =>
  byte-identical, asserted). Filled via `dataclasses.replace`, mirroring how the
  pitch room fills `script_brief`.
- Flag source = `OTR_ENABLE_STYLE_GRAMMAR` (env) read in the writer (not in the
  outline module), threaded onto the request — mirrors the pitch-room gate.

## D. Final-beat detection + injection (`nodes/_otr_outline.py`)

- In `generate_outline()`, after the phase skeletons are built, compute the final
  voiced CHARACTER beat coordinate ONCE: `(last_phase_idx, last_beat_idx)` from
  the last non-empty phase skeleton. (Do NOT use `validate_beat_roles` for this —
  it validates a post-hoc mapping, not a live coordinate.)
- `_build_beat_user_prompt(..., is_final_character_beat: bool = False)` — when
  `req.enable_style_grammar` AND `req.ending_tag` AND `is_final_character_beat`,
  APPEND the ending instruction (`ending_template`) to the beat prompt. Empty /
  not-final => byte-identical prompt (golden-asserted).
- Pipeline ORDER (pin): assign beat roles → ground crisis nouns (L1) → (if flag)
  inject ending template at the final character beat → `stamp_dialogue_slot_ids`.
  Never mutate `beat.intent` before role assignment.

## E. Announcer-outro gating

`_assemble_outline()` appends a voiced announcer close after the last character
beat (intent "Close on a concrete final image showing what changed..."). Do NOT
remove it (breaks budget validator #7). Gate only its INTENT:
- flag OFF: exact existing intent string (byte-identical).
- flag ON: a non-outcome intent — "Close the episode without explaining the
  outcome; identify the program only."
Metric language: "last voiced CHARACTER beat", not "final voiced beat".

## F. Telemetry (for validation)

Stamp on `meta.story_quality`: the selected `style_slug`, `ending_tag`, the
`domain`, and the crisis-noun substitution count at the final beat (reuse
`count_ungrounded_crisis`). Enables the soak metrics without new instrumentation.

## G. Interaction / ordering with existing levers

Style grammar + L1/L2 crisis-noun grounding (`OTR_STORY_QUALITY_L12`) form ONE
"story-grammar" bundle, turned on together (grammar fixes climax SHAPE, L1 fixes
trope VOCABULARY). T4 staging penalty complements §E. T2 critic adapter is
orthogonal measurement. The negative anti-trope ban is CUT (redundant with L1 +
the positive ending tag).

## H. Tests

- `ending_tag=""` / flag OFF => beat prompt + beat roles + announcer intent
  byte-identical (golden over a fixture outline).
- `validate_catalog()`: every entry has a valid tag + domain + template.
- selector determinism: same (premise, meta, cast_seed) => same slug; emergency
  slug only on disaster keywords; default pool excludes emergency.
- flag ON: announcer close intent becomes the non-outcome string; final character
  beat prompt carries the ending template.
- C7: full writer run, flag OFF => output byte-identical (audio gate holds).

## I. Validation soak (baseline first)

Record current-code baseline, then lever-ON: crisis-noun density at the final
beat (target 0), distinct `ending_tag` distribution (>= 80% non-doomsday), critic
`arc_verdict` mix, a ~6-episode graded A/B on a couple of seeds.

## Hard constraints (carried)

100% local default (frontier opt-in, separate); DARK/default-OFF/byte-identical;
deterministic + C7 seed path holds; edit canonical JSON in the same change as any
node/widget change; full suite + Bug Bible green; UTF-8 no BOM; SFW; no new heavy
model, no extra paid call.
