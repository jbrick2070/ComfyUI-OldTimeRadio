<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: build-ready as-is? no. The plan still has build-blocking ambiguity in the catalog data contract, selector determinism/domain taxonomy, and flag/bundle semantics; it also introduces a likely `ending_mode` compatibility regression against the current catalog helper.

MUST-FIX BEFORE BUILD:
1. [A] Catalog data contract is under-specified: the plan requires 100 `domain` + `ending_tag` assignments and 8 concrete `ENDING_TEMPLATES`, but provides neither the exact per-slug mapping nor the exact template strings. Two builders can make different valid catalogs and produce incompatible endings. Concrete fix: add a checked-in data diff/table listing, for every current `STYLE_CATALOG[*]["slug"]`, its exact `domain` and `ending_tag`, plus the exact 8 `ENDING_TEMPLATES` strings. `validate_catalog()` must assert `len(STYLE_CATALOG) == 100`, unique slugs, every slug has domain + ending_tag, every tag has a template, and every `ENDING_TAG_BY_SLUG` entry matches the catalog field.

2. [A] Fix-introduced regression: current `_otr_style_catalog.render_style_grammar()` indexes `s['ending_mode']`, and current entries have `ending_mode` + `tags`, not `ending_flavor`, `ending_tag`, or `domain`. Renaming `ending_mode` to `ending_flavor` will break existing callers unless all call sites are migrated. Concrete fix: either keep `ending_mode` as a backward-compatible alias with byte-identical `render_style_grammar()` output when the flag is OFF, or update `render_style_grammar()` to use `ending_flavor` and add a golden test proving pre-flag style grammar output is unchanged. Grounding: current `render_style_grammar()` emits `Ending mode: {s['ending_mode']}.`

3. [B] Selector algorithm is not precise enough to be deterministic. “deterministic index = sha256(...)” omits the integer conversion, modulo, and candidate ordering. Concrete fix: specify exact implementation, e.g. `candidates` in `STYLE_CATALOG` declaration order after filtering; `idx = int(hashlib.sha256(f"{cast_seed}:style:{domain}".encode("utf-8")).hexdigest(), 16) % len(candidates)`. Add test locking one known seed/domain to one known slug.

4. [B] Domain taxonomy is ambiguous and can produce empty candidate pools. Existing L1/L2 `select_domain()` returns domains like `education`, `paleontology`, `energy`, `astronomy`, `medicine`, `law`, `general`, etc.; current style catalog tags are genre/register labels like `suspense`, `mystery`, `emergency`, `drama`, not those domains. Concrete fix: state that style `domain` values must be exactly keys from `_otr_story_quality_l12.DOMAIN_PALETTE`, including `general`; `validate_catalog()` must assert at least one non-emergency candidate for every possible L1/L2 domain and define fallback order: domain candidates -> `general` candidates -> all non-emergency candidates -> all styles.

5. [B] Emergency gating is under-specified. “explicit disaster/rescue keywords” is not checkable and will diverge by implementor. Concrete fix: define an `EMERGENCY_KEYWORDS: tuple[str, ...]` in `_otr_style_catalog.py`, exact casefold matching rules, and exact fields searched in `premise/meta`. Add selector tests for one positive and one negative case.

6. [G/C/E] Flag semantics contradict each other. [C] introduces `OTR_ENABLE_STYLE_GRAMMAR`; grounding in `_otr_story_quality_l12.py` says L1/L2 runs only under `OTR_STORY_QUALITY_L12`; [G] says they ship as one bundle. The final beat id depends on L1/L2 `beat_role`, so style grammar cannot safely run independently. Concrete fix: define one effective helper used everywhere, e.g. `story_grammar_enabled()`, and state exact env semantics. If `OTR_ENABLE_STYLE_GRAMMAR` is the bundle flag, then when true the writer must always call `build_sq_data()` regardless of the old L1/L2 flag; if preserving `OTR_STORY_QUALITY_L12`, make it an alias with identical effect. OFF must mean both style ending and L1/L2 are skipped and no telemetry key is added.

7. [C] `final_char_beat_id` source is wrong as written. The plan says use `roles_by_beat`, but grounding shows `roles_by_beat` is local inside `_otr_story_quality_l12.build_sq_data()` and is not returned; the returned writer-visible object is `sq[beat_id]["beat_role"]`. Concrete fix: define final id as: scan `outline.beats` in order for `speaker_role == "character"` and `sq_by_beat[beat.beat_id]["beat_role"] == "irreversible_choice"`; assert exactly one when enabled; pass that beat id to the composer.

8. [E] Announcer-outro gate transport is still ambiguous and contradicts the sequencing constraint. [SEQUENCING] says `OutlineRequest` gets no new fields; [E] says the flag may reach `_assemble_outline` “via the request or a module-level read.” Concrete fix: choose one. Given the no-`OutlineRequest` constraint, use a module-level/env helper in `_otr_outline.py` with the same effective flag helper as the writer. Do not add fields to `OutlineRequest`. Add a test that flag OFF preserves the exact current intent string from grounding: `Close on a concrete final image showing what changed (use the central object if set); no moral, thesis, or news-summary tag.`

9. [A/C] Fail-soft ending tag lookup is contradicted by direct indexing. [A] says default to `_DEFAULT_ENDING_TAG = "revelation"` with LOUD warn, never raise; [C] says `ENDING_TAG_BY_SLUG[slug]`, which raises `KeyError`. Concrete fix: add `get_ending_tag(slug) -> str` and `get_ending_template(slug) -> str` helpers that use `.get(..., _DEFAULT_ENDING_TAG)`, log a warning on fallback, and never raise for unknown/missing slug.

SHOULD-FIX:
1. [D] The line-composer contract needs exact field names and defaults. “ending_template + a final-beat marker” is too loose for tests. Concrete fix: specify fields such as `ending_template: str = ""` and `is_final_character_beat: bool = False` or `final_char_beat_id: str = ""`; require empty/default values to be omitted from prompt text and serialization-compatible with existing golden fixtures. verify: actual `_otr_line_composer` request model shape.

2. [F] Crisis-noun telemetry is not fully specified. `count_ungrounded_crisis(intent, grounded)` requires a grounded palette, but [F] only says “reuse `count_ungrounded_crisis`.” Concrete fix: define `grounded = premise_noun_palette(roster, outline.premise, *premise_texts(meta))` and count on the final irreversible-choice beat’s post-`build_sq_data` intent.

3. [I] Add a regression test for `render_style_grammar()` compatibility. Current catalog helper is public and existing behavior may be used outside this new flag path. Test unknown slug still returns `""`; known slug still renders the same text when style grammar flag is OFF.

4. [A] `validate_catalog()` should also check emergency filtering invariants: every `ending_tag` is in `ENDING_TAGS`; every `domain` is in the L1/L2 domain set; every emergency style is still identified by existing `tags == "emergency"` or an explicitly documented replacement.

5. [H] Build order should run writer plumbing before or alongside line-composer golden tests. Chunk 3 can add dead fields, but the meaningful “final beat only” test needs writer-derived `final_char_beat_id` from [C]. Concrete fix: either split unit vs integration tests explicitly or move writer plumbing before the full prompt-injection tests.

OPTIONAL / NICE-TO-HAVE:
- Add `selected_style_debug` log line under flag ON with `domain`, candidate count, selected slug, ending tag, and final beat id.
- Add a small self-test in `_otr_style_catalog.py` analogous to `_otr_outline.py` for selector determinism and catalog validation.

CUT THESE:
1. [C] Cut the optional `meta.style` override. It is not required to stop the doomsday-button climax, it risks downstream visual/HUD drift, and “optional” leaves two valid incompatible builds. If desired later, ship as a separate visual-alignment change.

2. [G] Cut “T4 staging penalty complements §E” and “T2 critic = orthogonal” from this build plan. They do not affect the local deterministic ending injection and add cross-feature scope with no concrete implementation step here.

3. [J] Cut or defer “critic `arc_verdict` mix” from the required soak. The build goal is final-beat ending shape and crisis-noun density; critic mix is orthogonal and not grounded in the shown code.

4. [J] Cut the “>= 80% non-doomsday” distribution target unless “doomsday” is made a measurable tag/category. With the proposed 8 tags, none is literally named doomsday, so the metric is currently non-actionable.

VERIFY-AT-BUILD checklist:
1. [ASSUMPTION] Earlier UNVERIFIABLE flag: writer pipeline order. Verify in `OTR_LedgerScriptWriter` that the actual order is `generate_outline()` returns -> `build_sq_data()` runs/mutates beat intents -> line composer receives each beat request. Grounding supports L1/L2 behavior, but writer code is not shown.

2. [ASSUMPTION] Earlier UNVERIFIABLE flag: `cast_seed` availability. Verify `cast_seed` is known after outline generation and before line composition, and that the same seed path is used for selector determinism.

3. [C/G] Verify effective flag helper is shared by writer and `_otr_outline._assemble_outline`; OFF produces no `meta.story_quality` key and no line prompt/announcer intent changes.

4. [D] Verify `_otr_line_composer` can add the ending fields with defaults without changing prompt text, serialized request shape, or golden output when disabled. verify: actual LineRequest implementation.

5. [E] Verify announcer close is not removed and budget validator #7 still passes. Grounding shows validator #7 requires announcer count equals `budget.announcer_beats`.

6. [C] Verify final beat marker targets the last voiced CHARACTER beat with `beat_role == "irreversible_choice"`, not the final announcer beat. Grounding: `build_sq_data()` assigns `irreversible_choice` to the last character beat and announcer beats get empty `beat_role`.

7. [A/B] Verify `_otr_style_catalog.py` import remains pure/local: no model load, no I/O, no env read except inside explicit selector/gate helpers.

8. [A] Verify no `KeyError` path remains for missing `ending_tag`, missing template, or unknown slug; fallbacks warn and use `_DEFAULT_ENDING_TAG`.

9. [Hard constraints] Verify canonical JSON is unchanged for env-only gate; if node/widget surface changes are introduced, update canonical JSON in the same change.

10. [I] Run C7/full writer fixture with flag OFF and assert byte-identical writer output/audio gate, including line prompts and current announcer close intent.