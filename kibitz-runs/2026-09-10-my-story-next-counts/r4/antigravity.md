VERDICT: yes-with-fixes. The core architecture across Sprints A, B, and C has largely converged, but five residual code-level defects (missing character count parameters in treatment, act number validation blocking normalization, Pydantic schema min_length failing optional frames, title override fallback triggering unintended LLM regen, and an unaddressed regression in an existing pipeline test) must be resolved before build lock.

MUST-FIX BEFORE BUILD:

1. [Sprint A1, Item 1 & 2] Treatment Signature & Initial Prompt Missing Selected Character Count Authority
   - Defect: In nodes/_otr_my_story.py:529-592, `_pass_treatment` and `_make_treatment_validator` only accept `act_count: int` and rely on `interp.cast_plan.planned` (`planned = max(1, int(interp.cast_plan.planned))`). They do not accept the user's binding selected character count (`requested_characters`). Furthermore, the initial treatment prompt at lines 575-576 states `"Acts requested: %d"` but completely omits the requested character count, leaving the model uninformed of the binding character constraint until post-validation rejects it.
   - Fix: Update `_pass_treatment` and `_make_treatment_validator` to take `requested_characters: int` (forwarded from `raw_requested` / `num_characters` in `run_my_story_episode`). In `_make_treatment_validator`, validate `len(names) == requested_characters` (excluding announcer). Add `"Characters requested: %d (announcer excluded).\n"` to the initial user prompt in `_pass_treatment`. Pass a dedicated treatment repair prompt factory closure over `bundle`, `act_count`, and `requested_characters` to `structured_call` that emits the full `failed_output` (bypassing `default_repair_prompt_factory`'s 400-char truncation) and explicitly commands reorganizing events into the exact selected act and character counts.

2. [Sprint A1, Item 4] Act Number Validator Rejection Blocks Number Normalization
   - Defect: In nodes/_otr_my_story.py:604-605, `_make_act_validator` rejects any act where `int(model.n) != n`:
     `if int(model.n) != n: return "this is act %d, not act %d" % (n, model.n)`.
     This directly contradicts Section A1.4 ("Normalize accepted act replies to their requested slot before assembly/seal"). If the model returns valid dialogue with `n=0` or `n=2` for slot 1, the validator rejects it and forces an LLM repair turn rather than normalizing it.
   - Fix: Remove `if int(model.n) != n:` from `_make_act_validator`. Normalize `act.n = n` in Python upon accepting the response before appending to `acts` and assembling the ledger.

3. [Sprint A1, Item 6] `StoryFrame` Pydantic Schema `min_length=1` Triggers Validation Error on Optional Frame Elements
   - Defect: Section A1.6 states: "Frame text and music descriptions are optional... Skip blank spoken rows before assigning line numbers/boundaries." However, in nodes/_otr_my_story.py:286-293, `StoryFrame` defines:
     `announcer_intro: list[str] = Field(min_length=1)`
     `announcer_outro: list[str] = Field(min_length=1)`
     `coda: str = Field(min_length=1)`
     `music_open: str = Field(min_length=1)`
     `music_close: str = Field(min_length=1)`
     If a model returns empty strings or empty lists for any optional frame text or cues, Pydantic's schema validation raises `ValidationError` during parse, before `_make_frame_validator` ever runs.
   - Fix: Update `StoryFrame` schema in `_otr_my_story.py` to remove `Field(min_length=1)`: set `announcer_intro` and `announcer_outro` to `list[str] = Field(default_factory=list)`, and `coda`, `music_open`, `music_close` to `str = ""`. In `_make_frame_validator` (nodes/_otr_my_story.py:688-704), remove the `any(not text.strip() ...)` blank check.

4. [Sprint A1, Item 6] Missing Title Override Fallback Triggers Unintended Shared Title Regeneration Call
   - Defect: Section A1.6 states: "MyStoryTailParts.final_title_override is optional: a nonblank accepted title wins; missing/blank passes None to the shared producer... No extra title call or title rejection." But in nodes/_otr_writer_tail.py:735-745 and 807-832, passing `final_title_override=None` drops into the `else:` branch which invokes `_generate_title_from_script` via `slot_scheduler.helper_context("generate_title")` — executing an extra LLM call. Furthermore, in nodes/OTR_LedgerScriptWriter.py:1555-1558, `headline_override` is only populated when `seed_source == "original_llm"`, leaving `my_story_fields` (`seed_source == "my_story_fields"`) un-synchronized.
   - Fix:
     a. In `run_my_story_episode`, if `treatment.title` is blank, fallback deterministically to `bundle.normalized.headline or bundle.normalized.title or "My Story Episode"` so `final_title_override` is always a non-empty string, preventing an unwanted LLM title call in the shared tail.
     b. In `OTR_LedgerScriptWriter.py:1556`, update the headline condition to:
        `headline_override=(final_title if resolved["seed_source"] in ("original_llm", "my_story_fields") else "")`.

5. [Sprint B, Testing Scope] Existing Regression Test `test_original_radio_pipeline.py` Broken by Credits Overwrite
   - Defect: In tests/test_original_radio_pipeline.py:50, the test explicitly asserts:
     `assert "generated by machine" in d.get("credits_source_line", "")`.
     Sprint B overwrites `credits_source_line` for `meta.source_bank == "original"` to `"Story generation models used: <identities>"`. Running the full regression test suite will immediately fail on this assertion.
   - Fix: Add an explicit test migration task to Sprint B to update `test_original_radio_pipeline.py:50` to assert `d.get("credits_source_line", "").startswith("Story generation models used:")`.

SHOULD-FIX:

1. [Sprint A1, Item 7] `_assemble` Does Not Accept `include_act_breaks` and Silently Drops Surplus Cues
   - Defect: In nodes/_otr_my_story.py:816 and 896-906, `_assemble` does not accept `include_act_breaks: bool`. If `include_act_breaks` is False, it still emits cues if `frame.music_inter` is populated. If `frame.music_inter` has fewer cues than boundaries, it silently stops emitting cues without using an empty prompt fallback. Surplus cues are discarded with no record.
   - Fix: Add `include_act_breaks: bool` to `_assemble`. If False, emit 0 interstitial cues. If True, emit exactly `len(acts) - 1` cues (using `frame.music_inter[i]` or `""` if exhausted). Record any unplaced proposals in `meta["my_story"]["music_cue_disposition"]` with disposition `"unused_surplus"`.

2. [Sprint A1, Item 1 & 3] Telemetry Note Contradiction in `_otr_my_story.py`
   - Defect: In nodes/_otr_my_story.py:1033, `story["notes"]` contains the hardcoded string: `"Requested and actual counts are telemetry, never gates."` This directly contradicts the operator's binding count ruling.
   - Fix: Update the note in `_otr_my_story.py:1033` to: `"Selected act and character counts are binding constraints enforced at treatment acceptance."`

3. [Sprint C] Overlong Token Split Boundary in `_flow_col1`
   - Defect: In nodes/otr_credits_roll.py:697-708, existing `_wrap` appends an unbroken overlong word when `not cur`. The plan specifies splitting overlong tokens without losing characters. If an implementor uses character slicing without measuring glyph widths, wide glyphs (e.g. uppercase 'W', 'M') could still overflow `col1_w`.
   - Fix: Ensure the custom token-splitting loop in `_flow_col1` incrementally checks `_fw(draw, partial, font) <= col1_w` character by character before pushing a split segment, advancing `y += _fh(font)` for each line.

OPTIONAL / NICE-TO-HAVE:

1. [Sprint A1, Item 3] In `_make_treatment_validator`, explicitly guard against a character named `"ANNOUNCER"` in `treatment.cast` to prevent collisions with `ANNOUNCER_NAME` during voice assignment in `_assign_voices`.

CUT THESE:

1. [Lines 366-756 in input.md / GO_FORWARD_PLAN.md] Stale alpha.19-21 Registry Discussions and Fragmented Archive Notes
   - Why safe to cut: Lines 366-549 discuss obsolete registry reviews for alpha.19-21 (`pyproject.toml` is at alpha.29). Lines 552-756 contain broken, truncated sentence fragments copied from old archives that are completely outside the scope of Sprints A, B, and C. Cutting them prevents builder distraction and preserves document integrity.

VERIFY-AT-BUILD checklist:

1. [Canonical Links & Widgets] Verify `workflows/otr_canonical.json` retains 23 nodes, 63 links, 37 writer widgets, and link 291 connecting writer 1 to mux 85. No graph mutation permitted.
2. [Live Component Dry-Run] Run canonical component execution targeting `partial_execution_targets: ["62"]` using prompt `e11e5d86-2dda-41fa-b942-bb927a754b83` (requested 1 act, 2 characters, local Qwen3.5-4B). Verify the repair factory recovers 1 act and 2 characters and freezes cleanly.
3. [Variable Count Scaling] Verify test suite exercises requested act counts 1, 3, 6 and character counts 1, 2, 4, matching controls without Python text truncation.
4. [Credits PNG Visual Proof] Render `otr_credits_roll` with the exact long Lantern title at 832x480, 512x288, and 1080p. Confirm all hero glyphs stay strictly inside Column 1 (`x < 654`) and subtitle/metadata follow without overlap.
5. [OpenRouter Identity Receipt] Verify `make_openrouter_generate_fn` captures the server-reported concrete model slug in `receipt_out` on success and stamps `meta.model_call_provenance` without retaining references to closures, models, or tokenizers.
6. [Bank Attribution Isolation] Verify `credits_source_line` is overwritten ONLY when `meta.source_bank == "original"`. Confirm My Story retains the listener byline and Public Domain/Shakespeare retain source author credits.

Marked assumptions:
- [ASSUMPTION] When `treatment.title` is blank in My Story, using `bundle.normalized.headline or bundle.normalized.title` as the fallback `final_title_override` is preferred over allowing the shared producer to make an extra LLM call to `generate_title`.
