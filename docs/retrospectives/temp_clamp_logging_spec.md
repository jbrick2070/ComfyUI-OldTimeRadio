# Temperature-Clamp Logging Spec -- Sprint C retrospective §1 / §2 / §5 / §7 claim

**Triage branch:** `triage-sprint-c-retrospective-2026-05-15`
**Verification mode:** read-only against `nodes/_otr_story_brief.py` on `sprint-c-story-brief-v2`. No source modifications. Spec only.
**Source verified via:** `git cat-file -p sprint-c-story-brief-v2:nodes/_otr_story_brief.py` (file does not exist on `main` -- introduced at Sprint C C5a1, commit `87f01bd`).

## Retrospective claim being verified

§1 "Empirical Brittleness", §2 "Sprint Pressure and Accrued Technical Debt", §5 "The Blind Handoff Pattern", and §7 "Cross-Sprint Hand-off Integrity" all reference the same alleged silent-clamp pattern:

> "the reflection module guarantees structural JSON compliance by severely constraining the semantic diversity and creative coherence of the 8-key reflection pass. The pipeline continues to operate, presenting green status lights to the central orchestrator, while silently degrading the quality of the meta.story_brief."

> "Require the C5a1 exception block to emit a highly visible, persistent flag in the main generation log when temperature is clamped."

## File path verified

`nodes/_otr_story_brief.py` -- NOT `nodes/_otr_story_brief_reflection.py` as the triage prompt phrased it. The file exists only on Sprint C branches (`sprint-c-story-brief-v2` and descendants); `main` does not yet have it.

The reflection entrypoint is `run_story_brief_reflection(...)` at approximately `nodes/_otr_story_brief.py:580` (line numbers cited below are from the `sprint-c-story-brief-v2` blob `aeda67e...`).

## Verified facts

### 1. Three scoped try/except arms exist (E-17 / RR-B3 / L-6 confirmed)

The `run_story_brief_reflection` body wraps three distinct narrow `try` blocks at:

- Block 1 -- LLM call, `try` at line 603, `except Exception as exc` at line 609. Returns `_failure_sentinel(reason="technical_fn_exception")` on failure.
- Block 2 -- JSON parse, `try` at line 623, `except json.JSONDecodeError as exc` at line 625. Returns `_failure_sentinel(reason=REJECT_JSON_PARSE)`.
- Block 3 -- pydantic schema validation, `try` at line 640, `except ValidationError as exc` at line 642. On failure invokes `_repair_pass(...)` (line 649) and either accepts the repaired output or returns `_failure_sentinel(reason=REJECT_SCHEMA)`.
- Additional content-validation block at line 673-714: if pydantic shape passed but `_validate_brief` returns rejection reasons (named characters / dialogue verbs / etc.), `_repair_pass(...)` is invoked at line 680 and either accepted or returns `_failure_sentinel(reason="content_validation_failed_after_repair" | "repair_pass_exception")`.

The retrospective's "three narrow exception arms" claim aligns with the source.

### 2. The temperature clamp lives in `_repair_pass` -- exact site

`nodes/_otr_story_brief.py:487-490`:

```python
repair_temperature = min(
    reflection_temperature + _REPAIR_TEMPERATURE_BUMP,
    _REPAIR_TEMPERATURE_CEILING,
)
```

Constants (declared at module scope):
- `_REPAIR_TEMPERATURE_BUMP: float = 0.15` (line 65)
- `_REPAIR_TEMPERATURE_CEILING: float = 0.55` (line 60)
- `_REFLECTION_TEMPERATURE` is the base value passed in by callers; default is in the same section near line 49-50 (refinement section 3.2: "temperature 0.2-0.4 keeps the model anchored to the schema").

The retrospective's framing of "clamps temperature to 0.55" is **mathematically incomplete but directionally correct.** The actual computation is `min(reflection_temperature + 0.15, 0.55)`. The clamp pins at 0.55 only when `reflection_temperature >= 0.40`. The repair-temperature ceiling is hard at 0.55.

### 3. The clamp triggers on validation-rejection paths, NOT on a "CRITICAL" exception

The retrospective frames the clamp as "triggered by a CRITICAL prefix." The actual trigger is one of two validation rejection states:

- Schema validation rejection (Block 3, line 642-668). Pre-repair log at line 643-646: `log.warning("[OTR_StoryBrief] schema validation failed (%s); attempting repair pass", exc)`.
- Content validation rejection (post-schema, line 673-714). Pre-repair log at line 675-678: `log.info("[OTR_StoryBrief] content validation rejected: %s; attempting repair pass", content_reasons)`.

The "CRITICAL" prefix is a **prompt-engineering string** prepended to the user message inside `_build_repair_messages` (line 459-470), per R-06 / C0b refinement section 3.5:

```
CRITICAL: You previously failed validation because: <reasons>.

Rewrite this visual brief to obey the schema. ...
```

It is a textual instruction to the LLM, not an exception class. The retrospective conflates the two.

### 4. The clamp emits NO log line at the call site (confirmed silent)

Inside `_repair_pass` (lines 473-498):

```python
def _repair_pass(...):
    repair_temperature = min(
        reflection_temperature + _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING,
    )
    messages = _build_repair_messages(
        failed_output, rejection_reasons, base_user_message,
    )
    return technical_fn(
        messages,
        temperature=repair_temperature,
        max_new_tokens=_REFLECTION_MAX_NEW_TOKENS,
    )
```

Between the clamp computation (line 487) and the LLM call (line 494), there is **no `log.*` call.** The surrounding "attempting repair pass" log messages at lines 643-646 and 675-678 mention WHY the repair pass fires (the validation rejection reasons) but **do NOT report**:

- the resulting `repair_temperature` value,
- the base `reflection_temperature` value, or
- whether the clamp pinned at the 0.55 ceiling.

The retrospective's specific claim that "this temperature clamping operates entirely silently during a try/except fault recovery" is **VERIFIED.** A downstream researcher reading the log will see only "attempting repair pass" with rejection reasons; they will have no observability into whether the repaired output came from a 0.4 + 0.15 = 0.55-ceilinged retry or a 0.2 + 0.15 = 0.35 retry. Identical-looking log streams can mask very different generative-entropy paths.

## Minimal additive log patch spec (DO NOT APPLY -- spec only)

### Target file

`nodes/_otr_story_brief.py` on the Sprint C branch lineage (`sprint-c-story-brief-v2` HEAD = `a125a35`; the same file on any Sprint A branch cut after the C5a1 commit `87f01bd`).

### Exact insertion site

Between line 490 (end of the `min(...)` expression) and line 491 (the `messages = _build_repair_messages(...)` call). One new log line, two lines of source including the trailing blank already present.

### Proposed log line text (one option, follows existing convention)

```python
    repair_temperature = min(
        reflection_temperature + _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING,
    )
    log.info(
        "[OTR_StoryBrief] repair pass clamped: base=%.3f bump=%.3f "
        "ceiling=%.3f -> repair_temperature=%.3f reasons=%s",
        reflection_temperature, _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING, repair_temperature, rejection_reasons,
    )
    messages = _build_repair_messages(
        failed_output, rejection_reasons, base_user_message,
    )
```

Why `log.info` and not `log.warning`:

- The clamp itself is a designed pre-flight behavior, not an unexpected condition (warning territory).
- The reason for the repair pass is already logged at `log.warning` (schema failure) or `log.info` (content rejection) before this point. The new line is purely a temperature-observability addition.
- An `INFO`-level line keeps log-volume manageable -- Sprint A researchers performing visual inspections will see it; production noise stays bounded.

### Why this is purely additive (no-change-logs rule preserved)

Sprint C standing directive (closed-sprint doc §5):

> "**No-change-logs rule (operator directive):** existing runtime log strings stay byte-stable. ... New log lines added at C5a1 (reflection failure sentinels), C5a2 (eviction notice), C5c-C5f (story_brief_status observability), and C5g (mood-prefix status) follow the same format conventions as their neighboring log lines; no surrounding existing line is modified."

The proposed addition:

- Adds ONE new log line. Does not modify any existing log string.
- The existing log strings at lines 610-612, 626-629, 643-646, 660-662, 675-678, 694-697, 706-708 remain byte-stable.
- Uses the same `[OTR_StoryBrief]` prefix and `%`-formatting style already used by all 7 existing `log.*` calls in this module.
- Uses `log.info` (existing convention -- see line 675's `log.info`) rather than introducing a new severity class.

Compliant with the rule.

### Why this is a Sprint A change, not a Sprint C amend

- Sprint C is closed. Re-opening the closed branch to land a log-line addition would violate the closed-sprint discipline.
- The clamp-observability gap is **not a Sprint C audio-C7 or pytest-gate failure** -- the existing 2276-pytest suite holds with the silent clamp intact.
- The directly affected Sprint A task is S-A.3 motion-priority manual visual inspection (per closed-sprint §C-final.5 / §9 of the Sprint C plan). If Sprint A inspectors see semantically-impoverished outputs, the new log line lets them disambiguate "LLM was weak" from "repair-pass ran with clamped temperature."

The recommended landing is the first Sprint A runtime-verification commit (the same commit that captures the audio C7 b3sum baselines per the closed-sprint Post-State Contract).

### Test coverage to add alongside the patch

Two simple additions to `tests/test_story_brief_reflection_pure_c5a1.py` (the file added by C5a1 -- exact name to be confirmed at Sprint A time):

1. `test_repair_pass_emits_clamp_log` -- monkeypatch `nodes._otr_story_brief.log`, force a schema-rejection path, assert exactly one `log.info` call with substring `"repair pass clamped"` and the resulting `repair_temperature` formatted to 3 decimals.
2. `test_repair_pass_clamp_log_does_not_break_no_change_logs_rule` -- AST parse the module, collect all `log.*` call sites, assert the existing log strings are byte-identical to a pinned snapshot (the new line is the only delta).

Both tests are pure pytest -- no GPU, no runtime gate. They can land in the same Sprint A commit as the source patch.

## Cross-references to other deliverables

- The Blind Handoff Pattern (retrospective §7) explicitly requests this log line. Deliverable 4 (Sprint A acceptance rows) carries the corresponding row.
- Deliverable 1 found a separate silent-failure class around `widgets_values` canonical shapes. That fix and this one are independent but both expand observability into Sprint A's empirical-verification pass.

## Sources cited (in-repo, read-only)

- `nodes/_otr_story_brief.py` on `sprint-c-story-brief-v2` (blob `aeda67ee...`)
- `docs/AI_Production_Pipeline_Retrospective__Sprint_C.md` §1, §2 "Error-Masked Degradation", §5 "Defensive Logic Misalignment", §7 "The Blind Handoff Pattern"
- `docs/closed-sprints/2026-05-15-sprint-c-story-brief-v2.md` §1.2 E-17, E-18, R-06, §5 standing directives, §C-final.5 post-state contract
