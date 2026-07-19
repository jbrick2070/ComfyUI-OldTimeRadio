# 2026-07-18 SOFT-normalizer sprint plan

**Baseline:** HEAD `d6b0706e` (`v2.0-alpha`). **Docs window** produced this plan;
a coder window executes it. **Operator directive:** "if LLMs are killing the flow
with the ledger intact they should not as long as ledger is obeyed."

**Governing contract:** `docs/SOURCE_BANK_PREFLIGHT.md` Gate 3 -- "Python creates
only mechanical data such as IDs, order, references, enums, counts, hashes, and
validated routing metadata... mechanical serialization of already accepted
verbatim rows is allowed" (L146-149); "no model-produced or unused count field
can gate production" (L166-167). Both P2 nickname markers and P5 leading
vocatives are Python-fixable line-content quirks upstream of ledger closure.

**Advisory ladder pattern (copy source):** commit `ed7b37de` shipped the fable2
`advisory_budget_defects` ladder at `nodes/_otr_scifi_fable2.py:1949-2014`
(`if budget_defects and budget_rerolls < _MAX_BUDGET_REROLLS:` reroll, else
accept + record). Constant at `nodes/_otr_scifi_fable2.py:190`
(`_MAX_BUDGET_REROLLS = 2`). Downstream advisory recording at
`nodes/_otr_scifi_fable2.py:3893-3894` and `:3934` (`advisory_budget_defects`
key in p3/p5 return meta). Codex chunks below MIRROR this pattern.

**Existing normalizer references (grep-verified):**
- `_is_canonical_character_name` -- `nodes/_otr_scifi_codex.py:225-241`.
- `_CAST_NAME_PREFIX_RE` (`Dr. / Prof.` prefix regex) --
  `nodes/_otr_scifi_codex.py:218`.
- `_CAST_NAME_WORD_RE` (Title-Case token) -- `nodes/_otr_scifi_codex.py:219`.
- `_CAST_NAME_ACRONYM_RE` (2-3 letter acronym token) --
  `nodes/_otr_scifi_codex.py:222`.
- `repair_cast_plan_metadata` (existing typed repair for announcer name) --
  `nodes/_otr_scifi_codex.py:259-273`.
- `scrub_self_vocative` (deterministic scrub, self-vocative-safe, idempotent) --
  `nodes/_otr_line_hygiene.py:69-94`.
- `clean_spoken_character_line` (parenthetical + self-vocative combined scrub) --
  `nodes/_otr_line_hygiene.py:98-100`.
- Codex P2 invocation + typed-repair hook --
  `nodes/_otr_scifi_codex.py:3310-3315` + `:2838-2846`.
- Codex P5 post-validator + retry --
  `nodes/_otr_scifi_codex.py:3442` (invoke, `max_attempts=3` at `:3000`).
- Codex `_spoken_error` (line-text rejection surface) --
  `nodes/_otr_scifi_codex.py:2222-2242`.
- Codex `validate_spoken_text_and_roster` (P5/P7/P9 gate) --
  `nodes/_otr_scifi_codex.py:2244-2273` (P2 uses `_validate_cast_plan:243-256`).

## Bake-off unblock checklist (P2 + P5 -> Mistral 120/420/720w bake-off)

Each item must land GREEN before the local Mistral-Nemo bake-off restarts:

- [ ] Chunk 1 (P2 Title-Case normalizer) shipped + full suite green.
- [ ] Chunk 2 (P5 self-vocative advisory ladder) shipped + full suite green.
- [ ] Full Windows suite + Bug Bible pass (no new fails, `advisory_budget_defects`
      keys preserved). Record deltas vs the 8082 baseline in the commit message.
- [ ] Live proof leg: `scifi_codex_v4` @ 120w Mistral-Nemo. Expected: RESULT
      SUCCESS + obs_publish + asset on disk. Ledger `meta.scifi_codex` contains
      `p2_normalizations` (list, possibly empty) and `p5_advisory_defects`
      (list, possibly empty) -- both RECORDED, never gated.
- [ ] Live proof leg: `scifi_codex_v4` @ 420w Mistral-Nemo (bake-off arm 1).
- [ ] Live proof leg: `scifi_codex_v4` @ 720w Mistral-Nemo (bake-off arm 1).
- [ ] Repeat 420/720w for `scifi_fable2` (arm 2) and `scifi_codex` base (arm 3).
- [ ] Fable-BLIND grade the 3-way transcripts, publish in
      `docs/2026-07-17-model-bakeoff-scoreboard.md`.


## Chunk 1 -- P2 cast-name Title-Case mechanical normalizer

**Goal:** eliminate stochastic P2 failures on `Maxwell 'Max' Hart` /
`maxwell hart` / `Dr. Maxwell "Max" Hart` style names that satisfy `CastPlanV4`
schema but trip `_is_canonical_character_name`. Ledger-safe: character
description / gender / role / voice slot are UNTOUCHED; only `name` is
mechanically Title-Cased with quoted middle tokens (nicknames) stripped.

**Class:** SOFT (Gate 3 mechanical-serialization exception; ledger row shape
unchanged; failure was cosmetic post-validator only).

**Files to touch:**

- `nodes/_otr_scifi_codex.py` (add normalizer + wire into `repair_cast_plan_metadata`)
  - LOC: +30 lines (new helper `_normalize_cast_name_mechanical`) + 4 lines
    inside `repair_cast_plan_metadata` (:259).
- `tests/test_scifi_codex_lane.py` (new focused normalizer test)
  - LOC: +25 lines (one class, three parametrized cases).

**Design (verbatim for coder):**

```python
# Insert after _CAST_NAME_ACRONYM_RE at nodes/_otr_scifi_codex.py:222.
_CAST_NAME_QUOTED_MIDDLE_RE = re.compile(
    r"\s+[\'\"‘’“”][^\'\"‘’“”]{1,20}"
    r"[\'\"‘’“”]"
)


def _normalize_cast_name_mechanical(name: str) -> str:
    """Mechanical name normalizer -- Gate 3 mechanical-serialization only.

    Preserves ``Dr. `` / ``Prof. `` prefix and a single trailing 2-3 letter
    acronym token. Strips ASCII / smart-quoted nickname tokens
    (``Maxwell 'Max' Hart`` -> ``Maxwell Hart``). Title-Cases each remaining
    word token; leaves acronym tokens untouched. Returns the input unchanged
    on any failure -- deterministic + idempotent (safe to re-apply).
    """
    try:
        raw = str(name or "").strip()
        if not raw:
            return name
        prefix_match = _CAST_NAME_PREFIX_RE.match(raw)
        prefix = prefix_match.group(0) if prefix_match else ""
        body = raw[len(prefix):]
        body = _CAST_NAME_QUOTED_MIDDLE_RE.sub("", body)
        tokens = [tok for tok in body.split(" ") if tok]
        normalized = []
        for tok in tokens:
            if _CAST_NAME_ACRONYM_RE.fullmatch(tok):
                normalized.append(tok)
            else:
                # strip trailing punctuation, Title-Case, restore trailing
                stripped = tok.strip(".,;:!?")
                trail = tok[len(stripped):]
                normalized.append(stripped[:1].upper() + stripped[1:].lower() + trail)
        return prefix + " ".join(normalized)
    except Exception:  # noqa: BLE001
        return name
```

**Wiring (mechanical pre-normalization in `repair_cast_plan_metadata` at
:259-273):** BEFORE the existing announcer-name check, apply
`_normalize_cast_name_mechanical` to every `row.name` (announcer excluded, since
the fixed `ANNOUNCER` uppercase form is contract-locked). If the normalized
form changes the row, set `changed = True`. Keep the announcer branch as-is.
Return the modified cast when `changed`. This is EXACTLY the extension seam the
existing repair typed hook already uses at `nodes/_otr_scifi_codex.py:2838-2846`
(codex `elif pass_id == "P2":` typed-repair calls `repair_cast_plan_metadata`
before the LLM structural retry), so the normalizer runs on the FIRST P2
failure attempt without a new call site.


**Advisory recording:** thread the count of normalized names into the P2
meta so the ledger records it, mirroring `advisory_budget_defects`. Suggested
key at the P2 return meta site: `p2_normalizations = [old_name -> new_name,
...]` (list of str). Cite `_otr_scifi_fable2.py:2014` as the template.

**Test to add:** `tests/test_scifi_codex_lane.py::test_p2_cast_name_mechanical_normalizer`:

- Input row name `Maxwell 'Max' Hart` -> normalized `Maxwell Hart` and
  `_is_canonical_character_name("Maxwell Hart") is True`.
- Input row name `maxwell hart` -> normalized `Maxwell Hart`.
- Input row name `Dr. Ada "Doc" Chen` -> normalized `Dr. Ada Chen`.
- Announcer row is never touched (`ANNOUNCER` stays `ANNOUNCER`).
- Idempotent: `norm(norm(name)) == norm(name)`.

**Expected suite delta vs the ed7b37de baseline (8082 passed / 32 skipped /
1 xfailed):** +1 new test => 8083 / 32 / 1 (both fable2 `advisory_budget_defects`
tests and the codex reconcile tests already covered by ed7b37de remain green).

**Ledger invariant:** the mechanical name normalization changes only
`cast[i].name` (a string field already recorded verbatim in the ledger from
`_assemble_ledger` at `_otr_scifi_codex.py:3234-3240`). No new fields,
no new ID space, no reorder. Byte-identical audio is unaffected (name has no
TTS text path -- the spoken text on each line is separately validated and
carries no self-referential vocative from a name change; see Chunk 2).

---

## Chunk 2 -- P5 self-vocative advisory ladder

**Goal:** eliminate stochastic P5 failures on `Edna, remember the plan.`-style
lines where the character addresses themselves. `nodes/_otr_line_hygiene.py:69`
already has the deterministic scrub -- it is simply not wired into the codex
P5 accept path. Under the Gate 3 mechanical-serialization principle a bounded
LLM reroll gives the model a chance to write cleanly; on reroll-budget
exhaustion, `scrub_self_vocative` is applied mechanically and the residual
recorded as `advisory_self_vocative_scrubs`, mirroring the fable2
`advisory_budget_defects` ladder shipped in ed7b37de.

**Class:** SOFT (Gate 3 mechanical serialization; line-row schema and
`speaker_role`, `boundary`, graph closure UNTOUCHED).

**Files to touch:**

- `nodes/_otr_scifi_codex.py` (add the advisory ladder around P5/P7/P9
  post-validation; extend `_validate_script_post` to be advisory-aware).
  - LOC: +40 lines (new helper `_scrub_and_record_self_vocative` +
    ladder-aware wrapper) + 8 lines threading advisory into P5/P7/P9
    invocations at `:3442`, `:3444`, `:3447`.
- `tests/test_scifi_codex_lane.py` (advisory-record test + suite bounded-repair
  test).
  - LOC: +40 lines (two test cases).

**Design (verbatim for coder):**

```python
# Import at top of nodes/_otr_scifi_codex.py.
from ._otr_line_hygiene import scrub_self_vocative

# New helper. Insert directly after validate_spoken_text_and_roster at :2273.
def _scrub_self_vocative_advisory(
    script: "ScriptArtifactV4",
    cast: "CastPlanV4",
) -> tuple["ScriptArtifactV4", list[str]]:
    """Mechanically scrub leading/trailing self-vocative from character
    line text and RETURN (new_script, scrubbed_line_ids). Idempotent; safe
    to re-apply. Per Gate 3, this is a mechanical serialization of already
    accepted line rows, so it does not violate the no-Python-prose rule.
    """
    locked = {row.char_id: row.name for row in cast.cast}
    scrubbed_ids: list[str] = []
    new_lines = []
    for line in script.lines:
        if line.char_id.startswith("music_") or line.skip:
            new_lines.append(line)
            continue
        name = locked.get(line.char_id, "")
        cleaned = scrub_self_vocative(line.text, name)
        if cleaned != line.text and cleaned.strip():
            new_lines.append(line.model_copy(update={"text": cleaned}))
            scrubbed_ids.append(line.line_id)
        else:
            new_lines.append(line)
    if not scrubbed_ids:
        return script, []
    return script.model_copy(update={"lines": new_lines}), scrubbed_ids
```


**Wiring (mirrors ed7b37de advisory ladder):**

1. In `_validate_script_post` (`_otr_scifi_codex.py:3206-3222`), split the
   `_spoken_error` self-vocative signal into an ADVISORY class (line begins
   with a self-vocative -- mechanically scrubbable) vs a HARD class
   (stage-direction/label/all-caps/non-lexical -- not scrubbable).
   The advisory class returns a distinguishable marker string
   `advisory:self_vocative:<line_id>` instead of raising, so
   `invoke_codex_structured` treats it as an ADVISORY defect rather than a
   structural retry trigger.
2. After the LAST accepted P5/P7/P9 invocation (final
   `validate_spoken_text_and_roster` call at `_otr_scifi_codex.py:3448`),
   apply `_scrub_self_vocative_advisory` to the accepted script BEFORE the
   final validator runs. The final validator then either passes clean or
   raises on a residual HARD defect. Scrubbed line_ids are appended to
   `meta["scifi_codex"]["advisory_self_vocative_scrubs"]`.
3. Reroll budget: reuse the existing `max_attempts=3` at
   `_otr_scifi_codex.py:3000` (no new constant needed). Once exhausted, the
   ladder falls through to the mechanical scrub above; if the scrub cannot
   empty the leading vocative (line would become empty), keep the original
   line text and record the line_id in the advisory list without altering
   the row -- ledger stays intact and the scrub is recorded, not gated.

**Ledger invariant:** the mechanical scrub only ALTERS `line.text` on rows
where the leading vocative is redundant (character addressing themselves) and
the residual line remains non-empty. `line_id`, `beat_id`, `shot_id`,
`char_id`, `speaker_role`, `boundary`, `arc_phase`, `compose_flags`,
`beat_intent`, `dialogue_slot_id` are all preserved verbatim
(`nodes/_otr_scifi_codex.py:3244-3252`). Text-hash stamping in
`_assemble_ledger` at :3266 (`sha256(v.encode('utf-8'))`) recomputes over
the scrubbed text, matching the mechanical-normalization principle
(Gate 3 L146-149).

**Byte-identical audio invariant:** `test_audio_byte_identical` locks the TTS
worker output for a FIXED input string. Since Chunk 2 changes the input
string (line text) only on lines that would otherwise HAVE FAILED (retry
storm), the byte-identical test's captured input strings are unchanged --
the affected lines are new content on a previously-failing path, not
existing golden rows. This is the same invariant preserved by ed7b37de's
count-gate ladder (Bug Bible 17 passed with the advisory recording live).

**Advisory recording:** `meta["scifi_codex"]["advisory_self_vocative_scrubs"]`
= list of `line_id` values whose text was mechanically scrubbed. Empty on a
clean run. Recorded, never gated.

**Tests to add** (`tests/test_scifi_codex_lane.py`):

- `test_p5_self_vocative_advisory_scrub`: a hand-crafted `ScriptArtifactV4`
  fixture where line `l001` = `"Edna, remember the plan."` and its cast row
  is `Edna Sparks`. After `_scrub_self_vocative_advisory` the line text is
  `"remember the plan."` and the returned list contains `l001`. Idempotent:
  re-applying returns `([], script)`.
- `test_p5_self_vocative_ladder_records_advisory_after_reroll_exhaustion`:
  a stubbed `creative_fn` that returns a script with a leading vocative on
  every attempt. Expected: no `CodexSpokenTextError`; the accepted script has
  the vocative scrubbed; `meta["scifi_codex"]["advisory_self_vocative_scrubs"]`
  contains the affected `line_id`; ledger closure (`_assemble_ledger`)
  completes clean.
- `test_p5_hard_defects_still_fail_closed`: an all-caps-word or
  stage-direction line STILL raises `CodexSpokenTextError` (advisory ladder
  does not swallow HARD content defects). Cite the same
  `_spoken_error:2225-2232` branches as untouched.

**Expected suite delta vs the ed7b37de baseline (8082 / 32 / 1):** +3 new
tests => 8086 / 32 / 1 (with Chunk 1's +1 => final 8087 / 32 / 1).

**Bug Bible:** add one entry for `p5_self_vocative_advisory_scrub` under the
line-hygiene rubric (references `_otr_line_hygiene.py:69` scrub function).

---

## Consolidated commit / push plan

Two chunks -> two commits, both to `v2.0-alpha`. Each commit runs the FULL
Windows suite + Bug Bible + AST + no-BOM verify before push. One push attempt
per green chunk. Never `--no-verify`. Never force-push.

- **Chunk 1 commit message:**
  `fix(codex): P2 cast-name mechanical Title-Case normalizer (Gate 3 mechanical serialization) -- SOFT normalizer, ledger intact`
- **Chunk 2 commit message:**
  `fix(codex): P5 self-vocative advisory ladder (Gate 3 mechanical serialization) -- SOFT normalizer, ledger intact, byte-identical audio preserved`

**Post-Chunk-2 live proof: MANDATORY** -- `scifi_codex_v4` 120w Mistral-Nemo
leg RESULT SUCCESS + obs_publish + asset on disk BEFORE any of the 420/720w
bake-off legs launch. Record ledger `meta.scifi_codex.p2_normalizations` and
`meta.scifi_codex.advisory_self_vocative_scrubs` in the HANDOFF_LOG entry so
the record shows the normalizers fired (or fired empty on a clean model draw).

## Deferred (NOT this sprint)

- `unused_shot` / `cast_coverage` fatal-vs-advisory decision (raised in the
  gates-fix-plan panel questions 3). Both were flagged as candidate Gate-3
  relaxations but never a live blocker in the 2026-07-18 evening leg. Leave
  fatal; revisit only if the P2+P5 fixes surface a new short-episode failure.
- `//15` words-per-beat scaling floor (gates-fix-plan question 4). The
  ed7b37de reconcile already handles the pathological case; no live proof of
  a bad floor. Do not re-open unless a proven failure appears.
