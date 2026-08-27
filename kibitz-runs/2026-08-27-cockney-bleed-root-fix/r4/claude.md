TOOL CHECK: PASS
Read nodes/_otr_dialogue_policy.py (37 lines, full file), nodes/_otr_line_composer.py:1030-1160,
nodes/_otr_compose_exchange.py:265-1036, tests/test_otr_dialogue_policy.py (156 lines),
tests/test_phase1_composer_prompt.py:1-179, tests/test_compose_exchange.py:1-545,
tests/test_exchange_seam_lane2.py:1-60, nodes/OTR_LedgerScriptWriter.py:4925-4948,
scripts/otr_api.py grep for use_exchange / source_bank / lemmy_cameo.


VERDICT: yes-with-fixes

The plan is internally consistent, correctly grounded, and mechanically complete. Two SHOULD-FIX items
survived R1-R3. Neither blocks build, but each is a concrete source of implementer error on the first
attempt. No MUST-FIX items found.


MUST-FIX BEFORE BUILD:

None -- plan converged.


SHOULD-FIX:

[1] P2.1 item 6 -- "\n\n" prefix is described but not shown in the new constant text block.

Defect: Item 6 says "Preserve the leading `"\n\n"` separator" as text, then immediately shows
the replacement content as a plain prose block with no leading newlines. An implementer reading
quickly would set:

    _COCKNEY_ORTHOGRAPHY_RULE = (
        "For LEMMY's spoken lines only, convey his Cockney accent..."
    )

and drop the separator. The current constant at nodes/_otr_dialogue_policy.py:6-10 is:

    _COCKNEY_ORTHOGRAPHY_RULE = (
        "\n\nConvey the Cockney accent through phrasing..."
    )

The P0.7 invariant -- "No-Lemmy prompt assembly remains byte-identical" -- is not at risk (a
non-Lemmy call returns system_prompt or "" unchanged). But a Lemmy call with the missing prefix
would fuse the policy onto the last line of the system prompt, which is a real prompt defect
(grounding clause ends without newline; see _otr_compose_exchange.py:390
`system = base + f"  - {grounding_clause}\n"` -- that trailing \n is the ONLY separator on the
exchange path when the policy prefix is absent).

Fix: In P2.1 item 6, show the full string literal INCLUDING the prefix so there is no reading
ambiguity:

    _COCKNEY_ORTHOGRAPHY_RULE = (
        "\n\nFor LEMMY's spoken lines only, convey his Cockney accent through phrasing, ..."
    )

The "line wrapping may follow formatter output" note already present is sufficient for the rest.

[2] P3.3 item 4 -- "_fake_gen_valid" signature description is slightly inaccurate.

Defect: The plan says "Preserve its real callable signature `(messages, *, temperature,
max_new_tokens)`" (no defaults). The real signature at
tests/test_compose_exchange.py:467 is `(messages, *, temperature=0.0, max_new_tokens=0)`.
The recorder wrapper should replicate the ACTUAL signature including defaults; otherwise the
existing tests that pass _fake_gen_valid directly (not through the wrapper) still work but the
reader is given wrong information about what to copy.

Fix: state the correct signature with defaults:
`(messages, *, temperature=0.0, max_new_tokens=0)`.
This is minor but the plan calls for exact signature fidelity in the same sentence, so match it.


OPTIONAL / NICE-TO-HAVE:

- P2.1 item 3: A one-sentence inline note explaining WHY `str` needs an explicit isinstance check
  BEFORE the Sequence check ("because str IS a Sequence in Python") would help implementers who
  wonder why the two-branch test is needed. Not a correctness issue; the code is correct as
  written.

- P5.3 item 6 paragraph on the audit script: the two-stage grouping logic (manual _groupable
  filtering first, then group_voiced_beats on the filtered run) mirrors run_exchange_prepass
  exactly but is spread across several paragraphs. A one-line note pointing to
  _otr_compose_exchange.py:984-1001 (_compose_run) as the reference implementation would anchor
  an implementer who writes the audit wrong the first time.


CUT THESE:

[1] P3.3 item 3 -- "compare only the system strings because repair reasons intentionally change
the user message." This is a correct observation but the plan has already restricted the
assertion scope to `gen.calls[call_index]["messages"][0]["content"]` (the system message),
so the parenthetical is a restatement. Safe to cut; the access path is self-documenting.

[2] P3.3 item 5 -- the note "do not treat it as the Cockney-absence lock: today it checks only
the static prefix and grounding clause." This is accurate but the statement adds no build
instruction; the implementer is told to run the test unchanged and is not asked to add
anything. One sentence warning the reader off a wrong inference is appropriate, but the
"today it checks..." technical detail is redundant given items 1-4 already establish the
explicit lock. Safe to cut back to just "Keep green as a routing-seam regression."


VERIFY-AT-BUILD checklist:

VERIFY-1 [P3.1, P4.1]: After the fix, run
    rg -n "roster_has_lemmy" nodes tests
must return no matches. Any surviving reference means an import or call was not migrated.
Flagged UNVERIFIABLE in earlier rounds (could not confirm all callers without grounding); now
confirmed two production callers (nodes/_otr_line_composer.py:1049-1050 and
nodes/_otr_compose_exchange.py:392-393) and one test import
(tests/test_otr_dialogue_policy.py:5). The grep is the proof -- a zero-match return is the
only acceptable result.

VERIFY-2 [P2.1, P3.1]: The \n\n separator is present in the new constant. After the file edit:
    python -c "from nodes._otr_dialogue_policy import _COCKNEY_ORTHOGRAPHY_RULE; assert _COCKNEY_ORTHOGRAPHY_RULE.startswith('\n\n'), 'prefix missing'"
must exit 0. This is the machine-checkable form of SHOULD-FIX [1] above.

VERIFY-3 [P4.2]: Confirm active-speaker values at both call sites are real string speaker names
and not cast rows or shim objects. Spot-check by adding a temporary assertion inside the
predicate in a local test run (not committed):
    assert all(isinstance(s, str) for s in speakers)
This assertion is already part of the TypeError validation; it doubles as the confirm step here.

VERIFY-4 [P4.5]: Canonical workflow widget slot 13 is still `true` after the diff:
    python -c "import json; wf=json.load(open('workflows/otr_canonical.json')); [n for n in wf['nodes'] if n['id']==1][0]['widgets_values'][13]"
must return True. Confirmed: tests/test_workflow_json_guardrails.py:659-662 already pins this;
the workflow validation gate in P5.2 item 4 covers it in the full run.

VERIFY-5 [P4.8]: `git diff -- workflows/otr_canonical.json` must be empty. The plan explicitly
forbids any workflow change for this patch; the diff is the proof, not any inferred code path.

VERIFY-6 [P5.3 item 6]: At least one retained group in the live ledger includes both LEMMY and
another speaker. This was flagged UNVERIFIABLE at build time in earlier rounds because it depends
on the live episode cast. It remains a runtime-only check; the plan correctly gates the
qualification leg on this condition and specifies rerun if not met. Implementer must record
the accepted group's beat IDs in the PBUG receipt as specified.

VERIFY-7 [P3.2 item 1]: The P3.2 item 1 test (ALICE active with LEMMY in allowed_people, no
Cockney rule in system) MUST FAIL on the unmodified code and PASS after the fix. Run the focused
test against both old and new code to confirm it is a real mutation test. If it passes on old
code, the test is wrong. This was not explicitly flagged as a verify step in the plan.


Notes from grounding (claims confirmed against real files):

- nodes/_otr_dialogue_policy.py:13-26 confirms roster_has_lemmy accepts str and dict items and
  returns True for full-cast rosters. The defect is real.
- nodes/_otr_line_composer.py:1049-1051 confirms the roster = list(req.allowed_people or ()) +
  [req.speaker] construction. The full allowed_people set feeds the policy decision today.
- nodes/_otr_compose_exchange.py:391-393 confirms roster_items = list(cast or []) + [slot.speaker
  for slot in beat_group]. The full cast, not the active speakers, drives the current decision.
- nodes/_otr_compose_exchange.py:446-452 confirms failure_reasons go into the USER message, not
  the system message. The P3.3 item 3 assertion that first and repair calls carry identical system
  content is correct.
- nodes/_otr_compose_exchange.py:694-706 confirms the repair path calls _run_once with the same
  beat_group, validating P4.4.
- tests/test_compose_exchange.py:107-123 confirms _CountingGen stores dicts with "messages" key;
  access pattern gen.calls[call_index]["messages"][0]["content"] is correct.
- tests/test_phase1_composer_prompt.py:136-148 confirms _recording_creative stores
  [dict(m) for m in messages] per call; access path creative.state["calls"][call_index][0] is
  the system message dict. ✓
- tests/test_compose_exchange.py:467-475 confirms _fake_gen_valid is a plain function with no
  state attribute; the plan's assertion that it stores no calls is accurate.
- scripts/otr_api.py:844,858 confirms source_bank and lemmy_cameo are whitelisted; use_exchange
  is not (zero matches). P4.5 and P5.3 item 2 are grounded.
- nodes/OTR_LedgerScriptWriter.py:4932-4934 confirms exchange_prepass_audit.beat_ids is
  sorted(_ex_lines_by_beat_id.keys()). P5.3 item 6 filter logic is correct.
- nodes/_otr_compose_exchange.py:996-1000 confirms run_exchange_prepass calls group_voiced_beats
  on the already-_groupable-filtered run, matching the two-stage audit structure in P5.3 item 6.
