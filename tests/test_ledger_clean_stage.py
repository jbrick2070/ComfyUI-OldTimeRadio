"""THE CLEAN STAGE -- a model reads every line, and a model repairs it.

The contract under test, in the operator's own terms (2026-08-14):

  * the JUDGE is a MODEL. *"Your shim can clean a contained story but it
    won't fix the next one ... I'd rather a more intelligent LLM say 'do you
    see things acting, like a door closing? that's not dialogue'."* The
    pattern list is evidence handed to that model, never the authority;
  * only a MODEL rewrites prose, never Python. No shims, no assertions;
  * the repair THINKS -- best edit, not a strip -- and the judge re-reads it;
  * bounded retries, each TOLD what survived, then flag loudly and CONTINUE.
    Never a silent pass, never a hard stop;
  * on the fidelity lanes the author's own language cannot TRIGGER a pattern
    repair -- but the judge still reads every line.
"""
from __future__ import annotations

import json
from concurrent.futures import CancelledError
import pytest

from nodes import _otr_ledger_clean as lcl
from nodes import _otr_spoken_text_policy as policy
from nodes._otr_generation_budget import GenerationContextOverflowError


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _ledger(*texts: str, bank: str = "original") -> dict:
    """One announcer row plus one character row per supplied text."""
    lines = [{
        "line_id": "L000",
        "beat_id": "b000",
        "char_id": "announcer",
        "speaker": "ANNOUNCER",
        "speaker_role": "announcer",
        "text": "Tonight, from the lighthouse.",
    }]
    beats = [{"beat_id": "b000", "speaker": "ANNOUNCER"}]
    for i, text in enumerate(texts, start=1):
        lines.append({
            "line_id": f"L{i:03d}",
            "beat_id": f"b{i:03d}",
            "char_id": "c01",
            "speaker": "Nan Reyes",
            "speaker_role": "character",
            "text": text,
        })
        beats.append({
            "beat_id": f"b{i:03d}",
            "speaker": "Nan Reyes",
            "beat_intent": "admit the lamp has been dead for days",
        })
    return {
        "cast": [
            {"char_id": "c01", "name": "Nan Reyes"},
            {"char_id": "announcer", "name": "ANNOUNCER"},
        ],
        "beats": beats,
        "lines": lines,
        "meta": {"source_bank": bank},
    }


CLEAN_JUDGEMENT = {"segments_read": 2, "not_speech": []}


def _dirty_judgement(*quotes: str) -> dict:
    """A judgement naming one entry per bad SEGMENT, which is the point."""
    return {
        "segments_read": len(quotes) + 1,
        "not_speech": [
            {"quote": q, "why": "action"} for q in quotes
        ],
    }


class _Slot:
    """Stands in for the creative LLM slot, replying with scripted JSON.

    Replies are keyed on the LINE they are about, never on a call counter.
    The pass reads every voiced row and interleaves judge and repair calls,
    so a positional script silently answers about the wrong row -- which it
    did on the first cut of these tests, handing the announcer's row a
    verdict written for a character's.

    ``judgements`` and ``repairs`` map a fragment of the line under
    discussion to the replies for it, consumed in order. Any line with no
    entry is judged pure speech, which is what an unremarkable row is.
    """

    def __init__(self, judgements=None, repairs=None, brief="a fight over a lamp", authorizations=None):
        self.judgements = dict(judgements or {})
        self.repairs = dict(repairs or {})
        self.authorizations = dict(authorizations or {})
        self.authorization_calls = 0
        self.authorization_prompts = []
        self.brief = brief
        self.judge_calls = 0
        self.repair_calls = 0
        self.summary_calls = 0
        self.judge_prompts: "list[str]" = []
        self.repair_prompts: "list[str]" = []
        self.summary_prompts: "list[str]" = []
        self._used: "dict[tuple[str, str], int]" = {}

    @property
    def calls(self) -> int:
        return self.judge_calls + self.repair_calls

    def _reply(self, kind: str, table: dict, prompt: str, default):
        # The line under discussion is the one on the THE LINE: row, so a
        # fragment quoted back inside the story-so-far window cannot steal
        # the reply meant for the line being judged.
        subject = ""
        for line in prompt.splitlines():
            if line.startswith("THE LINE: "):
                subject = line[len("THE LINE: "):]
                break
        for fragment, replies in table.items():
            if fragment in subject:
                seen = self._used.get((kind, fragment), 0)
                self._used[(kind, fragment)] = seen + 1
                pool = list(replies)
                return pool[min(seen, len(pool) - 1)]
        return default

    def __call__(self, messages, **kwargs):
        text = (
            "\n".join(m.get("content", "") for m in messages)
            if isinstance(messages, list) else str(messages)
        )
        # THREE jobs now: the act brief runs once per arc phase before any
        # judging, so a fixture that does not know about it hands summary
        # calls the replies meant for the judge and every count drifts.
        if "In about TEN WORDS" in text:
            self.summary_prompts.append(text)
            self.summary_calls += 1
            return json.dumps({"going_on": self.brief})
        if "DO THIS, IN ORDER:" in text:
            self.judge_prompts.append(text)
            self.judge_calls += 1
            return json.dumps(
                self._reply("judge", self.judgements, text, CLEAN_JUDGEMENT))
        if "AUTHORIZE THE ORIGINAL COMPLAINT" in text:
            self.authorization_calls += 1
            self.authorization_prompts.append(text)
            return json.dumps(self._reply("authorization", self.authorizations, text,
                                          {"verdict": "unresolved", "spans": [], "reason": "No fixture authorization"}))
        self.repair_prompts.append(text)
        self.repair_calls += 1
        return json.dumps(
            self._reply("repair", self.repairs, text, {"text": "unset"}))


#: Stage business the pattern list happens to know about.
DIRTY = "(She turns from the window.) The lamp has not turned since Tuesday."
#: What a thinking repair gives back -- the action implied in the speech.
FIXED_REPLACEMENT = "I can't even look at the window."
FIXED = FIXED_REPLACEMENT + " The lamp has not turned since Tuesday."

#: THE CASE THE PATTERN LIST CANNOT SEE. "closes" is not in any verb list,
#: and never will be, because the next story invents a new one.
INVISIBLE = "The door closes behind him. I told you he would not stay."
INVISIBLE_REPLACEMENT = "He's gone."
INVISIBLE_FIXED = INVISIBLE_REPLACEMENT + " I told you he would not stay."


def _replacement_reply(*values):
    return {"replacements": [{"span_id": f"span_{i:03d}", "replacement": value}
                              for i, value in enumerate(values, 1)]}


def test_whole_row_false_accusation_preserves_valid_coda_and_outer_whitespace():
    original = "  Until next time.\t"
    ledger = _ledger(original, bank="my_story")
    slot = _Slot(judgements={original: [_dirty_judgement("Until next time.")]},
                 authorizations={original: [{"verdict": "already_spoken", "reason": "This is spoken dialogue"}]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="my_story")
    assert ledger["lines"][1]["text"] == original
    assert slot.authorization_calls == 1 and slot.repair_calls == 0
    assert receipt["rows"][0]["outcome"] == "already_spoken"
    assert receipt["rows"][0]["scope"]["authorization"]["calls"][0]["error"] is None


def test_genuine_whole_row_direction_remains_convertible_after_authorization():
    original = "The door closes behind him."
    ledger = _ledger(original)
    slot = _Slot(judgements={original: [_dirty_judgement(original)]},
                 authorizations={original: [{"verdict": "whole_row_direction"}]},
                 repairs={original: [{"text": "He's gone."}]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == "He's gone."
    assert receipt["rows"][0]["scope"]["mode"] == "whole"
    assert slot.authorization_calls == 1 and slot.repair_calls == 1


def test_scope_authorization_binds_once_and_reuses_bound_slot_for_repair():
    original = "Until next time."
    ledger = _ledger(original, bank="my_story")
    slot = _Slot(judgements={original: [_dirty_judgement(original)]})
    bindings, calls = [], []

    def bind(schema):
        bindings.append(schema)

        def bound(messages, **kwargs):
            calls.append(messages)
            return json.dumps({"verdict": "already_spoken",
                               "spans": None if len(calls) == 1 else []})
        return bound

    slot._otr_bind_schema = bind
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="my_story")
    assert bindings == [lcl._ScopeAuthorization]
    assert len(calls) == 2 and slot.authorization_calls == slot.repair_calls == 0
    assert ledger["lines"][1]["text"] == original
    assert "unclean_spoken_text" not in ledger["lines"][1].get("compose_flags", [])
    assert receipt["rows"][0]["outcome"] == "already_spoken"


def test_scope_binder_failure_propagates_without_unconstrained_fallback():
    original = "Until next time."
    slot = _Slot(judgements={original: [_dirty_judgement(original)]})

    def bind(schema):
        raise RuntimeError("schema owner unavailable")

    slot._otr_bind_schema = bind
    with pytest.raises(RuntimeError, match="schema owner unavailable"):
        lcl.run_ledger_clean(_ledger(original), slot_fn=slot, bank_id="original")
    assert slot.authorization_calls == slot.repair_calls == 0


def test_whole_row_authorization_can_narrow_to_a_real_local_direction():
    original = "Stay here. (He sighs)"
    ledger = _ledger(original)
    slot = _Slot(judgements={original: [_dirty_judgement(original)]},
                 authorizations={original: [{"verdict": "localized_defect", "spans": [
                     {"quote": "(He sighs)", "start_char": 11, "end_char": len(original)}]}]},
                 repairs={original: [_replacement_reply("")]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == "Stay here. "
    assert receipt["rows"][0]["scope"]["mode"] == "partial"


def test_disjoint_repairs_preserve_all_unedited_spaces_tabs_newlines_and_punctuation():
    original = "  (He sighs)\tI won't leave;  promise.\n(He nods)  "
    expected = "  All right.\tI won't leave;  promise.\nAgreed.  "
    ledger = _ledger(original)
    slot = _Slot(repairs={"(He sighs)": [_replacement_reply("All right.", "Agreed.")]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == expected
    assert len(receipt["rows"][0]["scope"]["approved_spans"]) == 2


def test_exact_pattern_spans_include_all_occurrences_without_summary_normalization():
    note = "(He\twhispers " + "very " * 15 + "quietly)"
    original = note + "\tStay.\n" + note
    findings = [item for item in policy.f1_finding_spans(original)
                if item["kind"] == "stage direction in brackets"]
    assert len(findings) == 2
    assert all(item["quote"] == note for item in findings)
    assert all(original[item["start_char"]:item["end_char"]] == note for item in findings)
    assert len(policy.f1_findings(original)[0].detail) <= 60 < len(note)
    spans = lcl._merge_repair_spans(original, [
        (item["start_char"], item["end_char"]) for item in policy.f1_finding_spans(original)])
    assert [span.quote for span in spans] == [note, note]


@pytest.mark.parametrize("with_offsets", [False, True])
def test_repeated_judge_quote_needs_the_intended_occurrence(with_offsets):
    quote = "The door closes."
    original = quote + " I wait. " + quote
    finding = {"quote": quote, "why": "direction"}
    if with_offsets:
        finding.update(start_char=original.rindex(quote), end_char=len(original))
    ledger = _ledger(original)
    slot = _Slot(judgements={original: [{"not_speech": [finding]}]},
                 repairs={original: [_replacement_reply("It's shut.")]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    if with_offsets:
        assert ledger["lines"][1]["text"] == quote + " I wait. It's shut."
        assert slot.authorization_calls == 0
    else:
        assert ledger["lines"][1]["text"] == original
        assert slot.repair_calls == 0
        assert receipt["rows"][0]["outcome"] == "unresolved"


def test_bounded_localizer_resolves_a_repeated_quote_without_authorizing_both():
    quote = "The door closes."
    original = quote + " I wait. " + quote
    ledger = _ledger(original)
    slot = _Slot(judgements={original: [_dirty_judgement(quote)]},
                 authorizations={original: [{"verdict": "localized_defect", "spans": [
                     {"quote": quote, "start_char": original.rindex(quote), "end_char": len(original)}]}]},
                 repairs={original: [_replacement_reply("It's shut.")]})
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == quote + " I wait. It's shut."
    assert slot.authorization_calls == 1 and slot.repair_calls == 1


@pytest.mark.parametrize("replacements", [
    [],
    [{"span_id": "another_span", "replacement": "Hello."}],
    [{"span_id": "span_001", "replacement": "Hello."}, {"span_id": "span_001", "replacement": "Again."}],
])
def test_invalid_replacement_id_sets_never_mutate_the_row(replacements):
    ledger = _ledger(DIRTY)
    slot = _Slot(repairs={DIRTY: [{"replacements": replacements}]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == DIRTY
    assert receipt["repaired"] == receipt["improved"] == 0
    assert receipt["unclean"] == 1
    assert receipt["rows"][0]["attempts"][0]["rejection_reason"]


def test_invalid_candidate_cannot_reach_clean_or_best_progress_acceptance(monkeypatch):
    def invalid_candidate(**kwargs):
        kwargs["proposal_receipt"]["replacements"] = _replacement_reply("")["replacements"]
        return "A replacement for every word in the line."
    monkeypatch.setattr(lcl, "_call_repair", invalid_candidate)
    ledger = _ledger(DIRTY)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=_Slot(), bank_id="original")
    assert ledger["lines"][1]["text"] == DIRTY
    assert receipt["repaired"] == receipt["improved"] == 0
    assert [item["outcome"] for item in receipt["rows"][0]["attempts"]] == ["scope_rejected"] * 2


def test_retry_feedback_cannot_widen_original_scope_and_reread_sees_candidate():
    protected = "The lamp has not turned since Tuesday."
    ledger = _ledger(DIRTY)
    slot = _Slot(judgements={FIXED: [_dirty_judgement(protected)]},
                 repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == FIXED
    scope = receipt["rows"][0]["scope"]["approved_spans"]
    assert len(scope) == 1 and scope[0]["quote"] == "(She turns from the window.)"
    assert slot.repair_calls == 2 and slot.authorization_calls == 0
    rereads = [prompt for prompt in slot.judge_prompts if "THE LINE: " + FIXED in prompt]
    assert rereads and all(">>> Nan Reyes: " + FIXED in prompt for prompt in rereads)
    assert all(">>> Nan Reyes: " + DIRTY in prompt for prompt in slot.repair_prompts)


def test_conserved_row_over_two_thousand_characters_is_not_a_schema_rejection():
    suffix = "\t" + "The lantern remains lit. " * 160 + "  "
    original = "(He sighs)" + suffix
    ledger = _ledger(original)
    slot = _Slot(repairs={"(He sighs)": [_replacement_reply("All right.")]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert len(original) > 2000
    assert ledger["lines"][1]["text"] == "All right." + suffix
    assert receipt["repaired"] == 1


def test_empty_local_deletion_cannot_make_the_final_spoken_row_empty():
    spans = lcl._merge_repair_spans("(He sighs) ", [(0, 10)])
    with pytest.raises(ValueError, match="final spoken row"):
        lcl._splice_replacements("(He sighs) ", spans, _replacement_reply("")["replacements"])


@pytest.mark.parametrize("job", ["judge", "authorization", "repair", "reread"])
@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError, CancelledError, GenerationContextOverflowError])
def test_cleaner_jobs_do_not_swallow_real_runtime_failures(job, error_type):
    original = "The door closes behind him."
    class _FailingJob(_Slot):
        def __call__(self, messages, **kwargs):
            prompt = "\n".join(message["content"] for message in messages)
            marker = {"judge": "DO THIS, IN ORDER:",
                      "authorization": "AUTHORIZE THE ORIGINAL COMPLAINT",
                      "repair": "HOW TO EDIT", "reread": "DO THIS, IN ORDER:"}[job]
            if marker in prompt and (job != "reread" or "THE LINE: He's gone." in prompt):
                raise error_type("runtime failed")
            return super().__call__(messages, **kwargs)
    slot = _FailingJob(judgements={original: [_dirty_judgement(original)]},
                       authorizations={original: [{"verdict": "whole_row_direction"}]},
                       repairs={original: [{"text": "He's gone."}]})
    with pytest.raises(error_type, match="runtime failed"):
        lcl.run_ledger_clean(_ledger(original), slot_fn=slot, bank_id="original")


def test_malformed_reread_cannot_be_clean_or_best_progress():
    class _MalformedReread(_Slot):
        def __call__(self, messages, **kwargs):
            prompt = "\n".join(message["content"] for message in messages)
            if "DO THIS, IN ORDER:" in prompt and "THE LINE: " + INVISIBLE_FIXED in prompt:
                return '{"not_speech": false}'
            return super().__call__(messages, **kwargs)
    slot = _MalformedReread(judgements={INVISIBLE: [_dirty_judgement("The door closes behind him.")]},
                            repairs={INVISIBLE: [_replacement_reply(INVISIBLE_REPLACEMENT)]})
    ledger = _ledger(INVISIBLE)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == INVISIBLE
    assert receipt["repaired"] == receipt["improved"] == 0
    assert receipt["unclean"] == 1 and slot.repair_calls == 2
    assert lcl.UNCLEAN_COMPOSE_FLAG in ledger["lines"][1]["compose_flags"]
    assert [item["outcome"] for item in receipt["rows"][0]["attempts"]] == ["reread_unresolved"] * 2


def test_pattern_repair_gets_a_real_reread_after_initial_judge_exhaustion():
    class _MalformedInitial(_Slot):
        def __call__(self, messages, **kwargs):
            prompt = "\n".join(message["content"] for message in messages)
            if "DO THIS, IN ORDER:" in prompt and "THE LINE: " + DIRTY in prompt:
                return '{"not_speech": false}'
            return super().__call__(messages, **kwargs)
    slot = _MalformedInitial(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    ledger = _ledger(DIRTY)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == FIXED
    assert receipt["rows"][0]["scope"]["initial_judge_valid"] is False
    assert any("THE LINE: " + FIXED in prompt for prompt in slot.judge_prompts)
    assert receipt["repaired"] == 1


@pytest.mark.parametrize("note", ["(He\tsighs)", "(He whispers " + "very " * 20 + "quietly)"])
def test_pattern_display_summary_cannot_override_exact_pattern_scope(note):
    original = note + " Stay. " + note
    slot = _Slot(repairs={note: [_replacement_reply("", "")]})
    ledger = _ledger(original)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == " Stay. "
    assert receipt["repaired"] == 1 and slot.authorization_calls == 0
    assert [span["quote"] for span in receipt["rows"][0]["scope"]["approved_spans"]] == [note, note]


def test_pattern_and_model_findings_both_keep_their_exact_edit_scope():
    original = "(He sighs) Stay here. The door closes."
    slot = _Slot(judgements={original: [_dirty_judgement("The door closes.")]},
                 repairs={original: [_replacement_reply("", "It's shut.")]})
    ledger = _ledger(original)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == " Stay here. It's shut."
    assert receipt["found_by_both"] == 1 and slot.authorization_calls == 0


@pytest.mark.parametrize("quotes", [("Until next", "next time."), ("Until", "next", "time.")])
def test_collective_whole_row_complaints_cannot_bypass_authorization(quotes):
    original = "Until next time."
    slot = _Slot(judgements={original: [_dirty_judgement(*quotes)]},
                 authorizations={original: [{"verdict": "already_spoken"}]},
                 repairs={original: [_replacement_reply("That's a wrap.")]})
    ledger = _ledger(original)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == original
    assert slot.authorization_calls == 1 and slot.repair_calls == 0
    assert receipt["rows"][0]["outcome"] == "already_spoken"


def test_normalized_whole_row_complaint_can_authorize_a_real_direction():
    original = "He\t walks away."
    slot = _Slot(judgements={original: [_dirty_judgement("He walks away.")]},
                 authorizations={original: [{"verdict": "whole_row_direction"}]},
                 repairs={original: [{"text": "He's gone."}]})
    ledger = _ledger(original)
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == "He's gone."
    assert slot.authorization_calls == slot.repair_calls == 1


@pytest.mark.parametrize("authorization", [
    {"verdict": "rewrite_it"},
    {"verdict": "localized_defect", "spans": []},
    {"verdict": "localized_defect", "spans": [{"quote": "Until", "start_char": 1, "end_char": 6}]},
    {"verdict": "localized_defect", "spans": [{"quote": "Until", "start_char": False, "end_char": 5}]},
    {"verdict": "localized_defect", "spans": [{"quote": "Until next"}, {"quote": "next time."}]},
    {"verdict": "already_spoken", "spans": [{"quote": "Until"}]},
])
def test_malformed_or_collectively_full_localization_keeps_the_original(authorization):
    original = "Until next time."
    slot = _Slot(judgements={original: [_dirty_judgement(original)]},
                 authorizations={original: [authorization]})
    ledger = _ledger(original)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == original
    assert slot.authorization_calls == 2 and slot.repair_calls == 0
    record = receipt["rows"][0]
    assert record["outcome"] == "unresolved" and receipt["unclean"] == 1
    assert all(call["error"] for call in record["scope"]["authorization"]["calls"])


def test_partial_ambiguous_complaint_cannot_authorize_whole_row_conversion():
    original = "The door closes. Stay. The door closes."
    slot = _Slot(judgements={original: [_dirty_judgement("The door closes.")]},
                 authorizations={original: [{"verdict": "whole_row_direction"}]})
    ledger = _ledger(original)
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == original
    assert slot.authorization_calls == 2 and slot.repair_calls == 0


def test_whole_row_localization_discards_earlier_collective_scope():
    original = "The door closes. Stay."
    slot = _Slot(judgements={original: [_dirty_judgement("The door closes.", "Stay.")]},
                 authorizations={original: [{"verdict": "localized_defect", "spans": [{"quote": "The door closes."}]}]},
                 repairs={original: [_replacement_reply("It's shut.")]})
    ledger = _ledger(original)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == "It's shut. Stay."
    assert [span["quote"] for span in receipt["rows"][0]["scope"]["approved_spans"]] == ["The door closes."]


def test_adjacent_scopes_are_distinct_replacements():
    original = "(He sighs)(He nods) Stay."
    slot = _Slot(repairs={original: [_replacement_reply("All right.", " Agreed.")]})
    ledger = _ledger(original)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == "All right. Agreed. Stay."
    assert len(receipt["rows"][0]["scope"]["approved_spans"]) == 2


def test_sentence_judging_preserves_second_occurrence_offsets(monkeypatch):
    quote = "The door closes."
    original = quote + " I wait. " + quote
    monkeypatch.setattr(lcl, "JUDGE_PER_SENTENCE", True)
    slot = _Slot(judgements={quote + " I wait. It's shut.": [CLEAN_JUDGEMENT],
                            quote: [CLEAN_JUDGEMENT, _dirty_judgement(quote)]},
                 repairs={original: [_replacement_reply("It's shut.")]})
    ledger = _ledger(original)
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == quote + " I wait. It's shut."
    assert slot.authorization_calls == 0


def test_votes_on_different_occurrences_do_not_authorize_either(monkeypatch):
    quote = "The door closes."
    original = quote + " I wait. " + quote
    monkeypatch.setattr(lcl, "JUDGE_VOTES", 2)
    slot = _Slot(judgements={original: [
        {"not_speech": [{"quote": quote, "start_char": 0, "end_char": len(quote)}]},
        {"not_speech": [{"quote": quote, "start_char": original.rindex(quote), "end_char": len(original)}]},
    ]})
    ledger = _ledger(original)
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["lines"][1]["text"] == original
    assert slot.repair_calls == slot.authorization_calls == 0


# ---------------------------------------------------------------------------
# the judge is a model, and that is the whole point
# ---------------------------------------------------------------------------


def test_the_judge_catches_what_no_pattern_list_can():
    """"The door closes behind him" -- the operator's own example.

    No pattern in the policy fires on it. If the patterns were the authority
    this line would be read aloud on air, closing door and all.
    """
    assert not policy.f1_findings(INVISIBLE), (
        "the fixture is meant to be invisible to the pattern list")

    ledger = _ledger(INVISIBLE)
    slot = _Slot(
        judgements={INVISIBLE: [_dirty_judgement("The door closes behind him.")]},
        repairs={INVISIBLE: [_replacement_reply(INVISIBLE_REPLACEMENT)]},
    )
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert ledger["lines"][1]["text"] == INVISIBLE_FIXED
    assert receipt["repaired"] == 1
    assert receipt["judge_only"] == 1, "no pattern found it; the judge did"
    assert receipt["rows"][0]["found_by"] == "judge"


def test_every_voiced_row_is_read_by_the_judge():
    """Detector-gating would mean the judge never sees the door close."""
    ledger = _ledger("I have not slept.", "The lamp is dead.")
    slot = _Slot()
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert slot.judge_calls == 3      # announcer + two character rows
    assert receipt["voiced_rows"] == 3
    assert receipt["repaired"] == 0
    assert receipt["unclean"] == 0


def test_the_patterns_are_offered_as_evidence_and_labelled_unreliable():
    ledger = _ledger(DIRTY)
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    # Select by THE LINE:, not by substring -- now that the window carries
    # the lines AFTER a row, the announcer's prompt also contains this text.
    judged = [p for p in slot.judge_prompts
              if ("THE LINE: " + DIRTY) in p][0]
    assert "crude pattern-matcher" in judged
    assert "It is often WRONG" in judged
    assert "stage direction in brackets" in judged


def test_the_patterns_still_fire_when_the_judge_misses_it():
    """A union, never a veto. The 2B model that shrugged does not get to."""
    ledger = _ledger(DIRTY)
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert receipt["pattern_only"] == 1
    assert receipt["repaired"] == 1
    assert ledger["lines"][1]["text"] == FIXED
    assert receipt["rows"][0]["found_by"] == "patterns"


def test_a_quote_that_is_not_in_this_line_is_dropped_not_refused():
    """A quote about a NEIGHBOUR must not send the repair after it.

    It is DROPPED rather than refused, and that distinction was a measured
    defect: refusing cost the structured-call ladder, and when the ladder
    exhausted the row lost its judge entirely and fell back to the pattern
    floor. On a 2B shown the surrounding lines, quoting a neighbour is
    common enough that the strict version turned the judge off on most rows.
    """
    ledger = _ledger(INVISIBLE)
    slot = _Slot(
        judgements={INVISIBLE: [{
            "segments_read": 2,
            "not_speech": [
                # about a different row entirely -- must be dropped
                {"quote": "Tonight, from the lighthouse.", "why": "action"},
                # about THIS line -- must survive in the same answer
                {"quote": "The door closes behind him.", "why": "sound"},
            ],
        }]},
        repairs={INVISIBLE: [_replacement_reply(INVISIBLE_REPLACEMENT)]},
    )
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert receipt["repaired"] == 1
    quotes = [f["quote"] for f in receipt["rows"][0]["complaint"]]
    assert quotes == ["The door closes behind him."], (
        "the neighbour quote should be dropped and the real one kept")


def test_the_best_rewrite_ships_when_it_cannot_be_made_spotless():
    """Progress, not perfection -- discarding it was losing real repairs.

    Measured in the lab: an eager 2B judge finds SOMETHING on nearly every
    rewrite, so an accept-only-when-spotless loop burns the budget and then
    ships the ORIGINAL -- throwing away a rewrite that had genuinely removed
    the stage direction.
    """
    two_faults = "(He sighs) The door closes behind him. I told you."
    half_fixed = " The door closes behind him. I told you."
    ledger = _ledger(two_faults)
    slot = _Slot(
        judgements={
            two_faults: [_dirty_judgement("(He sighs)",
                                          "The door closes behind him.")],
            half_fixed: [_dirty_judgement("The door closes behind him.")],
        },
        repairs={two_faults: [_replacement_reply("", "The door closes behind him.")]},
    )
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert receipt["improved"] == 1
    assert receipt["repaired"] == 0
    assert ledger["lines"][1]["text"] == half_fixed, "the better line ships"
    # Improved is not clean: the flag stays so nothing is quietly declared fixed.
    assert lcl.UNCLEAN_COMPOSE_FLAG in ledger["lines"][1]["compose_flags"]
    assert receipt["rows"][0]["outcome"] == "improved"


# ---------------------------------------------------------------------------
# EVERY segment, not just the first -- the operator's actual requirement
# ---------------------------------------------------------------------------


def test_every_non_speech_segment_in_one_line_is_named_and_repaired():
    """*"It reads the WHOLE line and looks for even SEGMENTS."*

    Measured live 2026-08-14: a single row carried a stage direction at BOTH
    ends with real dialogue between them. A judge that reports the first one
    leaves the second to be read aloud.
    """
    both_ends = (
        "(Montgomery sighs) I've already given you the reel. "
        "(Montgomery sighs again)"
    )
    fixed = "I'm tired of this. I've already given you the reel. Let it go."
    ledger = _ledger(both_ends)
    slot = _Slot(
        judgements={both_ends: [_dirty_judgement(
            "(Montgomery sighs)", "(Montgomery sighs again)")]},
        repairs={both_ends: [_replacement_reply("I'm tired of this.", "Let it go.")]},
    )
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert receipt["segments_named"] == 2
    # BOTH went into ONE repair call, as a numbered checklist. A per-segment
    # loop would edit the line twice against two partial views of it.
    assert slot.repair_calls == 1
    prompt = slot.repair_prompts[0]
    assert "1. '(Montgomery sighs)'" in prompt
    assert "2. '(Montgomery sighs again)'" in prompt
    assert "every one of these must be gone" in prompt
    assert ledger["lines"][1]["text"] == fixed


# ---------------------------------------------------------------------------
# the repair
# ---------------------------------------------------------------------------


def test_the_repair_is_told_the_judge_s_own_words():
    ledger = _ledger(INVISIBLE)
    slot = _Slot(
        judgements={INVISIBLE: [_dirty_judgement("The door closes behind him.")]},
        repairs={INVISIBLE: [_replacement_reply(INVISIBLE_REPLACEMENT)]},
    )
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert "The door closes behind him." in slot.repair_prompts[0]
    assert "Never just delete and staple" in slot.repair_prompts[0]
    # The judge's own standard, restated inside the repair, so the model
    # self-checks before the judge is spent on re-reading it.
    assert "read your rewrite once" in slot.repair_prompts[0]


def test_the_repair_is_shown_the_speaker_the_beat_intent_and_the_story_so_far():
    """The window IS the fix: an edit that cannot see the moment guesses."""
    ledger = _ledger("I have not slept.", DIRTY)
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    prompt = slot.repair_prompts[0]
    assert "Nan Reyes" in prompt
    assert "I have not slept." in prompt            # the story so far
    assert "admit the lamp has been dead" in prompt  # WHERE THE STORY IS
    assert "Tonight, from the lighthouse." in prompt  # the announcer open


def test_a_repaired_row_carries_metrics_for_the_line_that_actually_ships():
    """Stale counts would describe the line this pass just replaced."""
    ledger = _ledger(DIRTY)
    ledger["lines"][1]["word_count"] = 999
    ledger["lines"][1]["char_count"] = 999
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    from nodes._otr_text_metrics import (
        canonical_char_count,
        canonical_word_count,
    )
    row = ledger["lines"][1]
    assert row["text"] == FIXED
    assert row["word_count"] == canonical_word_count(FIXED)
    assert row["char_count"] == canonical_char_count(FIXED)


def test_the_judge_reads_the_repair_back_before_it_is_accepted():
    """A repair graded more weakly than the judgement can pass by moving the
    defect somewhere the weaker check cannot see."""
    ledger = _ledger(INVISIBLE)
    walked = "He walks out. I told you he would not stay."
    slot = _Slot(
        judgements={
            INVISIBLE: [_dirty_judgement("The door closes behind him.")],
            walked: [_dirty_judgement("He walks out.")],
        },
        repairs={INVISIBLE: [_replacement_reply("He walks out."), _replacement_reply(INVISIBLE_REPLACEMENT)]},
    )
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert slot.repair_calls == 2
    assert "He walks out." in slot.repair_prompts[1]
    assert "YOUR PREVIOUS ATTEMPT" in slot.repair_prompts[1]
    assert ledger["lines"][1]["text"] == INVISIBLE_FIXED
    assert receipt["repaired"] == 1


# ---------------------------------------------------------------------------
# bounded, and it never stops the render
# ---------------------------------------------------------------------------


def test_an_unfixable_row_ships_flagged_and_the_render_continues():
    ledger = _ledger(INVISIBLE)
    stubborn = INVISIBLE
    slot = _Slot(
        judgements={
            INVISIBLE: [_dirty_judgement("The door closes behind him.")],
            stubborn: [_dirty_judgement("The door closes behind him.")],
        },
        repairs={INVISIBLE: [_replacement_reply("The door closes behind him.")]},
    )
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert slot.repair_calls == 2, "bounded: it does not grind"
    # It SHIPS -- the original text, not a half-repair, and never an exception.
    assert ledger["lines"][1]["text"] == INVISIBLE
    assert lcl.UNCLEAN_COMPOSE_FLAG in ledger["lines"][1]["compose_flags"]
    assert receipt["unclean"] == 1
    assert receipt["rows"][0]["outcome"] == "unclean"


def test_a_real_provider_failure_retains_its_failure_type():
    class _Exploding:
        def __call__(self, messages, **kwargs):
            raise RuntimeError("the provider fell over")

    ledger = _ledger(DIRTY)
    with pytest.raises(RuntimeError, match="the provider fell over"):
        lcl.run_ledger_clean(ledger, slot_fn=_Exploding(), bank_id="original")
    assert ledger["lines"][1]["text"] == DIRTY


def test_python_never_edits_the_prose_itself():
    """No model, no rewrite -- the row ships flagged instead of stripped."""
    ledger = _ledger(DIRTY)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=None, bank_id="original")
    assert ledger["lines"][1]["text"] == DIRTY
    assert receipt["rows"][0]["outcome"] == "no_model"
    assert lcl.UNCLEAN_COMPOSE_FLAG in ledger["lines"][1]["compose_flags"]


def test_the_receipt_lands_on_meta_even_when_nothing_fired():
    ledger = _ledger("The lamp has not turned since Tuesday.")
    slot = _Slot()
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert ledger["meta"]["ledger_clean"]["policy"] == (
        policy.SPOKEN_TEXT_POLICY_ID)
    assert ledger["meta"]["ledger_clean"]["version"] == (
        lcl.LEDGER_CLEAN_VERSION)
    assert ledger["meta"]["ledger_clean"]["judge"] == "model"


# ---------------------------------------------------------------------------
# the fidelity lanes
# ---------------------------------------------------------------------------


def test_the_author_s_own_language_cannot_trigger_a_pattern_repair():
    """Fidelity outranks arc: a third person from the source is not a defect.

    The JUDGE still reads the line -- it can tell an author's sentence from a
    writer's leaked stage direction, which is exactly what a pattern cannot.
    """
    line = "She sighed, and the crown grew heavy on her brow."
    for bank in ("shakespeare", "public_domain"):
        ledger = _ledger(line, bank=bank)
        slot = _Slot()
        receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id=bank)
        assert slot.repair_calls == 0, bank
        assert ledger["lines"][1]["text"] == line, bank
        assert receipt["repaired"] == 0, bank
        assert slot.judge_calls == 2, bank   # it was still READ


def test_production_markup_is_still_a_defect_on_a_fidelity_lane():
    """No author ever wrote "MACBETH:" into a character's speech."""
    ledger = _ledger("MACBETH: Is this a dagger which I see before me?",
                     bank="shakespeare")
    fixed = " Is this a dagger which I see before me?"
    slot = _Slot(repairs={"MACBETH:": [_replacement_reply("")]})
    receipt = lcl.run_ledger_clean(
        ledger, slot_fn=slot, bank_id="shakespeare")
    assert slot.repair_calls == 1
    assert ledger["lines"][1]["text"] == fixed
    assert receipt["repaired"] == 1


def test_the_fidelity_carve_out_is_exactly_two_banks_and_only_markup():
    assert policy.FIDELITY_BANKS == {"shakespeare", "public_domain"}
    assert policy.repairable_kinds("shakespeare") == policy.MARKUP_KINDS
    assert policy.repairable_kinds("original") == {
        kind for kind, _pattern in policy.F1_PATTERNS}


# ---------------------------------------------------------------------------
# F2 is detected and reported, never repaired
# ---------------------------------------------------------------------------


def test_f2_is_reported_and_never_spends_a_repair_call():
    """A row that disagrees about who speaks is bookkeeping, not prose."""
    ledger = _ledger("The lamp has not turned since Tuesday.")
    ledger["lines"][1]["speaker"] = "Someone Else"
    slot = _Slot()
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert slot.repair_calls == 0
    assert receipt["f2_rows"] == 1
    assert receipt["f2"][0]["line_id"] == "L001"
    assert receipt["f2"][0]["findings"][0]["kind"] == (
        "row disagrees with its beat")


def test_a_row_naming_no_speaker_is_f2():
    ledger = _ledger("The lamp has not turned since Tuesday.")
    ledger["lines"][1]["speaker"] = ""
    ledger["beats"][1]["speaker"] = ""
    receipt = lcl.run_ledger_clean(ledger, slot_fn=None, bank_id="original")
    assert receipt["f2"][0]["findings"][0]["kind"] == "no speaker"


# ---------------------------------------------------------------------------
# skipped and non-voiced rows are not the clean stage's business
# ---------------------------------------------------------------------------


def test_music_and_skipped_rows_are_never_touched():
    ledger = _ledger(DIRTY)
    ledger["lines"].append({
        "line_id": "L900", "char_id": "music_open",
        "speaker_role": "music", "text": "",
    })
    ledger["lines"][1]["skip"] = True
    slot = _Slot()
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert slot.repair_calls == 0
    assert receipt["voiced_rows"] == 1     # the announcer row alone
    assert ledger["lines"][1]["text"] == DIRTY


# ---------------------------------------------------------------------------
# ONE definition of the pattern floor -- grader and pass may never diverge
# ---------------------------------------------------------------------------


def test_the_grader_imports_the_same_policy_the_clean_stage_uses():
    """If these ever diverge, a run repairs a line the grader still fails."""
    import importlib.util
    import sys
    from pathlib import Path

    view_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "otr_ledger_view.py"
    )
    spec = importlib.util.spec_from_file_location("_otr_view_probe", view_path)
    view = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the grader defines @dataclass types, and
    # dataclasses resolves annotations through sys.modules[cls.__module__].
    sys.modules["_otr_view_probe"] = view
    try:
        spec.loader.exec_module(view)
    finally:
        sys.modules.pop("_otr_view_probe", None)

    assert view.F1_PATTERNS is policy.F1_PATTERNS
    assert view.VOICED_ROLES is policy.VOICED_ROLES
    assert view.f1_findings is policy.f1_findings


# ---------------------------------------------------------------------------
# no shims -- the operator's standing rule, enforced against the source
# ---------------------------------------------------------------------------


def test_python_owns_no_opinion_about_prose():
    """Operator 2026-08-14: *"no shims or py asserts, I only want the LLM to
    clean things up."* One write of row text, and it writes the MODEL's line.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "nodes" / "_otr_ledger_clean.py"
    ).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # skip the module docstring
    # CODE ONLY. The comments explain why a bare `row["text"] = ` is wrong,
    # so a naive substring search finds the very thing it is checking for.
    code = "\n".join(
        line.split("#", 1)[0] for line in body.splitlines()
    )

    assert "assert " not in code
    assert "re.sub" not in code and ".sub(" not in code
    assert '["text"] =' not in code, "text is set through the canonical owner"
    # FOUR writes now -- the spotless F1 repair, the best-effort F1
    # improvement, the confirmed F2 reattribution, and the unconfirmed one --
    # and EVERY ONE writes a string the model returned. That is the whole
    # law: Python chooses which of the model's answers to keep, and never
    # writes a word of its own.
    assert code.count("set_line_text_metrics(row,") == 4


# ---------------------------------------------------------------------------
# BLINDNESS -- the pass must actually SEE the artifacts, on the REAL shape
# ---------------------------------------------------------------------------


def _production_shaped_ledger() -> dict:
    """A ledger shaped like the WRITER LANE actually writes one.

    Measured off a live `media_archive` episode 2026-08-14: `arc_phase` and
    `beat_intent` sit on the LINE row, and the beat rows carry transport
    only. The first cut of the pass read the beat, so every prompt shipped
    with an empty act and the model judged blind on all 16 rows -- with a
    green unit suite, because the old fixtures put the fields on the beat.
    This fixture is deliberately the production shape.
    """
    return {
        "cast": [
            {"char_id": "c01", "name": "Nan Reyes"},
            {"char_id": "announcer", "name": "ANNOUNCER"},
        ],
        "beats": [
            # Transport only -- exactly what the writer lane emits.
            {"beat_id": "b001", "char_id": "announcer",
             "line_ids": ["b001"], "speaker": "ANNOUNCER",
             "scene_id": None, "shot_id": None},
            {"beat_id": "b002", "char_id": "c01",
             "line_ids": ["b002"], "speaker": "Nan Reyes",
             "scene_id": None, "shot_id": None},
            {"beat_id": "b003", "char_id": "c01",
             "line_ids": ["b003"], "speaker": "Nan Reyes",
             "scene_id": None, "shot_id": None},
        ],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "char_id": "announcer",
             "speaker": "ANNOUNCER", "speaker_role": "announcer",
             "arc_phase": "setup", "beat_intent": "open the episode",
             "text": "Tonight, from the lighthouse."},
            {"line_id": "b002", "beat_id": "b002", "char_id": "c01",
             "speaker": "Nan Reyes", "speaker_role": "character",
             "arc_phase": "rising", "beat_intent": "admit the lamp is dead",
             "text": DIRTY},
            {"line_id": "b003", "beat_id": "b003", "char_id": "c01",
             "speaker": "Nan Reyes", "speaker_role": "character",
             "arc_phase": "turn", "beat_intent": "refuse to leave",
             "text": "I am not walking down those stairs."},
        ],
        "meta": {
            "source_bank": "media_archive",
            "arc_shape": "bittersweet_parting",
            "story_contract": {"label": "expedition camp radio log"},
        },
    }


def test_the_act_is_read_from_the_line_row_where_production_puts_it():
    ledger = _production_shaped_ledger()
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="media_archive")

    judged = [p for p in slot.judge_prompts
              if ("THE LINE: " + DIRTY) in p][0]
    assert "WHERE THE STORY IS:" in judged
    assert "rising" in judged, "arc_phase never reached the prompt"
    assert "admit the lamp is dead" in judged, "beat_intent never reached it"
    # Episode-level context the writer already stamps, for free.
    assert "expedition camp radio log" in judged
    assert "bittersweet_parting" in judged
    # And the repair sees it too.
    assert "WHERE THE STORY IS:" in slot.repair_prompts[0]


def test_the_receipt_proves_what_the_model_actually_saw():
    """Operator: *"not blind due to a coding error ... a telemetry test."*

    A green suite cannot see blindness -- only a count taken off the real
    artifact can. If any of these reads 0 on a live episode, the pass was
    working blind and the ledger now says so.
    """
    ledger = _production_shaped_ledger()
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    receipt = lcl.run_ledger_clean(
        ledger, slot_fn=slot, bank_id="media_archive")

    seen = receipt["context_seen"]
    assert seen["rows_with_arc_phase"] == 3
    assert seen["rows_with_beat_intent"] == 3
    assert seen["rows_with_cast_name"] == 3
    assert seen["rows_with_lines_before"] == 2   # every row but the first
    assert seen["rows_with_lines_after"] == 2    # every row but the last
    assert "expedition camp radio log" in seen["episode_context"]


def test_the_beat_row_still_works_for_the_lane_that_populates_it():
    """The codex lane DOES put these on the beat. Read the union, not one."""
    ledger = _production_shaped_ledger()
    for row in ledger["lines"]:
        row.pop("arc_phase", None)
        row.pop("beat_intent", None)
    for beat in ledger["beats"]:
        beat["arc_phase"] = "rising"
        beat["beat_intent"] = "admit the lamp is dead"

    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    receipt = lcl.run_ledger_clean(
        ledger, slot_fn=slot, bank_id="media_archive")
    assert receipt["context_seen"]["rows_with_arc_phase"] == 3
    assert receipt["context_seen"]["rows_with_beat_intent"] == 3


# ---------------------------------------------------------------------------
# it may never kill a render, and it must state its cost honestly
# ---------------------------------------------------------------------------


def test_a_junk_row_in_the_lines_array_cannot_kill_the_render():
    """`run_ledger_clean` is called UNWRAPPED from the writer tail.

    The row being judged was guarded, but its NEIGHBOURS were sliced raw out
    of `lines[]` and handed to `_is_voiced`, which does `.get()`. One stray
    string or null in that array -- an older or hand-edited ledger -- would
    have raised AttributeError straight out of the pass and killed the
    episode. `otr_ledger_view.grade()` filters the same array the same way,
    so the shape is not hypothetical.
    """
    ledger = _ledger(DIRTY, "And then we left.")
    ledger["lines"].insert(1, "a stray string where a row should be")
    ledger["lines"].insert(3, None)
    ledger["beats"].append("junk beat")

    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")
    assert receipt["voiced_rows"] == 3
    assert ledger["lines"][2]["text"] == FIXED


def test_the_receipt_counts_the_briefing_calls_it_actually_spent():
    """The writer tail promises this pass states its cost honestly.

    The per-act briefing runs BEFORE the receipt used to exist, so its calls
    were real spend that never appeared in `model_calls`.
    """
    ledger = _ledger("I have not slept.")
    slot = _Slot()
    receipt = lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert slot.summary_calls > 0
    assert receipt["briefing_calls"] == slot.summary_calls
    assert receipt["model_calls"] >= slot.summary_calls


def test_a_row_no_model_ever_saw_is_not_counted_as_a_failed_repair():
    ledger = _ledger(DIRTY)
    receipt = lcl.run_ledger_clean(ledger, slot_fn=None, bank_id="original")
    assert receipt["no_model"] == 1
    assert receipt["unclean"] == 0, (
        "'unclean' means a model tried and could not fix it")


def test_the_sha_check_proves_the_context_reached_the_prompt():
    """The check that catches "built it, then dropped it on the floor".

    A field COUNT cannot see this class: the string existed, it was passed
    into the builder, and the builder never rendered it. Hashing what we
    meant and looking for it in what we sent does see it.
    """
    ledger = _production_shaped_ledger()
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    receipt = lcl.run_ledger_clean(
        ledger, slot_fn=slot, bank_id="media_archive")

    verified = receipt["context_verified"]
    # A sight string and a short sha. 1 = saw it, 0 = did not.
    assert verified["sight"] == "1" * len(lcl.CONTEXT_FIELDS)
    assert verified["ok"] is True
    assert len(verified["sha"]) == 8
    assert verified["prompts"] > 0


def test_a_field_built_but_never_rendered_is_caught_and_named():
    """The literal bug this exists for, reproduced against the real checker."""
    built = [
        {"role": "system", "content": "you judge a line"},
        {"role": "user", "content": "THE LINE: I told you he would not stay."},
    ]
    landed = lcl.verify_context_landed(built, {
        "line": "I told you he would not stay.",
        # threaded into the builder, never rendered into the body
        "act": "this is the rising of the story",
    })
    assert landed["ok"] is False
    assert landed["missing"] == ["act"]
    # position 2 is `act` -- the one that did not arrive
    assert landed["sight"] == "11011", landed["sight"]


def test_the_sight_string_reads_left_to_right_in_a_fixed_order():
    """"11011" must mean the same thing in every log line, forever."""
    assert lcl.CONTEXT_FIELDS == (
        "line", "speaker", "act", "around", "complaint")

    built = [{"role": "user", "content": "THE LINE: hello there"}]
    all_there = lcl.verify_context_landed(built, {"line": "hello there"})
    assert all_there["sight"] == "11111", "a field not needed reads 1"

    lost_two = lcl.verify_context_landed(built, {
        "line": "hello there",
        "speaker": "Nan Reyes",       # built, never rendered
        "complaint": "1. '(he sighs)'",  # built, never rendered
    })
    assert lost_two["sight"] == "10110", lost_two["sight"]


def test_the_digest_is_stable_and_ignores_empty_fields():
    a = lcl.verify_context_landed(
        [{"role": "user", "content": "alpha beta"}],
        {"one": "alpha", "two": "beta", "three": ""},
    )
    b = lcl.verify_context_landed(
        [{"role": "user", "content": "alpha beta"}],
        {"one": "alpha", "two": "beta"},
    )
    assert a["sha"] == b["sha"], "an empty field must not change the digest"
    assert a["ok"] and b["ok"]


def test_the_prompt_markers_the_test_double_routes_on_are_pinned():
    """A canary for the fixture itself.

    `_Slot` routes judge / repair / briefing on two literal substrings. If a
    prompt is reworded and these drift, the suite fails in confusing ways
    somewhere else -- this test names the real cause in one line.
    """
    ledger = _ledger(DIRTY)
    slot = _Slot(repairs={DIRTY: [_replacement_reply(FIXED_REPLACEMENT)]})
    lcl.run_ledger_clean(ledger, slot_fn=slot, bank_id="original")

    assert slot.summary_calls and slot.judge_calls and slot.repair_calls, (
        "all three job kinds must have been routed")
    assert all("In about TEN WORDS" in p for p in slot.summary_prompts)
    assert all("DO THIS, IN ORDER:" in p for p in slot.judge_prompts)
    assert not any("DO THIS, IN ORDER:" in p for p in slot.repair_prompts)


# ---------------------------------------------------------------------------
# wiring -- code that is not wired in is dead code
# ---------------------------------------------------------------------------


def test_the_clean_stage_is_wired_into_the_one_shared_producer_boundary():
    """It must run in the tail EVERY source bank reaches, and run FIRST.

    Before ``run_ledger_cleanup``, because that pass re-stamps text metrics:
    a row rewritten here has to be measured after the rewrite, not before.
    """
    from tests.fixtures.writer_family import family_source

    # `_run_writer_tail` moved to nodes/_otr_writer_tail.py in lean-mean
    # order 9 slice 2, byte-identically; the pin follows the family.
    source = family_source()

    tail = source.split("def _run_writer_tail", 1)
    assert len(tail) == 2, "the shared writer tail has been renamed"
    body = tail[1]

    clean_at = body.find("run_ledger_clean(")
    cleanup_at = body.find("run_ledger_cleanup(")
    assert clean_at != -1, "the clean stage is not called from the tail"
    assert cleanup_at != -1
    assert clean_at < cleanup_at, (
        "the clean stage must run BEFORE the completion pass that re-stamps "
        "text metrics"
    )
    # The CREATIVE slot rewrites dialogue -- the tier that wrote the line
    # rewrites it, so the repaired line still sounds like its neighbours.
    call = body[clean_at:cleanup_at]
    assert "slot_fn=creative_generate_fn" in call
