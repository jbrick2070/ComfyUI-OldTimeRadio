"""PBUG-20261028-01 -- an ANNOUNCER sign-off below the CODA killed the episode.

FOUND LIVE. Overnight pair 13 (2026-10-28, `otr_8gb_animatediff`, writer
Gemma 4 E2B, news bank, one act) died in `_pass_script` after four attempts:

  - SKELETON_BREAK: ANNOUNCER line after the CODA (line 16)
  salvage also refused: (the same defect)

The writer heartbeats in the server log show the tails of the drafts. The last
two came back byte-identical, ending

  ... CODA: ...  MUSIC: ...fades to silence  ANNOUNCER: The shadow of
  innovation falls across the threshold of what is right.  END.

TWO CAUSES, neither of them the model's prose:

* THE REPAIR NOTE POINTED THE WRONG WAY. This defect was one of the
  `_FRAME_ORDER_MARKERS`, whose note was written for the opposite fault (an
  outro that opened too EARLY) and tells the model to "move the ANNOUNCER
  outro down so it comes after all the drama". For a sign-off that is already
  below the CODA that is the defect itself, so every rung kept it there.
* SALVAGE REFUSED A DRAFT THAT LOSES NOTHING. All the drama was intact; the
  announcer's last line sat one row too low. The announcer speaks the outro
  and the CODA alike, so salvage now keeps the line with the outro, ahead of
  the CODA, and receipts the move.
"""
from __future__ import annotations

from nodes._otr_scifi_news_pro_markup import (
    ANNOUNCER_NAME,
    NewsProParseDefect,
    parse_scifi_news_pro_markup,
)
from nodes import _otr_scifi_news_pro as scifi_news_pro
from nodes._otr_scifi_news_pro import (
    _frame_order_repair_note,
    _post_coda_announcer_repair_note,
)

D = NewsProParseDefect
CAST = ["Zeynep Temel", "Guowei Xu", "Oliver Grant"]
SIGN_OFF = "The shadow of innovation falls across the threshold of what is right."

# The refused draft, rebuilt from the leg's heartbeats and its defect line
# number: the ANNOUNCER sign-off is line 16, after the CODA and the closing
# MUSIC, and nothing else in the draft is malformed.
REFUSED_DRAFT = "\n".join([
    "TITLE: The Price of Touch",                                          # 1
    "MUSIC: Detroit techno, up and under",                                # 2
    f"{ANNOUNCER_NAME}: Tonight, a lab that can feel.",                   # 3
    "SCENE 1: a robotics lab at night",                                   # 4
    "Zeynep Temel: The data shows that if we release the core, anyone can build it.",  # 5
    "Guowei Xu: Control must be tempered by foresight.",                  # 6
    "Oliver Grant: Then sell it and address the ethical concerns later.",  # 7
    "Zeynep Temel: The value is not in the transaction, Oliver.",         # 8
    "Guowei Xu: This capture needs a framework we do not yet possess.",   # 9
    "Oliver Grant: We cannot wait for one.",                              # 10
    "Zeynep Temel: We cannot address them later. The design depends on its openness.",  # 11
    f"{ANNOUNCER_NAME}: The silence in the lab said what nobody would.",  # 12
    "CODA: The algorithms were released. The price was paid in silence.",  # 13
    "",                                                                   # 14
    "MUSIC: Detroit techno fades slowly to silence",                      # 15
    f"{ANNOUNCER_NAME}: {SIGN_OFF}",                                      # 16
    "END.",                                                               # 17
])


def test_the_refused_draft_reproduces_the_live_defect_exactly():
    parsed, defects = parse_scifi_news_pro_markup(REFUSED_DRAFT, CAST)
    assert parsed is None
    assert [(d.code, d.detail, d.line_no) for d in defects] == [
        (D.SKELETON_BREAK, "ANNOUNCER line after the CODA", 16)]


def test_the_repair_note_says_move_the_sign_off_UP():
    _parsed, defects = parse_scifi_news_pro_markup(REFUSED_DRAFT, CAST)
    note = _post_coda_announcer_repair_note(defects)
    assert "move that ANNOUNCER row UP" in note
    assert "(line 16)" in note
    assert "above the CODA" in note
    # THE LIVE DEFECT: the old note for this defect said the reverse.
    assert _frame_order_repair_note(defects) == "", (
        "the outro-too-early note tells the model to move the outro DOWN, "
        "which is the defect itself here")
    assert _post_coda_announcer_repair_note(()) == ""


def test_the_frame_order_note_still_serves_its_own_defect():
    """Splitting the post-CODA case out must not silence the original note."""
    _parsed, defects = parse_scifi_news_pro_markup("\n".join([
        "TITLE: T", "MUSIC: m", f"{ANNOUNCER_NAME}: Hello.",
        "SCENE 1: a room", "Guowei Xu: One.",
        f"{ANNOUNCER_NAME}: Too early.", "Guowei Xu: Two.",
        "CODA: c", "MUSIC: m", "END.",
    ]), CAST)
    assert "move the ANNOUNCER outro down" in _frame_order_repair_note(defects)
    assert _post_coda_announcer_repair_note(defects) == ""


def test_SALVAGE_delivers_the_episode_that_died():
    parsed, defects = parse_scifi_news_pro_markup(
        REFUSED_DRAFT, CAST, salvage=True)
    assert defects == (), f"salvage must deliver, got {defects}"
    # The sign-off keeps its words and its voice, spoken with the outro.
    assert parsed.announcer_outro[-1] == SIGN_OFF
    assert parsed.coda.startswith("The algorithms were released.")
    assert sum(len(s.lines) for s in parsed.scenes) == 7
    assert any("ANNOUNCER line after the CODA kept" in r
               for r in parsed.speaker_resolutions), (
        "the move must be receipted, never hidden")


class ScriptedWriter:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        self.prompts.append(list(messages)[-1]["content"])
        return self.replies.pop(0)


def _run(writer):
    return scifi_news_pro._run_markup_ladder(
        writer,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=CAST,
        initial_temperature=0.7,
    )


def test_the_right_rule_REACHES_the_repair_prompt():
    fixed = REFUSED_DRAFT.replace(f"\n{ANNOUNCER_NAME}: {SIGN_OFF}", "")
    writer = ScriptedWriter([REFUSED_DRAFT, fixed])
    _raw, parsed, _diag = _run(writer)
    assert parsed is not None and len(writer.prompts) == 2
    second = writer.prompts[1]
    assert "move that ANNOUNCER row UP" in second
    assert "move the ANNOUNCER outro down" not in second


def test_four_identical_refusals_now_deliver_instead_of_raising():
    """The live leg: the model repeated the shape on every rung."""
    writer = ScriptedWriter([REFUSED_DRAFT] * 4)
    _raw, parsed, diag = _run(writer)
    assert diag.get("salvaged") is True
    assert parsed.announcer_outro[-1] == SIGN_OFF
