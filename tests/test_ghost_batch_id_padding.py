"""A mis-padded opaque id is the SAME row, not an unknown one.

MEASURED, not imagined. Across the 27 episode ledgers on this box that carry
`ghost_prompt` objects, 127 of 263 beats fell to the deterministic path, and
the single most common non-content rejection was::

    attempt 2 rejected: Ghost batch row carries unknown id 'g0010'

25 beats -- a whole 25-beat episode, `signal_lost_the_weight_of_grief_
20260903_005108`, lost every authored prompt to it. `opaque_id` spells ordinal
10 as ``g010``; the batch prompt puts that exact string in front of the model;
and the model, having just written ``g000`` through ``g009``, continued the
pattern it saw and wrote ``g0010``. It only happens on episodes with more than
ten Ghost beats, which is why the 8-beat dailies never showed it.

An opaque id is a POSITIONAL INTEGER with a fixed prefix and nothing else -- it
carries no content and no meaning beyond "which row". So a digit run that reads
as the same integer names the same row, and refusing it throws away a whole
episode's authoring over a zero. That is the entire scope of this tolerance:

* the digits are read as an INTEGER and re-spelled with `opaque_id`;
* the result must still be an EXPECTED id, or it is still unknown;
* the duplicate check runs on the CANONICAL id, so two spellings of one row
  are a repeat, exactly as two copies of one spelling are;
* NOTHING ELSE relaxes -- an unknown prefix, a non-numeric tail, extra envelope
  keys, extra row fields, trailing prose, a repeated JSON key and a missing row
  are all still hard rejections.
"""
import pytest

from nodes._otr_video_engines import ghost_signal_author as gsa


def _envelope(pairs):
    import json
    return json.dumps({"shots": [{"id": i, "drawable_beat": b}
                                 for i, b in pairs]})


ELEVEN = [gsa.opaque_id(i) for i in range(11)]
LEAVES = ["the console dial settles into a steady glow %d" % i
          for i in range(11)]


def test_the_measured_failure_now_parses():
    """`g0010` for ordinal 10, the exact string off the failed episode."""
    ids = list(ELEVEN)
    ids[10] = "g0010"
    out = gsa.parse_batch_response(_envelope(zip(ids, LEAVES)), ELEVEN)
    assert set(out) == set(ELEVEN)
    assert out["g010"] == LEAVES[10]


@pytest.mark.parametrize("spelling", ["g0010", "g10", "g00010", "g000010"])
def test_every_spelling_of_ordinal_ten_names_row_ten(spelling):
    ids = list(ELEVEN)
    ids[10] = spelling
    out = gsa.parse_batch_response(_envelope(zip(ids, LEAVES)), ELEVEN)
    assert out["g010"] == LEAVES[10]


def test_the_exact_spelling_is_still_the_normal_case():
    out = gsa.parse_batch_response(_envelope(zip(ELEVEN, LEAVES)), ELEVEN)
    assert out == dict(zip(ELEVEN, LEAVES))


def test_a_padding_variant_of_a_row_the_batch_did_not_ask_for_is_still_unknown():
    """Tolerance is about SPELLING, never about admitting an extra row."""
    ids = list(ELEVEN)
    ids[10] = "g0099"          # reads as 99; this batch has 11 rows
    with pytest.raises(gsa.GhostAuthorParseError, match="unknown id"):
        gsa.parse_batch_response(_envelope(zip(ids, LEAVES)), ELEVEN)


@pytest.mark.parametrize("bogus", [
    "x010",        # wrong prefix
    "g010a",       # non-numeric tail
    "g 010",       # whitespace
    "G010",        # wrong case -- the prefix is literal
    "g",           # no digits at all
    "g-010",       # sign
    "g010.0",      # not an integer spelling
])
def test_nothing_but_a_bare_digit_run_is_repaired(bogus):
    ids = list(ELEVEN)
    ids[10] = bogus
    with pytest.raises(gsa.GhostAuthorParseError, match="unknown id"):
        gsa.parse_batch_response(_envelope(zip(ids, LEAVES)), ELEVEN)


def test_two_spellings_of_one_row_are_a_repeat_not_two_rows():
    ids = list(ELEVEN)
    ids[10] = "g0010"
    pairs = list(zip(ids, LEAVES)) + [("g010", "a second leaf for the same row")]
    with pytest.raises(gsa.GhostAuthorParseError, match="repeats id"):
        gsa.parse_batch_response(_envelope(pairs), ELEVEN)


def test_a_repaired_id_still_has_to_be_present_to_satisfy_the_batch():
    """Whole batch or nothing -- the tolerance does not create rows."""
    ids = list(ELEVEN)[:10]
    with pytest.raises(gsa.GhostAuthorParseError, match="missing id"):
        gsa.parse_batch_response(_envelope(zip(ids, LEAVES[:10])), ELEVEN)


# --------------------------------------------------------------------------- #
# Everything else this parser refuses, refused exactly as before.
# --------------------------------------------------------------------------- #

def test_the_strictness_that_matters_is_untouched():
    import json
    good = list(zip(ELEVEN, LEAVES))
    # an extra envelope key
    with pytest.raises(gsa.GhostAuthorParseError, match="exactly one key"):
        gsa.parse_batch_response(
            json.dumps({"shots": [{"id": i, "drawable_beat": b}
                                  for i, b in good], "note": "hi"}), ELEVEN)
    # an extra field on a row
    with pytest.raises(gsa.GhostAuthorParseError, match="exactly id"):
        gsa.parse_batch_response(
            json.dumps({"shots": [{"id": i, "drawable_beat": b, "why": "x"}
                                  for i, b in good]}), ELEVEN)
    # trailing prose after the object
    with pytest.raises(gsa.GhostAuthorParseError, match="trailing content"):
        gsa.parse_batch_response(
            _envelope(good) + "\n\nHope that helps.", ELEVEN)
    # not JSON at all
    with pytest.raises(gsa.GhostAuthorParseError, match="not JSON"):
        gsa.parse_batch_response('{"shots": [', ELEVEN)


def test_the_repair_is_reported_never_silent(caplog):
    import logging
    ids = list(ELEVEN)
    ids[10] = "g0010"
    with caplog.at_level(logging.WARNING,
                         logger="OTR.video.ghost_signal_author"):
        gsa.parse_batch_response(_envelope(zip(ids, LEAVES)), ELEVEN)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "g0010" in joined and "g010" in joined


def test_a_clean_batch_logs_no_repair(caplog):
    import logging
    with caplog.at_level(logging.WARNING,
                         logger="OTR.video.ghost_signal_author"):
        gsa.parse_batch_response(_envelope(zip(ELEVEN, LEAVES)), ELEVEN)
    assert not [r for r in caplog.records if "id" in r.getMessage()]


# --------------------------------------------------------------------------- #
# A JSON failure now says WHAT it choked on, not only where.
# --------------------------------------------------------------------------- #

def test_a_json_failure_carries_the_text_around_it():
    """21 real beats died on "not JSON" and nothing on disk said why.

    A `JSONDecodeError` names a column. The model's raw response is stored
    nowhere, so "line 1 column 852" could equally mean a truncated response, an
    unescaped quote inside a leaf, or prose that was never JSON -- three bugs
    with three different fixes. The reason now carries the text around the
    failure, so the NEXT occurrence is readable off the ledger.
    """
    broken = ('{"shots": [{"id": "g000", "drawable_beat": "the dial '
              'reads "OFF" and settles"}]}')
    with pytest.raises(gsa.GhostAuthorParseError) as caught:
        gsa.parse_batch_response(broken, ["g000"])
    message = str(caught.value)
    assert "not JSON" in message
    assert "around the failure" in message
    assert "<<HERE>>" in message
    assert "OFF" in message


def test_the_excerpt_is_bounded_and_stays_on_one_line():
    filler = "x" * 4000
    broken = '{"shots": [{"id": "g000", "drawable_beat": "%s\n\n' % filler
    with pytest.raises(gsa.GhostAuthorParseError) as caught:
        gsa.parse_batch_response(broken, ["g000"])
    message = str(caught.value)
    assert "\n" not in message and "\r" not in message
    assert len(message) < 400, len(message)


def test_a_failure_with_no_position_degrades_quietly():
    """The excerpt is an ADDITION; its absence must never be an error."""
    class _NoPos(ValueError):
        pass

    assert gsa._decode_excerpt("some body", _NoPos("boom")) == ""
    assert gsa._decode_excerpt("", ValueError("boom")) == ""
    assert gsa._decode_excerpt(None, ValueError("boom")) == ""
