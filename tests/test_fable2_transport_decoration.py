"""Markdown decoration on a structural label is TRANSPORT, not authorship.

THE LIVE FAILURE (2026-08-01). A local model wrapped its markup in Markdown --
``**TITLE:** ...``, ``**ANNOUNCER:** ...``, ``**END.**``. Every classifier in
``_otr_fable2_markup`` is ``^``-anchored, so a decorated line missed its own
classifier and fell through to the ``_RE_SPEAKER`` catch-all, whose group(1)
became a character literally named ``**TITLE``. Not in the cast -> UNKNOWN_SPEAKER
AND SKELETON_BREAK, on every line, for all four ladder rungs. Measured across one
canonical campaign: 106 UNKNOWN_SPEAKER + 106 SKELETON_BREAK, and 3 of 6 episodes
died in the writer.

The fix is a CLOSED grammar, not a Markdown sanitizer. These tests pin both what
it removes and -- more importantly -- what it must never touch.
"""

import hashlib

import pytest

from nodes._otr_fable2_markup import (
    _canonicalize_transport_line,
    normalize_fable2_markup_text,
    parse_fable2_markup,
)

_LINES = (
    "TITLE: The Quiet Relay",
    "MUSIC: low glass tones",
    "ANNOUNCER: A relay waits above the sleeping valley.",
    "SCENE 1: the mountain relay room",
    "{speaker}: The signal is steady now.",
    "ANNOUNCER: The relay returns to silence before dawn.",
    "CODA: The factual source beneath this fiction is:",
    "MUSIC: one low sustained note",
    "END.",
)


def _script(speaker="SELA"):
    """A STRUCTURALLY COMPLETE episode -- the skeleton the parser requires.

    Shaped after tests/test_45word_failure_regressions.py so these fixtures
    exercise decoration handling and NOTHING else. My first draft omitted the
    announcer outro and the closing MUSIC line, and failed for three reasons
    that had nothing to do with Markdown."""
    return "\n".join(row.format(speaker=speaker) for row in _LINES)


def _decorate(script):
    """Wrap every label the way the live model did: ``**LABEL:**`` / ``**END.**``."""
    out = []
    for line in script.splitlines():
        if line == "END.":
            out.append("**END.**")
            continue
        head, _, rest = line.partition(": ")
        out.append("**%s:** %s" % (head, rest))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# what it removes -- the forms actually observed in production
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,want", [
    # the wrapper spans the colon -- the shape the live model emitted
    ("**TITLE:** The Persistent Strawberry", "TITLE: The Persistent Strawberry"),
    ("**ANNOUNCER:** Good evening.", "ANNOUNCER: Good evening."),
    ("**SCENE 1:** The lighthouse", "SCENE 1: The lighthouse"),
    ("**MARKUS BUEHLER:** We tried.", "MARKUS BUEHLER: We tried."),
    # the wrapper closes before the colon
    ("**MUSIC**: sting", "MUSIC: sting"),
    # single-marker and underscore forms
    ("*MUSIC:* sting", "MUSIC: sting"),
    ("__CODA:__ later that night", "CODA: later that night"),
    ("_TITLE:_ Quiet", "TITLE: Quiet"),
    # standalone token, no colon -- verbatim from the live log
    ("**END.**", "END."),
])
def test_removes_balanced_decoration_from_labels(raw, want):
    got, notes = _canonicalize_transport_line(raw)
    assert got == want
    assert notes, "a removal must be REPORTED, never silent"


# --------------------------------------------------------------------------- #
# what it must NEVER touch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", [
    "TITLE: already clean",
    "ANNOUNCER: emphasis **inside** the line stays",
    "SELA: she said *quietly* and left",
    "**UNBALANCED: no closing marker",      # unbalanced -> stays a loud defect
    "**MIXED:__ payload",                   # mixed markers -> stays a defect
    "> quoted line",                        # not in the grammar
    "# heading",                            # not in the grammar
    "`code`",                               # not in the grammar
    "- bullet",                             # not in the grammar
])
def test_leaves_everything_else_byte_identical(raw):
    got, notes = _canonicalize_transport_line(raw)
    assert got == raw
    assert notes == ()


def test_spoken_payload_is_never_rewritten():
    """The parser's purity promise: it 'never rewrites a spoken word'.

    Emphasis INSIDE a spoken line is authorship. Only the proven outer wrapper
    around the LABEL comes off."""
    got, _ = _canonicalize_transport_line("**BO NI:** Hello **world**, it *is* me")
    assert got == "BO NI: Hello **world**, it *is* me"


def test_a_genuinely_unknown_speaker_still_raises_unknown_speaker():
    """Canonicalization must not weaken closed-roster enforcement.

    Decoration is stripped BEFORE the catch-all, so an undecorated name that is
    simply not in the cast still fails -- the diagnostic keeps its meaning."""
    parsed, defects = parse_fable2_markup(_script(speaker="STRANGER"), ["SELA"])
    assert parsed is None
    rendered = " ".join(str(d) for d in defects)
    assert "UNKNOWN_SPEAKER" in rendered, rendered
    assert "STRANGER" in rendered, rendered


def test_a_decorated_unknown_speaker_is_reported_by_its_REAL_name():
    """Before the fix every decorated label became an unknown speaker, so the
    real defect was buried under 106 false ones. A decorated name that is
    genuinely off-roster must now be named WITHOUT its decoration."""
    parsed, defects = parse_fable2_markup(
        _decorate(_script(speaker="STRANGER")), ["SELA"])
    assert parsed is None
    rendered = " ".join(str(d) for d in defects)
    assert "STRANGER" in rendered
    assert "**STRANGER" not in rendered


# --------------------------------------------------------------------------- #
# the two normalizers must not diverge (kibitz r1 MUST-FIX 3)
# --------------------------------------------------------------------------- #
def test_decorated_script_parses_and_hashes_the_same_text():
    """``normalize_fable2_markup_text`` builds normalized_source + its sha256 +
    the proof map; ``_normalize_line`` decides classification. If only one of
    them canonicalized, the accepted script would differ from the artifact
    hashed beside it -- a receipt describing a document nobody parsed."""
    clean = _script()
    decorated = _decorate(clean)

    parsed_cln, defects_cln = parse_fable2_markup(clean, ["SELA"])
    parsed_dec, defects_dec = parse_fable2_markup(decorated, ["SELA"])
    assert defects_cln == (), defects_cln
    assert defects_dec == (), defects_dec
    assert parsed_cln is not None and parsed_dec is not None

    n_cln = normalize_fable2_markup_text(clean)
    n_dec = normalize_fable2_markup_text(decorated)
    assert n_dec == n_cln
    assert (hashlib.sha256(n_dec.encode()).hexdigest()
            == hashlib.sha256(n_cln.encode()).hexdigest())


def test_the_exact_live_failure_now_parses():
    """The decorated form that killed 3 of 6 legs now parses clean."""
    parsed, defects = parse_fable2_markup(_decorate(_script()), ["SELA"])
    assert defects == (), defects
    assert parsed is not None
    assert parsed.title == "The Quiet Relay"
