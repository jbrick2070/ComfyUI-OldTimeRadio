"""Story-Quality R2 C1 + C2 -- specificity anchors + central object (pure).

Roundtable-converged (docs/2026-06-22-story-quality-c1c2/). Deterministic
derivations from the curated key_terms + zero-ripple prompt injectors. CPU.
UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_specificity import (  # noqa: E402
    MAX_ANCHORS,
    _cast_tokens,
    derive_central_object,
    derive_specificity_anchors,
    inject_anchors_into_header,
    inject_central_object_into_brief,
    sanitize_for_prompt,
)

_KEY_TERMS = [
    "WALLOPS ISLAND", "NASA", "Katalyst", "Swift", "Link",
    "three robotic arms", "a $500M telescope",
]
_CAST = [{"name": "CHRIS SHAW"}, {"name": "SKIP SPENDER"}]


class TestCastTokens:
    def test_full_name_and_long_subtokens(self):
        toks = _cast_tokens([{"name": "Chris Shaw"}, {"name": "A. J. Doe"}])
        assert "chris shaw" in toks and "chris" in toks and "shaw" in toks
        # short sub-tokens (<=2) are NOT added (avoids over-excluding)
        assert "a" not in toks and "j" not in toks
        assert "doe" in toks


class TestAnchors:
    def test_filters_cast_and_caps(self):
        out = derive_specificity_anchors(_KEY_TERMS, _CAST)
        # anchors = the first MAX_ANCHORS non-cast key_terms in order
        assert out == ["WALLOPS ISLAND", "NASA", "Katalyst", "Swift", "Link"]
        assert len(out) == MAX_ANCHORS
        assert "CHRIS SHAW" not in out and "SKIP SPENDER" not in out

    def test_object_phrase_kept_within_cap(self):
        out = derive_specificity_anchors(
            ["three robotic arms", "a $500M telescope", "Swift"], _CAST)
        assert "three robotic arms" in out and "a $500M telescope" in out

    def test_partial_cast_name_does_not_over_exclude(self):
        # 'a new car' must survive even though cast 'A. J. Doe' has token 'a'
        out = derive_specificity_anchors(["a new car", "the courthouse"],
                                         [{"name": "A. J. Doe"}])
        assert "a new car" in out

    def test_dedupe_and_empty(self):
        assert derive_specificity_anchors(["NASA", "nasa"], []) == ["NASA"]
        assert derive_specificity_anchors([], []) == []
        assert derive_specificity_anchors(None, None) == []


class TestCentralObject:
    def test_accepts_object_phrase(self):
        assert derive_central_object(_KEY_TERMS, _CAST) == "three robotic arms"

    @pytest.mark.parametrize("kts,expected", [
        (["NASA", "Swift"], ""),                       # all bare entities
        (["James Webb Telescope"], ""),                # Title-Case proper noun
        (["Olympic Analytical Laboratory"], ""),       # entity suffix
        (["2024", "500"], ""),                         # numeric
        (["a cracked phone"], "a cracked phone"),      # object phrase
        (["the missing ledger", "NASA"], "the missing ledger"),
    ])
    def test_rules(self, kts, expected):
        assert derive_central_object(kts, []) == expected

    def test_idempotent(self):
        a = derive_central_object(_KEY_TERMS, _CAST)
        b = derive_central_object(_KEY_TERMS, _CAST)
        assert a == b


class TestInjectors:
    def test_anchors_block_appended_and_omitted(self):
        h = "EPISODE CONTEXT"
        out = inject_anchors_into_header(h, ["three robotic arms", "Swift"])
        assert "Specificity anchors" in out and "three robotic arms" in out
        assert inject_anchors_into_header(h, []) == h        # unchanged when empty

    def test_central_object_not_appended_to_close(self):
        # operator 2026-06-26: the announcer's close ends on the brief news
        # summary, NOT a tacked-on random object -- a bare "Artemis program."
        # (proper-noun program) or even "A cracked phone." (a fragment, not a
        # complete sentence) dangles as an unneeded spoken reference. The object
        # is no longer appended; the brief is returned UNCHANGED. (The earlier
        # scaffold-label leak "Central object, if useful: ..." stays fixed too.)
        b = "The lab cleared its name."
        out = inject_central_object_into_brief(b, "a cracked phone")
        assert out == b                                      # no object appended
        assert "cracked phone" not in out
        assert "Central object" not in out                   # scaffold label never leaks
        assert "if useful" not in out
        # proper-noun / program names + empties: all leave the close unchanged.
        assert inject_central_object_into_brief(b, "Artemis program") == b
        assert inject_central_object_into_brief(b, "") == b
        assert inject_central_object_into_brief(b, "the pencil.") == b

    def test_sanitize_strips_newlines_no_word_slice(self):
        s = sanitize_for_prompt("an experimental\nquantum\tcomputer")
        assert "\n" not in s and "\t" not in s
        assert s == "an experimental quantum computer"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
