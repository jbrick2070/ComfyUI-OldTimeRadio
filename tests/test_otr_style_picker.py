"""tests/test_otr_style_picker.py -- unit tests for the two-pass
style picker (commit landing 2026-05-10).

Hermetic: no GPU, no actual LLM. generate_fn is mocked end-to-end.

Coverage map:
  - DESCRIPTOR_RE grammar acceptance/rejection
  - _sample_seeds determinism
  - _compute_article_hash stability
  - _parse_inventor_output: 5-line happy path; bullet-stripping;
    grammar fail; count fail; exact-duplicate fail; distinctness
    rule fail
  - _validate_chooser_output: exact match; bullet-stripped match;
    quoted match; non-candidate fail; empty fail
  - StylePick model: shape; chosen-must-match-grammar; candidates-
    grammar-and-distinctness validators
  - pick_style happy path: inventor produces 5, chooser picks one
  - pick_style inventor retry recovery
  - pick_style inventor all-fail -> StyleGenerationFailedError
  - pick_style chooser mismatch -> StyleGenerationFailedError
  - pick_style empty article precondition -> raise
  - pick_style seed pool too small -> raise

ADR alignment: same fail-loud discipline as
nodes/news_interpreter.py + the writer's prior _generate_style_via_llm
(commit 62e85f2). Pure-python tests; no torch / transformers
imported here.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Same sys.path setup as sibling tests so ``nodes`` is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_style_picker as _SP  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_TEN_PRESETS = [
    "closed_room_suspense",
    "detective_case_file",
    "pulp_serial_cliffhanger",
    "mission_control_procedural",
    "deep_space_distress_call",
    "noir_interrogation",
    "small_town_uncanny",
    "radio_newsroom_emergency",
    "haunted_broadcast_signal",
    "laboratory_containment",
]


def _five_valid_distinct() -> list[str]:
    """5 grammar-valid descriptors with distinct root sets (max 1
    shared root per pair). Used as the canned inventor output."""
    return [
        "unknown_origin_signal_log",
        "decommissioned_dish_archive",
        "midnight_newsroom_emergency",
        "vacuum_chamber_breach",
        "haunted_repeater_loop",
    ]


def _make_canned_generate_fn(responses):
    """Build a generate_fn that returns each canned response in
    order across calls. Raises after exhaustion so a buggy test
    doesn't silently re-use the last response."""
    iterator = iter(responses)

    def _gen(messages, *, temperature, max_new_tokens):
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError(
                "test exhausted canned responses; check call count"
            ) from exc

    return _gen


# ---------------------------------------------------------------------------
# Grammar regex
# ---------------------------------------------------------------------------


class TestDescriptorRegex:
    def test_accepts_two_words(self):
        assert _SP.DESCRIPTOR_RE.match("noir_interrogation")

    def test_accepts_five_words(self):
        assert _SP.DESCRIPTOR_RE.match("a_b_c_d_e")

    def test_rejects_one_word(self):
        assert not _SP.DESCRIPTOR_RE.match("noir")

    def test_rejects_six_words(self):
        assert not _SP.DESCRIPTOR_RE.match("a_b_c_d_e_f")

    def test_rejects_uppercase(self):
        assert not _SP.DESCRIPTOR_RE.match("Noir_Interrogation")

    def test_rejects_hyphens(self):
        assert not _SP.DESCRIPTOR_RE.match("noir-interrogation")

    def test_rejects_digits(self):
        assert not _SP.DESCRIPTOR_RE.match("noir_2_interrogation")

    def test_rejects_double_underscore(self):
        assert not _SP.DESCRIPTOR_RE.match("noir__interrogation")

    def test_rejects_trailing_underscore(self):
        assert not _SP.DESCRIPTOR_RE.match("noir_interrogation_")


# ---------------------------------------------------------------------------
# _sample_seeds determinism
# ---------------------------------------------------------------------------


class TestSampleSeeds:
    def test_same_rng_same_sample(self):
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        a = _SP._sample_seeds(rng_a, _TEN_PRESETS, 5)
        b = _SP._sample_seeds(rng_b, _TEN_PRESETS, 5)
        assert a == b
        assert len(a) == 5
        assert len(set(a)) == 5  # no duplicates

    def test_different_seeds_different_samples(self):
        a = _SP._sample_seeds(random.Random(1), _TEN_PRESETS, 5)
        b = _SP._sample_seeds(random.Random(2), _TEN_PRESETS, 5)
        # Different seeds usually produce different orderings; if
        # this test ever fails by random chance, both seeds map to
        # the same 5-of-10 permutation -- bump one seed.
        assert a != b

    def test_pool_too_small_raises(self):
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP._sample_seeds(random.Random(0), _TEN_PRESETS[:3], 5)
        assert "seed_pool too small" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _compute_article_hash stability
# ---------------------------------------------------------------------------


class TestArticleHash:
    def test_stable_across_calls(self):
        h1 = _SP._compute_article_hash("hello world")
        h2 = _SP._compute_article_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_text_different_hash(self):
        assert _SP._compute_article_hash("a") != _SP._compute_article_hash("b")


# ---------------------------------------------------------------------------
# _parse_inventor_output
# ---------------------------------------------------------------------------


class TestParseInventorOutput:
    def test_happy_path_5_clean_lines(self):
        raw = "\n".join(_five_valid_distinct())
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_strips_dash_bullets(self):
        raw = "\n".join(f"- {c}" for c in _five_valid_distinct())
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_strips_numbered_list(self):
        raw = "\n".join(f"{i+1}. {c}" for i, c in enumerate(_five_valid_distinct()))
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_strips_quotes(self):
        raw = "\n".join(f'"{c}"' for c in _five_valid_distinct())
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_lowercases(self):
        raw = "\n".join(c.upper() for c in _five_valid_distinct())
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _SP._parse_inventor_output("")

    def test_count_too_few_raises(self):
        raw = "\n".join(_five_valid_distinct()[:3])
        with pytest.raises(ValueError, match="3 parseable lines"):
            _SP._parse_inventor_output(raw)

    def test_count_too_many_raises(self):
        raw = "\n".join(_five_valid_distinct() + ["bonus_descriptor_a"])
        with pytest.raises(ValueError, match="6 parseable lines"):
            _SP._parse_inventor_output(raw)

    def test_grammar_invalid_line_raises(self):
        # Parser lowercases before regex check (tolerant to mixed-
        # case LLM output), so we need a string that fails the
        # regex even after .lower(). Hyphen is a realistic LLM
        # mistake -- ChatGPT-style models often emit hyphenated
        # tags by default.
        candidates = _five_valid_distinct()
        candidates[2] = "noir-interrogation-chamber"
        raw = "\n".join(candidates)
        with pytest.raises(ValueError, match="grammar"):
            _SP._parse_inventor_output(raw)

    def test_exact_duplicate_raises(self):
        candidates = _five_valid_distinct()
        candidates[2] = candidates[0]
        raw = "\n".join(candidates)
        with pytest.raises(ValueError, match="duplicate"):
            _SP._parse_inventor_output(raw)

    def test_distinctness_rule_violated_raises(self):
        # Two entries sharing 2 root words should fail.
        bad = [
            "noir_interrogation_chamber",
            "noir_interrogation_archive",  # shares 'noir' + 'interrogation'
            "deep_space_distress_call",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
        ]
        raw = "\n".join(bad)
        with pytest.raises(ValueError, match="share 2 root words"):
            _SP._parse_inventor_output(raw)


# ---------------------------------------------------------------------------
# _validate_chooser_output
# ---------------------------------------------------------------------------


class TestValidateChooserOutput:
    def test_exact_match(self):
        candidates = _five_valid_distinct()
        out = _SP._validate_chooser_output("vacuum_chamber_breach", candidates)
        assert out == "vacuum_chamber_breach"

    def test_strips_whitespace(self):
        candidates = _five_valid_distinct()
        out = _SP._validate_chooser_output("  vacuum_chamber_breach  \n", candidates)
        assert out == "vacuum_chamber_breach"

    def test_strips_quotes(self):
        candidates = _five_valid_distinct()
        out = _SP._validate_chooser_output('"vacuum_chamber_breach"', candidates)
        assert out == "vacuum_chamber_breach"

    def test_strips_bullet(self):
        candidates = _five_valid_distinct()
        out = _SP._validate_chooser_output("- vacuum_chamber_breach", candidates)
        assert out == "vacuum_chamber_breach"

    def test_takes_first_line(self):
        candidates = _five_valid_distinct()
        out = _SP._validate_chooser_output(
            "vacuum_chamber_breach\nbecause it matches the stakes",
            candidates,
        )
        assert out == "vacuum_chamber_breach"

    def test_non_candidate_raises(self):
        with pytest.raises(ValueError, match="not in the candidate list"):
            _SP._validate_chooser_output(
                "noir_thriller", _five_valid_distinct(),
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _SP._validate_chooser_output("", _five_valid_distinct())


# ---------------------------------------------------------------------------
# StylePick pydantic model
# ---------------------------------------------------------------------------


def _valid_stylepick_kwargs() -> dict:
    return {
        "chosen": "vacuum_chamber_breach",
        "candidates": _five_valid_distinct(),
        "seed_sample": _TEN_PRESETS[:5],
        "article_hash": "a" * 64,
        "model_id": "test-model",
        "temp_pass1": 0.6,
        "temp_pass2": 0.1,
        "pass1_attempts": 1,
        "pass1_duration_ms": 100,
        "pass2_duration_ms": 50,
    }


class TestStylePickModel:
    def test_happy_construction(self):
        pick = _SP.StylePick(**_valid_stylepick_kwargs())
        assert pick.chosen == "vacuum_chamber_breach"
        assert len(pick.candidates) == 5

    def test_chosen_grammar_invalid_raises(self):
        kw = _valid_stylepick_kwargs()
        kw["chosen"] = "not-snake-case"
        with pytest.raises(ValidationError):
            _SP.StylePick(**kw)

    def test_candidates_must_be_5(self):
        kw = _valid_stylepick_kwargs()
        kw["candidates"] = _five_valid_distinct()[:4]
        with pytest.raises(ValidationError):
            _SP.StylePick(**kw)

    def test_candidates_distinctness_enforced(self):
        kw = _valid_stylepick_kwargs()
        kw["candidates"] = [
            "noir_interrogation_chamber",
            "noir_interrogation_archive",  # shares 2 roots with above
            "deep_space_distress",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
        ]
        with pytest.raises(ValidationError, match="share 2 root"):
            _SP.StylePick(**kw)

    def test_pass1_attempts_capped(self):
        kw = _valid_stylepick_kwargs()
        kw["pass1_attempts"] = 99
        with pytest.raises(ValidationError):
            _SP.StylePick(**kw)


# ---------------------------------------------------------------------------
# pick_style end-to-end
# ---------------------------------------------------------------------------


class TestPickStyle:
    def test_happy_path(self):
        inventor_response = "\n".join(_five_valid_distinct())
        chooser_response = "vacuum_chamber_breach"
        gen = _make_canned_generate_fn([inventor_response, chooser_response])
        pick = _SP.pick_style(
            gen,
            article_text="A real news story about black holes.",
            seed_pool=_TEN_PRESETS,
            rng=random.Random("happy"),
            model_id="test-model",
        )
        assert pick.chosen == "vacuum_chamber_breach"
        assert pick.candidates == _five_valid_distinct()
        assert pick.pass1_attempts == 1
        assert len(pick.seed_sample) == 5
        assert pick.model_id == "test-model"
        assert len(pick.article_hash) == 64

    def test_inventor_retry_recovers(self):
        # First attempt invalid grammar; second attempt valid.
        bad = "\n".join(["BadCaps", "noir_interrogation", "deep_space_distress",
                         "vacuum_chamber_breach", "haunted_repeater_loop"])
        good = "\n".join(_five_valid_distinct())
        chosen = "midnight_newsroom_emergency"
        gen = _make_canned_generate_fn([bad, good, chosen])
        pick = _SP.pick_style(
            gen,
            article_text="A second test article.",
            seed_pool=_TEN_PRESETS,
            rng=random.Random("retry"),
            model_id="test-model",
        )
        assert pick.chosen == chosen
        assert pick.pass1_attempts == 2

    def test_inventor_all_fail_raises(self):
        bad = "only_one_line"
        gen = _make_canned_generate_fn([bad, bad, bad])
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(
                gen,
                article_text="Article that breaks the inventor.",
                seed_pool=_TEN_PRESETS,
                rng=random.Random("allfail"),
                model_id="test-model",
            )
        assert "after 3 attempts" in str(exc_info.value)

    def test_chooser_mismatch_raises(self):
        inventor_response = "\n".join(_five_valid_distinct())
        chooser_response = "rogue_descriptor_outside_pool"
        gen = _make_canned_generate_fn([inventor_response, chooser_response])
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(
                gen,
                article_text="Article whose chooser goes rogue.",
                seed_pool=_TEN_PRESETS,
                rng=random.Random("chooser-rogue"),
                model_id="test-model",
            )
        assert "chooser output validation failed" in str(exc_info.value)

    def test_chooser_llm_raise_propagates(self):
        inventor_response = "\n".join(_five_valid_distinct())

        def _gen(messages, *, temperature, max_new_tokens):
            # First call (inventor) returns valid; second call
            # (chooser) raises.
            if temperature >= 0.5:  # inventor temperatures
                return inventor_response
            raise RuntimeError("chooser model offline")

        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(
                _gen,
                article_text="Article whose chooser model dies.",
                seed_pool=_TEN_PRESETS,
                rng=random.Random("chooser-offline"),
                model_id="test-model",
            )
        assert "chooser LLM call failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_empty_article_raises_precondition(self):
        gen = _make_canned_generate_fn([])
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(
                gen,
                article_text="",
                seed_pool=_TEN_PRESETS,
                rng=random.Random("empty"),
                model_id="test-model",
            )
        assert "article_text is empty" in str(exc_info.value)

    def test_seed_pool_too_small_raises(self):
        gen = _make_canned_generate_fn([])
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(
                gen,
                article_text="Article with starved seed pool.",
                seed_pool=_TEN_PRESETS[:3],
                rng=random.Random("tiny"),
                model_id="test-model",
            )
        assert "seed_pool too small" in str(exc_info.value)

    def test_same_seed_same_sample(self):
        # Sample is rng-driven; same rng seed -> same sample even
        # across separate pick_style calls (writer's C7 guarantee).
        inv = "\n".join(_five_valid_distinct())
        chosen = "vacuum_chamber_breach"
        gen_a = _make_canned_generate_fn([inv, chosen])
        gen_b = _make_canned_generate_fn([inv, chosen])
        pick_a = _SP.pick_style(
            gen_a, article_text="x", seed_pool=_TEN_PRESETS,
            rng=random.Random(1234), model_id="m",
        )
        pick_b = _SP.pick_style(
            gen_b, article_text="x", seed_pool=_TEN_PRESETS,
            rng=random.Random(1234), model_id="m",
        )
        assert pick_a.seed_sample == pick_b.seed_sample
        assert pick_a.article_hash == pick_b.article_hash
