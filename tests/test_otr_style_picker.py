"""tests/test_otr_style_picker.py -- unit tests for the two-pass
style picker (commit landing 2026-05-10).

Hermetic: no GPU, no actual LLM. generate_fn is mocked end-to-end.

Coverage map:
  - DESCRIPTOR_RE grammar acceptance/rejection
  - _sample_seeds determinism
  - _compute_article_hash stability
  - _parse_inventor_output: 5-line happy path; bullet-stripping;
    overgeneration -> first-5 distinct (B 2026-06-04); malformed /
    duplicate / near-duplicate lines skipped; too-few-after-dedupe
    -> raise; returned 5 always satisfy StylePick
  - _validate_chooser_output: exact match; bullet-stripped match;
    quoted match; non-candidate fail; empty fail
  - StylePick model: shape; chosen-must-match-grammar; candidates-
    grammar-and-distinctness validators
  - pick_style happy path: inventor produces 5, chooser picks one
  - pick_style inventor retry recovery
  - pick_style inventor all-fail -> StyleGenerationFailedError
  - pick_style chooser mismatch -> fallback to first candidate (BUG-LOCAL-295)
  - pick_style chooser model raise -> fallback to first candidate (BUG-LOCAL-295)
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
        # 3 valid distinct descriptors -> fewer than 5 survive -> raise.
        raw = "\n".join(_five_valid_distinct()[:3])
        with pytest.raises(ValueError, match="only 3 distinct"):
            _SP._parse_inventor_output(raw)

    # --- B (2026-06-04): tolerant of overgeneration --------------------
    # The parser keeps the StylePick(min=max=5) contract by taking the
    # first 5 distinct grammar-valid descriptors and skipping malformed /
    # duplicate / near-duplicate lines, instead of hard-failing. mistral's
    # clean ~5-line output is unaffected (first-5 == all 5); gemma-style
    # overgeneration becomes survivable.

    def test_overgeneration_takes_first_5(self):
        # 8 valid distinct descriptors -> first 5 in order, no raise.
        extra = [
            "bonus_alpha_channel",
            "bonus_beta_relay",
            "bonus_gamma_outpost",
        ]
        raw = "\n".join(_five_valid_distinct() + extra)
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_overgeneration_with_noise_takes_first_5_valid(self):
        # gemma-style: prose, headings and malformed tags interleaved with
        # valid descriptors. Malformed lines are skipped; the first 5 valid
        # distinct survive.
        raw = "\n".join([
            "Here are some style descriptors for the article:",
            "unknown_origin_signal_log",
            "1. Some commentary that is not a descriptor at all",
            "decommissioned_dish_archive",
            "noir-interrogation-chamber",          # hyphens -> invalid
            "midnight_newsroom_emergency",
            "TooManyWordsButNoUnderscoresHere",     # invalid
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
            "trailing_extra_descriptor_six",
        ])
        out = _SP._parse_inventor_output(raw)
        assert out == _five_valid_distinct()

    def test_exact_duplicate_collapsed_then_first_5(self):
        # A duplicate is skipped (shares all roots); with 6 lines where one
        # duplicates an earlier one, 5 distinct survive and are returned.
        base = _five_valid_distinct()
        raw = "\n".join(base[:3] + [base[0]] + base[3:])  # dup of base[0]
        out = _SP._parse_inventor_output(raw)
        assert out == base

    def test_exact_duplicate_dropping_below_5_raises(self):
        # 5 lines where one duplicates another -> only 4 distinct -> raise.
        candidates = _five_valid_distinct()
        candidates[2] = candidates[0]
        raw = "\n".join(candidates)
        with pytest.raises(ValueError, match="only 4 distinct"):
            _SP._parse_inventor_output(raw)

    def test_near_duplicate_skipped_then_first_5(self):
        # A near-duplicate (shares >1 root with an accepted descriptor) is
        # skipped, not fatal. 6 lines, one near-dup -> 5 distinct survive.
        raw = "\n".join([
            "noir_interrogation_chamber",
            "noir_interrogation_archive",   # shares 2 roots -> skipped
            "deep_space_distress_call",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
            "midnight_newsroom_emergency",
        ])
        out = _SP._parse_inventor_output(raw)
        assert out == [
            "noir_interrogation_chamber",
            "deep_space_distress_call",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
            "midnight_newsroom_emergency",
        ]

    def test_near_duplicate_dropping_below_5_raises(self):
        # Near-dup skip leaves only 4 distinct -> raise.
        raw = "\n".join([
            "noir_interrogation_chamber",
            "noir_interrogation_archive",   # shares 2 roots -> skipped
            "deep_space_distress_call",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
        ])
        with pytest.raises(ValueError, match="only 4 distinct"):
            _SP._parse_inventor_output(raw)

    def test_returned_five_satisfy_stylepick(self):
        # The 5 returned from an overgenerating draw must always satisfy the
        # strict StylePick.candidates validator (grammar + distinctness).
        raw = "\n".join(_five_valid_distinct() + ["bonus_alpha_channel"])
        out = _SP._parse_inventor_output(raw)
        kw = _valid_stylepick_kwargs()
        kw["candidates"] = out
        kw["chosen"] = out[0]
        pick = _SP.StylePick(**kw)
        assert pick.candidates == _five_valid_distinct()


# ---------------------------------------------------------------------------
# _parse_inventor_descriptors -- B telemetry counts
# ---------------------------------------------------------------------------


class TestInventorParseDescriptors:
    """B telemetry (2026-06-04): _parse_inventor_descriptors surfaces
    per-draw counts (raw / valid / distinct / truncated) alongside the
    5 chosen candidates so over/under-generation is observable."""

    def test_clean_five_counts(self):
        raw = "\n".join(_five_valid_distinct())
        p = _SP._parse_inventor_descriptors(raw)
        assert p.candidates == _five_valid_distinct()
        assert (p.raw_count, p.valid_count, p.distinct_count,
                p.truncated_count) == (5, 5, 5, 0)

    def test_overgeneration_counts(self):
        extra = ["bonus_alpha_channel", "bonus_beta_relay",
                 "bonus_gamma_outpost"]
        raw = "\n".join(_five_valid_distinct() + extra)
        p = _SP._parse_inventor_descriptors(raw)
        assert p.candidates == _five_valid_distinct()
        assert (p.raw_count, p.valid_count, p.distinct_count,
                p.truncated_count) == (8, 8, 8, 3)

    def test_noise_lines_counted_raw_not_valid(self):
        raw = "\n".join([
            "Here are some descriptors:",       # raw, not valid
            "unknown_origin_signal_log",
            "decommissioned_dish_archive",
            "noir-interrogation-chamber",        # raw, not valid (hyphens)
            "midnight_newsroom_emergency",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
        ])
        p = _SP._parse_inventor_descriptors(raw)
        assert p.candidates == _five_valid_distinct()
        assert (p.raw_count, p.valid_count, p.distinct_count,
                p.truncated_count) == (7, 5, 5, 0)

    def test_near_dup_valid_but_not_distinct(self):
        raw = "\n".join([
            "noir_interrogation_chamber",
            "noir_interrogation_archive",    # valid, near-dup -> skipped
            "deep_space_distress_call",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
            "midnight_newsroom_emergency",
        ])
        p = _SP._parse_inventor_descriptors(raw)
        assert (p.raw_count, p.valid_count, p.distinct_count,
                p.truncated_count) == (6, 6, 5, 0)

    def test_too_few_raises(self):
        raw = "\n".join(_five_valid_distinct()[:4])
        with pytest.raises(ValueError, match="only 4 distinct"):
            _SP._parse_inventor_descriptors(raw)

    def test_output_wrapper_matches_candidates(self):
        raw = "\n".join(_five_valid_distinct() + ["bonus_alpha_channel"])
        p = _SP._parse_inventor_descriptors(raw)
        assert _SP._parse_inventor_output(raw) == p.candidates


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

    def test_telemetry_fields_default(self):
        # B telemetry fields are optional with safe defaults so existing
        # records and direct constructions stay valid.
        pick = _SP.StylePick(**_valid_stylepick_kwargs())
        assert pick.valid_count == 0
        assert pick.distinct_count == 0
        assert pick.truncated_count == 0
        assert pick.model_slug == ""

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
        pick = _SP.pick_style(creative_fn=gen, technical_fn=gen,
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

    def test_stamps_telemetry_counts_and_slug(self):
        # Overgenerating draw (5 clean + 1 extra) -> first 5 chosen;
        # truncated_count=1; model_slug passed through to StylePick.
        inv = "\n".join(_five_valid_distinct() + ["bonus_alpha_channel"])
        gen = _make_canned_generate_fn([inv, "vacuum_chamber_breach"])
        pick = _SP.pick_style(
            creative_fn=gen, technical_fn=gen,
            article_text="A telemetry article.",
            seed_pool=_TEN_PRESETS,
            rng=random.Random("telemetry"),
            model_id="openrouter:slot-a",
            model_slug="hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M",
        )
        assert pick.candidates == _five_valid_distinct()
        assert pick.valid_count == 6
        assert pick.distinct_count == 6
        assert pick.truncated_count == 1
        assert pick.model_id == "openrouter:slot-a"
        assert pick.model_slug == "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"

    def test_telemetry_clean_draw_defaults_slug_empty(self):
        # Clean 5-line draw: no truncation; model_slug defaults to "" when
        # the caller doesn't pass one (the writer supplies it in prod).
        inv = "\n".join(_five_valid_distinct())
        gen = _make_canned_generate_fn([inv, "vacuum_chamber_breach"])
        pick = _SP.pick_style(
            creative_fn=gen, technical_fn=gen,
            article_text="A clean article.",
            seed_pool=_TEN_PRESETS,
            rng=random.Random("clean-telemetry"),
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
        )
        assert pick.valid_count == 5
        assert pick.distinct_count == 5
        assert pick.truncated_count == 0
        assert pick.model_slug == ""

    def test_inventor_retry_recovers(self):
        # First attempt invalid grammar; second attempt valid.
        bad = "\n".join(["BadCaps", "noir_interrogation", "deep_space_distress",
                         "vacuum_chamber_breach", "haunted_repeater_loop"])
        good = "\n".join(_five_valid_distinct())
        chosen = "midnight_newsroom_emergency"
        gen = _make_canned_generate_fn([bad, good, chosen])
        pick = _SP.pick_style(creative_fn=gen, technical_fn=gen,
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
            _SP.pick_style(creative_fn=gen, technical_fn=gen,
                article_text="Article that breaks the inventor.",
                seed_pool=_TEN_PRESETS,
                rng=random.Random("allfail"),
                model_id="test-model",
            )
        assert "after 3 attempts" in str(exc_info.value)

    def test_chooser_mismatch_falls_back(self):
        # BUG-LOCAL-295: a chooser pick outside the candidate pool must
        # NOT halt the episode. After exhausting its retry budget the
        # picker falls back to the first candidate rather than raising.
        inventor_response = "\n".join(_five_valid_distinct())
        rogue = "rogue_descriptor_outside_pool"
        gen = _make_canned_generate_fn(
            [inventor_response, rogue, rogue, rogue]
        )
        pick = _SP.pick_style(creative_fn=gen, technical_fn=gen,
            article_text="Article whose chooser goes rogue.",
            seed_pool=_TEN_PRESETS,
            rng=random.Random("chooser-rogue"),
            model_id="test-model",
        )
        assert pick.chosen == _five_valid_distinct()[0]
        assert pick.candidates == _five_valid_distinct()

    def test_chooser_llm_raise_falls_back(self):
        # BUG-LOCAL-295: if the chooser model raises on every attempt,
        # fall back to the first candidate rather than crashing the run.
        inventor_response = "\n".join(_five_valid_distinct())

        def _gen(messages, *, temperature, max_new_tokens):
            # inventor temperatures are high; chooser temp is low.
            if temperature >= 0.5:
                return inventor_response
            raise RuntimeError("chooser model offline")

        pick = _SP.pick_style(creative_fn=_gen, technical_fn=_gen,
            article_text="Article whose chooser model dies.",
            seed_pool=_TEN_PRESETS,
            rng=random.Random("chooser-offline"),
            model_id="test-model",
        )
        assert pick.chosen == _five_valid_distinct()[0]

    def test_empty_article_raises_precondition(self):
        gen = _make_canned_generate_fn([])
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(creative_fn=gen, technical_fn=gen,
                article_text="",
                seed_pool=_TEN_PRESETS,
                rng=random.Random("empty"),
                model_id="test-model",
            )
        assert "article_text is empty" in str(exc_info.value)

    def test_seed_pool_too_small_raises(self):
        gen = _make_canned_generate_fn([])
        with pytest.raises(_SP.StyleGenerationFailedError) as exc_info:
            _SP.pick_style(creative_fn=gen, technical_fn=gen,
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
        pick_a = _SP.pick_style(creative_fn=gen_a, technical_fn=gen_a, article_text="x", seed_pool=_TEN_PRESETS,
            rng=random.Random(1234), model_id="m",
        )
        pick_b = _SP.pick_style(creative_fn=gen_b, technical_fn=gen_b, article_text="x", seed_pool=_TEN_PRESETS,
            rng=random.Random(1234), model_id="m",
        )
        assert pick_a.seed_sample == pick_b.seed_sample
        assert pick_a.article_hash == pick_b.article_hash


# ---------------------------------------------------------------------------
# A (2026-06-04): inventor GBNF decode grammar + threading
# ---------------------------------------------------------------------------


class TestInventorGBNF:
    """The inventor decode grammar constrains a grammar-capable backend to
    exactly _REQUIRED_CANDIDATE_COUNT snake_case descriptor lines."""

    def test_root_chains_exactly_n_lines(self):
        g = _SP._build_inventor_gbnf()
        root = next(ln for ln in g.splitlines() if ln.startswith("root ::="))
        # N 'line' refs separated by N-1 newline literals.
        assert root.count("line") == _SP._REQUIRED_CANDIDATE_COUNT
        assert root.count('"\\n"') == _SP._REQUIRED_CANDIDATE_COUNT - 1

    def test_line_rule_is_two_to_five_snake_words(self):
        g = _SP._build_inventor_gbnf()
        assert 'line ::= word "_" word ("_" word)? ("_" word)? ("_" word)?' in g
        assert "word ::= [a-z]+" in g

    def test_module_constant_built_once(self):
        assert _SP._INVENTOR_GBNF == _SP._build_inventor_gbnf()
        assert _SP._INVENTOR_GBNF.startswith("root ::=")


class TestInventorGrammarThreading:
    """A: the inventor passes its GBNF grammar ONLY to backends that
    advertise grammar support (the remote llama-server lane). A plain
    local/test backend is never handed an unexpected `grammar` kwarg, so
    mistral's path stays byte-identical."""

    def test_grammar_passed_when_capable_and_enabled(self, monkeypatch):
        monkeypatch.setenv("OTR_ENABLE_INVENTOR_GBNF", "1")
        seen = {}

        def cap_creative(messages, *, temperature, max_new_tokens, grammar=None):
            seen["grammar"] = grammar
            return "\n".join(_five_valid_distinct())
        cap_creative._otr_supports_grammar = True  # remote-lane marker

        def technical(messages, *, temperature, max_new_tokens, grammar=None):
            return "vacuum_chamber_breach"

        pick = _SP.pick_style(
            creative_fn=cap_creative, technical_fn=technical,
            article_text="A grammar-capable backend article.",
            seed_pool=_TEN_PRESETS, rng=random.Random("gbnf-cap"),
            model_id="openrouter:slot-a",
        )
        # The inventor (creative) call received the exactly-N grammar.
        assert seen["grammar"] == _SP._INVENTOR_GBNF
        assert pick.chosen == "vacuum_chamber_breach"

    def test_grammar_not_passed_when_flag_off(self, monkeypatch):
        # Default-off: even a grammar-capable backend gets NO grammar unless
        # OTR_ENABLE_INVENTOR_GBNF is set. B remains the safety net, and a
        # non-grammar /v1 (Ollama) is never handed an unknown field.
        monkeypatch.delenv("OTR_ENABLE_INVENTOR_GBNF", raising=False)
        seen = {"grammar": "SENTINEL"}

        def cap_creative(messages, *, temperature, max_new_tokens, grammar=None):
            seen["grammar"] = grammar
            return "\n".join(_five_valid_distinct())
        cap_creative._otr_supports_grammar = True

        def technical(messages, *, temperature, max_new_tokens, grammar=None):
            return "vacuum_chamber_breach"

        _SP.pick_style(
            creative_fn=cap_creative, technical_fn=technical,
            article_text="Grammar-capable but flag off.",
            seed_pool=_TEN_PRESETS, rng=random.Random("gbnf-off"),
            model_id="openrouter:slot-a",
        )
        assert seen["grammar"] is None  # not attached

    def test_grammar_not_passed_to_plain_backend(self):
        # A plain generate_fn with NO grammar kwarg: if the picker passed
        # grammar it would TypeError. The absent marker must suppress it.
        def plain(messages, *, temperature, max_new_tokens):
            if temperature >= 0.5:           # inventor (high temp)
                return "\n".join(_five_valid_distinct())
            return "vacuum_chamber_breach"   # chooser (low temp)

        pick = _SP.pick_style(
            creative_fn=plain, technical_fn=plain,
            article_text="A plain local backend article.",
            seed_pool=_TEN_PRESETS, rng=random.Random("gbnf-plain"),
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
        )
        assert pick.chosen in _five_valid_distinct()
