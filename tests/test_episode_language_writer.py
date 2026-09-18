"""The one switch on the writer: widget, fidelity gate, stamp, replay, authoring.

Covers the writer-side half of the multilingual one-switch
(``docs/2026-09-18-multilingual-notebooklm/coder_prompt_all_kokoro_day1.md``
items 2, 6 and 10). The registry itself is pinned by
``tests/test_episode_languages.py``.

THE ENGLISH REGRESSION GATE runs through every case here: an English or ``Off``
run must produce the same prompts, the same stamp absence and the same title
pass it produced before this feature existed.
"""
from __future__ import annotations

import pytest

from nodes import _otr_episode_languages as el
from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
from nodes.OTR_LedgerScriptWriter import _native_authoring_instruction
from nodes._otr_writer_tail import (
    _generate_title_from_script,
    _title_language_instruction,
)

NON_ENGLISH = ["Spanish", "Portuguese", "Italian", "French",
               "Hindi", "Japanese", "Mandarin"]
FIDELITY_BANKS = ("shakespeare", "public_domain")


# --------------------------------------------------------------------------- #
# the widget -- one control, trailing slot, admitted rows only
# --------------------------------------------------------------------------- #


def test_the_widget_is_declared_after_replay_from_and_before_gate_in():
    """BUG-LOCAL-097: gate_in is a forceInput socket and holds no saved slot,
    so declaring episode_language between them makes it the TRAILING value."""
    spec = W.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())
    assert order.index("replay_from") + 1 == order.index("episode_language")
    assert order.index("episode_language") + 1 == order.index("gate_in")
    assert order[-1] == "gate_in"


def test_the_widget_offers_off_plus_every_admitted_row():
    choices, meta = W.INPUT_TYPES()["optional"]["episode_language"]
    assert list(choices) == el.dropdown_choices()
    assert choices[0] == el.OFF_LABEL
    assert len(choices) == 9
    # A COMBO whose default is out-of-list is a load-time hazard.
    assert meta["default"] == "English" and meta["default"] in choices


def test_the_tooltip_says_what_stays_english():
    """The split is the product. A reader must not have to guess it."""
    _choices, meta = W.INPUT_TYPES()["optional"]["episode_language"]
    tip = meta["tooltip"]
    for phrase in ("music", "Off is not a language", "Kokoro", "as you typed"):
        assert phrase in tip, phrase


def test_the_widget_is_on_the_creative_whitelist_in_both_mirrors():
    """A headless leg sets a language; apply_profile never manages it."""
    import importlib.util
    from pathlib import Path
    from nodes import _otr_workflow_apply as WA
    assert "episode_language" in WA.CREATIVE_WHITELIST
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "otr_api_language_test", repo / "scripts" / "otr_api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    assert "episode_language" in api.CREATIVE_WHITELIST


# --------------------------------------------------------------------------- #
# the fidelity gate -- before any LLM call
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", NON_ENGLISH)
@pytest.mark.parametrize("bank", FIDELITY_BANKS)
def test_a_fidelity_bank_refuses_on_every_non_english_row(label, bank):
    row = el.row_by_label(label)
    with pytest.raises(el.EpisodeLanguageError) as caught:
        el.check_source_bank_admission(row, bank)
    message = str(caught.value)
    assert bank in message and label in message
    # The refusal must say what to do instead, not merely that it refused.
    assert "English" in message


@pytest.mark.parametrize("bank", FIDELITY_BANKS)
def test_english_admits_the_fidelity_banks(bank):
    el.check_source_bank_admission(el.row_by_label("English"), bank)


@pytest.mark.parametrize("bank", FIDELITY_BANKS + ("media_archive", "original"))
def test_off_admits_every_bank(bank):
    """Off is off: today's path, byte for byte."""
    el.check_source_bank_admission(None, bank)


@pytest.mark.parametrize("label", NON_ENGLISH)
@pytest.mark.parametrize("bank", ("media_archive", "original", "my_story",
                                  "scifi_news_pro"))
def test_the_generative_banks_are_admitted_on_every_row(label, bank):
    el.check_source_bank_admission(el.row_by_label(label), bank)


def test_a_blank_bank_id_is_not_a_refusal():
    """The roll sentinel resolves to a real id before this gate; a blank is
    never turned into a fail that names no bank."""
    el.check_source_bank_admission(el.row_by_label("Spanish"), "")
    el.check_source_bank_admission(el.row_by_label("Spanish"), None)


# --------------------------------------------------------------------------- #
# the stamp -- beside source_bank, before the freeze
# --------------------------------------------------------------------------- #


def test_off_stamps_nothing_at_all():
    """Not a null, not "off" -- no key. An absent stamp is how a reader tells
    "asked for today's path" from "chose English"."""
    assert el.resolve_ledger(el.OFF_LABEL) is None


@pytest.mark.parametrize("label", ["English"] + NON_ENGLISH)
def test_the_stamp_carries_iso_header_and_a_verifiable_receipt(label):
    stamp = el.resolve_ledger(label)
    assert set(stamp) == {"episode_language", "language_header",
                          "episode_language_receipt"}
    row = el.row_by_label(label)
    assert stamp["episode_language"] == row.iso
    assert stamp["language_header"] == row.native_header
    receipt = stamp["episode_language_receipt"]
    assert receipt["registry_id"] == "episode_languages"
    assert receipt["row_revision"] == row.row_revision
    assert len(receipt["row_sha256"]) == 64


def test_an_unstamped_ledger_paints_english():
    """Off, a legacy graph and a pre-feature frozen ledger all land here."""
    for meta in ({}, {"source_bank": "original"},
                 {"episode_language": ""}, None):
        assert el.iso_from_meta(meta) == "en"
        assert el.row_from_meta(meta).label == "English"


def test_a_ledger_naming_an_unknown_row_fails_closed():
    with pytest.raises(el.EpisodeLanguageError, match="no row in this pack"):
        el.iso_from_meta({"episode_language": "tlh"})


# --------------------------------------------------------------------------- #
# replay -- the ledger's iso wins, drift is named
# --------------------------------------------------------------------------- #


def _stamped(label):
    meta = {"source_bank": "original"}
    meta.update(el.resolve_ledger(label))
    return meta


@pytest.mark.parametrize("label", ["English"] + NON_ENGLISH)
def test_a_replay_agreeing_with_its_ledger_passes(label):
    check = el.replay_language_check(_stamped(label), el.resolve_label(label))
    assert check.iso == el.row_by_label(label).iso
    assert check.row_drift is None


def test_a_widget_naming_another_language_fails_the_replay():
    with pytest.raises(el.EpisodeLanguageError) as caught:
        el.replay_language_check(_stamped("Spanish"),
                                 el.resolve_label("Japanese"))
    message = str(caught.value)
    assert "Spanish" in message and "Japanese" in message
    # It must name the two ways out, not just the collision.
    assert el.OFF_LABEL in message


def test_off_never_compares_on_a_replay():
    """A replay under Off re-renders whatever the bundle recorded."""
    check = el.replay_language_check(_stamped("Hindi"),
                                     el.resolve_label(el.OFF_LABEL))
    assert check.iso == "hi" and check.row.label == "Hindi"


def test_a_legacy_unstamped_bundle_replays_under_english_and_refuses_spanish():
    legacy = {"source_bank": "original"}
    assert el.replay_language_check(legacy, el.resolve_label("English")).iso == "en"
    assert el.replay_language_check(legacy, el.resolve_label("")).iso == "en"
    with pytest.raises(el.EpisodeLanguageError, match="replay language mismatch"):
        el.replay_language_check(legacy, el.resolve_label("Spanish"))


def test_row_drift_warns_with_both_revisions_and_continues():
    """Warn, never silently claim identity. Missing row already failed above."""
    meta = _stamped("Spanish")
    meta["episode_language_receipt"] = dict(
        meta["episode_language_receipt"],
        row_revision=1, row_sha256="0" * 64,
    )
    check = el.replay_language_check(meta, el.resolve_label("Spanish"))
    drift = check.row_drift
    assert drift is not None
    assert drift["iso"] == "es" and drift["label"] == "Spanish"
    assert drift["recorded_row_revision"] == 1
    assert drift["current_row_revision"] == el.row_by_label("Spanish").row_revision
    assert drift["recorded_row_sha256"] == "0" * 64
    assert len(drift["current_row_sha256"]) == 64
    assert "current row" in drift["resolution"]


def test_an_identical_row_reports_no_drift():
    assert el.replay_language_check(
        _stamped("Mandarin"), el.resolve_label("Mandarin")).row_drift is None


def test_the_writer_calls_the_replay_check_before_the_cast_block():
    """A test that calls the helper proves the helper, never the wiring."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nodes"
           / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert "_EPLANG.replay_language_check(meta, _language)" in src
    assert src.index("_EPLANG.replay_language_check") < src.index("lock_cast(")
    # The fidelity gate must precede the LLM preflight and _resolve_inputs.
    assert src.index("_EPLANG.check_source_bank_admission") < src.index(
        "_resolve_inputs(")


# --------------------------------------------------------------------------- #
# authored, never translated
# --------------------------------------------------------------------------- #


def test_english_authoring_instruction_is_empty_so_prompts_stay_identical():
    """The English regression gate. Telling the model "write in English" would
    change every English prompt in the pack to say what it already assumed."""
    for meta in ({}, _stamped("English")):
        assert _native_authoring_instruction(meta) == ""
        assert _title_language_instruction(meta) == ""


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_a_non_english_row_supplies_its_own_native_instruction(label):
    meta = _stamped(label)
    instruction = _native_authoring_instruction(meta)
    assert instruction == el.row_by_label(label).authoring["writer_instruction"]
    assert instruction.strip()


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_the_title_rule_joins_the_native_voice_to_the_mechanical_rule(label):
    row = el.row_by_label(label)
    rule = _title_language_instruction(_stamped(label))
    assert row.authoring["writer_instruction"] in rule
    assert row.authoring["title_instruction"] in rule


def test_the_title_rule_degrades_instead_of_costing_an_episode_its_title():
    """A registry failure must never fail an episode over a LABEL."""
    assert _title_language_instruction({"episode_language": "tlh"}) == ""


def test_no_instruction_asks_the_model_to_translate():
    """Author native. There is no English draft behind any of these."""
    for row in el.reload_registry()[0]:
        for key in ("writer_instruction", "title_instruction"):
            text = row.authoring[key].lower()
            assert "translate this" not in text
            assert "translation of" not in text


# --------------------------------------------------------------------------- #
# the title pass carries the language, and English is byte-identical
# --------------------------------------------------------------------------- #


class _PromptCapture:
    def __init__(self):
        self.messages = None

    def __call__(self, messages, *, temperature, max_new_tokens, stop=None):
        self.messages = messages
        return "DETAILS: a\nCANDIDATES: b\nTITLE: Una Noche Larga"


SCRIPT = "ANNOUNCER: Buenas noches.\nADA: El faro no responde.\nTOM: Entonces vamos."


def _title_prompt(language_instruction=""):
    capture = _PromptCapture()
    _generate_title_from_script(
        capture, SCRIPT, premise="un faro calla",
        language_instruction=language_instruction,
    )
    return capture.messages


def test_the_english_title_prompt_is_byte_identical_to_the_pre_feature_prompt():
    assert _title_prompt("") == _title_prompt()


def test_a_native_title_rule_reaches_both_prompt_halves():
    rule = _title_language_instruction(_stamped("Spanish"))
    messages = _title_prompt(rule)
    system, user = messages[0]["content"], messages[1]["content"]
    assert rule in system
    assert rule in user
    # The scratchpad runs in the language too -- English candidates would put
    # the model one translation away from its own final line.
    assert "do not draft in English and translate" in user


def test_the_authored_title_survives_accents():
    """Any ASCII-strip on the title path is a defect."""
    capture = _PromptCapture()
    title = _generate_title_from_script(
        capture, SCRIPT, language_instruction="Escribe en español.")
    assert title == "Una Noche Larga"


# --------------------------------------------------------------------------- #
# the outline stages carry it, and English stays byte-identical
# --------------------------------------------------------------------------- #


def test_generate_outline_accepts_a_language_instruction_defaulting_empty():
    import inspect
    from nodes import _otr_outline as O
    sig = inspect.signature(O.generate_outline)
    assert sig.parameters["language_instruction"].default == ""


def test_the_writer_hands_the_outline_the_ledgers_language():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nodes"
           / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert "language_instruction=_native_authoring_instruction(meta)" in src
    # And the composition header -- the one block every compose pass reads.
    assert "canon_header = _language_prompt_lead" in src
