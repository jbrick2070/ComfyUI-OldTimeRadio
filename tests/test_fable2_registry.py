"""scifi_fable2 registry surfaces -- S0 inert pins (architecture doc s11/s13).

Covers: bank row (inserted BEFORE custom_source_bank; rss-payload-not-spark),
pipeline row (registry-legal slots only; executable flips at S1b), sidecar
registration, frame-deck lint (SFW deck law), detection-only story rules,
and the science_news untouched pin. Pure Python; no GPU, no LLM.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as ROUTING  # noqa: E402
from nodes import _otr_story_rules as RULES  # noqa: E402
from nodes import OTR_LedgerScriptWriter as W  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_PACK_DIR = _REPO / "nodes" / "story_packs" / "scifi_news_pro"

_FABLE2_SEAMS = frozenset({
    "fable2_dossier_system", "fable2_pitch_system", "fable2_select_system",
    "fable2_treatment_system", "fable2_news_read_system",
    "fable2_script_system", "fable2_critic_system",
    "fable2_revision_system", "fable2_casting_system",
})


@pytest.fixture(autouse=True)
def _fresh_registries():
    ROUTING._REGISTRY = None
    RULES._clear_caches()
    yield
    ROUTING._REGISTRY = None
    RULES._clear_caches()


# ---------------------------------------------------------------------------
# 1. Bank row
# ---------------------------------------------------------------------------

class TestBankRow:
    def test_registered_directly_before_custom_which_stays_last(self):
        # r3/M7: "+ Add Your Own" stays last; new lanes inserted BEFORE it.
        ids = ROUTING.list_bank_ids()
        assert ids[-1] == "custom_source_bank"
        # v4 campaign 2026-07-17: scifi_codex_v4 inserted directly before custom,
        # after the _v3 tail; custom stays last. Sonnet-bake-off rip 2026-07-18:
        # shakespeare_v3 is now the last _v3 lane before scifi_codex_v4.
        assert ids[-3:] == ("shakespeare_v3", "scifi_news",
                            "custom_source_bank")

    def test_row_shape_runnable_s1b(self):
        bank = ROUTING.get_bank("scifi_news_pro")
        assert bank.runnable is True                # S1b: runner shipped
        assert bank.fetcher == "science_rss"
        assert bank.interpreter == ""
        assert bank.default_story_model == "scifi_news_pro"
        assert bank.default_story_pipeline == "scifi_news_pro_multipass"
        assert bank.required_seams == ()
        assert "No fallback to legacy_many_pass" in bank.guide_ref

    def test_runnable_defaults_contract(self):
        # r3/M6: title_form_label present ahead of the runnable flip.
        d = ROUTING.get_bank("scifi_news_pro").defaults
        assert d["title_form_label"] == "science-fiction radio drama"
        assert d["coda_mode"] == "real_news_report"
        assert "machine" in d["credits_source_line"]

    def test_fable2_receives_rss_payload_not_spark(self):
        # r1/C3 pin: fetcher set + interpreter empty means the writer's
        # original_radio spark branch (empty fetcher AND interpreter) can
        # NEVER fire for this bank -- the shared fetch lane hands the runner
        # a validated RSS payload.
        assert W._bank_has_no_source_contract(
            ROUTING.get_bank("scifi_news_pro")) is False

    def test_run_gate_admits_runnable_bank(self):
        # S1b flip: the runnable gate now admits the bank (the runner
        # ships in the SAME change -- sweep rule b).
        bank = ROUTING.require_runnable_bank("scifi_news_pro")
        assert bank.source_bank_id == "scifi_news_pro"


# ---------------------------------------------------------------------------
# 2. Pipeline row
# ---------------------------------------------------------------------------

class TestPipelineRow:
    def test_row_shape_executable_s1b(self):
        pipe = ROUTING.get_pipeline("scifi_news_pro_multipass")
        assert pipe.executable is True              # S1b: runner shipped
        assert pipe.requires_source_contract is False
        assert pipe.declared_seams == _FABLE2_SEAMS
        assert any("NEVER routes to legacy_many_pass" in n
                   for n in pipe.notes)

    def test_pass_graph_order_and_slots(self):
        # r4/M2: registry-legal slots ONLY (creative | technical); the
        # judge/creative subtype language lives in descriptions.
        pipe = ROUTING.get_pipeline("scifi_news_pro_multipass")
        rows = [(p.pass_id, p.slot) for p in pipe.passes]
        assert rows == [
            ("dossier", "technical"),
            ("pitch_room", "creative"),
            ("pitch_select", "technical"),
            ("treatment", "creative"),
            ("news_read", "technical"),   # P2c, S1b read-split
            ("script", "creative"),
            ("critic", "technical"),
            ("revision", "creative"),
            ("casting_voices", "technical"),
            ("assemble", "technical"),
            ("safety_scan", "technical"),
        ]
        assert {slot for _pid, slot in rows} <= ROUTING._VALID_PASS_SLOTS

    def test_assemble_pass_is_marked_pure_python_metadata(self):
        pipe = ROUTING.get_pipeline("scifi_news_pro_multipass")
        p7 = next(p for p in pipe.passes if p.pass_id == "assemble")
        assert p7.seam_refs == ()
        assert "pure Python" in p7.description

    def test_p8_is_a_python_safety_scan_with_no_seam(self):
        """The LLM ledger audit is gone: no seam, no model, no veto."""
        pipe = ROUTING.get_pipeline("scifi_news_pro_multipass")
        p8 = next(p for p in pipe.passes if p.pass_id == "safety_scan")
        assert p8.seam_refs == ()
        assert "pure Python" in p8.description
        assert "fable2_audit_system" not in pipe.declared_seams

    def test_slot_enum_rejection(self):
        # A non-registry slot (e.g. "judge") must die at parse time.
        raw = {
            "story_pipeline_id": "bad_pipe", "label": "Bad",
            "executable": False, "requires_source_contract": False,
            "declared_seams": [],
            "passes": [{"pass_id": "p", "slot": "judge", "seam_refs": [],
                        "description": "not a legal slot"}],
        }
        with pytest.raises(ROUTING.RegistryValidationError, match="slot"):
            ROUTING._parse_pipeline(raw, "test fixture")


# ---------------------------------------------------------------------------
# 3. Pack + sidecar
# ---------------------------------------------------------------------------

class TestPackAndSidecar:
    def test_pack_resolves_with_all_nine_seams(self):
        pack = ROUTING.resolve_story_pack("scifi_news_pro")
        assert pack.story_pipeline_id == "scifi_news_pro_multipass"
        assert set(pack.prompt_stages) == _FABLE2_SEAMS
        for seam, text in pack.prompt_stages.items():
            assert text.strip(), f"seam {seam} is empty"

    def test_sidecar_registered(self):
        assert ROUTING._PACK_SIDECAR_FILENAMES_BY_BANK["scifi_news_pro"] == \
            frozenset({"frame_deck.json"})

    def test_pack_dir_carries_exactly_the_two_files(self):
        names = sorted(p.name for p in _PACK_DIR.glob("*.json"))
        assert names == ["frame_deck.json", "scifi_news_pro.json"]


# ---------------------------------------------------------------------------
# 4. Frame deck lint (SFW deck law; doc s13 S0)
# ---------------------------------------------------------------------------

# Weapons / horror / smoking vocabulary the deck (cards AND stances) must
# never carry. Word-boundary regex so e.g. "politeness" never false-hits.
_DECK_BANNED = re.compile(
    r"\b(gun|guns|knife|knives|rifle|pistol|blade|bomb|weapon|weapons|"
    r"kill|kills|murder|corpse|blood|horror|terror|haunted|haunting|"
    r"demon|ghost|zombie|monster|cigarette|tobacco|smoke|smoking|"
    r"war|battle|soldier)\b",
    re.IGNORECASE)


class TestFrameDeckLint:
    def _deck(self):
        return json.loads(
            (_PACK_DIR / "frame_deck.json").read_text(encoding="utf-8"))

    def test_deck_shape(self):
        deck = self._deck()
        assert deck["schema_version"] == "fable2_deck_v1"
        cards = deck["cards"]
        # 12 authored cards is the shippable S0 minimum (doc s14.10); the
        # P1 deal needs >= 3 distinct.
        assert len(cards) >= 12
        names = [c["name"] for c in cards]
        assert len(set(names)) == len(names), "card names must be unique"
        for card in cards:
            assert card["name"].strip() and card["shape"].strip()

    def test_stances_live_in_the_deck(self):
        # r2 anchor: stances are JSON content in frame_deck.json under the
        # "stances" key -- JSON owns content.
        deck = self._deck()
        stances = deck["stances"]
        assert len(stances) >= 3
        names = [s["name"] for s in stances]
        assert len(set(names)) == len(names)
        for stance in stances:
            assert stance["name"].strip() and stance["note"].strip()

    def test_deck_is_sfw(self):
        deck = self._deck()
        corpus = []
        corpus += [f"card {c['name']}: {c['shape']}" for c in deck["cards"]]
        corpus += [f"stance {s['name']}: {s['note']}"
                   for s in deck["stances"]]
        for text in corpus:
            hit = _DECK_BANNED.search(text)
            assert hit is None, f"banned term {hit.group(0)!r} in: {text}"


# ---------------------------------------------------------------------------
# 5. Story rules: detection-only
# ---------------------------------------------------------------------------

class TestStoryRules:
    def test_rules_resolve(self):
        rules = RULES.resolve_story_rules("scifi_news_pro")
        assert rules.rules_id == "scifi_news_pro"
        assert rules.cliche and rules.on_the_nose and rules.banned_thesis

    def test_detection_only_no_replacement_tables(self):
        # Operator law (LLM-first): Python judges, the LLM writes -- this
        # lane carries NO python rewriting vocabulary, ever.
        rules = RULES.resolve_story_rules("scifi_news_pro")
        assert rules.cliche_replacements == ()


# ---------------------------------------------------------------------------
# 6. Roster order (roster-trim 2026-07-17: 10 independent lanes + custom)
# ---------------------------------------------------------------------------

class TestRosterOrder:
    """Roster-trim 2026-07-17: the science_news family, all _v2 rows, and the
    orphan bases (public_domain_story/shakespeare/original_radio_v3) were ripped.
    Each surviving lane is an INDEPENDENT bank (own pack + story_rules, no family
    base-map). custom stays last. v4 campaign 2026-07-17: scifi_codex_v4 added as
    the first independent _v4 lane, before custom. Sonnet-bake-off rip 2026-07-18:
    four bake-off _v3 variants were retired, leaving two inline _v3 lanes."""

    def test_roster_order_holds(self):
        ids = ROUTING.list_bank_ids()
        assert ids == ("media_archive", "original_radio",
                       "scifi_news_pro",
                       "public_domain_story_v3", "shakespeare_v3",
                       "scifi_news",
                       "custom_source_bank")

    def test_fable2_seams_never_shadow_production(self):
        # The loader enforces this globally; pin it for the fable2 set
        # explicitly (a production-seam collision would re-route science).
        from nodes._otr_story_pack import PRODUCTION_SEAM_ALLOWLIST
        assert not (_FABLE2_SEAMS & PRODUCTION_SEAM_ALLOWLIST)

    def test_legacy_pipeline_row_untouched(self):
        pipe = ROUTING.get_pipeline("legacy_many_pass")
        assert pipe.executable is False
        assert pipe.requires_source_contract is True
        assert pipe.declared_seams == frozenset()
