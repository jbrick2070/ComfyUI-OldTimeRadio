"""PBUG-20260710-07 -- no silent cast-keyed role mutation (named regression pin).

PBUG-20260710-07 (scifi_fable2 30w live smoke roll 22, 2026-07-10): a postamble
ANNOUNCER row arrived ``speaker_role="character"`` with ``skip=True`` and a NULL
reason after a green 8-pass spine -- flipped by an UNIDENTIFIED cast-keyed
mutator with NO compose-flag breadcrumb (silent data corruption; PROMOTED
BUG-11.32, ROOT CAUSE OPEN in PROD_BUG_LOG).

Phase 0 of the v4 improvement campaign (2026-07-17) root-caused the class
STATICALLY: the announcer<->char_id ambiguity is resolved by the D3 pre-freeze
coerce sweep (``_otr_freeze_cascade.run_freeze_cascade`` ->
``production_ledger.coerce_speaker_role_for_char_id``, keyed on
``cast_ids_from_ledger``). It is already closed by three cooperating guards:
  (1) the announcer SENTINEL char_id + NAME-EXCLUSION keep a legitimate announcer
      line OUT of ``cast_ids``, so it is never coerced;
  (2) every coercion stamps a ``role_coerce:...reason=...`` breadcrumb on the
      row's ``compose_flags``;
  (3) ``meta["role_coercions"]`` records the count + line_ids audit.

This file PINS the two durable invariants so a future refactor cannot silently
reintroduce a blind cast-keyed flip (LESSONS_GATE_BRIEF section 4 item 2 /
PRODUCTION_SPRINT_LESSONS lesson 24, the "cast-keyed mutation must stamp a
reason" rule):

  INVARIANT A -- NO SILENT MUTATION: every cast-keyed role coercion carries a
    ``role_coerce:...reason=...`` breadcrumb AND a ``meta["role_coercions"]``
    audit. (The exact thing PBUG-20260710-07's symptom lacked: "reason null ...
    no compose-flag breadcrumb".)
  INVARIANT B -- ANNOUNCER-SENTINEL PROTECTION: a line carrying the announcer
    sentinel char_id (or a name=ANNOUNCER cast slot) is NEVER coerced to
    "character".

REGRESSION PIN ONLY -- no coerce machinery is added here. The root fix shipped
pre-campaign; adding more would be a shim (operator directive). The PBUG stays
ROOT-OPEN in PROD_BUG_LOG until a live v4 leg formally retires it via the
per-lane announcer-sentinel MINTING invariant (v4 campaign Phase 2). Complements
tests/test_d3_role_coercion.py (which pins the D3 mechanism); this file pins the
PBUG-20260710-07 CONTRACT. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import production_ledger as PL  # noqa: E402
from nodes import _otr_freeze_cascade as _LFC_ORCH  # noqa: E402


_CAST = [
    {"char_id": "c01", "name": "MALI"},
    {"char_id": "c02", "name": "MANFRED"},
    {"char_id": "announcer", "name": "ANNOUNCER"},
]


def _ledger_obj(data):
    obj = SimpleNamespace(data=data, episode_id=data.get("episode_id", "ep"))
    obj.save = lambda: None  # type: ignore[attr-defined]
    return obj


def _cascade_ledger_data():
    from nodes import _otr_ledger_freeze as _LFC
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_pbug_20260710_07",
        "cast": [
            {"char_id": "c01", "name": "MALI", "traits": "calm",
             "voice_preset": "v2/en_speaker_6"},
            {"char_id": "c02", "name": "MANFRED", "traits": "smug",
             "voice_preset": "v2/en_speaker_2"},
            {"char_id": "announcer", "name": "ANNOUNCER", "traits": "warm",
             "voice_preset": "v2/en_speaker_9"},
        ],
        "scenes": [], "shots": [],
        "beats": [{"beat_id": f"b00{i}"} for i in range(1, 5)],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "speaker_role": "announcer",
             "char_id": "announcer", "text": "Gather round, dear listeners.",
             "word_count": 4, "char_count": 28, "skip": False},
            {"line_id": "b002", "beat_id": "b002", "speaker_role": "character",
             "char_id": "c01", "text": "I have isolated the frequency at last.",
             "word_count": 7, "char_count": 37, "skip": False},
            # b011 culprit: cast char_id c02 but role announcer (must coerce back)
            {"line_id": "b011", "beat_id": "b003", "speaker_role": "announcer",
             "char_id": "c02", "text": "I sent a copy to The Chronicle.",
             "word_count": 7, "char_count": 31, "skip": False},
            # postamble announcer row on the sentinel char_id (must stay put)
            {"line_id": "b004", "beat_id": "b004", "speaker_role": "announcer",
             "char_id": "announcer", "text": "Stay tuned.",
             "word_count": 2, "char_count": 11, "skip": False},
        ],
        "music": [], "clips": [],
        "meta": {"episode_title": "Chandra's Echo", "style": "observatory_drama"},
    }


# ---------------------------------------------------------------------------
# INVARIANT A -- NO SILENT MUTATION (every cast-keyed flip carries a reason)
# ---------------------------------------------------------------------------

class TestInvariantANoSilentMutation:
    def test_helper_coercion_stamps_reason_token(self):
        line = {"line_id": "b011", "char_id": "c02", "speaker_role": "announcer"}
        out, changed = PL.coerce_speaker_role_for_char_id(
            line, {"c01", "c02"}, "pin")
        assert changed is True
        assert out["speaker_role"] == "character"
        flags = out.get("compose_flags") or []
        # PBUG-20260710-07: the flip must NOT be silent -- a reason is mandatory.
        assert any(f.startswith("role_coerce:") and "reason=cast_char_id" in f
                   for f in flags), (
            "cast-keyed role coercion must stamp a reason breadcrumb")

    def test_pre_freeze_sweep_records_audit_and_reason(self):
        # Site (c) -- the mandatory pre-freeze sweep. The mutation must ride an
        # auditable trail: per-line reason breadcrumb + meta["role_coercions"].
        led = _ledger_obj(_cascade_ledger_data())
        _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)
        rows = {r["line_id"]: r for r in led.data["lines"]}
        assert rows["b011"]["speaker_role"] == "character"
        rc = led.data["meta"].get("role_coercions")
        assert rc and rc["count"] == 1 and "b011" in rc["line_ids"]
        flags = rows["b011"].get("compose_flags") or []
        assert any(f.startswith("role_coerce:") and "reason=" in f
                   for f in flags)


# ---------------------------------------------------------------------------
# INVARIANT B -- ANNOUNCER-SENTINEL PROTECTION (never coerce the announcer)
# ---------------------------------------------------------------------------

class TestInvariantBAnnouncerSentinelProtection:
    def test_sentinel_char_id_line_never_coerced(self):
        # The per-lane announcer-sentinel invariant: a line on the sentinel
        # char_id "announcer" is out of cast_ids -> never flipped to character.
        line = {"line_id": "b001", "char_id": "announcer",
                "speaker_role": "announcer"}
        out, changed = PL.coerce_speaker_role_for_char_id(
            line, {"c01", "c02"}, "pin")
        assert changed is False
        assert out["speaker_role"] == "announcer"

    def test_named_announcer_cast_slot_excluded_from_cast_ids(self):
        # PBUG-07 direction / safety net: the announcer is often stored as an
        # ordinary cast slot (char_id=c01, name=ANNOUNCER) rather than the
        # sentinel. NAME-EXCLUSION must still keep it out of cast_ids, so its
        # intro line is protected even without a minted sentinel char_id.
        led = {"cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "tts_model": "kokoro"},
            {"char_id": "c02", "name": "NORA ECKELS", "tts_model": "bark"},
        ]}
        cast_ids = PL.cast_ids_from_ledger(led)
        assert cast_ids == {"c02"}
        intro = {"line_id": "b001", "char_id": "c01",
                 "speaker_role": "announcer"}
        out, changed = PL.coerce_speaker_role_for_char_id(intro, cast_ids, "pin")
        assert changed is False
        assert out["speaker_role"] == "announcer"

    def test_sweep_leaves_sentinel_announcer_rows_untouched(self):
        led = _ledger_obj(_cascade_ledger_data())
        _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)
        rows = {r["line_id"]: r for r in led.data["lines"]}
        # sentinel announcer rows keep their role; only the b011 culprit moved
        assert rows["b001"]["speaker_role"] == "announcer"
        assert rows["b004"]["speaker_role"] == "announcer"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
