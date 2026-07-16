"""scifi_fable2 S1a -- the writer tail extraction byte-identity pin.

The tail (J.5 title regen -> canon write -> K meta stamps -> K.5 visual
plan -> K.5.5/K.5.6 reflections -> story-spine orchestrator/unload ->
REJECT gate -> provenance -> L return assembly -> M save) now lives in
``OTR_LedgerScriptWriter._run_writer_tail(ctx)`` and consumes ONLY the
``WriterTailContext`` -- no closure over run() locals (architecture doc
2026-07-10-scifi-fable2, sections 11/13).

Pins (r2 anchor):
* the ctx field contract -- exactly the 17 pinned fields, pinned order;
* no-closure -- the tail has no freevars and takes (self, ctx) only;
* run() delegation -- run() builds the ctx (final_title_override=None,
  run_story_spine=True) and returns the tail's tuple; the tail body no
  longer lives inline in run();
* byte identity -- two independent, identical synthetic states through
  the tail produce byte-identical 5-tuples (same-run diff; NOT
  fixture-based -- the S1b fixture is captured from the S1a live smoke);
* ctx.run_story_spine=False gates the Wave-2 spine WITHOUT disturbing
  the legacy env default (doc s14 item 8);
* final_title_override precedence: user-typed episode_title > override
  (title_source="fable2_script_title", regen NEVER called) > LLM regen;
* refine_active=True writes the self._refine_last stash.

The LIVE legacy science smoke that must accompany this sprint is the
production-path half of the byte-identity gate; this file is the
in-suite half.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    WriterTailContext,
    _SlotScheduler,
)


@pytest.fixture(autouse=True)
def _preserve_current_ledger():
    """new_ledger() sets production_ledger._CURRENT (the process-wide
    current-ledger singleton). These tests build REAL ledgers in tmp
    dirs; leaking one as the ambient singleton breaks later tests that
    read it (e.g. the freeze-cascade C4 passthrough suite). Save and
    restore around every test."""
    from nodes import production_ledger as _PL
    saved = _PL._CURRENT
    yield
    _PL._CURRENT = saved


PINNED_FIELDS = (
    "led", "meta", "resolved", "outline_view", "canon",
    "episode_root", "episode_id", "contract", "style_grammar_on",
    "source_bank_row", "slot_scheduler", "creative_fn", "technical_fn",
    "cast_seed", "refine_active", "run_story_spine",
    "final_title_override",
)


@dataclass
class _StubPolicy:
    device: str = "cpu"
    attn_impl: str = "sdpa"
    quant_policy: str = "none"
    vram_ceiling_gb: float = 0.0
    gguf_n_ctx: int = 0
    gguf_quant: str = ""
    lane_allowlist: tuple = ()


def _stub_generate(*_a, **_k) -> str:
    """Deterministic no-op LLM: reflections take their sentinel path."""
    return ""


def _make_resolved(**over) -> dict:
    r = {
        "episode_title":          "The Byte Identity Hour",
        "temperature":            0.7,
        "top_p":                  0.9,
        "target_words":           30,
        "num_characters":         2,
        "creative_writing_model": "stub/creative-model",
        "technical_model":        "stub/technical-model",
        "creativity":             0.5,
        "act_count":              1,
        "include_act_breaks":     False,
        "optimization_profile":   "balanced",
        "seed_source":            "rss",
        "source_ref":             "",
        "news_seed":              "Test science wire item.",
        "perfect_run_spacesaver": False,
        "llm_policy":             _StubPolicy(),
    }
    r.update(over)
    return r


def _make_ctx(tmp_path: Path, monkeypatch, **over) -> WriterTailContext:
    """Build a complete synthetic WriterTailContext over a REAL Ledger
    in a tmp episode root. Two calls with equal args produce equal
    states (paths normalized by the caller)."""
    from nodes import production_ledger as _PL
    from nodes import _otr_canon as _OTRC
    from nodes import _otr_model_loader as _OTRML
    from tests.fixtures.ledger_stub import make_stub_ledger

    monkeypatch.setattr(
        _OTRML, "request_slot",
        lambda slot, model_id, policy=None, load_config=None: {
            "model": None, "tokenizer": None,
        },
    )

    episode_id = "s1a_tail_pin_ep"
    episode_root = tmp_path / "episodes" / episode_id
    episode_root.mkdir(parents=True, exist_ok=True)

    led = _PL.new_ledger(episode_id=episode_id, out_dir=str(episode_root))
    stub = make_stub_ledger()
    led.data["cast"] = stub["cast"]
    led.data["lines"] = stub["lines"]
    meta = led.data.setdefault("meta", {})

    outline = SimpleNamespace(
        premise="A lighthouse keeper hears a signal nobody sent.",
        title="Signal at the Light",
        setting="a remote lighthouse",
        time_of_day="night",
    )
    canon = _OTRC.episode_canon_from_outline_dict({
        "title":         outline.title,
        "premise":       outline.premise,
        "setting":       outline.setting,
        "time_of_day":   outline.time_of_day,
        "sound_palette": [],
    })

    sched = _SlotScheduler(
        creative_id="stub/creative-model",
        technical_id="stub/technical-model",
        top_p=0.9, min_p=0.0, repetition_penalty=1.0,
    )

    kwargs = dict(
        led=led,
        meta=meta,
        resolved=_make_resolved(),
        outline_view=outline,
        canon=canon,
        episode_root=episode_root,
        episode_id=episode_id,
        contract=None,
        style_grammar_on=False,
        source_bank_row=SimpleNamespace(source_bank_id="scifi_fable2", defaults={
            "title_form_label": "science-fiction radio drama",
            "hud_origin_label": "",
        }),
        slot_scheduler=sched,
        creative_fn=_stub_generate,
        technical_fn=_stub_generate,
        cast_seed=1234,
        refine_active=False,
        run_story_spine=True,
        final_title_override=None,
    )
    kwargs.update(over)
    return WriterTailContext(**kwargs)


def _normalize(out: tuple, tmp_path: Path) -> tuple:
    """Replace the per-test tmp root in string outputs so two runs over
    different tmp dirs compare byte-identical."""
    root = str(tmp_path)
    esc = json.dumps(root)[1:-1]  # JSON-escaped form (backslashes)
    fixed = []
    for item in out:
        if isinstance(item, str):
            item = item.replace(esc, "<ROOT>").replace(root, "<ROOT>")
        fixed.append(item)
    return tuple(fixed)


# ---------------------------------------------------------------------------
# Contract pins
# ---------------------------------------------------------------------------

def test_ctx_field_contract_exact():
    """WriterTailContext carries EXACTLY the 17 pinned fields, in the
    pinned order (doc s11). Any add/remove/reorder is an S1b+ design
    change and must update the architecture doc first."""
    names = tuple(f.name for f in dataclasses.fields(WriterTailContext))
    assert names == PINNED_FIELDS, (
        f"WriterTailContext field drift.\nExpected: {PINNED_FIELDS}\n"
        f"Got:      {names}"
    )


def test_tail_signature_and_no_closure():
    """The tail takes (self, ctx) only and closes over nothing -- the
    'consumes ONLY ctx' law is structural, not stylistic."""
    fn = OTR_LedgerScriptWriter._run_writer_tail
    params = list(inspect.signature(fn).parameters)
    assert params == ["self", "ctx", "tail_finalizer"], f"signature drift: {params}"
    assert fn.__code__.co_freevars == (), (
        f"tail grew a closure: {fn.__code__.co_freevars}"
    )


def test_run_delegates_to_tail():
    """run() ends by building the ctx from its locals (legacy pins:
    final_title_override=None, run_story_spine=True) and returning the
    tail's tuple; the tail body is no longer inline."""
    src = inspect.getsource(OTR_LedgerScriptWriter.run)
    assert "WriterTailContext(" in src
    assert "_run_writer_tail" in src
    assert "run_story_spine=True" in src
    assert "final_title_override=None" in src
    assert 'meta=led.data.setdefault("meta", {})' in src
    assert "J.5. Post-composition title regen" not in src, (
        "the tail body is still inline in run()"
    )


# ---------------------------------------------------------------------------
# Byte identity (same-run diff)
# ---------------------------------------------------------------------------

def test_tail_byte_identity_same_inputs(tmp_path, monkeypatch):
    """Two independent, identical synthetic states through the tail
    produce byte-identical 5-tuples. Any nondeterminism or hidden state
    the extraction smuggled in breaks this."""
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    out_a = _normalize(
        OTR_LedgerScriptWriter()._run_writer_tail(
            _make_ctx(root_a, monkeypatch)),
        root_a,
    )
    out_b = _normalize(
        OTR_LedgerScriptWriter()._run_writer_tail(
            _make_ctx(root_b, monkeypatch)),
        root_b,
    )

    assert len(out_a) == 5 and len(out_b) == 5
    for i, (a, b) in enumerate(zip(out_a, out_b)):
        assert a == b, f"tail output slot {i} drifted between equal states"
    # slot 4 broadcasts the resolved technical model id
    assert out_a[4] == "stub/technical-model"


def test_tail_preserves_selected_final_draft_seals_on_disk(
        tmp_path, monkeypatch):
    """The shared tail's final save cannot shed the C4b winner receipt."""
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    ctx = _make_ctx(
        tmp_path / "sealed", monkeypatch, run_story_spine=False,
        final_title_override="The Sealed Draft")
    ctx.led.data["meta"]["source_bank"] = "scifi_fable2"
    sealed = {
        "raw_sha256": "1" * 64,
        "normalized_sha256": "2" * 64,
        "parsed_sha256": "3" * 64,
        "proof_map_sha256": "4" * 64,
        "artifact_sha256": "5" * 64,
        "score": [0, 0, 0, 0, 0],
        "scoring_context_sha256": "6" * 64,
        "p3_attempts": [{"attempt": 1, "outcome": "accepted"}],
        "p5_attempts": [{"attempt": 1, "outcome": "accepted"}],
    }
    ctx.led.data["meta"]["fable2"] = {"final_draft": sealed}

    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert ctx.led.data["meta"]["fable2"]["final_draft"] == sealed
    saved = json.loads(Path(ctx.led.path).read_text(encoding="utf-8"))
    assert saved["meta"]["fable2"]["final_draft"] == sealed


@pytest.mark.parametrize(
    "source_bank",
    ("scifi_fable2", "scifi_codex", "scifi_sonnet"),
)
def test_content_owned_tail_stamps_delivery_before_finalizer(
    tmp_path, monkeypatch, source_bank,
):
    """Every sealed-content lane reaches its freeze hook with fresh delivery.

    The shared writer tail is the final producer boundary: it stamps
    pronunciation-safe delivery text after all writer-side work and before
    a lane finalizer can execute its Phase-10 freeze.
    """
    from nodes._otr_readiness import text_for_tts_source_sha256
    from nodes._otr_text_delivery import (
        CONTENT_OWNED,
        delivery_mode_for_meta,
        resolve_line_delivery,
    )

    class _DeliveryProbe:
        checked = False

        def before_save(self, *, ctx):
            cast_seed = (
                (ctx.led.data["meta"].get("cast_contract") or {})
                .get("cast_seed", ctx.led.data["meta"].get("episode_seed"))
            )
            assert cast_seed is not None
            rows = [
                row for row in ctx.led.data["lines"]
                if row.get("text", "").strip()
                and not row.get("skip")
                and row.get("speaker_role") in {"character", "announcer"}
            ]
            assert rows
            for row in rows:
                canonical, delivery = resolve_line_delivery(
                    row, CONTENT_OWNED,
                )
                assert canonical == row["text"]
                assert delivery == row["text_for_tts"]
                assert row["text_for_tts_source_sha256"] == (
                    text_for_tts_source_sha256(canonical)
                )
            self.checked = True

        def after_save(self, *, saved_path, ledger_data):
            assert saved_path
            assert ledger_data

    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    ctx = _make_ctx(tmp_path / source_bank, monkeypatch, run_story_spine=False)
    ctx.led.data["meta"]["source_bank"] = source_bank
    before = {
        row["line_id"]: row["text"]
        for row in ctx.led.data["lines"]
        if row.get("text", "").strip()
        and not row.get("skip")
        and row.get("speaker_role") in {"character", "announcer"}
    }
    probe = _DeliveryProbe()

    OTR_LedgerScriptWriter()._run_writer_tail(ctx, tail_finalizer=probe)

    assert probe.checked
    assert delivery_mode_for_meta(ctx.led.data["meta"]) == CONTENT_OWNED
    # The content-owned tail owes CreditsRoll a durable seed receipt, and
    # meta.episode_seed is the receipt it accepts.
    assert isinstance(ctx.led.data["meta"].get("episode_seed"), int)
    # It must NOT claim meta.cast_contract.cast_seed: that key asserts the
    # writer's seeded picker produced this cast and can be REPLAYED from it,
    # which is false for a lane that builds its own cast. CastLock replays the
    # picker whenever it sees cast_seed, and a lane-owned cast carries no
    # num_characters_request -- claiming it detonated the replay live with
    # "num_characters must be 1-6, got 0".
    assert "cast_seed" not in (ctx.led.data["meta"].get("cast_contract") or {})
    assert {
        row["line_id"]: row["text"]
        for row in ctx.led.data["lines"]
        if row.get("line_id") in before
    } == before


def test_legacy_tail_does_not_introduce_delivery_stamps(tmp_path, monkeypatch):
    """The shared content-owned repair does not alter legacy delivery."""
    from nodes._otr_text_delivery import LEGACY, delivery_mode_for_meta

    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    ctx = _make_ctx(tmp_path, monkeypatch, run_story_spine=False)
    for row in ctx.led.data["lines"]:
        row.pop("text_for_tts", None)
        row.pop("text_for_tts_source_sha256", None)
        row.pop("text_for_tts_receipt", None)

    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert delivery_mode_for_meta(ctx.led.data["meta"]) == LEGACY
    assert all(
        "text_for_tts" not in row
        for row in ctx.led.data["lines"]
    )


# ---------------------------------------------------------------------------
# run_story_spine gate (doc s14 item 8)
# ---------------------------------------------------------------------------

def test_run_story_spine_false_gates_spine(tmp_path, monkeypatch):
    """ctx.run_story_spine=False: the spine NEVER runs even when the env
    default would enable it; the unload branch stamps writer_llm_unload."""
    from nodes import _otr_story_spine as _OTRSPINE

    calls = []
    monkeypatch.setattr(_OTRSPINE, "enabled", lambda: True)
    monkeypatch.setattr(
        _OTRSPINE, "run_post_script_spine",
        lambda *a, **k: calls.append(1),
    )

    ctx = _make_ctx(tmp_path, monkeypatch, run_story_spine=False)
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert calls == [], "spine ran despite ctx.run_story_spine=False"
    assert "writer_llm_unload" in ctx.led.data["meta"], (
        "unload branch did not stamp writer_llm_unload"
    )


def test_run_story_spine_true_respects_env_default(tmp_path, monkeypatch):
    """Legacy posture: ctx.run_story_spine=True leaves the decision to
    the env-gated _OTRSPINE.enabled(), exactly as before the extraction."""
    from nodes import _otr_story_spine as _OTRSPINE

    calls = []
    monkeypatch.setattr(_OTRSPINE, "enabled", lambda: False)
    monkeypatch.setattr(
        _OTRSPINE, "run_post_script_spine",
        lambda *a, **k: calls.append("off"),
    )
    OTR_LedgerScriptWriter()._run_writer_tail(
        _make_ctx(tmp_path / "off", monkeypatch, run_story_spine=True))
    assert calls == [], "spine ran with enabled()=False"

    monkeypatch.setattr(_OTRSPINE, "enabled", lambda: True)
    monkeypatch.setattr(
        _OTRSPINE, "run_post_script_spine",
        lambda led, meta, outline, **k: calls.append("on"),
    )
    OTR_LedgerScriptWriter()._run_writer_tail(
        _make_ctx(tmp_path / "on", monkeypatch, run_story_spine=True))
    assert calls == ["on"], "spine did not run with enabled()=True"


# ---------------------------------------------------------------------------
# final_title_override precedence (r3/M3)
# ---------------------------------------------------------------------------

def _spy_title_regen(monkeypatch):
    import nodes.OTR_LedgerScriptWriter as _W
    calls = []

    def _rec(*a, **k):
        calls.append(1)
        return ""

    monkeypatch.setattr(_W, "_generate_title_from_script", _rec)
    return calls


def test_title_precedence_user_beats_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    regen_calls = _spy_title_regen(monkeypatch)
    ctx = _make_ctx(
        tmp_path, monkeypatch,
        final_title_override="The Authored Play Title",
    )  # resolved.episode_title stays user-typed
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)
    meta = ctx.led.data["meta"]
    assert meta["episode_title"] == "The Byte Identity Hour"
    assert meta["title_source"] == "user"
    assert regen_calls == [], "title regen ran despite a user title"


def test_title_override_wins_and_regen_never_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    regen_calls = _spy_title_regen(monkeypatch)
    ctx = _make_ctx(tmp_path, monkeypatch, final_title_override="The Authored Play Title")
    ctx.resolved["episode_title"] = ""  # no user title
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)
    meta = ctx.led.data["meta"]
    assert meta["episode_title"] == "The Authored Play Title"
    assert meta["title_source"] == "fable2_script_title"
    assert regen_calls == [], (
        "title regen ran despite final_title_override -- the fable2 lane "
        "NEVER regenerates its authored title"
    )
    assert ctx.canon.title == "The Authored Play Title"


def test_title_regen_path_unchanged_when_no_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    regen_calls = _spy_title_regen(monkeypatch)
    ctx = _make_ctx(tmp_path, monkeypatch)  # override=None (legacy)
    ctx.resolved["episode_title"] = ""
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)
    meta = ctx.led.data["meta"]
    assert regen_calls == [1], "legacy blank-title path must call regen"
    # stub regen returns "" -> outline fallback, exactly as before
    assert meta["episode_title"] == "Signal at the Light"
    assert meta["title_source"] == "outline_fallback"


# ---------------------------------------------------------------------------
# refine stash
# ---------------------------------------------------------------------------

def test_refine_stash_written_when_active(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    writer = OTR_LedgerScriptWriter()
    ctx = _make_ctx(tmp_path, monkeypatch, refine_active=True)
    out = writer._run_writer_tail(ctx)
    stash = writer._refine_last
    assert set(stash) == {
        "outline", "premise", "creative_fn", "cast_seed",
        "episode_root", "led", "script_text",
    }
    assert stash["cast_seed"] == 1234
    assert stash["script_text"] == out[0]
    assert stash["led"] is ctx.led


def test_no_refine_stash_when_inactive(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_STORY_SPINE", "0")
    writer = OTR_LedgerScriptWriter()
    OTR_LedgerScriptWriter()._run_writer_tail(
        _make_ctx(tmp_path, monkeypatch, refine_active=False))
    assert not getattr(writer, "_refine_last", None)
