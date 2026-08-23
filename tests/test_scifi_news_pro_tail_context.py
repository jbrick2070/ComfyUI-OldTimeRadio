"""Writer-tail ownership and persistence contracts."""
from __future__ import annotations

import builtins
import dataclasses
import dis
import inspect
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import OTR_LedgerScriptWriter as writer_module  # noqa: E402
from nodes import _otr_writer_inputs as writer_inputs_module  # noqa: E402
from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    WriterTailContext,
    _SlotScheduler,
)


@pytest.fixture(autouse=True)
def _preserve_current_ledger():
    from nodes import production_ledger as ledger_module

    saved = ledger_module._CURRENT
    yield
    ledger_module._CURRENT = saved


PINNED_FIELDS = (
    "led", "meta", "resolved", "outline_view", "canon",
    "episode_root", "episode_id", "contract", "style_grammar_on",
    "source_bank_row", "slot_scheduler", "creative_fn", "technical_fn",
    "run_story_spine", "final_title_override",
    # 2026-08-23: the visual-style roll receipt. It was being read off `run()`'s
    # locals from inside the tail, which is not a scope the tail has -- see the
    # field's own docstring and the bytecode guard below.
    "style_roll",
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


def _stub_generate(*_args, **_kwargs) -> str:
    return ""


def _make_resolved(**overrides) -> dict:
    resolved = {
        "episode_title": "The Byte Identity Hour",
        "temperature": 0.7,
        "top_p": 0.9,
        "target_words": 30,
        "num_characters": 2,
        "creative_writing_model": "stub/creative-model",
        "technical_model": "stub/technical-model",
        "creativity": 0.5,
        "act_count": 1,
        "include_act_breaks": False,
        "optimization_profile": "balanced",
        "seed_source": "rss",
        "source_ref": "",
        "news_seed": "Test science wire item.",
        "perfect_run_spacesaver": False,
        "llm_policy": _StubPolicy(),
    }
    resolved.update(overrides)
    return resolved


def _make_ctx(tmp_path: Path, monkeypatch, **overrides) -> WriterTailContext:
    from nodes import production_ledger as ledger_module
    from nodes import _otr_canon as canon_module
    from nodes import _otr_model_loader as loader
    from tests.fixtures.ledger_stub import make_stub_ledger

    monkeypatch.setattr(
        loader,
        "request_slot",
        lambda slot, model_id, policy=None, load_config=None: {
            "model": None,
            "tokenizer": None,
        },
    )

    episode_id = "writer_tail_pin"
    episode_root = tmp_path / "episodes" / episode_id
    episode_root.mkdir(parents=True, exist_ok=True)
    led = ledger_module.new_ledger(
        episode_id=episode_id, out_dir=str(episode_root))
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
    canon = canon_module.episode_canon_from_outline_dict({
        "title": outline.title,
        "premise": outline.premise,
        "setting": outline.setting,
        "time_of_day": outline.time_of_day,
        "sound_palette": [],
    })
    scheduler = _SlotScheduler(
        creative_id="stub/creative-model",
        technical_id="stub/technical-model",
        top_p=0.9,
        min_p=0.0,
        repetition_penalty=1.0,
    )
    values = {
        "led": led,
        "meta": meta,
        "resolved": _make_resolved(),
        "outline_view": outline,
        "canon": canon,
        "episode_root": episode_root,
        "episode_id": episode_id,
        "contract": None,
        "style_grammar_on": False,
        "source_bank_row": SimpleNamespace(
            source_bank_id="scifi_news_pro",
            default_story_pipeline="scifi_scifi_news_pro_circuit",
            defaults={
                "title_form_label": "science-fiction radio drama",
                "hud_origin_label": "",
            },
        ),
        "slot_scheduler": scheduler,
        "creative_fn": _stub_generate,
        "technical_fn": _stub_generate,
        "run_story_spine": False,
        "final_title_override": None,
    }
    values.update(overrides)
    return WriterTailContext(**values)


def _normalize(out: tuple, tmp_path: Path) -> tuple:
    root = str(tmp_path)
    escaped = json.dumps(root)[1:-1]
    return tuple(
        value.replace(escaped, "<ROOT>").replace(root, "<ROOT>")
        if isinstance(value, str) else value
        for value in out
    )


def test_ctx_field_contract_exact():
    assert tuple(
        field.name for field in dataclasses.fields(WriterTailContext)
    ) == PINNED_FIELDS


def test_tail_signature_and_no_closure():
    fn = OTR_LedgerScriptWriter._run_writer_tail
    assert list(inspect.signature(fn).parameters) == [
        "self", "ctx", "tail_finalizer"]
    assert fn.__code__.co_freevars == ()


# --------------------------------------------------------------------------- #
# THE GUARD THAT ACTUALLY CATCHES A BORROWED LOCAL
#
# `co_freevars == ()` above is real but it is NOT the invariant the tail's
# docstring claims ("consumes ONLY this context -- no closure over run()
# locals"). A CLOSURE only forms over an ENCLOSING function's locals, and
# `run()` does not enclose `_run_writer_tail` -- they are sibling methods. A
# sibling's local therefore compiles to LOAD_GLOBAL and can never appear in
# co_freevars, so that assertion could not fail however wrong the code was.
#
# It was wrong. `_run_writer_tail` read a bare `_style_roll` on the
# dynamic-style floor-fallback branch -- a local of `run()` -- and also called
# `random.Random` in a module that never imported `random`. Two unbound globals
# on ONE branch: reaching it raised NameError and took the tail down instead of
# falling back to a floor style. Both are fixed; this is the guard that would
# have said so.
# --------------------------------------------------------------------------- #
def _global_loads(code, found):
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            found.add(instruction.argval)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _global_loads(const, found)
    return found


def _functions_of(module):
    """Every function the MODULE ITSELF compiled, module-level and on classes.

    The `co_filename` check is load-bearing: `@dataclass` and `Protocol` attach
    generated `__repr__` / `__subclasshook__` bodies compiled inside the stdlib,
    whose globals resolve in dataclasses.py and typing.py rather than here.
    """
    own_file = module.__file__
    for name, obj in vars(module).items():
        candidates = []
        if isinstance(obj, types.FunctionType):
            candidates.append((name, obj))
        elif isinstance(obj, type) and getattr(obj, "__module__", "") == module.__name__:
            for method_name, member in vars(obj).items():
                function = (member.__func__
                            if isinstance(member, (classmethod, staticmethod))
                            else member)
                if isinstance(function, types.FunctionType):
                    candidates.append(("%s.%s" % (name, method_name), function))
        for label, function in candidates:
            if function.__code__.co_filename == own_file:
                yield label, function


@pytest.mark.parametrize("module", [writer_module, writer_inputs_module],
                         ids=["writer", "writer_inputs"])
def test_no_function_loads_a_global_its_module_does_not_define(module):
    """A borrowed local is a NameError with a delay on it."""
    offenders = {}
    for label, function in _functions_of(module):
        for name in _global_loads(function.__code__, set()):
            if not hasattr(module, name) and not hasattr(builtins, name):
                offenders.setdefault(name, set()).add(label)
    assert not offenders, (
        "%s loads globals it does not define -- each one raises NameError the "
        "moment its branch runs: %s"
        % (module.__name__,
           ", ".join("%s (in %s)" % (name, ", ".join(sorted(where)))
                     for name, where in sorted(offenders.items()))))


def test_the_guard_would_have_caught_the_borrowed_local(tmp_path):
    """The negative control. A guard nobody has watched fail is a guess.

    This is the shape of the real defect: one method binds a name, a SIBLING
    method reads it. No closure forms, `co_freevars` stays empty, and the old
    assertion sails straight past it.
    """
    source = (
        "class Writer:\n"
        "    def run(self):\n"
        "        _style_roll = object()\n"
        "        return self.tail()\n"
        "\n"
        "    def tail(self):\n"
        "        return _style_roll.seed\n"
    )
    path = tmp_path / "borrowed_local_module.py"
    path.write_text(source, encoding="utf-8")
    module = types.ModuleType("borrowed_local_module")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)

    assert module.Writer.tail.__code__.co_freevars == (), (
        "the OLD assertion passes on the defect -- that is the whole point")
    offenders = {}
    for label, function in _functions_of(module):
        for name in _global_loads(function.__code__, set()):
            if not hasattr(module, name) and not hasattr(builtins, name):
                offenders.setdefault(name, set()).add(label)
    assert "_style_roll" in offenders, offenders
    assert offenders["_style_roll"] == {"Writer.tail"}


def test_run_delegates_to_tail():
    source = inspect.getsource(OTR_LedgerScriptWriter.run)
    assert "WriterTailContext(" in source
    assert "_run_writer_tail" in source
    assert "run_story_spine=True" in source
    assert "final_title_override=None" in source
    assert "meta=meta" in source


def test_tail_byte_identity_same_inputs(tmp_path, monkeypatch):
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
    assert out_a == out_b
    assert len(out_a) == 5
    assert out_a[4] == "stub/technical-model"


def test_tail_preserves_lane_seals_on_disk(tmp_path, monkeypatch):
    ctx = _make_ctx(
        tmp_path,
        monkeypatch,
        final_title_override="The Sealed Draft",
    )
    ctx.led.data["meta"]["source_bank"] = "scifi_news_pro"
    sealed = {
        "raw_sha256": "1" * 64,
        "normalized_sha256": "2" * 64,
        "parsed_sha256": "3" * 64,
        "proof_map_sha256": "4" * 64,
        "artifact_sha256": "5" * 64,
        "p3_attempts": [{"attempt": 1, "outcome": "accepted"}],
    }
    ctx.led.data["meta"]["scifi_news_pro"] = {"final_draft": sealed}

    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert ctx.led.data["meta"]["scifi_news_pro"]["final_draft"] == sealed
    saved = json.loads(Path(ctx.led.path).read_text(encoding="utf-8"))
    assert saved["meta"]["scifi_news_pro"]["final_draft"] == sealed


@pytest.mark.parametrize("source_bank", ["scifi_news_pro"])
def test_content_owned_tail_stamps_delivery_before_finalizer(
        tmp_path, monkeypatch, source_bank):
    from nodes._otr_readiness import text_for_tts_source_sha256
    from nodes._otr_text_delivery import (
        CONTENT_OWNED,
        delivery_mode_for_meta,
        resolve_line_delivery,
    )

    class _DeliveryProbe:
        checked = False

        def before_save(self, *, ctx):
            rows = [
                row for row in ctx.led.data["lines"]
                if row.get("text", "").strip()
                and not row.get("skip")
                and row.get("speaker_role") in {"character", "announcer"}
            ]
            assert rows
            for row in rows:
                canonical, delivery = resolve_line_delivery(row, CONTENT_OWNED)
                assert canonical == row["text"]
                assert delivery == row["text_for_tts"]
                assert row["text_for_tts_source_sha256"] == (
                    text_for_tts_source_sha256(canonical))
            self.checked = True

        def after_save(self, *, saved_path, ledger_data):
            assert saved_path
            assert ledger_data

    ctx = _make_ctx(tmp_path / source_bank, monkeypatch)
    ctx.led.data["meta"]["source_bank"] = source_bank
    canonical = {
        row["line_id"]: row["text"]
        for row in ctx.led.data["lines"]
        if row.get("text", "").strip()
    }
    probe = _DeliveryProbe()
    OTR_LedgerScriptWriter()._run_writer_tail(ctx, tail_finalizer=probe)

    assert probe.checked
    assert delivery_mode_for_meta(ctx.led.data["meta"]) == CONTENT_OWNED
    assert isinstance(ctx.led.data["meta"].get("episode_seed"), int)
    assert "cast_seed" not in (
        ctx.led.data["meta"].get("cast_contract") or {})
    assert {
        row["line_id"]: row["text"]
        for row in ctx.led.data["lines"]
        if row.get("line_id") in canonical
    } == canonical


def test_legacy_tail_does_not_introduce_delivery_projection(
        tmp_path, monkeypatch):
    from nodes._otr_text_delivery import LEGACY, delivery_mode_for_meta

    ctx = _make_ctx(tmp_path, monkeypatch)
    for row in ctx.led.data["lines"]:
        row.pop("text_for_tts", None)
        row.pop("text_for_tts_source_sha256", None)
        row.pop("text_for_tts_receipt", None)

    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert delivery_mode_for_meta(ctx.led.data["meta"]) == LEGACY
    assert all("text_for_tts" not in row for row in ctx.led.data["lines"])


def test_run_story_spine_false_uses_unload_path(tmp_path, monkeypatch):
    from nodes import _otr_story_spine as spine

    calls = []
    monkeypatch.setattr(
        spine, "run_post_script_spine",
        lambda *args, **kwargs: calls.append("spine"),
    )
    ctx = _make_ctx(tmp_path, monkeypatch, run_story_spine=False)
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert calls == []
    assert "writer_llm_unload" in ctx.led.data["meta"]


def test_run_story_spine_true_runs_once(tmp_path, monkeypatch):
    from nodes import _otr_story_spine as spine

    calls = []
    monkeypatch.setattr(
        spine,
        "run_post_script_spine",
        lambda led, meta: calls.append((led, meta)),
    )
    ctx = _make_ctx(tmp_path, monkeypatch, run_story_spine=True)
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert calls == [(ctx.led, ctx.meta)]


def _spy_title_regen(monkeypatch):
    import nodes.OTR_LedgerScriptWriter as writer_module

    calls = []

    def _record(*args, **kwargs):
        calls.append(1)
        return ""

    monkeypatch.setattr(
        writer_module, "_generate_title_from_script", _record)
    return calls


def test_title_precedence_user_beats_override(tmp_path, monkeypatch):
    regen_calls = _spy_title_regen(monkeypatch)
    ctx = _make_ctx(
        tmp_path,
        monkeypatch,
        final_title_override="The Authored Play Title",
    )
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert ctx.led.data["meta"]["episode_title"] == "The Byte Identity Hour"
    assert ctx.led.data["meta"]["title_source"] == "user"
    assert regen_calls == []


def test_title_override_wins_without_regen(tmp_path, monkeypatch):
    regen_calls = _spy_title_regen(monkeypatch)
    ctx = _make_ctx(
        tmp_path,
        monkeypatch,
        final_title_override="The Authored Play Title",
    )
    ctx.resolved["episode_title"] = ""
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert ctx.led.data["meta"]["episode_title"] == "The Authored Play Title"
    assert ctx.led.data["meta"]["title_source"] == "scifi_news_pro_script_title"
    assert ctx.canon.title == "The Authored Play Title"
    assert regen_calls == []


def test_title_regen_falls_back_to_outline(tmp_path, monkeypatch):
    regen_calls = _spy_title_regen(monkeypatch)
    ctx = _make_ctx(tmp_path, monkeypatch)
    ctx.resolved["episode_title"] = ""
    OTR_LedgerScriptWriter()._run_writer_tail(ctx)

    assert regen_calls == [1]
    assert ctx.led.data["meta"]["episode_title"] == "Signal at the Light"
    assert ctx.led.data["meta"]["title_source"] == "outline_fallback"
