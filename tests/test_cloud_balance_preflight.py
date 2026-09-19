"""Queue-time wallet vs estimate -- socket-free. Inject balances and prices."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nodes._otr_shared import cloud_balance_preflight as cbp

REPO = Path(__file__).resolve().parents[1]


def _walk(prompt, _unique_id):
    return [n for n in (prompt or {}).values() if isinstance(n, dict)]


def _director(video, image="z_image_turbo"):
    return {
        "class_type": "OTR_VideoDirector",
        "inputs": {
            "gate_in": ["63", 0],
            "announcer_video_model": video,
            "music_video_model": video,
            "character_video_model": video,
            "announcer_image_model": image,
            "music_image_model": image,
            "character_image_model": image,
        },
    }


def _writer(handle, widget, slug, act_count=3, replay_from=""):
    return {
        "class_type": "OTR_LedgerScriptWriter",
        "inputs": {
            "gate_in": ["63", 0],
            "act_count": str(act_count),
            "replay_from": replay_from,
            "creative_writing_model": handle,
            "technical_model": handle,
            widget: slug,
        },
    }


def _prompt(*nodes):
    out = {"63": {"class_type": "OTR_WorkflowValidator", "inputs": {}}}
    for i, node in enumerate(nodes, start=70):
        out[str(i)] = node
    return out


def _prices(table):
    def fn(eid, _eng):
        return table.get(eid)
    return fn


def _balance_map(mapping):
    def fn(wallet):
        return mapping[wallet]
    return fn


def _run(prompt, balances, *, unit=None, writer=None, skip_writer=False):
    return cbp.ensure_prompt_cloud_balance(
        prompt, "63",
        resolve_engine=lambda _eid: object(),
        unit_usd_fn=unit or _prices({}),
        writer_usd_fn=writer or (lambda _h, _s: (0.0, "injected")),
        walk_fn=_walk,
        skip_writer=skip_writer,
        balance_fn=_balance_map(balances) if isinstance(balances, dict) else balances,
    )


def test_clip_seconds_follow_veo_menu():
    assert cbp.clip_seconds_for_act_count(1) == 4
    assert cbp.clip_seconds_for_act_count("1") == 4
    assert cbp.clip_seconds_for_act_count(3) == 8
    assert cbp.clip_seconds_for_act_count(5) == 8
    assert cbp.clip_seconds_for_act_count(0) == 8


def test_pad_is_fifteen_percent_or_one_dollar():
    assert cbp.pad_usd(40.0) == 6.0
    assert cbp.pad_usd(3.0) == 1.0


def test_parse_comfy_micros_are_cents():
    assert cbp.parse_comfy_balance({"effective_balance_micros": 200}) == 2.0
    assert cbp.parse_comfy_balance({"amount_micros": 5000}) == 50.0
    assert cbp.parse_comfy_balance({}) is None


def test_comfy_remaining_two_estimate_forty_refuses():
    lines = [cbp.CostLine("comfy", "cloud_wan_i2v", 40, 1.0, 40.0, "40 beats")]
    verdict = cbp.judge_wallet(
        "comfy", lines, cbp.BalanceResult(2.0, "", cbp.COMFY_HOST, "refuse"))
    assert verdict.severity == "refuse"
    assert verdict.needed_usd == 46.0
    prompt = _prompt(_director("cloud_wan_i2v"))
    with pytest.raises(ValueError, match="comfy wallet") as exc:
        _run(prompt, {"comfy": cbp.BalanceResult(2.0, "", cbp.COMFY_HOST, "refuse")},
             unit=_prices({"cloud_wan_i2v": 1.0}), skip_writer=True)
    assert "needed $46.00" in str(exc.value)
    assert "remaining $2.00" in str(exc.value)


def test_comfy_remaining_fifty_estimate_forty_passes():
    prompt = _prompt(_director("cloud_wan_i2v"))
    verdicts = _run(
        prompt,
        {"comfy": cbp.BalanceResult(50.0, "", cbp.COMFY_HOST, "refuse")},
        unit=_prices({"cloud_wan_i2v": 1.0}),
        skip_writer=True,
    )
    assert [v.severity for v in verdicts] == ["ok"]
    assert verdicts[0].needed_usd == 46.0


def test_openrouter_limit_remaining_short_refuses():
    prompt = _prompt(_writer(
        "openrouter:slot-a", "openrouter_slot_a_model", "google/gemini-3.5-flash"))
    with pytest.raises(ValueError, match="openrouter wallet"):
        _run(
            prompt,
            {"openrouter": cbp.BalanceResult(
                0.5, "", cbp.OPENROUTER_HOST, "refuse", "key limit_remaining")},
            writer=lambda _h, _s: (3.0, "injected writer"),
        )


def test_openrouter_uncapped_credits_403_warns():
    def getter(url, _token):
        if url == cbp.OPENROUTER_KEY_URL:
            return 200, {"data": {"limit_remaining": None}}
        if url == cbp.OPENROUTER_CREDITS_URL:
            return 403, {}
        raise AssertionError(url)

    result = cbp.openrouter_balance(get_json=getter, bearer=lambda: "k")
    assert result.severity == "warn"
    assert result.remaining_usd is None
    prompt = _prompt(_writer(
        "openrouter:slot-a", "openrouter_slot_a_model", "google/gemini-3.5-flash"))
    verdicts = _run(
        prompt,
        {"openrouter": result},
        writer=lambda _h, _s: (3.0, "injected writer"),
    )
    assert [v.severity for v in verdicts] == ["warn"]


def test_google_only_paid_media_never_refuses_on_balance():
    prompt = _prompt(_director("google_veo_video", image="google_image"))
    verdicts = _run(
        prompt, {"google": cbp.google_balance()},
        unit=_prices({"google_veo_video": 0.40, "google_image": 0.04}),
        skip_writer=True,
    )
    assert verdicts
    assert all(v.wallet == "google" for v in verdicts)
    assert all(v.severity == "warn" for v in verdicts)


def test_local_ltx_and_kokoro_make_no_query():
    seen = []

    def boom(wallet):
        seen.append(wallet)
        raise AssertionError("local graph must not query a wallet")

    prompt = _prompt(_director("ltx_video"), {
        "class_type": "OTR_CastLock",
        "inputs": {
            "char_voice_engine": "kokoro",
            "announcer_voice_engine": "kokoro",
        },
    })
    verdicts = _run(prompt, boom, skip_writer=True)
    assert verdicts == []
    assert seen == []


def test_comfy_balance_http_500_refuses():
    result = cbp.comfy_balance(
        get_json=lambda _url, _token: (500, {}),
        bearer=lambda: "k",
    )
    assert result.severity == "refuse"
    assert result.remaining_usd is None
    prompt = _prompt(_director("cloud_wan_i2v"))
    with pytest.raises(ValueError, match="remaining unknown"):
        _run(prompt, {"comfy": result},
             unit=_prices({"cloud_wan_i2v": 1.0}), skip_writer=True)


def test_one_act_prices_four_second_clips_not_eight():
    prompt = _prompt(
        _writer("google_api:slot-a", "google_api_slot_a_model",
                "gemini-flash-latest", act_count=1),
        _director("google_veo_video", image="google_image"),
    )
    lines = cbp.estimate_lines(
        prompt, "63",
        resolve_engine=lambda _eid: object(),
        unit_usd_fn=_prices({"google_veo_video": 0.40}),
        writer_usd_fn=lambda _h, _s: (0.0, "injected"),
        walk_fn=_walk,
        skip_writer=True,
    )
    video = [ln for ln in lines if ln.label == "google_veo_video"]
    assert video and "4s clip" in video[0].note
    assert "8s clip" not in video[0].note


def test_unknown_price_empty_wallet_refuses():
    lines = [cbp.CostLine("comfy", "cloud_mystery", 1, None, 0.0, "no price")]
    verdict = cbp.judge_wallet(
        "comfy", lines, cbp.BalanceResult(0.0, "", cbp.COMFY_HOST, "refuse"))
    assert verdict.severity == "refuse"


def test_unknown_price_with_remaining_warns():
    lines = [cbp.CostLine("comfy", "cloud_mystery", 1, None, 0.0, "no price")]
    verdict = cbp.judge_wallet(
        "comfy", lines, cbp.BalanceResult(10.0, "", cbp.COMFY_HOST, "refuse"))
    assert verdict.severity == "warn"


def test_validator_calls_balance_between_slugs_and_assets():
    src = (REPO / "nodes" / "_otr_workflow_validator.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_queue_time_readiness_gates"
    )
    names = []
    for node in ast.walk(helper):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            names.append(func.attr)
    assert names.index("ensure_prompt_cloud_slugs") < names.index(
        "ensure_prompt_cloud_balance")
    assert names.index("ensure_prompt_cloud_balance") < names.index(
        "ensure_prompt_visual_assets")


def test_module_imports_are_stdlib_only():
    src = (REPO / "nodes" / "_otr_shared" / "cloud_balance_preflight.py").read_text(
        encoding="utf-8")
    top = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            top.extend(imp.name for imp in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.append(node.module or "")
    assert set(top) <= {"__future__", "json", "logging", "urllib.error",
                        "urllib.request", "typing"}, top


def test_empty_wallet_with_unpriced_engine_names_the_engine():
    prompt = _prompt(_director("cloud_mystery_i2v"))
    with pytest.raises(ValueError, match="no price function") as exc:
        _run(prompt, {"comfy": cbp.BalanceResult(0.0, "", cbp.COMFY_HOST, "refuse")},
             skip_writer=True)
    assert "cloud_mystery_i2v" in str(exc.value)


def test_two_writer_slots_on_one_wallet_share_the_run_ceiling():
    prompt = _prompt({
        "class_type": "OTR_LedgerScriptWriter",
        "inputs": {
            "gate_in": ["63", 0],
            "creative_writing_model": "comfy:slot-a",
            "technical_model": "comfy:slot-b",
            "comfy_slot_a_model": "google/gemini-3.5-flash",
            "comfy_slot_b_model": "google/gemini-3.5-flash",
            "replay_from": "",
        },
    })
    lines = cbp.estimate_lines(
        prompt, "63", resolve_engine=lambda _eid: None, walk_fn=_walk,
        writer_usd_fn=lambda handle, _slug: (2.0 if handle.endswith("-a") else 5.0, "n"))
    assert len(lines) == 1
    assert lines[0].wallet == "comfy" and lines[0].total_usd == 5.0


def test_default_unit_usd_reads_real_adapter_price_functions(monkeypatch):
    for name in ("OTR_CLOUD_VIDEO_EST_USD", "OTR_CLOUD_LTX25_EST_USD_PER_S",
                 "OTR_CLOUD_LTX25_DURATION", "OTR_CLOUD_FLUX_PRO_EST_USD",
                 "OTR_SONILO_MIN_DURATION_S"):
        monkeypatch.delenv(name, raising=False)
    from nodes import _otr_audio_engines, _otr_image_engines, _otr_video_engines  # noqa: F401
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_video_engines import registry as vreg

    assert cbp.default_unit_usd("cloud_wan_i2v", vreg.get_engine("cloud_wan_i2v")) == pytest.approx(0.50)
    ltx8 = cbp.default_unit_usd("cloud_ltx25_foley_plus", vreg.get_engine("cloud_ltx25_foley_plus"), 8)
    ltx4 = cbp.default_unit_usd("cloud_ltx25_foley_plus", vreg.get_engine("cloud_ltx25_foley_plus"), 4)
    assert ltx8 >= 0.40 * 8 - 1e-9 and ltx4 < ltx8
    assert cbp.default_unit_usd("cloud_flux_pro", ireg.get_engine("cloud_flux_pro")) == pytest.approx(0.05)
    assert cbp.default_unit_usd("sonilo", areg.get_engine("sonilo")) == pytest.approx(30 * 0.15 / 60.0)
    assert cbp.default_unit_usd("cloud_elevenlabs", areg.get_engine("cloud_elevenlabs")) == pytest.approx(
        cbp.TTS_CHARS_PER_ACT * 0.24 / 1000.0)
    assert cbp.default_unit_usd("google_veo_video", vreg.get_engine("google_veo_video")) is None


def test_every_registered_comfy_wallet_engine_has_a_price():
    from nodes import _otr_audio_engines, _otr_image_engines, _otr_video_engines  # noqa: F401
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_video_engines import registry as vreg

    unpriced = []
    for registry in (vreg, ireg, areg):
        for engine_id in sorted(getattr(registry, "_registry", {}) or {}):
            if cbp.wallet_for_engine(engine_id) != "comfy":
                continue
            if cbp.default_unit_usd(engine_id, registry.get_engine(engine_id)) is None:
                unpriced.append(engine_id)
    assert unpriced == []
