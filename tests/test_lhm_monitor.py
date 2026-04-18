"""
tests/test_lhm_monitor.py  --  Day 13 LHM poller unit tests
============================================================

Drives ``otr_v2.hyworld.lhm_monitor`` via injected ``fetcher`` /
``sleep_fn`` / ``monotonic_fn`` / ``unix_fn`` hooks so the tests run
deterministically without touching the real LHM endpoint or wall
clock.

No GPU, no network, no torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from itertools import count
from pathlib import Path

import pytest

from otr_v2.hyworld import lhm_monitor as lm


# ---------------------------------------------------------------------
# fixture data -- a synthetic LHM JSON tree
# ---------------------------------------------------------------------


def _fake_lhm_tree(
    *, gpu_temp_c: float, vram_used_gb: float, ram_used_gb: float,
    cpu_temp_c: float = 55.0, vram_total_gb: float = 16.0,
    ram_total_gb: float = 32.0,
) -> dict:
    """Mimic the shape we walk in ``_extract_sample_from_tree``.

    Only the fields we consume are populated; the walker ignores the
    rest so a sparse tree is fine.
    """
    return {
        "Text": "",
        "Children": [
            {
                "Text": "NVIDIA GeForce RTX 5080 Laptop",
                "Value": "",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Value": "",
                        "Children": [
                            {"Text": "GPU Core", "Value": f"{gpu_temp_c} C",
                             "Children": []},
                        ],
                    },
                    {
                        "Text": "Memory",
                        "Value": "",
                        "Children": [
                            {"Text": "GPU Memory Used",
                             "Value": f"{vram_used_gb} GB",
                             "Children": []},
                            {"Text": "GPU Memory Total",
                             "Value": f"{vram_total_gb} GB",
                             "Children": []},
                        ],
                    },
                ],
            },
            {
                "Text": "13th Gen Intel Core i9 CPU",
                "Value": "",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Value": "",
                        "Children": [
                            {"Text": "CPU Package", "Value": f"{cpu_temp_c} C",
                             "Children": []},
                        ],
                    },
                ],
            },
            {
                "Text": "Generic Memory",
                "Value": "",
                "Children": [
                    {"Text": "Memory Used", "Value": f"{ram_used_gb} GB",
                     "Children": []},
                    {"Text": "Memory Available",
                     "Value": f"{ram_total_gb - ram_used_gb} GB",
                     "Children": []},
                ],
            },
        ],
    }


def _make_fetcher(tree: dict):
    """Return a fetcher closure that always serves ``tree``."""
    payload = json.dumps(tree).encode("utf-8")

    def _fetch(_url: str, _timeout_s: float) -> bytes:
        return payload

    return _fetch


# ---------------------------------------------------------------------
# module contract
# ---------------------------------------------------------------------


def test_module_imports_torch_free(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        class _Blocker:
            def find_module(self, name, path=None): return self if name == "torch" else None
            def load_module(self, name): raise ImportError("torch blocked")
        sys.meta_path.insert(0, _Blocker())
        from otr_v2.hyworld import lhm_monitor as lm  # noqa: F401
        assert hasattr(lm, "poll_once")
        assert hasattr(lm, "poll_loop")
        assert hasattr(lm, "summarize")
        assert hasattr(lm, "summarize_ndjson")
        assert hasattr(lm, "VRAM_CEILING_GB")
        print("OK")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert r.stdout.strip() == "OK"


def test_day_13_ceiling_constants() -> None:
    assert lm.VRAM_CEILING_GB == 14.5
    assert lm.RAM_CEILING_GB == 28.0
    assert lm.GPU_TEMP_CEILING_C == 85.0


# ---------------------------------------------------------------------
# poll_once
# ---------------------------------------------------------------------


def test_poll_once_extracts_all_four_metrics() -> None:
    tree = _fake_lhm_tree(
        gpu_temp_c=72.5, vram_used_gb=6.8, ram_used_gb=14.2, cpu_temp_c=61.0,
    )
    sample = lm.poll_once(
        fetcher=_make_fetcher(tree),
        now_mono=100.0, now_unix=1700000000.0,
    )
    assert sample.gpu_temp_c == pytest.approx(72.5)
    assert sample.gpu_vram_used_gb == pytest.approx(6.8)
    assert sample.ram_used_gb == pytest.approx(14.2)
    assert sample.cpu_temp_c == pytest.approx(61.0)
    assert not sample.unreachable
    assert sample.t_monotonic == 100.0
    assert sample.t_unix == 1700000000.0


def test_poll_once_records_unreachable_on_fetch_error() -> None:
    def _raising(_url: str, _timeout_s: float) -> bytes:
        raise OSError("simulated connection refused")

    sample = lm.poll_once(fetcher=_raising, now_mono=1.0, now_unix=2.0)
    assert sample.unreachable is True
    assert "network:OSError" in sample.reason
    assert sample.gpu_temp_c is None


def test_poll_once_records_unreachable_on_malformed_json() -> None:
    def _garbage(_url: str, _timeout_s: float) -> bytes:
        return b"not valid json {"

    sample = lm.poll_once(fetcher=_garbage, now_mono=1.0, now_unix=2.0)
    assert sample.unreachable is True
    assert "parse" in sample.reason or "error" in sample.reason


def test_poll_once_handles_mb_units() -> None:
    tree = _fake_lhm_tree(
        gpu_temp_c=60.0, vram_used_gb=5.0, ram_used_gb=10.0,
    )
    # Force VRAM into MB to exercise the GB/MB branch.
    tree["Children"][0]["Children"][1]["Children"][0]["Value"] = "5120 MB"
    sample = lm.poll_once(
        fetcher=_make_fetcher(tree), now_mono=0.0, now_unix=0.0,
    )
    assert sample.gpu_vram_used_gb == pytest.approx(5.0)  # 5120 MB == 5 GB


def test_sample_to_dict_is_json_serialisable() -> None:
    tree = _fake_lhm_tree(
        gpu_temp_c=70.0, vram_used_gb=8.0, ram_used_gb=12.0,
    )
    sample = lm.poll_once(fetcher=_make_fetcher(tree))
    d = sample.to_dict()
    for key in (
        "t_monotonic", "t_unix", "gpu_temp_c", "gpu_vram_used_gb",
        "ram_used_gb", "cpu_temp_c", "unreachable", "reason",
    ):
        assert key in d
    json.dumps(d)


# ---------------------------------------------------------------------
# poll_loop
# ---------------------------------------------------------------------


def test_poll_loop_writes_ndjson_and_returns_samples(tmp_path: Path) -> None:
    out = tmp_path / "lhm.ndjson"
    tree = _fake_lhm_tree(
        gpu_temp_c=70.0, vram_used_gb=8.0, ram_used_gb=12.0,
    )
    # Deterministic clocks: advance by 1 s per tick regardless of sleep.
    t_iter = count(start=0.0, step=1.0)
    samples = lm.poll_loop(
        out,
        interval_s=60.0,
        max_samples=5,
        fetcher=_make_fetcher(tree),
        sleep_fn=lambda _s: None,
        monotonic_fn=lambda: next(t_iter),
        unix_fn=lambda: 1700000000.0,
    )
    assert len(samples) == 5
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        d = json.loads(line)
        assert d["gpu_temp_c"] == pytest.approx(70.0)
        assert d["unreachable"] is False


def test_poll_loop_respects_duration(tmp_path: Path) -> None:
    """Duration limit trumps max_samples when reached first."""
    out = tmp_path / "lhm.ndjson"
    tree = _fake_lhm_tree(
        gpu_temp_c=65.0, vram_used_gb=5.0, ram_used_gb=10.0,
    )

    # Each call to monotonic_fn returns t += 30 s so after 4 polls
    # we've covered 120 s of simulated time.
    state = {"t": 0.0}

    def _mono() -> float:
        return state["t"]

    def _sleep(_s: float) -> None:
        state["t"] += 30.0  # fixture: 30 s per sleep call

    samples = lm.poll_loop(
        out,
        interval_s=30.0,
        duration_s=100.0,  # Should stop after ~4 polls (3 intervals)
        max_samples=100,
        fetcher=_make_fetcher(tree),
        sleep_fn=_sleep,
        monotonic_fn=_mono,
        unix_fn=lambda: 0.0,
    )
    assert 3 <= len(samples) <= 5


def test_poll_loop_respects_stop_when(tmp_path: Path) -> None:
    out = tmp_path / "lhm.ndjson"
    tree = _fake_lhm_tree(
        gpu_temp_c=65.0, vram_used_gb=5.0, ram_used_gb=10.0,
    )
    # stop_when trips on the 3rd check.
    calls = {"n": 0}

    def _stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    samples = lm.poll_loop(
        out,
        interval_s=1.0,
        max_samples=100,
        stop_when=_stop,
        fetcher=_make_fetcher(tree),
        sleep_fn=lambda _s: None,
        monotonic_fn=lambda: 0.0,
        unix_fn=lambda: 0.0,
    )
    # 2 polls before stop_when returns True on the 3rd pre-check.
    assert len(samples) == 2


def test_poll_loop_rejects_nonpositive_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lm.poll_loop(tmp_path / "x.ndjson", interval_s=0.0)


# ---------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------


def test_summarize_empty_returns_note() -> None:
    summary = lm.summarize([])
    assert summary.n_samples == 0
    assert any("no samples" in n for n in summary.notes)


def test_summarize_peak_mean_min() -> None:
    def _s(t, temp, vram, ram) -> lm.LhmSample:
        return lm.LhmSample(
            t_monotonic=t, t_unix=t,
            gpu_temp_c=temp, gpu_vram_used_gb=vram, ram_used_gb=ram,
        )

    samples = [
        _s(0.0, 50.0, 4.0, 10.0),
        _s(60.0, 75.0, 8.0, 15.0),
        _s(120.0, 65.0, 6.0, 12.0),
    ]
    summary = lm.summarize(samples)
    assert summary.n_samples == 3
    assert summary.gpu_temp_c["peak"] == pytest.approx(75.0)
    assert summary.gpu_temp_c["min"] == pytest.approx(50.0)
    assert summary.gpu_temp_c["mean"] == pytest.approx((50 + 75 + 65) / 3)
    assert summary.gpu_vram_used_gb["peak"] == pytest.approx(8.0)
    assert summary.ram_used_gb["peak"] == pytest.approx(15.0)
    assert summary.duration_s == pytest.approx(120.0)
    assert summary.vram_ceiling_breached is False
    assert summary.ram_ceiling_breached is False
    assert summary.gpu_temp_ceiling_breached is False


def test_summarize_flags_vram_ceiling_breach() -> None:
    s = lm.LhmSample(t_monotonic=0, t_unix=0, gpu_vram_used_gb=15.0)
    summary = lm.summarize([s])
    assert summary.vram_ceiling_breached is True
    assert any("VRAM ceiling" in n for n in summary.notes)


def test_summarize_flags_ram_ceiling_breach() -> None:
    s = lm.LhmSample(t_monotonic=0, t_unix=0, ram_used_gb=29.0)
    summary = lm.summarize([s])
    assert summary.ram_ceiling_breached is True


def test_summarize_flags_gpu_temp_ceiling_breach() -> None:
    s = lm.LhmSample(t_monotonic=0, t_unix=0, gpu_temp_c=88.0)
    summary = lm.summarize([s])
    assert summary.gpu_temp_ceiling_breached is True


def test_summarize_counts_unreachable() -> None:
    samples = [
        lm.LhmSample(t_monotonic=0, t_unix=0, unreachable=True, reason="network:OSError"),
        lm.LhmSample(t_monotonic=60, t_unix=0, gpu_temp_c=60.0, gpu_vram_used_gb=5.0, ram_used_gb=10.0),
        lm.LhmSample(t_monotonic=120, t_unix=0, unreachable=True, reason="parse:ValueError"),
    ]
    summary = lm.summarize(samples)
    assert summary.n_unreachable == 2
    assert summary.n_samples == 3


def test_summarize_ndjson_roundtrip(tmp_path: Path) -> None:
    """poll_loop output feeds summarize_ndjson losslessly."""
    out = tmp_path / "lhm.ndjson"
    tree = _fake_lhm_tree(
        gpu_temp_c=80.0, vram_used_gb=14.0, ram_used_gb=22.0,
    )
    t_iter = count(start=0.0, step=10.0)
    lm.poll_loop(
        out,
        interval_s=10.0,
        max_samples=3,
        fetcher=_make_fetcher(tree),
        sleep_fn=lambda _s: None,
        monotonic_fn=lambda: next(t_iter),
        unix_fn=lambda: 1700000000.0,
    )
    summary = lm.summarize_ndjson(out)
    assert summary.n_samples == 3
    assert summary.gpu_temp_c["peak"] == pytest.approx(80.0)
    assert summary.gpu_vram_used_gb["peak"] == pytest.approx(14.0)
    assert summary.vram_ceiling_breached is False  # 14.0 < 14.5
    assert summary.ram_ceiling_breached is False   # 22.0 < 28.0
    assert summary.gpu_temp_ceiling_breached is False


def test_summarize_ndjson_missing_file(tmp_path: Path) -> None:
    summary = lm.summarize_ndjson(tmp_path / "nonexistent.ndjson")
    assert summary.n_samples == 0
    assert any("log file missing" in n for n in summary.notes)


def test_summary_to_dict_json_serialisable() -> None:
    summary = lm.summarize([
        lm.LhmSample(t_monotonic=0, t_unix=0, gpu_temp_c=70.0,
                     gpu_vram_used_gb=8.0, ram_used_gb=12.0, cpu_temp_c=50.0),
    ])
    d = summary.to_dict()
    for key in (
        "n_samples", "n_unreachable", "duration_s",
        "gpu_temp_c", "gpu_vram_used_gb", "ram_used_gb", "cpu_temp_c",
        "vram_ceiling_breached", "ram_ceiling_breached", "gpu_temp_ceiling_breached",
        "vram_ceiling_gb", "ram_ceiling_gb", "gpu_temp_ceiling_c",
        "notes",
    ):
        assert key in d
    assert d["vram_ceiling_gb"] == 14.5
    assert d["ram_ceiling_gb"] == 28.0
    assert d["gpu_temp_ceiling_c"] == 85.0
    json.dumps(d)
