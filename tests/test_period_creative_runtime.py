"""Sprint D D4 -- runtime-gated period-creative pipeline tests.

Four assertions all gated under OTR_REGRESSION_RUNTIME=1:

  test_period_creative_modern_technical_vram_peak_under_14_5gb
      Full pipeline run with talkie on the creative slot and
      Mistral-Nemo on the technical slot. Asserts BOTH
      torch.cuda.max_memory_allocated AND max_memory_reserved
      stay under 14.5 GB during the slot-swap. Catches caching
      allocator bloat the allocated-only check would miss.

  test_period_creative_stable_across_two_runs_advisory
      @pytest.mark.xfail(strict=False). GPTQ split-K
      nondeterminism is a known property of int4 quantization;
      byte-deterministic output across fixed-seed re-runs is
      NOT achievable. xfail (not skip) so the test STILL RUNS
      under runtime gate and surfaces unexpected stability as a
      genuine improvement signal.

  test_period_creative_diction_guard_no_modernisms_on_no_news_fixture
      Run the period creative path against a fixed no-news fixture
      and regex-assert against modernisms (okay, guys, cool, hey,
      smartphone, internet, etc.). Exercises the diction layer in
      isolation -- no modern news input to muddy the signal.

  test_period_creative_modern_news_warning_emit
      Run against a modern-news fixture (post-1952 references).
      WARNING-only path: if anachronisms appear, emit UserWarning
      with the offending tokens. Documents the known limitation
      without failing the test.

All four sit in the runtime gate. Sprint A or operator-discretion
runs them with HF_TOKEN configured + talkie weights downloaded.
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OTR_REGRESSION_RUNTIME = os.environ.get("OTR_REGRESSION_RUNTIME") == "1"

TALKIE_REPO_ID = "talkie-lm/talkie-1930-13b-it"


# ---------------------------------------------------------------------------
# Fixture stubs (Sprint A lands the actual fixture authoring + writer
# harness; D4 stages the test surfaces).
# ---------------------------------------------------------------------------

NO_NEWS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "no_news_period_seed_v1.json"
MODERN_NEWS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "modern_news_seed_v1.json"


def _require_fixture(path: Path) -> None:
    if not path.is_file():
        pytest.skip(
            f"fixture {path.relative_to(REPO_ROOT)} not yet authored; "
            f"Sprint A precondition. The D4 runtime test stages here."
        )


# Modernism regex set per v3 plan D4. Each pattern surfaces a token
# that is forbidden by OTR_PERIOD_SYSTEM_PROMPT.
MODERNISMS = (
    r"\bokay\b",
    r"\bguys\b",
    r"\bcool\b",
    r"\bhey\b",
    r"\bsmartphone\b",
    r"\binternet\b",
    r"\bemail\b",
    r"\btext message\b",
)


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
def test_period_creative_modern_technical_vram_peak_under_14_5gb() -> None:
    """Full talkie creative + Mistral-Nemo technical pipeline run.
    Asserts peak allocated AND peak reserved both stay under
    14.5 GB during the slot-swap.
    """
    import torch  # noqa: PLC0415
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; runtime smoke needs GPU")
    pytest.skip(
        "Sprint A runtime harness for full-pipeline run not yet "
        "wired. Stages here at D4. Implementation: invoke writer "
        "with creative=talkie + technical=Mistral-Nemo + capture "
        "torch.cuda.max_memory_{allocated,reserved}() pre and post."
    )
    # When Sprint A lands the harness:
    # from <writer_runtime_harness> import run_full_pipeline
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # run_full_pipeline(creative=TALKIE_REPO_ID, technical=DEFAULT_LLM)
    # peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    # peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    # assert peak_allocated_gb < 14.5
    # assert peak_reserved_gb < 14.5


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
@pytest.mark.xfail(
    strict=False,
    reason="GPTQ int4 split-K kernel nondeterminism; byte-stability "
           "across fixed-seed runs is not guaranteed. xfail strict=False "
           "so stable runs surface as a genuine improvement signal.",
)
def test_period_creative_stable_across_two_runs_advisory() -> None:
    """Two runs at the same seed. Advisory-stability check.
    Marked xfail(strict=False) per v3 plan -- not a sprint blocker.
    """
    pytest.skip(
        "Sprint A runtime harness for fixed-seed talkie generation "
        "not yet wired. Stages here at D4."
    )
    # When Sprint A lands:
    # from <writer_runtime_harness> import run_fixed_seed_creative
    # out_a = run_fixed_seed_creative(repo_id=TALKIE_REPO_ID, seed=42)
    # out_b = run_fixed_seed_creative(repo_id=TALKIE_REPO_ID, seed=42)
    # assert out_a == out_b


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
def test_period_creative_diction_guard_no_modernisms_on_no_news_fixture() -> None:
    """Period creative output on a fixed no-news seed must not
    contain modernisms. Exercises the diction layer in isolation.
    """
    _require_fixture(NO_NEWS_FIXTURE)
    pytest.skip(
        "Sprint A runtime harness for fixture-based talkie creative "
        "run not yet wired. Stages here at D4. Implementation: "
        "invoke writer with the no-news fixture as seed, capture "
        "generated dialogue, sweep MODERNISMS regex."
    )
    # When Sprint A lands:
    # from <writer_runtime_harness> import run_period_creative_against_fixture
    # out = run_period_creative_against_fixture(NO_NEWS_FIXTURE)
    # for pattern in MODERNISMS:
    #     assert not re.search(pattern, out, re.IGNORECASE), (
    #         f"period creative produced modernism {pattern!r} on "
    #         f"the no-news fixture"
    #     )


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
def test_period_creative_modern_news_warning_emit() -> None:
    """Run against a modern-news fixture. Warning-only: if
    post-1952 anachronisms appear, emit UserWarning naming the
    tokens. Documents the limitation without failing.
    """
    _require_fixture(MODERN_NEWS_FIXTURE)
    pytest.skip(
        "Sprint A runtime harness for modern-news fixture run not "
        "yet wired. Stages here at D4. Implementation: run period "
        "creative against the modern-news fixture, scan output for "
        "post-1952 token patterns, warnings.warn on hit."
    )
    # When Sprint A lands:
    # from <writer_runtime_harness> import run_period_creative_against_fixture
    # out = run_period_creative_against_fixture(MODERN_NEWS_FIXTURE)
    # anachronisms = _find_post_1952_tokens(out)
    # if anachronisms:
    #     warnings.warn(
    #         f"period creative produced post-1952 tokens "
    #         f"{anachronisms} when given modern news input",
    #         UserWarning,
    #     )
