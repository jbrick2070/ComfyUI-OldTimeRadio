# -*- coding: utf-8 -*-
"""No VRAM arithmetic may refuse a load. An OOM is the only authority.

Operator directive, restated 2026-09-21: "I don't want any VRAM guards."
It is the same rule as 2026-08-29's "prove me wrong with an OOM but don't put
an artificial gate", and the reason is measured rather than stylistic: every
one of these estimates has been WRONG in the direction that refuses working
configurations --

  * a since-removed writer backend's estimate priced the whole file on the
    GPU and ignored its own partial-offload setting, so a 12B was refused
    at a configuration the card ran;
  * the fit gate priced a Q4_K_M as a Q8_0 (PBUG-20260829-08) and a 4096
    request at the row's declared 8192, refusing a load the card had
    already performed with all 48 layers resident;
  * the probe refusal killed the episode when the READING failed, which is
    not evidence about whether the model fits.

Two kinds of check live here. The AST tripwire asserts the VRAM path holds
no `raise`, so a future "small safety check" cannot quietly reintroduce a
gate. The behavioural test CALLS the admission gate with a model priced well
over its ceiling and asserts it still returns, because a review caught an
earlier test re-implementing the arithmetic inline and grading its own
copy -- it would have stayed green while the real code regained a guard.

KNOWN LIMIT of the tripwire, worth stating rather than pretending: it
looks at the lines around each `raise`, so a gate reintroduced inside a
separately-named helper would evade it. The behavioural tests are the
real protection; the tripwire is the cheap early warning.
"""
import ast
import pathlib

PACK = pathlib.Path(__file__).resolve().parents[1]
LOADER = PACK / "nodes" / "_otr_model_loader.py"


def _raises_inside_vram_block(path: pathlib.Path) -> list:
    """Every `raise` whose enclosing lines mention a VRAM reading."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        lo = max(0, node.lineno - 14)
        window = "\n".join(lines[lo:node.lineno]).lower()
        # The loader spells it check_vram_fit / VRAMFitFailedError, not
        # "vram-fit" -- a review caught that the hyphenated spelling matched
        # nothing there, so a reintroduced raise in the fit gate would have
        # slipped straight past this tripwire.
        if any(k in window for k in (
                "mem_get_info", "free_gb", "vram preflight", "vram-fit",
                "check_vram_fit", "vramfitfailederror", "fit_verdict",
                "vram_ceiling_gb", "estimated_needed_gb", "estimated_gb")):
            found.append((path.name, node.lineno,
                          lines[node.lineno - 1].strip()[:80]))
    return found


def test_the_fit_gate_never_raises():
    offenders = _raises_inside_vram_block(LOADER)
    assert offenders == [], (
        "the VRAM-fit estimate is a recommendation, not a capability "
        "refusal. Offending raises: %r" % (offenders,))


def test_the_gate_reports_FAIL_without_refusing():
    """THE BEHAVIOURAL HALF, on the gate that survives.

    A FAIL verdict is a recommendation. `_assert_policy_admits_vram` must log
    it and RETURN -- the runtime's own OOM is the only authority on whether a
    model fits. Called for real rather than re-derived, because a test that
    recomputes the arithmetic grades its own copy and stays green while the
    real code regains a guard.
    """
    import types

    from nodes import _otr_model_catalog as cat
    from nodes._otr_model_loader import _assert_policy_admits_vram

    verdict = cat.check_vram_fit(
        "meta-llama/Meta-Llama-3-70B-Instruct", 8192,
        ceiling_gb=6.8, safetensors_gb_hint=140.0)
    assert verdict.tier == "FAIL", (
        "fixture no longer prices as FAIL (%s at %s GB), so this test would "
        "pass without exercising the refusal path at all"
        % (verdict.tier, verdict.estimated_gb))

    policy = types.SimpleNamespace(vram_ceiling_gb=6.8)
    ctx = types.SimpleNamespace(value=8192, tier="UNKNOWN")
    # The assertion IS that this returns.
    _assert_policy_admits_vram(
        "meta-llama/Meta-Llama-3-70B-Instruct", ctx, policy)
