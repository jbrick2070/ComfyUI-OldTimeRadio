"""Freeze-policy resolution for the six live story banks.

The policy decides who may perform the bounded same-story safety cleanup. It
never converts word count, visual vocabulary, style, or craft observations into
an episode veto.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as LFC
from nodes import _otr_story_routing as ROUTING


@pytest.fixture(autouse=True)
def _fresh_registry():
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None


@pytest.mark.parametrize(
    "bank_id",
    ["media_archive", "original", "public_domain", "shakespeare"],
)
def test_inline_banks_receive_same_story_cleanup_policy(bank_id):
    policy = LFC.resolve_freeze_policy({"source_bank": bank_id})
    assert policy.name == "inline_safety_cleanup"
    assert policy.run_inline_safety_cleanup is True
    assert policy.terminal_error == ""


@pytest.mark.parametrize("bank_id", ["scifi_news", "scifi_news_pro"])
def test_fixed_topology_banks_are_content_owned(bank_id):
    policy = LFC.resolve_freeze_policy({"source_bank": bank_id})
    assert policy.name == "content_owned_readonly"
    assert policy.run_inline_safety_cleanup is False
    assert policy.terminal_error == ""


def test_untagged_ledger_uses_inline_compatibility_policy():
    policy = LFC.resolve_freeze_policy({})
    assert policy.name == "inline_safety_cleanup"
    assert policy.run_inline_safety_cleanup is True


def test_unknown_declared_bank_fails_as_structural_configuration():
    policy = LFC.resolve_freeze_policy({"source_bank": "no_such_bank"})
    assert policy.name == "policy_resolution_failed"
    assert policy.run_inline_safety_cleanup is False
    assert policy.terminal_error
