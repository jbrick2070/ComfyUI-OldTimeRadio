"""Phase A Chunk 3 -- extractor helper failure-mode coverage.

Structural failures (unknown seam, unknown triple) must raise RegistryError.
Happy-path coverage lives in test_extractor_coverage.py (Chunk 4).
"""
from pathlib import Path

from upstream_story_lab.contracts import PRODUCTION_TEMPLATE_SEAMS
from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry, RegistryError

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_seam_raises():
    reg = Registry(ROOT)
    packs = list(reg.packs)
    bank, model, pipeline = packs[0]
    try:
        get_pack_prompt_or_none(reg, bank, model, pipeline, "not_a_seam")
    except RegistryError:
        return
    raise AssertionError("expected RegistryError for unknown seam")


def test_unknown_bank_raises():
    reg = Registry(ROOT)
    try:
        get_pack_prompt_or_none(reg, "not_a_bank", "not_a_model",
                                "not_a_pipeline",
                                PRODUCTION_TEMPLATE_SEAMS[0])
    except RegistryError:
        return
    raise AssertionError("expected RegistryError for unknown triple")
