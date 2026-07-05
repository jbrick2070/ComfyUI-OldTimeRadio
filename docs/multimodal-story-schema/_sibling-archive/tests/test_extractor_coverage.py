"""Table-driven coverage: for every (bank, model, pipeline, seam) tuple in
the loaded registry, extractor returns str for populated seams, None for
absent/empty."""
from pathlib import Path

from upstream_story_lab.contracts import PRODUCTION_TEMPLATE_SEAMS
from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def test_extractor_coverage_all_packs():
    reg = Registry(ROOT)
    for pack_key in reg.packs:
        pack, _path = reg.packs[pack_key]
        for seam in PRODUCTION_TEMPLATE_SEAMS:
            got = get_pack_prompt_or_none(reg, *pack_key, seam)
            raw = pack.prompt_stages.get(seam, "").strip()
            expected = raw or None
            assert got == expected, (
                f"{pack_key} seam {seam}: got={got!r} expected={expected!r}"
            )
