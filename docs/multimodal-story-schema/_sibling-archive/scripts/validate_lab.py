"""Headless validation of the upstream story lab v2 (CI/sandbox use, no
ComfyUI required - kept per kibitz r1 judgment)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from upstream_story_lab.bridge import build_bridge_artifact, build_spec  # noqa: E402
from upstream_story_lab.compat import (  # noqa: E402
    MOTION_ROLE_KEYS,
    NEWS_BRIEFS_FIELDS,
    NEWS_SEED_KEYS,
    PRODUCTION_VISUAL_TAILS,
    extract_motion_role_keys,
    extract_news_briefs_fields,
    extract_news_seed_keys,
    extract_visual_tails,
)
from upstream_story_lab.preview import (  # noqa: E402
    scan_story_leakage,
    scan_visual_leakage,
)
from upstream_story_lab.registry import Registry  # noqa: E402


def main() -> int:
    registry = Registry(ROOT)
    mirror = ROOT / "production_mirror" / "nodes"

    assert tuple(extract_news_briefs_fields(mirror / "news_interpreter.py")) == NEWS_BRIEFS_FIELDS
    assert tuple(extract_news_seed_keys(mirror / "_otr_legacy_to_stage1_adapter.py")) == NEWS_SEED_KEYS
    assert tuple(extract_motion_role_keys(mirror / "_otr_video_engines" / "render_driver.py")) == MOTION_ROLE_KEYS
    assert extract_visual_tails(mirror / "_otr_story_brief_helpers.py") == PRODUCTION_VISUAL_TAILS

    # Full declared matrix (kibitz r3, Codex S2): every runnable pack x every
    # style - the same coverage the pytest matrix asserts, so a green here
    # really is matrix-green.
    specs = 0
    for (bank_id, model_id, pipeline_id), _ in sorted(registry.packs.items()):
        if bank_id == "custom_source_bank":
            continue
        for style_id in sorted(registry.styles):
            spec = build_spec(
                registry, source_bank_id=bank_id, story_model_id=model_id,
                story_pipeline_id=pipeline_id, visual_style_id=style_id,
            )
            if bank_id != "science_news":
                leaked = scan_story_leakage(registry, spec)
                assert not leaked, f"{bank_id}/{model_id} leaked {leaked}"
            leaked_visual = scan_visual_leakage(spec)
            assert not leaked_visual, (
                f"{bank_id}/{model_id}/{style_id} visual leaked {leaked_visual}"
            )
            build_bridge_artifact(spec)
            specs += 1

    manifests = registry.public_domain_manifests()

    print("OK upstream_story_lab v2")
    print(f"banks={sorted(registry.banks)}")
    print(f"story_packs={len(registry.packs)}")
    print(f"validated_specs={specs}")
    print(f"visual_styles={sorted(registry.styles)}")
    print(f"public_domain_manifests={len(manifests)}")
    print("mirror_drift=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
