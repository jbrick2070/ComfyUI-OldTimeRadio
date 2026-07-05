"""AST drift-proofing: pinned production shapes vs production_mirror.

Never imports mirrored modules (the mirror is dependency-incomplete by
design). When production moves and the mirror is refreshed, these fail with
re-pin messages instead of letting the bridge rot silently.
"""

from __future__ import annotations

import pytest

from upstream_story_lab.compat import (
    MOTION_ROLE_KEYS,
    NEWS_BRIEFS_FIELDS,
    NEWS_SEED_KEYS,
    NEWS_SEED_REQUIRED_KEYS,
    CompatDriftError,
    PRODUCTION_VISUAL_TAILS,
    canonical_json_hash,
    extract_motion_role_keys,
    extract_news_briefs_fields,
    extract_news_seed_keys,
    extract_visual_tails,
)


def test_news_briefs_fields_match_mirror(mirror_nodes) -> None:
    extracted = extract_news_briefs_fields(mirror_nodes / "news_interpreter.py")
    assert tuple(extracted) == NEWS_BRIEFS_FIELDS, (
        "NewsBriefs drifted - re-verify meta.news mirror and re-pin compat.py"
    )


def test_news_seed_keys_match_mirror(mirror_nodes) -> None:
    extracted = extract_news_seed_keys(
        mirror_nodes / "_otr_legacy_to_stage1_adapter.py"
    )
    assert tuple(extracted) == NEWS_SEED_KEYS
    assert set(NEWS_SEED_REQUIRED_KEYS).issubset(extracted)


def test_motion_role_keys_match_mirror(mirror_nodes) -> None:
    extracted = extract_motion_role_keys(
        mirror_nodes / "_otr_video_engines" / "render_driver.py"
    )
    assert tuple(extracted) == MOTION_ROLE_KEYS
    dead = {"sfx", "scene_broll", "background_abstract"}
    assert not dead.intersection(extracted), "a dead role came back"


def test_visual_tails_match_mirror_byte_identically(mirror_nodes) -> None:
    extracted = extract_visual_tails(mirror_nodes / "_otr_story_brief_helpers.py")
    assert extracted == PRODUCTION_VISUAL_TAILS


def test_extractors_fail_pointedly_on_missing_mirror(tmp_path) -> None:
    with pytest.raises(CompatDriftError, match="mirror file missing"):
        extract_news_briefs_fields(tmp_path / "nope.py")


def test_canonical_hash_ignores_whitespace_and_order() -> None:
    a = {"b": 1, "a": [1, 2]}
    b = {"a": [1, 2], "b": 1}
    assert canonical_json_hash(a) == canonical_json_hash(b)
