from pathlib import Path

from nodes import _otr_story_routing


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_source_bank_selector_has_only_canonical_ids():
    assert list(_otr_story_routing.list_bank_ids()) == [
        "media_archive",
        "original",
        "scifi_news_pro",
        "public_domain",
        "shakespeare",
        "custom_source_bank",
    ]


def test_active_operator_surfaces_use_bank_names():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    plan = (REPO_ROOT / "docs" / "GO_FORWARD_PLAN.md").read_text(
        encoding="utf-8"
    )
    assert "NewsPro lane" not in readme
    assert "Codex lane" not in readme
    assert "Sci-Fi Codex reverify tail" not in plan
    assert "NewsPro C5 consumers" not in plan
    assert "canonical scifi_news_pro full-media qualification leg" not in plan
