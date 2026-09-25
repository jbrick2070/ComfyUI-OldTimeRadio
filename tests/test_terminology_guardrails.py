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
        "my_story",
        "custom_source_bank",
    ]


