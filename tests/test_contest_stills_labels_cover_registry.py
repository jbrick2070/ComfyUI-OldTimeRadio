"""Every registered still engine has a label in the contest app.

The Radio Play Studio builds its stills dropdown from the live image registry, and a
choice with no ``choice.stills.<id>`` key is shown as the raw engine id. The i18n tests
check a fixture list, not the registry, so a new engine could ship unlabeled and nothing
failed (found by QA on qwen_image_21, 2026-11-22). This is that guard.
"""
from __future__ import annotations

import json
from pathlib import Path

from nodes._otr_image_engines import registry as ireg

_I18N = (Path(__file__).resolve().parents[1] / "contest" / "radio-play-studio"
         / "radio_play_studio" / "static" / "i18n")


def _keys(lang):
    return json.loads((_I18N / ("%s.json" % lang)).read_text("utf-8"))


def test_every_registered_still_engine_has_an_english_label():
    en = _keys("en")
    missing = [n for n in ireg.all_engine_names() if "choice.stills.%s" % n not in en]
    assert not missing, ("registered still engines with no contest label "
                         "(add choice.stills.<id> to every language file): %s" % missing)


def test_every_language_labels_the_same_still_engines_as_english():
    en = {k for k in _keys("en") if k.startswith("choice.stills.")}
    for path in sorted(_I18N.glob("*.json")):
        keys = {k for k in json.loads(path.read_text("utf-8"))
                if k.startswith("choice.stills.")}
        assert keys == en, "%s differs from en.json: %s" % (path.name, sorted(keys ^ en))
