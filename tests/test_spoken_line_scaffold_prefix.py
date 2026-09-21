# -*- coding: utf-8 -*-
"""A reasoning scaffold word must never be read aloud.

Measured 2026-09-21 across 61 episodes and 955 spoken lines: three announcer
lines shipped beginning with a bare `thought`, in Spanish, Portuguese and
Japanese episodes. A voice actor reads that as the English word "thought"
before the line starts. It is response transport, like the `[VOICE: ...]` tag,
so it is stripped in the same place.

The rule is case-based on purpose. A model emits the scaffold in lower case; a
real sentence capitalises its opening word. So `thought A luz...` is transport
and `Thought is the enemy here` is prose, and the second must survive.
"""
import pytest

from nodes._otr_line_composer import strip_line_formatting

LABELS = ("ANNOUNCER",)


@pytest.mark.parametrize("raw,expected", [
    # All three shipped to air.
    ("thought A luz do projetor finalmente se apaga, deixando apenas o nome.",
     "A luz do projetor finalmente se apaga, deixando apenas o nome."),
    ("thought Tras la devoción poética de un nombre en la arboleda sombría.",
     "Tras la devoción poética de un nombre en la arboleda sombría."),
    ("thought ヴェローナの夜に誓われた名もなき愛から",
     "ヴェローナの夜に誓われた名もなき愛から"),
    # An explicit separator is transport whatever the case.
    ("analysis: The lamp fails at midnight.", "The lamp fails at midnight."),
    ("thought: Nightfall.", "Nightfall."),
    ("reasoning - the keeper waits.", "the keeper waits."),
])
def test_a_scaffold_prefix_is_removed(raw, expected):
    assert strip_line_formatting(raw, LABELS) == expected


@pytest.mark.parametrize("line", [
    # Authored prose that happens to open with one of those words. Losing the
    # first word here would be a worse bug than the one this guard fixes.
    "Thought is the enemy here, and always was.",
    "Thoughts of home kept him awake.",
    "Analysis paralysis gripped the crew.",
    "Thinking men do not sail in this weather.",
    "Good evening. This is SIGNAL LOST.",
    # A QA pass caught these: the separator branch used to ignore case and
    # amputated the opening clause of ordinary prose.
    "Analysis: a word he despised, she said.",
    "Reasoning: it was the only way out.",
    "Thought-provoking silence filled the room.",
    "Thought — unbidden — filled her mind.",
])
def test_authored_prose_keeps_its_first_word(line):
    assert strip_line_formatting(line, LABELS) == line
