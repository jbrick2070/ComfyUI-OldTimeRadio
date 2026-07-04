"""nodes/_otr_period_prompts.py

1940s old-time-radio period-flavored LLM prompts.

The script writer LLM (Mistral-Nemo 12B per the OTR default) renders
modern English by default; OTR is supposed to feel like a transcript
of a 1940s anthology broadcast. This module gives that feeling
without swapping models or growing VRAM:

    - A reusable system prompt that imprints the period's diction,
      pacing, and broadcast conventions.
    - A small library of few-shot exemplars drawn from the public-
      domain corpus of OTR shows (Suspense, Lights Out, X Minus One,
      Inner Sanctum) reduced to short illustrative blocks.
    - A simple ``render_prompt(...)`` helper that assembles the
      system prompt + few-shot block + the runtime instruction the
      orchestrator already builds.

Status: pluggable. Designed to drop into the existing LLM call site
in ``story_orchestrator.py`` without touching the orchestrator's
control flow -- the orchestrator just imports
``OTR_PERIOD_SYSTEM_PROMPT`` (or calls ``render_prompt``) when
constructing its messages list. This module ships tonight as new code
behind the scenes; the integration step happens in the next session
once the in-flight FULL acceptance soak completes.

Module is stdlib-only -- no torch, no LLM HTTP, no Comfy graph
imports.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- The system prompt -------------------------------------------------

OTR_PERIOD_SYSTEM_PROMPT = """\
You are writing a 1940s American anthology radio drama in the spirit of
Suspense, Lights Out, Inner Sanctum, and Escape. Every line must read
the way it would sound coming through a single mono speaker on a tabletop
console radio in 1947. Follow these rules without exception:

DICTION
- Use period-natural American English from roughly 1938-1952. Words and
  phrases that did not exist in that decade are forbidden, including
  "okay" (use "all right"), "guys" (use "fellows" or names), "cool" as
  approval (use "swell" or "fine"), "hey" as a greeting (use "say" or
  the person's name), and any modern technology term (no phones unless
  rotary, no computers, no radar references unless military-correct).
- Sentence rhythm runs slightly formal, slightly clipped. Characters
  speak in complete sentences but lean on contractions naturally
  ("I've", "you'll", "don't"). They almost never trail off; they
  finish what they started.

BROADCAST CONVENTION
- The host is THE NARRATOR. Open with one paragraph of NARRATOR setup,
  scene-setting, and stakes. Close with one paragraph of NARRATOR
  outro that lands the moral or the lingering question.
- Stage directions go in [square brackets] and are functional only:
  [SOUND: door creaks open], [MUSIC: ominous brass sting]. Never
  describe character emotion in stage directions; show emotion through
  the line itself.
- Every spoken line is tagged CHARACTER:dialogue. The tag is one
  uppercase word, optionally with one period for "MR." or "DR." style
  honorifics: NARRATOR, MONTGOMERY, DR. KASKE, MISS WELLS. Do NOT use
  lowercase, do NOT use brackets around character names, do NOT prefix
  with numbers.

CONSTRAINTS
- Family-broadcast safe. No profanity, no sexual content, no graphic
  violence. Tension comes from menace, dread, and consequence -- not
  from gore.
- A complete arc in three parts: setup, confrontation, resolution.
  Even when the resolution is unsettling, it must be reached on the
  page, not deferred to a "to be continued."
- No Modernity. No references to events after 1952. No cell phones,
  no computers, no internet, no jet airliners, no Cold War vocabulary
  beyond what was in print at the time.

Stay inside this voice for every line you produce. If a request would
require you to break period, substitute the closest period-correct
analogue. Every dialogue line must read aloud cleanly through a
1940s tube radio.
"""


# --- Few-shot exemplars ------------------------------------------------


@dataclass(frozen=True)
class PeriodExemplar:
    """One short illustrative block of period dialogue.

    Kept short on purpose: few-shot blocks pay quadratic context cost
    in the LLM call, and Mistral-Nemo 12B's effective context cap
    leaves limited room for both exemplars and the actual outline.
    Each exemplar is roughly 6-10 lines of dialogue + minimum stage
    directions.
    """

    title: str
    show_inspiration: str  # e.g. "Suspense", "Lights Out"
    body: str


PERIOD_EXEMPLARS: list[PeriodExemplar] = [
    PeriodExemplar(
        title="The Lighthouse Keeper",
        show_inspiration="Suspense",
        body=(
            "NARRATOR: It was the night the storm came in off the bay, and "
            "Henry Cole sat alone in the lamphouse with a question he could "
            "not put down.\n"
            "[SOUND: wind battering glass]\n"
            "HENRY: Strange how a light that's burned for forty years can "
            "still find new ways to flicker.\n"
            "MARGARET: You're tired, Henry. Come down. Have your supper.\n"
            "HENRY: Not yet. The bell-buoy went silent at quarter past, and "
            "I haven't heard it pick back up.\n"
            "MARGARET: It's the storm, that's all.\n"
            "HENRY: It is, until it isn't. That's the trouble with this "
            "coast -- you never quite know which.\n"
        ),
    ),
    PeriodExemplar(
        title="The Wireless",
        show_inspiration="X Minus One",
        body=(
            "NARRATOR: Doctor Avery had, against the better judgment of "
            "every colleague who would still answer his calls, built the "
            "thing in his cellar.\n"
            "DR. AVERY: It receives, Miss Holt. It receives perfectly. The "
            "trouble is what it receives.\n"
            "MISS HOLT: I don't follow.\n"
            "DR. AVERY: The signal is dated next Tuesday.\n"
            "[SOUND: faint static, slow tone underneath]\n"
            "MISS HOLT: Doctor -- shut it off.\n"
            "DR. AVERY: I tried, last night. It plays whether the power is "
            "on or not.\n"
        ),
    ),
    PeriodExemplar(
        title="Last Train Out",
        show_inspiration="Escape",
        body=(
            "NARRATOR: Every Friday at five past eleven, the night train "
            "left Union Station for points west. Every Friday but one.\n"
            "CONDUCTOR: Ticket, please.\n"
            "STRANGER: I haven't got one.\n"
            "CONDUCTOR: Sir, you'll need to step off at the next platform.\n"
            "STRANGER: There won't be a next platform. Not for this train.\n"
            "[MUSIC: low brass sting]\n"
            "CONDUCTOR: Are you threatening me?\n"
            "STRANGER: I'm warning you. There's a difference, and I haven't "
            "got time to explain it.\n"
        ),
    ),
]


def render_few_shot_block(
    exemplars: list[PeriodExemplar] | None = None,
    max_exemplars: int = 2,
) -> str:
    """Render a few-shot block suitable to splice into the system or
    user prompt.

    Defaults to the first two exemplars (Lighthouse + Wireless) which
    together demonstrate both small-cast tension and a sci-fi twist
    -- the two modes OTR most commonly produces. Pass a custom list
    if a particular episode wants a different tonal anchor.

    Caps at ``max_exemplars`` so a caller can dial down the prompt
    cost on tight context budgets.
    """
    chosen = (exemplars if exemplars is not None else PERIOD_EXEMPLARS)[:max_exemplars]
    blocks: list[str] = []
    for ex in chosen:
        blocks.append(
            f"--- Exemplar: {ex.title} (in the spirit of {ex.show_inspiration}) ---\n"
            f"{ex.body.rstrip()}\n"
        )
    return "\n".join(blocks).rstrip()


def render_prompt(
    user_instruction: str,
    include_few_shot: bool = True,
    max_exemplars: int = 2,
    system_prompt: str = OTR_PERIOD_SYSTEM_PROMPT,
) -> tuple[str, str]:
    """Compose ``(system, user)`` strings ready to drop into a chat-
    style LLM call (the same shape Mistral-Nemo + most ladder models
    expect).

    Returns:
        (system_prompt, user_prompt) -- two strings.

    The system_prompt is always the period system prompt (callers
    can override). The user_prompt prefixes the few-shot block (when
    requested) and then the runtime instruction the orchestrator
    already constructs.
    """
    if include_few_shot:
        few_shot = render_few_shot_block(max_exemplars=max_exemplars)
        user_prompt = f"{few_shot}\n\n--- Now write the requested episode ---\n\n{user_instruction.strip()}"
    else:
        user_prompt = user_instruction.strip()
    return system_prompt, user_prompt


__all__ = [
    "OTR_PERIOD_SYSTEM_PROMPT",
    "PeriodExemplar",
    "PERIOD_EXEMPLARS",
    "render_few_shot_block",
    "render_prompt",
]
