"""Sprint 10A step 6-A -- Stage 2 multi-turn roleplay prompt builder.

Per docs/story-generator-final-plan.md Stage 2: per beat, a short
chat conversation walks the LLM into character across acknowledgment
turns, then asks for a single line at Turn 4.

  Turn 1 SYSTEM: 'You are {speaker_name}. {persona}. Reply "Ready"
                  when set.'
  Turn 1 ASSIST: 'Ready.'

  Turn 2 USER:   'Premise: {premise}. Arc so far: {arc}. Confirm.'
  Turn 2 ASSIST: 'Confirmed.'

  Turn 3 USER:   'Established facts: ... You: {persona}. Pronouns:
                  {pronouns}. Confirm.'
  Turn 3 ASSIST: 'Confirmed.'

  Turn 4 USER:   'Previous line ({previous_speaker}): "{previous_line}".
                  Your beat: {intent}. Length: ~{length_target} words.
                  Register: {emotional_register}. Speak your line.
                  Just the line.'
  Turn 4 ASSIST: [DIALOGUE -- this is what ships]

The prompt builder produces the full chat-message list. Only Turn 4's
output lands in the script -- Turns 1-3 are scaffolding: they reshape
the prompt into a chat-native sequence the model handles better than
a single monolithic prompt, AND they give validation breakpoints
(if the assistant doesn't return "Ready" / "Confirmed" at the right
turn the chain is considered broken and the caller can regenerate).

This module is PURE: no LLM, no I/O, no transformers imports. Tests
exercise the message-list shape; the actual generate calls land in
6-B (multi-turn call site + best-of-N selector).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ._otr_stage1_plan import Stage1Arc, Stage1Beat, Stage1CastMember


# ---------------------------------------------------------------------------
# Constants -- acknowledgment-token bank for the chain validators
# ---------------------------------------------------------------------------

# Turn 1 expected reply (case-insensitive). The acknowledgment chain
# is broken if the model returns something semantically different.
TURN1_EXPECTED: str = "ready"

# Turn 2 + Turn 3 expected reply (case-insensitive).
TURN23_EXPECTED: str = "confirmed"

# Accepted-variant patterns -- if the model says "I'm ready", "Yes,
# ready", "Got it" instead of literal "Ready", we still accept it
# (per the plan: 'if the model doesn't return "Ready" / "Confirmed"
# (or a close variant), the chain is broken'). Each pattern is a
# substring or regex applied to the assistant reply lowercased.
TURN1_ACCEPTED_VARIANTS: List[re.Pattern] = [
    re.compile(r"\bready\b"),
    re.compile(r"\bgot it\b"),
    re.compile(r"\bunderstood\b"),
    re.compile(r"\bok(?:ay)?\b"),
    re.compile(r"\byes\b"),
    re.compile(r"\binitial\w*\s+ready\b"),
]

TURN23_ACCEPTED_VARIANTS: List[re.Pattern] = [
    re.compile(r"\bconfirm(?:ed)?\b"),
    re.compile(r"\bunderstood\b"),
    re.compile(r"\bgot it\b"),
    re.compile(r"\bok(?:ay)?\b"),
    re.compile(r"\bagreed\b"),
    re.compile(r"\bnoted\b"),
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Stage2PromptChain:
    """The full Stage 2 multi-turn chat-message list for one beat.

    Consumers feed `system_and_priming_messages` to the LLM to elicit
    Turn 1/2/3 acknowledgments, then feed the FULL message list (with
    those acknowledgments appended as assistant turns) to elicit
    Turn 4 -- the dialogue line that ships.

    For best-of-N: the prompt chain is built ONCE per beat; the
    Turn 4 call is repeated N times with different sampling seeds /
    temperature. Each candidate runs through Stage 3 validators
    (validate_line) and the highest scorer ships.
    """

    # Messages 0..5 (3 user/system + 3 assistant priming turns).
    # Concatenating + a final Turn 4 user message yields the full
    # chat to feed model.generate.
    priming_messages: List[dict]

    # Turn 4 user prompt. Append this + an empty assistant turn to
    # priming_messages to elicit the dialogue line.
    turn4_user: dict

    # Beat metadata carried alongside the messages for the caller's
    # diagnostic logging.
    beat_id: str
    speaker_name: str
    target_words: int
    emotional_register: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _arc_summary(arc: Stage1Arc) -> str:
    """One-line arc summary for Turn 2."""
    return (
        f"Setup: {arc.setup} "
        f"Complication: {arc.complication} "
        f"Resolution: {arc.resolution}"
    )


def _format_running_facts(facts: List[str]) -> str:
    """Bullet-list running facts for Turn 3."""
    if not facts:
        return "(no established facts yet)"
    return "\n".join(f"  - {f}" for f in facts)


def build_stage2_prompt_chain(
    *,
    speaker: Stage1CastMember,
    beat: Stage1Beat,
    arc: Stage1Arc,
    premise: str,
    running_facts: List[str],
    previous_speaker: Optional[str] = None,
    previous_line: Optional[str] = None,
) -> Stage2PromptChain:
    """Assemble the 4-turn Stage 2 chat prompt for one beat.

    Args:
        speaker: the cast member speaking this beat (must NOT be a
            reserved speaker -- announcer/music beats don't use the
            multi-turn chain).
        beat: the Stage1Beat for this line.
        arc: the episode arc spine.
        premise: the episode premise.
        running_facts: the running_facts list, updated through the
            episode.
        previous_speaker: name of the speaker of the immediately
            preceding voiced line. None for the first dialogue line.
        previous_line: text of the immediately preceding voiced
            line. None for the first dialogue line.

    Returns:
        Stage2PromptChain ready for the multi-turn call.
    """
    # Turn 1: system role-bind + leading user prompt for the Ready ack.
    #
    # BUG-LOCAL-280 (2026-05-27, Sprint 10B Wave 0 smoke): Mistral-
    # Nemo's chat template enforces user/assistant alternation AFTER
    # the optional system message ("After the optional system message,
    # conversation roles must alternate user/assistant/..."). The
    # prior shape went system -> assistant directly (system prompt
    # asked for "Ready" + assistant priming reply with no user turn
    # in between), which raised TemplateError on EVERY Stage 2
    # generate. Fix: split the system prompt to drop the "Reply Ready"
    # instruction, and insert a leading USER turn that asks for the
    # Ready ack. Now: system -> user -> assistant -> user -> assistant
    # ... clean alternation.
    turn1_system = (
        f"You are {speaker.name}. {speaker.persona}"
    )

    # Turn 1 user: ask for the Ready ack (was inside the system prompt
    # pre-BUG-LOCAL-280; moved here to preserve role alternation).
    turn1_user = (
        f'Reply with the single word "Ready" when you are set to '
        f"play this character."
    )

    # Turn 1 assistant priming reply (this is what we EXPECT the LLM
    # to produce in a real call; in the prompt-chain object we
    # include it as the canonical priming for prefix-cache reuse).
    turn1_assist = "Ready."

    # Turn 2: premise + arc.
    turn2_user = (
        f"Premise: {premise}\n\n"
        f"Arc so far:\n{_arc_summary(arc)}\n\n"
        f'Reply with the single word "Confirmed" if you understand '
        f"the arc."
    )
    turn2_assist = "Confirmed."

    # Turn 3: established facts + persona + pronouns reinforcement.
    turn3_user = (
        f"Established facts (the dialogue must respect these):\n"
        f"{_format_running_facts(running_facts)}\n\n"
        f"Reminder of who you are:\n"
        f"  - Name: {speaker.name}\n"
        f"  - Gender: {speaker.gender}\n"
        f"  - Pronouns: {speaker.pronouns}\n"
        f"  - Role in this arc: {speaker.arc_role}\n"
        f"  - Voice texture: {speaker.persona}\n\n"
        f'Reply with the single word "Confirmed" if you have these '
        f"settled."
    )
    turn3_assist = "Confirmed."

    # 7-message priming chain post-BUG-LOCAL-280 (was 6 pre-fix).
    # Order satisfies Mistral-Nemo chat template alternation:
    # system -> user -> assistant -> user -> assistant -> user -> assistant.
    priming_messages: List[dict] = [
        {"role": "system",    "content": turn1_system},
        {"role": "user",      "content": turn1_user},
        {"role": "assistant", "content": turn1_assist},
        {"role": "user",      "content": turn2_user},
        {"role": "assistant", "content": turn2_assist},
        {"role": "user",      "content": turn3_user},
        {"role": "assistant", "content": turn3_assist},
    ]

    # Turn 4: previous-line context + beat + length + register.
    prev_clause = ""
    if previous_speaker and previous_line:
        prev_clause = (
            f'\nPrevious line ({previous_speaker}): "{previous_line}"\n'
        )
    turn4_user = (
        f"{prev_clause}"
        f"Your beat: {beat.intent}\n"
        f"Length target: about {beat.length_target_words} words.\n"
        f"Emotional register: {beat.emotional_register}\n\n"
        f"Speak your line. Just the line -- no stage directions, "
        f"no narration, no self-naming."
    )

    return Stage2PromptChain(
        priming_messages=priming_messages,
        turn4_user={"role": "user", "content": turn4_user},
        beat_id=beat.beat_id,
        speaker_name=speaker.name,
        target_words=beat.length_target_words,
        emotional_register=beat.emotional_register,
    )


def assemble_full_chat(chain: Stage2PromptChain) -> List[dict]:
    """Return the full chat-message list (priming + Turn 4 user).

    Caller appends an empty assistant turn (or just calls generate
    against this list) to elicit the dialogue line.
    """
    return chain.priming_messages + [chain.turn4_user]


# ---------------------------------------------------------------------------
# Acknowledgment-chain validators (the 'chain breakpoints' from the plan)
# ---------------------------------------------------------------------------


def accept_turn1_ack(reply: str) -> bool:
    """Return True iff the assistant's Turn 1 reply matches an
    accepted-variant pattern. Used by callers that want to validate
    the chain hasn't broken before paying the cost of Turn 4."""
    if not isinstance(reply, str):
        return False
    low = reply.strip().lower()
    if not low:
        return False
    return any(rx.search(low) for rx in TURN1_ACCEPTED_VARIANTS)


def accept_turn23_ack(reply: str) -> bool:
    """Same as accept_turn1_ack but for Turn 2 + Turn 3."""
    if not isinstance(reply, str):
        return False
    low = reply.strip().lower()
    if not low:
        return False
    return any(rx.search(low) for rx in TURN23_ACCEPTED_VARIANTS)
