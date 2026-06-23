# GROUNDING EXCERPTS (real OTR source -- verify panel claims against these)


## nodes/_otr_outline.py :1166-1268 (_build_beat_user_prompt -- the beat planner prompt)
```python
def _build_beat_user_prompt(
    req: OutlineRequest,
    macro: _MacroShape,
    phase_name: str,
    beat_speaker: str,
    beat_position: tuple[int, int],
    *,
    previous_beat_intent: Optional[str] = None,
    next_beat_speaker: Optional[str] = None,
    phase_summary: Optional[str] = None,
) -> str:
    """Stage 3 user prompt -- ask for intent + mood for one beat.

    Sprint 3B (2026-05-25): the beat is no longer fully isolated. The
    prompt now carries a 1-beat adjacency window so the LLM can write
    an intent that connects to its neighbours instead of a generic
    standalone beat:

      * `previous_beat_intent` -- the narrative intent of the beat
        immediately before this one (the real, already-generated
        intent: Stage 3 fleshes beats sequentially, so the previous
        beat's intent always exists by the time this one is built).
        Omitted entirely for the first voiced beat of the outline.
      * `next_beat_speaker` -- who speaks the beat immediately after
        this one. Stage 3 has not fleshed that beat yet, so its
        *intent* does not exist; the speaker (known from the phase
        skeleton) is the available forward signal -- enough for the
        LLM to land this beat as a hand-off to that speaker. Omitted
        entirely for the last voiced beat of the outline.
      * `phase_summary` -- a one-line statement of what the current
        arc phase is for (from ARC_PHASE_GUIDANCE).

    Each adjacency line is emitted ONLY when its value is present;
    a missing neighbour produces no line at all (never an empty or
    "None" placeholder). Adjacency window is 1 -- immediate
    neighbours only. target_words is intentionally NOT requested:
    Python owns the per-beat word allocation (see
    `_allocate_phase_target_words` and the Stage 3 block in
    generate_outline). Cross-beat coherence beyond the 1-beat window
    is still handled by the combiner + budget validators downstream.
    """
    beat_idx, beat_total = beat_position
    parts = [
        f"Title: {macro.title}",
        f"Premise: {macro.premise}",
        f"Setting: {macro.setting}",
        "",
        f"Phase: {phase_name}",
    ]
    if phase_summary:
        parts.append(f"Phase focus: {phase_summary}")
    parts.append(f"Beat {beat_idx + 1} of {beat_total} in this phase")
    parts.append(f"Speaker: {beat_speaker}")
    # Adjacency window (1): only emit a line when the neighbour exists.
    if previous_beat_intent:
        parts.append(f"Previous beat intent: {previous_beat_intent}")
    if next_beat_speaker:
        parts.append(f"Next beat is spoken by: {next_beat_speaker}")
    parts.extend([
        "",
        "Task: write the intent (one sentence, NOT dialogue) and a "
        "mood descriptor for this beat. The intent MUST be an ACTION "
        "UNDER PRESSURE -- the speaker DOES something with stakes "
        "(reveal, refuse, demand, bargain, accuse, conceal, choose, "
        "threaten, confess), not merely discusses, reflects, or "
        "describes. RAISE THE STAKE: this beat's pressure must be higher "
        "than the previous beat's -- escalate, never tread water. It "
        "should follow on from the previous beat and set up the next "
        "where those are given. "
        # D2 (2026-06-22, story-quality lift): antagonist-stance consistency.
        # The weak-end failure (b-Chandra's Echo) was the antagonist reversing
        # his stance toward the protagonist with no turn beat. JSON-free,
        # no cross-run state -- a best-effort generation nudge.
        "KEEP STANCE CONSISTENT: each character's stance toward the "
        "protagonist and the central conflict must stay true to the want "
        "they have shown so far. A reversal -- an adversary relenting, an "
        "ally turning on them -- is allowed ONLY as a deliberate turn this "
        "beat earns and shows, never an unmotivated flip from the previous "
        "beat. Return only the JSON object.",
    ])
    return "\n".join(parts)


# C0 (story-quality R2): an action-under-pressure beat intent leads with (or
# contains) a stakes verb. Used as a measurement signal by the story-quality
# scan; NOT a hard outline-failing gate (a strict structured_call post_validator
# would flake outlines on weak models -- the prompt constraint is the lever).
_ACTION_PRESSURE_RE = re.compile(
    r"\b(reveal|refus|deny|denies|denied|demand|bargain|accus|conceal|hid|"
    r"hides|choos|chose|chooses|threaten|confess|betray|expos|defy|defies|"
    r"defied|insist|warn|confront|sacrific|risk|gambl|force)\w*",
    re.IGNORECASE,
)


def intent_is_action_under_pressure(intent) -> bool:
    """True when a beat intent reads as an action under pressure (carries a
    stakes verb). Pure; never raises."""
    try:
        return bool(_ACTION_PRESSURE_RE.search(str(intent or "")))
    except Exception:  # noqa: BLE001
        return False


```


## nodes/_otr_line_composer.py :1065-1245 (_build_user_prompt -- composer prompt incl L2 deflection)
```python
def _build_user_prompt(req: LineRequest) -> str:
    """Render the per-beat user prompt for the composer.

    Phase 4 v4 (2026-05-11): block order tightened for future KV-cache
    reuse. Every block that stays byte-identical across all composer
    calls in an episode lives in the STATIC PREFIX:

        STYLE
        THEME                (Commit 2: meta.news first-sentence theme)
        EPISODE CONTEXT      (canon_header)
        NAMED ENTITIES       (people + things, sorted)
        CAST                 (full voice cards, all characters)
        OUTLINE              (full spine, plain - no per-call arrow)

    Blocks that change per call live in the PER-BEAT TAIL:

        CURRENT BEAT         (single spine row for the beat we write)
        POSITION             (Commit 4: phase, beat N of M, next phase)
        SOUND IN THE ROOM    (Commit 2: beat.sfx_cue)
        LAST SPOKEN          (last_lines rolling window; scene-local
                              via Commit 3)
        WRITE LINE           (role induction + beat + mood + word count
                              + "Speak now.")

    Optional blocks are dropped entirely when their LineRequest field
    is empty so unit tests that pin a specific minimal shape keep
    working. NAMED ENTITIES fires when allowed_people OR
    allowed_things is non-empty.

    The role-induction sentence "You are <SPEAKER>." (plus optional
    "You are responding to <PREV_SPEAKER>.") sits immediately above
    the generation target. Small instruct-tuned LLMs in the 7B-14B
    class hold a per-call role much more reliably when the directive
    is one block above the response slot vs upstream in the system
    prompt.
    """
    parts: list[str] = []

    # ===== STATIC PREFIX (byte-stable across an episode) =====

    if req.style_descriptor:
        parts.append(f"STYLE: {req.style_descriptor}")
        parts.append("")

    # THEME emits when the writer threads a non-empty theme via
    # `LineRequest.theme` (Commit 2 in the v4 plan).
    if req.theme:
        parts.append(f"THEME: {req.theme}")
        parts.append("")

    parts.append("EPISODE CONTEXT")
    parts.append(req.canon_header)

    # NAMED ENTITIES split (Commit 1 in the v4 plan). The writer
    # populates allowed_people / allowed_things separately on every
    # real call. allowed_roster is still consumed by the phantom-gate
    # check downstream (detect_phantom_names); the prompt-rendering
    # side is split-only.
    if req.allowed_people or req.allowed_things:
        parts.append("")
        parts.append("NAMED ENTITIES IN THIS WORLD")
        if req.allowed_people:
            parts.append(
                "  People: " + ", ".join(sorted(req.allowed_people))
            )
        if req.allowed_things:
            parts.append(
                "  Places, agencies, things: "
                + ", ".join(sorted(req.allowed_things))
            )
        parts.append(
            'Generic roles ("the tech", "the lab", "mission control") '
            "are fine. Do not invent any other proper name."
        )

    # CAST replaces single-speaker CHARACTER when all_voice_cards is
    # threaded. Falls back to the speaker-only voice card on legacy
    # callers (Commit 2 wires the full-cast path in the writer).
    if req.all_voice_cards:
        parts.append("")
        parts.append("CAST")
        parts.append(req.all_voice_cards)
    elif req.character_voice_card:
        parts.append("")
        parts.append(f"CHARACTER: {req.character_voice_card}")

    if req.outline_spine:
        parts.append("")
        parts.append(req.outline_spine)

    # ===== PER-BEAT TAIL (changes every call) =====

    # CURRENT BEAT — single spine row for the beat we are writing
    # right now. The outline above stays plain (no arrow) for KV
    # stability; this block names which row we are on. Writer
    # pre-renders the string via `render_current_beat(outline,
    # beat.beat_id)` and threads it on `req.current_beat_block`.
    if req.current_beat_block:
        parts.append("")
        parts.append(req.current_beat_block)

    # CONTINUITY CONSTRAINTS -- Sprint 5A (2026-05-25). A per-speaker,
    # per-beat hard-constraint block the writer renders from the episode
    # ContinuityState (who knows what, by which beat -- see
    # `_otr_continuity.render_continuity_slice`). Lives in the per-beat
    # tail because it changes per call, and sits ABOVE POSITION /
    # WRITE LINE so the constraint frames the beat before the model
    # writes. The slice string already carries its own
    # "CONTINUITY CONSTRAINTS ..." header. Empty string -> block dropped
    # (no continuity signal for this speaker at this beat), so every
    # caller / test that omits the field is unaffected.
    if req.continuity_slice:
        parts.append("")
        parts.append(req.continuity_slice)

    # POSITION supersedes the old generic ARC PHASE block (Commit 4
    # in the v4 plan). Emits the position string verbatim. Legacy
    # arc_phase-only callers still get a fallback ARC PHASE block so
    # this commit does not regress them in isolation.
    if req.position:
        parts.append("")
        parts.append(f"POSITION: {req.position}")
    elif req.arc_phase:
        guidance = ""
        try:
            from . import _otr_episode_budget as _OTRB  # type: ignore
            guidance = _OTRB.ARC_PHASE_GUIDANCE.get(req.arc_phase, "")
        except Exception:  # noqa: BLE001
            guidance = ""
        parts.append("")
        if guidance:
            parts.append(f"ARC PHASE: {req.arc_phase}")
            parts.append(f"  {guidance}")
        else:
            parts.append(f"ARC PHASE: {req.arc_phase}")

    # SOUND IN THE ROOM — Commit 2 in the v4 plan. Threaded from
    # beat.sfx_cue so the line can react to the sound environment.
    if req.sfx_cue:
        parts.append("")
        parts.append(f"SOUND IN THE ROOM: {req.sfx_cue}")

    # ===== Sprint 3 (2026-05-28): DRAMATIC FRAME (magnetic pole) =====
    # The block sits ABOVE the rolling window so the next_turn the
    # beat must reveal is the last directive the model reads before
    # the LAST SPOKEN buffer. Each line is conditionally emitted so
    # legacy callers (Sprint 2 Optional fields all empty) still
    # render exactly the pre-Sprint-3 prompt -- the entire block is
    # dropped when none of the Sprint 3 fields are set.
    _dramatic_lines: list[str] = []
    if req.dramatic_question:
        _dramatic_lines.append(
            f"DRAMATIC QUESTION: {req.dramatic_question}"
        )
    # L2 authoring contract (story-quality v2, R3 2026-06-22). Under the flag,
    # for a high-tension character beat that already carries subtext, WITHHOLD
    # the literal Objective and ask for the deflection instead -- the universal
    # weak-writer failure was collapsing to terse imperative command-shouting
    # ("Override the protocols!") that states the goal outright. The gate is a
    # conjunction of DETERMINISTIC inputs (the flag + speaker_role + the pinned
    # beat_tension + whether the beat HAS subtext) -- never inferred from
    # generated text. Flag OFF (default) => the whole branch is dead and the
    # block below renders the pre-R3 prompt byte-for-byte.
    _sqv2_deflect = (
        req.story_quality_v2_enabled
        and req.speaker_role == "character"
        and req.beat_tension >= OBJECTIVE_DEFLECTION_TENSION_MIN
        and bool((req.beat_subtext or "").strip())
    )
    _this_beat_lines: list[str] = []
    if req.beat_objective and not _sqv2_deflect:
        _this_beat_lines.append(f"  Objective: {req.beat_objective}")
    if req.beat_obstacle:
        _this_beat_lines.append(f"  Obstacle:  {req.beat_obstacle}")
    if req.beat_turn:
        _this_beat_lines.append(f"  Turn:      {req.beat_turn}")
    if req.beat_subtext:
        _this_beat_lines.append(f"  Subtext:   {req.beat_subtext}")
    if 1 <= req.beat_tension <= 5:
        _this_beat_lines.append(f"  Tension:   {req.beat_tension}/5")
    if _sqv2_deflect:

```


## nodes/_otr_outline.py :744-872 (arc_phases budget + arc_phase validators)
```python
def _get_budget(req: "OutlineRequest"):
    """Return the EpisodeBudget on req, or None.

    Stored as `object` on OutlineRequest so the module can be imported
    without coupling to _otr_episode_budget at load time. We check
    duck-typing here (presence of arc_phases attribute is sufficient).
    """
    b = getattr(req, "budget", None)
    if b is None:
        return None
    if (hasattr(b, "arc_phases") and hasattr(b, "per_phase_words")
            and hasattr(b, "per_phase_beats")):
        return b
    return None


def _format_episode_budget_block(req: "OutlineRequest") -> str:
    """Render the EPISODE BUDGET block. Empty string when no budget."""
    b = _get_budget(req)
    if b is None:
        return ""
    arc_phases = list(b.arc_phases)
    per_phase_words = list(b.per_phase_words)
    per_phase_beats = list(b.per_phase_beats)
    words_lo, words_hi = b.words_per_beat_range
    lines: list[str] = [
        "EPISODE BUDGET -- hit these numbers:",
        f"- Total spoken words: ~{b.target_words} (within 15%)",
        f"- Structure: {b.act_count} act"
        f"{'s' if b.act_count != 1 else ''} -> {', '.join(arc_phases)}",
    ]
    phase_words = ", ".join(
        f"{name} ~{w}"
        for name, w in zip(arc_phases, per_phase_words)
    )
    lines.append(f"- Words per phase: {phase_words}")
    phase_beats = ", ".join(
        f"{name} {n}"
        for name, n in zip(arc_phases, per_phase_beats)
    )
    lines.append(f"- Voiced beats per phase: {phase_beats}")
    lines.append(f"- Each voiced beat: {words_lo}-{words_hi} words")
    lines.append(
        f"- Music inter beats: {b.music_inter_count} "
        f"({'one between each pair of phases' if b.music_inter_count > 0 else 'continuous flow, no music_inter'})"
    )
    lines.append(
        f"- Announcer beats: {b.announcer_beats} (open + close)"
    )
    lines.append(
        "- Every voiced beat MUST carry an `arc_phase` field set to "
        f"one of: {', '.join(arc_phases)}."
    )
    return "\n".join(lines)


def validate_outline_against_budget(
    outline: "Outline",
    req: "OutlineRequest",
    *,
    word_drift_warn_ratio: float = 0.25,
) -> Optional[str]:
    """Run the Phase 2A outline validators.

    Returns None on pass. Returns an error string on the FIRST hard
    failure (suitable for the reroll-then-repair loop). Validator #1
    (total word drift) is WARN-only per §6.E -- never fails. Per
    §6.G announcer + music + sfx beats are EXCLUDED from word and
    per-phase budgets but are still counted by validators #6 / #7.

    S28 cleanbreak: budget is now required at OutlineRequest
    construction time, so _get_budget always returns a populated
    EpisodeBudget here. The pre-S28 `if b is None: return None`
    branch is extinct.

    Validator list (re-numbered after §6.C dropped per-character
    distribution):
      #1  total word drift (WARN >25%, no reroll per §6.E)
      #2  per-phase word totals within [0.80, 1.20] of target
      #3  per-phase voiced-beat counts within [target-1, target+2]
      #4  per-voiced-beat target_words ∈ words_per_beat_range
      #5  arc_phase monotonic ordering (no interleaving)
      #6  count(music_inter beats) == budget.music_inter_count
      #7  count(announcer beats) == budget.announcer_beats
      #8  every speaker ∈ character_cast ∪ {ANNOUNCER}
          (existing cast-membership check; KEPT)
    """
    b = _get_budget(req)
    # S28 cleanbreak: dropped `if b is None: return None` no-op
    # fallback. Producer contract guarantees b is non-None here.

    voiced = [
        beat for beat in outline.beats
        if beat.speaker_role == "character"
    ]
    announcer_beats = [
        beat for beat in outline.beats
        if beat.speaker_role == "announcer"
    ]
    music_inter_beats = [
        beat for beat in outline.beats
        if beat.speaker_role == "music_inter"
    ]

    # Wiring-review #13 (2026-05-11): validate arc_phase
    # existence + allowed-value-membership + monotonic ordering
    # BEFORE running per-phase word totals or per-phase beat
    # counts. Otherwise an unknown / missing phase value silently
    # miscounts under the per-phase aggregations (validators 2 + 3)
    # and the reroll prompt fires for the wrong reason.

    arc_phases = list(b.arc_phases)
    per_phase_words = list(b.per_phase_words)
    per_phase_beats = list(b.per_phase_beats)
    phase_index = {ph: i for i, ph in enumerate(arc_phases)}

    # --- arc_phase: existence + value + monotonic order (was #5) ---
    last_idx = -1
    for beat in voiced:
        ph = (beat.arc_phase or "").strip()
        if not ph:
            return (
                f"Beat {beat.beat_id} is missing arc_phase. Every "
                f"voiced beat MUST carry one of: "
                f"{', '.join(arc_phases)}."
            )
        if ph not in phase_index:
            return (
                f"Beat {beat.beat_id} has arc_phase={ph!r}; not in "

```


## allowed_things / entity check (real ledgers -- does the palette source carry CONFLICT objects?)

### pending_20260623_063433
  premise: 
  meta.allowed_roster = ["ANNOUNCER", "CHANDRA X-RAY OBSERVATORY", "GALACTIC CENTER", "KEARNEY HOUTEN", "MILKY WAY GALAXY", "NASA", "NIA FROST", "STANLEY HUDSON", "SUPERNOVA REMNANT"]
### signal_lost_the_scorchedearth_switch_20260623_060918
  premise: 
  meta.allowed_roster = ["ANNOUNCER", "DONNA VOSS", "KARNACKI ECKELS", "LOW-EARTH ORBIT", "M\u0100HIA PENINSULA", "NIRAN HALLOWAY", "ROCKET LAB", "US SPACE FORCE", "VICTUS HAZE PUMA"]
### signal_lost_venting_o2_20260623_055159
  premise: 
  meta.allowed_roster = ["ANNOUNCER", "CARL TERWILLIGER", "EDWARD SATO", "MALI SCOTT", "MILITARY EXERCISES", "ROCKET LAB", "STEALTH LAUNCH", "US SPACE FORCE", "VICTUS HAZE PUMA"]
### signal_lost_knob_at_full_open_20260623_051721
  premise: 
  meta.allowed_roster = ["ANNOUNCER", "EL NI\u00d1O", "GLOBAL TEMPERATURES", "LEMMY", "MANFRED VOLKOV", "MINDY STONE"]
