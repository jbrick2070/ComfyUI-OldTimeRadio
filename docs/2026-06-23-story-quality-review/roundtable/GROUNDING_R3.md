# GROUNDING R3 (wiring shapes -- verify against these)


## nodes/_otr_outline.py :84-135 (class Beat -- Pydantic; add optional fields w/ defaults)
```python
class Beat(BaseModel):
    """One beat of the outline. Lines are generated 1:1 from beats."""

    beat_id: str = Field(
        ...,
        pattern=r"^b\d{3}$",
        description="Stable ID, format 'b001', 'b002', monotonic per outline",
    )
    speaker: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Character name in ALL CAPS, or 'NARRATOR' for music/sfx beats",
    )
    speaker_role: SpeakerRole = Field(
        ...,
        description="Routing role; see _otr_speaker_role for HuMo vs LTX-radio dispatch",
    )
    intent: str = Field(
        ...,
        min_length=4,
        max_length=200,
        description="What this beat accomplishes narratively, one sentence",
    )
    target_words: int = Field(
        ...,
        ge=3,
        le=80,
        description="Approximate word count for the dialogue line",
    )
    mood: str = Field(
        ...,
        min_length=2,
        max_length=40,
        description="Tone descriptor, e.g. 'tense', 'wry', 'foreboding'",
    )
    sfx_cue: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Optional [SFX:] hint for the surrounding line",
    )
    arc_phase: str = Field(
        default="setup",
        max_length=40,
        description=(
            "Phase 2A (2026-05-11): narrative phase label from "
            "EpisodeBudget.arc_phases (setup / complication / "
            "resolution / climax / etc.). Required-with-default per "
            "the post-Phase-3 review pass (Strategy A). A 12B model "
            "like Mistral-Nemo frequently omits Optional pydantic "
            "fields; making the field required with a 'setup' "
            "default guarantees it is always populated, and the "
```


## nodes/_otr_line_composer.py :581-700 (class LineRequest -- dataclass fields)
```python
class LineRequest:
    """Per-beat input for compose_line.

    Fields are duplicated from Beat (rather than passing the Beat directly)
    to keep this module's import surface stdlib-only at module load. The
    caller maps Beat fields into LineRequest.

    Phase 0 (2026-05-11): `allowed_roster` field added for the name-
    gate check. Orchestrator MUST populate it on every call — built
    once via build_allowed_roster after cast lock + news_interpreter.
    The empty-frozenset default is retained ONLY as a dataclass-
    ordering artifact (non-defaulted fields can't follow defaulted
    ones).

    Phase 1 (2026-05-11): three new context fields replace the
    composer's previous "speaker + intent + mood + canon header +
    last 3 lines" diet:

      style_descriptor          full snake_case style for the episode
                                (from _otr_style_picker). Empty string
                                skips the STYLE block entirely.
      outline_spine             one-line-per-beat compact rendering
                                of the whole outline so the composer
                                can see the arc it is participating
                                in. Empty string skips the OUTLINE
                                block entirely. Renderer ships in this
                                module (`render_outline_spine`).
      character_voice_card      one-line `name (gender, traits)` blurb
                                for the speaker of this beat. Empty
                                string skips the CHARACTER block. Built
                                from cast_rows via `build_voice_card`.

    Prompt placement is static-first / variable-second (style +
    canon_header + outline_spine + allowed_roster are stable across
    every composer call in the episode; voice_card + last_lines +
    speaker + intent change per call). Once KV-cache reuse lands in
    the loader (deferred; tracked in the ADR), the cached prefix
    covers everything up to the CHARACTER block.

    All Phase 1 fields default to empty strings so unit tests and
    early-stage callers that don't have them yet keep working.
    """

    speaker: str
    intent: str
    mood: str
    target_words: int
    canon_header: str               # from render_episode_canon_header()
    last_lines: list[tuple[str, str]]  # [(speaker, text), ...] most recent last; empty for first beat
    allowed_roster: frozenset[str] = field(default_factory=frozenset)
    # Phase 1 (2026-05-11) -- composer prompt enrichment + sliding window.
    style_descriptor: str = ""
    outline_spine: str = ""
    character_voice_card: str = ""
    # Phase 2A (2026-05-11) -- arc_phase awareness. When non-empty,
    # the per-beat prompt grows by an ARC PHASE block carrying the
    # ARC_PHASE_GUIDANCE one-liner for the current phase so the
    # composer steers by narrative phase, not just mood.
    arc_phase: str = ""
    # Phase 4 v4 (2026-05-11) -- prompt revision pass. All defaults
    # are empty so every existing test / caller keeps working; each
    # block in `_build_user_prompt` gates on the corresponding field
    # being non-empty.
    #
    #   allowed_people / allowed_things  Split roster for prompt
    #       rendering. Cast names ("ALICE") and journalistic terms
    #       ("CERN") render in distinct buckets. `allowed_roster`
    #       remains the union and stays the input to the phantom gate;
    #       these two fields are render-only. When both are empty the
    #       composer falls back to the legacy combined ALLOWED NAMES
    #       block driven by `allowed_roster`.
    #   prev_speaker  Name of the character who spoke the immediately
    #       preceding line. Renders in the WRITE LINE role-induction
    #       block as "You are responding to <name>." Empty drops that
    #       sentence (first line of a scene, post-music marker).
    #   current_beat_block  Pre-rendered CURRENT BEAT block (one
    #       outline-spine row for the beat we are writing now). The
    #       writer computes this once per beat via
    #       `render_current_beat(outline, beat.beat_id)`. Keeping the
    #       outline_spine itself plain (no arrow) lets the static
    #       prefix stay byte-stable across every call in an episode
    #       so a future KV-cache reuse pass lands without re-encoding
    #       the spine.
    #   theme  One-sentence theme from `meta.news.script_brief`
    #       (Commit 2 in the v4 plan). Optional flavor, not the
    #       structural-direction outline.
    #   all_voice_cards  Newline-joined voice cards for the whole
    #       cast (Commit 2). When set, replaces single-speaker
    #       CHARACTER block with CAST. Falls back to
    #       `character_voice_card` when empty.
    #   sfx_cue  `beat.sfx_cue` for this beat (Commit 2). Renders as
    #       SOUND IN THE ROOM in the per-beat tail.
    #   position  "<phase>, beat N of M. Next phase: <next>." string
    #       (Commit 4). Replaces the generic per-phase ARC_PHASE_GUIDANCE
    #       one-liner with a position-specific directive. Falls back
    #       to the legacy ARC PHASE block driven by `arc_phase` when
    #       empty.
    allowed_people: frozenset[str] = field(default_factory=frozenset)
    allowed_things: frozenset[str] = field(default_factory=frozenset)
    prev_speaker: str = ""
    current_beat_block: str = ""
    theme: str = ""
    all_voice_cards: str = ""
    sfx_cue: str = ""
    position: str = ""
    # LFC sprint commit 3, section 6.1 (2026-05-11). speaker_role lets
    # polish_line branch its system prompt -- character beats get
    # the strict "no narration" prompt; announcer beats get the
    # narration-allowed prompt that still strips bracket stage
    # directions and asterisk action. Default "character" so legacy
    # callers / tests see the original prompt unchanged.
    speaker_role: str = "character"
    # Sprint 5A (2026-05-25) -- continuity slice. The writer renders a
    # per-speaker, per-beat hard-constraint block from the episode
    # ContinuityState (_otr_continuity.render_continuity_slice) and
    # threads the prompt-ready string here. Empty string means no
    # continuity signal for this speaker/beat -- `_build_user_prompt`
    # drops the block entirely. Default "" keeps every existing caller
    # and test working unchanged.
    continuity_slice: str = ""
```


## nodes/_otr_episode_budget.py :190-240 (class EpisodeBudget -- arc_phases; DO NOT overload)
```python
class EpisodeBudget:
    """Authoritative budget for one episode.

    Built once by `compute_episode_budget` from
    (target_words, act_count, include_act_breaks, num_characters).

    Consumers:
      * outline LLM prompt          per_phase_words / per_phase_beats /
                                     words_per_beat_range / arc_phases /
                                     music_inter_count / announcer_beats
      * 8 outline validators         all fields
      * composer prompt (arc_phase)  arc_phases (looked up by beat index)
    """

    act_count: int
    arc_phases: tuple[str, ...]
    per_phase_words: tuple[int, ...]
    per_phase_beats: tuple[int, ...]
    words_per_beat_range: tuple[int, int]
    music_inter_count: int
    announcer_beats: int            # always 2 (open + close)
    cast_size: int
    target_words: int               # echoed for downstream convenience


def _max_target_words_for_act_count(act_count: int) -> int:
    """Largest target_words this act_count can hold without a per-phase
    budget violation, given BEAT_WORD_HARD_MAX. Used only to make the
    fail-fast guard message actionable; it is not itself a gate."""
    cfg = ACT_COUNT_CONFIG[act_count]
    limits = []
    for frac, nb in zip(cfg["act_word_fractions"], cfg["voiced_beats_per_act"]):
        if frac <= 0.0:
            continue
        # Floor feasibility per phase: nb * HARD_MAX >= 0.80 * frac * tw.
        limits.append((nb * BEAT_WORD_HARD_MAX) / (0.80 * frac))
    return int(min(limits)) if limits else 0


def compute_episode_budget(
    target_words: int,
    act_count: int,
    include_act_breaks: bool,
    num_characters: int,
) -> EpisodeBudget:
    """Validate widget combo and derive the EpisodeBudget.

    Raises InvalidEpisodeBudgetError on any of:
      * target_words < 30
      * act_count out of range [1, 7]
      * act_count < default_act_count(target_words)
```


## nodes/_otr_ledger_scrub.py :981-1011 (L5a: story_quality telemetry aggregation -- EP16 undercount)
```python
    # ---- 4b. story-quality-v2 telemetry (gated) -------------------------
    # Aggregate the per-line breadcrumbs into a meta summary, ONLY when the flag
    # is on (never written when off -> the flag-off ledger carries no new meta
    # key). Counts ride the row compose_flags (the FIXED schema); the scrub is
    # the natural aggregation point -- it has just recorded the L7 splits and
    # sees the L1 objective_literal_retry flags carried up from compose (C0
    # preserves them through set_lines / reroll).
    if _sqv2_on:
        _l1_rerolls = _l7_splits = _l7_split_failures = 0
        for r in lines:
            if not isinstance(r, dict):
                continue
            for _f in r.get("compose_flags") or []:
                if not isinstance(_f, str):
                    continue
                if _f == "objective_literal_retry":
                    _l1_rerolls += 1
                elif _f.startswith("action_split:"):
                    _l7_splits += 1
                elif _f.startswith("action_split_failed:"):
                    _l7_split_failures += 1
        _meta = led.get("meta")
        if not isinstance(_meta, dict):
            _meta = {}
            led["meta"] = _meta
        _meta["story_quality"] = {
            "l1_rerolls": _l1_rerolls,
            "l7_splits": _l7_splits,
            "l7_split_failures": _l7_split_failures,
        }

```


## nodes/_otr_ledger_reviewer.py :1136-1144 (compute_edit_cap) + :2030-2045 (too_many_edits set)
```python
def compute_edit_cap(voiced_beats: int) -> int:
    """edit_cap = min(8, max(3, voiced_beats // 3)).

    Per §3 Phase 3 -- scales with episode size so a 19-beat 7-act
    episode can accommodate ~6 plausible rewrites without flipping
    `too_many_edits`, while a 6-beat 1-act episode caps at 3.
    """
    return min(8, max(3, voiced_beats // 3))


# ...
        _stamp_word_counts_safe(led)
        led.save()
        return disp

    edits_applied = apply_doctor_edits(
        candidate, doctor_report, edit_cap=edit_cap,
    )
    if edits_applied == -1:
        led.data.clear()
        led.data.update(original_snapshot)
        meta_after = led.data.setdefault("meta", {})
        meta_after["reviewer_verdict"] = "too_many_edits"
        disp = ReviewerDisposition(
            verdict="too_many_edits",
            pre_audit_violations=pre_audit_violations,
            pre_audit_repairs_applied=repairs_applied,
```


## nodes/_otr_freeze_cascade.py :593-605 (reviewer-terminal stops the cascade) + :730-766 (where story critic runs)
```python
      * Existing review_ledger runs next. Only Phases 1 + 2 fire
        inside it after S33 B3 (Phase 9 retired per refined
        no-auditors rule). The composite phase_name string
        `phase_1_2_9_reviewer_composite` is retained for forensic
        continuity.
      * If the reviewer verdict is a terminal failure
        (too_many_edits / needs_full_rerun), the cascade stops there:
        the ledger has already been restored to its pre-review state,
        and Phase 10 would either re-flag the same pre-existing gap
        or stamp frozen_clean on an unaltered ledger -- both
        misleading. S33 B2 (2026-05-15) retired the two rollback-gate
        verdicts (cast_unrecoverable, post_audit_failed) per the
        refined no-auditors rule.

# ...
    # structurally sound, so the critic spends its whole budget on the
    # one thing no earlier pass judges: DRAMATIC QUALITY (continuity,
    # voice drift, flat lines, arc verdict -- audit Finding C). For 5B
    # the report is ADVISORY: it changes no line text. It is stamped on
    # meta for Sprint 5C (targeted reroll reads `reroll_targets`) and
    # Sprint 6 (render coupling reads `render_priority` / `flat_lines` /
    # `arc_verdict`). run_story_critic NEVER raises -- on any failure it
    # returns StoryCriticReport.clean(), so a critic failure can never
    # break the freeze (Prime Directive 1). The `# LLM slot: technical`
    # tag lives at the structured_call site inside run_story_critic;
    # the critic reuses the technical model already resident in the
    # cascade -- no new widget, no VRAM swap (Prime Directive 6).
    #
    # BUG-LOCAL-273 fix: refresh ledger_data + meta from led.data BEFORE the
    # critic stamp. The Phase 1+2 reviewer above calls Ledger.save()
    # internally; production_ledger.py:1012 rebinds self.data to a new
    # merged dict on every save, detaching the cascade's L570/L576 locals.
    # Without this refresh the stamp lands on an orphaned dict and the
    # reroll/render_plan/HuMo never see it (live run
    # signal_lost_bioluminescent_trench_descent_20260525_182002: critic
    # produced 4 reroll_targets / arc_verdict=uneven; reroll + render_plan
    # both saw clean()/defaults).
    ledger_data = led.data
    meta = ledger_data.setdefault("meta", {})
    story_critic_report = _OTRSC.run_story_critic(
        generate_fn,
        ledger_data,
        ledger_data.get("cast", []) or [],
    )
    meta["story_critic_report"] = story_critic_report.model_dump()
    log.info(
        "[LFC] Sprint 5B story critic: arc_verdict=%s, %d reroll "
        "target(s), %d flat line(s), %d continuity issue(s), %d line(s) "
        "in render priority (advisory -- no line text changed)",
        story_critic_report.arc_verdict,
        len(story_critic_report.reroll_targets),
        len(story_critic_report.flat_lines),
```
