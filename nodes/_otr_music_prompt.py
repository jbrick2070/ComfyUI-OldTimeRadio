"""nodes/_otr_music_prompt.py -- the single source of truth for theme-music cue
prompts, routed through the Meta brief protocol.

Every music cue prompt (opening / closing / interstitial) is composed HERE from
the story/Meta brief on the ledger meta, read via the brief-reader protocol
(`_otr_brief_reader._read_brief_field`) -- never by poking meta directly with a
local template. This is the same downstream-consumer contract the visual slots
(FLUX / LTX / HuMo / portraits) follow, so music generation pulls period,
setting, and mood from the same propagating creative brief as every other
creative call.

TWO PRODUCTS since 2026-09-11 (operator, after listening to `moonlit_deception`:
"the 'music' needs to be improved, doesn't sound like music ... sounds like
radio hiss, which is what we asked. So I am asking to go over ALL musical
prompts and make them more musical ... ideally it is relevant to the story"):

* ``compose_music_prompt(meta, cue_id) -> (row_text, duration)`` -- the ROW
  text: the ledger's ``generation_prompt`` / ``description`` for the cue, the
  text hashed into ``cue_spec_sha256``, and what a reader of the ledger sees.
  The brief's mood words, the musical DEVICES they call for, the story's
  period idiom, the setting it evokes, the cue's arc, and the instrumental-only
  tail. Story-relevant, and it reads as a musical instruction.
* ``compose_engine_prompt(meta, row_text) -> EnginePrompt`` -- what the ENGINE
  hears: the story palette's instruments first (`_otr_music_palette`), a
  clean-studio production anchor, then the row text -- or an AUTHORED row's
  text verbatim on the my_story / scifi_news_pro lanes -- capped at the
  smallest engine budget (Sonilo refuses above 1000 characters) by trimming
  the row text at a clause boundary, never raising. Plus the one negative
  prompt for every engine that takes one. The per-engine "analog tape warmth
  / vintage / radio" anchors that used to be prepended were the withdrawn ask
  and appear nowhere.

Mood resolution cascade (preserved from the audited musicgen path so a given
brief yields the same music): v2 `music_mood_terms` (top 3) -> v1
`story_brief_terms.atmosphere` (top 3) -> keyword-mined produced logline /
`news.script_brief` -> neutral "atmospheric".

The last step of that cascade is SKIPPED on a `groove_arc` palette (2026-09-12):
"atmospheric" is a texture instruction, and with no brief it was the FIRST
thing in the row -- standing in front of "Detroit techno at 128 BPM". A real
brief's mood words are still carried on every bank, groove or not, because
those are harmonic rather than textural and a dance cue can wear them.

PURE: no I/O, no GPU, no engine imports. Consumers (the theme node) import from
here; this module imports only the brief reader and the palette. UTF-8 no BOM,
ASCII-only, no em-dashes.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._otr_brief_reader import _read_brief_field, spoken_term
from ._otr_music_palette import mood_devices, story_palette, tempo_phrase

# Three fixed cues + durations (seconds). Durations are part of cue identity;
# keep stable (mirrors the legacy MusicGen cue durations).
CUE_DURATIONS: dict[str, int] = {"opening": 12, "closing": 8, "interstitial": 4}

# Universal instrumental-only tail, applied last so it always lands at the end.
_PROMPT_TAIL = ", instrumental only, no dialogue, no vocals"

# Per-cue ARC -- what the music DOES over its length, in musical terms. The
# "intro" / "outro" words are kept on purpose: the SA3 engine's context window
# still falls back to them for a caller that hands over no placement.
_CUE_CHARACTER: dict[str, str] = {
    # "settles into a STEADY theme" asked for the repetition this module
    # spent 2026-09-12 removing; a theme that FLOWS is the same musical idea
    # without the word that invites a two-bar loop.
    "opening":      "a rising overture that settles into a flowing theme, "
                    "instrumental intro",
    "closing":      "a final statement of the theme resolving to a warm held "
                    "chord, instrumental outro",
    "interstitial": "a brief melodic bridge that hands off cleanly, short "
                    "instrumental transition",
}

#: THE ARC FOR A PALETTE THAT CARRIES `groove_arc` -- today Detroit techno and
#: Chicago house, the two the operator's ear rejected on 2026-09-12.
#:
#: THE CONTROL-FLOW FACT, WHICH IS MEASURED AND NOT A THEORY: `_CUE_CHARACTER`
#: was appended AFTER the rhythmic branch, unconditionally, so a 128 BPM drum
#: machine was asked for "a rising overture" and, closing, to resolve "to a
#: warm held chord" -- even though the same function already refuses to hand a
#: rhythmic palette the orchestral `mood_devices` and `tempo_phrase`. The arc
#: language walked past the guard built to stop exactly that.
#:
#: WHETHER THAT WORDING IS WHY THE CUES CAME BACK AS PADS IS A HYPOTHESIS, not
#: a demonstrated cause (codex contrarian round, same day). What is established
#: is his verdict on the output and the contradiction in the prompt; a seed-
#: matched before/after audition is what would settle causation, and it is his
#: ear that decides. Competing explanations that remain untested include the
#: checkpoint (post-trained SA3 ignores cfg and negatives) and the sampler.
#:
#: REPETITION IS THE POINT HERE, WHICH INVERTS THE SUSTAINED LANE'S RULE.
#: The opening above says "flowing" rather than "steady" to avoid inviting a
#: two-bar loop; techno IS a held groove, and the rhythmic negative already
#: drops the whole anti-loop half for these banks, so this arc is free to ask
#: for the steadiness the other one has to avoid.
_CUE_CHARACTER_RHYTHMIC: dict[str, str] = {
    "opening":      "the groove established in the first bar and held, "
                    "instrumental intro",
    "closing":      "a last pass of the groove ending clean on the downbeat, "
                    "instrumental outro",
    "interstitial": "a short rhythmic break that hands off cleanly, short "
                    "instrumental transition",
}

#: What every engine hears after the instruments: a clean recording, not a
#: degraded one. This is the ONLY production language in any music prompt.
PRODUCTION_ANCHOR = "clearly recorded, clean balanced studio mix, natural room"

#: The one negative prompt, for the engines that take one. It names the
#: withdrawn texture explicitly so the model steers away from it.
NEGATIVE_PROMPT_DEFAULT = (
    "noise, static, hiss, white noise, radio static, crackle, distortion, "
    "clipping, silence, speech, vocals, singing, lyrics, spoken word, "
    # The loop half, added 2026-09-12. Stable Audio Open is built to make
    # loops and one-shots, so a cue has to say that it is not one. RANKED
    # first among the levers tried, on a lab bench over four seeds (closing
    # cue envelope periodicity 0.78 -> 0.26) -- DIRECTION ONLY: that bench
    # could not reproduce a shipped cue from its own receipt (correlation
    # 0.92), so the number that counts is the canonical one. On the first
    # canonical leg with the whole change the two cues measured 0.109 and
    # 0.286 against 0.485 and 0.713 the night before, and the repeat lag
    # moved from 0.25 s to 2.2-4.2 s (`scripts/otr_music_ab.py`).
    "loop, looping, repetitive, ostinato, sequencer, arpeggiator, drum "
    "machine, metronome, click track, drum loop, beat")

#: The smallest engine budget in the pack (`eng_cloud_sonilo` refuses a
#: longer prompt with a ValueError, and a render must never die on length).
ENGINE_PROMPT_MAX_CHARS = 1000


@dataclass(frozen=True)
class EnginePrompt:
    """What one engine call hears: the positive ``text``, the ``negative``
    prompt, and the ``palette_key`` of the ensemble that leads the text."""
    text: str
    negative: str
    palette_key: str


# Keyword-mined mood tags from news.script_brief (last-ditch fallback when the
# brief carries no music_mood_terms and no atmosphere). Case-insensitive.
_MOOD_TAGS: dict[str, str] = {
    "betrayal":   "minor mode, unresolved tension",
    "discovery":  "rising figure, slight upward motion",
    "loss":       "subdued, slow decay",
    "urgent":     "urgent rising line, restless bass",
    "isolation":  "sparse texture, wide stereo field",
    "danger":     "building tension, dissonant cluster",
    "mystery":    "harmonic ambiguity, slow modulation",
    "triumph":    "resolving cadence, brighter register",
    "conflict":   "opposing voices, clashing harmony",
    "silence":    "minimal density, long pauses",
}


#: THE NEGATIVE FOR A BANK WHOSE GENRE HAS A BEAT (operator, 2026-09-12).
#: What is still a defect in any music stays: noise, hiss, static, crackle,
#: distortion, clipping and anyone singing. What goes is the entire
#: anti-rhythm half, because on a techno, house or salsa bank those words
#: describe the brief. Carrying the underscore's negative onto a dance
#: floor is the same self-cancelling request that cost this project a
#: night: a cue that asked for "raw distorted TR-909" while the negative
#: banned distortion, and tore every time.
NEGATIVE_PROMPT_RHYTHMIC = (
    "noise, static, hiss, white noise, radio static, crackle, distortion, "
    "clipping, silence, speech, vocals, singing, lyrics, spoken word, "
    "out of tune, sloppy timing, muddy mix"
)


def negative_for(palette) -> str:
    """The negative prompt this ensemble should hear."""
    return (NEGATIVE_PROMPT_RHYTHMIC if palette.rhythmic
            else NEGATIVE_PROMPT_DEFAULT)


def _mood_suffix(script_brief: str) -> str:
    """Mine mood tags from the news script_brief. Returns a comma-prefixed
    suffix (e.g. ', minor mode, unresolved tension') or '' if nothing matches."""
    if not script_brief:
        return ""
    low = script_brief.lower()
    tags: list[str] = []
    seen: set[str] = set()
    for keyword, tag in _MOOD_TAGS.items():
        if keyword in low and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return (", " + ", ".join(tags)) if tags else ""


def compose_music_prompt(meta: dict, cue_id: str) -> tuple[str, int]:
    """Compose a music cue's ROW text from the Meta brief, returning (prompt,
    duration_sec). Reads every brief field through the brief-reader protocol;
    never crashes on an absent / malformed brief (falls through to the house
    palette + the cue's arc, plus a neutral "atmospheric" -- except on a
    `groove_arc` palette, which omits that word; see the module docstring).
    """
    terms = (meta.get("story_brief_terms") or {}) if isinstance(meta, dict) else {}
    if not isinstance(terms, dict):
        terms = {}
    setting_raw = terms.get("setting") or []
    if not isinstance(setting_raw, list):
        setting_raw = []
    # Normalised: this is composed into the music TEXT prompt below, and the
    # brief emits identifier case (PBUG-20260903-04).
    setting_terms = [spoken_term(t) for t in setting_raw if spoken_term(t)]

    # Mood: v2 music_mood_terms (via the protocol reader) -> v1 atmosphere ->
    # keyword-mined produced logline / news.script_brief.
    mood_terms: list[str] = []
    music_mood_raw = _read_brief_field(meta, "music_mood_terms", default=[])
    if isinstance(music_mood_raw, list):
        mood_terms = [str(t).strip() for t in music_mood_raw if str(t).strip()]
    if mood_terms:
        mood_terms = mood_terms[:3]
    else:
        atmosphere_raw = terms.get("atmosphere") or []
        if not isinstance(atmosphere_raw, list):
            atmosphere_raw = []
        atmosphere = [str(t).strip() for t in atmosphere_raw if str(t).strip()]
        if atmosphere:
            mood_terms = atmosphere[:3]
        else:
            # Meta split (2026-07-09): the last-ditch mood mining prefers
            # the PRODUCED story's logline (a summary of the actual
            # episode) over the pre-generation source digest; the digest
            # survives only as the final floor for old ledgers.
            produced = (meta.get("produced_story") or {}) if isinstance(meta, dict) else {}
            if not isinstance(produced, dict):
                produced = {}
            news_meta = (meta.get("news") or {}) if isinstance(meta, dict) else {}
            if not isinstance(news_meta, dict):
                news_meta = {}
            seed_text = (
                produced.get("logline") or news_meta.get("script_brief") or ""
            )
            kw = _mood_suffix(seed_text).lstrip(", ").strip()
            mood_terms = [t.strip() for t in kw.split(",") if t.strip()] if kw else []

    setting_str = ", ".join(setting_terms[:2]) if setting_terms else ""
    palette = story_palette(meta)

    parts: list[str] = []
    # The brief's own words first (story relevance a reader can check).
    #
    # THE NEUTRAL FLOOR IS A SUSTAINED WORD, so a `groove_arc` palette does
    # not get it (operator's ear, 2026-09-12). With no brief there are no mood
    # terms, and "atmospheric" then LED the prompt -- a texture instruction
    # standing in front of "Detroit techno at 128 BPM" and agreeing with the
    # pads that used to close the palette.
    #
    # `groove_arc` AND NOT `rhythmic`, deliberately: jazz and salsa are
    # rhythmic too and he judged their cues RIGHT, floor and all, so they keep
    # the word. This condition is narrower than "a bank with a genre" and the
    # two must not be conflated.
    #
    # The brief's real words are kept on every bank: moods like "ominous,
    # uneasy" are harmonic, not textural, and a dance cue can carry them. Only
    # the invented floor is dropped.
    if mood_terms:
        parts.append(", ".join(mood_terms))
    elif not palette.groove_arc:
        parts.append("atmospheric")
    if palette.rhythmic:
        # THE GENRE IS THE MUSICAL INSTRUCTION on a bank that has one, and
        # the orchestral devices and the anti-rhythm tempo phrase would
        # both argue with it -- "slow tempo, sustained and taut, no rubato"
        # is a fine thing to tell a string section and a contradiction to
        # tell a drum machine. The idiom carries the BPM instead, which is
        # the one musical-time fact the model answers accurately.
        parts.append(palette.idiom)
    else:
        parts.extend(mood_devices(mood_terms))
        # The ONLY musical-time information the model gets: duration is a
        # separate return value and never reaches the text, so without this
        # the model chooses its own pace, and at this length it chooses a
        # loop.
        parts.append(tempo_phrase(mood_terms))
        parts.append(palette.idiom)
    if setting_str:
        parts.append(f"evokes {setting_str}")
    parts.append((_CUE_CHARACTER_RHYTHMIC if palette.groove_arc
                  else _CUE_CHARACTER)[cue_id])
    prompt = ", ".join(parts) + _PROMPT_TAIL
    return prompt, CUE_DURATIONS[cue_id]


def _trim_at_clause(text: str, budget: int) -> str:
    """``text`` cut to at most ``budget`` characters at a clause boundary when
    one sits in the second half of the cut, else at the budget."""
    cut = text[:max(0, budget)]
    at = max(cut.rfind(", "), cut.rfind("; "), cut.rfind(". "))
    if at > budget // 2:
        cut = cut[:at]
    return cut.rstrip(" ,;.")


def compose_engine_prompt(meta: dict, row_text: str) -> EnginePrompt:
    """What the engine hears for one cue: the story palette's instruments, the
    production anchor, then ``row_text`` -- the composed row text on the
    legacy lane, or an AUTHORED row's ``generation_prompt`` verbatim on the
    my_story / scifi_news_pro lanes (the palette and anchor only ever go in
    FRONT of it). Capped at ``ENGINE_PROMPT_MAX_CHARS`` by trimming the row
    text at a clause boundary; never raises."""
    palette = story_palette(meta)
    head = f"{palette.instruments}, {PRODUCTION_ANCHOR}. "
    body = str(row_text or "").strip()
    budget = ENGINE_PROMPT_MAX_CHARS - len(head)
    if len(body) > budget:
        if body.endswith(_PROMPT_TAIL):
            # A composed row overflowing keeps its instrumental-only tail:
            # the clauses that go are in the middle, never the instruction.
            body = (_trim_at_clause(body[:-len(_PROMPT_TAIL)],
                                    budget - len(_PROMPT_TAIL)) + _PROMPT_TAIL)
        else:
            body = _trim_at_clause(body, budget)
    return EnginePrompt(text=(head + body).strip(),
                        negative=negative_for(palette),
                        palette_key=palette.key)
