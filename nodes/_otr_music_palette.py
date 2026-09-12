"""nodes/_otr_music_palette.py -- what the music is PLAYED ON, per story.

Operator, 2026-09-11 (after listening to `moonlit_deception`): *"the 'music'
needs to be improved, doesn't sound like music ... sounds like radio hiss,
which is what we asked. So I am asking to go over ALL musical prompts and make
them more musical ... ideally it is relevant to the story."* And: it DID relate
to the story for a while (April / May) before drifting back to tape hiss.

A text-to-music model plays what it is told to play. Told "atmospheric, evokes
a moonlit wood, slow build, analog tape warmth" it renders texture; told
"lute, viol consort, recorder, Elizabethan consort music, minor key, tremolo
strings" it renders music. This module is the STORY -> ENSEMBLE step: a named
set of instruments and an idiom for the story's period and bank, and the
musical DEVICES the brief's mood words call for. `_otr_music_prompt` composes
them into the row text and the engine prompt; no engine prepends its own.

Keyed ONLY on ledger-stable facts -- `source_bank`, `source_meta` (the year,
read leniently: tonight's ledger says `"c. 1595"`), and the brief's
`music_mood_terms` -- so the same seed plays the same ensemble on replay. The
visual style roll is deliberately NOT an input: it is drawn from OS entropy
(`style_roll.seed_source`) and would make the instrumentation unreplayable.

PURE and total: no I/O, no engine imports, never raises on any meta shape --
a missing or malformed field is the house orchestra, never a dead render.
UTF-8 no BOM, ASCII-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


#: A cue is 4 to 12 seconds long and it is BACKGROUND. Two things follow,
#: and both were learned the hard way on 2026-09-12 when the music finally
#: became audible and turned out to be looping:
#:
#: * SUSTAINED INSTRUMENTS LEAD. A text-to-audio model reads the first
#:   tokens hardest, and a plucked or struck instrument (lute, harpsichord,
#:   pizzicato, any drum) has a fixed transient it can only repeat. Ask for
#:   those first in a 12-second window and the model writes a two-bar loop,
#:   which is exactly what Stable Audio Open is built to do best.
#: * ONE PACE PER CUE. Asking for "slow held chords" and "brushed drums"
#:   in the same breath is not a richer instruction, it is a contradictory
#:   one, and the model resolves it by picking the rhythmic half.
#:
#: Plucked instruments are not banned -- a lute IS the sound of 1595 -- they
#: are placed after the sustained ones and qualified ("gentle lute").
@dataclass(frozen=True)
class Palette:
    """A named ensemble: ``key`` is the stable receipt label, ``instruments``
    the comma list the ENGINE hears first, ``idiom`` the period phrase the ROW
    text carries so a reader of the ledger knows what was asked for.

    ``rhythmic`` says this ensemble IS its groove -- techno, house, salsa, a
    jazz quartet with a drummer. It switches three things that the rest of
    this module gets right only for sustained underscore: the rhythm section
    leads the prompt instead of being pushed behind the strings, the cue
    receives a negative prompt that does not forbid its own genre, and the
    BANK beats the period band so a public-domain story from 1890 still gets
    the house music it was promised rather than a string quartet.
    """
    key: str
    instruments: str
    idiom: str
    rhythmic: bool = False


#: The house sound of the show: a 1940s radio-drama pit orchestra.
HOUSE_PALETTE = Palette(
    "radio_orchestra",
    "warm strings, muted brass, clarinet, soft upright bass",
    "1940s radio drama orchestra",
)

EARLY_CONSORT = Palette(
    "early_consort",
    "viol consort, recorders, soft bowed strings, gentle lute",
    "Elizabethan consort music",
)
BAROQUE_CHAMBER = Palette(
    "baroque_chamber",
    "bowed strings, oboe, bassoon, gentle harpsichord",
    "baroque chamber music",
)
ROMANTIC_CHAMBER = Palette(
    "romantic_chamber",
    "string quartet, clarinet, French horn, soft piano",
    "Romantic-era chamber music",
)
ELECTRIC_COMBO = Palette(
    "electric_combo",
    "Hammond organ, sustained electric guitar, soft bass guitar",
    "1960s instrumental combo",
)
SCIFI_ORCHESTRA = Palette(
    "scifi_orchestra",
    "theremin, sustained strings, low brass, distant french horn",
    "1950s science-fiction radio orchestra",
)

#: THE PER-BANK GENRES (operator, 2026-09-12). Each leads with the thing
#: that actually defines it -- the drum machine, the congas, the rhythm
#: section -- because for these the groove is the subject and not the
#: backing. That is the opposite of the rule above, and the `rhythmic` flag
#: is what makes the difference explicit instead of accidental.
#:
#: The tempo is named in the idiom because a text-to-audio model answers a
#: BPM number accurately: measured 2026-09-12 over five renders, prompts
#: written at 124-135 BPM came back at 125.0, 125.0, 127.7, 130.4 and 133.3.
DETROIT_TECHNO = Palette(
    "detroit_techno",
    "Roland TR-909 drum machine, deep analog sub bass, detuned synth stabs, "
    "warm Juno pads",
    "Detroit techno at 128 BPM, hypnotic machine funk",
    rhythmic=True,
)
JAZZ_QUARTET = Palette(
    "jazz_quartet",
    "brushed drums, walking upright bass, piano comping, tenor saxophone",
    "small-group jazz quartet, relaxed swing",
    rhythmic=True,
)
SALSA_CONJUNTO = Palette(
    "salsa_conjunto",
    "congas and timbales, piano montuno, upright bass tumbao, bright brass "
    "section",
    "salsa conjunto at 100 BPM, clave-driven",
    rhythmic=True,
)
CHICAGO_HOUSE = Palette(
    "chicago_house",
    "Roland TR-707 drum machine, rolling bass line, warm piano chords, "
    "soft string pads",
    "Chicago house at 122 BPM, soulful and steady",
    rhythmic=True,
)

#: Period bands by the source's year: (exclusive upper bound, palette).
_PERIOD_BANDS = (
    (1650, EARLY_CONSORT),
    (1800, BAROQUE_CHAMBER),
    (1900, ROMANTIC_CHAMBER),
    (1960, HOUSE_PALETTE),
)
#: Past the last band.
_MODERN_PALETTE = ELECTRIC_COMBO

#: A bank with no usable year still has a period of its own.
_BANK_PALETTES = {
    "shakespeare": EARLY_CONSORT,
    "scifi_news_pro": DETROIT_TECHNO,
    "media_archive": JAZZ_QUARTET,
    "original": SALSA_CONJUNTO,
    "public_domain": CHICAGO_HOUSE,
}


def bank_music_table() -> list[tuple[str, str, bool]]:
    """``(bank, idiom, rhythmic)`` for every bank with a DECLARED palette,
    sorted by bank -- the five in ``_BANK_PALETTES``. ``my_story`` is not
    here because it has no fixed palette: it takes the ``music_style`` widget
    and falls back to the house orchestra, which the docs state beside this
    table. This is the ONE place the docs read the shipped defaults from, so
    README and every generated launch recipe say what the composer does rather
    than what somebody remembered (the defaults were undocumented from
    2026-09-12 dawn until the same evening); a test pins both against it."""
    return sorted((bank, palette.idiom, palette.rhythmic)
                  for bank, palette in _BANK_PALETTES.items())

#: The tempo words each device group asks for. TEMPO IS PER GROUP, not per
#: pace class (Fable, 2026-09-12): "unhurried, expressive rubato" is right
#: for grief and wrong for dread, because suspense is slow AND RIGID -- rubato
#: removes the pulse, and the pulse is the thing a listener feels. The pace
#: class below still exists, but only to decide which devices CONTRADICT each
#: other. Deliberately words and never a BPM: a background cue must not imply
#: a click track. None of them may say "steady", which is the word that asks
#: for a repeating figure.
_RUBATO = "slow tempo, unhurried, expressive rubato"
#: Tension WITHOUT a pulse. The first attempt at this line said "a held and
#: unwavering pulse" and measured the worst loopiness of the whole campaign --
#: 0.864 on a real canonical leg (shadows_in_the_mist, ominous/foreshadowing),
#: against 0.485 for the episode that started the complaint. Of course it did:
#: "unwavering pulse" IS a request for a repeating figure, which is the one
#: thing this table exists to avoid. Suspense is carried by sustain and
#: harmonic tension, not by a beat you could set a metronome to.
_TAUT_SUSTAIN = "slow tempo, sustained and taut, no rubato"
_DRIFTING = "slow tempo, floating and unmetered"
_MOVING = "moving tempo, flowing line"
_BROAD = "unhurried tempo, broad phrasing"

#: Mood words -> the musical devices that express them, each tagged with the
#: PACE it implies and the TEMPO it asks for. Matched at a word start, first group wins per term; the
#: short ambiguous stems are bounded so "warm" never reads as "war" and
#: "moonlight" never reads as "light".
#:
#: THE PACE TAG IS LOAD-BEARING (2026-09-12). `mood_devices` may return two
#: devices, and before this tag its only filter was dedup -- so a brief
#: reading "melancholic, playful" asked one 12-second cue for "slow cello,
#: held minor chords" AND "pizzicato strings, brushed drums" at once. The
#: model resolved that contradiction by playing the rhythmic half, in a loop.
#: Two devices may now be returned only when their paces agree, or when one
#: of them is neutral.
#:
#: NO PERCUSSION ANYWHERE IN THIS TABLE, and that is deliberate rather than
#: squeamish: a drum in a 4-to-12-second background cue can only be a loop,
#: and every percussion phrase that used to live here ("brushed drums",
#: "snare accents", "timpani") was in a cue the operator heard as a tape
#: deck. The emotions they carried are expressed by register and articulation
#: instead, which is what an orchestrator would reach for at this length.
_MOOD_DEVICES = (
    (r"\b(?:magic|enchant|dream|fairy|moonli|wonder|spell|fantas)",
     "shimmering sustained strings, soft celesta colour", "slow", _DRIFTING),
    (r"\b(?:tense|tension|suspens|dread|fear|danger|menac|uneasy|anxi|nervous)",
     "minor key, low brass swells, slow rising strings", "slow", _TAUT_SUSTAIN),
    (r"\b(?:sad\b|grief|loss\b|mourn|melanchol|sorrow|sombre|somber|elegiac|lament"
     r"|hopeless|loveless|despair|forlorn|lonel)",
     "a slow cello line over held minor chords", "slow", _RUBATO),
    (r"\b(?:dark\b|brood|sinister|ominous|grim\b|forebod|malevol)",
     "low strings, bass clarinet, slow dissonant chords", "slow", _TAUT_SUSTAIN),
    (r"\b(?:urgent|urgency|chase|frantic|action|battle|conflict|war\b|fight|pursuit|storm)",
     "urgent strings climbing over a restless bass line", "fast", _MOVING),
    (r"\b(?:grand\b|heroic|triumph|regal|majest|noble|royal|ceremon|epic\b)",
     "broad brass and soaring strings, a noble melody", "neutral", _BROAD),
    (r"\b(?:myster|eerie|uncanny|strange|haunt|secret|shadow|deceit|decept|intrigue)",
     "sustained strings, a sparse questioning melody", "slow", _TAUT_SUSTAIN),
    (r"\b(?:warm\b|warmth|tender|love\b|lovely|lover|loving|romanc|romantic|gentle"
     r"|hope\b|hopeful|affection|joy)",
     "major key, legato strings, soft woodwinds", "slow", _RUBATO),
    (r"\b(?:playful|comic|comed|whims|mischie|merry|jest|light-?hearted|witty)",
     "a light dancing woodwind melody, bright major colour", "fast", _MOVING),
    (r"\b(?:calm\b|serene|peace|pastoral|quiet|tranquil|still\b|reflective)",
     "gentle flute melody over soft strings", "slow", _RUBATO),
)
_COMPILED_MOOD_DEVICES = tuple(
    (re.compile(pattern, re.IGNORECASE), device, pace, tempo)
    for pattern, device, pace, tempo in _MOOD_DEVICES)

#: WHICH TEMPO WINS when a brief's devices ask for different ones (codex,
#: 2026-09-12). They used to be read in the order the BRIEF happened to list
#: its moods, which is the same order-dependence the majority-pace rule was
#: written to remove: ["grief","tense"] and ["tense","grief"] are the same
#: cue and were getting different tempo words.
#:
#: The order below is an ARGUMENT, not a measurement, and it is written down
#: here rather than left implicit in the mood table so that a reader can
#: disagree with it in one place. A cue of 4 to 12 seconds falls into a
#: repeating figure when the text leaves it room, so when two devices
#: disagree the phrase that denies metre most explicitly wins: unmetered
#: first, then sustained-and-taut, then rubato. The two remaining phrases
#: say least about repetition and come last.
_TEMPO_PRIORITY = (_DRIFTING, _TAUT_SUSTAIN, _RUBATO, _MOVING, _BROAD)
_TEMPO_RANK = {phrase: rank for rank, phrase in enumerate(_TEMPO_PRIORITY)}

#: WHICH PACE WINS A TIED VOTE. Its own constant, because `_MOOD_DEVICES`
#: order already has a job -- a term is classified by the FIRST pattern it
#: matches, so a phrase like "a dark and playful malice" is read by table
#: position -- and a tie-break that borrowed that order would silently
#: change classification whenever someone reordered the table for the other
#: reason (codex r2, 2026-09-12).
#: Slow wins, for the same reason as `_TEMPO_PRIORITY`: at four to twelve
#: seconds the fast reading is the one that comes back as a repeating
#: figure, so an evenly-split brief takes the reading that cannot loop.
_PACE_TIE_ORDER = ("slow", "fast")

#: The fallback when no device matched at all. A text-to-audio model is given
#: no metre otherwise -- duration is a separate argument and never reaches the
#: text -- and an unpaced request at this length comes back as a repeating
#: figure.
_PACE_TEMPO = {"slow": _RUBATO, "fast": _MOVING, "neutral": _BROAD}

#: When no mood word maps: still a musical instruction, never a texture, and
#: never "steady" -- that word asks for exactly the repetition this module
#: spent 2026-09-12 removing.
DEFAULT_DEVICE = "a clear unhurried melody over warm harmony"
DEFAULT_PACE = "slow"

_FOUR_DIGITS = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def year_of(source_meta) -> int | None:
    """The source's year as an int, read leniently from ``source_meta`` --
    ``1606``, ``"1606"``, ``"c. 1595"``, ``"1895-1897"`` all answer; anything
    without a plausible four-digit year answers ``None``. Never raises."""
    if not isinstance(source_meta, dict):
        return None
    for key in ("year", "date", "published", "first_performed", "period"):
        value = source_meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue  # a nested object is not a date, whatever digits it holds
        match = _FOUR_DIGITS.search(str(value))
        if match:
            year = int(match.group(1))
            if 1000 <= year <= 2100:
                return year
    return None


def bank_of(meta) -> str:
    """The source bank name the ledger meta carries, lower-cased, or ``""``."""
    if not isinstance(meta, dict):
        return ""
    for value in (meta.get("source_bank"), meta.get("bank"),
                  (meta.get("story_routing") or {}).get("bank")
                  if isinstance(meta.get("story_routing"), dict) else None,
                  (meta.get("source_meta") or {}).get("bank")
                  if isinstance(meta.get("source_meta"), dict) else None):
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


#: WHICH WAY AN UNRECOGNISED STYLE FALLS, stated as a rule rather than left
#: to a word list (codex r1, 2026-09-12). The first cut claimed to be
#: "biased toward rhythmic on doubt" and did the opposite: anything the
#: regex missed came back sustained and collected the anti-loop negative.
#: So "shoegaze" or "musique concrete" would have been asked for and then
#: forbidden their own drums -- the self-cancelling request that tears.
#:
#: The rule now: a typed style is SUSTAINED only when it names something
#: sustained, and everything else is treated as having a groove. That is
#: the asymmetry the measurements support. Wrongly sustained costs a cue
#: the anti-loop wording and risks a loop, which is a disappointment;
#: wrongly rhythmic asks for a beat while banning beats, which produced a
#: 400 ms broadband burst every time it was measured.
_RHYTHM_WORDS = re.compile(
    r"\b(?:techno|house|salsa|jazz|funk|disco|drum|drums|percussion|beat|"
    r"beats|groove|rhythm|rhythmic|bpm|dance|swing|reggae|ska|hip.?hop|"
    r"breakbeat|jungle|garage|electro|bossa|samba|mambo|cumbia|afrobeat|"
    r"march|marching|tango|polka|rock|metal|punk|bluegrass|banjo|"
    r"tabla|gamelan|taiko|conga|bongo|timbale|snare|kick|808|909|707)\b",
    re.IGNORECASE)

#: The other side of the rule: text that names sustained music, and is
#: therefore safe to give the anti-loop negative to.
_SUSTAINED_WORDS = re.compile(
    r"\b(?:drone|ambient|chant|chanting|plainsong|gregorian|choir|choral|"
    r"cello|violin|viola|strings|quartet|quintet|orchestra|orchestral|"
    r"consort|chamber|organ|harp|pad|pads|atmosphere|atmospheric|soundscape|"
    r"sustained|held|legato|nocturne|lullaby|hymn|requiem|adagio|"
    r"harmonium|accordion|theremin|flute|clarinet|oboe|bassoon)\b",
    re.IGNORECASE)


def custom_palette(style_text) -> "Palette | None":
    """The operator's typed style as a Palette, or ``None`` for a blank.

    The text is used VERBATIM as both the instruments and the idiom -- it is
    what the person actually wants to hear, and second-guessing it with a
    keyword table would be a worse instruction than their own words. Total
    over any input: a non-string, a blank, or whitespace is ``None``.
    """
    text = str(style_text or "").strip()
    if not text:
        return None
    text = " ".join(text.split())[:200]
    # A named groove is rhythmic. Otherwise it is sustained ONLY if it
    # says something sustained; anything unrecognised falls to rhythmic,
    # because that negative cannot contradict whatever was asked for.
    if _RHYTHM_WORDS.search(text):
        rhythmic = True
    elif _SUSTAINED_WORDS.search(text):
        rhythmic = False
    else:
        rhythmic = True
    return Palette("custom", text, text, rhythmic=rhythmic)


def story_palette(meta) -> Palette:
    """The ensemble for this story.

    A DECLARED GENRE BEATS THE PERIOD BAND (operator, 2026-09-12). The banks
    he named a genre for get it whatever year the source carries, because
    "public domain is Chicago house" is a statement about the BANK and most
    public-domain sources are Victorian -- read year-first, every one of them
    would have come back a string quartet and the instruction would have had
    no visible effect at all.

    Everything else is unchanged: year first, then bank, then the house
    orchestra. Total over every meta shape.
    """
    meta = meta if isinstance(meta, dict) else {}
    # ONLY MY STORY MAY NAME ITS OWN MUSIC (operator, 2026-09-12: "only
    # 'my story' allows an original music prompt"). Every other bank has a
    # fixed musical identity and that is the point of having one -- a
    # Shakespeare episode scored as surf rock is not a feature, and the sci-fi
    # news lane being Detroit techno every week is what makes it recognisable.
    # My Story is the bring-your-own lane, so it is the one that takes a
    # bring-your-own score, exactly as it already takes an authored prompt.
    bank = bank_of(meta)
    if bank == "my_story":
        typed = custom_palette(meta.get("music_style"))
        if typed is not None:
            return typed
    declared = _BANK_PALETTES.get(bank)
    if declared is not None and declared.rhythmic:
        return declared
    year = year_of(meta.get("source_meta"))
    if year is not None:
        for upper_bound, palette in _PERIOD_BANDS:
            if year < upper_bound:
                return palette
        return _MODERN_PALETTE
    return declared if declared is not None else HOUSE_PALETTE


def mood_devices(mood_terms, *, limit: int = 2) -> list[str]:
    """The musical devices the brief's mood words call for -- at most
    ``limit`` distinct phrases, in mood order; the default device when none
    of the words map. Total over junk input, ``limit`` included.

    THE DEVICES FOLLOW THE BRIEF; THE PACE AND TEMPO DO NOT. Which devices
    survive, and in what order, is the brief's own emphasis -- that is what
    "in mood order" means and it is deliberate. The cue's PACE and TEMPO are
    properties of the whole brief and are decided without reference to the
    order its words happen to arrive in (codex, 2026-09-12).

    A ``limit`` below one still returns one device, because a cue with no
    device has no prompt.

    ONE PACE PER CUE. A second device joins only when its pace agrees with
    the first, or when either is neutral. Before that rule a brief reading
    "melancholic, playful" asked a 12-second cue to be slow and held AND
    pizzicato with brushed drums, and the model answered with a loop
    (operator, 2026-09-12: "a loop-a-loop tape deck").
    """
    return _mood_devices_with_pace(mood_terms, limit=limit)[0]


def mood_pace(mood_terms) -> str:
    """The pace the brief's mood words imply -- ``"slow"`` / ``"fast"`` /
    ``"neutral"``. Drives the tempo phrase, which is the only musical-time
    information the model receives.

    Public because the pace is the FACT and the phrase is one rendering of
    it -- the tests assert on the fact."""
    return _mood_devices_with_pace(mood_terms, limit=2)[1]


def tempo_phrase(mood_terms) -> str:
    """The words that tell the model how fast to play -- dread gets a taut
    sustain where grief gets rubato. Never a BPM number: a background cue
    must not imply a click track.

    THE SAME MOODS GIVE THE SAME TEMPO IN ANY ORDER (codex, 2026-09-12).
    It used to take the first SURVIVING device's tempo, so the answer moved
    with the brief's word order and could contradict the cue's own pace:
    ["heroic","playful"] resolved to a FAST cue and then asked for heroic's
    unhurried phrasing. The tempo now comes from every matching device that
    AGREES with the winning pace -- including ones the ``limit`` dropped,
    since the tempo is a property of the brief and not of how many devices
    fit in the prompt -- ranked by `_TEMPO_PRIORITY`."""
    return _mood_devices_with_pace(mood_terms, limit=2)[2]


def _mood_devices_with_pace(mood_terms, *, limit: int = 2):
    """``(devices, pace, tempo)`` -- the shared body of the readers above.

    THE MAJORITY PACE WINS, not the first term's (Fable, 2026-09-12). The
    brief lists its mood words in no particular order, so letting the first
    one decide gave a comedy a rubato lullaby when "pastoral" happened to
    precede "playful". Every term is matched first, the pace the most terms
    agree on is chosen, and only then are devices kept -- the ones that agree
    with that pace, plus neutral ones, up to ``limit``. A device that
    contradicts the majority is dropped rather than blended, because one
    4-to-12-second cue cannot be two paces at once.

    NEITHER ARBITRATION MAY READ THE BRIEF'S ORDER (codex, 2026-09-12).
    Both of them used to, quietly: a tied vote fell to whichever term was
    listed first, and the tempo came from whichever surviving device was
    listed first. Both now fall back to a declared constant instead --
    `_PACE_TIE_ORDER` and `_TEMPO_PRIORITY`. The DEVICE LIST still follows
    the brief, and that is the one place where its order is signal rather
    than noise.
    """
    matched = []
    votes = []
    for term in (mood_terms or []):
        text = str(term or "")
        if not text.strip():
            continue
        for pattern, device, pace, tempo in _COMPILED_MOOD_DEVICES:
            if pattern.search(text):
                # EVERY MATCHING TERM VOTES, even when two of them share a
                # device: "playful" and "merry" are one device and two votes,
                # and counting devices instead read a two-thirds majority as a
                # tie. The device list stays deduplicated; the ballot does not.
                if pace != "neutral":
                    votes.append(pace)
                if device not in [m[0] for m in matched]:
                    matched.append((device, pace, tempo))
                break
    if not matched:
        return [DEFAULT_DEVICE], DEFAULT_PACE, _PACE_TEMPO[DEFAULT_PACE]
    winner = _winning_pace(votes)
    tempo = _arbitrate_tempo([(p, t) for _d, p, t in matched], winner)
    cap = _device_limit(limit)
    devices = []
    for device, pace, _tempo in matched:
        if len(devices) >= cap:
            break
        if pace != "neutral" and pace != winner:
            continue
        devices.append(device)
    return devices, winner, tempo


def _device_limit(limit) -> int:
    """How many devices a caller may have -- at least one, whatever it asks.

    A cue with no device has no prompt, so ``limit=0`` cannot mean zero; and
    junk cannot raise, because this runs inside a render.
    """
    try:
        return max(1, int(limit))
    except (TypeError, ValueError):
        return 2


def _winning_pace(paces) -> str:
    """The pace the most mood TERMS asked for -- one ballot per matching
    term. A tie falls to `_PACE_TIE_ORDER`, never to whichever word the brief
    listed first.
    """
    if not paces:
        return "neutral"
    most = max(paces.count(pace) for pace in set(paces))
    tied = [pace for pace in set(paces) if paces.count(pace) == most]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda pace: _PACE_TIE_ORDER.index(pace)
               if pace in _PACE_TIE_ORDER else len(_PACE_TIE_ORDER))


def _arbitrate_tempo(candidates, winner: str) -> str:
    """The one tempo phrase for a cue, from every matched device's
    ``(pace, tempo)``.

    THE TEMPO MAY NEVER CONTRADICT THE PACE, which is what made this an
    order bug rather than a style one: a FAST cue was being told to phrase
    broadly because a neutral device happened to be listed first. Devices
    that agree with the winning pace decide it; the neutral ones are read
    only when nothing else is left.
    """
    agreeing = [tempo for pace, tempo in candidates if pace == winner]
    if not agreeing:
        agreeing = [tempo for _pace, tempo in candidates]
    if not agreeing:
        return _PACE_TEMPO.get(winner, _PACE_TEMPO[DEFAULT_PACE])
    return min(agreeing, key=lambda phrase: _TEMPO_RANK.get(phrase, len(_TEMPO_RANK)))
