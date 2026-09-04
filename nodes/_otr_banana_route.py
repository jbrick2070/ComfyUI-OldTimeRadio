"""The banana route -- a house-style comedic transform for VISUAL prompts.

Weapon nouns in still/video prompts become bananas (operator ruling
2026-08-06, "visuals only"): the announcer says "he drew his revolver" over a
shot of a man wielding a banana, and the spoken script is never touched. This
is an authorial style option, NOT a content guardrail -- the 2026-08-05 rip of
the visual-safety clauses stands, and this module must never grow stakes
rewrites (murder/blood/kill stay untouched: bananas replace the INSTRUMENTS,
never the stakes).

Contract: docs/2026-08-06-BUILD-SPEC-banana-route.md (the v3 contract -- scoped card shielding).
Applied at exactly two funnels -- the still dispatcher
(otr_image_gen_dispatcher, before the prompt content hash) and the video
render driver (after _apply_visual_safety_prompt) -- gated by:

    env switch ON  AND  (bank not a fidelity lane  OR  include override)

| env key | default | meaning |
|---|---|---|
| OTR_BANANA_STILLS | ON | transform at the still funnel |
| OTR_BANANA_VIDEO | ON | transform at the video funnel |
| OTR_BANANA_INCLUDE_FIDELITY_BANKS | OFF | force the route on shakespeare/public_domain |

Env is read per call; on a RESIDENT server an env change needs a fresh boot
(no widget exists by operator ruling -- no canonical-workflow change, ever).

THE DEFAULT GRAPH NOW FEEDS THIS ROUTE (banner corrected 2026-08-28 -- the
paragraph here described a retired graph for months): the shipped canonical
selects `still_flat (16:9)` for all three roles, which MINTS STILLS from
text prompts, so the route is live by default. The corpus agrees:
`banana_route` appears in 320 ledgers and `still_flat` in 319 of them. The
old text claimed the canonical ran procedural viz_* engines that mint no
stills -- true once, long gone.

Split-switch honesty: stills-on/video-off (or inverse) is a real capability
only on t2v lanes. On i2v lanes the anchor still carries the look, so a split
state invites the model to morph a banana anchor back toward the gun the text
names. The still funnel is the load-bearing joke site; the video funnel is
mostly consistency.

The STILL LANE IS INTENTIONALLY UNCAPPED, and that asymmetry is deliberate --
do not "fix" it by calling :func:`cap_phrase_safe` from the still dispatcher.
The video funnel re-caps because specific video branches ADVERTISE a character
budget (``finish_visual_prompt``'s 188/620, the 240-char LTX motion budget) and
a substitution that grows the text after that cap would break a promise the
branch made. No still composer publishes such a number, so a still cap would
invent a truncation contract nobody owns and would move every still prompt
hash for nothing.

Quote shielding exists for a script-integrity reason, not politeness, and it
is SCOPED TO CARD TEXT -- it is not a general "quotes are sacred" rule. Two
card shapes are shielded, both composed by ``compose_still_word_prompt``:

* WORD mode -- ``a title card displaying the words "<spoken line>"``. The
  quoted span is script RENDERED as picture text, so transforming it would put
  a substituted word on the one audience-readable surface.
* MUSIC mode -- ``an abstract picture evoking "<episode title>"``. The card is
  wordless, but the title is still script: the credits roll DISPLAYS it and
  the announcer SPEAKS it, so a bananafied evocation would contradict the
  episode.

Everywhere else quotes are decorative and are NOT shielded. A writer LLM
styling an ordinary prompt as ``a man carrying a "revolver"`` used to have
that revolver survive untransformed -- the route silently under-firing. So
``apply()`` takes ``shield_quoted_card_text``: the still dispatcher passes
True only for objects stamped ``source == "still_word"``, and the video funnel
passes False (no card composes on that lane -- a card's words travel in the
minted still, never in a video text prompt).

Hash contract: ``sha256_before``/``sha256_after`` are raw-UTF-8 hex digests of
the prompt string. They are deliberately NOT the dispatcher's
``_prompt_content_hash`` (which json-wraps the text first). Two hashes, two
purposes, never compared.

Pure; stdlib only (re, os, hashlib, dataclasses) + the zero-dependency
``_otr_bank_variants.base_source_bank_id`` leaf. No import-time side effects,
no NODE_CLASS_MAPPINGS entry.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

try:  # package import (production)
    from ._otr_bank_variants import base_source_bank_id
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_bank_variants import base_source_bank_id  # type: ignore

#: Versions the COMPLETE transform algorithm -- the substitution table AND the
#: quote-shield SCOPE. "3" is the scoped-shield revision: quoted spans are
#: protected only on still_word card prompts, where "2" protected them
#: everywhere. Bumped so a v2 and a v3 receipt can never claim the same
#: contract. NOTE: this is NOT the ``otr-banana-v2:`` variety hash namespace
#: below -- that string is a hash domain separator and moving it would
#: reshuffle every episode's fruit picks.
TABLE_VERSION = "3"

#: COPIED from nodes/_otr_casting.py:1238 per the operator's ruling ("copy the
#: existing idiom -- three lines, not new machinery"). Drift is closed by a
#: TEST, not a shared module: the suite asserts this frozenset equals
#: _otr_casting._LEMMY_EXCLUDED_SOURCE_BANK_IDS, so the tree keeps ONE answer
#: to "what is a fidelity lane".
_BANANA_EXCLUDED_SOURCE_BANK_IDS = frozenset({"public_domain", "shakespeare"})

# --------------------------------------------------------------------------
# Variety pools (per-episode, shape-class; index 0 of every pool is the v1
# replacement so an empty variety key is byte-identical to the base table).
# v1 SHIPS SIDEARM + LONG ONLY; everything else is PINNED (QA ruling 8).
# Pool entries carry explicit (singular, plural) forms -- no pluralizer.
# --------------------------------------------------------------------------

_VARIETY_POOLS: dict = {
    "SIDEARM": (("banana", "bananas"), ("red banana", "red bananas")),
    "LONG": (("long banana", "long bananas"), ("plantain", "plantains")),
}
#: Canonical class order for the receipt string -- one format, everywhere.
_CLASS_ORDER = ("SIDEARM", "LONG")

# Row kinds: ("class", cls, 0|1) -> pool pick (singular|plural form);
# ("wrap", cls, 0|1) -> the gunman phrase around the pool pick;
# ("lit", singular, plural picked by index) -> literal replacement.
_ROWS: tuple = (
    # -- SIDEARM (pooled) --
    ("gun", ("class", "SIDEARM", 0)), ("guns", ("class", "SIDEARM", 1)),
    ("handgun", ("class", "SIDEARM", 0)), ("handguns", ("class", "SIDEARM", 1)),
    ("pistol", ("class", "SIDEARM", 0)), ("pistols", ("class", "SIDEARM", 1)),
    ("revolver", ("class", "SIDEARM", 0)), ("revolvers", ("class", "SIDEARM", 1)),
    ("six-shooter", ("class", "SIDEARM", 0)), ("six-shooters", ("class", "SIDEARM", 1)),
    ("firearm", ("class", "SIDEARM", 0)), ("firearms", ("class", "SIDEARM", 1)),
    ("weapon", ("class", "SIDEARM", 0)), ("weapons", ("class", "SIDEARM", 1)),
    ("blaster", ("class", "SIDEARM", 0)), ("blasters", ("class", "SIDEARM", 1)),
    # -- the gunman wrapper (operator ruling (c): wielding, never holding --
    # the pose is the gag; the man stays a man, the banana moves to his hand)
    ("gunman", ("wrap", "SIDEARM", 0)), ("gunmen", ("wrap", "SIDEARM", 1)),
    # -- LONG (pooled) --
    ("rifle", ("class", "LONG", 0)), ("rifles", ("class", "LONG", 1)),
    ("carbine", ("class", "LONG", 0)), ("carbines", ("class", "LONG", 1)),
    ("musket", ("class", "LONG", 0)), ("muskets", ("class", "LONG", 1)),
    ("assault rifle", ("class", "LONG", 0)), ("assault rifles", ("class", "LONG", 1)),
    ("sniper rifle", ("class", "LONG", 0)), ("sniper rifles", ("class", "LONG", 1)),
    # -- PINNED: scatter / rapid-fire --
    ("shotgun", ("lit", "bunch of bananas")), ("shotguns", ("lit", "bunches of bananas")),
    ("machine gun", ("lit", "bunch of bananas")), ("machine guns", ("lit", "bunches of bananas")),
    ("tommy gun", ("lit", "bunch of bananas")), ("tommy guns", ("lit", "bunches of bananas")),
    ("submachine gun", ("lit", "bunch of bananas")), ("submachine guns", ("lit", "bunches of bananas")),
    # -- PINNED: tiny / antique --
    ("derringer", ("lit", "banana")), ("derringers", ("lit", "bananas")),
    ("flintlock", ("lit", "banana")), ("flintlocks", ("lit", "bananas")),
    ("blunderbuss", ("lit", "banana")), ("blunderbusses", ("lit", "bananas")),
    # -- PINNED: energy --
    ("ray gun", ("lit", "banana beam")), ("ray guns", ("lit", "banana beams")),
    ("death ray", ("lit", "banana beam")), ("death rays", ("lit", "banana beams")),
    ("disintegrator", ("lit", "banana beam")), ("disintegrators", ("lit", "banana beams")),
    # -- PINNED: blades (a sabre is already banana-shaped; the yellow crescent
    # IS the joke, so blades never rotate)
    ("knife", ("lit", "banana")), ("knives", ("lit", "bananas")),
    ("dagger", ("lit", "banana")), ("daggers", ("lit", "bananas")),
    ("switchblade", ("lit", "banana")), ("switchblades", ("lit", "bananas")),
    ("sword", ("lit", "banana")), ("swords", ("lit", "bananas")),
    ("sabre", ("lit", "banana")), ("sabres", ("lit", "bananas")),
    ("saber", ("lit", "banana")), ("sabers", ("lit", "bananas")),
    ("rapier", ("lit", "banana")), ("rapiers", ("lit", "bananas")),
    ("cutlass", ("lit", "banana")), ("cutlasses", ("lit", "bananas")),
    ("bayonet", ("lit", "banana")), ("bayonets", ("lit", "bananas")),
    ("machete", ("lit", "banana")), ("machetes", ("lit", "bananas")),
    ("straight razor", ("lit", "banana")), ("straight razors", ("lit", "bananas")),
    ("ice pick", ("lit", "banana")), ("ice picks", ("lit", "bananas")),
    # -- PINNED: clubs (bare "club" is EXCLUDED -- card suits, night clubs)
    ("truncheon", ("lit", "banana")), ("truncheons", ("lit", "bananas")),
    ("billy club", ("lit", "banana")), ("billy clubs", ("lit", "bananas")),
    # -- PINNED: thrown / misc --
    ("hand grenade", ("lit", "banana")), ("hand grenades", ("lit", "bananas")),
    ("grenade", ("lit", "banana")), ("grenades", ("lit", "bananas")),
    ("bazooka", ("lit", "banana")), ("bazookas", ("lit", "bananas")),
    ("brass knuckles", ("lit", "banana peels")),
)
# RESERVED, deliberately absent: bomb, missile, torpedo, cannon (v1 is
# unambiguous hand props; `cannon -> banana cannon` would break single-pass
# closure and the closure test will catch anyone who tries).
# EXCLUDED, deliberately absent (collision with OTR's own composed prompt
# vocabulary -- measured in the 7b sweep): shot/shoot/shooting, tank, axe,
# hatchet, blackjack, bare club, poison, harpoon, dynamite, gunfire, gunshot,
# at gunpoint. All verbs, all gore: instruments, never stakes.

#: The gunman phrase templates (singular, plural).
_WRAP_TEMPLATES = ("man wielding a {}", "men wielding {}")


def select_varieties(variety_key: str) -> dict:
    """Per-episode shape-class picks. Pure and deterministic.

    Empty key -> index 0 everywhere (the exact v1 table). Each class hashes
    independently so adding a class later never reshuffles existing ones.
    Variety rotates PER EPISODE (key = the ledger's freeze_timestamp), never
    per re-render -- re-rendering a frozen ledger reproduces the same fruits,
    which is what makes still/video coherence structural.
    """
    picks = {}
    for cls in _CLASS_ORDER:
        pool = _VARIETY_POOLS[cls]
        if not variety_key:
            idx = 0
        else:
            digest = hashlib.sha256(
                ("otr-banana-v2:%s:%s" % (variety_key, cls)).encode("utf-8")
            ).hexdigest()[:8]
            idx = int(digest, 16) % len(pool)
        picks[cls] = idx
    return picks


def varieties_receipt(picks: dict) -> str:
    """The canonical compact receipt string, one format everywhere."""
    return ",".join(
        "%s=%s" % (cls, _VARIETY_POOLS[cls][picks.get(cls, 0)][0])
        for cls in _CLASS_ORDER)


def _resolve_replacement(spec, picks: dict) -> str:
    kind = spec[0]
    if kind == "lit":
        return spec[1]
    cls, form = spec[1], spec[2]
    pool_entry = _VARIETY_POOLS[cls][picks.get(cls, 0)]
    if kind == "class":
        return pool_entry[form]
    # wrap: the gunman phrase around the class pick
    return _WRAP_TEMPLATES[form].format(pool_entry[form])


def _table_for(picks: dict) -> tuple:
    """The concrete (source, replacement) table for an episode's picks,
    longest-source-first (enforced here AND asserted by test)."""
    rows = [(src, _resolve_replacement(spec, picks)) for src, spec in _ROWS]
    rows.sort(key=lambda r: (-len(r[0]), r[0]))
    return tuple(rows)


#: The base (index-0 everywhere) table, exported for tests and docs.
BANANA_TABLE: tuple = _table_for({cls: 0 for cls in _CLASS_ORDER})


def _match_case(replacement: str, source_match: str) -> str:
    """Case preservation: lowercase -> canonical replacement as-is;
    Title-case -> first alphabetic char of the WHOLE replacement uppercased;
    ALL-CAPS -> the entire replacement uppercased; anything else -> canonical."""
    if source_match.isupper() and len(source_match) > 1:
        return replacement.upper()
    if source_match[:1].isupper():
        for i, ch in enumerate(replacement):
            if ch.isalpha():
                return replacement[:i] + ch.upper() + replacement[i + 1:]
        return replacement
    return replacement


# --------------------------------------------------------------------------
# Quote shielding -- the state machine (QA ruling 9: same-style pairing only).
# --------------------------------------------------------------------------

_OPENERS = {'"': '"', "“": "”"}  # straight->straight, curly->curly


def _is_escaped(text: str, i: int) -> bool:
    """A delimiter is escaped iff its preceding backslash run has ODD length."""
    n = 0
    j = i - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return n % 2 == 1


def _shielded_spans(text: str) -> list:
    """[(start, end)] of quoted spans, delimiters included. Same-style pairing
    only; an unmatched opener (or closer) is LITERAL text -- the scan resumes
    after it, never silently shielding the rest of the prompt."""
    spans = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _OPENERS and not _is_escaped(text, i):
            closer = _OPENERS[ch]
            j = i + 1
            while j < n:
                if text[j] == closer and not _is_escaped(text, j):
                    break
                j += 1
            if j < n:
                spans.append((i, j + 1))
                i = j + 1
                continue
        i += 1
    return spans


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class BananaResult:
    text: str
    substitutions: int
    table_version: str
    sha256_before: str
    sha256_after: str
    varieties: str


def apply(text: str, *, variety_key: str = "",
          shield_quoted_card_text: bool = True) -> BananaResult:
    """Transform every weapon noun; pure, idempotent, unconditional.

    The enable gate lives at the call sites (env + bank idiom); apply() itself
    always transforms. Idempotence is by CONSTRUCTION -- no replacement
    contains any source term (the closure test enumerates every episode
    table) -- so historical adapter re-calls of the visual-safety seam and any
    accidental double application are harmless no-ops.

    ``shield_quoted_card_text`` selects whether QUOTED SPANS are protected, and
    it is a policy the CALLER owns because only the caller knows what it is
    composing (see the module docstring for the two shielded card shapes):

    * True -- quoted spans pass through byte-identical. Correct ONLY for
      still_word card prompts, where the quoted text is script.
    * False -- quotes are treated as decoration and their contents transform.
      Correct for portraits, scene stills and every video prompt.

    The default is True so that a caller who forgets fails toward UNDER-firing
    (the route does less, harmlessly) rather than toward transforming card
    script, which is the one outcome the visuals-only ruling forbids. Both
    production funnels pass it EXPLICITLY regardless; a new caller transforming
    ordinary visual prompts must pass ``shield_quoted_card_text=False``.
    """
    original = str(text or "")
    picks = select_varieties(variety_key)
    table = _table_for(picks)
    spans = _shielded_spans(original) if shield_quoted_card_text else []

    # Split into (segment, shielded) pieces preserving order.
    pieces = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            pieces.append((original[cursor:start], False))
        pieces.append((original[start:end], True))
        cursor = end
    if cursor < len(original):
        pieces.append((original[cursor:], False))

    total = 0
    out_parts = []
    for segment, shielded in pieces:
        if shielded:
            out_parts.append(segment)
            continue
        for source, replacement in table:
            pattern = re.compile(
                r"\b" + re.escape(source) + r"\b", re.IGNORECASE)

            def _repl(m, _replacement=replacement):
                return _match_case(_replacement, m.group(0))

            segment, n = pattern.subn(_repl, segment)
            total += n
        out_parts.append(segment)
    result = "".join(out_parts)
    return BananaResult(
        text=result,
        substitutions=total,
        table_version=TABLE_VERSION,
        sha256_before=_sha(original),
        sha256_after=_sha(result),
        varieties=varieties_receipt(picks),
    )


# --------------------------------------------------------------------------
# The phrase-safe post-transform cap (QA ruling 3).
# --------------------------------------------------------------------------

def _all_episode_pick_combos() -> list:
    """Every possible per-episode pick dict -- the REAL Cartesian product over
    the pools (QA ruling 8: enumerate, never assert a magic count). Shared by
    the cap's phrase inventory and the closure test."""
    import itertools
    ranges = [range(len(_VARIETY_POOLS[cls])) for cls in _CLASS_ORDER]
    return [dict(zip(_CLASS_ORDER, combo))
            for combo in itertools.product(*ranges)]


#: Every multi-word replacement phrase any episode table can produce; the cap
#: must never cut inside one.
_PHRASES = tuple(sorted(
    {rep for picks in _all_episode_pick_combos()
     for _s, rep in _table_for(picks) if " " in rep},
    key=len, reverse=True))

_TRAILING_CLAUSE = "no on-screen text"


def _last_index_ci(text: str, needle: str) -> int:
    """Index of the LAST case-insensitive occurrence of ``needle``, computed
    against the ORIGINAL string so the index is always valid for it.

    ``str.lower()`` can CHANGE LENGTH (U+0130 lowercases to two characters), so
    an index taken from a lowered copy can be off by the difference and slice
    mid-clause. An empty needle returns -1 rather than matching: ``rfind("")``
    answers ``len(text)``, which would collapse the whole tail into the
    protected region."""
    if not needle:
        return -1
    found = -1
    for match in re.finditer(re.escape(needle), text, re.IGNORECASE):
        found = match.start()
    return found


def _phrase_starts_covering(body_lower: str, cut: int) -> list:
    """Start offsets of every replacement-phrase occurrence that STRICTLY
    covers ``cut`` -- i.e. every phrase the cut would land inside."""
    starts = []
    for phrase in _PHRASES:
        i = body_lower.find(phrase)
        while i >= 0:
            if i < cut < i + len(phrase):
                starts.append(i)
            i = body_lower.find(phrase, i + 1)
    return starts


def _retreat_to_phrase_boundary(body: str, cut: int) -> int:
    """Walk ``cut`` left until it sits outside EVERY replacement phrase.

    Retreating ONCE is not enough, and this was measured rather than reasoned:
    the shipped single-retreat form truncated inside a phrase in 68 of 3,641
    cases. Two mechanisms defeat it. The phrase inventory OVERLAPS in ordinary
    text -- "...red banana peels off his mask" contains both
    ``man wielding a red banana`` and ``banana peels``, so retreating out of the
    second lands inside the first. And the separator strip can walk the cut back
    INTO a phrase, because a space is a legal interior character of every
    multi-word entry. So: retreat to the LEFTMOST covering start, strip
    separators, re-check, and repeat until the cut stops moving. ``cut``
    decreases monotonically, so this terminates."""
    lowered = body.lower()
    while cut > 0:
        starts = _phrase_starts_covering(lowered, cut)
        if not starts:
            break
        retreated = min(starts)
        while retreated > 0 and body[retreated - 1] in ", ":
            retreated -= 1
        if retreated >= cut:
            break
        cut = retreated
    return cut


def _longest_complete_phrase_prefix(body: str, budget: int) -> str:
    """The longest leading run of ``body`` that ends on a COMPLETE replacement
    phrase and still fits ``budget`` (``""`` when none does).

    Used only when the retreat consumed the entire body. Re-splitting a phrase
    to salvage some text is exactly what the retreat exists to prevent, so the
    salvage has to end on a whole phrase or not happen at all."""
    lowered = body.lower()
    best = ""
    for phrase in _PHRASES:
        i = lowered.find(phrase)
        while i >= 0:
            end = i + len(phrase)
            if end <= budget and end > len(best):
                best = body[:end]
            i = lowered.find(phrase, i + 1)
    return best


def cap_phrase_safe(text: str, max_chars: int,
                    protected_clause: str | None = None) -> str:
    """Word-boundary cap that never splits a replacement phrase and keeps the
    composing branch's engineered trailing clause.

    THE CONTRACT. A branch that capped its own prompt publishes the number it
    capped to AND the clause it engineered to protect; the banana funnel re-caps
    to that number and holds that clause intact, and does nothing else. A branch
    that publishes nothing is never capped at all.

    ``protected_clause`` is matched ANYWHERE, because a branch's promise covers
    everything from its clause to the end of the string -- that is how the
    brief+beat motion clause keeps the era tail that ``finish_visual_prompt``
    splices in AFTER it. ``_TRAILING_CLAUSE`` is the fallback and is honoured
    only when TRAILING, because a mid-prompt occurrence is content rather than a
    render constraint (the ruling ``finish_visual_prompt`` already made). An
    absent branch clause falls back rather than protecting nothing, so a branch
    whose composer already trimmed its clause never comes out WORSE than before.

    The funnel re-caps only when the transform CROSSES the published budget
    (``pre_len <= max_chars < post_len``), never on a shrink and never on a
    prompt that was already over budget before the route touched it. The
    returned string is always ``<= max_chars`` regardless, because this is
    exported and called directly by tests -- the postcondition must not depend
    on caller discipline.
    """
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    suffix = ""
    body = text
    idx = _last_index_ci(text, str(protected_clause or ""))
    if idx < 0 and text[-len(_TRAILING_CLAUSE):].lower() == _TRAILING_CLAUSE:
        idx = len(text) - len(_TRAILING_CLAUSE)
    if idx >= 0:
        # keep the clause and whatever separator precedes it
        sep_start = idx
        while sep_start > 0 and text[sep_start - 1] in ", ":
            sep_start -= 1
        suffix = text[sep_start:]
        body = text[:sep_start]
    budget = max_chars - len(suffix)
    if budget <= 0:
        # The clause alone does not fit. Keep the clause -- the branch
        # engineered it -- and drop the body. Unreachable from the funnel,
        # which caps only when the pre-transform text was already within budget.
        out = suffix.lstrip(", ")
    else:
        if len(body) > budget:
            cut = body.rfind(" ", 0, budget + 1)
            if cut <= 0:
                cut = budget
            trimmed = body[:_retreat_to_phrase_boundary(body, cut)].rstrip(", ")
            if not trimmed:
                trimmed = _longest_complete_phrase_prefix(body, budget)
            body = trimmed
        out = (body + suffix).lstrip(", ") if not body else body + suffix
    if not out:
        out = text[:max_chars]
    return out[:max_chars] if len(out) > max_chars else out


# --------------------------------------------------------------------------
# The gate: env switches + the fidelity-bank idiom.
# --------------------------------------------------------------------------

#: ``""`` is deliberately NOT here. ``raw is None`` below already covers
#: "unset", so an empty or whitespace-only value is a MALFORMED knob -- in
#: practice a launcher line with a trailing space -- and BUILD-SPEC section 1
#: sends anything outside these two sets to the default plus one warning. A
#: present-but-empty OTR_BANANA_INCLUDE_FIDELITY_BANKS used to read as TRUE and
#: silently bananafy the shakespeare / public_domain lanes.
_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

#: Distinct (env name, normalized token) pairs already warned about, so a
#: malformed knob costs ONE line per process instead of one per read. The gate
#: runs per still, per video request AND per beat at ShotLock cast-time
#: preflight, so an undeduped warning is hundreds of lines on one episode --
#: against BUILD-SPEC section 1's "one warning naming the key". Same idiom as
#: ``_otr_roster_gender._UNMAPPED_SEEN``; tests reset the entry they assert on.
_WARNED_MALFORMED_ENV: set = set()


def _bool_env(name: str, default: bool) -> bool:
    """Guarded boolean env read (BUILD-SPEC section 1): a malformed knob is
    IGNORED with one warning, never fatal."""
    raw = otr_env.get(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    seen_key = (name, token)
    if seen_key not in _WARNED_MALFORMED_ENV:
        _WARNED_MALFORMED_ENV.add(seen_key)
        import logging
        logging.getLogger("OTR").warning(
            "[banana_route] %s=%r is not boolean-like; IGNORING it and using "
            "the default (%s). A malformed knob must never lose a render.",
            name, raw, "on" if default else "off")
    return default


def banana_stills_enabled() -> bool:
    return _bool_env("OTR_BANANA_STILLS", True)


def banana_video_enabled() -> bool:
    return _bool_env("OTR_BANANA_VIDEO", True)


def include_fidelity_banks() -> bool:
    return _bool_env("OTR_BANANA_INCLUDE_FIDELITY_BANKS", False)


def source_bank_excludes_banana(source_bank_id) -> bool:
    """True when the bank's family is a source-faithful adaptation.

    Normalized through ``base_source_bank_id`` so bake-off variants
    (``shakespeare_v2``, ``public_domain_v3``) inherit the exclusion --
    fidelity is a family behaviour, not a per-row opt-in (the _LEMMY idiom,
    copied per operator ruling; equality with the casting frozenset is
    test-enforced)."""
    normalized = base_source_bank_id(str(source_bank_id or "").strip().lower())
    return normalized in _BANANA_EXCLUDED_SOURCE_BANK_IDS


def banana_gate(ledger_meta, *, lane: str) -> bool:
    """The effective per-funnel gate: env switch AND bank policy.

    ``lane`` is ``"stills"`` or ``"video"``. A ledger with no
    ``meta.source_bank`` (hand-built harness requests) is NOT excluded -- the
    global default applies."""
    if lane == "stills":
        if not banana_stills_enabled():
            return False
    elif lane == "video":
        if not banana_video_enabled():
            return False
    else:
        raise ValueError("banana_gate lane must be 'stills' or 'video', got %r"
                         % (lane,))
    bank = ""
    if isinstance(ledger_meta, dict):
        bank = str(ledger_meta.get("source_bank") or "")
    if bank and source_bank_excludes_banana(bank):
        return include_fidelity_banks()
    return True


def receipt_keys(result: BananaResult) -> dict:
    """The six receipt keys for an ON-path transform (a real BananaResult).
    OFF paths use :func:`off_receipt` instead -- the two are the whole
    receipt vocabulary; there is no third shape."""
    return {
        "banana_route": "on",
        "banana_table_version": result.table_version,
        "banana_substitutions": result.substitutions,
        "banana_sha256_before": result.sha256_before,
        "banana_sha256_after": result.sha256_after,
        "banana_varieties": result.varieties,
    }


def off_receipt(prompt: str, *, variety_key: str = "") -> dict:
    """The OFF-path receipt: which table and which fruits WOULD have applied
    (QA fix for OFF-run forensics), with before == after and zero
    substitutions. Pure; does NOT transform."""
    digest = _sha(str(prompt or ""))
    return {
        "banana_route": "off",
        "banana_table_version": TABLE_VERSION,
        "banana_substitutions": 0,
        "banana_sha256_before": digest,
        "banana_sha256_after": digest,
        "banana_varieties": varieties_receipt(select_varieties(variety_key)),
    }


__all__ = [
    "TABLE_VERSION", "BANANA_TABLE", "BananaResult",
    "apply", "select_varieties", "varieties_receipt", "cap_phrase_safe",
    "banana_stills_enabled", "banana_video_enabled", "include_fidelity_banks",
    "source_bank_excludes_banana", "banana_gate", "receipt_keys",
    "off_receipt", "_all_episode_pick_combos",
]
