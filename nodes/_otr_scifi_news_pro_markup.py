"""Structural parser for the scifi_news_pro whole-play markup.

Python validates only delimiter order, nonempty fields, scene numbering, and
closed-roster identity. Ordinary authored casing, Unicode names, punctuation,
quotes, parentheses, brackets, line count, and coda style are preserved and
never influence acceptance. Transport whitespace around rows and delimiters is
not part of spoken prose.

The parser never repairs or rewrites a spoken word. A structurally malformed
artifact may receive a bounded same-story format repair upstream.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass

try:
    from ._otr_text_metrics import canonical_word_count
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_text_metrics import canonical_word_count  # type: ignore

__all__ = [
    "NewsProParseDefect",
    "ParseDefect",
    "ParsedLine",
    "ParsedScene",
    "ParsedScript",
    "SpeakerRoster",
    "build_speaker_roster",
    "normalize_scifi_news_pro_markup_text",
    "parse_scifi_news_pro_markup",
    "render_defects",
    "speaker_identity_key",
]

ANNOUNCER_NAME = "ANNOUNCER"

# --- line classifiers (structural delimiters; first match wins) -------------
_RE_TITLE = re.compile(r"^TITLE:\s*(.+)$", re.IGNORECASE)
_RE_MUSIC = re.compile(r"^MUSIC:\s*(.+)$", re.IGNORECASE)
_RE_SCENE = re.compile(r"^SCENE\s+(\d{1,2}):\s*(.+)$", re.IGNORECASE)
_RE_CODA = re.compile(r"^CODA:\s*(.+)$", re.IGNORECASE)
#: THE TERMINAL DELIMITER, AND IT USED TO DEMAND A PERIOD NOBODY ASKED FOR.
#:
#: This was ``^END\.\s*$``. A model that wrote a bare ``END`` fell past it, past
#: ``_RE_SPEAKER`` (which needs a colon), onto ``BAD_LINE_SHAPE`` -- and because
#: ``on_end`` never fired, the end-of-text check then added ``MISSING_END`` too.
#: TWO reported defects, ONE missing character. `scifi_news_pro` died on it at
#: 3.3 minutes with ``BAD_LINE_SHAPE: END`` (PBUG-20260815-03).
#:
#: FOUR ACCEPTED FORMS, and no more: ``END``, ``END.``, ``[END]``, ``[END.]``.
#: The bracketed pair is admitted because the lane's own house style brackets
#: transport elsewhere, and the bold-unwrap path (shape 4) already delivers
#: ``**END**`` here as a bare ``END``.
#:
#: WHAT STAYS A LOUD DEFECT, deliberately: an UNPAIRED bracket (``[END`` or
#: ``END]``), anything with trailing content (``END. Fade out.``), and any
#: content-bearing variant. Widening a terminal delimiter into "anything
#: containing END" is how a structural marker stops being structural -- the
#: point of this grammar is that the parser can tell the end of the script from
#: a line of dialogue about endings.
_RE_END = re.compile(r"^(?:END\.?|\[END\.?\])\s*$", re.IGNORECASE)
_RE_SPEAKER = re.compile(r"^([^:\r\n]+):\s*(\S(?:.*\S)?)$")


#: The ONLY emphasis markers this canonicalizer will remove, longest first so
#: ``**`` is tried before ``*``. A CLOSED grammar on purpose (kibitz r1): a
#: general Markdown sanitizer would be exactly the "silent massaging" the ladder
#: exists to prevent. Backticks, headings, bullets and HTML stay LOUD defects.
_TRANSPORT_MARKERS = ("**", "__", "*", "_")

#: The structural delimiters a WHOLE-LINE emphasis wrapper is allowed to hide.
#: Shape 4 (below) unwraps only when the result matches one of these, so the
#: rule stays "balanced emphasis around TRANSPORT is transport" rather than
#: becoming a general Markdown sanitizer. Ordered as the parser tries them.
_TRANSPORT_CLASSIFIERS = (_RE_TITLE, _RE_MUSIC, _RE_SCENE, _RE_CODA, _RE_END)


def _is_transport(line: str) -> bool:
    """True iff ``line`` is one of the structural delimiters."""
    return any(rx.match(line) for rx in _TRANSPORT_CLASSIFIERS)


def _family_balanced(inner: str, marker: str) -> bool:
    """Both LENGTHS of ``marker``'s family are balanced inside ``inner``.

    THE DEFECT THIS CLOSES (QA, 2026-08-03). The first shape-4 draft asked
    ``inner.count(marker) % 2 == 0``, which cannot tell ``*`` from ``**``
    because one is a substring of the other. A single unclosed ``**`` is TWO
    ``*`` characters, so it counted as EVEN and passed; a lone ``*`` inside a
    ``**`` wrapper was invisible to a ``"**"`` substring count and also passed.
    Measured escapes, each verified to reach an accepted field:

        *SCENE 5: **bold thing*            -> SCENE 5: **bold thing
        **SCENE 5: a *lonely marker here** -> SCENE 5: a *lonely marker here
        _SCENE 5: __vault door_            -> SCENE 5: __vault door

    Those strays land in ``parsed.title`` and ``scenes[].setting``, which feed
    the final title override and the shot-direction description -- exactly the
    "silently rewritten into accepted prose carrying a stray marker" outcome
    this module's docstring promises never happens.

    Count the DOUBLE form first, remove it, then count what single characters
    remain. A legitimately nested pair (``The **vault** deep``) still passes.
    """
    long_form = marker[0] * 2
    if inner.count(long_form) % 2:
        return False
    return inner.replace(long_form, "").count(marker[0]) % 2 == 0


def _canonicalize_transport_line(line: str) -> "tuple[str, tuple[str, ...]]":
    """Strip a BALANCED emphasis wrapper from a structural label. Report it.

    THE DEFECT THIS CLOSES (live, 2026-08-01). A local model emitted its markup
    wrapped in Markdown -- ``**TITLE:** ...``, ``**ANNOUNCER:** ...``,
    ``**END.**``. Every classifier here is ``^``-anchored, so a decorated line
    misses ``_RE_TITLE`` and then MATCHES the ``_RE_SPEAKER`` catch-all, whose
    group(1) becomes a character literally named ``**TITLE``. That is not in the
    cast, so each line raised UNKNOWN_SPEAKER *and* SKELETON_BREAK -- 106 of each
    across one campaign, and three of six episodes died in the writer.

    WHAT IS AND IS NOT AUTHORSHIP. ``_normalize_line`` already owns "transport"
    normalization, and emphasis wrapped around a DELIMITER is transport: it
    carries no story. Emphasis inside a spoken line IS authorship and is left
    byte-identical -- ``**BO NI:** Hello **world**`` canonicalizes to
    ``BO NI: Hello **world**``, keeping the payload's own markers.

    BALANCED ONLY. Four recognized shapes:

        <M>LABEL:<M> payload   ->  LABEL: payload      (wrapper spans the colon)
        <M>LABEL<M>: payload   ->  LABEL: payload      (wrapper before the colon)
        <M>TOKEN<M>            ->  TOKEN               (standalone, e.g. END.)
        <M>LABEL: payload<M>   ->  LABEL: payload      (wrapper spans the LINE)

    SHAPE 4 (2026-08-03, live). The 30-word sweep lost two legs to a model that
    wrote its structure as whole-line markdown -- ``**SCENE 5: The vault**``,
    ``**MUSIC**``, ``**CODA**``, ``**TITLE: ...**``. The wrapper spans label AND
    payload, so shapes 1-3 all miss: the body opens with a space after the colon
    (not the marker) and the label does not end with the marker. The line then
    fell to the ``_RE_SPEAKER`` catch-all and produced a character literally
    named ``**SCENE 5``, i.e. ``UNKNOWN_SPEAKER`` four attempts running until
    the markup ladder exhausted and the leg died in the writer having never
    reached a video engine.

    Shape 4 is ORDERED LAST and GATED ON TRANSPORT. Last, because checked
    earlier it would mangle the case this function exists to preserve:
    ``**BO NI:** Hello **world**`` also starts and ends with ``**``, and a naive
    outer strip yields ``BO NI:** Hello **world``. Gated, because unwrapping
    anything whose result is not a structural delimiter would turn this closed
    grammar into the general Markdown sanitizer the ladder exists to prevent --
    so ``**She turns: the room is empty**`` is left ALONE and stays loud.

    It also requires the REMAINDER to carry an EVEN number of that marker, so a
    line like ``**BO NI: Hello **world**`` (unbalanced interior) is never
    silently rewritten into accepted prose carrying a stray marker.

    An unbalanced, mixed or payload-internal marker is NOT touched and stays a
    loud defect. Roster-independent by design: it never consults the cast, so a
    genuinely unknown speaker still reaches UNKNOWN_SPEAKER after decoration is
    removed, and the closed-roster diagnostic keeps its meaning.

    **DO NOT "FIX" A STRAY UNMATCHED LEADING MARKER HERE.** It is the obvious
    next step -- ``*Ada: Hello`` misses the roster only because of one stray
    ``*`` -- and it was proposed and REJECTED on 2026-08-12 (PBUG-20260812-03),
    for two reasons:

    * it breaks this function's contract. Balanced, transport-only, malformed
      stays loud. Stripping unmatched decoration is the general Markdown
      sanitizer the paragraphs above exist to refuse.
    * **it would silently disarm the repair note that DOES fix this.** Strip the
      marker from ``*SFX:`` and you get ``SFX:`` -- still not a roster name, so
      still UNKNOWN_SPEAKER, but it no longer LOOKS like a stage direction. The
      writer's ``_standalone_stage_direction_repair_note`` keys on exactly that
      shape, so it would stop firing and the ladder would fall back to the
      generic instruction that already burned four attempts on a live leg.

    The stray-marker case is handled where it belongs: the writer's repair rung
    checks the decorated label against the roster and tells the model to restore
    the canonical label while keeping the dialogue.

    Returns ``(line, notes)``; every removal is reported, never hidden.
    """
    s = str(line).strip()
    if not s:
        return s, ()
    for marker in _TRANSPORT_MARKERS:
        if not s.startswith(marker):
            continue
        body = s[len(marker):]
        colon = body.find(":")
        if colon < 0:
            # standalone token -- the whole line is wrapped (``**END.**``)
            if body.endswith(marker):
                inner = body[:-len(marker)].strip()
                if inner:
                    return inner, (
                        "removed balanced %s around standalone %r"
                        % (marker, inner),)
            return s, ()
        label, rest = body[:colon], body[colon + 1:]
        if rest.startswith(marker):                     # <M>LABEL:<M> payload
            out = (label + ":" + rest[len(marker):]).strip()
            return out, ("removed balanced %s around label %r (wrapper spanned "
                         "the colon)" % (marker, label.strip()),)
        if label.endswith(marker):                      # <M>LABEL<M>: payload
            out = (label[:-len(marker)] + ":" + rest).strip()
            return out, ("removed balanced %s around label %r"
                         % (marker, label[:-len(marker)].strip()),)
        # SHAPE 4, LAST and TRANSPORT-GATED -- see the docstring.
        if body.endswith(marker):                       # <M>LABEL: payload<M>
            inner = body[:-len(marker)].strip()
            if inner and _family_balanced(inner, marker) and _is_transport(inner):
                return inner, (
                    "removed balanced %s around whole transport line %r"
                    % (marker, inner),)
        return s, ()
    return s, ()


#: A speaker label that re-states the character's ROLE after the name --
#: "Commander Vance (Space Force Tactician)". The writer does this when the
#: cast card is fresh in its context, and every such line used to raise
#: UNKNOWN_SPEAKER against a roster holding the bare name.
_RE_ROLE_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")


def _strip_role_parenthetical(name: str) -> str:
    """``"Vance (Tactician)"`` -> ``"Vance"``. Strips ONE trailing group.

    Only a TRAILING parenthetical, and only one, so a name that is entirely
    parenthesised or carries an interior group is left alone rather than being
    mangled into something that might collide with a different cast member. The
    caller keeps the exact-match attempt first; this is the fallback.
    """
    stripped = _RE_ROLE_PARENTHETICAL.sub("", str(name)).strip()
    return stripped or str(name).strip()


# --- speaker identity: ONE matcher, shared with the writer -------------------
#
# WHY THIS LIVES HERE AND WHY IT IS THE ONLY COPY (Bug Bible 12.132, verify
# condition 3). Until 2026-08-24 the "is this label a cast member" rule existed
# in TWO compositions -- this module's `on_speaker` and the writer's
# `_resolves_to_cast` -- plus TWO hand-written copies of the identity key.
# `_resolves_to_cast`'s own docstring claimed it was "imported rather than
# reimplemented", but only the HELPERS were imported; the LADDER was copied.
# That is not a style complaint: `_resolves_to_cast` decides whether the repair
# rung tells the model "restore this real character's label" or "fold or omit
# this row", so a parser that accepts a label the note believes is illegal
# instructs the model to DELETE A LINE THE PARSER WOULD HAVE TAKEN.
#
# THE LAYER MATTERS. `_canonicalize_transport_line` above normalizes LINE TEXT
# and is deliberately closed -- it feeds `normalize_scifi_news_pro_markup_text`,
# whose output is hashed into the receipt, and its dated refusal
# (PBUG-20260812-03) still stands there. Everything below normalizes only a
# LOOKUP CANDIDATE. It never edits the line, never reaches the hashed artifact,
# and on failure the defect still carries the RAW supplied label -- which is
# exactly what keeps `_standalone_stage_direction_repair_note` firing on
# `*SFX`. The apparent conflict between the refusal and Bible 12.132 is a
# layer confusion; resolved by putting the normalization here.


def speaker_identity_key(value: str) -> str:
    """Case/spacing-insensitive speaker identity; display text stays canonical.

    THE one identity rule. The writer imports this rather than restating it.
    """
    return " ".join(str(value).split()).casefold()


#: Titles that may precede a name. A CLOSED set on purpose -- an open
#: "first token is an honorific" heuristic would treat a one-word character
#: name as a title and alias the character to nothing.
_HONORIFICS = frozenset((
    "dr", "doctor", "mr", "mrs", "ms", "miss", "prof", "professor",
    "capt", "captain", "cmdr", "commander", "sgt", "sergeant",
    "lt", "lieutenant", "col", "colonel", "gen", "general",
    "adm", "admiral", "maj", "major", "sr", "sister", "fr", "father",
    "rev", "reverend", "chief", "officer", "agent", "sir", "dame",
    "lady", "lord", "madam", "madame", "detective", "inspector",
    "nurse", "coach", "judge", "mayor", "governor", "senator",
))

#: The emphasis markers a LABEL may be wearing, longest first so ``**`` is
#: tried before ``*``. Same closed family as ``_TRANSPORT_MARKERS`` -- a label
#: is not a licence to run a general Markdown sanitizer either.
_LABEL_MARKERS = ("**", "__", "*", "_")

_RE_TRAILING_COMMA_CLAUSE = re.compile(r",[^,]*$")


def _strip_label_decoration(label: str) -> str:
    """Remove emphasis markers wrapping a speaker LABEL. Candidate only.

    ``**DR. CHEN**`` -> ``DR. CHEN``; ``**ANNOUNCER`` -> ``ANNOUNCER``.

    UNMATCHED LEADING MARKERS ARE STRIPPED HERE, and that does NOT reopen
    PBUG-20260812-03. That refusal protects `_canonicalize_transport_line`,
    which rewrites the line and the hashed artifact, and its second stated
    reason was that stripping there would disarm the stage-direction repair
    note. Neither applies at this layer: nothing is rewritten, and an
    unresolvable candidate still reports the RAW label, so ``*SFX`` still
    reaches the note looking exactly like a stage direction.
    """
    token = str(label).strip()
    changed = True
    while changed and token:
        changed = False
        for marker in _LABEL_MARKERS:
            if token.startswith(marker) and len(token) > len(marker):
                token = token[len(marker):].strip()
                changed = True
                break
            if token.endswith(marker) and len(token) > len(marker):
                token = token[:-len(marker)].strip()
                changed = True
                break
    return token or str(label).strip()


def _strip_trailing_delivery_tag(label: str) -> str:
    """``"ELI, whispering"`` -> ``"ELI"``. Strips ONE trailing comma clause.

    A DELIVERY TAG IS NOT THE ONLY THING A COMMA MEANS, which is why this can
    never run before the exact match. ``DR. ORION NINE, SENIOR SIGNAL ANALYST``
    is a legal CANONICAL roster label carrying a comma (and a passing test in
    `tests/test_scifi_news_pro_markup.py`), so the exact rung claims it first
    and this function is never consulted for it. Here it only ever proposes a
    candidate, and a candidate that hits nothing changes nothing.
    """
    token = str(label).strip()
    if "," not in token:
        return token
    stripped = _RE_TRAILING_COMMA_CLAUSE.sub("", token).strip()
    return stripped or token


def _label_candidates(label: str) -> "tuple[str, ...]":
    """Ordered lookup candidates for ``label``; the RAW label is always first.

    Breadth-first over the three strippers so single-defect labels resolve
    before compound ones, and so a compound label like
    ``**DR. CHEN**, urgent`` -- decoration AND a delivery tag, which is the
    shape that actually killed the 2026-08-24 leg -- is reachable by composing
    them in either order without hard-coding a sequence.
    """
    ordered: "list[str]" = []
    seen: "set[str]" = set()
    frontier = [" ".join(str(label).split())]
    while frontier:
        current = frontier.pop(0)
        key = speaker_identity_key(current)
        if not current or key in seen:
            continue
        seen.add(key)
        ordered.append(current)
        for strip in (_strip_role_parenthetical,
                      _strip_trailing_delivery_tag,
                      _strip_label_decoration):
            nxt = strip(current)
            if nxt and speaker_identity_key(nxt) not in seen:
                frontier.append(nxt)
    return tuple(ordered)


def _proposed_aliases(name: str) -> "tuple[str, ...]":
    """Short forms a script writer would plausibly use for ``name``.

    CLOSED and structural -- never fuzzy. ``test_unknown_speaker_is_hard_no_remap``
    pins "no near-miss remap" as deliberate policy, and an edit-distance rule
    would merge a genuinely invented character as readily as it fixes a typo.
    Every form here is a SUBSTRING-BY-TOKEN of the canonical name.

    ``Dr. Haorong Chen`` -> ``Dr. Chen``, ``Chen``, ``Haorong``
    ``DR. ORION NINE, SENIOR SIGNAL ANALYST`` -> ``DR. ORION NINE``, ``DR. NINE``,
    ``NINE``, ``ORION``
    """
    full = " ".join(str(name).split())
    if not full:
        return ()
    out: "list[str]" = []
    head = full.split(",", 1)[0].strip()
    if head and head != full:
        out.append(head)          # the name proper, role clause dropped
    words = head.split()
    honorific = ""
    if len(words) > 1 and words[0].rstrip(".").casefold() in _HONORIFICS:
        honorific = words[0]
        words = words[1:]
    if len(words) >= 2:
        # A SURNAME IS NOT ALWAYS ONE WORD, and assuming it was cost a live
        # episode (2026-08-24). This took `words[-1]`, so
        # `Dr. Domitilla Del Vecchio` proposed `Vecchio` and never
        # `Del Vecchio` -- the model used the surname a person would use, the
        # roster could not express it, the label went unresolved, and salvage
        # minted a second character for someone already in the cast. Same hole
        # for `van Helsing`, `de Gaulle`, `von Braun`, `Mac Alister`.
        #
        # TRAILING n-grams, not ALL n-grams. A surname is a SUFFIX of the name,
        # so `Del Vecchio` and `Vecchio` are proposed while `Domitilla Del` --
        # a contiguous n-gram nobody would ever use as a label -- is not.
        # Over-proposing is not free: every junk alias is another chance to
        # collide with a real character and be suppressed by the ambiguity
        # guard, which would lose a GOOD alias to protect a fake one.
        #
        # Still CLOSED, still structural, still SUBSTRING-BY-TOKEN: every form
        # below is a contiguous run of the canonical name's own tokens. No
        # edit distance, no particle vocabulary to keep one culture short.
        for start in range(1, len(words)):
            suffix = " ".join(words[start:])
            if honorific:
                out.append(f"{honorific} {suffix}")
            out.append(suffix)
        out.append(words[0])          # the given name on its own
    elif len(words) == 1 and honorific:
        out.append(words[0])
    return tuple(dict.fromkeys(w for w in out if w))


class SpeakerRoster:
    """The ONE authority on whether a supplied label names a cast member.

    Built once per parse and shared with the writer's repair rung, so the
    parser and the note can never disagree about what is a legal speaker.

    THE LADDER, and exact identity ALWAYS wins first: a roster that genuinely
    carries parentheses, commas or asterisks matches itself before any relaxed
    rung runs.

    THE AMBIGUITY GUARD IS THE WHOLE SAFETY ARGUMENT. An alias enters the index
    only if EXACTLY ONE cast member proposes it and it collides with no exact
    key. When two members claim one alias, NEITHER gets it -- both degrade to
    exact-only and their short-form lines fail loudly, exactly as they do
    today. Suppression, never a coin flip: silently merging two characters
    would mis-cast a voice and corrupt the ledger, which is far worse than the
    refusal it replaces.
    """

    def __init__(self, cast_names, extra_aliases=None) -> None:
        self.cast_names = tuple(str(name).strip() for name in cast_names)
        self.exact: "dict[str, str]" = {
            speaker_identity_key(ANNOUNCER_NAME): ANNOUNCER_NAME,
        }
        self.blank_labels = 0
        self.ambiguous_labels: "list[tuple[str, str]]" = []
        #: Salvage-adopted strangers only, keyed by identity. Held SEPARATE
        #: from `exact` because reverse containment (`adopt`) may run against
        #: these and must never run against the locked cast.
        self._adopted: "dict[str, str]" = {}
        for name in self.cast_names:
            key = speaker_identity_key(name)
            if not key:
                self.blank_labels += 1
                continue
            prior = self.exact.get(key)
            if prior is not None:
                self.ambiguous_labels.append((prior, name))
                continue
            self.exact[key] = name

        claims: "dict[str, set[str]]" = {}
        for name in self.exact.values():
            if name == ANNOUNCER_NAME:
                continue
            for spoken_label in _proposed_aliases(name):
                claims.setdefault(
                    speaker_identity_key(spoken_label), set()).add(name)
        # AUTHORED ALIASES, from the model that invented the cast (operator
        # 2026-08-24: deterministic token rules are "too strict and may not
        # catch edge cases"). They are ADDITIVE and arrive as DATA -- the
        # parser stays pure, so one script always parses one way -- and they
        # are held to exactly the same ambiguity guard below. An alias naming
        # nobody on the roster is dropped rather than trusted.
        for owner, aliases in dict(extra_aliases or {}).items():
            canonical = self.exact.get(speaker_identity_key(owner))
            if canonical is None or canonical == ANNOUNCER_NAME:
                continue
            for spoken_label in aliases or ():
                key = speaker_identity_key(spoken_label)
                if key:
                    claims.setdefault(key, set()).add(canonical)
        self.aliases: "dict[str, str]" = {}
        self.suppressed_aliases: "tuple[str, ...]" = tuple(sorted(
            key for key, owners in claims.items()
            if len(owners) > 1 or key in self.exact))
        for key, owners in claims.items():
            if len(owners) == 1 and key not in self.exact:
                self.aliases[key] = next(iter(owners))

    def adopt(self, name: str) -> str:
        """Register a SALVAGE-ADOPTED speaker and return the name their line
        belongs to -- which may be a stranger adopted EARLIER.

        RETURNS a name because adoption is BIDIRECTIONAL. Registering forward
        only was order-dependent, and the kibitz r1 panel (Cursor and Opus,
        independently) caught it in the first cut of this method:
        `_proposed_aliases('ELOTWIZ')` is EMPTY -- one token, no honorific --
        so `DR. MICHAEL ELOTWIZ` arriving first adopts a person whose surname
        resolves, but `ELOTWIZ` arriving FIRST adopts a person with no aliases
        at all, and the longer label then matches nothing and mints a SECOND
        stranger. Measured on the real parser: long-then-short gave one
        speaker, short-then-long gave two. Same script, different order,
        different cast.

        So the reverse direction is checked too: if a proposed alias of the
        NEW label is already an ADOPTED name, the new label is another way of
        naming that same stranger.

        ADOPTED NAMES ONLY, never the locked cast, and that restriction is
        load-bearing. Locked names are authoritative and their aliases are
        already registered forward at construction; running containment
        backwards against them would let a script's `Dr. Michael Chen` swallow
        a locked `Chen` who is a different character. Strangers have no such
        authority -- they exist only because the parser could not place them --
        so collapsing two spellings of one stranger loses nothing.

        THE DEFECT THIS CLOSES (live, 2026-08-24). Adoption recorded the new
        character in `p.adopted` and never told the roster, so the roster the
        NEXT label is resolved against still did not contain them. A model
        that mislabelled one character twice therefore minted TWO strangers,
        not one. Two real characters arrived at casting as four speakers and a
        finished episode was discarded.

        So the typo half of that failure was never "unresolvable by policy"
        at all. The typo makes ONE stranger, which is unavoidable and honest;
        the missing write-back is what split that stranger in two, and that
        is a plain bug.

        SAME RUNGS, SAME GUARD, NO NEW POLICY. The adopted name goes in as an
        exact key and its `_proposed_aliases` forms go through the identical
        ambiguity check the locked cast uses -- an alias claimed by two
        characters is registered for NEITHER, and an alias that collides with
        any exact name loses. No fuzzy matching is introduced or implied.
        """
        canonical = " ".join(str(name).split())
        key = speaker_identity_key(canonical)
        if not key:
            return canonical
        settled = self.exact.get(key)
        if settled is not None:
            return settled

        # REVERSE CONTAINMENT, adopted strangers only -- see the docstring.
        # Exactly one claimant, or nothing: two adopted strangers both
        # answering to this label is precisely the ambiguity the forward guard
        # refuses, and it is refused identically here.
        claimants = {
            owner for owner in (
                self._adopted.get(speaker_identity_key(label))
                for label in _proposed_aliases(canonical)
            ) if owner
        }
        if len(claimants) == 1:
            existing = next(iter(claimants))
            self.exact[key] = existing
            return existing

        self.exact[key] = canonical
        self._adopted[key] = canonical
        # An alias this new name would claim, but which is already an exact
        # name or already claimed by someone else, must NOT be stolen.
        for label in _proposed_aliases(canonical):
            alias_key = speaker_identity_key(label)
            if not alias_key or alias_key in self.exact:
                continue
            prior = self.aliases.get(alias_key)
            if prior is None:
                self.aliases[alias_key] = canonical
            elif prior != canonical:
                # Ambiguous now that a second claimant exists: neither gets
                # it, exactly as the constructor's guard decides.
                del self.aliases[alias_key]
        return canonical

    def resolve(self, supplied: str) -> "tuple[str | None, str]":
        """``(canonical_name_or_None, how)``. Never raises, never rewrites."""
        candidates = _label_candidates(supplied)
        for index, candidate in enumerate(candidates):
            hit = self.exact.get(speaker_identity_key(candidate))
            if hit is not None:
                return hit, ("exact" if index == 0
                             else f"normalized {candidate!r}")
        for candidate in candidates:
            hit = self.aliases.get(speaker_identity_key(candidate))
            if hit is not None:
                return hit, f"alias {candidate!r}"
        return None, "unresolved"

    def names_a_cast_member(self, supplied: str) -> bool:
        """Whether ``supplied`` resolves at all -- the writer's question."""
        return self.resolve(supplied)[0] is not None


def build_speaker_roster(cast_names, extra_aliases=None) -> SpeakerRoster:
    """Public constructor; the writer calls this rather than rebuilding it."""
    return SpeakerRoster(cast_names, extra_aliases)


#: LABELS THAT NAME A SOUND, NOT A PERSON. A CLOSED vocabulary.
#:
#: THE RULE IS THE WORD, NOT THE PUNCTUATION (operator, 2026-08-24: *"they
#: should not chunk off dialogue, we should just never do any SFX"*). The first
#: draft of this dropped any DECORATED unresolvable label, which is wrong and
#: was caught immediately: ``(SOMEONE NEW): I have something to say.`` is a
#: character the model invented, wearing brackets, WITH DIALOGUE IN IT.
#: Dropping it would delete a real spoken line -- the one thing salvage exists
#: to prevent. Decoration is not evidence that something is not a person.
#:
#: A cue word IS that evidence. ``SFX: a door slams`` carries no dialogue: it
#: describes a sound, and this pipeline HAS NO SOUND EFFECTS. The `[SFX: ...]`
#: ledger token was removed 2026-07-01 and the whole SFX bed subsystem was
#: ripped 2026-08-06, so a cue row can only ever become a character reading a
#: stage direction aloud in their own voice.
#:
#: MUSIC is absent on purpose -- it is a real structural delimiter with its own
#: classifier, and listing it here would be dead code.
_SOUND_CUE_LABELS = frozenset((
    "sfx", "s f x", "sound", "sounds", "sound effect", "sound effects",
    "soundeffect", "sound fx", "fx", "foley", "effect", "effects",
    "env", "environment", "ambience", "ambient", "ambiance", "atmos",
    "atmosphere", "noise", "stinger", "cue", "sound cue", "audio",
    "sfx cue", "background", "bg",
))


def _undecorated_speaker_name(label: str) -> str:
    """A display name fit for a cast row, a caption and a credit line.

    ``(SOMEONE NEW)`` -> ``SOMEONE NEW``; ``**THORNE**`` -> ``THORNE``.
    Falls back to the original if stripping would leave nothing, because a
    nameless character is worse than a decorated one.
    """
    bare = _strip_label_decoration(str(label).strip()).strip("()[]<>*_ \t")
    bare = " ".join(bare.split())
    return bare or " ".join(str(label).split())


def _is_sound_cue_label(label: str) -> bool:
    """``SFX``/``*SFX*``/``[ENV]`` -> True. ``(SOMEONE NEW)`` -> False.

    Only ever consulted AFTER the resolver has failed, so a real cast member
    -- decorated or not -- has already resolved and can never reach this.
    Decoration is stripped first so the cue is recognised however it is
    dressed, but the VERDICT comes from the vocabulary alone.
    """
    # Brackets and parentheses are stripped here as well as emphasis markers,
    # because a cue is written every one of these ways in the wild:
    # ``SFX:``, ``*SFX:``, ``[SFX]:``, ``(sound)``. This is a local strip for
    # a VOCABULARY TEST only -- it proposes nothing to the roster and cannot
    # admit anybody.
    bare = _strip_label_decoration(str(label).strip()).strip("()[]<>*_ \t")
    # A cue label is the word itself, not a sentence that mentions it.
    return speaker_identity_key(bare) in _SOUND_CUE_LABELS


def _normalize_line(line: str) -> "tuple[str, tuple[str, ...]]":
    """Normalize transport whitespace and balanced label emphasis.

    Authored content is untouched: see :func:`_canonicalize_transport_line` for
    the exact, closed grammar and why a spoken word can never be rewritten."""
    return _canonicalize_transport_line(line)


def normalize_scifi_news_pro_markup_text(text: str) -> str:
    """Return the parser's normalized proof artifact.

    SHARES ``_canonicalize_transport_line`` WITH THE PARSER ON PURPOSE (kibitz
    r1 MUST-FIX 3). This text is what ``normalized_source``, its sha256 and the
    proof map are built from, while ``_normalize_line`` decides classification.
    Two independent normalizers would let the accepted script diverge from the
    artifact hashed beside it -- a receipt describing a document nobody parsed."""
    return "\n".join(
        _canonicalize_transport_line(raw)[0] for raw in str(text).splitlines())


class NewsProParseDefect(enum.Enum):
    """Defect classes for the markup ladder (doc section 6, exact set)."""

    MISSING_TITLE = "MISSING_TITLE"
    DUPLICATE_TITLE = "DUPLICATE_TITLE"
    # Missing END is structural. Every retry uses the same provider-capacity
    # contract; the parser never infers a word/token shortfall.
    MISSING_END = "MISSING_END"
    CONTENT_AFTER_END = "CONTENT_AFTER_END"
    BAD_LINE_SHAPE = "BAD_LINE_SHAPE"
    UNKNOWN_SPEAKER = "UNKNOWN_SPEAKER"
    SCENE_ORDER = "SCENE_ORDER"
    EMPTY_SCENE = "EMPTY_SCENE"
    SKELETON_BREAK = "SKELETON_BREAK"
    CAST_MEMBER_SILENT = "CAST_MEMBER_SILENT"
    MULTIPLE_CODA = "MULTIPLE_CODA"


@dataclass(frozen=True)
class ParseDefect:
    """One collected defect: class + human detail + 1-based source line."""

    code: NewsProParseDefect
    detail: str = ""
    line_no: "int | None" = None

    def __str__(self) -> str:
        where = f" (line {self.line_no})" if self.line_no is not None else ""
        detail = f": {self.detail}" if self.detail else ""
        return f"{self.code.value}{detail}{where}"


@dataclass(frozen=True)
class ParsedLine:
    """One constituent spoken line with canonical roster identity."""

    speaker: str
    text: str


@dataclass(frozen=True)
class ParsedScene:
    n: int
    setting: str
    lines: "tuple[ParsedLine, ...]"


@dataclass(frozen=True)
class ParsedScript:
    title: str
    music_open: str
    music_inter: "tuple[tuple[int, str], ...]"  # (scene_after, cue_text)
    music_close: str
    announcer_intro: "tuple[str, ...]"
    scenes: "tuple[ParsedScene, ...]"
    announcer_outro: "tuple[str, ...]"
    coda: str
    character_word_count: int
    announcer_word_count: int
    # Retained for receipt compatibility; prose normalization is retired.
    #
    # SPEAKER RESOLUTIONS DO NOT BELONG HERE, and putting them here cost a live
    # leg on 2026-08-24. `_parsed_payload` SEALS this field, and the seal is
    # re-verified by re-parsing the raw source. The build parses against the
    # treatment's full `cast_names`; the seal check parses against
    # `_speakers_in_order(parsed)` -- a SMALLER roster. Same script, same
    # speakers, but a label can reach its character by a different RUNG, so the
    # receipt STRING differs and the seal reports "parsed artifact seal is
    # stale" for a draft nothing touched. A seal must depend on the script, not
    # on how the parser got there.
    normalizations: "tuple[str, ...]" = ()
    #: How each non-exact speaker label reached its character. A RECEIPT, and
    #: deliberately OUTSIDE the sealed payload for the reason above.
    speaker_resolutions: "tuple[str, ...]" = ()
    #: Speakers admitted by salvage that the locked roster did not contain.
    #: The producer owns giving each one a voice and a cast row.
    adopted_speakers: "tuple[str, ...]" = ()
    #: Rows salvage discarded, and locked cast who never spoke. Empty on every
    #: honest parse -- a non-empty value means the episode was salvaged.
    dropped_rows: "tuple[str, ...]" = ()


def render_defects(defects: "tuple[ParseDefect, ...]") -> str:
    """Full defect list as structural-retry quotable text (one per line)."""
    return "\n".join(f"- {d}" for d in defects)


def _wc(text: str) -> int:
    return canonical_word_count(text)


# --- state machine states ----------------------------------------------------
_EXPECT_TITLE, _PREAMBLE, _SCENES, _POSTAMBLE, _DONE = range(5)


class _Parse:
    """Mutable walk state (module-private; the public surface is pure)."""

    def __init__(self, cast_names, extra_aliases=None, salvage=False) -> None:
        self.state = _EXPECT_TITLE
        self.defects: "list[ParseDefect]" = []
        self.roster = SpeakerRoster(cast_names, extra_aliases)
        self.cast_names = self.roster.cast_names
        self.resolutions: "list[str]" = []
        self.salvage = bool(salvage)
        self.adopted: "list[str]" = []
        self.dropped: "list[str]" = []
        for _ in range(self.roster.blank_labels):
            self.skeleton("cast roster contains a blank speaker label")
        for prior, name in self.roster.ambiguous_labels:
            self.skeleton(
                f"cast roster labels {prior!r} and {name!r} are "
                "ambiguous under case-insensitive identity"
            )
        self.title: "str | None" = None
        self.title_first = True
        self.music_open: "str | None" = None
        self.music_inter: "list[tuple[int, str]]" = []
        self.music_close: "str | None" = None
        self.intro: "list[str]" = []
        self.scenes: "list[tuple[int, str, list[ParsedLine]]]" = []
        self.outro: "list[str]" = []
        self.coda: "str | None" = None
        self.saw_end = False

    #: RETIRED 2026-08-24. The identity rule now lives once, module level, as
    #: `speaker_identity_key`, which the writer imports instead of keeping its
    #: own hand-written copy. Kept as a thin alias only so a caller reaching
    #: for the old private name gets the same answer rather than an
    #: AttributeError; there is no second implementation behind it.
    _speaker_key = staticmethod(speaker_identity_key)

    def defect(self, code: NewsProParseDefect, detail: str = "",
               line_no: "int | None" = None) -> None:
        self.defects.append(ParseDefect(code, detail, line_no))

    def skeleton(self, detail: str, line_no: "int | None" = None) -> None:
        self.defect(NewsProParseDefect.SKELETON_BREAK, detail, line_no)

    # -- per-shape handlers ---------------------------------------------------

    def on_title(self, text: str, no: int) -> None:
        if self.title is not None:
            self.defect(NewsProParseDefect.DUPLICATE_TITLE, text, no)
            return
        self.title = text
        if self.state != _EXPECT_TITLE:
            self.title_first = False
        self.state = max(self.state, _PREAMBLE)

    def on_music(self, text: str, no: int) -> None:
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self.state = max(self.state, _PREAMBLE)
            if self.music_open is None:
                if self.intro:
                    self.skeleton(
                        "opening MUSIC must come before the announcer intro",
                        no)
                self.music_open = text
            else:
                self.skeleton("extra MUSIC line in the preamble", no)
        elif self.state == _SCENES:
            self.music_inter.append((self.scenes[-1][0], text))
        elif self.state == _POSTAMBLE:
            if self.coda is None:
                self.skeleton("closing MUSIC must come after the CODA", no)
            if self.music_close is None:
                self.music_close = text
            else:
                self.skeleton("extra MUSIC line in the postamble", no)

    def _close_scene(self, no: "int | None") -> None:
        if self.scenes and not self.scenes[-1][2]:
            self.defect(NewsProParseDefect.EMPTY_SCENE,
                        f"SCENE {self.scenes[-1][0]} has no spoken lines", no)

    def _check_preamble_complete(self, no: int) -> None:
        if self.music_open is None:
            self.skeleton("opening MUSIC line missing", no)
        if not self.intro:
            self.skeleton("announcer intro missing", no)

    def on_scene(self, n: int, setting: str, no: int) -> None:
        if self.state == _POSTAMBLE:
            self.skeleton("SCENE header after the announcer outro began", no)
            return
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self._check_preamble_complete(no)
            self.state = _SCENES
        else:
            self._close_scene(no)
        expected = (self.scenes[-1][0] + 1) if self.scenes else 1
        if n != expected:
            self.defect(NewsProParseDefect.SCENE_ORDER,
                        f"SCENE {n} where SCENE {expected} was expected", no)
        self.scenes.append((n, setting, []))

    def on_coda(self, text: str, no: int) -> None:
        if self.coda is not None:
            self.defect(NewsProParseDefect.MULTIPLE_CODA, text, no)
            return
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            self.skeleton("CODA before any scene", no)
            self.state = _PREAMBLE
        elif self.state == _SCENES:
            self._close_scene(no)
            # In salvage the frame was deliberately held open so post-outro
            # drama could still land, so outro text already collected is NOT
            # a missing outro. Only a genuinely absent one is reported.
            if not (self.salvage and self.outro):
                self.skeleton("announcer outro missing before the CODA", no)
            self.state = _POSTAMBLE
        self.coda = text

    def on_end(self, no: int) -> None:
        self.saw_end = True
        if self.state == _SCENES:
            self._close_scene(no)
        if not self.scenes:
            self.skeleton("no scenes before END.", no)
        if not self.outro:
            self.skeleton("announcer outro missing", no)
        if self.coda is None:
            self.skeleton("CODA missing", no)
        if self.music_close is None:
            self.skeleton("closing MUSIC line missing", no)
        self.state = _DONE

    def on_speaker(self, name: str, text: str, no: int) -> None:
        supplied_name = " ".join(name.split())
        # ONE MATCHER (`SpeakerRoster.resolve`), shared with the writer's
        # repair rung. Exact identity wins first; the relaxed rungs -- role
        # parenthetical, emphasis decoration, trailing delivery tag, and the
        # unambiguous alias index -- only ever propose a lookup candidate.
        #
        # THE DEFECT KEEPS THE RAW LABEL. Bible 12.132 asks for exactly this:
        # normalize before comparing, but report what the model actually wrote,
        # so `_standalone_stage_direction_repair_note` still sees `*SFX` as a
        # stage direction and a reader can tell a formatting artifact from a
        # genuinely absent roster entry.
        canonical_name, how = self.roster.resolve(supplied_name)
        if canonical_name is None:
            if _is_sound_cue_label(supplied_name):
                # A SOUND CUE IS NEVER A CHARACTER, in salvage or out of it.
                # THIS PIPELINE HAS NO SOUND EFFECTS -- the ledger token went
                # 2026-07-01 and the SFX subsystem was ripped 2026-08-06 -- so
                # a cue row can only become a character reading a stage
                # direction aloud in their own voice, or an SFX row in a
                # ledger that has nowhere to put one.
                #
                # DROPPED, NOT ADOPTED, and it costs no dialogue: a cue row
                # describes a sound, it does not speak. Rows that DO carry
                # dialogue -- including a character the model invented -- are
                # adopted below and keep every word.
                #
                # Deliberately NOT gated on salvage. An SFX row must never
                # reach the ledger by any path, and on the honest path this
                # still reports UNKNOWN_SPEAKER via the fall-through below,
                # so the repair rung is still told to fix it.
                if self.salvage:
                    self.dropped.append(
                        f"line {no}: sound cue, not a character: "
                        f"{supplied_name}")
                    return
            if self.salvage:
                # ADOPTION (operator, 2026-08-24): "sometimes a wrong name
                # populated but shouldn't kill the whole episode." A speaker
                # the roster cannot place is a character the model wrote, so
                # in salvage the episode KEEPS the line and the character is
                # admitted for real -- the producer deals them a voice and
                # they reach the ledger. A slightly wrong name in a delivered
                # episode beats a perfect refusal, which is THE LAW: an audit
                # may improve a story, it may never fail one.
                # ADOPT THE UNDECORATED FORM. The label arrives wearing
                # whatever the model put on it, and that name becomes a real
                # CAST ROW -- it reaches the voice deal, the ledger, the
                # captions and the credits. Adopting `(SOMEONE NEW)` verbatim
                # would print the brackets in the credit roll and read as a
                # broken episode rather than a salvaged one. The raw label is
                # still recorded in the receipt line below, so the adoption
                # stays traceable to exactly what the model wrote.
                # TELL THE ROSTER, and take back the name it settled on.
                # Without this the adopted character is unfindable by their own
                # surname and the next mention mints ANOTHER stranger; the
                # RETURN value is what makes it order-independent, attaching a
                # longer label to a stranger already adopted under a shorter
                # one -- see `SpeakerRoster.adopt`.
                canonical_name = self.roster.adopt(
                    _undecorated_speaker_name(supplied_name))
                if canonical_name not in self.adopted:
                    self.adopted.append(canonical_name)
                self.resolutions.append(
                    f"line {no}: speaker {supplied_name!r} ADOPTED as "
                    f"{canonical_name!r} -- not on the locked roster; "
                    "admitted as a speaking character")
            else:
                self.defect(
                    NewsProParseDefect.UNKNOWN_SPEAKER, supplied_name, no
                )
                canonical_name = supplied_name
        elif how != "exact":
            # Every non-exact resolution is receipted, never hidden.
            self.resolutions.append(
                f"line {no}: speaker {supplied_name!r} resolved to "
                f"{canonical_name!r} by {how}")
        if canonical_name == ANNOUNCER_NAME:
            if self.state in (_EXPECT_TITLE, _PREAMBLE):
                self.state = max(self.state, _PREAMBLE)
                self.intro.append(text)
            elif self.state == _SCENES:
                if self.salvage:
                    # THE FRAME STAYS OPEN IN SALVAGE. A mid-scene ANNOUNCER
                    # row is what actually killed the 2026-08-24 leg: it
                    # closed the story frame, and every later character line
                    # became "after the last scene". Here the outro text is
                    # KEPT (the ledger needs a non-empty outro) but the scenes
                    # stay open, so the drama that follows still lands.
                    self.outro.append(text)
                    return
                self._close_scene(no)
                self.state = _POSTAMBLE
                self.outro.append(text)
            elif self.state == _POSTAMBLE:
                if self.coda is not None:
                    self.skeleton("ANNOUNCER line after the CODA", no)
                self.outro.append(text)
            return
        # character line
        if self.state in (_EXPECT_TITLE, _PREAMBLE):
            if self.salvage:
                # IMPLICIT SCENE (operator, 2026-08-24 -- "you never know
                # what crazy stuff a model will throw at the parser").
                # Salvage must not depend on recognizing which SPECIFIC way
                # a scene header failed to parse -- a bare "SCENE 1:", a
                # bracketed "SCENE [1]:", or any shape nobody has seen yet
                # all land here the same way: real dialogue, no scene ever
                # opened for it to belong to. Without this, salvage refused
                # every one of them outright ("no scene contains a spoken
                # line"), because `self.scenes` stayed empty regardless of
                # how much real dialogue followed. Open an implicit scene 1
                # from the first placeable line, the same way an unresolved
                # speaker is ADOPTED rather than dropped -- this is a
                # RECEIPT, not a defect, so it does not block salvage the
                # way `self.skeleton(...)` would.
                self._check_preamble_complete(no)
                self.resolutions.append(
                    f"line {no}: character line ({canonical_name}) arrived "
                    "before any SCENE header parsed -- salvage opened an "
                    "implicit SCENE 1"
                )
                self.scenes.append((1, "", []))
                self.state = _SCENES
                self.scenes[-1][2].append(
                    ParsedLine(speaker=canonical_name, text=text)
                )
                return
            self.skeleton(
                f"character line ({canonical_name}) before SCENE 1", no
            )
            self.state = _PREAMBLE
        elif self.state == _SCENES:
            self.scenes[-1][2].append(
                ParsedLine(speaker=canonical_name, text=text)
            )
        elif self.state == _POSTAMBLE:
            self.skeleton(
                f"character line ({canonical_name}) after the last scene", no
            )


def parse_scifi_news_pro_markup(text: str, cast_names, extra_aliases=None,
                                salvage: bool = False) -> (
        "tuple[ParsedScript | None, tuple[ParseDefect, ...]]"):
    """Parse whole-play scifi_news_pro markup against the legal cast.

    ``cast_names`` is the treatment's canonical display-name roster;
    ANNOUNCER is implicitly legal. Matching ignores case and repeated spacing,
    while the returned artifact uses the canonical roster spelling.
    Returns ``(ParsedScript, ())`` on a clean parse or ``(None, defects)``
    with EVERY defect collected. Pure: no I/O, no mutation of arguments,
    never rewrites a spoken word.

    ``extra_aliases`` maps a canonical roster name to additional labels the
    cast's own author says it will use for that character. Additive, and held
    to the same ambiguity guard as the derived ones.

    ``salvage`` IS NOT A LOOSER PARSER -- IT IS THE LAST RUNG OF THE PRODUCER.
    It is False for every honest attempt, and the writer turns it on only when
    the repair ladder has already spent its attempts and the alternative is
    delivering NO EPISODE. Operator, 2026-08-24: *"accepts sometimes a wrong
    name populated but shouldn't kill the whole episode."* In salvage:

    * a speaker the roster cannot place is ADOPTED as a real character
      (``script.adopted_speakers``) rather than refused;
    * an unlabelled row is DROPPED and recorded (``script.dropped_rows``)
      rather than refused -- this is the one place the module may discard a
      row, and it discards only rows that name no speaker and therefore
      cannot be performed by anyone;
    * a mid-scene ANNOUNCER row keeps its text but does not close the frame.

    Everything else still refuses. Salvage cannot invent a TITLE, a CODA, an
    END or a scene, so a draft with no story in it still fails loudly.
    """
    p = _Parse(cast_names, extra_aliases, salvage)
    normalizations: "list[str]" = []
    for no, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if p.state == _DONE:
            p.defect(NewsProParseDefect.CONTENT_AFTER_END, line[:80], no)
            continue
        line, notes = _normalize_line(line)
        normalizations.extend(f"line {no}: {n}" for n in notes)
        if not line:
            continue  # the whole line was decoration
        m = _RE_TITLE.match(line)
        if m:
            p.on_title(m.group(1).strip(), no)
            continue
        m = _RE_MUSIC.match(line)
        if m:
            p.on_music(m.group(1).strip(), no)
            continue
        m = _RE_SCENE.match(line)
        if m:
            p.on_scene(int(m.group(1)), m.group(2).strip(), no)
            continue
        m = _RE_CODA.match(line)
        if m:
            p.on_coda(m.group(1).strip(), no)
            continue
        if _RE_END.match(line):
            p.on_end(no)
            continue
        m = _RE_SPEAKER.match(line)
        if m:
            p.on_speaker(m.group(1), m.group(2).strip(), no)
            continue
        if p.salvage:
            # An unlabelled row names no speaker, so no voice can perform it
            # and no ledger row can own it. In salvage it is dropped and
            # RECORDED rather than allowed to refuse the whole episode.
            p.dropped.append(f"line {no}: {line[:80]}")
            continue
        p.defect(NewsProParseDefect.BAD_LINE_SHAPE, line[:80], no)
        if p.state == _EXPECT_TITLE:
            p.state = _PREAMBLE

    # ---- end-of-text checks -------------------------------------------------
    if p.title is None:
        p.defect(NewsProParseDefect.MISSING_TITLE)
    elif not p.title_first:
        p.skeleton("TITLE is not the first line")
    if not p.saw_end:
        # The missing delimiter is the actionable structural defect. Suppress
        # derivative postamble messages until an END line actually arrives.
        if p.state == _SCENES:
            p._close_scene(None)
        p.defect(NewsProParseDefect.MISSING_END)
    spoken = {ln.speaker for _n, _s, lines in p.scenes for ln in lines}
    for name in p.cast_names:
        if name not in spoken:
            if p.salvage:
                # A locked cast member who never spoke cannot be conjured
                # into speech without authoring dialogue, which is forbidden.
                # In salvage they simply do not appear in this episode; the
                # producer drops their cast row so the ledger's speaker set
                # still matches exactly, with no hole and no silent voice.
                p.dropped.append(f"cast member never spoke: {name}")
                continue
            p.defect(NewsProParseDefect.CAST_MEMBER_SILENT, name)

    if p.salvage and not any(lines for _n, _s, lines in p.scenes):
        # Salvage still refuses a draft with no drama in it. Delivering an
        # episode of nothing is not the operator's "wrong name populated".
        # Checked on SPOKEN LINES, not on whether a SCENE header exists: a
        # draft can open "SCENE 1:" and then contain nothing but unlabelled
        # narration, which salvage drops -- leaving a scene with no drama in
        # it and nobody to perform.
        p.defects[:] = [d for d in p.defects
                        if d.code is not NewsProParseDefect.EMPTY_SCENE]
        p.skeleton("salvage cannot proceed: no scene contains a spoken line")

    if p.defects:
        return None, tuple(p.defects)

    character_words = sum(
        _wc(ln.text) for _n, _s, lines in p.scenes for ln in lines)
    announcer_words = (sum(_wc(t) for t in p.intro)
                       + sum(_wc(t) for t in p.outro)
                       + _wc(p.coda or ""))
    script = ParsedScript(
        title=p.title or "",
        music_open=p.music_open or "",
        music_inter=tuple(p.music_inter),
        music_close=p.music_close or "",
        announcer_intro=tuple(p.intro),
        scenes=tuple(ParsedScene(n=n, setting=s, lines=tuple(lines))
                     for n, s, lines in p.scenes),
        announcer_outro=tuple(p.outro),
        coda=p.coda or "",
        character_word_count=character_words,
        announcer_word_count=announcer_words,
        normalizations=tuple(normalizations),
        speaker_resolutions=tuple(p.resolutions),
        adopted_speakers=tuple(p.adopted),
        dropped_rows=tuple(p.dropped),
    )
    return script, ()
