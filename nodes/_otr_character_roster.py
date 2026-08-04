"""Parse a play's cast list into character records, with gender.

Why this exists
---------------
Casting currently gives Shakespeare characters a gender by DICE ROLL. Measured:
`gender_of_first_name` returns "unknown" for JULIET, ROSALIND, CELIA, TITANIA,
BEATRICE and every other name in the vendored corpus -- they are not in the
1930s first-name lexicon -- so the statistical allocator's rolled gender stands
and a female lead draws a male voice on roughly half of seeds.

The answer was sitting in the source the whole time. Every play ships a cast list
("Characters in the Play" in Folger, "THE PERSONS IN THE PLAY" in Gutenberg) whose
descriptions state relation and rank outright:

    ORLANDO, youngest son of Sir Rowland de Boys
    ROSALIND, daughter to Duke Senior
    PHOEBE, a disdainful shepherdess

That block is also what makes the roster AUTHORITATIVE for parsing: a speech
prefix outside the roster is not a character, which is how structural headings
("FIRST ACT", "TABLEAU") stop being mistaken for speakers without a frequency
rule -- a rule that would have deleted BEATRICE's single speech from the scene
named after her.

This runs at VENDOR time, and its output is written into the provenance sidecar,
so the render path reads a confirmed record and never infers.

Honest unknowns
---------------
A character whose description carries no gendered word is recorded as "unknown"
rather than guessed. An auditable gap is worth more than a confident error: the
caller can prefer passages that avoid unknown speakers, or assign from whichever
voice pool has room and say so in the ledger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CharacterRecord",
    "parse_character_roster",
    "infer_gender",
]

# The Folger cast block opens with this heading and its underline; Gutenberg uses
# an all-caps variant. Either way the block ends at the first act/scene heading.
_ROSTER_HEADINGS = (
    "characters in the play",
    "the persons in the play",
    "persons in the play",
    "dramatis personae",
    "characters",
)
_END_OF_ROSTER = re.compile(r"^\s*(ACT\b|SCENE\b|PROLOGUE\b|FIRST ACT\b)", re.IGNORECASE)

# "NAME, description" -- but also "NURSE to Juliet", which carries no comma and
# was being dropped entirely, and a bare "ROMEO" with no description at all.
_ENTRY_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z' .\-]*[A-Z]|[A-Z])"
    r"\s*(?:,\s*(?P<desc>.+)|\s+(?P<desc2>[a-z].*))?$"
)

# A cast list genders a description-less entry through the NEXT one: Folger lists
# "ROMEO" bare, then "MONTAGUE, his father" -- and that "his" is Romeo's. Same for
# "JULIET" followed by "CAPULET, her father". Without reading this back-reference,
# the two most famous names in the corpus both come back unknown.
_BACKREF_RE = re.compile(r"^(?P<pronoun>his|her|their)\s+\S", re.IGNORECASE)

# A line that merely introduces a group ("Attending Duke Frederick:") -- its
# members follow indented, and they inherit nothing but their own text.
_GROUP_HEADER_RE = re.compile(r"^\s*\S.*:\s*$")

# Relation words are the strongest signal and they gender THIS character, even
# when the description also names someone else ("daughter to Duke Senior" is a
# woman, despite the Duke). Ordered scan, first definite hit wins.
_MALE_RELATION = (
    "son", "brother", "father", "husband", "nephew", "uncle", "grandfather",
    "widower", "bridegroom", "godson", "stepson", "boy", "lad", "youth",
)
_FEMALE_RELATION = (
    "daughter", "sister", "mother", "wife", "niece", "aunt", "grandmother",
    "widow", "bride", "goddaughter", "stepdaughter", "girl", "maiden",
)
# Rank and role. Period-reliable in this corpus; weaker than relation.
_MALE_TITLE = (
    "duke", "king", "prince", "lord", "sir", "earl", "count", "baron", "knight",
    "friar", "priest", "abbot", "monk", "shepherd", "servingman", "manservant",
    "footman", "butler", "gentleman", "god", "emperor", "master", "mr",
    # Offices that were male by period convention in this corpus.
    "governor", "constable", "watchman", "sexton", "signior", "signor",
    "senator", "tribune", "soldier", "sailor", "boatswain", "clown", "fool",
    "steward", "porter", "herald", "apothecary", "schoolmaster", "tutor",
    "thane", "commander", "murderer", "nobleman", "wrestler", "courtier",
    # "Scottish Nobles" in this corpus are men; period convention, not a claim
    # about the word in general.
    "noble",
)
_FEMALE_TITLE = (
    "duchess", "queen", "princess", "lady", "dame", "countess", "baroness",
    "abbess", "nun", "shepherdess", "waiting-woman", "gentlewoman", "goddess",
    "empress", "mistress", "nurse", "maid", "governess", "mrs", "miss", "madam",
    "witch", "sorceress", "seamstress", "milkmaid", "matron",
)

# "Three Witches, the Weird Sisters" is ONE roster entry, but the play speaks as
# FIRST WITCH, SECOND WITCH and THIRD WITCH. The same shape covers "Three
# Murderers". Expanding it is what gets the prophecy on the heath -- the marquee
# scene of the corpus -- its cast.
_COUNTED_ENTRY_RE = re.compile(
    r"^(?P<count>two|three|four|five|six)\s+(?P<role>[A-Za-z][A-Za-z'\-]*)\b",
    re.IGNORECASE,
)
_ORDINALS = ("FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH")
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

_WORD_RE = re.compile(r"[a-z']+")


# The cast list and the speech prefixes DISAGREE on names, and the roster is the
# formal one: "SIGNIOR BENEDICK, a gentleman from Padua" speaks as BENEDICK, and
# "COUNT CLAUDIO" as CLAUDIO. Matching only on the full name left the leads of
# Much Ado, Lear and Twelfth Night all absent_from_roster. Honorifics are stripped
# to build the alias -- but gender is still read from the FULL name, because
# "SIGNIOR" and "COUNT" are exactly what genders them.
_HONORIFICS = frozenset({
    "signior", "signor", "count", "don", "donna", "lord", "lady", "sir", "dame",
    "duke", "duchess", "king", "queen", "prince", "princess", "friar", "master",
    "mistress", "doctor", "captain", "sergeant", "the", "young", "old", "first",
    "second", "third",
})


@dataclass(frozen=True)
class CharacterRecord:
    """One character as the play's own cast list describes them."""

    name: str
    description: str
    gender: str            # "male" | "female" | "unknown"
    gender_source: str     # "relation" | "title" | "group" | "back_reference" | "unknown"
    aliases: tuple[str, ...] = ()

    @property
    def is_known_gender(self) -> bool:
        return self.gender in ("male", "female")

    def matches(self, speech_prefix: str) -> bool:
        """Does this record belong to a speech prefix from the play text?"""
        candidate = str(speech_prefix or "").strip().upper()
        return bool(candidate) and (
            candidate == self.name.upper()
            or candidate in {form.upper() for form in self.aliases}
        )


def _counted_aliases(entry_text: str) -> tuple[str, ...]:
    """Ordinal speech prefixes implied by a counted collective entry.

    "Three Witches, the Weird Sisters" -> FIRST WITCH, SECOND WITCH, THIRD WITCH.
    """
    match = _COUNTED_ENTRY_RE.match(str(entry_text or "").strip())
    if match is None:
        return ()
    count = _COUNT_WORDS.get(match.group("count").lower(), 0)
    plural = match.group("role")
    # The entry names the role in the PLURAL ("Witches", "Murderers") while the
    # speech prefix uses the singular ("FIRST WITCH"). Take the shortest
    # singular form, since "Witches" must reduce to "Witch", not "Witche".
    singular = min(_singular_forms(plural.lower()), key=len)
    role = singular.upper()
    return tuple(f"{_ORDINALS[i]} {role}" for i in range(min(count, len(_ORDINALS))))


def _aliases_for(name: str) -> tuple[str, ...]:
    """Short forms a speech prefix might plausibly use for this roster name."""
    words = str(name or "").split()
    out: list[str] = []
    stripped = [w for w in words if w.lower().strip(".,'") not in _HONORIFICS]
    if stripped and stripped != words:
        out.append(" ".join(stripped))
    if len(stripped) > 1:
        out.append(stripped[-1])
    elif len(words) > 1 and not stripped:
        # An all-honorific name like "FIRST LORD" -- keep it whole.
        out.append(" ".join(words))
    unique_forms: list[str] = []
    for form in out:
        if form and form.upper() != str(name).upper() and form not in unique_forms:
            unique_forms.append(form)
    return tuple(unique_forms)


def _singular_forms(word: str) -> tuple[str, ...]:
    """The word plus its likely singular, so a GROUP header still genders.

    Group headers are written in the plural -- "Waiting gentlewomen to Hero:",
    "Lords attending Duke Senior" -- and the members beneath them inherit that
    description. Matching only singular forms left MARGARET and URSULA
    genderless under a heading that calls them gentlewomen.
    """
    forms = [word]
    if word.endswith("women"):
        forms.append(word[:-5] + "woman")
    elif word.endswith("men"):
        forms.append(word[:-3] + "man")
    elif word.endswith("esses"):
        forms.append(word[:-2])
    elif word.endswith("es") and len(word) > 4:
        forms.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        forms.append(word[:-1])
    return tuple(forms)


def _scan(words: list[str], male: tuple[str, ...], female: tuple[str, ...]) -> str:
    """First definite gendered word wins; ambiguity yields ''."""
    for word in words:
        forms = _singular_forms(word)
        if any(form in male for form in forms):
            return "male"
        if any(form in female for form in forms):
            return "female"
    return ""


def infer_gender(name: str, description: str) -> tuple[str, str]:
    """Return (gender, source) for one character.

    Relation words outrank titles because a relation always describes THIS
    character, while a title may belong to somebody named in passing.
    """
    desc_words = _WORD_RE.findall(str(description or "").lower())
    verdict = _scan(desc_words, _MALE_RELATION, _FEMALE_RELATION)
    if verdict:
        return verdict, "relation"

    # Titles, in the description first and then in the name itself -- Folger
    # writes DUKE FREDERICK and SIR OLIVER MARTEXT with the rank in the name.
    verdict = _scan(desc_words, _MALE_TITLE, _FEMALE_TITLE)
    if verdict:
        return verdict, "title"
    name_words = _WORD_RE.findall(str(name or "").lower())
    verdict = _scan(name_words, _MALE_TITLE, _FEMALE_TITLE)
    if verdict:
        return verdict, "title"
    return "unknown", "unknown"


def parse_character_roster(play_text: str) -> tuple[CharacterRecord, ...]:
    """Every character the play's own cast list names, in listed order.

    Returns an empty tuple when the text has no recognizable cast block -- a
    vendored SCENE has none, only a whole play does. That is a normal absence,
    not a failure, so the caller decides what to do about it.
    """
    lines = str(play_text or "").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower().rstrip(":") in _ROSTER_HEADINGS:
            start = index + 1
            break
    if start is None:
        return ()

    records: list[CharacterRecord] = []
    seen: set[str] = set()
    group_context = ""
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if _END_OF_ROSTER.match(stripped):
            break
        if _GROUP_HEADER_RE.match(line) and "," not in stripped:
            # "Lords attending Duke Senior in exile:" -- a grouping label, not a
            # person, but it DESCRIBES the indented members listed beneath it,
            # who otherwise carry no description at all. Without this, JAQUES
            # and AMIENS come back genderless despite the play plainly calling
            # them Lords.
            group_context = stripped.rstrip(":").strip()
            continue
        # A counted collective ("Three Witches, the Weird Sisters") is written in
        # mixed case, so the all-caps entry pattern never sees it. It must be
        # read first, and it stands for several speakers rather than one.
        counted = _counted_aliases(stripped)
        if counted:
            head, _, tail = stripped.partition(",")
            gender, source = infer_gender(head, tail.strip() or head)
            if head not in seen:
                seen.add(head)
                records.append(
                    CharacterRecord(
                        name=head.strip(),
                        description=tail.strip(),
                        gender=gender,
                        gender_source=source,
                        aliases=counted,
                    )
                )
            group_context = ""
            continue

        match = _ENTRY_RE.match(stripped)
        if match is None:
            group_context = ""
            continue
        name = match.group("name").strip()
        description = (match.group("desc") or match.group("desc2") or "").strip()
        if name in seen:
            continue
        seen.add(name)

        # Back-reference: this entry's leading possessive genders the PREVIOUS
        # character, who had no description of their own. "MONTAGUE, his father"
        # makes ROMEO male; "CAPULET, her father" makes JULIET female.
        backref = _BACKREF_RE.match(description)
        if backref and records and not records[-1].is_known_gender:
            pronoun = backref.group("pronoun").lower()
            if pronoun in ("his", "her"):
                previous = records[-1]
                if not previous.description:
                    records[-1] = CharacterRecord(
                        name=previous.name,
                        description=previous.description,
                        gender="male" if pronoun == "his" else "female",
                        gender_source="back_reference",
                        aliases=previous.aliases,
                    )
        # An indented, description-less entry belongs to the group above it.
        indented = line[:1].isspace()
        gender, source = infer_gender(name, description)
        if gender == "unknown" and indented and group_context:
            gender, source = infer_gender(name, group_context)
            if gender != "unknown":
                source = "group"
        if not indented:
            group_context = ""
        records.append(
            CharacterRecord(
                name=name,
                description=description,
                gender=gender,
                gender_source=source,
                aliases=_aliases_for(name),
            )
        )
    return tuple(records)
