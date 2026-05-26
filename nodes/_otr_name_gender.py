"""Sprint 10A step 4 -- First-name gender oracle.

Deterministic name -> gender lookup over the project's
config.cast_pools.FIRST_NAMES list. The Stage 1 cast audit
(_otr_stage1_cast_audit.py) uses this to catch name/gender
inversions where the LLM emits a male-coded name with
gender='female' or vice versa.

The pool is curated by hand: each entry in cast_pools.FIRST_NAMES
gets an explicit gender tag. Names commonly used for both genders
are tagged "unisex" -- a unisex entry never produces a mismatch
finding (the LLM's gender choice is accepted as authoritative for
names that legitimately go either way).

For first names that DO NOT appear in cast_pools.FIRST_NAMES
(i.e. names the LLM invents from outside the pool), lookup returns
"unknown" -- the audit treats that as a soft signal, NOT a
mismatch. The LLM's gender choice is taken on faith for novel
names. A future step (Stage 3 validators) may add a stronger
heuristic for unknown names.

This module is PURE: no I/O, no LLM, no transformers imports.
"""

from __future__ import annotations

from typing import Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENDER_LITERAL = Literal["male", "female", "unisex", "unknown"]


# ---------------------------------------------------------------------------
# Curated first-name gender table
# ---------------------------------------------------------------------------
#
# Keys are lowercased first names that appear (or could appear) as
# cast member first names. Sourced from config/cast_pools.FIRST_NAMES
# plus a handful of names that the LLM tends to invent. The
# consistency test
# (tests/test_stage1_cast_audit.py::TestNameGenderCoverage) walks
# cast_pools.FIRST_NAMES and asserts every entry has a row here.
#
# Genders:
#   "male"   -- conventionally male-coded; LLM emitting 'female' is a
#               mismatch (caught by the audit). The original
#               operator-flagged ROBINSON VOSS -> female case sits in
#               this bucket.
#   "female" -- conventionally female-coded; mirror case.
#   "unisex" -- used either way. LLM's emitted gender is accepted
#               without raising a mismatch.
#
# The genuinely ambiguous-or-mythical entries (LEVIATHAN, DJINN,
# MORK, CAVOR, KARNACKI, TARKON, NEMO -- fictional / alien /
# mythological names) are tagged "unisex" rather than guessing.
# Picking the audit-conservative option here keeps the false-positive
# rate on the gate near zero.
#
# Coverage philosophy: when in doubt, prefer "unisex" over guessing
# a gender. The audit's job is to catch CLEAR inversions
# (ROBINSON=female), not to police edge cases.

NAME_GENDER: dict[str, GENDER_LITERAL] = {
    # ---- 1950s Americana Noir ----
    "vance":      "male",     # surname-as-first
    "stone":      "unisex",   # genuinely unisex when used as first name
    "margot":     "female",
    "nora":       "female",
    "sully":      "male",
    "mac":        "male",
    "hayes":      "male",     # surname-as-first
    "cole":       "male",     # operator-flagged inversion
    "drake":      "male",
    "quinn":      "unisex",
    "reese":      "unisex",
    "kane":       "male",
    "carter":     "male",
    "blake":      "unisex",   # increasingly unisex but lean male
    # ---- Afrofuturism ----
    "malik":      "male",
    "zuri":       "female",
    "chidi":      "unisex",
    "ayo":        "unisex",
    "oya":        "female",
    "kael":       "male",
    "tariq":      "male",
    "nia":        "female",
    # ---- Neo-Tokyo Cyberpunk ----
    "ren":        "unisex",
    "akira":      "male",
    "kenji":      "male",
    "yuki":       "unisex",
    "sora":       "unisex",
    "jiro":       "male",
    "rei":        "unisex",
    "hiro":       "male",
    # ---- Thai Density ----
    "krit":       "male",
    "mali":       "female",
    "niran":      "male",
    "sunan":      "unisex",
    "dao":        "female",
    "pim":        "unisex",
    "som":        "unisex",
    # ---- Russian Dieselpunk ----
    "lev":        "male",
    "anya":       "female",
    "dmitri":     "male",
    "sergei":     "male",
    "volkov":     "male",      # surname-as-first
    "mira":       "female",    # operator-flagged inversion
    "yuri":       "male",
    # ---- Simpsons ----
    "nelson":     "male",
    "martin":     "male",
    "carl":       "male",
    "lenny":      "male",
    "montgomery": "male",
    "seymour":    "male",
    "edna":       "female",
    "ned":        "male",
    "barney":     "male",
    "moe":        "male",
    "kent":       "male",
    "rod":        "male",
    "todd":       "male",
    "jimbo":      "male",
    "dolph":      "male",
    "kearney":    "male",
    # ---- Pulp adventure ----
    "dale":       "male",
    "tommy":      "male",
    "pinky":      "unisex",
    # ---- Public domain classics ----
    "alice":      "female",
    "allan":      "male",
    "ayesha":     "female",
    "cavor":      "unisex",    # fictional / coined
    "dracula":    "male",
    "edward":     "male",
    "griffin":    "male",      # surname-as-first
    "gulliver":   "male",
    "henry":      "male",
    "james":      "male",
    "john":       "male",
    "karnacki":   "unisex",    # fictional / coined
    "leviathan":  "unisex",    # mythical
    "mina":       "female",
    "nemo":       "unisex",    # fictional / Latin "no one"
    "phileas":    "male",
    "quasimodo":  "male",
    "robinson":   "male",      # **operator-flagged inversion case**
    "sherlock":   "male",
    "smee":       "male",
    "tarkon":     "unisex",    # fictional / coined
    "victor":     "male",
    "watson":     "male",
    "wendy":      "female",
    # ---- Peter O'Toole ----
    "lawrence":   "male",
    "reginald":   "male",      # operator-flagged inversion
    "anton":      "male",
    "priam":      "male",
    "maurice":    "male",
    "alan":       "male",
    # ---- Jim Carrey ----
    "truman":     "male",
    "fletcher":   "male",
    "joel":       "male",
    "stanley":    "male",
    "walter":     "male",
    "ace":        "male",
    "lloyd":      "male",
    "bruce":      "male",
    # ---- Robin Williams ----
    "mork":       "unisex",    # fictional alien
    "adrian":     "male",      # almost always male in US; female in UK rarely
    "sean":       "male",
    "andrew":     "male",
    "parry":      "male",
    "malcolm":    "male",
    "daniel":     "male",
    "chris":      "unisex",
    # ---- The Office ----
    "michael":    "male",
    "pam":        "female",
    "ryan":       "male",
    "kevin":      "male",
    "kelly":      "female",    # lean female in US
    "meredith":   "female",
    "toby":       "male",
    "darryl":     "male",
    "erin":       "female",
    "creed":      "male",
    "oscar":      "male",
    "phyllis":    "female",
    # ---- Real actors ----
    "steve":      "male",
    "rainn":      "male",
    "jenna":      "female",
    "mindy":      "female",
    "ellie":      "female",
    "rashida":    "female",
    "ed":         "male",
    # ---- Classic fiction ----
    "clarisse":   "female",
    "doug":       "male",
    "travis":     "male",
    "charlie":    "unisex",
    "will":       "male",
    "faber":      "male",
    "rick":       "male",
    "palmer":     "male",
    "glen":       "male",
    "isidore":    "male",
    "bob":        "male",
    "donna":      "female",
    "juliana":    "female",
    "manfred":    "male",
    "leo":        "male",
    # ---- Richard Pryor ----
    "gus":        "male",
    "monty":      "male",
    "duane":      "male",
    "rufus":      "male",
    "leroy":      "male",
    "skip":       "male",
    "grover":     "male",
    # ---- Robin Williams (additional) ----
    "peter":      "male",
    "sailor":     "male",
    "djinn":      "unisex",    # mythological
    # ---- LEMMY (special) ----
    "lemmy":      "male",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup_first_name_gender(full_name: str) -> GENDER_LITERAL:
    """Look up the canonical gender for a cast member's full name.

    Splits on whitespace and looks up the FIRST token (case-
    insensitive). Returns one of "male" / "female" / "unisex" /
    "unknown".

    Args:
        full_name: cast name, typically "FIRST LAST" but tolerant of
            "FIRST", "FIRST MIDDLE LAST", or punctuation. Empty
            strings + None-equivalents return "unknown".

    Returns:
        GENDER_LITERAL.

    Behavior:
        * "ROBINSON VOSS" -> "male" (first token 'robinson')
        * "Mira Drake"    -> "female" (first token 'mira')
        * "Zog Frellnax"  -> "unknown" (LLM-invented name not in pool)
        * ""              -> "unknown"
    """
    if not isinstance(full_name, str):
        return "unknown"
    s = full_name.strip()
    if not s:
        return "unknown"
    # First whitespace-separated token; strip surrounding punctuation
    # (apostrophes, periods, commas) that names sometimes carry.
    first = s.split()[0].strip("'\"-.,;:")
    if not first:
        return "unknown"
    return NAME_GENDER.get(first.lower(), "unknown")


def is_inversion(canonical: GENDER_LITERAL, llm_gender: str) -> bool:
    """Return True if the LLM's gender choice contradicts the canonical
    name-gender lookup.

    Rules:
        * canonical == "unknown" -> not an inversion (LLM choice accepted).
        * canonical == "unisex"  -> not an inversion (name goes either way).
        * canonical != llm_gender (after lowercasing) -> inversion.
    """
    if canonical in ("unknown", "unisex"):
        return False
    if not isinstance(llm_gender, str):
        return False
    return canonical != llm_gender.strip().lower()
