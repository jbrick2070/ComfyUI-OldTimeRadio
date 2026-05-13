# Audit S13.1 — Cast contract structural-token verification

**Date:** 2026-05-12
**Branch:** `v2.0-alpha`
**Predecessor:** `7ea481e` (S12.4 hash-length pin)
**Verdict:** PORT REQUIRED — five tokens slipped through; heuristic ported + extended; new contract assertion shipped.

## Method

Per the S10-S15 plan's S13.1 spec:

1. Construct fixtures with each of `TITLE`, `NOTE`, `TARGET`,
   `STYLE`, `NARRATOR` as a character `name` in `ledger.cast[]`.
2. Run the active cast contract:
   `_otr_casting._assert_voice_preset_invariant` plus any sibling
   cast validators (`_assert_unique_bark_voices`).
3. Record which structural tokens slip through.

## Pre-S13.1 result

**ALL FIVE tokens slipped through.** The pre-S13.1 cast contract
checked only:

- `_assert_voice_preset_invariant`: empty / non-`v2/*` voice_preset
- `_assert_unique_bark_voices`: voice preset collisions across rows

Neither has any opinion on the *name* shape. A cast row like
`{"name": "TITLE", "voice_preset": "v2/en_speaker_0", ...}` passes
both assertions cleanly.

This means: an LLM hallucination that emits `TITLE` (or any of the
other four) as a character name renders as a Bark voice line in
production with no contract pushback. The user only notices
post-render — too late to abort cleanly.

## Port + extension

Recovered `_SFX_CAST_BLOCKLIST_PATTERNS` and
`_looks_like_non_character_cast_name` from
`nodes/story_orchestrator.py` at commit `b6fb314^` (the parent of
S7.1, where the helpers were deleted). Ported verbatim into
`nodes/_otr_casting.py` as `_NON_CHARACTER_CAST_PATTERNS`, plus:

**Bug fix:** original `\bV\.O\.\b` and `\bO\.S\.\b` patterns had a
trailing `\b` after the final `.` that never matched (Python regex
word boundary doesn't fire after a non-word char at end-of-string).
Faithful port + bugfix here drops the trailing `\b` so `JOHN V.O.`
actually matches.

**Extension:** five new patterns explicitly anchored to the S13.1
tokens:

```python
r"^TITLE$", r"^NOTE$", r"^TARGET$", r"^STYLE$",
```

(NARRATOR was already covered by the original `\bNARRATOR\b`.)

Anchored as exact-match (`^...$`) on uppercase to minimize
false positives on real character names that happen to contain
the substring (e.g., a character named "Anna Title-Holder" should
NOT trigger).

## Contract assertion wired into `lock_cast`

```python
_assert_no_structural_tokens_in_cast(cast)
```

Lives alongside `_assert_voice_preset_invariant` and
`_assert_unique_bark_voices` in the lock_cast finalization block.
Raises `CastingFailedError` with a structural-token diagnostic
naming the offending row(s).

## Risk asymmetry (per the plan's inline rationale)

> The false-positive cost (a real character named "Style" gets
> rejected) is far lower than the false-negative cost (an LLM
> hallucination renders as a voice line in production).

Concrete numbers:

| Mode             | Cost                                       | Recovery                                |
|------------------|--------------------------------------------|-----------------------------------------|
| False-positive   | One reroll with a different name           | Auto-handled by the cast LLM retry path |
| False-negative   | Broken episode that ships before catch     | Manual investigation + re-render        |

The asymmetry is roughly 10:1 in favor of false-positive
acceptance. The patterns are aggressive accordingly. If a future
story legitimately needs a character named one of these tokens,
the right move is a case-sensitive whitelist check, not pattern
relaxation.

## Post-S13.1 result

Every structural token now raises `CastingFailedError`. Test
parametrized over `("TITLE", "NOTE", "TARGET", "STYLE", "NARRATOR")`:
5/5 pass. Plus 5 sanity tests:

- Real character names PASS the guard (no false positives on
  OSCAR SIRIKIT, WILL SMITHERS, JIMBO HALPERT, WENDY HUDSON,
  STANLEY CRANSTON, MINA SPENDER, BABA YAGA, LEMMY KILMISTER).
- ANNOUNCER slot exempted (canonical narrator slot, not artefact).
- Legacy SFX-cue artefacts still caught (BUG-LOCAL-090 and
  BUG-LOCAL-097 patterns preserved through the port: ALARM
  BLARING, SFX EXPLOSION, MUSIC QUEUE, KEVIN VOICEOVER,
  JOHN V.O., OFF SCREEN).

10 tests in `tests/test_cast_contract_rejects_structural_tokens.py`,
all PASS.

## Files

- `nodes/_otr_casting.py` -- port + extend + wire (in lock_cast).
  `__all__` extended with two new exports.
- `tests/test_cast_contract_rejects_structural_tokens.py` (NEW) --
  10 tests covering token rejection, real-name acceptance, helper
  classification, ANNOUNCER exemption, legacy artefact preservation.
- `docs/audit-S13.1.md` (this) -- audit record per plan
  ("audit doc records the verification result").

## Acceptance

- [x] Parametrized test exists.
- [x] Every structural token raises.
- [x] Audit doc records the verification result.
- [x] Heuristic ported into `_otr_casting.py` per the plan's
  "if any token slips through" branch.
- [x] Same commit ships the port + parametrized tests + dedicated
  regressions for previously-slipping tokens (per plan's
  "same commit ships" gate).
