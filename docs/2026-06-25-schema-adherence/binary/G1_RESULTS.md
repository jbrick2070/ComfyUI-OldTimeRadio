# G1 RESULTS -- abstain-residual count -> DROP the binary lane

**Gate:** `binary/pass01_plan.md` G1 -- "MEASURE the abstain residual: what fraction
of spans land in NEITHER deterministic detector (the true ABSTAIN set)? If ~0, the
binary lane is unnecessary -> stop." Offline, no-GPU.

**Verdict: the genuine residual is ~0 -> DROP the dialogue/stage-direction binary
lane (Lever 2).** The whole schema-adherence sprint's load-bearing goal (Lever 1)
is already shipped; with G1 clearing to ~0, Lever 2 is not built.

## Method

Tool: `scripts/_otr_g1_abstain_residual.py` (pure, offline; reuses the REAL
detectors). Corpus: all 638 shipped episode ledgers under
`output\otr\episodes\**\*_ledger.json`. For every `speaker_role == "character"`
line (5,513 total):
- **caught** = the production deterministic detectors fire on the frozen text --
  `detect_stage_business_for_reroll` (Tier-2) OR a confident `split_stage_business`
  action. Both gate on `is_third_person_action_clause` (lead verb in the finite
  `_NARRATION_VERBS` whitelist, <= 12 words, no 1st/2nd-person pronoun).
- **broad** = a deliberately OVER-broad, verb-AGNOSTIC superset heuristic
  (`_relaxed_action_clause`: any 3rd-person-ish action verb -- whitelist OR a
  present-3sg `-s` verb OR an `-ing` gerund -- <= 20 words, no 1st/2nd pronoun)
  flags an outside-quote span or an undelimited punctuation chunk.
- **residual candidate** = broad AND NOT caught -> a direction-SHAPED span the
  production detectors miss.

The frozen corpus is POST-scrub, so the operative number is what REMAINS in shipped
dialogue (a genuine leak the detectors missed), not what was stripped upstream.

## Raw numbers

| metric | value |
|---|---|
| ledgers scanned | 638 |
| character lines | 5,513 |
| production-caught on frozen (reroll-only leaks the floor declined) | 154 |
| broad residual CANDIDATES | 841 (15.3% of character lines) |
| genuine leaked stage directions (40-sample inspection) | **0 / 40** |

## The 15.3% is ~entirely FALSE POSITIVES (the decisive finding)

A 40-line cross-corpus sample of the 841 candidates contains **ZERO genuine leaked
stage directions.** Every flagged line is legit spoken content that the verb-
agnostic heuristic over-flagged:
- **Names ending in `s`** read as a "verb": `"Reeves, the thermal regulators
  aren't bypassing..."` -> flagged on `Reeves` (a vocative, not an action). (~7/40.)
- **Ordinary 3rd-person dialogue verbs**: `gets`, `defies`, `seems`, `changes`,
  `feels`, `blooms` in normal sentences (`"This luminescence defies every
  textbook..."`, `"This changes everything."`).
- **In-character SPOKEN commands** (radio-drama astronaut narrating aloud):
  `"Initiating final descent."`, `"Overriding AI..."`, `"Engaging manual
  jettison, Yuri."`, `"Aborting jettison!"`, `"Accessing mainframe."` -- these are
  DIALOGUE the character speaks, not stage directions.

This is exactly why the production `is_third_person_action_clause` uses the precise
`_NARRATION_VERBS` whitelist: a verb-agnostic classifier (which is what an LLM
binary span call approximates) over-flags clean dialogue massively. Worse, the
in-character spoken commands show the binary lane's DOWNSIDE: an LLM that
classified `"Initiating final descent."` as stage-direction and stripped it would
DAMAGE the script.

## Conclusion

- The genuine abstain residual (real leaked directions the two-tier deterministic
  detectors miss) is **~0** across 5,513 shipped character lines. The detectors +
  the shipped freeze floor already catch the real stage business; what remains is
  clean dialogue.
- Per the G1 gate: **DROP the binary dialogue/stage-direction lane (Lever 2).** G2
  (byte-identity of abstain) is moot -- there is no lane.
- The `_NARRATION_VERBS` whitelist is the correct precise design; if a real future
  leak with a non-whitelist verb is ever captured, the cheap fix is to ADD that
  verb to the whitelist, not to introduce an LLM span classifier that re-litigates
  clean dialogue (and risks stripping in-character spoken commands).

## Reproduce

`cd` repo; `.venv\Scripts\python.exe scripts\_otr_g1_abstain_residual.py`
(stdout = JSON with the counts + a 40-line sample; stderr = the one-line summary).
The sample is what the verdict above was eyeballed against.
