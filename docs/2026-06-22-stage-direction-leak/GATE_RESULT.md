# Precision gate result (Chunk 2) -- PASS

`scripts/stage_direction_scan.py` over the OTR episodes tree (489 frozen
ledgers), non-mutating.

- flagged character lines: **27** across ~8 episodes (the leak is WIDESPREAD,
  not one-off -- validates the sprint).
- `would_mutate` (the destructive floor would strip): **20**.
- detect-only (floor conservatively declined -> reroll handles): **7** (longer
  multi-clause leads, e.g. "sets down microscope, picks up a pen Then..." /
  "looks directly at Duane, concern in her eyes ...").

## Manual review of all 20 `would_mutate` rows -> 20/20 TRUE POSITIVES, 0 false positives
Every mutation strips a genuine LEADING stage direction down to the spoken
dialogue, verbatim, including correct handling of an opening dialogue quote:
- "twirls his pen nervously Look, Pinky..." -> "Look, Pinky..."
- "pauses, sets pen down Alright, Pinky..." -> "Alright, Pinky..."
- "clenches jaw You mean..." -> "You mean..."
- "slams fist on table Reeves, you're playing with fire!..." -> "Reeves, ..."
- "sipping coffee, casually Zuri, I heard..." -> "Zuri, I heard..."
- "clenching microphone stand Brothers and sisters..." -> "Brothers and sisters..."
- "squints at screen \"Hmm, it seems..." -> "\"Hmm, it seems..." (quote kept)
- "taps on the tablet \"We've got a unique opportunity..." -> "\"We've got..."
- ...and 12 more, all leading-stage-direction -> dialogue.
Full JSONL: `gate_scan_mutating.jsonl`.

## Outcome
ZERO false positives among the destructive strips -> `BARE_STAGE_FLOOR_ACTIVE = True`
is VALIDATED. The freeze floor (Chunk 4) ships ACTIVE. (Multi-clause/long leads
that the floor declines are the reroll gate's job, Chunk 3.)
