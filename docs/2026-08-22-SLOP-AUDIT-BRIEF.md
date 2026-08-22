# Slop audit brief -- OTR, 2026-08-22

**Reviewer: read the REAL files. Ground every claim in a path and a line.
A claim you cannot ground is worse than silence.**

You are auditing a working ComfyUI custom-node pipeline for AI-generated slop
and cleanup targets. The suite is green (11,827 passing) and the code RUNS in
production daily, so "it looks unusual" is not a finding. Only report things
that are wrong, dead, or lying.

## What counts as a finding (ranked)

**1. A DOCSTRING OR COMMENT THAT CONTRADICTS ITS CODE.** This is the highest-
value class here and it is known to have hits -- three were found by accident in
one day:

* `scripts/otr_gpu_soak_matrix.py` claimed "no heavy local video model is in the
  rotation" long after `PROFILES` had them (fixed today).
* A test gate fired on the docstring that WARNED about the idiom it forbids.
* Two tests grepped their own module's prose and flagged it as the violation.

So: where does prose assert a constraint, a count, a default or a behaviour the
code no longer has? Cite both sides.

**2. DEAD OR UNREACHABLE CODE.** A branch nothing can enter, a parameter no
caller passes, a constant nobody reads, a helper with no call sites, an
`except` that cannot be raised into. Prove it is unreachable rather than merely
unused-looking -- this repo has guarded imports and duck-typed dispatch, so
"no static caller" is not proof on its own.

**3. A DECLARATION THAT NOTHING READS.** The repo's own worst bug of the month
was `still_word` declaring `StillPlanRow(kind="portrait", required="never")`
that no consumer ever consulted, so a portrait was minted for every cast member
for months. Look for the same shape: a field, capability or config key that is
declared, documented, tested for its VALUE, and never actually consumed.

**4. DUPLICATED LOGIC THAT HAS ALREADY DRIFTED.** Two copies of the same rule
where the copies now disagree. Drifted copies are a finding; identical copies
are at most a note.

**5. COPY-PASTE ARTIFACTS.** A comment describing a different function, a
variable named for what it used to hold, an error message naming the wrong
engine or node.

## What is NOT a finding -- do not report these

* Style, naming, formatting, line length, type hints, f-string vs %.
* "Add tests" / "add error handling" / "consider refactoring". Not asked.
* Long comments. This codebase deliberately records WHY, including operator
  rulings and the history of bugs. Density is intentional; do not call it noise.
* Anything requiring a rewrite of working, shipped behaviour.
* Speculation about performance without a measurement.

## Where to look first

* `nodes/_otr_video_engines/` -- the video adapters and `render_driver.py`
  (large, edited by many sessions).
* `nodes/_otr_image_engines/` -- the image adapters.
* `nodes/otr_image_gen_dispatcher.py`, `nodes/otr_shot_lock.py`,
  `nodes/otr_silent_composite.py`.
* `scripts/` -- harnesses accumulate stale docstrings fastest.

## Output

A short markdown list. For each finding:

```
FILE:LINE  <one-line claim>
  evidence: <the prose or code that is wrong, quoted>
  contradicted by: <the code that disproves it, quoted, with its line>
  class: docstring-lie | dead-code | unread-declaration | drifted-duplicate | copy-paste
```

Rank most-certain first. If a section is clean, say so in one line. Do not pad.
