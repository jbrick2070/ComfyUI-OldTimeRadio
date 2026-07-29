# r1 FINAL -- the motion floor, judged

Judge: Claude/Opus (driver seat; anchor at `r1/anchor_claude.md`), HEAD
`f6977e3d`. Panel: codex `gpt-5.6-sol` high (returned, grounded below);
agy still running at synthesis time -- its claims fold into r2 rather than
holding this round, and that is recorded rather than hidden.

## THE FINDING THAT REFRAMES THE RULING

**The green episode everyone has been citing as proof was seven dead-flat
stills and a 52-second frozen console, and it passed every gate in the build.**

`GO_FORWARD_PLAN.md:12-14` records the first-ever live proof of the encoder
chunks: `obs_publish OK`, 14,637,297 bytes, **`engine_histogram
{"still_flat": 7}`**, and `appended 52.0s console`. `cheap_families.py:330`
documents `still_flat` in its own words as *"A DEAD-FLAT still: the selected
image held STATIC (no pan/zoom...)"*.

So the operator is not describing a hypothetical. He watched an episode that
was a still for its entire length -- beats AND credits -- and every check in
this repo called it green. That is the defect. **The rule has to bite on
whether the FRAMES DIFFER, not on whether a video engine produced them**,
because `still_flat` emits N identical frames and satisfies every frame-count,
coverage and contract check we own.

## WHAT IS ALREADY DONE, AND MUST NOT BE REBUILT

**Claim 3 of the brief is already the shipped behaviour, measured.** For every
contract permitting a tail trim, `partition_beat` renders the smallest legal
length at or above the target and trims the surplus
(`coverage_plan.py:287-297`). Codex found this independently; my anchor found
it independently; an audit of the LIVE registry settles it:

    all 31 registered engines cover a 1-second beat with REAL VIDEO
    humo/wan_i2v  min 33 -> render 33 (1.3s), trim 8
    google_veo    menu (100,150,200) -> render 100 (4.0s), trim 75
    word_razzle   menu (125,200) -> render 125 (5.0s), trim 100
    allow_tail_trim=False on a video engine: NONE

`google_veo` rendering 4.0 s for a short beat IS the operator's "if the
minimum is four seconds, we should have video for four seconds", shipped since
2026-07-25. **Case 1 of the 2026-07-28 still-floor ruling ("target frames <
min_frames") is unreachable for every registered engine.** The beat half of
this campaign is therefore a ROSTER TEST plus a doc correction, not a
mechanism. Harness: `tmp/_mf_audit.py`.

## THE POLICY, RESTATED SO IT IS ENFORCEABLE

Codex's MUST-FIX 1 and 2 are accepted: the brief conflated two different
things and used an unenforceable bound. Two separate invariants:

**A. NATIVE COVERAGE (beats).** A beat routed to a MOTION engine is covered by
native rendered frames for its whole duration -- already true, now pinned.

**B. THE STATIC-RUN CEILING (everywhere).** No delivered artifact may hold one
source frame longer than `round(4 * final_fps)` frames. This is measured on
DELIVERED PIXELS, so it catches `still_flat`, a frozen credits backdrop and a
held last frame with one rule, and it cannot be gamed by re-routing.

The bound is a frame count derived from fps, not "2-4 seconds" in prose.

**The explicit still ENGINES are not outlawed by B -- they are governed by
it.** `still_flat` / `still_word` are declared static routes and stay legal
for what they are; what stops is an EPISODE made of them, and a 52-second
hold. Their cadence policy is a separate declared route (codex CUT 3 accepted:
"several short stills" is not an escape hatch).

## THE CREDITS FIX (accepted, codex's shape over mine)

`N = min(body_duration, credits_duration)`. Cut that tail from the assembled
body, **play it forward once**, and loop only when the body is shorter than
the roll, with a short crossfade at the seam. Everything stays inside the
presentation-only boundary at `otr_credits_roll.py:1421-1447`.

Better than my "cut N and loop": the honest pass plays first, and N is derived
from two numbers the node already has rather than being a new constant.
Rejected from my own anchor: the synthesized "slow drift" (codex CUT 2) --
it converts a still into cosmetic movement and adds a second visual authority.

## THE TEST THAT WOULD HAVE PASSED THE BUG -- codex's best catch

`tests/test_credits_roll_spec.py:446-470` proves col-3 TEXT scrolls **over a
deliberately constant backdrop**. A naive whole-frame motion check therefore
goes green on the exact frozen-background defect, because the overlay moves.
**Grade the backdrop BEFORE the overlays, as its own component.** This is now
a hard requirement on WIRE-W5, alongside per-beat frame provenance, extension
mode, and longest-repeated-frame run.

## THE ONE REAL CONTRADICTION -- decide before WIRE-W3b

`coverage_plan.py:1-6` forbids ping-pong as coverage; `eng_wan_ti2v.py:695-735`
deliberately mirror-extends a VRAM-limited native render, and
`PROD_BUG_LOG.md:2679-2719` (`PBUG-20260723-02`) preserves that for the 8 GB
tier. Both cannot be true of one receipt. **OPERATOR DECISION, stated as a
fork rather than resolved here:**

- (a) native multi-render coverage supersedes the 8 GB exception; or
- (b) WAN ping-pong survives as a NAMED `repeated_motion` mode that does not
  claim strict native coverage, and the grader reads the name.

Default if unruled: **(b)** -- it preserves a shipped tier, and (a) is a
GPU-budget decision no panel should take on the operator's behalf.

## agy's r1, GROUNDED -- one real catch, one useful dissent, three rejects

agy landed after first synthesis. Judged claim by claim rather than whole:

**REJECTED -- it argues to KEEP the thing the operator just ruled out.** agy's
MUST-FIX 1 recommends retaining the held final frame "as an explicit, bounded
UI presentation exception". That is the 52-second still, and the operator's
direction is not ambiguous. Recorded because a panel arguing FOR the defect is
worth knowing about, not because it survives.

**ACCEPTED (real, and it sharpens the fix).** agy is right that a tail cut
assumes the body is longer than N, and that a short episode or test clip would
fail extraction. Codex's `N = min(body_duration, credits_duration)` already
absorbs it; the edge case is now explicit rather than incidental, and "body
shorter than the roll" is a named case with its own acceptance.

**ACCEPTED AS A CONCERN, REJECTED AS A FIX.** agy is right that a provider
minimum can discard 75%+ of generated frames on a short beat (it meets codex's
MUST-FIX 8 from the other side). Both its proposed fixes are refused:
restricting `allow_tail_trim` to "low-overhead engines" would re-open exactly
the over-segmentation WIRE-W1 just closed (an engine without tail trim gets
refused or split), and "adjust short beat durations to match engine
`min_frames`" inverts the contract -- the beat length comes from the AUDIO,
and bending the story's timing to suit a model is the tail wagging the dog.

**THE USEFUL DISSENT -- the panel genuinely split on the ping-pong.** codex
defaults to (b), a named `repeated_motion` compatibility mode. agy demands (a),
forbid it outright, on the grounds that mirror-extension is visible
oscillation and not forward motion. **agy has precedent on its side that
neither seat cited:** `a1d810f1` already banned mirroring on the lip-sync lane
after the operator's "no render backwards, that doesn't work"
(`allow_mirror=False` + `MirrorExtensionForbidden`). So backwards motion is
ALREADY refused where it was ever eyeballed. That materially strengthens (a)
and it is now in the operator's fork below rather than buried.

**MISREAD -- do not act on it.** agy cites `audio_conditioned_video` at
`otr_credits_roll.py:182`; that line is inside `_fam`, a family-name helper in
the credits node, and the credits node knows nothing about audio conditioning.
Its `announcer_voice.py` / `batch_character_voices.py` cites are for a
mouth-still classification that WIRE-W7 already owns.

**REJECTED -- it violates a standing contract.** agy's SHOULD-FIX 1 would
degrade a failed backdrop to "a solid darkened canvas". That is
credits-over-black, which the LOOK CONTRACT names as a bounce. The
presentation-only boundary already returns the finished body with a zero tail,
which is the honest degradation. Its `credits_backdrop_mode` widget is also
refused: the operator has ruled, and a mode selector re-opens a settled
decision while forcing a canonical-JSON change for a presentation toggle.

## CARRIED TO r2

- codex SHOULD-FIX 3: `OTRCreditsRoll` declares no `IS_CHANGED`
  (`otr_credits_roll.py:1361-1414`), so a re-run on the same body path could
  serve cached credits. [ASSUMPTION -- verify inputs are unique per run.]
- codex MUST-FIX 5: an uncoverable beat on a MOTION route must REFUSE, not
  silently become a still, or the delivered `engine_id` lies to WIRE-W5.
- Cost envelope: quantify discarded rendered seconds where a provider minimum
  overshoots a short beat (Veo 100, Pixverse 125). Contract tests now; live
  cloud stays parked pending spend approval.
- agy's "short episode / body shorter than N" case, as its own acceptance row.

## OUT OF SCOPE, EXPLICITLY (codex CUT 4, accepted)

WIRE-W4's audio slicer and WIRE-W7's mouth-still ownership are UNCHANGED by
this ruling. Those stills are INIT IMAGES a video render conditions on --
`accepts_still` governs minting and is explicitly distinct from continuity
(`frame_contract.py:13-18`). They are not a substitute for video.
