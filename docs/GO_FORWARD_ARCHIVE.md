# GO_FORWARD -- ARCHIVE

**Archived, never deleted.** `docs/GO_FORWARD_PLAN.md` is FORWARD-ONLY by its own
first line, and its 2026-08-16 self-audit found the rule broken in the most
misleading way available: *"the bulk of the remaining length is DONE narrative
inline inside OPEN sections"*. A section headed **OPEN, IN PRIORITY ORDER** was
235 lines of which roughly two thirds described work that had SHIPPED.

**THIS FILE IS NOT READ TO RESUME.** A new window reads the plan. This is where
the receipts live once the thing they describe is closed -- kept because the
audit's other finding was that *"roughly a third of these sections are standing
operator rulings phrased as 'do not re-open', and losing one costs more than the
length does."* So nothing here was summarised away: every block below is the
VERBATIM text that was in the plan, and every RULING it carried was lifted back
into the plan before the block moved.

Moved 2026-08-23 by the daylight coder window.

---

## GHOST PROMPT V2 -- built, proven and shipped (closed 2026-08-22)

**GHOST PROMPT V2 IS BUILT, PROVEN AND SHIPPED (closed 2026-08-22, Opus coder
window).** Code + tests: `a8fad82c`, pushed on `v2.0-alpha`, HEAD == origin.
Receipt: `docs/2026-08-22-ghost-prompt-v2-publish-receipt.md`.

**THE DEFECT IS ON THE RECORD IN ITS OWN BYTES.** The v1 baseline was rendered
and published FIRST, then all eight of its prompts were reconstructed from the
pre-change composer (`git show e1265208:...`) and HASH-MATCHED 8/8 against the
trace. It really did ship cast names inside the picture, ending mid-clause:

    moves with mali vance demands dr sterling hand, Tense mood, scene, ...
    moves with gulliver reeves forcefully seizes the shredded, ...

and both announcer beats plus both music beats were byte-identical to each
other. That is what v2 replaces.

**THREE LIVE ARMS, ALL PUBLISHED TO `otr/obs/`:**
* **A** `signal_lost_disc_of_destiny_20260822_163533` -- v1 prompts, the baseline.
* **B1** `signal_lost_disc_of_destiny_20260822_171254` -- v2, deterministic leaves.
* **B2** `signal_lost_turntables_lament_the_last_spin_20260822_174415` -- v2,
  eight rows `source=writer_llm` on `google/gemma-4-12b-it`, zero fallback,
  zero replay.

**A vs B1 IS THE SAME-SEED A/B AND IT IS EXACT** -- measured, not assumed: same
voiced-text SHA, same episode seed, same roles, same frame counts, same eight
video seeds, same cast, same negative, and **nothing else differs but the
prompt**. Prompt length fell from 208-317 characters to 164-198, every one
measured at 32-43 installed SD1 tokens in a single window. **This is the pair to
eyeball** -- same script, same seeds, one variable.

**A vs B2 is NOT same-script, and the receipt says so rather than implying it.**
The technical slot also drives the writer's structured passes, so pinning it to
gemma changed the story. B2 proves the LLM TREATMENT, not a pixel comparison.
A same-script LLM arm is **not obtainable without reverting code** -- the v1
content route no longer exists -- which is exactly why the A arm was captured
before any code changed.

**THE MODEL CHOICE WAS MEASURED, NOT ASSUMED, AND THE FIRST LIVE LEG IS WHY.**
Mistral-Nemo answered in a perfectly valid envelope but wrote four-word
abstractions (`signal oscillates, broadcast begins`; `tension builds`), which
the validator rejected twice and the batch fell to the deterministic pools --
machinery correct, instruction too thin. The batch template gained worked
examples AT the target length plus named counter-examples. Then both candidates
were put to the REAL batch prompt directly, two minutes instead of two renders:
**gemma-4-12b 8/8 accepted, Mistral-Nemo 4/8** (three rejected for putting a
hand into `object` mode). A word COUNT is a number a model does not feel; a
sentence at the target length is one it can match.

**ONE THING IS STILL THE OPERATOR'S CALL AND IS DELIBERATELY NOT DONE:**
`config/profiles/otr_ghost_signal_v3.json` still pins `technical_model` to
Mistral-Nemo. B2 pinned gemma PER-LEG via `--set` rather than editing the
shipped profile, because promoting it would change SCRIPTS on this lane and
story output is a closed subject. The 8/8-vs-4/8 measurement is the case for
promoting it; say the word and it is a one-line profile edit.

**A POST-CODING QA PASS CAUGHT A REAL BLOCKER BEFORE THIS LANDED,** and it is
worth remembering as a class: the writer-release assertion had been written as a
`finally:` INSIDE node 92's `except Exception -- never break the render`, so a
writer that failed to release would have been logged as a handled warning and
the episode would have walked into `run_real_episode` holding writer weights on
a 16 GB card. **A guard inside the thing that swallows guards is not a guard.**
The suite was green straight through it, so the replacement test is STRUCTURAL:
it walks the AST and refuses the call inside any broad-`except` try.

**Gates:** suite **12225 passed / 134 skipped / 1 xfailed**; Bug Bible
**22/26/3**; canonical validator **23 nodes / 57 links** with the blob still
`c27dff36`; 54 variants / 0 failures; forbidden sweep 0 runtime hits; BOM /
0-byte / AST clean. `pyproject.toml` untouched (it is a publish trigger).


---

## A. GHOST SIGNAL -- built, proven and shipping (closed 2026-08-22)

**A. GHOST SIGNAL IS BUILT, PROVEN AND SHIPPING (closed 2026-08-22).**
Three green chunks, all pushed on `v2.0-alpha`:
* `01317eec` -- dependency + schema lock (ADE pinned at
  `9257651221002dcba0a12f9cff37e1944e58fb60`, both artifacts hash-verified
  before boot, `/object_info` captured from a post-install clean boot).
* `d0b3a65b` -- the wired implementation slice (adapter, pure composer, durable
  subject sigil, clean-Lanczos delivery mode, registry/profile/G4/G3.7, 353
  focused tests).
* The publish receipt + `shipping` promotion (this session's third chunk).
**Live proof:** episode `signal_lost_the_constables_knock_20260822_050116`,
8/8 beats on `animatediff15_video`, `RESULT SUCCESS` + `obs_publish OK`, 21:33,
1920x1080 @ 25 fps in the LIVE `otr/obs`. Every section-10 pass condition
verified -- receipt at `docs/2026-08-22-ghost-signal-publish-receipt.md`.
* **The first leg FAILED and it was worth it.** A bookend beat with the motion
  pass off pulls the pack's own subject (163-178 chars) AND its motion register
  (130-209); on `recur_frac` that composes to ~474 against a 320 ceiling, and
  `_trim_to` could not shrink either because both are comma-free prose. The
  composer refused at cast-time preflight before a weight loaded. Root-fixed:
  word-boundary fallback, no dangling function words, and step 4 of the trim
  order now shrinks pack surfaces. **Every unit budget test had handed the
  composer an authored clause, which short-circuits the pack register entirely
  -- that is the test-design lesson, not a Ghost-specific one.**
* **THE LOOK IS ACCEPTED, DO NOT CHASE IT.** The motion reads fast (12.5 fps of
  AnimateDiff held to 25). Operator, watching the published episode: *"i was
  expecting experimental vj"* and *"its perfect"*. Not an open defect.
* **NO VRAM CLAIM.** Admission stays unenforced; the lane may OOM. A single
  5872 MiB / 100% reading was observed and is NOT a qualified cost row.
* **G3.7 note worth keeping:** `_portrait_free_roles_from_policy` is INERT for a
  lane with an EMPTY `still_plan` -- it looks for a portrait row saying "never".
  A no-still lane is covered by the stronger `accepts_still = False` gate at the
  image dispatcher instead. Adding a portrait/never row to make the role set
  light up would be a declaration the lane cannot honour.
* **OWED, and honestly stated:** the independent finished-diff review seat went
  UNFILLED. The Agent tool was disabled for the session and the substituted
  Codex CLI lane thrashed in a PowerShell error loop (1.4 MB of tool errors, no
  report) and was killed. The live leg is stronger evidence and it caught the
  real blocker -- but it is not a review, and this line says so rather than
  implying coverage.


---

## A-ORIGINAL -- the superseded pre-build plan, kept for its receipts

**A-ORIGINAL (superseded, kept for the receipts). GHOST SIGNAL (AnimateDiff) -- CODE-AND-WIRE-READY, r4 COMPLETE, NOT BUILT.**
The operator's ultra-low-VRAM video lane. A separate session took it through a
full four-round Kibitz arc; **no code, dependency, workflow, render, test,
measurement, commit or push has been performed.** This is the next build.
* Plan: `docs/2026-08-22-GHOST-SIGNAL-CODING-PLAN.md`
* Judgment: `kibitz-runs/2026-08-22-ghost-signal-code-ready-plan/r4/judgment.md`
* Receipt: `kibitz-runs/2026-08-22-ghost-signal-code-ready-plan/r4/final.md`
* **The receipt is BYTE-IDENTICAL to the plan -- verified,
  sha256 `06104e911e599f7630563eaab5ae5cb5aa92c40e0331d3d0e253ba6c4150e81b`.**
* Upstream: `ComfyUI-AnimateDiff-Evolved` (Kosinkadink) is the canonical repo;
  the pinned graph uses the verified `ADE_AnimateDiffLoaderGen1` and
  `ADE_StandardStaticContextOptions`.
* The plan FREEZES: all eight nodes with every input/output/widget and all ten
  links; staged executor aliases and low-VRAM release ordering; fresh
  audio-length source animation per beat at 12.5 fps held twice to 25 fps;
  ledger motion + the selected visual-style pack as prompt authorities;
  character sigils where available and strong abstract treatments otherwise;
  fixed 512x288 generation enlarged directly to 1920x1080 with clean Lanczos;
  the complete profile, canonical-workflow wiring audit, implementation
  sequence, tests and one publish smoke.
* **G3.7 applies from day one:** Ghost Signal must declare its `still_plan`
  (kind / cardinality / aspect / required / framing), and the portrait-free
  role set will honour it automatically -- no edit to the image phase.


---

## B -- closed 2026-08-22 (and half of it was already closed)

**B IS CLOSED, AND HALF OF IT WAS ALREADY CLOSED BEFORE THIS WINDOW LOOKED
(verified 2026-08-22, not assumed).** The asymmetry the item was really about --
*"a refusal currently fails the WHOLE EPISODE"*, harsher than the operator's
*"I accept some errors"* ruling -- was fixed in the video-lane session and is
live in `nodes/otr_image_gen_dispatcher.py`: a refusing engine records
`reason="model_refusal"` skip evidence (`:1609`) and the completeness gate
tolerates exactly that reason and no other (`:1754-1761`). The decision the item
asked for was therefore already made, in favour of DEGRADE. **Grep the code
before re-planning a carried item; this one was stale in the queue.**
What remains is optional and low-value under the standing ruling: root-causing
the SEED-dependent Ideogram music-card refusal. It costs GPU cycles the operator
explicitly said not to burn on blemishes, so it is not queued.


---

## C -- closed 2026-08-22 (`2aa13b35`)

**C IS CLOSED (2026-08-22, `2aa13b35`).** The shipped card that read *"...of a
subterranean"* and stopped. The reduction was working as designed and the design
was one stage short: a word-boundary cut is honest about never splitting a word
and says nothing about landing mid-thought. `_still_word_fit_card` now takes
whole COMMA CLAUSES before any word cut, then drops dangling function words:

    before: The signal came from the flooded lower levels of a subterranean
    after:  The signal came from the flooded lower levels

**The clause stage is the one that actually rescues it**, because the trailing
word there was an ADJECTIVE and no function-word list catches that without a
parts-of-speech tagger this repo does not ship. **SCOPED TO REDUCTIONS ON
PURPOSE:** an authored line that ends on "the" is the AUTHOR's line and THE LAW
forbids rewriting a story for style, so the contract is "a line this function
shortened comes back a complete thought; a line under both caps comes back
byte-identical however it ends", and both halves are pinned. The rule now lives
once in `nodes/_otr_shared/text_tails.py`; the Ghost composer had its own copy
and imports the shared one instead.


---

## E -- the CLOSED half (2026-08-22); the enforcement half stays in the plan

**E IS HALF CLOSED, AND THE HALF THAT IS NOT SAYS SO OUT LOUD (2026-08-22).**
`docs/IMAGE_GEN_PREFLIGHT.md` IG2.2 read *"Every engine SERVES every role the
menu offers it for"*. It does not check that and never did -- it reads `roles`
off the registry, renders nothing, and `ideogram4_local` is the proof: declares
all three roles, passes IG2.2, refuses at render on real card prose. The gate
and its test now say **DECLARES**, which is what they check.
**The enforcement half needs an arc and is DELIBERATELY NOT DONE SOLO.** What
would close it is a per-KIND declaration -- "every engine serves every request
KIND its lane asks for" -- and no image engine declares a kind surface today.
Adding one is a design choice with more than one defensible answer (what the
kinds are, who owns the role-to-kind map, what a partial declaration does to the
menu), so by the 2026-08-17 routing rule it wants a full arc, and it belongs
with item D rather than bolted onto a gate doc.

