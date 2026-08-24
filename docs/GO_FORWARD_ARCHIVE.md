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

---

## Moved 2026-08-23 (second pass -- "only truly go forward items")

### ALSO DONE AND PUSHED 2026-08-22 -- THE VIDEO-LANE SESSION (Opus coder window)

Everything below is committed, pushed, and lockstep-verified. Suite 12,084
passed / 0 failed at the last full run. **A new window does not need to rebuild
any of it -- read this to know what already exists.**

**1. `animatediff15_v3_haunted_video` -- v3 + the removable domain adapter.**
The clean v3 lane plus AnimateDiff v3's optional `v3_sd15_adapter.ckpt` (97 MB)
on the MODEL path via stock `LoraLoaderModelOnly`. Two design questions were
settled by READING THE ARTIFACT, not its documentation: it carries 256 tensors,
all UNet attention, ZERO text-encoder -- hence model-only and CLIP untouched --
and its keys are the legacy diffusers attn-processor spelling that ComfyUI maps
natively, so no conversion and no custom loader. **The operator and Gemini
independently preferred its output.** Apache-2.0 end to end: this is the first
Ghost configuration that could be published.

**2. Two CADENCE peers, `animatediff15_h3_video` / `animatediff15_h5_video`.**
The hold factor is now a class attribute every call site reads through `self`.
It was three module functions hard-wired to hold-2, so a peer declaring hold-5
would have rendered hold-2 under its own receipt -- **G1.3 for the third time in
one day.** Golden's cadence is pinned across 16 frame counts against an
INDEPENDENTLY RECOMPUTED reference, not by calling the new code with hold=2.

**3. `OTR_WRITER_SEED` -- the reason four comparisons failed today.**
`gen_kwargs` carried `do_sample=True` with no seed, so every leg wrote a
different episode and every impression formed while comparing two legs turned
out to be about the SCRIPT. Opt-in, off in production, keyed on the PROMPT not a
call counter (a counter makes seeds order-dependent, so one conditional pass
shifts everything after it and the reproduction drifts silently).

**4. The cast-seed guard.** The operator asked whether GULLIVER REEVES in every
episode was random. It was not: four bake-off legs ran with a leaked
`OTR_CAST_SEED=42` -- BUG-LOCAL-269 through a different door. The resolver now
warns LOUD when the seed is set and `OTR_C7` is not, naming the cast it will
produce, and a test sweeps all 37 launcher scripts. **The launcher now states
its seed mode in BOTH branches; silence in the production branch is what let
this ride an entire bake-off.**

**5. A preflight model gate on the canonical runner.** A haunted leg ran the
script pass, the whole voice pass and part of the video pass before
`assert_usable` reported the adapter missing -- the weight was on disk but under
a root the headless model-paths config does not name. The runner now asks the
running server what it can SEE via `/object_info` before submitting. Same
failure now costs 5 seconds instead of 428.

**6. Preflight G1.3** -- a per-artifact constant (byte floor, filename, recipe
receipt) must be a CLASS attribute a sibling can override. Both directions are
pinned: below its own artifact, and within ~15% of it.

**7. Model-refusal degrade** -- a refusing image model no longer kills the
episode; the beat records skip evidence and the completeness gate tolerates it.

**LIVE ARTIFACTS -- eight episodes in `otr/obs/` from this session**, including
the complete cadence ladder on ONE module and ONE style (13 -> 8 -> 5 motion
windows per beat): `magnifying_the_past` (hold-2), `reel_478_extraterrestrial`
(hold-3), `feline_fallen_from_the_stars` (hold-5). Hold-5 rendered in 825s
against hold-2's 1676s.

**MEASURED, so nobody re-litigates it:** the domain adapter does NOT wash out
colour. Five measurements across two styles, never once below its clean
counterpart (archival 2.50 vs 2.07; anime 6.19 and 12.88 vs 5.13). Separately,
the single v2 anime episode measured SATAVG 51.97 against 5-13 for every v3
episode -- a possible real module trait, still n=1, worth one seeded A/B.

**THREE PROBLEM STATEMENTS written for later pickup:**
* `docs/2026-08-22-GHOST-CADENCE-PROBLEM-STATEMENT.md` -- long beats fuse
  THIRTEEN sliding motion windows; motion energy per beat is an uncontrolled
  function of how long the writer's sentence happened to be.
* `docs/2026-08-22-PHASE3-TRANSMISSION-STATE-PROBLEM-STATEMENT.md` -- rebuilt
  around the operator's own idea (drive contamination from the per-line VOICE
  AROUSAL the repo already computes) rather than `arc_phase`. No LLM, no new
  ledger field; still needs one declared `VideoRequest` field.
* `docs/2026-08-22-GHOST-PROMPT-PROBLEM-STATEMENT.md` -- the source the Ghost
  Prompt V2 plan cites. Its finding 1 (Ghost emits a 21-char cue where the style
  pack authors 262 chars, and the STILL lane uses the rich tails) is marked
  WONTFIX for the prompt sprint and remains available afterwards.

### SHIP INTENT -- OPERATOR, 2026-08-22 EVENING (read before touching the Ghost lane)

**`v3_haunted` + Prompt v2.1 IS THE PRESUMPTIVE SHIP CONFIGURATION.** In his
words: *"i think v3_haunted 2.1 will ship and we may ditch the rest we shall
see."* Not a verdict -- an intent -- so the peers are RETAINED until he says
otherwise. Do not delete `animatediff15_v3_video`, the cadence peers or the v2
peer on the strength of this line.

**THE DOMAIN ADAPTER IS WHAT MAKES A STYLE PACK LAND, and it took his eye to
see it.** He compared two anime episodes -- one looks anime, one does not:

    signal_lost_whiskers_in_the_stacks_20260822_150524      v3_HAUNTED  -> anime
    signal_lost_turntables_lament_the_last_spin_20260822_185304  v3 clean -> not

Read straight off the per-beat clip filenames. Every arm rendered on 2026-08-22
except the top-ranked one used the CLEAN engine, because that is what
`otr_ghost_signal_v3` pins -- which is why a whole day of style comparisons was
quietly handicapped.

**TWO AXES, AND THEY ARE ORTHOGONAL. Do not conflate them again:**
* **Subject legibility = the PROMPT.** Proved on an engine-matched same-seed
  A/B (both clean v3): v1 4/4 recognisable, v2 0/4, v2.1 6/8 with people back.
* **Style capture = the ADAPTER.** Proved by the two anime episodes above.

**FRAME RATE / SMOOTHNESS IS PARKED, NOT FORGOTTEN (operator, hard).** He asked
whether the frame rate should drop so it reads smoother and less experimental,
then answered himself: *"maybe not i dont know noty wortyjh chasng now clos eto
release ... 2.1 we can chase fream,earets"*. **So do not touch fps, cadence or
the hold factor before release.** It is a real want for AFTER 2.1, and the
recipes-are-not-on-the-table directive still governs how it is approached.

**STILL HONESTLY UNANSWERED -- his own question, and nobody should pretend
otherwise:** which engine is better SYNCHED to the audio and the beats. He said
plainly he cannot tell, and that clean v3 might synch better if better prompted
while lacking the anime feel. The two episodes he compared differ in engine AND
prompts AND script, so nothing isolates it. Answering it properly wants a
matched pair -- same script, same seeds, haunted vs clean, prompts held
constant. It has NOT been run.

**The dropdown now says which profile to pick** (`SHIP CANDIDATE -- pick this
one` vs `clean, NO adapter (weaker style capture)`), because the default being
non-obvious is exactly what cost a day of confounded comparisons.

## HOW TO READ THIS FILE (lean pass 2026-08-16 evening)

**Start at THE LIVE QUEUE, at the top of this file (2026-08-20).** It is the
only authority on what is OPEN and in what order; the numbered bodies further
down are detail you read when you pick an item up, not a second to-do list.
The older "THE QUEUE, DRIVER-SET 2026-08-17" block below is SUPERSEDED as an
ordering authority -- it still says "G IS NEXT", and G was closed by its own
newer body. Keep it for its rulings, not its order.

The file was audited end to end on 2026-08-16 and it had drifted badly from its
own forward-only rule. What the audit found, so the next reader is not misled:

* **Eleven internal cross-references are already BROKEN** -- they point at
  headings that no longer exist (`section 0`, `0-BIS`, `0-TER`, `0-QUATER`,
  `0A`, `WHAT IS ACTUALLY LEFT`, `NEXT CODING QUEUE`, `STORY LAB RECOVERY
  BASE`, `ON DECK item 5`). If a pointer sends you nowhere, that is why: the
  target was removed and the pointer was not. Do not go hunting.
* **The bulk of the remaining length was DONE narrative inline inside OPEN
  sections** -- receipts, shipped-work paragraphs and superseded framings mixed
  into live items. **THE ARCHIVE SPLIT THIS DEMANDED IS DONE (2026-08-23):**
  `docs/GO_FORWARD_ARCHIVE.md` now holds the closed items VERBATIM and is **not
  read to resume**. It was done the careful way this note asked for -- every
  ruling those items carried was lifted back into the plan BEFORE the block
  moved, because roughly a third of these sections are standing operator rulings
  phrased as "do not re-open", and losing one costs more than the length does.
  The section headed OPEN went from 235 lines to 117 and now contains only work
  that is actually open.
* **Where a heading says something SHIPPED or CLOSED, believe it and move on.**
  The value in those sections is the ruling or the trap attached to them, never
  the receipt.

**Two contradictions the audit found are now resolved in favour of the newer
statement, and both old ones are struck:** the suite receipt (this file now has
ONE, in BASELINES below), and the Lemmy-vs-render-proofs ordering (THE QUEUE
below supersedes the 2026-08-13 ruling's ordering half; that ruling's
non-ordering content still binds).

### THE STORY LAB IS RETIRED -- two guardrails survive it

The lab ([`jbrick2070/ComfyUI-OTR-UpstreamStoryLab`](https://github.com/jbrick2070/ComfyUI-OTR-UpstreamStoryLab),
`main` = `7df7c80`) is READ-ONLY reference. **Do not develop in it. Do not ship
its duplicate workflow, production mirror or bridge into OTR.**

Its two detector files were inventoried and NOT ported, deliberately:
`spoken_text_policy.py` and `ledger_verifiers.py` are both REGEX detectors --
the lab's own header says *"a future fuzzy or model-assisted policy requires a
new policy"* -- so porting them would not solve the generalization problem the
model judge was built for.

**The canonical `scifi_news` episode topology STANDS and is still the
contract:** opening music -> ANNOUNCER introduction -> character drama
with interstitial music only where the script asks -> ANNOUNCER source-backed
real-news summary -> closing music. Both announcer bookends and the opening and
closing music are structural reservations independent of `target_words`. The
opening establishes story, place and time and connects the premise to the news;
the ending summarizes the real news and distinguishes fact from fiction.

### 1. Z-Image generic reference conditioning is CLOSED and OUT

The permanent matched harness is `scripts/otr_zimage_reference_ab.py`; its live
artifacts are under
`output/otr/episodes/zimage_reference_ab_20260820/stills/{off,on}`. Each arm used
the same installed NVFP4 UNET, Qwen FP8 encoder, VAE, prompt, negative, seed 7,
1472x832 canvas and eight-step recipe on a separate fresh server boot. The OFF
arm was clean. The ON `graph.json` structurally proves the exact dual
`ReferenceLatent` chain reaches both sampler conditionings; its separate
`SUCCESS` receipt and fresh output prove that submitted graph executed. The ON
pixel output reproduced the square grid across the walls and clothing. This is
a by-eye pixel verdict; no gridscore number is evidence.

**DECISION:** generic `ReferenceLatent` is not an approved semantic path for the
installed Z-Image Turbo checkpoint and is OUT of production. A node accepting a
graph proves structural compatibility, not training compatibility. The engine
now advertises `accepts_reference_image=False`; `engine_version` is `2` so every
possibly gridded v1 still misses cache. The portrait-derived deterministic seed
remains enabled, so character scenes stamp `portrait_anchor_mode='seed'`, never
blank. **Correction from the 2026-08-21 mixed canonical QA:** that seed prevents
random anchor selection but does not guarantee face/costume identity across
beats; the open visual-identity item is in THE CURRENT STEP. The diagnostic
graph remains only so future weights can be retested against the same
single-variable harness; it is not a production fallback.

## HISTORICAL CODING BACKLOG -- subordinate to the current runway at the top

This 2026-08-04 inventory may still contain real work, but it is not the active
order. Re-ground any item against HEAD only after the current top runway reaches
it. The 2026-08-11 review routing at the top governs; no full arc is implied.

1. **Style/identity campaign, items 1-4 (one campaign, ~1 day).** Highest
   leverage: fixes the credits style line for all six banks uniformly.
   Re-verified: `run_story_brief_reflection` is `_otr_story_brief.py:446`;
   `_build_left` is `video_engine.py:1442`; the treatment line renders as
   padded `Style    :` at `video_engine.py:1762` (an earlier "no longer greps"
   note here was a driver grep miss, corrected by the r2 panel -- BOTH
   consumers are live and both move in the repoint). Sharper finding on
   item 4: `ending_template` is NOT "computed and never read" -- the catalog
   computes it (`_otr_style_catalog.py:906`) AND the composer reads it
   (`_otr_line_composer.py:809-810`); what is missing is the THREAD between
   them -- no call site passes it into a LineRequest. That is a DECISION GATE
   inside the campaign, not a confirmed build step: wire the thread or rip the
   dead ends, decided with the panel at build. Same for the ghost-name fork:
   **default = scrub briefs after cast lock** (conservative -- no unlocked name
   reaches the listener); the operator may overrule to propagate pitch names.
   `style_seed_env` confirmed validator-only (`capability_profiles.py:116`).
   Item 5 (the 120-key `meta` rip) stays a gated block of its own, NOT part of
   this campaign.
2. **The P0 repair rung tells the truth (2 items, ~0.5 day).**
   `repair_literal_source_metadata` (`_otr_scifi_source_repair.py`, called from
   `_otr_scifi_codex.py`): (a) emit a receipt per pruned span/evidence-row/fact
   -- silent pruning violates the plan's own Invariant 3; (b) give the repairer
   the `allowed_source_fields` allowlist or prune per row, so one bad rehome
   stops poisoning the whole artifact. Suite-provable with fixtures. The HTML
   block-join separator stays an OPERATOR decision (coordinate-system change)
   and is NOT part of this chunk.
3. **Script-parse repair: fix the SPEC, then code it (~0.5 day docs + panel,
   then 1-2 days code).** The claim that increments 1-5 are code-ready is
   STALE: `docs/2026-08-03-script-parse-repair-CODE-READY.md` itself says r3
   returned seven must-fixes that invalidate the call/trace design and
   everything after its STATUS block is a draft. The next chunk is the spec
   correction folding those seven in (a kibitz arc IS the vehicle); only then do
   increments 1-5 become codeable.
4. **Passage-lane craft scoring + the stichomythia floor (~1 day).**
   `_otr_passage_selector.py` has no scoring functions -- the Fable craft
   criteria (French-scene boundaries, entrance/exit starts, couplet ends,
   continuation-word penalties) are all unbuilt. Score, keep top-K, seeded hash
   within the class. Same chunk: the per-beat word floor excludes stichomythia,
   so a merge rule or floor exemption in `_otr_episode_budget`. Pure Python +
   tests. **Sequencing (r1): runs AFTER the ON DECK sprint lands and re-grounds
   against whatever item 1 changed** -- both touch the fidelity selection
   surfaces.
5. **Cast-list parser: the two weak plays (~0.5 day).** Midsummer 1/12 and
   Comedy of Errors 1/7 gendered -- mechanicals/servants in shapes
   `_otr_character_roster.py` does not read yet. Vendored texts are on disk;
   offline; suite-provable against the sidecars.
6. **Small-items batch (~0.5 day, ONE campaign over the batch, one commit each):**
   * `OTRImageGenDispatcher` (`otr_image_gen_dispatcher.py:1412`) has no
     `IS_CHANGED` while depending on external file existence -- confirmed by
     grep, none in the file. Decide the CONTRACT before coding (r2): either
     fingerprint EVERY actual external dependency or deliberately force
     re-runs for this side-effectful node -- a partial path fingerprint still
     serves stale results.
   * Rotated server logs have no retention policy.
   * The `provider_side` three-part-rule regression (picked AND forced
     `cloud_kling_avatar`).
   * The shared `row_is_active(...)` evaluator over captured state -- confirmed
     absent from the tree -- closing the four env-read sites named in OPEN BUGS.

---

## Sprint items 2 and 3 -- closed on re-grounding, 2026-08-23

Neither was done by the window that archived them; both were already true at
HEAD and the plan had not caught up. Verbatim as they stood:

### 2. THE NON-COMMERCIAL NOTICE REACHES NO HUMAN SURFACE (~30 min)

Fully scoped, ledger-clean, and the smallest real win on the board.
`nodes/OTR_LedgerScriptWriter.py:3590` stamps `meta["noncommercial_notice"]` (via
`_otr_provenance.noncommercial_notice`, `:124`) and logs it. **Nothing renders
it.** `nodes/otr_credits_roll.py:516` reads only `credits_source_line`.

Add a sibling printed-credits item beside that block -- `:516-518` is the exact
three-line shape to copy -- plus an integration test. The ledger field already
exists and already has an owner, so this adds a CONSUMER, not a field: no
ownership question to answer. Fires on Folger sources.

**Acceptance (r1 + r2 + r3 + r4):** `meta.noncommercial_notice` present -> ONE
rendered credits item; absent when empty, exact text, exactly ONCE. The
existing source item renders as `>> SOURCE: ...` (`otr_credits_roll.py:510-518`)
-- state the notice's literal prefix the same way, and the notice renders even
when a malformed legacy ledger lacks `credits_source_line`; ADJACENCY (source
line immediately followed by the notice, each its own `intercept` entry)
applies when both exist. No new wrapping helper: the existing intercept renderer already
measures and wraps every entry through `_wrap`
(`otr_credits_roll.py:1131-1135`). Test the ORDERED flow list (do not convert
`col3_flow` to a dict -- duplicate `"intercept"` keys collapse). Integration
fixture proves the Folger wording survives flow construction unchanged.
Legibility on canvas is eyeballed on the next permitted render, not claimed
from the test.

### 3. THE TEST-ORDERING POLLUTION (~30 min)

`tests/test_public_domain_sources.py` pollutes
`tests/test_public_domain_interpreter.py::test_empty_cast_is_rejected_and_retried_to_failure`.
Confirmed 2026-08-04: fails when the two run adjacently, passes **11/11** when
the interpreter file runs alone, invisible in full-suite order. Pre-existing --
already proven by stashing and reproducing at the prior commit.

Worth the half hour because it costs a real signal: any targeted run touching
those two files reports a red line that has to be re-diagnosed as benign every
time. **Build shape (r2 -- the r1 "cleanup fixture" idea was WRONG and is
withdrawn):** the mechanism is MODULE-IDENTITY breakage, not leaked state.
`test_module_import_is_lazy` (`tests/test_public_domain_sources.py:223-233`)
calls `importlib.reload(pd)` twice, which REPLACES the module's class objects,
while the interpreter test file imported exception classes at collection time
-- so `except OldClass` no longer matches instances raised by the reloaded
module. No cleanup fixture can restore class identity. Fix (r3-refined): run
the lazy-import assertion in a SUBPROCESS -- `sys.executable`, repo-root
`cwd`, `check=True`, fresh import with the read guard installed -- and pin
BOTH test-order permutations as regressions. The private-module-name
alternative is CUT: it risks exercising fallback import paths instead of the
production `nodes._otr_public_domain_sources` package identity.

## Sprint item 5 -- closed 2026-08-23 (the commercial-clean JOIN)

Verbatim row as it stood before the fix, kept because its reasoning about WHICH
layer owns which licence is the half that still teaches:

> ### 5. 1,090 CAST ROWS CLAIM A NON-COMMERCIAL MODEL IS COMMERCIALLY CLEAN
>
> `eng_indextts2.py:55` says `commercial_clean = False` (bilibili non-commercial);
> all 40 bank rows say `true`; `cast_lock.py` trusts the bank row. The row flag is
> the CLIP's licence and the engine flag is the MODEL's -- genuinely different
> facts, both already in the right layers. **Stamp the JOIN. Do NOT edit the 40
> bank rows** (`otr_dl_indextts2_refs.py:11-17` documents them as clip provenance;
> the ingest mints three rows across three engines from one PD clip).
>
> **Must heal ATOMICALLY or it creates the defect it fixes:** the stamp
> (`cast_lock.py:742`), the `gated` counter (`:575/:614/:661/:670`) AND the three
> report strings (`:578/:618/:673`) -- otherwise the report prints `clean=True`
> beside a ledger saying `False`. Resolve ONE profile by `(role, engine)` --
> role-scoped, never engine-name-scoped. **Enforcement stays OFF.**
> Prospective-only for the 1,090 frozen ledgers.

The atomicity warning was the useful part and it was honoured structurally
rather than by care: at both report sites the verdict is computed once into a
local that feeds the counter AND the string, and an AST ratchet refuses any new
direct read of the clip flag. See the plan's closed stub for what shipped.
