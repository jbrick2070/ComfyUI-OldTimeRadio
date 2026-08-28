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

## Sprint item 1 -- DEFERRED by operator ruling 2026-08-23 (public_domain source grounding)

Retired from the go-forward on the operator's instruction: *"yes you can retire
the public domain from the go forward, mark it as deferred."* The ruling that
killed its premise, in his words: *"public domain does not need to get author's
words unless they are dialogue, it can paraphrase"* and *"please do not chase as
long as it carries the story and some dialogue if present."*

Read the plan's DEFERRED stub first -- it states the only scope that would ever
be sanctioned (source DIALOGUE, never a prose window) and records that the
coordinate-system infrastructure the text below asks for was already built.

Verbatim, as it stood:

> ### 1. THE PUBLIC_DOMAIN LANE IS TOLD TO CARRY WORDS IT IS NEVER SHOWN (the session's main work -- r1 panel re-scoped 2026-08-04)
>
> The headline defect, and the one that manufactured "Arkham, Massachusetts" over
> H. G. Wells. The pack orders the model to carry the author's language:
>
> * `nodes/story_packs/public_domain/faithful_radio_adaptation.json:13`
>   (`exchange_system`) -- "Where the source gives these characters words, CARRY
>   THEM. Keep their diction, their rhythm, their argument."
>
> And `nodes/_otr_compose_exchange.py` (994 lines) has **ZERO** references to
> `source_text`, `full_text`, `source_meta` or `excerpt` -- verified by grep.
> **The instruction is bound to an absent document.** A model told to carry words
> it cannot see will invent words and believe it complied.
>
> **SCOPE RULING (r1, grounded against `docs/2026-08-03-fidelity-pass-ownership.md`
> line 25): this item is PUBLIC_DOMAIN ONLY.** The ownership table rules
> `exchange_compose` **NOT RUN** on the Shakespeare verbatim lane ("It exists to
> author dialogue. There is no dialogue to author."), so enhancing the composer
> for Shakespeare invests in a pass the verbatim executor removes. Shakespeare
> gets exactly ONE change this sprint: its dangling comma (item 3). The keystone
> "compile source speech, do not generate it" (THE ADAPTATION DESIGN) binds the
> VERBATIM lane; `public_domain` is the operator-ruled FUZZY PROSE lane, where
> grounding the generative composer is the correct move, not a contradiction.
>
> **PREMISE CORRECTION 2026-08-23 (re-grounded at HEAD before any code, on the
> operator's "make sure it's not already done or invalid" instruction). THE DEFECT
> IS REAL AND UNFIXED -- `nodes/_otr_compose_exchange.py` still has ZERO
> references to `source_text` / `full_text` / `source_meta` / `excerpt` /
> `canonical_body` / `source_window`, so the pack's CARRY-THEM instruction is
> still bound to an absent document. BUT LEG (a)'s COORDINATE SYSTEM IS ALREADY
> BUILT, and the text below describing it as work to do is stale:**
> * `nodes/_otr_source_document.py` EXISTS and carries `SourceSpan` with half-open
>   Unicode `start_char`/`end_char` offsets plus a `canonical_body`.
> * `nodes/_otr_public_domain_sources.py` already separates the UNCAPPED
>   `normalize_public_domain_body` from `_project_to_payload_window(body,
>   max_chars=12000)` -- the exact refactor r3 asked for -- and exposes
>   `source_document_from_text` / `build_source_document`.
> * The document ALREADY REACHES THE WRITER: `OTR_LedgerScriptWriter.py:3718`
>   does `resolved.get("source_document")` -- but uses `_sd.canonical_body` ONLY
>   to derive a sound world for the style contract. The composer is never passed
>   any of it.
> * Provenance sidecars went from 1 to 95 on disk, so the hash-discipline
>   paragraph below is also measuring a corpus that has moved.
> **So what remains is the SELECTOR plus the CONSUMER wiring, not the coordinate
> system.** The item is much cheaper than it reads. It is also a PROMPT change,
> which is the one thing standing ruling 2a says to prove on an artifact first,
> and it sits next to the operator's "story quality is done" directive -- the
> fidelity reading (a Wells adaptation inventing "Arkham, Massachusetts" is a
> correctness fault on a fidelity lane) is why it is still listed as open rather
> than struck.
>
> Three legs, ALL required -- the panel killed the raw-injection shape:
>
> **(a) A BOUNDED source window over the COMPLETE canonical body -- never the
> payload's `full_text`, which is itself truncated.** r2 correction of this
> plan's own premise: `canonicalize_public_domain_text(..., max_chars=12000)`
> (`_otr_public_domain_sources.py:337-343`) truncates at 12,000 CHARS, and
> `payload_from_manifest_unit` stores THAT as `full_text` -- while the corpus
> runs **916 words (`cradle_protocol`) to 25,200 words (`beckoning_fair_one`)**
> across 65 units. So "the material already arrives, it needs passing" is false
> for large sources: the payload carries a prefix. The selector reads the
> complete canonical body from the SOURCE layer, separated from the interpreter
> excerpt. Hash discipline: exactly ONE of 65 units ships a provenance sidecar
> (`time_machine__arrival.provenance.json`), and its `body_sha256` covers
> normalized RAW bytes, not the canonicalized body -- two NON-interchangeable
> fields. Derive a `canonical_body_sha256` at fetch/selection time, bind
> selection + receipts to it, and do NOT call it authenticated provenance. Do
> NOT migrate the 65 closed manifests for it (`_SOURCE_KEYS`/`_UNIT_KEYS` closed
> at `:48-63`); carry it in `source_meta` and snapshots. Coordinate system (r3):
> refactor `canonicalize_public_domain_text` into an UNCAPPED normalization
> owner plus a separate 12,000-char legacy payload projection; spans are
> half-open Unicode char offsets (`start_char`/`end_char`) into the uncapped
> string; `canonical_body_sha256 = sha256(canonical_body.encode("utf-8"))`;
> stamp normalization + selector versions. Transport (r3): `SourceFetchResult`
> exposes only payload/source_meta/source_rights and `_resolve_inputs` collapses
> to a three-tuple, and the snapshot envelope is the SEVEN-KEY payload
> (`_otr_source_snapshot.py:48-50`) whose `full_text` is the truncated prefix --
> so extend the PUBLIC-DOMAIN snapshot with the CANONICAL BODY as the SOLE
> replay authority (r4 cut the "or exact selected text" alternative -- selected
> text cannot recreate pre-outline grounding or select windows for a NEWLY
> generated outline), under a versioned body/hash/normalization contract. A
> legacy seven-key snapshot FAILS with a typed grounding-version error -- but
> ONLY when the snapshot's bank is `public_domain`/adaptation (r4, both lanes
> converged): the seven-key envelope is the UNIVERSAL loader, and an
> unconditional rejection would break every other bank's existing snapshots and
> bake-off replays. Keep the full document OUT of meta/ledger (`source_meta` is
> copied into durable metadata at `:3548`). Budget: capacity
> is EVERY backend, not GGUF alone -- the fitting seam
> (`_otr_generation_budget.py:132`) spans GGUF (`estimate_prompt_tokens`,
> estimator, `_otr_gguf_backend.py:1264-1273`), OpenRouter, Google and Comfy --
> so select the window against the COMPLETE assembled message (system seam,
> cast, prior lines, contracts, source block, output reservation), reserve
> conservatively with stated margin, and refuse `prompt_no_room`
> deterministically BEFORE provider execution; receipts distinguish
> estimated_prompt_tokens / requested_output / context cap / margin / estimator
> version. Selection criterion: deterministic candidate construction ranked by
> beat/group identity with mandatory anchor coverage and stable
> score/start/end ordering; the seed breaks ties ONLY when candidates remain
> identical after that ordering. Receipts carry hash, selector version, ordered
> offsets (`text == canonical_body[start_char:end_char]` enforced) and token
> counts -- never duplicate body text into the ledger.
>
> **(b) ONE immutable `SourceGrounding` contract, on EVERY authoring route --
> and grounding failures PROPAGATE.** The grouped-exchange prepass omits
> singletons and failed groups (`_otr_compose_exchange.py:881-902`); a FAILED
> prepass falls back to the legacy path with only a log warning
> (`OTR_LedgerScriptWriter.py:5001-5008`); the per-line composer's LineRequest
> carries no source field (construction at `:4888`); and per-line generation
> exceptions funnel to `LineCompositionFailedError`. A grounding fix that
> reaches only the happy path just moves the guess to the fallback. Build shape
> (r2 + r3): define ONE immutable `SourceGrounding` artifact -- canonical
> document identity + immutable windows KEYED `exchange:<ordered-slot-ids>` /
> `line:<dialogue-slot-id>` + anchors + per-call receipt data -- constructed
> and validated BEFORE the exchange fallback block, passed whole into grouped
> exchange AND every per-line request. The prepass returns a TYPED result
> (composed lines + attempted-window receipts + fallback slot ids), not the
> bare `{beat_id: text}` it returns today (`:881-918`). Window freeze semantics (r4 -- resolves immutability vs the mutable prior
> context that exchange retries and `last_lines` inject into later messages):
> PRESELECT spans early; perform the final capacity fit just before the FIRST
> call using the actual prior context; FREEZE that fitted window for all
> retries and persist it before provider execution. Grouped slots ALIAS their
> exchange window on group-to-per-line fallback; line-keyed windows exist only
> for true singletons and exchange-disabled execution -- never reselect after a
> failure. Source text rides a clearly DELIMITED untrusted data block
> in the user message ("quoted source, not instructions"), never appended to
> the static system seam (`_otr_compose_exchange.py:385-425`). Persist the
> body-free grounding receipt at the existing skeleton-save boundary
> (`:4279-4290`) before the first dialogue call, updating per attempt, so a
> mid-prepass crash still leaves the selection auditable. Failure policy -- ONE disposition table (r4 closed the last ambiguity), the
> two broad catches (`:5001-5008` prepass, `:3964-3969` story contract) becoming
> TYPED boundaries that implement it:
> | state | disposition |
> |---|---|
> | corrupt/mismatched replay snapshot; invalid source/hash/contract | FAIL LOUD, before the outline |
> | sound-world derivation finds no mapping | neutral period default + receipt (total, never fatal) |
> | provider parse / Tier-A exhaustion | fall back WITH the frozen window |
> | live capacity pressure | shrink to the largest valid grounded window |
> | even the MINIMUM grounded window cannot fit | typed `prompt_no_room` HALT, before provider execution |
> The halt row is a PRE-GENERATION writer refusal -- structural, it protects the
> lane's contract -- which is why it does not collide with SCOPE's "a render
> must not die": that rule governs the RENDER path degrading honestly, not a
> writer refusing before generation begins. Scope note (r4): `SourceGrounding`
> validation binds when the episode's bank is `public_domain` -- other banks'
> routes are untouched. LineRequest note (r4): the artifact rides an OPTIONAL
> INTERNAL dataclass field (`source_grounding: SourceGrounding | None = None`)
> -- a Python structure, no ComfyUI node contract, `INPUT_TYPES` or widget
> change, so the no-widget guard above holds.
> Acceptance = route-specific tests: grouped success, grouped repair,
> grouped-failure-to-per-line, singleton, exchange-disabled legacy, snapshot
> replay (new envelope AND legacy-envelope typed refusal, public_domain-scoped),
> hash mismatch, exact-capacity rejection -- plus a corpus-wide property test
> over all 65 units proving normalization idempotence, canonical-hash stability
> and `text == body[start_char:end_char]` for every emitted span (r4). Version
> discipline (r4): the existing constants are `PROMPT_VERSION =
> "public_domain_interpreter_v2"` / `SCHEMA_VERSION = "public_domain_briefs_v1"`
> (`_otr_public_domain_sources.py:36-38`); name and bump every changed one, and
> give SourceDocument / SourceOverview / SourceGrounding / normalization /
> selector / snapshot their own explicit versions.
>
> **(c) World anchors, DERIVED FIRST -- and the sound world gets ONE owner that
> feeds every surface.** Prefer deriving a typed grounding sidecar from EXISTING
> metadata + the selected spans. New manifest fields are a LAST resort:
> `_SOURCE_KEYS`/`_UNIT_KEYS` are closed frozensets
> (`_otr_public_domain_sources.py:48-63`, same for `_SCENE_KEYS`), so new fields
> mean a schema version + migration across all 65 units. AND the competing frame
> must actually be disabled, not outvoted: the adaptation `sound_world` is a
> content-blind draw (`OTR_LedgerScriptWriter.py:3962`, palettes at
> `_otr_style_catalog.py:442-463` -- grate/mantel/teacup over whatever source
> rolled it). r2 sharpened the shape: the catalog renders the drawn sound world
> into `contract.grammar` SEPARATELY from the `contract.sound_world` stamp and
> the canon derivation, so a stamp-only fix leaves the prompt grammar still
> carrying the contradictory palette. ONE source-aware derivation function must
> feed the stamp, the grammar and canon for `style_pool_class == "adaptation"`
> (arc_shape gate at `:4325` is the shipped precedent), with an explicit neutral
> period default when no mapping exists -- and it runs BEFORE the grammar is
> built (or the grammar re-renders from the final contract), or the prompt
> grammar keeps the contradictory palette while the stamp looks fixed (r3, both
> lanes independently). DECIDE whether derivation failure is fatal: today's
> broad catch silently disables the whole story contract. Reconcile with the
> EXISTING anchors owner: `meta["specificity_anchors"]`
> (`OTR_LedgerScriptWriter.py:4259-4266`) already derives and injects an anchor
> projection -- the new source anchors REPLACE it or deterministically merge
> into it, never run beside it as a second independent voice. Do NOT delete the
> adaptation styles -- operator-authored 2026-07-14; fix the DRAW and the
> plumbing, not the styles.
>
> **Two receipts, named now so neither is overstated later:**
> * `code-complete + suite-green` -- the most a session without the live leg can claim.
> * `production-qualified` -- only after a canonical `public_domain` leg passes a
>   rubric: no unsupported foreign place/character/object; the source's setting
>   and principal event retained; provenance receipt complete; `obs_publish OK`;
>   asset on disk.
>
> **Two rules from the 08-03 craft brief, both hard-won, both easy to violate:**
> 1. **Never name the feared failure.** Writing "no Arkham" into a prompt IMPLANTS
>    Arkham. Forbid by CATEGORY, never by example.
> 2. **Every fidelity instruction must be PAIRED with the material it binds to.**
>    An unpaired "carry the words" is the bug, not the fix.
>
> **Size honesty (r1) and CHUNK ORDER (r3 -- the naive order was CYCLIC):** this
> is THE SESSION, not 90 minutes. r3 caught a dependency cycle in the obvious
> build order: the sound world feeds `contract.grammar`, the grammar is consumed
> by the OUTLINE (`OTR_LedgerScriptWriter.py:3948-3963` -> `:4129`), and beats do
> not exist until the outline returns -- so a sound world derived from
> beat-keyed windows is impossible. Build in THIS order, one green pushed chunk
> each:
> **CHUNKS 1 AND 2 ARE DONE AND PROVEN ON RENDERS. Chunk 3 (the grounding supply line) is PARKED under the story-quality directive -- the Source-grounding note in section 3 above is authoritative; a contributor may pick it up.**
>
> **Carried into chunk 3 from the chunk-2 QA (do not lose):** snapshot replay
> has no whole-body carrier, so an adaptation lane replaying a frozen source
> falls back to the drawn palette and a live run and its replay produce
> different sound worlds. The tempting fix -- rebuild the document from the
> snapshot's `full_text` -- is WRONG and was rejected: that field is the
> truncated projection, so it would mint a document whose total-coverage
> guarantee describes a prefix. The correct fix is the snapshot-envelope
> extension already specified in 1(a) below.
>
> 1. **Uncapped `SourceDocument` + a pre-outline `SourceOverview`** (r4): split
>    the normalization owner, then derive deterministic COVERING windows with
>    exact-span evidence for cast, setting, principal turns and ending. This is
>    what grounds the PRE-OUTLINE authors -- the interpreter today reads the
>    CAPPED payload (`_otr_public_domain_sources.py:520-543`, running at
>    `OTR_LedgerScriptWriter.py:3748-3757`) before contract (`:3948`) and
>    outline (`:4129`); beat-keyed grounding alone arrives too late for them.
>    Transport (r4): ONE transient typed field --
>    `SourceFetchResult.source_document` -> typed normalized result ->
>    `resolved["source_document"]` -- MECHANICALLY excluded from meta/ledger
>    serialization; snapshot replay reconstructs the same type.
> 2. **Contract / grammar / outline from the overview's document-level
>    anchors**: the one derivation function runs BEFORE grammar build (or
>    grammar re-renders from the final contract), feeding stamp + grammar +
>    canon. Pre-outline derivation uses DOCUMENT-level anchors only -- selected
>    spans do not exist yet (r4 wording fix).
> 3. **Beat-keyed window selector + `SourceGrounding` threading + typed failure
>    boundaries** (post-outline, when beats exist). The route matrix must NAME
>    the announcer routes -- intro / rewrite / outro authoring at
>    `OTR_LedgerScriptWriter.py:5104-5116`, `:5272-5285`, `:5357-5409`
>    (verify-at-build) -- and decide per route: grounded, or constrained to
>    already-grounded accepted fields.
> No node signature, widget, link or schema change is intended anywhere in this
> item -- the canonical JSON stays byte-identical through the sprint; if any
> chunk turns out to need an INPUT_TYPES change, section-0 same-commit rules
> apply and the plan must say so first. The bench items were conditional filler
> and are now unreachable; that is fine.
>
> **Ceiling to be honest about:** this can be built and unit-tested here, but its
> real proof is a render. Renders HAVE RESUMED (2026-08-05), so the
> `production-qualified` leg is runnable whenever a render window is free; until
> it runs, claim only `code-complete + suite-green`.


---

# ARCHIVED 2026-08-24 -- pruned from the go-forward plan

Moved on the operator's instruction to make the plan *"accurate, truly go forward items"*. Verbatim, nothing edited.
**This file is not read to resume.**


## Ghost ship intent -- superseded 2026-08-23

### GHOST SHIP INTENT -- SUPERSEDED 2026-08-23 (the section it replaces is in the archive)

The 2026-08-22 SHIP INTENT block said *"the peers are RETAINED until he says
otherwise. Do not delete `animatediff15_v3_video`, the cadence peers or the v2
peer on the strength of this line."* **He said otherwise on 2026-08-23** --
*"delete any animatediff that are not haunted"* -- and they are gone
(`187380d0`). The block is archived rather than edited because a superseded
instruction that still reads as current is the most dangerous kind of stale
note: a future window would have followed it and re-created the six lanes.

**What survives from it:** `v3_haunted` + Prompt v2.1 IS the ship
configuration. That was an intent then and it is the state now.



## PBUG-20260824-01 original problem statement (Class A now fixed)

### >>> THE ORIGINAL PROBLEM STATEMENT (2026-08-24) -- superseded in part, kept for its evidence <<<

**Operator: *"lets mark scifi news writer error as our next plan ... we need to
help fix scifi_news_pro."*** This is the top of the queue.

**THE MEASUREMENT, and it is the reason this is now a priority rather than a
shrug.** The overnight writer-gate loop (2026-08-24 00:36-08:22 PDT, 10 full
passes over all five banks, `scripts/otr_overnight_loop.sh`, log at
`tmp/otr_overnight_loop.log`) put every bank through the canonical workflow ten
times. Four banks were near-perfect. `scifi_news_pro` **FAILED 6 of 10 passes**
-- 60%. Every other bank's failures over the same window: zero, except the
single shakespeare failure that was root-caused and fixed the same night
(PBUG-20260802-02 third manifestation). This lane is the outlier by an enormous
margin, and it is the ONE lane that is DISPATCHED rather than inline
(`nodes/_otr_lane_specs.py` -- `scifi_news_pro_multipass` is the only entry in
LANE_SPECS; everything else runs Section I).

**PASS ORDER, verified in `run_scifi_news_pro_episode`:** `_pass_treatment`
(:3607) -> `_pass_news_read` (:3621) -> `_pass_script` (:3636). **The failure
DURATION identifies which pass died**, which is how the two classes below were
separated without needing every leg log:

**CLASS A -- the script/markup pass (~4 min to fail, 3 confirmed occurrences).**
`UNKNOWN_SPEAKER` plus `SKELETON_BREAK`. The model emits speakers that are not
in the locked cast and structure the skeleton forbids. Real captures:
* `UNKNOWN_SPEAKER: DR. LEE` x3 + `SKELETON_BREAK: character line (DR. LEE)
  after the last scene`
* `UNKNOWN_SPEAKER: **ANNOUNCER` -- **markdown bold leaking into the speaker
  token** -- plus `SKELETON_BREAK: character line (**ANNOUNCER) before SCENE 1`
  and `SKELETON_BREAK: announcer intro missing`, and
  `UNKNOWN_SPEAKER: DR. RAPHAEL ZUFFERERY`
* `UNKNOWN_SPEAKER: THOR`, `UNKNOWN_SPEAKER: LUCAS`, `SKELETON_BREAK` on both
  plus `Dr. Schmidt` after the last scene
The `**ANNOUNCER` capture is the most actionable single clue in this entry: a
speaker the parser SHOULD recognize, rejected only because the model wrapped it
in markdown. That is a transport/normalization gap, not a story problem, and it
is cheap to test.

**CLASS B -- the news_read pass (~1.5-2.9 min to fail, 1 fully captured).**
`NewsProTreatmentError` from `_pass_news_read` (`nodes/_otr_scifi_news_pro.py:1802`),
after **2 attempts**: *"the closing read is a FACTUAL report and it names
invented characters (Laura Goodkind). Report only what the source says, using
the source's own names."* The validator
(`_make_news_read_validator`, :1748-1755) is CORRECT -- a factual news close
must not cite the drama's fictional cast -- but the pass is being asked to
write a factual read while the fictional cast names sit in its own prompt
(`_pass_news_read` builds `FICTIONAL CAST NAMES (never use these in the factual
read): ...`). Telling a small model "never say X" while showing it X is a known
weak instruction shape. Worth checking whether the ladder's 2 attempts is the
real budget and whether the repair prompt actually names the offending token.

**WHAT IS NOT KNOWN, stated honestly:** passes 7 and 8 failed with the bare
label `WRITER` and no captured reason. Their durations (2.2 and 1.5 min) put
them in Class B's profile, but that is INFERENCE, not evidence. The reason is
lost because **`tmp/_bankgate_<bank>.log` is overwritten by every pass** -- a
real harness gap this entry surfaces as a side finding: the overnight loop
destroys the evidence for every failure except the most recent one per bank.
Anyone taking this item should fix that FIRST (append, or stamp the pass number
into the filename), or they will be re-running the loop to recover data that
was already collected once.

**SCOPE QUESTION FOR WHOEVER TAKES IT -- decide before coding.** Two classes,
two mechanisms, one lane. They are NOT obviously one fix, and the last time two
`scifi_news_pro`-adjacent symptoms were filed as "one fault, two doors"
(PBUG-20260802-02's original entry) that framing was wrong and had to be
corrected the same day. Treat A and B as separate until proven otherwise.

**DO NOT let this become story-quality work.** The operator's 2026-08-04
directive stands: scripts are ACCEPTED as they are. This item is about a lane
that REFUSES TO PRODUCE AN EPISODE 60% of the time -- a structural/renderability
defect, explicitly inside the "any structural or ledger fault" carve-out. The
fix is to make the lane produce a valid ledger, not to make its prose better.


**THE CLOSED ITEMS MOVED OUT ON 2026-08-23.** Ghost Prompt V2, item A (Ghost
Signal), A-ORIGINAL, B, C and E's closed half were receipts of SHIPPED work
sitting inside a section headed OPEN -- exactly what the 2026-08-16 self-audit
flagged and did not dare fix blind. They are VERBATIM in
`docs/GO_FORWARD_ARCHIVE.md`, which is **not read to resume**. What follows is
what is actually open, plus every ruling those items carried, because the
rulings are the half that still binds.

**RULINGS LIFTED OUT OF THE ARCHIVED ITEMS -- STILL IN FORCE:**

* **THE GHOST SIGNAL LOOK IS ACCEPTED. DO NOT CHASE IT.** The motion reads fast
  (12.5 fps of AnimateDiff held to 25). Operator, watching the published
  episode: *"i was expecting experimental vj"* and *"its perfect"*. Not a defect.
* **NO VRAM CLAIM ON THE GHOST LANE.** Admission stays unenforced and the lane
  may OOM. A single 5872 MiB / 100% reading was observed and is NOT a qualified
  cost row.
* **A REDUCTION COMES BACK A COMPLETE THOUGHT; AN AUTHORED LINE COMES BACK
  BYTE-IDENTICAL.** `_still_word_fit_card` takes whole comma clauses before any
  word cut. Scoped to REDUCTIONS on purpose: a line that ends on "the" because
  the AUTHOR wrote it that way is the author's, and THE LAW forbids rewriting a
  story for style. Both halves are pinned. The rule lives once, in
  `nodes/_otr_shared/text_tails.py`.
* **GREP THE CODE BEFORE RE-PLANNING A CARRIED ITEM.** Half of item B was
  already fixed before the window that "opened" it looked -- a refusing engine
  had recorded `reason="model_refusal"` skip evidence and the completeness gate
  already tolerated exactly that reason. The decision the item asked for had
  been made, in favour of DEGRADE.
* **G3.7:** `_portrait_free_roles_from_policy` is INERT for a lane with an EMPTY
  `still_plan` -- it looks for a portrait row saying "never". A no-still lane is
  covered by the stronger `accepts_still = False` gate at the image dispatcher.
  Adding a portrait/never row to light it up would be a declaration the lane
  cannot honour.
* **AND ONE HONEST LINE KEPT ON THE RECORD:** Ghost Signal shipped with its
  independent finished-diff review seat UNFILLED -- the Agent tool was disabled
  and the substituted Codex lane thrashed and was killed. The live leg is
  stronger evidence and caught the real blocker, but it is not a review, and the
  plan said so rather than implying coverage.

**QUEUE STATE, updated 2026-08-24 by the OVERNIGHT coder window (state only;
plan authorship stays with the planner window):**
* **SHAKESPEARE IS FIXED AND PROVEN LIVE (`8ca3f13a`).** PBUG-20260802-02's
  third manifestation -- a locked cast member (MARIA) served zero dialogue
  under a tight beat budget, refused by the freeze gate. Root cause:
  `_otr_outline._phase_check` validates cast membership one-directionally
  (invented = used - locked) and never checks starvation. Fixed with
  `nodes/_otr_cast_coverage_repair.py`, a repair pass in the writer's own tail
  BEFORE the freeze cascade -- the gate stays refuse-only per the standing
  "gates refuse, producers repair" convention. 18 new tests; suite
  **12053 / 120 skipped / 1 xfailed**, zero regressions. Bible **12.131**.
  **Live proof: shakespeare PASSED in both of the last two full overnight
  passes** (different random scene draws each time).
* **THE OVERNIGHT LOOP RAN 10 FULL PASSES** (00:36-08:22 PDT,
  `scripts/otr_overnight_loop.sh`, log `tmp/otr_overnight_loop.log`), five banks
  per pass, and published **43 episodes to `otr/obs` (121 -> 164)**. Stopped on
  operator instruction after pass 10 with a success in hand; pass 11's
  media_archive leg was interrupted mid-flight and correctly reported itself as
  INTERRUPTED rather than as a node exception.
* **THAT LOOP IS WHAT PRODUCED THE NEW TOP ITEM.** `scifi_news_pro` failed
  **6 of 10 passes (60%)** while every other bank failed zero. Recorded as
  **PBUG-20260824-01** (`452132d0`) with both failure classes and their real
  captured errors; Class A's markdown-leak half promoted as Bible **12.132**.
  See the NEXT ITEM block at the top of this file.
* **A HARNESS GAP WAS FIXED IN PASSING (`f84906c8`).** `tmp/_bankgate_<bank>.log`
  was overwritten every pass, which is why two of the six `scifi_news_pro`
  failure reasons are lost. The loop now archives leg logs to
  `tmp/legs/passNNN/`, so the next diagnosis needs ONE loop, not another night.
* **REGISTRY (2026-08-24, no code change):** node extraction is import-based and
  runs in Comfy-Org's own Cloud Build; our pack loads 25/25 nodes under a
  faithful local reproduction of that container, so the empty node panel is NOT
  ours. Their automatic backfill scheduler is provisioned `paused = true` with a
  leap-day cron, so it does not self-resolve. Problem statement ready to send at
  `docs/2026-08-24-comfy-registry-problem-statement.md`; `2.0.0-alpha.7` was
  published and sits at `NodeVersionStatusPending` (their queue, not a failure).

**PRIOR QUEUE STATE, 2026-08-23 by the DAYLIGHT coder window:**
* **THE LEAN-MEAN CAMPAIGN IS EFFECTIVELY COMPLETE.** Orders 1-6 closed
  overnight, 7 cancelled by ruling. This window closed **8** (the shared
  ffprobe boundary + eleven callers, `00ac7df8` / `8dd9f2cf`), **9** (the
  writer split, `8182b38c` / `d99c1adc` -- 1,927 lines out, 7,418 -> 5,490,
  byte-identical with a sha256 per block), and **11** (the `scripts/` owner
  table, `e541db1d`). Truth table:
  `docs/2026-08-23-lean-mean-progress.md`.
* **ORDER 10 REMAINS HELD and a coder should not unhold it.** Workflow-atomic,
  and a blanket variant regeneration still reverts the operator's hand-edited
  `ghost_signal_v3` ship-candidate settings. **Order 12** stays blocked on
  `otr_cloud_lanes` ratification.
* **A LIVE BUG THE SEAM FOUND (`0acdc993`).** `_run_writer_tail` read
  `_style_roll` -- a local of `run()`, a SIBLING method -- and called
  `random.Random` in a module that never imported `random`. Both on the
  dynamic-style FLOOR FALLBACK branch, so a failed style reflection raised
  NameError instead of degrading to a floor style. It survived because the
  invariant was tested with `co_freevars == ()`, which **cannot fail** on a
  sibling's local (that compiles to LOAD_GLOBAL). The replacement walks the
  bytecode for global loads the module does not define, with a negative-control
  test. NOT filed as a PBUG: the proof is a disassembly, not a live artifact.
* **OPERATOR RULINGS 2026-08-23, executed (`cc36c64b`).** *"Rip it out"* --
  `otr_hazard.py` gone. *"Soak op delete, we'll make a new soak op"* --
  `soak_operator.py` (a 304-line legacy shim since BUG-LOCAL-002 gutted it in
  May) and its orphaned `watcher_overrides.json` gone; the shim's one live
  function moved byte-identically to `scripts/treatment_scanner.py`, the
  address the shim's own docstring had always named.
* **THE LIVE RENDER IS PAID. `otr/obs` went 116 -> 117 on 2026-08-23.**
  `signal_lost_shadows_of_the_vault_20260823_223635_silent_procgen_blended_captioned_with_credits_final.mp4`,
  `RESULT SUCCESS`, via `scripts/otr_writer_bank_gate.py --acts 1` on the
  canonical workflow (profile `otr_w45_still_flat`, bank `media_archive`). The
  gate then continued to bank `original` on its own. **The all-banks sweep is
  RUNNING, not unrun** -- read `tmp/_bankgate_<bank>.log` for where it got to,
  and the obs count for what it published. One caution learned here: the gate
  launched as a background job reports its PARENT shell exiting while the render
  child keeps working, so a bare exit code is not the leg's verdict. Read the
  leg log for `RESULT SUCCESS` and the obs count, per the obs directive.
* **F is HALF PROVEN, HALF BLOCKED** (unchanged). `otr_g4_wan_ti2v` proven end
  to end; `otr_upscale_ship` blocked on the operator
  (`docs/2026-08-23-item-F-upscale-ship-writer-failure.md`).
* **D is PARKED by the operator** (unchanged). Evidence banked at
  `docs/2026-08-22-negative-channel-declaration/driver_anchor.md`.
B, C and E's doc half closed 2026-08-22.


**STILL THE OPERATOR'S CALL, from the archived Ghost Prompt V2 item:**
(SUPERSEDED 2026-08-23: that profile retired with the non-haunted lanes, so this
call is now about `otr_ghost_signal_v3_haunted.json` if he still wants it. The
measurement below stands either way.)
`config/profiles/otr_ghost_signal_v3_haunted.json` pins `technical_model` to
Mistral-Nemo. B2 pinned gemma PER-LEG via `--set` rather than editing the
shipped profile, because promoting it would change SCRIPTS on this lane and
story output is a closed subject. The case for promoting it is measured --
**gemma-4-12b 8/8 accepted, Mistral-Nemo 4/8** on the real batch prompt -- and
it is a one-line profile edit whenever he says the word.

**GHOST IS ONE LANE -- DONE 2026-08-23 (`187380d0`).** Operator: *"I def dont
need 6 shit lanes"* / *"delete any animatediff that are not haunted"*. Six became
`animatediff15_v3_haunted_video`. The two parent CLASSES survive UNREGISTERED
because the winner inherits them; only the true leaves (v2, the h3/h5 cadence
pair) lost code. Five NAMED tombstones in `RETIRED_ENGINE_IDS`, one of which
carried the lane's published proof. Five profiles and three variant sets went
with them.

**TWO THINGS FROM THAT CHANGE ARE STILL LIVE:**
* **THE HAUNTED LANE HAS NO `otr/obs` RECEIPT OF ITS OWN.** The Ghost proof on
  record (`signal_lost_the_constables_knock_20260822_050116`, 8/8 beats) ran on
  the BASE lane that just retired. The survivor's evidence is that the operator
  and Gemini independently preferred its output -- a judgement, not a published
  episode. **One live leg is the next thing this lane owes**, and it is now the
  only Ghost lane there is to run.
* **STILLS ON THE ANIMATEDIFF LANE -- PARKED BY THE OPERATOR, 2026-08-23.**
  *"Perhaps animatediff was not using stills but maybe it should be"*, then
  *"not sure if I can accept stills"*, then *"image in we can park"*. The lane
  declares no stills today and `test_ghost_signal_lane` pins that the image
  dispatcher mints none for any ghost role; the G3.7 note above explains why the
  portrait-policy seam is inert for it rather than broken. Whether it SHOULD
  consume a still is a design question with more than one defensible answer --
  a full arc when he picks it up, not a knob.

**AND ONE HAZARD LOST ITS SUBJECT.** `build_variants.py --check` read 54 variants
/ 2 FAILURES all week, both on `otr_ghost_signal_v3` -- the hand-edited ship
candidate a blanket regeneration would have reverted, which is the stated reason
order 10 is HELD. That profile retired with its siblings and the check now reads
**51 variants, 0 failures**. Order 10's other half (widget position is persisted
production data) is untouched and it stays HELD, but the drift it was waiting on
is gone.

**D. The guides/dialect/recipe refactor (operator's architecture, staged).**
*"the PROMPT is the LANGUAGE from the video lane; the DIALECT is the
instructions for prompting the image lane -- double vs single quotes, neg prompt
vs no neg prompt, temperature, seed, knobs."* Three layers, all existing repo
vocabulary: **prompt** (lane) / **dialect** (engine phrasing) / **recipe**
(engine parameters -- the repo already says `recipe`). Today the composer
PRE-JOINS everything and each engine regexes it back apart; `ideogram4_local`
already implements a dialect without it being called one. Its own item, its own
arc -- do NOT bundle it with a small fix. `docs/2026-08-22-golden-rule-any-engine-any-slot/PLAN.md` section 8.


**E -- THE ENFORCEMENT HALF, STILL OPEN.** (The closed half, which corrected
IG2.2 from "SERVES" to "DECLARES", is in the archive.)
**The enforcement half needs an arc and is DELIBERATELY NOT DONE SOLO.** What
would close it is a per-KIND declaration -- "every engine serves every request
KIND its lane asks for" -- and no image engine declares a kind surface today.
Adding one is a design choice with more than one defensible answer (what the
kinds are, who owns the role-to-kind map, what a partial declaration does to the
menu), so by the 2026-08-17 routing rule it wants a full arc, and it belongs
with item D rather than bolted onto a gate doc.


**F. Regression sweep over shipped video profiles** (carried): only
`otr_w45_wan_ti2v` is proven; `otr_g4_wan_ti2v` and `otr_upscale_ship` remain
unexercised.



## Coding-sprint items 1-5 -- all closed or removed by ruling

### 1. -- REMOVED. PUBLIC_DOMAIN IS LEFT AS IS (operator, 2026-08-23)

*"no more chasing public domain, leave it as is, remove any public domain
updates from go forward."* Nothing about this lane is go-forward work. The
ruling that settles it lives in `docs/OTR_STANDING_RULINGS.md`; the retired
campaign text is in `docs/GO_FORWARD_ARCHIVE.md`. Do not re-add a row here.

### 2 and 3 -- BOTH CLOSED, re-verified against HEAD 2026-08-23

Neither was done by this window; both had simply gone stale in the file. Full
text of the two rows is in `docs/GO_FORWARD_ARCHIVE.md`.

* **2. The non-commercial notice -- DONE (2026-08-07).** The row said *"Nothing
  renders it. `otr_credits_roll.py:516` reads only `credits_source_line`."* False
  at HEAD: `otr_credits_roll.py:554` reads `meta["noncommercial_notice"]` and
  appends its own `intercept` entry. It was built to the row's exact acceptance
  conditions -- a SEPARATE `if` rather than an `elif` (so a malformed legacy
  ledger missing `credits_source_line` still shows the rights warning), no
  second prefix (the string already begins "NON-COMMERCIAL SOURCE:"), `.strip()`
  so a whitespace-only field cannot emit a bare `>>`, and adjacency falling out
  of the append order. Covered in three test files.
* **3. The test-ordering pollution -- NO LONGER REPRODUCES.** Running
  `tests/test_public_domain_sources.py` and `tests/test_public_domain_interpreter.py`
  ADJACENTLY -- the exact condition the row describes -- gives 28 passed. The
  module-identity breakage it diagnosed (a double `importlib.reload` replacing
  class objects the other file had already imported) is not observable at HEAD.

### 4. THE SPOKEN-CITATION CODA IS CLOSED -- re-verified against HEAD 2026-08-23

The operator asked whether this was already fixed. It was, and this row had gone
stale: **all four owed items are done**, and the last fragment was closed in the
same check.

* **B4, extract the coda helper -- DONE.** It is
  `OTR_LedgerScriptWriter._compose_and_stamp_announcer_close`, a MODULE-LEVEL
  function (the row's old line range, `:5463-5588`, no longer exists). **And the
  TRAP was honoured:** `news_meta` stays in the CALLER -- it is defined at
  `:5393` inside `run()` and read below there, exactly as the warning required,
  and the extracted function's own docstring records why.
* **B6, bump `CURRENT_SCHEMA_VERSION` -- DONE.** It reads `l4-2026-08-07`, and
  `scripts/audit_spoken_citations.py` requires `spoken_coda_source` on anything
  newer while holding the COMPLETE pre-l4 lineage as legacy. The comment there
  even guards the boundary string against being "helpfully" updated, which would
  invert the audit over 1,587 historical ledgers.
* **Writer-level routing tests -- DONE.** `tests/test_announcer_close_routing_matrix.py`
  (10 tests) walks both fidelity banks x {owned fact, silent}, asserts the coda
  is PRESENT rather than merely URL-free, carries it through to a real
  `Dialogue:` cue, and uses `media_archive` as the control exactly as specified.
* **Bug Bible coverage -- DONE.** `BUG_BIBLE.yaml:5249` carries the rule: "the
  edited function has no callers on the production path".

**AND THE PARKED WORKTREE ITEM IS CLOSED TOO (2026-08-23).** The deletion that
another session staged and stood down was never merged -- that worktree is clean
at an old commit -- so it was re-verified from scratch and done here. Both
symbols were confirmed dead at HEAD: `finalize_news_coda_surface` had ZERO
callers, and `news_coda_spoken_reduction` was gated on two compose flags,
`news_coda_fact_reduced` and `news_coda_fact_deferred_to_credits`, that are READ
in that one place and SET NOWHERE in the tree. The branch could never fire; the
else-arm popped a key nothing had put there. Removed, with a tombstone comment
naming the reason -- the same shape as the `news_coda_fallback` receipt above
it, which tested for a string "no composer has ever emitted".

### 5. THE COMMERCIAL-CLEAN JOIN IS CLOSED (done 2026-08-23; full row in the archive)

Built to the row's own shape: the JOIN is stamped, the 40 bank rows were NOT
touched, resolution is one profile by `(role, engine)` through
`EngineProfileResolver.profile_for`, and enforcement stays OFF (the `gated`
count still feeds one non-blocking I-8 warning).

**Where it landed.** `nodes/cast_lock.py` gained
`_delivered_commercial_clean(entry, ref)` -- the clip's licence AND the model's,
joined -- and it is now the ONLY reader of a bank row's flag in that file. All
five `gated` counters, both `clean=` report strings and the `_stamp` funnel go
through it. The two report sites compute the value ONCE into a local that feeds
both the counter and the string, so the report can no longer disagree with the
ledger printed beside it.

**The row's premise held, with two corrections.** The engine flag is at
`eng_indextts2.py:159`, not `:55` (line drift). And the authority for the model's
licence is not the adapter class attribute but the ENGINE-PROFILE layer, which
already knew: `char_indextts2_v1` carries
`commercial_clean: false  # Bilibili license -- non-commercial use gated`. The
cast layer had been contradicting the profile layer about the same audio.

**A second suspected site was checked and is NOT broken.**
`allowed_for_release` (`_otr_voice_node_common.py:1389`) derives from
`effective_license_state(profile)` -- the profile layer -- so the release
manifest was already truthful, and the release gate is armed nowhere but in its
own tests. No change made there.

**One deliberate design call:** the join DOWNGRADES only on a KNOWN-gated model.
An engine with no curated profile leaves the clip flag standing, so a partial
install behaves exactly as it did before. Guarded by an AST ratchet plus a
negative control in `tests/test_cast_lock_commercial_clean_join.py` (10 tests):
nothing outside the join helper may read a bank row's clip licence again.


---

## CLOSED 2026-08-25 -- `scifi_news_pro` Class A + Class B (was the top-of-file NEXT ITEM)

Moved verbatim from `docs/GO_FORWARD_PLAN.md`'s `>>> NEXT ITEM <<<` banner.
Class B had actually shipped same-day as Class A (`b19a11ef`) but this
banner was never updated to say so -- a full day of drift, caught when a
window about to re-code Class B read the real file first. See
`docs/PROD_BUG_LOG.md` PBUG-20260824-01 for the closing receipt: 17 PASS /
0 FAIL for `scifi_news_pro` counted correctly from AFTER both fixes landed
(the original 60% measurement predates both).

### >>> NEXT ITEM: `scifi_news_pro` -- CLASS B IS OPEN, AND CLASS A OWES A RATE <<<

**Class A shipped and is LIVE-PROVEN** (`a19f3df2`). The receipt is in
`docs/PROD_BUG_LOG.md` under PBUG-20260824-01; do not restate it here.
First post-fix pass: `RESULT SUCCESS` in 12.8 min, episode published to
`otr/obs`, salvage NOT used. **What remains is open work, and it is two things.**

**1. CLASS A OWES A RATE, NOT A PASS.** One green leg proves the lane can
produce again. It does NOT retire a defect that was measured at **6 failures in
10 live passes**. Let the loop run and compare the `scifi_news_pro` failure rate
against that 60%. Leg logs now archive per pass to `tmp/legs/passNNN/`, so a
failure can be diagnosed from ONE loop instead of another night.
**Do not mark PBUG-20260824-01 closed on a single episode.**

**2. CLASS B (`_pass_news_read`) IS UNTOUCHED AND IS THE REAL NEXT ITEM.**
`NewsProTreatmentError` after 2 attempts: *"the closing read is a FACTUAL report
and it names invented characters (Laura Goodkind)."* The validator
(`_make_news_read_validator`, `nodes/_otr_scifi_news_pro.py:1748-1755`) is
CORRECT -- a factual news close must not cite the drama's fictional cast. The
suspect shape is the prompt: `_pass_news_read` builds
`FICTIONAL CAST NAMES (never use these in the factual read): ...`
(`:1778-1779`), i.e. it hands a small local model the exact tokens it must not
emit, and that block is the ONLY channel by which those names reach its
context at all.

* **All three r1 lanes independently ruled Class B a DIFFERENT fault** --
  different pass, different ladder (typed `structured_call`, 2 attempts),
  different error type, different validator, no shared code path with the
  markup parser. **It must never be reported as covered by the Class A fix.**
* The repo has already ruled the mirror case: `news_close_read` is excluded
  from the SCRIPT prompt because it is *"a distractor a small local model does
  not reliably resist"* (`:1819-1829`). The candidate fix is the same move --
  give the names to the VALIDATOR only (it already receives them, `:1732-1735`)
  and remove them from the model's context, after which the failure becomes
  impossible rather than merely less likely.
* **[ASSUMPTION] until measured:** that the listing CAUSES the copying. Verify
  with matched prompt variants before changing that pass.

---

## CLOSED (already, before 2026-08-25) -- overnight loop PATH-inheritance gotcha

Also found stale during the 2026-08-25 plan-drift catch above: this item's
"durable fix, not yet applied" is already applied. `scripts/otr_overnight_loop.sh:26`
carries `export PATH="/c/Program Files/Git/usr/bin:$PATH"`, and the loop's
own log shows real, non-empty timestamps throughout
(`[07:36:33Z] pass 1: launching bank gate (obs=121)`), confirming coreutils
resolve correctly. Moved verbatim, not re-verified further.

**THE GOTCHA, found 2026-08-24 by reading the log header rather than trusting
that the process was alive.** Launching `scripts/otr_overnight_loop.sh` via
`Start-Process` inherits the caller's PATH. When that PATH lacks Git's
`usr\bin`, the loop STILL RUNS -- the bank gate is a Windows python call and
works perfectly -- but every coreutil inside the supervisor silently returns
nothing. The tell was a header reading `[] pass 1: launching bank gate (obs=)`
instead of a real timestamp.


## MOVED FROM GO_FORWARD_PLAN 2026-08-27 (verbatim, nothing edited)

Removed from the forward plan because it is CLOSED or historical, not
because it is unimportant. The plan is open work only; this is the record.

### FOLEY/MIME row -- the 2026-08-26 build history and superseded exit text

**THE CODE IS BUILT, GREEN AND PUSHED (2026-08-26, `a7675d37`). THIS IS NO
LONGER A CODER ITEM.** What is left is a live leg and a pair of ears, which is
a RENDER window's job. A coder window taking this row should skip to the next
one unless it is also driving the GPU.

**WHAT IS ALREADY MEASURED, so it is not re-measured:** a live canonical leg on
`otr_ltx25_high_foley_plus` decoded 12 beats at
`decode_peak_mb` **2883-3269 MiB (avg ~3090)** against the 14.5 GiB
(14848 MiB) stop. Every stem came out at exactly `186240 = 97 x 1920` samples
on a runtime-read 48000 Hz rate. **The two-pass split is proven and the VRAM
question is CLOSED** -- the audio decode costs ~21% of the ceiling, in
isolation, after `reclaim_idle_models`. Do not re-open it.

**WHAT IS STILL OWED, and it is the whole remaining exit:**
* the FOLEY leg's own tail -- `foley_bed=mixed beats=N/N lanes=...`,
  `foley_loudness=...`, and `obs_publish OK`. The leg was still rendering at
  handoff (see HANDOFF_LOG for its state);
* a MIME leg on `otr_ltx25_high_mime`. That profile pins mime on the CHARACTER
  role ONLY and leaves announcer/music on the SILENT lane deliberately: mime
  zeroes the master over its own beats, so it needs speaking neighbours in the
  same master WAV to be measured against. All three roles on mime would silence
  the episode and prove nothing. Watch for `foley_muted_s=`;
* `ltx25_video` proven STILL silent on the same HEAD;
* **the listening test, which no receipt can stand in for.** The lab heard LTX
  foley and rated it the model's strong suit -- but off a SINGLE-STAGE graph.
  These lanes harvest the REFINED stage-2 latent, which nobody has heard. If it
  disappoints, harvesting stage one's `separate` slot 1 instead is a one-line
  change and the rest of the path is identical.

**Below is the original coding brief, kept because it records WHY the build is
shaped the way it is.** It is history now, not instructions.

**DESIGN IS DONE AND CONVERGED. This was a CODING item, not a design one.**
Four rounds, Codex + Cursor, artifacts in `kibitz-runs/2026-08-26-foley-bed/`
(gitignored -- read them off disk, they do not travel).

* **THE SPEC:** `kibitz-runs/2026-08-26-foley-bed/r4/final.md`. It is
  CUMULATIVE -- the r1 body, then r2/r3/r4 amendment sections that OVERRIDE
  earlier text wherever they disagree. Read it whole before typing.
* **THE RULINGS:** `docs/2026-08-26-foley-bed-OPERATOR-RULINGS.md` (committed).
  Four of them, all binding, none for the panel to relitigate.

**WHAT IT IS.** LTX 2.5 already generates its own audio -- footsteps, room
tone, a score -- while rendering the picture, and the adapter throws it away at
`LTXVSeparateAVLatent` (`eng_ltx25.py:6`). This keeps it and mixes it under the
episode master at a fixed **0.20 foley / 0.80 master**. A second lane, **mime**,
is the SAME mechanism at **1.00/0.00**.

**THE WORD THAT MUST NOT BE CONFUSED.** The **SFX bed** was a different feature
(separately GENERATED effects from a dedicated model), **ripped 2026-08-06 and
permanently dead**. The **FOLEY bed** is the video model's own output. Naming
anything `sfx_*` is a defect -- `tests/test_rip_sfx_bed_guard.py` guards it.

**WHAT THE ARC ALREADY CAUGHT, so it is not re-learned on the GPU:**
* **A VRAM blocker with 0.02 GiB of headroom.** Wiring the decode into the same
  graph makes the audio VAE a second consumer, so `free_after_use` can no
  longer drop it before sampling. **Two passes**: harvest the latent, reclaim,
  then a second tiny graph.
* **The audio latent is DESTROYED before a second pass could reach it** --
  `wrapper_bridge` keeps only `{"unet","modality","decode"}`. Needs two new
  parent hooks, both no-ops on the silent lane.
* **The canonical workflow JSON DOES change** -- `OTR_EpisodeAssembler` has no
  way to know it is on a foley route. Append `video_policy_json`, wire node 87
  -> node 7. Append-only (BUG-LOCAL-097), same change as the code (section 0).
* **A stem written to tmp is deleted before the mux reads it** -- `mux()`
  sweeps `_shared/tmp`. Write straight to `otr/episodes/<ep>/audio/`.
* **`tests/test_rip_sfx_bed_guard.py:262-271` fails on first compile** unless
  rewritten in the same change (it requires the connector tooltip to say
  "retired").

**EXIT:** a live canonical leg published to `otr/obs/` with `obs_publish OK`,
the foley proven decoded and mixed (not adapter self-report), loudness and peak
receipts, and `ltx25_video` proven still silent. Measure audio-decode VRAM on
the two-stage graph BEFORE registering -- over 14.5 GiB is an operator stop.

**STATUS 2026-08-26: THE CODE IS BUILT AND GREEN. THE EXIT IS NOT MET -- no
live leg has run, so NOTHING here is qualified.**

**BOTH LANES SHIPPED, NOT ONE.** The operator overrode the spec's mime
deferral mid-build (*"foley and mime we need this feature for both"*), so
`ltx25_foley_plus` (0.20 foley / 0.80 master, global) and `ltx25_mime`
(1.00 / 0.00, per-window) are both registered and public. That is RULING 6 in
`docs/2026-08-26-foley-bed-OPERATOR-RULINGS.md`; the connector decision is
RULING 5. **The `kibitz-runs/` spec still says "mime CUT" and "ONLY
ltx25_foley_plus" -- it is superseded on that point and only that point.**

Landed in one change: the two parent seams plus both lanes and the second-pass
decode (`eng_ltx25.py`), the stem format, the lane gain table and the mix
envelope (`_otr_video_engines/foley_stems.py`, new), the coverage cutter and
manifest threading (`render_driver.py`), the pre-loudness provisional master
and its ledger flavour stamp (`scene_sequencer.py`), the mix and the single
delivery gain (`otr_master_audio_mux.py`), and three appended links in
`workflows/otr_canonical.json`.

**WHAT A RENDER WINDOW STILL OWES, and it is the whole exit condition:**
* Pin a role to `ltx25_high_foley_plus (16:9)` and run ONE canonical leg; then
  the same for `ltx25_high_mime (16:9)`. **Mime is ROLE-WIDE** -- every beat of
  the chosen role goes silent -- so pick the role deliberately.
* Read `decode_peak_mb` off the per-beat `FOLEY decode:` log line. That is the
  audio-decode VRAM measurement, and it is logged on EVERY beat rather than
  measured once. **Over 14.5 GiB is an operator stop.** Note the tighter
  context: the G8 solo smoke measured the shared picture graph at 16152 MiB
  in-pipeline on a 16303 MiB card -- about 150 MiB of headroom -- which is
  exactly why the audio decode is a second graph that runs only after
  `reclaim_idle_models`.
* Prove the bed by LISTENING, and by the mux's `foley_bed=mixed beats=N/N
  lanes=...` + `foley_loudness=...` (+ `foley_muted_s=...` on a mime leg)
  receipt lines -- never by the adapter's self-report.
* Prove `ltx25_video` is STILL silent on the same HEAD.
* **The stage-2 foley QUALITY is still an assumption.** The lab's golden foley
  recipe is single-stage; these lanes harvest the REFINED audio latent. Nobody
  has heard it.

### SUPERSEDED BANNER -- its re-triage instruction was already carried out

### >>> SUPERSEDED BANNER (kept verbatim -- its re-triage instruction was carried out above) <<<

**Found 2026-08-25, before re-coding what this banner used to name.** The
`scifi_news_pro` Class A/B item (was here) and the loop PATH-inheritance
item (was right after it) were BOTH already fixed and committed hours
earlier the same day -- Class B by `b19a11ef`, the PATH fix already sitting
in `scripts/otr_overnight_loop.sh:26`. Neither banner was ever updated to
say so. Full receipts moved verbatim to `docs/GO_FORWARD_ARCHIVE.md`
(search "CLOSED 2026-08-25"); the scifi_news_pro closure is also in
`docs/PROD_BUG_LOG.md` PBUG-20260824-01 (17 PASS / 0 FAIL, counted correctly
from after both fixes landed).

**Do not trust the rest of this file's OPEN sections at face value either
without a quick grep-the-real-code check first** -- this is now a confirmed,
repeatable failure mode for this file (two stale banners found in one pass),
not a one-off. The CURRENT RUNWAY table below (operator-ordered 2026-08-13)
is the next candidate; spot-check row 2 (LEMMY Phases 2-4) against the real
files before starting it, the same way this catch was made.

**WHAT THE 2026-08-25 EVENING WINDOW ACTUALLY DID, so the next window does not
re-triage it:** it worked the 4060 crash reports the operator pasted, not this
banner. Four commits, all pushed and lockstep-verified:
`be0ab7fb` (cast_lock's unguarded `config` import, PBUG-20260825-02),
`2c524732` (unquantized loads inheriting a 4-bit-sized VRAM cap,
PBUG-20260825-03), `063fcfc3` + `fb67d059` (PBUG-20260825-04, the
BUG-LOCAL-098 tripwire and the orphan-lifecycle races behind it), plus the
`2.0.0-alpha.9` registry release. The orphan work left TWO deliberate
deferrals, both now filed under OPEN BUGS as "The orphan-lifecycle pair" --
read that row before touching `_otr_model_loader.py`'s cache lifecycle, since
three successive review rounds each found a new race in the previous cut of
the same fix.

---

## COCKNEY BLEED -- the root cause, as the plan carried it (closed 2026-08-27)

**CLOSED BY `a967b47c`.** Kept verbatim below because it is the explanation of
WHY the scoping was wrong, and because it contains one sentence that turned out
to be WRONG -- worth preserving rather than quietly deleting.

**THE CORRECTION.** The block below ends by saying the fix is *"to make LEMMY
the grammatical subject of the accent sentence and leave the orthography
sentence global"*. The converged plan superseded that second half and the
shipped code follows the plan, not this paragraph. "Global" was only ever meant
to mean *global within a Lemmy-containing call* -- phonetic spelling is unwanted
from anyone. Left literally, it would have kept appending orthography bytes to
NON-Lemmy calls, which breaks the byte-identity half of the acceptance contract
for exactly the lines the fix exists to leave alone. A non-Lemmy active set now
receives ZERO policy bytes, and the orthography clause rides inside the single
scoped block.

Everything else in the block held up under the code: the two production callers,
the subjectless first sentence, and the reading that `roster_has_lemmy()` gated
WHETHER rather than WHO.

### >>> BACKGROUND ONLY: COCKNEY BLEED -- THE ROOT CAUSE, KEPT FOR THE READER <<<

**Operator-observed on published episodes, 2026-08-26:** *"when lemmy is in the
scene everyone starts talking like lemmy with a cockney speech, not just
lemmy."*

**ROOT-CAUSED, and the fix is a scoping change rather than a rewrite.**
`nodes/_otr_dialogue_policy.py:6-10`. `append_dialogue_policy()` appends
`_COCKNEY_ORTHOGRAPHY_RULE` to the WHOLE system prompt whenever
`roster_has_lemmy()` is true, and the rule's first sentence is unscoped:
*"Convey the Cockney accent through phrasing, idiom, cadence, and rhythm."*
Nothing in it names LEMMY, so the writer reads it as a scene-wide instruction
and re-registers the entire cast.

The irony worth preserving: the rule's real job is its SECOND sentence -- use
standard English spelling, no phonetic misspellings -- which is doing exactly
what it should. The first sentence was meant as context for it and became an
accent order. `roster_has_lemmy()` gates WHETHER the rule is added; nothing
gates WHO it applies to.

So the fix is to make LEMMY the grammatical subject of the accent sentence and
leave the orthography sentence global (phonetic spelling is unwanted from
anyone). It is NOT a prose-quality item -- a character's voice contradicting
the source is a correctness defect, and the 2026-08-04 "story quality is done"
ruling explicitly leaves that class open.

**THIS ROW IS NO LONGER THE WORK -- IT IS THE EXPLANATION.** The fix is
CODE-READY at the TOP of this file
(`docs/2026-08-27-cockney-bleed/CODE_READY_PLAN.md`, design commit `f92530e2`).
Take it from there; this section survives only so the reader understands WHY the
scoping is wrong without re-deriving it. It is the LEMMY ACCENT BLEED and
nothing else.

**The non-audio dialogue-prompt work it used to point at is SHIPPED** (2026-08-27,
`e923a9f3`) -- a different defect that merely shared the word "dialogue". Its
snapshot `C:\Users\jeffr\AppData\Local\Temp\otr-speak-act-kibitz-20260826-2125`
must still not be deleted; what it now owes is the live proof in the row below,
not more code.

---

## CLOSED 2026-08-28 -- foley audibility, the voice-bleed fix, and the LTX 2.5 foley+mime qualification window

Archived verbatim from the top of GO_FORWARD_PLAN.md. Both rows are now resolved: see PBUG-20260827-03 and PBUG-20260828-01 in `docs/PROD_BUG_LOG.md` for the closing receipts, and the live queue described in the replacement row for the six-leg listening proof still in flight when this was archived.

### >>> TAKE THIS FIRST: MAKE THE FOLEY BED AUDIBLE <<<

**OPERATOR, 2026-08-27, after listening: *"I think foley is our biggest gap
now."* That settles the order -- this row is above everything else.**

**THE BED IS MIXED, LEVELLED, GREEN ON EVERY RECEIPT, AND INAUDIBLE.**
PBUG-20260827-03, Bible `12.137`. Published episode
`signal_lost_ink_and_martyrdom_20260827_071626`. Measured: the bed sits
**37-58 dB under the programme, median 45** (audible is 15-25 under).

**THE CAUSE, and it is one line of code.**
`FOLEY_LANE_GAINS['ltx25_foley_plus'] = (0.20, 0.80)` in
`nodes/_otr_video_engines/foley_stems.py` is a BARE MULTIPLIER on stems whose
level is never measured. LTX 2.5 emits foley at **-30 to -55 dBFS RMS**, so
0.20 subtracts another 14 dB from something already inaudible.

**ALREADY RULED OUT, with evidence -- do not re-derive:** the mux ran (the two
masters differ; the mp4 carries AAC 48 kHz stereo); the video DID generate audio
(every stem carries signal, loudest peak -12.8 dBFS); it was the right engine on
all 12 beats. **And the A/B that settles it: `ltx25_mime` at gain 1.00, on
equally quiet stems, published the same day and the operator said *"sounds
great."*** Same engine, same mux, same stem levels -- only the gain differs.

**THE DECISION THAT BLOCKS THE FIX, AND IT IS THE OPERATOR'S.** A single larger
constant CANNOT work: the stems span 21.2 dB RMS, so a value that lifts the
quietest puts the loudest forward of the dialogue. Both panel lanes (Codex
`gpt-5.6-sol`, Cursor Grok 4.6 High) converged on referencing each stem before
applying the ratio -- which **RULING 2 forbids by name**
(`docs/2026-08-26-foley-bed-OPERATOR-RULINGS.md`: *"The foley stem gets no
normalization pass of its own"*). That ruling was made before anyone knew the
stems arrive 30-55 dB down. **Amend it or the bed stays inaudible; there is no
third option that respects it as written.**

**THE OPERATOR'S OWN HYPOTHESIS, WORTH TESTING FIRST BECAUSE IT IS FREE.**
*"Shouldn't we be rendering a foley?"* -- the motion bake-in (`65538f41`,
`f46abe03`) landed AFTER every episode rendered so far, and LTX 2.5 scores the
event it can SEE. More visible action may produce LOUDER stems, which could
raise the bed without touching RULING 2 at all. **One foley leg on the new
prompts distinguishes the two causes**, and no episode has yet rendered with
them.

* Run it: `-Profile otr_ltx25_high_foley_plus -Acts 2`.
* Then measure the stems, do not guess: compare raw stem RMS against the
  -30/-55 dBFS baseline recorded in PBUG-20260827-03.
* **The 2-second harness exists:** `scripts/otr_replay_foley_mix.py
  <episode_dir> [--inject-unpositioned]` replays the mix from disk artifacts, so
  a terminal-node fault never again costs a 3-hour render.

**WHAT WOULD PROVE IT FIXED:** per-beat `RMS(bed in window) - RMS(programme in
window)` inside the intended band, and an operator listening pass. Both panel
lanes were explicit that `foley_bed=mixed` is PLUMBING, not audibility -- the
receipt that hid this bug must not be the receipt that closes it.

---

### >>> THEN: COCKNEY BLEED -- CODE SHIPPED, A LIVE LEG AND A BIBLE ROW OWED <<<

**THE CODE IS DONE AND PUSHED (`a967b47c`).** Roster semantics are gone: the
Cockney rule is scoped to the ACTIVE SPEAKER -- `(req.speaker,)` per line,
`tuple(slot.speaker for slot in beat_group)` per exchange -- the rule names
LEMMY as its grammatical subject and fences every other character's register,
and `append_dialogue_policy` refuses roster-shaped values instead of widening.
Receipt: `PBUG-20260827-02` in `docs/PROD_BUG_LOG.md`. Suite 12377 passed /
121 skipped / 1 xfailed; Bug Bible 22 passed; canonical workflow untouched and
re-validated at 23 nodes / 60 links.

**WHAT IS STILL OWED -- TWO THINGS, and the second is cheap.** FIRST, the live
canonical leg. The 5080 was rendering the mime + H3 chain for the whole coder
window, so nothing has yet proven production reachability or given the operator
a LISTENING gate.
Run it from `docs/2026-08-27-cockney-bleed/CODE_READY_PLAN.md` P5.3 exactly --
`media_archive`, `lemmy_cameo=always include`, three acts, `-Port` omitted so
the wrapper picks a free ephemeral port -- and require the applied-patch receipt
to show both widgets before accepting the leg.

**DO NOT RE-DERIVE THE FIX AND DO NOT RE-OPEN THE ARC.** The captured-prompt
tests are the deterministic scoping gate and they already pass; the live leg
proves reachability and sound, which is a different claim. A small lexical
sample can never prove bleed impossible and must not become a dialogue
blacklist. If bleed somehow survives, P5.3 item 10 says where to look next --
the labeled full-cast voice cards and the rolling prior context -- before
anyone widens the patch.

**SECOND: THE BUG BIBLE CANDIDATE, and the coverage scan is ALREADY DONE so the
qualifying window does not pay for it twice.** Deferred deliberately -- the
`PROD_BUG_LOG.md` amendment puts a single promotion at WRAP-UP, and a
cross-project rule should not be minted while its production proof is
outstanding. Promote it in the same window that qualifies the leg.

* **The class:** a style or policy instruction whose GATE is a membership test
  over a population (*is X anywhere in this cast?*) while its SCOPE over
  subjects is never written down, delivered as a SUBJECTLESS imperative into a
  prompt that renders several subjects in one call. Absence is the correct
  answer for a lone non-target subject; a NEGATIVE clause is the only thing
  that works when target and non-target share a single call.
* **Checked against `BUG_BIBLE.yaml` (315 entries, guide HEAD `91e4cea`) and
  `otr_coverage_index.yaml`. Two neighbours, neither of them a cover:** `07.29`
  is a shared prompt builder invoked with the wrong scope PROFILE, and it is
  image-generation -- it names the scope failure but not the gate-versus-scope
  confusion and not the missing grammatical subject. `12.136` is yesterday's
  rule and keys on the routed ENGINE'S CAPABILITY, a different axis entirely.
  `12.114` is the reserved-identity ASSET leak -- the voice, not the prompt.
* **The verify half worth carrying, because it is the part that nearly fooled
  this window:** a presence-and-absence pair proves nothing until it is run RED
  against the old code. One of these tests passes on the unfixed implementation
  for an ACCIDENTAL reason -- `_normalize_cast` had already turned cast rows
  into objects the old str-or-dict detector could not see -- so it pins a
  forward invariant and is not evidence of a fix. Its docstring says so.

---

### >>> NEXT: QUALIFY THE LTX 2.5 FOLEY BED + MIME -- A RENDER WINDOW <<<

**FOLEY IS QUALIFIED AS OF 2026-08-27. MIME AND THE LISTENING TEST ARE NOT.**
A live canonical leg on `otr_ltx25_high_foley_plus` ran 3h19m09s and published
`signal_lost_ink_and_martyrdom_20260827_071626` to `otr/obs/`:
`RESULT SUCCESS`, `obs_publish OK`,
`foley_bed=mixed beats=12/13 lanes=ltx25_foley_plus:12 master_gain=0.80`,
`foley_loudness=lufs measured=-12.29 -> target=-14.0 gain_db=-1.71
peak_dbfs=-3.52`, and -- the line that matters --
`foley_unpositioned=1 (no master-mix slot; normal for music_inter bridges)`,
which is PBUG-20260826-02's killer beat being skipped instead of killing the
episode. 37 decodes, zero fatal markers.

**WHAT IS STILL OWED:** the MIME leg (running at time of writing), and **the
listening test, which no receipt can stand in for** -- `foley_bed=mixed` proves
the bed was decoded, placed and levelled, not that it sounds right under the
dialogue.

**A TERMINAL-NODE FAULT NO LONGER COSTS A WHOLE RENDER.**
`scripts/otr_replay_foley_mix.py <episode_dir> [--inject-unpositioned]` replays
the foley mix from disk artifacts in about two seconds. Use it before spending
three hours.


---

---

# ARCHIVED 2026-08-28 -- the lean-and-mean reorganization

**Why these blocks are here.** The operator asked for the go-forward to be
cleaned up *"lean and mean"* and reorganized *"on the most logical [basis] so
we can code the most and test the least."* `GO_FORWARD_PLAN.md` was rebuilt
that day into three working queues (no-render work / render work batched by
the leg that proves it / operator questions in one pass). Everything below
was REMOVED from the plan in that pass and is preserved VERBATIM: closed
receipts from the 2026-08-27/28 sessions, the superseded 2026-08-13 runway
table, stale scheduling text, and re-triage notes whose conclusions were
folded into the live queues. Nothing was deleted. Every operator ruling these
blocks carried was lifted into the plan's "SECTION 4 -- STANDING RULINGS
LIFTED FROM ARCHIVED BLOCKS" BEFORE the move. Blocks that stayed live were
retitled/renumbered in place and are NOT duplicated here.

## 1. The file-header prune note of 2026-08-24 (its standing warning was kept in the new header)

**Pruned again 2026-08-24** on the same instruction -- *"update the go forward
plan so it's accurate, truly go forward items"*. 402 lines of closed receipts
and superseded text went to the archive VERBATIM, nothing edited and nothing
deleted: a superseded problem statement, five coding-sprint rows that were
already closed, a superseded ship-intent block, and two stale queue-state
receipts. **Most of that was written into this file the same day it was removed
-- by the window that closed the work.** Recording what you just finished
inside a section headed OPEN is the easiest way to make this file lie, and it
is the specific failure the 2026-08-16 self-audit flagged and did not dare fix
blind. Receipts belong in `docs/PROD_BUG_LOG.md` and the archive; rulings
belong in `docs/OTR_STANDING_RULINGS.md`; only what is still TO DO belongs here.

## 2. The 2026-08-27/28 foley / voice-bleed / named-sounds receipts (top-of-queue block; the LISTENING TEST paragraphs stayed live as THE CURRENT STEP)

### >>> TAKE THIS FIRST: NAME THE MIME SOUND, THEN LISTEN <<<

**FOLEY AUDIBILITY IS RESOLVED, THE VOICE BLEED IT UNCOVERED IS FIXED, AND SIX
LEGS ARE RENDERING RIGHT NOW TO PROVE BOTH.** PBUG-20260827-03 closed
2026-08-28: it was never the mixer. `65538f41`/`f46abe03` (per-lane motion
prompts) landed after every prior episode rendered, and once the dispatch
seam actually reached the engines (`24f85f95`, `34253841`) the SAME
`FOLEY_LANE_GAINS` 0.20 took raw stems from -34.4..-56.2 dBFS to
-14.0..-41.5, and bed-vs-programme from 45 dB under to -6.6..-21.4. Ruling 2
never needed amending. The operator called it before the measurement did:
*"shouldn't we be rendering a foley?"*

**WHAT THAT AUDIBLE BED THEN REVEALED: voices, because the tail named no
sound.** `signal_lost_a_name_stripped_bare_20260827_195142` -- operator:
*"is speech the best word, or dialogue is bleeding into the prompt"*.
Production asked for `"matched environmental foley... ambient room tone"`, a
CATEGORY that leaves the model to choose, and with a face in frame it chose
voice. Fixed 2026-08-28, PBUG-20260828-01: a cue table now names the sounds
the beat's own action makes (`nodes/_otr_video_engines/eng_ltx25.py`), foley
and mime share the identical string (*"the only difference between foley and
mime is the mux layer"*), and a scoped kibitz r3 review (codex) caught and
fixed three real defects before ship -- a weapon-noun collision with the
banana route, a false-positive "finished" receipt, and `document` matching
`documentary`. Full receipt: `kibitz-runs/2026-08-28-named-sounds/`.
Commits `d3cca496`..`a196fd90`, pushed, HEAD == origin verified, suite
12365/121/1, 0 regressions.

**FOUND AND FIXED IN THE SAME PASS, UNRELATED BUT ADJACENT:** the style-cue
prefix was landing in front of `minimax_h3`'s required verbatim opener on any
non-default style pack, silently breaking "must begin exactly" -- confirmed
live on the anime pack, `d3cca496`. Not a production incident (the default
pack's cue is empty, so it never fired); a review catch, not a Bug Bible
candidate.

## 3. Stale scheduling text from the H3 acceptance row (overtaken by the overnight queue; the durable one-GPU warning was kept in the live Section 2 intro)

**THE BOX IS FREE, AND THE FOLEY RE-RUN OUTRANKS THIS ROW.** The foley
qualification leg that held port 8000 all night DIED at its terminal node after
3h17m22s and published nothing (PBUG-20260826-02, fixed in `499312bb`). Its two
python processes from 21:43 are still resident holding port 8000 with the GPU
already back to ~1.2 GB; whoever goes next resets them per `CLAUDE.md`
section 4.

**Order matters here.** Foley/mime qualification is the NEXT ITEM at the top of
this file and its blocker was only just cleared, so the foley re-run takes the
box first. This H3 row is a single leg and can follow it. Do NOT start both --
two windows resetting one GPU is how each kills the other's leg.

**There is no untreated foley BEFORE artifact to preserve** -- that leg never
reached the mix. The H3 before/after A/B is unaffected: its BEFORE sample is the
published Caretaker episode named above.

## 4. The 2026-08-25 LEMMY re-triage result (its two live conclusions -- the fastwan_8gb leg and the Phases 2-4 question -- were lifted into Batch R4 and Section 3 of the live plan)

**THE RE-TRIAGE THE PREVIOUS BANNER DEMANDED WAS DONE 2026-08-25 (late). Its
result is below; the old banner text follows underneath, unedited.**

**Runway row 2 (LEMMY Phases 2-4 + "its three live PBUGs") IS MOSTLY CLOSED and
the row is STALE.** Checked against the real tree, not the banner:
* **PBUG-20260811-01 -- CLOSED 2026-08-16, MIS-ATTRIBUTED.** The cameo never
  killed the writer; `lemmy_force` was INERT on that lane at the repro commit.
  Row 2's clause 5 asks to "resolve the fable2 BAD_LINE interaction" -- a bug
  closed as never having been about the cameo. **Withdrawn premise.**
* **PBUG-20260811-03 -- CLOSED 2026-08-18**, fixed and live-proven on a
  forced-cameo leg (`da44f642` + `7faf3bf7`).
* **PBUG-20260811-02 -- the ONLY one still OPEN.** Root cause established, the
  repair is WRITTEN, and it is not a coding item: it needs a canonical
  `fastwan_8gb` leg with 60-SECOND opening AND closing cues (long enough to
  chunk at `_MUSIC_MAX_CHUNK_DUR_S = 22.0`). **That is a RENDER window, not a
  coder slot.**
* **Clause 4 is moot** -- `scifi_news` no longer exists (live banks are
  media_archive, original, scifi_news_pro, public_domain, shakespeare,
  custom_source_bank).
* **"Phases 2-4" are STILL undefined anywhere in the repo** -- the phase
  numbering lives only in a gitignored `kibitz-runs/` directory. Per
  `docs/2026-08-16-lemmy-open-changes-PROBLEM-STATEMENT.md`, the six row-2
  exit clauses are the only readable statement of intent. **Asking a window to
  "complete Phases 2-4" is not an actionable exit condition** -- retire the
  numbering or recover it, but do not let it keep sending windows in circles.

## 5. The superseded CURRENT RUNWAY table (operator-ordered 2026-08-13; rows 1-2 superseded/closed, row 3 lifted to Batch R6, row 4's work to Section 1.5 and its ruling to Section 4, row 5's pointer to "After this queue")

## CURRENT RUNWAY -- OPERATOR-ORDERED 2026-08-13. WORK IT TOP TO BOTTOM.

**ROW 1 BELOW IS SUPERSEDED by PRIORITY 1 above (operator 2026-08-14).** Its
order was "resume the upstream Story Lab and A/B against it"; the lab is now
parked, read-only and being retired, and the story work happens in production.
Rows 2-5 are unchanged and still run in their listed order, behind PRIORITY 1.

This block is authoritative. It supersedes every older order, count, Lemmy gate,
and review-routing sentence lower in this file. Re-ground the active row against
the Windows tree before editing; when a row is coded, fully tested, committed and
pushed, move its receipt to `docs/HANDOFF_LOG.md` and remove it from this forward
queue in the same push.

| # | Active work | Exit condition |
|---:|---|---|
| 1 | **SUPERSEDED -- what remains is PRIORITY 1 at the top of this file** | Do NOT restart from the Story Lab. |
| 2 | **Give LEMMY a fighting chance: complete Phases 2-4 and its three live PBUGs** | Preserve the Cockney floor with one upstream engine-policy authority wired through the canonical workflow, CastLock and renderer; qualify real routes by operator-audition receipts; close the six-engine gender-only pin gap; restore or explicitly decline `scifi_news` cameo policy; resolve the fable2 BAD_LINE interaction; re-observe the missing closing before diagnosing. No silent substitute and no defined-but-unwired policy. |
| 3 | **Run seven fresh post-change 45-word render proofs** | All seven exact public engine IDs pass against the post-bugfix/post-Lemmy HEAD with `COVERS`, `RESULT SUCCESS`, server `Prompt executed` + `obs_publish OK`, and the canonical OBS asset on disk. See **WHAT IS ACTUALLY LEFT** below. |
| 4 | **Narrow learned-upscale hardening only** | Harden the two `SpandrelEsrgan._resolve_model` edge cases if still reproducible. The multi-GPU learned-upscale stage itself is CLOSED and must not be reopened. |
| 5 | **Handoff after executable rows 1-4** | Continue in `ROADMAP.md`: lean-mean -> RunPod/AMD/Mac -> install -> product docs/v2 release. This row is a pointer, not work that precedes lean-mean. Lean-mean scope and coding order live only in `docs/LEAN_MEAN_CLEANUP.md`. |

## 6. THE CODING SPRINT header of 2026-08-04 (stale framing; its cross-references to sections 0 / 0-BIS / 0-QUATER were already among the eleven broken pointers the 2026-08-16 audit recorded -- the targets left this file long ago. Its live items -- the gender ladder and the H3 standing context -- remain in the plan)

## THE CODING SPRINT (operator directive 2026-08-04; re-sized by the r1-r4 arc)

Item 1 is the structural work and consumes most of a session; items 2-3 are
small and share one campaign. Items 8 and 9 are DONE (receipts in
`docs/HANDOFF_LOG.md`). The live open work is sections 0 (video matrix pattern,
did NOT converge), 0-BIS (no-mirror, CODE-READY), 0-QUATER's deferred
shield-scoping chunk (own kibitz arc), and the 0-QUINQUE MiniMax ruling.
Work by priority, not by number -- the numbering is historical.

**RENDERS HAVE RESUMED** (2026-08-05). The 08-04 "no render runs this session"
line is spent -- it governed that session only, and the 08-05 handoff opens on a
live-proof obligation. Reset per `CLAUDE.md` section 4 before any leg: selective
CIM kill by CommandLine, never a blanket python kill (it severs the MCP tooling).

Everything below was verified against the real files on 2026-08-04, is
non-GPU, and is provable by the suite alone. Work them in order; each ends
green and pushed on its own.

## 7. Item 6, the G15 scene-coherence vacuity fix -- FIXED 2026-08-28 (the STILL OWED arming question stayed live as Section 3, question A)

### 6. ~~A TERMINAL FREEZE GATE THAT HAS NEVER READ A POPULATED FIELD~~ -- FIXED 2026-08-28

**The join and the vacuity refusal are done.** `find_scene_coherence_issues`
(`nodes/_otr_scene_guard.py`) now joins `beat_id -> beats[].scene_id`, the
join the schema actually has -- `lines[].scene_id` is confirmed dead, no
writer has ever populated it. `_check_g15_scene_coherence`
(`nodes/_otr_ledger_freeze.py`) writes the full `{required, checked, verdict,
issues}` shape into `report.info`, and an armed gate that examines zero real
linkages now fails loud instead of passing silently -- distinguished from a
ledger with no scenes at all, which stays a legitimate clean skip (a
pre-existing, different state; conflating the two would have been a wider
behavior change than asked). Commit `e2807dcc`, reviewed via scoped kibitz r2
(codex, 6 MUST-FIX, all grounded and folded in --
`kibitz-runs/2026-08-28-scene-coherence-vacuity/`), 28 tests including a
named regression guard, full suite 12374/121/1, 0 regressions, HEAD == origin
verified.

**STILL OWED, and it is the ONLY thing left on this item:** whether any bank
should actually ARM `defaults.scene_coherence_check`. Nothing does today --
the fix changes a function with zero live callers in current production, by
design, so this shipped at zero risk. GO_FORWARD's original text said
"measure OFFLINE over the published corpus first, then arm in ONE change" --
that measurement was never attempted tonight and stays open. Whoever picks
this up next decides first whether any bank should arm it at all before
running that measurement.

## 8. The "Two carried items" heading note (both items were relocated live: the Gutenberg fetch to Section 3 question G, the Shakespeare verbatim executor to Section 5)

### Two carried items with no home of their own

(Titled "Bench leftovers" until 2026-08-23 -- a name that now reads as the
retired VIDEO bench and has nothing to do with it. The block it referred to was
an older conditional list, gone long before. Renamed rather than moved: both
items below are real and open.)

## 9. The OPEN BUGS section header and the 2026-08-11 bank-sweep trio (two of three CLOSED per the re-triage above; PBUG-20260811-02 lives on as Batch R4)

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not. That
split is why the two eyeball-era entries at the end are PARKED rather than live.

### The 2026-08-11 bank-sweep trio (LEMMY sprint, all three OPEN)

Found by a six-bank live render sweep, not by tests. Full detail lives in
`docs/PROD_BUG_LOG.md` and `docs/2026-08-11-FINDING-lane-cast-contract-divergence.md`;
these rows exist so a window working THIS list actually sees them.

* **PBUG-20260811-03 -- `scifi_news` lost the LEMMY cameo it was built for.**
  ROOT CAUSE ESTABLISHED: `scifi_news` is a CONTENT-OWNED lane
  (`delivery_mode_for_meta(meta) == CONTENT_OWNED`, measured off the sweep's own
  ledger; `original` is `legacy`). Content-owned runners build their own cast and
  never run the writer's seeded picker, and `lock_cast()` is what applies the
  cameo -- so it cannot fire there. The empty `cast_contract` is the same
  deliberate decision: that block stamps `meta.episode_seed` and withholds
  `cast_seed`, because claiming one on a lane-owned cast detonated CastLock's
  replay before (`num_characters must be 1-6, got 0`).
  **THE OBVIOUS FIX IS THE WRONG ONE** -- routing content-owned lanes back
  through `lock_cast()` is precisely what that comment warns against. The repair
  belongs in the lane runner. **Operator row 15.**
  *Worst of the three by exposure:* nothing fails and nothing logs, so every
  `scifi_news` episode since the redesign has shipped with no cast contract.
* **PBUG-20260811-01 -- forcing the cameo kills the `scifi_fable2` writer on
  `scifi_news_pro`.** `pass 'script' failed after 4 attempt(s): markup ladder
  exhausted; BAD_LINE`. Reproduced at 30 AND 90 target words, so NOT a word
  squeeze; with the cameo on its natural roll the writer passes cleanly. Root
  cause not established.
* **PBUG-20260811-02 -- `scifi_news_pro` dies at node 92 with no materialized
  still for beat `music_closing_001`** (`still-spine handoff missing materialized
  scene still ... engine still_flat`), on the same profile where five other banks
  produced one. Seen ONCE. Re-run before treating the cause as understood.

## 10. The orphan-lifecycle pair intro and the CLOSED GGUF generation-deadline row (the registry half stays live in Section 1.6; the row's reachability lesson is lifted into Section 4)

### The orphan-lifecycle pair (deferred 2026-08-25, both DESIGN items, neither a grep-and-fix)

Both fall out of PBUG-20260825-04, whose four landed fixes shipped in
`fb67d059` after a full kibitz r1-r4 arc (Codex r2/r3, Cursor r4, Fable r1).
The arc found a new race in each of the first two cuts of the same fix, so
**do not treat either item below as mechanical** -- each is a genuine design
choice with more than one defensible answer, which per CLAUDE.md means a full
arc BEFORE code, not after.

- **THE GENERATION DEADLINE NOW COVERS THE GGUF LANE -- CLOSED 2026-08-25
  (evening).** Left this row in place rather than deleting it, because the
  DEFERRAL'S OWN SEVERITY CALL WAS WRONG and that is the reusable part. It
  said "VERIFY FIRST, it may be live rather than theoretical: check whether
  the current production technical-slot catalog row is `gguf_native`". That
  check was run and answered NO -- the canonical technical slot resolves to
  the transformers `google/gemma-4-12b-it` row -- and the honest-looking
  conclusion "latent, not live" was WRONG, because it asked only about the
  UNPROFILED canonical run. **Six committed `status="shipping"` profiles
  (`otr_g4_fastwan`, `_humo`, `_ltx_8gb`, `_ltx_audio_in`, `_ltx_video`,
  `_wan_ti2v`) pin `technical_model` to `unsloth/gemma-4-12b-it-GGUF`, and
  profile `status` is validated but is NOT an application gate** -- so real
  shipping runs were hitting the uncovered lane the whole time. *A
  reachability question answered against the default path only is not
  answered.*
  Shipped: deadline-conditional streaming in `_otr_gguf_backend` (no
  deadline -> the identical non-streaming call, `stream` absent entirely;
  a deadline -> stream and stop between chunks), plus ONE shared absolute
  `time.monotonic()` deadline computed BEFORE worker submission, a pre-call
  admission check, a parent recheck after `future.result()`, and the legacy
  `GemmaHeartbeatStreamer` migrated to the same clock. Receipts in
  `docs/PROD_BUG_LOG.md` (PBUG-20260825-04, deferral 1) and
  `kibitz-runs/2026-08-25-gguf-deadline/`.

## 11. The PARKED re-observe heading (its body rides Batch R5 live) and the original PARKED D2 heading

### PARKED -- unverified at HEAD, re-observe on the next real render legs
(The 2026-07-24 "after SFX" checkpoint is VOID -- SFX is parked. The re-observe
now rides whatever real render legs come next, D2 included.)

## PARKED -- D2 (renders have resumed; run when a render window is free for fail-hunting soak legs)

## 12. The original "After this queue" block (its content survives in the live plan's Section 2 deferred items and closing section)

## After this queue

One coder window at a time; every chunk = focused tests + full suite + Bug Bible
+ commit AND push + `HEAD == origin/v2.0-alpha`.

When the executable rows in the authoritative table above are exhausted,
continue with `ROADMAP.md`.
Lean-mean is not an item in this queue: `docs/LEAN_MEAN_CLEANUP.md` is its sole
current scope, blast-radius, coding-order, and verification authority.

Open judgment question (render-window, not a coder slot): the LOCAL mistral/gemma
writer matrix. The Sonnet arm of the creative-writer question is answered
(`docs/2026-07-17-model-bakeoff-scoreboard.md`); the local roster comparison
never ran.

## 13. Bug Bible receipts: the two promotions completed 2026-08-28 and the older promotion receipts

| ~~`PBUG-20260823-01` (preflight gate vocabulary collision)~~ | **DONE 2026-08-28.** Promoted as Bible `12.139`, survival-guide `7121254`, count 317 -> 318 (README bumped in all THREE places it appears -- a third hyphenated "317-entry" occurrence was missed on the first pass and caught by the regression suite's own count check before commit, not after). Coverage-index row added. Regression suite 22 passed / 26 skipped / 3 xfailed. No overlap found against `12.79` (VCS-diff narrowness) or the sys.modules stub-ownership entry -- genuine gap. |
| ~~`PBUG-20260823-02` (watcher timeout worded as render death)~~ | **DONE 2026-08-28.** Promoted as Bible `12.140`, survival-guide `67ad867`, count 318 -> 319 (all three README places). Coverage-index row added. Regression suite 22 passed / 26 skipped / 3 xfailed. Zero overlap hits in either file -- clean gap. |

**PROMOTED 2026-08-25 (evening): Bible `12.134`, survival-guide `6633ef6`, count
312 -> 313** (README bumped in all three places, coverage-index row added, Bible
regression re-run green 22/26/3). Source: **PBUG-20260825-04**, the
BUG-LOCAL-098 tripwire firing loud on a 4060 load that had in fact succeeded --
admissible because it surfaced as a real production traceback, promotable
because the fix is verified and its coverage is automatable
(`tests/test_bug098_orphan_race.py`). The reusable half is deliberately NOT
"the threshold was wrong": the guard sampled `torch.cuda.memory_allocated()`,
a PROCESS-WIDE counter, and reported the delta as one model's footprint, so an
abandoned worker freeing tensors concurrently drove it negative. *A diagnostic
that gates on a shared, process-wide quantity cannot make a claim about one
component of that process* -- and the tell is that the check LOOKS
model-scoped because it brackets one model's load. Checked against
`otr_coverage_index.yaml` and the Bible first: `12.46` covers the orphan
thread PINNING VRAM, which is the adjacent-but-different half, so this is a
genuine gap rather than a second entry for a covered class.
**Also fixed in the same commit:** `otr_coverage_index.yaml` had a
pre-existing unquoted `Root cause: ` colon-space on one record, so the index
-- whose entire purpose is to be machine-readable so the 4M-token scrape is
never repaid -- did not parse at all. Quoted; it now loads (429 records) and
its header metadata is re-synced to the Bible HEAD.

**PROMOTED 2026-08-18 (evening): Bible `12.114`, survival-guide `b9aada7e`, count
292 -> 293** (README bumped in all three places, coverage-index row added, Bible
regression re-run green 20/26/3). Source: **PBUG-20260817-08**, the Lemmy cameo
voice -- admissible because it surfaced on two live published episodes, and
promotable now only because the fix is verified. The entry carries TWO reusable
halves: *a reservation that exists as a convention in one subsystem is invisible
to another subsystem enumerating the same catalogue*, and the diagnostic trap
that cost more time than the bug -- ***a post-fix sighting is not proof the fix
failed; check process age first.*** Its verify section also pins *sweep the
SELECTOR, not the helper* and *count a corpus by ROLE before concluding anything
about pool concentration*.

**NOTHING ELSE WAS PROMOTED 2026-08-18, deliberately, and
the reasoning is worth keeping.** The evidence-guards work EXECUTED an existing entry's verify
steps rather than discovering a new class: `12.111` verify step 3 is what turned
up G1's partial guard, and `12.111`'s own `cause` section already describes that
failure verbatim -- *"Refusing only when a specific file (`MANIFEST.json`) exists
leaves every sibling artifact unprotected ... the separate `_KEY` directory"*. An
entry that predicts the defect you then find does not need a second entry.
The window's other finding -- `engine_impl_version` structurally unfillable
because no adapter defines `impl_version` -- is **static-audit only**, so the
admission rule bars it regardless of how real it is. And nothing had actually
rotted: all eleven cited artifacts re-hashed clean, so there is no live artifact
to admit. **Preventive work with no live failure produces no Bible row.**

**NOTHING WAS PROMOTED 2026-08-17 (item B window), deliberately.** The window's
findings are all static-audit -- the positive-prose ban, the seven-call-site
video negative, the B6 gate gap, the traceroute's coverage blind spot -- and the
admission rule reserves the Bible for defects verified by a live artifact.
`PBUG-20260817-01` was re-proved on pixels here but is ALREADY covered by Bible
`12.108`, whose tag list literally includes `prompt-audit-cannot-see-pixels`.
Bible stays **287**, README stays 287 in all three places; the Three-File
Contract is intact.
