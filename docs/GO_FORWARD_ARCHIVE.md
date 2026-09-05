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

# ARCHIVED 2026-09-01 -- the forward-only pass

Cut from `docs/GO_FORWARD_PLAN.md` on the operator's instruction ("clean up the go
forward so it's strategized, clean and only go forward, not a log"). Every block below
is VERBATIM; nothing was deleted. Open work that these blocks carried was re-homed in
the live plan before the cut (the block headings say where).

## 1. Plan header history (the 2026-08-23 split note, the 2026-08-28 lean-and-mean note, and the HOW TO READ THIS FILE block; their surviving rule is the new header's one paragraph)

Split 2026-08-23 on the operator's instruction: *"go forward should only have the
go forward plans. Only."*

**Reorganized 2026-08-28 (the lean-and-mean pass)** on the operator's
instruction: *"clean up the go forward so it's lean and mean, and then analyze
and reorganize it on the most logical [basis] so we can code the most and test
the least."* The organizing fact: most items here need a LIVE RENDER LEG to
prove, and a leg costs 1-3 HOURS on one GPU -- testing is the bottleneck,
coding is not. So the file is now three working queues plus support sections:

* **Section 1 -- NO-RENDER WORK.** Everything provable without a GPU leg,
  front-loaded so it can all ship in coding sittings.
* **Section 2 -- RENDER WORK, BATCHED BY THE LEG THAT PROVES IT.** One leg
  proves several items; each batch says so explicitly.
* **Section 3 -- WAITING ON THE OPERATOR.** Every open question in one list,
  answerable in one pass instead of interrupt-by-interrupt.
* **Section 4** -- rulings lifted verbatim from archived blocks.
  **Section 5** -- parked/deferred. **Section 6** -- standing traps.

Receipts from the 2026-08-27/28 sessions and superseded queue blocks moved to
the archive VERBATIM, nothing deleted; every operator ruling they carried was
lifted into Section 4 FIRST. The standing warning from the 2026-08-24 prune
(now itself in the archive) still binds: **recording what you just finished
inside a section headed OPEN is the easiest way to make this file lie.**
Receipts belong in `docs/PROD_BUG_LOG.md` and the archive; rulings belong in
`docs/OTR_STANDING_RULINGS.md`; only what is still TO DO belongs here.

## HOW TO READ THIS FILE (three files since 2026-08-23)

**THE PLAN IS THIS FILE AND IT IS ONLY OPEN WORK.** Operator: *"go forward
should only have the go forward plans. Only."* Two companions carry what used to
be tangled in here:

* `docs/OTR_STANDING_RULINGS.md` -- the laws, the standing operator rulings, the
  review routing, the model/credit ladder, how to talk to the operator, the
  obs-path override, window packing, tombstones and pointers. **Read it. It is
  not optional and it is not history** -- it is the set of constraints the next
  piece of work has to satisfy. When something in this plan says "THE LAW" or
  "the REVIEW ROUTING block", that is where it now lives.
* `docs/GO_FORWARD_ARCHIVE.md` -- closed receipts, verbatim. **Not read to
  resume.**

`CLAUDE.md` is unchanged and remains the highest authority.

## 2. THE CURRENT STEP preamble of 2026-08-31 (two days on rented hardware; the constraints and the queue stayed live)

*Rewritten 2026-08-31. Receipts live in `docs/PROD_BUG_LOG.md` and the archive.
Pod knowledge: `docs/RUNPOD_INSTALL.md`. Machine guide: `docs/MACHINE_MATRIX.md`.*

Two days on rented hardware found three OTR bugs the dev box cannot surface
about itself -- all fixed, receipts filed -- left the provisioner incomplete,
and measured nothing of the lane matrix. The 5080 kept shipping throughout.

## 3. Coding-order rows 1-2 (sanctioned gap C0-C7 and Spandrel), both DONE 2026-08-28

1. ~~Sanctioned gap~~ **DONE 2026-08-28, fully.** C0-C6 shipped `cd96e9b3`;
   C7 proved the chain END TO END (`83be6c74`) -- an all-refused episode
   floors every beat and publishes degraded, an unsanctioned absence still
   fails loud at the spine. The predicted hole was already closed by C2.
2. ~~Spandrel~~ **DONE 2026-08-28** -- neither edge case reproduced; the
   untested warn-cap got tests (`ef957ecf`).

## 4. The dead-code campaign narrative (rounds 1-4 and the V4 adjudication receipts; the live rule and the V5 pointer stayed in the plan)

**RUNNING BESIDE THE ORDER -- THE DEAD-CODE CAMPAIGN (operator standing
instruction 2026-08-28: keep hunting "until there are no more dead code
candidates"; STOP RULE = two independent blind deep sweeps returning zero
CONFIRMED findings).** FIVE rounds done through 2026-08-28 evening: rounds
1-3 (~2,600 lines, eleven lying comments, two real bug fixes, seven widget
migrations verified by the operator's own UI open), round 4 (`d084585d` --
`_voice_backends` deleted, Chatterbox/Dia `_load_wav` consolidated,
cue-manifest helpers retired), and the V4 master report fully adjudicated
and executed (`18ff0533` + `0ce621a0`: 18/19 findings CONFIRMED by a
7-verifier grounding pass, ten ready findings + six operator-ruled
retirements executed, three operator-ruled KEEPs recorded, the 3D spike lab
deferred to the 3D-retirement boundary). The V5 sweep (18 findings,
`docs/2026-08-28-dead-code-hunt-v5/`) is under adjudication. The live hunt
prompt is `docs/DEAD_CODE_HUNT_PROMPT_V5.md`. A SEPARATE pre-ship pass, the
KNOB CENSUS (`docs/KNOB_CENSUS_PROMPT.md`), tables every WORKING widget
against corpus evidence + a think-like-a-human judgment; the operator rules
per row -- census informs what the 4060 template pins.

## 16. Section 1.0 THE 4060 FRICTIONLESS SET as it stood on 2026-08-28 (kokoro was declared 2026-08-29, the template shipped 2026-08-29 and moved into workflows/ on 2026-09-01; the two remaining bullets were restated in the live plan)

Four small items, all verified against the tree on 2026-08-28, plus one trap:

* **`workflows/variants/otr_4060_floor.json`: `"quant_policy": "none"` ->
  `"bnb_nf4"`.** One line. The single thing blocking the 4060 from loading
  its 10 GB writer; the old justification for "none" is provably stale
  (accelerate + bitsandbytes are declared deps now).
* **Declare `kokoro` in `requirements.txt`** and fix the stale repo hint at
  `nodes/_otr_audio_engines/eng_kokoro.py:139` (`1038lab/KokoroTTS` in the
  error message vs `hexgrad/Kokoro-82M` actually loaded at `:159`). Two
  lines; removes the biggest fresh-install voice friction.
* **Ship a template in `example_workflows/`** -- the 4060 floor graph
  already exists as a variant; it just does not ship where "download and
  click run" looks.
* **README model table** from the compatibility workbook's Baseline Combos
  tab (`outputs/20260828-ungated-models/` -- the LIVING fact sheet: edit
  cells in place, never add a changelog tab). Include the HF-token
  two-tier story: defaults need no account ever; gated upgrades = account +
  license click-through + `HF_TOKEN` once (the resolver already finds it).
* **THE TRAP: `pyproject.toml` edits AUTO-FIRE a registry publish.** The
  `kokoro` line also belongs in its static deps -- but that edit WAITS and
  rides the deliberate republish: operator deletes the flagged node entry
  (his click, not ours) -> ONE commit bumps the version + adds kokoro to
  pyproject -> clean publish -> clean 4060 install of the shipped template.

Waiting on externals, not on us: Codex deep research on the HF-token
problem statement (`docs/2026-08-28-hf-token-feature/PROBLEM_STATEMENT.md`)
and the operator's registry node-delete click.

## 5. Section 1.1 THE SANCTIONED-GAP CONTROL PATH (C0-C6 shipped cd96e9b3, C7 proven 83be6c74; the row said 'archive on the next tidy')

### 1.1 THE SANCTIONED-GAP CONTROL PATH -- r2, the coding plan, is the next step

**Live-proven necessary on 2026-08-26.** Ideogram is not seed-deterministic:
after the music-card fix it went from refusing every music card to **6 of 7**,
and that ONE refusal still killed a 30-minute episode at the still-spine gate.
No amount of prompt work takes a stochastic refusal to zero.

The dispatcher already says "the episode continues" and the composite already
floors an `exists=False` row -- **nothing in between mints that row**. Spec and
r1 judgment: `kibitz-runs/2026-08-25-model-refusal-required-still/`.
Accounting for it landed 2026-08-26 (`a2837b05`) and is deliberately inert
until this exists.

**THE BLOCKER IS CLEARED -- r2 CAN RUN (2026-08-27).** r1 ended on one open
item it refused to decide for the operator: what an episode should do when
EVERY required still is sanctioned-gapped, given that node 92's success check
is `clip_count > 0`. **Asked and ruled 2026-08-27: it PUBLISHES.** The full
ruling, its reasoning and -- importantly -- what it does NOT license are in
`docs/OTR_STANDING_RULINGS.md` under *"AN ALL-REFUSED EPISODE STILL PUBLISHES"*.
Read it there before r2, because the ruling is narrower than its headline: it
permits publishing an all-refused episode, it does NOT permit REPORTING one as
a clean render, and the `required_scene_targets` ledger-completeness law is
untouched.

**SHIPPED 2026-08-28 (`cd96e9b3`). The control path EXISTS.** C0-C6 of the
hardened plan landed together, full suite 12390 passed / 121 skipped /
1 xfailed, no regressions. **SUPERSEDED: C7 was proven END TO END in `83be6c74` (verified 2026-08-31) -- this row's remaining work is closed. Archive on the next tidy.** The original text follows: what remained on this row was C7 ONLY -- the
end-to-end test through the real spine and composite, which is EXPECTED to
find a hole (`assemble_silent_timeline` raises "manifest has no renderable
beats" on an all-gap episode; that hole is the work). Everything else here is
history and moves to the archive on the next tidy.

The panel earned its keep: all three lanes independently returned NO on the
same finding -- `validate_and_repair_still_spine` runs ONE LINE before
`run_real_episode`, so every downstream change was dead code until C0 taught
the validator to accept a gap. They also caught the driver's own claim that
the `a2837b05` accounting was correct and out of scope: it is correct only
while no gap row can arrive, and the proposed predicate would have reported a
crashed all-missing render as publishable-degraded.

**The original next step was r2, the coding plan**, per r1's own stated roster
(r1 Codex+Fable -> r2 Codex -> r3 Codex+Cursor -> r4 agy Pro). Nothing else
about this row is waiting on the operator.

**r1 WAS WRITTEN 2026-08-25 AND THE TREE MOVED THE NEXT DAY. RE-GROUNDED
2026-08-27 against HEAD -- do not hand r2 the stale finding list.** `a2837b05`
landed on 2026-08-26 and added 85 lines to `otr_video_render_batch.py`, so one
of r1's four findings is already closed and one of the survivors got SHARPER.

| r1 finding | status at HEAD, verified by reading the file |
|---|---|
| 1. Nothing mints the `exists=False` row between a sanctioned dispatch and the spine | **STILL THE ROW.** This is the item itself. |
| 2. The skip branch never reaches the renderer loop | **STILL OPEN**, unverified in this pass -- r2 grounds it. |
| 3a. The manifest loop counts a gap as a delivered receipt (`:146-150`) | **ALREADY FIXED** by `a2837b05`. `_clip_delivered_motion(clip)` (`:134-153`, `exists` alone, deliberately) now routes an undelivered beat to `sanctioned_gap_shot_ids` at `:213` instead of minting a receipt for it. Do NOT re-fix this. |
| 3b. `delivered_frames_ok` is True over an absent clip (`:750-770`) | **STILL LIVE**, and here is the exact mechanism: a gap has no `source == "clip"` segment, so `segs` is empty, `status` becomes `no_clip_segment` -- and `no_clip_segment` is the one status that flips NOTHING. `ok_all` is only cleared by `held_last_frame`, or by `not positioned and segs and delivered != tgt`, whose `segs` guard is falsy for exactly this case (`otr_silent_composite.py:766-769`). |
| 4. An all-refused episode reports FAILURE (`clip_count > 0`) | **STILL LIVE, now at `otr_video_render_batch.py:640`** -- and `a2837b05` made it BITE rather than merely lurk. |

**FINDING 4 DESERVES ITS OWN PARAGRAPH, because the fix for 3a is what armed
it.** `clip_count` is `len(receipts)` (`:129`), and since `a2837b05` receipts
correctly EXCLUDE sanctioned gaps. So an all-refused episode now has a genuinely
empty receipt list and `"ok": manifest["clip_count"] > 0` genuinely evaluates
False. Before that commit the gap rows were counted as receipts, so the same
episode would have reported ok=True by ACCIDENT -- for the wrong reason, off a
receipt that lied. **The correct accounting collides head-on with the 2026-08-27
ruling that such an episode must publish**, which is not a regression in
`a2837b05` but the point at which an existing contradiction became honest enough
to see. r2 fixes the success predicate, NOT the accounting.

**The payload-never-empty guarantee already anticipates this** and is worth
reading before designing (`otr_video_render_batch.py:203-209`):
`OTR_CreditsRoll._require` rejects `{}`/`[]`/`None`/`""`, so an all-gap episode
returning an empty payload would convert a publishable degraded episode into a
hard mux-time failure -- "the exact outcome the sanctioned gap exists to
prevent", in the code's own words. Whoever writes r2 starts from there.

## 6. Section 1.3 gender ladder: the r2 and r3 review narratives and the answered Section 3 question B cross-reference (the spec pointer, the two 2026-08-28 rulings and the next step stayed live)

**Codex r2 verdict: NO. Eleven must-fixes, at `kibitz-runs/2026-08-05-gender-ladder/r2/codex.md`.**
The diagnosis survived; the mechanism did not. The three that matter:

1. **The web search would silently do nothing.** The spec passes a tools/plugins
   argument to `OpenRouterBackend.generate`, which swallows unknown kwargs through
   `**_ignored` -- no error, no search, a confident answer from a model that never
   looked. That is the same silent-no-op class as the defect above.
2. **"LLM extraction over the FULL unit text" cannot run.** `beckoning_fair_one.txt`
   is 143,176 bytes and 58 of 65 source files exceed 12,000 bytes, against a
   32,768 estimated-token per-call cap.
3. **Blanket surname aliases are identity-unsafe.** Two rows sharing a surname
   with the same gender currently produce a confident pin rather than an
   abstention.

**r3 RAN 2026-08-06 and it is STILL NO** -- Codex NO with 7 must-fixes, agy
yes-with-fixes with 9. That is three NOs across two rounds, and r3's findings invalidate
the CODING PLAN rather than its line numbers, so the standing re-ground rule applies:
**the next step is a SPEC REWRITE folding r2 + r3, then r3 again. Not r4.**

Both lanes independently found (a) a manifest sequencing deadlock -- the stamper is
specced to run per-unit inside the vendor fetch loop, but the manifest is written only
AFTER the loop, so it can never see the unit it was called for; and (b)
`RosterGenderVerdict` has no `gender_source` / `gender_confidence` fields, so the
ladder's whole output cannot be carried without changing every verdict-construction
path. Codex also confirmed the r2 finding that the OpenRouter backend still has no
web/plugin parameter, so the web-search tier would silently do nothing.

Judgment: `kibitz-runs/2026-08-06-2026-08-06-gender-ladder-r3/r3/judgment.md` (LOCAL
ONLY). **Trap: commit `496d9d57` inserted ~90 lines near the top of
`nodes/_otr_roster_gender.py`, so every cite into that file in the r3 review has
SHIFTED. Re-pin before acting.** The rewrite should also CONSUME the
`normalize_gender` boundary item 8 installed there rather than adding a second
normalization path.

**Found while grounding, and it reopens an operator ruling:** 32 of 85 Shakespeare
roster rows are `unknown` TODAY -- 38% of the lane assumed solved. Comedy of Errors
ships 7 characters, every one unknown. The narrower ruling that fits the evidence:
Shakespeare's KNOWN rows stay untouchable, but tiers 3-4 may fill only its
`unknown` rows. That fixes 32 rows without ever second-guessing a parsed
dramatis personae. **Operator decision, not a driver call** -- cross-listed as
Section 3, question B.

## 7. Sections 1.4 (OTR_ENABLE_LTX_I2V retired 2026-08-28) and 1.5 (learned-upscale hardening closed 2026-08-28; its ruling is in Section 4)

### 1.4 ~~RETIRE `OTR_ENABLE_LTX_I2V`~~ -- DONE 2026-08-28, tombstone

Shipped the same night it was scoped: `192d3aa2` (family = image_to_video,
declarative still requirement, both driver maps, the enum token, content
oracle) + `f9eab3f6` and `063e8c0b` (the seven test sites the full suite then
found, including two prompt-only fixtures that now SUPPLY a still instead of
disabling the requirement, and the soak rotation's text_to_video chair
re-seated with `animatediff15_v3_haunted_video`). Full suite green. The
operator's ruling is now simply how the lane works; there is no flag to
retire and no text-only LTX path to reason about.

### 1.5 ~~NARROW LEARNED-UPSCALE HARDENING~~ -- CLOSED 2026-08-28: NEITHER EDGE CASE REPRODUCES

The row was conditional -- "harden the two `SpandrelEsrgan._resolve_model`
edge cases **if still reproducible**" -- so the first step was the
reproduction, not a fix. Both were attempted on the real module (CPU, no
model load) and both behave correctly at HEAD:

| edge case | result |
|---|---|
| `folder_paths.get_folder_paths` raises a non-ImportError | Resolver survives and falls back to the repo-relative dir. **One warning across five calls**, not five -- the `_RESOLVE_WARNED` cap works. |
| An unreadable candidate (`PermissionError`) | **Propagates** rather than masquerading as absence, exactly as the docstring claims. Already pinned by `test_upscale_cache_fingerprint.py:243-249`. |
| (checked while there) shallow checkout, <5 path parents | No `IndexError`; resolves. |

The hardening this row asked for was done in earlier passes and the code
documents it -- including an explicit note that a classification pass was
written, found unreachable, and REMOVED rather than kept as reassuring dead
code.

**What the verification did surface, and it is now fixed:** the warn cap
itself had NO test. Removing the guard that keeps a broken `folder_paths`
from logging once per prompt evaluation (it is called from `IS_CHANGED`)
would have failed nothing. Three tests added in
`tests/test_upscale_cache_fingerprint.py`: warns once not once-per-call, the
set stays bounded, and the repo fallback still resolves.

The ruling attached to this row stands and is lifted into Section 4: **the
multi-GPU learned-upscale stage itself is CLOSED and must not be reopened.**

## 8. Section 2 Batch R0 (the six overnight 1-act legs; closed 2026-08-31)

### Batch R0 -- CLOSED (2026-08-31). Those legs published and were judged; the pointer to THE CURRENT STEP is dead because the rewrite removed that content. Archive on the next tidy.

### Batch R0 -- ALREADY RENDERED: the six overnight 1-act legs. One morning pass over `otr/obs/` proves six lanes

See THE CURRENT STEP at the top of this file. One pass proves: the mime
prompt, the foley bed under dialogue, plain ltx25 video, H3 silent, H3
audio-in, and the ltx_audio_in baseline -- and may satisfy Batch R1 for free
(check the H3 legs' prompt receipts before spending R1's leg).

## 9. Section 2 Batch R2: the 2026-08-27 re-grounding narrative (what was measured, where the SCENE/PORTRAIT claim came from, two corrections); the measurement step and the preserved fork stayed live

**THIS ROW USED TO ORDER A FULL ARC. IT NO LONGER DOES, because the premise it
rested on was never measured.** Re-grounded 2026-08-27 against the evidence
file, the PBUG and the canonical JSON, after the operator asked the reasonable
question -- *"i am not aware of this bug, maybe it has been fixed, i wonder when
it came up"* -- and the honest answer turned out to be worth the check.

**WHAT IS ACTUALLY MEASURED.** All SIX refusal events in
`docs/2026-08-26-ideogram4-card-refusal-evidence.md` are the same beat type:

```
ideogram4_local still_music_opening_001 min=79.0 std=10.5
ideogram4_local still_music_closing_001 min=80.0 std=10.5
ideogram4_local still_music_closing_001 min=78.0 std=10.2
ideogram4_local still_music_opening_001 min=80.0 std=10.2
ideogram4_local still_music_closing_001 min=80.0 std=10.3
ideogram4_local still_music_opening_001 min=87.0 std=10.5
```

Zero SCENE refusals. Zero PORTRAIT refusals. **And the music route -- the only
route that ever refused -- was FIXED on 2026-08-26 (`ae7e7b6a`) and proven on
two published episodes with zero refusals, on the weakest writer and the
strongest alike.**

**WHERE THE SCENE/PORTRAIT CLAIM CAME FROM.** `ae7e7b6a`'s own message: the two
routes are *"still `elements: []` and therefore still expected to refuse"*.
**Expected to. Not observed to.** It is an inference from structural similarity
to a route that has since been repaired, and it hardened into this row as
though it were a finding.

**TWO MORE CORRECTIONS TO WHAT THIS ROW USED TO SAY.**
* *"the three lanes production renders with"* is wrong. `ideogram4_local`
  appears **ZERO times** in `workflows/otr_canonical.json`; the canonical names
  `z_image_turbo`, three times. The engine is OPT-IN by construction --
  `default_roles = ()`, and its own comment says *"z_image_turbo stays the
  shipped default; no model is 'primary'"*. It ran in the sweep only because
  profile `otr_soak_llmsweep_02` selects it deliberately.
* The other four local engines went **91 mints, zero refusals** across that same
  sweep (flux2_klein 35/0, z_image_turbo 32/0, flux_gen1 16/0, lumina_image
  8/0). Nothing about this is a general stills defect.

## 10. Section 3 question J as it stood on 2026-08-31 (delete-and-republish, who finds out why; superseded by the 2026-09-01 secret-scanner finding)

### J. THE REGISTRY IS FLAGGED -- two questions, and this is the third time it has been raised

**Verified live 2026-08-31:** `latest_version` is NULL, two versions, **zero
Active**, both `NodeVersionStatusFlagged`. Nobody can install OTR through
ComfyUI-Manager by any route -- not on a pod, not on a desktop. For a pack
intended to go open source, that is the single largest adoption blocker, and it
has been sitting as a sub-bullet of the capstone behind every other bug.

README now leads with the git clone instead (`786b3fa4`), so a new user has a
route that works. These two remain, and neither is scheduled because neither has
been asked:

* **J1. Delete and republish now, yes or no?** Node-delete is a HARD delete that
  frees every version string for reuse; version-delete is soft and BURNS the
  string. Only the operator's browser session can do it -- the publish token
  returns 401.
* **J2. Who owns finding out WHY alpha.13 and alpha.14 were flagged?** The
  scanner is a private repo and its findings are not visible to us. Without an
  answer, a republish may simply be flagged again for the same unknown reason,
  and we would have spent a version string to learn nothing. (answer these in ONE pass)

Nothing in this section needs a render or a coder until answered. Defaults are
stated where a default exists; silence keeps the default.

## 11. Section 3 questions B and C (both answered 2026-08-28; the rulings live at the top of Section 1.3 and in Batch R3)

* **(B) ANSWERED 2026-08-28: YES -- fill only the unknowns.** KNOWN rows stay
  untouchable. Ruling recorded at the top of Section 1.3 together with the
  web-tier replacement (LLM decides, cached name index).
* **(C) ANSWERED BY EVENTS 2026-08-28: the numbering is retired.** The LEMMY
  trio it tracked is fully closed on live evidence (see Batch R3's note);
  nothing remains for a phase number to point at.

## 12. The 2026-09-01 ship-audit session entry and its evening addendum (decisions, landed commits 64d81ca7 / 8cfe0007 / bef7928d / 6e54f9ae, Bible 12.144, the clean-room leg result). Every open item was re-homed: registry -> Section 3 J; the 3.13 voice -> Section 3 K; the 8 GB abort -> Section 3 L and Batch R7; the non-mechanical blockers -> Section 1.9; the parked AnimateDiff image input -> Section 5

## 2026-09-01 -- Ship audit, registry flag, and the 4060 clean-room install (Fable 5.1 session)

Receipts: `docs/ship-audit-2026-09-01/` -- SHIP_LIST.md (71 confirmed + 51 disputed
findings, ranked), NOVELTY_DELTA.md (32 new to the record, 14 known-open, 25 adjacent,
0 regressions), FINDINGS.json (every finding with file:line and both reviewers' notes),
4060_CLEANROOM.md (the friction log), pyproject_alpha15.patch (the release commit, NOT applied).

### DECISIONS (settled by evidence, no operator input needed)
1. **The registry gate is a SECRET scanner, not an exec linter.** Comfy-Org's own backend
   (`registry_svc.go:1392-1455`) posts the zip to a `SecretScannerURL`; any non-empty
   response = Flagged, and the reason goes to their private Discord. This is why the
   08-28/08-29 exec/subprocess hunt in `.comfyignore` found nothing.
2. The one shipped string that matches a published secret rule was README.md:164
   (`hf_` + 38 x's, gitleaks' huggingface window). It entered on 2026-08-29, so it cannot
   explain the FIRST flags (alpha.9-.11, 08-25/26), but it IS in flagged alpha.14 and absent
   from Active alpha.8. FIXED: now `hf_your_token_here` with a one-line reason.
3. alpha.15 = ONE operator push (pyproject.toml is a release trigger): the prepared patch
   bumps the version, adds pycairo/pillow/aiohttp (imported by shipped nodes, declared only in
   requirements.txt), marks bitsandbytes `sys_platform != 'darwin'` and kokoro
   `python_version < '3.13'`. If alpha.15 is Flagged too, republish the alpha.8 tree
   (commit e44235f5) byte-identical as alpha.16 as the control: Active means the trigger is
   in the alpha.9+ delta and can be bisected; Flagged means the ruleset moved and that is
   the evidence to hand Comfy-Org. Never version-delete (soft delete burns the string).
4. LANDED this session (mechanical, one right answer, 207 targeted tests + Bug Bible green,
   5080 path proven unchanged by printing before/after): unguarded `torch.cuda.ipc_collect()`
   in `load_llm` (`_otr_model_loader.py`, killed every CUDA-less writer), `_detect_host()`
   now emits "linux" (`_otr_workflow_validator.py`, both ROCm profiles failed their own stamp
   on the host they target; "any" profiles unaffected), GGUF `n_gpu_layers` default now
   offloads on mps as well as cuda (`_otr_gguf_backend.py`, two sites; cuda still -1, cpu 0).
5. README 2b now names ComfyUI-GGUF (pinned commit + `patches/` patch) for `ltx25_*` and
   `flux2_klein`, states that HuMo 1.7B / Wan / ltx_8gb / H3 / still / viz need NO extra pack
   on ComfyUI 0.34+ (verified on a clean portable), and carries the Python 3.13 kokoro note.

### THE 4060 CLEAN ROOM (fresh portable v0.34.0, Python 3.13.14, fresh HF cache, git clone)
6. **BLOCKER, NEW: `pip install -r requirements.txt` failed outright on Python 3.13** --
   the interpreter ComfyUI Desktop and the portable both ship. `kokoro>=0.7.16` is not
   installable on 3.13 by any pip route (numpy==1.26.4 pin on 0.7.16; Requires-Python <3.13
   on every newer kokoro/misaki; spacy/thinc/blis source builds behind `misaki[en]`). pip is
   all-or-nothing, so NOTHING installed. FIXED the install half: the kokoro line now carries
   `python_version < "3.13"`; the other 17 requirements install.
7. **OPERATOR DECISION -- what is the default voice on Python 3.13?** Kokoro is the shipped
   announcer default and the 8 GB class char voice, and it now silently does not exist on
   every mainstream Windows install. Options, in rough order of effort: (a) bark for both
   voices in the 8 GB / 3.13 variants (installs everywhere, auto-downloads, slower, older
   sound); (b) a `kokoro-onnx` backend for the kokoro engine (pure onnxruntime, supports
   3.13, same voices, new code, a design item -> kibitz arc); (c) tell 3.13 users to use
   Python 3.12 (not possible on Desktop/portable). The clean-room leg used (a).
8. Stock boot is clean on a stock console (no PYTHONUTF8): 25/25 nodes, 23.6 s prestartup
   (it fetches 28 Kokoro voice files at boot even when kokoro cannot be installed), 4.6 s
   import. The pack's weight preflight refused the queue BEFORE the writer and named every
   missing file -- good; its restart hint cites `scripts/_otr_headless_model_paths.yaml`,
   which registry installs do not carry.
9. Node packs: LTX 2.5 needs ComfyUI-GGUF at 6ea2651e + the patch + a restart (README said
   nothing until now). HuMo 1.7B needs no pack at all. H3 is operator-only and not a
   from-nothing candidate.
10. Weights: no 8 GB profile exists for LTX 2.5 / HuMo 1.7B; their pinned tiers live only in
    the pod provisioner (does not ship) and the Windows fetcher has no lane for them and
    defaults its models root to C:\ComfyUI-Models. This test fetched them with the
    provisioner's pins (bytes + sha256 verified): LTX 2.5 23.9 GB, HuMo 1.7B 13.6 GB,
    z_image int8 14.6 GB, writer E2B 6.0 GB.
11. Leg results: see 4060_CLEANROOM.md "Leg results" (appended as they land).

### NEXT STEPS
12. 5080: apply `pyproject_alpha15.patch` and push (the release commit; operator's call on
    timing). Then the alpha.8 control if Flagged.
13. 5080: the remaining blockers from SHIP_LIST.md section 2 that are NOT mechanical and
    want a kibitz arc: indextts2 as the canonical default char voice with reference WAVs
    that never ship; the 8 GB `ltx_8gb` profiles paired with a 14.5 GB writer;
    `needs_fp8_te`/`needs_fp4_te` never consulted by `_fit_reason`; the janitor
    `audio_slices` sweep (9.3 GB, 6.7 s per boot -- three lines, but it widens what gets
    auto-deleted, so it needs a test).
14. 5080: stop citing scripts/ and docs/ in shipped error text -- 8 LTX sites, 3 TTS worker
    paths, spandrel, cloud image, 16 OpenRouter/credits sites (SHIP_LIST.md section 3).
15. 4060: after the clean-room legs, fold the measured 8 GB results into
    `config/machine_classes.json` (only what actually rendered), regenerate
    docs/MACHINE_MATRIX.md, and the README newbie pass. Do not advertise a lane the
    clean room did not finish.
16. Mac/AMD (horizon, operator not hopeful): SHIP_LIST.md section 6 lists the six-step
    audio-only Mac path; four of the six are now landed or prepared (guards above, bnb
    marker, pyproject deps). Still open: llama-cpp-python install text per platform,
    the credits font resolver on macOS.

### PARKED (operator idea, 2026-09-01)
17. Image input for the AnimateDiff haunted lane (an i2v anchor for the 8 GB floor). Not
    started; ship-readiness first.

### ADDENDUM 2026-09-01 (evening) -- what landed after items 1-17, and what is left

Reviewed from HEAD 93a37aa1 (Codex's consolidation) after its all-clear. Docs-only commit.

**Landed and pushed (each reviewed by one independent Sonnet pass, lockstep-verified):**
- `64d81ca7` -- items 2, 4, 5, 6 above: README token literal, three platform guards,
  kokoro `python_version < "3.13"` marker, README 2b node-pack table, ship-audit receipts.
  Tests: 207 + 44 targeted, Bug Bible regression 22 passed.
- `8cfe0007` -- one template folder (`example_workflows/` collided with `workflows/` at the
  same gallery URL, first mount won, `otr_canonical` listed and 404'd on click; now
  `workflows/otr_4060_floor.json` + `tests/test_workflow_templates_single_folder.py`);
  the three TTS worker scripts ship (`.comfyignore` `scripts/*` + negations, bundle
  simulated with pathspec: exactly three scripts/ files ship); `.comfyignore`, `.gitignore`,
  `.gitattributes`, `SKILL.md`, `_START_HERE.md`, `workflows/external_examples/` stop
  shipping; item 14 DONE (12 nodes/ files stop citing unshipped scripts/ and docs/, GitHub
  URLs + inlined steps; wrapper_bridge names ComfyUI-GGUF for the GGUF loaders, test added).
  Tests: 20 + 164 targeted, `build_variants --check` 90/0.
  CORRECTION by Codex (`bef7928d`): that commit also re-worded eng_indextts2.py, which is a
  signed voice-route fingerprint (9 guards failed). Codex restored the file byte-for-byte
  and ships the installer instead (`!scripts/_otr_indextts2_install.ps1`). Lesson kept in
  memory: grep tests/ for sha256 before touching anything under nodes/, run the full suite.
- `6e54f9ae` -- item 16 mostly DONE: `OTR_CREDITS_FONT` override + macOS font candidates
  (5080 still resolves consola.ttf), platform-aware llama-cpp-python hint (darwin/linux,
  upstream's `GGML_METAL` / `GGML_HIP` flags, pin labelled a Windows measurement) + README
  table, BOM-safe HF token-file read. Tests: 161 + 22 targeted.
- Bug Bible `8956de15` (Bible repo main): entry 12.144 + coverage-index row for
  PBUG-20260901-04 (interpreter-capped optional dependency vetoes the whole install).

**4060 clean-room leg results (item 11):** fresh portable v0.34.0, Python 3.13.14, stock
launch. Writer (E2B, 26 min incl. 6 GB download), bark voices, image prompts all ran. The
FIRST z_image_turbo still (int8 convrot) aborted the whole ComfyUI process at sampler step
5/8 under DynamicVRAM (`Fatal Python error: Aborted`, stack in comfy/ldm/lumina/model.py;
log kept at C:\OTR-CleanRoom\server_run1_zimage_abort.log). Same shape as the drill's
PBUG-03, now reproduced on a never-touched install. The video stage never started, so
LTX 2.5 and HuMo 1.7B on 8 GB remain UNMEASURED. A retry with `--disable-dynamic-vram
--lowvram` was queued and then stopped on the operator's instruction; the clean room stays
on disk (70.9 GB) for that retry. Matrix unchanged: nothing rendered end to end.

**Remaining blockers, ranked:**
1. Registry: apply `docs/ship-audit-2026-09-01/pyproject_alpha15.patch` and push (operator;
   release trigger). If Flagged again, republish the alpha.8 tree byte-identical as the
   control (item 3).
2. Default voice on Python 3.13 (item 7) -- unchanged, operator decision.
3. 8 GB: the shipped default image engine aborts the process under a stock launch. Either
   the 8 GB dropdown set avoids z_image stills, or the launch flags become documented
   requirements. Needs the ComfyUI-side report with the faulthandler stack.
4. `otr_canonical.json` default char voice is indextts2 with reference WAVs that never ship
   (item 13); the `ltx_8gb` profiles pair an 8 GB engine with a 14.5 GB writer (item 13).
5. Design items still needing an arc: `needs_fp8_te`/`needs_fp4_te` in `_fit_reason`; the
   janitor `audio_slices` sweep (9.3 GB, 6.7 s per boot); cloud spend ceilings on the
   cpu_floor / otr_mac_mps Google routes (SHIP_LIST.md sections 2, 3, 5).

**Next action:** operator pushes alpha.15. Then, when the 4060 is free, rerun
`otr_cleanroom_8gb_ltx25` from C:\OTR-CleanRoom with the low-VRAM flags and record the
first measured 8 GB result for a non-AnimateDiff video lane.

## 13. Batch R3: the ALREADY DONE paragraph (clean sweep, PASS-tier gate, LLM preflight guide, Q6_K removal; ruling in OTR_STANDING_RULINGS)

**ALREADY DONE (pushed, green, lockstep-verified):** the clean sweep itself
(`Qwen/Qwen2.5-14B-Instruct` ripped with its blast radius), the PASS-tier
invariant gate, `docs/LLM_PREFLIGHT_GUIDE.md`, and the `Q6_K` dropdown removal.
The ruling is recorded in `docs/OTR_STANDING_RULINGS.md` ("ONLY EASY-TO-LOAD
LLMs SHIP"). All 7 surviving local rows are verified present on disk.

## 14. Batch R3 tail: the retired LEMMY Phases 2-4 numbering note (the trio closed on live evidence; PBUG-02 has its own exit condition in Batch R4)

**The old LEMMY "Phases 2-4" numbering is RETIRED (2026-08-28).** The trio it
tracked collapsed on live evidence: -01 closed 08-16 (mis-attributed), -03
closed 08-18, -03 EXTENDED closed 08-28 (`a469ffb2` -- 48 post-fix
scifi_news_pro ledgers, zero empty cast contracts, and a natural-roll LEMMY
with 8 lines published to obs on 08-26). Nothing remains for a phase number
to point at; PBUG-02 below is the sole survivor and has its own exit
condition.

## 15. Section 6: the moot bake-off runner note (both runners deleted 2026-08-23; the one-sentence lesson stayed live)

- **MOOT since 2026-08-23 -- kept as one line so the finding is not re-derived.**
  This row noted that two `scripts/` bake-off runners aborted a whole sweep on an
  encoder count mismatch, and called it the correct direction that an operator
  should know before an overnight run. **Both runners were deleted with every
  other bake-off** ("delete any animatediff..." was the Ghost half; "I think I am
  done with all bakeoffs" was this one). The finding still generalises: a runner
  that discards the encoder's return value and recomputes the count independently
  will disagree with it silently. Worth carrying into any replacement sweep.

## 17. 2026-09-02 -- the forward-only rewrite: blocks removed from the plan (verbatim, for the record)

The plan was cut to a queue in the operator's order (alpha.15 push -> kokoro-onnx -> the
correctness bugs -> the 8 GB ship set -> the 4060 template test) and every narrative,
receipt and settled ruling moved out. Rulings and standing traps went to
`docs/OTR_STANDING_RULINGS.md`; the blocks below had no other home and are kept here
as they stood, under the heading each carried in the plan.

### Section 1.3 gender ladder: the 2026-08-05 live evidence, the split-is-the-diagnosis paragraph and the SPEC v1 summary (v2 folds it)

Live 2026-08-05: `EBENEZER SCROOGE` = female, `JACOB MARLEY` = other,
`HENRY HARTWICK OGLETHORPE` = female. Meanwhile MACBETH, BANQUO, PROSPERO and
MIRANDA are all correct.

**The split IS the diagnosis, and it means the render code is not broken.**
Shakespeare ships 14 provenance sidecars carrying `characters` with genders; the
prose lane has ONE tracked sidecar and its `characters` key is `None`. The pin
chain already exists and is lane-neutral (`_otr_roster_gender.py`, 12.6 KB, on
disk). Shakespeare is right because the DATA is there. Prose is rolled because it
is not. **This is a vendor-time data gap with a working consumer** -- the exact
inverse of the Item 7 bug, where the value existed and nothing read it.

Spec: `docs/2026-08-05-character-gender-ladder-SPEC.md` (Fable, driver-grounded).
A four-tier TOTAL ladder -- roster -> pronouns in the source text ->
character-in-work web lookup -> name-frequency percentage -- stamping `gender`
(always populated, never `unknown`, because a voice must still be cast) plus
`gender_source` and a confidence. Operator rulings baked in: Shakespeare's KNOWN
rows are untouchable, the announcer stays randomly male/female by design, and the
invented lanes (`original`, `scifi_news`, `scifi_news_pro`, `media_archive`) keep
rolling -- their characters do not exist, so a name search there risks matching a
real person.

### Section 1.3 gender ladder: the spec-rewrite re-grounding narrative

**THE REWRITE IS WRITTEN: `docs/2026-08-28-character-gender-ladder-SPEC-v2.md`
(local; `docs/2026-*` is gitignored).** It folds r2 + r3 and answers B1-B4
explicitly. Two of the four blockers are DISSOLVED by the rulings below rather
than engineered around, and one shrank on re-grounding: r3 said carrying the
ladder's output meant changing every verdict-construction path, but there are
exactly SIX and all six are in `nodes/_otr_roster_gender.py` (298, 310, 311,
334, 364, 468), with ZERO in `tests/` -- so with defaults the change is
additive. Next step is ONE review round against the r2+r3 finding lists, then
code.

### Section 1.6 header: the 2026-07-27 cite-drift examples

**EVERY LINE CITE IN THIS SECTION IS SUSPECT.** Each one checked during the
2026-07-27 triage had moved: `_is_cloud_video_engine` is `render_driver.py:1599` not
`1274-1295`; the "NO FALLBACK to text-only" refusal is `:2148` not `1801-1817`;
`_use_i2v` is `eng_ltx_video.py:583` not `559-572`. The defects are mostly still
real; their coordinates are not. **Re-pin a row's cite when you touch it.**
Path note (verified 2026-08-04): engine adapters live under
`nodes/_otr_video_engines/` (and `_otr_audio_engines/`, `_otr_image_engines/`)
-- bare `eng_*.py` cites in these rows are shorthand for those paths.

### Section 1.6 P0 cluster: the campaign-log measurement of the HTML block-join defect

- **`full_text` reaches the span coordinate system carrying HTML BLOCK JOINS WITH
  NO SEPARATOR, and on the live evidence this is the DOMINANT P0 failure cause.**
  Measured in the campaign logs: `'...Field of Martian PolygonsNASA/JPL-'`,
  `'...and the School ofEngine'`, `'...what you're doing.Let's s'`, `'...(AMR).The
  resea'`. The RSS adapter strips tags without inserting whitespace, so two elements
  fuse into one token. `_normalize_span_source_text` collapses whitespace RUNS but
  cannot insert a space that was never there, so the model quotes the sentences a
  reader sees and they are not byte-exact in the stored text -- exactly the
  "non-literal source span" rejection that killed 12 of the 15 P0 legs.
  **Deliberately NOT fixed by A-3:** inserting separators is a WIDER change to the
  coordinate system `source_digest` pins -- an operator decision, and it belongs in
  the source adapter rather than the codex normalizer. Owed: which adapter builds
  `full_text`, whether a separator can be inserted at admission without breaking any
  accepted ledger, and a fixture from these four strings. (Cross-listed as
  Section 3, question D.)

### Section 1.6 orphan-occupancy registry: the PBUG-20260825-04 arc history and what the closed fixes bought

##### The orphan-occupancy registry (design item -- full arc BEFORE code)

Falls out of PBUG-20260825-04, whose four landed fixes shipped in
`fb67d059` after a full kibitz r1-r4 arc (Codex r2/r3, Cursor r4, Fable r1).
The arc found a new race in each of the first two cuts of the same fix, so
**do not treat this item as mechanical** -- it is a genuine design
choice with more than one defensible answer, which per CLAUDE.md means a full
arc BEFORE code, not after. (Its sibling, the GGUF generation deadline, CLOSED
2026-08-25 -- receipt in the archive; its lesson is lifted into Section 4.)

- **THE ORPHAN-OCCUPANCY REGISTRY -- still deferred, now on its third
  independent confirmation.** `has_local_resident_llm()` reports "nothing
  resident" the instant a timeout invalidates the cache dict, even while the
  orphan worker is still actively running CUDA kernels on the model that
  entry described. `nodes/otr_shot_lock.py:1781` and
  `nodes/otr_video_render_batch.py:289` both trust that signal before
  starting visual/video work. The r1 panel (Codex + Cursor + Fable) deferred
  this unanimously; r3 and r4 each re-raised it and each time it was
  re-confirmed as correctly out of scope for the cache-bookkeeping fixes.
  Shape: a process-global, lock-protected registry of in-flight generations,
  registered before invalidation and cleared via `Future.add_done_callback`,
  with fail-fast admission on `request_slot` and the visual-entry guards
  reading real occupancy instead of the dict's cleared-or-not state.
  **What this session's fixes did and did not buy:** they close the concrete
  cache-bookkeeping windows (no abandoned publish, no laundered
  invalidation, no torn read, no unconditional teardown of a foreign live
  entry); they do NOT make orphan GPU occupancy visible to a downstream
  visual stage. That remains exactly as exposed as before.

### Section 1.6 coverage cluster: the three-silent-mechanisms row and the _should_loop_fill row (CLOSED on re-verification 2026-09-02: the loop-fill and held-frame paths were RETIRED 2026-08-02 -- `_should_loop_fill` is a named no-op and a short clip now raises `ClipUnderrunsItsBeat`; `encode_frames_to_silent_mp4` proves its count through `proven_frame_count`)

- **THREE silent coverage mechanisms exist, not one** (found 2026-07-25).
  Mechanism 1, the engine mirror/ping-pong (`wrapper_bridge.extend_frames_to_target`),
  is GONE from `eng_ltx_8gb` -- pinned behaviourally by a test that detonates the
  helper and renders successfully. It REMAINS in `eng_wan_ti2v`, deliberately and
  permanently: WAN renders a short native clip on purpose and fills the beat with it,
  which is the shipped 8GB tier contract `PBUG-20260723-02` protects. **Still open:**
  composite loop-fill (`otr_silent_composite._should_loop_fill`, which also SUPPRESSES
  its own underrun warning once it activates) and held-last-frame. For `ltx_8gb` the
  composite path is now de facto unreachable -- the adapter returns exactly the
  requested count or raises -- but not structurally impossible:
  `encode_frames_to_silent_mp4` reports the size of the array it piped into ffmpeg
  rather than re-probing what ffmpeg wrote, so an encode-side drop could still
  under-report. PRE-EXISTING; close it when the assembly boundary is next opened.
- **`_should_loop_fill` names the permanent fix and it is now being built**
  (`otr_silent_composite.py:244-266`): *"The real fix is phrase-chunking (render the
  beat's correct duration so it never underruns) -- tracked as a follow-up."* The
  coverage block IS that follow-up.

### Section 1.6 coverage cluster: the CanonicalClip.frame_count row (CLOSED on re-verification 2026-09-02: every clip writer ffprobes, tests/test_terminal_frame.py gates the roster 17/17, the residual self-declared surface was empty) and the ping-pong cross-list (its live proof is the deferred capped-14B HuMo leg)

- **`CanonicalClip.frame_count` -- "the integer timing authority" -- HALF CLOSED**
  (`58e288af` + `40780b82`, count closed @ `48e3c6fb` without paying a decode). Every
  module that writes a clip now ffprobes it, and a roster gate in
  `tests/test_terminal_frame.py` fails by name for any module that writes a clip
  without proving it. What this proves and what it does not: it proves the muxer wrote
  what it was piped, which is the right question for a clip written by ONE ffmpeg
  pass; it does NOT prove decodability, which is why `assemble_beat_segments` still
  decode-counts every ASSEMBLED beat and must keep doing so. **Re-verify before
  acting:** this row's "still self-declared elsewhere" pointers (the four `viz_*` and
  four `still_*` engines) were both closed afterwards, so the remaining open surface
  may be empty.
- **Ping-pong on a capped HuMo beat played lip sync BACKWARDS** -- FIXED in code @
  `a1d810f1`, but the finding is STATIC (no live artifact), so it is NOT a PBUG row. A
  capped-14B leg would reproduce it. Kept here so the live proof is not forgotten
  (listed under Section 2, deferred render items).

### Section 1.6 coverage cluster: three recorded limits moved to OTR_STANDING_RULINGS.md KNOWN OPEN (the 7d-preflight canvas trap, odd-canvas evenness, the roster-gate string-constant limit)

- **THE 7d-PREFLIGHT THAT "PROVED THE GPU" RAN AT THE WRONG CANVAS** (found
  2026-07-27, B5 panel; verified -- and it corrects a claim this file once made).
  `render_single` and both HTTP entry points use the older ledger-free `build_request`,
  which never reaches the canvas seam and defaults to `OTR_VIDEO_RENDER_CANVAS`
  (832x480). So the "GPU IS PROVEN" leg (`ltx_8gb`, 25 frames, 3004 MB) exercised
  832x480, not the production canvas. `render_single` parity is explicitly deferred by
  the O1 judgment; what must NOT happen is another "proof" through that harness being
  read as a production proof. (The 512x288 canvas itself HAS since rendered live --
  bench arm D, three cells -- but through a DIRECT-NODE graph, which proves the canvas
  and the recipe, not the seam.)
- **Odd-canvas evenness is validated at the ENCODER, not where the canvas is chosen.**
  The stride defect itself is closed (`b1f2ee86`): `ffmpeg_silent_mp4_cmd` declares the
  REAL width/height and `encode_frames_to_silent_mp4` REFUSES an odd canvas by name,
  because yuv420p subsamples chroma 2x2 and cannot represent an odd dimension. Still
  true and NOT fixed: neither `WanInitImageMixin._dims()` nor the `Canvas` schema
  validates evenness, so an odd canvas is caught late rather than at the choice. No
  live producer builds one today (832x480, 512x288, 1472x832 are all even).
- **KNOWN LIMIT of the widened roster gate**, recorded so it is not rediscovered as a
  surprise: the codec flag is matched as a STRING CONSTANT, so a flag assembled at
  runtime (an f-string, `"-c:%s" % stream`) or the stream-index spelling `-c:0` is
  invisible to the sweep. Nothing in the tree does that today; an encoder that ever
  needs to must be pinned in `_ENTRY_POINT_PROOFS` by hand, which the inventory test
  makes a visible decision. Separately, ONE mutant survives the round by construction:
  deleting the self-proving membership assertion is catchable only by a meta-test of
  that assertion.

### Section 1.7 adaptation design: the keystone-correction rationale

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md`.

**The keystone correction: compile source speech, do not generate it.** A ledger row
that merely POINTS at a source segment proves structure, not meaning --
`PRODUCTION_SPRINT_LESSONS.md` lesson 11 documents that exact failure class.
Source-owned text must be materialized deterministically from an authenticated
segmented artifact and verified against it. "Summarize into X words" then means
SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not paraphrasing -- which also removes
the VRAM hazard, since no model sits in the source-speech path.

Settled by arithmetic: an episode cannot exceed **1,520 words** (19 voiced beats at
act_count 7, `BEAT_WORD_HARD_MAX` 80), so full-scene performance is impossible
without redesigning beat topology. Build target is the 300-word unit.

### Batch R2: the fork argument preserved for the arc that may not be needed

**THE FORK, PRESERVED FOR THE ARC THAT MAY NOT BE NEEDED.** The lens receives
only three things -- the prose, `kind` and `role`
(`ideogram4_local.py:644-646`). There is no subject field to derive an anchor
from, and this repo has ALREADY ruled on extracting one from the prose:
`_wrapped_caption` (`ideogram4_local.py:372`) says the composer emits *"a
comma-joined five-layer string behind a style prefix, which is a convention, not
a grammar, so any attempt to re-extract subject / setting / elements from it
mis-fires."* Two defensible answers, not equal:
* **(a)** extract a subject noun from the prose -- lens-local and small, but it
  is the option the codebase already tried and wrote off, and a wrong noun
  INVENTS CONTENT, which the source-fidelity rule forbids in as many words;
* **(b)** a new metadata channel so the producer hands the lens a real subject
  -- more wiring, but the anchor is derived rather than guessed.

Evidence: `docs/2026-08-26-ideogram-music-card-PROBLEM-STATEMENT.md` and
PBUG-20260826-01.

### Batch R3: the operator-quote charter and the done-before-this-row paragraph

**One sweep of four legs (after Leg 0, Section 1.2) proves all 7 surviving
local LLM rows in BOTH slots, plus the gemma Q8_0 negative probe -- and each
leg doubles as a ledger-cleanup live-watch and an eyeball re-observation
chance.**

**THE OPERATOR'S WORDS, which are this item's charter:**
*"when all this LLM coding is done we should look and do more coding and retest
all local LLMs on a 1 runthrough that should catch it"*; *"if it doesn't fit
nicely or requires Ollama rip it from the dropdown and blast radius"*; *"clean
sweep I only want easy to load LLMs"*; *"there should be an LLM preflight guide
-- preflight guides for adding all your own components"*; *"all models should
live out here `C:\ComfyUI-Models`"*; and *"all LLMs should either be able to
play creative or technical equally. If they're not, and they were not tested or
not implemented, and they serve no worth, we should rip them out."*

Done before this row: the clean sweep, the PASS-tier invariant gate,
`docs/LLM_PREFLIGHT_GUIDE.md`, the `Q6_K` removal (ruling: `docs/OTR_STANDING_RULINGS.md`,
"ONLY EASY-TO-LOAD LLMs SHIP"); all 7 surviving local rows are on disk.

### Batch R5: the D2 statistics, the 08-04 postmortem recap and the story-engine preamble

### Batch R5 -- OPPORTUNISTIC: D2 fail-hunting still legs; every clean leg is also an eyeball re-observation

Run when a render window is free and nothing above needs the box.

**D2 (from the long-standing parked row -- renders have resumed):**
Reset per AGENTS.md section 4, boot headless, run **320-word `public_domain` or
`shakespeare` still legs until one fails** (~1 in 6). Three legs on 08-04 all
published, which at that rate is a ~58% chance of zero -- neither confirmed nor
cleared.

**Either outcome is valid.** A publish is a clean leg; a **fail-closed with
complete evidence is the PROOF D1 WORKS** and is the outcome you want. When it
fails the server log names the branch itself -- arm, token, index, canonical
`prompt_hash`, repr-escaped excerpt -- plus a compact JSON `MISSING_TARGET`
record emitted BEFORE the raise (the canonical runner truncates the exception at
500 chars, `scripts/otr_api.py:749`). The log survives reboot;
`scripts/otr_rotate_log.ps1` rotates instead of truncating. D3 then fixes THAT
branch at its root and `PROD_BUG_LOG.md` gets a mechanism, not a guess.

**Do NOT:** weaken the completion gate, revive the portrait-init fallback, or
rebuild the withdrawn "give the collapse guard a still owner" fix -- the 08-04
postmortem disproved that chain (70 whiffs and 69 cast-time deferrals across 11
passes that ALL published).

Record: `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`,
`docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.

**The two eyeball re-observations that RIDE these legs (and any other real
render leg -- they cost zero extra legs):**

Both were eyeball observations against a story engine that has since had its LLM
vetoes ripped, THE LAW imposed, six banks renamed onto new packs, word-fit ceilings
retired, the repair-first plan landed, and a ledger cleanup pass added. Neither has a
reproduction at current HEAD, and under the standing rule a finding with no
reproduction is not a row. **Do NOT schedule coder time against either.** They are
settled by the operator eyeballing a real render leg: still there -> re-admit
as a FRESH dated row with that leg as evidence; gone -> the LAW-era work already fixed
it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`).
  Episodes START a story instead of admitting you into one; the announcer takes debate
  turns instead of framing. Operator eyeball 2026-07-11. If it survives re-observation
  the fix is still seam + score contract + fail-closed validator, never Python
  authorship.
- **Name-splice defect #2.** v4-campaign Phase 0 record in HANDOFF_LOG; its timebox
  predates THE LAW.

### Batch R6 as it stood (seven 45-word proofs from a broken cross-reference)

### Batch R6 -- SEVEN 45-word engine proofs (from the archived 2026-08-13 runway table, row 3 -- CONFIRM STILL OWED before spending seven legs)

Lifted verbatim from the archived table: **Run seven fresh post-change 45-word
render proofs.** Exit condition: All seven exact public engine IDs pass against
the post-bugfix/post-Lemmy HEAD with `COVERS`, `RESULT SUCCESS`, server
`Prompt executed` + `obs_publish OK`, and the canonical OBS asset on disk.

**Flagged 2026-08-28:** the row's "See WHAT IS ACTUALLY LEFT below" pointer is
one of the eleven cross-references the 2026-08-16 audit already recorded as
BROKEN (target removed before the audit; see the archive). The engine list must
be re-derived from the live registry, and whether all seven proofs are still
owed at current HEAD has not been re-verified. Confirm before rendering.

### Batch R7: the clip-by-clip Leg C .. C5 narrative and the README-pass DONE block (receipts: 4060_CLEANROOM.md, PBUG-20260902-01/-02, 66da15da, 756c64f4)

### Batch R7 -- THE 4060 CLEAN-ROOM RETRY: first measured 8 GB result for a non-AnimateDiff video lane

The clean room is built and stays on disk: `C:\OTR-CleanRoom` on the 4060 (fresh
portable v0.34.0, Python 3.13.14, OTR clone, ComfyUI-GGUF pinned + patched, 52 GB of
pinned weights placed, two headless profiles `otr_cleanroom_8gb_ltx25` and
`otr_cleanroom_8gb_humo17` on bark voices). Server and legs start through Task
Scheduler (`_boot_stock.cmd STOCK LOWVRAM`, `_leg_stock.cmd`), never from an SSH
session. Friction log: `docs/ship-audit-2026-09-01/4060_CLEANROOM.md`.

* Leg A: `otr_cleanroom_8gb_ltx25`, `public_domain`, 1 act, with
  `--disable-dynamic-vram --lowvram --disable-pinned-memory`. The stock launch aborted
  in the first z_image still (Section 3 L); this leg is what tells whether the lane
  itself fits after the still is out of the way.
* Leg B: `otr_cleanroom_8gb_humo17` behind it (no extra node pack; 13.6 GB of
  Comfy-Org weights).
* Leg C: RAN 2026-09-02 (profile `otr_cleanroom_8gb_klein_ltx25`, Klein Q4 GGUF stills +
  LTX 2.5). Under the STOCK loader Klein did NOT abort -- the Z-Image abort is
  engine-specific -- and minted a clean 832x480 still, but ONE still took ~42 min: the
  7.7 GB bf16 Qwen3-4B encoder stayed on the card and the DiT loaded with "0.00 MB usable"
  (120 s per step). The lowvram flag pair (Leg C2) was IDENTICAL (143 s per step), so the
  flags were never the answer. ROOT CAUSE FIXED in 9b90189a: the three local image engines
  ran the graph without `free_after_use` (every video engine has it), so the encoder never
  left the card before the sampler; fix proven byte-identical on the 5080 with 5-7 GB lower
  peak (PBUG-20260902-01, Bible 12.145). Leg C3 (that fix, STOCK flags) was UNCHANGED:
  under ComfyUI 0.34 DynamicVRAM the dropped encoder leaves an orphaned VBAR that only
  another dynamic model's pressure reclaims, and the GGUF DiT is a classic patcher. Measured
  in-process on the clean room and fixed in ad6a635f: `run_graph(..., evict_after_use=
  {"clip"})` unloads the named dynamic patcher through ComfyUI's own registry at its drop
  (free 620 MB -> 6998 MB; a 2-step Klein render 9.4 s vs ~4 min). Four-round arc on two
  substitute seats, `docs/2026-09-02-encoder-eviction/driver_anchor.md`; 5080 proof on both
  paths byte-identical. Leg C4 (ad6a635f, STOCK flags) FAILED the same way, and the
  instrumented Leg C4b showed why: the encoder had never occupied the card (0 resident
  pages at its drop; 48 MB free) -- the WRITER LLM that composed the still prompts was
  still resident, and nothing in the general path released it before the image stage
  (PBUG-20260902-02). Fixed in da2b7a36: `OTR_ImageGenDispatcher` calls the canonical
  `free_otr_pipeline_residue()` once per dispatch before the first local still. Measured
  on the 5080 first: the 12B writer (7.4 GB) had been co-resident with every still there
  too (`allocated 7387 -> 6`, `free 14.4 GB after`, five stills then minted cleanly). Leg C5
  (da2b7a36, STOCK flags, 06:09) PASSED on the 4060: `free 6.9 GB after`, `Requested to
  load Flux2 / loaded completely; 5560.68 MB usable, 2591.65 MB loaded`, 1.07 s per step
  (~21 s a still, was ~42 min), nine stills in one server process, then on into LTX 2.5.
  Klein on 8 GB under stock launch flags is MEASURED. And Leg A's own question got its
  first answer on the same leg: LTX 2.5 (Q3_K_M DiT, 12B encoder pinned to CPU by the
  engine) renders on the 4060 under stock flags -- two-stage passes at 1664x960, 1018 s
  for the first clip and 827-851 s after (DiT half offloaded), six clips by 07:47 with
  ~20 in the episode, so ~14 min a clip and a five-hour episode. It WORKS on 8 GB; it is
  not a daily driver at that pace. PROVEN (matrix) waits for that episode to reach the
  clean room's obs (~10:30-11:00 on 2026-09-02) and for an operator eyeball.
* Record ONLY what publishes (`RESULT SUCCESS` + `obs_publish OK` + the file) into
  `config/machine_classes.json` (`proven[]` or `known_limits`), regenerate
  `docs/MACHINE_MATRIX.md`. Do not advertise a lane the clean room did not finish.
  DONE 2026-09-02 (66da15da, 756c64f4): the README newbie pass landed from a four-reader
  audit (three templates named, the Python 3.13 voice fix by node and dropdown, IndexTTS2's
  own installer, the 8 GB Klein / LTX 2.5 facts, AMD and Mac rows, the LTXVideo pin and
  patch, the measured `known_limits` line and a regenerated matrix); the provisioner now
  pins AnimateDiff-Evolved (release 1.6.0) like the other two packs. Still owed from the
  audit: `pyproject.toml`'s kokoro marker (in the alpha.15 patch, the operator's registry
  push) and a shipped 8 GB Klein + LTX 2.5 profile (Section 1.0).

### Batch R1 as it stood: the H3 leg command and its 2026-08-27 correction (the leg was spent by two post-fix episodes)

### Batch R1 -- ONE leg on `otr_w45_minimax_h3_video` proves the H3 prompt-policy fix

**The code is shipped, green and pushed (`e923a9f3`, suite 12332/121/1). What
is missing is the one thing CPU tests cannot supply: a published episode.**

A vanilla canonical run does NOT prove it -- the canonical defaults to the
still floor, so no character beat reaches H3 at all. The leg has to select
`minimax_h3_video` deliberately.

**THE COMMAND IS READY -- do not improvise a harness.**
`scripts/otr_headless_canonical.ps1` is the sanctioned wrapper: it resets
selectively, boots the UTF-8 launcher, and ALWAYS loads the real canonical.

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_w45_minimax_h3_video -Acts 1
```

**CORRECTED 2026-08-27, after the wrong version of this command failed a live
leg.** The first version here said to pin the engine with
`-Set "OTR_VideoDirector.character_video_model=..."`. That is REFUSED by
design: `patch_creative` whitelists CREATIVE widgets only (writers, seeds,
banks), and the video-model widgets are MANAGED -- engine routing goes through
a PROFILE (`scripts/otr_api.py:831`, `CREATIVE_WHITELIST`).
`otr_w45_minimax_h3_video` is the sanctioned profile: all three roles on
`h3_low_video` AND the h3 boot contract the engine ENFORCES
(`--reserve-vram 12`, `--disable-pinned-memory`) -- a default boot would have
been refused even if the patch had landed. Writers DO ride as `-Set`
(whitelisted), e.g.
`-Set "OTR_LedgerScriptWriter.technical_model=google/gemma-4-12b-it (11.9 GB)"`.

* Reset per `CLAUDE.md` section 4 (selective kill by CommandLine, port 8000
  empty, GPU back to ~1.5 GB); the wrapper does this, but verify it happened.
* **A FRESH EPISODE ID IS MANDATORY.** `request_hash` excludes prompt bytes, so
  an existing clip is cache-eligible and an old SPEAKING clip would be reused
  -- a false pass that looks exactly like a real one.
* Publish to `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs` and confirm the
  asset on disk with a current timestamp plus `obs_publish OK`.
* **Read the prompt receipt, which is the actual verdict:** nonverbal action
  and camera PRESENT; the beat's exact dialogue and any speaking / lip-sync /
  mouth anchor ABSENT.

**The BEFORE sample already exists and must be preserved:**
`signal_lost_the_caretakers_clause_20260826_155835` in `otr/obs/` -- every beat
on `minimax_h3_video`, rendered before the fix. Do not overwrite it or reuse
its episode identity; it is half of the A/B.

### Deferred render items: the LOCAL mistral/gemma writer matrix (forbidden by the 2026-08-04 story-quality directive)

- **The LOCAL mistral/gemma writer matrix** (render-window judgment question,
  not a coder slot). The Sonnet arm of the creative-writer question is answered
  (`docs/2026-07-17-model-bakeoff-scoreboard.md`); the local roster comparison
  never ran.

### Section 3 (F) detail: the hop-by-hop WAN chain, the July 8 GB writer diagnosis and the A2 causal-chain correction

- **WAN 8-GB low-VRAM launch contract -- CODE-COMPLETE and PROOF-INCOMPLETE. It is
  not a coding item; the one thing blocking it is an OPERATOR DECISION.**

  **Already BUILT and WIRED end to end** (verified hop by hop): `otr_8gb_wan.json`
  `video.max_render_frames=17` -> `capability_profiles` optional-key validator ->
  `_otr_workflow_apply.py:532` flatten -> `workflows/variants/otr_8gb_wan.json`
  node-87 widget slot 14 = 17 -> `otr_video_director.py:423` policy stamp ->
  `otr_shot_lock.py:1722` `ledger.video.max_render_frames` ->
  `render_driver.py:3328` per-adapter policy -> `motion_common.profile_max_render_frames()`
  -> `eng_wan_ti2v._floor_length` hard cap (`:730`) and `_planned_length` refusal
  (`:785`), with `render_driver.py:3845` refusing on drift. Landed `f914f0a4`, dead
  node-87 widget repaired `7f4644a1` + `8f41af27`, WAN deliberately excluded from
  `frame_contract.PLANNING_CAP_ENGINES` by `b23fc035`, recipe frozen `71753cb4` /
  `8424f369`, whole-beat single-UNET-load `439ce8c7`. Regression net:
  `tests/test_remaining_video_contracts.py:16-194` (nine hop-by-hop tests) plus
  `tests/test_multiclip_effective_contract.py:216,234`.

  **THE ONE OPERATOR DECISION (the actual blocker).** The ceiling reaches a leg ONLY
  through a variant workflow or a hand-set widget: `otr_canonical.json` node 87 ships
  `max_render_frames=0`, so a plain canonical WAN run is UNPINNED and inherits
  `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure shape. The obvious patch
  (pin 17 in the canonical) is WRONG: the canonical serves every tier, and 17 is the
  8-GB tier's number, so pinning it would cap LTX/HuMo 16-GB legs too. The channel
  that carries 17 today is `config/profiles/*.json`, which is on the RETIREMENT list
  -- so writing new behaviour onto it is forbidden. **Decision needed: after profile
  retirement, who owns a tier's native render ceiling?** The shape that fits the
  per-adapter-ownership doctrine is that `eng_wan_ti2v` DECLARES its own tier ceiling
  (a capability-row field), the widget becomes an operator OVERRIDE with 0 meaning
  "use the adapter's own contract", and the profile channel stops mattering. That is
  a real design change with a live-behaviour blast radius on any card with headroom
  (the VRAM predictor currently gets to ask for more than 17 and often can), so it is
  NOT being written on assumption. Ratify the shape first.

  **Also open, all PROOF obligations rather than build work:** the 18-engine GPU
  campaign is engine COVERAGE, not an 8-GB qualification; and a render on a PHYSICAL
  8 GB card is still owed -- the four-arm bench PREQUALIFIED on a 16 GB card told to
  reserve 8 GiB, which is not the same claim.

  **One untested edge, cheap to close whenever this reopens:** WAN is out of
  `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan CAN contradict by
  design, and `_planned_length` hard-refuses mid-episode when they do -- but no test
  asserts a 17-frame tier survives a multi-segment beat. `:216`/`:234` in
  `test_multiclip_effective_contract.py` pin the topology, not that outcome.
- **THE 8 GB PROFILE FAMILY CANNOT RUN ITS OWN WRITER** (found 2026-07-27;
  LIVE-REPRODUCED TWICE on two different banks, then confirmed by a two-strikes
  kibitz panel -- codex `gpt-5.6-sol` high and agy independently reached the same
  diagnosis). `config/profiles/otr_8gb_ltx.json` pairs a 12B GGUF writer
  (`gemma-4-12b-it-Q4_K_M`, 6.63 GB of weights) with `llm.gguf_n_ctx: 2048` under a
  declared `vram_ceiling_gb: 6.8`. The pipeline's own smallest prompt needs **2064
  input tokens** and P0 reserves 2800 output (`_P0_BASE_OUTPUT_TOKENS`), so the leg
  dies in `OTR_LedgerScriptWriter` before any render. Live preflight, verbatim:
  `Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)`. **ctx is the SYMPTOM; the
  writer MODEL is the cause** -- 4096 puts it near 9.4-9.5 GB, OOM on the very card
  the tier exists for. Every 2048-ctx profile (`otr_8gb_ltx`, `otr_8gb_wan`,
  `8gb_lite`, `cpu_floor`, `otr_amd8_rocm`, `otr_cloud_lanes`) is `status=draft` and
  every one pairs 2048 with the 12B; the only `status=shipping` profile is
  `16gb_full` (4096 + Mistral-Nemo). **NOT a one-line profile edit:** the GGUF
  registry ships exactly two rows (`unsloth/gemma-4-12b-it-GGUF`,
  `unsloth/Qwen3-8B-GGUF`); `google/gemma-2-2b-it` is in the TRANSFORMERS catalog, a
  different lane -- agy proposed it and was wrong, recorded so nobody re-derives it.
  **Largely mooted by profile retirement:** with no profile passed, the canonical
  JSON's own `gguf_n_ctx=4096` / Q8_0 binds and the leg runs. **Fix the profiles or
  finish retiring them; do not leave both.**
- **A2 -- HELD pending the profile retire-now vs retire-later scope. The profile's
  `llm` section silently overrides the canonical JSON, and the applied-overrides echo
  HIDES it.** Held because its entire subject is `apply_profile_to_workflow` and the
  printed echo -- a channel directed to be retired, so building on it now may be work
  on something scheduled for deletion. The fix SHAPE is correct and ready when the
  scope is settled. The profile's `llm.*` values win over the widgets the operator set
  in `otr_canonical.json` (which ships `creative`/`technical` = `google/gemma-4-12b-it`,
  `gguf_n_ctx=4096`, `gguf_quant=Q8_0`, `llm_vram_ceiling_gb=14.5`), while
  `scripts/otr_api.py:817` flattens only `role_overrides` / `slot_overrides` /
  `features` + two `seed_policy` keys for the printed summary -- so the run reports
  "16 overrides" while ALSO having replaced the entire LLM configuration. **Causal
  chain corrected** (triage 2026-07-27, codex; grounded): the override does NOT come
  from the validator's `OTR_ACTIVE_PROFILE` export -- it happens at submission,
  `scripts/otr_canonical_api_run.py:157` -> `apply_profile_to_workflow`; and the real
  applier (`nodes/_otr_workflow_apply.py:492-540`) ALREADY flattens `llm`; only the
  printed echo is stale. **Fix: generate the echo FROM the applier's flattened map.**
  Adding `llm` to the echo by hand leaves the next drift intact.

### Section 3 J preamble, K (ruled) and L (measured) as they stood

#### J. THE REGISTRY -- one push, then one control experiment

The gate is a SECRET scanner, not an exec linter (Comfy-Org backend,
`registry_svc.go:1392-1455`): any non-empty scanner response flags the version and
the reason goes to their private Discord. Two versions exist, both Flagged, no
Active, no rollback target (the node hard-delete freed alpha.8's string).

* **J1. Push alpha.15** -- `docs/ship-audit-2026-09-01/pyproject_alpha15.patch`
  applies cleanly (version bump, pycairo/pillow/aiohttp, bitsandbytes
  `sys_platform != 'darwin'`, kokoro `python_version < '3.13'`). The README's
  token-shaped literal, the only shipped string matching a published secret rule,
  is already gone (`64d81ca7`). Operator's push; the file is a release trigger.
* **J2. If alpha.15 flags, run the control:** republish the alpha.8 tree
  (`e44235f5`) byte-identical as alpha.16. Active means the trigger is in the
  alpha.9+ delta and can be bisected; Flagged means the ruleset moved and that
  result is the evidence to hand Comfy-Org. Never version-delete (soft delete burns
  the string).

#### K. DEFAULT VOICE -- RULED 2026-09-01: kokoro-onnx is the go-to

Operator: *"kokoro onnx is our new go-to."* Same Kokoro-82M voices, ONNX Runtime instead of
the torch `kokoro` package, so it installs on Python 3.13 (the interpreter ComfyUI Desktop and
the portable ship) and on Linux and Mac. Build row: Section 1.11. Until it lands, the 3.13
sets run bark. Nothing else is open here.

#### L. THE 8 GB IMAGE ENGINE ABORTS THE PROCESS UNDER A STOCK LAUNCH

On a never-touched portable install with stock flags, the first z_image_turbo still
(int8 convrot) aborted the whole ComfyUI process at sampler step 5/8 under
DynamicVRAM (`Fatal Python error: Aborted`, stack in comfy/ldm/lumina/model.py; the
drill's PBUG-03 shape). The known workaround is a launch flag pair
(`--disable-dynamic-vram --lowvram`), i.e. a special thing a newcomer does not know.
Decide: the 8 GB dropdown set avoids z_image stills, or the launch flags become a
documented requirement for 8 GB, or both until ComfyUI answers the faulthandler
report. Default if unruled: the flags are documented in README for 8 GB and the
retry (Batch R7) measures the lanes behind them.
Half-answered by 1.12 (2026-09-01): the 8 GB set now runs Klein stills, so it avoids
z_image by ruling rather than by this abort; whether the abort is engine-specific is
what Leg C measures. The launch-flag question stays open for anyone who picks Z-Image
on 8 GB from the dropdown.

### Section 3 question (A) shipping history and the (I) reframing narrative

* **(A) Arm `defaults.scene_coherence_check` on any bank?** The G15 vacuity fix
  shipped 2026-08-28 (`e2807dcc`; receipt in the archive) but nothing arms the
  gate today -- the fix changed a function with zero live callers in current
  production, by design, so it shipped at zero risk. GO_FORWARD's original text
  said "measure OFFLINE over the published corpus first, then arm in ONE
  change" -- that measurement was never attempted and stays open. Whoever picks
  this up next decides first whether any bank should arm it at all before
  running that measurement. (The measurement itself is no-render work once
  ruled.)

### Section 3 standing context for (I) as it stood

#### Standing context for question (I): MINIMAX H3 -- A SPRINT SERIES ON THE VIDEO PATHS (operator, 2026-08-09)

**THE RULING IS IN, AND IT DISSOLVES THE OLD QUESTION.** This section used to
say the next step was an operator ruling on "does H3 belong in the video
dropdown given the 4 s floor vs the sub-4 s beats". That framing is RETIRED.
The operator's 2026-08-09 direction: H3 is **"a series of sprints all to refine
the video paths"** -- scope TBA.

**What that changes for a window picking this up:**
* It is NOT a yes/no dropdown admission any more, so do not go looking for a
  verdict to record. There is nothing blocked on the operator here.
* The unit of work is a SPRINT against the video paths, not a one-shot chunk.
  Expect several, each with its own kibitz gate, each landing green and pushed
  on its own.
* The 4 s floor is now an INPUT to that refinement -- a constraint the video
  paths have to accommodate or explicitly route around -- rather than a
  disqualifier that settles admission.

**Scope is TBA and that is deliberate.** Do NOT invent the sprint list. When the
operator names the first sprint, write it into Section 1 or 2 of this file as
its own row, and leave this section as the standing context.

**Grounding that survives the reframing:**
* Problem statement `docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md` is
  UNTRACKED and another window's working file -- never stage, edit or delete it
  from a different window. Read it; do not touch it.
* The matrix-pattern spec already names MiniMax as a churn driver
  (`docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` section 5), so a video-path
  sprint will likely collide with the un-converged matrix work (section 0). Read
  that section's "what survived all four rounds" before designing anything.
* The recipes are NOT on the table (standing directive). A video-path sprint
  refines PATHS -- routing, canvas negotiation, admission, extension -- never
  the shipped render recipe.

### THE QUEUE item 2 as it stood: the ghost pool r1 judgment with code excerpts

**2. GHOST POOL -- r1 IS IN, AND IT DISSOLVES THE BUG.** Panel fired
2026-08-31 (Codex `no`, Antigravity `yes-with-fixes`, Cursor `no`); artifacts in
`kibitz-runs/2026-08-31-ghost-clause-pool/r1/` (gitignored, local only).

**Uniqueness is measured on the wrong thing.** Verified against the files, not
taken on the panel's word:

    GHOST_V2_SLOTS = ("pack_cue", "motif", "leaf", "law")   prompt.py:824
    motif_for_character(components, mode, seed_int)          author.py:644
    key = leaf.casefold()                                    shot_lock.py:2313

The picture is four slots; the duplicate check reads one. Two beats with the
same leaf but different CHARACTERS compose different prompts and render
different pictures -- and the rule rejects them anyway. That is why 6 -> 18
failed and why 50 would fail too. **The pool is not too small; the test is too
strict, in a way that does not match what a viewer would notice.**

**BUILD:** uniqueness on the FINALIZED POSITIVE PROMPT, applied identically to
writer output, replay and the deterministic path -- capacity becomes clauses x
motifs and the arithmetic failure stops existing. Then a bounded progression,
total by construction: unused finalized prompt -> reuse a leaf where a different
motif keeps the prompt new -> reuse the least-recent signature, deterministic on
`episode_seed + beat_id`, never adjacent. The allocator returns a PER-BEAT reuse
disposition appended to that beat's existing `fallback_reason` (ShotLock
currently stamps one batch-wide reason over every object, which would erase the
original model-failure reason).

**Only pool exhaustion becomes recoverable.** My anchor said "a fallback that
can raise is not a fallback" and that was overbroad: `GhostAuthorError` has 10
raise sites covering unknown mode, missing bookend motif, invalid role and an
empty `motif_cue`. Those are structural corruption and stay loud --
OOM-or-nothing governs render refusals, not schema violations.

**CUT, so nobody rebuilds them:** the combinatorial generator (a second grammar
engine before the proxy is fixed), act-scoped uniqueness (verified: no
authoritative act field exists -- the spec carries beat, role, mode, motif,
ordinal and arc cue; ShotLock supplies only a mapped `arc_phase`), and "loud
handover" (it is controlled reuse under a second name).

**DONE WHEN:** >18 same-mode beats complete; mixed replay plus fresh authoring
completes; all three paths share the invariant; adjacent finalized prompts never
repeat; same seed gives identical output AND receipts; every beat keeps a valid
`ghost_prompt`; then the failing five-act topology through
`workflows/otr_canonical.json` with `obs_publish OK` and the file verified on
disk. **Existing tests encode the obsolete absolute-leaf rule
(`test_ghost_prompt_v2_lane.py:399-405, 437-451`;
`test_ghost_signal_author.py:925-931`) and must be REPLACED with the new
invariant, not deleted.**

*Open from r1:* whether the anchor's "roughly 70 minutes" belongs to this
failure or to a `GhostCadenceError` (unseparated); and the panel's own
assumption that "no adjacent repeat" is the right viewer threshold, which wants
checking against frames rather than more reasoning.

### Section 1 coding order (2026-08-28) and its reordering rationale

**THE CODING ORDER (2026-08-28, operator-set and REORDERED the same day on
his instruction: the 4060 set moves LAST -- "I don't want to test [the] 4060
until we are done with all other bugs." The 4060 test is the CAPSTONE: it
runs once, on a tree with every other bug already closed, so the frictionless
verdict is not polluted by known-open defects):**

1. **The kokoro-onnx backend** (1.11 below, and item 1 of THE CURRENT STEP) -- the
   operator's ruled default voice; a design item, so its kibitz arc comes first.
2. **Gender-ladder: SPEC v2 -> one review round -> CODE** (1.3; the spec is
   written and pushed, the code is not started; three NOs in its history
   mean the round is not optional).
3. **Local-LLM sweep Leg 0** (1.2 -- ~15-20 min, in-process; needs an IDLE
   GPU, do not contend with a render).
4. **Image defaults** (1.12 below) -- APPLIED 2026-09-01: Klein in every 8 GB / 12 GB /
   AMD profile, AMD images-only, Z-Image cfg 2.0 -> 1.0 on the operator's A/B. Batch R7
   Leg C then found and fixed the 8 GB residency defects (9b90189a + ad6a635f: the text
   encoder leaves the card before the sampler; da2b7a36: the dispatcher releases the
   writer LLM before the first local still -- the first-order cause, on the 5080 too);
   Leg C5 is the 8 GB measurement of the shipped code. **Auto-download** (1.13 below)
   -- design row, confirm to schedule.
5. **Ship-audit blockers** (1.9 below) -- the non-mechanical survivors of the
   2026-09-01 audit; each is a design item with more than one defensible answer.
6. **Docs deletion pass** (1.10 below): stale docs go unless they carry a video-model
   recipe; no new guides.
7. **The 4060 frictionless set** (1.0 below) -- LAST coding item, then the
   republish sequence it gates: operator applies the alpha.15 patch -> clean
   publish -> the 4060 template test.
8. Handoff bookkeeping.

### Section 1.0 the 4060 frictionless set as it stood

#### 1.0 THE 4060 FRICTIONLESS SET (the CAPSTONE -- runs LAST, after every other bug closes; the path to "download the template, click run" on his 4060)

What is still open on this set (the kokoro declaration, the shipped template and the
README table landed 2026-08-29 to 2026-09-01; receipts in the archive):

* **`workflows/variants/otr_4060_floor.json` `quant_policy`** -- verify what the
  variant carries today before touching it; the 2026-08-28 note said `"none"` ->
  `"bnb_nf4"` was the one line blocking the 10 GB writer, and the 2026-09-01
  clean room ran the floor's E2B writer unquantized at ~2-4 tok/s on 8 GB.
* **README model table** from the compatibility workbook's Baseline Combos tab
  (`outputs/20260828-ungated-models/`, the LIVING fact sheet: edit cells in place,
  never add a changelog tab). The HF-token two-tier story is now in README 2c/3.
* **THE TRAP: `pyproject.toml` edits AUTO-FIRE a registry publish.** The prepared
  release commit is `docs/ship-audit-2026-09-01/pyproject_alpha15.patch` (version
  bump, pycairo/pillow/aiohttp, bitsandbytes and kokoro markers). ONE operator push.
* **The clean-room retry** (Section 2, Batch R7) is what proves this set.

### Section 1.12 image engine defaults: the ruling facts, the DONE 2026-09-01 block and the Leg C bullet

#### 1.12 IMAGE ENGINE DEFAULTS -- RULED 2026-09-01: Klein 4B Q4 GGUF for the low-VRAM set

Operator: *"Klein 4B Q4 GGUF can be the default for Mac / AMD and 30/40/50 series low VRAM
JSON."* Z-Image-Turbo stays the 16 GB NVIDIA default (45 published episodes; nvfp4 on
Blackwell). Facts the row rests on: Klein is 4B, Apache-2.0, a Comfy-Org repackage, zero
refusals across 35 stills in the last sweep; the Q4 GGUF DiT is 2.6 GB
(`Latentiq/FLUX.2-klein-4B-GGUF`, ungated) plus the 8 GB `qwen_3_4b` encoder (offloads before
sampling) and the 0.34 GB VAE; it loads through `UnetLoaderGGUF`, so the ComfyUI-GGUF pack the
LTX 2.5 lane already requires is the one prerequisite (README 2b names it).

DONE 2026-09-01 (the 5080 window, one commit; the canonical keeps `z_image_turbo`):
* Image slots (`announcer_image`, `music_image`, `character_image`) are `flux2_klein` in all
  19 low-VRAM profiles (`otr_nvidia_8gb_*`, `otr_8gb_*`, `8gb_lite`, `otr_4060_*`,
  `otr_nv40_12gb`, `otr_amd8_rocm`, `otr_amd16_rocm`) and `config/machine_classes.json`
  carries `image: flux2_klein` for the 8gb, 12gb and amd classes; variants and
  `docs/MACHINE_MATRIX.md` regenerated. `otr_mac_mps` and `cpu_floor` are untouched: both
  are `draft`, already images-only (`still_motion`), and run `google_image` because no local
  image engine declares `mps` or `cpu` in the registry yet (next bullet).
* Mac and AMD ship IMAGES ONLY: both AMD profiles run `still_motion` for the character lane
  and the procedural viz lanes for announcer and music; the amd class `video` is
  `still_motion` and the matrix reads it.
* **Z-Image recipe: cfg 2.0 -> 1.0** (`nodes/_otr_image_engines/z_image_turbo.py`, the
  reference A/B harness pin, and its tests). A same-seed A/B on nvfp4 AND bf16 showed the
  "bloody faces" look tracks cfg 2.0, not the 4-bit weight; the operator picked the cfg 1.0
  frames. The negative is inert at cfg 1.0, as it is for Flux; `OTR_ZIMAGE_CFG` overrides.
  Receipts: `docs/2026-09-01-16GB-IMAGE-ENGINE-PROBLEM-STATEMENT.md`,
  `docs/ship-audit-2026-09-01/image-jury/zab_*.png`.

Still owed:
* Registry: `nodes/_otr_image_engines/registry.py` gains `mps` on the `flux2_klein` row ONLY
  after one measured render on Apple Silicon; then `otr_mac_mps` flips to Klein. ROCm already
  qualifies (presents as cuda).
* Batch R7 Leg C (RAN 2026-09-02, see Batch R7): Klein renders on 8 GB under the stock
  loader and the Z-Image abort is engine-specific; the 42-min-per-still residency defect it
  exposed is fixed in three commits -- 9b90189a and ad6a635f (the encoder leaves the card
  at its drop, classic and DynamicVRAM paths; a real second-order defect, proven on the
  5080) and da2b7a36 (the first-order one: the dispatcher releases the writer LLM before
  the first local still; the 5080 had a 7.4 GB writer co-resident with every still too).
  Klein on 8 GB under stock launch flags is now MEASURED (Leg C5: ~21 s a still, 1.07 s
  per step, two stills in one process). The matrix says PROVEN only for what PUBLISHED;
  the 8 GB row's image column flips when a Klein episode reaches obs, which now depends
  on the video lane behind it (LTX 2.5 on 8 GB: Leg A's question, being answered by the
  same Leg C5 as it runs on).
* The cfg 1.0 promotion step: the same four A/B cells on three real episode prompts

### Section 4 as it stood (the four lifted rulings; now in OTR_STANDING_RULINGS.md)

## SECTION 4 -- STANDING RULINGS LIFTED FROM ARCHIVED BLOCKS (2026-08-28)

Lifted VERBATIM before their blocks moved to the archive. The full
standing-rulings authority remains `docs/OTR_STANDING_RULINGS.md`.

* From the archived 2026-08-13 runway table, row 4: **"The multi-GPU
  learned-upscale stage itself is CLOSED and must not be reopened."** (The
  narrow `SpandrelEsrgan._resolve_model` hardening closed 2026-08-28; archived.)
* From the same table, row 1: **"Do NOT restart from the Story Lab."** (The lab
  is parked, read-only and being retired; the story work happens in
  production.)
* **Operator acceptance 2026-08-28: LTX 2.5 vocalizing its prompt is CLOSED --
  it is not an open investigation.** Do not re-open it from older text. The
  named-sounds cue table, the joint-AV identity/prose removal, the runtime
  identity-leak guard and the ASR stem auditor are the shipped answer; receipts
  in the archive, `docs/PROD_BUG_LOG.md` and `git log` (`d3cca496`..`5cd4dcc8`).
* From the archived GGUF generation-deadline row (CLOSED 2026-08-25), the
  reusable lesson the row was kept for: **"A reachability question answered
  against the default path only is not answered."**

---

### Section 5 parked voice reference bank: the measured table and why-parked narrative

#### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected.** Operator: *"park it on go forward."* Raised
after the operator observed we should have far more voices than the writer is
being offered. He was right, by a wide margin.

##### The finding, measured live

| pool | count | what it serves |
|------|-------|----------------|
| Bark `VOICE_PROFILES` (`config/cast_pools.py`) | **10** (6M/4F) | what the writer's casting menu offers |
| Kokoro presets on disk | 4 | ANNOUNCER only, a separate namespace |
| **Voice reference bank** (`config/voice_reference_bank.json`) | **204 declared, 153 resolvable on disk** (97M/106F/1N) | IndexTTS2 / Dia / Chatterbox cloning |

`_otr_voice_bank.default_char_engine()` returns **`indextts2`**, promoted to the
shipped character-voice default on 2026-06-04. It is a zero-shot CLONING engine:
`requires_voice_ref = True`, `voice_ref_kind = "wav_path"`. Every reference clip
is a distinct voice.

So the writer casts from **10 Bark presets** while the engine that actually
speaks the characters draws from a **153-voice reference bank**.
`_otr_casting.py` states it outright -- *"Open-character voices are always drawn
from the Bark pool (VOICE_PROFILES in config/cast_pools.py), so the tts_model is
Bark by construction"* -- and `_assert_unique_bark_voices` enforces uniqueness
across those 10.

##### Why this is parked rather than done

`MAX_SPEAKING_CAST = 10` was set from the Bark pool and is therefore a Bark
artifact, not a real ceiling. But raising the constant alone achieves NOTHING:
`_deal_voice_menu` builds the menu from `VOICE_PROFILES` and refuses with
*"voice stock capacity 10 < cast size N"*. The actual work is pointing the
casting menu at the reference bank when the character engine is a cloning
engine.

**That is why it is parked and not done unilaterally.** It changes what
`voice_preset` and `tts_model` MEAN on every cast row, and cast rows are ledger
JOIN KEYS -- `cast[].name` / `char_id` / `voice_preset` / `voice_ref_id` /
`voice_engine`, joined from `lines[].speaker` and `beats[].char_id`. Under the
operator's hard rule (*the writer must OBEY the ledger for downstream content; a
hole in the ledger is a broken render*), a change of that shape needs every
field's owner enumerated BEFORE the call is moved, not after.

### Section 6 standing traps as they stood (now in OTR_STANDING_RULINGS.md KNOWN OPEN)

## SECTION 6 -- STANDING TRAPS AND RECORDED LIMITS (carried knowledge; no scheduled work)

#### Test-harness and tooling

- **The B7 forbidden sweep cannot see an UNTRACKED file, so a new test file passes the
  gate and fails one commit later.** `tests/test_b7_forbidden_sweep.py` builds its
  input from `git diff s29-clean-slate-gate -- *.py`, which covers tracked files only.
  A new test file added and gated in the same session is green; the moment it is
  committed it enters the diff, and a forbidden runtime identifier in it turns HEAD red
  with nothing else changed. Cost one red HEAD. **Not fixed, because the fix is a
  judgment call:** sweeping the working tree instead of the diff would widen the gate
  to every untouched file in the repo. Cheap mitigation until then -- re-run the full
  suite once after the FIRST commit of any new test file.
- A runner that discards the encoder's return value and recomputes the frame count
  independently will disagree with it silently (the 2026-08-23 bake-off finding; the
  runners are gone, the lesson carries into any replacement sweep).
- **LATENT, not reachable today: the fewest-segments partitioner can accept a
  disproportionate trim on a WIDE discrete menu.** WIRE-W1 makes `partition_beat` take
  the lowest segment count that covers, including via a permitted tail trim. On a
  ladder that is always the right trade; on a DISCRETE menu whose largest entry dwarfs
  its smallest it need not be -- covering 1019 frames from a `(10, 999)` menu, two
  segments give `[999, 999]` and discard 979 frames where three give `[999, 10, 10]`
  exactly. **A bound was written, MEASURED and REVERTED, and the measurement is the
  point:** rejecting a trim of a whole smallest-clip turned `[12, 12]` into
  `[12, 4, 4]` on a `min=4 max=12 quantum=8` ladder -- a third render and a third model
  load to recover four frames -- across 4,885 cases in the sweep grid. The widest
  shipped menus are Veo's `(100, 150, 200)` and Pixverse's `(125, 200)`, whose worst
  real trim is 25 frames. Revisit only if an adapter declares a menu with an extreme
  ratio; the reasoning is recorded in `coverage_plan.partition_beat` so the next reader
  does not re-derive the bound and re-ship the regression.

---

### Open risks: the check_compatibility argument in full (now Section 3 question K)

- **`check_compatibility` is RESERVED, not wired** -- operator/planner decision flagged,
  with a 2-of-2 recommendation to RIP (codex and Fable independently, Claude grounded
  both). The argument that decided it: the "it reserves the name" benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation provably
  ignores whatever a client puts under that name (`tests/test_otr_check_cli.py:335`
  asserts a bundle whose `check_compatibility` is a plain integer activates). The only
  artifact that reserves the name is the `EXTENDING_OTR.md` paragraph, which exists
  either way. Case AGAINST: churn on landed green code for zero behaviour change, and the
  plan of record already names the future consumer (randomizer eligibility). Blast radius
  if ripped: ~5 code sites, 2 test files, 3 docs; no workflow JSON, no routing, no
  source-payload consumer. **Not a coder chunk** -- either ratify the inert constant or
  schedule the rip as a planner chunk. Proposed doctrine line: a name published to
  clients before its consumer exists lives in the client-facing DOC as "reserved, no
  contract, ignored if defined" and nowhere in executable code, because code that names
  an interface is read as enforcing it.

---

## 2026-09-03 -- Prompt v3 Half A, the registry route gate, and three replay bugs

Moved out of GO_FORWARD_PLAN 2026-09-03 (forward-only rule). Closed receipts; not read to resume.
Full records: `docs/2026-09-02-animatediff-ledger-experiments/prompt-rule/` (driver anchor,
operator rewrites, measured slot budget, as-built change list, lane audit),
`kibitz-runs/2026-09-02-prompt-v3-crux/` (r1-r4 + QA judgments), and PBUG-20260903-01/02/03.

**A HUNG CANONICAL LEG WAS FOUND RUNNING AND LEFT ALONE (5080 window, 2026-09-02 22:00).**
`scripts/otr_canonical_api_run.py --profile otr_ltx25_high_foley_plus --act-count 1
--source-bank "roll (any eligible bank)" --visual-style "roll (any style)" --timeout 10800`
started at 20:32 and is still resident, but it is doing nothing: the worker has used
**1.8 CPU-seconds in ninety minutes**, holds a 48 MB working set (it never imported torch),
has **no child process**, and nothing is listening on port 8000. It created
`otr/episodes/pending_20260902_213847/` at 21:38 and has written not one file since. By the
operator's own five-minute rule that leg is a fail.

**Not killed, deliberately.** It is the other window's process, it holds no VRAM (the GPU sat
at ~3 GB desktop baseline throughout), and it blocks nothing -- so killing it would destroy
someone else's context for no gain. It will expire on its own `--timeout 10800` at 23:32.
It also costs no expected episodes: the operator froze GPU work tonight ("don't waste any GPU
runs until the prompting is fixed"), so an idle box is what he asked for. Fifteen episodes did
publish to `otr/obs/` today, the last at 21:08.

Whoever owns that window should read its own launcher output; from outside the process there
is nothing further to learn.

**ITEM 3 -- PROMPT v3 HALF A IS BUILT AND REVIEWED (2026-09-03, overnight).** The
four-round arc closed CONVERGED and the code followed it: r1 Fable cold plus
Antigravity, r2 Codex, r3 Cursor, r4 Sonnet, then a scoped finished-diff QA. Eight
grounded reversals of the driver's own plan, of which three would have shipped a
broken or pointless change:

* **the coat is in the LEAF, not only in the motif** (r3) -- dropping the motif
  alone would have changed the prompt and not the picture, and read as a null
  result for the wrong reason;
* **the mode law was never framing boilerplate** (r4) -- the comment above it
  records 4-of-4 recognisable subjects with a concrete noun in the prompt and
  0-of-4 without, so deleting it is only safe because the crux kernel is a total
  ladder that can never come back empty;
* **the seed is not derived from the prompt** (r2) -- `request_hash` mixes brief,
  cast, beat and character, so the A/B is same-seed by construction and needs no
  derivation machinery at all. The corollary trap: the crux must never be written
  into `story_brief_terms`, which IS hashed.

**What it composes:** pack cue, crux kernel (the episode's own `key_objects` in
its own setting), the episode's light, a world motion, and a vantage mapped from
the stored mode. It reads no `motif_cue`, no `drawable_beat` and no mode law, and
it changes nothing the author stored -- so a frozen episode replays with a
byte-identical seed and a different picture. Measured 32.9 tokens against v2's
40.2, inside a 77-token window.

**Proven on CPU:** the full suite (only the three known worktree artefacts, all
94 of them green in the main checkout), 5,314 fuzz combinations over hostile
ledger shapes, the real render path driven with four ledger variants, and 363
targeted tests.

**The GPU freeze lifts by its own terms** -- the prompting is fixed -- and the
first run is ONE leg, not two: the published "The Faded Ledger" already IS the v2
arm (same plan, same seeds, same engine), so only the v3 replay has to render.
Bundle frozen and the runner dry-run clean. `otr_verify_replay --ab` asserts
equal seeds and different prompt shas.

**Half B remains open** and is where the operator's motion rule lands: the beat's
own dialogue in front of the writer, `world/thing/hand` vocabulary, subject
coverage. Half A buys the story's object, place and light on every beat; Half B
buys its motion. Item 3b (the other lanes) is amended -- ADD a crux clause beside
the appearance on silent image-to-video lanes, never drop the face on redundancy
alone; `wan_ti2v`'s 83-word face inside a 100-word cap is the one place something
must give.

**THE A/B IS ON SCREEN (2026-09-03 00:20).** Two episodes in `otr/obs/` under the
same title, told apart by their timestamp:

* **21:08** -- the old prompt (`signal_lost_the_faded_ledger_20260902_210812...`), 145.4 MB
* **23:59** -- Prompt v3 (`signal_lost_the_faded_ledger_20260902_235930...`), 146.8 MB

Same plan, same seeds, same 107.0s, comparable size at every pipeline stage.
`scripts/otr_prompt_ab_report.py` returns **seeds identical on 8 of 8, prompts
differ on 8 of 8**, and every prompt composed live at exactly the token count
predicted on CPU. Whatever differs on screen differs because of the words.

**It took four legs, and three of them were bug reports** -- PBUG-20260903-01,
-02 and -03, all in replay code written for the instrument this week, all the
same shape: a replay skips a stage and something downstream still needs what
that stage produced. **Two of the three published green**, which is the finding
worth carrying: `obs_publish OK` is not proof of an episode. The stage table
(render / blend / caption / credits / mux, each with its duration) is what
caught them, and a stage that changes the DURATION was the defect every time.

**Queued behind it:** fresh 1-act episodes on `video_art` and `recur_frac`, the
two packs the operator asked about. Those are new stories rather than the same
one twice, so they show v3 under a different pack; they are not a comparison.

**Next, in order:** his eye on the pair; then Half B (the beat's own dialogue in
front of the writer, which is where his motion rule lands); then item 3b on the
other lanes as amended. The first thing to tune if the pictures read as busy is
the LIGHT slot -- his own rewrites carry three units where v3 carries four, and
the light is already first in the drop order.

**THE TWO STYLE LEGS LANDED (2026-09-03 01:42).** Fresh 1-act episodes on the two
packs the operator asked about, both `RESULT SUCCESS`, both verified as real
full-length episodes rather than trusted on their exit code:

* **The Strain of a Sleeve** -- `video_art`, public_domain, 8 clips, 100.2s, 118.5 MB.
  *"video-art feedback style. a spinning turntable in the riverbank, a low
  shimmer moving on it, lit against the dark, the light moving"*
* **The Weight of Grief** -- `recur_frac`, scifi_news_pro, 25 clips, 247.3s, 339.1 MB.
  *"recursive fractal light field. a bakelite radio set in the clinical
  laboratory, fluorescent hum, turning a slow quarter and stopping, the object
  large in the frame"*

Both carry their pack cue, both draw the story's own `key_objects`, and both
titles came from the story -- no harness label reached a title card.

**FIVE FILES ARE IN `otr/obs/` AND TWO OF THEM ARE THE KNOWN-BROKEN ONES.** The
7.9 MB `..._231401` and `..._233738` are the PBUG-20260903-02 and -03 casualties:
one second of picture each. They are LEFT IN PLACE deliberately -- nothing is
swept out of obs -- but they are named here so nobody opens one and concludes
Prompt v3 is broken. **The v3 arm to watch is `..._235930`, 146.8 MB.**

**One wart worth a line, not a fix tonight.** The kernel joins subject and place
with a fixed `"in the"`, which reads correctly almost everywhere and awkwardly on
a few settings ("a spinning turntable **in the** riverbank"). The operator's own
rewrites vary the preposition ("at a reservoir", "in a large water reservoir").
A small per-setting preposition choice would fix it; it is cosmetic and it waits
for his eye on the pictures first.


## STATUS -- WHAT THE 2026-09-03/04 SESSION SETTLED (archived 2026-09-04 evening; done-work narrative, moved whole -- the one forward pointer it carried, item 3d, lives in the queue)

## STATUS -- WHAT THE 2026-09-03/04 SESSION SETTLED (read before re-deriving any of it)

**2026-09-04, second half -- the runpod-found fixes landed in the commit that
carries this paragraph, as ONE invariant: a machine fact has ONE owner, and a
test proves the copies agree.** Full four-round `kibitz-plugin:kibitz` arc
(Codex gpt-5.6-sol, Antigravity Gemini 3.8 Flash, Cursor grok-4.6 -- r4's
Cursor seat by a Sonnet substitute plus the operator's manual paste), a Sonnet 5
QA pass on the named functions (clean; verified live that with nothing pinned
every answer on the 5080 resolves to the same binary and the same tree -- the
only string-level change is a resolved path where a bare literal used to be), a
Fable gate (ship-with-fixes, applied), two manual passes (Cursor, Antigravity)
on the pushed tree, then the suite -- **13484 passed, 0 failed** -- and the
live publish matrix (three pairwise-distinct roots, a pinned ffmpeg copy, the
shipping 8 GB profile; its receipt is the docs commit that tombstones 1.4a
row A's "still owed" line). What landed, in 1.4a's rows:
ffmpeg has one resolver (`_otr_shared/ffmpeg.py`; measured live, 9 of 10 sites
had ignored `OTR_FFMPEG` on a box with ffmpeg on PATH -- PBUG-20260904-02); the
output root has one owner and `OTR_OBS_DIR` is a declared publication root
(R-A); the remote video adapters DECLARE `session_residency = "remote"` (the
cloud base's six rows plus the two Google adapters -- the seven that split a
long beat are the roster's named gap; Kling inherits it and never splits)
and BeatSession no longer asks them for handles they do not have (R-B, the
word_razzle pod failure -- PBUG-20260904-03); the models root has one spelling;
the widget-drift gate has no silent exemption; the nvenc probe is cached per
binary. Three AST guards (`test_ffmpeg_single_resolution`,
`test_output_root_single_owner`, `test_models_root_single_spelling`) make the
claim a test, not a docstring. Bible 12.151-12.154 promoted. The pod
(`k0ph3m9491yp6s`) was stopped at 7:15 AM after its chain went idle; its seven
overnight episodes are in `otr/obs`. Side find: the 5080's own Q4_K_M writer
copy predates the 2026-08-29 pin (PBUG-20260904-04) -- re-downloaded. Open
threads live in 1.4a row G.

**The big one: character directives were being thrown away, 100% of the time.**
`derive_creative_directives` batches up to 15 beats and asks for one JSON row
each, but reached the model through a wrapper hardcoding `max_new_tokens=300`.
A row costs ~73 tokens, so a full batch needed ~1,100 and got 300: the reply
stopped mid-object, the parser returned `{}`, and the reseed loop retried the
SAME prompt under the SAME ceiling. Every character beat fell through to the
deterministic template -- a character PORTRAIT with no action verb in it -- which
is why the picture never matched the story. Fixed (PBUG-20260903-07) and proven
live: 14 of 14 beats from the writer, zero warnings, where it had been 0 of 14.

**Also fixed and pushed:** `"camera locked"` as the default when no camera was
given (3 lanes); the compactor silently dropping the CAMERA clause over a
12-word cap (2 of 32 registers, both in packs the operator watches); `Voice:`
reaching picture models in 90% of joint-AV prompts; the one-sentence camera ask,
now requiring a STRATEGIC move and banning framing-only answers; the wan/fastwan
bookends, which previously shipped a static seed with ZERO verbs.

**A portability bug only a second machine could find.** `scope_draw._has_nvenc`
asked whether ffmpeg was COMPILED with nvenc, not whether it can ENCODE -- the
third copy of a mistake already fixed twice, and `encode_sink`'s docstring
claimed to be the only such decision, which is why it survived. On a rented 4090
that lists the encoder and cannot open a session, `viz_camera` went FAIL -> PASS
on the fix. Any AMD / Mac / container host hit the same wall.

**`ideogram4_local` was never Blackwell-only** -- only its default was. It now
walks a precision ladder (nvfp4 -> fp8_scaled -> int8_convrot), all three in the
same UNGATED Comfy-Org/Ideogram-4 repo. NOT an 8 GB lane: fp8 is ~29 GB all-in.

**Published filenames now say what made the episode** --
`<title>_<ts>__<style>__<video>__<image>__<tts>__<bank>_final.mp4`. The old tail
was compositing noise and actively misleading: this session read `procgen` as a
render engine and built a wrong diagnosis on it. Obs copy only; archival
untouched.

**Rented-4090 lane matrix (1 act each), and read the failures correctly:**

| outcome | lanes |
|---|---|
| PASS | viz_camera, viz_green, still_flat, still_motion, still_pan, still_word, animatediff15_v3_haunted, ltx_8gb |
| TIMEOUT (driver's 40 min, render survived -- NOT a defect) | wan_ti2v, fastwan_8gb |
| OOM at decode (known 24 GB Ada negative) | ltx25_foley_plus, ltx25_mime, ltx25_video |
| dependency absent on that pod | ltx_video, mesh_stage (portable Blender) |
| REAL defect -- FIXED 2026-09-04 | word_razzle -- refused every beat long enough to split, because `BeatSession` demanded a handle identity from an adapter that holds no handles. The seven remote adapters now DECLARE `session_residency = "remote"` and the demand is skipped for exactly them, fail-closed (kibitz runpod-found-fixes R-B; PBUG-20260904-03) |

A 3-act AnimateDiff episode also passed unattended in 33.7 min.

**STILL OPEN, and today's audit CONFIRMED it rather than fixing it:** item 3d
below -- ten of eleven lanes lead the prompt with the cast's face paragraph via
`motion_common.compose_parts`, 83 words of a hard 100-word cap on `wan_ti2v`, so
the camera clause falls off the end. That is a SHARED-composer change touching
eleven lanes on both machines and is deliberately not a drive-by.

Eight pod-driving lessons landed in `docs/RUNPOD_INSTALL.md` section 7A and its
failure atlas.

---


## ARCHIVED 2026-09-04 evening from ## THE QUEUE -- READ THIS FIRST (ordered by the operator, 2026-09-02)

* **alpha.17 published 2026-09-03 17:56Z, commit `524426ee`, Action green, and VERIFIED
  AGAINST THE ARTIFACT rather than the commit.** The CDN zip
  (`.../2.0.0-alpha.17/node.zip`, 6,397,276 bytes, 814 files) was downloaded and grepped:
  `OTR_ENABLE_HTTP_RENDER_ROUTES` present in `__init__.py`, the pycairo marker and the
  `kokoro-onnx` line present in the shipped `requirements.txt`, and none of `tests/`,
  `.claude/`, `.github/`, `kibitz-runs/` leaked. The registry recorded **19 dependencies**
  (alpha.16 recorded 18) -- the static-list check that alpha.3 taught us to run.
  **ITS SCAN HAS SINCE LANDED (2026-09-03 19:06Z): `Flagged`, 158 findings, ALL
  `severity: info`, ZERO `critical`** -- re-read directly from
  `?include_status_reason=true`, not carried over from the doc. The delta from
  alpha.16's 157 is exactly +1 `python_environment_manipulation` (102 -> 103), which is
  the route gate's own `OTR_ENABLE_HTTP_RENDER_ROUTES` env read that alpha.17 added.

* **pycairo was the ONLY source-build-only dependency -- audited, not assumed
  (2026-09-04).** Every line in `requirements.txt` was evaluated against a
  linux / cp310 / x86_64 marker environment (what the extract container is) and then
  checked against PyPI's own file list for that release. All 17 selected packages resolve
  to either a manylinux wheel or a pure-python wheel; `kokoro-onnx` and `pycairo` are
  correctly skipped by their markers. The transitive suspect named in the handoff is also
  clear: `misaki` is pure-python and `spacy` / `thinc` / `blis` all carry cp310 manylinux
  wheels.

* **alpha.16 scanned 0 critical / 157 info (2026-09-03), the first zero-critical scan
  since alpha.8.** The two `critical` `prohibited-string` findings that flagged alpha.9
  through alpha.15 were the writer declaring the session-bearer hidden input beside
  `api_key_comfy_org` (PBUG-20260902-04); removing it cleared both.
  `tests/test_registry_prohibited_strings.py` guards every shipped file. Confirmed still
  true of alpha.17: the four remaining `auth_token_comfy_org` occurrences in `nodes/` are
  byte-identical to alpha.16 and are a consumer-side NAME CHECK, not a declaration.

* **THE NODE COUNT WAS A DEPENDENCY BUG, NOT AN APPROVAL PROBLEM -- FIXED IN `aff1f9c4`.**
  The "N Nodes" badge is fed by Algolia's `comfy_nodes` array, which is populated by
  Comfy-Org's `node-pack-extract`: a Cloud Build container that boots a headless CPU
  ComfyUI on Linux, runs a bare `pip install -r requirements.txt` under `set -e`, and
  reads `/object_info`. **It is independent of Flagged/Active, so no approval would ever
  have populated it.** Measured: `/versions/2.0.0-alpha.16/comfy-nodes` returns
  `{"comfy_nodes": null, "totalNumberOfPages": 0}` where
  `comfyui-videohelpersuite/versions/1.7.9` returns real rows. Cause: `pycairo` publishes
  21 Windows wheels, 1 sdist and **zero Linux wheels**, so that install had to build from
  source against `libcairo2-dev` and died. `requirements.txt` now marks it
  `sys_platform == 'win32'`, mirrored into `pyproject.toml`.
  **Cost, measured not asserted:** one function. The only `import cairo` in
  `scope_draw.py` is inside `paint_mandala`, so importing the module never imports cairo;
  `freq_bars_green`, `cfr_flags`, `find_ffmpeg`, `encode_silent_mp4` are cairo-free, so
  the scope overlays, caption burn, silent composite and encode sink are untouched.
  `viz_mxc_mandala` guards `load()` and `assert_usable()` and appears zero times in
  `otr_canonical.json`.
  The re-check ran: `/versions/2.0.0-alpha.17/comfy-nodes` is still `null`.
  The paragraph that stood here read that as "something else in the install also fails
  on Linux -- next suspects are `kokoro`'s spacy/thinc/blis chain against the 600 s
  extract timeout". **That is wrong, and acting on it would have burned a coder window
  on infrastructure we do not own.** It offered a binary -- our fix worked, or our
  install still fails -- and omitted the third option, which is the true one.
  `?include_status_reason=true` exposes a field the theory was built without:
  **`comfy_node_extract_status`**, which has TERMINAL values (`success` and
  `invalid_format` both observed). Measured across **360 registry packs**: only 64 have
  ever recorded a `success`, and **the most recent successful extract ANYWHERE in that
  sample is 2026-04-28**. rgthree (Active, 47 prior successes) has versions sitting
  `pending` since 2026-05-09; kjnodes (Active, 24 successes) since 2026-05-05;
  florence2 since 2026-05-06. **Comfy-Org's `node-pack-extract` has produced no
  successful extract for any pack in over four months.** OTR records zero successes
  because OTR's first publish was 2026-08-22 -- four months AFTER the pipeline last
  worked. It never had a chance, and no `requirements.txt` change reaches it.

* **ALPHA.17 SCANNED 2026-09-03 19:06Z: 158 findings, ALL `info`, ZERO `critical`, status
  `Flagged`.** The credential fix held across the bump. The delta from alpha.16's 157 is
  exactly +1 `python_environment_manipulation` (102 -> 103), which is the route gate's own
  `OTR_ENABLE_HTTP_RENDER_ROUTES` env read that alpha.17 added -- i.e. the one new finding
  is the feature we shipped, behaving as designed. Flagged with zero criticals is the
  expected outcome, not a failure.

**2. THE KOKORO-ONNX BACKEND -- DONE 2026-09-02, receipts archived.** Both proofs
published and eyeballed ("this episode is perfect, voices great"). Its one leftover, the
kokoro-onnx dependency line reaching registry installs, rides the alpha.17 bump in item 1.
Detail in `docs/GO_FORWARD_ARCHIVE.md` and `docs/2026-09-02-kokoro-onnx/`; row in
Section 1.1.

* **3a. Character gender ladder -- DONE 2026-09-02.** Receipt archived; row in Section 1.2.

  The AnimateDiff lane was composing an
  invented costume plus a leaf written from that costume plus a framing law -- none of the
  three knew the story, which is why it drew a charcoal coat and a satchel for an episode
  about film canisters. Half A (`b198026a`'s parent merge `b97cce6d`) composes pack cue +
  crux kernel (the episode's own `key_objects` in its own setting) + light + world motion +
  a vantage mapped from the stored mode, reading neither the stored motif nor the leaf nor
  the law. Render-time only, so a frozen episode replays with a byte-identical seed --
  `render_request_hash` never included the prompt. **Proven:** same-seed A/B in `otr/obs/`
  (21:08 old vs 23:59 new, seeds identical 8/8, prompts differ 8/8), every prompt composed
  live at exactly the token count predicted on CPU, 32.9 tokens against v2's 40.2.
  Operator verdict on the video_art leg: "surprisingly good... it's about Huckleberry Finn
  and there's actually a river!"

Klein stills run on 8 GB under stock launch flags (Leg C5: ~21 s a still) and LTX 2.5
renders there at ~14 min a clip; the Leg C5 episode PUBLISHED 2026-09-02 12:10
(`signal_lost_rationed_breath_20260902_060027`, 24 clips, 6 h 35 min; receipt in
`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`). (a) DONE 2026-09-02: the 8 GB row is PROVEN with its image lane on the
all-kokoro clean-room episode `signal_lost_the_ledger_of_shadows_20260902_134447` (the
operator's eyeball), receipt merged into the row's 4060 entry (7 episodes), matrix
regenerated;

**7. DOCS DELETION PASS -- DONE 2026-09-02.** `docs/` went 644 tracked files to 237 across
three deletion-only commits (45895801, 2bf15784, 607a5ee7). Kept set in Section 1.8.


## ARCHIVED 2026-09-04 evening from ### 1.1 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (design item, kibitz arc BEFORE code)

**STATUS 2026-09-02: BUILT, proofs pending (see queue item 2).** Shape as shipped:
`nodes/_otr_audio_engines/_kokoro_backends.py` (torch path moved verbatim; ONNX path
CPU by design, explicit provider, 4-thread cap, voices from the existing `.pt` files
through a digest-named npz); `eng_kokoro.py` selects per `load()` via
`OTR_KOKORO_BACKEND=auto|torch|onnx`; the prefetch fetches `onnx/model.onnx` at boot only
when the ONNX backend will be used; canonical nodes 80/81 default both voice slots to
kokoro on `kokoro_builtin`; a character-side engine-agreement guard; five profiles fixed;
complementary `kokoro` / `kokoro-onnx` markers in `requirements.txt`; the generated "Voice
engines" table in the matrix. Design record: `docs/2026-09-02-kokoro-onnx/`.

Measured 2026-09-01: `kokoro-onnx>=0.6.1` + `onnxruntime` pip-install clean on Python 3.13
(kokoro-onnx pins `>=3.10,<3.14`; espeak-ng ships bundled for win / mac / linux; the torch
`kokoro` package stays `<3.13`). Receipts: PBUG-20260901-04 and `docs/OTR_STANDING_RULINGS.md`
"ALL AUDIO LANES SHIP ON KOKORO".


## ARCHIVED 2026-09-04 evening from ### 1.2 CHARACTER GENDER LADDER (queue item 3a) -- the SPEC REWRITE is written; next is ONE review round, then code

**DONE 2026-09-02.** Round: Antigravity r2 on the driver anchor
(`docs/2026-09-02-gender-ladder/driver_anchor.md`, sections 8 and 9 carry the fold and the
proof) -- 7 must-fixes, all grounded and taken. Code: tiers 3-4 in
`scripts/otr_stamp_character_genders.py` (recall into a committed per-bank
`character_gender_index.json`, the first-name pool, the body-hash-anchored merge, the
Shakespeare scene stamper), the verdict fields and the alias-aware join in
`nodes/_otr_roster_gender.py`, greedy decoding at temperature 0 in the constrained closure.
Proof: `signal_lost_intensity_in_the_drawingroom_20260902_152901` -- ELIZABETH BENNET female,
MR. DARCY male, COLONEL FITZWILLIAM male, all `llm_recall` / `recalled` in
`meta.cast_source_contract.evidence`; the leg before it (`unhand_me_sir_20260902_151527`) is
the before-picture where Darcy rolled female because the join ignored the aliases.

Evidence and the v1 spec: `docs/2026-08-05-character-gender-ladder-SPEC.md` and
`docs/GO_FORWARD_ARCHIVE.md` (the prose lane stamps `characters: None`; the Shakespeare
sidecars carry genders; the consumer `_otr_roster_gender.py` is lane-neutral and correct).
v2 folds v1; the rulings above are baked into v2.


## ARCHIVED 2026-09-04 evening from ### 1.4 OPEN DEFECTS THAT ARE CODING WORK (queue item 3c; a leg may prove some later, none needs a leg to FIX)

**A. VERIFIED, HIGH -- the output tree is decided TWICE, and the copies disagree.**
`nodes/_otr_paths.py` is the documented owner (`OTR_OUTPUT_DIR` is "highest
precedence", :74-106). `nodes/otr_master_audio_mux.py` imports NOTHING from it and
carries its own `_episodes_root()` (:695) and `_obs_dir()` (:709), neither of which
reads `OTR_OUTPUT_DIR`. Proven by EXECUTION, not by reading:

```
OTR_OUTPUT_DIR=D:\RenderOutput , folder_paths=C:\ComfyRoot\output
_otr_paths.otr_episodes_root()  -> D:\RenderOutput\otr\episodes
mux._episodes_root()            -> C:\ComfyRoot\output\otr\episodes   AGREE? False
_otr_paths.otr_obs_dir()        -> D:\RenderOutput\otr\obs
mux._obs_dir()                  -> C:\ComfyRoot\output\otr\obs        AGREE? False
```

**Blast radius is the terminal publisher.** `_episodes_root()` gates
`_inflight_episode_for_stem()` (:797) -- a mismatch silently rejects a valid
in-flight ledger and falls back to filename-suffix peeling the module itself calls
fragile. `_obs_dir()` is where `_publish_to_obs()` writes the deliverable. The same
bare-`folder_paths`-or-`"."` idiom is copy-pasted again in `otr_caption_burn.py`
(:349) and `otr_silent_composite.py` (:1692).

**Why it has not bitten:** both launchers hand-pin `OTR_OUTPUT_DIR` =
`--output-directory` = `OTR_OBS_DIR` in lockstep -- by human discipline in two
separate shell scripts, not by any code invariant.

**CORRECTED BY KIBITZ r1 (2026-09-04) -- the first draft of this row was wrong
in three ways, and it matters because the first fix proposed would have broken a
contract test.** Full record in
`kibitz-runs/2026-09-04-runpod-found-fixes/r1/{judgment,final}.md` (local, gitignored):

* **Four owners, not two.** `_otr_ledger.py:182` reads `OTR_OBS_DIR` too, and
  `__init__.py:97-108` WRITES `OTR_OUTPUT_DIR` at import inside ComfyUI -- that
  pin, not "the launcher was skipped", is the mechanism behind the Desktop /
  bare-`main.py` split.
* **Two concepts, deliberately named apart.** The *episode workspace root*
  (`_episodes_root`) IS a true duplicate that ignores `OTR_OUTPUT_DIR` -- delegate
  it. The *publication root* (`_obs_dir`) is NOT: `_otr_paths.otr_obs_dir()`
  (:401-429) never reads `OTR_OBS_DIR`, which is a first-class override read by
  seven files including five launchers and exists to point OUTSIDE the tree (the
  documented two-tree split). The mux is right to honour it; `_otr_paths` is the
  incomplete one. "Delegate the mux to `_otr_paths`" for obs would have broken the
  split AND tripped `_validate_contract` (:220-236 raises on an out-of-tree path).
* **A contract test pins it.** `tests/test_output_tree_contract.py:48-50` lists
  `otr_obs_dir` in `_NOARG_HELPERS` and asserts in-tree. Any change ships WITH
  that test updated, or does not ship.
* **Open ruling, deciding in r2:** an explicit pin SKIPS the in-tree assert
  (driver's first lean) or registers as a *declared second authorized root* the
  way `_otr_ledger.py:168-191` already does (panel 2-1, and now the driver's lean
  -- it keeps the validator meaningful instead of holing it).
* Both docstrings -- mux "ONE owner", `_otr_paths` "THE obs dir" -- are false and
  must become true or go; this is the nvenc shape.
* **Acceptance:** Path equality is necessary and NOT sufficient. A live headless
  publish with `OTR_OUTPUT_DIR != folder_paths.get_output_directory()` landing in
  the watched tree. Caption-burn / silent-composite fallbacks are a rostered
  follow-up, not equals of `_publish_to_obs`.

**RESOLVED 2026-09-04 (kibitz r2-r3, built the same day).** Ruling R-A: the
pin is honoured INSIDE `_otr_paths.otr_obs_dir()` and skips the in-tree assert
for that path only (returned as typed, no `resolve()`); `_validate_contract`
untouched; the ledger already authorized both roots. The mux's two functions
are thin delegates (names kept -- the publication tests monkeypatch them).
r3 grew the scope: the canonical graph's node 84 (`OTR_SilentComposite`, empty
`output_path`) writes via `_default_out`, and node 93 writes beside it, so the
"rostered follow-up" above would have left the canonical chain split -- silent
composite, caption burn AND `portrait_ledger._output_base` now delegate too.
Enforced by `tests/test_output_root_single_owner.py` (AST, named allowlist:
`eng_mesh_stage` because ComfyUI's SaveGLB refuses any path outside
`folder_paths`' own dir -- NOT a leftover twin, do not "retire" it -- and
`vram_context_test`, a diagnostic outside the contract). conftest now strips
`OTR_OBS_DIR` at import (the launch recipe sets it on this box). The `__init__`
pin is preserved deliberately; removing it is its own design item.
**PAID 2026-09-04 09:43-10:04 PDT -- the live publish matrix, measured.**
`workflows/otr_canonical.json`, one act, on the 5080 with `--output-directory
C:\Users\jeffr\otr_accept\A`, `OTR_OUTPUT_DIR=...\otr_accept\B` and
`OTR_OBS_DIR=C:\Users\jeffr\Documents\ComfyUI\output\otr\obs` (pairwise
distinct; the shipped soak `.cmd` locksteps all three, so a launcher derived
from it ran), `OTR_FFMPEG` pinned to a COPY of the binary while the real one
stayed on PATH, and a watcher recording every ffmpeg process with its parent.
Result: `RESULT SUCCESS`; nothing new under A; silent, blended and final all
under B; the watched folder went 113 -> 114 with the mux's LOUD publish line;
every ffmpeg the server spawned ran from the pinned copy (the PATH binary
appeared only under pytest parents from the concurrent suites). The leg ran
the draft `8gb_lite` profile on the canonical default bank (`media_archive`),
not the shipping 8 GB row: the launcher's profile substitution did not take,
and the receipt says what ran. Five attempts, each a measurement: attempts 3
and 4 surfaced PBUG-20260904-05; attempt 5 also exposed PBUG-20260904-06 (the
ledger's `meta.paths.obs_final` still pointed at the planned archival-name
file, not the published one) -- fixed in its own commit with a pairing test.

**B. VERIFIED, LATENT -- ffmpeg is resolved nine ways.** Three are BYTE-IDENTICAL
copy-paste (`otr_caption_burn.py:53`, `otr_master_audio_mux.py:49`,
`otr_silent_composite.py:123`) and each self-documents that this defect class has
already hit the pack twice. The same three correctly delegate `_ffprobe_bin()` to
`_otr_shared/ffprobe.py`, so the pack knows the right shape and did not use it here.
Two have live gaps, dormant only because nothing calls them:
`_otr_shared/encode_sink.py::find_ffmpeg` (:20) never checks `OTR_FFMPEG`, and
`_otr_shared/content_oracle.py::_ffmpeg` (:88) has no PATH fallback at all (that
module has zero importers). The rest disagree on what to return when nothing
resolves (`None` / `""` / bare `"ffmpeg"`).

**RESOLVED 2026-09-04 -- and it was not latent (PBUG-20260904-02).** The count
was twelve, not nine, plus three sites that ran a literal `"ffmpeg"` with no
resolver at all (`otr_post_upscale_procgen_blend` -- canonical node 93 -- ,
`eng_google_lyria`, `foley_stems`). Six copies read the caller's own bare
default as an operator choice, so on any box with ffmpeg on PATH the pin was
never consulted: measured live on the 5080, 9 of 10 resolvers ignored
`OTR_FFMPEG`. The cloud preflights refused env-only installs. ONE owner now:
`nodes/_otr_shared/ffmpeg.py::resolve_ffmpeg` (explicit choice -> pin -> PATH
-> the Windows install dirs -> `None`; reuses the probe's bare-name rule);
every runtime site a thin adapter with its own refusal; `resolve_ffprobe`'s
sibling steps go through it. Guard: `tests/test_ffmpeg_single_resolution.py`,
three AST rules, NO allowlist. Matrix: `tests/test_ffmpeg_resolution_precedence.py`,
PATH left available in every case. The nvenc cache is keyed per binary and the
second cache in front of it (`video_engine._check_nvenc`) is gone.
Out of scope, named: `scripts/otr_ingest_pd_voices.py:101`,
`scripts/otr_macbeth_probe.py:603/620/1196` (scripts cannot import the owner
without the package; convert when a script is next touched).

**C. VERIFIED, LATENT -- the models root is decided twice.**
`_otr_gguf_backend.py::_models_root()` (:891) existence-gates the legacy
`C:\ComfyUI-Models` before falling back to `folder_paths.models_dir`.
`_otr_video_engines/wan_shared.py::configured_models_root()` (:35) returns that
literal UNCONDITIONALLY with no existence check and no `folder_paths` fallback --
while its docstring claims "folder_paths is the authority and is probed first",
which is true of its callers and false of the function. Harmless today because
every caller probes `folder_paths` itself first, but the docstring invites
standalone use, and a direct caller would break on exactly the fresh-install box
this repo's notes keep warning about.

**RESOLVED 2026-09-04 -- delegated, not reworded.** `configured_models_root()`
now returns `str(_models_root())` through a LAZY import (the module's V-12
cold-import statement holds, and `tests/test_models_root_single_spelling.py`
pins both the equality under every env state and the absence of a top-level
import). The "ONE spelling" sentence is true instead of admitting it was false.
The third convention, `_otr_paths.comfy_models_dir()` / `OTR_MODELS_DIR`, stays
parked (cursor r3: do not open a third env in this diff). A FOURTH spelling
belongs to the same parked item (agy r4): `_otr_image_engines/flux2_klein.py:209-215`
probes `folder_paths` first, then the two env vars, and never the legacy tree --
the inverse of `_models_root`'s order. When the merge happens it has four
owners to retire, not three.

**D. VERIFIED, RISKY -- the widget-drift HARD GATE has a silent escape hatch.**
`_otr_workflow_validator.py:172` -- `try: expected = _expected_slot_count(...)
except Exception: continue`. This backs the gate at :497 that exists to catch the
BUG-210/253/281 silent-widget-shift class. Any node whose `INPUT_TYPES()` raises is
silently exempted with no log, and the gate can report `widget_vector_drift=0`
having never checked it. Owed: log the skip at minimum, and decide whether an
un-introspectable node should fail the gate rather than pass it.

**RESOLVED 2026-09-04 (`ad89a45b`).** A raising `INPUT_TYPES()` is a named drift
FINDING (node id, type, exception class, repr'd message), so the hard gate
fails; every class reaching that line is ours (third-party types are skipped
above it), so there was no foreign-node cost to weigh. Precise blast radius,
found while writing the public-path test: `validate_workflow_contract` already
raised on a raising `INPUT_TYPES()` for `OTR_`-prefixed types BEFORE the drift
gate, so the node's own `validate()` was fail-closed all along -- the hatch
mattered for the STANDALONE callers of `widget_vector_drift` (the canonical-graph
integrity test could report zero drift on a class it never checked) and for any
mapped class without the prefix. The `validate_anyway` tooltip no longer claims
the switch "never blocks the episode". 7 tests in
`tests/test_widget_drift_gate_no_silent_exemption.py`.

**E. VERIFIED, DEAD -- 12 symbols with exactly one repo-wide reference (their own
definition), confirmed by `git grep -n -w -F`:** `MAX_HEADLINE_CLEAN_CHARS`
(`_otr_scifi_p0_contract.py:23`), `PLATE_DIRNAME`
(`eng_ghost_signal_stillin_lab.py:72`), `RADIO_HOST_FACE_NEG`
(`otr_meta_brief_image_prompt.py:322`), `CRT_RED`/`CRT_WHITE`/`CRT_MAGENTA`
(`video_engine.py:68-71`), `HARNESS_VERSION`/`MACHINE_CEILING_MB`/`NVML_FLOOR_MB`
(`scripts/_otr_b_spikes/_b_harness.py`), `ffprobe_timebase`
(`scripts/audit_otr_full_run.py:101`), `corpus_ledgers`
(`scripts/otr_clean_stage_lab.py:529`), `_negative_phrases`
(`scripts/otr_style_traceroute.py:213` -- and its logic is duplicated inline in the
function directly below it, which is the slop shape exactly).
`MAX_HEADLINE_CLEAN_CHARS` was queried as a possible UNWIRED cap rather than dead
code; the file answers it at :50 -- "headline/summary are short and bounded by the
fetcher" -- so the bound exists upstream and the constant is vestigial. Clean delete.

**F. VERIFIED, SMALL -- vestigial rows.** `_otr_structured_call.py:110` calls
`REPAIR_TEMPERATURE` the shared public export and `_REPAIR_TEMPERATURE` a compat
alias; it is backwards (the underscore name is the one every call site and test
uses, the public one has zero consumers). `otr_shot_lock.py:3396` re-imports
`_stamp_durable` into a scope that already bound it at :3207. `scripts/otr_asset_index.py:264`
has an `elif engine.startswith("humo")` that cannot fire (only `eng_humo.py` exists,
so the preceding `== "humo"` always wins). `scripts/otr_machine_matrix.py:124`
`load_classes(_profiles=None)` never reads the parameter and both call sites pass
nothing. `_otr_paths.py:449` `episodes_for_obs_dir()` claims to be kept "for
back-compat with existing imports" and has zero callers.

**RIPPED 2026-09-04 (Phase 0 of the registry collapse; operator directive:
an orphan is ripped 100% or wired back in).** Every row-E symbol and every
row-F item above was re-verified on the current tree with `git grep -n -w
-F` before the cut and returns zero references after it. Two corrections
from the re-verification: `episodes_for_obs_dir()` also had an `__all__`
entry (gone with the function), and `scripts/otr_asset_index.py`'s
`elif engine.startswith("humo")` is LIVE -- the registry carries `humo_clip`
and `humo_diet` -- so it is kept; the audit's "only `eng_humo.py` exists"
was true of files and false of engine ids. `REPAIR_TEMPERATURE` was folded
into `_REPAIR_TEMPERATURE`, the spelling every call site and test used.

**PHASE 0b, SAME DAY -- the chain, and the thing that nearly went wrong.**
Operator: "even though there may be dependency B, what does dependency B lead
to, dependency C -- follow the dependency chain to find truly rippable code."
`scripts/dead_code_closure.py` now walks the reference graph to a FIXPOINT and
prints each round; Phase 0 had already proved the need by orphaning
`_derive_beats` when it removed `corpus_ledgers`. Calibration took two fixes
before the tool could be trusted: a class registered by a DECORATOR is reached
by the side effect and never named again (47 engine adapters), and with tests
as roots the first cut never opened `tests/` at all. That took 289 candidates,
five of five hand-checked LIVE, down to 55 with every known-live symbol clear.
A 73-agent fan-out then verified each one against the dynamic paths a static
sweep cannot see; 36 of 37 were confirmed dead and one was RESCUED --
`CharacterAliases` is reached by a pydantic forward-reference string.

**AND THEN THE QA PASS CAUGHT WHAT ALL 73 AGENTS MISSED.** Fifteen of those
"dead" symbols are protected by `docs/OTR_STANDING_RULINGS.md`, written the day
before, and `p0_source_char_budget` is the diagnosed-but-unwired fix for OPEN
`PBUG-20260729-03`. Every agent had asked whether code reaches the symbol; none
had asked whether the repo already ruled on it. All fifteen were restored before
the commit, and the sweep now READS those rulings and reports such candidates
under a separate heading naming the doc that speaks for them. Thirteen cleanly
cleared symbols shipped; the rest stay, each with its ruling.

* `otr_post_upscale_procgen_blend.py` keeps a `sys.path` insert of `nodes/` for
  ComfyUI's flat loading. Both tool owners it uses (`_otr_shared.ffmpeg`,
  `_otr_shared.ffprobe`) now import package-relative first with the flat
  fallback, and its unused flat `_otr_paths` import is gone -- the flat-first
  spelling had made a SECOND module instance of the owner that no fixture could
  reach (it kept its own Windows-dir tuple and answered a test the fixture
  thought it had muted). Only the insert itself remains; it is what the flat
  fallbacks need.

**FIXED IN THIS SESSION -- the audit's best catch was the driver's own slop from
the night before.** Four rows, all introduced 2026-09-03 and corrected 09-04:
`_obs_basename`'s broad except returned silently while all eight other fallbacks in
that file log -- the exact shape that had already hidden a missing `re` import for
one dev cycle; `_resolve_writer_llm`'s docstring justified `max_new_tokens` by "the
one caller whose reply length scales with a batch" after the same commit taught that
caller to bypass the wrapper; the ideogram ladder comment claimed "all four slots in
three precisions" when the real shape is 3/3/2/1, which also hides a genuine
mixed-precision requirement (no int8 text encoder exists); and four `_DEFAULT_*`
aliases were added with a comment justifying them as the error message's lead name,
which the literal refusal text never reads.


## ARCHIVED 2026-09-04 evening from ### 1.5 THE 8 GB SHIP SET -- promote what the clean room proved (queue item 4)

Measured 2026-09-02 on the physical RTX 4060 under plain stock launch flags (receipts:
`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`, PBUG-20260902-01 / -02, Bible 12.145 /
12.146): Klein 4B Q4 GGUF stills at ~21 s a still after three residency fixes (`9b90189a`,
`ad6a635f`, `da2b7a36`, each measured on the 5080 first and byte-identical there); LTX 2.5
(Q3_K_M DiT, 12B encoder pinned to CPU) at ~14 min a clip, so a ~5-hour episode. It WORKS
on 8 GB; it is not a daily driver at that pace.


## ARCHIVED 2026-09-04 evening from ### 1.8 DOCS (queue item 7) -- a DELETION pass, not a review (operator ruling 2026-09-01)

**DONE 2026-09-02 (45895801, 2bf15784, 607a5ee7; 644 -> 237 tracked docs).** How it
was run: a scripted citation scan over tests, the Bible coverage index, `nodes/`,
`scripts/`, README, CLAUDE.md, this plan, the rulings and the ship audit produced 378
uncited dated docs plus 98 uncited named ones; a 13-agent read of every dated candidate
returned 373 DELETE / 5 KEEP; the driver then grounded the list against the real tree.
Citations from the append-only history logs (`HANDOFF_LOG`, `PROD_BUG_LOG`,
`GO_FORWARD_ARCHIVE`) did not keep a doc -- git history holds all of them. Kept on
purpose: the five recipe carriers (`2026-07-31-PROBLEM-STATEMENT-under-8gb-still-to-video`
bench table, `2026-08-02-MEASUREMENT-M2-humo-vram-ladder`,
`2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames`, `2026-08-16-video-sprint-PLAN` LTX 2.3
distilled recipe, `2026-08-01-fastwan-8gb-MODEL-MANIFEST` with the LoRA sha256 and the
licence gap -- these still owe a move into the adapter comments or the matrix); the
08-26 foley-bed operator rulings; every doc a test, the code, this plan, the ship audit,
`WRITER_INPUT_MATRIX` or `LANE_BUILD_LESSONS` cites (the 09-02 kokoro-onnx and
encoder-eviction receipts among them); `COMFY_TEMPLATE_DIFF_PROTOCOL`, `WIDGET_OWNERSHIP_LEGEND`, `SPEC_haunted_image_to_video`
and its design review, `DEAD_CODE_EXECUTION_PLAN`, the licence attestations, the lane
receipts, the ship-audit folder, the schema examples and the skills backup.


## ARCHIVED 2026-09-04 evening from ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 2026-09-02; awaits his pick)

**Status 2026-09-02 (evening).** Operator: "I leave it up to you to synthesize and code maybe
1-3 of the best options." Item 0, the instrument, is CODED after a full four-round arc
(`instrument/driver_anchor.md`, sections 1-13): every rendered clip carries a versioned ACTUAL
receipt hashed into `actual_request_sha` and stamped durably as `meta.render_trace`; the writer's
trailing `replay_from` widget (and `otr_canonical_api_run.py --replay-from`) re-renders a frozen
bundle (`scripts/otr_freeze_replay_bundle.py`) through the whole canonical graph with no writer,
TTS, music or stills, node 7 byte-copying the SHA-verified master; `scripts/otr_verify_replay.py`
is the offline A/A verifier. Item 1, the adapter sweep, is RUNNING on the 5080 overnight profile
(two styles x 1.0 / 0.5 / 0.25 / 0.0, titles "Adapter <k> <style>", published to `otr/obs/` for
the operator's eye). Next: the live replay proof (render, freeze, replay twice, verify) once the
sweep releases the GPU, then item 2, the still-in lab peer -- registered only after the
`docs/VIDEO_LANE_PREFLIGHT.md` gates 1-8 and `tests/test_lane_preflight_matrix.py`, as the
operator's standing rule for any new video pack requires.

The first three sweep legs were launched with `--title "Adapter <k> anime"` so the
leg could be told apart in `otr/obs/`; that flag rides the writer's `episode_title` widget and
so became the on-screen TITLE CARD of three published episodes (the known harness-label path,
CLAUDE.md "fix the title at the source"). Legs 4-8 run without `--title`.

**First read of the sweep (operator, 2026-09-02 evening, on the three anime legs 1.0 / 0.5 /
0.25):** "the anime looks improved; which one is better, not sure at the moment." He reviews
the full candidate set when he is back; the driver's standing instruction meanwhile is "keep
coding, do your best" and to decide whether a blinded A/B is needed. The instrument's
render-trace rows on legs 4-8 (adapter_strength per shot) are what makes any later A/B
attributable; the first three legs pre-date the merge and carry no trace.

**Item 1 CLOSED by the operator's eye (2026-09-02, ~20:30), six of eight legs watched.**
Storybook engraving: 0.5 (`the_last_reading_190630`) "very storybook and did the best of
maintaining the style"; 1.0 (`the_trembling_ground_182739`) "cool rendered 3D realistic
storybook stuff but drifted into the army men in green coats". Anime: 1.0 "more anime than
0.0 but drifted into real-life stuff for a while"; 0.5 "some anime but some people in blue
coats"; 0.25 "maybe the most anime"; 0.0 "definitely animation, not strikingly anime, but
cool".

**Item 2 CODED (2026-09-02, later evening), after a full four-round arc (Fable cold +
Antigravity r1, Codex r2, Cursor r3, Sonnet r4 -- converged; Sonnet QA on the finished diff).**
`animatediff15_v3_stillin_lab_video`: the haunted lane started from an in-family 512x288
plate minted in-graph from the pack's full language and the ledger's world, repeated in Python
to the sampler batch, sampled at a strict `OTR_STILLIN_LAB_DENOISE` (default 0.65); a lab id
on `otr_stillin_lab_5080` (`draft`, never a default); the replay instrument gained a derived
bundle (`--derive-engine`) so ONE ledger renders on both Ghost siblings; the verifier gained
the plate-hash rule; `scripts/otr_stillin_probe_report.py` measures motion energy against the
lane's own A/A band. The design and receipts are `still-in-peer/driver_anchor.md` sections
1-14.


## ARCHIVED 2026-09-04 evening from ### Batch R7 -- THE 4060 CLEAN ROOM: the legs still owed

* **Leg C5** (Klein + LTX 2.5, stock flags): PUBLISHED 2026-09-02 12:10, 24 clips,
  `RESULT SUCCESS` + `obs_publish OK`


## ARCHIVED 2026-09-04 evening from ### 4.Y BRING `word_razzle` HOME -- it is the one cloud lane whose NAME hides it (operator, 2026-09-03)

**Separately noted -- FIXED 2026-09-04 (kibitz runpod-found-fixes, ruling R-B;
PBUG-20260904-03).** `word_razzle` failed the 2026-09-03 pod matrix with
`SessionIdentityUnavailable` -- it would render 2 segments from one set of
handles but declares no `session_identity()`. NOT a credentials problem and NOT
pod-specific: the same refusal fires on the 5080 for any beat long enough to
need two segments, on every one of the seven remote adapters. (This paragraph
said "already fixed" for a day while the refusal was untouched; the roster
test's tripwire was the only honest record.)

The fix is a DECLARATION, not an inference: `_CloudVideoBase` (five rows),
`google_omni_video` and `google_veo_video` set `session_residency = "remote"`,
and `beat_session.holds_local_handles()` skips the identity demand for exactly
them -- fail-closed, so an absent or misspelt attribute still reads as local and
the demand stands. No `session_identity()` was added to any cloud adapter: an
identity of handles that do not exist would be theater. The roster test is keyed
on the declaration and a control pins that no engine declares remote AND a local
isolation. Eight local engines still declare the identity (ghost_signal, humo,
ltx25, ltx_8gb, ltx_av, ltx_video, minimax_h3, wan_ti2v).

## ARCHIVED 2026-09-04 evening -- lines the forward/archive split did not carry

The forward-only rewrite split each section into work-still-to-do and finished
narrative. These ruling-bearing lines landed in neither half, so they are kept here
VERBATIM with their surrounding lines, per the 2026-08-16 rule that losing a ruling
costs more than the length does.

**From # OTR Go-Forward Plan (plan line 9):**

```
Layout: THE QUEUE (ordered; item 1 is the next thing to do) -> Section 1 the build rows
behind the queue, in queue order -> Section 2 render work batched by the leg that proves
it -> Section 3 rulings owed by the operator, each with its default -> Section 4 parked
-> Section 5 Bug Bible promotion field -> Section 6 open risks. Standing traps, recorded
limits and lifted rulings live in `docs/OTR_STANDING_RULINGS.md` ("KNOWN OPEN" and the
```

**From ## WHERE TO PICK UP (written 2026-09-04 evening; the last CODE head is `3d4a7077`) (plan line 61):**

```
is the receipt (about 9 `info` findings expected, from 158).

**DECISIONS OWED BY THE OPERATOR -- skip these, do not guess:**
* `nodes/_otr_shared/content_oracle.py` has ZERO production importers (eight test
  sites; the two "importers" were comment mentions). Rip it, wire its luma-floor and
```

**From ### 1.8 DOCS (queue item 7) -- a DELETION pass, not a review (operator ruling 2026-09-01) (plan line 1095):**

```
`docs/RUNPOD_INSTALL.md` is the sole RunPod manual and recovery guide (Codex owns it);
bugs and history live only in `docs/PROD_BUG_LOG.md`, the archive and
`docs/OTR_STANDING_RULINGS.md` (never in this plan). Working rule for the pass: a dated spec, plan, handoff,
kickoff, brief or log under `docs/` goes unless it is cited by a test, by the Bible
coverage index, or by a shipping doc, or it records a video-model recipe (measured
```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1261):**

```
mints a new ledger, so a same-ledger A/A cannot run), a blinded two-null scorecard; (1) the
adapter strength swept PER STYLE including 0.0 (zero code; env override exists); (2) a still-in
LAB PEER engine with the lane's own in-family 512x288 plate (never a flip of the shipping
engine's contract, never a Klein gate on the 4060), one subject-free scene plate per beat with the
plan untouched (the earlier "cycle frozen to figure" was superseded in the item's own arc,
```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1262):**

```
adapter strength swept PER STYLE including 0.0 (zero code; env override exists); (2) a still-in
LAB PEER engine with the lane's own in-family 512x288 plate (never a flip of the shipping
engine's contract, never a Klein gate on the 4060), one subject-free scene plate per beat with the
plan untouched (the earlier "cycle frozen to figure" was superseded in the item's own arc,
`still-in-peer/driver_anchor.md` section 7 D4); (3) the pack's
```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1268):**

```
FrameContract design arc). Deferred: speech-energy scaling, SparseCtrl, per-style checkpoints,
FreeInit. Cut: the style-aware roll, CameraCtrl, IP-Adapter / PIA / Lightning. Nothing in the
shipping recipe moves until the operator picks the first arm; the still-in idea stays parked
(Section 4) until then.

```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1295):**

```
`meta.render_trace[*].sampler_inputs.adapter_strength`. The three already-published episodes
stay in `otr/obs/` (nothing is ever tidied out of it). SOURCE FIX, still open: give the harness a
`run_label` of its own that reaches the log and the ledger meta but never `episode_title` -- a
small design item (touches the API runner, the whitelist and the mux's filename), arc before code.

```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1327):**

```
default moves to 0.25 -- one constant. **The coats finding:** across the traced episodes the
only prompts that say "coat" say "cream coat" and "charcoal coat" (the cast's own looks, both
characters), never green or blue; the green army coats and blue uniforms are the base model's
prior for "coat" under an engraving or anime medium, amplified by the adapter. The GARMENT
WORD is a style lever -- an input to item 3 (pack language): the motif composer should map
```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1345):**

```
one style per invocation, anime and storybook_engraving first, every leg to `otr/obs/` with
the story's own title -- and the operator's eye on the triptychs. Stop rules stand (section 7
D9): plates that do not read as the style end E1 (next arm E11); no denoise that both moves
and keeps the plate ends E1 (next arm the E2 probe).

```

**From ### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 202 (plan line 1352):**

```
and ruled: draw what the line is ABOUT in the story's one setting, mix the visual style with
the CRUX of the story on every beat, the world's own motion, no costume or prop the dialogue
never names, fewer variables ("too many variables -> humans and bags"), get creative, and
"yes, I want to see more STORY, not just the characters." Mechanically confirmed: the lane's
character motif is a costume template with fallback pools (coat; lantern / key / ledger /
```

**From ## SECTION 2 -- RENDER WORK, BATCHED BY THE LEG THAT PROVES IT (test the least) (plan line 1386):**

```
scheduling note; still true). Reset per `CLAUDE.md` section 4 before every
leg. A leg that does not reach `otr/obs/` did not pass. **Every canonical leg
below is also a free chance to** (a) re-observe the two parked eyeball items
(Batch R5) and (b) watch the two ledger-cleanup behaviour changes named in
Open risks that have no live receipt yet.
```

**From ## SECTION 4 -- PARKED / DEFERRED (out of the working queue) (plan line 1701):**

```
---

## SECTION 4 -- PARKED / DEFERRED (out of the working queue)

### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK
```


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.2 CHARACTER GENDER LADDER (queue item 3a) -- the SPEC REWRITE is written; next is ONE review r

### 1.2 CHARACTER GENDER LADDER (queue item 3a) -- the SPEC REWRITE is written; next is ONE review round, then code

Left open, recorded in PBUG-20260815-04's follow-up: a given-name alias can
match a different character with that surname (COLONEL FITZWILLIAM via "fitzwilliam"; right
here by coincidence) -- a surname-only alias for the short_form tier is the next fork.

Operator rulings folded: ARIEL / PUCK / ROBIN stay on the roll (locked index entries); Dr. Lira Kell is
female (locked).

**TWO OPERATOR RULINGS, 2026-08-28, and they reshape the spec:**

* **Shakespeare: fill ONLY the 32 `unknown` roster rows.** KNOWN rows from the
  parsed dramatis personae stay untouchable; the ladder's lower tiers may fill
  the blanks.
* **THE WEB-SEARCH TIER IS REPLACED, not plumbed.** Operator's design, his
  words: *"just have the LLM decide -- ask what the likely gender of this
  person name is, have the LLM decide, and keep that in an index of names."*
  So tier 3 becomes an LLM VERDICT ON THE NAME (the model already knows
  Scrooge and Marley from training -- no live search needed), cached in a
  PERSISTENT name index so each name is asked once, ever. Tier 4
  name-frequency stays as the deterministic floor beneath it, keeping the
  ladder TOTAL when the LLM call fails. This dissolves both review rounds'
  biggest must-fix (the silent no-op web call): there is no web call. On his
  "is it not easy to query a search engine?" -- keyless search-engine querying
  is the fragile part (scraping is blocked/ToS; keyless APIs are thin); the
  RSS precedent covers feeds because feeds are MEANT to be fetched. The
  LLM-ask design avoids the whole problem, offline-first.
  The invented lanes (original, scifi_news_pro, media_archive) KEEP ROLLING
  by the standing ruling -- their characters do not exist, so no lookup of
  any kind applies.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.3 GHOST POOL -- uniqueness on the finalized prompt (queue item 3b; r1 is in, build)

r1 verdict (2026-08-31, Codex `no` / Antigravity `yes-with-fixes` / Cursor `no`; artifacts
`kibitz-runs/2026-08-31-ghost-clause-pool/r1/`, 5080-local; narrative in
`docs/GO_FORWARD_ARCHIVE.md`): **the pool is not too small, the duplicate check is.** The
picture is four slots (`GHOST_V2_SLOTS`) and the check reads one (`key = leaf.casefold()`
in `otr_shot_lock.py`), so two beats with the same leaf and different characters are
rejected although they render different pictures. That is why 6 -> 18 failed and why 50
would fail too.

(The r1 anchor's "roughly 70 minutes" failure was the zero-frame beat, PBUG-20260831-01; not this row.)


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.4 OPEN DEFECTS THAT ARE CODING WORK (queue item 3c; a leg may prove some later, none needs a l

#### The P0 / source-span cluster (2026-07-30)

- **`full_text` HTML block joins fuse tokens** (`...PolygonsNASA/JPL-`, `...School
  ofEngine`, `...doing.Let's s`, `...(AMR).The resea`): the dominant P0 span-rejection
  cause on live evidence. The RSS adapter strips tags without inserting whitespace, and
  `_normalize_span_source_text` can collapse runs but cannot insert a space that was never
  there. Owed, after Section 3 question B is ruled: name the adapter that builds
  `full_text`, insert the separator at admission without breaking any accepted ledger's
  `source_digest`, and pin a fixture from those four strings. Belongs in the source
  adapter, not the codex normalizer. History: `docs/GO_FORWARD_ARCHIVE.md`.

- **The deterministic P0 rung PRUNES SILENTLY, which violates the plan's own
  Invariant 3.** `repair_literal_source_metadata` drops an unsupported span, then its
  evidence row, then the fact -- and emits no receipt. An accepted P0 index simply
  has fewer facts than the model wrote, and nothing says which were dropped or why.

- **The deterministic P0 rung is ALL-OR-NOTHING across an artifact, and can poison
  its own good work.** It is handed `a0_payload` (all seven keys) while
  `_validate_fact_index` restricts spans to `allowed_source_fields` (the projection).
  A quote rehomed into a field the projection omitted makes `post_validator` reject
  the WHOLE repaired artifact -- "cites source field ... outside the supplied P0
  evidence" -- so one unlucky rehome discards every correct prune in the same pass.
  Either give the repairer the allowlist or prune per row.

- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked.

#### The orphan-occupancy registry (design item -- full arc BEFORE code)

Deferred three times as correctly out of scope for the
cache-bookkeeping fixes (PBUG-20260825-04, `fb67d059`; arc history in the archive), and
each cut of that fix found a new race, so this is a genuine design choice: full arc first.
`da2b7a36` (the dispatcher frees the writer before the first local still) narrows the
image-stage exposure but does not build the registry.

#### Coverage, canvas and clip-contract

- **The route lock is ONE NODE TOO LATE for the image phase** (found 2026-07-25, node
  order confirmed against the canonical JSON: ...). The landed fix closed the spine-validation gap; the image phase still
  relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine seam").

- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment item 1).
  B5 made this non-load-bearing for the render (the engine declares its own
  canvas now), so it is no longer urgent -- the drift guard in
  `tests/test_ltx_8gb_canonical_canvas.py` covers the disagreement that matters.

- **`ltx_av` underruns long beats** (found 2026-07-25, codex; confirmed).

#### Voice engines

- (PBUG-20260902-03; ROOT-PINNED 2026-09-02 14:20). The 4060 Leg C5 episode
  lost both announcer lines to a 1.4 kHz tone; a direct engine probe on the same box
  then rendered the exact announcer line as speech and a 9-word character line as
  seven seconds of noise plus a 2.5 kHz tone. So it is the engine's roll, not the
  announcer graph, not the knobs, not line length (the 180-character chunker already
  exists). ... test = a stub engine that returns a tone first
  and speech second yields speech in one retry.

#### Routing, env-capture and the credits card

- **Env-read sites missing from the S0b inventory** (was four; the `OTR_ENABLE_LTX_I2V`
  site was DELETED by the 2026-08-28 retirement, so two remain)

- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At
  512x288 (the ltx_8gb tier) col1 is 65px past its footer even with every ledger row
  this policy may drop already dropped; at 640x360 it is 12px over. Both are drawn
  anyway (a terminal node never destroys a finished episode) and LOGGED at ERROR
  naming the canvas -- the old behaviour was drawn, clipped by PIL, silent. At 288
  lines the three-column console is already a polite fiction: col3's scrolling
  transcript is as unreadable as anything col1 clips.

#### 1.4a DEAD CODE / DUPLICATED-DECISION AUDIT (2026-09-04) -- three Sonnet lanes, driver-verified

Every row below was re-checked against the real files by the driver before it was
written down; the REJECTED list is part of the record precisely so a discarded
claim is not re-raised as a fresh finding.

**STANDING RULINGS FROM THE AUDIT (rows A-F are built; the receipts are in the archive,
these sentences are not).**

**G. FOLLOW-UPS THE 2026-09-04 ARC SURFACED AND DID NOT BUILD -- each with the
reason it waits.**
* `tests/test_openrouter_slug_curation.py::test_routers_appear_in_both_slot_dropdowns_and_auto_leads`
  passes on this box only because of an UNTRACKED catalog cache ("no OpenRouter
  models cached -- run refresh_catalog_cache" in a clean worktree).
* The google omni / veo / image adapter tests resolve `comfy_output_dir()` without
  pinning `OTR_OUTPUT_DIR`, so on the real tree they write provider bytes under the
  live `output/otr/episodes/_shared/tmp/`; in a worktree the Tier-3 walk-up lands
  on `C:\Users\output` and they fail.
* **UNFOUND EXPORTER (2026-09-04):** ... (import, `pytest_configure`, `pytest_sessionstart`), `tests/__init__.py`,
  `pyproject.toml`, any registered plugin (anyio, hydra, langsmith, pytest_asyncio)
  or by importing the mux in plain Python. Consequence: a test that injects an
  output root ONLY through a `folder_paths` stub is overridden once the reader it
  drives delegates to the owner (three `test_video_render_path_cw4` tests broke this
  way and now pin the env too).
* **NEW BUILD ITEM, arc opened 2026-09-04 -- collapse the registry scan from
  158 findings to about five by the one-owner rule** ... Measured from
  alpha.17's real payload: the env rule fires ONCE PER FILE (103 findings, 103
  files), so one `os.environ` owner takes it to 1; the subprocess rule fires per
  site (35 in 20 files), so one process runner takes it to ~2.
* (d) From the `-02` window:
* **The draft 8 GB profiles cannot write on two of the three banks
  (PBUG-20260904-05):** ... `media_archive` FITS: attempt 5 of the publish matrix
  above rendered and published on `8gb_lite` with that bank, so the fault is
  the bank-plus-context pairing, not the profile alone. The row that SHIPS,
  `otr_4060_12b_gguf_offload`,
  runs the same 12B at `gguf_n_ctx: 4096` (measured 7.8 of 8.2 GB) and fits.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.1 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (design ite

Shape (settle the details in the arc, do not re-derive these):
* A second backend inside `nodes/_otr_audio_engines/eng_kokoro.py` (247 lines; today it loads
  `KPipeline`). ONNX path: load model + voices, phonemize with espeak-ng, run onnxruntime,
  return 24 kHz audio.
* Weights: kokoro-onnx expects `kokoro-v1.0.onnx` + one `voices-v1.0.bin` (an npz keyed by
  voice, GitHub release `model-files-v1.1`); the HF mirror `onnx-community/Kokoro-82M-v1.0-ONNX`
  (ungated; model 86 MB q8f16 to 326 MB full, 55 per-voice `.bin` files at 0.5 MB) uses a
  different per-voice format.
* it is an 82M model and faster than realtime on CPU.
* That dissolves the indextts2-default half of the ship-audit finding (the
  reference-WAV preflight is the bullet below)
* (the surviving half of the ship-audit indextts2 finding)
* so README can say "want a better TTS, install it yourself, here is what runs where" and point at one table.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.9 ONE MANIFEST, PREFLIGHT AUTO-DOWNLOAD (queue item 8; design row; operator asked "and auto do

Today only the writer LLMs, bark, musicgen and the kokoro voices fetch themselves; every
image engine, every video engine, Stable Audio 3, the cloning TTS engines and the reference
WAVs are manual placement behind two fetchers under `scripts/` that the registry bundle does
not ship, and README's "auto-fetch" wording for the 8 GB lane is what the drill measured as
manual.

(The first-run download bill is README's number, not this file's.)


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.10 SHIP-AUDIT SURVIVORS (2026-09-01) -- each needs a design decision, not a grep (queue item 8

### 1.10 SHIP-AUDIT SURVIVORS (2026-09-01) -- each needs a design decision, not a grep (queue item 8)

Receipts and every file:line: `docs/ship-audit-2026-09-01/SHIP_LIST.md` (71 confirmed,
51 disputed for the operator to rule, section 8). The mechanical items are landed; the voice item rides Section 1.1, the 8 GB writer item
Section 1.5, Mac / AMD Section 1.13. These are not mechanical and each wants a kibitz arc
before code:
1. **Runtime writes inside the pack directory** (cloud-media billing ledger,
   OpenRouter catalog cache): a registry update wipes them. Route through
   `nodes/_otr_paths.otr_shared_cache_dir()`; needs a migration note for existing
   ledgers.
   Bites exactly the registry-install users the queue's first items create.
2. **`_fit_reason` never consults `needs_fp8_te` / `needs_fp4_te`**
   (`nodes/_otr_shared/capability_profiles.py`), so fp8 and NVFP4 engines qualify on the
   ROCm tiers whose `dtype_policy` forbids them. Two clauses keyed on `dtype_policy`.
3. **The janitor cannot sweep `tmp/audio_slices`** (`nodes/_otr_janitor.py`: directory
   granularity, newest-child mtime): 9.3 GB measured, and the boot sweep stats 21,440
   files (6.7 s) every ComfyUI start. Three lines, but it widens what gets auto-deleted,
   so it lands with a test.
4. **Cloud spend with no ceiling**: `cpu_floor` and `otr_mac_mps` route every image role
   to the paid Google API on the mere presence of a key, and the BYO-key lane has no
   reserve/bill/ledger path (`eng_google_image.py`). Ledger it or make it an explicit
   opt-in.
5. **`eng_ltx_video` / `eng_ltx_av` reload ~14 GiB of weights per beat**; the
   `prepare()` + `external_results` pattern the sibling lanes use is the fix.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.11 THE ADAPTATION DESIGN (queue item 8; hardened, NOT yet built; multi-session -- start only w

Keystone rationale (removed as explanation, not work): "(which also removes the VRAM hazard: no model sits in the source-speech path)."

From step 2: "-- the failure that killed `scifi_news` in the six-bank run."

From step 5: "(`visual_style_policy`, the other half of this item, was ripped 2026-08-04.)"


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.12 STYLE / IDENTITY DECISION WORK (queue item 8; backlog; not the next coder window)

Grounded by the 2026-08-03 four-agent forensics; every line has a file:line in the
session traces.

1. **"Invent one and tag it"**: ... (`_otr_story_brief.py:513` -- proven content-loyal on
   both specimens) ... Highest-leverage item here: it fixes the credits line for all six
   banks uniformly.
2. **Rename `meta.style` -> `meta.story_scaffold`**: "Consumers move in ONE atomic change" narrative prose form; ledger validator note "(r3)".
3. **Ghost-name reconciliation fork**: original prose form.
4. **Dead fields found**: original prose form.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for. Scope
   as its own rip with the ledger law (every field one owner).


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.5 THE 8 GB SHIP SET -- promote what the clean room proved (queue item 4)

Klein is already the image default in all
19 low-VRAM profiles and the 8gb / 12gb / amd classes (ruling 2026-09-01, `c0ebe31f`;
`docs/OTR_STANDING_RULINGS.md` "IMAGE ENGINE DEFAULTS BY MACHINE CLASS").

1. **The PROVEN flip.** The Leg C5 episode is in the clean room's obs
   (`signal_lost_rationed_breath_20260902_060027`, published 2026-09-02 12:10, 24 LTX 2.5
   clips, 12 Klein stills, 6 h 35 min). When the operator has watched it: add the receipt to `config/machine_classes.json` (`proven[]`
   on the 8gb class, image column included; `known_limits` keeps the pace), regenerate
   `docs/MACHINE_MATRIX.md` and the README block (`scripts/otr_machine_matrix.py`), and
   the 8gb `proof_summary` stops saying "image lane unexercised". Record only what
   published.

(from item 3) (14.5 GB ceiling next to the only genuinely 8 GB video engine). ... (agy proposed it and was wrong). "Finish retiring the profiles" is no longer an option: `95feac86` keeps `config/profiles` as lab presets and makes the matrix the record.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 1.6 THE 4060 TEMPLATE TEST SET -- what is still open before the capstone (queue item 5)

* **One JSON for now (operator 2026-09-02).** `workflows/otr_4060_floor.json` is
  removed from the gallery (it shipped 2026-08-29 to 2026-09-02 as a bark-voiced
  zero-download floor, before kokoro ran on 3.13); `config/profiles/otr_4060_floor.json`
  and its generated variant stay as lab presets.
* (from the README bullet) -- or the operator rules that README's injected class table and
  per-lane facts (66da15da) already cover it and the bullet goes.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Batch R1 -- THE H3 PROMPT-POLICY VERDICT: read the receipts first, render only if they do not an

### Batch R1 -- THE H3 PROMPT-POLICY VERDICT: read the receipts first, render only if they do not answer

The fix (`e923a9f3`, 2026-08-27) already has two post-fix `minimax_h3_video` episodes in
the matrix's PROVEN row: `signal_lost_the_poise_of_stone_20260827_143538` and
`signal_lost_reel_of_resistance_20260828_121427` (the operator watched the latter). Owed
is the VERDICT, still marked "STILL OWED" under PBUG-20260827-01 in `docs/PROD_BUG_LOG.md`:
the positive video prompt shows nonverbal action and camera PRESENT, and the beat's exact
dialogue and any speaking / lip-sync / mouth anchor ABSENT. The ledger on disk stores only
`prompt_sha8` for the positives, so read the render receipt or the server log. Clean
receipts close the row. Render a fresh leg ONLY if neither episode's receipt answers it:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_w45_minimax_h3_video -Acts 1
```

Engine routing is a PROFILE, never `-Set` (`patch_creative` whitelists creative widgets
only; writers may ride as `-Set`); the profile carries the h3 boot contract
(`--reserve-vram 12`, `--disable-pinned-memory`); a FRESH episode id is mandatory because
`request_hash` excludes prompt bytes, so a cached SPEAKING clip would be a false pass. The
BEFORE sample `signal_lost_the_caretakers_clause_20260826_155835` (every beat on H3,
pre-fix, under `output/otr/episodes/`) stays untouched: it is half of the A/B.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Batch R2 -- ONE image-mode sweep on `otr_soak_llmsweep_02` settles SCENE + PORTRAIT `elements: [

Re-run the image sweep's profile `otr_soak_llmsweep_02` against post-`ae7e7b6a`
HEAD

Full argument:
`docs/2026-08-26-ideogram-music-card-PROBLEM-STATEMENT.md` (5080-local) and
PBUG-20260826-01.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Batch R3 -- FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator directive 2

After Leg 0 (Section 1.7), four canonical legs prove all 7 surviving local LLM rows in
BOTH slots plus the gemma Q8_0 negative probe -- and each leg doubles as a ledger-cleanup
live-watch and an eyeball re-observation chance.

**THE SWEEP ITSELF is what is open. Design is done (11-agent fan-out, 2026-08-25), build
is not.** The honest shape, which is NOT one render:

* **The operator's creative/technical parity rule is the sweep's acceptance
  criterion.** Structurally both slots already build from the IDENTICAL
  `dropdown_choices()` list, so no row is slot-restricted; what is unproven is
  whether each row can actually do the TECHNICAL job (constrained JSON / GBNF).

**Full design (coverage matrix, per-row assertions, skip-reporting rules, risks)
is in the 2026-08-25 workflow result; re-derive from this row if it is lost.**


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Batch R5 -- OPPORTUNISTIC: the cfg 1.0 promotion cells, D2 fail-hunting still legs, and the eyeb

* **The cfg 1.0 promotion A/B** (Section 1.5 item 4): four cells x three real episode
  prompts to a NEW dir under `docs/ship-audit-2026-09-01/image-jury/`, then the operator's
  eyeball. Confirms cfg 1.0 on real content or sends it back.
* **D2:** reset per CLAUDE.md section 4, boot headless, run 320-word `public_domain` or
  `shakespeare` still legs until one fails (~1 in 6). A publish is a clean leg; a
  fail-closed with the compact JSON `MISSING_TARGET` record in the SERVER log (arm, token,
  index, canonical `prompt_hash`, repr-escaped excerpt; the canonical runner truncates the
  exception at 500 chars) is the PROOF D1 WORKS -- D3 then fixes THAT branch at its root and
  `docs/PROD_BUG_LOG.md` gets a mechanism, not a guess. Do NOT weaken the completion gate,
  revive the portrait-init fallback, or rebuild the withdrawn "give the collapse guard a
  still owner" fix (the 08-04 postmortem disproved that chain). Receipts:
  `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`,
  `docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.
* **Two eyeball items ride ANY real render leg, at zero extra legs:** the announcer
  framing defect (`docs/2026-07-11-announcer-framing-defect.md`: episodes START a story
  instead of admitting you into one) and name-splice defect #2. Both predate THE LAW and
  have no reproduction at HEAD, so no coder time: still there -> re-admit as a FRESH dated
  row with that leg as evidence (the framing fix stays seam + score contract + fail-closed
  validator, never Python authorship); gone -> tombstone.

Removed detail: "the canonical runner truncates the exception at 500 chars"; "is the PROOF D1 WORKS" (D1 is shipped).


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Batch R7 -- THE 4060 CLEAN ROOM: the legs still owed

### Batch R7 -- THE 4060 CLEAN ROOM: the legs still owed

The clean room stays on disk: `C:\OTR-CleanRoom` on the 4060 (portable v0.34.0, Python
3.13, OTR clone at `da2b7a36`, ComfyUI-GGUF pinned + patched, pinned weights, bark
voices). Server and legs start through Task Scheduler, never from an SSH session. The
clean-room profiles are untracked stand-ins on the 4060; the shipped equivalent is
Section 1.5 item 2. Friction log and every leg receipt (R1, R2, C through C6):
`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`.

* **Leg C5** (Klein + LTX 2.5, stock flags): the operator eyeball is Section 1.5 item 1. Leg A's question is
  answered by this leg (LTX 2.5 works on 8 GB, not a daily driver); Leg C6 (fp8 encoder)
  is not needed.
* **Leg B -- OPEN:** `otr_cleanroom_8gb_humo17` (HuMo 1.7B, 13.6 GB of Comfy-Org weights,
  no extra node pack) -- the faster 8 GB video candidate if LTX 2.5's ~14 min a clip is
  too slow to ship as the 8 GB video default. Pull the clone to HEAD first (later commits
  are docs and the provisioner).
* **Z-Image from the dropdown on 8 GB:** one stock-flag `z_image_turbo` still
  post-`da2b7a36`. The R1 abort (`Fatal Python error: Aborted` at sampler step 5/8 under
  DynamicVRAM) was never re-tested after the residency fixes, which the clean-room doc
  calls only its "likely root". If it still aborts: document the
  `--disable-dynamic-vram --lowvram` pair in README's 8 GB row for that dropdown choice
  (README carries no such text today) and file the faulthandler report with ComfyUI. The
  shipped 8 GB set runs Klein by ruling and is unaffected either way.
* **The registry-install template test** (Section 1.6) runs here once a version is
  Active.
* Record ONLY what publishes (`RESULT SUCCESS` + `obs_publish OK` + the file) into
  `config/machine_classes.json`, regenerate `docs/MACHINE_MATRIX.md`. Do not advertise a
  lane the clean room did not finish.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Batch R8 -- THE NEXT POD SESSION: the acid test and a looped lane sweep on one rental (queue ite

### Batch R8 -- THE NEXT POD SESSION: the acid test and a looped lane sweep on one rental (queue item 9)

The volume stays (it holds the warm cache, the expensive thing to recreate); the pod stays
STOPPED until this batch runs. Codex owns it and `docs/RUNPOD_INSTALL.md`. Sequence on the
one rental: saved template -> `scripts/otr_provision.py` -> one published episode with
ZERO hand steps (the acid test the DOER was built for: Stable Audio 3, index-tts in its own
`uv` venv at `INDEXTTS2_PYTHON = "3.10"`, the reference WAVs -- all installed by the
provisioner, none by hand), THEN the looped lane sweep (`scripts/otr_pod_lane_soak.sh`,
`scripts/otr_pod_overnight_sweep.sh`) for the HuMo 14B / LTX 2.5 / indextts2
second-machine receipts. Receipts file under the 16gb class in `config/machine_classes.json`
`engine_evidence`: there is no 24 GB class or profile by ruling (`95feac86`; Section 3 L
asks whether a row is wanted).


## ARCHIVED 2026-09-04 evening (compaction pass) from ### The registry rows, carried forward (the 2026-09-02 queue's item 1)

**1. THE REGISTRY -- alpha.17 IS PUBLISHED AND VERIFIED (2026-09-03). Two SEPARATE goals
live here and conflating them is what made this feel stuck: (a) an INSTALLABLE listing,
which needs an admin approval we cannot self-serve, and (b) the "N Nodes" count on the
card, which is a different pipeline entirely -- and which, MEASURED 2026-09-03, is
STALLED ON COMFY-ORG'S SIDE and is not ours to fix at all (see the ANSWERED block
below; the newest successful extract in a 360-pack sample is 2026-04-28).** The old
control experiment (republish alpha.8 byte-identical) is CANCELLED -- the cause was found
directly instead.

* So the credential fix held across the bump, Flagged-on-info-only is the expected outcome,
  and **every precondition on the manual review request is now satisfied** -- it is ready
  to file on the operator's go and nothing else gates it.
* **THE SCAN COLLAPSE IS THE CODER'S NEXT CHUNK** -- see WHERE TO PICK UP at the
  top: plan closed at r4, orphan rips shipped, owners on disk; the ratchet commit is
  next. Nothing in it needs the operator except the review filing above.
* **ANSWERED 2026-09-03 18:30Z, AND THE ANSWER CLOSES (b) AS NOT-OURS.**
  It simply was not the node-count blocker.
* **STILL FLAGGED IS THE EXPECTED OUTCOME for (a), not a failure.** Active needs ZERO
  findings of any severity or an admin batch approval; 157 info-level YARA hits (env
  reads, ffmpeg subprocess, opt-in cloud lanes) keep it Flagged forever on the automated
  path. The gate is a literal `if issues == ""` on the scanner body -- one info finding
  and 157 are identical. There is no publisher self-service route to Active, confirmed by
  reading `Comfy-Org/registry-backend`. Corroborated independently: rgthree's two newest
  publishes are Flagged too, so the ruleset tightened in late August and comparing against
  older Active versions of popular packs proves nothing.
* **THE NODE-COUNT HALF IS ALREADY FILED, AND HAS BEEN SINCE 2026-08-24: the operator
  opened `Comfy-Org/registry-backend#203`.** It is still OPEN with no comments, labels or
  assignee. It names a cause this window's 480-pack survey could not have found on its own
  -- **a backfill cron scheduled for February 29th**, a date that does not occur in 2026 and
  next occurs in 2028. The operator added the survey as a comment on 2026-09-03 and
  corrected the issue's date window: extraction did not stop in February, it stopped
  **2026-04-28**, and 19 packs' newest success falls in April with nothing after.
* Verified on `comfyui-video-xy-plot`: all five of its versions carry their own
  `{"message": "subprocess: ffprobe", "by": "dr.lt.data@gmail.com"}` note, four of
  them published on the same day. So approving alpha.17 buys nothing for
  alpha.18, and this branch shipped five versions in five days. Asking now would
  spend a reviewer's time on a version superseded within the week -- which is the
  behaviour that actually annoys a maintainer, as distinct from asking at all.
  `docs/2026-09-03-registry-review-request-READY.md` carries the title, the body,
  alpha.17's version id, counts read from its own scan, the zero-is-unreachable
  measurement, and the video-xy-plot precedent.
* **The old NEXT line, kept so the parking is legible:** file the manual review
  request. Draft ready and retargeted at alpha.17 in
  `docs/2026-09-02-registry-manual-review-request.md`. **Precondition 2 (grep the
  published zip for the route gate) IS NOW SATISFIED** -- see the verification above, so
  strike that step. **BOTH REMAINING PRECONDITIONS ARE NOW SATISFIED
  (2026-09-03):** alpha.17's scan landed at 0 critical, and every finding count in the
  draft has been re-read from alpha.17's own scan (158, not alpha.16's 157). The draft is
  ready to post as-is. Filing is a public post on Comfy-Org's tracker and needs the
  operator's explicit go -- it is the only thing still holding it.
* Publishing a small pack under his own publisher
  to confirm a dependency set actually installs in their Linux container is ordinary use
  and would have caught the pycairo bug for free, before it cost four version strings.

**3b. PROMPT v3 "DRAW THE CRUX" -- HALF A SHIPPED AND PROVEN 2026-09-03.**
  **HALF B -- THE OPERATOR RULED ITS SHAPE 2026-09-03 AND r1 INVERTED ITS
  MOTIVATING CASE.** the r1 panel caught that the item's own worked example had been read backwards.
  r1 also corrected three inherited claims. r2 is running.
  **The old framing, kept only so a reader of the history understands what
  changed:** subject-appropriate motion needs the beat's own dialogue in front
  of the writer -- Half A buys the story's object, place and light on every beat and cannot
  buy its motion. He named the gap himself on the ledger episode: "I don't see any trucks
  though, it does mention a truck once" -- `truck` is in one spoken line and not in
  `key_objects`, so Half A had no path to it.

**3d. Item 3b-of-the-old-numbering** -- (83 words on one real episode) measurement detail.

**4.** what the clean room proved becomes a saved dropdown set.

**9.** there is no 24 GB class or profile by ruling (95feac86).

**WATCH:**
* **`obs_publish OK` is not proof of an episode (2026-09-03, cost two GPU legs).** Three
  replay bugs shipped that morning; TWO of them published green with a broken file --
  one second of picture in an 85-second episode, because a placeholder wire set the
  procgen overlay's length and `PostUpscaleProcgenBlend` takes the shorter input. What
  caught it was measuring the published file's DURATION at every pipeline stage.
  PBUG-20260903-01/02/03.
* `video_rotation.sh` has run since 2026-08-31 out of session `8a385813` (that runs in GitHub
  Actions against the repo, not this box).
* Zero-frame beat: the root fix landed (`415b1ba0`, PBUG-20260831-01: an untimed
  `music_open`/`music_close` with no timed mirror got no duration; it was also the ghost
  pool r1's "roughly 70 minutes" death).
* Whether a PRUNED P0 index is ever accepted has not been measured live (the deterministic
  rung became reachable at `47c554fa` after the campaign stopped).
* Token rotation: (Codex's security note, relayed in-session 2026-09-02 and recorded nowhere else).


## ARCHIVED 2026-09-04 evening (compaction pass) from ### J. THE REGISTRY -- the control experiment, only if alpha.15 flags

Two versions exist, both Flagged (a private secret scanner, not an exec linter; CLAUDE.md
7A), no Active, no rollback target (the node hard-delete freed alpha.8's string). The
alpha.15 push is queue item 1; the README's token-shaped literal, the only shipped string
matching a published secret rule, is already gone (`64d81ca7`).

* **If alpha.15 flags, run the control:** republish the alpha.8 tree (`e44235f5`)
  byte-identical as alpha.16. Active means the trigger is in the alpha.9+ delta and can be
  bisected; Flagged means the ruleset moved and that result is the evidence to hand
  Comfy-Org. Never version-delete (a soft delete burns the string).
* **Version sequencing:** alpha.15 = the marker patch; alpha.16 = the control, only if
  needed; the kokoro-onnx dependencies (Section 1.1) ride the NEXT bump after both, never
  while a version is Pending.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### The question list

Removed from "### The question list" (finished-work reports, commit shas, past evidence, and superseded prose), verbatim:

(A) "The G15 vacuity fix (`e2807dcc`) is live code with zero callers."

(B) "Inserting separators is a WIDER change to the coordinate system `source_digest` pins -- it belongs in the source adapter, and it is the DOMINANT P0 failure cause on live evidence."

(D) "After profile retirement, who owns a tier's native render ceiling? Since `95feac86` the machine matrix, not a profile, is the stranger-facing channel; profiles are lab presets."

(H) "Since 08-15 a research_only source WITHHOLDS the OBS copy instead of killing the finished render."

(I) "Found by the five-bank beat test: a `pirate_radio_resistance_drama` premise was drawn over a film-reel standoff seeded by a real Library of Congress item on 'Midnight' (1939) -- the operator caught it on screen. Second specimen of the content-blind-draw class; the scaffold-off rule so far was stated only for `original`."

(J) "skipping `finish_visual_prompt` and the `image_grade_tail` append by the 2026-07-02 look direction"

(K) "codex and Fable both said RIP, Claude grounded both." "Argument in the archive."

(L) "The matrix says 16 GB+ is the top tier by design (`95feac86`) and LTX 2.5 at 1664x960 OOMed on a rented 24 GB 4090 (`docs/RUNPOD_INSTALL.md`)."


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Detail for question (D): the 8 GB / profile cluster -- one decision underneath

- **WAN 8 GB ceiling: CODE-COMPLETE, PROOF-INCOMPLETE, and ONE decision blocks it.** The
  17-frame ceiling reaches a leg only through a variant workflow or a hand-set widget:
  `otr_canonical.json` node 87 ships `max_render_frames=0`, so a plain canonical WAN run
  is UNPINNED and inherits `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure
  shape. Pinning 17 in the canonical is WRONG (it would cap the 16 GB LTX / HuMo legs).
  Proposed shape: `eng_wan_ti2v` DECLARES its own tier ceiling as a capability-row field,
  the widget becomes an operator OVERRIDE with 0 meaning "use the adapter's contract", and
  the profile channel stops mattering -- a real design change with a live blast radius on
  any card with headroom, so ratify before code. The wired chain is pinned by
  `tests/test_remaining_video_contracts.py` (nine hop-by-hop tests) and
  `tests/test_multiclip_effective_contract.py`; commit history in the archive. Also owed
  after the ruling: a render on a PHYSICAL 8 GB card (the four-arm bench prequalified on a
  16 GB card told to reserve 8 GiB, which is not the same claim; the 18-engine campaign is
  coverage, not an 8 GB qualification). One untested edge, cheap to close when this
  reopens: WAN is out of `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan
  CAN contradict by design and `_planned_length` hard-refuses mid-episode -- no test asserts
  a 17-frame tier survives a multi-segment beat.
- **A2 -- the applied-overrides echo hides the profile's `llm.*` override.**
  `nodes/_otr_workflow_apply.py` already flattens `llm`; `scripts/otr_api.py` echoes only
  role / slot / features plus two seed keys, so a run reports "16 overrides" while also
  having replaced the entire LLM configuration. Fix: generate the echo FROM the applier's
  flattened map (never add keys by hand). HELD on question (D), because its whole subject
  is the profile channel.
- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by
  coincidence** (found 2026-07-27, B6 panel, two lenses independently). The coverage
  PLANNER reads `config/profiles/otr_8gb_ltx.json` `video.max_render_frames`, and
  `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The ADAPTER's own
  pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Today both land on 161 (profile
  unpinned, env unset), so nothing breaks. But `workflows/variants/otr_8gb_ltx.env.json`
  ships `OTR_LTX_8GB_MAX_FRAMES=97` and NOTHING currently reads that file. The day a
  launcher honours it without also pinning the profile, the planner emits a 98-161
  frame segment and the adapter refuses it MID-EPISODE -- after the stills are minted
  and, on a multi-segment beat, after the 6.34 GiB checkpoint is hoisted.
  **Deliberately NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame
  beat partitions, which is a production planning decision, not a cleanup. The preset
  carries a `_ceiling_note` saying do not export it alone. Compare WAN, which B3
  wired correctly: `otr_8gb_wan.json` sets BOTH `launch.env.OTR_WAN_TI2V_MAX_FRAMES`
  and `video.max_render_frames`.

Archived-only detail (finished work / provenance removed from the go-forward rows):
- "CODE-COMPLETE, PROOF-INCOMPLETE" status framing.
- The wired chain is pinned by `tests/test_remaining_video_contracts.py` (nine hop-by-hop tests) and `tests/test_multiclip_effective_contract.py`; commit history in the archive.
- The four-arm bench prequalified on a 16 GB card told to reserve 8 GiB; the 18-engine campaign ran as coverage.
- Found 2026-07-27, B6 panel, two lenses independently.
- Compare WAN, which B3 wired correctly: `otr_8gb_wan.json` sets BOTH `launch.env.OTR_WAN_TI2V_MAX_FRAMES` and `video.max_render_frames`.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

Measured table and the full argument: `docs/GO_FORWARD_ARCHIVE.md`. (204-entry reference bank: 153 resolvable on disk, 97M/106F/1N; `MAX_SPEAKING_CAST = 10` is a Bark artifact; writer pool is 6M/4F.)

Related and already shipped: `num_characters` is now a REQUEST rather than a cap
(operator directive, all banks) -- see `tests/test_cast_size_is_a_request.py`.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 4.X OTR-LITE -- a second, frictionless pack AFTER v2 ships (operator idea, 2026-09-03)

### 4.X OTR-LITE -- a second, frictionless pack AFTER v2 ships (operator idea, 2026-09-03)

**Not work yet, and explicitly not before v2 ships.** The operator, thinking it
over away from the desk: *"once we ship v2 we ship an OTR-Lite, similar
architecture but only the most frictionless auto-download non-gated models, and
maybe just maybe we can figure an ffmpeg-less solution for a truly streamlined
workflow."*

**Two halves, and the second one pays a debt nobody connected to it.**

* **Non-gated auto-download only.** Already half-mapped:
  `scripts/otr_fetch_lane_weights.py` offers UNGATED sources by design and
  deliberately refuses to paper over the one gated repo (Lightricks/LTX-2.5), so
  its lane list is effectively the candidate set. Measured on the 4090 pod
  2026-09-03, the fully-frictionless bundle is `haunted + one z_image precision
  + stable_audio_3` -- about 20 GB for one complete episode.
* **ffmpeg-less.** This is not only an install-friction win. `subprocess` calls
  to ffmpeg/ffprobe are **35 of the pack's 158 Comfy Registry scan findings**
  (`python_command_injection_risk`), plus most of the 12
  `python_url_command_execution` ones. An in-process encode path shrinks the
  largest non-`os.environ` finding class at the same time as removing the binary
  dependency -- the two goals are the same work. PyAV (`av>=16.0.0`) is already a
  ComfyUI CORE dependency, so it ships on every install.

**THE ONE KNOWN BLOCKER, and it is where this starts.** This PyAV build has no
`libass` and no `drawtext`, so CAPTION BURN and CREDITS cannot move in-process
as-is. The mux, the silent composite and the probe already have PyAV routes. So
the first question of an OTR-Lite effort is captions and credits without
libass/drawtext -- not the mux, which is the part that looks hard and is not.

**Related evidence already on disk:** `docs/RUNPOD_INSTALL.md` section 7A (what a
second machine actually trips over), and the registry finding counts in
`docs/2026-09-03-registry-review-request-READY.md`.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### 4.Y BRING `word_razzle` HOME -- it is the one cloud lane whose NAME hides it (operator, 2026-09-

**The transparency problem, stated exactly.** 8 of the 30 registered video
engines render provider-side:

```
cloud_kling_avatar  cloud_seedance_2  cloud_wan_i2v  cloud_wan_i2v_audio
cloud_vidu_q2_pro_fast_720p          google_omni_video  google_veo_video
word_razzle   <- the outlier: nothing in the name says it phones out
```

Seven of the eight self-label via a `cloud_`/`google_` prefix. `word_razzle`
reads like a local text-effect lane and is not one. That is the misleading row,
and it is also the one with a plausible local replacement.

**WHY IT IS NOW FEASIBLE, AND IT MAY NOT EVEN BE A NEW ENGINE.** word_razzle is
"animate a word-card still into a living period poster" -- a cloud i2v taking
(init image + prompt + seed + duration + motion_mode). Both halves already exist
locally:

* **The card.** `ideogram4_local` is the pack's spelling champion -- its own
  recipe note records that every card in the campaign with perfect spelling
  rendered at mu 0.5. It did NOT exist when word_razzle was built as a cloud
  lane in 2026-07-03, which is the whole reason that lane is cloud.
* **The motion.** `still_motion`, `still_pan` and `ltx_video` already animate a
  still. A word card is a still.

**So the first question of the arc is whether this is an ENGINE at all, or just
a PROFILE** -- `character_image: ideogram4_local` + an existing local i2v in
`character_visual`. A profile costs nothing; a new engine id trips five
generated fixtures, two literal rosters, the terminal-frame proof rule and
`docs/VIDEO_LANE_PREFLIGHT.md` gates 1-8. Those are very different prices for
the same outcome, which is exactly why this is a design item and not a
drive-by.

**Owed before code:** a kibitz arc on that question. Also decide what happens to
the cloud `word_razzle` row -- retire it, or rename it `cloud_word_razzle` so the
roster is honest either way. Renaming alone would fix the transparency defect
even if the local lane never lands.

**Not done, and the predicate question if anyone wants it:** a RECIPE identity
for remote lanes (model id + resolution + params), so a mid-beat env flip is
caught for a cloud lane the way weight drift is caught for a local one. Nobody
has asked for long cloud beats; log it, do not build it.


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Bug Bible promotion field -- pending actions only

| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |

(12.151-12.156 promoted 2026-09-04 -- PBUG-20260904-01..04 and -06; receipts in
HANDOFF_LOG. The 12.139 / 12.140 promotions completed 2026-08-28 and the 2026-08-25 /
2026-08-18 / 2026-08-17 promotion receipts are in the archive.)


## ARCHIVED 2026-09-04 evening (compaction pass) from ### Open risks

- **NO CLIENT BANK HAS EVER RUN LIVE.** Every extensibility wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven path end
  to end (fetch -> interpret -> writer -> cleanup -> tail -> publish).
- **CLIENT-AUTHORED PYTHON executes in-process** (wave 3).
- **The client-facing surface is LIVE TEXT, not just docs:** the `custom_source_bank`
  row's `guide_ref` is raised to the operator by `require_runnable_bank`, and the
  `source_bank` tooltip repeats it.
- **The ledger-cleanup pass runs on EVERY bank, not just client banks** (`3d97a130`). It
  is a no-op on a complete ledger and costs no LLM call there, but two shipped-lane
  behaviours did change and are worth watching on the next live legs: (a) unsafe spoken
  language on a `content_owned_readonly` bank is now REPAIRED at the writer tail instead
  of reaching G9 untouched, so a leg that used to die at freeze may now ship a sanitized
  line; (b) a blank `meta.episode_title` is now filled at the tail instead of exploding
  later in `otr_credits_roll`. Both are the intended direction under THE LAW; neither has
  a live receipt yet.
- Lean-mean has one current ordered campaign in `docs/LEAN_MEAN_CLEANUP.md`.

## ARCHIVED 2026-09-04 evening -- ruling fragments the compaction pass did not carry verbatim

The compaction rewrote each section into short to-do rows. Most rulings were carried in
the rewrite, but these lines are not present VERBATIM in the compacted plan, so they are
kept here with their neighbours -- a reworded ruling is not proof the original survived.

**From ### 1.3 GHOST POOL -- uniqueness on the finalized prompt (queue item 3b; r1 is in, build (line 108 of the pre-compaction plan):**

```
bounded progression, total by construction: unused finalized prompt -> reuse a leaf where a
different motif keeps the prompt new -> reuse the least-recent signature, deterministic on
`episode_seed + beat_id`, never adjacent. The allocator appends a PER-BEAT reuse
disposition to that beat's existing `fallback_reason` (ShotLock stamps one batch-wide
reason today, which would erase the original model-failure reason). Only pool exhaustion
```

**From #### The P0 / source-span cluster (2026-07-30) (line 174 of the pre-compaction plan):**

```
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do not
  raise the minimum word target as a capacity workaround.

```

**From ### 1.1 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (line 424 of the pre-compaction plan):**

```
  PyPI wheels); `onnxruntime-gpu` optional. Keep the model on CPU by default when a video
  engine holds the GPU; it is an 82M model and faster than realtime on CPU.
* SHIP SCOPE (operator ruling 2026-09-01: "we can ship all audio lanes with kokoro"): in
  the same change, `workflows/otr_canonical.json` and every generated variant default BOTH
  voice slots to kokoro (`char_voice_engine`, `announcer_voice_engine`, `voice_bank`
```

**From ### 1.1 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (line 429 of the pre-compaction plan):**

```
  `kokoro_builtin`); indextts2, chatterbox, dia and bark stay in the dropdowns as
  install-it-yourself upgrades. That dissolves the indextts2-default half of the ship-audit finding (the
  reference-WAV preflight is the bullet below) and the Section 4 voice-bank item stays parked.
* THE COMPATIBILITY TABLE IS GENERATED, NOT HAND-KEPT: extend
  `scripts/otr_machine_matrix.py` to emit a "Voice engines" table into
```

**From ### 1.9 ONE MANIFEST, PREFLIGHT AUTO-DOWNLOAD (queue item 8; design row; operator asked  (line 461 of the pre-compaction plan):**

```
2. Preflight resolves the selected dropdowns to artifacts, prints the total, refuses early
   if disk is short, downloads into the running ComfyUI's models tree through `folder_paths`
   (never `C:\ComfyUI-Models`) with `.part` files, hash verification and resume; the fetcher
   already has that code and moves under `nodes/` so it ships.
3. Gated rows (LTX 2.5) refuse BEFORE downloading, naming the terms URL and the token step.
```

**From ### 1.9 ONE MANIFEST, PREFLIGHT AUTO-DOWNLOAD (queue item 8; design row; operator asked  (line 464 of the pre-compaction plan):**

```
   already has that code and moves under `nodes/` so it ships.
3. Gated rows (LTX 2.5) refuse BEFORE downloading, naming the terms URL and the token step.
4. A `download_policy` widget: auto (default), ask (list sizes only), never (air-gapped).
The manifest carries the kokoro-onnx weights, so this lands after Section 1.1. Design item:
kibitz arc before code. (The first-run download bill is README's number, not this file's.)
```

**From ### 1.11 THE ADAPTATION DESIGN (queue item 8; hardened, NOT yet built; multi-session --  (line 496 of the pre-compaction plan):**

```

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md` (5080-local).
Keystone: compile source speech from an authenticated segmented artifact, never generate
it; "summarize into X words" means SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not
paraphrasing (which also removes the VRAM hazard: no model sits in the source-speech path).
```

**From ### 1.12 STYLE / IDENTITY DECISION WORK (queue item 8; backlog; not the next coder windo (line 549 of the pre-compaction plan):**

```
   (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`
   reads it too; missing them fails ledger validation on the first episode.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast` (names
   are a pure pool draw; `source_character_names` deliberately None for invention
   lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence:
```

**From ### Batch R2 -- ONE image-mode sweep on `otr_soak_llmsweep_02` settles SCENE + PORTRAIT  (line 719 of the pre-compaction plan):**

```
* **If they refuse:** the fork below is real and gets the full arc, now with
  numbers instead of an inference.
* **If they do not:** this row collapses to a documentation correction and costs
  nothing further. That is the likelier outcome and it is why no arc runs first.

```

**From ### Batch R3 -- FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator (line 731 of the pre-compaction plan):**

```
PBUG-20260826-01.

### Batch R3 -- FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator directive 2026-08-25)

After Leg 0 (Section 1.7), four canonical legs prove all 7 surviving local LLM rows in
```

**From ### Batch R3 -- FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator (line 737 of the pre-compaction plan):**

```
live-watch and an eyeball re-observation chance. Charter: `docs/OTR_STANDING_RULINGS.md`
"ONLY EASY-TO-LOAD LLMs SHIP" -- every surviving row does creative AND technical, or it
is ripped on a MEASURED failure, never on assumption. All 7 rows are on disk;
`docs/LLM_PREFLIGHT_GUIDE.md` is the preflight guide.

```

**From ### The registry rows, carried forward (the 2026-09-02 queue's item 1) (line 897 of the pre-compaction plan):**

```
  and if the container's preinstalled CPU torch does not satisfy the constraint, pip would
  download multiple GB inside the 600 s extract timeout. Unknowable without the image.
* **ANSWERED 2026-09-03 18:30Z, AND THE ANSWER CLOSES (b) AS NOT-OURS. DO NOT CHASE
  KOKORO.**
  **`aff1f9c4` STAYS.** The pycairo marker is correct on its own merits -- a package
```

**From ### The registry rows, carried forward (the 2026-09-02 queue's item 1) (line 924 of the pre-compaction plan):**

```
  next occurs in 2028. The operator added the survey as a comment on 2026-09-03 and
  corrected the issue's date window: extraction did not stop in February, it stopped
  **2026-04-28**, and 19 packs' newest success falls in April with nothing after. Do NOT
  open a second issue on this; comment on #203.
* **PARKED BY THE OPERATOR 2026-09-03: THE REVIEW REQUEST IS THE LAST THING WE DO,
```

**From ### The registry rows, carried forward (the 2026-09-02 queue's item 1) (line 1013 of the pre-compaction plan):**

```
**4. THE 8 GB SHIP SET -- what the clean room proved becomes a saved dropdown set.**
(b) a shipped `otr_8gb_klein_ltx25` profile and variant (the clean-room profile is
untracked on the 4060 -- promote it, do not rewrite it); (c) the eight profiles that still
pair an 8 GB ceiling with the 12B writer get the E2B writer the 8gb class ships. Section
1.5.
```

**From ### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE B (line 1216 of the pre-compaction plan):**

```
### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected** (operator: *"park it on go forward"*). The writer casts
from 10 Bark presets (6M/4F, `config/cast_pools.py`) while the cloning engines draw from a
204-entry reference bank (153 resolvable on disk, 97M/106F/1N,
```

**From ### Bug Bible promotion field -- pending actions only (line 1358 of the pre-compaction plan):**

```
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |
| **Seedance softener mangles authored prompts (2026-08-17)** | CANDIDATE only: fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever produces the artifact; nearest coverage `12.108` does not cover blind-regex rewriting of authored text. Detail: `docs/GO_FORWARD_ARCHIVE.md` |
```

## ARCHIVED 2026-09-04 late -- the KOKORO-ONNX design row (DONE: backend shipped in `70a1d33b`, the voice-engines table is generated by `scripts/otr_machine_matrix.py`, the deps rode alpha.17; the operator caught it still listed as open)

### 5.4 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (design item, kibitz arc BEFORE code)

Settle the details in the arc, do not re-derive these.

* **Add an ONNX backend to `nodes/_otr_audio_engines/eng_kokoro.py`** so a default voice installs on every box: load model + voices, phonemize with espeak-ng, run onnxruntime, return 24 kHz audio. Prefer the torch backend only when `kokoro` is importable (3.12 boxes); ONNX everywhere else; bark stays the zero-dependency fallback.
* **Pick ONE weights source and wire the fetch.** kokoro-onnx wants `kokoro-v1.0.onnx` + one `voices-v1.0.bin` (npz keyed by voice, GitHub release `model-files-v1.1`); the ungated HF mirror `onnx-community/Kokoro-82M-v1.0-ONNX` (86 MB q8f16 to 326 MB full, 55 per-voice `.bin` at 0.5 MB) uses a different per-voice format. Fetch via the existing `_otr_kokoro_voice_prefetch` machinery, and map announcer/character voice ids onto the same names the torch path uses so the cast ledger does not change.
* **Registry deps:** `kokoro-onnx>=0.6.1` and `onnxruntime>=1.20.1` in both manifests (plain PyPI wheels); `onnxruntime-gpu` optional. Keep the model on CPU by default when a video engine holds the GPU (82M model, faster than realtime on CPU).
* **Default both voice slots to kokoro in the same change** -- `workflows/otr_canonical.json` and every generated variant: `char_voice_engine`, `announcer_voice_engine`, `voice_bank` `kokoro_builtin`; indextts2, chatterbox, dia and bark stay in the dropdowns as install-it-yourself upgrades.
  * SHIP SCOPE (operator ruling 2026-09-01: "we can ship all audio lanes with kokoro").
  * The Section 4 voice-bank item stays parked.
* **THE COMPATIBILITY TABLE IS GENERATED, NOT HAND-KEPT:** extend `scripts/otr_machine_matrix.py` to emit a "Voice engines" table into `docs/MACHINE_MATRIX.md` from `nodes/_otr_audio_engines/registry.py` (device_backends, requires_sidecar, requires_vendor, practical_without_gpu, model_requirements) plus a per-engine "ships with the pack / install on your own" column, so README can point at one table.
* **Cloning engines preflight their reference WAVs:** preflight the resolved `ref_path` files in `OTR_CastLock` BEFORE the writer call whenever a cloning engine IS selected, so an opt-in indextts2 / chatterbox / dia user fails in seconds, not after a whole script.
* **DONE WHEN:** a clean 3.13 portable install renders a 1-act episode with kokoro voices for announcer and characters through `workflows/otr_canonical.json` and publishes to `otr/obs/`, and the same commit passes on the 5080's 3.12 venv with the torch path still selected.
* **BLOCKED ON / release gate:** the `pyproject.toml` dependency edit auto-fires a registry publish. It rides the bump AFTER alpha.15 (and alpha.16 if the control runs) resolves -- never while a version is Pending, and never reusing a string (CLAUDE.md 7A).


---

# Archived 2026-09-05 -- superseded GO_FORWARD sections

Replaced because they described a world that had ended: the scan-collapse
ratchet commit was named as the NEXT commit hours after it shipped, and item 4
still instructed the next window to post a review draft that never mentions the
registry BAN. Kept verbatim for the record; the archive is not read to resume.

## (archived) item 4 (registry)

## 4. THE REGISTRY -- publish the first NON-ALPHA, file the review the same day

Everything above changes shipped code; everything below needs the published pack. The
review request is a PUBLIC post and needs the operator's explicit go.

**Standing constraints:** the 5080 loop is untouched -- nothing ships that reduces tomorrow's `obs` count. The pod stays STOPPED until item 9. One coder window per file owner (CLAUDE.md section 1); every chunk = focused tests + full suite + Bug Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

**1. THE REGISTRY.** Two separate goals: (a) an INSTALLABLE listing (needs an admin approval we cannot self-serve), (b) the card's "N Nodes" count (a different pipeline, stalled on Comfy-Org's side -- not ours).
* **NEXT CODER CHUNK: the scan collapse ratchet commit.** Plan closed at r4, orphan rips shipped, owners on disk; the ratchet commit is what remains. Needs nothing from the operator.
* **PREP STATUS 2026-09-04 -- three of the five handed-over items are DONE:**
  (b) `viewer/` is already excluded (`.comfyignore:167`, 2026-09-03) -- nothing owed;
  (d) the `AUTH_TOKEN_COMFY_ORG` literal is ALREADY SCRUBBED -- `grep` finds ZERO in
  `nodes/` and `__init__.py`; what remains is the lowercase field NAME
  `auth_token_comfy_org`, which is pinned data about Comfy's own partner nodes, not
  a credential; (e) the draft polish is folded into the v2 draft below.
  STILL OPEN: (a) re-check `GET /nodes/comfyui-old-time-radio/versions/<v>/comfy-nodes`
  after a publish, and (c) the post itself.
* **THE DRAFT IS NOW `docs/2026-09-04-registry-review-request-READY-v2.md`.** The
  2026-09-03 draft is the alpha.17 record and MUST NOT be posted -- its numbers are
  stale in our favour. The v2 draft carries the measured collapse (103 env files -> 4,
  35 spawn sites -> 3, three singletons -> 0) and changes the ask from "please
  overlook 158 findings" to "we collapsed what could be collapsed; here is the diff
  and why the rest is the render path". It has THREE placeholders only the operator
  can fill: `<VERSION>`, `<SCAN COUNTS>` from the real post-publish scan, and a
  re-verified `0 critical`. Do NOT predict the scanner's counts -- our measured
  source-side change is a different claim, and conflating them is what would make a
  reviewer distrust the rest.
* **PARKED BY THE OPERATOR 2026-09-03: THE REVIEW REQUEST IS THE LAST THING WE DO, AFTER WE SHIP AND DROP THE ALPHA TAG.** Not a deferral -- a sequencing decision, because admin approval is PER VERSION.
  **File it on the first NON-ALPHA version, the same day it publishes**, so the version under review is the version a stranger would install. Include the question about node-level review; it costs them less than five separate ones.
  Draft is ready and needs no further work: `docs/2026-09-03-registry-review-request-READY.md`. **Re-read it before posting** -- the version id, the counts and the "0 critical" claim all move with each new publish, and a request quoting a superseded version is worse than none. Filing is a public post and needs the operator's explicit go.
* **DO NOT CHASE KOKORO.** The old control experiment (republish alpha.8 byte-identical) is CANCELLED.
  **`aff1f9c4` STAYS.** The pycairo marker is correct on its own merits and is what makes a Linux `pip install -r requirements.txt` work at all. The Linux pod keeps the engine because `scripts/otr_pod_provision.sh` apt-installs the headers and pip-installs pycairo explicitly -- do NOT "simplify" that line away.
  **Reproduce before re-opening this:** compare the newest `success` timestamp across a broad pack sample against OTR's first publish date. If a pack ever extracts successfully again, the question becomes live; until then the card's node count is not a work item.
* **STILL FLAGGED IS THE EXPECTED OUTCOME for (a), not a failure.** There is no publisher self-service route to Active.
* Do NOT open a second issue on the node count; comment on `Comfy-Org/registry-backend#203` (open, operator-filed).
* Open risk, unsettleable from here: kokoro pulls `torch`; if the container's preinstalled CPU torch does not satisfy the constraint, pip downloads multiple GB inside the 600 s extract timeout. Unknowable without the image.
* **A REGISTRY-TESTER REPO (operator idea, 2026-09-03): worth it for INSTALL verification, NOT for scanner probing.** Iterating publishes to map which YARA patterns trip is the part to avoid: it is noise in a real security queue, and it is unnecessary -- `?include_status_reason=true` already returns all findings itemized by rule.
* DONE WHEN a version reads `Active` and a Manager install on the 4060 lands the OTR nodes. Never version-delete: a soft delete burns the string permanently.


## (archived) item 1 (scan collapse)

## 1. THE SCAN COLLAPSE -- the MIGRATION is done; one acceptance step remains

**Batches (a)-(d) and the ratchet commit are SHIPPED** (receipts in
`docs/HANDOFF_LOG.md`). env offender FILES 103 -> 2, spawn files 20 -> 1, all three
singletons cleared, six of twelve "url command" hits gone. Both PENDING sets now
contain ONLY the two BLOCKED files below, so the guards are green with nothing left
to migrate.

**TWO FILES ARE PERMANENTLY BLOCKED, each with its reason recorded in the guards'
own `BLOCKED` tables. Do not sweep either up mechanically:**
* `nodes/_otr_audio_engines/eng_indextts2.py` -- byte-hashed by Lemmy's voice
  qualification. Rides along with the next re-audition wanted for other reasons;
  ruled and closed, see
  `docs/2026-09-04-PROBLEM-STATEMENT-indextts2-fingerprint-lock.md`.
* `nodes/_otr_writer_heartbeat.py` -- a LEAF by contract; a pack import
  reintroduces the cycle that once left two generate transports blind.

**THE ACCEPTANCE LEG IS DONE (2026-09-04 evening) -- ITEM 1 IS CLOSED.**

A one-act canonical leg on `workflows/otr_canonical.json`, run AFTER the security
build so it doubles as that build's live proof:

    RESULT SUCCESS   prompt 010a3848   Prompt executed in 00:10:17
    obs_publish OK -> otr/obs/a_name_stripped_bare_20260904_185159__paper_origami
                      __still_flat__z_image_turbo__kokoro__shakespeare_final.mp4
    13,062,017 bytes on disk, 1920x1080 h264 + aac
    v=88.280s a=67.071s tail_budget=21.2s declared -- the A/V gap is IN FAMILY
      (the five previously published episodes run 22-29s)

Every stage the security build touched actually RAN, which is the point of the
leg: ProcgenBlend blended, CaptionBurn burned 11 caption events through the
`ass=` filtergraph, CreditsRoll appended 21.2s, and MasterAudioMux reported
**audio_byte_identical OK** -- the exact invariant that would have broken had the
widget been severed inside `_ffmpeg_bin` instead of at the execute method.

ZERO deprecation warnings (every shipped widget carries the bare default, as
measured) and ZERO refusals from any new guard.

**A NOTE THE NEXT WINDOW NEEDS.** The registry floor is NOT zero and never was: the
closed plan says "the gate is ZERO findings or a manual admin approval; nothing here
reaches zero (ffmpeg is a subprocess and that is the render path)". The collapse buys
a report a human can read in one screen -- about eleven lines instead of 158, and the
`credential-access` tag on two files instead of eleven. The route to Active is the
manual review in item 4.


## (archived) item 1A (security)

## 1A. THE REGISTRY BAN -- BOTH SURFACES ARE CLOSED. What is left is the PUBLISH.

**SHIPPED 2026-09-04 in `a9e0383e`.** Suite 13659 passed / 126 skipped /
1 xfailed. Receipt in `docs/HANDOFF_LOG.md`.

The ban verdict on alpha.13/.14 named two surfaces and **both were real**:

| verdict clause | the actual surface | status |
|---|---|---|
| "no-auth route" | `POST /otr/video_render_single` + `/otr/video_render_soak` shipped UNCONDITIONALLY in alpha.13 (`f9986b9f`), reading caller-supplied paths from an unauthenticated body | closed `b198026a`, 2026-09-03 -- in alpha.16+ |
| "node widget" | the `ffmpeg` STRING widget on 5 nodes reached `argv[0]`, beat the operator's `OTR_FFMPEG` pin, and yielded a SECOND binary via the ffprobe sibling rule | closed `a9e0383e` |

Also closed in the same commit, found by our own review rather than the
scanner: a `replay_from` widget that accepted a UNC path (SMB/NTLM coercion
needing nothing planted locally), the wildcard CORS header on the
unconditionally-registered ledger GET, an unescaped basename interpolated into
an ffmpeg filtergraph (4 sites, 2 helper copies), and cwd-relative binary
resolution.

**STILL OPEN, in priority order:**

1. ~~THE PUBLISH~~ **DONE. `2.0.0-alpha.19` is published and Pending**
   (`531ddda4`, workflow green, 19 dependencies recorded). And alpha.18's scan
   finally landed, which measures the collapse for the first time:
   **158 findings -> 12**, all `info`, zero critical. env manipulation 103 -> 4,
   command-injection 35 -> 3, url-command-execution 12 -> 0, and the
   bytecode / windows-process / sensitive-file singletons all to 0. (Correction
   worth keeping: the url-command drop was the SPAWN-OWNER collapse, not the
   error-string rewording an earlier note credited.)
2. **File the re-review request.** `docs/2026-09-04-registry-review-request-SHORT.md`
   is rewritten for the real story ("you were right on both counts, both are
   closed"). **DO NOT SEND either older draft** -- they never mention the ban,
   and they claim credit for two `critical` findings that were already gone as
   of alpha.16 and for a `python_url_command_execution` drop that never happened.
   Posting is a PUBLIC ACT and is the operator's alone.
3. **Media-path containment (NOT started).** `_validate_contract`
   (`_otr_paths.py:224`) is the pack's own containment guard and is called by
   NOTHING outside that file. Fifteen STRING path widgets exist; the four
   output-path ones plus `out_suffix`
   (`otr_post_upscale_procgen_blend.py:757`, which walks out of the episode dir
   if it carries a separator) let a workflow read one arbitrary file and write a
   transcode anywhere writable. Fable classified this as DATA LOSS, not RCE, and
   both r1 lanes cut an unrestricted write-audit from the security build's
   scope -- so it is its own row, and it wants an arc: containment could break a
   legitimate render that writes outside `otr/`.
   The UNC half of this class is ALREADY closed by `proc.py`'s remote-argument
   rule.
4. **Schema removal of the five now-inert `ffmpeg` widgets.** Deliberately
   deferred: keeping the field is what made the security fix a ZERO workflow-JSON
   change. Measured 2026-09-04: **101** workflow files (not 63 -- that number was
   never verified), 465 `ffmpeg` widget values, every one the bare literal
   `"ffmpeg"`. Removal is the 3-part edit (`widgets_values`, the `inputs`
   descriptor, every later link `dst_slot` repaired BY IDENTITY) plus
   `build_variants.py --all` and the four verification tests.
5. **`otr_video_render_batch` exposes the same GPU harness** (`render_single` /
   `run_gpu_soak`) that the env-gated POST routes call, as an ordinary node with
   no gate. Not obviously a defect -- every node is invocable by a workflow --
   but it means gating the routes did not remove the capability. Decide whether
   that matters before claiming the harness is "off by default".

**THE METHOD THAT FOUND ALL OF THIS, worth repeating:** read the registry's own
`status_reason` instead of guessing from the scan counts. The counts were never
the blocker.


## (archived) WHERE TO PICK UP

## WHERE TO PICK UP

**State:** `v2.0-alpha`, HEAD == origin, clean tree. Suite baseline **13480 passed / 126
skipped / 1 xfailed**. Bible **335 entries**. No resident server; VRAM at the desktop
baseline. Codex CLI standard credits are OUT until **2026-09-07 08:34 PDT** and Spark
overflows its context on a 12 KB plan -- fill that seat with Sonnet/Opus or a second agy
model (Gemini 3.1 Pro (High) and 3.8 Flash are different reviewers) and name the roster.

**NEXT COMMIT: the ratchet commit of the scan collapse** (item 1). Read
`kibitz-runs/2026-09-04-registry-findings-collapse/r4/final.md` whole before typing. The
owners `nodes/_otr_shared/env.py`, `nodes/_otr_shared/proc.py` and the shared ratchet
`tests/fixtures/ratchet.py` are on disk and inert until the migration starts.

**Only the operator does these -- stop and ask:** the `pyproject.toml` version bump (a
publish is his eyeball), the review request (a public post), and the rulings in item 6.

**Keep this file clean:** when a row is finished, delete it from here and write the receipt
in `docs/HANDOFF_LOG.md`. This file is the to-do list, not the log.

