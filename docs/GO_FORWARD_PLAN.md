# OTR Go-Forward Plan

**Forward-only, and now literally so.** Open work and nothing else.

* **Standing rulings, laws, review routing and the credit ladder:**
  `docs/OTR_STANDING_RULINGS.md` -- **read it, it is not optional.** The plan says
  what to do; that file says what you may not do while doing it.
* **Closed receipts:** `docs/GO_FORWARD_ARCHIVE.md` (not read to resume).
* **The highest authority is still `CLAUDE.md`**, unchanged.

Split 2026-08-23 on the operator's instruction: *"go forward should only have the
go forward plans. Only."*

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

## THE CURRENT STEP -- READ THIS FIRST

### OPEN, IN PRIORITY ORDER

### >>> COCKNEY BLEED -- CODE SHIPPED, A LIVE LEG AND A BIBLE ROW OWED <<<

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

### >>> OWED: THE LIVE H3 ACCEPTANCE PROOF FOR THE PROMPT-POLICY FIX <<<

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

---

### >>> THEN: SCENE + PORTRAIT `elements: []` -- MEASURE IT BEFORE DESIGNING IT <<<

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

**SO THE NEXT STEP IS A MEASUREMENT, NOT A PANEL. It is a RENDER item.**
Re-run the image sweep's profile `otr_soak_llmsweep_02` against post-`ae7e7b6a`
HEAD -- `scripts/otr_bank_engine_sweep.py`, image mode, which walks every bank
against both engine profiles -- and read whether SCENE and PORTRAIT beats refuse
at all now.
* **If they refuse:** the fork below is real and gets the full arc, now with
  numbers instead of an inference.
* **If they do not:** this row collapses to a documentation correction and costs
  nothing further. That is the likelier outcome and it is why no arc runs first.

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

---

### >>> ALSO OPEN: THE SANCTIONED-GAP CONTROL PATH <<<

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

**Next step is r2, the coding plan**, per r1's own stated roster
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

---

### >>> NEXT ITEM: THE LOCAL-LLM ACCEPTANCE SWEEP (operator directive 2026-08-25) <<<

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

**THE OPERATOR REDIRECTED THE WINDOW MID-SESSION, and these are his words:**
*"when all this LLM coding is done we should look and do more coding and retest
all local LLMs on a 1 runthrough that should catch it"*; *"if it doesn't fit
nicely or requires Ollama rip it from the dropdown and blast radius"*; *"clean
sweep I only want easy to load LLMs"*; *"there should be an LLM preflight guide
-- preflight guides for adding all your own components"*; *"all models should
live out here `C:\ComfyUI-Models`"*; and *"all LLMs should either be able to
play creative or technical equally. If they're not, and they were not tested or
not implemented, and they serve no worth, we should rip them out."*

**ALREADY DONE (pushed, green, lockstep-verified):** the clean sweep itself
(`Qwen/Qwen2.5-14B-Instruct` ripped with its blast radius), the PASS-tier
invariant gate, `docs/LLM_PREFLIGHT_GUIDE.md`, and the `Q6_K` dropdown removal.
The ruling is recorded in `docs/OTR_STANDING_RULINGS.md` ("ONLY EASY-TO-LOAD
LLMs SHIP"). All 7 surviving local rows are verified present on disk.

**STILL OPEN -- THE SWEEP ITSELF. Design is done (11-agent fan-out, 2026-08-25),
build is not.** The honest shape, which is NOT one render:
* **7 local rows / 2 model slots = 4 canonical legs MINIMUM**, plus a **Leg 0**
  in-process preflight (no ComfyUI; `request_slot` -> ~40-token generate ->
  `_self_unload` per row, with `reset_peak_memory_stats()` around each). Leg 0
  is one command, ~15-20 min, and is what fails loudly on a dead row -- but a
  leg that never reaches `otr/obs/` did not pass, so the 4 canonical legs are
  the real proof.
* **Every leg must PIN `--source-bank` to the scifi lane.** Canonical ships
  `'roll (any eligible bank)'`, and `_otr_scifi_news_pro.py` is the only runner
  code-verified to drive BOTH slots. Unpinned, a leg can land on a lane that
  never touches the technical slot and the sweep proves nothing about that row.
* **`gguf_quant` is ONE per-run widget**, and `unsloth/Qwen3-8B-GGUF` ships
  only `Q4_K_M` -- so any leg carrying it runs Q4_K_M.
* **A KNOWN FALSE-GREEN TO DESIGN AROUND:** `meta.slot_calls_by_slot` is
  incremented ONLY inside `_SlotScheduler._account_and_get_entry`
  (`OTR_LedgerScriptWriter.py:627`). SIX `request_slot` sites live outside it
  (`story_orchestrator.py:1224`/`:1351`, `otr_shot_lock.py:1002`,
  `OTR_LedgerFreezeCascade.py:282`, `OTR_LedgerScriptWriter.py:4393`,
  `_otr_motion_clause.py:361`). The counter proves IN-WRITER generation only;
  reading it as full-row exercise is a false green.
* **The operator's creative/technical parity rule is the sweep's acceptance
  criterion.** Structurally both slots already build from the IDENTICAL
  `dropdown_choices()` list, so no row is slot-restricted; what is unproven is
  whether each row can actually do the TECHNICAL job (constrained JSON / GBNF).
  A row that cannot do both, and was never tested or implemented, is a RIP
  candidate under his rule -- but rip only on a measured failure, never on
  assumption.
* **A negative probe worth running deliberately:** the gemma GGUF row at
  `Q8_0` / `n_ctx=4096` needs ~14.70 GiB FREE against a 15.92 GiB card with
  ComfyUI resident, and `_otr_gguf_backend.py` compares against
  `mem_get_info()` FREE with "NO silent context downgrade". Either outcome is
  informative; record both.

**Full design (coverage matrix, per-row assertions, skip-reporting rules, risks)
is in the 2026-08-25 workflow result; re-derive from this row if it is lost.**

---


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

## ON DECK -- WHAT REMAINS OF CONTINUITY CORRECTNESS

### 0-QUINQUE. MINIMAX H3 -- A SPRINT SERIES ON THE VIDEO PATHS (operator, 2026-08-09)

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
operator names the first sprint, write it into THE QUEUE at the top of this file
as its own row, and leave this section as the standing context.

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

### 6. A TERMINAL FREEZE GATE THAT HAS NEVER READ A POPULATED FIELD

`find_scene_coherence_issues` reads `lines[].scene_id`; the `scifi_news` lane
writes `beats[].scene_id`. 55 ledgers assert the check, 0 carry the field, 55
pass. Nothing in `nodes/` writes `lines[].scene_id` on ANY lane -- the check
never had a producer.

Join per line: `beat_id` -> beat -> `scene_id` -> declared scene. Add a **VACUITY
refusal** (an armed gate that examined zero linkages FAILS -- that is how this
survived 55 episodes). **Split request from verdict:** keep a
configuration-derived `scene_coherence_required` and write
`{required, checked, verdict, issues}` into `report.info` -- `run_gap_audit` is
READ-ONLY (`_otr_ledger_freeze.py:664-698`), so the gate must not mutate the
ledger; the phase wrappers already persist the report. Measure OFFLINE over the
published corpus first, then arm in ONE change -- no intermediate flag-off ship.
**CORRECTED 2026-08-23 -- this row's premise moved under it.** It used to say
"replace the stale hard-coded bank list at `tests/test_scene_guard_v4.py:89-99`
with registry-derived coverage (it omits `scifi_news`, the one bank that enables
the flag)". Both halves are now false: **`scifi_news` NO LONGER EXISTS** (the
live banks are media_archive, original, scifi_news_pro, public_domain,
shakespeare, custom_source_bank, and that test's list is exactly the first five),
and **NO bank sets `defaults.scene_coherence_check` at all** -- the writer reads
it at `OTR_LedgerScriptWriter.py:3176` and nothing supplies it. So the gate is
not merely inert on current banks; it has no consumer anywhere. What survives is
the DESIGN above (the join, the vacuity refusal, request-vs-verdict) and the
lesson beneath it. Whoever arms this decides first whether any bank should.

**The vacuity class is now proven twice** -- this gate, and the freeze test at
`test_g9_sfw_ship_stop.py` that filtered on a retired code prefix (fixed
`4506b1ed`). Any NEW armed gate ships with a vacuity assertion.

### 7. CHARACTER GENDER IS ROLLED ON PROSE LANES -- Scrooge shipped female (spec REWRITE owed; r2 and r3 both returned NO)

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
dramatis personae. **Operator decision, not a driver call.**

### Two carried items with no home of their own

(Titled "Bench leftovers" until 2026-08-23 -- a name that now reads as the
retired VIDEO bench and has nothing to do with it. The block it referred to was
an older conditional list, gone long before. Renamed rather than moved: both
items below are real and open.)

The first: **the three works that refuse to vendor** (`ghost_ship` gid 11045,
`purple_cloud` 11229, `beleaguered_city` 11521 --
`scripts/otr_vendor_public_domain_library.py:303/341/542` against the parser
at `:594-686`) **needs one Gutenberg fetch, so it is operator-opt-in only** --
not schedulable inside an offline sprint.

**Do NOT start the Shakespeare verbatim executor in this session.** It is a
multi-session structural change gated on the ownership table
(`docs/2026-08-03-fidelity-pass-ownership.md`) with four overwrite paths to close
first, and starting it half-way is worse than not starting it.

## PARKED -- D2 (renders have resumed; run when a render window is free for fail-hunting soak legs)

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

## THE ADAPTATION DESIGN (hardened, NOT yet built)

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

**NEXT, IN ORDER:**

1. **The segmented source artifact** (schema, spans, hashes,
   `body[start:end] == segment.text`, omission receipts) and the pass-to-field
   ownership table -- **nothing else codes until that table exists.**
2. **Cast from the selected cut.** Real scenes carry 3-12 speakers against a
   6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so which
   speakers appear must follow from the cut that fits the word budget. Coupled hard
   to the capacity guard: at act_count 1 there are exactly THREE voiced beats, so a
   4-person cast is a mathematically guaranteed `CastVoiceCoverageError` -- the
   failure that killed `scifi_news` in the six-bank run. `compute_episode_budget`
   must also receive the TRUE locked cast.
3. **Loosen the count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067` hard-
   raises on any locked != requested) and change the pack text that tells the model
   to drop figures.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner -- and
   bind its output to the verified body hash.
5. **Schema migration** to retire `cast_hints`; still required by the validators and
   by `public_domain_manifest_schema.json`, so manifests and tests migrate in the
   same change. (`visual_style_policy`, the other half of this item, was ripped
   2026-08-04.)

**KNOWN AND NOT FIXED:** `canonicalize_shakespeare_text` truncates at 12,000 chars
and the interpreter sees only the first 5,000, so a 3,445-word scene reaches the
brief as ~880 words, silently. Belongs with the artifact work, where each beat is
fed its own segment rather than a blind prefix.

## STYLE / IDENTITY DECISION WORK (backlog; not the next coder window)

Grounded by the 2026-08-03 four-agent forensics; every line has a file:line in the
session traces.

1. **"Invent one and tag it"**: add a derived style/genre field to
   `run_story_brief_reflection` (`_otr_story_brief.py:513` -- proven content-loyal on
   both specimens), stamp beside `story_brief`, repoint the treatment `Style:` line
   (`video_engine.py:1762`) and the HUD (`video_engine.py:1336` -> `_build_left`
   `:1592`) at it. Highest-leverage item here: it fixes the credits line for all six
   banks uniformly.
2. **Rename `meta.style` -> `meta.story_scaffold`** (operator: too many metas; the
   field is neither scifi nor a description). Consumers move in ONE atomic change:
   writer stamps, credits `_story_style_receipt`, `visual_plan.style`,
   `video_engine.py:1336`, tests -- AND the ledger validators (r3):
   `_otr_ledger_consistency.py` pins the field in its matrix
   (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`
   reads it too; missing them fails ledger validation on the first episode.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast` (names
   are a pure pool draw; `source_character_names` deliberately None for invention
   lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence:
   Evelyn/Leonard as offscreen lore; Fogbound Rails bio still opens "Lizzie Gray".
4. **Dead fields found**: `ending_template` computed but zero LineRequest call sites
   pass it; `seed_policy.style_seed_env` validated but unconsumed; `dramatic_state`
   derived PRE-dialogue goes stale in the treatment.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for. Scope
   as its own rip with the ledger law (every field one owner).

## OPEN OPERATOR QUESTIONS (flagged, awaiting a ruling)

* A research_only source now WITHHOLDS the OBS copy instead of killing
  the finished render (chunk 0.5 behaviour change, live since 08-15).
  If the operator wants the old kill-the-render behaviour back, it is a
  one-line revert -- say so.

* **Does `media_archive` want the catalog premise at all**, or the same
  scaffold-off treatment as `original`? Found by the five-bank beat test: a
  `pirate_radio_resistance_drama` premise was drawn over a film-reel standoff
  seeded by a real Library of Congress item on 'Midnight' (1939) -- the operator
  caught it on screen. Second specimen of the content-blind-draw class. The
  scaffold-off rule so far was stated only for `original`.
* **Rename the un-namespaced `OTR_WAN_*` knobs?** `eng_wan_i2v`'s six frozen knobs
  are `OTR_WAN_STEPS` / `_CFG` / `_SHIFT` / `_SAMPLER` / `_SCHEDULER` / `_NEGATIVE`
  -- no `I2V` namespace, unlike every sibling. Default if unruled: leave them. The
  freeze already removed the power that made the missing namespace dangerous (they
  are consent-act-only now and cannot bind a production leg), and a rename would
  silently break operator muscle memory for a sweep.
* **`style_tail_policy`'s closed enum cannot express a SHIPPED path.**
  `VALID_STYLE_TAIL_POLICIES` has `full` and `minimal_clean`, but
  `build_radio_host_prompt`'s `ltx_radio_mouth` branch
  (`otr_meta_brief_image_prompt.py:394-401`) RETURNS EARLY with
  `"%s, warm dramatic lighting"`, skipping both `finish_visual_prompt(...,
  era_profile="still")` and the `image_grade_tail` append -- deliberately, per the
  2026-07-02 operator look direction. The `ltx_audio_in` bookend row nonetheless
  declares `style_tail_policy="full"`. Adding an enum token is an operator call:
  either add a third token for "canonical warm, no era tail, no grade tail", or
  ratify that the `ltx_radio_face` path is EXEMPT from the plan's style-tail
  authority. Default if unruled: the exemption, because it changes no behaviour.
* **After profile retirement, who owns a tier's native render ceiling?** The full
  statement of this one is in the WAN 8-GB row under OPEN BUGS -- it is the single
  blocker on a code-complete block.
* **`check_compatibility`: ratify the inert constant, or schedule the rip?** See
  Open risks.

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not. That
split is why the two eyeball-era entries at the end are PARKED rather than live.

**EVERY LINE CITE IN THIS SECTION IS SUSPECT.** Each one checked during the
2026-07-27 triage had moved: `_is_cloud_video_engine` is `render_driver.py:1599` not
`1274-1295`; the "NO FALLBACK to text-only" refusal is `:2148` not `1801-1817`;
`_use_i2v` is `eng_ltx_video.py:583` not `559-572`. The defects are mostly still
real; their coordinates are not. **Re-pin a row's cite when you touch it.**
Path note (verified 2026-08-04): engine adapters live under
`nodes/_otr_video_engines/` (and `_otr_audio_engines/`, `_otr_image_engines/`)
-- bare `eng_*.py` cites in these rows are shorthand for those paths.

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

### The P0 / source-span cluster (2026-07-30)

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
  accepted ledger, and a fixture from these four strings.
- **The deterministic P0 rung PRUNES SILENTLY, which violates the plan's own
  Invariant 3.** `repair_literal_source_metadata` drops an unsupported span, then its
  evidence row, then the fact -- and emits no receipt. An accepted P0 index simply
  has fewer facts than the model wrote, and nothing says which were dropped or why.
  Under "fail loud, not fatal" the degrade is the right direction and the silence is
  not.
- **The deterministic P0 rung is ALL-OR-NOTHING across an artifact, and can poison
  its own good work.** It is handed `a0_payload` (all seven keys) while
  `_validate_fact_index` restricts spans to `allowed_source_fields` (the projection).
  A quote rehomed into a field the projection omitted makes `post_validator` reject
  the WHOLE repaired artifact -- "cites source field ... outside the supplied P0
  evidence" -- so one unlucky rehome discards every correct prune in the same pass.
  Either give the repairer the allowlist or prune per row.
- **Nothing measures whether a pruned P0 index is ACCEPTED** (recorded, no action
  owed yet). No live leg has ever run with the deterministic rung reachable (it became
  reachable at `47c554fa`, after the campaign stopped), and the rejection logs carry
  only a truncated `raw head` plus no source payload, so the question cannot be
  answered offline. A-1's instrumentation is what makes the next campaign able to
  answer it.
- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0 after
  two attempts on non-literal fact source spans; provider/model convergence, extends
  BUG-11.35. NOT a word/length gate. Blocks the last 120w receipt and the
  `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed in tree, reverify
  still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do not
  raise the minimum word target as a capacity workaround.

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

### The 8 GB / profile cluster

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

### Coverage, canvas and clip-contract

- **The route lock is ONE NODE TOO LATE for the image phase** (found 2026-07-25, node
  order confirmed against the canonical JSON: `87 VideoDirector -> 88 ImageDirector ->
  89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher -> 92 VideoRenderBatch`).
  `resolve_final_shot_engines` runs at node 92, but stills are minted at 91 and image
  PROMPTS at 89. The landed fix closed the spine-validation gap; the image phase still
  relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine seam").
  **Chunk 1 of the coverage block is the fix.** Note node 89 precedes node 90, so
  hoisting to ShotLock still does not put MetaBrief downstream of the authority --
  that needs a VideoDirector-time freeze and is NOT in scope. (This is also the
  "image-phase still ownership" item from the campaign queue.)
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
- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment item 1).
  `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated from a possibly-empty
  policy. B5 made this non-load-bearing for the render (the engine declares its own
  canvas now), so it is no longer urgent -- the drift guard in
  `tests/test_ltx_8gb_canonical_canvas.py` covers the disagreement that matters. Close
  it when the general canvas resolver lands.
- **Odd-canvas evenness is validated at the ENCODER, not where the canvas is chosen.**
  The stride defect itself is closed (`b1f2ee86`): `ffmpeg_silent_mp4_cmd` declares the
  REAL width/height and `encode_frames_to_silent_mp4` REFUSES an odd canvas by name,
  because yuv420p subsamples chroma 2x2 and cannot represent an odd dimension. Still
  true and NOT fixed: neither `WanInitImageMixin._dims()` nor the `Canvas` schema
  validates evenness, so an odd canvas is caught late rather than at the choice. No
  live producer builds one today (832x480, 512x288, 1472x832 are all even).
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
- **KNOWN LIMIT of the widened roster gate**, recorded so it is not rediscovered as a
  surprise: the codec flag is matched as a STRING CONSTANT, so a flag assembled at
  runtime (an f-string, `"-c:%s" % stream`) or the stream-index spelling `-c:0` is
  invisible to the sweep. Nothing in the tree does that today; an encoder that ever
  needs to must be pinned in `_ENTRY_POINT_PROOFS` by hand, which the inventory test
  makes a visible decision. Separately, ONE mutant survives the round by construction:
  deleting the self-proving membership assertion is catchable only by a meta-test of
  that assertion.
- **`ltx_av` underruns long beats** (found 2026-07-25, codex; confirmed). It caps at
  `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py:58`, default 497, env-overridable) and clamps
  at `:950-953`. It is NOT "renders to target natively" as three earlier docs claimed.
- **Ping-pong on a capped HuMo beat played lip sync BACKWARDS** -- FIXED in code @
  `a1d810f1`, but the finding is STATIC (no live artifact), so it is NOT a PBUG row. A
  capped-14B leg would reproduce it. Kept here so the live proof is not forgotten.
- **`docs/ENGINE_MATRIX.md` reports the DECLARED contract only** (found 2026-07-27).
  Correct today and consistent with its own stated design (every number read from the
  live registry). But the moment a profile pins an `ltx_8gb` ceiling, the matrix keeps
  printing `9-161 step 8` for a tier whose real window is narrower, and the `--check`
  drift gate cannot notice because it diffs the registry, which the effective contract
  never touches. Owed at the prequalification step, not before.

### Routing, env-capture and the credits card

- **`wants_talking_prompt()` escapes any routing freeze.** It calls
  `_recipe_config(self._recipe())` and `_recipe()` (`eng_ltx_av.py:402-432`) re-reads
  `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET name on EVERY call by documented
  design ("Read fresh every call"). So a `required="when_engine_talking"` row evaluated
  through the hook re-reads the environment after capture. S0b-core needs ONE shared
  `row_is_active(...)` evaluator over captured state, with the talking result inside
  `ltx_resolved`.
- **`provider_side` is a THREE-part rule, not an attribute.** `_is_cloud_video_engine`
  accepts a `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`.
  `cloud_kling_avatar` has no `provider_side` attribute and is caught by the id prefix
  alone, so an `engine_facts` builder using a bare `getattr` would classify it local
  and let the radio-host redirect send a cloud avatar to local LTX. Needs a regression
  on picked AND forced `cloud_kling_avatar`.
- **Four env-read sites missing from the S0b inventory:** `eng_ltx_video.py:541-564`
  (`OTR_ENABLE_LTX_I2V`), `render_driver.py:1176-1203` and
  `otr_meta_brief_image_prompt.py:297-300` (`OTR_ENABLE_HUMO_HOSTS`), and
  `eng_ltx_av.py:352-353,403-432` (recipe/UNET re-read outside `assert_usable`).
- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At
  512x288 (the ltx_8gb tier) col1 is 65px past its footer even with every ledger row
  this policy may drop already dropped; at 640x360 it is 12px over. Both are drawn
  anyway (a terminal node never destroys a finished episode) and LOGGED at ERROR
  naming the canvas -- the old behaviour was drawn, clipped by PIL, silent. At 288
  lines the three-column console is already a polite fiction: col3's scrolling
  transcript is as unreadable as anything col1 clips. This is a DESIGN job -- a card
  laid out for a small canvas -- not more ladder heroics.

### Test-harness and tooling

- **The B7 forbidden sweep cannot see an UNTRACKED file, so a new test file passes the
  gate and fails one commit later.** `tests/test_b7_forbidden_sweep.py` builds its
  input from `git diff s29-clean-slate-gate -- *.py`, which covers tracked files only.
  A new test file added and gated in the same session is green; the moment it is
  committed it enters the diff, and a forbidden runtime identifier in it turns HEAD red
  with nothing else changed. Cost one red HEAD. **Not fixed, because the fix is a
  judgment call:** sweeping the working tree instead of the diff would widen the gate
  to every untouched file in the repo. Cheap mitigation until then -- re-run the full
  suite once after the FIRST commit of any new test file.
- **MOOT since 2026-08-23 -- kept as one line so the finding is not re-derived.**
  This row noted that two `scripts/` bake-off runners aborted a whole sweep on an
  encoder count mismatch, and called it the correct direction that an operator
  should know before an overnight run. **Both runners were deleted with every
  other bake-off** ("delete any animatediff..." was the Ghost half; "I think I am
  done with all bakeoffs" was this one). The finding still generalises: a runner
  that discards the encoder's return value and recomputes the count independently
  will disagree with it silently. Worth carrying into any replacement sweep.
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

### PARKED -- unverified at HEAD, re-observe on the next real render legs
(The 2026-07-24 "after SFX" checkpoint is VOID -- SFX is parked. The re-observe
now rides whatever real render legs come next, D2 included.)

Both were eyeball observations against a story engine that has since had its LLM
vetoes ripped, THE LAW imposed, six banks renamed onto new packs, word-fit ceilings
retired, the repair-first plan landed, and a ledger cleanup pass added. Neither has a
reproduction at current HEAD, and under the standing rule a finding with no
reproduction is not a row. **Do NOT schedule coder time against either.** They are
settled by the operator eyeballing a real render leg after SFX: still there -> re-admit
as a FRESH dated row with that leg as evidence; gone -> the LAW-era work already fixed
it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`).
  Episodes START a story instead of admitting you into one; the announcer takes debate
  turns instead of framing. Operator eyeball 2026-07-11. If it survives re-observation
  the fix is still seam + score contract + fail-closed validator, never Python
  authorship.
- **Name-splice defect #2.** v4-campaign Phase 0 record in HANDOFF_LOG; its timebox
  predates THE LAW.

### Carried administrative rows

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until ratified
  at the next operator fan-out (green codex leg `c1f3891f` is the retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema `.v4`
  literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |
| **`PBUG-20260823-01` (preflight gate vocabulary collision)** | **PENDING single-entry promotion** per the 08-07 amendment: live-verified, fixed `b11a4269`, automatable coverage already in-repo (`tests/test_preflight_required_models_are_gateable.py`). Candidate rule: a gate must never treat "absent from an enumeration that could not contain it" as refutation. Check `otr_coverage_index.yaml` + Bible for overlap, then Three-File Contract in ONE survival-guide commit |
| **`PBUG-20260823-02` (watcher timeout worded as render death)** | **PENDING single-entry promotion**: live-verified on the exposing leg, fixed `cebe7c75`, coverage `tests/test_canonical_runner_timeout_is_not_a_death.py`. Candidate rule: a watcher's timeout is a fact about the WATCHER, never worded as a fact about the work. Same overlap check first |
| **Seedance softener mangles authored prompts (2026-08-17)** | **CANDIDATE, not admissible yet.** A blind regex pass over authored text produced "Dial slowly sweeps wildly" and inverted "vibrates aggressively" -> "vibrates subtly" on the DEFAULT pack's most energetic beat. Provable statically and now fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever runs and produces the artifact. Nearest existing coverage is `12.108`'s `self-veto-resolution` / `phrase-not-word-matching` tags, which do NOT cover blind-regex rewriting of authored text |

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

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval queue is
`docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented fixture creates a row.

## Open risks

- **NO CLIENT BANK HAS EVER RUN LIVE.** Every extensibility wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven path end
  to end (fetch -> interpret -> writer -> cleanup -> tail -> publish). Treat the first
  live client-bank leg as a qualification, not a formality. Deferred power-user tiers
  (client own-runner + staging, dependency manifest, standalone story_rules) are
  explicitly OUT of v1 and are a NEW block if the operator ever wants them.
- **CLIENT-AUTHORED PYTHON executes in-process** (wave 3). The posture that must hold in
  every future change: `--activate` is the consent act; the seam fails LOUD
  (`UserBankExecutionError`) and never substitutes; client code never touches the
  canonical ledger; owner IDENTITY is verified so a bank can only run its OWN bundle; the
  shipped fetcher/interpreter registries are never widened to admit a client id. Do not
  relax any of these for convenience.
- **The client-facing surface is LIVE TEXT, not just docs:** the `custom_source_bank`
  row's `guide_ref` is raised to the operator by `require_runnable_bank`, and the
  `source_bank` tooltip repeats it. Any future change to the activation path (folder
  name, CLI verb, restart behaviour) must update `nodes/story_packs/banks.json`, that
  tooltip and `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
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
- **The ledger-cleanup pass runs on EVERY bank, not just client banks** (`3d97a130`). It
  is a no-op on a complete ledger and costs no LLM call there, but two shipped-lane
  behaviours did change and are worth watching on the next live legs: (a) unsafe spoken
  language on a `content_owned_readonly` bank is now REPAIRED at the writer tail instead
  of reaching G9 untouched, so a leg that used to die at freeze may now ship a sanitized
  line; (b) a blank `meta.episode_title` is now filled at the tail instead of exploding
  later in `otr_credits_roll`. Both are the intended direction under THE LAW; neither has
  a live receipt yet.
- No code lands mid-sweep of an active qualification campaign (the 420-rung
  uniform-code-confound lesson).
- There is no standalone SFX provider layer to rebuild. Current video clips are
  silent and the terminal mux uses the frozen upstream master audio. The future
  direction in `ROADMAP.md` is to retain and mix selected video-generation audio
  as inexpensive ambience; do not revive the fast-moving provider/bed stack or
  claim that future path is already wired.
- Lean-mean has one current ordered campaign in `docs/LEAN_MEAN_CLEANUP.md`.
  The retired FRONT/TAIL and SW-1 execution model must not be revived.

## PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected.** Operator: *"park it on go forward."* Raised
after the operator observed we should have far more voices than the writer is
being offered. He was right, by a wide margin.

### The finding, measured live

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

### Why this is parked rather than done

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

### What the work is, when it is taken up

1. Enumerate every consumer of a cast row's voice fields -- casting, TTS
   dispatch, per-beat audio slicing, credits, portraits, captions, `obs_publish`
   -- and name the new owner of each field. Exactly one owner each.
2. Make the casting menu engine-aware: Bark presets when the character engine is
   Bark, reference-bank entries when it is a cloning engine. Gender and
   `commercial_clean` already exist on bank rows.
3. Replace `_assert_unique_bark_voices` with an engine-agnostic
   one-voice-per-character invariant. The rule itself is right and must survive:
   two characters sharing a voice is a correctness defect.
4. Derive `MAX_SPEAKING_CAST` from the ACTIVE engine's pool instead of a
   constant. `tests/test_cast_size_is_a_request.py` already asserts the constant
   matches the live stock, so it will report the drift rather than hide it.
5. Prove on `scifi_news_pro` (the only bank on the fable2 writer) with a cast
   larger than 10 and complete speaker-to-`char_id` equality in the ledger.

Related and already shipped: `num_characters` is now a REQUEST rather than a cap
(operator directive, all banks) -- see `tests/test_cast_size_is_a_request.py`.

---

