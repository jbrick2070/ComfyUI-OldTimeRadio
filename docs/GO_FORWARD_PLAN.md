# OTR Go-Forward Plan

> **"As long as it doesn't crash when it's not supposed to."**
> -- the operator, 2026-09-11. That is the bar, and it is the same rule as
> "obs is the success signal": a leg passed if it reached `otr/obs/` without an
> unexpected crash. What it LOOKS like is not the test.

**The one rule for this file: ONLY UNFINISHED WORK BELONGS HERE.** When work
finishes, its receipt moves to [HANDOFF_LOG](HANDOFF_LOG.md) or its own evidence
folder and the row leaves this page. Read AGENTS.md, CLAUDE.md and
[standing rulings](OTR_STANDING_RULINGS.md) first; this file does not restate them.
Measured status and commits: [session_handoff.md](../session_handoff.md).

## 0. The bar

**"As long as it doesn't crash when it's not supposed to."** (Operator, 2026-09-11.)

That is the priority filter for everything below. This is a fun experimental app and
**exactness is not the goal** -- operator, same day: *"I realize my visual pack and
story source combined is quite complex. I'm not expecting anything exact."* So:

- **Crash-class and durability-class defects are the work.** An uncaught exception, a
  live asset written where a sweeper can delete it, an identity that silently
  resolves outside its episode.
- **Aesthetic drift is NOT the work.** A Shakespearean staging of a modern dinner is
  emergent behaviour from combining two independent surfaces, not a broken contract.
  Do not reopen it. See
  [ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md).
- **AN ARC IS CODING (operator ruling 2026-09-11).** Operator: *"when I say code I
  include arc -- I consider arc coding."* So section 4 is not deferred work sitting
  behind the real work; **section 4 IS the remaining work**, and "do all the coding
  before testing" means do the arcs. Arcs cost no GPU and no money, so the only
  budget they spend is wall clock.
- **NEVER ASSUME A FONT IS INSTALLED (operator ruling 2026-09-11).** Operator:
  *"don't assume people have fonts installed. I'm open to some bad formatting as
  long as it doesn't crash."* This REVERSES the earlier no-fallback policy for
  credits (*"a point-size-less bitmap hero is unacceptable"*, Fable risk #3): that
  ruling optimised for how the card looks, and it was killing FULLY RENDERED
  episodes at the credits stage on any box without DejaVu/consola/Menlo. An ugly
  card ships; a `CreditsDataError` ships nothing. The principle generalises: a
  missing PRESENTATION resource degrades, it never refuses.
- **Prompts are hand-crafted per model and CHARACTER-BUDGETED.** Do not swashbuckle
  them. Adding conditional nuance spends budget that does not exist and reintroduces
  the per-prompt subtleties the operator has already rejected; removing words edits a
  tuned recipe. Both directions are closed.

## 1. The sequence

**ARCS -> CODING -> SHAKESPEARE -> TESTING, ABSOLUTELY LAST (operator directive
2026-09-11 -- hard, and it governs everything else in this file).**

| phase | what it means | done when |
|---|---|---|
| 1. ARCS | Every row whose design has more than one defensible answer gets its r1 BEFORE any code. An arc IS coding; it is $0 and costs a wait, not a budget. | No row in 5A/5B is waiting on a design decision. |
| 2. CODING | Build what the arcs settled, plus every row that never needed one. | Section 5 is empty of buildable work. |
| 3. SHAKESPEARE | Row 3.6, named by the operator as its own phase rather than folded into general coding. | A verbatim Shakespeare episode is buildable. |
| 4. TESTING | Freeze ONE hash, then four machines qualify it. | Section 4's coverage is owed against a real head. |

**A DEFERRED ARC IS DEFERRED CODING, and that is the mistake this table exists to
stop.** The head was frozen once at `3e692dc5` with two arcs pushed past it and 3.6
cut from the week -- which is testing before coding is complete, by the plan's own
definition. Withdrawn. Before that, `f5f40bd4` was frozen before coding finished and
was broken inside the hour by `a074f56e`. **Two premature freezes in one day: do not
cut a third until phases 1-3 are actually empty.**

**A GATE ON EVIDENCE THE WAVE PRODUCES IS NOT A VALID DEFERRAL EITHER, now that
testing is last.** Ghost Half B was held for the 4060's `kernel_source` hit-rate; the
Sci-Fi repair cap was held as "render-affecting, lands after the wave". Both reasonings
assumed the wave ran first. It does not. An item that can only be decided by live
evidence must either be decided WITHOUT it or be explicitly cut from this round -- it
may not sit in a queue waiting for a phase that comes after it.

**What genuinely still defers, and it is now a two-item list:** a row blocked on an
operator ruling, and a row deliberately cut from this round with the reason written in
it. Nothing else.

**Then, and only then: four machines qualify ONE frozen commit.** The 5080 and RunPod
are driven from the coder window; the 4060 and the Mac are driven by Cowork natively on
those boxes, from
[NATIVE_PLAN_4060_AND_MAC.md](2026-09-11-four-machine-test-wave/NATIVE_PLAN_4060_AND_MAC.md).

**After the wave, the next day begins in `otr/obs/`, not in the editor.** Count what
landed against the legs promised, read the four phone-homes, and triage any crash-class
failure FIRST. Only when that triage is empty does new work start.

## 2. Tonight -- the four-machine wave (the TEST phase)

Copy-paste prompts, one block per machine:
[PROMPTS.md](2026-09-11-four-machine-test-wave/PROMPTS.md). That file restates the
six-act fixture on purpose -- it has to be self-contained to paste.

**Freeze ONE head and qualify it.** Four machines qualify one hash. Every render-path
push stacked on top adds a suspect to the 5080's Leg A, the one leg that must pass;
and if the wave goes badly, a render-inert delta means the suspect list is one fix
plus the platform -- a single bisect, not six. Write the frozen hash into the
`WAVE HEAD:` line in PROMPTS.md before anything starts.

Every lane: pulls first and reports the HEAD it ran, writes findings to `docs/`,
**does NOT push** (the push stays with the 5080), phones home per leg, and reports
its rolled `visual_style`.

| Lane | Legs |
|---|---|
| 5080 | Leg A, the PBUG-20260911-03 scopes requalification (**the one that must pass**), verified on disk not from logs, with the consumer run at bypass true AND false. Leg B, the six-act Jeffrey/Codex fixture. |
| 4060 | One-act and three-act on `otr_4060_12b_gguf_offload` (status shipping) for the legs that must land, THEN the draft profiles as well -- the operator struck the blanket forbid on 2026-09-11 (*"you kept telling me this won't work"*), and PBUG-20260904-05 records `8gb_lite` PUBLISHING on `media_archive` and refusing only on the two banks whose prompts exceed its 2,048-token context. A 20s context refusal is a defect to report with its token numbers, not a failed leg. Scopes path on 8 GB. `kernel_source` counts. Fresh-install friction to `4060_DRILL_LOG.md`. |
| Virtual Mac | One full canonical on Apple Silicon. Then look hard at title/caption/credits TEXT in the delivered frames -- eight macOS episodes shipped with the hero title off the right edge. |
| RunPod | Second installed model family at one/three/six acts, verifying actual runtime IDs not dropdown labels. One fresh Original publication for the credits proof. **Stop the pod when done.** |

Not covered by the wave: **listening** (the operator's own ear; no agent here can
ingest audio, and reading TTS text or checking a waveform is not listening).

## 3. Open defects with an owner

| Defect | State | Next action |
|---|---|---|
| Caption burn kills an episode over its TITLE CARD | When a hero title was planned and the ffmpeg `ass` burn fails, `OTRCaptionBurn.burn` raises RuntimeError and ends the whole prompt -- at the END of a ~30-minute render, discarding the script, cast, voices, audio master and every clip over a cosmetic card. The trigger is a PLATFORM gap: `resolve_ffmpeg()` falls through to the imageio-ffmpeg build wherever there is no system ffmpeg, and a build without libass fails this and nothing else. **"That is the Mac and the container" was a FORECAST and it is mostly wrong -- measured 2026-09-11.** The 5080's resolved ffmpeg is a winget build configured `--enable-libass` with the `ass` filter present, and PBUG-20260909-04 records the Mac publishing EIGHT episodes with burned hero titles, so libass was there too. `resolve_ffmpeg` tries imageio-ffmpeg LAST (`_otr_shared/ffmpeg.py:88-94`). The real exposure is a bare container. **ARC DEFERRED PAST THE WAVE for that reason** -- one lane, driven from the coder window, and checkable in one command before its leg rather than arced blind. | **A degrade was written and REVERTED the same hour -- do not simply re-apply it.** The branch catches EVERY ValueError from `burn_captions_on_video`, and `tests/test_caption_burn_fails_closed_on_title.py` proves the cost: an unknown caption STYLE also lands there, and degrading a misconfiguration would ship untitled episodes forever in silence. The fix needs the burn's failures CLASSIFIED -- capability gap (degrade) vs misconfiguration (refuse) vs resource death (stay fatal, the operator wants OOM loud) -- and that is a design choice with more than one defensible answer, so it takes an arc before code. Also true and separate: the node's `report` output is UNWIRED in the canonical graph (node 86, `links: []`), so any degradation receipt must live in the LOG, not the report string. |
| `google_image` hard-fails where a sanctioned gap exists | A 200 OK Gemini response with no image block raises `GoogleAPIRequestShapeError`, and `dispatch_images` only routes an exception into the sanctioned-gap path when `is_model_refusal` is set -- which ONLY `Ideogram4RefusalError` does anywhere in the tree. So an empty response takes the NO FALLBACK branch and kills the render, where the identical shape from ideogram4 floors just that beat. | Small and already-precedented: set the flag so it routes into `skip_evidence_by_oid` / `STATUS_SANCTIONED_GAP`, which already gives the ledger row a real owner. NOT done tonight because `google_image` is a cloud lane and the four-machine wave is local -- it reaches no leg in it. Config-selectable in `otr_cloud_hq`, `otr_cloud_low`, `google_omni_all`, `google_omni_media`. |
| **3.6a THE VERBATIM LANE NEVER SEES ITS SOURCE** -- the headline finding, verified | `LineRequest.source_block` defaults to `""` (`_otr_line_composer.py:292`) and the line prompt only includes source text `if req.source_block:` (`:908`). There are exactly TWO `LineRequest(` construction sites in the tree: the writer's main one (`OTR_LedgerScriptWriter.py:5179`), which **never sets `source_block`**, and the cast-coverage gap-repair (`_otr_cast_coverage_repair.py:201-210`), which does. So on the ORDINARY dialogue path no source text reaches the model at all -- it writes Shakespeare's lines from intent and mood. That is why a "verbatim" episode invents its dialogue. | This is a CORRECTNESS defect, not story quality -- CLAUDE.md's 2026-08-04 carve-out names source-contradiction explicitly. It is also the thing 3.6 is actually about, and it is far narrower than the archive's five steps. **Do NOT reach for `compose_exchange` because it has a working `source_block` parameter:** it is still an LLM being asked to CARRY the words, not a deterministic splice, so wiring it would not satisfy the 2026-08-03 keystone and could reproduce the very defect it looks like it fixes. |
| **3.6b `_otr_passage_selector.py` IS BUILT AND UNWIRED** | 14 KB, three commits all dated 2026-08-03 -- the same day as the keystone ruling -- with its own `tests/test_passage_selector.py`. `select_passage()` is a deterministic, seeded, verbatim window selector over Folger-format speeches that is beat-topology aware (spans a long speech across consecutive beats against `BEAT_WORD_HARD_MAX`) and `cast_ceiling`-aware, so it solves the archive's Step 2 ("cast must follow from the cut") as a side effect of its own search. **Zero production callers** -- the only two hits outside the module are COMMENTS in `_otr_episode_budget.py`. | **This is the third time this lane has built to spec and never wired:** `SourceOverview` (removed 2026-09-05 for zero consumers), `select_grounding` (zero callers, tests only), now `select_passage`. Per the standing rule an orphan is ripped 100% with a grep receipt or wired in deliberately -- and this one should be WIRED, because it is most of what 3.6 needs. Verify its contract against 3.6a's seam before trusting it; built-and-never-run is not the same as working. |
| Cast-count vocabulary disagrees across three surfaces | `_otr_casting.py` stamps the already-CLAMPED value as `num_characters_request`; `OTR_LedgerScriptWriter.py` builds an `EpisodeBudget` from the UNCLAMPED request; `_otr_episode_budget.py` labels that `cast_size`. For a request of 8 those surfaces say six and eight. Pre-existing, surfaced by the 2026-09-11 contrarian pass on the cast-count fix. | Not crash-class, and deliberately NOT folded into that fix -- keeping it separate is what makes it possible to tell whether the boundary correction worked. **Do not naively retarget the replay field:** `cast_lock.py:629` consumes `num_characters_request` to replay casting, so changing its meaning breaks replay. Decide which surface owns "the count that happened" and document the legacy replay meaning. |
| Packed set models FLAGGED | `scripts/otr_registry_scan_oracle.py` reports **4 findings in the shipped set**, all pre-existing and none from today: `_otr_shared/ffmpeg.py:249`, `motion_common.py:544` and `:630` (`$subprocess_direct`), and `prestartup_script.py:150` (`$env_read/mod`). A clean scan is what auto-promotes a version to Active; any finding leaves it Flagged, and a Flagged version is the "Cannot resolve install target" the Manager reports. | Measured, not predicted -- the oracle models the scanner, so the real status still comes from the registry. Decide per finding before the next `pyproject.toml` bump: the three `subprocess.run` calls are real platform probes (`vm_stat`, `sysctl`) and an ffmpeg spawn, so they need a judgement, not a respell. Nothing here blocks the wave; it blocks a publish. |
| OpenRouter catalog cache PATH | The "can never warm it" half is closed -- `scripts/otr_openrouter_refresh.py` is allow-listed and ships from the next `pyproject.toml` bump. What remains is that the cache still lives at `_otr_openrouter_backend.py:792`, inside the installed pack, so a registry UPDATE wipes it and the user re-runs the refresh. | **A three-way tier fork, which is why it did not ship with the allow-list.** `otr_shared_cache_dir()` matches the `source_banks` precedent and its "never the only copy" contract (a catalog IS regenerable); `otr_state_dir()` is where the billing ledger went; the external models root is the GGUF precedent. Plus a migration call on an existing warm cache. Per the 2026-08-17 amendment that is an arc. **DEFERRED PAST THE WAVE, deliberately:** it is not crash-class -- the cost is one re-run of a documented command after a registry update -- and it reaches no leg in this wave, since every lane runs local models. The arc costs $0 and a wait, so it runs when the wave is not waiting on it. |

## 4. Qualification coverage still owed

Every qualifying run loads `workflows/otr_canonical.json` through
`scripts/otr_canonical_api_run.py` -- no `--workflow` override, no replay substitute,
no `partial_execution_targets`. Assets go straight to canonical episode paths; final
publication must exist in `otr/obs`.

| Coverage | Required evidence |
|---|---|
| Repair requalification | Full canonical on the pushed correction: scopes path, supplied facts, actual voices, credits, pictures, durable OBS file. |
| Six-act repeatability | Three fresh full canonical runs. Record observed loads/reuse; claim clean boot and resident reuse only where the runtime actually supports it. |
| Source and cast variety | One-act monologue, three-act ensemble, detailed six-act. Requested vs actual cast, supplied/neutral byline, breaks on and off. May overlap repeatability only where the recorded fixture genuinely covers both. |
| Second model family | One/three/six acts on a compatible installed family. Verify actual runtime IDs. |
| Original credits | One fresh Original publication: observed creative model agrees across wire, saved ledger and rendered credit. Only live proof is pending. |
| Listening | Opening/middle/ending on at least two publications including a six-act. Operator's ear only. |
| 2.2 Ghost CUDA | Live five-act forced-Ghost CUDA publication with stored prompt/admission/reuse inspection. No new Ghost schema field. Needs a CUDA host. |
| 5.7 chunked music | Full canonical on `otr_8gb_fastwan` (**that is the real profile id -- "fastwan_8gb" appears nowhere on disk, and the profile is status `draft`**) with long opening/closing cues, to prove the existing chunked-music repair. One leg, not a design row. |
| Inherited regression debt | **Load-bearing, not a footnote** -- it is what makes any "suite is green" claim mean something. Re-ground 51 OTR and 10 Bible failures against baseline IDs **and normalized payloads**. The Bible strict metadata validator separately has 149 unchanged issues. Never quarantine silently; never call the suite all-green. |

**The six-act fixture:** Jeffrey and Codex getting closer to release across one
continuous evening in Jeffrey's workspace. Jeffrey is the physically present adult;
Codex is a named AI dramatic speaker heard through the computer speakers, with only
an on-screen interface -- no embodied human, age or gender invented. Only those two
dramatic voices; reports from other machines are displayed text or discussed by them,
never new speaking characters. The house announcer is production framing, outside
dramatic cast, turns and ending. Request four characters to exercise the flexible
actual-two cast. Six acts, normal breaks; breaks are framing, not new times or places.

For every attempt record source fields, code/canonical/graph hashes, actual
model/quantization/profile, prompt ID, elapsed time, memory and loads, all repair
attempts, requested vs actual acts and cast, ledger seals and final paths. Preserve
terminal failure evidence before asserting anything. Keep the full denominator.

## 5. The remaining work

Every row was re-grounded against current code and put to a file-grounded reviewer
before landing here. Arc receipts, if a row's history is wanted:
`kibitz-runs/2026-09-11-arc-*/r1/` and `docs/2026-09-11-arcs/*/`.

### 5A. CODE IT -- the fork is settled

| Row | What to write |
|---|---|
| 3.1 Ghost Half-B, LLM tier | The DETERMINISTIC tier is in: the beat's own text now ranks `meta.key_objects`, with a negation guard. Its measured ceiling is 26.3% of beats, so what remains is the tier the operator actually chose -- extend the batched Ghost author so a beat naming no listed object still gets a physical-artifact subject. Rules: photographable thing only, never an abstraction; beat reference RANKS, never SOURCES; a noun named only inside a negation is not an artifact. Not a new retry loop and not a gate. **Gated on two things, deliberately.** It is a design choice with more than one defensible answer, so it takes a full arc before code -- and it changes the batched Ghost author's PROMPT, which the operator has ruled on twice (*"be very careful about swashbuckling updating prompts"*, *"we are constrained by characters"*). And the 4060's legs are the first live exercise of the deterministic tier, so their `kernel_source` counts are the first real hit-rate measurement -- the number that should shape the prompt. Build it after the wave reports, not before. |
| 3.6 Shakespeare -- **PHASE 3, its own phase** | The operator named this directly on 2026-09-11 (*"arcs - coding - Shakespeare and testing absolutely last"*), so it is in scope, not cut. **The old "wiring, not design" label was wrong** -- it came from a plan rewrite rather than from the code. The 2026-08-03 arc settled the KEYSTONE (compile source speech deterministically, never generate it) and `_otr_source_document.py` provides the artifact, but `select_grounding` has zero production callers and GO_FORWARD_ARCHIVE:8777 lists five steps whose first is marked BLOCKS EVERYTHING: loosening a hard-raise cast-count invariant and retiring `cast_hints` through a schema migration. | **r1 IN FLIGHT 2026-09-11**, asking the two questions that decide the size: is step 1 still blocking against the CURRENT files, and is there a smallest useful slice that delivers a real verbatim episode without the full five steps and without a schema migration? The cast-count hard-raise is itself suspect under "only an out of memory should fail", and relaxing it may unblock step 1 cheaply. Size it from the arc, then build. Not a quick item, and no longer a reason to skip it. |
| 3.7 meta ownership | Fork NONE. Fix the stale comments; the ownership split is already correct in code. |
| 3.2 composer | Fork NONE, no crash risk. Fix the stale docs. |
| 2.4 audit tail, model-root | The four-owner model-root merge, or nothing. Do NOT rip `comfy_models_dir()` / `resolve_hf_model_path()` on a caller count: they are a dead CHAIN, the archive PARKED the convention, and a FOURTH spelling lives in `flux2_klein.py:209-215`. Refusal reasoning recorded in `afe3bfed`. |

### 5B. NEEDS A MEASUREMENT OR A DECISION, not another round

| Row | What it actually needs |
|---|---|
| 2.4 routing/canvas | The fork survives, but the 193-frame ceiling **cannot be asserted as production-proven** -- it is a lab-warm isolation number, and this lane has a live receipt of production peaks exceeding lab peaks. Needs a canonical-path qualification leg first. Then a TWO-file diff (`frame_contract.PLANNING_CAP_ENGINES` **and** `otr_16gb_ltx_audio_in.json` `video.max_render_frames`) -- changing one alone narrows nothing. **Not render-inert**: `assert_coverage_plans` refuses any ltx_audio_in ledger planned before the change and rendered after, so in-flight episodes need replanning at cutover. |
| 3.5 per-beat reload | **Direction undercut by prior art.** Moving this same encoder off-GPU was already tried and REVERTED on live-measured evidence that it did not move the peak (PBUG-20260616-01 / BUG-07.17). The generic scope hook also closes on engine-CHANGE, not every beat, so "even consecutive same-engine beats start cold" was wrong. Per-lane encoder device defaults differ (`ltx_video` GPU vs `ltx_av` CPU). Before any caching work, decide whether it is worth doing at all given the revert. Not OOM-proven -- wall-clock cost. |
| 2.4 voice/credits | **The instrument does not fit the defect.** `high_band_edge_ratio` detects edge squeal; PBUG-20260902-03 documents a SUSTAINED TONE, and an independent synthetic reproduction of the exact documented frequencies scored ~0 against the real function. Closing this means designing and empirically qualifying a NEW whole-clip speech-shape scorer that does not exist anywhere in the tree. Real work, for a non-crash defect -- weigh against the bar before starting. |
| 2.4 source | Blocked on the operator's digest ruling (section 6). |
| 3.4 clean install | r1 verdict was **yes-with-fixes** -- the only one. Keep the existing download scope; the early-tool-check proposal needs narrowing. Low priority, non-crash. |

## 6. Blocked on the operator -- each unblocks with one word

2.4 source digest ruling (HTML block joins) · 4 / 4B registry review note for posting (**guard: revalidate the old
discriminator/bisection hypothesis before ANY destructive strip-down**) · 5.1-5.2 release after physical 8 GB proof ·
6 unruled product choices and Bible fan-out candidates.

**WITHDRAWN 2026-09-11, same day, before it cost a ruling.** A question was queued
here about whether a run whose obs COPY fails on I/O may still report success. It rested
on a premise the operator corrected and the code disproves: *"otr/obs is ALWAYS the
local output folder -- don't confuse my testing setup with what ships. Everyone this
ships to will see the otr subdirectory in their output folder; our workflow should
create the folders within their output."*
That is exactly what the code does. `otr_obs_dir()` resolves `<output>/otr/obs` unless
`OTR_OBS_DIR` is pinned, and that pin exists ONLY for this box's two-tree split (the
headless server renders into the ComfyUI-Installs tree while the operator watches
`Documents\ComfyUI`). `comfy_output_dir()` tier 2 is
`folder_paths.get_output_directory()` -- ComfyUI's own API, so a shipping user's OWN
output folder. `_otr_paths` resolves and never creates; the mux calls
`makedirs(..., exist_ok=True)` for the episode dir (`:1358`) and the obs dir (`:1388`).
So there is no network share in this path for anyone. The "unmounted drive" trigger was
this box's test rig mistaken for the product. What is left is a LOCAL `os.makedirs`
failing -- a full disk or a permissions fault -- which is genuine resource death and
SHOULD stay fatal under the operator's own rule. No ruling needed; the row is closed.

## 7. Hardware and publication state

- **Mac, 4060, RunPod: RELEASED** for tonight's wave. Their own proven routes and
  owners remain authoritative. Findings to `docs/`; they do not push.
- **Registry versions, tags, public posts and release promotions** need the operator's
  approval process. Pushes to `v2.0-alpha` are required and are not a release. Never
  send an old review draft.
- **2.5 clone-reference preflight is a conditional tombstone.** Attempted and reverted
  2026-09-04 after a grounded review killed it on three counts, all still true. Not
  scheduled. Do not revive without new evidence.

## 8. Constraints specific to this plan

Only the ones not already in CLAUDE.md or the standing rulings.

- Full listener source, no RSS. Cast count is flexible and records requested vs
  actual; the house announcer is excluded from dramatic cast.
- **WE DO NOT CHASE ACT COUNT, exactly as we do not chase word count (operator
  ruling 2026-09-11).** Operator: *"acts we ask for, we don't chase -- just like
  words. The LLM does its own thing. I don't have enough horsepower or tokens to
  chase it. I'd need to build a better act structure and I don't want to now."*
  The requested act count is a REQUEST to the model, not a gate on its output.
  **A3 -- adding a selected-act vs parsed-scene comparison to Sci-Fi's markup
  ladder -- is CLOSED by this ruling and was removed from the design rows above.**
  Do not add a scene-count check, a tolerance band, a repair rung keyed on act
  count, or any rejection derived from act count. A better act STRUCTURE is a
  future project the operator has explicitly declined to start; it is not a
  validation problem and must not be approached as one.
- Model checking returns actually-applied rewrites under a fixed total attempt budget
  including repairs. No separate chunker, report-only checker, recursive loop, or
  fourth/fifth round.
- An exhausted optional correction still yields a usable ledger with unresolved
  evidence recorded. Only existing usable-ledger requirements may refuse after
  applicable repair. Provider, storage, cancellation and real OOM failures stay
  truthful. **No predictive word, duration, story-length or cast-size rejection.**
- Supplied byline or neutral listener attribution for My Story; observed creative
  model for Original; genuine authors preserved in adaptation banks.
- No replay-system, migration or re-render project. Saved input means fresh
  generation. Do not hide test publications from OBS.

## 9. Parked -- preserved, not work

`5.3`/`5.6` unqualified installed family and GGUF opt-in combinations ·
`5.4`-`5.5` H3 policy receipts and the image-mode refusal question ·
`5.8`-`5.10` cfg promotion comparisons, failed-proof re-derivations and clean-room
legs · `5.11`-`5.12`/AMD scoped pod and platform acceptance after access, including
`3.5` dtype capability admission on ROCm · `3.5` explicit cloud billing/opt-in:
`config/profiles/cpu_floor.json` still routes all three image roles to
`google_image` with no ceiling (`otr_mac_mps.json` was re-pinned to `sd15`, so
tonight's Mac lane is not exposed) ·
`7` operator-parked casting/adaptation/AnimateDiff ideas, OTR-Lite after v2,
word_razzle placement · `8` live client-bank proof and remaining
cleanup-tail/zero-frame/RenderError cases · **Release runway**: LEAN_MEAN_CLEANUP,
representative platform acceptance, clean install, product documentation,
operator-authorized release.

Detail for every row on this page lives in
[the archive](GO_FORWARD_ARCHIVE.md#2026-09-11----pre-opus-queue-preserved-verbatim).
Historical status, version and model claims there are **not current instructions** --
re-ground against current code before coding, and remove a row from this page once
closure is established. Do not revive the deleted ROADMAP.md or start a parallel queue.
