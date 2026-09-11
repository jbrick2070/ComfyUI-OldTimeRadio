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
  card ships; a `CreditsDataError` ships nothing. **Coded 2026-09-11** --
  `_load_font` degrades to PIL's embedded font and logs the remedy. Generalise the
  principle: a missing PRESENTATION resource degrades, it never refuses.
- **Prompts are hand-crafted per model and CHARACTER-BUDGETED.** Do not swashbuckle
  them. Adding conditional nuance spends budget that does not exist and reintroduces
  the per-prompt subtleties the operator has already rejected; removing words edits a
  tuned recipe. Both directions are closed.

## 1. The sequence -- ARC, then CODE, then TEST ON ALL FOUR

Operator, 2026-09-11: *"arcs -- rebase go-forward to strategically optimise: arc,
code, test on all 4."* That is the spine of this file, and an arc IS coding.

**But the honest optimisation is not a straight line, and pretending otherwise costs
a day.** Tonight's wave does not depend on any arc below -- it qualifies the scopes
fix and gathers evidence -- and it is itself the EVIDENCE SOURCE several arcs need.
Serialising everything behind "all arcs done" would idle four machines tonight and
then arc blind tomorrow. So:

| Phase | What | Why here |
|---|---|---|
| **A. ARC now** | 3.3 orphan occupancy, 3.5 per-beat reload, Sci-Fi cap value, runtime pack writes (all Tier 1), then Tier 2 rows that need no live evidence. | Crash/OOM class first, per the bar. Costs no GPU and no money, so it runs alongside the wave without competing for anything. |
| **B. CODE what each arc converges on** | Implement, one CLI review per change, focused + full suite compared against baseline IDs **and normalized payloads**, then push. | An arc that never becomes a diff bought nothing. |
| **C. TEST on all four** | The wave in section 2. **Freeze one hash first.** | Four machines qualify ONE commit. Every render-path push after the freeze adds a suspect to the one leg that must pass. |
| **D. The evidence-fed arcs** | 3.8 fonts (needs the Mac's text leg), 3.4 clean-install (needs the 4060 and RunPod pull steps), and the visual-continuity A/B (needs several rolled styles side by side). | Arcing these BEFORE the wave re-derives what the wave will hand you for free. |

**The one ordering rule that matters:** anything render-affecting either lands before
the freeze or waits for phase D. Nothing render-affecting goes in between.

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
| 4060 | One-act and three-act on `otr_4060_12b_gguf_offload` (status shipping -- **NOT** `8gb_lite` or `otr_4060_floor`, both draft; `8gb_lite` refuses in 20s on two of three banks, PBUG-20260904-05). Scopes path on 8 GB. Fresh-install friction to `4060_DRILL_LOG.md`. |
| Virtual Mac | One full canonical on Apple Silicon. Then look hard at title/caption/credits TEXT in the delivered frames -- eight macOS episodes shipped with the hero title off the right edge. |
| RunPod | Second installed model family at one/three/six acts, verifying actual runtime IDs not dropdown labels. One fresh Original publication for the credits proof. **Stop the pod when done.** |

Not covered by the wave: **listening** (the operator's own ear; no agent here can
ingest audio, and reading TTS text or checking a waveform is not listening).

## 3. Open defects with an owner

| Defect | State | Next action |
|---|---|---|
| PBUG-20260911-03 scopes path | Code fixed, offline-qualified, pushed. NOT closed. | 5080 Leg A. Nothing left to code. |
| Sci-Fi repair-turn cap VALUE | Half open. The CATCH shipped 2026-09-11 -- an overflow on the repair turn no longer kills the episode. The remaining half is that `_draft_fits_repair_turn` still predicts fit against a flat `HARD_VRAM_CONTEXT_LIMIT` rather than the transport's real window. | RENDER-AFFECTING: the cap decides repair-vs-cold-regeneration, so it changes which script comes back. Not before a wave. It is also not one fix -- local, GGUF-native and OpenRouter each resolve capacity differently -- so it needs an arc, not a constant swap. |
| OpenRouter catalog cache | Half of this row SHIPPED (`d912188f`): the billing ledger now lives under `otr_state_dir()` with a copy-forward. What remains is the CATALOG cache at `_otr_openrouter_backend.py:792`, still inside the pack. **And it is worse than "gets wiped": a registry-installed pack can NEVER WARM IT.** `refresh_catalog_cache` has zero callers in `nodes/` -- nothing self-warms -- and its only caller, `scripts/otr_openrouter_refresh.py`, is stripped by `.comfyignore:94` (`scripts/*`) and never allow-listed back. So every registry user with OpenRouter enabled silently runs at `DEFAULT_CONTEXT_WINDOW = 8192` instead of the model's real window, and the empty-cache sentinel tells them to "run refresh_catalog_cache", which they have no way to do. Not crash-class -- it truncates output. Cheap: allow-list the refresh script, which is render-inert (`.comfyignore` changes what SHIPS, not what renders, and only `pyproject.toml` fires a publish). |

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

## 5. The remaining work, after the arcs (2026-09-11)

Eight arcs ran r1 with one reviewer each (codex), and every verdict was then verified
against the real files before anything was folded in. **41 of 42 panel claims held.**
Anchors had been built from adversarially-verified grounding and were STILL wrong in
seven of eight cases -- establishing the facts and reasoning correctly from them are
different skills, and that gap is what the arcs caught.

**r2 was judged unnecessary on all eight.** Each row now turns on a decision, a
measurement, or a concrete diff -- not on a design argument. Running r2 would be
ceremony. Receipts: `kibitz-runs/2026-09-11-arc-*/r1/`, anchors in
`docs/2026-09-11-arcs/*/`.

### 5A. CODE IT -- fork settled or collapsed

| Row | What to write |
|---|---|
| 3.1 Ghost Half-B | Fork already ruled by the operator (`OTR_STANDING_RULINGS.md:759`, 2026-09-03): extend with the beat's own dialogue. Implement the existing ruling. |
| 3.6 Shakespeare | Keystone design settled by a full kibitz arc on 2026-08-03: compile source speech deterministically, never generate it. `_otr_source_document.py` already provides the artifact. Wiring, not design. |
| 3.7 meta ownership | Fork NONE. Fix the stale comments; the ownership split is already correct in code. |
| Sci-Fi cap VALUE | A correct per-provider value already exists in `cache_entry["context_cap"]`. Thread it instead of the flat constant. **RENDER-AFFECTING** -- lands after a wave, never before a freeze. |
| 3.2 composer | Fork NONE, no crash risk. Fix the stale docs. |
| 2.4 audit tail, model-root | **DO NOT RIP -- the panel was wrong here and the check is recorded so nobody re-derives it.** `resolve_hf_model_path()` is genuinely uncalled, but `comfy_models_dir()` IS called, by `resolve_hf_model_path` itself -- a dead CHAIN, not two free symbols. More importantly the archive rules it PARKED: *"the third convention, `_otr_paths.comfy_models_dir()` / `OTR_MODELS_DIR`, stays parked (cursor r3: do not open a third env in this diff)"*, and a FOURTH spelling exists in `_otr_image_engines/flux2_klein.py:209-215` -- *"when the merge happens it has four owners to retire, not three."* Ripping one of four owners of an unfinished consolidation is exactly the half-removal the operator's *"an orphan is ripped 100% or wired back in"* directive forbids. This row is the four-owner merge, or it is nothing. |

### 5B. NEEDS A MEASUREMENT OR A DECISION, not another round

| Row | What it actually needs |
|---|---|
| 2.4 routing/canvas | The fork survives, but the 193-frame ceiling **cannot be asserted as production-proven** -- it is a lab-warm isolation number, and this lane has a live receipt of production peaks exceeding lab peaks. Needs a canonical-path qualification leg first. Then a TWO-file diff (`frame_contract.PLANNING_CAP_ENGINES` **and** `otr_16gb_ltx_audio_in.json` `video.max_render_frames`) -- changing one alone narrows nothing. **Not render-inert**: `assert_coverage_plans` refuses any ltx_audio_in ledger planned before the change and rendered after, so in-flight episodes need replanning at cutover. |
| 3.5 per-beat reload | **Direction undercut by prior art.** Moving this same encoder off-GPU was already tried and REVERTED on live-measured evidence that it did not move the peak (PBUG-20260616-01 / BUG-07.17). The generic scope hook also closes on engine-CHANGE, not every beat, so "even consecutive same-engine beats start cold" was wrong. Per-lane encoder device defaults differ (`ltx_video` GPU vs `ltx_av` CPU). Before any caching work, decide whether it is worth doing at all given the revert. Not OOM-proven -- wall-clock cost. |
| 2.4 voice/credits | **The instrument does not fit the defect.** `high_band_edge_ratio` detects edge squeal; PBUG-20260902-03 documents a SUSTAINED TONE, and an independent synthetic reproduction of the exact documented frequencies scored ~0 against the real function. Closing this means designing and empirically qualifying a NEW whole-clip speech-shape scorer that does not exist anywhere in the tree. Real work, for a non-crash defect -- weigh against the bar before starting. |
| 2.4 source | Blocked on the operator's digest ruling (section 6). |
| 3.4 clean install | r1 verdict was **yes-with-fixes** -- the only one. Keep the existing download scope; the early-tool-check proposal needs narrowing. Low priority, non-crash. |

### 5C. CLOSED, DISSOLVED, OR NOT CODE

| Row | Disposition |
|---|---|
| 3.8 fonts | **DISSOLVED by the arc.** The shared-resolver side does not survive: the four resolvers have genuinely different jobs (measured-monospace, libass-declared-family, deliberately-refusing-branded, proportional-unmeasured), and a unified Python table would not close the real gap because captions and titles are drawn by libass, which never sees the Python side. Removed. It did surface one real operator decision -- see section 6. |
| A2 follow-up | **CLOSED.** Both sub-items already deliberate and tested; the row text was stale. |
| 2.2 Ghost CUDA | Live five-act forced-Ghost CUDA publication. Needs a CUDA host. No code, no arc. |

## 6. Blocked on the operator -- each unblocks with one word

2.4 source digest ruling (HTML block joins) · 4 / 4B registry review note for posting (**guard: revalidate the old
discriminator/bisection hypothesis before ANY destructive strip-down**) · 5.1-5.2 release after physical 8 GB proof ·
6 unruled product choices and Bible fan-out candidates.

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
