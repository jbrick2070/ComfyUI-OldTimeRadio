# OTR Go-Forward Plan

This is the main and sole executable work queue. Only unfinished work belongs
here. Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md)
before acting. Current measured status, commits and test receipts are in
[session_handoff.md](../session_handoff.md) and [HANDOFF_LOG](HANDOFF_LOG.md).
When work finishes, move its receipt to the log/archive and remove it here.

## Next owner and sequencing

Opus takes the next coding sprints when Jeffrey starts that window. One
production coder at a time, branch v2.0-alpha. No build or GPU run is active.

**CODE ALL DAY, TEST TONIGHT ON FOUR MACHINES (operator directive 2026-09-11).**
Operator: *"I'd rather spend all day coding and test tonight rather than code and
test code and test, so I can have all 4 machines -- 4060, 5080, RunPod and virtual
Mac -- at once."* So coding chunks BATCH; they do not alternate with legs. The Mac,
the 4060 and RunPod are RELEASED for that evening wave (the earlier hold is lifted
by the same directive), and their copy-paste prompts live in
[the four-machine wave](2026-09-11-four-machine-test-wave/PROMPTS.md). Those lanes
write findings to `docs/` and DO NOT PUSH -- the push stays with this box so four
machines cannot collide on one branch -- and they phone home per leg rather than
streaming. Work the coding scopes below before starting that wave. Review
genuine design choices through R1 arc, R2 implementation, R3 wiring and R4
convergence; a deterministic conformance fix needs the finished-diff review
specified in CLAUDE.md. Obtain actual Sonnet QA after any production revision,
ground its findings, and record the reviewers who really ran. Substitute an
unavailable reviewer under the standing rule; do not fabricate consensus.

## Sprint 1 -- CODED 2026-09-11; only the live receipt remains

PBUG-20260911-03 is fixed in code and offline-qualified; it is NOT closed, because
no render has exercised it yet. The remaining work is one line in the qualification
table below ("Repair requalification") and the 5080's Leg A in the four-machine
wave -- there is no coding left here.

What shipped: the scopes MP4 now resolves its episode identity at entry and writes
through `otr_composited_dir(manifest episode_id)`; the shared-scratch selection and
the ambient system-temp fallback are both gone. Two further defects were found and
fixed in the same chunk -- `_validate_episode_id` accepted a dots-only id that
collapsed to a directory OUTSIDE any episode, and both real-producer tests ran with
an unpinned output root, so after the re-homing they would have minted phantom
episodes in the live production tree on every suite run. `key` also seeds the
idle-scope RNG, so old and new derivations were compared across 18 inputs to prove
ZERO seed drift; rendered pixels are unchanged. Full detail, reviewers and receipts:
the PBUG-20260911-03 entry in [PROD_BUG_LOG](PROD_BUG_LOG.md), Bible rules 01.05 and
01.06.

## Sprint 2 -- resolve the observed visual continuity defect at its owner

Use canonical09's stored prompts, image receipts and 16 dramatic stills in
[the evidence folder](2026-09-11-my-story-5080-qualification/pairlock_09_receipt.md).
Trace why a continuous present-day dinner alternates period/modern clothing
and seating/room details. The roll legitimately selected
shakespeare_stage_realism; do not misdiagnose another source bank as leaking.
Check actual pixels before claiming a face, age, presence or setting defect.
Close any unsupported claim with evidence instead of engineering around it.

This is continuity/source correctness, not better-prose work. Reuse the existing
scene/portrait/style owners and bounded checker+rewriter. Decide where stable
scene facts and visual treatment meet; complete R1-R4 before any new ownership,
schema or wiring design. Do not solve it with forbidden-word lists, a new
report-only checker, a separate chunker, added retry loops, or a late subjective
publication veto. Preserve flexible source cast and off-camera framing.

Acceptance: regression evidence discriminates the actual cause; a fresh full
canonical run retains source people/current-time action and stable scene facts.
An applied prompt correction proves application, not correct pixels. Inspect
the resulting images. Do not keep rerolling and hide failed attempts.

## Coding batch before the evening wave -- re-grounded 2026-09-11

Every backlog row below was re-read against the CURRENT tree and then adversarially
verified; the verification REFUTED most of the first pass's optimism, so this list
is deliberately shorter than the backlog table suggests. Sizes are S/M/L.

**Needs its design arc FIRST (the arc is $0, needs no GPU, and is the honest way to
fill a coding day):** 3.8 shared font-family/file resolution (M, and the only row
whose CODE_NOW classification survived verification outright); 3.2 shared
silent-video composer face/crux (L, touches eleven lanes on both machines, so
section 0B applies); 3.3 orphan generation occupancy (L, several narrow fixes each
surfaced a new race -- that is the two-strikes signal); 3.1 Ghost v3 Half-B (L,
restart the arc, r2 never converged); 3.4 clean-install manifest/fetch (L, and its
"blocked on 1.1" clause is stale -- there is no 1.1 row any more).

**Mechanical, no arc needed:** the confirmed sub-items of 3.5 (redirect the runtime
pack cache roots through `_otr_paths`), A2 follow-up (thread the creative slot's
resolved capacity into Sci-Fi's repair-turn fit, which still budgets against a flat
VRAM constant), and the file-scoped half of the 2.4 audit tail.

**A3 needs one short review before code, not a full arc:** exact-match versus a
tolerance band for the selected-act/scene comparison. Selected acts are binding, so
this is not a word-count or duration gate and must route into the EXISTING ladder --
never a new standalone checker.

**Verified already closed; removed from the backlog table below:** image-phase route
ownership (route_freeze.py is the single authority and all four call sites use it)
and long hero-title containment (PBUG-20260911-01, fixed today).

## Qualification after coding and QA

Every qualifying run loads workflows/otr_canonical.json through
scripts/otr_canonical_api_run.py, with no --workflow override, replay substitute
or partial_execution_targets. Configure installed models using the approved
runtime/profile inputs and save the dumped API graph. All assets go straight to
their canonical episode paths; final publication must exist in otr/obs.

| Remaining coverage | Required evidence |
|---|---|
| Repair requalification | Full canonical episode on the pushed correction, with scopes path, supplied facts, actual voices, credits, pictures and durable OBS file checked. |
| Six-act Jeffrey/Codex repeatability | Three fresh full canonical runs. Include clean boot and resident reuse only where the actual runtime supports it; record observed loads/reuse. |
| Source and cast variety | One-act monologue, three-act ensemble and detailed six-act source; requested/actual cast differences, supplied/neutral byline, breaks on/off. These may overlap repeatability runs only when the recorded fixture genuinely covers both. |
| Second installed model family | One/three/six acts with a compatible locally installed family. Verify actual runtime IDs, not dropdown labels alone. |
| Original credits | One fresh Original full publication: observed creative model agrees across wire, saved ledger and rendered credit. Only live proof is pending. |
| Listening | Opening/middle/ending audition on at least two publications, including six acts. This session cannot ingest audio; do not label waveform checks or reading TTS text as listening. |
| Inherited regression debt | Re-ground the 51 OTR and ten Bible failures against recorded baseline IDs AND normalized payloads. The Bible strict metadata validator also has 149 unchanged issues; its baseline/candidate logs are in the handoff receipt. Fix genuine open causes in separate qualified chunks; never silently quarantine them or call the suite all-green. |

The next six-act fixture is Jeffrey and Codex getting closer to release during
one continuous evening in Jeffrey's workspace. Jeffrey is the physically present
adult; Codex is a named AI dramatic speaker heard through the computer speakers,
with only an on-screen interface, no embodied human, age or gender invented.
Only these two dramatic voices. Reports from other machines are displayed text
or discussed by them, not new speaking characters. The standard house announcer
is production framing, outside dramatic cast, turns and ending. Request four
characters to exercise the flexible actual-two cast; select six acts and normal
act breaks. Breaks are broadcast framing, not new times or locations.

Story arc: check the real canonical; reproduce a lost supplied fact; repair
through existing owners; use a bounded checker+rewriter; compare the recorded
reports; finish together appreciating progress while honestly leaving remaining
release checks open. This fictional plot does not guarantee an actual model
error/repair in the test. Record actual applied corrections, no-ops and failures.
No test has been queued for this fixture.

For every attempt record source fields, code/canonical/graph hashes, actual
model/quantization/profile settings, prompt ID, elapsed time, memory and loads,
all repair attempts, requested/actual acts/cast, ledger seals and final paths.
Preserve terminal failure evidence before assertions. Diagnose a live failure
before further generation. Keep the full denominator and do not edit production
while a run is active. Use shipped watchdog/reset procedures; never blanket-kill
Python processes. Do not change story controls merely to meet a time target.

## Held hardware and publication work

- Mac and 4060: RELEASED 2026-09-11 for the evening four-machine wave, with
  prompts in [the wave doc](2026-09-11-four-machine-test-wave/PROMPTS.md). Their own
  proven routes and owners remain authoritative, they write findings to `docs/`, and
  they DO NOT PUSH -- the push stays with the 5080.
- RunPod: authorized by the operator 2026-09-11 for the evening wave ("this 5080
  is yours, you can spin RunPod"); it carries the second-model-family and Original-
  credits legs. Stop the pod when its legs finish. Pull the qualified repository/image before testing, record the actual
  revision, use the real canonical, and stop rented compute afterward. No rental
  is active. Historical pod results are not current access or qualification.
- Registry versions, public review messages, tags and release promotions require
  the operator's existing approval process. Pushes to v2.0-alpha are required;
  they are not a release or registry publish. Never send old review drafts.

## Constraints on the remaining sprints

- Full listener source, no RSS. Selected acts are binding. Cast count is flexible
  and records requested/actual values; the house announcer is excluded.
- Use existing author/correction/ledger/freeze owners. Model checking returns
  actual applied rewrites with a fixed total attempt budget including repairs.
  No separate chunker, report-only checker, recursive loop or fourth/fifth round.
- Exhausted optional correction retains a usable ledger with unresolved evidence.
  Only existing usable-ledger requirements may refuse after applicable repair;
  provider, storage, cancellation and real OOM failures stay truthful. No
  predictive word, duration, story-length or cast-size rejection.
- Supplied byline or neutral listener attribution for My Story; observed creative
  model attribution for Original; genuine authors preserved in adaptation banks.
- No replay-system/migration/re-render project. Saved input means fresh generation.
  Do not hide test publications from OBS. No better-prose campaign.
- Windows venv, UTF-8, pytest -q -p no:cacheprovider. Code chunks need focused/full
  OTR tests and Bug Bible, compared against the recorded inherited baseline.
  Validate canonical JSON roundtrip, live widgets/input names and link integrity.
  Commit and immediately push each qualified chunk; verify HEAD equals origin,
  nonempty/UTF-8/no-BOM files and touched Python AST. No production code in this
  documentation handoff.

## Later open work -- stable IDs, not the next sprint

This index preserves the backlog under this plan. The detailed historical scope
and all rulings were moved verbatim to
[the pre-Opus archive](GO_FORWARD_ARCHIVE.md#2026-09-11----pre-opus-queue-preserved-verbatim).
Those historical status/version/model claims are not current instructions.
Re-ground a retained candidate against current code and receipts before coding;
remove it here when closure is established. Do not revive the deleted ROADMAP.md
or create a parallel executable queue.

| Stable row | Remaining action / scope |
|---|---|
| 2.2 | After My Story: five-act forced-Ghost CUDA canonical publication, stored prompt/admission/reuse inspection; no new Ghost schema field. |
| A2 follow-up | Separately assess unknown native capacity, remote token-estimate accounting and Sci-Fi's legacy character estimate; do not reopen the shared native capacity/EOS fix. |
| A3 | Review selected-act acceptance in Sci-Fi's existing markup repair, including salvage. Preserve flexible casts, whole failed drafts and bounded existing retries. |
| 2.4 source | HTML block joins without changing accepted source digests (ruling needed); scifi_news P0 literal-span convergence/live reverify; scifi_news_pro provider/output capacity and P9/GGUF follow-ups. No deterministic source-prune rung. |
| 2.4 routing/canvas | ShotLock write-side canvas validation, ltx_av long-beat underruns and matrix declared-vs-effective limits; wants_talking_prompt capture needs design. Image-phase route ownership is CLOSED (route_freeze.py owns it; all four call sites verified 2026-09-11). |
| 2.4 voice/credits | Opt-in Bark non-speech output repair with a bounded keep-best policy; small-canvas credits layout. Long hero-title containment is CLOSED (PBUG-20260911-01, 2026-09-11). |
| 2.4 audit tail | Re-ground test catalog/output-root dependencies, hidden output-env exporter and protected model-root consolidation. Do not repeat the completed scan collapse or worktree-credit fix. |
| 2.5 | If scheduled, effective clone-reference preflight before the writer, using real dispatch fallback/policy resolution; CastLock is too late and declared paths are insufficient. |
| 3.1 | Resume the saved Ghost v3 Half-B arc: the current author excludes beat dialogue and still picks objects by ordinal. The physical-artifact subject remains open; Half A/pool proof does not close it. No claim that old reviewers are still running. |
| 3.2 | Shared silent-video composer face/crux decision; preserve audio-in/text-to-video requirements and measure truncation at the actual engine. |
| 3.3 | Orphan generation occupancy versus cleared model-cache state; independent design, actual in-flight ownership. |
| 3.8 | Shared torch-free font-family/file resolution across captions, titles, credits and scopes; symptom containment does not close this design. |
| 3.4 | Reconcile remaining manifest/queue-time download/ffprobe clean-install gaps with already-shipped auto-downloaders. Do not rebuild implemented fetch paths. |
| 3.5 | Re-ground surviving runtime pack writes, dtype capability admission, janitor granularity, explicit cloud billing/opt-in and per-beat model reloads. Each true design survivor needs its own arc. |
| 3.6 | Shakespeare authenticated segmented/verbatim source artifact and field-owner table first; preserve Public Domain paraphrase/flexible cast and current no-word-gate rulings. |
| 3.7 | Remaining content-derived style, pitch/cast name reconciliation, dead-field ownership and meta cleanup. meta.style still has readers/writers, but meta.story_scaffold already means a separate control; resolve ownership instead of blindly renaming into it. |
| 4 / 4B | Re-read actual registry latest/status_reason/node extraction and compare exact published bundle to commit. Resolve remaining cnr_id/install evidence and prepare one current review note for operator posting. Revalidate the old discriminator/bisection hypothesis before any destructive strip-down. |
| 5.1-5.2 | Physical 8 GB combination/cold-install proof, measured profile choices and model matrix updates after operator release. One canonical JSON; lab profiles are not new shipping graphs. |
| 5.3 / 5.6 | Re-ground unqualified installed LLM family/slot combinations and GGUF opt-in availability. Do not rerun the completed historical Leg0 merely because an old task says next. |
| 5.4-5.5 | Read H3 policy receipts before another run; settle outstanding image-mode elements/refusal question with a real selected route. |
| 5.7 | Full canonical fastwan_8gb with long opening/closing music cues to prove existing chunked-music repair. |
| 5.8-5.10 | Remaining cfg promotion comparisons, failed-proof re-derivations, clean-room legs and eyewitness checks; do not call old component evidence full media proof. |
| 5.11-5.12 / AMD | Scoped pod/platform acceptance after access and prerequisite checks, actual ROCm/bitsandbytes capability before claims; retain deferred render blockers. |
| 6 | Still-unruled product choices and Bible fan-out candidates: confirm each remains unanswered; no assumed permission for version bumps, public posting or destructive installation. |
| 7 | Operator-parked casting/adaptation/AnimateDiff image-input ideas, OTR-Lite after v2, word_razzle placement and remote-recipe identity; activate only within their recorded scope. |
| 8 | Live client-bank end-to-end proof, coordinated activation instructions, remaining cleanup-tail/zero-frame/RenderError cases and live-backed Bible deltas. Preserve protected code and source contracts. |
| Release runway | Existing LEAN_MEAN_CLEANUP scope after this queue, representative platform acceptance, clean install, product documentation and operator-authorized release. Optional App/player, gentle stable speaker placement and useful native-video ambience remain separate designs. |

Stable-ID renumbering and the detailed withdrawn/rejected proposals remain in
the archive and standing rulings. Preserve those decisions when resuming a row;
their existence is not authorization to repeat completed work.
