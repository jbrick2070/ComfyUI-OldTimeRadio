# OTR Go-Forward Plan

This is the main and sole executable work queue. Only unfinished work belongs
here. Read AGENTS.md, CLAUDE.md and [standing rulings](OTR_STANDING_RULINGS.md)
before acting. Current measured status, commits and test receipts are in
[session_handoff.md](../session_handoff.md) and [HANDOFF_LOG](HANDOFF_LOG.md).
When work finishes, move its receipt to the log/archive and remove it here.

## Next owner and sequencing

Opus takes the next coding sprints when Jeffrey starts that window. One
production coder at a time, branch v2.0-alpha. No build or GPU run is active.
Work the coding scopes below before starting further generation. Review
genuine design choices through R1 arc, R2 implementation, R3 wiring and R4
convergence; a deterministic conformance fix needs the finished-diff review
specified in CLAUDE.md. Obtain actual Sonnet QA after any production revision,
ground its findings, and record the reviewers who really ran. Substitute an
unavailable reviewer under the standing rule; do not fabricate consensus.

## Sprint 1 -- own the scopes video under its episode

Fix OPEN PBUG-20260911-03 at the existing producer,
nodes/otr_scene_aware_scopes.py. It currently writes a retained scopes MP4 into
shared scratch and falls back to system temp. Use the validated episode path
authority (nodes/_otr_paths.py, otr_composited_dir) and the actual manifest
episode_id. Keep the returned-path consumer contract. Do not move files after
render, change another episode's identity or reopen the already-fixed rename
transaction. Preserve all existing published/diagnostic assets.

Expected files: nodes/otr_scene_aware_scopes.py,
tests/test_node_temp_hygiene.py and focused real-producer path coverage; relevant
Bible01.02/12.66 rule, coverage and index together. The old hygiene assertion
requires shared scratch and must learn the current asset-owner contract. The
Bible currently indexes this occurrence as OPEN; it does not claim a fix or
new executable path check.

Canonical node94 receives the episode manifest from node92 through link271;
its returned path reaches node93 through link273. No output-directory widget
exists. Verify the live INPUT_TYPES and links. No graph mutation is presently
required; if node/wiring/widget/config changes become necessary, apply them in
workflows/otr_canonical.json in the same chunk. Never create a second graph.

Acceptance: the real producer returns a nonempty file under the validated
active episode directory; invalid identity or output failure cannot send it
to shared/system temp. Exercise the existing consumer with bypass off and on
without changing published audio or frame contracts. A full canonical live
receipt must show the scopes path, final episode assets and OBS publication.
Do not introduce a story-rejection gate or a new output-path owner.

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

- Mac and 4060: no contact, activation or test prompts until Jeffrey explicitly
  releases them. Their own proven routes and owners remain authoritative.
- RunPod: scoped use is authorized but blocked by missing authenticated current
  access. Pull the qualified repository/image before testing, record the actual
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
| 2.4 routing/canvas | Image-phase route ownership, ShotLock write-side canvas validation, ltx_av long-beat underruns and matrix declared-vs-effective limits; wants_talking_prompt capture needs design. |
| 2.4 voice/credits | Opt-in Bark non-speech output repair with a bounded keep-best policy; small-canvas credits layout. Long hero-title containment is not pending code. |
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
