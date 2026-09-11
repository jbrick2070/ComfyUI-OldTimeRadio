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
- **Prompts are hand-crafted per model and CHARACTER-BUDGETED.** Do not swashbuckle
  them. Adding conditional nuance spends budget that does not exist and reintroduces
  the per-prompt subtleties the operator has already rejected; removing words edits a
  tuned recipe. Both directions are closed.

## 1. Tonight -- the four-machine wave (nearest deadline)

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

## 2. Open defects with an owner

| Defect | State | Next action |
|---|---|---|
| PBUG-20260911-03 scopes path | Code fixed, offline-qualified, pushed. NOT closed. | 5080 Leg A. Nothing left to code. |
| Sci-Fi repair-turn capacity overflow | **Open, crash-class.** `nodes/_otr_scifi_news_pro.py` never catches `PromptContextOverflowError` / `GenerationContextOverflowError` (`nodes/_otr_generation_budget.py`), so a too-generous cap propagates uncaught mid-episode. | Splits in two. **The catch is shippable** -- it fires only where today it crashes, so every currently-succeeding episode is unchanged. **Changing the cap VALUE is not** -- it decides repair-vs-cold-regeneration and therefore which script comes back. Take the catch; leave the value. |
| 8 GB draft-profile refusal | Known (PBUG-20260904-05). Decides whether the 4060 publishes anything tonight. | Handled in the wave by naming the shipping profile. No code. |
| Runtime writes INSIDE the pack directory | **Open, durability-class -- this is the bar's own second bullet.** `nodes/_otr_openrouter_backend.py:792` still resolves its catalog cache to `Path(__file__)/../models`, and `nodes/_otr_shared/cloud_media_backend.py:209` puts `billing_ledger.jsonl` (a real-money audit trail, and the ONLY copy) under `<repo>/otr/cache/`. A registry update or reinstall wipes both. | NOT the mechanical edit it looks like -- grounded 2026-09-11: relocating the catalog default cold-starts a populated cache, which drops `resolve_context_window` to the 8192 default and changes the writer's token budget. So it needs a migration step, and the billing ledger should NOT land in a tier whose contract says "never the only copy". Do this AFTER the wave, with the migration. |

## 3. Qualification coverage still owed

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

## 4. Design rows awaiting an arc

None of these get coded without their arc first. Arcs cost no GPU and no money.

| Row | Scope |
|---|---|
| 3.5 per-beat model reload | A ~14 GiB LTX reload per beat. Framed as an **OOM surface** rather than a speed complaint -- under the section 0 bar it earns its place because reloading that much per beat is where a render falls over, not because it is slow. It sits here, below section 2, deliberately. |
| 3.5 janitor granularity | The janitor cannot sweep `tmp/audio_slices` -- no such reference exists in `nodes/_otr_janitor.py`, and 9.3 GB was measured sitting there. Durability. It widens an auto-delete, so it lands with a test and never without one. |
| 3.8 | Shared torch-free font-family/file resolution across captions, titles, credits, scopes. The Mac's text-rendering leg tonight is its evidence. |
| 3.2 | Shared silent-video composer face/crux. Touches eleven lanes on both machines, so CLAUDE.md section 0B applies -- prove the unchanged machine is unchanged, measured. **Guard: preserve the audio-in / text-to-video requirements, and measure truncation at the ACTUAL engine, not from a declared limit.** |
| 3.3 | Orphan generation occupancy vs cleared model-cache state. Several narrow fixes each surfaced a new race; that is the two-strikes signal. |
| 3.1 | Ghost v3 Half-B: the author still picks objects by ordinal and excludes beat dialogue. Restart the arc; r2 never converged. |
| 3.4 | Clean-install manifest / queue-time download / ffprobe gaps. Its "blocked on 1.1" clause is STALE -- there is no 1.1 row. The 4060 and RunPod pull steps tonight produce the real gap list, so do not arc it before that. |
| 3.6 | Shakespeare segmented/verbatim source artifact and field-owner table. Explicitly not next. |
| 3.7 | Pitch/cast name reconciliation and dead-field ownership. `meta.style` still has readers and writers; `meta.story_scaffold` already means a separate control -- resolve ownership, do not blind-rename. **"Content-derived style" is STRUCK** -- deriving style from content is picking a winner between pack and story, which [ARC_CLOSED](2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md) settled. Do not reopen it. |
| 2.4 routing/canvas | ShotLock write-side canvas validation; ltx_av long-beat underruns; matrix declared-vs-effective limits; `wants_talking_prompt` capture. |
| 2.4 voice/credits | Opt-in Bark non-speech repair with a bounded keep-best policy; small-canvas credits layout (known -- report it, do not chase it). |
| 2.4 audit tail | Remaining output-root/env-exporter and protected model-root items. The cold-cache test dependency is FIXED; the google/veo unpinned-fixture and worktree-credit claims were re-grounded and REFUTED -- do not re-derive them. |
| 2.4 source | HTML block joins pending an operator digest ruling; scifi_news P0 literal-span convergence; scifi_news_pro provider/output capacity and P9/GGUF follow-ups. No deterministic source-prune rung. |
| A3 | Selected-act acceptance in Sci-Fi's markup repair. Needs one short review, not a full arc: exact match vs tolerance band. Routes into the EXISTING ladder; never a new checker. Stated once, here. |
| A2 follow-up | Unknown native capacity and remote token-estimate accounting. The Sci-Fi character-estimate half is tracked as the crash-class row in section 2. **Guard: do NOT reopen the shared native-capacity/EOS fix -- it landed and is receipted.** |
| 2.2 | Five-act forced-Ghost CUDA publication with stored prompt/admission/reuse inspection. No new Ghost schema field. Needs a CUDA proving host. |

## 5. Blocked on the operator -- each unblocks with one word

2.4 source digest ruling (HTML block joins) · A3 exact-match vs tolerance ·
4 / 4B registry review note for posting (**guard: revalidate the old
discriminator/bisection hypothesis before ANY destructive strip-down**) · 5.1-5.2 release after physical 8 GB proof ·
6 unruled product choices and Bible fan-out candidates.

## 6. Hardware and publication state

- **Mac, 4060, RunPod: RELEASED** for tonight's wave. Their own proven routes and
  owners remain authoritative. Findings to `docs/`; they do not push.
- **Registry versions, tags, public posts and release promotions** need the operator's
  approval process. Pushes to `v2.0-alpha` are required and are not a release. Never
  send an old review draft.
- **2.5 clone-reference preflight is a conditional tombstone.** Attempted and reverted
  2026-09-04 after a grounded review killed it on three counts, all still true. Not
  scheduled. Do not revive without new evidence.

## 7. Constraints specific to this plan

Only the ones not already in CLAUDE.md or the standing rulings.

- Full listener source, no RSS. **Selected acts are binding.** Cast count is flexible
  and records requested vs actual; the house announcer is excluded from dramatic cast.
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

## 8. Parked -- preserved, not work

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
