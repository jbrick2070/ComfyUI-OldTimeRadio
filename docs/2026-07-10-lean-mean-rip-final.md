# OTR Lean-Mean Rip -- De-Slop / Ship-Shape Campaign Plan

**Date:** 2026-07-10 (analysis-only session; NO code changes tonight -- operator directive)
**Status:** CODE-READY, VERIFIED AGAINST REAL POST-PORTABILITY CODE. Portability
S0-S6 SHIPPED same night (through `20185542`, operator ratifications folded, 7 variants
emitted, suite 7251 green) -- so a 2-agent verification sweep re-checked EVERY concrete
claim in this doc against that HEAD: 48 confirmed, 5 stale line-cites re-pointed, 4
factual corrections folded (all marked VERIFIED-2026-07-10 inline). R4 Fable gate:
CONVERGED after must-fixes MF-1..MF-7 (folded). The REVISE-AFTER register (section 2)
is now largely RESOLVED with real answers. Decision boxes D-1..D-6: OPERATOR RATIFIED
2026-07-10 (D-2 with a future-upscale codicil, section 7). CLEARED TO EXECUTE
post-portability; execution-time gates = R-4 seam re-survey + R-7 re-grep only.
**Method:** 3 mechanical inventory agents (sonnet, file:line grounded) -> 3 Fable
architect fan-out takes (deletion-first / boundary-consolidation / ship-integrity) ->
Claude judge synthesis -> grounding audit (12-point claim verification) -> Fable final
gate. Round artifacts (campaign log + r4 judgment) live in the LOCAL gitignored working
folder `docs/2026-07-10-lean-mean-rip/` per repo convention.
**Goal (operator):** make the repo shippable and efficient, remove AI-slop code, end-state
as if designed by one senior architect -- lean and mean.

## 0. Page One -- What This Buys

Gross deletion ~32-33k LOC (~12% of 264k), plus consolidation of five half-finished
patterns into finished ones. Repo lands at roughly 230k LOC with: zero dark scaffolds,
zero unwired cockpit switches, one ffprobe boundary instead of ~16, one registry base
instead of two-and-a-half, a writer package instead of a 6,765-line file, half of
scripts/ no longer ambiguous to strangers, and a test suite whose green count stops
including dead code's private tests.

Current baseline (measured 2026-07-10): 789 py files / 264,479 LOC -- nodes/ 225/120,442;
tests/ 464/111,540; scripts/ 65/19,966; visual/ 25/10,053. Suite 7,136 green.

Panel convergence was strong: all three architects independently ruled RIP (not wire)
on OTR_RTXUpscale, delete (not keep-dark) on the dark engine scaffolds, dissolve
story_orchestrator, and finish-or-delete every half-consolidation. Conflicts were
adjudicated by the judge against grounded evidence (section 8).

## 1. Standing Deletion Policy (adopt as house law)

1. DELETE IS DEFAULT; git is the archive. A rip commit + Bug Bible entry is the
   tombstone. Keep-dark/quarantine requires a NAMED future consumer in a converged plan
   doc plus an expiry date (the hy3:free pattern). No name + no date = delete now.
2. Dead test = (no production import) AND (absent from otr_canonical.json) AND (unnamed
   in current converged plans). Three of three: delete, same commit as its subject.
3. Tombstone tests are demolition scaffolding: they die at the first tag after their rip
   shipped. Reintroduction guards live ONCE in a single denylist test file, not as
   per-rip 300-line files.
4. An abstraction earns existence only if >=2 production call sites adopt it IN ITS
   LANDING COMMIT. Never again merge an unadopted "single source of truth."
5. Every wave: full suite + Bug Bible + headless boot + OTR_WorkflowValidator +
   commit AND push, same session. Graph-touching waves add a 30-word live smoke.

## 2. REVISE-AFTER-PORTABILITY Register (re-verify before executing any wave)

This plan was written with portability ASSUMED LANDED; it then LANDED the same night
(S0-S6 through `20185542`) and the register was re-verified against the real tree:

- R-1. RESOLVED-CONFIRMED: writer = 34 widgets (INPUT_TYPES :2115-2773; canonical
  node-1 widgets_values length 34); config/profiles/widget_mapping.json version:2 with
  perfect_run_spacesaver in exempt_widget_names (:445). W5's arithmetic holds.
- R-2. RESOLVED-CONFIRMED: scripts/build_variants.py exists (--check at :329);
  workflows/variants/ holds 7 emitted variant+recipe pairs (nv50 16gb_full, 8gb_lite,
  cpu_floor, nv40_12gb, amd16, amd8, mac_mps). Cloud variant still REFUSED pending
  OpenRouter pin ratification -- the last ratify gate.
- R-3. RESOLVED-CONFIRMED: S0 fixes verified in code (loader guard _otr_model_loader
  :143; dispatcher gen_fn=None raises :455; _detect_host mps/vendor validator
  :249-256; installer trio _otr_{indextts2,dia,chatterbox}_install.ps1 shipped).
  Nothing graduates into W0.
- R-4. STILL LIVE: S1 threading landed (nodes/_otr_shared/llm_policy.py; instantiated
  writer :1395); the writer grew to 6,876 LOC and the slot-plumbing/seeding seam
  boundaries MOVED -- SW-1's line ranges MUST be re-surveyed at execution.
- R-5. RESOLVED-CONFIRMED: gate_in wired as link 279 into the writer (forceInput,
  consumes no widgets_values slot); validator asserts master_hash. See W5 for the
  now-QUANTIFIED dst_slot obligation this creates.
- R-6. RESOLVED: test_openrouter_* is still exactly 13 files post-S6; W8 triage base
  unchanged. (Cloud pins ratification may add coverage later -- re-glance then.)
- R-7. STILL LIVE: line numbers are 2026-07-10 measurements against `20185542` --
  re-grep before cutting, always. Suite baseline at verification: 7251/31/1.

## 3. Doctrine Quick Fixes -- W0 (small, high-value, land first)

- W0a. README.md:44-45 and :163 claim "the pipeline falls back to a guaranteed CRT
  floor." VERIFIED-2026-07-10, RESOLVED: the CRT floor is a REAL, wired, sanctioned
  node -- SignalLostVideoRenderer (video_engine.py:2028, registered as
  OTR_SignalLostVideo, root __init__.py:163). So: REWORD both sentences to "policy
  floor" / "explicit signal-lost route" -- never the word "fallback" -- do NOT delete.
  The README scripts a stranger's expectations at the exact moment they hit a missing
  model.
- W0b. nodes/video_engine.py:2291-2306 -- exception swallow falls back to a hardcoded
  legacy audio path (output/otr/audio/). Direct house-law violation. Becomes a raise;
  test harness needs go through conftest stubs, not tolerated prod fallbacks.
- W0c. nodes/video_engine.py:1815-1821 -- `except Exception: pass` around a JSON parse,
  zero logging. Raise or log-and-raise.
- W0d. Bare `except:` clauses -- VERIFIED-2026-07-10 count is TWO, not four:
  story_orchestrator.py:183 and _otr_model_loader.py:275 (the VRAM evict wrap; S0's
  touch did not fix it). _otr_freeze_cascade.py is already clean (every clause is
  `except Exception  # noqa: BLE001`). Both -> `except Exception` minimum, narrowed
  where obvious.
- W0e. nodes/_otr_casting.py:962-966 "Defensive fallback" on gender-pool exhaustion
  degrades quietly -> fail loud (bounded-repair may retry upstream, but never silent).
- W0f. Delete scripts/normalize_workflow_widgets.py (~200 LOC) -- self-flagged "KNOWN
  OVER-AGGRESSIVE -- DO NOT RUN AUTONOMOUSLY," it broke 5 guardrail tests in May, and it
  sits next to the ONE canonical JSON. A loaded gun, not history.
- W0g. Delete scripts/_tmp_video_art_*.json (3 temp probe artifacts committed against
  the delete-temp-probes law). Verify scripts/_otr_canonical_api_prompt*.json are
  regenerated by the harness each run; if so gitignore them.

W0 is code-touching (b-e) -> full suite + Bug Bible + one 30-word smoke.

## 4. Deletion Waves W1-W8 (each = one commit-sized chunk with its test deltas)

**W1. Inert text (~670 LOC, zero graph contact).**
Delete the writer's 403-line `__main__` self-test harness -- VERIFIED-2026-07-10 now at
OTR_LedgerScriptWriter.py:6474-6876 (file grew 6765->6876 with S5a's widgets; harness
LOC unchanged). Unreachable; shadows the real suite. Delete story_orchestrator.py:185-252
(68 lines: _truncate_at_sentence_boundary / _tail_at_sentence_boundary /
_inject_scene_transitions each defined TWICE at module scope; the FIRST bodies are
silently shadowed dead code -- the live _inject_scene_transitions is the LATER one and
they emit different cue text, so delete first bodies only). Guard added same commit:
ruff F811 (or an AST no-duplicate-top-level-defs meta-test) so shadowed twins can never
recur silently.

**W2. Dark engine scaffolds + their tests (~1.8k LOC).**
Delete files (NOT keep-dark): nodes/_otr_video_engines/{eng_character_3d.py (441),
eng_still_parallax.py (300), eng_triposr.py (159)}, nodes/_otr_image_engines/
{hidream_i1.py (100), sd35_large.py (105)}; tests test_video_character_3d (385),
test_video_still_parallax (261), test_video_triposr (43). Grounded: zero references in
otr_canonical.json; registry __init__ files carry tombstone comments only; the 8-tier
portability matrix names none of them. Keep-dark is the worst state: CAPABILITIES v2
changes the engine contract, so dark scaffolds rot into things that LOOK resurrectable
but need rewrites anyway. Shrink the __init__ tombstone essays to one line each.
TEST DELTAS (R4 gate MF-4, scope corrected by VERIFIED-2026-07-10): the hidream_i1/
sd35_large imports in test_capability_profiles.py:69-71 and test_workflow_apply.py:80-82
are FUNCTION-scoped (inside the _image_registry()/_registry_engine_ids() helpers), NOT
module scope -- so deletion breaks the tests that CALL those helpers, not whole-file
collection. Same fix (strip the imports same commit), smaller blast radius than the
gate stated. Also sweep test_model_slot_audit.py:103,
test_registry_is_the_menu_guard.py:63, test_image_dep_pilot.py:35,:78 (absence-asserts
survive; comments rot). NOTE the W2->C2 dependency: the image twins hold the only
non-None requires_flag values (hidream_i1.py:54, sd35_large.py:58) -- C2's vestige kill
assumes W2 already landed.

**W3. Unadopted "single-source-of-truth" modules + their tests (~2.5k LOC).**
Zero PRODUCTION importers confirmed for all six: nodes/_otr_shared/{sidecar.py (136),
slot_matrix.py (88), content_oracle.py (198), nsfw_frame_qc.py (198),
cloud_media_cache.py (264)} and nodes/_otr_audio_cache.py (324). But the R4 gate (MF-3)
REFUTED "only their own tests reference them" -- SEVEN LIVE test files import from the
kill list, so a blind delete is a collection-error wave. Per-module rulings:
- sidecar.py + cloud_media_cache.py: DELETE (VERIFIED-2026-07-10: _otr_shared/
  sidecar.py has ZERO importers of any kind, not even a private test; cloud_media_cache
  keeps only its own test, which dies with it).
- nsfw_frame_qc.py: DELETE + surgery on test_video_survival_guide_vectors.py:27.
- _otr_audio_cache.py: DELETE + surgery on test_release_gate.py:127 -- the release-gate
  contract is LIVE; stub/inline the AudioCacheRecord shape there.
- slot_matrix.py + content_oracle.py: RE-ADJUDICATED (gate catch) -- live guards consume
  them (test_rip_sfx_broll_guard.py:21; the three live viz-engine contract tests
  test_video_viz_{camera:16,rainbow:20,mandala:26}.py that C1 depends on; plus
  test_slot_matrix_soak.py:31-32). They are test-support modules living in nodes/:
  default = RELOCATE to tests/ support (or excise the assertions), decided at execution.
  A module only tests import is test infrastructure, not production surface.
EXCEPTIONS: (a) nodes/_otr_probe.py is NOT deleted here -- superseded by C5; it dies in
C5's landing commit. (b) HARD HAZARD: the LIVE audio sidecar helper is
nodes/_otr_audio_engines/_otr_sidecar.py (imported as _SC by eng_dia.py:35 and
eng_chatterbox.py:30) -- the dead twin _otr_shared/sidecar.py even names it in its
docstring. Kill list uses FULL PATHS ONLY. (c) cloud_media_cache: if the cloud-audio S1
build (parked plan d2fc8d77) starts before this wave, re-check adoption; plan law says
the rebuild targets schema v2, not a pre-v2 cache.

**W4a. The visual/ POC tree (~10.4k LOC).**
Delete the entire visual/ package (25 files, 10,053 LOC) -- the 2026-04-15
sidecar-subprocess POC superseded by the wired Image/Video platform. Grounded: zero
non-test importers; registration is a 5-entry lazy-import table in the ROOT
__init__.py:172-184 (OTR_VisualBridge/Poll/Renderer/PromptCoercion/ExtractFluxPrompt) --
remove those entries same commit. Test deltas: delete
test_visual_prompt_coercion_contract.py (94) and test_wedge_probe.py (imports visual/);
INSPECT test_b5_visual_polish_collapse.py:218 at execution -- if it pins the collapse
itself, fold its assertion into the W7 denylist; if it guards live b5 behavior, surgery.
NOTE (grounding correction): test_image_platform_c1.py does NOT import visual/ -- no
surgery needed there, contrary to one panel claim.
GATE ADDENDA: surviving repo scanners hardcode "visual" in directory tuples -- strip
"visual" from those tuples in this same commit (or verify each tolerates the missing
dir, as test_story_brief_helpers_c5b.py:256 does via is_dir()). Add the five
OTR_Visual* node types to the validator's DELETED_NODE_TYPES denylist per the repo's
existing tombstone pattern. Internal-consistency note: deleting visual/renderer.py
removes one of C5's listed ffprobe migration sources -- C5's sweep list shrinks by one.

**W4b. Unwired registered nodes (~2.2k LOC).**
Delete: OTR_SaveToEpisodeWorkspace (+ its 267-line test; superseded by dispatcher-side
persistence), OTR_ProjectStateLoader (nodes/project_state.py; zero refs anywhere),
OTR_VRAMContextTest (its own docstring says superseded), OTR_VideoProbe
(nodes/otr_video_probe.py; render_driver fills real host_caps post-portability -- the
probe's job no longer exists). KEEP: OTR_VRAMGuardian with a WRITTEN EXPIRY -- it earns
its slot during 8-tier VRAM bring-up, dies at portability ship (or move behind an
OTR_DEBUG_NODES=1 registration gate). Registration edits in the same commit.
TEST SURGERY (grounding correction -- do NOT delete wholesale):
tests/test_video_platform_aseam.py is ~98% LIVE coverage (VideoDirector widget vector,
ShotLock clip budget, render_driver terminal-fail, role_compat/resolver/schemas/
gpu_residency/portrait_ledger). Remove ONLY test_probe_emits_usable_list (:299-306),
the OTR_VideoProbe entries in the two parametrized lists (:265-273, :291-296), PLUS
(gate catch) the module-scope import at :28 (`from nodes.otr_video_probe import
OTRVideoProbe`) and the cold-import subprocess string at :76 -- the ranges alone leave
an ImportError. Add the four ripped node types to the validator DELETED_NODE_TYPES
denylist.

**W5. OTR_RTXUpscale rip -- the one graph-touching wave (~1.4k LOC). PANEL UNANIMOUS.**
Rip, do not wire: the node (nodes/rtx_upscale.py, 922 LOC) delivers a 1080p ceiling the
ffmpeg composite chain already owns; wiring it would create a second scaling authority.
The cockpit lie dies with it: writer widget perfect_run_spacesaver exists ONLY to serve
RTXUpscale's cleanup, and OTR_PostUpscaleProcgenBlend's docstring narrates a dead
1472x832-to-RTX-VSR pipeline (fix the docstring same commit).
POSITIONAL REALITY (re-verified against the REAL post-S5 canonical, 34-entry array):
perfect_run_spacesaver = INPUT_TYPES index 9, canonical node-1 widgets_values[9]=false
-- CONFIRMED unchanged after S5a (portability appended at the end). Pinned by
test_workflow_json_guardrails (:591, and the slot-9 assert now at :773 -- the old :735
cite drifted after S5 fixture updates), test_openrouter_slot_widgets_s2:30,
test_otr_api_companions (:38, :101, and the third pin now at :240), test_meta_paths
(:154,:161,:181 -- dict keys, no positional drift). Removing slot 9 shifts every later
index left by one -- BUG-LOCAL-097 class if done casually.
RULING: one DECLARED SCHEMA EPOCH commit -- INPUT_TYPES removal + canonical
widgets_values slot-9 removal + widget_mapping v2 exempt-list edit + otr_api parity
fixtures + all FIVE pinning test files (the four above PLUS -- gate catch MF-2 --
tests/test_filename_pattern_audit.py, which read_text()s nodes/rtx_upscale.py and
asserts its glob strings at :189-196 with an allowlist row at :82: FileNotFoundError on
delete) + validator fixtures + registration removal + node file deletion + docstring
fix + a validator DELETED_NODE_TYPES row -- then `build_variants.py` regenerates ALL
variant JSONs + recipes mechanically (this is why the wave waits for portability:
variants are generated, so the rip costs one canonical edit + one regenerate, never
nine hand-edits).
SECOND POSITIONAL AXIS (gate catch MF-1 -- render-breaker class, NOW QUANTIFIED against
the real post-S5 canonical): the widget ALSO exists as a node-1 inputs[] entry
({"widget":{"name":"perfect_run_spacesaver"},"link":null} between creativity and
min_p), and LINKS address inputs by dst_slot = index into that array. VERIFIED
2026-07-10: node-1 inputs[] mirrors widgets_values 1:1 for indices 0-33, plus gate_in
appended at index 34 carrying link 279 -- the ONLY live link into node 1 (all 34
widget-mirror entries swept: every other link is null). THE EXACT OBLIGATION: the epoch
commit removes inputs[] index 9 AND renumbers link 279's dst_slot 34 -> 33, then runs a
link-referential-integrity audit. widgets_values was never the only positional array.
Append-only law's PURPOSE (protect saved graphs from silent drift) is honored: the
saved-graph population is canonical + generated variants + the parity copy, all
regenerated in-commit; policy is the validator REFUSES old-layout graphs loudly rather
than shifting them. Verification: widget-count-vs-INPUT_TYPES audit, JSON round-trip,
full suite, ONE 30-word live smoke.
OPERATOR DECISION BOX (see section 7): default window is post-portability per your
sequencing lock; the panel's alternative (pre-S5, while the writer is still 28 widgets
and no variants exist) is cheaper but resequences portability -- your call at execution.

**W6. Scripts purge (~9.5k LOC).**
Of 65 top-level scripts, ~32 are concluded history. HYBRID POLICY (judge ruling between
delete-all and archive-all): DELETE concluded one-shots whose outcome is recorded in
docs/memory -- finished bakeoffs (build/run_ltx_av_q_bakeoff, run_wan_ti2v_bakeoff + the
two bakeoff workflow JSONs living in scripts/), closed-bug doctors (otr_gemma4_doctor),
superseded pre-cleanbreak render chain (render_episode_concat/render_flux_batch/
render_humo_batch -- __init__ tombstones the node types they fed), soak_operator
(superseded per otr_api.py's own docstring), scripts/vram_context_test.py, concluded
probe-eval scripts, scripts/_otr_b_spikes/ (5 probes + harness gating a 3D path
rescoped away 2026-06-08). MOVE to scripts/dev/: reusable dev tooling a stranger
shouldn't run but the lab still uses (dep pilots, consult/roundtable runners like
_consult_openai.py, stage/story scan tools if still consulted).
DO NOT MOVE: run_codex_agent.ps1 + run_agy_agent.ps1 (kibitz panel launchers -- the
skill harness outside this repo invokes them BY PATH; grounded: no in-repo referencers,
but the external skill dependency makes their paths load-bearing). Live ops stay:
otr_api.py, otr_canonical_api_run.py, otr_headless_canonical.ps1, watchdog, zombie
killer, tail logs, serve_ledger, tree doctor, openrouter refresh, hf_download_driver,
the 3 TTS sidecar workers, downloaders, audit_otr_full_run -- PLUS (gate catch MF-6,
all load-bearing): _otr_soak_server_launch.cmd (invoked by otr_headless_canonical.ps1
:32 and read by test_determinism_env_launcher.py:19 -- required by every headless-boot
gate THIS PLAN mandates), otr_ia2v_server_boot.cmd (:20 chains to it; ia2v is the
production recipe), otr_mesh_stage_blender.py (live engine worker;
test_video_mesh_stage.py:43), otr_video_soak.py (test_video_soak_fixture.py:21),
otr_pin_partner_nodes.py (test_partner_nodes_pin.py:18, test_audit_i2v.py:18),
profile_scope_render.py, otr_visual_smoke.py -- AND (verification catch, new tonight):
scripts/build_variants.py (the variant generator itself -- load-bearing for W5/R-2) +
the three sidecar installer scripts _otr_{indextts2,dia,chatterbox}_install.ps1
(shipped by portability S0; referenced by engine error messages).
TEST-DELTA TABLE REQUIRED (gate catch MF-5 -- this wave had none): every deleted or
moved script lists its test coupling in the same commit. Known couplings:
test_treatment_scanner_unicode.py:22 imports soak_operator (dies with it);
test_render_humo_batch_plan / test_render_episode_concat_discovery /
test_render_flux_batch / test_build_silent_test_episode load the render trio (die with
it -- and build_silent_test_episode.py itself gets re-triaged live-vs-dead first); the
dev/ moves break path pins in test_audio_dep_pilot.py:18, test_image_dep_pilot.py:17,
:93, test_video_dep_pilot.py:17, test_stage_direction_scan.py:14,
test_story_quality_scan.py:18 (+_r2) -- update pins or move tests with their scripts.
Deleted-script names land in test_canonical_headless_api.py's existing
RETIRED_FULL_WORKFLOW_HARNESSES denylist (:28-61) or the W7 table.
Policy line for README: "If it's in scripts/, a stranger may run it today; scripts/dev/
is lab tooling." Also: `__pycache__` litter in scripts/ -- gitignore-verify.
NOTE (operator 2026-07-10): docs/ENGINE_MATRIX.md -- a GENERATED engine x backend
visibility table emitted by build_variants.py from the registries -- lands in the NEXT
coder session, BEFORE this campaign (spec in GO_FORWARD_PLAN). W6's README policy line
should link it; C2/W2 registry changes will regenerate it automatically via --check.

**W7. Tombstone retirement (net ~3.0k LOC).**
Grounded kill list (16 files, 3,179 LOC): test_no_orchestrator_legacy_symbols (442),
test_legacy_audit_clean (374), test_no_phase_9_call_b3 (323), test_no_rollback_gates_b2
(273), test_no_phantom_handlers_b4 (234), test_legacy_contract_retired (249),
test_lfc_phase_extinction (202), test_no_ltx_style_brief_c3b (206),
test_fetch_science_news_no_legacy_wrapper (219), test_openrouter_model_gone (201),
test_no_cleanup_model_id (106), test_batch_dispatch_retired (88),
test_kokoro_legacy_node_retired (69), test_bark_legacy_node_retired (64),
test_init_aliases_empty (65), test_no_ollama_backend (64).
DOUBLE-COUNT NOTE (gate): test_openrouter_model_gone sits in both this list and W8's
13-file cluster -- it retires HERE; W8's arithmetic recounts at R-6.
Replacement: ONE tests/test_removed_surfaces.py denylist table (~150 LOC) asserting
banned symbols/imports/node-types stay absent -- one row per retired surface, Bug Bible
cross-referenced. Age gate: a tombstone survives until the first tag after its rip
shipped; anything the portability build itself lands stays until portability's tag.
W2/W3/W4 rips add their rows here rather than new files.

**W8. OpenRouter test diet (~1.7k of 2.6k LOC) -- LAST, gated.**
The cluster survives as a category (the cloud tier ships openrouter slots); 13 files for
one lane is sprawl. Keep contract/auth/fail-loud (402/429) core ~900 LOC; delete
permutation replays and anything resolving ~latest aliases LIVE in tests (flake bombs).
GATE: only after the otr_cloud_lanes variant emits and one cloud smoke runs green
(R-6 re-triage first).

## 5. Consolidation Campaign C1-C7 (finish every half-consolidation or delete it)

The repo's disease is not size: it is five half-finished consolidations. Rule: FINISH or
DELETE -- no third state. The consolidation unit here is the FUNCTION, not the base
class (the registry doctrine's own words: "inheritance is never required").

- C1. Viz engine family: move the ~90-line near-identical scaffolding (_ref_path,
  _canvas_dims, _build_render_request, _clip_from_raw, prepare/teardown) from
  eng_viz_camera/eng_viz_mandala/eng_visualizer/eng_viz_rainbow into motion_common as
  FUNCTIONS (it already exists and eng_viz_camera already imports it). Engine-specific
  by design: draw function, look constants, fps/aspect/roles. Also kill the vestigial
  `fallback_engine = None` field on viz engines (a fallback field in a no-fallback
  house). ~360 dup LOC collapses.
- C2. Audio registry onto _otr_shared/engine_registry_base.py (video+image already use
  it; audio's hand-rolled copy was migration caution, now expired -- its docstring
  admits it). VESTIGE SCOPE CORRECTED (gate catch MF-7 -- "two vestiges" was wrong by
  an order of magnitude): requires_flag spans the audio base (base.py:47), the audio
  descriptor (registry.py:46), NINE audio engine files, and is exercised as a tested
  FEATURE (test_audio_engine_registry.py:26 stub-flag gating;
  test_audio_engine_adapters.py:64-70 None-asserts); the only non-None values live in
  the W2-doomed image twins (hidream_i1.py:54, sd35_large.py:58). So: W2 lands FIRST,
  then C2 rips requires_flag + GATED_BY_FLAG across base + registry + engines + 3-4
  test files + dep-pilot getattr sites, one commit, blast radius stated. Audio keeps
  generate_voice/generate_clip as protocol extensions. VERIFIED-2026-07-10: S3's
  registry-v2 atomic commit did NOT unify audio -- video+image registries import the
  base (:25 each), audio still hand-rolls; C2 is fully open, exactly as planned.
- C3. Local image family: give it the base/helpers its cloud half already has
  (_CloudImageBase exists; the 6 local files carry a byte-identical 4-line _role_of).
  Mirror the existing shape; invent nothing.
- C4. TTS subprocess scaffolding: one _otr_audio_engines/subprocess_tts.py helpers
  module (venv resolution, worker launch, wav load) for chatterbox/dia/indextts2.
  Drift is already real: eng_indextts2.py:245-247 hand-rolls what _SC.remove_quietly
  centralizes. Engine-specific: env vars, worker filenames, timeouts.
- C5. ONE FFPROBE BOUNDARY (grounding upgraded this: ~11 files / ~16 independent
  ffprobe-invoking implementations with FOUR different binary-resolution strategies --
  env+PATH, which-only, ffmpeg-sibling, bare "ffprobe" no check). Build
  nodes/_otr_shared/ffprobe.py: binary resolution (env OTR_FFPROBE > PATH > ffmpeg
  sibling) + duration + dims/fps/pix_fmt + stream counts. Migrate: otr_credits_roll
  (:908), otr_master_audio_mux (:39), otr_silent_composite (:33), cloud_media_canonical
  (:402), wan_shared (:52), audit_otr_full_run (:71-104), remaining engine adapters
  found by the grep sweep. _otr_probe.py dies here (superseded; its API was
  duration-only and insufficient -- adjudicated against the "adopt as-is" take).
  Guard: ratchet test on `ffprobe` grep-count outside the boundary module.
- C6. THE ONE-SHAPE RULING (166 isinstance-dict-else sites / 56 files): canonicalize AT
  CONSTRUCTION, at module boundaries -- the moment a line-request/profile/render-request
  crosses a boundary it becomes the pydantic model (schemas.py already exists);
  downstream is attribute-only and a dict RAISES (fail-loud satisfied). Producer-first
  migration, then delete consumer branches shape-by-shape, starting with the three
  shapes behind the writer-54 / video_engine-20 / orchestrator-18 except-cluster.
  Guard: a RATCHET manifest test -- per-file isinstance-dance counts frozen, fails on
  increase, ratchets down as files migrate. Dicts remain legal INSIDE a module. This is
  an ongoing campaign, not a wave.
- C7. Test scaffolding (additive only, zero churn to 7,136 green): (a) rootdir path
  config (pytest.ini/pyproject or conftest) kills the 70-file sys.path.insert
  copy-paste as one-line diffs; (b) ONE shared stub-LLM fixture in conftest --
  MANDATORY for new tests, existing tests migrate only when already touched.
  conftest.py:44-70 already documents the materialized order-dependence bug that
  hand-rolled stubs caused; extend that lesson, don't restructure. DO NOT merge the
  many small single-assertion test files -- fragmentation is cosmetic; merging is churn
  with regression risk and no behavior gain.

## 6. Structural Campaign SW (the two giants become designed-on-purpose)

- SW-1. OTR_LedgerScriptWriter.py (VERIFIED-2026-07-10: now 6,876 LOC after S5a; the
  S1 policy threading moved the slot-plumbing/seeding seam boundaries -- _SlotScheduler
  still :422 but downstream grew; RE-SURVEY all seam ranges before cutting, per R-4)
  -> package nodes/_otr_writer/ with
  OTR_LedgerScriptWriter.py remaining as a FACADE re-exporting the class (node
  registration + ledger-stamped import paths stay byte-identical):
  slot_plumbing.py (today :345-766 -- NOTE R-4: post-S1 this code has already changed;
  re-survey), seeding.py (:767-1416 title-gen + input/RSS-seed resolution),
  original_radio.py (:1417-2071), widgets.py (see SW-2), writer_node.py (INPUT_TYPES
  shell + run() only).
  run(): extract the HALVES, keep the spine -- line-request building and the refine
  machinery become pure functions with explicit parameters; the beat loop stays as the
  one visible ~1000-line spine. The God-method's sin is implicit state threading, not
  sequence. Full shatter mid-campaign is how a local gets lost. 80% of the win, 20% of
  the risk.
- SW-2. INPUT_TYPES: from a 659-line imperative method (VERIFIED-2026-07-10 now
  :2115-2773 after S5a's six widgets) to a module-level ordered WIDGETS table (name,
  type, default, tooltip, choice-builder) + a ~20-line renderer. The
  append-only law becomes a MACHINE CHECK: a frozen name-order manifest test that fails
  on any non-append edit. (Portability added widgets 28-33 by hand-count; after SW-2
  the next addition is a table row.) BUG-LOCAL-097 stops being folklore.
- SW-3. story_orchestrator.py (3,034) stops existing as a name. Split by subsystem:
  news_ingest.py (:1267-1873, fully self-contained -- cleanest first cut),
  cast_naming.py (:408-821 + :2150-2338), qa_filters.py (:822-1266), dialogue_post.py
  (:2070+); timeout/VRAM infra merges into the existing runtime layer, never a new
  "misc". PRECONDITION: W1's shadowed-twin resolution ships first (own commit) so a
  dead twin can't resurrect into a new module.
- SW-4. Sentinel widgets ("(enable OpenRouter)" x2 idx17-18, "(enable Comfy Credits)"
  x2 idx19-20, "(select Google API model)" x2 idx25-26 -- grounded): error messages
  cosplaying as data. Replacement: real enum value `disabled` + tooltip naming the env
  key + queue-time fail-loud raise naming that key. VALUES change, positions don't --
  safe under append-only; rides the W5 epoch commit or portability S5 (profiles own
  those slot values post-portability), whichever executes first.

## 7. Operator Decision Boxes -- ALL SIX RATIFIED (operator, 2026-07-10)

The operator approved every default below, with one codicil on D-2 (see it). The
campaign is now fully cleared to execute post-portability; remaining execution-time
gates are only R-4 (seam re-survey) and R-7 (re-grep before cutting).

- D-1. W5 window: DEFAULT post-portability (your sequencing lock; variants regenerate
  mechanically). ALTERNATIVE the panel prefers on cost: pre-S5, while writer=28 widgets
  and zero variants exist. Choosing it resequences portability -- your call.
- D-2. RTXUpscale rip vs wire: panel UNANIMOUS rip (composite chain owns 1080p). It is
  922 LOC you once built -- eyeball before the epoch commit.
  RATIFIED WITH CODICIL (operator 2026-07-10): rip NOW; a FUTURE system-agnostic
  multi-GPU upscale campaign is planned for after this rip lands. Design constraints
  recorded for that campaign: (1) it is a REBUILD against the portability stack --
  registry rows with device_backends per engine (nvidia/amd/mps/cpu), per-tier profile
  values, fail-loud on unsupported hardware -- NOT a resurrection of the vendor-locked
  RTX-VSR node (git keeps the reference); (2) it plugs into the EXISTING name-only
  `upscale_stage` profile reservation the portability plan deliberately kept (panel
  tried to cut it 3x); (3) HONEST-SWITCH LAW: its widgets/profile fields land in the
  SAME COMMIT as a working engine, `off` is a real enum value, selecting an unsupported
  engine raises -- no cockpit switch ever again precedes or outlives its machine (the
  spacesaver lesson); (4) new widgets APPEND at the end per positional law, variants
  regenerate via build_variants.py.
- D-3. scripts/ hybrid (delete concluded / dev/ for lab tooling / live stays): ratify
  the split list at execution; anything you still run moves at most one directory.
- D-4. Tombstone age-gate policy (section 1.3): adopt as standing law?
- D-5. W8 OpenRouter diet: approve after cloud-lane smoke, per gate.
- D-6. VRAMGuardian expiry: dies at portability ship, or debug-gate registration?

## 8. Judge's Corrections to the Panel (what grounding caught -- method note)

- test_video_platform_aseam.py is NOT wholesale-deletable (~98% live coverage; surgery
  only). The deletion-first take overreached; corrected in W4b.
- test_image_platform_c1.py does NOT import visual/ -- the claimed surgery there is
  unnecessary; the REAL visual-importing tests are prompt_coercion_contract, wedge_probe,
  b5_visual_polish_collapse (handled in W4a).
- _otr_probe.py "adopt as-is" (boundary take) loses to adopt-AND-EXTEND: its API is
  duration-only with no binary resolution; C5 builds the real boundary and supersedes it.
- ffprobe duplication is ~16 implementations across ~11 files, not 3 -- C5 upgraded from
  nice-to-have to a named campaign.
- README CRT-floor claim requires verify-then-fix (W0a), not blind deletion: confirm
  whether a sanctioned floor path survives before rewording.
- R4 FABLE GATE (7 must-fixes, all folded in): node-1 inputs[]/dst_slot second
  positional axis on W5 (render-breaker class); test_filename_pattern_audit reads
  rtx_upscale.py as text; SEVEN live test files import the W3 kill list (slot_matrix +
  content_oracle re-adjudicated as test-support); module-scope image-scaffold imports
  in two live platform test files (W2); W6 lacked a test-delta table and its keep-list
  missed the headless soak launcher chain this plan's own gates depend on; C2 vestige
  blast radius corrected. Gate verdict: CONVERGED after must-fixes -- no new round.
- CODE-READINESS VERIFICATION (2-agent sweep vs HEAD `20185542`, post-portability,
  same night): 48 claims CONFIRMED; 5 stale line-cites re-pointed (W1 harness
  :6474-6876; guardrails slot-9 pin :773; companions third pin :240; SW-1/SW-2
  ranges); 4 corrections folded (bare-except count is 2 not 4; W2 test imports are
  function-scoped; W6 keep-list gained build_variants.py + the installer trio; dead
  sidecar.py has zero importers at all). W0a resolved: CRT floor is the wired
  OTR_SignalLostVideo node -> REWORD not delete. W5's dst_slot obligation quantified:
  remove inputs[9], renumber link 279 dst_slot 34->33 (only live link into node 1).
  S1's llm_policy.py is a live down payment on C6's canonicalize-at-boundary pattern.

## 9. Risk Register (merged panel + judge; each risk names its catching guard)

1. W5 positional shift on an un-regenerated graph/variant -> widget-count-vs-INPUT_TYPES
   audit + guardrail slot pins + validator refusing old layouts + one live 30w smoke.
2. Sidecar name collision (_otr_shared/sidecar.py dead vs _otr_audio_engines/
   _otr_sidecar.py LIVE) -> full-path kill lists; test_tts_engine_sidecars fails
   instantly if wrong file dies; headless boot before push.
3. String/dynamic imports invisible to grep -> headless boot (real registration import
   chain) is the true gate on W2/W3/W4; boot before every push.
4. Tombstone retirement + multi-window resurrection (documented repo failure mode) ->
   single denylist file + Bug Bible entry per rip + age gate.
5. One-shape flip turning tolerated dicts into mid-episode crashes masked by the 709
   except-Exception sites -> producer-first order + per-shape 30-word smoke + ratchet.
6. Writer facade breaking ledger-stamped import paths -> facade keeps byte-identical
   import surface; canonical-JSON contract validation in the same commit.
7. Deleting an orphan a parked plan expects (cloud_media_cache vs cloud-audio S1) ->
   W3 exception (c): diff kill list against parked converged plans at execution.
8. Over-sanitizing kills lab velocity -> scripts/dev/ split not deletion for lab
   tooling; kibitz launchers pinned in place (external skill dependency).
9. Suite shrink hides real coverage loss -> every test deletion lists what contract now
   covers the surface (or states "surface deleted with it"); aseam surgery pattern is
   the template.
10. This doc's line numbers rot as portability lands -> R-7: re-grep before every cut.

## 10. Execution Order (post-portability; each chunk = commit+push, suite+Bible green)

W0 (doctrine fixes) -> W1 (inert text + F811 guard) -> W2 (dark engines) -> W3 (orphan
modules) -> W4a (visual/) -> W4b (unwired nodes + aseam surgery) -> W7 (tombstone
consolidation, absorbing W2-W4 rows) -> W6 (scripts split) -> W5 (RTXUpscale epoch +
SW-4 sentinels if not already ridden with S5) -> C1-C5 (family consolidations) -> SW-1/
SW-2/SW-3 (giants; re-survey seams post-S1 first) -> C6 (one-shape ratchet, ongoing) ->
C7 (test scaffolding, additive) -> W8 (OpenRouter diet, after cloud smoke).
Estimated 12-16 working sessions at this repo's demonstrated chunk cadence. Every
session ends pushed to v2.0-alpha; the operator eyeball gates tags only.
