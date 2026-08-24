# `scripts/` -- the owner table (lean-mean order 11, 2026-08-23)

**THIS IS NOT A KILL LIST, and the campaign says so in its own exit condition:**
*"Each deletion has an owner/caller/test record and accepted loss; no active
bench, doctor, recovery, render path, or protocol-specific fixture
disappears."* Order 11's scripts half is documentation-shaped by design. Nothing
here has been deleted. Four files are put to the operator with the evidence
already gathered, and everything else is recorded so the next window does not
re-derive it.

The old bulk kill list is explicitly forbidden by the campaign
(`docs/LEAN_MEAN_CLEANUP.md` section 2.4, `scripts/` row: *"Do not reuse the old
bulk kill list; active bakeoff, doctor, render, soak, and recovery tools are
protected until proven otherwise"*). This table replaces it with per-file
evidence.

## How the evidence was gathered, and where it can lie

Every file under `scripts/` was counted against a corpus of `nodes/`, `tests/`,
`scripts/`, `docs/`, `workflows/`, `config/`, `.github/` and `kibitz-runs/` --
by filename, by repo-relative path, by both slash conventions, and (for `.py`)
by import spelling. Git supplied the last-touch date and tracked status.

**THE METHOD'S BLIND SPOT, stated up front because it fired once.** A filename
scan cannot see a script invoked through a constructed path or a glob, so a
zero-reference row is a QUESTION, never a verdict. The reverse also happened:
`watcher_overrides.json` scores zero references AND its own `_schema` string
says *"See soak_operator.py:apply_watcher_overrides for the full allowlist"* --
a function that **no longer exists in that file**. So its zero is real, and the
config it documents has no reader. That is the single clearest finding in this
pass, and it came from following the file's own words rather than the count.

## The shape of it

| class | files | what it means |
|---|---:|---|
| WIRED | 95 | named by code or by a test -- a caller exists |
| DOCUMENTED ONLY | 27 | named only by docs/handoffs; the doc IS the caller |
| UNREFERENCED, TRACKED | 11 | the only real questions -- one row each below |
| UNTRACKED | 34 | operator working files; not shipped, not in git |

The 34 untracked files are **not the pack's problem** -- they are never
published (`.comfyignore` and git tracking both exclude them) and they are the
operator's own bench scratch. They are listed for completeness only. Eight are
`_tmp_google_all_real*` run logs and **four of those are zero bytes**.

## UNREFERENCED AND TRACKED -- the eleven, with a verdict each

**KEEP, and the zero is expected.** These are run BY A HUMAN or by a standing
rule, never by code, so a reference count was never going to find them:

* `otr_reset_gpu_box.ps1` (2026-08-20) -- **implements CLAUDE.md section 4**,
  the selective reset every headless run is required to perform. Its own header
  says "SELECTIVE BY COMMANDLINE, NEVER A BLANKET PYTHON KILL". A recovery tool
  is explicitly protected by the campaign's exit condition.
* `otr_video_lane_sweep.sh` (2026-08-22) -- one-act live coverage sweep across
  the per-engine video profiles. That is item F's shape and item F is OPEN.
* `otr_ideogram4_refusal_repro.py` (2026-08-22) -- a CONTAINED repro for the
  seed-dependent Ideogram music-card refusal, which is the half of item B that
  is still open. The repro for a live open question is not debt.
* `otr_gender_blind_control.py` and `otr_gender_probability_lab.py`
  (both 2026-08-15) -- the two instruments built to measure a model BEFORE
  trusting it against 132 shipped gender pins. Voice/character gender remains an
  operator-eyeball item, so the instruments stay with it.
* `serve_ledger.py` (2026-04-24) -- a tiny HTTP viewer that shows the current
  production ledger in a browser. Old, dependency-free, and the sort of thing
  that is only missed the day it is gone.
* `bark_artifact_scan.py` (2026-06-21) -- the high-band squeal QA scan. Its
  dependency is **live and tested**: `nodes/_otr_bark_lib.high_band_edge_ratio`
  exists and `tests/test_bark_artifact_metric.py` imports it. Working
  instrument, not an orphan.
* `build_ltx_av_q_bakeoff_workflow.py` (2026-07-07) -- the GENERATOR for the
  ltx_av_q bake-off graph. Its output (`otr_ltx_av_q_bakeoff_distilled_native.json`)
  sits beside it and its consumer `run_ltx_av_q_bakeoff.py` is named by 28
  documents. Deleting a generator while keeping its output is how a graph
  becomes unreproducible.

**PUT TO THE OPERATOR -- three, and only three. TWO WERE RULED ON THE SAME DAY:**

1. **`watcher_overrides.json`** -- **DELETED, 2026-08-23.** Operator: *"soak op
   delete, we'll make a new soak op."* The audit's finding was that its declared
   reader `soak_operator.py:apply_watcher_overrides` no longer exists; the
   reason turned out to be that BUG-LOCAL-002 gutted that 1,500-line runner back
   on 2026-05-02 and the config was never removed with it. It went with the shim.
2. **`otr_hazard.py`** -- **DELETED, 2026-08-23.** Operator: *"don't worry, rip
   it out, it was my idea to help things."* The "BIG RED LIGHT: this tree is
   mid-surgery" banner outlived its surgery.

   **And the third file the ruling reached, which this audit had classed WIRED:**
   `soak_operator.py` itself. It was a LEGACY SHIM -- 304 lines holding exactly
   ONE function, `scan_treatment`, kept alive only by
   `tests/test_treatment_scanner_unicode.py`. Its own docstring named the right
   home (*"prefer adding to scripts/treatment_scanner.py"*) and that module had
   never been created. So the function moved there BYTE-IDENTICALLY (sha256
   `9c0a4c1dedc87de2c5cc6ce49a22eea93d65ab2be4782ca828219075210cd83e`), the test
   imports it from its new address, and the shim is gone. The BUG-LOCAL-033 fix
   it carries -- accepting U+2500 separators and U+2192 cast arrows -- is
   untouched, because a scanner that loses it starts a false-positive flag storm
   on every real treatment.

3. **`_consult_question_ltx23_res4lyf.md`** (2026-07-07, 8 KB) -- a consult
   question about integrating LTX 2.3 + RES4LYF, addressed to an outside reader.
   The LTX 2.5 lane shipped since. Kept or archived, it is a document, not code.

## The standalone ffprobe callers -- order 8's deliberate remainder

Order 8 consolidated ffprobe resolution for the eleven **node/runtime** callers
and stopped at the repo edge on purpose: *"Leave standalone script callers
unchanged until their order-11 owner audit proves a cold-import-safe
adoption."* This is that audit, and the answer is that **every one of them is a
live tool** -- not one is an orphan, and several are among the most
heavily-documented files in `scripts/`:

| script | tests | docs | last touched |
|---|---:|---:|---|
| `audit_otr_full_run.py` | 2 | 25 | 2026-05-08 |
| `otr_h3_mime_runner.py` | 1 | 2 | 2026-08-12 |
| `otr_macbeth_probe.py` | 2 | 11 | 2026-08-08 |
| `otr_measure_av_offset.py` | 0 | 7 | 2026-08-02 |
| `otr_talking_radio_probe_eval.py` | 1 | 4 | 2026-07-02 |
| `otr_w45_campaign.py` | 1 | 27 | 2026-08-14 |
| `render_episode_concat.py` | 1 | 0 | 2026-05-02 |
| `render_humo_batch.py` | 4 | 1 | 2026-05-10 |
| `run_humo_bakeoff.py` | 0 | 15 | 2026-07-09 |
| `run_ltx_av_q_bakeoff.py` | 0 | 28 | 2026-07-03 |
| `run_video_arm_bakeoff.py` | 2 | 39 | 2026-08-02 |
**WHAT THAT MEANS FOR THE ADOPTION, and it is the opposite of a cleanup.**
These eleven are not candidates for deletion, so migrating them to
`nodes/_otr_shared/ffprobe.py` is a real code chunk with a real risk: a
standalone script that imports from `nodes/` acquires the pack's import graph,
and several of these run OUTSIDE a ComfyUI process. The boundary module is
stdlib-only and cold-import clean by construction (proven in a subprocess by
`tests/test_ffprobe_boundary.py`), so the adoption IS safe -- but it should be
its own chunk with its own suite run, not a tail bolted onto this audit.

**One concrete win is already known and worth carrying into that chunk:**
`run_video_arm_bakeoff.py:1430-1435` carries its own copy of the rational
frame-rate split, and it is one of the three copies order 8 named. The others
are now gone.

## DOCUMENTED ONLY -- named by handoffs, not by code

A doc naming a script is a caller: it is how the operator finds the tool six
weeks later. `otr_check.bat` is named by **25** documents, `run_video_arm_bakeoff.py`
by 39, `run_ltx_av_q_bakeoff.py` by 28, `otr_w45_campaign.py` by 27. None of
these are candidates for anything.

| script | size | docs naming it | last touched |
|---|---:|---:|---|
| `otr_check.bat` | 1 KB | 25 | 2026-07-24 |
| `otr_vendor_public_domain_library.py` | 51 KB | 18 | 2026-08-04 |
| `otr_writer_bank_gate.py` | 11 KB | 12 | 2026-08-16 |
| `otr_dl_indextts2_refs.py` | 27 KB | 11 | 2026-08-18 |
| `otr_measure_av_offset.py` | 38 KB | 7 | 2026-08-02 |
| `_otr_b_spikes` | 51 KB | 5 | 2026-06-07 |
| `otr_gemma4_doctor.py` | 6 KB | 5 | 2026-07-20 |
| `otr_queue_smoke.py` | 3 KB | 4 | 2026-08-14 |
| `_otr_style_authority_smoke.py` | 18 KB | 3 | 2026-08-17 |
| `build_story_only.py` | 3 KB | 3 | 2026-07-16 |
| `download_ltx_2_3.ps1` | 4 KB | 3 | 2026-05-11 |
| `otr_ltx_mad.py` | 2 KB | 3 | 2026-06-12 |
| `otr_ltx_motion_smoke.py` | 9 KB | 3 | 2026-08-17 |
| `otr_name_randomness_lab.py` | 6 KB | 3 | 2026-08-15 |
| `otr_openrouter_refresh.py` | 2 KB | 3 | 2026-06-01 |
| `otr_voice_identity_2x2.ps1` | 8 KB | 3 | 2026-08-18 |
| `_consult_openai.py` | 15 KB | 2 | 2026-06-29 |
| `run_agy_agent.ps1` | 4 KB | 2 | 2026-06-27 |
| `download_models.sh` | 6 KB | 1 | 2026-05-01 |
| `otr_build_obs_listen_page.py` | 10 KB | 1 | 2026-08-18 |
| `otr_run_watcher.ps1` | 5 KB | 1 | 2026-07-11 |
| `otr_story_score.py` | 13 KB | 1 | 2026-08-16 |
| `otr_suite_ltx_verify.ps1` | 230 B | 1 | 2026-06-14 |
| `otr_tail_logs.py` | 4 KB | 1 | 2026-05-31 |
| `otr_title_identity_acceptance.py` | 6 KB | 1 | 2026-08-17 |
| `otr_w45_overnight.py` | 10 KB | 1 | 2026-08-14 |
| `test_prompt_import_isolation.py` | 6 KB | 1 | 2026-07-24 |

## WIRED -- a caller exists in code or tests

| script | tests | code callers | docs | last touched |
|---|---:|---|---:|---|
| `build_variants.py` | 4 | `boot_contracts.py`, `capability_profiles.py` | 148 | 2026-08-16 |
| `otr_api.py` | 13 | `_otr_workflow_apply.py`, `_otr_workflow_validator.py` | 123 | 2026-08-14 |
| `_otr_soak_server_launch.cmd` | 4 | `boot_contracts.py`, `_otr_w45_boot.ps1` | 87 | 2026-08-22 |
| `otr_canonical_api_run.py` | 3 | `_tmp_d1_live_legs.ps1`, `otr_gpu_soak_matrix.py` | 85 | 2026-08-23 |
| `render_humo_batch.py` | 4 | `eng_humo.py`, `production_ledger.py` | 1 | 2026-05-10 |
| `audit_otr_full_run.py` | 2 | `_otr_ledger.py`, `audit_spoken_citations.py` | 25 | 2026-05-08 |
| `otr_g1_lemmy_audition.py` | 1 | `cast_pools.py`, `_otr_evidence_citations.py` | 29 | 2026-08-18 |
| `otr_pin_partner_nodes.py` | 2 | `eng_cloud_image.py`, `cloud_media_invoke.py` | 1 | 2026-07-09 |
| `_otr_headless_model_paths.yaml` | 2 | `eng_ltx_av.py`, `_otr_soak_server_launch.cmd` | 7 | 2026-08-19 |
| `_otr_indextts2_worker.py` | 2 | `eng_indextts2.py`, `_otr_voice_route.py` | 10 | 2026-06-05 |
| `ensure_upscale_models.py` | 2 | `__init__.py`, `eng_spandrel_esrgan.py` | 19 | 2026-08-17 |
| `otr_headless_canonical.ps1` | 1 | `_tmp_run_adapter_2x2.ps1`, `_tmp_run_bakeoff_bank.ps1` | 41 | 2026-08-15 |
| `otr_style_traceroute.py` | 2 | `eng_ltx_video.py`, `eng_wan_i2v.py` | 11 | 2026-08-17 |
| `run_ltx_av_q_bakeoff.py` | 0 | `otr_image_gen_dispatcher.py`, `build_ltx_av_q_bakeoff_workflow.py` | 28 | 2026-07-03 |
| `vram_context_test.py` | 3 | `gpu_residency.py`, `vram_context_test.py` | 9 | 2026-04-29 |
| `audit_spoken_citations.py` | 2 | `_otr_ledger.py`, `audit_voice_gender_consistency.py` | 41 | 2026-08-07 |
| `bench_graphs` | 0 | `dmd_sampler.py`, `eng_fastwan_8gb.py` | 12 | 2026-07-31 |
| `bench_helper` | 2 | `dmd_sampler.py`, `run_video_arm_bakeoff.py` | 13 | 2026-07-31 |
| `hf_download_driver.py` | 1 | `download_4060_nano_models.ps1`, `download_ltx_0_9_8.ps1` | 14 | 2026-07-20 |
| `otr_fetch_public_domain.py` | 2 | `_otr_roster_gender.py`, `otr_stamp_character_genders.py` | 29 | 2026-08-15 |
| `otr_ledger_view.py` | 2 | `_otr_spoken_text_policy.py`, `otr_clean_stage_lab.py` | 8 | 2026-08-14 |
| `otr_render_watchdog.ps1` | 1 | `dmd_sampler.py`, `run_4060_8gb_suite.ps1` | 14 | 2026-07-13 |
| `render_flux_batch.py` | 1 | `flux_gen1.py`, `render_episode_concat.py` | 1 | 2026-05-10 |
| `run_video_arm_bakeoff.py` | 2 | `eng_fastwan_8gb.py`, `otr_rotate_log.ps1` | 39 | 2026-08-02 |
| `_otr_evidence_citations.py` | 2 | `otr_lemmy_cross_engine_audition.py` | 4 | 2026-08-18 |
| `_otr_idx_download_weights.py` | 1 | `eng_indextts2.py`, `_otr_indextts2_install.ps1` | 0 | 2026-07-10 |
| `_otr_indextts2_install.ps1` | 1 | `eng_indextts2.py`, `_otr_idx_download_weights.py` | 3 | 2026-07-10 |
| `build_silent_test_episode.py` | 1 | `production_ledger.py`, `render_flux_batch.py` | 14 | 2026-04-25 |
| `grade_episode.py` | 2 | `acceptance.py` | 31 | 2026-08-06 |
| `otr_audio_dep_pilot.py` | 1 | `otr_image_dep_pilot.py`, `otr_video_dep_pilot.py` | 3 | 2026-06-29 |
| `otr_lemmy_listen_page.py` | 1 | `_otr_evidence_citations.py`, `otr_g1_listen_page.py` | 11 | 2026-08-18 |
| `otr_ltx_av_q_bakeoff_distilled_native.json` | 0 | `build_ltx_av_q_bakeoff_workflow.py`, `run_humo_bakeoff.py` | 5 | 2026-06-27 |
| `otr_macbeth_probe.py` | 2 | `validate_canonical_workflow.py` | 11 | 2026-08-08 |
| `otr_video_dep_pilot.py` | 1 | `otr_image_dep_pilot.py`, `otr_video_gpu_smoke.py` | 7 | 2026-06-30 |
| `otr_w45_campaign.py` | 1 | `otr_hazard.py`, `otr_w45_overnight.py` | 27 | 2026-08-14 |
| `run_wan_ti2v_bakeoff.py` | 0 | `otr_rotate_log.ps1`, `otr_wan_ti2v_bakeoff_gguf.json` | 25 | 2026-07-07 |
| `soak_operator.py` | 1 | `otr_api.py`, `watcher_overrides.json` | 16 | 2026-08-06 |
| `_otr_chatterbox_install.ps1` | 0 | `eng_chatterbox.py`, `_otr_indextts2_install.ps1` | 2 | 2026-06-05 |
| `_otr_chatterbox_worker.py` | 1 | `eng_chatterbox.py` | 2 | 2026-07-10 |
| `_otr_dia_worker.py` | 1 | `eng_dia.py` | 2 | 2026-07-09 |
| `_otr_w45_boot.ps1` | 0 | `otr_rotate_log.ps1`, `otr_w45_overnight.py` | 11 | 2026-08-01 |
| `audit_voice_gender_consistency.py` | 1 | `grade_episode.py` | 29 | 2026-08-06 |
| `audit_wrong_person_census.py` | 1 | `_otr_name_authority.py` | 14 | 2026-08-20 |
| `download_ltx_0_9_8.ps1` | 0 | `eng_ltx_8gb.py`, `download_4060_nano_models.ps1` | 8 | 2026-07-20 |
| `otr_check.py` | 1 | `otr_check.bat` | 28 | 2026-07-24 |
| `otr_gpu_soak_matrix.py` | 1 | `build_variants.py` | 12 | 2026-08-22 |
| `otr_h3_mime_runner.py` | 1 | `otr_zimage_reference_ab.py` | 2 | 2026-08-12 |
| `otr_headless_process.psm1` | 1 | `otr_headless_canonical.ps1` | 1 | 2026-07-13 |
| `otr_lemmy_cross_engine_audition.py` | 1 | `cast_pools.py` | 11 | 2026-08-18 |
| `otr_lemmy_production_audition.py` | 1 | `cast_pools.py` | 17 | 2026-08-18 |
| `otr_mesh_stage_blender.py` | 1 | `eng_mesh_stage.py` | 1 | 2026-06-30 |
| `otr_stamp_character_genders.py` | 0 | `_otr_roster_gender.py`, `otr_fetch_public_domain.py` | 13 | 2026-08-15 |
| `otr_talking_radio_probe_eval.py` | 1 | `eng_ltx_video.py` | 4 | 2026-07-02 |
| `otr_video_gpu_smoke.py` | 1 | `build_humo_bakeoff_workflow.py` | 4 | 2026-07-03 |
| `otr_video_soak.py` | 1 | `build_humo_bakeoff_workflow.py` | 7 | 2026-08-23 |
| `otr_visual_smoke.py` | 1 | `otr_video_render_batch.py` | 4 | 2026-08-12 |
| `otr_voice_identity_acceptance.py` | 1 | `otr_voice_identity_2x2.ps1` | 8 | 2026-08-18 |
| `otr_zimage_reference_ab.py` | 1 | `z_image_turbo.py` | 5 | 2026-08-21 |
| `profile_scope_render.py` | 2 | -- | 14 | 2026-07-08 |
| `validate_canonical_workflow.py` | 2 | -- | 29 | 2026-08-09 |
| `verify_google_slugs.py` | 1 | `google_slug_verifier.py` | 13 | 2026-08-10 |
| `_consult_nvidia.py` | 0 | `_consult_round_robin.py` | 2 | 2026-05-01 |
| `_consult_round_robin.py` | 0 | `_consult_nvidia.py` | 1 | 2026-05-22 |
| `_otr_a_s2_probes` | 0 | `README.md` | 1 | 2026-07-03 |
| `_otr_dia_install.ps1` | 0 | `eng_dia.py` | 3 | 2026-07-09 |
| `_otr_lumina_image_smoke.py` | 0 | `_otr_style_authority_smoke.py` | 1 | 2026-08-17 |
| `_otr_mirror_clone_refs.py` | 1 | -- | 19 | 2026-08-16 |
| `_otr_single_engine_smoke.py` | 0 | `otr_humo_vram_ladder.py` | 31 | 2026-08-12 |
| `bark_preset_audition.py` | 1 | -- | 10 | 2026-06-17 |
| `build_humo_bakeoff_workflow.py` | 0 | `run_humo_bakeoff.py` | 1 | 2026-07-07 |
| `build_video_evidence_manifest.py` | 0 | `motion_common.py` | 13 | 2026-08-22 |
| `download_humo_models.ps1` | 1 | -- | 7 | 2026-07-10 |
| `download_video_stack_weights.ps1` | 0 | `hf_download_driver.py` | 4 | 2026-04-18 |
| `kill_otr_zombies.ps1` | 0 | `otr_reset_gpu.ps1` | 5 | 2026-04-21 |
| `otr_campaign_receipt.ps1` | 1 | -- | 2 | 2026-07-23 |
| `otr_clean_stage_lab.py` | 0 | `_otr_ledger_clean.py` | 6 | 2026-08-16 |
| `otr_cloud_s0_smoke.py` | 0 | `otr_macbeth_probe.py` | 9 | 2026-07-06 |
| `otr_g1_listen_page.py` | 1 | -- | 1 | 2026-08-18 |
| `otr_gender_secondopinion_lab.py` | 0 | `otr_gender_blind_control.py` | 0 | 2026-08-15 |
| `otr_humo_vram_ladder.py` | 0 | `otr_rotate_log.ps1` | 9 | 2026-08-02 |
| `otr_ia2v_server_boot.cmd` | 0 | `otr_rotate_log.ps1` | 7 | 2026-07-02 |
| `otr_image_dep_pilot.py` | 1 | -- | 2 | 2026-06-07 |
| `otr_ingest_pd_voices.py` | 1 | -- | 13 | 2026-08-05 |
| `otr_ltx25_encoder_load_audit.py` | 1 | -- | 9 | 2026-08-20 |
| `otr_ltx25_two_stage_audit.py` | 1 | -- | 10 | 2026-08-20 |
| `otr_reset_gpu.ps1` | 0 | `otr_voice_identity_2x2.ps1` | 4 | 2026-08-17 |
| `otr_rotate_log.ps1` | 0 | `_otr_soak_server_launch.cmd` | 16 | 2026-08-04 |
| `otr_tree_doctor.py` | 1 | -- | 0 | 2026-06-11 |
| `otr_verify_voice_cast_mode.py` | 1 | -- | 3 | 2026-08-18 |
| `otr_wan_smoke.py` | 0 | `eng_wan_i2v.py` | 3 | 2026-07-03 |
| `otr_wan_ti2v_bakeoff_gguf.json` | 0 | `run_wan_ti2v_bakeoff.py` | 10 | 2026-07-07 |
| `render_episode_concat.py` | 1 | -- | 0 | 2026-05-02 |
| `run_codex_agent.ps1` | 0 | `run_agy_agent.ps1` | 2 | 2026-06-27 |
| `run_humo_bakeoff.py` | 0 | `otr_rotate_log.ps1` | 15 | 2026-07-09 |
| `setup_cloud.sh` | 0 | `download_models.sh` | 4 | 2026-05-01 |

## UNTRACKED -- operator working files, never shipped

Not in git, excluded from the published pack. Listed for completeness; four of
the `_tmp_google_all_real*` logs are zero bytes.

| file | size |
|---|---:|
| `_otr_canonical_api_prompt.json` | 10 KB |
| `_otr_canonical_api_prompt_fastwan30.json` | 10 KB |
| `_otr_canonical_api_prompt_live_30word.json` | 9 KB |
| `_otr_prompt_g4_probe.json` | 10 KB |
| `_otr_prompt_w45_probe.json` | 10 KB |
| `_otr_ref_audit_report.json` | 26 KB |
| `_otr_upscale_checkpoint_smoke.py` | 14 KB |
| `_tmp_d1_live_legs.ps1` | 3 KB |
| `_tmp_google_all_real.err.log` | 0 B |
| `_tmp_google_all_real.out.log` | 2 KB |
| `_tmp_google_all_real_25flash.err.log` | 0 B |
| `_tmp_google_all_real_25flash.out.log` | 2 KB |
| `_tmp_google_all_real_31lite.err.log` | 0 B |
| `_tmp_google_all_real_31lite.out.log` | 2 KB |
| `_tmp_google_all_real_latest.err.log` | 0 B |
| `_tmp_google_all_real_latest.out.log` | 2 KB |
| `_tmp_google_all_real_retry.err.log` | 2 KB |
| `_tmp_google_all_real_retry.out.log` | 949 B |
| `_tmp_launch_bakeoff_bank.ps1` | 577 B |
| `_tmp_launch_bakeoff_queue.ps1` | 502 B |
| `_tmp_launch_bakeoff_tail_queue.ps1` | 499 B |
| `_tmp_launch_hy3_tests.ps1` | 490 B |
| `_tmp_run_adapter_2x2.ps1` | 3 KB |
| `_tmp_run_bakeoff_bank.ps1` | 520 B |
| `_tmp_run_bakeoff_queue.ps1` | 1 KB |
| `_tmp_run_bakeoff_tail_queue.ps1` | 1 KB |
| `_tmp_test_hy3_dropdown.ps1` | 589 B |
| `_tmp_video_art_google_dryrun.json` | 9 KB |
| `_tmp_video_art_local1_prompt.json` | 9 KB |
| `_tmp_video_art_local_dryrun.json` | 9 KB |
| `download_4060_nano_models.ps1` | 2 KB |
| `run_4060_8gb_suite.ps1` | 6 KB |
| `test_4060_h3.ps1` | 2 KB |
| `test_4060_nano.ps1` | 2 KB |

## What this pass did NOT do

No file was deleted, moved or renamed. The test-root half of order 11 stays
closed on its own evidence: there is no repo-root `conftest.py` and no
`pythonpath` key in `pyproject.toml`, so the 175 per-file `sys.path.insert`
calls across the test tree are the LOAD-BEARING import mechanism rather than
redundancy. That row's premise was false at HEAD and the plan already says so.

**Two of the three were ruled on the day this table was written** and are
deleted (`watcher_overrides.json`, `otr_hazard.py`), taking the
`soak_operator.py` shim with them and promoting its one live function to
`scripts/treatment_scanner.py`. `_consult_question_ltx23_res4lyf.md` is still
open. Everything else is recorded, and the ffprobe adoption for standalone scripts is a real chunk with a
real gate, not a sweep.
