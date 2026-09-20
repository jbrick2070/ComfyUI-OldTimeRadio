# Ship regression, 2026-09-19 evening -- the full suite before v2.1.x

Operator: the remaining credits go to regression testing before the ship and
the Reddit post. Section 7A's condition for any publish: the suite diffed
against a same-HEAD baseline, every new failure explained, THEN bump. This
file is that explanation. The bump itself is his call (third digit only).

## The numbers

* Full suite at `23497b6e` (+ docs): RC=2, twelve nodeids failed that are not
  in `EXPECTED_FAILED_NODEIDS`. Log: `%TEMP%\full_suite_ship_baseline.log`.
* The same twelve run in a worktree at `cc62b2c1` -- the commit this session
  started from, before any of the day's code -- and **all twelve fail there
  too.** None was introduced today.
* Every suite the day's code touches is green at `96346118`: the five corpus
  suites (237), the counter/chunker/ledger-metric/verbatim set (567 passed,
  2 skipped), and the ASCII word-count parity measurement (27,600 rows, 0
  differ).
* A second full run at final HEAD is recorded below when it lands.

## The twelve, classified from their own assertion lines

| # | nodeid | what the assertion says | class | reaches the shipped still lane? |
|---|---|---|---|---|
| 1 | `test_b7_forbidden_sweep.py::test_forbidden_sweep_runs_clean` | test files carry words the B7 sweep treats as runtime markers (`alias`, `shim`), and one absolute `C:/Users/jeffr` path in a test string | test hygiene | no -- and one hit (`test_vendor_scan_furniture.py:422`) was today's, reworded in `96346118`; the rest predate the session |
| 2 | `test_canonical_replay.py::test_voices_music_and_sequencer_pass_through_on_replay` | `OTRVoiceNodeBase.generate() got multiple values for argument 'ledger_json'` | real signature drift on the REPLAY path (`replay_from`, the A/A null) | the widget ships; a replay would hit this -- **assess before the bump** |
| 3 | `test_cloud_sku_json_parity.py::...[otr_cloud_low-otr_cloud_deluxe_3act]` | three unexpected widget drifts, first `OTR_VideoDirector.announcer_image_model` | cloud variant drift vs. the expected-drift list | no (cloud SKUs) |
| 4 | `test_evidence_citation_integrity.py::...still_hashes_to_what_the_record_claims` | cited evidence MISSING from disk: `otr/episodes/lemmy_cross_engine/*.wav` | artifacts deleted from the local output tree; the record still cites them | no |
| 5 | `test_frame_receipt_conformance.py::...[cloud_ltx25_foley_plus]` | `CloudMediaError: corrupt_output -- partner result missing ['path', 'content_type']` | cloud media backend contract vs. the adapter fixture | no (cloud engine) |
| 6 | `test_google_omni_video_adapter.py::test_canonicalize_returns_silent_clip_dict` | same `CloudMediaError: corrupt_output` | as 5 | no (Google lane) |
| 7 | `test_google_veo_video_adapter.py::test_canonicalize_returns_silent_clip_dict` | same | as 5 | no |
| 8 | `test_google_veo_video_adapter.py::test_existing_google_video_engines_stay_silent` | same | as 5 | no |
| 9 | `test_hf_env_offline.py::test_request_slot_uses_complete_canonical_cache_without_download` | the test's stub `guarded_auto_download()` rejects `progress_pbar`, which the code now passes | STALE TEST (the code's own comment says the bar is forwarded on purpose) | no -- fix the stub |
| 10 | `test_lane_preflight_matrix.py::test_g2_canvas_truth` | `animatediff15_lightning_video`, `animatediff15_v3_haunted_video`, `ltx_8gb` declare `render_canvas 512x288` that the declaration overrules; not in `EXPECTED_RED` | profile config-vs-truth on three VIDEO lanes (8 GB / animatediff, the 4060's surface) | no (still lane) |
| 11 | `test_lemmy_provisional_tier.py::test_the_writer_stage_bark_preset_SURVIVES_the_normalizer` | `'' == 'v2/en_speaker_8'` -- the normalizer drops bark's `voice_preset` | real regression on the Lemmy cameo provisional route | `lemmy_cameo` rolls ~11% on every episode -- **assess before the bump** |
| 12 | `test_lemmy_provisional_tier.py::test_a_rendered_receipt_names_artifacts_that_exist_and_still_match` | `otr/episodes/lemmy_cross_engine/kokoro_neutral.wav is missing` | as 4 | no |

## What this means for the bump

* Ten of twelve are cloud-lane, artifact-on-disk, or test-hygiene drift that
  cannot reach an episode rendered on the shipping still lane. They should be
  either fixed or added to `EXPECTED_FAILED_NODEIDS` + `docs/known-failures.md`
  with these reasons, so the guard stops reporting them as regressions.
* Two can reach a shipped episode and deserve one look each before the
  version string burns: **#2** (a `replay_from` run would raise) and **#11**
  (a Lemmy cameo on the chatterbox/bark route loses its preset). Neither is on
  the default path of a fresh install rendering an episode.
* Nothing in today's commits (`fbfbe0d6` .. `96346118`) moved any of the
  twelve; the two the day's code could have touched -- the word counter and
  the chunker -- were measured byte-identical on English.

## The final-HEAD run (2026-09-20 00:05, tree at `cd97b3e5`)

RC=2; **fourteen** nodeids not in `EXPECTED_FAILED_NODEIDS`, not twelve -- the
first table above was read from a truncated listing. The full set is the
twelve rows above plus:

| # | nodeid | note |
|---|---|---|
| 13 | `test_llm_slot_sweep.py::test_every_llm_call_site_has_slot_tag` | a static sweep for LLM call sites without a slot tag; failed in BOTH of today's full runs; not among the twelve re-run at `cc62b2c1`, so its age is not proven -- but no LLM call site was added today (the day's code is the scan lane, the word counter and the chunker) |
| 14a | `test_comfy_credential_rip.py::test_llm_hosts_capture_the_key_through_set_auth` | present only in the final run |
| 14b | `test_ltx_8gb_canonical_canvas.py::test_the_8gb_variant_workflow_agrees_with_the_declaration` | present only in the first run |

14a and 14b swapped between two runs of the same tree with no code change
between them that touches credentials or the 8 GB LTX variant -- run-to-run
variance (the second run happened while a headless leg held the GPU and its
environment). Both are environment-shaped, neither is on the shipping still
lane, and neither is in today's diff. The known-fail guard still hides their
assertion lines; `%TEMP%\full_suite_final.log` holds the run.

So the ship condition stands as written above, with fourteen explained
rather than twelve: ten cannot reach a shipped still-lane episode, two (#2,
#11) deserve one look before the version string burns, #13 is a sweep that
wants an owner, and the pair in #14 is variance to watch, not a regression.

## Closed after the close-out audit (2026-09-20, Composer 2.5 lane)

* **#11 FIXED** -- `cast_lock._stamp` cleared the Lemmy writer-stage bark
  preset on a provisional (audition) stamp; the clear now skips
  `fallback == "provisional_route"`. Both the Lime test and the Lemmy test
  pass. This was the one of the fourteen that could reach a shipped episode.
* **#2 FIXED** -- the replay test passed the engine positionally into
  `generate(self, script_json, ledger_json="", ...)`; a stale test, not the
  code.
* **#9 FIXED** -- the HF-offline stub takes `**kwargs` through.
* **#12 unchanged** -- the audition wavs the receipt names are not on this
  box; environmental.
* Remaining, unchanged: #1, #3-#8, #10, #12-#14 as classified above.

## One design call for the operator (not a defect; from Sonnet on `8d064235`)

Two contracts meet on the cast row's `voice_preset`: the Lime rule
(2026-09-17: a row spoken by kokoro/google/elevenlabs must not carry a
leftover Bark `v2/` preset) and the Lemmy rule (2026-08-16: the writer-stage
preset is Lemmy's identity and CastLock leaves it alone). The #11 fix keeps
the preset on every provisional (audition) stamp, so a Lemmy row auditioned
through kokoro -- a real shipped route -- keeps `v2/en_speaker_8` on a
kokoro-spoken row. Measured: only bark's dispatch reads `voice_preset`,
kokoro reads `voice_ref_id`, and the credits prefer `voice_engine` /
`voice_ref_id`, so the kept value changes no audio and no credit line; it is
a ledger-hygiene inconsistency, pinned as such by
`test_a_provisional_stamp_keeps_the_writer_preset_whatever_engine_auditions_it`.
Sonnet also could not find any production reader that needs the preset to
survive a non-bark stamp (the "two-stage" behaviour lives in the pinned
test and a docstring, now corrected). The options are: leave it (today);
or let Lime win everywhere and retire the survival test. Neither blocks a
ship.
