# OUTPUT TREE CONTRACT (operator law, 2026-06-11) + OH tickets — build-ready

**The contract:** `<comfy output>/otr/` contains EXACTLY two top-level entries:

    otr/
      episodes/<episode_id>/...   <- the ASSET OF RECORD for everything episode-scoped,
                                     in logical subfolders: ledger + script/treatment,
                                     stills/, portraits/, videos/ (per-line pieces),
                                     composited/, upscaled/, meshes/ (3D assets),
                                     captions, manifests
      episodes/_shared/           <- RESERVED system tiers (never an episode):
        cache/   (content-addressed cross-episode copies; never the only copy)
        tmp/     (scratch; TEMP/TMP/OTR_GPU_LEASE_DIR; janitor-swept)
        state/   (per-machine state, e.g. news_history.json)
      obs/                        <- FINAL deliverable videos ONLY, flat, OBS-watched

Anything else at the top level = a FAILED run (hygiene gate) / FAILED test (CI).

## Tickets (one coder window, after the current GPU queue drains)

- **OH-0 (hunt, first):** identify the ACTUAL writer of live `otr/stills/` (pass00's
  attribution was corrected in judgment — suspects: ST-3 cache materialization or a
  direct join bypassing `_otr_paths.py`). Fix it to the contract as part of OH-1.
- **OH-1 (code, one chunk):** `_otr_paths.py` — add `otr_shared_cache_dir()` /
  `otr_shared_tmp_dir()` / `otr_state_dir()` -> `episodes/_shared/...`; production
  write helpers RAISE LOUD on empty episode_id (kill the `_legacy_*` fallbacks);
  `_validate_contract()` wraps OUTPUT helpers only (resolve() + relative_to; exempt
  input/models/log/HF resolvers). Every `episodes/` walker skips `_`-prefixed entries
  (ledger auto-pick named). README.txt dropped into `_shared/`.
- **OH-2 (same chunk):** launcher env repoint (`_otr_soak_server_launch.cmd`:
  TEMP/TMP/OTR_GPU_LEASE_DIR -> `_shared/tmp`) + hygiene gates updated together
  (`scripts/_otr_soak_capstone.py`: fail any top-level otr/* outside episodes|obs;
  `_shared` never an episode; obs flat) + new `tests/test_output_tree_contract.py`
  (top-level set; zero top-level files; helper-contract parametrized over output
  helpers) + ONE narrow CI test banning `comfy_output_dir()/"otr"` composition
  outside `_otr_paths.py`.
- **OH-3 (ops chunk):** janitor — stale-tmp sweep (age threshold, only under
  `_shared/tmp`, skip-locked, log every unlink; the ONE sanctioned auto-delete);
  runs at server boot + post-publish.
- **OH-4 (ops chunk, operator-gated):** migration + attic — preflight QUIESCENCE
  (no active renders/ffmpeg; tmp idle past threshold); MOVE live cache/state into
  `_shared` (content-addressed collision policy: verify hash/size, quarantine
  conflicts); move dead debris (audio 7.7GB, videos, script_gates, blend_test,
  _legacy_stills, qa_waveforms, qa_frames, portraits, aship, aship_test, _lane1) to
  `<output>/otr_attic_<timestamp>/` OUTSIDE otr/; print the dry-run table first
  (`otr_tree_doctor --dry-run`); OPERATOR approves deletions — nothing else
  auto-deletes.
- **OH-5 (docs):** section-0 LIVE STATUS tick + tracker row + this contract linked
  from VIDEO_BUILD_HANDOFF.md HARD RULES.

## Sequencing (binding)

Run AFTER the current queue drains (7-leg sweep -> supervised wan batch -> 0-E
Phase B) and BEFORE item 5 (the 3D sprints must be born under the locked contract —
their meshes/ land per-episode from day one). Quiet-box guardrail applies.

Evidence: `roundtable/pass00_plan.md` (+ operator clarification), `pass01/` reviews,
`pass01_judgment.md`. Spend: ~$0.10.
