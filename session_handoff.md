# Session Handoff -- OTR v2.0-alpha audio/voice CLEAN-BREAK -- 2026-06-03

## Core goal
Finish `docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md` (the SSOT) under
the CLEAN-BREAK directive: a model-agnostic per-role engine registry (character
voice / announcer / music) as the SOLE audio path, wired into the ONE workflow
`workflows/otr_scifi_16gb_full.json`. Work the plan's clean-break sprints to
completion, and **remove each legacy item IN LOCKSTEP with building its
replacement** -- build -> wire -> full suite green -> delete the legacy + all refs
in the SAME change -> green again -> guard test that fails if it reappears. The
full CLEAN-BREAK directive is in the EXECUTION-PLAN header (do not re-derive it).

## Tech stack & constraints
Python 3.12 + torch, Windows, RTX 5080 16 GB. Branch `v2.0-alpha`, never `main`.
Hard rules live in CLAUDE.md (auto-loads) -- the ones that cause rework if
forgotten:
- Tests + git on the WINDOWS HOST via **Desktop Commander cmd** (NOT PowerShell
  for git; NOT the GitHub connector -- context-only). venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Full regression after EVERY code change: `python -m pytest -q -p no:cacheprovider`
  (~37 s; must stay fully green). Baseline this session: **3727 passed, 12 skipped,
  0 failed**.
- Commit via `.git\COMMIT_EDITMSG` + `git commit -F` (cmd); `git add` explicit
  paths, never `-A`. ASCII-only `.py`, no BOM, never the word "dummy".
- After every push verify: local HEAD == origin HEAD, no 0-byte files, no BOM,
  AST parses, node classes registered.
- **Audio is king** (Prime Directive #1): never silently drop or degrade audio
  behavior to satisfy a deletion. `baseline_v2` (render-twice from the NEW
  engines, operator GPU) is THE reference -- no v1.7 byte-identity.

## What's done & decided (this session; HEAD `904103f` == origin/v2.0-alpha)
- Verified clean start: HEAD was `c178648`, suite green 3727/12/0.
- **`bf10ef7` -- sprint 1a PREP (the safe, headless half).** Relocated the pure
  Bark per-line helpers (`_clean_text_for_bark`, `_chunk_text_for_bark`,
  `_generate_single_line`) **byte-exactly** (AST `get_source_segment`) from
  `nodes/batch_bark_generator.py` into `nodes/_otr_bark_lib.py`.
  `batch_bark_generator.py` re-exports the three names (every import/patch target
  still resolves: `test_core` parity import, `test_bark_ledger` patches
  `nodes.batch_bark_generator._generate_single_line`). `story_orchestrator`
  sources `_generate_single_line` from the lib. `_generate_single_line` got a lazy
  `import torch`. Zero behavior change. This makes bark inference delegation-free
  so `eng_bark` can stop constructing the heavy batch node.
- **`904103f` -- the decision brief** `docs/2026-06-03-bark-cleanbreak-1a__decision-brief.md`
  (READ THIS FIRST for sprint 1a): ready-to-apply code for the rest of 1a.
- Findings that shape every audio-node deletion (1a/1b/1c):
  - The **freeze-halt safety gate** (`freeze_verdict=='needs_full_rerun'`,
    BUG-276/300) is enforced ONLY inside the four legacy audio nodes
    (`batch_bark_generator`, `kokoro_announcer`, `musicgen_theme`,
    `batch_audiogen_generator`). **No v2 node re-homes it** (checked `cast_lock.py`,
    `scene_sequencer.py`, `_otr_freeze_cascade.py` -- the cascade only *stamps*
    it). Deleting any legacy audio node drops its copy. **Re-home it ONCE in
    `OTR_CastLock`** (runs first in the audio chain) and it covers all three
    deletions. The per-node `bypass_freeze_halt` widget cannot survive (v2 nodes
    forbid extra widgets, E.4) -> becomes env `OTR_BYPASS_FREEZE_HALT`.
  - The per-line **ledger timing write-back** (BUG-096 `dur_s`/`start_s`) is
    REDUNDANT: `scene_sequencer.py:768-903` already writes authoritative
    timings from the assembled audio. Safe to drop from every legacy audio node.
  - A per_line engine is **NOT byte-identical** to its batch path -> the new
    reference is `baseline_v2` (operator GPU capture), per the directive.
  - **voice_preset routing SOLVED:** add a `voice_ref_field` attr per adapter
    ("voice_preset" for bark, default "voice_ref_path" for cloning engines); the
    dispatch reads the right cast field into the existing ref slot. Non-breaking
    for chatterbox/indextts2. (Appendix A of the brief.)

## State of the art
HEAD `904103f`, suite green. Engine adapters in `nodes/_otr_audio_engines/`:
`bark`(still `interface="batch"` -- delegates), `chatterbox`/`indextts2`/
`stable_audio`(self-contained per_line/clip G1 bodies, flag-gated default-off),
`kokoro`/`musicgen`(still `interface="batch"`). Shared dispatch:
`nodes/_otr_voice_node_common.py` (`_delegate_batch` at ~217 for batch,
`_render_per_line` for per_line). The 4 legacy audio nodes still exist and are
reached via batch delegation only (not graph-node instances in full.json).
The eng_bark per_line body, the dispatch patch, and the CastLock freeze-halt
re-home are all written out ready-to-apply in the decision brief (appendices
A/B/C).

**Clean-break sprints remaining** (each LOCKSTEP; the freeze-halt re-home in
CastLock is the shared prerequisite that unblocks 1a/1b/1c deletions):
- **1a (bark)** -- PREP done. Remaining: resolve Gate A + Gate B (below), then
  apply brief appendices A+B+C, flip `eng_bark.interface="per_line"`, delete
  `batch_bark_generator.py` + bark refs (`__init__.py`, `_otr_legacy_manifest.LEGACY_AUDIO_NODES`,
  `config/legacy_invocation_manifest.json` bark entry only), convert the 6 bark
  test files (list in brief), add the reappearance guard test, suite green,
  capture `baseline_v2`, commit+push.
- **1b (kokoro)** -- `eng_kokoro` self-contained announcer body; delete
  `kokoro_announcer.py` + refs (drop its freeze-halt copy; CastLock covers it).
- **1c (musicgen)** -- `eng_musicgen` self-contained clip body; delete
  `musicgen_theme.py` + `batch_audiogen_generator.py`, then REMOVE the batch
  dispatch entirely (`_otr_voice_node_common` ~217 + `stable_audio_theme.py` ~172)
  -- last batch user, I-3 retired.
- **2** -- remove the writer bark voice_preset stamp in `_otr_casting`
  (`python_assign_voice_preset` + `_assert_voice_preset_invariant` + uniqueness
  guard); OTR_CastLock owns casting (bank `voice_ref_id`); bark draws from the bank.
- **3** -- remove R0a legacy seeding + `config/legacy_invocation_manifest.json`;
  `baseline_v2` replaces the render-twice-LEGACY tests.
- **4** -- promotion: flip full.json engine-widget defaults to the new engines per
  role; retire `OTR_ENABLE_*` gating. Gated on F.
- **5** -- F probes for indextts2 + stable_audio (isolated venvs); flip
  `supports_external_generator` + reconcile bodies vs real signatures.

## Immediate next steps
1. Open the decision brief `docs/2026-06-03-bark-cleanbreak-1a__decision-brief.md`.
2. **Gate A (operator decision):** confirm the freeze-halt re-homes to `OTR_CastLock`
   + `OTR_BYPASS_FREEZE_HALT` env (recommended). Implement appendix C in
   `nodes/cast_lock.py`; repoint `tests/test_bark_freeze_halt_bypass.py` to the new
   home + env contract. (This same change later serves 1b/1c.)
3. Apply appendix A (dispatch `voice_ref_field`) + appendix B (`eng_bark` per_line
   body); flip `eng_bark.interface` to `per_line`. Suite green.
4. In the SAME change: delete `batch_bark_generator.py` + all bark refs; convert
   the 6 bark test files (brief has the list); add the guard test that fails if
   `BatchBarkGenerator`/`batch_bark_generator` reappears. Suite green.
5. **Gate B (operator GPU):** capture render-twice `baseline_v2` for bark; wire it
   in. Commit + push. Then proceed to 1b -> 1c -> 2 -> 3 -> 4 -> 5.

## Open questions
- Gate A: confirm CastLock (vs voice-base vs drop) as the freeze-halt home.
- `eng_bark` temperature: hardcode 0.7 or thread `profile.default_params['text_temp']`
  through `generate_voice`? (Brief appendix B note.)
- Verify-at-build: `assert_model_available(char_bark_v1)` must be lenient with an
  empty `model_path` (Bark loads from HF cache) so the per_line bark path does not
  raise MISSING_MODEL at render.

---
## Resume instructions
Open a fresh window with the project mounted, attach this file, and say:
"Read this handoff + docs/2026-06-02-audio-voice-overhaul__EXECUTION-PLAN.md +
docs/2026-06-03-bark-cleanbreak-1a__decision-brief.md. Verify HEAD ==
origin/v2.0-alpha and the suite is green (3727/12/0). Continue the EXECUTION-PLAN
under the CLEAN-BREAK directive, removing legacy in lockstep, until all sprints
are done -- starting by finishing sprint 1a (Gate A then the flip+delete).
Acknowledge when ready."
