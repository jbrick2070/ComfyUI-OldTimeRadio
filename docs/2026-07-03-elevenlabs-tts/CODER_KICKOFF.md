# CODER KICKOFF — Cloud ElevenLabs TTS + Sonilo music (paste into a fresh window)

You are picking up a scoped, gate-converged build in the OTR ComfyUI custom-node
pack. Everything below is code-ready; your job is to IMPLEMENT it, sprint by
sprint. Read `CLAUDE.md` at the repo root first — operator directives win.

## Context (already done — do NOT re-litigate)
- Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch `v2.0-alpha`.
- SOURCE OF TRUTH = `docs/2026-07-03-elevenlabs-tts/BUILD_PLAN.md` (contracts C1-C10,
  sprints S0-S7, verify-at-build items, operator decisions). Read it fully before coding.
- Goal: make ElevenLabs cloud TTS (char + announcer) and cloud Sonilo music
  FULLY-CLOUD engines via the existing `invoke_partner_node` backend (Comfy credits),
  fail-loud, no fallback, dropdown-is-enable, audio spine frozen.
- Gate chain COMPLETE and CONVERGED: roundtable R1 (model pick) + kibitz r2 (codex) +
  Fable x2 (final GO) + two independent code-scan reviews (codex + antigravity) all
  agree: the plan's anchors are real, nothing is built yet, and the MUST-FIX list IS
  the build order. Do not re-review; build.

## Hard rules (from CLAUDE.md)
- YOU run everything via Desktop Commander (Windows venv), never hand the operator a command.
- File I/O through the file tools (Read/Write/Edit) on the REAL Windows files; the
  Linux/bash mount LAGS — never trust it for current state.
- Test runner: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`, `$env:PYTHONUTF8=1`,
  `pytest -q -p no:cacheprovider`. Bug Bible lives in the separate repo
  `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (cd there, use
  the RELATIVE path `tests\bug_bible_regression.py`).
- Run the regression suite + Bug Bible + the B7 forbidden-sweep after EVERY code change.
- Commit AND push to `v2.0-alpha` per green chunk, same session. Verify HEAD==origin,
  no BOM, AST parse. UTF-8 no BOM, SFW, no "dummy" (use "placeholder"/"stub").
- PowerShell: chain with `;` not `&&`; never `python -c "..."` with nested quotes —
  write a temp `.py`, run it, delete it. First quoting error = switch to a temp .ps1.
- Any node/widget/wiring change lands IN `workflows/otr_scifi_16gb_full.json` in the
  SAME change (append-only positional widgets), then re-validate.

## Non-negotiable invariants (fail the build if broken)
- `test_audio_byte_identical` stays green (cloud is dropdown-opt-in; DEFAULTS NEVER CHANGE).
- No silent fallback to a local engine on any cloud-selected line/cue — fail LOUD.
- Determinism: casting keys on `OTR_CAST_SEED`; deterministic request + ledger hash
  (not provider byte-identity).
- Adapter-registration + its CAPABILITIES row land in the SAME COMMIT
  (`test_capability_profiles.py:213` is bidirectional set-equality).
- Conformance xfail removal happens ONLY in the sprint that registers that adapter
  (elevenlabs S1, sonilo S5) — never earlier, or S0 goes red.
- S6 ships NO master engine-value edit (`test_capability_profiles.py:173-202` asserts
  master==profile). S7 acceptance runs PROFILE-LESS (else `16gb_full.json` reverts the pick).

## START HERE — Sprint S0 (pure code, no render; the gate for everything)
Do these, run the suite + Bug Bible green, commit+push, THEN proceed to S1:
1. `nodes/_otr_shared/cloud_media_canonical.py:127` — implement `canonicalize_audio`
   (replace `_not_built_yet`): WAV 44.1kHz, stereo policy, loudness matched to the
   LOCAL lane's real reference (resolve `LOUDNESS_REFERENCE_SOURCE` at `:68` — find the
   assembler's existing loudness constant/module; do NOT invent a new LUFS convention),
   +/-250ms tolerance with head/tail silence padding, emit `actual_duration_s`.
2. `nodes/_otr_engine_profiles.py:35` — add `"cloud"` to `_VALID_RUNTIMES`; declare the
   new `EngineProfile` fields (`partner_row`, `provider_id`, `required_param_defaults`,
   `auth_required`, `billing_category`, `canonicalizer`, `error_policy`, valid roles) —
   the model is `extra="forbid"`, so undeclared fields raise.
3. Thread `provider_voice_id` end-to-end: `config/voice_bank_entry_schema.json` (add the
   property), `VoiceBankEntry` dataclass (`_otr_voice_bank.py:76`), `_entry_from_dict()`
   (`:158`), CastLock `_stamp()` (`cast_lock.py:650-654`), the durable cast stamp
   (`production_ledger.py`), and reserve it for the admission gate.
4. V3-expand + re-pin the ElevenLabs `model` / `output_format` / `apply_text_normalization`
   combos (`scripts/otr_pin_partner_nodes.py` regenerates ALL of `partner_nodes.yaml` —
   diff for unrelated image/video row drift and run the image/video conformance suites
   in the same chunk). Do NOT touch the conformance xfails in S0.

Then S1..S7 exactly as BUILD_PLAN.md sequences them. Use
`docs/2026-07-03-elevenlabs-tts/CODE_REVIEW_PROMPT.md` to self-review each sprint.

## Operator decisions still open (ask before the sprint that needs them)
Announcer pinned vs shuffled (S2); voice-pool size + ToS review (S2); library-only vs
clone (defer clone); sonilo vs stability + which cue roles (S5); confirm fail-loud
overrides PD1 for cloud char lines (S3); provider-native duration vs post-trim (S5).
Recommended defaults are in BUILD_PLAN.md §6.

## Verify-at-build (runtime-only)
ElevenLabs voice-id vs display-label (decides C4 capture test); whether the local lane
has an explicit loudness constant; V3 re-pin row-drift. Headless S7 needs `OTR_COMFY_API_KEY`.
