# Hardened plan -- finish deleting the voice-route subsystem

HEAD to build on: current `main`. Written against `e8960496`; every
`:line` below is from that head and must be re-read before editing.
Re-grounded 2026-09-25 at `d167f0ab`: the module, the `cast_pools`
policy block, the bank field and `lemmy_route_qualified` are all still
present. `tests/test_stale_ledger_voice_guard_removed.py` is already
gone, so section 2.5 is moot -- put the section 2.7 pin in
`test_voice_casting_identity.py`. Docs paths moved from `docs/` to
`apple/` with the 2026-09-25 retirement.
Do not start from the 51b6c146 plan. Do not re-rip 91ad5961.

Hardened 2026-09-24. Scoped r2 (Cursor driver; Claude CLI + GPT-5.6-sol
substitute; Codex CLI quota-held until 2026-09-29). Not a four-round arc.

---

## Law of this chunk

Delete the unused route module and everything that only compiles if it
exists. Do not leave a field, helper, fixture, or comment that still
talks as if qualified routes were a live contract. Do not migrate
IndexTTS2 env/process ownership. Do not populate `EXPECTED_FAILED_NODEIDS`.
Do not bump `pyproject.toml`.

---

## 1. One atomic production deletion

In a single commit (or a pair that cannot be merged independently):

### 1.1 Module and policy

- Delete `nodes/_otr_voice_route.py`.
- In `config/cast_pools.py`, delete one contiguous block:
  `QUALIFICATION_RECEIPT_REQUIRED_FIELDS`, `LEMMY_AUDITION_LINES`, every
  `PROVISIONAL_*` name, `provisional_route_problems`, and
  `LEMMY_VOICE_POLICY`. Partial deletion is an import-time NameError on
  every `from config import cast_pools`.
- `sha256_of_file`: no production import from the route module. Duplicates
  in `_otr_voice_node_common.py` and `foley_stems.py` stay.

### 1.2 Dead bank exception field (was fork 3F -- closed: RIP)

No production reader since `ae883a2d`. A no-op key named "qualified route"
will be re-wired. Rip all of:

- `nodes/_otr_voice_bank.py`: parse of `unavailable_qualified_route_ids`,
  `_BANK_UNAVAILABLE_ROUTE_IDS_CACHE`, helper
  `unavailable_qualified_route_ids`
- `scripts/otr_make_portable_voice_bank.py`:
  `_PRIVATE_ROUTES_UNAVAILABLE_IN_PORTABLE_BANK`, the write of that key,
  `_validate_generated_bank`'s id check, notes that mention the private
  Index route as a route-id exception

Replacement already exists: portable bank drops private IndexTTS2 rows;
recurring table + missing row is an ordinary draw; `reserved_for` still
travels with the chatterbox/dia clone rows.

### 1.3 Comments that would put the subsystem back

- `nodes/_otr_audio_engines/__init__.py:38-47` -- drop fingerprint / Lemmy
  route wording. Keep the try/except import. Real remaining reasons the
  adapter is .comfyignore'd: YARA, sidecar venv, cloning refs.
- `.comfyignore:156-168` -- rewrite the comment the same way; do NOT stop
  excluding `eng_indextts2.py`.
- `nodes/cast_lock.py:107-110` -- the leftover `voice_route` is cleared
  because it is stale identity, not because the voice node raises ENGINE
  DISAGREEMENT. That raise is gone.
- `nodes/cast_lock.py:906` -- drop `_otr_voice_route.py` from the call-site
  list. Also drop the `_lemmy_voice_policy` comment at :926; there is no
  such function.
- `tests/test_output_tree_containment.py:339-346` -- the UNC test stays;
  rewrite the comment so it does not cite `_otr_voice_route.py:1074`.
- `config/voice_reference_bank.json` `notes` -- stop naming
  `resolve_and_verify_reference inside generate()`.

---

## 2. Tests -- name every file, every fate

### 2.1 `tests/conftest.py`

Delete `lemmy_route_qualified`. Do not touch `EXPECTED_FAILED_NODEIDS`
(empty frozenset on purpose).

### 2.2 `tests/test_cast_lock_config_import_portability.py`

Drop `nodes/_otr_voice_route.py` from `_FILES` and from the docstring.
The docstring also names `_lemmy_voice_policy` which does not exist --
fix that in the same edit. Keep `_otr_scifi_news_pro.py` and
`_otr_voice_node_common.py`.

### 2.3 `tests/test_voice_identity_fix.py`

DELETE the route/fingerprint block (~632-853) and `_routed_cast_row` plus
both tests at ~1106-1138, including
`test_a_current_ledger_route_is_still_honoured` (inverse of the new law).
KEEP the PBUG-20260817-09 seed / emotion-cap / kill-switch tests
(everything before 632, and 863-1073).

### 2.4 `tests/test_make_portable_voice_bank.py` -- the whole file is in scope

KEEP (retarget if they mention route ids):

- portable WAV contract: distinct male/female, atomic publish, too-short
  refusal, reservation travels with chatterbox/dia rows (`:42-98` and
  siblings)
- `test_cli_emits_runtime_override_and_schema_valid_bank` -- drop the
  `unavailable_qualified_route_ids` assertion at `:242-243`; keep CLI /
  schema
- `test_exact_exception_is_safe_in_preserve_ledger_mode` (`:443-479`) --
  KEEP the Lime v2/ clear; drop route-exception framing

DELETE:

- `:64-67` `LEMMY_VOICE_POLICY` / `unavailable_qualified_route_ids`
  assertion inside the otherwise-kept first test
- `:249-289` malformed / sha-bound exception-list tests
- `:292-331` `test_exact_portable_exception_skips_private_route_and_casts_generic_lemmy`
  (uses the doomed fixture; asserts `lemmy_route_tier` / `_id` /
  `_reason_code` which nothing writes)
- `:334-440` the three `VoiceRouteError` tests, including the
  monkeypatch of missing `cast_lock._lemmy_voice_policy`

If a leftover `copy` import has no users, drop it.

### 2.5 `tests/test_stale_ledger_voice_guard_removed.py`

KEEP `test_the_raising_guard_is_gone_and_stays_gone` and
`test_no_production_code_CALLS_OR_IMPORTS_the_deleted_guard`.
DELETE the four fingerprint tests (`:110-214`) and the unused
`VOICE_ROUTE` path constant (`:56`). Rewrite the module docstring so it
does not describe `select_policy_route` as live.

### 2.6 `tests/test_env_single_owner.py` and `tests/test_process_single_owner.py`

Remove `nodes/_otr_audio_engines/eng_indextts2.py` from `BLOCKED`.
Leave it in `PENDING`. One-line note: fingerprint reason died with the
route subsystem; migration is a different commit. Do not invent a new
unblock condition. Do not migrate the adapter here.

### 2.7 Already-covered leftover dict; do not add a lock-then-dispatch test

`tests/test_voice_casting_identity.py:139-156`
(`test_a_stale_route_field_is_cleared_on_a_re_stamp`) already proves
lock strips `voice_route` in both policies. Rewrite its docstring: the
voice node does not raise; the field is shed as stale identity.
`:297` `test_a_locked_row_sheds_every_retired_route_field` already lists
the three retired literals.

ADD one pin that lock cannot prove: a source-inspection (or a call that
never goes through `lock`) that `nodes/_otr_voice_node_common.py` has no
`voice_route` token, so a leftover dict on a frozen ledger cannot raise
in dispatch. Put it in `test_voice_casting_identity.py` or the surviving
stale-ledger file. Do not call `lock()` first.

### 2.8 Small tails

- `tests/test_otr_dialogue_policy.py:101-110` -- comment with no tests
  under it. Delete the tail.
- `tests/test_per_line_audio_meta.py:176` -- "slated for deletion" is
  already past; `test_voice_route_reference_contract.py` is gone. Past
  tense.
- `tests/test_tts_voice_preflight_matrix.py` -- P2.1 is already retargeted
  at `RECURRING_CHARACTER_VOICES`. Rename the test / `routes` locals so
  they do not say "route". Gate 4 section at `:255-258` is empty; delete
  the heading.

---

## 3. Docs -- same work, no publish

- Listen record, same commit as the policy deletion: append it to
  `apple/RIGHTS_DECISION_LEMMY_VOICE.md` (dated docs were retired, so no
  new dated file). Quote the 2026-08-18 listen path, the winning clone
  ids, and the hashes that lived on the approved IndexTTS2 / chatterbox /
  dia records. Add a supersession line under
  `apple/OTR_STANDING_RULINGS.md` (the Lemmy listen ruling) -- the listen stands; the receipt
  machinery is retired; identity is `RECURRING_CHARACTER_VOICES` +
  `reserved_for`.
- README "Known failures" section: drop the portable-bank route-test
  class and the "cite audition wavs by hash" class (those tests/files are
  gone). No `pyproject.toml` bump. In-repo README will be ahead of the
  Active zip until the next approved patch. State that in the commit.
- `apple/TTS_VOICE_PREFLIGHT.md`: retire Gate 4 entirely (P4.1-P4.6). Fix
  P2.1 wording to the recurring-table rule. While in the file, replace
  P6.1's "100% local, no API keys" sentence with the 2026-09-24
  offline-first / cloud-additive line from the ComfyUI `CLAUDE.md`.
- The ship-regression receipt doc was retired; the commit message names
  the nodeids this work deletes instead.
- `apple/OTR_STANDING_RULINGS.md` (known-failure class list) and `apple/known-failures.md`:
  refresh the class list against a set-diff at current HEAD, then again
  after the deletion commit.

---

## 4. Sequencing and proof

1. Capture the failing nodeid set at the head you start from before
   touching anything. An older worktree is the wrong baseline.
2. One green chunk: production deletion + tests + comments + the dated
   archive + README/preflight/ship-regression class list.
3. Scoped suite, then full-suite set-diff against the capture from step 1.
   Every disappeared failure must be a nodeid this commit deleted; every
   new failure must be explained.
4. Push when that scoped suite is green. Live `otr/obs/` leg follows;
   it is not a push gate.

Scoped tests, named: `test_make_portable_voice_bank.py`,
`test_voice_identity_fix.py`, `test_voice_casting_identity.py`,
`test_cast_lock_config_import_portability.py`,
`test_tts_voice_preflight_matrix.py`, `test_env_single_owner.py`,
`test_process_single_owner.py`, `test_stale_ledger_voice_guard_removed.py`,
`test_otr_dialogue_policy.py`, plus grep that `nodes/` has no
`_otr_voice_route` import.

---

## 5. Out of scope

- Cache-off ledger stamps. Own row.
  `test_end_to_end_google_tts_cache_off_byte_identity` forbids widening
  `_persist_ledger_stamps`.
- HF_HOME Windows pin. Different CODE row.
- Migrating `eng_indextts2.py` onto the env/process owners.
- Filling `EXPECTED_FAILED_NODEIDS`.
- A registry publish.
