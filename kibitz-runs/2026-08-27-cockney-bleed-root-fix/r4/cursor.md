VERDICT: yes-with-fixes — helper + two callers + no-workflow contract have converged; three leftover ambiguities would NameError the new tests or fabricate/drop the mixed-group receipt.

MUST-FIX BEFORE BUILD:
1. [P3.2 / P3.3] Defect: P3.1 is the only place that keeps `from nodes._otr_dialogue_policy import _COCKNEY_ORTHOGRAPHY_RULE`. `tests/test_phase1_composer_prompt.py:8-16` and `tests/test_compose_exchange.py:28` do not import it, but P3.2/P3.3 tell those files to assert the constant. That is a collection/NameError, not a weak assertion. Survived R1–R3. Fix: add that exact import in both integration files; assert `_COCKNEY_ORTHOGRAPHY_RULE` in `messages[0]["content"]` after `role == "system"`.
2. [P5.3.6] Defect: “construct slots carrying beat id, dialogue slot id, and speaker, then call `group_voiced_beats`” has two incompatible builds. `VoicedSlot` (`nodes/_otr_compose_exchange.py:100-110`) has no `beat_id`. `group_voiced_beats` returns the original objects (`759-824`) and only reads `.speaker` (`792-794`). Using `VoicedSlot` drops beat ids, so the later subset check against `meta.exchange_prepass_audit.beat_ids` cannot run. R3-introduced. Fix: use `SimpleNamespace(beat_id=..., dialogue_slot_id=..., speaker=...)` (or any object with those three attrs). Do not use `VoicedSlot` for the live audit.
3. [P5.3.6] Defect: “Index `ledger.lines[]` by `beat_id`” allows `{row["beat_id"]: row for row in lines}`. Live ledger `lines[]` includes music rows with missing `beat_id` (`music_opening_001` / `music_closing_001` have no `beat_id`; music_inter rows do). A dict comp silently collapses blanks/duplicates and can attach the wrong speaker/slot. R3-introduced. Fix: fail the receipt unless each `beats[].beat_id` used in reconstruction maps to exactly one `lines[]` row with that `beat_id`; blank or duplicate identity is a run break, not last-write-wins.

SHOULD-FIX:
1. [P2.1] Helper docstring still reads roster-era (`nodes/_otr_dialogue_policy.py:32`). Require it to say `active_speakers` = current output speakers only, values = `LineRequest.speaker` (`nodes/_otr_line_composer.py:248`) and `VoicedSlot.speaker` (`nodes/_otr_compose_exchange.py:109`). Smallest lock against a later “pass the full cast for safety” regression.
2. [P3.3 item 3] Pin the existing stateful Tier-A in `test_repair_triggers_once_then_succeeds` (`tests/test_compose_exchange.py:220-242`). `_always_fail` still yields two generate calls, but `_raw_for` is MARLOW/REESE-only (`85-92`). One sentence: local LEMMY/MARLOW raw + fail-then-ok checker; compare system strings only (`failure_reasons` go to USER a`nodes/_otr_compose_exchange.py:446-457`).
3. [P6.1 vs `docs/GO_FORWARD_PLAN.md:167-168`] P2.1 supersedes “leave the orthography sentence global,” but P6.1 only says “update GO_FORWARD.” An implementer can add a PBUG and leave the old sentence as live instruction. Fix: replace that paragraph in the same records change.
4. [P5.3.2 / P5.3.6] If `media_archive` fetch/source fails, stop that leg; do not retarget `original` inside the same receipt. If no retained mixed LEMMY+other group, cap at one rerun then fail the mixed-exchange qualification (do not loop).
5. [P2.1] After removing `Any`, `Dict`, `Iterable`, `Union`, delete the empty `from typing import` line or the file will not parse.

OPTIONAL / NICE-TO-HAVE:
- Parametrize P3.1 TypeError cases 5–7.
- State the expected production diff: `_otr_dialogue_policy.py`, `_otr_line_composer.py`, `_otr_compose_exchange.py`, plus the three named test files and later records. Any other production module needs a new grounded reason.
- P2.1 helper docstring warning is enough; do not import `cast_pools` into the stdlib-only helper. `LEMMY_PROFILE["name"] == "LEMMY"` is `config/cast_pools.py:376`.

CUT THESE:
1. `roster_has_lemmy` compatibility alias — would keep the roster category P2.1 deletes. `rg` shows only `_otr_dialogue_policy.py` and `tests/test_otr_dialogue_policy.py` in code.
2. A checked-in qualification script for P5.3.6 — one deleted probe is enough; the grouping helper is already production code.
3. A second live `use_exchange=False` leg — already forbidden by P5.3.9; `use_exchange` is not on `CREATIVE_WHITELIST` (`scripts/otr_api.py:831-859`).
4. `append_dialogue_policy(None, ...)` tests — both callers resolve a str first (`nodes/_otr_line_composer.py:1040-1051`,`nodes/_otr_compose_exchange.py:389-393`). `system_prompt or ""` is enough.
5. Post-gen dialect scrubber / vocabulary blacklist — already P2.4; would be a shim on a prompt-scope bug.

VERIFY-AT-BUILD checklist:
- [R3] Live `beats[]` still lack `dialogue_slot_id` (confirmed on `signal_lost_glass_shards_and_broken_promises_20260825_094527` ledger `beats[]`); join from `lines[]` while walking `beats[]` order, not `lines[]` file order.
- [R3] Live `meta.exchange_prepass_audit` exists (same ledger, `meta` ~line 2295: `beats_composed` + `beat_ids`). Confirm the new run writes the same keys.
- [R3] Direct PowerShell `-Set @(...)` prints both `OTR_LedgerScriptWriter.source_bank='media_archive'` and `OTR_LedgerScriptWriter.lemmy_cameo='always include'` (`scripts/otr_canonical_api_run.py:78-89, 385-388`). `_parse_value` keeps `always include` as a string (`46-51`).
- [R2/R3] Wrapper default `Port=0` (`scripts/otr_headless_canonical.ps1:31,173-189`). Record the logged ephemeral port / `COMFYUI_URL`; do not claim :8000. Still confirm :8000 is empty and VRAM is at desktop baseline before boot (wrapper only logs VRAM at `162-170`).
- [R3] `cast` / persona USER roster unchanged: `build_exchange_prompt` still builds per-slot `roster_block` from `persona_by_name` (`335-348`); only `roster_items` at `391-393` loses `cast`. Existing exchange tests staying green is the check.
- [R2] `compose_line` returns `LineResult`; capture via `_recording_creative.state["calls"]` as a list of message-lists (`tests/test_phase1_composer_prompt.py:136-145, 109-1161` in `_otr_line_composer.py`). Do not treat `compose_line` as returning a raw string.
- [R1] If scoped system tests pass and live fallback/singleton lines still bleed, inspect USER `all_voice_cards` (`OTR_LedgerScriptWriter.py:4575-4578, 4848`; `_otr_line_composer.py:724-730`; Lemmy card includes “Cockney” at `config/cast_pools.py:382-384`). Do not widen this patch on that evidence alone.
- [this round] `rg -n "append_dialogue_policy|roster_has_lemmy" nodes tests` after the atomic change: only the new keyword call sites + tests.
- [this round] `git diff -- workflows/otr_canonical.json` empty; writer slot 13 remains `true` (`tests/test_workflow_json_guardrails.py:659-663`). No `INPUT_TYPES` / `NODE_CLASS_MAPPINGS` / widget / `IS_CHANGED` / VRAM change.
- [this round] `scifi_news_pro` still returns at `OTR_LedgerScriptWriter.py:3376-3431` before the inline composers; helper not imported there. `tests/test_scifi_news_pro_lemmy_cameo.py` unchanged.
- [P1.5] Evidence tree is real: `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\signal_lost_glass_shards_and_broken_promises_20260825_094527\audio\` exists. New qualification still requires `RESULT SUCCESS`, `Prompt executed`, `obs_publish OK`, and the asset under `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs`.
- verify: focused/full/Bible/workflow gates on the frozen diff (not re-audited here).
- [ASSUMPTION] Production `lines[]` dialogue beats remain 1:1 with `beats[].beat_id` on a healthy run; music_open/close rows without `beat_id` are ignored by a beats[] walk if uniqueness is enforced as in must-fix 3.

Domain: no new node, no `NODE_CLASS_MAPPINGS` work, no tensor layout, no model residency, no `IS_CHANGED`. Callers already lazy-import the helper inside the function body. Canonical JSON is correctly untouched.
