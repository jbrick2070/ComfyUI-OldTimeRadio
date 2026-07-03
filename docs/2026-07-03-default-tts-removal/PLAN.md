# Code-Ready Plan — Remove the dead `default_tts` widget (OTR_SceneSequencer)

**Date:** 2026-07-03
**Branch:** v2.0-alpha
**Type:** Routine dead-widget prune (mechanical). Same class as `allow_auto_fallback` / `episode_duration_target`.
**Grounding:** Claude anchored against the real Windows files (below). Panel = codex + antigravity (kibitz local). Antigravity was dead this session (zero output), so the panel ran codex-only; Claude's own grounded anchor covered the wiring (r3) round independently. Fable = one medium-level confirmation pass at the end only.

> Note on Fable: codex flagged (correctly, per CLAUDE.md §9) that a routine dead-widget prune does not warrant Fable. The operator explicitly requested a Fable confirmation pass for this plan, which overrides the §9 default — so it runs, framed as belt-and-suspenders confirmation, not as the driver.

---

## 1. Finding (proven, not assumed)

`default_tts` on `OTR_SceneSequencer` is a **fully dead widget**, not a fallback:

- Declared as a combo widget at `nodes/scene_sequencer.py:634-637` — combo `["bark", "parler", "kokoro"]`, default `"bark"`.
- Accepted as a param at `nodes/scene_sequencer.py:688` in `sequence(...)` with an explicit inline comment: *"kept: widget INPUT accepted by node contract; per-line TTS routing reads voice_assignments in the ledger, not this widget."*
- The `sequence()` run body **never reads `default_tts`** — verified by grep across the file; the only two references are the INPUT_TYPES declaration (:634) and the signature default (:688).
- Not a fallback: a fallback would be consumed when a line lacks an assignment. `SceneSequencer.sequence()` does not read `default_tts` at all; per-line TTS routing is decided upstream by the per-role voice nodes (character/announcer/music) off the canonical cast. (Note: `voice_assignments` is NOT the stored authority — it is `voice_assignments_from_cast()` at `nodes/_otr_ledger_consumers.py:164`, a derived legacy view of `led["cast"]`. The Sequencer contributes nothing to routing either way.)
- Stale-on-top-of-dead tell: the combo still lists `parler`, which is no longer a registered engine.

**Conclusion:** clean removal. No behavioral change to audio routing **for the canonical workflow**.

**Compatibility boundary (explicit):** `OTR_SceneSequencer` is a public exported class (`__init__.py:160`). Deleting `default_tts` from the `sequence(...)` signature changes the node's accepted-input contract, so any *stale saved graph or API prompt still passing `default_tts`* would need a re-save/migration. This migration covers the canonical `workflows/otr_scifi_16gb_full.json` only; external/older graphs carrying the promoted `default_tts` input must be re-saved. (In-repo there are no other graphs — confirmed by the repo-wide grep in §2.5.)

## 2. Removal surface (all in ONE commit — fail-closed, per §0)

The applier maps a profile key onto this widget, so the widget, its mapping, and all three profile keys MUST move together or the profile applier fails-closed on a missing target. Surface:

1. **Node code** — `nodes/scene_sequencer.py`
   - Delete the `default_tts` widget block, `:634-637`.
   - Delete the `default_tts="bark",` param + its inline comment from the `sequence(...)` signature, `:688`.
   - Verify no other reference remains in the file (grep `default_tts` → 0 hits after).

2. **Widget mapping** — `config/profiles/widget_mapping.json`
   - Delete the `"slot_overrides.sequencer_default_tts"` block, `:95-102` (registry `audio`, target `["OTR_SceneSequencer","default_tts"]`).
   - Check the trailing comma on the preceding block (`:94`) so the JSON stays valid.

3. **Profiles (all 3)** — delete the `"sequencer_default_tts": "bark"` key:
   - `config/profiles/16gb_full.json:22`
   - `config/profiles/8gb_lite.json:22`
   - `config/profiles/cpu_floor.json:19`
   - Check the trailing comma on the preceding key in each (`slot_overrides` block) so JSON stays valid.

4. **Workflow JSON of record** — `workflows/otr_scifi_16gb_full.json`, node **id 3** (`OTR_SceneSequencer`)
   - Current `widgets_values = ['[]', 0, 999, '', 'bark', 0]` mapping positionally to
     `script_json, start_line, end_line, output_dir, default_tts, dialogue_offset_ms`.
   - `default_tts` is **index 4** (value `'bark'`). Remove that one array entry → `['[]', 0, 999, '', 0]`.
     This is a MID-LIST removal (dialogue_offset_ms follows), so the value MUST be removed in the SAME edit as the INPUT_TYPES widget or positional drift (BUG-LOCAL-097) shifts `dialogue_offset_ms`.
   - `default_tts` is also promoted to an **input socket** on node 3 (`inputs[]` contains `{"name":"default_tts","widget":{"name":"default_tts"}}`). Remove that input entry too.
     - Before removing: confirm no `links[]` entry targets that input slot (widget-as-input with no wire is expected). If a link exists, it must be deleted and `last_link_id` left untouched.
     - After removing the input entry, verify remaining input `slot`/ordering is still internally consistent (litegraph tolerates gaps, but keep the array clean). Re-check every other node's link `dst_slot` into node 3 is unaffected (they reference named sockets `script_json`/audio buses, not `default_tts`).

5. **Repo-wide grep (not just tests/)** — grep `default_tts|sequencer_default_tts` across the WHOLE repo (py/json), excluding `docs/` and `kibitz-runs/`. Grounded current hit list (2026-07-03):
   - `nodes/scene_sequencer.py:634,688` (the widget + param — removed in §2.1).
   - `config/profiles/{16gb_full,8gb_lite,cpu_floor}.json` + `config/profiles/widget_mapping.json:95,100` (removed in §2.2/2.3).
   - `workflows/otr_scifi_16gb_full.json` node 3 (removed in §2.4).
   - `scripts/_otr_overnight_story_soak.py:201` — a **stale comment** ("# sequencer_default_tts -- that SceneSequencer combo only accepts ..."). Not a live input, so it will not break the run, but delete/refresh it so the comment stops referencing a removed widget.
   - Any `tests/` file asserting the widget exists, the combo values, the profile key, or a SceneSequencer **widget/input count** (a hard-coded count is now N-1). Grounded: `tests/` has zero `default_tts` hits today; the widget-order guard is `TestWidgetOrderVsInputTypes` in `test_workflow_json_guardrails.py:1147` (zips saved input order vs live INPUT_TYPES — passes only when code + JSON move together).
   - `ROADMAP.md:712` lists `default_tts` in the SceneSequencer HIDE list — do the one-line doc touch-up in the SAME commit.
   - **Run the post-removal `default_tts` grep with ignore-rules OFF** (`rg -uu` or raw PowerShell `Select-String`). `scripts/_otr_overnight_story_soak.py` is gitignored, so a default ripgrep skips it and a gitignored live script could falsely pass a "0 hits" check.
   - The `OTR_WorkflowValidator` widget-count-vs-INPUT_TYPES audit must still pass (that's the point — code + JSON move together).

## 3. Validation gate (after the edit, before commit)

Per CLAUDE.md §0 + §3:
- `OTR_WorkflowValidator` on `otr_scifi_16gb_full.json` — widget count == live INPUT_TYPES for node 3; every wired input-name in INPUT_TYPES; link referential integrity.
- JSON round-trip load of the workflow + all 3 profiles + widget_mapping.json (parse-clean, UTF-8 no BOM).
- Profile-applier coherence for all 3 tiers — run each committed profile through `apply_profile` (`nodes/_otr_workflow_apply.py:442`) and `cross_validate_profile` (`nodes/_otr_shared/capability_profiles.py:298`); a leftover `sequencer_default_tts` mapping/profile key with the widget gone is exactly what these catch. Expect clean (no unmapped/orphan target).
- Full regression suite + Bug Bible (Windows venv), `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`.
- AST parse on `scene_sequencer.py`.
- Then commit AND push to v2.0-alpha, same session; verify HEAD == origin, no BOM.
- **0-byte check scoped to touched files only.** The repo already contains pre-existing 0-byte files at root (e.g. `agy_*.log`, `test_ci.out`, `_otr_latentsync_worker.err`), so a blanket "no 0-byte files" gate is a false alarm here — assert only that the files touched by THIS change are non-empty.

## 4. Risk / non-goals

- **Risk: low.** No routing path reads the value; removal is confined to declaration + config surface + one positional JSON slot.
- **The only real trap** is the positional `widgets_values` / promoted-input pair on node 3 — handled explicitly in §2.4.
- **Non-goal:** do not touch per-line TTS routing, the voice nodes, or `voice_assignments`. This prune does not alter which engine any line uses.
- **Non-goal:** the second INPUT_TYPES in this file (`:1047`, a different node class) is unaffected — no `default_tts` there.

## 5. Open questions for the panel

1. Any consumer of `sequencer_default_tts` outside this repo (headless applier scripts, soak harness args) that would break on the key's disappearance? (Grep showed doc/soak `.out` references only — confirm none are live inputs.)
2. Is the promoted `default_tts` input on node 3 referenced by any link anywhere in the graph? (Expected: no.)
3. Any test that asserts an exact SceneSequencer widget/input **count** rather than the named widget?

## 5b. Panel sign-off

- **codex (r1):** yes-with-fixes. All 5 findings grounded + folded (0-byte gate scope, compat boundary, voice_assignments wording, repo-wide grep + soak comment, named applier gates). Antigravity was down this session — panel ran codex-only; Claude's grounded anchor covered the wiring round.
- **Fable (medium confirmation, operator-requested):** CONFIRM-WITH-NITS. Verified all four load-bearing claims directly (dead widget, complete surface, node-3 index-4/no-link wiring, no test-count breakers). Two non-gating nits folded: grep-with-ignore-off, and the ROADMAP.md:712 doc touch-up.
- **Net:** ready to build. No blockers.

## 6. Verification note (what Claude read)

`nodes/scene_sequencer.py` (INPUT_TYPES 600-645, `sequence()` 684-709, grep for `default_tts` = 2 hits only); `config/profiles/widget_mapping.json:88-103`; `config/profiles/{16gb_full,8gb_lite,cpu_floor}.json` (sequencer_default_tts lines 22/22/19); `workflows/otr_scifi_16gb_full.json` node id 3 parsed live via the Windows venv (widgets_values + inputs, above); repo-wide grep for `default_tts` / `sequencer_default_tts`. ROADMAP.md:712 already lists `default_tts` under SceneSequencer "HIDE" — consistent with retirement.
