# Code-Ready Plan — Remove the dead `default_tts` widget (OTR_SceneSequencer)

**Date:** 2026-07-03
**Branch:** v2.0-alpha
**Type:** Routine dead-widget prune (mechanical). Same class as `allow_auto_fallback` / `episode_duration_target`.
**Grounding:** Claude anchored against the real Windows files (below). Panel = codex + antigravity (kibitz local). Fable = one medium-level confirmation pass at the end only (per CLAUDE.md §9: not narrative, not a high-stakes structural rip — Fable is belt-and-suspenders confirmation here, not the driver).

---

## 1. Finding (proven, not assumed)

`default_tts` on `OTR_SceneSequencer` is a **fully dead widget**, not a fallback:

- Declared as a combo widget at `nodes/scene_sequencer.py:634-637` — combo `["bark", "parler", "kokoro"]`, default `"bark"`.
- Accepted as a param at `nodes/scene_sequencer.py:688` in `sequence(...)` with an explicit inline comment: *"kept: widget INPUT accepted by node contract; per-line TTS routing reads voice_assignments in the ledger, not this widget."*
- The `sequence()` run body **never reads `default_tts`** — verified by grep across the file; the only two references are the INPUT_TYPES declaration (:634) and the signature default (:688).
- Not a fallback: a fallback would be consumed when a line lacks an assignment. Per-line TTS routing is fully owned by the per-role voice nodes (character/announcer/music) + the ledger `voice_assignments`. The Sequencer contributes nothing to routing.
- Stale-on-top-of-dead tell: the combo still lists `parler`, which is no longer a registered engine.

**Conclusion:** clean removal. No behavioral change to audio routing.

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

5. **Tests** — grep `default_tts` and `sequencer_default_tts` across `tests/`:
   - Update/remove any test asserting the widget exists, the combo values, the profile key, or a SceneSequencer **widget count** (a hard-coded count will now be N-1).
   - The `OTR_WorkflowValidator` widget-count-vs-INPUT_TYPES audit must still pass (that's the point — code + JSON move together).

## 3. Validation gate (after the edit, before commit)

Per CLAUDE.md §0 + §3:
- `OTR_WorkflowValidator` on `otr_scifi_16gb_full.json` — widget count == live INPUT_TYPES for node 3; every wired input-name in INPUT_TYPES; link referential integrity.
- JSON round-trip load of the workflow + all 3 profiles + widget_mapping.json (parse-clean, UTF-8 no BOM).
- Profile-applier dry pass for all 3 tiers — no "target widget not found" (proves the mapping/profile/widget removal is coherent).
- Full regression suite + Bug Bible (Windows venv), `$env:PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`.
- AST parse on `scene_sequencer.py`.
- Then commit AND push to v2.0-alpha, same session; verify HEAD == origin, no 0-byte files, no BOM.

## 4. Risk / non-goals

- **Risk: low.** No routing path reads the value; removal is confined to declaration + config surface + one positional JSON slot.
- **The only real trap** is the positional `widgets_values` / promoted-input pair on node 3 — handled explicitly in §2.4.
- **Non-goal:** do not touch per-line TTS routing, the voice nodes, or `voice_assignments`. This prune does not alter which engine any line uses.
- **Non-goal:** the second INPUT_TYPES in this file (`:1047`, a different node class) is unaffected — no `default_tts` there.

## 5. Open questions for the panel

1. Any consumer of `sequencer_default_tts` outside this repo (headless applier scripts, soak harness args) that would break on the key's disappearance? (Grep showed doc/soak `.out` references only — confirm none are live inputs.)
2. Is the promoted `default_tts` input on node 3 referenced by any link anywhere in the graph? (Expected: no.)
3. Any test that asserts an exact SceneSequencer widget/input **count** rather than the named widget?

## 6. Verification note (what Claude read)

`nodes/scene_sequencer.py` (INPUT_TYPES 600-645, `sequence()` 684-709, grep for `default_tts` = 2 hits only); `config/profiles/widget_mapping.json:88-103`; `config/profiles/{16gb_full,8gb_lite,cpu_floor}.json` (sequencer_default_tts lines 22/22/19); `workflows/otr_scifi_16gb_full.json` node id 3 parsed live via the Windows venv (widgets_values + inputs, above); repo-wide grep for `default_tts` / `sequencer_default_tts`. ROADMAP.md:712 already lists `default_tts` under SceneSequencer "HIDE" — consistent with retirement.
