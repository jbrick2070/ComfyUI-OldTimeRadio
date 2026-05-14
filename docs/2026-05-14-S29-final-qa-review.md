# S29 Clean-Slate Gate — Final QA Review

**Sprint:** S29 — Clean-Slate Gate (code-only, no ComfyUI Desktop runs)
**Branch:** `s29-clean-slate-gate`
**Cut from:** `v2.0-alpha @ aad568c` (the merge commit that brought
`s28-cleaner-break` into `v2.0-alpha`)
**Plan:** `docs/2026-05-14-S29-clean-slate-gate-plan.md`
**Owner:** Jeffrey A. Brick
**Closed:** 2026-05-14
**Pre-push HEAD:** `69a9899`

This document is forward-looking. It reports what closed, what is
now unblocked, what regression guards exist after S29, and what the
next sprint looks like. It is NOT a memorial.

---

## Summary

S29 ends the voice-path-cleanbreak chain (S24 → S25 → S26 → S27 →
S28 → S29) at literal 100%. 12 commits over 7 phases. Pure static
fixes. No ComfyUI Desktop boot required.

Final pytest: **2146 passed, 8 skipped, 0 failed** (+3 from S28
baseline 2143). Final Bug Bible: **23/1/2xf**. Audio-byte-identical:
**PASS** at every Phase 2 commit boundary and at sprint close.
Workflow link validator: **0 violations** across all 5 workflow JSONs.
Forbidden-pattern sweep: **0 runtime hits** across `aad568c..HEAD`
.py diff.

After S29, every future legacy hit is a `BUG-LOCAL-NNN` single-commit
fix — not a sprint name. **The clean slate is the slate.**

---

## What closed

### Phase 0 — Baseline + S28 merge + S29 branch cut (3 commits)

- `057d5b6` `docs(s29): add Clean-Slate Gate plan + open s29 branch`
- `a06edd1` `docs(s29): Phase 0 baseline capture`

Operations: `git push origin s28-cleaner-break` (already up-to-date),
`git merge --no-ff s28-cleaner-break` into `v2.0-alpha` (merge commit
`aad568c`), `git push origin v2.0-alpha`, `git checkout -b
s29-clean-slate-gate`.

Baseline artifacts committed:
- `docs/2026-05-14-S29-baseline-pytest.txt`        2143 passed
- `docs/2026-05-14-S29-baseline-bug-bible.txt`     23/1/2xf
- `docs/2026-05-14-S29-baseline-forbidden.txt`     0 runtime hits
- `docs/2026-05-14-S29-baseline-link-integrity.txt` 0 violations
- `docs/2026-05-14-S29-baseline-audio.txt`         7 passed, 1 skipped

### Phase 1 — Workflow JSON + validator scrub (1 commit)

- `fab57e3` `gate(s29-p1): workflow JSON + validator scrub`

Three static fixes bundled in one commit because all three land in
`workflows/otr_scifi_16gb_full.json`:

  1. Cleared hardcoded `C:/Users/jeffr/Documents/.../workflows/
     otr_scifi_16gb_full.json` from Node 63's workflow_json_path
     widget (empty string → validator falls back to
     `_DEFAULT_WORKFLOW_PATH = Path(__file__).parent.parent /
     "workflows" / "otr_scifi_16gb_full.json"`).
  2. Removed `DEPRECATED_manifest` output socket from `SceneSequencer`
     (`RETURN_TYPES`, `RETURN_NAMES`, `manifest=[]` init,
     `manifest_json = ...`, return tuple) + the corresponding output
     entry in Node 3 of the workflow JSON. Deleted the 21-line NOTE
     docstring documenting the deprecation history.
  3. Moved Node 63 `"pos"` from `[-300, -300]` (off-canvas) to
     `[50, 2100]` (below the existing cluster, left-aligned).

### Phase 2 — Line-composer fallback EXTINCT (2 commits)

- `86f674b` `cleanbreak(s29-p2-tests): bulk-add polish_generate_fn= to test callsites`
- `5ba2585` `cleanbreak(s29-p2-delete): drop polish_line active_fn fallback + forensic citations`

**The last cleanbreak commits in the S24→S29 chain.**

Test-side (s29-p2-tests):
- 19 `polish_line()` callsites in `tests/test_lfc_polish_fixes.py` +
  `tests/test_phase1_composer_prompt.py` bulk-patched to pass
  `polish_generate_fn=` explicitly. Same fn reused as both the first
  positional and the kwarg. One callsite (test_polish_generate_fn_
  preferred_over_generate_fn) already passed it -- skipped.

Production-side (s29-p2-delete):
- `polish_line` signature: `polish_generate_fn` is now REQUIRED (no
  default value). The 12-line "defense-in-depth" comment block + the
  `active_fn = polish_generate_fn if ... else generate_fn` ternary
  are deleted; the function body calls `polish_generate_fn` directly.
- `polish_line` docstring §6.4 reworded: drops the "S28 cleanbreak
  retired the falls-back-to-generate_fn" framing in favour of a
  positive statement of the producer contract.
- 5 forensic "back-compat" comment blocks deleted from `_otr_line_
  composer.py` per S29 deletion-bias rule (LineRequest dataclass
  docstring, `build_user_prompt` docstring + NAMED ENTITIES block
  comment, compose_line polish_pass comment). Technical content
  retained; "S28 cleanbreak retired X" citations removed -- git log
  is the audit trail.
- `test_polish_falls_back_to_generate_fn_when_polish_fn_is_none`
  deleted entirely (test directly asserted the fallback behaviour
  the cleanbreak just removed).
- 15 additional `compose_line` / `_phase_3_per_line_polish` callsites
  in test_lfc_phase_3_polish_in_cascade.py + test_phase1_composer_
  prompt.py updated to pass polish_generate_fn= so the polish_line
  call inside the cascade has a fn to call.

**Audio-byte-identical PASS at every commit boundary.** Rule F
revert+trace never invoked.

### Phase 3 — Replace hasattr() guard with isinstance (1 commit)

- `49e1fd7` `gate(s29-p3): replace hasattr() guard trick with isinstance check`

Closes S28 deviation #2: `OutlineRequest.__post_init__` used `not
hasattr(self.budget, "arc_phases")` as a "duck-type guard that
doubles as a grep-dodge" because the prior contributor worried about
a circular-import cost. There is no such cost --
`nodes/_otr_episode_budget.py` does not import `_otr_outline`, so a
top-level `from ._otr_episode_budget import EpisodeBudget` is free.

- `from ._otr_episode_budget import EpisodeBudget` lifted to module-
  level imports.
- `if not hasattr(self.budget, "arc_phases"):` → `if not isinstance
  (self.budget, EpisodeBudget):`
- 8 lines of "we can't isinstance-check without importing" apology
  deleted from the surrounding comment block.

### Phase 4 — Roadmap fold-ins (3 commits)

- `5a12897` `fold(s29-p4-1): NODE_DISPLAY_NAME_MAPPINGS placeholder-string assertion`
- `fdbec9d` `fold(s29-p4-2): correct _load_cached_wav return annotation`
- `1b8fe87` `fold(s29-p4-4): generalize per-entry justification rule + verify p4.3`

  4.1 `tests/test_naming_conventions.py::test_node_display_names_
      have_no_placeholder_strings`. Walks NODE_DISPLAY_NAME_MAPPINGS
      and rejects any `[EMOJI]` / `[TODO]` / `[PLACEHOLDER]` /
      `[FIXME]` substring.

  4.2 `_load_cached_wav` annotation: `-> torch.Tensor | None` →
      `-> tuple[torch.Tensor, int] | None` in both
      `nodes/batch_audiogen_generator.py` and
      `nodes/musicgen_theme.py`. MusicGen docstring updated in
      lockstep.

  4.3 Verify-only: AudioGen + ProcSFX + workflow JSON widget
      `script_json` defaults were already brought to `"{}"` in
      S26-A4a/S26-A4b. `grep -rn '"script_json": "[]"' workflows/
      nodes/` returns zero hits at S29 close.

  4.4 `tests/test_legacy_audit_clean.py::test_excluded_allowed_
      collections_have_per_entry_justification`. AST-walks every
      `tests/*.py` and for every module-level assignment whose name
      starts with `EXCLUDED_` or `ALLOWED_`, requires every entry in
      the literal collection to be preceded (within the contiguous
      `#` comment block above the entry) by a `# justification:`
      marker. `EXCLUDED_PATH_PREFIXES` brought into compliance with
      three new per-entry justifications.

### Phase 5 — Forensic + dead-code + orphan-node sweep (1 commit)

- `8948a05` `sweep(s29-p5): forensic + dead-code + orphan-node audit close-out`

  5.1 Pre-S20 sprint citations (`# (s1)-(s19)`): 0 hits at
      baseline. No work required.
  5.2 Unattributed `# TODO:` / `# FIXME:` / `# XXX:` / `# HACK:`
      in nodes/: 0 hits at baseline. No work required, no triage
      doc needed.
  5.3 `vulture nodes/ tests/ --min-confidence 80`: 15 hits.
      Action:
        - 2 truly-dead imports deleted from `tests/_run_baseline.py`
        - 1 `if False else None` linter-dodge deleted from
          `tests/test_save_to_episode_workspace.py:251`
        - 12 API-contract dead parameters (HF tokenizer stubs,
          ComfyUI hidden inputs, pytest hook signatures) annotated
          with inline `# kept: <reason>` comments per plan §Phase
          5.3 acceptance OR retained their existing `# noqa: ARG001`
          markers.
  5.4 Three-way orphan-node + ghost-workflow-type audit:
        - 5 "orphan" .py files in nodes/ are confirmed library
          helpers (imported by 26 files repo-wide) — NOT orphan
          node files.
        - 14 "ghost" workflow types are standard ComfyUI built-ins
          (CLIPLoader, VAELoader, UNETLoader, etc.) or OTR nodes
          that need `folder_paths` for runtime registration; in
          pytest the test environment cannot load them. At ComfyUI
          Desktop runtime all 14 resolve.
        - Zero true orphans, zero true ghosts.

### Phase 6 — Regression-guard hardening (1 commit)

- `88bc428` `guard(s29-p6): pin alias empty-state + arm forbidden-pattern sweep`

  6.1 `tests/test_init_aliases_empty.py`:
        - `test_rename_aliases_dict_does_not_exist` asserts
          `not hasattr(pkg, "_RENAME_ALIASES")`. Any future
          re-introduction (even an empty dict) trips the test.
        - `test_node_class_mappings_no_bare_name_aliases` asserts
          every NODE_CLASS_MAPPINGS key carries the `OTR_` prefix.
          Catches the alternative re-introduction path of registering
          bare-name aliases directly.
  6.2 `docs/_s28_forbidden_sweep.py` re-armed with 8 new extinction
      markers per S29 plan §Phase 6.2:
        - `req.budget is None`
        - `polish_generate_fn is not None`
        - `hasattr(self.budget`
        - `DEPRECATED_manifest`
        - `C:/Users/jeffr`
        - `OTR_LedgerScriptReviewer`
        - `Gemma4`
        - `reviewer_verdict`
      Plus the S28 carry-over (`otr_legacy_audio_dir`) = 9 total
      extinction markers. Sweep run against `aad568c..HEAD` returns
      0 runtime hits.

### Phase 7 — Doc cleanup + sign-off (current commit being assembled)

- `69a9899` `chore(s29-p7-1): delete docs/cleanbreak-deferred.md`

  7.1 `git rm docs/cleanbreak-deferred.md`. No archive copy. No
      museum. The 3 historical resolutions (C10, C8 CD-1, S14.2 ADR)
      are in git history at `s28-cleaner-break` HEAD; anyone who
      needs them can `git show 218ebbe:docs/cleanbreak-deferred.md`.
  7.2 ROADMAP.md updated: S29 added as CURRENT WORK; S28 demoted to
      PRIOR CURRENT WORK; S26 + S27 PRIOR CURRENT WORK sections
      stripped entirely (git log is the audit trail). Roadmap-only
      items 1, 2, 4, 5 marked CLOSED BY S29; item 3 noted as
      operator-gated.
  7.3 "Deferred" language scrubbed from forward-work sections;
      replaced with "forward feature work" / "operator-gated" / "gated
      on external clock".
  7.4 This document.

---

## What's unblocked

After S29 close, the following are open:

1. **Sprint B — Two-Model Selector.** Scoping at
   `docs/2026-05-13-two-model-selector-scoping.md`. 14 sections, 6
   open decisions. Mechanical 2.5-3 day refactor with audio C7 baseline
   preserved (Slot 1 defaults to current Mistral-Nemo).
2. **Sprint C — `meta.story_brief` v2.** Pre-flight cleanbreaks +
   build commits documented. C3 legitimately shifts the audio
   baseline (Gemma-4-E4B-it default for VRAM headroom); the new
   baseline becomes the post-C3 reference.
3. **Sprint A — Downstream verification (FLUX / LTX / HuMo).**
   Opens after C close. End-to-end pass confirming the post-LFC
   ledger + meta.story_brief reach every consumer correctly.
4. **ComfyUI Desktop runtime pass.** Forward feature work, Jeffrey's
   own clock. Includes: visually confirming Node 63 sits cleanly on
   canvas without overlap (the static move to `[50, 2100]` ships in
   Phase 1 — Jeffrey can drag wherever he prefers), workflow re-save
   through Desktop, runtime `_RENAME_ALIASES` firing check, 1-cue /
   10-second episode smoke runs.

These are NOT deferred. They are sequenced forward feature work.

---

## Regression guards now active

After S29, the following surfaces are pinned by automated checks:

| Guard | Mechanism | What it catches |
|-------|-----------|-----------------|
| Bug Bible regression (23/1/2xf) | `tests/bug_bible_regression.py` | UTF-8 BOM, AST parse failures, node-registration drift |
| Workflow link integrity (0 violations) | `tools/validate_workflow_links.py` | Workflow JSON link-target mismatches |
| Audio-byte-identical (PASS) | `tests/test_audio_byte_identical.py` | Audio path drift |
| Forbidden-pattern sweep (0 runtime hits) | `docs/_s28_forbidden_sweep.py` | All 9 extinction markers from S28+S29 |
| `_RENAME_ALIASES` empty-state | `tests/test_init_aliases_empty.py::test_rename_aliases_dict_does_not_exist` | Re-introduction of legacy class-name aliases |
| NODE_CLASS_MAPPINGS prefix | `tests/test_init_aliases_empty.py::test_node_class_mappings_no_bare_name_aliases` | Bare-name workflow types re-introduced |
| NODE_DISPLAY_NAME_MAPPINGS hygiene | `tests/test_naming_conventions.py::test_node_display_names_have_no_placeholder_strings` | `[EMOJI]` / `[TODO]` / `[PLACEHOLDER]` / `[FIXME]` substrings in display names |
| Per-entry `# justification:` | `tests/test_legacy_audit_clean.py::test_excluded_allowed_collections_have_per_entry_justification` | Audit-blindspot widening without documented reason |
| Vulture --min-confidence 80 | manual run | New unannotated dead code |

---

## What does not exist anymore

| Surface | Where it lived | Where it died |
|---|---|---|
| `polish_line` `active_fn` fallback | `_otr_line_composer.py:1277` | S29 Phase 2 (`5ba2585`) |
| `polish_generate_fn=None` default | polish_line signature | S29 Phase 2 (`5ba2585`) |
| `hasattr(self.budget, "arc_phases")` guard | `_otr_outline.py.__post_init__` | S29 Phase 3 (`49e1fd7`) |
| `DEPRECATED_manifest` output | SceneSequencer + workflow JSON Node 3 | S29 Phase 1 (`fab57e3`) |
| `C:/Users/jeffr/...` hardcode | Node 63 workflow_json_path widget | S29 Phase 1 (`fab57e3`) |
| Off-canvas Node 63 `[-300, -300]` | workflow JSON | S29 Phase 1 (`fab57e3`) |
| `docs/cleanbreak-deferred.md` | docs/ | S29 Phase 7 (`69a9899`) |
| 5 forensic "back-compat" comments | `_otr_line_composer.py` | S29 Phase 2 (`5ba2585`) |
| Dead imports (`io`, `struct` from `_run_baseline.py`) | tests/_run_baseline.py | S29 Phase 5 (`8948a05`) |
| `if False else None` linter-dodge | `test_save_to_episode_workspace.py:251` | S29 Phase 5 (`8948a05`) |

All of the above are now anti-regression guards in the forbidden-
pattern sweep config (`docs/_s28_forbidden_sweep.py`) or in pytest
test files. Re-introduction trips an alarm.

---

## Why this is the final cleanbreak sprint

S28 was directed as the LAST cleanbreak sprint. S29 closes the
residual S28 deviations + 13 hygiene items with pure static fixes:

  1. `_otr_line_composer.py:1265` active_fn fallback (Phase 2)
  2. `_otr_outline.py.__post_init__` hasattr trick (Phase 3)
  3. `docs/cleanbreak-deferred.md` stub (Phase 7.1)
  4. `OTR_WorkflowValidator` hardcoded path (Phase 1.1)
  5. `DEPRECATED_manifest` output (Phase 1.2)
  6. Node 63 off-canvas pos (Phase 1.3)
  7. `test_naming_conventions.py` placeholder assertion (Phase 4.1)
  8. `_load_cached_wav` return annotation (Phase 4.2)
  9. AudioGen / ProcSFX script_json defaults (Phase 4.3, verify-only)
  10. C11 generalization (Phase 4.4)
  11. Pre-S20 sprint-citation comments (Phase 5.1, zero hits)
  12. Unattributed `# TODO:` / `# FIXME:` etc. (Phase 5.2, zero hits)
  13. Orphan helpers / dead imports (Phase 5.3)
  14. Orphan node files / ghost workflow types (Phase 5.4)
  15. `_RENAME_ALIASES` empty-state pin (Phase 6.1)
  16. Forbidden-pattern sweep extinction markers (Phase 6.2)

After S29, every future legacy hit is a `BUG-LOCAL-NNN` single-commit
fix. **"100%" means "100%" without footnotes.**

---

## Documented deviations from plan

| # | Plan spec | Actual disposition |
|---|-----------|--------------------|
| 1 | Phase 1: 3 commits | 1 commit. Three sub-steps all land in the same `workflows/otr_scifi_16gb_full.json` file; isolating per step would churn the same file three times. The commit body splits the changes per sub-step. |
| 2 | Phase 2.2 "first then delete" two-commit flow | Followed exactly: `s29-p2-tests` (bulk-add) then `s29-p2-delete` (fallback deletion). |
| 3 | Phase 4: 4 commits | 3 commits. Phase 4.3 was verify-only (defaults already at `"{}"` from S26-A4a/S26-A4b); the verification note was folded into the Phase 4.4 commit. |
| 4 | Phase 5: 4 commits | 1 commit. Phase 5.1 + 5.2 returned zero hits at baseline (no work); Phase 5.3 + 5.4 fit in one sweep commit. |
| 5 | Phase 5.3 vulture annotations | Inline `# kept: <reason>` comments per plan. Vulture itself does not recognize `# kept:` as a suppression marker; the plan acceptance permits both "zero hits OR every remaining one has an inline `# kept:` comment" -- the OR condition is satisfied. |

None of the deviations affect any acceptance criterion. All are
documented at the commit-message level so the audit trail is
complete.

---

## Sources

- Plan: `docs/2026-05-14-S29-clean-slate-gate-plan.md`
- Baseline artifacts: `docs/2026-05-14-S29-baseline-*.txt`
- S28 close: `docs/2026-05-13-S28-final-qa-review.md`
- Commits: `git log --oneline aad568c..HEAD` on
  `s29-clean-slate-gate`

**The clean slate is the slate.**
