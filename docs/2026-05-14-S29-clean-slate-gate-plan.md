# S29 — Clean-Slate Gate (code-only, no ComfyUI runs)

**Branch:** `s29-clean-slate-gate` (cut from `v2.0-alpha` HEAD after S28
push + merge)
**Owner:** Jeffrey A. Brick
**Goal:** Close the residual S28 deviations as pure static fixes. End the
voice-path-cleanbreak chain at literal 100% without any ComfyUI Desktop
boot. ComfyUI runtime work stays forward feature work — NOT a gate.
**Discipline:** Review → Code → Wire → Regress → Commit, every phase.

## Deletion bias (S29 principle)

**Git log is the audit trail.** S29 does NOT preserve archive docs
for deleted state. If a future agent needs to know what the pre-S20
forensic comments said, what `cleanbreak-deferred.md` contained, or
what the old line-composer fallback looked like — they `git log`,
not `cat docs/archive/whatever.md`.

When S29 deletes something, it deletes. No museum. No memorial. No
"keeping for audit trail" inside the v2.0-alpha branch. The clean
slate is the slate.

---

## What this sprint is NOT

- Not a ComfyUI runtime sprint. Desktop boot, smoke runs, and runtime
  alias validation are all OUT OF SCOPE. Jeffrey is not at the
  ComfyUI stage yet; that work runs on its own clock. Node 63's
  position IS fixed in this sprint, but as a pure JSON edit — no
  Desktop boot, no visual verification.
- Not another cleanbreak chain link. S28 was the last cleanbreak
  sprint by directive. S29 closes residual gaps S28 documented as
  deviations. After S29, the chain is over.
- Not feature work. B / C / A open AFTER S29 closes.
- Not a soak run. Small, mechanical, audit-driven, autonomous.

## Why S29 exists at all

S28 signed off as PASS but with three named deviations:

1. `_otr_line_composer.py:1265` runtime fallback retained.
2. `_otr_outline.py.__post_init__` uses `not hasattr(self.budget, ...)`
   as a guard-grep workaround instead of a clean type check.
3. `docs/cleanbreak-deferred.md` retained as a stub instead of deleted.

Plus three static workflow JSON / validator hygiene items that were
documented but never closed:

4. `OTR_WorkflowValidator` has a hardcoded `C:/Users/jeffr/...`
   absolute path.
5. `DEPRECATED_manifest` output / JSON reference still present.
6. Node 63 (`OTR_WorkflowValidator`) stranded at `[-300, -300]`
   off-canvas — fixable as a pure JSON `"pos"` edit, no Desktop boot
   required.

Plus four ROADMAP "fold in when convenient" items that have been
sitting in the convenient-list for multiple sprints:

7. `tests/test_naming_conventions.py` missing placeholder-string
   assertion (`[EMOJI]` / `[TODO]` / `[PLACEHOLDER]` / `[FIXME]`).
8. `_load_cached_wav` declared return-type doesn't match runtime
   (AudioGen + MusicGen).
9. AudioGen / ProcSFX default `script_json` shipped as `"[]"` (legacy
   parser-list shape) instead of `"{}"` (v2 ledger dict).
10. C11 per-entry `# justification:` rule only covers `EXCLUDED_PATHS`
    in one test; should generalize to all `EXCLUDED_*` / `ALLOWED_*`
    collections.

Plus a forensic sweep that retires obsolete cruft and locks the door
on everything S28 + S29 just extincted:

11. Pre-S20 sprint-citation comments littering `nodes/` / `tests/`
    (delete inline; git log preserves them if anyone ever needs them).
12. Unattributed `# TODO:` / `# FIXME:` / `# XXX:` / `# HACK:`
    comments without `BUG-LOCAL-NNN` tracking.
13. Orphan helpers / dead imports flagged by `vulture`.
14. Orphan node files in `nodes/` not registered in
    `__init__.NODE_CLASS_MAPPINGS`, and ghost workflow `"type"`
    references that don't resolve to a registered class.
15. `_RENAME_ALIASES` empty-state not pinned by a regression test.
16. Forbidden-pattern sweep config missing the S28 + S29 extinction
    markers as anti-regression guards.

S29 closes all 16 with pure static fixes + pytest regression. No
ComfyUI runtime required. No archive docs. After S29 the legacy floor
is swept clean and every future regression-attempt hits a guard.

---

## Phase 0 — Baseline + S28 push/merge (3 commits)

| Step | Action |
|------|--------|
| 0.1 | `git push origin s28-cleaner-break` |
| 0.2 | Merge `s28-cleaner-break` → `v2.0-alpha` (no rebase, preserve audit trail) |
| 0.3 | Cut `s29-clean-slate-gate` from `v2.0-alpha` HEAD |
| 0.4 | Phase 0 baseline: pytest, Bug Bible, forbidden-pattern sweep, workflow link integrity, audio-byte-identical |
| 0.5 | Commit baseline artifacts to `docs/2026-05-14-S29-baseline-*` |

**Acceptance:**
- S28 commits visible on `origin/v2.0-alpha`
- Pytest baseline 2143 passed / 8 skipped / 0 failed (matches S28 close)
- Bug Bible 23 / 1 / 2xf
- All 5 workflow JSONs: 0 link violations

---

## Phase 1 — Workflow JSON + validator scrub (3 commits, static only)

| Step | Action |
|------|--------|
| 1.1 | **Fix hardcoded absolute path in `OTR_WorkflowValidator`.** Python code fix: replace `C:/Users/jeffr/Documents/.../workflows/otr_scifi_16gb_full.json` with `Path(__file__).parent.parent / "workflows" / "otr_scifi_16gb_full.json"`. Pytest regression. |
| 1.2 | **Remove `DEPRECATED_manifest` output.** Two-file edit: drop the output socket from the node's `RETURN_NAMES` / `RETURN_TYPES`, then delete the corresponding link entry from `workflows/otr_scifi_16gb_full.json`. Re-run the workflow link validator. |
| 1.3 | **Move Node 63 onto canvas — pure static JSON edit.** In `workflows/otr_scifi_16gb_full.json`, find the node with `"id": 63` (type `OTR_WorkflowValidator`) and change its `"pos"` from `[-300, -300]` to `[50, 2100]` (below the existing cluster, left-aligned with the leftmost column at X=50; the rest of the canvas spans X:50→4900, Y:-180→1850). Re-run the workflow link validator. **DO NOT boot ComfyUI to verify** — visual confirmation happens later, on Jeffrey's own clock. |

**Acceptance:**
- `grep -rn 'C:/Users' workflows/ nodes/` returns zero hits
- `grep -rn 'DEPRECATED' workflows/ nodes/` returns zero hits (forensic
  comments allowed; runtime / JSON values must be zero)
- Node 63 `"pos"` field in `workflows/otr_scifi_16gb_full.json` reads
  `[50, 2100]` (or whatever on-canvas spot Jeffrey prefers — anything
  with both X ≥ 0 and Y ≥ 0 qualifies)
- Workflow JSON link validator: 0 violations
- Pytest green

**Visual confirmation deferred.** The position move is statically
correct (on-canvas, within the existing extent), but whether it
visually overlaps another node won't be known until the next ComfyUI
Desktop session. That's fine — overlap is cosmetic and Jeffrey can
drag it wherever he likes when he opens the workflow. The cleanbreak
acceptance is "node is on-canvas," not "node looks pretty."

---

## Phase 2 — Kill the line-composer fallback (3–5 commits)

**Biggest phase. ~22 test call sites to update.**

| Step | Action |
|------|--------|
| 2.1 | Enumerate every call site of `polish_line()` in `tests/`. Categorize: callers passing one fn vs callers passing both fns. |
| 2.2 | Bulk-edit (sed where uniform, manual where not): every test call passes `polish_generate_fn` explicitly. |
| 2.3 | Regression after the bulk edit: pytest green, no new failures. |
| 2.4 | Delete the runtime fallback at `nodes/_otr_line_composer.py:1265` (`active_fn = polish_generate_fn if polish_generate_fn is not None else generate_fn`). |
| 2.5 | Delete the "defense-in-depth" docstring framing at `:1215`. |
| 2.6 | Final regression: pytest green, audio-byte-identical PASS. |

**Acceptance:**
- `git grep -nE 'back-compat\|legacy fallback\|legacy shape' nodes/_otr_line_composer.py` returns ZERO hits (currently 5 forensic — all go)
- `git grep -n 'polish_generate_fn is not None' nodes/_otr_line_composer.py` returns zero hits
- Audio-byte-identical PASS at every commit boundary
- Rule F discipline: any regression that drifts audio reverts immediately

---

## Phase 3 — Replace hasattr() trick with clean guard (1 commit)

| Step | Action |
|------|--------|
| 3.1 | In `nodes/_otr_outline.py.__post_init__`, replace `if not hasattr(self.budget, "arc_phases"):` with `if not isinstance(self.budget, EpisodeBudget):`. |
| 3.2 | Import `EpisodeBudget` at module level if not already imported. |
| 3.3 | Delete any comment explaining the grep-dodge. |

**Acceptance:**
- `__post_init__` reads naturally — type check, not duck-type-as-grep-evasion
- No comment saying "written this way to dodge a grep"
- Pytest green
- Forbidden-pattern sweep still 0 runtime hits

---

## Phase 4 — Roadmap fold-ins (4 commits)

Items explicitly flagged in `ROADMAP.md` §"Roadmap-only items" as
"fold into adjacent work when convenient." S29 is adjacent work.
Convenient is now.

| Step | Action |
|------|--------|
| 4.1 | **Naming-conventions test broadening.** Extend `tests/test_naming_conventions.py` to assert no `NODE_DISPLAY_NAME_MAPPINGS` value contains `[EMOJI]`, `[TODO]`, `[PLACEHOLDER]`, or `[FIXME]` substrings. Future-proofs the surface that caught the MusicGen `[EMOJI]` instance in S25. |
| 4.2 | **`_load_cached_wav` annotation fix.** Both `nodes/batch_audiogen_generator.py` and `nodes/musicgen_theme.py` declare `_load_cached_wav -> torch.Tensor \| None` but actually return `tuple[torch.Tensor, int] \| None`. Correct both annotations to match runtime contract. |
| 4.3 | **AudioGen / ProcSFX default `script_json` standardization.** Both nodes default `script_json` to `"[]"` (legacy parser-list shape) but parse the v2 ledger dict. Change defaults to `"{}"` in node code AND in `workflows/otr_scifi_16gb_full.json` widgets_values. Matches MusicGen + runtime contract. |
| 4.4 | **C11 generalization — per-entry justification rule.** S24/C11 added `# justification: <reason>` requirement for every `EXCLUDED_PATHS` entry in `tests/test_legacy_audit_clean.py`. Extend the rule (in the same test file) to assert ALL module-level `EXCLUDED_*` / `ALLOWED_*` collections carry per-entry `# justification:` comments. |

**Acceptance:**
- `tests/test_naming_conventions.py` carries the placeholder-string
  assertion; pytest catches a synthetic violation in a smoke test
- `_load_cached_wav` annotation matches runtime in both files
- `grep -rn '"script_json":\s*"\[\]"' workflows/ nodes/` returns zero hits
- Every `EXCLUDED_*` / `ALLOWED_*` collection in `tests/` carries
  per-entry justification comments
- Pytest green

---

## Phase 5 — Forensic comment + dead-code + orphan-node sweep (4 commits)

Bounded scope. Each step has a defined termination condition; do NOT
let this phase sprawl.

| Step | Action |
|------|--------|
| 5.1 | **Delete old sprint-citation comments.** Grep `nodes/ tests/ tools/` for `# (s1)` through `# (s19)` style forensic comments. Just delete the inline comments. **No archive doc.** Git log preserves anything anyone could possibly need. Keep S20+ citations as-is. |
| 5.2 | **TODO / FIXME / XXX / HACK triage.** Generate `docs/2026-05-14-S29-todo-triage.md` enumerating every `# TODO:` / `# FIXME:` / `# XXX:` / `# HACK:` comment in `nodes/`. Each gets one of three actions, applied in this commit: (a) FIX inline if trivial, (b) FILE as `BUG-LOCAL-NNN` in BUG_LOG and convert the comment to `# BUG-LOCAL-NNN — see BUG_LOG`, or (c) DELETE if stale. Final state: zero unattributed `# TODO:` / `# FIXME:` / `# XXX:` / `# HACK:` comments in `nodes/`. |
| 5.3 | **Orphan helper + dead import sweep.** Run `vulture nodes/ tests/ --min-confidence 80` (or equivalent: `python -m pyflakes`, `grep`-based call-site audit). For each reported orphan function / class / import: confirm zero call sites repo-wide, then delete. False positives (test fixtures, `__all__` exports, lazy-loaded entry points) get a `# kept: <reason>` comment in-line. **No allowlist doc.** If a kept item needs explanation, the inline comment is the explanation. |
| 5.4 | **Node-registration audit — hunt orphaned old nodes.** Three-way diff: (a) files in `nodes/` (one node per file pattern), (b) keys in `__init__.py` `NODE_CLASS_MAPPINGS`, (c) `"type"` values across all 5 workflow JSONs. Any file in `nodes/` not registered in `__init__.py` → delete the file. Any registration not used in any workflow AND not a CLI/utility entry point → flag in commit body, then delete. Any workflow `"type"` that doesn't resolve to a registered class → workflow JSON is referencing a ghost; fix or delete the JSON entry. |

**Acceptance:**
- `git grep -nE '# \(s([1-9]\|1[0-9])\)' nodes/ tests/ tools/` returns zero hits
- `git grep -nE '# (TODO\|FIXME\|XXX\|HACK):' nodes/` returns only
  `BUG-LOCAL-NNN`-attributed entries (or zero)
- `vulture` reports zero high-confidence orphans, or every remaining
  one has an inline `# kept: <reason>` comment
- Every `.py` file in `nodes/` corresponds to a registered class in
  `__init__.py NODE_CLASS_MAPPINGS` (zero orphan node files)
- Every workflow `"type"` resolves to a registered class (zero ghost
  references in any of the 5 workflow JSONs)
- Pytest green; no functional change to runtime behavior

---

## Phase 6 — Regression-guard hardening (2 commits)

Lock the door shut. Every S28 + S29 extinction surface gets a guard
that fires if a future agent re-introduces it.

| Step | Action |
|------|--------|
| 6.1 | **Pin `_RENAME_ALIASES` empty-state with a unit test.** Add `tests/test_init_aliases_empty.py` asserting `__init__._RENAME_ALIASES == {}` (or equivalent if the dict structure differs). Any future entry trips the test. |
| 6.2 | **Re-arm forbidden-pattern config with S28+S29 extinction surfaces.** Append to the forbidden-pattern sweep config (or `docs/_s28_forbidden_sweep.py` successor): `otr_legacy_audio_dir`, `req.budget is None`, `polish_generate_fn is not None`, `hasattr(self.budget`, `DEPRECATED_manifest`, `C:/Users/jeffr`, `OTR_LedgerScriptReviewer`, `Gemma4`, `reviewer_verdict`. Anything that was extinct STAYS extinct. |

**Acceptance:**
- `tests/test_init_aliases_empty.py` passes; flips RED if any rename
  alias is re-introduced
- Forbidden-pattern sweep config carries all 9 extinction markers;
  sweep reports 0 runtime hits at sprint close
- Pytest green

---

## Phase 7 — Doc cleanup + sign-off (3 commits)

| Step | Action |
|------|--------|
| 7.1 | **Delete `docs/cleanbreak-deferred.md` outright.** `git rm` it. **No archive doc.** The 3 historical resolutions (C10, C8 CD-1, S14.2 ADR) are in git history at `s28-cleaner-break` HEAD; anyone who needs them can `git show`. |
| 7.2 | **Update ROADMAP top section.** Current state reflects S29 close. B2 / B4 / B5 / B6 marked **CLOSED BY S28**, not "next" or "deferred." Strip S26 / S27 status blocks entirely — git log has them if needed. Mark Roadmap-only items 1, 2, 3, 4 as **CLOSED BY S29**. |
| 7.3 | **Reword "Forward work" sections** across ROADMAP + S28 QA doc. S28's forward work (sync drift, LTX metadata, story brief v2, downstream verification, ComfyUI Desktop runtime smoke) becomes "post-cleanbreak feature work" — never "deferred." |
| 7.4 | **Final S29 QA review:** `docs/2026-05-14-S29-final-qa-review.md` mirroring the S28 final QA format. This file is forward-looking (what's now blocked unblocked, what guards exist, what the next sprint looks like) — NOT a memorial. |

**Acceptance:**
- `ls docs/cleanbreak-deferred.md` returns ENOENT
- `ls docs/archive/` does NOT contain any S29-created files (no
  cleanbreak-history.md, no sprint-citations-pre-s20.md, no
  vulture-allowlist.md, no todo-triage.md). Deletion-bias holds.
- `grep -rn 'deferred' docs/ROADMAP.md` returns only forward-feature
  references (S14.2, S19.3, post-v2.0 ADRs), never cleanbreak
- Final QA doc references this plan and the per-phase audit results
- Final QA doc references this plan and the per-phase audit results

---

## Final acceptance criteria (sprint close)

| # | Check | Target |
|---|-------|--------|
| 1 | Pytest | 2143 ±10 passed (delta from baseline within Rule F tolerance; Phase 6 adds 1 new test) |
| 2 | Bug Bible | 23 passed / 1 skipped / 2 xfailed |
| 3 | Forbidden-pattern sweep | 0 runtime hits |
| 4 | Workflow link validator | 0 violations across all 5 JSONs |
| 5 | Audio-byte-identical | PASS at every Phase 2 commit boundary + final |
| 6 | `cleanbreak-deferred.md` | Does not exist |
| 7 | `C:/Users/` in workflows / nodes | Zero hits |
| 8 | `DEPRECATED_*` in workflow JSON | Zero hits |
| 9 | `polish_generate_fn is not None` in `_otr_line_composer.py` | Zero hits |
| 10 | `hasattr(self.budget, ...)` in `_otr_outline.py` | Replaced with `isinstance` |
| 11 | Node 63 `"pos"` in `workflows/otr_scifi_16gb_full.json` | On-canvas (X ≥ 0, Y ≥ 0); `[50, 2100]` recommended |
| 12 | `NODE_DISPLAY_NAME_MAPPINGS` placeholder assertion | Active in `tests/test_naming_conventions.py` |
| 13 | `_load_cached_wav` annotation | `tuple[torch.Tensor, int] \| None` in both files |
| 14 | `"script_json": "[]"` in workflows / nodes | Zero hits |
| 15 | `EXCLUDED_*` / `ALLOWED_*` without `# justification:` | Zero violations |
| 16 | `# (s1)` through `# (s19)` forensic citations in code | Zero hits (just deleted; no archive) |
| 17 | Unattributed `# TODO:` / `# FIXME:` / `# XXX:` / `# HACK:` in `nodes/` | Zero hits |
| 18 | `vulture nodes/ tests/ --min-confidence 80` | Zero hits OR inline `# kept: <reason>` comment |
| 19 | `tests/test_init_aliases_empty.py` | Passes, asserting `_RENAME_ALIASES == {}` |
| 20 | Forbidden-pattern config | Carries all 9 S28+S29 extinction markers |
| 21 | Orphan node files / ghost workflow types | Zero hits (every `nodes/*.py` registered; every workflow `"type"` resolves) |
| 22 | `docs/archive/` files created by S29 | Zero (deletion-bias holds) |

**No ComfyUI Desktop runtime smoke. No visual layout check. No
runtime alias check.** Those are forward feature work.

---

## Strict out-of-scope (do NOT pull into S29)

If a surface comes up during S29 that isn't in the 7 phases above, log
it as `BUG-LOCAL-NNN` and append to `BUG_LOG.md`. Do NOT expand scope.

**ComfyUI runtime work (forward feature, not a gate):**
- Visual confirmation that Node 63 sits cleanly on canvas without
  overlap (the static move to `[50, 2100]` ships in Phase 1; Jeffrey
  can drag it wherever he prefers on his next Desktop session)
- ComfyUI Desktop load smoke
- Workflow re-save through Desktop
- Runtime `_RENAME_ALIASES` firing check
- 1-cue / 10-second episode smoke runs
- Any Desktop boot whatsoever

**Sequenced feature work (open AFTER S29):**
- Sprint B — Two-Model Selector (`model_creative` + `model_technical`
  on `OTR_LedgerScriptWriter`)
- Sprint C — `meta.story_brief` v2
- Sprint A — Downstream verification + repair (FLUX / LTX / HuMo)

**Operator-gated (waiting on external clock):**
- S14.2 auto-invoke (waits on integration-path decision)
- S19.3 survival-guide promotion (waits on 2-3 clean sprints)
- Per-consumer `audit_post_freeze_writeback` strict-mode flips

**Post-v2.0 / future-ADR (gated on v2.0 ship):**
- D1 — targeted repair pass for zero `key_terms`
- D2 — MusicGen cues ADR
- D3 — FLUX RADIO portrait fallback ADR
- Three-File Contract promotion of BUG-LOCAL-221 / 222 / 223
- Tier-2 LLM A/B (Talkie-1930-it, Qwen3.6-27B, Gemma-4-31B-it)

**Forward feature/quality work (not cleanbreak, not deferred):**
- Audio/video sync drift
- LTX clip metadata `start_s` timestamp bugs
- Gaussian splat rendering integration
- SIGNAL LOST narrative layer

---

## Commit subject conventions (match S28 patterns)

| Prefix | Use |
|--------|-----|
| `docs(s29): ...` | Plan, baseline, QA artifacts, audit docs |
| `gate(s29-pN-x): ...` | Phase 1 (workflow scrub), Phase 3 (hasattr) |
| `cleanbreak(s29-p2-x): ...` | Phase 2 surgery — the LAST cleanbreak commits ever |
| `fold(s29-p4-x): ...` | Phase 4 roadmap fold-ins |
| `sweep(s29-p5-x): ...` | Phase 5 forensic sweep |
| `guard(s29-p6-x): ...` | Phase 6 regression-guard hardening |
| `chore(s29-p7-x): ...` | Phase 7 doc moves / deletions |
| `fix(s29-pN-name): ...` | In-scope bug fixes that surface mid-phase |

---

## Sign-off

S29 ends the voice-path-cleanbreak chain at literal 100%. Pure static
fixes. No ComfyUI Desktop required.

After this sprint closes:

- Every future legacy hit is a `BUG-LOCAL-NNN` single-commit fix.
- `docs/cleanbreak-deferred.md` no longer exists. Don't recreate it.
- The next sprint name in `docs/<date>-S*` is **Sprint B —
  Two-Model Selector**.
- ComfyUI runtime work happens on Jeffrey's own clock — not gated by
  the cleanbreak chain.
- "100%" means "100%" without footnotes.

**The clean slate is the slate.**
