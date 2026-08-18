# PLAN after r1 -- audition evidence guards

Hardened from `docs/2026-08-18-audition-evidence-guards/driver_anchor.md` by the
r1 judgment in this folder. Forward-only: this is the plan, not a changelog.

## THE PROBLEM, IN ONE PARAGRAPH

`config/cast_pools.py` cites audio evidence **by sha256** in permanent records.
Three scripts produce that evidence, and they are meant to be re-run. The moment
a record cited the output, the output became an append-only archive and the
script's default path became a destructive one. Two of the three scripts can
destroy a citation today; nothing has actually rotted yet, so this is
**preventive**.

## GROUND TRUTH (verified at HEAD 345ea230, do not re-derive)

| instrument | guard today | verdict |
|---|---|---|
| `scripts/otr_lemmy_production_audition.py` | refuses any non-empty output dir **and** its `_KEY` sibling (`:246-254`), has `--out-dir` (`:223`) | **CORRECT -- the reference shape** |
| `scripts/otr_g1_lemmy_audition.py` | guards on `MANIFEST.json` only (`:253-254`); `_KEY_DIR` created `exist_ok=True` (`:156`) and unguarded; `--overwrite` escape (`:236`) | **PARTIAL -- the trap 12.111 names** |
| `scripts/otr_lemmy_cross_engine_audition.py` | none; `_OUT_DIR` is a module constant (`:57-59`), no flag | **UNGUARDED** |
| `scripts/otr_lemmy_listen_page.py` | `_CAMPAIGN_DIR` hardcoded (`:36`); already refuses to clobber `DECISIONS.json` (`:334-336`) | **BLOCKS the fix** -- cannot target a new campaign |

**Citations, all re-hashed and all matching:** cross-engine `MANIFEST.json`
`ac55c90c...` + six clips; G1 `34dd4c9d...`; production-ceiling `344ccdf8...`.

**Two behaviours that decide the design:**
1. `render()` (`:288-297`) resets `manifest["engines"][engine]` to a fresh row
   with `"clips": {}` and re-renders. There is **no skip-if-complete branch**.
   What resumes is the manifest *merge across runs*, which is why
   `--engine dia` preserves the other engines.
2. `_save_manifest()` (`:268`) restamps `generated_utc` on **every** call and is
   called in the per-engine `finally` (`:329`). **Any** run against the cited
   directory rewrites the manifest and breaks its cited hash -- even one that
   renders nothing.

## THE DESIGN (settled at r1)

**Refuse non-empty, with an idempotent `--resume` -- and no policy import.**
The driver's original cite-aware Option D is withdrawn: `cast_pools.py` carries
four different citation field shapes (`:389-390`, `:537-538`, `:860`, `:995`), so
a cite-scanner inside an instrument would silently miss a fifth and reproduce the
partial-guard defect it is meant to cure. Citation integrity is enforced by
**tests**, not by the instrument.

### D1 -- `scripts/otr_lemmy_cross_engine_audition.py`
* Add `--out-dir` (bare name resolves under `otr/episodes/`, matching
  `otr_lemmy_production_audition.py:234-236`). `_OUT_DIR` stops being a module
  constant read by the writers; thread it or set it once in `main()`.
* **Refuse** to render into an existing non-empty directory unless `--resume`.
  Reject a path that exists and is not a directory.
* **`--resume` must be idempotent, and this is the crux.** For each selected
  engine: if the manifest row is complete AND both clips exist AND both hash to
  what the manifest records, **skip the engine entirely** -- do not load the
  adapter, do not generate, do not rewrite the row.
* **Write no manifest when nothing was rendered.** Track whether any clip was
  actually written this run; if none was, leave `MANIFEST.json` untouched so
  `generated_utc` cannot move. This is what makes a resume against the cited
  directory a true no-op.
* Preserve the existing recovery ergonomics: the `INCOMPLETE:` message
  (`:333-337`) must now name the flag combination that actually works.

### D2 -- `scripts/otr_g1_lemmy_audition.py`
* Widen the guard to the production shape: check **both** `_OUT_DIR` and
  `_KEY_DIR` for existence-and-non-emptiness before rendering; reject a
  non-directory path.
* **Remove `--overwrite`.** `--out-dir` already serves every legitimate use, and
  a single flag that disarms protection on evidence carrying a permanent
  archival promise is not defensible. (Confirm at r2/r3 that nothing scripted
  passes it.)

### D3 -- `scripts/otr_lemmy_listen_page.py`
* Accept `--campaign-dir` / `--out-dir` so a new audition directory can be
  listened to. Keep the existing `DECISIONS.json` non-clobber behaviour.

### D4 -- tests (this is where citation integrity lives)
1. **Double-run refusal**, per 12.111 verify step 1: run the instrument twice
   against the same temp directory; assert the second exits non-zero and writes
   nothing. **Assert on exit code and file mtimes, never on stdout text.**
2. **Partial-state refusal**, verify step 2: a directory holding only clips and
   no manifest; and a `_KEY` directory that exists while the primary does not.
3. **Idempotent resume:** a complete directory + `--resume` renders nothing,
   exits zero, and leaves every mtime *and* the manifest bytes unchanged.
4. **G1's missing rot check** -- the real step-4 gap. The cross-engine routes
   already have one (`tests/test_lemmy_provisional_tier.py:758-796`, passes
   today). `tests/test_voice_identity_fix.py:740-761` only asserts G1's
   *declared* hash string; add a check that opens
   `g1_lemmy_test_a/MANIFEST.json` and re-hashes it, skipping honestly when the
   artifacts are not on the box (reuse the `_output_root` probe pattern).
5. **Roster pin**, not an AST scanner: parameterize the guard tests over the
   three known instruments and pin the roster, so a fourth evidence writer trips
   it and forces a decision.

## OUT OF SCOPE (stated so it is not silently re-litigated)
* **No re-render.** The evidence is intact; re-rendering to test a guard would
  destroy the thing under discussion. Every guard test uses temp directories.
* **No verdict, no qualification.** The three provisional routes stay
  `rendered_pending_listen`.
* **`bark_preset_audition.py`** writes `manifest.json` to
  `docs/2026-06-17-bark-voice/audition/` (`:38`), but nothing cites it by hash.
  Out of scope.
* **Not a workflow change.** These are standalone scripts; nothing here touches
  `workflows/otr_canonical.json`. Confirm explicitly at r3.
* **12.111 verify step 5** (byte-identical-under-changed-code) is **not provable
  without a re-render**, which is forbidden above. Say so; do not claim it.

## GATES
Full suite green against the measured baseline **11028 / 110 / 1**; Bug Bible
regression green; **Sonnet 5 QA on the finished diff BEFORE the push**; then
commit and push, and correct the stale BASELINES figure in `GO_FORWARD_PLAN.md`
in the same change.
