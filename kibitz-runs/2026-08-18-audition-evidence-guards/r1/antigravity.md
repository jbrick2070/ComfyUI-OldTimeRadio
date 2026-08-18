VERDICT: yes-with-fixes. The core guard strategy (Option D) introduces an inverted dependency coupling CLI scripts to policy configurations while overlooking that the current render loop does not actually skip existing clips during resume.

MUST-FIX BEFORE BUILD:
1. [Section 3 / Section 2B] Flawed Resumption Model in Render Loop
   - Defect: Section 2B assumes `scripts/otr_lemmy_cross_engine_audition.py` is safely resumable because `_load_manifest()` merges JSON data. However, `render()` (scripts/otr_lemmy_cross_engine_audition.py:286-321) iterates over `engines`, unconditionally resets `manifest["engines"][engine] = row`, and overwrites clips on disk via `_write_clip()`. Re-running `--render` without manually filtering `--engine <failed_engine>` wipes out previously rendered clips and invalidates their hashes.
   - Concrete Fix: Implement true resumption inside `render()`: when `--resume` is passed, check if `manifest["engines"][engine]` is already complete and its clip files exist on disk with matching hashes before invoking `adapter.generate_voice()`.

2. [Section 3] Inverted Architecture Coupling in Option D (Script-to-Policy Dependency)
   - Defect: Option D proposes importing and parsing `config/cast_pools.py` inside `otr_lemmy_cross_engine_audition.py` to dynamically detect if output paths are cited. This creates a reverse dependency where build/rendering tools depend on policy schema, fails if `cast_pools.py` syntax/structure changes, and provides zero protection for uncited in-progress runs or non-policy citations.
   - Concrete Fix: Drop the runtime `cast_pools.py` import check from the audition script. Adopt strict non-empty directory refusal (Option C) as the primary guard: refuse execution if `--out-dir` exists and is non-empty unless `--resume` is explicitly passed. Enforce citation integrity via static regression tests (Section 5 Acceptance Step 4).

3. [Section 4 / Section 2A] Omission of `otr_lemmy_listen_page.py` Parameterization
   - Defect: Section 4 excludes `scripts/otr_lemmy_listen_page.py` from scope. However, `otr_lemmy_listen_page.py` hardcodes `_CAMPAIGN_DIR = os.path.join(_EPISODES, "lemmy_cross_engine")` (scripts/otr_lemmy_listen_page.py:36). If `--out-dir` is added to `otr_lemmy_cross_engine_audition.py`, the listen page generator cannot target or evaluate any new audition output directory.
   - Concrete Fix: Include `scripts/otr_lemmy_listen_page.py` in scope to accept `--out-dir` / `--campaign-dir` matching the audition script's output destination.

4. [Section 2D / Section 3] Incomplete Guard Surface on G1 Audition Sibling
   - Defect: `scripts/otr_g1_lemmy_audition.py` only checks for `MANIFEST.json` and creates `_KEY_DIR` unconditionally with `exist_ok=True` (scripts/otr_g1_lemmy_audition.py:156, 253-254), leaving blinding keys and orphan WAVs vulnerable to silent overwriting.
   - Concrete Fix: Bring `otr_g1_lemmy_audition.py` to full parity with `otr_lemmy_production_audition.py:246-254` by checking both `_OUT_DIR` and `_KEY_DIR` for non-emptiness before rendering, and remove or deprecate `--overwrite` in favor of `--out-dir`.

SHOULD-FIX:
1. [Section 5, Step 4] POSIX vs Windows Path Normalization in Standing Re-Hash Validator
   - Defect: Receipts in `config/cast_pools.py` use forward slashes (e.g. `otr/episodes/lemmy_cross_engine/MANIFEST.json` at config/cast_pools.py:1095), whereas Windows filesystem lookups resolve with backslashes.
   - Concrete Fix: Ensure the acceptance step 4 re-hash harness normalizes relative paths with `os.path.normpath` or `pathlib.Path` before reading file bytes.

2. [Section 2A] Atomic Directory Creation Timing
   - Defect: `render()` in `scripts/otr_lemmy_cross_engine_audition.py:282` creates `_OUT_DIR` before preflight checks or server warning gates complete, leaving dirty empty directories if aborted.
   - Concrete Fix: Defer `os.makedirs(_OUT_DIR)` until after preflight resolution and resident server checks have passed.

OPTIONAL / NICE-TO-HAVE:
1. [Section 2A] Relative `--out-dir` Resolution
   - Allow bare folder names passed to `--out-dir` in `otr_lemmy_cross_engine_audition.py` to resolve under `_EPISODES`, matching `otr_lemmy_production_audition.py:234-236`.

CUT THESE (scope / over-engineering):
1. [Section 5, Step 3] Generic AST Scanner for "Unguarded Evidence Writers"
   - Why safe to cut: Building an AST/heuristic test that attempts to detect all arbitrary evidence writers across the codebase is brittle, prone to false positives, and over-engineered. A deterministic parameterized test verifying the three known audition CLI instruments (`production`, `cross_engine`, `g1`) is sufficient, maintainable, and completely covers the defect class.

2. [Section 3] Runtime Citation Introspection in CLI Instruments (Option D Citation Check)
   - Why safe to cut: Dynamic reflection into `config/cast_pools.py` at runtime adds unnecessary fragility. Strict non-empty directory blocking + explicit `--resume` provides 100% of the required overwrite safety without runtime coupling.

[ASSUMPTION] Assumed that `scripts/bark_preset_audition.py` is an uncommitted local-temp runner (writes WAVs to `<temp>/bark_audition`) and does not write to `output/otr/episodes`, thus requiring no sibling guard changes.
