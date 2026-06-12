<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document is still a decision prompt, not a build-ready plan; it contains at least one code-grounding contradiction and lacks the exact path changes/gate updates needed to avoid breaking current writers.

MUST-FIX BEFORE BUILD:
1. [Q1 / Hard constraints] Pick Option B, not A or C. Option A (`otr/.system/...`) violates the operator’s explicit “episodes/ + obs/ ONLY” top-level contract; on Windows a dot-dir is not reliably hidden from Explorer. Option C kills the cross-episode stills cache despite the stated hard constraint that cache reuse is valuable. Concrete fix: make the sanctioned layout:
   - `otr/episodes/`
   - `otr/obs/`
   - `otr/episodes/_shared/cache/`
   - `otr/episodes/_shared/tmp/`
   - `otr/episodes/_shared/state/`
   and define `_shared` as reserved non-episode infrastructure.

2. [Live inventory: stills / Grounding: `_otr_paths.py`] The attribution says `otr_save_to_episode_workspace` falls back to `otr_stills_dir("") = top level`, but the shown code returns `<output>/otr/_legacy_stills`, not `<output>/otr/stills`. Concrete fix: correct the inventory/root-cause statement before coding, and verify the actual writer responsible for live `otr/stills`; do not migrate based on the stated fallback until that writer is identified. verify: search for writers to `otr/stills`, `"stills"`, `otr_stills_dir("")`, and any cache materialization path.

3. [Q2 / Grounding: `_otr_paths.py`] Existing helpers already violate the proposed contract:
   - `otr_audio_dir("")` returns `otr/_legacy_audio`
   - `otr_stills_dir("")` returns `otr/_legacy_stills`
   - `otr_portraits_dir("")` returns `otr/_legacy_portraits`
   - `otr_state_dir()` returns `otr/state`
   Concrete fix: in the same ticket that adds the guard, change these to either:
   - require a non-empty `episode_id` and raise loudly for production write helpers, or
   - route sanctioned shared tiers through new helpers such as `otr_shared_cache_dir()`, `otr_tmp_dir()`, `otr_state_dir()` returning under `otr/episodes/_shared/...`.
   Do not add the guard before changing these helpers or the build will fail immediately.

4. [Q2] “No helper may return a path outside the contract” is overbroad as written. `_otr_paths.py` also intentionally returns input/model/log/HF paths outside `otr`: `comfy_input_dir()`, `comfy_models_dir()`, `comfyui_log_path()`, `resolve_hf_model_path()`. Concrete fix: scope the assertion to OTR output helpers only, e.g. helpers whose returned path is under `comfy_output_dir() / "otr"`. Exempt non-output resolvers explicitly.

5. [Q2 / Q3 / Hard constraints] Moving tmp requires launch/environment changes before enforcement. The live inventory says `TEMP`, `TMP`, and `OTR_GPU_LEASE_DIR` currently point at `otr/tmp`; ffmpeg children and atomic-publish staging depend on that. Concrete fix: update the launcher/env setup first to point all three to `otr/episodes/_shared/tmp`, then update atomic-publish staging to use that helper, then enable the top-level guard.

6. [Hard constraints / Q2] The hygiene gate update is underspecified and can false-fail once cache/tmp/state move under `episodes`. The gate “counts files under server output”; putting cache/tmp/state under `episodes/_shared` will increase counts unless excluded. Concrete fix: update the gate in the same chunk to:
   - fail any top-level `otr/*` entry except `episodes` and `obs`;
   - ignore or separately report `otr/episodes/_shared/{cache,tmp,state}`;
   - keep OBS flatness checks on `otr/obs` only;
   - prevent treating `_shared` as an episode.

7. [Q3 / Q4] Migration during active renders is unsafe and the document does not define a quiescence condition. The live inventory says `stills` and `tmp` were written “TODAY (mid-render)”. Concrete fix: land OH-1 only after the queued current-tree renders drain, or explicitly stop ComfyUI/render workers before migration. Add a preflight that fails if active render/ffmpeg/python children are detected or if files under old `otr/tmp` changed within the stale threshold.

8. [Q3] “Archive-then-delete” must not create another illegal top-level directory under `otr`. Concrete fix: put the operator-approved archive outside `otr`, e.g. `<output>/otr_archive_YYYYMMDD_HHMMSS/` or an operator-specified external path, then delete approved debris from `otr` only after verification.

SHOULD-FIX:
1. [Q2] The audit requirement “grep/AST sweep for hardcoded `otr/<dir>` strings” is too narrow. It will miss `Path("otr") / "tmp"`, `os.path.join("otr", "tmp")`, backslash paths, f-strings, and variables. Concrete fix: add CI checks for:
   - `comfy_output_dir() / "otr"` outside `_otr_paths.py`;
   - string/path joins containing `otr`, `episodes`, `obs`, `tmp`, `state`, `stills`, `audio`, `videos`;
   - Windows backslash variants;
   - allowlist `_otr_paths.py`, tests, and migration tooling.

2. [Q2] Runtime path guard needs normalization details. Concrete fix: compare `path.resolve(strict=False)` against resolved allowed roots and require `relative_to()` either `otr/episodes` or `otr/obs`. Also reject `..` traversal and symlink escapes where possible.

3. [Q1] Define `_shared` as reserved so episode discovery does not treat it as a renderable episode. Concrete fix: any code walking `otr/episodes/*` must skip entries starting with `_`, especially `_shared`. verify: ledger auto-pick walkers currently scanning `otr_episodes_root()`.

4. [Q3] The janitor needs safety rules. Concrete fix: stale-tmp sweep should only delete under `otr/episodes/_shared/tmp`, only files older than an operator-set threshold, skip locked/open files, log every deletion, and never recurse outside the resolved tmp root.

5. [Q3] Cache migration needs collision policy. Concrete fix: when moving content-addressed stills from old cache locations into `episodes/_shared/cache`, dedupe by hash/name, verify identical size/hash before overwriting, and quarantine conflicts for operator review.

6. [Q4] The recommendation should be encoded as sequencing, not prose. Concrete fix:
   - OH-1 lands after current queue drains.
   - No 7-leg sweep / WAN batch / 0-E Phase B starts until OH-1 guard passes against the migrated tree.
   - If an emergency interruption is chosen, stop all writers first and run migration preflight.

OPTIONAL / NICE-TO-HAVE:
- Add a one-shot `otr_tree_doctor --dry-run` that prints current illegal top-level entries, proposed moves, estimated bytes, and exact archive destination.
- Add a README fragment under `otr/episodes/_shared/README.txt` explaining that `_shared` is system-owned and not an episode.
- Add metrics for cache hits/misses before and after migration so the stills-cache benefit remains visible.

CUT THESE (over-engineering):
1. [Q1 Option A] Cut `.system`. It violates the strict top-level operator view and adds an exception the hygiene gate must remember forever.
2. [Q1 Option C] Cut per-episode-only cache for this ticket. It solves top-level cleanliness by destroying cross-episode reuse, which the hard constraints explicitly value.
3. [Q2] Cut any broad “assert every `_otr_paths.py` return is under episodes/obs” implementation. It will incorrectly catch model/input/log/HF helpers that are not OTR output asset paths.
4. [Q3] Cut automatic deletion of dead debris. The document already says operator approval is required; implement dry-run/archive list first, deletion only after approval.