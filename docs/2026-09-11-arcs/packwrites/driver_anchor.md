# Driver anchor -- runtime-pack-writes: runtime writes inside the installed pack

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **DURABILITY**.

---

## 1. What is TRUE TODAY, grounded against current code

CONFIRMED CURRENT, file:line matches the row exactly. `_catalog_cache_path()` (nodes/_otr_openrouter_backend.py:787-793) resolves the OpenRouter catalog to `<repo>/models/openrouter_models.json` unless `OTR_OPENROUTER_CACHE_DIR` is set (line 791: `base = Path(override) if override else Path(__file__).resolve().parent.parent / "models"`, line 792 IS that line). `resolve_cache_root()` (cloud_media_backend.py:209-213) resolves Comfy-Cloud state to `<repo>/otr/cache/cloud_media` unless `OTR_CLOUD_MEDIA_CACHE_DIR` is set; `billing_ledger.jsonl` lives at `self.cache_root / "billing_ledger.jsonl"` (line 381).

Both roots are ALREADY fully untracked by git: `.gitignore:15` ignores all of `models/` (re-pinned explicitly at :208 "so the cache file is unmistakably non-versioned"), and `.gitignore:283` ignores `/otr/cache/` outright. Neither file has ever shipped in a git clone or in the Comfy Registry zip (.comfyignore layers only on top of git-tracked content). Effect of an in-place GIT update (ComfyUI-Manager's mechanism for a git-tracked install: `git pull`): untracked/ignored files are left alone -- neither cache is touched. Effect of a REGISTRY reinstall (rebuild from the versioned zip): since neither file was ever part of the zip, the rebuilt directory is simply in the same state as a brand-new install -- both caches ABSENT, not corrupted.

Neither code path treats "absent" as an error: `load_catalog_cache()` (:809-822) returns a well-formed empty catalog on a missing file; `_atomic_write_catalog` (:1017-1023) and `ledger_append`/`ledger_path` (cloud_media_backend.py:378-395) both lazily `mkdir(parents=True, exist_ok=True)`. A cold cache after any install path is a designed, safe state -- it never crashes, which already satisfies the operator's stated bar on its own.

The token-budget side effect the row names is real and precise: `resolve_context_window()` (:851-887) reads the cached model's `context_length`; on a cache miss it falls back to `row_default or DEFAULT_CONTEXT_WINDOW` (`DEFAULT_CONTEXT_WINDOW = 8192`, line 150) and logs a loud warning naming the consequence. The `load()` call site (:1150-1161) feeds that into `cache_entry["context_cap"]`, consumed at :1276 and :1297 by the writer's output-token-budget fitter. So a cold catalog silently (well, loudly-logged) clamps every OpenRouter model -- regardless of its real window -- to an assumed 8,192-token budget until `refresh_catalog_cache()` runs. And that refresh has exactly ONE caller in the whole tree: `scripts/otr_openrouter_refresh.py:44`. No node registration calls it. `scripts/*` is excluded from the registry bundle by `.comfyignore` with NO exception carved for this script (unlike the TTS-worker and Blender-stage scripts, which each get an explicit `!scripts/...` line) -- so a registry-installed pack has zero shipped way to ever warm this cache; it is 8192-clamped forever until an operator hand-invokes the backend module.

The billing ledger is write-only in this codebase: `ledger_append` is called from exactly two sites (`cloud_media_invoke.py:768,817`), both writers; nothing reads `billing_ledger.jsonl` back. Runtime budget enforcement (`CloudMediaSession.reserve()`/`_spent_usd`, cloud_media_backend.py:282,294-307) is a purely in-memory per-session counter, never seeded from the file. So losing the ledger file has ZERO effect on spend-gating correctness -- the only real loss is the operator's own historical audit trail, unrecoverable once gone. That validates the row calling it "the ONLY copy."

`nodes/_otr_paths.py:471-476` already defines `otr_shared_cache_dir()` -> `<ComfyUI output>/otr/episodes/_shared/cache/`, entirely OUTSIDE the pack directory (survives any custom-node reinstall) with one existing precedent user of exactly this shape: `_otr_public_domain_sources.py:769-772`'s `source_banks` cache. It is not swept by the janitor (zero hits in `_otr_janitor.py`). Its own docstring states the tier's contract: "A cache entry is NEVER the only copy." That is precisely the contract the ledger would violate if simply dropped there -- the GO_FORWARD_PLAN row's "should NOT land in a tier whose contract says never the only copy" is a correct, code-grounded read of this exact docstring, not editorializing. No sibling "durable, non-cache, non-sweepable, outside-the-pack" tier currently exists in `_otr_paths.py` for the ledger to move to.

**Key files:** `nodes/_otr_openrouter_backend.py:150 (DEFAULT_CONTEXT_WINDOW), 787-793 (_catalog_cache_path), 809-822 (load_catalog_cache), 851-887 (resolve_context_window), 1017-1023 (_atomic_write_catalog), 1032 (refresh_catalog_cache), 1150-1161 (load() call site feeding context_cap)`, `nodes/_otr_shared/cloud_media_backend.py:209-213 (resolve_cache_root), 279,282,294-307 (in-memory spend state, never ledger-seeded), 378-395 (ledger_path/ledger_append)`, `nodes/_otr_paths.py:471-476 (otr_shared_cache_dir, 'never the only copy' contract docstring)`, `nodes/_otr_public_domain_sources.py:769-772 (existing precedent user of otr_shared_cache_dir)`, `scripts/otr_openrouter_refresh.py:44 (sole caller of refresh_catalog_cache; not shipped in the registry bundle)`, `.gitignore:15,208,283 (both paths already fully untracked)`, `.comfyignore (scripts/* exclusion with no carve-out for otr_openrouter_refresh.py)`, `docs/GO_FORWARD_PLAN.md:92 (live row) and docs/GO_FORWARD_ARCHIVE.md:4832-4835,8771 (prior framing)`

**Blast radius:** Both modules are shared infrastructure loaded on every machine that exercises the OpenRouter writer slot or the Comfy-Cloud image/video/TTS lanes: the 5080 and 4060 dev boxes, RunPod, Mac, and any third-party git-clone or registry install. It's pure stdlib Path resolution (no CUDA/dtype/VRAM-tier branch), so it carries none of the numeric-parity risk CLAUDE.md section 0B flags for tier-sensitive shared code like `_plan_max_memory` -- but it DOES change cold-start behavior (and therefore the OpenRouter writer's assumed context/output-token budget, per the mechanism above) for every machine on its first run after the change, until each machine's catalog is repopulated at the new location.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- GO_FORWARD_PLAN.md:92 and GO_FORWARD_ARCHIVE.md:4833/8771 say "a registry update or reinstall wipes both" as if the two operations are equivalent. Only 'reinstall' (rm + fresh zip extract) is strongly grounded: both files are 100% untracked (.gitignore:15,208,283) and never bundled (.comfyignore's own header: 'untracked files are already excluded by the publisher'), so a from-scratch rebuild leaves the pack exactly as a new install -- both caches absent. Whether ComfyUI-Manager/comfy-cli's in-place 'update' on an existing registry install preserves or deletes pre-existing untracked files is implemented in that external tool, not this repo -- UNVERIFIED here. A git-tracked install's 'update' (git pull in place) provably does NOT touch either file: git pull never deletes untracked/ignored paths.
- Neither doc row mentions that both cache roots already have a working env-var override today with zero code change -- OTR_OPENROUTER_CACHE_DIR (nodes/_otr_openrouter_backend.py:791) and OTR_CLOUD_MEDIA_CACHE_DIR (nodes/_otr_shared/cloud_media_backend.py:210) -- a real, currently-available interim mitigation for an operator who wants either file outside the pack directory before any migration lands.

## 3. The fork -- this is what the round must pressure-test

Two files need DIFFERENT target tiers, plus a migration decision for data that already exists on disk today -- not a one-line path swap. (1) Catalog cache (openrouter_models.json): leave in-pack and accept the already-safe, loudly-logged cold-start (status quo, zero code change) vs. move to `otr_shared_cache_dir()` (matches the source_banks precedent, survives reinstall, unswept) vs. move under the `C:\ComfyUI-Models`-style external root (matches the GGUF-weights precedent, CLAUDE.md section 6A). Whichever is chosen, decide whether to copy an operator's existing warm cache forward on first run after the code change, or accept one more cold start (self-heals via refresh_catalog_cache, never crashes) -- and separately whether the registry-bundle gap (refresh script unshipped) gets closed in the same change (comfyignore exception, like the worker scripts) or left as a known limitation. (2) Billing ledger: cannot use `otr_shared_cache_dir()` (proven contract violation above) -- needs either a NEW non-cache "durable records" tier added to `_otr_paths.py` (sibling to cache/tmp under otr_shared_root(), explicitly documented as never-swept/never-disposable) or a different external root entirely, plus a decision on whether/how to carry forward any ledger history an operator already has at the old in-pack path so the migration itself doesn't read as data loss.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
