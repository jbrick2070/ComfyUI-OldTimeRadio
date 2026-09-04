# Driver anchor, r3 -- registry findings collapse (Claude / Fable 5.1, sole judge)

Written BEFORE the r3 fan-out, against the r2 `final.md` and the real tree.
Round focus: wiring, integration, sequencing.

VERDICT: yes-with-fixes. The owners wire cleanly (both are stdlib leaves and
the package marker is empty), but the plan's ORDER is wrong in one place --
the twelve subprocess-patching tests are listed as per-batch work when they
can and should be rewritten once, first -- and the boot-order question it
raises for the root file is answered by the tree, not open.

MUST-FIX BEFORE BUILD (driver's own):
1. [Sequencing] Rewrite the twelve subprocess-patching test files ONCE, in
   batch (a), to patch the REAL module: `import subprocess;
   monkeypatch.setattr(subprocess, "run", spy)` (and the two `mock.patch`
   forms likewise). Because `proc.run` looks `subprocess.run` up on the
   module at call time, a test patched that way is correct BEFORE its module
   migrates (the module still calls `subprocess.run` -> the spy) and AFTER
   (the owner calls `subprocess.run` -> the spy). That decouples the test
   rewrite from the batch order entirely; the r2 final's "per batch" wording
   is replaced.
2. [Wiring] Every migrated module imports the owners in the pack's existing
   two-form shape -- `try: from ._otr_shared import env, proc /
   except ImportError: from _otr_shared import env, proc` -- because tests
   import many modules FLAT (`tests/test_obs_published_filename.py` and the
   ffprobe boundary tests do), and `otr_master_audio_mux.py:42-45` /
   `:51-54` are the precedent. A packaged-only import breaks the flat suite
   on the first batch.
3. [Boot order, answered] The root `__init__.py` CAN import the env owner
   before its first write: `nodes/__init__.py` is a one-line package marker,
   `nodes/_otr_shared/__init__.py` has no top-level imports, and the root
   file already imports `.nodes._otr_shared.hf_token` at `:80` ahead of the
   node loader. So its two writes (`:51` setdefault, `:107` pin) migrate in
   batch (d) with `from .nodes._otr_shared import env` placed above line 51.
   `prestartup_script.py` runs as a bare module with no package context and
   keeps its writes -- that is the ONLY root exception, by decision.
4. [Wiring] The ffmpeg owner's `env.get(FFMPEG_ENV)` keeps the constant
   `OTR_FFMPEG` inside `ffmpeg.py` and `which()` inside the two tool owners,
   so `tests/test_ffmpeg_single_resolution.py` stays green by construction;
   the env owner must never spell the constant. Run the test in batch (a),
   do not reason about it.
5. [Sequencing] The ratchet SETS are edited in the SAME commit as the batch
   they describe, and the FULL suite (about seven minutes on this box) runs
   per commit -- not only the guards. Nine commits is about an hour of suite
   time; that is the price of "green per commit" and it is paid.
6. [Integration] The final proof is a canonical leg (`workflows/otr_canonical.json`,
   one act, this box) that PUBLISHES to `otr/obs/` after batch (d) -- the
   operator's success signal. A green suite is not the receipt; the mp4 in
   the watched folder is. Then, and only then, the operator bumps
   `pyproject.toml` for the scan that is the registry receipt.

SHOULD-FIX:
1. [Integration] The allowlist on Linux: `ffmpeg` (no `.exe`), `python3.12`
   (the pod's interpreter), `nvidia-smi`, `blender` all pass the basename
   rule; the unit test should include a Linux-shaped absolute path for each.
2. [Sequencing] Batch (a) also carries the three singletons and the six
   strings; the D change (`gpu_residency.py`) needs its Windows-without-psutil
   test to monkeypatch the import, not uninstall anything.
3. [Peers] Both boxes push. `ListAgents` before each batch and message any
   live peer window the batch's file list; `nodes/` and `tests/` are this
   box's surface, but a peer editing `docs/PROD_BUG_LOG.md` at the same time
   is normal and `merge=union` handles it.

CHECKED-CLEAN by the driver: no workflow JSON, widget, or node signature
changes anywhere in this plan (nothing for the widget-drift or link-index
gates to catch); `.comfyignore` unaffected; `scripts/` untouched;
`psutil` 7.2.2 importable in the venv (the core requirements file for the
booted install is `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\requirements.txt`
-- see the grep receipt in the r3 judgment).

UNVERIFIABLE until built: the residual count; whether the `.pth` chunked
read clears the file rule; whether the scanner sees `proc.py`'s two sites as
two findings or more if it also matches the re-exported names.
