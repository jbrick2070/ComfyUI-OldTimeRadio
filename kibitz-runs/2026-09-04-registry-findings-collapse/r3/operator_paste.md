OPERATOR'S MANUAL PASTE, r3 (lane not stated by the operator; pasted 2026-09-04 ~10:33 PDT with the r2 final as input, per the MANUAL_PROMPT_r3 brief). Verbatim.

VERDICT: yes-with-fixes. Owners/ratchets/spelling-only hold; batch globs, spawn-name case, and the stdlib-subprocess patch target will go red between commits.

MUST-FIX BEFORE BUILD:
1. [Build shape (d)] Batch (d) is `nodes/otr_*.py + OTR_*.py + two roots`. That misses unprefixed top-level offenders the pending set cannot drop: `nodes/video_engine.py:1082` `proc = subprocess.Popen(`; `nodes/production_ledger.py:222` `out = subprocess.check_output(`; env readers `nodes/scene_sequencer.py:368` `os.environ.get("OTR_MASTER_TARGET_LUFS"` and `nodes/cast_lock.py:499` `os.environ.get("OTR_BYPASS_FREEZE_HALT"`. Last batch then cannot empty `_PENDING_MIGRATION`. Fix: (d) = every remaining path under `nodes/` plus the two roots, not those globs.

2. [B + test_terminal_frame] `_SPAWN_CALLS = ("Popen", "run", "call", "check_call", "check_output")` at tests/test_terminal_frame.py:239. Plan `proc.popen` is lowercase; the sweep keys `func.attr in _SPAWN_CALLS` at :389-391. `proc.popen(...)` is not a spawn, so :495 `assert "encode_silent_mp4" in found.get("scope_draw.py"` goes red the moment scope_draw migrates. Same commit: add `"popen"` to `_SPAWN_CALLS`, and treat every import that binds the name `proc` (not only `from ._otr_shared import proc` / `from ._otr_shared.proc import run, popen`). Real forms: `from . import proc` inside `_otr_shared`, `from .._otr_shared import proc` in engines, `from _otr_shared import proc` flat. Today only `import subprocess` / `from subprocess import` at :358-365 count.

3. [B test patches] Do not patch stdlib `subprocess.run`. tests/test_credits_roll_spec.py:567-582 saves `cr.subprocess.run` then passthrough-calls it for non-scroll argv; tests/test_ffprobe_boundary.py:18 imports subprocess, :165 patches `ffp.subprocess.run`, :163 returns `subprocess.CompletedProcess`, :177 raises `subprocess.TimeoutExpired`. A process-wide stdlib patch intercepts those constructors/helpers. Patch `proc.run` / `proc.popen` (or `nodes._otr_shared.proc.subprocess.run` if lookup stays `subprocess.run` at call time). Re-exports MUST be identity aliases (`TimeoutExpired = subprocess.TimeoutExpired`), because those tests raise/construct the stdlib types. Cut two of the "twelve" from this arc: tests/test_soak_title_provenance.py:44 loads `scripts/otr_gpu_soak_matrix.py`; tests/test_w45_campaign_bank_pinning.py:33 loads `scripts/otr_w45_campaign.py`. scripts/ is out of scope; leave those patches on the script module.

4. [Build shape (a) vs (b)] One file, one batch. (a) currently edits engine-package files that (b) migrates again: `nodes/_otr_upscale_engines/eng_spandrel_esrgan.py:259` `Path(model_path).read_bytes()`; `nodes/_otr_video_engines/eng_ltx25.py:287` `ggml_module = __import__("sys").modules.get(base_cls.__module__)`; `nodes/_otr_video_engines/wan_shared.py:202-232` the six strings. Two boxes both push `v2.0-alpha`; 4060 owns 8 GB profile/engine text. Move those three into the (b) commit that already owns the file. (a) = `nodes/_otr_shared/**` only (gpu_residency.py lives there:77-97). Ratchet commit adds env.py/proc.py/guards and does not touch engine files.

5. [A/B consumer import + name collision] `nodes/_otr_shared/ffmpeg.py:28-34` is the dual/flat recipe (`from .ffprobe import` / `from _otr_shared.ffprobe import` / `from ffprobe import`). Every env/proc consumer must copy that shape; `from . import env` alone dies on flat load. Do not import the owner as `env` next to an `env` parameter: `nodes/_otr_shared/route_freeze.py:64-72` `def routing_env_snapshot(env=None):` / `src = os.environ if env is None else env`; tests/test_route_freeze.py:25 `rf.routing_env_snapshot({})`. Keep the injectable mapping; default path may `snapshot()`. Same shadow at `nodes/_otr_video_engines/motion_common.py:69` `environ = os.environ if env is None else env`. `from ._otr_shared.env import get, pin, snapshot` (or `import env as otr_env`).

6. [A vs existing ffmpeg ratchet] tests/test_ffmpeg_single_resolution.py:48-57 flags a List/Tuple whose first Constant is `"ffmpeg"` / `"ffmpeg.exe"` outside ffmpeg.py/ffprobe.py. A proc.py allowlist `("ffmpeg", "ffprobe", ...)` fails that test the commit proc.py lands. Use a dict name->reason (plan already wants reasons), not a tuple starting with ffmpeg. Run that test on env.py AND proc.py.

SHOULD-FIX:
7. [Local proxy] `grep -rl 'os\.environ\|getenv'` misses aliases: `__init__.py:510` `import os as _otr_ro` then :560 `_otr_ro.environ.get("OTR_ENABLE_HTTP_RENDER_ROUTES", "0")`; `nodes/_otr_freeze_cascade.py:911-912` `import os as _os_d3` / `_os_d3.environ.get("OTR_TEST_MODE")`. Empty pending sets on the AST guards are the proxy; drop or widen the grep. Walker must be generic on every `import os as X`, not hardcoded `_otr_ro`. Also `__init__.py:437` `import os as _otr_os`.

8. [A boot order] Plan cites `__init__.py:51` as the OTR_OUTPUT_DIR pin. :51 is `os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")`. Pin is :97-107 `if not os.environ.get("OTR_OUTPUT_DIR"):` / `os.environ["OTR_OUTPUT_DIR"] = _otr_out`. Import is already proven before the pin: :80 `from .nodes._otr_shared.hf_token import ensure_hf_token`. Last-batch `__init__.py` must import env before :51 (first write), not before :97. `nodes/__init__.py` is the one-line marker; `nodes/_otr_shared/__init__.py` has no top-level imports -- that pair is enough.

9. [Guards timing] Teach test_terminal_frame both subprocess and proc from the ratchet commit so (a) cannot forget the sweep. Ship the network five-file allowlist in that same commit (no pending; those files stay). Freeze the pending-set test as 5080-owned until empty; 4060 adds no new `os.environ` / `subprocess.run` sites in `nodes/`.

10. [test_master_mux] tests/test_master_mux_terminal_knob.py:94 `"environ" in ast.dump(fn) or fn.attr == "getenv"`. After mux `float(env.get(...))` that dump has no "environ"; if mux uses `from ._otr_shared.env import get` then `float(get(...))` is a Name, not an Attribute. Bind the predicate to the mux's real import; land it in the same commit as `nodes/otr_master_audio_mux.py`.

OPTIONAL:
- Plan "with proc.popen(...) as p": no `with subprocess.Popen` under nodes/ today. Returning the real Popen is enough (encode_sink.py:183 `self.proc = subprocess.Popen(`).
- `snapshot()` for blender is redundant: eng_mesh_stage.py:215 already `env = dict(base_env)` before :809 `build_blender_env(os.environ)`. Copy is behavior-equal; keep snapshot() for route_freeze default only if you want one mapping helper.

CUT THESE:
1. Patching stdlib `subprocess` as the cross-test strategy -- leaks into fixture ffmpeg/git/python -c; owner patch is the smaller fix.
2. Migrating soak/w45 subprocess patches -- subjects are scripts/, not this arc.
3. Importing env.py from prestartup_script.py -- no package context; five writes stay inline by decision.
4. urllib helper -- already withdrawn; network stays five named file exceptions.

VERIFY-AT-BUILD:
- Each batch: pending set shrinks by exactly the files in that commit (both directions); converted file left in the set fails.
- proc allowlist: every argv0_receipt.txt basename (ffmpeg, ffprobe, python*, git, nvidia-smi, blender) passes; unlisted raises the named error. [ASSUMPTION] pod `OTR_*_VENV` is `.../bin/python3` -- startswith("python") after .exe strip covers it; a wrapper whose basename is not python*/ffmpeg/... will refuse.
- After pending empty: load `workflows/otr_canonical.json`, log has `obs_publish OK`, file exists under `otr/obs/`. Green suite is not acceptance.
- Next published scan quotes ~9 info findings; pyproject.toml untouched.
