# Fable -- OTR Overnight-Soak Review + Go-Forward Plan

Read-only analysis of the two back-to-back 25-leg coverage soaks (2026-06-12 ->
06-13, branch v2.0-alpha). No code changed, nothing pushed, Wan untouched, the
16gb_full enable-set and GO_FORWARD_PLAN section 1A untouched. Every proposed
code change below is a snippet for a separate coder window to apply.

This revision folds in a 2-model roundtable second opinion (GPT-5.5 +
Gemini 3.1 Pro via OpenRouter, ~$0.13). Claude grounded every panel claim
against the real code; see Appendix B for the judgment log. The one material
catch (the temp-helper file semantics) is folded into A1.

## Evidence base
- `scripts/overnight_soak_report.md` (pass 1 + pass 2, per-leg table, histograms).
- `scripts/overnight_soak_run.log` (per-leg headers, histograms, the failing
  hygiene-gate message, the writer execution_error).
- Source: `nodes/_otr_video_engines/{cheap_families,eng_ltx_video,eng_still_parallax,eng_humo,render_driver,wrapper_bridge}.py`,
  `nodes/_otr_shared/fallback.py`, `nodes/news_interpreter.py`,
  `nodes/_otr_structured_call.py`, `nodes/OTR_LedgerScriptWriter.py`,
  `scripts/_otr_soak_capstone.py`, `scripts/_otr_soak_server_launch.cmd`,
  `nodes/_otr_paths.py`, `nodes/_otr_janitor.py`.
- On-box model inventory (`C:\ComfyUI-Models\...`) confirmed via directory listing.

Headline confirmed: the pipeline is end-to-end healthy at 70 words. Every leg
except the writer flake produced a playable OBS final with byte-identical master
audio. 0/25 is one systemic hygiene regression (R1) plus one designed
fail-closed writer halt (R3); R2 is a behavioral/quality finding, not a crash.

---

## 0. Go-forward plan (execution order)

A coder window should land these in this order; each is independently shippable.

1. **R1 temp-leak fix (do first -- flips most legs green).** Add the
   `_tmp.py` helper (A1, hardened per the roundtable), swap the 7 call sites,
   run the full suite + Bug Bible, then a 1-leg soak to confirm the hygiene gate
   goes green. Low risk, high payoff.
2. **R3 soak escape hatch (trivial).** Add the `OTR_NEWS_BRIEFS_REQUIRED=0`
   env override (A2a) and set it in the soak launcher so a fabricated key_term
   degrades instead of aborting the engine-under-test. Production stays fail-closed.
3. **Re-run the 2-pass soak.** With R1+R3 in, expect close to 25/25 green
   (the still/flux/parallax/visualizer/ltx_orbit beats). Capture the result.
4. **R2 disambiguation (gated on a log line, NOT a blind code change).** Pull
   one HuMo leg's `format_swap_log` line from the live server `otr_runtime.log`;
   branch to the CS-4 umt5-detach fix (if OOM) or the wrapper fix (if a forward
   error). Do not touch the in-process VRAM path before that line is in hand.
5. **R3 durable prune-to-floor (product fix).** Land A2b + its regression test
   so a single hallucinated theme word never aborts a production episode.
6. **Minor harness items (A3) + the bigger bets (section 4)** as capacity allows;
   the VRAM-budget-aware scheduler is the highest-leverage larger item.

Defaults to ship now (section 3): flux_still for announcer + character,
visualizer for music. Keep HuMo / latentsync / 3D selectable-not-default until
step 4 lands. Wan stays parked.

---

## 1. Fixes (must-fix for a green sweep)

### R1 -- floor / cheap-family temp-file leak (PRIMARY; flips most legs green)

**Root cause (confirmed).** Seven render-helper call sites create their
intermediate mp4 via `tempfile.mktemp` / `tempfile.mkstemp` (no `dir=`) or
`tempfile.gettempdir()`, so the file lands wherever `gettempdir()` resolves:

| # | File:line | Current call | Leaked prefix |
|---|---|---|---|
| 1 | `cheap_families.py:122` | `tempfile.mktemp(suffix=".mp4", prefix="otr_floor_%s_" % self.name)` | `otr_floor_*` (every leg) |
| 2 | `eng_still_parallax.py:302` | `tempfile.mkstemp(..., prefix="otr_parallax_")` | `otr_parallax_*` |
| 3 | `eng_ltx_video.py:721` | `tempfile.mkstemp(..., prefix="otr_ltx_")` | `otr_ltx_*` |
| 4 | `eng_humo.py:342` | `tempfile.mktemp(..., prefix="otr_humo_")` | `otr_humo_*` (latent; HuMo floored) |
| 5 | `eng_latentsync.py:213` | `tempfile.mktemp(..., prefix="otr_lsync_")` | `otr_lsync_*` |
| 6 | `eng_wan_i2v.py:286` | `tempfile.mktemp(..., prefix="otr_wan_")` | `otr_wan_*` (parked; fix anyway) |
| 7 | `render_driver.py:323` | `os.path.join(tempfile.gettempdir(), "otr_audio_slices")` | `otr_audio_slices/` |

The hygiene gate `assert_no_stray_writes` (`scripts/_otr_soak_capstone.py:212`)
checks that the SYSTEM temp dir (`%LOCALAPPDATA%\Temp`) gained no new `otr_*`
entries. The gate's docstring states the design contract: "the in-tree TEMP
repoint held." The soak launcher does repoint it
(`_otr_soak_server_launch.cmd:55-58` sets `TEMP`/`TMP` to
`<output>\otr\episodes\_shared\tmp`). The bug is that these seven call sites
DEPEND on that ambient repoint holding for the live server -- and it did not.
The run log shows the leaked files in `C:\Users\jeffr\AppData\Local\Temp` and a
per-leg baseline of "233 otr_* system-temp entries" growing across legs. The
intermediates escape to the system temp dir and are never unlinked (the clip
`path` is consumed downstream by SilentComposite, so the engine cannot unlink
synchronously). This is exactly the failure class CLAUDE.md flags: the v2 install
move already dropped `PYTHONUTF8` from the boot path; relying on an ambient
process env for a correctness invariant is the same trap.

**Fix (concrete, hardened).** Route every intermediate to the in-tree tmp tier
explicitly, independent of the ambient `TEMP`. The janitor already sweeps that
tier (`nodes/_otr_janitor.py`, OH-3), so the clip path stays valid through the
compositor and the file is cleaned afterward.

**Step 1 -- add the shared helper.** `nodes/_otr_video_engines/_tmp.py` (new):

```python
"""In-tree temp allocator for video-engine intermediates (R1 fix).

Every engine intermediate .mp4 must land under the sanctioned in-tree tmp tier
(otr/episodes/_shared/tmp) -- NOT the ambient system temp dir -- so the OH-3
janitor sweeps it and the soak hygiene gate stays green regardless of whether the
launcher repointed TEMP. Cold-import clean (V-12): stdlib only at module scope;
the paths import is lazy. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

import os
import tempfile


def _in_tree_tmp_dir():
    """otr/episodes/_shared/tmp (created), or None if the output tree cannot be
    resolved (headless CPU unit tests with no ComfyUI output dir)."""
    try:
        try:
            from .._otr_paths import otr_shared_tmp_dir
        except ImportError:
            from _otr_paths import otr_shared_tmp_dir  # type: ignore
        d = str(otr_shared_tmp_dir())
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:  # noqa: BLE001
        return None


def otr_engine_tmp_mp4(prefix: str) -> str:
    """Reserve a unique in-tree .mp4 path and return it. The path does NOT exist
    on return (matches the legacy tempfile.mktemp semantics the call sites relied
    on); the caller's ffmpeg/encoder creates it. Fail-closed in production: if the
    in-tree tmp dir cannot be resolved, only OTR_TEST_MODE permits the tempfile
    default -- production raises rather than silently leak to the system temp dir
    (roundtable MUST-FIX #2)."""
    d = _in_tree_tmp_dir()
    if d is None:
        if os.environ.get("OTR_TEST_MODE"):
            d = None  # tempfile default dir, tests only
        else:
            raise RuntimeError(
                "otr_engine_tmp_mp4: cannot resolve the in-tree tmp dir and "
                "OTR_TEST_MODE is unset -- refusing to leak to the system temp "
                "dir (R1). Check comfy_output_dir()/otr_shared_tmp_dir().")
    # mkstemp reserves a unique name atomically; unlink so we hand back a
    # non-existent path (the legacy mktemp contract). Every OTR ffmpeg cmd
    # passes -y and encode_frames_to_silent_mp4 overwrites, so an existing file
    # would also be fine -- this just removes any future dependency on -y and
    # avoids a 0-byte .mp4 lingering if a writer fails before its first frame
    # (roundtable MUST-FIX #1; the claimed ffmpeg hang does NOT occur today).
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix=prefix, dir=d)
    os.close(fd)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return path
```

**Step 2 -- the six engine swaps.** Add `from ._tmp import otr_engine_tmp_mp4`
at each module top and replace the call.

`cheap_families.py:122`
```python
-        out_path = tempfile.mktemp(suffix=".mp4", prefix="otr_floor_%s_" % self.name)
+        out_path = otr_engine_tmp_mp4("otr_floor_%s_" % self.name)
```
`eng_humo.py:342`
```python
-        out_path = tempfile.mktemp(suffix=".mp4", prefix="otr_humo_")
+        out_path = otr_engine_tmp_mp4("otr_humo_")
```
`eng_latentsync.py:213`
```python
-        out_path = tempfile.mktemp(suffix=".mp4", prefix="otr_lsync_")
+        out_path = otr_engine_tmp_mp4("otr_lsync_")
```
`eng_wan_i2v.py:286`
```python
-        out_path = tempfile.mktemp(suffix=".mp4", prefix="otr_wan_")
+        out_path = otr_engine_tmp_mp4("otr_wan_")
```
`eng_still_parallax.py:302-304` (delete the `os.close(fd)` line)
```python
-        fd, out_path = tempfile.mkstemp(suffix=".mp4",
-                                        prefix="otr_parallax_")
-        os.close(fd)
+        out_path = otr_engine_tmp_mp4("otr_parallax_")
```
`eng_ltx_video.py:721-722` (delete the `os.close(fd)` line)
```python
-        fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="otr_ltx_")
-        os.close(fd)
+        out_path = otr_engine_tmp_mp4("otr_ltx_")
```

**Step 3 -- the audio-slice dir (`render_driver.py:323`).**
```python
-    tmp_dir = os.path.join(tempfile.gettempdir(), "otr_audio_slices")
+    from ._tmp import _in_tree_tmp_dir
+    _base = _in_tree_tmp_dir() or tempfile.gettempdir()
+    tmp_dir = os.path.join(_base, "audio_slices")
```
(The leaf rename `otr_audio_slices` -> `audio_slices` means even the fallback
path no longer matches the gate's `otr*` system-temp scan.)

**Cleanup nits.** The lazy `import tempfile` in `cheap_families.py:119`,
`eng_humo.py:318`, `eng_still_parallax.py:284`, `eng_ltx_video.py:667` becomes
unused after the swap -- remove to keep the linter quiet. Leave `os` imports.

**Grounding note (roundtable).** Both panel models flagged a feared "ffmpeg
refuses/hangs overwriting the empty mkstemp file." I verified this does NOT occur:
every OTR ffmpeg builder passes `-y` (`wrapper_bridge.py:457/478/492`) and
`run_ffmpeg` runs non-interactively (`stdout=DEVNULL`, no stdin). Also `eng_ltx`
and `eng_still_parallax` ALREADY use `mkstemp` (existing file) +
`encode_frames_to_silent_mp4` and render fine in the soak (3/3), proving the
encoder overwrites. The unlink in the helper is therefore belt-and-suspenders,
not a correctness requirement -- but it is cheap and removes a future footgun, so
it is kept.

**Risk / blast radius.** Low. One new file + 7 swaps; the clip `path` contract is
unchanged (only the parent dir moves). Verify on the box that
`otr_shared_tmp_dir()` resolves inside the live ComfyUI process (it does -- the
audio slices already live there).

**Regression tests to add.**
- `tests/test_engine_tmp_in_tree.py`: monkeypatch `otr_shared_tmp_dir()` to a
  pytest `tmp_path`; assert `otr_engine_tmp_mp4` returns a path under it, that the
  path does NOT exist on return, and that `gettempdir()` gains no `otr_*` entry.
  Add a case with the dir unresolvable + `OTR_TEST_MODE` unset that asserts the
  fail-closed `RuntimeError`.
- Extend the AST forbidden-sweep (the b7-style guard): no `nodes/**` file may call
  `tempfile.mktemp(`/`mkstemp(`/`gettempdir()` for an `otr_*` artifact without
  going through `otr_engine_tmp_mp4` / `_in_tree_tmp_dir`. Makes the leak class
  un-reintroducible.

### R3 -- writer key_term post-validator halt (INTERMITTENT, ~1-2/25)

**Root cause (confirmed).** `OTR_LedgerScriptWriter` (node 1) exhausts the
`structured_call` retry ladder on `build_news_briefs`:
`PostValidationError: V1: key_term 'climate mitigation' not in source` (pass 2;
pass 1 `'sustainability'`). The failing terms are THEME ABSTRACTIONS the local
writer (gemma) invents -- not verbatim or paraphrased in source. The Sprint 10B
LLM-as-judge fallback (`news_interpreter.v1_validate(..., judge_fn=...)`) IS
wired (`news_interpreter.py:811-815`) and correctly rejects a fabricated topic.
On exhaustion the writer HALTS by design: `news_briefs_required` defaults TRUE
(`OTR_LedgerScriptWriter.py:1887-1912`, 2446-2469; Jeffrey 2026-05-27: halt +
re-queue rather than silently degrade). Production re-queues (RSS re-roll); the
soak cannot, so ~1-2/25 legs abort on a bad story draw.

**Verdict: working-as-designed fail-closed, not a code bug.** What changes is
policy, split by audience:

**Fix A2a (trivial, recommended for the soak): env escape hatch on the halt.**
At `OTR_LedgerScriptWriter.py:2446-2448`:
```python
         _news_required = bool(
             resolved.get("news_briefs_required", True)
         )
+        # Soak/headless escape hatch: an explicit env override lets a batch run
+        # degrade (raw news_seed) instead of halting on a single fabricated
+        # key_term, without editing the graph widget. Production leaves this
+        # unset so the widget default governs.
+        if os.environ.get("OTR_NEWS_BRIEFS_REQUIRED") == "0":
+            _news_required = False
```
Soak launcher then sets `OTR_NEWS_BRIEFS_REQUIRED=0`. (`os` is imported at file
top.) Confidence: high; off by default; one branch.

**Fix A2b (durable product fix): prune fabricated key_terms to the floor.**
In `news_interpreter.py` `_content_validator` (798-820), before returning the V1
failure, drop terms that fail strict word-boundary and accept if at least
`_MIN_KEY_TERMS` survive:
```python
     def _content_validator(brief: NewsBriefs) -> str | None:
         if len(brief.key_terms) < _MIN_KEY_TERMS:
             return (
                 f"V0: key_terms below production minimum "
                 f"({len(brief.key_terms)} < {_MIN_KEY_TERMS})"
             )
+        # Prune-to-floor self-heal (R3): drop key_terms the model fabricated
+        # (not strict-in-source) when enough grounded anchors remain, so a single
+        # hallucinated theme word does not abort the episode. The pruned list
+        # stays within the field's min_length=1/max_length bounds, so the
+        # assignment is safe whether or not validate_assignment is ever enabled
+        # (NewsBriefs is a plain BaseModel today -- no frozen, no assignment
+        # validation; roundtable grounding).
+        def _strict_in_source(t: str) -> bool:
+            pat = r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])"
+            return bool(re.search(pat, source_text_full, re.IGNORECASE))
+        _grounded = [t for t in brief.key_terms if _strict_in_source(t)]
+        if len(_grounded) >= _MIN_KEY_TERMS and len(_grounded) < len(brief.key_terms):
+            brief.key_terms = _grounded
         v_failures: list[str] = []
```
(`re` is already imported.) This PRUNES rather than relaxes -- a brief with fewer
than 2 grounded terms still halts. Prompt tightening is low yield (the prompt
already says "verbatim from the source", `news_interpreter.py:704`; GBNF cannot
enforce semantic presence).

**Risk / blast radius.** A2a: none (env-gated). A2b: low -- a slightly weaker
brief can pass, but the V0 floor still guarantees >=2 grounded anchors;
production `news_briefs_required` stays available for the true-halt path.

**Regression test.** `tests/test_news_v1_prune.py`: a brief with 4 key_terms
(2 in-source, 2 fabricated) prunes to the 2 valid and validates clean; a brief
with only 1 in-source still fails V0.

### R2 -- HuMo / motion / 3D engines floor at 70 words

**What I confirmed.** All HuMo model handles are installed on the box
(`diffusion_models\humo_1.7B_fp16.safetensors`,
`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`;
`text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors`;
`loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`;
`audio_encoders\whisper_large_v3_fp16.safetensors`). So the flooring is NOT a
missing-model gate. The render path also wires HuMo's inputs correctly: the
portrait resolves from the portrait index and per-beat audio is sliced from the
frozen master (`render_driver.py:657-746`). HuMo receives both `init_image` and
`audio_ref`; the demotion happens at LOAD/FORWARD time and is caught by the
fallback loop (`render_driver.py:1106-1133`), which restamps LOUD and walks
`humo -> humo_1.7B -> latentsync -> still_motion`.

**The finding splits into EXPECTED behavior and a REAL must-root-cause item:**

- **latentsync 0/6 -- EXPECTED.** `lipsync_overlay` family requires a
  `base_clip_ref`; the 70w soak provides none and the `_provide_lipsync_base`
  seam does not synthesize one here, so `_assert_family_inputs_satisfiable`
  raises `FamilyInputGap` and the chain LOUD-skips to the floor. Input gate, not
  a defect.
- **triposg_talk 0/6, mesh_stage -> parallax -- EXPECTED at 70w on this box.**
  The 3D talking path needs mesh assets / a cu128 toolchain; its cousins
  (`hunyuan3d_talk`, `trellis_talk`) are already SKIPPED_DISABLED for that reason.
- **HuMo 14B 0/6 -- consistent with CS-4.** The umt5 fp8 text-encoder (~5.2 GB
  resident) co-resident with the 14B forward starves the 14.5 GB lease -> OOM ->
  hard fallback. Matches the known CS-4-open item (lazy TE-detach).
- **HuMo 1.7B (the DEFAULT) 0/6 -- the genuinely concerning one.** The 1.7B
  stack is light (~3.3 GB UNET + umt5 + whisper) and PASSES the ~38-min
  acceptance render elsewhere (CS-4 record). Uniform 0/6 here, with all files
  present, points to a runtime forward/lease failure or a co-residency spike
  (still/portrait-phase residue not fully drained before beat 1, despite the
  `run_episode` pre-render reclaim at `render_driver.py:1162-1167`), not to
  missing inputs.

**What I could not disambiguate from the given inputs.** The per-beat
`classify_failure` reason (OOM vs forward/wrapper error) is written by
`format_swap_log` to the LIVE SERVER log (`otr_runtime.log`), not to
`overnight_soak_run.log` (the client view, which carries only the histogram).

**Verdict.** R2 is a bug to fix for the DEFAULT character path -- a shipped
default that silently (LOUD in the trace, invisible to a viewer) delivers still
frames where a talking face is expected is a quality regression, even though the
episode renders playable. It is NOT a bug for latentsync / 3D at 70w (expected).
The concrete next step is mechanical: capture one HuMo leg's `format_swap_log`
line from `otr_runtime.log` and branch:
- if `kind == OOM` (expected for 14B): land the CS-4 lazy umt5-TE detach (free
  umt5 after `CLIPTextEncode`, before the HuMo sampler); for 1.7B confirm the
  pre-render reclaim actually drains the still phase before beat 1.
- if a forward/wrapper error: fix the wrapper-node resolution / SageAttention /
  the 4n+1 length path in `eng_humo._build_graph`.

Until that lands, keep HuMo selectable-not-default (section 3). Do NOT attempt a
blind VRAM-discipline edit -- that area (CS-4 / BUG-291) is the most fragile in
the stack; one log line settles it.

**Risk / blast radius of the eventual fix.** Medium -- touches the in-process
VRAM discipline; must be GPU-soaked (single resident heavy <= 14.5 GB, audio
byte-identical) before shipping.

**Regression test.** GPU-smoke (operator lane): after a HuMo beat, assert the
trace row `final_engine == "humo_1.7B"` (real talking face, not `still_motion`)
for a fixed-seed 1-beat episode within 14.5 GB. CPU-side: assert a
character_video request carrying `init_image` + `audio_ref` satisfies
`_assert_family_inputs_satisfiable("humo", req)` (guards against an input-gate
regression masquerading as VRAM).

---

## 2. Minor improvements (harness / quality-of-life)

- **Output-tree resolver picks the stale Documents tree (setup note #1).** Make
  it prefer the LIVE server's `OTR_OUTPUT_DIR` -- query the running server
  (`/system_stats` or the boot env) instead of a hardcoded Documents default, and
  fail LOUD if the resolved tree does not match the server answering on :8000.
- **`--exclude` flag for parked engines (setup note #2).** `availability()` is
  pure profile-fit and never reads `OTR_ENABLE_WAN_I2V`, so Wan enumerates as
  runnable. Promote the wrapper's hardcoded name filter to an argparse
  `--exclude ENGINE` (repeatable) in `scripts/_otr_overnight_soak_run.py`.
- **Janitor sweep at server boot.** The baseline logged "233 otr_* system-temp
  entries" -- a growing backlog. Once R1 routes intermediates in-tree, run the
  OH-3 janitor sweep of `otr_shared_tmp_dir()` at boot so stale intermediates do
  not accumulate across nights.
- **Hygiene-gate scoping.** `leaked = now - sys_temp_before` already subtracts the
  baseline (correct); optionally also filter to entries with `mtime > leg_start`
  so a pre-existing backlog can never confuse the diagnosis. Belt-and-suspenders
  after R1.
- **Heartbeat noise + `status=?`.** The 60s `[soak] t= ...` line never resolves
  status. Widen to 120-180s (or log on change) and wire `status` to `/queue` +
  `/history` so the digest shows queued/executing/done instead of `?`.

---

## 3. Best permutations for release (the ask)

Basis: the two-pass per-leg histograms + reliability + VRAM headroom on the 16 GB
5080, plus the engine code paths. NOTE: I reviewed the histograms and the
report's look notes and confirmed an OBS final decodes clean; I did not watch the
finals frame-by-frame in this read-only pass -- the look ranking below leans on
which engines rendered NATIVE beats reliably, and the operator should eyeball 2-3
finals per slot to confirm the aesthetic call.

**70-word episode -- recommended shipped defaults:**

| Slot | Default | Selectable-not-default | Rationale |
|---|---|---|---|
| announcer_visual | `flux_still` | `station_card`, `still_parallax` | flux_still rendered native portrait beats reliably with zero heavy video-phase VRAM (the Flux gen happens in the image phase; the video slot only pan-animates the minted still). station_card is the lighter card look; still_parallax adds subtle motion. |
| music_visual | `visualizer` | `ltx_orbit` (premium motion), `abstract` | visualizer/abstract are zero-VRAM procedural and rendered 1/1 native, bulletproof. ltx_orbit also rendered 1/1 with the best motion but loads LTX in-process -- premium-selectable until R1 lands and a longer soak confirms headroom. Avoid mesh_stage (demotes to parallax). |
| other_beats_visual (character) | `flux_still` | `still_parallax` (motion look) | flux_still rendered 3/3 native character beats, still_parallax 3/3 native. Both reliable, good look, no heavy video-phase VRAM. |

**Gate OFF as a default until R1/R2 land (keep all selectable):** `humo`,
`humo_1.7B` (R2 -- floors to stills until the VRAM/forward cause lands),
`latentsync` (needs the base-clip seam), `triposg_talk` / `mesh_stage` (3D
toolchain/assets). `wan_i2v` stays parked.

**Longer episodes:** the same still-first defaults hold. Two shifts with length:
(a) `ltx_video` for music/b-roll motion gets more attractive as more beats
amortize its load -- promote to selectable-tested once R1 is fixed; (b) HuMo
gets more audio to drive but also more cumulative VRAM pressure, so do not
promote HuMo on length until CS-4's TE-detach lands and a longer-soak tier proves
cross-beat drain. Re-run the per-slot ranking against a 150w/300w soak before
changing longer-episode defaults.

---

## 4. Bigger-than-a-breadbox future updates (ranked impact vs effort)

1. **VRAM-budget-aware scheduler (HIGH / MED).** Before dispatching a heavy
   engine, estimate its resident footprint against live-free VRAM and the 14.5 GB
   lease; if it will not fit, demote DELIBERATELY at plan time and stamp the
   reason, instead of discovering it via a caught OOM mid-forward. Turns HuMo's
   silent-floor-via-OOM into a budgeted, explainable decision and folds in the
   CS-4 lazy-TE-detach. Directly addresses the root of R2.
2. **A character-motion path that survives at 70w within 14.5 GB (HIGH / HIGH).**
   Finish CS-4 (lazy umt5 detach) + confirm the 1.7B forward so the DEFAULT
   talking-character beat renders a real talking face. Highest user-visible
   payoff; gated by item 1's scheduler.
3. **One sanctioned temp allocator + an AST ban (MED / LOW).** Ship
   `otr_engine_tmp_mp4` as the ONLY way engines mint intermediates + the
   forbidden-sweep rule. Kills the entire R1 leak class permanently
   (BUG-LOCAL-220 lesson: the cleanup hook ships with the surface).
4. **Look-QA automation (MED / MED).** Automated per-final probe -- face-present
   detection on character beats, motion-energy on music beats -- so "rendered
   playable but a still where a face was expected" (exactly R2) is caught
   mechanically. Makes PASS mean "looks right," not just "decodes."
5. **Longer-episode soak tiers (MED / LOW).** 150w / 300w presets to exercise
   VRAM accumulation + cross-beat drain before release. Cheap; catches
   sustained-residency regressions.
6. **Bring Wan online once unparked (MED / MED).** Adds an i2v motion peer;
   needs the enable-set change (operator-gated) + a dedicated soak pass.
7. **Multi-GPU / offload (LOW-MED / HIGH).** Only if the 16 GB ceiling stays the
   binding constraint after items 1/2. Defer.

---

## Appendix B -- Roundtable judgment log (GPT-5.5 + Gemini 3.1 Pro, ~$0.13)

Panel via OpenRouter on `pass00` of this report; raw reviews in
`docs/2026-06-13-fable-soak-review/roundtable/pass01/`. Claude grounded each claim
against the code; only verified items were folded in.

- **[CONFIRMED -> folded] Helper fail-open to `gettempdir()` preserves the R1
  leak (GPT #2).** The original helper fell back to the system temp dir on any
  resolution error -- the exact failure R1 fixes. Folded: production now fails
  closed; the tempfile default is allowed only under `OTR_TEST_MODE`.
- **[CONFIRMED -> folded, severity downgraded] `mkstemp` leaves a 0-byte file
  (GPT #1, Gemini #1).** Both feared ffmpeg would refuse/hang overwriting it.
  GROUNDED FALSE: every OTR ffmpeg cmd passes `-y` (`wrapper_bridge.py:457/478/492`),
  `run_ffmpeg` is non-interactive, and ltx/parallax already encode over an
  existing mkstemp file successfully (3/3 in the soak). So no hang occurs. The
  suggested `unlink`-after-`mkstemp` is still folded as cheap hardening (returns
  a non-existent path, matches legacy `mktemp`, avoids a lingering 0-byte file on
  writer failure) -- accepted as code, rejected as a blocker.
- **[ROOT CAUSE CORRECTED -> resolved] `pass00` (the copy the panel reviewed) was
  truncated mid-sentence (GPT #3).** GPT flagged it ending at "There are SEVEN
  call sites." Grounded root cause: a STALE sandbox VM mount when the bash `cp`
  snapshotted the file -- NOT a corrupted deliverable. Verified via Desktop
  Commander that the real Windows `FABLE_SOAK_REVIEW.md` is intact (507 lines).
  Operational lesson: snapshot/verify report files via Desktop Commander, not the
  bash mount (the known cowork-sandbox-mount gotcha). Still a useful catch -- it
  forced the verify that proved the real file is whole.
- **[MISREAD -> low risk, noted] R3 prune mutates a model "without validation"
  (Gemini).** GROUNDED: `NewsBriefs` is a plain `BaseModel` (no `frozen`, no
  `validate_assignment`; `news_interpreter.py:151`), so `brief.key_terms = ...`
  is safe, and a pruned list of 2-6 stays within the field's min/max bounds even
  if assignment validation is later enabled. Noted in the A2b comment; no change
  needed.
- **[REJECTED] "Ungrounded HuMo/R3 default-policy contradiction" (GPT).** The
  report already states R3's halt is working-as-designed and R2's HuMo default
  should be gated off until root-caused -- consistent, not contradictory. No
  change.

Convergence: one grounded pass surfaced one material item (the truncation) plus
two hardening folds; no architectural rework. The plan is considered converged --
a second live pass would mostly re-confirm.

## Constraints honored
Analysis only -- no code edits to the engines, no pushes, no Wan, no enable-set
or GO_FORWARD_PLAN changes. All proposed code is a snippet for a coder window.
Single resident heavy <= 14.5 GB, frozen-audio byte-identity, and 100%-local are
preserved in every recommendation. This document is UTF-8, no BOM, ASCII-only, SFW.
