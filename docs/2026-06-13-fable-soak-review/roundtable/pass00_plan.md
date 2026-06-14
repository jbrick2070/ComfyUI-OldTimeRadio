# Fable -- OTR Overnight-Soak Review + Recommendations

Read-only analysis of the two back-to-back 25-leg coverage soaks (2026-06-12 ->
06-13, branch v2.0-alpha). No code changed, nothing pushed, Wan untouched, the
16gb_full enable-set and GO_FORWARD_PLAN section 1A untouched. Every proposed
code change below is written as a snippet for a separate coder window to apply.

## Evidence base
- `scripts/overnight_soak_report.md` (pass 1 + pass 2, per-leg table, histograms).
- `scripts/overnight_soak_run.log` (per-leg headers, histograms, the failing
  hygiene-gate message, the writer execution_error).
- Source: `nodes/_otr_video_engines/{cheap_families,eng_ltx_video,eng_still_parallax,eng_humo,render_driver}.py`,
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

## 1. Fixes (must-fix for a green sweep)

### R1 -- floor / cheap-family temp-file leak (PRIMARY; flips most legs green)

**Root cause (confirmed).** Four engine render helpers create their intermediate
mp4 with `tempfile.mktemp` / `tempfile.mkstemp` and pass NO `dir=` argument, so
the file lands wherever `tempfile.gettempdir()` resolves:

| File:line | Call | Leaked prefix |
|---|---|---|
| `nodes/_otr_video_engines/cheap_families.py:122` | `tempfile.mktemp(suffix=".mp4", prefix="otr_floor_%s_" % self.name)` | `otr_floor_*` (every leg -- the floor always runs) |
| `nodes/_otr_video_engines/eng_still_parallax.py:302` | `tempfile.mkstemp(suffix=".mp4", prefix="otr_parallax_")` | `otr_parallax_*` |
| `nodes/_otr_video_engines/eng_ltx_video.py:721` | `tempfile.mkstemp(suffix=".mp4", prefix="otr_ltx_")` | `otr_ltx_*` |
| `nodes/_otr_video_engines/eng_humo.py:342` | `tempfile.mktemp(suffix=".mp4", prefix="otr_humo_")` | `otr_humo_*` (latent -- did not fire only because HuMo floored, see R2) |

A fifth site of the same class: `render_driver.py:323` builds the per-beat audio
slice dir as `os.path.join(tempfile.gettempdir(), "otr_audio_slices")` -- another
`otr_*` entry in the ambient temp dir.

The hygiene gate `assert_no_stray_writes` (`scripts/_otr_soak_capstone.py:212`)
checks that the SYSTEM temp dir (`%LOCALAPPDATA%\Temp`) gained no new `otr_*`
entries. The gate's own docstring states the design contract: "the in-tree TEMP
repoint held." The soak launcher does repoint it
(`_otr_soak_server_launch.cmd:55-58` sets `TEMP`/`TMP` to
`<output>\otr\episodes\_shared\tmp`). The bug is that these five call sites
DEPEND on that ambient repoint holding for the live server -- and it did not.
The run log shows the leaked files are `otr_floor_still_kenburns_*.mp4` in
`C:\Users\jeffr\AppData\Local\Temp`, and the per-leg baseline already counts
"233 otr_* system-temp entries" growing across legs. So the intermediates are
escaping to the system temp dir and never being unlinked (the clip `path` is
consumed downstream by SilentComposite, so the engine cannot unlink synchronously).

This is exactly the failure class CLAUDE.md already flags: the v2 install move
dropped `PYTHONUTF8` from the boot path; relying on an ambient process env for a
correctness/hygiene invariant is the same trap. The fix should not depend on a
launcher remembering to set `TEMP`.

**Fix (concrete).** Route every engine intermediate to the in-tree tmp tier
explicitly via `dir=`, independent of the ambient `TEMP`. The janitor already
owns deletion of that tier (`nodes/_otr_janitor.py`, the OH-3 sanctioned
auto-delete sweeps `otr_shared_tmp_dir()`), so the intermediates get cleaned and
the clip path stays valid through the compositor.

Add one shared helper (new, in `nodes/_otr_video_engines/` or `_otr_shared/`):

```python
# nodes/_otr_video_engines/_tmp.py  (new)
import os, tempfile

def otr_engine_tmp_mp4(prefix):
    """Create an empty intermediate .mp4 under the in-tree shared tmp tier
    (never the ambient system temp dir). Returns an absolute path; the OH-3
    janitor sweeps this tier. Falls back to gettempdir() only if the output
    tree cannot be resolved (headless/CPU unit tests)."""
    try:
        try:
            from .._otr_paths import otr_shared_tmp_dir
        except ImportError:
            from _otr_paths import otr_shared_tmp_dir  # type: ignore
        d = str(otr_shared_tmp_dir())
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = None  # last-resort: tempfile default (unit tests w/o an output tree)
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix=prefix, dir=d)
    os.close(fd)
    return path
```

Then replace the four engine sites with `otr_engine_tmp_mp4("otr_floor_%s_" %
self.name)` / `"otr_parallax_"` / `"otr_ltx_"` / `"otr_humo_"`, and point the
audio-slice dir at `otr_shared_tmp_dir()/audio_slices` instead of
`gettempdir()`. This also retires the deprecated bare `mktemp` (a known
security/clobber footgun) in favor of `mkstemp`.

**Risk / blast radius.** Low. Five helper functions; the returned clip `path`
contract is unchanged (only the parent directory moves). The one thing to verify
on the box: `otr_shared_tmp_dir()` resolves correctly when these engines run
inside the live ComfyUI process (it depends on `comfy_output_dir()`); it does, as
the same tree already holds the audio slices today.

**Regression test to add.**
- `tests/test_engine_tmp_in_tree.py`: monkeypatch `otr_shared_tmp_dir()` to a
  pytest `tmp_path`, run each engine's `render_clip` (or `otr_engine_tmp_mp4`)
  on a CPU stub, assert the returned path is under `tmp_path` AND that
  `tempfile.gettempdir()` gained no `otr_*` entry during the call.
- Extend the existing AST forbidden-sweep (the b7-style guard) with a rule:
  no `nodes/**` file may call `tempfile.mktemp(` or `tempfile.mkstemp(` /
  `tempfile.gettempdir()` for an `otr_*`-prefixed artifact without a `dir=`
  argument. This makes the leak class permanently un-reintroducible.

### R3 -- writer key_term post-validator halt (INTERMITTENT, ~1-2/25)

**Root cause (confirmed).** The abort is `OTR_LedgerScriptWriter` (node 1)
exhausting the `structured_call` retry ladder on `build_news_briefs`:
`PostValidationError: V1: key_term 'climate mitigation' not in source` (pass 2;
pass 1 was `'sustainability'`). The failing key_terms are THEME ABSTRACTIONS the
local writer LLM (gemma) invents -- a topic label that is genuinely not a verbatim
or paraphrased span of the source article. The Sprint 10B LLM-as-judge semantic
fallback (`news_interpreter.v1_validate(..., judge_fn=...)`) IS wired
(`news_interpreter.py:811-815`) and correctly REJECTS these (the message has no
"(strict + LLM-judge)" suffix in the strict-only branch, but the judge path also
returns a real failure for a fabricated topic -- it is not in the article).

On exhaustion the writer HALTS the graph **by design**: the `news_briefs_required`
widget defaults TRUE per the operator directive baked into the tooltip
(`OTR_LedgerScriptWriter.py:1887-1912`, Jeffrey 2026-05-27: "the whole workflow
needs to stop and re-roll news until it works and stamps the ledger"). In live
production the operator re-queues, which pulls a fresh RSS article (the re-roll).
In the soak there is no re-queue and the story RNG is unique per leg, so ~1-2 of
25 legs draw a story the writer fabricates a theme term for, and the leg aborts.

**Verdict: this is working-as-designed fail-closed, not a code bug.** What to
change is policy, and it splits cleanly by audience:

- **For the soak (recommended, do first):** the soak is testing RENDER engines,
  not the writer. A bad brief should degrade, not abort the engine-under-test.
  Set `news_briefs_required=False` for soak legs (graceful-degrade to raw
  news_seed). Pure harness change, zero production impact. Removes R3 from the
  sweep entirely.

- **For the product (recommended durable fix; coder window):** add a
  prune-to-floor self-heal in `_content_validator` (`news_interpreter.py:798`).
  When V1 fails on only a SUBSET of key_terms and at least `_MIN_KEY_TERMS` (2)
  in-source terms survive, DROP the fabricated terms and accept the pruned brief
  rather than re-rolling/halting. Keep the hard halt only when fewer than the
  floor survive. This preserves the V0 anchor floor (>=2 source-grounded terms)
  while making a single hallucinated label non-fatal.

  ```python
  # sketch inside _content_validator, after v1 failures are known:
  valid_terms = [t for t in brief.key_terms if _strict_in_source(t, source_text_full)]
  if len(valid_terms) >= _MIN_KEY_TERMS and len(valid_terms) < len(brief.key_terms):
      brief.key_terms = valid_terms          # prune fabricated; keep grounded
      return None                            # accept the pruned brief
  ```

- Prompt tightening is low yield: the prompt already says "verbatim from the
  source" (`news_interpreter.py:704`); gemma ignores it a few percent of the
  time, and GBNF cannot enforce semantic presence.

**Risk / blast radius.** Harness option: none. Prune option: low -- a slightly
weaker brief can pass, but the V0 floor still guarantees >=2 grounded anchors;
production `news_briefs_required` can stay ON for the true-halt path.

**Regression test to add.** `tests/test_news_v1_prune.py`: a brief with 4
key_terms (2 in-source, 2 fabricated) prunes to the 2 valid and validates clean;
a brief with only 1 in-source still fails V0 (below the floor) and halts.

### R2 -- HuMo / motion / 3D engines floor at 70 words

**What I confirmed.** All HuMo model handles are installed on the box
(`C:\ComfyUI-Models\diffusion_models\humo_1.7B_fp16.safetensors`,
`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`;
`text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors`;
`loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`;
`audio_encoders\whisper_large_v3_fp16.safetensors`). So the flooring is **NOT a
missing-model gate.** The render path also wires HuMo's inputs correctly: the
portrait resolves from the portrait index and the per-beat audio is sliced from
the frozen master (`render_driver.py:657-746`). HuMo therefore receives both
`init_image` and `audio_ref`; the demotion happens at LOAD/FORWARD time and is
caught by the fallback loop (`render_driver.py:1106-1133`), which restamps LOUD
and walks `humo -> humo_1.7B -> latentsync -> still_kenburns`.

**The finding splits into EXPECTED behavior and a REAL must-root-cause item:**

- **latentsync 0/6 -- EXPECTED.** latentsync is the `lipsync_overlay` family,
  which requires a `base_clip_ref` (a base talking clip to lip-sync onto). The
  70-word soak provides none and the `_provide_lipsync_base` seam does not
  synthesize one in this config, so `_assert_family_inputs_satisfiable` raises
  `FamilyInputGap` and the chain LOUD-skips to the floor. This is an input gate,
  not a defect.
- **triposg_talk 0/6, mesh_stage -> parallax -- EXPECTED at 70w on this box.**
  The 3D talking path needs mesh assets / a cu128 toolchain; its cousins
  (`hunyuan3d_talk`, `trellis_talk`) are already SKIPPED_DISABLED for exactly
  that reason. Demoting to parallax/still is the designed contingency.
- **HuMo 14B 0/6 -- consistent with CS-4.** The umt5 fp8 text-encoder (~5.2 GB
  resident, per the CS-4 finding) co-resident with the 14B forward starves the
  14.5 GB lease -> OOM -> hard fallback. This matches the known CS-4-open item
  (lazy TE-detach).
- **HuMo 1.7B (the DEFAULT) 0/6 -- the genuinely concerning one.** The 1.7B
  stack is light (~3.3 GB UNET + umt5 + whisper) and PASSES the standalone
  ~38-min acceptance render elsewhere (per the CS-4 record). Its uniform 0/6
  here, with all files present, points to a runtime forward/lease failure or a
  co-residency spike (still/portrait-phase residue not fully drained before the
  first video beat, despite the `run_episode` pre-render reclaim at
  `render_driver.py:1162-1167`) rather than to missing inputs.

**What I could not disambiguate from the given inputs.** The per-beat
`classify_failure` reason (OOM vs a forward/wrapper error) is written by
`format_swap_log` to the LIVE SERVER log (`otr_runtime.log`), not to
`overnight_soak_run.log` (the client view, which only carries the histogram).
The client log shows the histograms and the gate failure but never a swap line.

**Verdict.** R2 is a bug to fix for the DEFAULT character path -- a shipped
default that silently (LOUD in the trace, but invisible to a viewer) delivers
still frames where a talking face is expected is a quality regression, even
though the episode renders playable. It is NOT a bug for latentsync / 3D at 70w
(expected). The concrete next step to CLOSE it is mechanical: capture the
server-side swap detail for one HuMo leg (`otr_runtime.log`, the
`format_swap_log` lines: `from_engine -> to_engine kind detail`) and branch:
- if `kind == OOM` (expected for 14B): land the CS-4 lazy umt5-TE detach (free
  the umt5 encoder after `CLIPTextEncode`, before the HuMo sampler) so the 14B
  fits; for the 1.7B confirm the pre-render reclaim actually drains the still
  phase before beat 1.
- if a forward/wrapper error: fix the wrapper-node resolution / SageAttention /
  the 4n+1 length path in `eng_humo._build_graph`.

Until that lands, keep HuMo selectable-not-default (see section 3).

**Risk / blast radius of the eventual fix.** Medium -- it touches the in-process
VRAM discipline (the exact area CS-4 / BUG-291 live in); must be GPU-soaked
(single resident heavy <= 14.5 GB, audio byte-identical) before shipping.

**Regression test to add.** A GPU-smoke assertion (operator lane): after a HuMo
beat, assert the trace row's `final_engine == "humo_1.7B"` (i.e. the default
character beat rendered a real talking face, not `still_kenburns`) for a
fixed-seed 1-beat episode within the 14.5 GB ceiling. CPU-side, add a unit test
that a character_video request carrying both `init_image` and `audio_ref`
satisfies `_assert_family_inputs_satisfiable("humo", req)` (guards against an
input-gate regression masquerading as VRAM).

---

## 2. Minor improvements (harness / quality-of-life)

- **Output-tree resolver picks the stale Documents tree (setup note #1).** The
  harness auto-resolved to `Documents\ComfyUI\output` (the pre-Desktop-v2 path),
  so every leg orphan-rejected its report until `OTR_SOAK_SERVER_OUTPUT` was
  pinned. Make the resolver prefer the LIVE server's `OTR_OUTPUT_DIR` -- query
  the running server (`/system_stats` or the env it booted with) instead of a
  hardcoded Documents default, and fail LOUD if the resolved tree does not match
  the server that is actually answering on :8000.
- **`--exclude` flag for parked engines (setup note #2).** `availability()` is
  pure profile-fit and never reads `OTR_ENABLE_WAN_I2V` (that env only gates
  load/render), so Wan enumerates as runnable regardless. Today the wrapper
  filters it by name; promote that to a first-class `--exclude wan_i2v` harness
  flag (and/or have `enumerate_options` honor the enable-env for flag-gated
  engines) so future parked engines do not need a code edit to skip.
- **Janitor sweep at server boot.** The per-leg baseline logged "233 otr_*
  system-temp entries" -- a growing backlog from prior runs. Once R1 routes
  intermediates in-tree, run the OH-3 janitor sweep of `otr_shared_tmp_dir()` at
  server boot so stale slices/intermediates do not accumulate across nights.
- **Hygiene-gate scoping.** The gate's `leaked = now - sys_temp_before` already
  subtracts the baseline, which is correct; but consider additionally filtering
  the leaked set to entries with `mtime > leg_start` so a pre-existing backlog
  can never confuse the diagnosis. (After R1 this is belt-and-suspenders.)
- **Heartbeat noise + `status=?`.** The `[soak] t= NNNs status=? vram_peak=...`
  line prints every 60s and never resolves status. Widen the cadence to 120-180s
  (or log only on change) and wire `status` to `/queue` + `/history` so the
  digest shows real progress (queued/executing/done) instead of `?`.

---

## 3. Best permutations for release (the ask)

Basis: the two-pass per-leg histograms + reliability + VRAM headroom on the 16 GB
5080, plus the engine code paths. NOTE: I reviewed the histograms and the
report's look notes and confirmed an OBS final decodes clean; I did not watch the
finals frame-by-frame in this read-only pass -- the look ranking below leans on
which engines rendered NATIVE beats reliably, and the operator should eyeball the
2-3 finals per slot to confirm the aesthetic call.

**70-word episode -- recommended shipped defaults:**

| Slot | Default | Selectable-not-default | Rationale |
|---|---|---|---|
| announcer_visual | `flux_still` | `station_card`, `still_parallax` | flux_still rendered native portrait beats reliably with zero heavy video-phase VRAM (the Flux gen happens in the image phase; the video slot only Ken-Burns-animates the minted still). station_card is the lighter card look; still_parallax adds subtle motion. |
| music_visual | `visualizer` | `ltx_orbit` (premium motion), `abstract` | visualizer/abstract are zero-VRAM procedural and rendered 1/1 native, bulletproof. ltx_orbit also rendered 1/1 and gives the best motion (orbit over a still) but loads LTX in-process -- keep it as the premium-selectable until R1 lands and a longer soak confirms headroom. Avoid mesh_stage (demotes to parallax). |
| other_beats_visual (character) | `flux_still` | `still_parallax` (motion look) | flux_still rendered 3/3 native character beats, still_parallax 3/3 native parallax. Both reliable, good look, no heavy video-phase VRAM. |

**Gate OFF as a default until R1/R2 land (keep all selectable):** `humo` and
`humo_1.7B` (R2 -- floors to stills until the VRAM/forward root-cause lands),
`latentsync` (needs the base-clip seam), `triposg_talk` / `mesh_stage` (3D
toolchain/assets). `wan_i2v` stays parked.

**Longer episodes:** the same still-first defaults hold. Two things shift with
length: (a) `ltx_video` for music/b-roll motion gets more attractive because more
beats amortize its in-process load -- promote it to selectable-tested once R1 is
fixed; (b) HuMo gets more audio to drive per beat but also more cumulative VRAM
pressure, so do NOT promote HuMo on length until CS-4's TE-detach lands and a
longer-soak tier (section 4) proves cross-beat drain. Re-run the per-slot ranking
against a 150w/300w soak before changing longer-episode defaults.

---

## 4. Bigger-than-a-breadbox future updates (ranked impact vs effort)

1. **VRAM-budget-aware scheduler (HIGH impact / MED effort).** Before dispatching
   a heavy engine, estimate its resident footprint against live-free VRAM and the
   14.5 GB lease; if it will not fit, demote DELIBERATELY at plan time and stamp
   the reason, instead of discovering it via a caught OOM mid-forward. This turns
   HuMo's silent-floor-via-OOM into a budgeted, explainable decision and folds in
   the CS-4 lazy-TE-detach. Directly addresses the root of R2.
2. **A character-motion path that survives at 70w within 14.5 GB (HIGH / HIGH).**
   Finish CS-4 (lazy umt5 detach) + confirm the 1.7B forward so the DEFAULT
   talking-character beat actually renders a talking face. Highest user-visible
   payoff; gated by item 1's scheduler to stay within budget.
3. **One sanctioned temp allocator + an AST ban (MED / LOW).** Ship the
   `otr_engine_tmp_mp4` helper from R1 as the ONLY way engines mint
   intermediates, and add the forbidden-sweep rule banning bare
   `mktemp`/`mkstemp`/`gettempdir` for `otr_*` artifacts in `nodes/**`. Kills the
   entire R1 leak class permanently (the BUG-LOCAL-220 lesson: the cleanup hook
   ships with the surface).
4. **Look-QA automation (MED / MED).** An automated per-final probe -- face-present
   detection on character beats, motion-energy on music beats -- so "rendered
   playable but it is a still where a face was expected" (exactly R2) is caught
   mechanically, not by eyeball. Makes the soak's PASS mean "looks right," not
   just "decodes."
5. **Longer-episode soak tiers (MED / LOW).** Add 150w / 300w presets to the
   harness to exercise VRAM accumulation + cross-beat drain before release. Cheap
   to add; catches the regressions that only appear under sustained residency.
6. **Bring Wan online once unparked (MED / MED).** Adds an i2v motion peer;
   requires the enable-set change (operator-gated) + a dedicated soak pass.
7. **Multi-GPU / offload (LOW-MED / HIGH).** Only if the 16 GB ceiling stays the
   binding constraint after item 1/2 land. Defer.

---

## Appendix A -- Detailed, copy-paste code fixes (the easy wins)

Line numbers are from the current v2.0-alpha working tree. A coder window should
re-grep before editing in case of drift. Every edit below is ASCII, no-BOM safe.

### A1. R1 temp-leak -- COMPLETE fix (high confidence, fully traced)

There are SEVEN call sites, all t