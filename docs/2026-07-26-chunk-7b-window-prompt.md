# NEXT WINDOW PROMPT -- OTR multi-clip chunk 7b

Paste the block at the very bottom into a fresh window. Everything above it is
context for a human deciding whether that is still the right next step.

---

## Where the build actually is

`HEAD == origin == 6558bed0` on `v2.0-alpha`. Suite **6891 passed / 27 skipped /
1 xfailed**. Canonical workflow `5377914B` byte-identical -- nothing in chunks
1-7 touches it.

Multi-clip beat coverage chunks **1a/1b/1c/2/3/3b/4/5/6a/6b/6c/6d/7a are DONE**,
green and pushed, with nine adversarial QA rounds behind them.

**Chunk 7a changed the plan's shape**, so ignore any older doc that says chunk 7
starts by "opting `ltx_8gb` in". There is nothing left to opt in to. The
operator's ruling, verbatim:

> "this architecture should work with all video and still models. There's no
> gate with opt in or opt out. If there is, we need to remove that. Everything
> gets an equal term... I don't like any hidden opt-ins. It either works or it
> fails."

So `supports_multi_clip` is deleted from `FrameContract`, from `join_mode_for`
and from `validate_coverage_plan`. All 31 registered engines carry a static
`FrameContract`. Multi-clip is universal. The only thing an engine still EARNS
is the CHAIN, via `continuity=strict_first_frame`.

## What 7a landed (read these before touching anything)

| file | what it is |
|---|---|
| `nodes/_otr_video_engines/frame_contract.py` | the declaration surface. `min_frames / max_frames / quantum / discrete_frames / native_fps / allow_tail_trim / continuity`. `can_split()` is derived arithmetic, `can_chain()` rests on continuity alone |
| `nodes/_otr_video_engines/coverage_plan.py` | the exact-sum partitioner. Refuses rather than drifts |
| `docs/ENGINE_MATRIX.md` | **GENERATED** -- the per-model requirements record. Never hand-edit |
| `tools/engine_matrix.py` | generates it; `--check` is a drift gate wired into the suite |
| `tests/test_engine_contract_roster.py` | a registered engine with no contract fails BY NAME |
| `tests/test_multiclip_goes_live.py` | the multi-segment path driven with REAL engines, no stubs |

Two field changes worth knowing: `discrete_durations` was renamed
`discrete_frames` (the field is FRAMES; every provider publishes SECONDS and
`(4, 6, 8)` is a well-formed frame menu no validator can reject), and
`native_fps` was added so the rate those frames are counted at is stated rather
than implied.

## The hard-won lesson from 7a -- do not skip the panel

Two adversarial QA panels found **six real defects in code that was already
green and already mutation-proven**. The stubs from chunks 3-6 passed all six.

1. Declaring ceilings while the opt-in stayed shut made an ordinary 8-second
   beat fatal -- no legal single render, no multi-clip escape, and
   `partition_beat` took the whole episode's plan-build down with it.
2. Veo was declared at the PROVIDER's 24 fps. Clips are resampled to the canvas
   and counted at 25, so 96/144/192 was unreachable and would have refused
   every real Veo beat. It is 100/150/200.
3. `humo_14B_169` inherited a 177 ceiling; its real cap is 49.
4. Cloud lanes declared `quantum=1` when only whole seconds are reachable.
5. `jump_segment_still_path` demanded a still for EVERY segment >= 1 -- but a
   chained successor deliberately owns none. Six of seven sampled engines died
   at segment 1, *after* segment 0 had rendered on the GPU.
6. Audio-driven lanes would have shipped garbled lip-sync, because nothing
   slices audio per segment.

**The cadence that caught them:** build -> suite green -> mutation-prove ->
fan out (agy via kibitz + Sonnet lenses + codex) -> judge every claim against
real code yourself -> fix survivors -> re-mutate -> re-gate -> push.

## THE ENV AUDIT -- fresh intel, this is 7b's actual work list

A thorough sweep found **15 environment variables that can move a declared
contract fact**, not the 3 the old plan named. Confirmed read sites, all in
`nodes/_otr_video_engines/`:

| env var | engine(s) | read when | contradicts | clamped? |
|---|---|---|---|---|
| `OTR_LTX_MAX_FRAMES` | `ltx_video` | render | `max_frames=169` | floor only, **no ceiling** |
| `OTR_LTX_MIN_DECODE_FRAMES` | `ltx_video` | render | min/max | self-clamped to the *resolved* cap, not the literal |
| `OTR_LTX_LOOP_MIN_DECODE_FRAMES` | `ltx_video` | render | `max_frames` | floor only |
| `OTR_LTX_LOOP_VIA_REVERSE` | `ltx_video` | render | `max_frames` | **default ON**; boomerang makes a 169 target render 193 frames with no env set at all |
| `OTR_LTX_AV_MAX_FRAMES` | `ltx_audio_in` | **import** | `max_frames=497` | none; bare `int()` |
| `OTR_LTX_8GB_MAX_FRAMES` | `ltx_8gb` | render | `max_frames=161` | `[9,16384]`, no clamp to 161 |
| `OTR_HUMO_14B_SAFE_FRAMES` | `humo_14B_169` | render | `max_frames=49` | none; bare `int()` |
| `OTR_WAN_TI2V_MAX_FRAMES` | `wan_ti2v` | render | `max_frames=177` | **correctly clamped** -- the only one that is |
| `OTR_CLOUD_PIXVERSE_DURATION` | `word_razzle` | render | `discrete_frames=(125,200)` | **none** -- returns the raw int, bypassing the 5/8 menu entirely; non-int silently swallowed |
| `OTR_CLOUD_SEEDANCE_DURATION` | `cloud_seedance_2` | render | min/max | clamped, matches contract today |
| `OTR_CLOUD_WAN_DURATION` | `cloud_wan_i2v*` | render | min/max | clamped, matches today |
| `OTR_CLOUD_VIDU_Q2_DURATION` | `cloud_vidu*` | render | min/max | clamped, matches today |
| `OTR_GOOGLE_VEO_DURATION_S` / `_VIDEO_DURATION_S` | veo x4 | render | `discrete_frames` | re-bucketed, cannot escape the set |
| `OTR_FORCE_ENGINE_MAP` | ANY | route | swaps the whole contract | well-formed specs accepted with **no** check the forced engine's contract fits the plan |
| `OTR_ENABLE_HUMO_HOSTS` | HuMo -> `ltx_audio_in` | route | indirect engine swap | string equality only |

Also found: **`OTR_ACTIVE_PROFILE`** selects a capability-profile JSON whose
`max_render_frames` is documented as "the profile-carried twin of
`OTR_WAN_TI2V_MAX_FRAMES`" -- a second, non-env-named lever on the same ceiling
via `motion_common.profile_max_render_frames()`. It lives outside
`_otr_video_engines/` and was not traced end to end. **Trace it in 7b.**

**No `OTR_*FPS*` variable exists anywhere.** Every engine's rate is a hardcoded
class constant. That is worth knowing and worth keeping true.

## 7b -- the scope

**One shared refusal, not fifteen.** An engine whose environment disagrees with
its declared static contract must fail LOUD, before any GPU work, naming the
variable and both numbers. A contract that moves with the environment is a
partition the image phase could not have planned for -- stills are minted
against the contract before the render phase begins.

Design notes, offered not imposed:

* The refusal wants one helper both the engines and a suite-wide test can call,
  so "the env agrees with the contract" is asked once and answered once.
* `OTR_LTX_AV_MAX_FRAMES` resolves at IMPORT, so its check can be a module-level
  assertion. The others resolve per render and need a render-time gate.
* `OTR_WAN_TI2V_MAX_FRAMES` is already clamped correctly -- the fix there is to
  make it refuse rather than silently lower, or to declare that lowering is
  legal. Pick one and say why.
* `OTR_LTX_LOOP_VIA_REVERSE` is the ugly one: it is ON by default and already
  makes `ltx_video` return `2N-1` frames, so `ltx_video` violates its own
  declared 169 ceiling **today, with no env set**. That is arguably 7c's
  (it is a fallback), but the contract is wrong until one of them lands. Decide
  explicitly and record the decision.
* `OTR_FORCE_ENGINE_MAP` swapping in an engine whose contract cannot execute the
  already-stamped plan is a real hole. `resolve_final_shot_engines` already runs
  `assert_coverage_plans` AFTER the override (that ordering was a QA4 fix), so
  the plan IS re-validated against the forced engine -- confirm that, and if it
  holds, say so rather than adding a second check.

## Then 7c and 7d

**7c -- rip the fallbacks.** No fallbacks; all video models obey the new paths.
The operator has explicitly accepted that the build temporarily does not work:

> "we have to rip out the dead code. There's no fallbacks. All video models must
> obey these new paths... I realize it's going to temporarily not work until you
> build the code. If that's accepted. And that's why I asked you before that we
> record the requirements for each path."

The requirements ARE now recorded (`docs/ENGINE_MATRIX.md`), which was the
precondition. The list:

* ping-pong `extend_frames_to_target` -- `eng_wan_ti2v.py:521-533`,
  `eng_ltx_8gb.py:426-437`
* composite loop-fill -- `otr_silent_composite._should_loop_fill`
* held-last-frame
* **provider-side clamps, added by 7a's audit:**
  `_CloudVideoBase._duration_seconds` ends `max(min_s, min(max_s, secs))`;
  `word_razzle` does `8 if secs > 5 else 5`; Veo's `_duration_s` discards the
  requested length outright at 1080p/4k. Same defect, provider side.
* `trim_tail` is computed on single-segment plans and never applied --
  `render_beat_coverage` early-returns to the historical path. **Pre-existing
  drift**, not a regression: `wan_i2v` already quantized 50 to 53 and shipped
  53, and the composite absorbed it. Wiring the trim and removing the
  absorption belong together.
* `ltx_video`'s boomerang (see 7b note above)
* the adapter-side half of chunk 5 (r4's shape, still owed): each segment graph
  takes the prepared handles as LITERALS and omits its loader nodes

**7d -- the live slice.** A 169-frame beat (`161 + (9-1)`; 169 mod 8 == 1 is why
that number) -- >= 2 forward-only clips, ONE heavy load, no ping-pong -- plus a
162-frame CPU tail-trim case. Acceptance: `RESULT SUCCESS` + `obs_publish OK` +
the asset on disk confirmed with `Test-Path`. **Nothing has rendered through
this machine yet.** 7d is where it first does.

## Standing rules that have bitten before

* **Never blanket-kill Python.** Selective kill by CommandLine via CIM only. A
  blanket kill severs the MCP servers and, in a remote window, the bridge you
  are watching through. CLAUDE.md section 4.
* **Preserve other windows' dirty `tmp/` paths** -- `_chain_720.ps1`,
  `_rearm_gate.ps1`, `_status_bake.ps1`. Never reset/stash/checkout them away.
* **Pathspec-only commits.** Commit AND push every green chunk; verify
  `HEAD == origin`. One push attempt max, then hand over a PowerShell block.
* **Never read `/mnt/user-data/uploads/`** for this repo -- lagging snapshot.
  All reads via Desktop Commander on the Windows path.
* The repo's KNOWN-FAIL-GUARD conftest hook **suppresses pytest tracebacks**.
  To see one, write a temp `.py` under `tmp/` that calls the code directly and
  prints `traceback.print_exc()`.
* Test venv is `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` --
  the repo's own `.venv` has no pytest.
* UTF-8 no BOM, ASCII, SFW, never the word "dummy".
* $0 external spend. 100% local, offline-first.
* **THE LAW:** an audit may improve a story, never fail one for length,
  language, style, visual vocabulary, or quality.
* Two strikes then `/kibitz` before a third fix attempt.

## Model budget

Rung 1 local Qwen -> rung 2 **agy (Gemini 3.6 Flash High, $0)** -> rung 3
**Codex `gpt-5.6-sol` high** -> rung 4 Claude -> rung 5 OpenRouter -> rung 6
Fable. The operator has explicitly cleared fanning out to Sonnet subagents, agy,
and codex. Use them -- between them they have found real defects in already-green
code six times.

Kibitz: `python kibitz/scripts/kibitz.py --doc <plan>.md --round r3 --only agy`
(and `--only codex`). `KIBITZ_AGY_MODEL` defaults to "Gemini 3.6 Flash (High)".

---

# THE PROMPT -- paste from here down

```
Resume the OTR build -- run the otr-handoff skill; you are CODER WINDOW A per
GO_FORWARD "Window packing". Read docs/GO_FORWARD_PLAN.md IN FULL (the CURRENT
STEP section is authoritative and was rewritten 2026-07-26 -- older text about
"the first adapter opt-in" is dead), the last few docs/HANDOFF_LOG.md entries,
docs/2026-07-26-chunk-7b-window-prompt.md, and git log/status.

State your MODEL & CREDIT BUDGET rung, then the current step, then a 5-line
scope/acceptance summary, then STOP until I confirm.

CURRENT STEP IS CHUNK 7b: the env-vs-contract refusal. Chunk 7a gave all 31
engines a static FrameContract and DELETED the supports_multi_clip opt-in --
there is no opt-in left anywhere, and there must not be. An engine whose
ENVIRONMENT disagrees with its DECLARED contract must now fail LOUD, before any
GPU work, naming the variable and both numbers. A contract that moves with the
environment is a partition the image phase could not have planned for.

An audit already found FIFTEEN env vars that can move a declared contract fact,
not the three the old plan named -- the full table with file:line, read-time
(import vs render) and clamping behaviour is in
docs/2026-07-26-chunk-7b-window-prompt.md. Start from that table; do not
re-derive it, but DO verify any row you are about to act on. Two rows need a
decision recorded rather than just a fix: OTR_LTX_LOOP_VIA_REVERSE (default ON,
already makes ltx_video return 2N-1 frames, so it violates its own 169 ceiling
today with no env set) and OTR_FORCE_ENGINE_MAP (check whether the existing
post-override assert_coverage_plans already closes the hole before adding a
second check). Also trace OTR_ACTIVE_PROFILE, which reaches the same wan_ti2v
ceiling through motion_common.profile_max_render_frames() and was not traced
end to end.

ONE shared refusal, not fifteen -- "the env agrees with the contract" gets asked
once and answered once, by a helper both the engines and a suite-wide test call.

Execute in order, one green pushed chunk at a time (commit AND push,
pathspec-only, verify HEAD == origin). Mutation-prove every fix -- a test that
has not been shown to fail for the right reason is decorative. Then fan out for
QA before pushing: agy via kibitz (--only agy, $0), codex gpt-5.6-sol, and
Sonnet subagent lenses; judge every claim against real code YOURSELF before
acting on it. Two QA panels found six real defects in chunk 7a's already-green,
already-mutation-proven code, and the stubs passed all six.

Two strikes then /kibitz. THE LAW holds. Never blanket-kill Python -- selective
CIM kill by CommandLine only. Preserve other windows' dirty tmp/ paths
(tmp/_chain_720.ps1, tmp/_rearm_gate.ps1, tmp/_status_bake.ps1). Never read
/mnt/user-data/uploads/ for this repo; all reads via Desktop Commander on
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. The test
venv is C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe. UTF-8 no BOM,
ASCII, SFW, never the word "dummy". $0 external spend.
```
