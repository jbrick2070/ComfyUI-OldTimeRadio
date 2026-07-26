# JUDGMENT -- the 7b blockers arc (r1 -> r4), and what it changed

Full 4-round kibitz arc, 8 agent calls (agy Gemini 3.6 Flash High + codex
`gpt-5.6-sol` high, both pinned and verified per round). Driver anchor written
before each fan-out; every claim below checked against source.

Base at arc start `c8cf0b07`; at arc end **`8f41af27`**, suite
**6925 -> 6983 passed / 27 skipped / 1 xfailed**, Bible 17, link validator 0
violations.

---

## 1. WHAT LANDED

| slice | commit | what |
|---|---|---|
| C1 | `7f4644a1` | node 87's `max_render_frames` input descriptor in `otr_canonical.json` -- the profile-ceiling channel was dead at its first hop |
| C2 | `ac609d25` | the plan-vs-output proof's fail-OPEN predicate closed; an unreadable count is now a failed verification |
| C1b | `8f41af27` | the SAME dead widget in all ELEVEN shipped variant workflows, including both 8GB tiers |

All three mutation-proven, C2 and C1b with explicit CONTROLS (a tightening that
also refuses honest input is not a fix).

**The single most valuable thing the arc produced** is C1b's discovery, and it
came from an agy r4 "verify-at-build" line rather than from a MUST-FIX:
`variants/otr_8gb_wan.json` carried an orphan widget value of **17**, not the
harmless `0` default -- matching `config/profiles/otr_8gb_wan.json:56`, the ONLY
shipped profile that pins `max_render_frames`. The WAN 8GB launch contract's
ceiling had been deliberately configured and silently ignored since it shipped,
because the value had no descriptor to arrive through. That is exactly the
failure `test_floor_max_override_is_an_absolute_hard_cap` was written after.
It was found because the wiring script REFUSED an unexpected value instead of
assuming one.

## 2. LIVE FACTS ESTABLISHED (receipts, not claims)

* The server path
  `C:\Users\jeffr\ComfyUI-Installs\...\custom_nodes\ComfyUI-OldTimeRadio` is a
  **junction** to this repo -- identical SHA-256, same HEAD. Every "live proof"
  in this build rests on that, and it was unverified until now.
* **First live render of this architecture: PASS.** `ltx_8gb`, 25 frames,
  20.8s, `frame_count=25` exactly as asked, VRAM peak 3004 MB. Labelled
  `7d-preflight`, NOT qualification.
* Boot is clean and 18s on an auto-resolved port; the canonical JSON loads.
* Registry-probed 7d lanes: `ltx_8gb` 169 -> `[161,9]`, `wan_i2v` 209 ->
  `[177,33]`, `wan_ti2v` 193 -> `[177,17]`. **All viz engines -- the canonical
  defaults -- have NO ceiling, so the default route can never exercise
  multi-clip.** `ltx_video` is disqualified: its default boomerang returns 193
  for a 169 ask.

## 3. THE ARCHITECTURE, AS SETTLED

1. **`frame_contract_for` stays STATIC.** `tools/engine_matrix.py:145-152`
   generates a committed doc from it under a `--check` gate.
2. **A per-ENGINE policy registry, not per-family.** Seedance, Cloud Wan, Vidu,
   Pixverse and Kling have different ranges, quantizers and duration inputs
   inside ONE adapter module (`eng_cloud_video.py:623-1011`). Keyed by engine
   ID, with alias resolution through `public_engines.resolve_engine_id`
   (`:67`), and required to equal `registry.all_engine_names()` exactly.
   **The universal `env -> profile -> literal` rule is CUT** -- the adapters
   differ historically on purpose and flattening them would silently change
   untested lanes.
3. **Three control modes: `exact` / `menu` / `provider_chosen`.** Google Omni
   advertises an arithmetic contract but its outbound request carries no
   duration control (`eng_google_omni_video.py:180-194`), so strict equality
   would fail only AFTER a paid call. `provider_chosen` refuses exact coverage
   at preflight.
4. **Typed taxonomy:** `UnresolvableEngineError` (unregistered/stub, may stay
   caught) vs `InvalidEngineConfigError` (terminal). Three swallow sites:
   `frame_contract.py:243`, `otr_shot_lock.py:1154`, `render_driver.py:3435`.
   A stamped plan must never degrade to `contract=None`.
5. **A separate `frame_policy_snapshot` with TWO consumers.** Never inside
   `routing_env_snapshot` (whose fingerprint feeds a terminal route-drift
   check). It must be fingerprinted into `IS_CHANGED` **and** stamped on the
   ledger and re-validated at the render boundary -- because headless runs
   never consult `IS_CHANGED` (`render_driver.py:3324-3326`).
6. **The receipt is a SIBLING, never a mutation of `coverage_plan`.** Five
   existing readers deserialize the current shape
   (`render_driver.py:743-765, 820-846, 2861-2864, 3447-3450`;
   `otr_image_gen_dispatcher.py:514-567`). Attach it to the returned clip and
   keep `render_beat_coverage`'s four-value return.
7. **ONE receipt builder for BOTH paths.** `render_beat_coverage` returns early
   for single-segment beats, so a receipt wired into the multi-segment tail
   would miss every beat production actually runs.

## 4. OPEN BLOCKERS -- verified, none fixed yet

**O1 -- THE CANVAS, and it is the 7d blocker.** Both seats, independently.
`build_request_from_shot` overwrites `req["canvas"]` to `1472x832` for every
non-face engine (`render_driver.py:2268-2273`), with deliberate per-engine
branches following for `ltx_video` (832x480) and `ltx_av` -- **but none for
`ltx_8gb`**. `OTR_VideoRenderBatch` passes no canvas at all
(`otr_video_render_batch.py:372-373`) and `build_request` hard-codes 25 fps
(`render_driver.py:221-256`). So the `otr_8gb_ltx` profile's 512x288 render
canvas (`config/profiles/otr_8gb_ltx.json:64-66`) is displaced by 1472x832 --
on the tier that exists BECAUSE 8GB cannot afford the big canvas.

NOT FIXED DELIBERATELY. The two seats prescribe different remedies (codex:
assert the final request is exactly 512x288/25; agy: respect the ledger canvas
with env fallback), the surrounding comments document per-engine render
canvases that exist for real quality reasons (BUG-LOCAL-412: LTX-2B "re-noises
into mush at 1472x832"), and `render_frames` is a hot path every engine
traverses. The clean shape is probably an `ltx_8gb` branch consuming the
already-stamped `ledger.video.canonical_canvas`
(`otr_shot_lock.py:1537-1541`) -- codex CUT 3 is right that no new widget is
needed -- but "probably" is not the standard for a hot path.

**O2 -- a THIRD validation bypass.** Exported `run_episode` renders without
`resolve_final_shot_engines` or `assert_coverage_plans`
(`render_driver.py:3038-3052, 3143-3152`), and the soak calls it directly
(`:4106-4129`). `run_real_episode` validates first (`:3274-3281`), but direct
callers bypass it. This answers r3's open question 5.4: yes, there is one.

**O3 -- `run_graph` cannot accept preloaded results**
(`wrapper_bridge.py:172-184, 301-336, 364-372`), so 7c's "remove the embedded
loaders" is not a first step. Required order: add
`external_results={node_id: normalized_output_tuple}` support -> run loaders
once in `prepare` -> segment graphs omit those specs but keep wires to the
external IDs -> teardown drops handles once -> only then delete the embedded
loaders and the ping-pong -> finally assert `checkpoint_loader_calls == 1` and
`clip_loader_calls == 1`.

**O4 -- the 169-frame beat's production seam needs a schema extension.**
`derive_opening_music_beat` computes `frames = round(first_start * fps)`
(`otr_shot_lock.py:514-541`), so 7.26s - 500ms crossfade = 6.76s x 25 = exactly
169, on the `music_visual` role that `otr_8gb_ltx` already routes to `ltx_8gb`.
But `opening_duration_sec` / `crossfade_ms` are not accepted by the profile
schema (`capability_profiles.py:158-167`, `_otr_workflow_apply.py:519-523`).
**`render.frame_budget` is NOT the mechanism** -- that widget is inert in
episode mode (`otr_video_render_batch.py:189-192`).

## 5. CLAIMS THAT DID NOT SURVIVE

**Driver, refuted by the panel (3):** live VRAM silently shortens renders
(S4 made it RAISE); the single path asks for the beat target rather than
`plan.segments[0].render_frames` (`segment_render_frames` answers from the plan
for index 0); `render.frame_budget: 49` caps episode beats (inert in episode
mode).

**Panel, refuted by the driver (4):** clamp the boomerang to the ceiling
(`test_loop_source_length_no_freeze_shortfall` pins the opposite for exactly
169 -- it would trade a ceiling violation for a returning freeze); re-partition
against a forced engine at render time (silent re-plan after stills are
minted); `FrameContract` needs `to_dict` for frozensets/enums (it holds a tuple
and a str, both JSON-safe); **raise `ltx_8gb`'s ceiling to 169 so the opening
beat stays single-segment** (this inverts the objective -- 169 is chosen
BECAUSE it splits `[161,9]`, and that split IS the proof).

Also rejected: a `frame_budget: 169` profile variant (inert); a tagged
`duration_estimate` fallback (a tagged fallback is still a fallback); a second
force-map check (`test_the_legacy_path_validates_the_plan_against_the_FINAL_engine`
already covers it); pre-B1 migration shims for hypothetical saved graphs.

## 6. ORDER FROM HERE

O1 canvas -> C3 (per-engine registry + taxonomy + canvas resolution) ->
C4a receipt carrier (sibling, both paths) -> C5a offline policy/preflight tests
(exact/menu single-call, provider_chosen zero-call, transport-spied) ->
O2 close the third bypass -> C6 stamped-vs-live + projection after terminal
injection -> O4 profile schema for the opening seam -> **7c** in the O3 order
-> C4b strict receipt population -> **7d** live, local-only.

7d acceptance, restored in full: 169 visible from `[161,9]` with one overlap
drop, >= 2 forward-only clips, ONE loader phase, no ping-pong/replan/
truncation/hold, plus the 162-frame CPU tail-trim case; then `RESULT SUCCESS`,
`obs_publish OK`, and both canonical assets confirmed on disk.
