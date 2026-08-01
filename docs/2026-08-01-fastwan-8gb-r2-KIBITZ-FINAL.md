# fastwan_8gb -- r2 KIBITZ FINAL (rulings + grounded claim ledger)

Durable record of the r2 round. Panel: codex `gpt-5.6-sol` (high), antigravity
`Gemini 3.5 Flash (High)`. Driver and sole judge: Claude (Cowork).
Working artifacts (gitignored, local only):
`kibitz-runs/2026-07-31-fastwan-8gb-engine/r2/`.
Input under review: `docs/2026-08-01-fastwan-8gb-WIRING-PLAN-r2.md` (partly superseded
by this file). Predecessor: r1 final, recorded in
`docs/HANDOFF_2026-08-01-fastwan-8gb-WIRING.md`.

---

Panel: codex `gpt-5.6-sol` (high), antigravity `Gemini 3.5 Flash (High)`. Driver and
sole judge: Claude. Claim ledger: `judgment.md`. Every line below is grounded.

---

## THE THREE QUESTIONS, RULED

### Q1 -- LOW-VRAM CANVAS: declare 832x480 on BOTH adapters. The reason is boot-independence, not VRAM.

**The premise this question was asked on is refuted.** The 2026-07-23 leg -- the one
whose 8251-9811 MB was cited as proof that the incumbent renders at 1472x832 --
**rendered at 832x480**. Measured, not inferred: all seven surviving per-clip mp4s in
`output/otr/episodes/signal_lost_the_damp_grave_of_julian_vane_20260723_182900/clips/`
ffprobe at 832x480. Channel C (`OTR_VIDEO_LANDSCAPE_CANVAS=832x480`,
`config/profiles/otr_8gb_wan.json:82`) bound correctly on that boot.

**The VRAM spread is a measurement-scope change, the third candidate GO_FORWARD item 8
named and told us not to skip past.** `2b095143` (2026-07-20) threaded the real
render-window NVML peak into `wan_ti2v`; before it the path fell back to an
instantaneous post-render read that under-reports. All six low readings (3372-4046 MB)
are June, before the fix; the single high reading is 2026-07-23, after it. Compounding
it, `VramPeakProbe` samples MACHINE-WIDE NVML while the bench's 6563.1 MiB is a
`peak_delta` -- an absolute against a delta is a category difference, not a gap.

**RULING.** Declare `render_canvas = (832, 480)` on `fastwan_8gb` AND on `wan_ti2v`.
This changes no pixels -- it is already what renders -- and that is the point: today
the correct canvas arrives through a **boot-coupled env channel**, which is the
`PBUG-20260723-02` dead-channel class. A leg submitted to a server booted without the
tier's `launch.env` silently renders at 1472x832. The declaration is boot-independent
and is applied LAST in `build_request_from_shot` (`render_driver.py:2544-2557`).

Consequent rulings:
- **Do NOT touch `_FRAME_COST_REF_PIXELS`.** It is the calibration reference the cost
  row was fitted at (`motion_common.py:253-262`); rewriting it silently rescales every
  prediction.
- **`_TI2V_COST_REF_W/H` alignment is OPTIONAL**, not a production fix. Production
  always passes real dims (`eng_wan_ti2v.py:944` -> `:959`), so the module constants
  are only defaults for dimension-omitting callers. Align them to the declaration for
  consistency, in one line, with a comment saying they are defaults only.
- **Canvas is not the biggest low-VRAM lever.** The text encoder (3.861 GiB) is LARGER
  than the UNET (3.549 GiB); co-resident total 8.722 GiB
  (`docs/2026-07-31-wan-8gb-parameter-analysis.md:97-113`). `ltx_8gb` measured encoder
  placement as its decisive knob. Encoder residency is owed its own pass; it is not
  this build's job, but a "most efficient for low VRAM" answer that stops at canvas is
  incomplete and should be recorded as such.
- **Render-small-then-upscale is already the shipped path** and the operator's
  preference is satisfied: `rtx_upscale.py:63,69` documents 832x480 -> 1920x1080. But
  **no test covers the full peak across render + VAE decode + upscale**, and the
  upscaler performs separate chunked allocations. Owed before any "the smaller canvas
  preserves the saving" claim.

### Q2 -- The question was mis-framed. Ping-pong is driven by the TIER CEILING, not the VRAM predictor.

`_floor_length` does not choose a shorter affordable render.
`compute_real_frame_budget` (`motion_common.py:321-363`) returns the static snapped
target or **raises `MotionBudgetError`**; its docstring is explicit that the
shrink-toward-floor behaviour was deliberately removed on 2026-07-10 ("the render NEVER
resizes itself"). The only thing that shortens a render is the hard cap:
`OTR_WAN_TI2V_MAX_FRAMES` -> the tier profile's `video.max_render_frames` -> the engine
max (`eng_wan_ti2v.py:724-733`).

So "does a 3-step engine still want to render short and mirror?" is really **"what
should FastWan's tier ceiling be?"**

**RULING.**
1. **FastWan gets its OWN ceiling channel.** Inheriting `OTR_WAN_TI2V_MAX_FRAMES` means
   the WAN tier's ceiling silently caps FastWan and FastWan has no key of its own. Add
   `OTR_FASTWAN_8GB_MAX_FRAMES` and its own profile row, and parameterize the env key
   and the diagnostic engine name (the refusal text hardcodes "wan_ti2v" at `:778`,
   `:795`, so FastWan would today refuse under the wrong engine's name).
2. **Render native at or below the highest CANONICALLY proven rung; ping-pong only
   above it.** FastWan being ~2.8x faster per clip is an argument for raising the
   ceiling, and the ceiling is exactly where that argument belongs.
3. **Do NOT promote the 81-frame bench result into the production cap.** A bench cell
   never qualifies an engine (CLAUDE.md s0A). Land the adapter under a conservative cap,
   then raise it only after a canonical render proves that rung green -- two separate
   chunks.

### Q3 -- The mechanical half is settled. The stability half does not block, but its gate is decoded-frame hashes.

**Mechanical.** `SamplerCustom` takes `noise_seed`, confirmed against the working graph
(`arm_c_fastwan_lora_gguf.json` node `9`: `{"add_noise": true, "noise_seed": 42,
"sampler": ["9a",0], "sigmas": ["9b",0]}`). The incumbent wires `"seed"` into `KSampler`
(`eng_wan_ti2v.py:866`). Wire `"noise_seed": int(plan.get("seed", 0))` and pin it with
a test asserting the built graph carries `noise_seed == plan["seed"]` and no `seed` key
on the sampler node.

**Stability.** Codex grounds seed-stability in `default_noise_sampler` producing a
seeded generator whose successive calls give fresh deterministic draws. I did not read
that file on this box, so it stands as a **hypothesis, not a finding**.

**RULING: it does not block the long-haul run, but it gates qualification.** The gate
is two cold canonical renders producing identical **decoded-frame hashes** -- not
mp4 bytes, which carry encoder nondeterminism and would make the test flaky (this is
why antigravity's "byte-identical output" formulation is rejected in favour of codex's).
Add: same-seed replay equality, different-seed divergence, and distinct successive
re-noise draws.

---

## MUST-FIX, CONSOLIDATED -- what the r1 handoff did not have

1. **`_vaedecode_inputs` hardcodes the sampler node id.** `eng_wan_ti2v.py:884` builds
   `{"samples": W("ksampler", 0), ...}`. A subclass whose `_build_graph` renames that
   node to `sampler_custom` wires the decode to a node that is not in the graph.
   Smallest fix: **keep the logical node id `"ksampler"`** for FastWan's `SamplerCustom`
   node so the inherited decode wiring stays correct, or override `_vaedecode_inputs`.
   Pick one and pin it with a graph-shape test.
2. **The recipe cannot be overridden by a class attribute.** `_tiled_vae`
   (`:534-540`), `_resolve_render_config` (`:618-640`), `_negative_prompt`,
   `_tile_geometry` and `_recipe_departures` read MODULE-level `WAN_TI2V_RECIPE`,
   `PREQUALIFICATION_ENV` and `_RECIPE_ENV_KEYS`. Declaring `RECIPE_FASTWAN_8GB` on the
   subclass has no effect. Either refactor those accessors to read class-level
   definitions (preferred -- one seam, behaviour-preserving for `wan_ti2v`) or override
   every one of them. **The r1 override table is incomplete on this point and must be
   rewritten before code.**
3. **The LoRA patcher is untracked and will leak.** `render_clip` tracks only
   `results.get("unet", ...)` (`:985-988`) and `keep={"unet","vae",self._TERMINAL}`
   (`:978`); `prepare` builds a loader-only graph with one logical result. Override or
   parameterize `prepare` so the session graph is base loader -> `LoraLoaderModelOnly`
   -> exposed patched result; register BOTH through `on_result`; retain both in
   `prepared["patchers"]`; extend the non-session path too. Test teardown after success
   AND after a failure between the two loads.
4. **`ManualSigmas` takes a STRING.** The working graph uses
   `{"sigmas": "1.0, 0.757, 0.522, 0.0"}`. The plan's Python tuple is not the wire
   format. Serialize canonically, and validate descending values, terminal zero, and
   `steps == len(sigmas) - 1`. **Also add `shift = 5.0`** -- it is in the recipe and
   absent from the override table.
5. **Three telemetry fields cannot reach the ledger at all.** `_clip_from_raw`
   (`wan_shared.py:533-573`) returns `vram_peak_mb`, `recipe`, `native_frame_count`,
   `extension_mode` -- and nothing else. `recipe` already flows (threaded 2026-07-27;
   it reads null on the 2026-07-23 leg only because that render predates it). So the
   operator-authorized backfill needs **`quant`, `use_lora`, `render_canvas`** added to
   `_clip_from_raw`, which is SHARED by both WAN adapters. Types: `quant: str`,
   `use_lora: str|null` (the filename, never a bool), `render_canvas: str` as
   `"832x480"`. Round-trip assertions against the durable ledger for both engines.
6. **Node-key collision.** The bench helper already owns `OTR_DMDRestartSamplerSelect`
   in its own `NODE_CLASS_MAPPINGS` (`otr_bakeoff_helper/__init__.py:397-413`). Moving
   the sampler into the pack means **deleting** the bench helper's class and mapping
   entry, not importing and re-registering it -- otherwise two packs claim one key.
   Give the production node a real category, not `"OTR/bakeoff"`.
7. **Registration is explicit.** Adapters carry `@register` (`eng_wan_ti2v.py:234`,
   `eng_ltx_8gb.py:483`); the DMD node needs a root `_NODE_MODULES` entry. Plus the
   `CAPABILITIES` row, or `audit_engine_roster` reports `unexpected` and CI fails.
8. **Fail preflight on identity, not on a hardcoded token.** The base adapter permits
   loader-mode and filename overrides, so a fixed `"Q5_K_M"` receipt could describe a
   differently named model. Refuse unless base, encoder, VAE, LoRA, transition impl,
   hashes and loader modes all match the frozen recipe. **`commercial_clean` is
   inherited `True` while the Kijai extraction's licence chain is unresolved** -- resolve
   it before registration or mark FastWan gated.
9. **The sampler fallback becomes a named refusal.** `otr_bakeoff_helper:326-370` logs
   `noise_scaling=FALLBACK-flow` and proceeds. Validate `model_sampling.noise_scaling`
   once before the loop; raise a defined error class with the original exception
   chained. Carry the same refusal into the bench copy.

## CUTS

- **The `pass`-body skeleton step** (r1 BUILD STEP 1). Its purpose was to discover the
  override surface; MUST-FIX 2 and 3 have now discovered it from the code. Building a
  registered-but-incomplete adapter only risks a selectable stub.
- **Any recalibration of `_FRAME_COST_REF_PIXELS`,** and the "mandatory second canvas
  channel" framing. Demoted to a one-line default alignment.
- **Retained against codex's cut:** the `session_identity` LoRA override stays.
  `_aux_loader_files` covers the file, but `lora_strength` is a recipe knob that is not
  a file, and a strength change must still open a new beat session.

## STILL OWED BEFORE THE LONG-HAUL RUN

1. The canonical-workflow gate: a render through the registered adapter with
   `RESULT SUCCESS` + `obs_publish OK`. Not a bench cell.
2. Decompose the ~9.8 GB machine-wide live figure into this render's share versus other
   resident state, clamped. Until then "A and C are VRAM-identical" transfers to
   production only as a claim about bench deltas.
3. Same-seed decoded-frame-hash replay (Q3).
4. Full-pipeline peak across render + decode + upscale (Q1).
5. FastVideo revision + sigma digest pinned in a durable manifest, and the exact licence
   chain with its notice-compliance gap recorded rather than summarized as "verified".
6. Repeated-session teardown with no VRAM or host-RAM growth across clips, including
   switching between `wan_ti2v` and `fastwan_8gb` in one session. Host RAM is uncovered
   by any model: FastWan measured 14.7-15.2 GB against the incumbent's 10.5-11.1 GB.


---

# r2 JUDGMENT -- grounded claim ledger

Driver/judge: Claude (Cowork). Panel: codex `gpt-5.6-sol` (high), antigravity
`Gemini 3.5 Flash (High)`. Anchor written before fan-out; the canvas probe in its
addendum was run during fan-out and before any agent file was opened.

Every verdict below was checked against the real Windows files.

| # | Claim | Seat | Verdict |
|---|---|---|---|
| 1 | `compute_real_frame_budget` never shortens -- it returns the static snapped target or RAISES `MotionBudgetError` | codex 1 | **CONFIRMED** `motion_common.py:321-363`. Docstring: "NEVER a VRAM-adaptive resize"; the shrink-toward-floor behaviour was deliberately killed 2026-07-10 (S4) |
| 2 | Production passes real width/height, so `_TI2V_COST_REF_W/H` are defaults for dimension-omitting callers only | codex 1 | **CONFIRMED** `eng_wan_ti2v.py:944` (`self._dims(request)`) -> `:959-960` |
| 3 | FastWan inheriting `_floor_length` prices frames at 3.07x and forces premature mirror-filling | agy 2, **driver anchor MF3** | **REFUTED** by 1 + 2. Never reached in production; and the failure mode would be a raise, not a short render. My own anchor was wrong here and codex caught it |
| 4 | `_vaedecode_inputs` hardcodes `W("ksampler", 0)`, so a subclass that renames the sampler node crashes | agy 3 | **CONFIRMED** `eng_wan_ti2v.py:884`. **Antigravity's best find; codex missed it entirely** |
| 5 | Recipe accessors read MODULE-level `WAN_TI2V_RECIPE` / `PREQUALIFICATION_ENV` / `_RECIPE_ENV_KEYS`, not subclass attributes | codex 3 | **CONFIRMED** `eng_wan_ti2v.py:534-540` (`_tiled_vae`), `:618-640` (`_resolve_render_config`). A class-attribute recipe cannot take effect |
| 6 | `_clip_from_raw` cannot carry `quant` / `use_lora` / `render_canvas` to the ledger | codex 5, agy 5 | **CONFIRMED with refinement** `wan_shared.py:533-573` returns `vram_peak_mb`, `recipe`, `native_frame_count`, `extension_mode`. `recipe` IS threaded (2026-07-27) -- it reads null on the 2026-07-23 leg only because that render predates the threading. **Three fields need new plumbing, not four** |
| 7 | `prepare` / `render_clip` track only the `"unet"` result, so a LoRA patcher is untracked and leaks | codex 2, agy 1 | **CONFIRMED** `eng_wan_ti2v.py:985-988` (`results.get("unet", ...)`), `keep={"unet","vae",self._TERMINAL}` at `:978` |
| 8 | FastWan inherits `OTR_WAN_TI2V_MAX_FRAMES` and emits `wan_ti2v`-named diagnostics | codex 6 | **CONFIRMED** `eng_wan_ti2v.py:724-733`; error text hardcodes "wan_ti2v" at `:778`, `:795` |
| 9 | Adapters need an explicit `@register`; the bench helper already owns the `OTR_DMDRestartSamplerSelect` key | codex 7 | **CONFIRMED** `eng_wan_ti2v.py:234`, `eng_ltx_8gb.py:483`; bench helper `__init__.py:397-413` |
| 10 | `ManualSigmas` takes a comma-separated STRING, not a tuple | codex 4 | **CONFIRMED against the working graph** -- `arm_c_fastwan_lora_gguf.json` node `9b`: `{"sigmas": "1.0, 0.757, 0.522, 0.0"}`. The plan's Python tuple is not the wire format |
| 11 | `SamplerCustom` consumes `noise_seed`, not `seed` | agy 4, codex 8, plan s7 | **CONFIRMED** bench node `9`: `{"add_noise": true, "noise_seed": 42, "sampler": ["9a",0], "sigmas": ["9b",0]}` |
| 12 | `default_noise_sampler` gives seeded, deterministic successive draws, so the restart transition is expected seed-stable | codex 8 | **PLAUSIBLE, not verified on this box.** Cited from the ComfyUI install; I did not read that file. Treated as a hypothesis that a replay test must confirm, per the skill's UNVERIFIABLE rule |
| 13 | A `CAPABILITIES` row is required or the roster audit fails CI | agy 7 | **CONFIRMED** (already grounded r1) `registry.py:575-609` |
| 14 | Changing `wan_ti2v`'s canvas invalidates its shipped VRAM profile and needs re-pinning | agy S1 | **CONFIRMED as a risk, but MOOT** -- the measured canvas already IS 832x480, so declaring it changes no pixels. See ledger row 16 |
| 15 | `_FRAME_COST_REF_PIXELS` must stay `1472*832` because it is the calibration reference | codex CUT 3 | **CONFIRMED** `motion_common.py:253-262` documents it as the telemetry reference the cost row was fitted at. Rewriting it silently rescales every prediction |
| 16 | The incumbent renders production at 1472x832; that explains the live-vs-bench VRAM gap | **plan s3, handoff s1, r1 F3** | **REFUTED BY MEASUREMENT (driver).** All 7 surviving clips of the 2026-07-23 leg ffprobe at **832x480**. The gap is a measurement-scope change: `2b095143` (2026-07-20) threaded the true render-window NVML peak; all six low readings predate it, the single high one postdates it. `VramPeakProbe` is also MACHINE-WIDE while the bench figure is a delta |
| 17 | Cut the `pass`-body skeleton step -- it risks registering an incomplete selectable adapter | codex CUT 1 | **ACCEPTED.** Rows 5 and 7 mean the override surface is now known from the code; the skeleton would only re-derive it |
| 18 | Cut the separate `session_identity` override if `_aux_loader_files` includes the LoRA | codex CUT 2 | **REJECTED.** `wan_shared.py:332` builds the receipt from loader names + aux assets, but `session_identity` gates BEAT-SESSION reuse. `lora_strength` is a recipe knob that is not a file, so a strength change must still open a new session. Keep the override, narrowed to strength |

## Panel performance

- **codex** carried the round. Rows 1, 2, 5, 8, 9, 10 are all real and all missed by
  both the plan and my anchor. Row 1 refuted my own MUST-FIX 3.
- **antigravity** contributed one finding neither codex nor I had (row 4, a genuine
  crash) and one confirmed telemetry gap (row 6). Its row 3 repeated my anchor's error.
- **driver** contributed row 16, which refutes the plan's own headline and r1's
  "finding of the round", and rows 6/18's refinements.

Net: 2 driver claims refuted (rows 3, 16 -- one of them my own), 1 panel claim
downgraded to hypothesis (row 12), 1 panel cut rejected (row 18).
