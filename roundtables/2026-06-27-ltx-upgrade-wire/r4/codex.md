VERDICT: yes-with-fixes. The plan is close, but the VRAM peak is not actually wired to the smoke/report path, decode invalid-env behavior is ambiguous, and the alpha test is worded against an impossible post-overlay alpha state.

MUST-FIX BEFORE BUILD:
1. [Smoke / VRAM verify] Defect: the plan says “Record recipe + decode knobs + NVML render-phase peak,” but the current report path records `ep["vram_peak_mb"]` from `render_driver.run_episode`, which is fed by `render_shot` returning `_mc.vram_used_mb()` after render, not the `VramPeakProbe` peak. See `nodes/_otr_video_engines/render_driver.py:1496`, `nodes/_otr_video_engines/render_driver.py:1619-1627`, `nodes/otr_video_render_batch.py:215-223`, and `nodes/_otr_video_engines/eng_ltx_av.py:621-634`. Concrete fix: return `vram_peak_mb` from `eng_ltx_av.render_clip`, preserve it through `_clip_from_raw`, and have `render_shot` prefer `clip["vram_peak_mb"]` over post-render `_mc.vram_used_mb()`.

2. [Decode env] Defect: “fail-loud/clamp” is a build-blocking ambiguity; those produce incompatible outputs. Current `_build_graph` hard-codes valid values at `nodes/_otr_video_engines/eng_ltx_av.py:556-559`, so the new parser must define exactly one invalid-env behavior. Concrete fix: absent env defaults to 128/32; present but non-int, <=0, or `overlap >= size` raises a named error before building the `VAEDecodeTiled` node. Do not clamp.

3. [Tests: Alpha] Defect: the test says edges stay transparent after `format=rgba->scale->unsharp->overlay`, but the real graph flattens through `overlay` and `format=yuv420p` at `nodes/otr_silent_composite.py:455-457`; the output cannot retain alpha. Concrete fix: test source-over math over a contrasting plate: semi-transparent edge pixels in the flattened output must show the expected blended background contribution, not an alpha channel.

4. [VRAM verify] Defect: the cleanup wording can be implemented with `results`/`images` undefined if `_wb.run_graph` or `results[self._TERMINAL][0]` raises. Current code only has valid `results` after `nodes/_otr_video_engines/eng_ltx_av.py:621-623`. Concrete fix: initialize `results = images = frames = path = None`; stop the probe in `finally`; run `_retain_model_patchers` only when `results` exists; encode only when `images` exists; always call `reclaim_idle_models`; assert peak only after cleanup.

SHOULD-FIX:
1. [VRAM verify] Use `_MC.VramPeakProbe(interval_s=0.1)` to match the existing WAN peak probe pattern at `nodes/_otr_video_engines/eng_wan_ti2v.py:455-461`; default 1.0s in `motion_common.py:255` is coarser than the “real peak” claim.

2. [Smoke] Update the smoke script header and constants when generalizing `_preflight_distilled_native_graph`; it currently documents and enforces distilled-native-specific setup at `scripts/run_otr_30word_smoke.py:9-13`, `scripts/run_otr_30word_smoke.py:38-52`, and `scripts/run_otr_30word_smoke.py:196-245`.

3. [Smoke] The plan says keep the LTX enablement check, but current smoke requires `OTR_ENABLE_LTX_AV == "1"` at `scripts/run_otr_30word_smoke.py:202-203`, while engine usability treats unset as enabled and only disables on `"0"` at `nodes/_otr_video_engines/eng_ltx_av.py:338-340`. [ASSUMPTION] If the smoke launcher always sets `OTR_ENABLE_LTX_AV=1`, this is not blocking; otherwise accept unset or state the launcher requirement.

OPTIONAL / NICE-TO-HAVE:
- Add recipe and decode knobs to the engine log line near `nodes/_otr_video_engines/eng_ltx_av.py:580-584`, so logs self-identify the exact graph even if the smoke report path changes.

CUT THESE:
1. [Decode env] Cut “clamp” as an option. It adds a second runtime behavior and can hide a bad operator env; fail-loud covers the goal.
2. [Unchanged] Keep companion-drift manifest hardening cut; this plan does not change the manifest schema.

VERIFY-AT-BUILD checklist:
1. FFmpeg `unsharp` exists on the resolved `fb` from `_ffmpeg_bin`, not a hardcoded `ffmpeg`; sharpen paths only. See `nodes/otr_silent_composite.py:79-81` and `nodes/otr_silent_composite.py:640-647`.
2. Semi-transparent RGBA foreground over a contrasting still plate blends correctly after `format=rgba -> scale -> unsharp -> overlay -> yuv420p`; verify composited pixels, not output alpha.
3. Built LTX graph defaults to spatial 512/64 and temporal 128/32, and env overrides are read at `_build_graph` runtime without module reload.
4. Invalid decode env values fail before `VAEDecodeTiled`.
5. Smoke report records actual recipe, decode knobs, and render-phase NVML peak propagated from `VramPeakProbe`, and peak is checked against `_MC.dynamic_vram_ceiling_mb()`.
6. Canonical workflow JSON remains unchanged unless a widget/input is added; if changed, re-run `OTR_WorkflowValidator`, JSON round-trip, link/widget audit. Link 246 currently wires `OTR_SignalLostVideo` to `OTR_SilentComposite.base_video_path`.
7. Real canonical-workflow smoke writes an asset under the canonical episode/output tree and confirms it exists.
8. Full regression suite, Bug Bible, AST/no-BOM/no-0-byte checks, commit + push to `v2.0-alpha`, and verify HEAD == origin.