<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no — current architecture mixes incompatible role/input/fallback semantics in one adapter and lacks an explicit in-process stop rule.

MUST-FIX BEFORE BUILD:

1. [Q1/Q2/M1] files: `nodes/_otr_video_engines/eng_ltx_av.py`, `nodes/_otr_shared/role_compat.py`, `nodes/_otr_video_engines/schemas.py` — one `ltx_av` adapter spanning lip-sync and music-reactive motion is architecturally wrong with the current registry protocol. `role_compat.engine_fits_role()` requires static `required_inputs`; `MUSIC_VISUAL` currently supplies no `audio_ref`, while talking roles need `init_image` + `audio_ref` (`role_compat.py`). A single `required_inputs=("text_prompt","audio_ref")` both fails closed for music and under-specifies talking-head init-image requirements. Fix: split into two registered thin adapters sharing a private core:
   - `ltx_av_talk`: roles `("announcer_visual","character_video")`, family `audio_driven_face`, required_inputs `("text_prompt","audio_ref","init_image")`, fallback `humo`.
   - `ltx_av_music`: role `("music_visual",)`, family `audio_conditioned_video`, required_inputs `("text_prompt","audio_ref")`, fallback direct to `still_kenburns` unless another landscape-safe fallback is chosen.
   If the driver cannot populate `audio_ref` for music, drop `music_visual` from v1 instead of weakening required inputs. VERIFY-AT-BUILD: where `render_driver.py` attaches `audio_ref` to music requests.

2. [Q1/schemas] files: `nodes/_otr_video_engines/schemas.py`, `nodes/_otr_video_engines/registry.py` — `audio_conditioned_video` is not currently a valid family. `schemas.py` rejects unknown `family_hint` via `FAMILIES`, and the guard asserts `FAMILIES == FAMILY_REQUIRED_INPUTS`. `registry.py` docstring also enumerates allowed families. Fix: add `audio_conditioned_video` only if used, with `FAMILY_REQUIRED_INPUTS["audio_conditioned_video"] = ("text_prompt","audio_ref")`; update the registry docstring. Do not reuse `audio_driven_face` for `music_visual`; its existing schema requirement is `("audio_ref","init_image")`, and semantically it is face-specific.

3. [Q4/M0/M2] files: sprint plan, `nodes/_otr_video_engines/eng_ltx_av*.py` — isolation decision has no hard stop rule. Grounding shows in-process engines rely on cold-import cleanliness and lazy heavy imports (`eng_ltx_video.py`, `eng_humo.py`, `motion_common.py`), while `MotionEngineBase` is explicitly in-process. Fix: M0 must STOP the in-process lane if the IA2V path requires installing or changing Python packages in the main cu130 ComfyUI venv, imports model/audio packages at module scope, or depends on nodes absent from either installed Desktop or headless ComfyUI. At that point either close the lane or open a separate `sidecar_required` design; do not “temporarily” pip-install into cu130. VERIFY-AT-BUILD: actual latentsync sidecar interface if a cu128 sidecar is proposed.

4. [Q8/M2] files: `nodes/_otr_video_engines/eng_ltx_av*.py`, fallback tests — fallback `ltx_av -> humo` cannot be shared across all three roles. `fallback.py` only walks a single-linked `fallback_engine`; it has no role or aspect awareness. `eng_humo.py` registers roles only `announcer_visual` and `character_video`, native 480x832 portrait with pillarbox behavior, so it is not a valid `music_visual` fallback. Fix: with the split above, set `ltx_av_music.fallback_engine = "still_kenburns"` direct. For `ltx_av_talk`, `fallback_engine = "humo"` is acceptable only as a LOUD talking-head downgrade.

5. [Q8/tests] files: fallback-chain tests / plan — the documented chain is stale. Grounding shows actual HuMo fallback is `humo -> humo_1.7B -> latentsync -> still_kenburns`, not `humo -> latentsync -> still_kenburns` (`eng_humo.py`). Therefore `ltx_av_talk -> humo` resolves to five engines total: `ltx_av_talk -> humo -> humo_1.7B -> latentsync -> still_kenburns`. This is mechanically allowed because `fallback.py` has `DEFAULT_MAX_HOPS = 16`, but tests and acceptance greps must expect the real chain.

SHOULD-CONSIDER:

1. [Q4/adapter shape] files: `nodes/_otr_video_engines/motion_common.py`, `eng_ltx_av*.py` — do not add a lifecycle member for audio conditioning. Existing `VideoRequest.audio_ref`, `required_inputs`, and `render_clip(request, prepared)` are sufficient per `schemas.py` and `registry.py`. Keep audio-specific extraction local to the adapter, like HuMo’s `_ref_path()` pattern. Do not mutate `MotionEngineBase`.

2. [Q3/init_image] files: `eng_ltx_av_talk.py`, tests — make talking adapter fail before graph execution if `init_image` is absent. `schemas.py` enforces this automatically if `family_hint="audio_driven_face"`; also keep an adapter-side named error like HuMo’s `render_clip()` check.

3. [Q6/touch list] files: adapter package init / registration tests — additive touch list is incomplete until you verify how adapters are imported. Grounding shows adapters self-register on import via `@register`, but the package import file is not shown. VERIFY-AT-BUILD: whether `nodes/_otr_video_engines/__init__.py` or another loader must import `eng_ltx_av_talk` / `eng_ltx_av_music`.

4. [Q6/touch list] files: tests — required minimal tests: schema accepts `audio_conditioned_video`; role_compat offers `ltx_av_music` only after `MUSIC_VISUAL` supplies `audio_ref`; Director rejects incompatible picks at execute time; fallback chains terminate at `still_kenburns`; no existing engine defaults change.

5. [Yvann lane] Cut from this sprint.
   It is music-only and does nothing for announcer/character lip-sync.
   It adds a new custom-node dependency, triggering V-12/b7 review.
   It competes with the hard LTX-AV probe for time.
   It is not audio-conditioned generation, only audio-modulated scheduling.
   Revisit only if LTX-AV is closed as INERT and music visuals remain a separate goal.

CUT THESE:

1. [Q8] Cut `ltx_av -> humo` for `music_visual`. It is incompatible by role and changes from full-frame landscape to portrait pillarbox with no fallback-layer aspect guard.

2. [Q1] Cut the single all-role `ltx_av` adapter. Two thin adapters sharing one private graph/core are simpler than inventing role-dependent `required_inputs` semantics across registry, schemas, and Director.

3. [Q5] Cut Yvann-Nodes from M0-M5. It is a separate music-reactive feature, not an architecture dependency for LTX-AV.

OPEN-QUESTIONS:

1. [Q2] VERIFY-AT-BUILD: does `render_driver.py` already attach per-beat `audio_ref` to `music_visual` requests? If not, either add it deliberately with tests or drop `ltx_av_music` from v1.

2. [Q4] VERIFY-AT-BUILD: do installed Desktop and headless ComfyUI both expose the IA2V/reference-audio nodes needed for the selected graph? If not, which build is authoritative for M0?

3. [Q4] VERIFY-AT-BUILD: if sidecar is needed, is there an existing video sidecar protocol reusable from latentsync, or would this be new architecture?

4. [Q8] Decide whether `ltx_av_talk -> humo` is an acceptable LOUD aspect downgrade. Mechanically safe under `fallback.py`; visually a policy decision.

5. [Q6] VERIFY-AT-BUILD: exact additive registration/import file for new adapters; not shown in grounding.