VERDICT: yes-with-fixes. The plan is close, but visual_format precedence, format required_inputs, and line-to-character context still have build-blocking ambiguity.

MUST-FIX BEFORE BUILD:
1. [1c visual_format SEMANTICS] `direct()` comparing role widget values to registry defaults will misclassify profile-applied defaults as explicit picks. Profiles patch `OTR_VideoDirector` role widgets directly in `nodes/_otr_workflow_apply.py:433-479`, and `config/profiles/16gb_full.json` sets non-registry defaults such as `viz_green` and `humo_14B_169`. Concrete fix: define explicitness against the effective profile/default source, not registry default. Add a build test where `16gb_full + visual_format=evidence_board` overrides all inheritable profile role values, while an explicit per-role override still wins.

2. [1b/1c REGISTRATION CHECKLIST] “honest required_inputs” is ambiguous for format engines. If a builder declares `required_inputs=("format_ctx",)`, role compatibility will reject the engine because accepted tokens are only `text_prompt/init_image/audio_ref/base_clip_ref` in `nodes/_otr_shared/role_compat.py:67-69` and unknown tokens fail closed at `nodes/_otr_shared/role_compat.py:124-128`. Concrete fix: state `fmt_evidence_board.required_inputs = ()`, `fmt_tin_toy.required_inputs = ()`, and `FAMILY_REQUIRED_INPUTS["format_composite"] = ()`; `format_ctx` is a schema field checked by `assert_usable`, not a role-compat token.

3. [1b FORMAT CONTEXT] `lines[]{speaker,start_s,end_s,audio_path}` is under-specified for correct board crop selection. The real render path resolves canonical `char_id` from the shot/line at `nodes/_otr_video_engines/render_driver.py:991-995`; display speaker strings and aliases are not safe keys into `board_manifest.cast[].char_id`. Concrete fix: make `FormatContext.lines[]` include `line_id` or `beat_id` plus canonical `char_id`; engines must join mouths/crops by `char_id`, not speaker text.

4. [1b CACHE KEYS vs 3/F1-a] F1-a says “sepia/polaroid prompt tail,” but 1b says sepia/polaroid styling is local PIL post-processing on the raw portrait and must preserve `portrait_hash`. Those are incompatible. Concrete fix: remove the prompt-tail wording; cast polaroids are raw portrait images with local sepia/border treatment only.

SHOULD-FIX:
1. [Status] Header still says `r1-hardened` although sections 1b/1c include r2/r3 hardening. Update status to r4 convergence-ready to avoid handing stale state to the builder.

2. [3/F1 Acceptance, 4/F2 Acceptance] Name the actual smoke entrypoints, not “30w smoke script.” Repo contains `scripts/run_otr_30word_smoke.py`; use exact script names and require canonical asset existence checks after each run per `CLAUDE.md` section 6.

OPTIONAL / NICE-TO-HAVE:
- [5] Specify the checked-in fixture GLB path for V1, or add “verify: fixture GLB path exists before V1.”

CUT THESE:
None. The remaining probes are tied to prior unverifiable risks and are not obvious over-engineering.

VERIFY-AT-BUILD checklist:
- Confirm `format_composite` added to `nodes/_otr_video_engines/schemas.py` `FAMILIES` and `FAMILY_REQUIRED_INPUTS`, with schema round-trip.
- Confirm `FormatContext` is a `_Forbid` submodel on `VideoRequest`, and `build_request_from_shot` copies it into format requests.
- Confirm visual_format/profile/explicit precedence with tests covering `16gb_full` role overrides.
- Confirm `cloud_kling_lipsync` payload maps PATH mp4/wav to Comfy `VIDEO`/`AUDIO` for pinned fields `video`, `audio`, `voice_language` from `nodes/_otr_shared/partner_nodes.yaml:135-145`.
- V1: Kling lipsync accepts still-frame silent input clips for crop and full-frame tin-toy face.
- V2: chosen multiview-to-GLB row exports Blender-importable GLB at pin time.
- V3: 4K board composite paste/scale/rounding sanity.
- V4: Kling crop/paste lands within +/-2px and passes portrait face-similarity.
- Workflow JSON: `OTR_WorkflowValidator`, JSON round-trip, link integrity, widget-count vs live `INPUT_TYPES`, and append-only `widgets_values` audit on `workflows/otr_scifi_16gb_full.json`.
- Headless acceptance preflight: `OTR_ENABLE_COMFY_CLOUD_MEDIA`, credentials, budget guard, and Kling concurrency.