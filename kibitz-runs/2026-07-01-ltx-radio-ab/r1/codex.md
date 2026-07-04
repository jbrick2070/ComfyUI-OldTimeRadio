VERDICT: no. The A/B story is plausible, but it is not build-ready until it defines the concrete routed engine, asset identity, and frozen-run protocol; otherwise the toggle can be dead or can compare more than the init still.

MUST-FIX BEFORE BUILD:
1. [Design] The toggle is dead unless the bookend engine is actually `ltx_audio_in`. The real workflow currently has node 87 `announcer_video_model` and `music_video_model` set to `"viz_green"`, not `ltx_audio_in` (`workflows/otr_scifi_16gb_full.json:1`). `ltx_audio_in` is only an engine/default-role capability in code (`nodes/_otr_video_engines/eng_ltx_av.py:16-20`, `tests/test_ltx_audio_in_engine.py:78-80`), not the saved workflow default. Concrete fix: make the A/B protocol explicitly patch or override announcer/music video slots to `ltx_audio_in` for both legs, and if that is a persistent widget/default change, update `workflows/otr_scifi_16gb_full.json` in the same change.

2. [Preconditions / Design] The plan assumes a `radio_host_portrait` still exists, but the current code path indexes portrait init images by `object_id`/`char_id` and `build_request_from_shot` starts from `char_id` lookup (`nodes/_otr_video_engines/render_driver.py:401-421`, `nodes/_otr_video_engines/render_driver.py:918-923`). Current main-feature code creates a synthetic announcer portrait keyed from announcer line `char_id`, normally `"announcer"` (`nodes/otr_meta_brief_image_prompt.py:437-449`, `nodes/otr_meta_brief_image_prompt.py:842-850`), not a visible `radio_host_portrait` object. [ASSUMPTION] If another window will add `radio_host_portrait`, the A/B doc must name that post-main contract. Concrete fix: define the exact ledger row contract before build: either `object_id="radio_host_portrait", kind="portrait"` and a dedicated lookup, or reuse `object_id="announcer"` deliberately; do not leave both names in play.

3. [Design / ASPECT] The plan says option (b) needs a wide face-radio still, but does not say how that wide still will be minted or located. Current `ltx_audio_in` unconditionally conditions wide engines on the beat scene still branch and clears portrait leakage (`nodes/_otr_video_engines/render_driver.py:1001-1024`). Concrete fix: specify a new `ltx_audio_in` bookend branch that, only when `OTR_LTX_RADIO_FACE=1` and role is announcer/music, resolves a wide `radio_host_portrait`/announcer portrait row instead of the scene still, and fail loud if the row is absent or not wide.

4. [A/B protocol] “Render the same baked episode/brief twice” is not strong enough to guarantee a clean A/B. If the face leg causes a new still to be minted or changes image-policy aspect, the comparison includes image-generation differences, not just `ltx_audio_in` init-image selection. Concrete fix: freeze one ledger/audio/story and ensure both candidate stills already exist in that same ledger before running video twice; then vary only `OTR_LTX_RADIO_FACE`.

5. [Design] The new `OTR_LTX_RADIO_FACE` toggle conflicts conceptually with the main feature’s HuMo-host switch. Current render policy hard-redirects HuMo-family engines on announcer/music to `ltx_audio_in` (`nodes/_otr_video_engines/render_driver.py:821-857`); the main plan names `OTR_ENABLE_HUMO_HOSTS` as the opt-in reversal (`docs/2026-07-01-brief-driven-radio-host/PLAN_HARDENED.md`). Concrete fix: declare the mode matrix: this A/B only applies when final routed engine is `ltx_audio_in`; if HuMo-hosts are enabled, `OTR_LTX_RADIO_FACE` is ignored or rejected loudly.

SHOULD-FIX:
1. [Goal / A/B protocol] “Pick the cooler default by eyeball” is under-specified. Concrete fix: state the decision rule: safe default remains faceless unless the face leg wins on readable radio identity, non-uncanny motion, and no false lip-sync expectation across at least the named S-F smoke brief.

2. [Design] Calling option (b) a “HuMo-style radio-FACE still” blurs provenance and runtime. The runtime engine is LTX, not HuMo; HuMo remains only the source/style of the still. Concrete fix: rename it “wide radio-host portrait still for LTX init” and reserve “HuMo” for actual `audio_driven_face` rendering.

3. [Invariants] “Default 0 = no change to current ltx look” is only true after `ltx_audio_in` is actually selected. Concrete fix: restate as “no change relative to an `ltx_audio_in` bookend render with the same ledger and same scene still.”

OPTIONAL / NICE-TO-HAVE:
- Add manifest stamping for each leg: routed engine, init image path, init kind, init dimensions, `OTR_LTX_RADIO_FACE`, and whether HuMo-host routing was disabled.
- Include one neutral/non-paranoid brief in addition to the tense S-F fixture so the default is not chosen from the wildcard’s best case only.

CUT THESE (scope / over-engineering):
1. [Open questions] Cut per-episode/per-role granularity for this stage. A global env toggle is enough for a two-leg A/B and avoids adding user-facing policy before the visual result is proven.
2. [Design] Cut any attempt to touch mesh, viz, HuMo routing, or character beats in this addendum. The necessary surface is one `ltx_audio_in` announcer/music init-image selection branch plus the frozen A/B harness.