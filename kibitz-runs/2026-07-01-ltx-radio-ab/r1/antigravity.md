VERDICT: yes-with-fixes. The A/B toggle logic is clean, but image phase minting and aspect resolution must be aligned with render-time engine expectations to prevent missing or pillarboxed assets.

MUST-FIX BEFORE BUILD:
1. [Design] Defect: Decoupling of image generation and rendering breaks face still availability. The prompt generator ([nodes/otr_meta_brief_image_prompt.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py)) only mints `radio_host_portrait` if `_humo_hosts_enabled()` is True. If `OTR_ENABLE_HUMO_HOSTS=0` (default) but the operator runs `OTR_LTX_RADIO_FACE=1` during the render phase, the face still is not in the ledger, causing lookup to fail.
   - Fix: Modify `nodes/otr_meta_brief_image_prompt.py::_humo_hosts_enabled` to check both environment variables: `return os.environ.get("OTR_ENABLE_HUMO_HOSTS", "0") == "1" or os.environ.get("OTR_LTX_RADIO_FACE", "0") == "1"`.
2. [Design / Aspect] Defect [ASSUMPTION]: Aspect ratio mismatch when reusing the same ledger across sweeps. If a ledger is generated with `OTR_ENABLE_HUMO_HOSTS=1`, the face still is portrait. Running `OTR_LTX_RADIO_FACE=1` (which dispatches the wide `ltx_audio_in` engine) on this ledger will feed a portrait still to a wide engine, causing pillarboxing.
   - Fix: Document in the A/B protocol section of the spec that comparing portrait HuMo hosts vs wide LTX face hosts requires separate image generation runs to correctly resolve the still's aspect ratio at mint time.
3. [Design / render_driver.py] Defect: Hard crash when the face still is missing. If `OTR_LTX_RADIO_FACE=1` is active but the face still is absent, `build_request_from_shot` in [nodes/_otr_video_engines/render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py) will resolve `init_image` to `""`, causing `ltx_audio_in` (which is `_is_talk=True` in [nodes/_otr_video_engines/eng_ltx_av.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py)) to fail with a hard `GraphExecutionError` mid-render.
   - Fix: In `nodes/_otr_video_engines/render_driver.py`, if `OTR_LTX_RADIO_FACE` is active but the face still is missing, log a LOUD warning and fall back to the scene still (`_still`) to satisfy `ltx_audio_in`'s init requirement and avoid rendering a black frame.

SHOULD-FIX:
1. [Design] Defect: Non-lip-sync disclosure documentation location is unspecified.
   - Fix: Add docstring clarification to the class docstring for `LtxAudioInEngine` in [nodes/_otr_video_engines/eng_ltx_av.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py) and in the project `README.md`.

OPTIONAL / NICE-TO-HAVE:
1. [Goal / Open questions] Run a single-frame smoke test of LTX motion on a wide face still to verify that LTX motion doesn't warp or smear face geometry unnaturally.

CUT THESE:
1. [Design] Granular toggles (per-episode or per-role). Keep it as a simple global environment variable `OTR_LTX_RADIO_FACE` as it is sufficient for the A/B sweep.
