Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
git add -- `
  nodes/_otr_shared/cloud_media_canonical.py `
  nodes/_otr_video_engines/eng_cloud_video.py `
  nodes/_otr_video_engines/eng_google_veo_video.py `
  nodes/_otr_video_engines/eng_google_omni_video.py `
  nodes/_otr_comfy_backend.py `
  tests/test_cloud_video_adapters.py `
  tests/test_comfy_slot_widgets.py `
  docs/PROD_BUG_LOG.md
git commit -m @"
fix(cloud): cap Vidu fps-resample surplus to the plan length

Cheap-cloud 1-act died at assembly (127 vs 125) after paying for Grok, Luma, and Vidu. canonicalize_video now keeps segment.render_frames after fps=25. Headless Comfy Credits reads OTR_COMFY_API_KEY when api_key_comfy_org is not injected. 5080 local path unchanged.
"@
git status -sb
