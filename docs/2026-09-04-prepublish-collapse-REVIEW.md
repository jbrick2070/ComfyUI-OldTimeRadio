# Pre-publish review: the registry scan collapse, as shipped

**PURPOSE: this is about to be PUBLISHED to the Comfy Registry as a new alpha.**
A published version string is burned permanently -- version-delete is a soft
delete that never frees the string. So this is the last cheap moment to find a
defect. Review the SHIPPED code, not a plan.

## What was done

Every machine fact got ONE owner, to collapse a Comfy-Registry security scan from
158 findings. Six commits, `4a8e063d`..`17079b7f`, all on `v2.0-alpha`:

* `nodes/_otr_shared/env.py` -- the ONLY module that may spell `os.environ`.
  `get/pin/setdefault/unpin/snapshot`. Reads are LIVE, never cached. It is a
  SPELLING, not a schema: a caller's default, cast and precedence stay at the
  call site.
* `nodes/_otr_shared/proc.py` -- the ONLY module that may spawn. `run()/popen()`,
  every kwarg forwarded, real subprocess objects returned, no exception wrapped,
  IDENTITY re-exports of PIPE/DEVNULL/STDOUT/Popen/CompletedProcess/
  CalledProcessError/TimeoutExpired, plus a basename allowlist: exact
  {git, nvidia-smi, blender} and prefixes {ffmpeg, ffprobe, python}.
* ~100 files migrated onto them. Three AST guards fail the build if a new
  `os.environ` or `subprocess` site appears under `nodes/`.

Measured result: files spelling `os.environ` 103 -> 4; subprocess spawn SITES
35 in 20 files -> 3 in 2 files; `kernel32.OpenProcess`, `__import__("sys")` and a
whole-file `Path.read_bytes()` sha256 all gone.

TWO FILES ARE DELIBERATELY NOT MIGRATED, each with a recorded reason:
`eng_indextts2.py` (byte-hashed by a voice-qualification fingerprint -- editing
any byte demotes a voice the operator approved by ear) and
`_otr_writer_heartbeat.py` (a leaf whose test forbids ANY pack import, because a
pack import reintroduces a cycle that once left two generate transports blind).

## What I want you to hunt, in priority order

1. **ANY behaviour change.** The migration claims to be semantics-neutral. Find a
   site where the migrated call does NOT mean what the original meant: a lost
   default, a changed cast, a dropped `.strip()`, an altered `or` chain, changed
   argv or kwargs, a read that was live and is now a snapshot (or vice versa).
2. **THE SPAWN ALLOWLIST REFUSING A REAL BINARY AT RENDER TIME.** This already bit
   once: `imageio_ffmpeg` ships `ffmpeg-win-x86_64-v7.1.exe`, which the exact-match
   allowlist refused, and the caller swallowed it and degraded SILENTLY. Trace
   every migrated spawn's argv[0] and find another one that cannot pass.
3. **Anything that would EMBARRASS US IN A PUBLISHED VERSION** -- a leftover debug
   path, a swallowed error that hides a real failure, an import that only works on
   this box, a test seam left pointing at a module that no longer has it.
4. **The two owners' own contracts.** Can `env.py` return something
   `os.environ.get` would not? Can `proc.py` change what a caller receives? Is any
   re-export not an identity alias?
5. **The guards.** Can a new `os.environ` or `subprocess` site get past the AST
   ratchets in `tests/test_env_single_owner.py` and
   `tests/test_process_single_owner.py`? Aliasing, importlib, getattr, star-import,
   a rebound module name.

Ground every claim in the real files. Say CONFIRMED / MISREAD / UNVERIFIABLE.
Suite is 13573 passed / 126 skipped / 1 xfailed, so a claim that something is
broken needs to explain why the suite is green.

## Diffstat of the shipped migration

```
 __init__.py                                        | 34 +++++++++++--
 nodes/OTR_LedgerScriptWriter.py                    | 26 +++++-----
 nodes/_otr_audio_engines/_otr_sidecar.py           |  6 ++-
 nodes/_otr_audio_engines/base.py                   |  7 ++-
 nodes/_otr_audio_engines/eng_chatterbox.py         | 18 +++++--
 nodes/_otr_audio_engines/eng_cloud_elevenlabs.py   | 12 +++--
 nodes/_otr_audio_engines/eng_cloud_sonilo.py       |  8 ++-
 nodes/_otr_audio_engines/eng_dia.py                | 22 +++++---
 nodes/_otr_audio_engines/eng_google_lyria.py       | 21 +++++---
 nodes/_otr_audio_engines/eng_google_tts.py         | 18 ++++---
 nodes/_otr_audio_engines/eng_kokoro.py             | 11 ++--
 nodes/_otr_audio_engines/eng_musicgen.py           |  7 ++-
 nodes/_otr_audio_engines/eng_stable_audio.py       |  7 ++-
 nodes/_otr_audio_engines/eng_stable_audio_3.py     | 22 ++++----
 nodes/_otr_banana_route.py                         |  8 ++-
 nodes/_otr_bark_lib.py                             | 12 +++--
 nodes/_otr_cast_env.py                             | 10 ++--
 nodes/_otr_comfy_backend.py                        | 16 +++---
 nodes/_otr_config.py                               |  7 ++-
 nodes/_otr_determinism.py                          |  8 ++-
 nodes/_otr_engine_profiles.py                      |  7 ++-
 nodes/_otr_freeze_cascade.py                       |  8 ++-
 nodes/_otr_gguf_backend.py                         | 23 +++++----
 nodes/_otr_google_api/client.py                    | 18 ++++---
 nodes/_otr_google_api/models.py                    |  8 ++-
 nodes/_otr_hf_auth.py                              |  8 ++-
 nodes/_otr_hf_env.py                               | 12 +++--
 nodes/_otr_image_engines/eng_cloud_image.py        | 24 +++++----
 nodes/_otr_image_engines/eng_google_image.py       | 16 +++---
 nodes/_otr_image_engines/flux2_klein.py            | 17 ++++---
 nodes/_otr_image_engines/flux_gen1.py              | 18 ++++---
 nodes/_otr_image_engines/hidream_i1.py             |  7 ++-
 nodes/_otr_image_engines/ideogram4_local.py        |  7 ++-
 nodes/_otr_image_engines/lumina_image.py           | 25 +++++----
 nodes/_otr_image_engines/sd35_large.py             |  7 ++-
 nodes/_otr_image_engines/z_image_turbo.py          | 27 ++++++----
 nodes/_otr_janitor.py                              |  8 ++-
 nodes/_otr_kokoro_voice_prefetch.py                | 15 ++++--
 nodes/_otr_ledger.py                               | 16 ++++--
 nodes/_otr_media_archive_sources.py                | 12 +++--
 nodes/_otr_model_catalog.py                        | 34 +++++++------
 nodes/_otr_model_loader.py                         |  7 ++-
 nodes/_otr_openrouter_backend.py                   | 13 +++--
 nodes/_otr_original_radio.py                       |  8 ++-
 nodes/_otr_paths.py                                | 20 +++++---
 nodes/_otr_public_domain_sources.py                |  7 ++-
 nodes/_otr_rolls.py                                | 11 +++-
 nodes/_otr_scifi_news_pro.py                       |  8 ++-
 nodes/_otr_shared/cloud_media_backend.py           | 19 ++++---
 nodes/_otr_shared/cloud_media_canonical.py         | 29 ++++++++---
 nodes/_otr_shared/cloud_model_ids.py               | 10 +++-
 nodes/_otr_shared/encode_sink.py                   | 21 +++++---
 nodes/_otr_shared/ffmpeg.py                        | 10 +++-
 nodes/_otr_shared/ffprobe.py                       | 27 +++++++---
 nodes/_otr_shared/gpu_residency.py                 | 59 ++++++++++++++++------
 nodes/_otr_shared/hf_token.py                      | 15 ++++--
 nodes/_otr_shared/proc.py                          | 47 +++++++++++++----
 nodes/_otr_shared/route_freeze.py                  | 14 ++++-
 nodes/_otr_shared/scope_draw.py                    | 17 +++++--
 nodes/_otr_source_snapshot.py                      |  7 ++-
 nodes/_otr_sys_specs.py                            | 14 +++--
 nodes/_otr_upscale_engines/eng_spandrel_esrgan.py  | 23 ++++++++-
 nodes/_otr_video_engines/_tmp.py                   |  6 ++-
 nodes/_otr_video_engines/eng_cloud_video.py        | 29 ++++++-----
 nodes/_otr_video_engines/eng_fastwan_8gb.py        |  8 ++-
 nodes/_otr_video_engines/eng_ghost_signal.py       |  7 ++-
 .../eng_ghost_signal_official.py                   |  8 ++-
 .../eng_ghost_signal_stillin_lab.py                |  7 ++-
 nodes/_otr_video_engines/eng_google_omni_video.py  | 12 +++--
 nodes/_otr_video_engines/eng_google_veo_video.py   |  8 ++-
 nodes/_otr_video_engines/eng_humo.py               | 47 ++++++++++-------
 nodes/_otr_video_engines/eng_ltx25.py              | 21 +++++---
 nodes/_otr_video_engines/eng_ltx_8gb.py            | 35 +++++++------
 nodes/_otr_video_engines/eng_ltx_av.py             | 41 ++++++++-------
 nodes/_otr_video_engines/eng_ltx_video.py          | 49 ++++++++++--------
 nodes/_otr_video_engines/eng_mesh_stage.py         | 54 ++++++++++++--------
 nodes/_otr_video_engines/eng_minimax_h3.py         |  9 +++-
 nodes/_otr_video_engines/eng_visualizer.py         |  9 +++-
 nodes/_otr_video_engines/eng_viz_camera.py         |  9 +++-
 nodes/_otr_video_engines/eng_viz_mandala.py        |  9 +++-
 nodes/_otr_video_engines/eng_viz_rainbow.py        |  9 +++-
 nodes/_otr_video_engines/eng_wan_ti2v.py           | 23 +++++----
 nodes/_otr_video_engines/ghost_signal_author.py    | 11 ++--
 nodes/_otr_video_engines/motion_common.py          | 20 +++++---
 nodes/_otr_video_engines/render_driver.py          | 34 ++++++++-----
 nodes/_otr_video_engines/wan_recipe.py             | 18 ++++---
 nodes/_otr_video_engines/wan_shared.py             | 29 ++++++++---
 nodes/_otr_video_engines/wrapper_bridge.py         | 22 +++++---
 nodes/_otr_voice_bank.py                           | 11 ++--
 nodes/_otr_voice_node_common.py                    | 11 ++--
 nodes/_otr_workflow_validator.py                   | 10 ++--
 nodes/_otr_writer_inputs.py                        | 10 ++--
 nodes/_otr_writer_tail.py                          | 10 ++--
 nodes/_otr_writer_vram.py                          |  8 ++-
 nodes/cast_lock.py                                 |  7 ++-
 nodes/otr_caption_burn.py                          | 14 +++--
 nodes/otr_credits_roll.py                          | 22 +++++---
 nodes/otr_image_gen_dispatcher.py                  | 13 +++--
 nodes/otr_master_audio_mux.py                      | 20 +++++---
 nodes/otr_meta_brief_image_prompt.py               | 11 ++--
 nodes/otr_post_upscale_procgen_blend.py            | 24 +++++----
 nodes/otr_shot_lock.py                             | 15 +++---
 nodes/otr_silent_composite.py                      | 26 ++++++----
 nodes/otr_video_render_batch.py                    |  8 ++-
 nodes/production_ledger.py                         | 21 ++++++--
 nodes/scene_sequencer.py                           | 14 +++--
 nodes/video_engine.py                              | 24 ++++++---
 107 files changed, 1232 insertions(+), 564 deletions(-)

```
