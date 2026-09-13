# The JSON matrix

Every profile that emits a saved graph, by machine class. `(canonical)`
in the voice column means the profile does not override it, so it splices
kokoro from the canonical -- which is what you want.

## 8 GB  --  18 profiles (18 machine, 0 lab/cloud)

| profile | kind | status | writer | quant | video lane | image | voice | music | packs | GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `8gb_lite` | machine | draft | gemma-4-E2B-it | none | still_motion | flux2_klein | bark | stable_audio_3 | GGUF | 18.2 |
| `otr_4060_12b_gguf_offload` | machine | shipping | gemma-4-E2B-it | none | animatediff15_v3_haunted_video | flux2_klein | kokoro | musicgen | AnimateDiff-Evolved,GGUF | 16.3 |
| `otr_4060_floor` | machine | draft | gemma-4-E2B-it | bnb_nf4 | viz_camera | flux2_klein | bark | musicgen | GGUF | 16.6 |
| `otr_4060_h3_nano` | machine | draft | gemma-4-E2B-it | none | minimax_h3_video | flux2_klein | kokoro | stable_audio_3 | GGUF | 55.9 |
| `otr_4060_haunted_e4b` | machine | draft | gemma-4-E4B-it | bnb_nf4 | animatediff15_v3_haunted_video | flux2_klein | kokoro | musicgen | AnimateDiff-Evolved,GGUF | 16.3 |
| `otr_4060_haunted_local` | machine | draft | gemma-4-E2B-it | none | animatediff15_v3_haunted_video | flux2_klein | kokoro | musicgen | AnimateDiff-Evolved,GGUF | 16.3 |
| `otr_4060_high_probe` | machine | draft | Qwen3.5-4B | none | animatediff15_v3_haunted_video | flux2_klein | kokoro | musicgen | AnimateDiff-Evolved,GGUF | 16.3 |
| `otr_4060_nano` | machine | draft | gemma-4-E2B-it | none | ltx_8gb | flux2_klein | kokoro | stable_audio_3 | GGUF | 30.1 |
| `otr_4060_nano_local` | machine | draft | gemma-4-E2B-it | none | ltx_8gb | flux2_klein | kokoro | musicgen | GGUF | 28.8 |
| `otr_8gb_fastwan` | machine | draft | gemma-4-E2B-it | none | fastwan_8gb | flux2_klein | bark | stable_audio_3 | GGUF | 28.2 |
| `otr_8gb_ltx` | machine | draft | gemma-4-E2B-it | none | ltx_8gb | flux2_klein | bark | stable_audio_3 | GGUF | 34.3 |
| `otr_8gb_ltx25_foley` | machine | draft | Qwen3.5-4B | bnb_nf4 | ltx25_foley_plus | sd15 | (canonical) | musicgen | GGUF | 26.7 |
| `otr_8gb_ltx25_mime` | machine | draft | Qwen3.5-4B | bnb_nf4 | ltx25_mime | sd15 | (canonical) | musicgen | GGUF | 26.7 |
| `otr_8gb_wan` | machine | draft | gemma-4-E2B-it | none | wan_ti2v | flux2_klein | bark | stable_audio_3 | GGUF | 27.5 |
| `otr_amd8_rocm` | machine | draft | Llama-3.2-3B-Instruct | none | still_motion | z_image_turbo | kokoro | musicgen | - | 21.8 |
| `otr_nvidia_8gb_h3` | machine | draft | gemma-4-E2B-it | none | minimax_h3_video | flux2_klein | kokoro | musicgen | GGUF | 54.6 |
| `otr_nvidia_8gb_haunted` | machine | shipping | gemma-4-E2B-it | none | animatediff15_v3_haunted_video | flux2_klein | kokoro | musicgen | AnimateDiff-Evolved,GGUF | 16.3 |
| `otr_qwen2507_haunted_proof` | machine | draft | Qwen3.5-4B | none | animatediff15_v3_haunted_video | z_image_turbo | kokoro | musicgen | AnimateDiff-Evolved | 25.4 |

## 12 GB  --  3 profiles (3 machine, 0 lab/cloud)

| profile | kind | status | writer | quant | video lane | image | voice | music | packs | GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `otr_4060_haunted_12b` | machine | draft | gemma-4-12b-it | bnb_nf4 | animatediff15_v3_haunted_video | flux2_klein | kokoro | musicgen | AnimateDiff-Evolved,GGUF | 16.3 |
| `otr_4060_viz_12b` | machine | draft | gemma-4-12b-it | bnb_nf4 | viz_camera | flux2_klein | kokoro | musicgen | GGUF | 12.7 |
| `otr_nv40_12gb` | machine | draft | gemma-4-12b-it | bnb_nf4 | wan_ti2v | flux2_klein | indextts2 | stable_audio_3 | GGUF | 34.4 |

## 16 GB  --  94 profiles (33 machine, 61 lab/cloud)

| profile | kind | status | writer | quant | video lane | image | voice | music | packs | GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `16gb_full` | machine | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | viz_camera | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_16gb_ltx_audio_in` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx_audio_in | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 49.3 |
| `otr_16gb_ltx_video` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 48.9 |
| `otr_5080_haunted_12b_overnight` | machine | draft | gemma-4-12b-it | bnb_nf4 | animatediff15_v3_haunted_video | z_image_turbo | kokoro | musicgen | AnimateDiff-Evolved | 25.4 |
| `otr_amd16_rocm` | machine | draft | Qwen3.5-4B | none | still_motion | z_image_turbo | kokoro | stable_audio_3 | - | 23.0 |
| `otr_bark_announcer_acceptance` | machine | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | indextts2 | stable_audio_3 | - | 38.0 |
| `otr_g4_fastwan` | machine | shipping | gemma-4-12b-it | bnb_nf4 | fastwan_8gb | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 44.1 |
| `otr_g4_humo` | machine | shipping | gemma-4-12b-it | bnb_nf4 | humo | z_image_turbo | indextts2 | stable_audio_3 | - | 60.9 |
| `otr_g4_ltx_8gb` | machine | shipping | gemma-4-12b-it | bnb_nf4 | ltx_8gb | z_image_turbo | indextts2 | stable_audio_3 | - | 50.2 |
| `otr_g4_ltx_audio_in` | machine | shipping | gemma-4-12b-it | bnb_nf4 | ltx_audio_in | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 49.3 |
| `otr_g4_ltx_video` | machine | shipping | gemma-4-12b-it | bnb_nf4 | ltx_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 48.9 |
| `otr_g4_wan_ti2v` | machine | shipping | gemma-4-12b-it | bnb_nf4 | wan_ti2v | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 43.5 |
| `otr_ghost_signal_v3_haunted` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | animatediff15_v3_haunted_video | z_image_turbo | indextts2 | stable_audio_3 | AnimateDiff-Evolved | 37.8 |
| `otr_h3_low_audio_in` | machine | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | h3_low_audio_in | z_image_turbo | indextts2 | stable_audio_3 | - | 76.6 |
| `otr_h3_low_video` | machine | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | h3_low_video | z_image_turbo | indextts2 | stable_audio_3 | - | 76.0 |
| `otr_ideogram4_local_still_word` | machine | draft | gemma-4-12b-it | bnb_nf4 | still_word | ideogram4_local | kokoro | musicgen | - | 19.8 |
| `otr_ltx25_foley_flux2klein` | machine | shipping | gemma-4-12b-it | bnb_nf4 | ltx25_high_foley_plus | flux2_klein | indextts2 | stable_audio_3 | GGUF | 47.3 |
| `otr_ltx25_high_foley_plus` | machine | shipping | gemma-4-12b-it | bnb_nf4 | ltx25_high_foley_plus | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 56.3 |
| `otr_ltx25_high_mime` | machine | shipping | gemma-4-12b-it | bnb_nf4 | ltx25_high_mime | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 78.5 |
| `otr_ltx25_high_video` | machine | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 56.3 |
| `otr_rot_h3_lumina` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | h3_low_video | lumina_image | indextts2 | stable_audio_3 | - | 67.2 |
| `otr_rot_humo_klein` | machine | draft | gemma-4-12b-it | bnb_nf4 | humo | flux2_klein | indextts2 | stable_audio_3 | GGUF | 51.8 |
| `otr_rot_ltx25_foley_fluxgen1` | machine | draft | gemma-4-12b-it | bnb_nf4 | ltx25_high_foley_plus | flux_gen1 | indextts2 | stable_audio_3 | GGUF | 50.1 |
| `otr_rot_ltx25_mime_klein` | machine | draft | gemma-4-12b-it | bnb_nf4 | ltx25_high_mime | flux2_klein | indextts2 | stable_audio_3 | GGUF | 69.5 |
| `otr_rot_ltx25_video_klein` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | flux2_klein | indextts2 | stable_audio_3 | GGUF | 47.3 |
| `otr_rot_ltx25_video_lumina` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | lumina_image | indextts2 | stable_audio_3 | GGUF | 47.5 |
| `otr_rot_wan_ideogram4` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | wan_ti2v | ideogram4_local | indextts2 | stable_audio_3 | GGUF | 41.5 |
| `otr_rot_wan_klein` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | wan_ti2v | flux2_klein | indextts2 | stable_audio_3 | GGUF | 34.4 |
| `otr_runpod_starter` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | wan22_high_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 43.5 |
| `otr_sd15_stills` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | sd15 | indextts2 | stable_audio_3 | - | 16.9 |
| `otr_stillin_lab_5080` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | animatediff15_v3_stillin_lab_video | z_image_turbo | indextts2 | stable_audio_3 | AnimateDiff-Evolved | 37.8 |
| `otr_upscale_ltx_probe` | machine | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 48.9 |
| `otr_upscale_ship` | machine | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | wan_ti2v | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 43.5 |
| `google_omni_all` | cloud | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | google_omni_video | google_image | google_tts | google_lyria | - | 0 |
| `google_omni_media` | cloud | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | - | google_image | (canonical) | - | - | 0 |
| `google_veo_all` | cloud | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | google_veo_video | google_image | google_tts | google_lyria | - | 0 |
| `google_veo_media` | cloud | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | - | google_image | (canonical) | - | - | 0 |
| `otr_lemmy_kokoro_diag` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | kokoro | stable_audio_3 | - | 23.0 |
| `otr_rot_tts_ann_chatterbox` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 59.0 |
| `otr_rot_tts_ann_dia` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 62.0 |
| `otr_rot_tts_bark` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | bark | stable_audio_3 | GGUF | 49.4 |
| `otr_rot_tts_chatterbox` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | chatterbox | stable_audio_3 | GGUF | 48.2 |
| `otr_rot_tts_dia` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | dia | stable_audio_3 | GGUF | 51.2 |
| `otr_rot_tts_kokoro` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_high_video | z_image_turbo | kokoro | stable_audio_3 | GGUF | 45.2 |
| `otr_sbcov_1` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_pan | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_sbcov_2` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | flux2_klein | indextts2 | stable_audio_3 | GGUF | 25.1 |
| `otr_sbcov_3` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_motion | lumina_image | indextts2 | stable_audio_3 | - | 25.3 |
| `otr_sbcov_4` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_pan | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_sbcov_5` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | flux2_klein | indextts2 | stable_audio_3 | GGUF | 34.4 |
| `otr_sbcov_6` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_motion | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_soak_llmsweep_01` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | flux2_klein | indextts2 | stable_audio_3 | GGUF | 57.3 |
| `otr_soak_llmsweep_02` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | indextts2 | stable_audio_3 | - | 57.5 |
| `otr_soak_llmsweep_03` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | lumina_image | indextts2 | stable_audio_3 | GGUF | 48.5 |
| `otr_soak_llmsweep_04` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | flux_gen1 | indextts2 | stable_audio_3 | GGUF | 57.3 |
| `otr_soak_llmsweep_05` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 54.7 |
| `otr_soak_llmsweep_06` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | flux2_klein | indextts2 | stable_audio_3 | GGUF | 57.3 |
| `otr_soak_llmsweep_07` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 54.7 |
| `otr_soak_still_flat_flux_gen1` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | flux_gen1 | indextts2 | stable_audio_3 | - | 27.9 |
| `otr_soak_still_flat_z_image_turbo` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_soak_still_motion_flux2_klein` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_motion | flux2_klein | indextts2 | stable_audio_3 | GGUF | 25.1 |
| `otr_soak_still_motion_lumina_image` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_motion | lumina_image | indextts2 | stable_audio_3 | - | 25.3 |
| `otr_soak_still_pan_flux_gen1` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_pan | flux_gen1 | indextts2 | stable_audio_3 | - | 27.9 |
| `otr_soak_still_pan_ideo` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_pan | ideo | indextts2 | stable_audio_3 | - | 14.9 |
| `otr_soak_still_word_flux2_klein` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_word | flux2_klein | indextts2 | stable_audio_3 | GGUF | 25.1 |
| `otr_soak_still_word_z_image_turbo` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_word | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_soak_word_razzle_ideo` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | word_razzle | ideo | indextts2 | stable_audio_3 | - | 14.9 |
| `otr_soak_word_razzle_lumina_image` | lab | draft | Mistral-Nemo-Instruct-2407 | bnb_nf4 | word_razzle | lumina_image | indextts2 | stable_audio_3 | - | 25.3 |
| `otr_w45_animatediff15_lightning_video` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | animatediff15_lightning_video | z_image_turbo | indextts2 | stable_audio_3 | AnimateDiff-Evolved | 37.3 |
| `otr_w45_animatediff15_v3_haunted_video` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | animatediff15_v3_haunted_video | z_image_turbo | indextts2 | stable_audio_3 | AnimateDiff-Evolved | 37.8 |
| `otr_w45_fastwan` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | fastwan_8gb | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 44.1 |
| `otr_w45_humo` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | humo | z_image_turbo | indextts2 | stable_audio_3 | - | 60.9 |
| `otr_w45_humo_14b_169` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | humo_14B_169 | z_image_turbo | indextts2 | stable_audio_3 | - | 60.9 |
| `otr_w45_humo_1_7b` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | humo_1.7B | z_image_turbo | indextts2 | stable_audio_3 | - | 46.7 |
| `otr_w45_humo_1_7b_169` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | humo_1.7B_169 | z_image_turbo | indextts2 | stable_audio_3 | - | 46.7 |
| `otr_w45_ltx25_foley_plus` | lab | shipping | gemma-4-12b-it | bnb_nf4 | ltx25_foley_plus | z_image_turbo | (canonical) | musicgen | GGUF | 44.0 |
| `otr_w45_ltx25_mime` | lab | shipping | gemma-4-12b-it | bnb_nf4 | ltx25_mime | z_image_turbo | (canonical) | musicgen | GGUF | 44.0 |
| `otr_w45_ltx25_video` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx25_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 56.3 |
| `otr_w45_ltx_8gb` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx_8gb | z_image_turbo | indextts2 | stable_audio_3 | - | 50.2 |
| `otr_w45_ltx_audio_in` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx_audio_in | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 49.3 |
| `otr_w45_ltx_video` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | ltx_video | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 48.9 |
| `otr_w45_mesh_stage` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | mesh_stage | z_image_turbo | indextts2 | stable_audio_3 | - | 38.7 |
| `otr_w45_minimax_h3_audio_in` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | h3_low_audio_in | z_image_turbo | indextts2 | stable_audio_3 | - | 76.6 |
| `otr_w45_minimax_h3_video` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | h3_low_video | z_image_turbo | indextts2 | stable_audio_3 | - | 76.0 |
| `otr_w45_still_flat` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_flat | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_still_motion` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_motion | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_still_pan` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_pan | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_still_word` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | still_word | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_viz_camera` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | viz_camera | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_viz_camera_kokoro_all` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | viz_camera | z_image_turbo | kokoro | stable_audio_3 | - | 23.0 |
| `otr_w45_viz_green` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | viz_green | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_viz_mxc_cpu` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | viz_mxc_cpu | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_viz_mxc_mandala` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | viz_mxc_mandala | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |
| `otr_w45_wan_ti2v` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | wan_ti2v | z_image_turbo | indextts2 | stable_audio_3 | GGUF | 43.5 |
| `otr_w45_word_razzle` | lab | shipping | Mistral-Nemo-Instruct-2407 | bnb_nf4 | word_razzle | z_image_turbo | indextts2 | stable_audio_3 | - | 34.1 |

## Mac  --  1 profiles (1 machine, 0 lab/cloud)

| profile | kind | status | writer | quant | video lane | image | voice | music | packs | GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `otr_mac_mps` | machine | shipping | Qwen3.5-4B | none | viz_camera | sd15 | kokoro | stable_audio_3 | - | 5.8 |

## CPU  --  4 profiles (1 machine, 3 lab/cloud)

| profile | kind | status | writer | quant | video lane | image | voice | music | packs | GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `cpu_floor` | machine | draft | Llama-3.2-3B-Instruct | none | still_motion | google_image | kokoro | musicgen | - | 2.5 |
| `otr_cloud_hq` | cloud | draft | openrouter:slot-a | none | google_veo_video | google_image | google_tts | google_lyria | - | 0 |
| `otr_cloud_lanes` | cloud | draft | openrouter:slot-a | none | cloud_wan_i2v_audio | cloud_nano_banana_2 | elevenlabs | sonilo | - | 0 |
| `otr_cloud_low` | cloud | draft | openrouter:slot-a | none | cloud_wan_i2v | google_image | google_tts | google_lyria | - | 0 |

