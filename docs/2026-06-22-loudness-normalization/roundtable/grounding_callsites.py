# GROUNDING -- verbatim call-site region from nodes/scene_sequencer.py (lines ~721-776).
# Confirms scope: :726 is SFX (stays peak); :747/:753/:775 are the three DIALOGUE voice paths.

            elif item_type == "sfx":
                desc = item.get("text", "") or "unknown sound"
                if sfx_clip_idx < len(sfx_clips):
                    clip_np, clip_sr = sfx_clips[sfx_clip_idx]
                    sfx_segment = _resample_audio(clip_np, clip_sr, sample_rate)
                    sfx_segment = _normalize_clip(sfx_segment, target_peak=0.85)   # :726 SFX -> STAYS peak
                    sfx_clip_idx += 1
                    sfx_pos = max(0, current_sample_pos + sfx_offset_samples)
                    sfx_timeline.append((sfx_pos, sfx_segment, desc))
                    render_log.append(f"[{global_idx}] SFX Overlay: {desc}")
                else:
                    render_log.append(f"[{global_idx}] SFX: {desc} (MISSING)")
                segment_np = np.zeros(int(sample_rate * 0.1), dtype=np.float32)

            elif item_type == "dialogue":
                character_name = _OTRLC.speaker_name(led, item)
                line = item.get("text") or ""
                is_announcer = (speaker_role == "announcer")

                if is_announcer and announcer_clip_idx < len(announcer_clips):
                    clip_np, clip_sr = announcer_clips[announcer_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)        # :747 announcer (Kokoro) -> RMS path
                    announcer_clip_idx += 1
                    render_log.append(f"[{global_idx}] ANNOUNCER (Kokoro): {line[:40]}...")
                elif tts_clip_idx < len(tts_clips):
                    clip_np, clip_sr = tts_clips[tts_clip_idx]
                    segment_np = _resample_audio(clip_np, clip_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)        # :753 pre-rendered TTS -> RMS path
                    tts_clip_idx += 1
                    render_log.append(f"[{global_idx}] {character_name}: {line[:40]}...")
                else:
                    # inline-Bark fallback
                    bark_np, bark_sr = _generate_bark_for_line(line, preset)
                    segment_np = _resample_audio(bark_np, bark_sr, sample_rate)
                    segment_np = _normalize_clip(segment_np)        # :775 inline-Bark -> RMS path
                    render_log.append(f"[{global_idx}] {character_name}: {line[:40]}...")

# Master applied ONCE in assemble() at :1122:
#   episode_waveform, _makeup_db = _master_loudness(episode_waveform)
# _master_loudness reads OTR_MASTER_MAKEUP_DB (default 4.0). Plan: default it to 0.0 when
# OTR_SEGMENT_LOUDNORM=='rms' and the env is unset (transparent peak ceiling, no double-gain).
