<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

` to the fallback engine.
I will flag this as a SHOULD-FIX: "Passing `voice_preset or "v2/en_speaker_6"` to the fallback engine assumes the fallback is always Bark. If `missing_ref_fallback` is dynamic, resolve the fallback's reference using `getattr(_fallback_fb, "voice_ref_field", "voice_ref_path")` instead of hardcoding the Bark preset."

Let's check the Dia worker text prep:
`per line: build [S1] <target text>; if a clone transcript is supplied prepend it ...`
Wait, the Dia model expects `[S1] <target text>`.
Does the adapter do this, or the worker?
The plan says: "Worker `scripts/_otr_dia_worker.py`: ... per line: build `[S1] <target text>`; if a clone transcript is supplied prepend it"
If the worker does this, the adapter just sends the raw text. That's fine.

What about `eng_dia.py` adapter?
"roles=("char_voice","announcer_voice")"
Wait, Dia is dialogue-native. Does it support announcer? Yes, the plan says so.

Let's check `eng_chatterbox.py` adapter:
"requires_flag="OTR_ENABLE_CHATTERBOX""
This matches the plan.

Let's review the MUST-FIX items I've found:
1. [Part C] Dia transcript lookup by `voice_ref_id` is impossible in the adapter because `generate_voice` only receives `ref_clip_path`. Fix: Key `dia_ref_transcripts.json` by the reference WAV filename (e.g., `ix_male_warm.wav`) so the adapter can look it up via `os.path.basename(ref_clip_path)`.
2. [Part B] Chatterbox dynamic `sample_rate`: Dropping the class-level `sample_rate = 24000` and returning `m.sr` dynamically will crash `pack_audio_batch` with a `ValueError` if `m.sr` differs from the default 24000