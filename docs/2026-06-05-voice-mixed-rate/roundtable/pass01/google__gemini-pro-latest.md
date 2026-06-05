<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. Direction A is the only path that fixes the crash while preserving the bit-exact output of the primary engine and the byte-identical pure-bark legacy path.

MUST-FIX BEFORE BUILD:
1. [Candidate Fix Directions / _otr_voice_node_common.py] The bark fallback appends a 24000 Hz clip to a list that will be packed at 22050 Hz (indextts2's rate), crashing `pack_audio_batch`. 
   Fix: Implement Direction A. In `_otr_voice_node_common.py` around line 406, locally `import torchaudio.functional as F`, check if `audio["sample_rate"] != sr`, and if so, resample `audio["waveform"]` to `sr` on CPU before appending. Update `audio["sample_rate"] = sr`. (Downsampling bark is required; upsampling indextts2 would violate the C7 bit-exact invariant for the primary engine).
2. [Questions for the panel / Pipeline] Hidden dependency: If Node 81 (indextts2) outputs 22050 Hz and Node 82 (kokoro announcer) outputs 24000 Hz, the downstream SceneSequencer will receive mixed rates across different batches. 
   Fix: Ensure SceneSequencer (or the downstream muxer) resamples incoming batches to a canonical project rate before concatenation along the time dimension, otherwise the timeline will play at the wrong speed. [ASSUMPTION: SceneSequencer currently concatenates without resampling].

SHOULD-FIX:
1. [Questions for the panel / _otr_voice_node_common.py] All-fallback edge case: If an episode has *only* fallback lines, `sr` is still correctly set to the primary engine's rate (22050) before the loop, meaning all 24000 Hz bark clips will be resampled to 22050. Add a regression test specifically for a 100% fallback episode to ensure this setup logic doesn't drift and cause an accidental 24000 Hz batch.
2. [Questions for the panel / _otr_audio_engines/base.py] Contract design: Keep `pack_audio_batch`'s strict `raise` as the default to prevent silent degradation, but add an explicit `allow_resample=False` kwarg. If a future caller explicitly opts in, it can safely pack mixed sources without duplicating the resampling logic.

OPTIONAL / NICE-TO-HAVE:
- Log a warning when a fallback clip is resampled (e.g., "Resampling bark fallback from 24000 to 22050 to match primary engine"), noting the minor quality/timing degradation.
- `torchaudio.functional.resample` is the correct choice. Bark's output is a CPU tensor (`torch.from_numpy` in `eng_bark.py`), so the resampler will run deterministically on CPU and consume no VRAM, satisfying the 16 GB ceiling.

CUT THESE (over-engineering):
1. [Candidate B] Making `pack_audio_batch` resample by default. Safe to cut because it masks genuine rate-mismatch bugs (like an engine returning the wrong rate) and weakens a safety contract that successfully caught this exact bug.
2. [Candidate C] Project-wide canonical voice rate. Safe to cut because it requires touching every engine adapter, re-baselining all C7 audio fixtures, and destroys the byte-identical legacy bark path.
3. [Candidate D] Avoiding the mix entirely by demoting the whole episode to bark. Safe to cut because it violates the user's casting intent (silently dropping indextts2 for characters that *do* have references).