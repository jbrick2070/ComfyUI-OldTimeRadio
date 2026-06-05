<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready with fixes. The direction A (resample bark-fallback clip) is correct and minimal; it needs concrete implementation details for the resampler and dependency handling.

MUST-FIX BEFORE BUILD:
1. [Fallback resample site] In `_otr_voice_node_common.py` ~line 405, after `audio = _bark_fb.generate_voice(...)`, resample `audio` from its native 24000 Hz to the target `sr` (22050). Use `torchaudio.functional.resample` on the waveform tensor, ensuring CPU tensor for determinism (`audio["waveform"].cpu()`). After resampling, update `audio["sample_rate"] = sr` before appending. Example patch:
   ```python
   import torchaudio.functional as F_ta
   wf = audio["waveform"].cpu()
   resampled = F_ta.resample(wf, orig_freq=audio["sample_rate"], new_freq=sr)
   audio["waveform"] = resampled.to(device=audio["waveform"].device) if audio["waveform"].is_cuda else resampled
   audio["sample_rate"] = sr
   ```
   This preserves deterministic output (CPU resampling, same input -> same bytes) and does not alter the indextts2 clips.

2. [Dependency availability] The codebase currently does not import `torchaudio`. Add a lazy import guard in the fallback branch (e.g., `try: import torchaudio.functional as F_ta; except ImportError: raise RuntimeError(...)`). Alternatively, list `torchaudio` as a required dependency (it is present in standard ComfyUI torch 2.10 installs; verify with `pip freeze`). The error message should instruct the operator to install torchaudio if missing.

3. [Determinism guarantee] Ensure the resampler does not use nondeterministic GPU ops by forcing the tensor to CPU before calling `torchaudio.functional.resample`. The bark output is already CPU (`torch.from_numpy`), but be explicit.

4. [Byte-identical invariant] The resampling must only execute when the fallback branch is taken (engine in `_OTR_CLONE_ENGINES` and no `voice_ref`). The existing condition already enforces this. After the fix, an all-indextts2 or all-bark episode remains untouched.

SHOULD-FIX:
* Add a regression test that exercises: (a) an episode with all indextts2 lines having refs (no resampling), (b) an episode where some lines lack refs (mixed rates, fallback triggered), (c) zero in-role lines, (d) mono policy. Verify audio output byte-identical in case (a) and deterministic in (b) across runs.
* Consider extracting a `resample_audio(audio, target_sr)` utility into `_otr_audio_utils.py` for reuse if future engines require cross-rate mixing.

OPTIONAL / NICE-TO-HAVE:
* Add a warning log when a resample occurs (e.g., “resampled bark fallback from 24000 to 22050 for line X”) to aid debugging.
* If torchaudio is unavailable, attempt a fallback to `scipy.signal.resample_poly` (if scipy present) to avoid a hard failure in constrained environments, but this is lower priority.

CUT THESE (over-engineering):
* Directions B, C, D are not needed now. The packer’s strict raise remains a valuable tripwire for other genuinely unintended mixed-rate bugs. Changing it would weaken safety. A global canonical rate would touch many engines and break byte‑identical legacy paths. Auto‑reuse/casting policies (D) do not guarantee all characters get refs; they would not eliminate the fallback in the reported scenario (6 chars, only 4 refs).

[ASSUMPTION] The environment has `torchaudio` installed as part of the pytorch 2.10 / cu130 stack. Verify `torchaudio.__version__` in the actual ComfyUI venv. If absent, the fix must either add it or use an alternative resampler (e.g., a simple hand-rolled linear interpolation) but that risks quality.