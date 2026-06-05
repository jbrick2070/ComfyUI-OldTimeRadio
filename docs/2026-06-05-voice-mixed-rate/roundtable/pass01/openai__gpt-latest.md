<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Direction A is the safest, but it is not build-ready unless the resample is lazy, CPU deterministic, shape-preserving, and the bark fallback teardown is made exception-safe.

MUST-FIX BEFORE BUILD:
1. [_otr_voice_node_common.py:_render_per_line fallback branch, approx. `clips.append(audio)` after `_bark_fb.generate_voice`] Direction A must resample the bark fallback AUDIO dict before append, not after packing. Concrete fix: after bark generation, convert the returned AUDIO to the primary `sr` used later by `pack_audio_batch(clips, sample_rate=sr, mono=mono)`. Preserve the AUDIO contract: `{"waveform": [B,C,T], "sample_rate": sr}`. Do not change `pack_audio_batch` default behavior.

2. [_otr_voice_node_common.py:_render_per_line fallback branch] Do not import torchaudio or any resampling backend at module import time. Concrete fix: put the resample helper import inside the fallback-only execution path or inside a lazily-called helper in `_otr_audio_utils.py`. This preserves the documented C-5 invariant in `_otr_voice_node_common.py` module docstring: “Import-time is side-effect-free; engine libraries are lazy-imported INSIDE generate.”

3. [_otr_voice_node_common.py:_render_per_line fallback branch + `_bark_fb.unload()` block] The existing `_bark_fb.unload()` is not in a `finally`; adding a resampler introduces another failure point that can leave Bark resident if resampling or a later line raises. Concrete fix: wrap the per-line loop / fallback adapter use so `_bark_fb.unload()` runs in a `finally`, analogous to outer `_teardown(adapter)` in `generate()`. Current code only unloads after the loop:
   - `_bark_fb = None`
   - fallback may `_bark_fb.load()`
   - after loop: `if _bark_fb is not None: ... _bark_fb.unload()`
   Any exception before that block skips unload.

4. [_otr_audio_engines/base.py:pack_audio_batch] Do not implement Direction B as the default. `pack_audio_batch` currently documents and enforces: “All items must share one sample rate (raises otherwise); pass sample_rate to assert a specific rate.” The raise at:
   - `mismatched = {r for r in rates if r != sr}`
   - `raise ValueError("pack_audio_batch: mixed sample rates...")`
   is the correct tripwire. Concrete fix: leave this strict precondition intact for normal callers. If an opt-in is added later, it must be explicit and non-default, e.g. a separate helper or clearly named argument, not silent behavior change.

5. [_otr_audio_utils.py or local helper] The resampler must canonicalize first and preserve arbitrary `[B,C,T]`, not assume Bark’s current mono shape. `eng_bark.py:generate_voice` currently returns `wav.reshape(1, 1, -1)`, but the helper should not bake that in. Concrete fix: run `canonical_audio(audio)`, resample along last dim, return same batch/channel dimensions with `sample_rate=target_sr`.

6. [_otr_voice_node_common.py:_render_per_line] The target sample rate should be the resolved primary profile rate `sr`, not Bark’s native rate and not `max(rates)`. This is the `sr` assigned from `profile.sample_rate` before request construction and later passed to `pack_audio_batch`. Concrete fix: for indextts2 primary, downsample only Bark fallback clips from 24000 to 22050; do not upsample all indextts2 clips to 24000.

7. [_otr_voice_node_common.py:_render_per_line] Do not change pure Bark or all-primary output. The fix must be inside the `self.ROLE == "char_voice" and engine in _OTR_CLONE_ENGINES and not voice_ref` branch only. Concrete fix: no changes to `eng_bark.py`, `eng_indextts2.py`, or the non-fallback append path:
   - primary path remains `audio = adapter.generate_voice(...)`; `clips.append(audio)`
   - pure Bark engine path remains native 24000
   - all-indextts2-with-refs remains native 22050

8. [Regression tests] Add a deterministic mixed-rate regression that reproduces the exact failure: primary engine `indextts2` at 22050, at least one line with usable `voice_ref`, at least one line without usable ref causing Bark fallback at 24000. Assert:
   - no `pack_audio_batch: mixed sample rates` error
   - output `sample_rate == 22050`
   - output waveform rank is `[B,C,T]`
   - `B == number of rendered in-role lines`
   - mono policy still yields `C == 1` when `stereo_policy="mono_safe"`.

SHOULD-FIX:
1. [_otr_audio_utils.py] Prefer a shared helper such as `resample_audio(audio, target_sr)` over inline fallback-only tensor manipulation. This keeps the append-site fix small while avoiding duplicated AUDIO-contract logic. It should be lazy-importing internally if it uses torchaudio.

2. [Resampler choice] Use `torchaudio.functional.resample` lazily on CPU if torchaudio is guaranteed in this ComfyUI environment. It is the best fit among the listed options: no new pip dependency, better quality than hand-rolled linear interpolation, lighter than `torchaudio.transforms.Resample`, and avoids SciPy dependency risk. Concrete implementation constraints:
   - canonicalize to torch tensor first
   - move to CPU for resampling if necessary
   - use float32
   - resample along last dimension
   - return a CPU or original-device tensor only if downstream expects that; current `pack_audio_batch` creates CPU `torch.zeros`, so CPU is acceptable
   - no CUDA resampler

3. [Resampler dependency] [ASSUMPTION] The Blackwell venv includes torchaudio because `eng_indextts2.py:_load_wav` explicitly avoids `torchaudio.load` due to torchcodec, not because torchaudio is absent. Verify this before relying on `torchaudio.functional.resample`. If torchaudio is not guaranteed, declare it as an existing required runtime dependency or provide a deterministic torch-only fallback. Do not let “missing ref” become “missing torchaudio” hard-fail.

4. [_otr_voice_node_common.py logging] The existing final log says `packed {n} clips at {sr} Hz`. After the fix, add a warning/count when fallback clips were resampled, e.g. “resampled N bark fallback clips 24000 -> 22050”. This is useful because the output will be 22050 even for an all-fallback indextts2 episode.

5. [All-fallback test] Add a test where selected engine is indextts2/chatterbox but every character lacks a usable ref. Expected result: all Bark fallback clips are resampled to the primary profile `sr` and packed successfully. This is distinct from pure Bark engine selection, which must remain 24000 and byte-identical.

6. [Single-line test] Add a single fallback line test. It catches implementations that only resample when both rates appear, or that accidentally rely on batch max length behavior.

7. [Zero-line / zero-clip paths] Keep the existing `empty_audio_batch(sr)` behavior in `_render_per_line`: no in-role lines returns `empty_audio_batch(sr)` before any fallback work. Also preserve `generate()`’s `sr_hint` fallback for `audio_out is None`. Tests should assert empty char batch uses the selected adapter/profile rate, not Bark’s fallback rate.

8. [SceneSequencer / EpisodeAssembler / HuMo] Verify downstream mixed-rate handling outside this packer. The provided grounding only shows the voice nodes and packer. It does not show SceneSequencer, EpisodeAssembler, HuMo, announcer, or music mux code. Verify: whether later concatenation/mux accepts char voice at 22050, announcer Kokoro at its own rate, and theme/music at another rate, or whether those stages already resample. Do not infer safety from this local fix.

OPTIONAL / NICE-TO-HAVE:
- Add an explicit non-default helper/API for future callers, e.g. `resample_then_pack_audio_batch(items, target_sr, mono=True)`, but keep `pack_audio_batch` strict.
- Include audio duration assertions in tests: resampled Bark fallback duration should stay within one sample of the original duration after 24000 -> 22050 conversion.
- Add a tiny synthetic AUDIO-unit test using sine/impulse tensors rather than invoking Bark/IndexTTS2 models.

CUT THESE (over-engineering):
1. [Direction B] Cut default resampling inside `pack_audio_batch`. It weakens the current documented contract in `_otr_audio_engines/base.py` and would hide future adapter bugs. The local failure source is known: Bark fallback appended into an indextts2 batch without conversion.

2. [Direction C] Cut project-wide canonical voice-rate conversion for this bug. It touches every adapter and risks breaking the byte-identical Bark path, while the crash is caused by one fallback branch.

3. [Direction D] Cut whole-episode demotion to Bark or auto-enabling voice reuse as the fix. It changes casting semantics and can silently replace an intended indextts2 render with Bark. It also does not address the underlying invariant violation: mixed-rate clips must be normalized before packing.

4. [Hand-rolled resampler] Cut a custom kernel unless torchaudio is unavailable. It is more likely to introduce quality, determinism, and shape bugs than `torchaudio.functional.resample`, and it adds maintenance surface for a simple 24000 -> 22050 fallback conversion.