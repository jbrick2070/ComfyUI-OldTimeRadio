<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: ..."
        *   "MUST-FIX BEFORE BUILD:"
        *   "SHOULD-FIX:"
        *   "OPTIONAL / NICE-TO-HAVE:"
        *   "CUT THESE (over-engineering):"
        *   "Mark [ASSUMPTION] anywhere you are inferring..."

    Let's review the `eng_indextts2.py` stdout issue carefully.
    ```python
    line = proc.stdout.readline()
    try:
        ready = json.loads(line) if line.strip() else {"ready": False, "error": "no readiness line"}
    ```
    If `torch` prints `UserWarning: ...`, `line` will be that warning. `json.loads` will raise `ValueError`. The `except ValueError:` block sets `ready = {"ready": False, "error": "bad readiness line..."}`.
    Then `if not ready.get("ready"): raise RuntimeError(...)`.
    So yes, it hard-crashes on the first warning printed to stdout. This is a fatal flaw for a subprocess worker running PyTorch.

    Let's review the sample rate issue carefully.
    ```python
    # Mixed-rate fix (BUG-LOCAL voice): bark renders at its native
    # rate (24000), but this batch packs at the primary engine's sr
    if int(audio.get("sample_rate", sr)) != sr:
        audio = resample_audio(audio, sr)
    clips.append(audio)
    continue
    ```
    This is inside the `if engine in _OTR_CLONE_ENGINES and not voice_ref:` block.
    Outside that block:
    ```python
    with deterministic_inference(engine_seed, warn_only=True):
        audio = adapter.generate_voice(prepared, voice_ref, None, engine_seed)
    clips.append(audio)
    ```
    If `adapter.generate_voice` returns `sample_rate=24000` but `sr=48000` (from profile), it appends it directly. `pack_audio_