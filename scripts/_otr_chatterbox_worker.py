"""OTR <-> Chatterbox Path-B worker. Runs in the isolated chatterbox .venv.

Protocol (matches nodes/_otr_audio_engines/eng_chatterbox.py):
  launched : python _otr_chatterbox_worker.py
  readiness: ONE json line on stdout -> {"ready": true} (or {"ready": false, "error": ...})
  request  : one json line on stdin  ->
      {"text","ref_clip","exaggeration":float,"cfg_weight":float,
       "temperature":float,"seed":int,"out_path","verbose":bool}
  response : one json line on stdout  ->
      {"ok": true, "out_path": <wav>, "sample_rate": 24000} | {"ok": false, "error": ...}
  stop     : {"stop": true} on stdin

IO discipline: identical to the indextts2 worker -- fd1 is redirected to fd2 so all
chatterbox / torch / tqdm output (python AND C level) lands on the engine's stderr
file and can NEVER corrupt the JSON protocol channel; the protocol JSON is written
to the saved real-stdout fd. Import is side-effect-free (the fd dance + model load
happen in main(), not at module load). UTF-8, ASCII-only source.
"""
import os
import sys
import json
import inspect
import traceback

_proto = None
_TARGET_SR = 24000


def emit(obj):
    _proto.write(json.dumps(obj, ensure_ascii=True) + "\n")
    _proto.flush()


def _seed_everything(seed):
    # The main process's deterministic_inference() cannot seed THIS subprocess,
    # so seed every RNG here, per request, before the forward.
    try:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _supported_kwargs(fn, **kwargs):
    """Subset of kwargs whose names the REAL generate() accepts. A kwarg the
    installed chatterbox version does not take is dropped instead of crashing the
    forward (introspected here, in the venv that actually has the model)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def main():
    global _proto
    # Protect the protocol channel BEFORE importing the heavy model stack.
    _saved_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _proto = os.fdopen(_saved_fd, "w", buffering=1, encoding="utf-8")

    try:
        import torch  # noqa: F401  (surfaces a venv/torch problem as a clean error)
        import torchaudio
        from chatterbox.tts import ChatterboxTTS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxTTS.from_pretrained(device=dev)
    except Exception as e:
        emit({"ready": False, "error": "%s: %s" % (type(e).__name__, e),
              "tb": traceback.format_exc()[-800:]})
        return

    emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            emit({"ok": False, "error": "bad request json: %s" % e})
            continue
        if req.get("stop"):
            break
        try:
            ref = req.get("ref_clip")
            out_path = req["out_path"]
            if not ref or not os.path.exists(ref):
                emit({"ok": False, "error": "ref_clip missing: %r" % ref})
                continue
            _seed_everything(int(req.get("seed", 0)))
            kwargs = _supported_kwargs(
                model.generate,
                audio_prompt_path=ref,
                exaggeration=float(req.get("exaggeration", 0.5)),
                cfg_weight=float(req.get("cfg_weight", 0.4)),
                temperature=float(req.get("temperature", 0.6)),
            )
            import torch
            wav = model.generate(req.get("text") or "", **kwargs)
            src_sr = int(getattr(model, "sr", _TARGET_SR) or _TARGET_SR)
            if not isinstance(wav, torch.Tensor):
                wav = torch.as_tensor(wav)
            wav = wav.detach().to("cpu", dtype=torch.float32)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)            # [1, T]
            # Guarantee the declared 24000 rate so the main-venv batch packs
            # single-rate (pack_audio_batch contract); chatterbox base is 24000.
            if src_sr != _TARGET_SR:
                wav = torchaudio.functional.resample(wav, src_sr, _TARGET_SR)
            # Save via soundfile, NOT torchaudio.save: torchaudio 2.x routes save
            # through torchcodec (not installed in this isolated venv) -> the cu128
            # Blackwell torch hits "TorchCodec is required". soundfile is direct.
            import soundfile as sf
            sf.write(out_path, wav.numpy().T, _TARGET_SR)  # [C, T] -> [T, C]
            if not os.path.exists(out_path):
                emit({"ok": False, "error": "generate produced no file at %s" % out_path})
                continue
            emit({"ok": True, "out_path": out_path, "sample_rate": _TARGET_SR})
        except Exception as e:
            emit({"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                  "tb": traceback.format_exc()[-600:]})


if __name__ == "__main__":
    main()
