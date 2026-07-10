"""OTR <-> Dia Path-B worker. Runs in the isolated dia .venv.

Protocol (matches nodes/_otr_audio_engines/eng_dia.py):
  launched : python _otr_dia_worker.py --model <hf_id>
  readiness: ONE json line on stdout -> {"ready": true} | {"ready": false, "error": ...}
  request  : one json line on stdin  ->
      {"text","ref_clip","ref_transcript","seed":int,"out_path","verbose":bool}
  response : one json line on stdout  ->
      {"ok": true, "out_path": <wav>, "sample_rate": 44100} | {"ok": false, "error": ...}
  stop     : {"stop": true} on stdin

Dia is dialogue-native: each per-line render is one [S1] turn. Voice cloning
prepends the reference transcript when supplied (audio_prompt-only otherwise).
Same fd1->fd2 discipline as the indextts2 worker so model / torch prints never
corrupt the JSON channel; the protocol JSON goes to the saved real-stdout fd.
UTF-8, ASCII-only source.
"""
import os
import sys
import json
import argparse
import traceback

_proto = None
_SR = 44100


def emit(obj):
    _proto.write(json.dumps(obj, ensure_ascii=True) + "\n")
    _proto.flush()


def _seed_everything(seed):
    # The main process cannot seed THIS subprocess; seed here per request. Dia
    # keeps speaker consistency via the audio prompt + a fixed seed.
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


def _strip_lead_tag(s):
    s = (s or "").strip()
    for tag in ("[S1]", "[S2]"):
        if s.startswith(tag):
            s = s[len(tag):].strip()
    return s


def _build_prompt(text, transcript):
    # Dia format: always begin with [S1]. Cloning prepends the ref transcript
    # (also [S1]) before the target so the model continues in that voice. Tolerate
    # a transcript / text that already carries a leading tag (avoid "[S1] [S1]").
    text = _strip_lead_tag(text)
    transcript = _strip_lead_tag(transcript)
    if transcript:
        return "[S1] %s [S1] %s" % (transcript, text)
    return "[S1] %s" % text


def _ensure_rate(out_path):
    """Guarantee the saved WAV is exactly _SR so the main-venv batch packs
    single-rate (pack_audio_batch contract). DAC output is 44100; this is cheap
    insurance against a build that writes a different rate."""
    import soundfile as sf
    try:
        if int(sf.info(out_path).samplerate) == _SR:
            return
    except Exception:
        pass  # info failed -> fall through to read + (re)write so we never
              # report 44100 for an unverified file (a corrupt file raises in
              # sf.read below -> the worker emits ok:false, fail-closed)
    import numpy as np
    data, sr = sf.read(out_path, dtype="float32", always_2d=True)  # [T, C]
    try:
        import torch
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(data.T), sr, _SR)
        sf.write(out_path, wav.T.numpy(), _SR)
    except Exception:
        ratio = _SR / float(sr)
        n_out = int(round(data.shape[0] * ratio))
        idx = np.linspace(0, data.shape[0] - 1, n_out)
        res = np.stack([np.interp(idx, np.arange(data.shape[0]), data[:, c])
                        for c in range(data.shape[1])], axis=1).astype("float32")
        sf.write(out_path, res, _SR)


def _save(model, out, out_path):
    """Save Dia output. Prefer the documented model.save_audio(path, out); fall
    back to soundfile if a build returns a raw array/tensor instead."""
    try:
        model.save_audio(out_path, out)
        if os.path.exists(out_path):
            return
    except Exception:
        pass
    import numpy as np
    import soundfile as sf
    arr = out
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().to("cpu").numpy()
    except Exception:
        pass
    arr = np.asarray(arr, dtype="float32").squeeze()
    sf.write(out_path, arr, _SR)


def _load_audio_prompt_codes(model, ref_path):
    """Load a WAV reference without TorchCodec/FFmpeg and return Dia DAC codes.

    Dia's public ``generate(audio_prompt=<path>)`` path delegates to
    ``torchaudio.load``. Recent torchaudio builds route that through TorchCodec,
    which is brittle on Windows unless shared FFmpeg DLLs and ABI-matched
    torchcodec wheels are present. Our reference bank is WAV-only, so soundfile
    is the smaller, deterministic loader here.
    """
    import soundfile as sf
    import torch

    data, sr = sf.read(ref_path, dtype="float32", always_2d=True)  # [T, C]
    audio = torch.from_numpy(data.T).contiguous()                  # [C, T]
    if int(sr) != _SR:
        try:
            import torchaudio
            audio = torchaudio.functional.resample(audio, int(sr), _SR)
        except Exception:
            import numpy as np
            ratio = _SR / float(sr)
            n_out = int(round(audio.shape[-1] * ratio))
            idx = np.linspace(0, audio.shape[-1] - 1, n_out)
            arr = audio.numpy()
            res = np.stack([
                np.interp(idx, np.arange(audio.shape[-1]), arr[c])
                for c in range(arr.shape[0])
            ], axis=0).astype("float32")
            audio = torch.from_numpy(res)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    return model._encode(audio.to(model.device))


def main():
    global _proto
    _saved_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _proto = os.fdopen(_saved_fd, "w", buffering=1, encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nari-labs/Dia-1.6B-0626")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401  (surfaces a venv/torch problem as a clean error)
        from dia.model import Dia
        compute = "float16" if torch.cuda.is_available() else "float32"
        model = Dia.from_pretrained(args.model, compute_dtype=compute)
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
            prompt = _build_prompt(req.get("text"), req.get("ref_transcript"))
            # audio_prompt-only is the official clone path this pass; a supplied
            # transcript is already folded into the prompt by _build_prompt.
            ref_codes = _load_audio_prompt_codes(model, ref)
            out = model.generate(prompt, audio_prompt=ref_codes)
            _save(model, out, out_path)
            if not os.path.exists(out_path):
                emit({"ok": False, "error": "generate produced no file at %s" % out_path})
                continue
            _ensure_rate(out_path)
            emit({"ok": True, "out_path": out_path, "sample_rate": _SR})
        except Exception as e:
            emit({"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                  "tb": traceback.format_exc()[-600:]})


if __name__ == "__main__":
    main()
