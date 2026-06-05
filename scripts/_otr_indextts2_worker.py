"""OTR <-> IndexTTS2 Path-B worker. Runs in the isolated index-tts .venv.

Protocol (matches nodes/_otr_audio_engines/eng_indextts2.py):
  launched : python _otr_indextts2_worker.py --model-dir <checkpoints> [--fp16]
  readiness: ONE json line on stdout -> {"ready": true} (or {"ready": false, "error": ...})
  request  : one json line on stdin  ->
      {"text","ref_clip","emo_vector":[8],"emo_alpha":1.0,"seed":int,"out_path","verbose":bool}
  response : one json line on stdout  ->
      {"ok": true, "out_path": <wav>, "sample_rate": 22050}  |  {"ok": false, "error": ...}
  stop     : {"stop": true} on stdin

IO discipline: the parent reads our stdout pipe for protocol JSON ONLY. main()
saves the real stdout fd aside and redirects fd1 -> fd2, so ALL incidental output
(indextts / torch / tqdm, python AND C level) lands on the engine's stderr file
and can never corrupt the JSON channel. Import is side-effect-free (the fd dance
happens in main(), not at module load). UTF-8, ASCII-only source.
"""
import os
import sys
import json
import argparse
import traceback

_proto = None


def emit(obj):
    _proto.write(json.dumps(obj, ensure_ascii=True) + "\n")
    _proto.flush()


def _seed_everything(seed):
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


def main():
    global _proto
    # Protect the protocol channel BEFORE importing the heavy model stack.
    _saved_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _proto = os.fdopen(_saved_fd, "w", buffering=1, encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401  (import surfaces a venv/torch problem as a clean error)
        from indextts.infer_v2 import IndexTTS2
        cfg = os.path.join(args.model_dir, "config.yaml")
        tts = IndexTTS2(cfg_path=cfg, model_dir=args.model_dir, use_fp16=bool(args.fp16))
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
            kwargs = dict(
                spk_audio_prompt=ref,
                text=req.get("text") or "",
                output_path=out_path,
                emo_alpha=float(req.get("emo_alpha", 1.0)),
                verbose=bool(req.get("verbose", False)),
            )
            ev = req.get("emo_vector")
            if ev and any(float(x) != 0.0 for x in ev):
                kwargs["emo_vector"] = [float(x) for x in ev]
            tts.infer(**kwargs)
            if not os.path.exists(out_path):
                emit({"ok": False, "error": "infer produced no file at %s" % out_path})
                continue
            emit({"ok": True, "out_path": out_path, "sample_rate": 22050})
        except Exception as e:
            emit({"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                  "tb": traceback.format_exc()[-600:]})


if __name__ == "__main__":
    main()
