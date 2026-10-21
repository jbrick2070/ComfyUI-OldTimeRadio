"""Voice the demo's narration with Kokoro (ONNX, CPU): one wav per scene line."""
import json
import wave
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

ROOT = Path(r"C:\ComfyUI-Models\TTS\KokoroTTS")
OUT = Path(__file__).resolve().parent / "narration"
OUT.mkdir(exist_ok=True)
VOICE = "am_michael"

LINES = {
    "s2": "Press one button, and your own GPU writes, casts, voices, scores, and films a whole radio play.",
    "s3": "Pick a classic, or tell your own story, with a card for every character. Press make. "
          "The bar learns how long each stage takes on your machine, and a reload never loses the run.",
    "s5": [
        "Under the hood, it's my Old-Time Radio node pack on ComfyUI, driven through the Comfy S.D.K. and comfy A.P.I. proxy.",
        "Gemma 4 writes the story.",
        "Kokoro gives each character a voice.",
        "Stable Audio 3 scores it.",
        "Flux 3 Klein paints every beat.",
        "L.T.X. 2.5 films every beat, with its own sound.",
        "And F.F.mpeg takes it to 1080p, with no model at all.",
    ],
    "s6": "Other tools make clips. This makes a whole story: beginning, middle, and end, "
          "every character keeping one voice and one look. All open weights, all on one laptop.",
    "s7": "Radio Play Studio. Open source, on your own GPU.",
}

voices = sorted(ROOT.joinpath("voices").glob("_onnx_voices.*.npz"))[0]
tts = Kokoro(str(ROOT / "onnx" / "model.onnx"), str(voices))
lengths = {}
GAP_S = 0.3   # breath between the sentences of a multi-sentence scene
marks = {}
for key, text in LINES.items():
    parts = text if isinstance(text, list) else [text]
    chunks, offsets, rate = [], [], 24000
    for i, part in enumerate(parts):
        samples, rate = tts.create(part, voice=VOICE, speed=1.0, lang="en-us")
        offsets.append(round(sum(len(c) for c in chunks) / rate, 2))
        chunks.append(np.asarray(samples, dtype=np.float32))
        if i + 1 < len(parts):
            chunks.append(np.zeros(int(GAP_S * rate), dtype=np.float32))
    samples = np.concatenate(chunks)
    marks[key] = offsets
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(OUT / f"{key}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    lengths[key] = round(len(pcm) / rate, 2)
    print(key, lengths[key], "s")
(OUT / "lengths.json").write_text(json.dumps({"lengths": lengths, "sentence_starts": marks}, indent=1),
                                  encoding="utf-8")
print("sentence starts:", marks.get("s5"))
print("words:", sum(len((" ".join(t) if isinstance(t, list) else t).split()) for t in LINES.values()))
