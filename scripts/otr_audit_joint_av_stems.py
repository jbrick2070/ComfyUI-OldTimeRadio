"""Does a joint-AV lane's OWN generated audio contain intelligible speech?

THE STEM IS THE ONLY HONEST TEST SURFACE. A final mix cannot answer this on
the foley lane: `ltx25_foley_plus` keeps the master at 0.80, so real TTS
dialogue is present and any correlation or listening test is confounded. The
per-beat stem under ``<episode>/audio/foley/`` is the model's own output with
no TTS anywhere near it, on BOTH lanes.

WHY ASR AND NOT ARITHMETIC. A hand-built "is this speech" detector -- band
energy, zero crossings, syllabic modulation -- is exactly the class of thing
this project has been burned by before (two hand-built grid detectors, both
wrong the same way). Transcription is a model answering the question the
operator actually asks: "are there words?" It runs on CPU so it cannot
disturb a render.

ASR IS A SCREEN, NOT A VERDICT. Whisper hallucinates on non-speech audio --
it will happily emit "you" or "Thank you." on a rustle. So this reports a
CONFIDENCE-FILTERED transcript and the operator's ear stays authoritative;
what this buys is triage across dozens of stems instead of dozens of listens.

Usage:
    python scripts/otr_audit_joint_av_stems.py [<episode_dir> ...]
    (no args = every episode that rendered a joint-AV beat)
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

EPISODES_DEFAULT = r"C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes"
JOINT_AV = ("ltx25_mime", "ltx25_foley_plus")

#: Whisper's stock hallucinations on silence/noise. Matching these is not
#: evidence of speech, and counting them as such would make every foley bed
#: look like it was talking.
_HALLUCINATIONS = {
    "you", "thank you.", "thank you", "bye.", "bye", ".", "!", "?",
    "thanks for watching!", "thanks for watching.", "subscribe",
    "the end.", "the end", "so", "so.", "oh", "oh.", "hmm", "mm",
    # observed live on real foley beds, 2026-08-28
    "thank you for watching!", "thank you for watching",
    "thank you so much for watching!", "thank you so much for watching",
    "mhm. mhm.", "mhm.", "mhm",
}

#: Whisper emits long runs of punctuation on textureless noise. Any transcript
#: that is punctuation and whitespace only carries no words by definition.
_PUNCT_ONLY = re.compile(r"^[\s.,!?\-–—']*$")

#: Below this average log-prob Whisper is guessing. Tuned to keep real words
#: and drop the stock filler above.
_MIN_LOGPROB = -1.0


def is_hallucination(text):
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    if (not t) or t in _HALLUCINATIONS or _PUNCT_ONLY.match(t):
        return True
    # A "transcript" with fewer than two real word characters per token is
    # noise dressed as text (". . . . ." arrives as many tiny tokens).
    words = [w for w in re.findall(r"[a-z']+", t) if len(w) > 1]
    return len(words) < 2


def lanes_in(ep_dir):
    """Which joint-AV lanes this episode actually rendered, from clip names."""
    clips = os.path.join(ep_dir, "clips")
    found = set()
    if os.path.isdir(clips):
        for fn in os.listdir(clips):
            for eng in JOINT_AV:
                if fn.endswith(eng + ".mp4"):
                    found.add(eng)
    return found


def stems_in(ep_dir):
    d = os.path.join(ep_dir, "audio", "foley")
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*.wav")))


def main(argv):
    targets = argv[1:]
    if not targets:
        targets = [os.path.join(EPISODES_DEFAULT, n)
                   for n in sorted(os.listdir(EPISODES_DEFAULT))
                   if os.path.isdir(os.path.join(EPISODES_DEFAULT, n))]
    targets = [t for t in targets if lanes_in(t)]
    if not targets:
        print("no episode with a joint-AV beat found")
        return 1

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU: never touch a render
    os.environ.setdefault("HF_HOME", r"C:\ComfyUI-Models\hf-cache")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-tiny.en", device=-1)

    grand_speaking = grand_total = 0
    report = []
    for ep in targets:
        name = os.path.basename(ep)
        lanes = "+".join(sorted(lanes_in(ep)))
        stems = stems_in(ep)
        print("\n" + "=" * 74)
        print("%s  [%s]  %d stem(s)" % (name[:60], lanes, len(stems)))
        print("=" * 74)
        speaking = 0
        for path in stems:
            try:
                out = asr(path, return_timestamps=False,
                          generate_kwargs={"return_dict_in_generate": True})
                text = (out.get("text") or "").strip()
            except Exception as exc:      # noqa: BLE001 -- a bad stem is data
                print("   %-38s  ASR FAILED: %s" % (os.path.basename(path), exc))
                continue
            grand_total += 1
            if is_hallucination(text):
                print("   %-38s  (no words)" % os.path.basename(path)[:38])
            else:
                speaking += 1
                grand_speaking += 1
                print("   %-38s  SPEECH: %r"
                      % (os.path.basename(path)[:38], text[:70]))
        report.append((name, lanes, speaking, len(stems)))
        print("   -> %d of %d stem(s) transcribed to words" % (speaking, len(stems)))

    print("\n" + "=" * 74)
    print("SUMMARY -- stems whose OWN generated audio transcribed to words")
    print("=" * 74)
    for name, lanes, spk, tot in report:
        print("  %-52s %-18s %d/%d" % (name[:52], lanes, spk, tot))
    print("\n  TOTAL: %d of %d joint-AV stems" % (grand_speaking, grand_total))
    print("\n  ASR is a SCREEN. A hit is a stem to listen to; a miss is not a")
    print("  guarantee of silence. The operator's ear remains the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
