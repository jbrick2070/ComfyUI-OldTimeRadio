"""G1 Test A -- the blinded three-arm Lemmy audition on IndexTTS2.

WHAT THIS IS FOR. Branch A of the Lemmy cross-engine plan cannot start until G1
passes, and G1 needs two things: about twenty minutes of GPU, and about ten
minutes of an operator's ears. This script does the first half and stops. It
renders six clips, hides which arm is which, and writes a sealed key. It does not
score anything, it does not decide anything, and it never writes a qualification
receipt -- only a human listening session can do that, which is the whole reason
the plan is built the way it is.

THE THREE ARMS (plan section 6, hashes frozen there and re-proved here):

    A  candidate   2_algenib_cockney.wav      the Cockney clone reference
    B  incumbent   vz_donor_marshal_indian    what 33 ledger rows actually used
    C  control     1_algenib_plain.wav        same speaker, no Cockney

A-vs-B answers "is the candidate a better route than the historic one". A-vs-C
answers "did the accent survive the clone, and is it still Algenib". The floor
question -- gravelly, Cockney, intelligible -- is absolute and applies to A alone.

WHY THE INCUMBENT IS IN AT ALL. Lemmy was never redrawn per episode: 33 of the 35
reference-carrying LEMMY rows name the SAME reference, because every one had
`meta.episode_seed=None` and the selector derived an identical seed. He was
ACCIDENTALLY PINNED. So this is a route comparison against a real incumbent, not
a hunt for something new.

BLINDING. Output files are named `arm1` / `arm2` / `arm3`; the mapping is written
to `KEY.json` in a sibling directory so it is not in the folder being listened
to. The shuffle is seeded and the seed is recorded, so the run reproduces exactly
while still being blind to the listener.

    python scripts/otr_g1_lemmy_audition.py            # preflight only
    python scripts/otr_g1_lemmy_audition.py --render   # preflight, then render

Background it and poll the log: six IndexTTS2 clips exceed the ~60s MCP ceiling.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

#: The audition assets, frozen 2026-08-08. Paths are absolute because these are
#: fixed evidence, not configuration.
_AUDITION_DIR = os.path.join(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes", "voice_audition_cockney")

#: Where the deliverable lands. Rendered assets are deliverables and go to their
#: canonical path the FIRST time -- never staged in tmp to be moved later.
_OUT_DIR = os.path.join(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes", "g1_lemmy_test_a")
_KEY_DIR = os.path.join(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes", "g1_lemmy_test_a_KEY")

#: One seed for the render (so every arm is generated under identical
#: conditions) and one for the label shuffle (so the blinding is reproducible).
RENDER_SEED = 20260810
SHUFFLE_SEED = 8102026

ARMS = {
    "A": {
        "role": "candidate",
        "file": os.path.join(_AUDITION_DIR, "2_algenib_cockney.wav"),
        "sha_prefix": "47E733", "sha_suffix": "A60DB2",
        "note": "Google TTS Algenib speaking Cockney -- the clone reference",
    },
    "B": {
        "role": "incumbent",
        "bank_ref_id": "vz_donor_marshal_indian",
        "note": "the accidentally-pinned historic IndexTTS2 reference",
    },
    "C": {
        "role": "control",
        "file": os.path.join(_AUDITION_DIR, "1_algenib_plain.wav"),
        "sha_prefix": "D48AAD", "sha_suffix": "283CFB",
        "note": "same speaker, NO Cockney -- the identity/accent control",
    },
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def preflight() -> dict:
    """Prove all three references BEFORE the model is loaded. Raises on any
    mismatch: an audition whose inputs are not the frozen ones is not evidence."""
    os.environ.setdefault("OTR_TEST_MODE", "0")
    resolved: dict = {}

    for arm in ("A", "C"):
        spec = ARMS[arm]
        path = spec["file"]
        if not os.path.isfile(path):
            raise SystemExit("arm %s reference is missing: %s" % (arm, path))
        digest = sha256_of(path)
        if not (digest.startswith(spec["sha_prefix"])
                and digest.endswith(spec["sha_suffix"])):
            raise SystemExit(
                "arm %s reference does NOT match the frozen hash: %s\n"
                "  expected %s...%s" % (arm, digest, spec["sha_prefix"],
                                        spec["sha_suffix"]))
        resolved[arm] = {"path": path, "sha256": digest}
        print("  arm %s  %-28s %s  OK" % (arm, os.path.basename(path), digest[:16]))

    # Arm B resolves through the BANK, and must byte-match its bank hash.
    from nodes._otr_voice_bank import load_voice_bank
    from nodes._otr_voice_node_common import _resolve_ref_to_disk

    bank, _ = load_voice_bank()
    entry = next((e for e in bank
                  if e.voice_ref_id == ARMS["B"]["bank_ref_id"]
                  and e.engine == "indextts2"), None)
    if entry is None:
        raise SystemExit("arm B: %r is not an indextts2 bank entry"
                         % ARMS["B"]["bank_ref_id"])
    disk = _resolve_ref_to_disk(entry.ref_path)
    if not (disk and os.path.exists(disk)):
        raise SystemExit("arm B reference does not exist on disk: %s" % (disk,))
    digest = sha256_of(disk)
    claimed = (entry.ref_sha256 or "").upper()
    if claimed and digest != claimed:
        raise SystemExit(
            "arm B bytes do NOT match the bank hash\n  actual %s\n  bank   %s"
            % (digest, claimed))
    resolved["B"] = {"path": disk, "sha256": digest,
                     "bank_ref_id": entry.voice_ref_id}
    print("  arm B  %-28s %s  OK" % (entry.voice_ref_id, digest[:16]))
    return resolved


def render(resolved: dict) -> int:
    from config.cast_pools import LEMMY_AUDITION_LINES as LINES
    from nodes._otr_audio_engines import get_engine

    os.makedirs(_OUT_DIR, exist_ok=True)
    os.makedirs(_KEY_DIR, exist_ok=True)

    # Blind the arms. Seeded so the run reproduces; the mapping goes to the KEY
    # directory, not the listening directory.
    labels = ["arm1", "arm2", "arm3"]
    order = ["A", "B", "C"]
    random.Random(SHUFFLE_SEED).shuffle(order)
    label_of = dict(zip(order, labels))

    engine = get_engine("indextts2")
    print("\nengine=%s sample_rate=%s" % (type(engine).__name__, engine.sample_rate))

    import soundfile as sf

    manifest = {
        "test": "G1 Test A -- blinded three-arm Lemmy audition",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": "indextts2",
        "engine_impl_version": str(getattr(engine, "impl_version", "") or ""),
        "sample_rate": int(engine.sample_rate),
        "render_seed": RENDER_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "lines_version": LINES["lines_version"],
        "neutral_line": LINES["neutral_line"],
        "emotional_line": LINES["emotional_line"],
        "references": {a: resolved[a]["sha256"] for a in ("A", "B", "C")},
        "clips": [],
    }

    try:
        for arm in ("A", "B", "C"):
            ref = resolved[arm]["path"]
            for kind, text in (("neutral", LINES["neutral_line"]),
                               ("emotional", LINES["emotional_line"])):
                name = "%s_%s.wav" % (label_of[arm], kind)
                out = os.path.join(_OUT_DIR, name)
                print("  rendering %s (arm %s, %s) ..." % (name, arm, kind))
                prepared = engine.prepare_text(text, None) \
                    if hasattr(engine, "prepare_text") else text
                audio = engine.generate_voice(prepared, ref, None, RENDER_SEED)
                wav = audio["waveform"].cpu().numpy().T      # [T, C]
                sf.write(out, wav, int(audio["sample_rate"]))
                manifest["clips"].append({
                    "file": name,
                    "line_kind": kind,
                    "sha256": sha256_of(out),
                    "sample_rate": int(audio["sample_rate"]),
                })
                print("      -> %s  %s" % (out, sha256_of(out)[:16]))
    finally:
        unload = getattr(engine, "unload", None)
        if callable(unload):
            unload()

    with open(os.path.join(_OUT_DIR, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    key = {
        "WARNING": "Do not open until after listening. This reveals the arms.",
        "mapping": {label_of[a]: {"arm": a, **{k: v for k, v in ARMS[a].items()
                                               if k in ("role", "note")}}
                    for a in ("A", "B", "C")},
        "reference_sha256": {a: resolved[a]["sha256"] for a in ("A", "B", "C")},
        "shuffle_seed": SHUFFLE_SEED,
    }
    with open(os.path.join(_KEY_DIR, "KEY.json"), "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=1)

    print("\nclips  -> %s" % _OUT_DIR)
    print("KEY    -> %s  (do not open until after listening)" % _KEY_DIR)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="actually load IndexTTS2 and render (default: preflight only)")
    args = ap.parse_args(argv)

    print("G1 Test A preflight -- proving the three frozen references")
    resolved = preflight()
    if not args.render:
        print("\nPreflight OK. Re-run with --render to produce the clips.")
        return 0
    return render(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
