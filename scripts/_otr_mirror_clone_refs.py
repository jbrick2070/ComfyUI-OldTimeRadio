"""Mirror the indextts2 CC0 char_voice references onto the chatterbox + dia clone
engines (idempotent). The 36 CC0 reference WAVs are clone-engine-agnostic, so each
clone engine gets the same voice pool by re-tagging the engine -- no new files, no
downloads. Run with the ComfyUI venv python; the bank JSON hot-reloads (no restart).

  python scripts/_otr_mirror_clone_refs.py [--dry-run]

Drops any pre-existing chatterbox/dia rows first (including the old placeholder rows
that pointed at nonexistent /refs/chatterbox/*.wav), then regenerates:
  - 36 chatterbox char_voice rows  (cb_*)
  - 36 dia char_voice rows         (dia_*)
  - 1  chatterbox announcer row    (cb_announcer_male -> a real male CC0 ref)
UTF-8, no BOM, ASCII-only.
"""
import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANK = os.path.join(_REPO, "config", "voice_reference_bank.json")
_MIRROR_ENGINES = ("chatterbox", "dia")
_PREFIX = {"chatterbox": "cb_", "dia": "dia_"}
_COMMON = ("gender", "timbre", "roles", "age_band", "ref_path", "ref_sha256",
           "commercial_clean")
_ANNOUNCER_REF_ID = "vz_bill_boerst"  # a real on-disk male CC0 ref


def _new_id(prefix, orig_id):
    base = orig_id[3:] if orig_id.startswith("vz_") else orig_id
    return prefix + base


def _mirror_row(src, engine):
    row = {"voice_ref_id": _new_id(_PREFIX[engine], src["voice_ref_id"]),
           "engine": engine}
    for k in _COMMON:
        row[k] = src[k]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(_BANK, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    voices = data["voices"]

    # Keep everything that is NOT a chatterbox/dia row (indextts2, kokoro, ...).
    kept = [v for v in voices if v["engine"] not in _MIRROR_ENGINES]
    idx_char = [v for v in kept
                if v["engine"] == "indextts2" and "char_voice" in v.get("roles", [])]

    mirrored = []
    for engine in _MIRROR_ENGINES:
        for src in idx_char:
            mirrored.append(_mirror_row(src, engine))

    # One chatterbox announcer row off a real male CC0 ref (so announcer-via-
    # chatterbox is not dangling; dia stays char_voice-only this pass).
    ann_src = next((v for v in idx_char if v["voice_ref_id"] == _ANNOUNCER_REF_ID), None)
    if ann_src is not None:
        mirrored.append({
            "voice_ref_id": "cb_announcer_male",
            "engine": "chatterbox",
            "gender": "male",
            "timbre": ["authoritative", "resonant"],
            "roles": ["announcer_voice"],
            "age_band": "adult",
            "ref_path": ann_src["ref_path"],
            "ref_sha256": ann_src["ref_sha256"],
            "commercial_clean": True,
        })

    new_voices = kept + mirrored
    ids = [v["voice_ref_id"] for v in new_voices]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit("duplicate voice_ref_id after mirror: %s" % dupes)

    print("kept=%d  mirrored=%d (chatterbox/dia char + 1 announcer)  total=%d"
          % (len(kept), len(mirrored), len(new_voices)))
    if args.dry_run:
        print("[dry-run] no write")
        return
    data["voices"] = new_voices
    with open(_BANK, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=True)
        fh.write("\n")
    print("wrote %s" % _BANK)


if __name__ == "__main__":
    main()
