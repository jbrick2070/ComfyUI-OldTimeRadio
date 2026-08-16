"""Mirror the indextts2 CC0 char_voice references onto the chatterbox + dia clone
engines. The CC0 reference WAVs are clone-engine-agnostic, so each clone engine gets
the same voice pool by re-tagging the engine -- no new files, no downloads. Run with
the ComfyUI venv python; the bank JSON hot-reloads (no restart).

  python scripts/_otr_mirror_clone_refs.py [--dry-run]

IDEMPOTENT BY OWNERSHIP (rewritten 2026-08-16). It replaces only the
`(engine, voice_ref_id)` keys it generates, in place, and passes every other row
through untouched -- so a second run over its own output is byte-identical.

The earlier version dropped EVERY chatterbox/dia row and rebuilt from the
indextts2 rows, which destroyed anything it did not itself produce. By 2026-08-16
the bank held three announcer rows it does not generate, each pinned by
assertions, and a re-run invited by the word "idempotent" would have deleted all
three. A generator may only own the keys it can actually recreate.

Regenerates:
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


#: Source-id prefixes stripped before the mirror prefix is applied, so the
#: mirrored id reads `cb_lemmy_algenib_cockney_v1` rather than
#: `cb_idx_lemmy_algenib_cockney_v1`. `idx_` was added 2026-08-16 when the first
#: non-`vz_` indextts2 row (the qualified Lemmy clone) became mirrorable.
_STRIPPED_SOURCE_PREFIXES = ("vz_", "idx_")


def _new_id(prefix, orig_id):
    for stripped in _STRIPPED_SOURCE_PREFIXES:
        if orig_id.startswith(stripped):
            return prefix + orig_id[len(stripped):]
    return prefix + orig_id


def _mirror_row(src, engine):
    row = {"voice_ref_id": _new_id(_PREFIX[engine], src["voice_ref_id"]),
           "engine": engine}
    for k in _COMMON:
        row[k] = src[k]
    return row


def plan_rows(voices):
    """Pure planner: the FULL new voices list, plus what changed.

    THE OWNERSHIP RULE, and why this was rewritten 2026-08-16. The original
    version dropped EVERY chatterbox/dia row and rebuilt from the indextts2
    rows -- so any row it did not itself produce was destroyed. The bank had
    since gained three announcer rows it does not generate
    (`cb_announcer_female`, `dia_announcer_male`, `dia_announcer_female`), each
    pinned by assertions, and a re-run invited by the word "idempotent" would
    have deleted all three.

    A generator may only own the keys it can actually recreate. So this
    replaces its OWN `(engine, voice_ref_id)` keys in place, leaves every other
    row untouched, and appends genuinely new mirrors at the end. A second run
    over its own output is byte-identical, which is what idempotent has to mean
    before the word is used.
    """
    idx_char = [v for v in voices
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

    # The keys this generator OWNS. Everything else in the bank is somebody
    # else's row and is passed through untouched.
    managed = {(row["engine"], row["voice_ref_id"]): row for row in mirrored}

    new_voices = []
    emitted = set()
    for row in voices:
        key = (row["engine"], row["voice_ref_id"])
        if key in managed:
            new_voices.append(managed[key])       # refresh in place, keep order
            emitted.add(key)
        else:
            new_voices.append(row)                # not ours -- leave it alone
    added = [row for key, row in managed.items() if key not in emitted]
    new_voices.extend(added)

    preserved = [row["voice_ref_id"] for row in voices
                 if row["engine"] in _MIRROR_ENGINES
                 and (row["engine"], row["voice_ref_id"]) not in managed]
    return {
        "voices": new_voices,
        "mirrored": mirrored,
        "added": [row["voice_ref_id"] for row in added],
        "preserved": sorted(preserved),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(_BANK, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    voices = data["voices"]

    plan = plan_rows(voices)
    new_voices, mirrored = plan["voices"], plan["mirrored"]

    ids = [v["voice_ref_id"] for v in new_voices]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit("duplicate voice_ref_id after mirror: %s" % dupes)

    print("mirrored=%d  added=%d  preserved-unmanaged=%d  total=%d"
          % (len(mirrored), len(plan["added"]), len(plan["preserved"]),
             len(new_voices)))
    if plan["added"]:
        print("  added: %s" % ", ".join(plan["added"]))
    if plan["preserved"]:
        print("  preserved (not generated here, left untouched): %s"
              % ", ".join(plan["preserved"]))
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
