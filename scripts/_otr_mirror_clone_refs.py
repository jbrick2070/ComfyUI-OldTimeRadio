"""Mirror the indextts2 CC0 char_voice references onto the chatterbox + dia clone
engines. The CC0 reference WAVs are clone-engine-agnostic, so each clone engine gets
the same voice pool by re-tagging the engine -- no new files, no downloads. Run with
the ComfyUI venv python; the bank JSON hot-reloads (no restart).

  python scripts/_otr_mirror_clone_refs.py [--dry-run]

IDEMPOTENT BY OWNERSHIP (rewritten 2026-08-16). It refreshes only the
`(engine, voice_ref_id)` keys it generates, MERGING over what is already there,
and passes every other row through untouched -- so a second run over its own
output is byte-identical.

OWNERSHIP HAD TO BE FIXED TWICE, and the second time is worth reading before
trusting this script again. The original version dropped EVERY chatterbox/dia row
and rebuilt from the indextts2 rows, destroying anything it did not itself
produce -- three announcer rows pinned by nine assertions. That was fixed at the
ROW level: own only the keys you can recreate.

It was still destroying data at the FIELD level, and the receipt could not see
it. `mirrored=83 added=2 preserved-unmanaged=3` prints identically whether or not
a field is lost, because it counts ROWS. A real run against the real bank showed
the truth: `speaker_id` was stripped from all eight mirrored rows that had one
(it was not on the seven-field allow-list, which was written before the field
existed), and the hand-improved `cb_announcer_male` was reverted to the literal
below. So ownership is now field-aware in both directions -- a mirror takes every
field but its identity, and a row this script merely bootstraps is created only
when it is missing.

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

#: The only two fields a mirror does NOT take from its source. Everything else
#: rides along, and that is a deliberate reversal of the earlier allow-list.
#:
#: WHY AN ALLOW-LIST WAS WRONG. It named seven fields and the bank has since grown
#: more, so every field added after it was written was SILENTLY DROPPED from all
#: 83 mirrored rows on every run. `speaker_id` was the one that mattered: it
#: records the real human behind a reference, and it exists because ref_path
#: collision cannot catch two recordings of one person -- LibriVox's Mark F. Smith
#: has a plain and a grandfatherly take in two different files. Without it one
#: narrator can be cast as two characters in the same episode, on chatterbox and
#: dia only, which is a casting defect nobody would trace back to a generator.
#:
#: A deny-list of exactly the identity pair is the right shape: a mirror IS its
#: source, re-tagged, so any future bank field mirrors correctly without anyone
#: remembering to come back here.
_IDENTITY_FIELDS = ("voice_ref_id", "engine")
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
    """The source row, re-tagged for ``engine``. Every field but the identity."""
    row = {"voice_ref_id": _new_id(_PREFIX[engine], src["voice_ref_id"]),
           "engine": engine}
    for key, value in src.items():
        if key not in _IDENTITY_FIELDS:
            row[key] = value
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

    A generator may only own the keys it can actually recreate -- and only the
    FIELDS it can actually derive, which is the half the first repair missed. So
    this MERGES its derived fields over its own `(engine, voice_ref_id)` keys,
    leaves every other row and every field it does not produce untouched, and
    appends genuinely new mirrors at the end. A second run over its own output is
    byte-identical, which is what idempotent has to mean before the word is used.
    """
    idx_char = [v for v in voices
                if v["engine"] == "indextts2" and "char_voice" in v.get("roles", [])]
    on_disk = {(v["engine"], v["voice_ref_id"]) for v in voices}

    mirrored = []
    for engine in _MIRROR_ENGINES:
        for src in idx_char:
            mirrored.append(_mirror_row(src, engine))

    # One chatterbox announcer row off a real male CC0 ref, so announcer-via-
    # chatterbox is not dangling (dia stays char_voice-only this pass).
    #
    # CREATED ONLY WHEN ABSENT, and that is the fix to the second thing this
    # generator used to destroy. This row is a BOOTSTRAP, not a mirror -- it is
    # hand-written here rather than derived from anything -- and the copy on disk
    # had since been re-pointed at a better announcer reference with curated
    # timbre and style tags. Refreshing it "in place" threw that away and put back
    # the literal below. A generator may recreate a row it can derive; it may not
    # overwrite a row somebody improved by hand.
    ann_src = next((v for v in idx_char if v["voice_ref_id"] == _ANNOUNCER_REF_ID), None)
    if ann_src is not None and ("chatterbox", "cb_announcer_male") not in on_disk:
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
            # MERGE, do not REPLACE. The generator owns every field it derives,
            # so those refresh; a field the existing row has and the source does
            # not is somebody else's addition and survives. Replacing outright
            # made "ownership" mean the whole row, which is how a curated field on
            # a generated row disappeared without anything reporting it.
            new_voices.append({**row, **managed[key]})
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
