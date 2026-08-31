#!/usr/bin/env python
"""Generate `docs/MACHINE_MATRIX.md` -- what runs on which machine.

    python scripts/otr_machine_matrix.py            # write the doc
    python scripts/otr_machine_matrix.py --stdout   # print it

WHY THIS IS A MATRIX AND NOT PROSE. Operator, 2026-08-31, after reading the
restructured pod guide: *"i feel its more of a history and not a true guide to
all computers -- they don't need a story, just a matrix of what we think will
work and what has been tested."* He is right. A reader arriving with a card in
their machine wants one row, not a narrative of how the row was discovered.

WHY IT IS GENERATED. Every cell comes from `config/profiles/*.json`, which
already carries `status`, `platform`, `gpu_vendor`, `device_backend` and
`llm.vram_ceiling_gb` alongside the actual engine picks. A hand-written
compatibility table is the single most rot-prone document a project can own; a
generated one cannot disagree with the profiles because it IS the profiles.

THE TWO CONFIDENCE LEVELS ARE NOT THE SAME CLAIM, and the table says which:
  * `shipping` / `draft` come from the profile's own `status` field -- the
    project's standing judgement about whether a combination is ready.
  * PROVEN means an episode actually rendered and published, with the evidence
    named in the notes below the table. That is a much stronger claim than
    `shipping`, and only a handful of rows have it.
Nothing here is inferred from "it looks like it should fit". A blank is an
honest unknown.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: Machine classes are DECLARED in config/machine_classes.json, not here.
#: Facts in a generator rot invisibly -- nobody greps a script for truth -- so
#: the judgement lives in data and this module only validates and renders it.
_CLASS_FILE = os.path.join(_REPO, "config", "machine_classes.json")

#: Not a profile: a mapping config that lives in the same directory and would
#: otherwise render as a "draft profile not vouched for".
_NOT_PROFILES = {"widget_mapping"}


class ClassValidationError(Exception):
    """A declared class contradicts the profile it names.

    Raised rather than warned. A compatibility table that quietly disagrees
    with the profiles is worse than no table: README told users an 8 GB card
    had "rendered nothing" for days after nine episodes published from one.
    """


def load_classes(profiles):
    """Declared classes, each validated against the profile it names."""
    try:
        doc = json.load(io.open(_CLASS_FILE, encoding="utf-8"))
    except OSError:
        return []
    by_id = {p["id"]: p for p in profiles}
    out, problems = [], []
    for row in doc.get("classes", []):
        pid = row.get("recommended")
        if not pid:
            out.append((row, None))          # a DECLARED gap, which is allowed
            continue
        prof = by_id.get(pid)
        if prof is None:
            problems.append("%r names profile %r, which does not exist"
                            % (row.get("label"), pid))
            continue
        want_vendor = row.get("gpu_vendor")
        if want_vendor and prof["vendor"] not in (want_vendor, "?"):
            problems.append("%r is %s but %r declares gpu_vendor %s"
                            % (pid, prof["vendor"], row.get("label"), want_vendor))
        vram = prof.get("vram")
        lo, hi = row.get("vram_min_gb"), row.get("vram_max_gb")
        if vram and lo is not None and hi is not None and not (lo <= float(vram) <= hi):
            problems.append("%r has vram_ceiling_gb %s, outside %r's %s-%s"
                            % (pid, vram, row.get("label"), lo, hi))
        out.append((row, prof))
    if problems:
        raise ClassValidationError(
            "config/machine_classes.json contradicts the profiles:\n  - "
            + "\n  - ".join(problems))
    return out

#: Proof now lives on the PROFILE, in its `proven` list, not here. It is a fact
#: about the profile, it accrues per-hardware (the haunted lane is proven on 8,
#: 16 and 24 GB), and under the two-box ownership split the machine that proved
#: it owns that file anyway. This module only reads and renders it.
#:
#: Keeping it honest matters more than keeping it full: PROVEN is the difference
#: between "we think" and "we know", and a padded list destroys the only reason
#: the table is worth reading.
def merged_row(row):
    """defaults <- row, through the ONE merge the profile builder uses.

    Reused from otr_machine_profile rather than repeated here: a second
    merge would be a second definition of what a machine class means, and
    the doc and the applied profile could then disagree without either
    being wrong on its own terms.
    """
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from otr_machine_profile import _merged, load_matrix
    return _merged(load_matrix(), row)


def proven_receipts(obj) -> list:
    """Proof receipts, read from a MATRIX ROW.

    Deliberately not from a profile file. Operator, 2026-08-31: "no profiles,
    no proven in the code, they just live there" -- there being
    config/machine_classes.json. Putting receipts on profiles also broke
    `build_variants --check` on an unknown-key guard, which was that guard
    correctly refusing a shape nobody had declared.
    """
    return [r for r in ((obj or {}).get("proven") or []) if isinstance(r, dict)]


def proven_summary(prof) -> str:
    rows = proven_receipts(prof)
    if not rows:
        return ""
    total = sum(int(r.get("episodes") or 0) for r in rows)
    hw = ", ".join(str(r.get("hardware", "?")) for r in rows)
    return "%d episode(s) on %s" % (total, hw)

#: Measured peaks worth stating, and the exact conditions. A VRAM number without
#: its conditions is how somebody buys the wrong card.
MEASURED = [
    ("animatediff15_v3_haunted_video", "1 act, 8 clips, 24 GB rented card",
     "2058 s, peak 15,990 MB, published"),
    ("ltx25_high_video", "16 GB card", "peak 14.48 GiB -- 5080-class only"),
    ("minimax_h3_video", "16 GB card clamped to --reserve-vram 12",
     "peak 7.28 GiB VRAM, 27.56 GiB HOST RAM, cold pass"),
]


def load_profiles() -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(_REPO, "config/profiles/*.json"))):
        try:
            d = json.load(io.open(path, encoding="utf-8"))
        except Exception:
            continue
        ro = d.get("role_overrides", {}) or {}
        so = d.get("slot_overrides", {}) or {}
        out.append({
            "id": d.get("id") or os.path.basename(path)[:-5],
            "status": d.get("status", "?"),
            "vram": (d.get("llm", {}) or {}).get("vram_ceiling_gb"),
            "vendor": d.get("gpu_vendor", "?"),
            "backend": d.get("device_backend", "?"),
            "platform": d.get("platform", "any"),
            "video": ro.get("character_visual") or so.get("video_render_engine") or "-",
            "image": ro.get("character_image") or "-",
            "voice": so.get("char_voice_engine") or "-",
            "music": so.get("music_engine") or "-",
            "proven": d.get("proven") or [],
        })
    return out


def _proven_ids(classes) -> set:
    """Profile ids a matrix row vouches for. Empty now that classes are
    self-contained, kept so the per-tier tables still mark a proven row."""
    return {row.get("recommended") for row, _p in (classes or [])
            if row.get("proven") and row.get("recommended")}


def _tier(vram):
    """Two tiers, deliberately.

    Splitting 16 / 24 / 32 GB invents a distinction the profiles do not make:
    nothing declares a ceiling above 16, so a 24 GB card runs exactly what a
    16 GB card runs and a separate row for it is an empty promise. Operator,
    2026-08-31: *"i'd rather just a 16+ tier."*

    The real question a bigger card raises is not "which tier" but "does
    anything here actually SPEND the extra memory" -- which is a profile
    question, called out under the table rather than faked as a tier.
    """
    if not vram:
        return "unstated"
    return "8 GB" if float(vram) <= 9 else "16 GB+"


_ORDER = ["8 GB", "16 GB+", "unstated"]


def render() -> str:
    profs = load_profiles()
    L = []
    A = L.append
    A("# Machine matrix -- what runs where\n")
    A("**GENERATED by `scripts/otr_machine_matrix.py` from "
      "`config/profiles/*.json`. Do not hand-edit; regenerate.**\n")
    A("Find your machine in the first table. That is the whole answer; "
      "everything after it is detail you only need if the answer is no.\n")

    # ---- the one table anybody actually needs ------------------------------
    classes = load_classes(profs)
    A("## What works on what machine\n")
    A("| your machine | writer | video | voice | music | image | status |")
    A("|---|---|---|---|---|---|---|")
    for row, prof in classes:
        label = row.get("label", "?")
        # Read the ROW, never a profile it may not name. Every machine value
        # lives in config/machine_classes.json now (operator, 2026-08-31: "no
        # profiles in the code"), and no row carries `recommended`, so the old
        # profile-linked branch printed "no profile yet" for ALL FOUR classes
        # while the matrix held real, proven values. The front-door table of
        # the guide told every newcomer that nothing runs anywhere.
        cells = merged_row(row)
        conf = ("**PROVEN** -- %s" % proven_summary(row)
                if proven_receipts(row)
                else "`%s`, unproven" % cells.get("status", "draft"))
        A("| **%s** | %s | %s | %s | %s | %s | %s |" % (
            label, cells.get("writer", "--"), cells.get("video", "--"),
            cells.get("char_voice", "--"), cells.get("music", "--"),
            cells.get("image", "--"), conf))
    A("")
    A("**Use the profile named for your machine** -- pass it to `--profile`, or "
      "pick the matching entries in the dropdowns. The engine names above are "
      "exactly the dropdown text.\n")
    for row, prof in classes:
        if prof is not None:
            A("* **%s** -> `%s`" % (row.get("label"), prof["id"]))
            if row.get("note"):
                A("  %s" % row["note"])
    A("")

    A("## How to read the confidence column\n")
    A("| value | means |")
    A("|---|---|")
    A("| **PROVEN** | an episode actually rendered and published. Evidence named below. |")
    A("| `shipping` | the profile is considered ready. Not the same as proven. |")
    A("| `draft` | exists, not vouched for. Try it; expect to debug. |")
    A("")
    A("Nothing here is inferred from \"it looks like it should fit\". A blank is "
      "an unknown, recorded as one.\n")

    by = {}
    for p in profs:
        by.setdefault(_tier(p["vram"]), []).append(p)

    for tier in _ORDER:
        rows = by.get(tier) or []
        if not rows:
            continue
        prov = sum(1 for r in rows if r["id"] in _proven_ids(classes))
        ship = sum(1 for r in rows if r["status"] == "shipping")
        A("## %s  --  %d profile(s), %d shipping, %d proven\n"
          % (tier, len(rows), ship, prov))
        # Only PROVEN and shipping rows are tabled. A tier holding 76 drafts is
        # a dump, not a guide: a reader picking a row cannot tell which of 76 to
        # trust. Drafts are counted and folded away, listed by the engine they
        # select, which is the only thing anyone scans them for.
        headline = [r for r in rows
                    if r["id"] in _proven_ids(classes) or r["status"] == "shipping"]
        drafts = [r for r in rows if r not in headline]
        if headline:
            A("| profile | video | voice | music | image | confidence |")
            A("|---|---|---|---|---|---|")
            for r in sorted(headline,
                            key=lambda x: (x["id"] not in _proven_ids(classes), x["id"])):
                conf = "**PROVEN**" if r["id"] in _proven_ids(classes) else "`%s`" % r["status"]
                A("| `%s` | %s | %s | %s | %s | %s |" % (
                    r["id"], r["video"], r["voice"], r["music"], r["image"],
                    conf))
            A("")
        else:
            A("**No shipping or proven profile at this tier.**\n")
        if drafts:
            eng = sorted({r["video"] for r in drafts})
            A("<details><summary>%d draft profile(s) here -- not vouched for"
              "</summary>\n" % len(drafts))
            A("Video engines they select: %s\n"
              % ", ".join("`%s`" % e for e in eng))
            A("| profile | video | voice |")
            A("|---|---|---|")
            for r in sorted(drafts, key=lambda x: x["id"]):
                A("| `%s` | %s | %s |" % (r["id"], r["video"], r["voice"]))
            A("\n</details>\n")

    A("## A bigger card does not currently get you more\n")
    A("The tier is `16 GB+` because that is the truth: nothing in "
      "`config/profiles/` declares a VRAM ceiling above 16, so a 24 GB or "
      "32 GB card runs exactly what a 16 GB one runs.\n")
    A("**A draft now exists that asks for more.** `otr_rented24_heavy` raises "
      "the writer ceiling 14.5 -> 22 GB and drops NF4 quantisation so the 12b "
      "writer sits resident, lifts the render frame cap 81 -> 121, and uses "
      "`indextts2` cloning for BOTH character and announcer where the 16 GB "
      "tier settles for kokoro on the announcer. It is `draft` and nothing has "
      "published from it -- proof follows the profile, never the reverse, "
      "because a lane is proven BY running under one.\n")
    A("**That matters when you are paying by the hour.** A rented 24 GB card "
      "ran the 16 GB haunted profile and peaked at 15,990 MB -- two thirds of "
      "the memory it was being billed for. A bigger card earns its cost only by "
      "running what a smaller one cannot: the large video lanes, voice cloning "
      "rather than the cheap fallback, the better image engines. None of that "
      "is proven yet, so this is a note about what to build, not a setting to "
      "change.\n")

    A("## The proven rows, and what proves them\n")
    for prof in sorted(profs, key=lambda x: x["id"]):
        for r in proven_receipts(prof):
            A("* **`%s`** on %s -- %s episode(s), %s. %s"
              % (prof["id"], r.get("hardware", "?"),
                 r.get("episodes", "?"), r.get("date", "?"),
                 r.get("evidence", "")))
    A("")
    A("## Measured peaks, with their conditions\n")
    A("A VRAM number without its conditions is how somebody buys the wrong card.\n")
    A("| engine | conditions | measured |")
    A("|---|---|---|")
    for eng, cond, meas in MEASURED:
        A("| `%s` | %s | %s |" % (eng, cond, meas))
    A("")
    A("### Read every VRAM number with suspicion\n")
    A("**A peak is mostly what the allocator GRABBED, not what the model "
      "NEEDED.** These cards are built for games: they take as much as is "
      "going, most of them grab close to the maximum available AT LOAD, and "
      "they recover from pressure rather than refusing. Torch's caching "
      "allocator then never hands it back. So a run that peaked at 15,990 MB "
      "on a 24 GB card is NOT evidence it needs 16 GB -- the same graph under "
      "`--reserve-vram` fits in far less, which is exactly how the H3 row "
      "below was measured.\n")
    A("The honest use of these numbers is COMPARATIVE -- which lane is "
      "heavier than which -- not a shopping threshold. A lane marked "
      "PROVEN on 8 GB is worth more than any peak figure, because a card "
      "actually did it.\n")
    A("**Host RAM is the limit people miss.** The clamped H3 run peaked at "
      "27.56 GiB of SYSTEM memory against 7.28 GiB of VRAM. A machine with "
      "16 GiB of RAM will struggle regardless of its GPU, and no VRAM table "
      "warns you.\n")
    A("## What is NOT here\n")
    A("* No row means an episode will look good -- only that the combination "
      "loads and runs. Output quality is a separate judgement.\n")
    A("* A `draft` row is not a promise. Most have never been run end to end.\n")
    A("* Non-NVIDIA is largely unexplored. Profiles exist "
      "(`otr_amd8_rocm`); none is proven.\n")
    A("* Where a lane needs weights, see "
      "[MODEL_ASSET_INDEX.md](MODEL_ASSET_INDEX.md) for the exact files and "
      "where to get them. That file is generated too, and is the only place "
      "file names and sizes are recorded.\n")
    return "\n".join(L) + "\n"


_BEGIN = "<!-- BEGIN GENERATED: machine-matrix -->"
_END = "<!-- END GENERATED: machine-matrix -->"


def headline_block() -> str:
    """Just the class table -- what gets injected into README.

    README is the universal front door and already answers "what do I run on my
    machine". Two hand-eyed surfaces answering one question is how it came to
    claim an 8 GB card had "rendered nothing" while nine episodes had published
    from one. So the answer is generated ONCE and injected, rather than told
    twice.
    """
    full = render()
    start = full.index("## What works on what machine")
    end = full.index("## How to read the confidence column")
    return full[start:end].rstrip() + "\n"


def inject_readme(check_only: bool = False) -> bool:
    """Put the class table into README between markers. True if in sync."""
    path = os.path.join(_REPO, "README.md")
    s = io.open(path, encoding="utf-8").read()
    block = _BEGIN + "\n\n" + headline_block() + "\n" + _END
    if _BEGIN in s and _END in s:
        head, rest = s.split(_BEGIN, 1)
        _stale, tail = rest.split(_END, 1)
        out = head + block + tail
    else:
        hook = "## Which video models fit your card"
        if hook not in s:
            print("  README hook not found; nothing injected")
            return True
        out = s.replace(hook, block + "\n\n" + hook, 1)
    if out == s:
        return True
    if check_only:
        return False
    io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="fail if the doc or README block is out of date; "
                         "writes nothing. For CI and the test suite.")
    args = ap.parse_args(argv)
    text = render()
    dest = os.path.join(_REPO, "docs", "MACHINE_MATRIX.md")

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        try:
            on_disk = io.open(dest, encoding="utf-8").read()
        except OSError:
            on_disk = ""
        doc_ok = on_disk == text
        readme_ok = inject_readme(check_only=True)
        if doc_ok and readme_ok:
            print("  in sync")
            return 0
        if not doc_ok:
            print("  STALE: docs/MACHINE_MATRIX.md differs from the profiles")
        if not readme_ok:
            print("  STALE: README's machine-matrix block differs")
        print("  run: python scripts/otr_machine_matrix.py")
        return 1

    io.open(dest, "w", encoding="utf-8", newline="\n").write(text)
    inject_readme()
    print("wrote %s (%d bytes) and injected the README block" % (dest, len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
