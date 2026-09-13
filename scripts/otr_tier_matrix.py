"""Render docs/TIER_MATRIX.md and the README's tier-matrix block from the
shipping profiles, and check that neither has drifted.

    python scripts/otr_tier_matrix.py          # write both
    python scripts/otr_tier_matrix.py --check  # exit 1 when either is stale

WHY GENERATED. The hand-typed matrix shipped on 2026-09-13 with the acts and
characters wrong on 14 of 18 rows, a lane label the profiles never used, and
four Mac graph names that did not exist -- three drifts from one file that
nobody regenerates. Every cell here is read from config/profiles/<id>.json for
the ids in build_variants.SHIPPING_SET, so the matrix cannot say something
the graphs do not. The README carries the same tables between
`<!-- BEGIN GENERATED: tier-matrix -->` and its END marker.
"""
from __future__ import annotations

import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from build_variants import SHIPPING_SET  # noqa: E402
from otr_dropdown_matrix import packs_by_engine  # noqa: E402

_DOC = os.path.join(_REPO, "docs", "TIER_MATRIX.md")
_README = os.path.join(_REPO, "README.md")
_BEGIN = "<!-- BEGIN GENERATED: tier-matrix -->"
_END = "<!-- END GENERATED: tier-matrix -->"

#: Machine classes in reading order: (graph prefix, heading).
_ARCHES = (
    ("otr_8gb_", "8 GB NVIDIA"),
    ("otr_16gb_", "16 GB NVIDIA"),
    ("otr_mac16_", "Apple Silicon, 16 GB"),
    ("otr_amd_", "AMD ROCm (experimental -- no receipts)"),
    ("otr_cpu_", "CPU only"),
)

#: Tiers in reading order, named by what the episode is made of.
_TIERS = ("low", "still", "video", "foley", "mime", "animatediff")

_COLUMNS = ("tier", "graph", "writer", "quant", "lanes (announcer / music / character)",
            "image", "weights", "also install", "acts", "chars", "status")


def _load_profile(profile_id: str) -> dict:
    path = os.path.join(_REPO, "config", "profiles", "%s.json" % profile_id)
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _internal(engine: str) -> str:
    from nodes._otr_shared.public_engines import resolve_engine_id
    return resolve_engine_id(engine)


def _weights(lanes: list) -> str:
    """auto / manual / none, for the video lanes a graph selects.

    `none` when no lane consumes weights (the procedural visualisers and the
    still lanes), `auto` when every weight-bearing lane is in the pack's own
    auto-fetch set, `manual` otherwise -- the launch recipe next to the graph
    names what to fetch.
    """
    from nodes._otr_visual_assets import _COVERED
    bearing = [_internal(l) for l in lanes
               if not (l.startswith("viz_") or l.startswith("still_"))]
    if not bearing:
        return "none"
    if all(e in _COVERED for e in bearing):
        return "auto"
    return "manual"


def _row(profile_id: str, tier: str, packs: dict) -> list:
    p = _load_profile(profile_id)
    llm = p.get("llm") or {}
    roles = p.get("role_overrides") or {}
    feats = p.get("features") or {}
    lanes = [roles.get(k, "") for k in
             ("announcer_visual", "music_visual", "character_visual")]
    lanes_cell = (lanes[0] if len(set(lanes)) == 1 else " / ".join(lanes))
    image = roles.get("character_image") or "(canonical)"
    # The image pick is dormant when the registry proves every lane on the
    # row mints no still (the visualisers AND the AnimateDiff lanes, which
    # render from the text prompt alone) -- the same proof the preflight
    # uses to skip the download, not a name heuristic.
    from nodes._otr_visual_assets import _proven_no_still
    picked = [l for l in lanes if l]
    if picked and all(_proven_no_still(_internal(l), None) for l in picked):
        image = "none (dormant)"
    also = sorted({packs[_internal(l)] for l in lanes if _internal(l) in packs})
    writer = str(llm.get("creative_model", "(canonical)")).split("/")[-1]
    return [
        "**%s**" % tier,
        "`%s`" % profile_id,
        writer,
        str(llm.get("quant_policy", "(canonical)")),
        lanes_cell,
        image,
        _weights([l for l in lanes if l]),
        ", ".join(also) or "nothing",
        str(feats.get("act_count", "(canonical)")),
        str(feats.get("num_characters", "(canonical)")),
        str(p.get("status", "")),
    ]


def render_tables() -> str:
    """The per-machine tables, one heading per machine class."""
    packs = packs_by_engine()
    shipping = set(SHIPPING_SET)
    out = []
    for prefix, heading in _ARCHES:
        out.append("### %s" % heading)
        out.append("")
        out.append("| " + " | ".join(_COLUMNS) + " |")
        out.append("|" + "---|" * len(_COLUMNS))
        for tier in _TIERS:
            pid = prefix + tier
            if pid in shipping:
                out.append("| " + " | ".join(_row(pid, tier, packs)) + " |")
            else:
                out.append("| %s | _not built_ |%s" % (tier, " |" * (len(_COLUMNS) - 2)))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def render_doc() -> str:
    return (
        "# The shipping set\n"
        "\n"
        "Generated by `scripts/otr_tier_matrix.py` from `config/profiles/` for the\n"
        "ids in `build_variants.SHIPPING_SET`; `--check` fails when this file or the\n"
        "README block is stale. Kokoro voices and MusicGen on every graph, upscaler\n"
        "off, all inherited from the canonical. Weights: `auto` fetch themselves at\n"
        "the first queue, `manual` are listed in the graph's `.launch.md`, `none`\n"
        "means the lanes need no video weights.\n"
        "\n"
        + render_tables()
    )


def _split_readme(text: str):
    b = text.find(_BEGIN)
    e = text.find(_END)
    if b < 0 or e < 0 or e < b:
        raise SystemExit("README.md has no tier-matrix markers -- add %s / %s "
                         "where the tables belong" % (_BEGIN, _END))
    return text[:b + len(_BEGIN)], text[e:]


def readme_with_block(text: str) -> str:
    head, tail = _split_readme(text)
    return head + "\n" + render_tables() + tail


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    doc = render_doc()
    with io.open(_README, encoding="utf-8") as fh:
        readme = fh.read()
    new_readme = readme_with_block(readme)
    if check:
        stale = []
        try:
            with io.open(_DOC, encoding="utf-8") as fh:
                if fh.read() != doc:
                    stale.append("docs/TIER_MATRIX.md")
        except FileNotFoundError:
            stale.append("docs/TIER_MATRIX.md (missing)")
        if new_readme != readme:
            stale.append("README.md tier-matrix block")
        if stale:
            print("STALE: " + ", ".join(stale))
            print("run: python scripts/otr_tier_matrix.py")
            return 1
        print("tier matrix is in sync (%d graphs)" % len(SHIPPING_SET))
        return 0
    with io.open(_DOC, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    if new_readme != readme:
        with io.open(_README, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new_readme)
    print("wrote docs/TIER_MATRIX.md and the README tier-matrix block "
          "(%d graphs)" % len(SHIPPING_SET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
