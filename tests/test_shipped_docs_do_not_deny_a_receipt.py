"""A shipped page must not deny a receipt the matrix records.

THE DEFECT THIS EXISTS FOR, measured 2026-09-19. An outside tester published an
episode from `otr_amd_still` on a Radeon on 2026-09-14. Commit `0fc0fb90` fixed
"six places in the shipped docs" that said AMD had never run. It missed eight
more, across five shipped files and two generators, and they shipped for five
days saying the opposite of `README.md`'s own headline:

    apple/ROCM.md                  seven passages, contradicting its own top
    apple/AGENT_INSTALL.md         "otr_amd_still is draft and has none"
    apple/WRITERS.md               "nobody has published an episode from AMD yet"
    README.md                      "AMD has no receipts yet"
    config/profiles/...json        a display_name PRINTED INTO EVERY RUN'S LOG
    scripts/otr_dropdown_matrix.py "Nothing in this repo has an AMD receipt"
    scripts/otr_tier_matrix.py     "AMD ROCm (experimental -- no receipts)"

The cost was not tidiness. An agent reading the shipped tree twice told the
operator that AMD and Mac were unproven, because a hand-written sentence asserting
PROOF reads as current and carries no date. A procedure that goes stale fails
loudly when someone runs it; a proof claim that goes stale is simply believed.

WHAT THIS CHECKS, and deliberately nothing more: no shipped page may pair a
denial phrase with a machine class that has at least one `proven` cell in
`docs/dropdown_matrix.json`. It is lexical and it is crude. A sentence that
truthfully scopes a gap -- "the 8 GB AMD profile is untested" -- names a tier
rather than the class, and passes. If a legitimate sentence ever trips it, add it
to ALLOWED with the reason, rather than loosening the phrase list: the phrases are
the signal.
"""
from __future__ import annotations

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Files a stranger or an agent reads. `docs/` is excluded from the package
#: (`.comfyignore`), so it is out of scope by the operator's own ruling.
SHIPPED = sorted((REPO / "apple").glob("*.md")) + [REPO / "README.md"]

#: Prose that asserts an ABSENCE of proof. Every one of the fourteen sites above
#: contained one of these.
DENIAL = re.compile(
    r"no receipts?\b|has none\b|never (?:once )?(?:been )?(?:run|executed|touched)"
    r"|nobody has published|never run on|has never|no .{0,16}receipt\b",
    re.IGNORECASE,
)

#: How a machine class is spoken about in prose, mapped to the KEY that class
#: uses inside `dropdown_matrix.json` (`amd`, `mac16`, `nv8`, `nv16`, `cpu`).
CLASS_WORDS = {
    "amd": "amd",
    "radeon": "amd",
    "rocm": "amd",
    "apple silicon": "mac16",
    "mac": "mac16",
}

#: Sentences that trip the rule and are TRUE. Keep the reason with the text.
ALLOWED = (
    # ROCm's own scope note: the two LAB profiles genuinely have no receipt --
    # only `otr_amd_still` does. Keeping them distinct is the proven/measured
    # discipline, not an oversight.
    "still read `UNVERIFIED on hardware` in their own files and genuinely are",
    # The writer table's Llama row. "Nobody has published an episode with it
    # yet" is about that MODEL and is true; the machine words on the line are
    # the model's own fit tags (`mac16 nv8 nv16 nv24`), not a claim about the
    # machine. This is the false positive the rule was expected to produce, and
    # it is cheaper to name than to teach the regex about fit tags.
    "unsloth/Llama-3.2-3B-Instruct",
)


def _proven_classes() -> set:
    """Machine classes with at least one `proven` engine cell."""
    data = json.loads((REPO / "docs" / "dropdown_matrix.json").read_text(encoding="utf-8"))
    out = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and value.strip().lower() == "proven":
                    out.add(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    return out


def test_no_shipped_page_denies_a_receipt_the_matrix_records():
    proven = _proven_classes()
    assert proven, "dropdown_matrix.json recorded no proven cell at all"

    offenders = []
    for path in SHIPPED:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not DENIAL.search(line):
                continue
            if any(ok in line for ok in ALLOWED):
                continue
            lowered = line.lower()
            for word, key in CLASS_WORDS.items():
                if word in lowered and key in proven:
                    offenders.append(
                        "%s:%d %s" % (path.relative_to(REPO), number, line.strip()[:110]))
                    break

    assert not offenders, (
        "a shipped page denies a receipt the generated matrix records -- fix the "
        "page, or add the line to ALLOWED with its reason:\n  " + "\n  ".join(offenders))
