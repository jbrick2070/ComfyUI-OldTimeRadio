"""The spoken-citation audit must actually catch the defect it was written for.

An audit script nobody calls is how this bug survived in the first place:
`provenance_coda_line` was composed, stamped, and read by nothing for months.
So the predicate gets its own tests, and they assert it FIRES -- not merely that
it runs.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_spoken_citations.py"

sys.path.insert(0, str(REPO / "scripts"))
import audit_spoken_citations as AUDIT  # noqa: E402

sys.path.insert(0, str(REPO))
from nodes import _otr_ledger as _OTRL  # noqa: E402

#: The complete pre-l4 lineage, from `_otr_ledger.CURRENT_SCHEMA_VERSION`'s own
#: docstring. Every one of these must sort as legacy: a ledger written before
#: the receipt existed cannot be faulted for lacking it.
HISTORICAL_VERSIONS = (
    "l1-2026-04-24",
    "l2-2026-04-25",
    "l3-2026-04-28",
    "l3-2026-05-02",
    "l3-2026-05-08",
    "l3-2026-05-14",
)


FOLGER_RIGHTS = {
    "source_label": "Folger Shakespeare",
    "license_label": "CC BY-NC 3.0 (Folger Shakespeare Library)",
    "source_url": "https://www.folger.edu/explore/shakespeares-works/macbeth/",
    "license_url": "https://creativecommons.org/licenses/by-nc/3.0/",
}

#: The public-domain rights sidecar carries NO license_label -- see
#: `_otr_public_domain_sources.source_rights_from_unit`. That empty value is the
#: trap the predicate has to survive.
GUTENBERG_RIGHTS = {
    "source_label": "Project Gutenberg",
    "license_status": "public_domain_us",
    "source_url": "https://www.gutenberg.org/files/1342/pg1342.txt",
    "license_url": "",
}


def _ledger(rights, *texts, schema="l3-2026-05-14", provenance=None,
            coda_source=None):
    meta = {"source_rights": dict(rights)}
    if provenance is not None:
        meta["provenance"] = provenance
    if coda_source is not None:
        meta["spoken_coda_source"] = coda_source
    return {
        "schema_version": schema,
        "meta": meta,
        "lines": [
            {"line_id": "l%03d" % i, "speaker_role": "announcer", "text": t}
            for i, t in enumerate(texts, 1)
        ],
    }


class TestForbiddenValues:
    def test_the_empty_licence_label_is_dropped(self):
        """An empty needle is a substring of every string.

        The public-domain sidecar has no `license_label`, so `normalize_provenance`
        yields "". Left in, it would match every line in the corpus and the audit
        would report the entire library as leaking -- measuring nothing.
        """
        needles = AUDIT.forbidden_values(_ledger(GUTENBERG_RIGHTS, "A quiet line."))
        assert "" not in needles
        assert all(n.strip() for n in needles)

    def test_a_clean_line_survives_the_empty_label_trap(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "The traveller returned, limping.")
        violations, _ = AUDIT.audit_ledger(ledger)
        assert violations == []

    def test_the_hostname_is_derived_from_the_url(self):
        needles = AUDIT.forbidden_values(_ledger(FOLGER_RIGHTS, "x"))
        assert "www.folger.edu" in needles


class TestItCatchesTheRealShippedLeaks:
    """Each text below is the shape of a line that actually went to air."""

    @pytest.mark.parametrize("rights,text", [
        (GUTENBERG_RIGHTS,
         "This has been SIGNAL LOST. Adapted from H.G. Wells' 'The Time Machine',"
         " published in 1895 and available at https://www.gutenberg.org/x."),
        (FOLGER_RIGHTS,
         "From tonight's echoing halls: Source: Folger Shakespeare. Date/Rights:"
         " c. 1606 | CC BY-NC 3.0. URL: https://www.folger.edu/x"),
        # The SHORT licence form. The stored label is the full
        # "CC BY-NC 3.0 (Folger Shakespeare Library)", so exact-substring on the
        # sidecar alone misses this -- which is how it shipped.
        (FOLGER_RIGHTS,
         "Beyond tonight's garden: Folger Shakespeare, CC BY-NC 3.0"),
        # Spelled out, as the TTS-facing text actually rendered it.
        (FOLGER_RIGHTS,
         "From tonight's enchanted isle: Folger Shakespeare Library,"
         " noncommercial use, CC BY-NC three"),
    ])
    def test_a_spoken_citation_is_a_violation(self, rights, text):
        ledger = _ledger(rights, "Clean opening line.", text)
        violations, _ = AUDIT.audit_ledger(ledger)
        assert violations, "audit missed a citation that shipped: %r" % text

    @pytest.mark.parametrize("coda", [
        "Tonight's tale was adapted from a work in the public domain.",
        "Tonight's tale was adapted from a work dedicated to the public domain.",
        "Tonight's tale was adapted from Folger Shakespeare.",
        "Tonight's tale was an original work, created for this broadcast.",
    ])
    def test_the_deterministic_coda_itself_is_never_a_violation(self, coda):
        """The fix must pass its own audit.

        Every line here is real output of `_otr_provenance.spoken_coda_line`. An
        audit that flags the correct replacement would block the fix instead of
        the defect -- note "public domain" is deliberately NOT a forbidden token.
        """
        ledger = _ledger(FOLGER_RIGHTS, "A character speaks.", coda)
        violations, _ = AUDIT.audit_ledger(ledger)
        assert violations == [], "the clean coda was flagged: %r" % coda

    def test_a_bare_url_is_caught_even_without_a_matching_needle(self):
        ledger = _ledger({}, "Read more at https://example.org/thing.")
        violations, _ = AUDIT.audit_ledger(ledger)
        assert violations and "<bare-url>" in violations[0]["matched"]

    def test_our_own_prompt_scaffold_is_caught(self):
        """A label-only echo must fail even when no URL survives."""
        ledger = _ledger({}, "Source: Folger Shakespeare. Date/Rights: c. 1606.")
        violations, _ = AUDIT.audit_ledger(ledger)
        assert violations

    def test_skipped_and_non_speech_rows_are_ignored(self):
        ledger = _ledger(FOLGER_RIGHTS)
        ledger["lines"] = [
            {"line_id": "m1", "speaker_role": "music_open",
             "text": "https://www.folger.edu/x"},
            {"line_id": "s1", "speaker_role": "announcer", "skip": True,
             "text": "https://www.folger.edu/x"},
        ]
        violations, _ = AUDIT.audit_ledger(ledger)
        assert violations == [], "a muted or non-spoken row is not a spoken leak"


class TestTheReceiptBoundary:
    """Absence of the receipt is history on a legacy ledger and a REGRESSION on a
    current one. Without that boundary a build that silently stopped stamping it
    would look exactly like the 1,587 ledgers that never could."""

    def test_a_legacy_ledger_without_the_receipt_is_not_a_failure(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.",
                         schema="l3-2026-05-14",
                         provenance={"status": "public_domain_us"})
        _, receipt_errors = AUDIT.audit_ledger(ledger)
        assert receipt_errors == []

    def test_a_current_provenance_ledger_must_carry_the_receipt(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.",
                         schema="l4-2999-01-01",
                         provenance={"status": "public_domain_us"})
        _, receipt_errors = AUDIT.audit_ledger(ledger)
        assert receipt_errors, "a dropped receipt must not pass as legacy"

    def test_a_receipt_outside_the_closed_vocabulary_fails(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.",
                         schema="l4-2999-01-01",
                         provenance={"status": "public_domain_us"},
                         coda_source="whatever_i_felt_like")
        _, receipt_errors = AUDIT.audit_ledger(ledger)
        assert receipt_errors

    def test_a_valid_current_ledger_passes_clean(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.",
                         schema="l4-2999-01-01",
                         provenance={"status": "public_domain_us"},
                         coda_source="provenance")
        violations, receipt_errors = AUDIT.audit_ledger(ledger)
        assert violations == [] and receipt_errors == []


class TestTheBoundaryIsWiredToTheREALVersion:
    """The class above proves the boundary WORKS, using an invented
    `l4-2999-01-01`. It cannot prove the boundary is drawn in the right PLACE.

    That is the #1 trap of this bump: a completionist find-replace on the
    l3 literal would put the REAL current version inside
    `LEGACY_SCHEMA_VERSIONS`, which un-legacies 1,587 historical ledgers and
    legacies every post-bump one -- inverting the audit completely. Every test
    above would stay green through it, because they hardcode a version string
    the production constant never mentions.

    So these assertions read the shipped constant instead.
    """

    def test_the_current_version_is_not_legacy(self):
        assert _OTRL.CURRENT_SCHEMA_VERSION not in AUDIT.LEGACY_SCHEMA_VERSIONS, (
            "the CURRENT version landed in the legacy set -- the boundary is "
            "inverted and the receipt requirement can never fire again")

    def test_the_current_version_requires_the_receipt(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.",
                         schema=_OTRL.CURRENT_SCHEMA_VERSION,
                         provenance={"status": "public_domain_us"})
        _, receipt_errors = AUDIT.audit_ledger(ledger)
        assert receipt_errors, (
            "a provenance-owned ledger at the CURRENT version dropped its "
            "receipt and the audit said nothing")

    def test_the_current_version_accepts_a_stamped_receipt(self):
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.",
                         schema=_OTRL.CURRENT_SCHEMA_VERSION,
                         provenance={"status": "public_domain_us"},
                         coda_source="provenance")
        violations, receipt_errors = AUDIT.audit_ledger(ledger)
        assert violations == [] and receipt_errors == []

    @pytest.mark.parametrize("schema", HISTORICAL_VERSIONS)
    def test_every_historical_version_is_legacy(self, schema):
        assert schema in AUDIT.LEGACY_SCHEMA_VERSIONS, (
            f"{schema} is a real lineage version; ledgers written at it "
            "predate the receipt and must not be faulted for lacking it")
        ledger = _ledger(GUTENBERG_RIGHTS, "A clean line.", schema=schema,
                         provenance={"status": "public_domain_us"})
        _, receipt_errors = AUDIT.audit_ledger(ledger)
        assert receipt_errors == [], f"{schema} was audited as current"

    def test_the_legacy_set_carries_no_version_that_never_existed(self):
        """It listed "l2-2026-05-02" -- l2-2026-04-25 and l3-2026-05-02 mashed
        together. Harmless in effect, wrong in fact, and the kind of wrong that
        hides a real gap: four genuine versions were missing beside it."""
        assert set(AUDIT.LEGACY_SCHEMA_VERSIONS) == set(HISTORICAL_VERSIONS)

    def test_the_previous_current_version_stays_legacy(self):
        """l3-2026-05-14 is the string that HOLDS the boundary. Replacing it
        with the new value during a sweep is the inversion this class exists
        to catch."""
        assert "l3-2026-05-14" in AUDIT.LEGACY_SCHEMA_VERSIONS


class TestTheCommandLineContract:
    """Exit codes match scripts/audit_otr_full_run.py: 0 clean, 1 violations,
    2 operational."""

    def _run(self, root):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root), "--json"],
            capture_output=True, text=True, cwd=str(REPO),
        )

    def test_clean_tree_exits_zero(self, tmp_path):
        ep = tmp_path / "ep_a" / "audio"
        ep.mkdir(parents=True)
        (ep / "ep_a_ledger.json").write_text(
            json.dumps(_ledger(GUTENBERG_RIGHTS, "A quiet line.")),
            encoding="utf-8")
        result = self._run(tmp_path)
        assert result.returncode == AUDIT.EXIT_CLEAN, result.stdout + result.stderr

    def test_leaking_tree_exits_one(self, tmp_path):
        ep = tmp_path / "ep_b" / "audio"
        ep.mkdir(parents=True)
        (ep / "ep_b_ledger.json").write_text(
            json.dumps(_ledger(
                FOLGER_RIGHTS,
                "Adapted from Folger Shakespeare, CC BY-NC 3.0")),
            encoding="utf-8")
        result = self._run(tmp_path)
        assert result.returncode == AUDIT.EXIT_VIOLATIONS
        payload = json.loads(result.stdout)
        assert payload["findings"][0]["episode"].endswith("ep_b_ledger.json")

    def test_a_missing_root_is_operational_not_clean(self, tmp_path):
        result = self._run(tmp_path / "does_not_exist")
        assert result.returncode == AUDIT.EXIT_OPERATIONAL

    def test_output_order_is_deterministic(self, tmp_path):
        for name in ("ep_c", "ep_a", "ep_b"):
            ep = tmp_path / name / "audio"
            ep.mkdir(parents=True)
            (ep / ("%s_ledger.json" % name)).write_text(
                json.dumps(_ledger(FOLGER_RIGHTS, "Folger Shakespeare, CC BY-NC 3.0")),
                encoding="utf-8")
        first = json.loads(self._run(tmp_path).stdout)["findings"]
        second = json.loads(self._run(tmp_path).stdout)["findings"]
        assert [r["episode"] for r in first] == [r["episode"] for r in second]
        assert [r["episode"] for r in first] == sorted(r["episode"] for r in first)
