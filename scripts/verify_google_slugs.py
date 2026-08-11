"""Check this pack's shipped Google model ids against Google's live catalog.

A thin wrapper. Every decision lives in
`nodes/_otr_shared/google_slug_verifier.py`, which is import-safe and fully
tested with injected I/O; this file only supplies the real key, the real HTTP
getter, the real sleeper and the real clock.

    python scripts/verify_google_slugs.py

Exit codes (stable, each covered by a test):
    0  everything shipped was found, or nothing needed checking
    2  shipped concrete ids are ABSENT from a completed catalog
    3  no API key -- not run
    4  auth or transport failure
    5  malformed catalog response

IT NEVER WRITES A DATE BACK. It prints the delta and stops. A date in
`nodes/_otr_slug_provenance.py` means a human looked; a script that stamps them
unattended manufactures the evidence-shaped field this tooling exists to catch.

Not collected by pytest -- the filename is deliberately outside `tests/`.
"""
from __future__ import annotations

import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes import _otr_slug_provenance as prov                    # noqa: E402
from nodes._otr_shared import google_slug_verifier as core        # noqa: E402
from nodes._otr_shared.slug_inventory import collect_inventory    # noqa: E402


def main(argv=None) -> int:
    records = collect_inventory()
    as_of = datetime.date.today()

    # The backlog and the ages print BEFORE the key is resolved, so a box with
    # no credentials still gets the offline half of the report instead of a bare
    # "not run".
    offline = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=None, as_of=as_of,
        is_pointer=prov.is_pointer, unverified_kind=prov.UNVERIFIED,
    )
    print(core.format_report(offline, catalog=None,
                             lane_authority=prov.LANE_AUTHORITY, final=False))

    from nodes._otr_google_api.client import get_json, resolve_api_key

    try:
        key = resolve_api_key()
    except Exception as exc:                                      # noqa: BLE001
        print("\nNO KEY (%s) -- catalog not fetched." % type(exc).__name__)
        return core.EXIT_NOT_RUN
    if not key:
        print("\nNO KEY -- catalog not fetched.")
        return core.EXIT_NOT_RUN

    print("\nFetching Google catalog ...")
    catalog = core.fetch_catalog(
        lambda path: get_json(path, _api_key=key), sleep=time.sleep,
    )
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=catalog, as_of=as_of,
        is_pointer=prov.is_pointer, unverified_kind=prov.UNVERIFIED,
    )
    print(core.format_report(report, catalog=catalog,
                             lane_authority=prov.LANE_AUTHORITY))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
