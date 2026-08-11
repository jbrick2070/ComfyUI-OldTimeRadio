"""Pure core for the Google slug verifier. Import-safe; does no I/O of its own.

WHY A CORE AND A SCRIPT. Two rules were in tension: "no test may import the
verifier script" (a script that runs work at import time is a landmine in a test
suite) and "test pagination, retry, every exit code, and prove zero network".
Splitting them settles it -- everything decidable lives here, behind injected
`get_json` / `resolve_key` / `sleep` / `as_of`, and `scripts/verify_google_slugs.py`
is a thin `main()` that supplies the real ones. The tests import ONLY this module
and inject fakes, so they exercise the real logic and touch no socket.

WHAT IT WILL NOT DO. It never writes a date back into
`nodes/_otr_slug_provenance.py`. A date is a human claim that someone looked; a
script stamping dates unattended manufactures precisely the evidence-shaped field
this whole chunk was written to kill. It prints the delta; a human commits it.

AGE NEVER FAILS. A calendar-triggered red on an offline-first pack would fire on
a fresh clone in 2027, on a box nobody had touched, and the first thing anyone
would do is delete the check. Both review lanes independently proposed a hard
failure, which makes it the intuitive answer rather than the right one. Age is
REPORTED. Future-dated entries DO fail, because that is malformed evidence rather
than old evidence.
"""
from __future__ import annotations

import datetime
import urllib.parse
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

# Stable, tested exit codes. Callers switch on these, so they may not drift.
EXIT_OK = 0
EXIT_SHIPPED_IDS_MISSING = 2
EXIT_NOT_RUN = 3
EXIT_AUTH_OR_TRANSPORT = 4
EXIT_MALFORMED_CATALOG = 5

#: Three TOTAL attempts, matching the client's own DEFAULT_MAX_RETRIES = 2.
MAX_ATTEMPTS = 3

PAGE_SIZE = 200
_MODELS_PREFIX = "models/"


class CatalogFetch(NamedTuple):
    """The result of trying to list a catalog.

    ``complete`` is the field everything else hinges on. Only a run that reached
    a terminal page -- one with no ``nextPageToken`` -- may authorize a "missing"
    verdict. A partial fetch that happens to omit an id proves nothing, and
    reporting it as missing would be a confident wrong answer.
    """

    ids: frozenset
    complete: bool
    pages: int
    error: Optional[str] = None
    exit_code: int = EXIT_OK


class VerifierReport(NamedTuple):
    missing: Tuple[Tuple[str, str], ...]
    orphaned: Tuple[Tuple[str, str], ...]
    unverified: Tuple[Tuple[str, str], ...]
    pointers: Tuple[str, ...]
    ages: Tuple[Tuple[Tuple[str, str], int], ...]
    future_dated: Tuple[Tuple[str, str], ...]
    exit_code: int


def normalize_model_name(name) -> Optional[str]:
    """Catalog name -> bare id, or None if it is not a usable name.

    Strips EXACTLY ONE ``models/`` prefix. A blank or non-string entry is
    rejected rather than coerced: a catalog that returns junk is malformed, and
    silently dropping it would let a partial answer look complete.
    """
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    if stripped.startswith(_MODELS_PREFIX):
        stripped = stripped[len(_MODELS_PREFIX):]
    return stripped or None


def _is_retryable(exc: BaseException) -> bool:
    """Transport / 429 / 5xx only.

    NEVER auth, and never a malformed response. Retrying a 401 three times turns
    one clear failure into three, and retrying malformed JSON cannot fix it.

    THE CLIENT HAS ALREADY DECIDED THIS, so read its answer instead of inventing
    a second one. `_attach_evidence` (`_otr_google_api/client.py`) stamps
    `retryable` and `http_status` on every exception it raises, from one
    taxonomy. An earlier cut of this function read a `status` attribute that no
    exception in this codebase carries, so `getattr(..., None)` was always None,
    every failure looked like a transport blip, and auth errors were retried
    three times -- the exact thing the docstring above promises not to do. Two
    classifications that must agree, with nothing forcing them to, is the defect
    this whole chunk was written to kill; it is not repeated inside it.
    """
    decided = getattr(exc, "retryable", None)
    if isinstance(decided, bool):
        return decided

    status = getattr(exc, "http_status", getattr(exc, "status", None))
    if status is None:
        return True                       # transport-level: worth one more go
    try:
        status = int(status)
    except (TypeError, ValueError):
        return False
    return status == 429 or 500 <= status < 600


def fetch_catalog(get_json: Callable[[str], dict], *,
                  sleep: Callable[[float], None] = lambda _s: None,
                  page_size: int = PAGE_SIZE,
                  max_pages: int = 50) -> CatalogFetch:
    """Page the model list. All I/O arrives through ``get_json``.

    ``pageSize`` is held across EVERY request, including token-bearing ones --
    dropping it mid-walk silently changes the page shape and can turn one walk
    into two different listings.
    """
    ids: set = set()
    token = ""
    pages = 0

    while True:
        path = "/v1beta/models?pageSize=%d" % int(page_size)
        if token:
            path += "&pageToken=" + urllib.parse.quote(token, safe="")

        payload = None
        last_exc: Optional[BaseException] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                payload = get_json(path)
                last_exc = None
                break
            except Exception as exc:      # noqa: BLE001 -- classified below
                last_exc = exc
                if not _is_retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                    break
                sleep(0.5 * (2 ** attempt))

        if last_exc is not None:
            # EXHAUSTION IS NOT "MISSING". The ids were never looked at; saying
            # they are absent would invent a fact out of a network problem.
            return CatalogFetch(frozenset(ids), False, pages,
                                error="%s: %s" % (type(last_exc).__name__, last_exc),
                                exit_code=EXIT_AUTH_OR_TRANSPORT)

        if not isinstance(payload, dict):
            return CatalogFetch(frozenset(ids), False, pages,
                                error="response is %s, not an object"
                                      % type(payload).__name__,
                                exit_code=EXIT_MALFORMED_CATALOG)
        models = payload.get("models")
        if not isinstance(models, list):
            return CatalogFetch(frozenset(ids), False, pages,
                                error="'models' is %s, not a list"
                                      % type(models).__name__,
                                exit_code=EXIT_MALFORMED_CATALOG)

        for entry in models:
            name = entry.get("name") if isinstance(entry, dict) else None
            norm = normalize_model_name(name)
            if norm:
                ids.add(norm)

        pages += 1
        token = payload.get("nextPageToken") or ""
        if not token:
            return CatalogFetch(frozenset(ids), True, pages)
        if pages >= max_pages:
            # A token that never terminates. Incomplete, and never "missing".
            return CatalogFetch(frozenset(ids), False, pages,
                                error="exceeded %d pages without a terminal page"
                                      % max_pages,
                                exit_code=EXIT_MALFORMED_CATALOG)


def age_in_days(when: Optional[str], as_of: datetime.date) -> Optional[int]:
    """Days between a recorded date and ``as_of``; negative means FUTURE-dated.

    ``as_of`` is injected. Reading the clock inside a pure function is what makes
    a test a time bomb, and this project has enough of those.
    """
    if not when:
        return None
    try:
        recorded = datetime.date.fromisoformat(when)
    except (TypeError, ValueError):
        return None
    return (as_of - recorded).days


def build_report(records: Sequence, provenance: Dict, *,
                 catalog: Optional[CatalogFetch],
                 as_of: datetime.date,
                 is_pointer: Callable[[str], bool],
                 unverified_kind: str) -> VerifierReport:
    """Classify everything. Pure -- no clock, no network, no printing.

    THE THREE BUCKETS ARE NOT THE SAME QUESTION, and conflating them is how a
    guard starts lying:

    * ``missing``            shipped concrete ids absent from a COMPLETED
                             catalog. Requires completeness; without it, empty.
    * ``provenance-orphaned``provenance identities the inventory no longer ships.
    * unshipped catalog entries are NEITHER -- Google listing a model this pack
      does not offer is not a finding about this pack.
    """
    shipped_pairs = set()
    pointers = set()
    for rec in records:
        if is_pointer(rec.provider_id):
            pointers.add(rec.provider_id)
            continue
        shipped_pairs.add((rec.provider_id, rec.authority_lane))

    missing: List[Tuple[str, str]] = []
    exit_code = EXIT_OK
    if catalog is not None and catalog.error:
        exit_code = catalog.exit_code
    elif catalog is not None and catalog.complete:
        for rec in records:
            if is_pointer(rec.provider_id) or not rec.catalog_target:
                continue
            if rec.provider_id not in catalog.ids:
                pair = (rec.provider_id, rec.authority_lane)
                if pair not in missing:
                    missing.append(pair)
        if missing:
            exit_code = EXIT_SHIPPED_IDS_MISSING

    orphaned = sorted(set(provenance) - shipped_pairs)
    unverified = sorted(
        key for key, rec in provenance.items()
        if rec.verification_kind == unverified_kind
    )

    ages: List[Tuple[Tuple[str, str], int]] = []
    future: List[Tuple[str, str]] = []
    for key, rec in sorted(provenance.items()):
        days = age_in_days(rec.verified_on, as_of)
        if days is None:
            continue
        if days < 0:
            future.append(key)            # malformed evidence, not old evidence
        else:
            ages.append((key, days))

    return VerifierReport(
        missing=tuple(sorted(missing)),
        orphaned=tuple(orphaned),
        unverified=tuple(unverified),
        pointers=tuple(sorted(pointers)),
        ages=tuple(ages),
        future_dated=tuple(sorted(future)),
        exit_code=exit_code,
    )


#: What each exit code MEANS, in the words a person reading a terminal needs.
#: A non-zero exit reads as "the tool broke" unless the tool says otherwise, and
#: exit 2 here means the opposite -- the tool worked and found something.
EXIT_MEANING = {
    EXIT_OK: "OK -- every shipped id was found, or nothing needed checking",
    EXIT_SHIPPED_IDS_MISSING:
        "FINDING (not a crash) -- the check ran fine and shipped ids are absent "
        "from the catalog. This is the tool doing its job; act on the list above",
    EXIT_NOT_RUN: "NOT RUN -- no API key, so the catalog was never fetched",
    EXIT_AUTH_OR_TRANSPORT: "FAILED -- auth or transport error, nothing was checked",
    EXIT_MALFORMED_CATALOG: "FAILED -- the catalog response was malformed",
}


def format_report(report: VerifierReport, *, catalog: Optional[CatalogFetch],
                  lane_authority: Dict[str, str], final: bool = True) -> str:
    """Human-readable text. The report lives HERE, not in a pytest print --
    a print under `pytest -q` capture is invisible, which is what silently
    gutted the original mitigation.

    ``final=False`` renders the short PRE-FLIGHT form: the offline half only, no
    catalog section and no verdict trailer. The first cut printed the whole
    report twice -- once before resolving the key and once after -- which reads
    as a glitch rather than as two phases, and made a working run look broken.
    """
    if not final:
        out = ["Google slug verification -- pre-flight (offline, no catalog yet)"]
        out.append("")
        out.append("UNVERIFIED backlog: %d identity(ies)" % len(report.unverified))
        by_lane: Dict[str, int] = {}
        for _pid, lane in report.unverified:
            by_lane[lane] = by_lane.get(lane, 0) + 1
        for lane in sorted(by_lane):
            out.append("  %-13s %3d   authority: %s"
                       % (lane, by_lane[lane], lane_authority.get(lane, "?")))
        if report.ages:
            out.append("Dated entries: %d, oldest %d day(s). Age NEVER fails."
                       % (len(report.ages), max(d for _k, d in report.ages)))
        if report.future_dated:
            out.append("FUTURE-DATED (malformed evidence): %s"
                       % ", ".join("%s@%s" % k for k in report.future_dated))
        return "\n".join(out)

    out: List[str] = ["Google slug verification"]

    out.append("")
    out.append("UNVERIFIED backlog: %d identity(ies)" % len(report.unverified))
    by_lane: Dict[str, int] = {}
    for _pid, lane in report.unverified:
        by_lane[lane] = by_lane.get(lane, 0) + 1
    for lane in sorted(by_lane):
        out.append("  %-13s %3d   authority: %s"
                   % (lane, by_lane[lane], lane_authority.get(lane, "?")))

    if report.ages:
        oldest = max(days for _k, days in report.ages)
        out.append("")
        out.append("Dated entries: %d, oldest %d day(s). Age NEVER fails a build."
                   % (len(report.ages), oldest))

    if report.pointers:
        out.append("")
        out.append("Pointers (no version claim, never dated): %s"
                   % ", ".join(report.pointers))

    if catalog is None:
        out.append("")
        out.append("Catalog: NOT FETCHED.")
    else:
        out.append("")
        out.append("Catalog: %d id(s) over %d page(s), complete=%s"
                   % (len(catalog.ids), catalog.pages, catalog.complete))
        if catalog.error:
            out.append("  ERROR: %s" % catalog.error)
        elif not catalog.complete:
            out.append("  INCOMPLETE -- no 'missing' verdict may be drawn.")

    if report.future_dated:
        out.append("")
        out.append("FUTURE-DATED (malformed evidence): %s"
                   % ", ".join("%s@%s" % k for k in report.future_dated))

    if report.missing:
        out.append("")
        out.append("SHIPPED IDS ABSENT from a completed catalog: %d"
                   % len(report.missing))
        for pid, lane in report.missing:
            out.append("  %-44s (%s)" % (pid, lane))

    if report.orphaned:
        out.append("")
        out.append("PROVENANCE-ORPHANED (no longer shipped): %s"
                   % ", ".join("%s@%s" % k for k in report.orphaned))

    out.append("")
    out.append("No dates were written. A date is a human claim; commit it yourself.")
    out.append("exit=%d  %s" % (report.exit_code,
                                EXIT_MEANING.get(report.exit_code, "")))
    return "\n".join(out)


__all__ = [
    "EXIT_OK", "EXIT_SHIPPED_IDS_MISSING", "EXIT_NOT_RUN",
    "EXIT_AUTH_OR_TRANSPORT", "EXIT_MALFORMED_CATALOG", "MAX_ATTEMPTS",
    "CatalogFetch", "VerifierReport", "normalize_model_name", "fetch_catalog",
    "age_in_days", "build_report", "format_report",
]
