#!/usr/bin/env python
"""Open every vendored-translation lead and verdict it. Nothing else.

Operator ruling 2026-09-18 and `apple/SHAKESPEARE_CORPUS_SPEC.yaml`: *"nothing enters the pipeline on the strength
of this document"*. v1 scored 28 cells READY without opening the pages, so this
script exists to replace claims with measurements before anyone vendors a line.

    $env:PYTHONUTF8=1
    python scripts/otr_shakespeare_corpus_gate.py --leads config/source_banks/shakespeare/translations/leads.json
    python scripts/otr_shakespeare_corpus_gate.py --leads ... --fetch --iso fr

Without ``--fetch`` it reports what the leads file already claims and opens
nothing, so every lead it has not excluded comes back PARTIAL -- honest about
having measured nothing. (It used to verdict the RIGHTS in that mode; rights
verdict nothing now -- operator 2026-09-18.) With
``--fetch`` it downloads each lead once into ``--cache`` and parses from local
(re-scraping during parser iteration gets you blocked), then writes a JSON
report plus a readable table.

It writes NO manifest. Vendoring is a separate, deliberate step: this tells you
which leads are worth the transcription work, and the operator decides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes import _otr_verbatim_corpus as CORPUS  # noqa: E402

#: The spec's own requirement: a descriptive agent with a contact address, and
#: respect for maxlag. A bare python-urllib agent gets a Wikimedia block.
USER_AGENT = (
    "OTR-corpus-gate/1.0 (ComfyUI-OldTimeRadio; "
    "https://github.com/jbrick2070/ComfyUI-OldTimeRadio)"
)

#: Polite floor between fetches to one host.
FETCH_DELAY_S = 1.0


def _cache_name(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".txt"


#: MediaWiki embeds the revision this render came from in its JS config. That
#: number is what pins the bytes: a Wikisource page edited tomorrow is a
#: different text under the same URL, and a manifest that cannot say WHICH
#: revision it vendored cannot prove what it shipped.
_WG_REVISION = re.compile(r'"wgCurRevisionId"\s*:\s*(\d+)')
_OLDID = re.compile(r"[?&]oldid=(\d+)")


def revision_id(url: str, text: str, headers: "dict | None" = None) -> str:
    """A version this fetch can be pinned to, or "".

    MediaWiki's own revision id first (exact), then an explicit ``oldid`` in
    the URL, then the transport's validators -- an ETag or Last-Modified is
    weaker than a revision number but still pins the bytes for a host that
    offers nothing better.
    """
    match = _WG_REVISION.search(text or "")
    if match:
        return "rev:" + match.group(1)
    match = _OLDID.search(url or "")
    if match:
        return "oldid:" + match.group(1)
    for key in ("ETag", "Last-Modified"):
        value = str((headers or {}).get(key) or "").strip().strip('"')
        if value:
            return "%s:%s" % (key.lower(), value)
    # A host with no version of its own (Project Gutenberg, archive.org) still
    # gets pinned: the bytes we actually read are the edition we vendored, and
    # the manifest carries the same digest. Weaker than a revision NUMBER --
    # it cannot tell you the page changed, only that it differs from ours --
    # and that is exactly what a re-run needs to know.
    if text:
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return ""


def fetch(url: str, cache_dir: str, *, refresh: bool = False) -> dict:
    """Fetch once into the cache; return what the gate measures.

    Never raises for a network-shaped reason -- an unreachable lead is a
    verdict, not a crash.
    """
    url = CORPUS.strip_tracking(url)
    path = os.path.join(cache_dir, _cache_name(url))
    meta_path = path + ".headers.json"
    if os.path.isfile(path) and not refresh:
        with open(path, "r", encoding="utf-8") as fh:
            body = fh.read()
        headers = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    headers = json.load(fh)
            except (OSError, ValueError):
                headers = {}
        return {"status": 200, "final_url": url, "text": body,
                "bytes": len(body.encode("utf-8")), "encoding": "utf-8",
                "headers": headers, "cached": True}
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # Gutenberg throttles a burst; a transport failure is a RETRY, not a
    # verdict. Measured 2026-09-18: two ebook ids failed mid-run and both
    # fetched cleanly on their own seconds later.
    last_error = ""
    for attempt in range(3):
        try:
            return _open(req, url, path, meta_path, cache_dir)
        except urllib.error.HTTPError as exc:
            return {"status": int(exc.code), "final_url": url, "text": "",
                    "bytes": 0, "encoding": "", "headers": {},
                    "error": "HTTP %d" % exc.code, "cached": False}
        except Exception as exc:  # noqa: BLE001 -- DNS, TLS, timeout, refusal
            last_error = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(FETCH_DELAY_S * (attempt + 1) * 3)
    return {"status": 0, "final_url": url, "text": "", "bytes": 0,
            "encoding": "", "headers": {}, "error": last_error, "cached": False}


def _open(req, url, path, meta_path, cache_dir) -> dict:
    """One attempt. Raises; `fetch` owns the retry and the verdict."""
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        final = resp.geturl()
        status = resp.status
        headers = {k: v for k, v in resp.headers.items()
                   if k in ("ETag", "Last-Modified", "Content-Type")}
    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        charset = "utf-8"
        body = raw.decode("utf-8", errors="replace")
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(headers, fh)
    time.sleep(FETCH_DELAY_S)
    return {"status": status, "final_url": final, "text": body,
            "bytes": len(raw), "encoding": charset, "headers": headers,
            "cached": False}


#: A Wikimedia Commons / NDL / archive.org FILE page describes a scan; the
#: scan itself is a PDF or DjVu. Either way there is no text to measure, and
#: the honest verdict is "transcribe this", not "wrong source".
_SCAN_MEDIA = ("application/pdf", "image/", "application/octet-stream",
               "image/vnd.djvu")
_SCAN_PAGE = re.compile(
    r"(commons\.wikimedia\.org/wiki/file:|\.pdf\b|\.djvu\b)", re.I)


def _looks_like_a_scan(text: str, headers: "dict | None") -> bool:
    media = str((headers or {}).get("Content-Type") or "").lower()
    if any(media.startswith(m) or m in media for m in _SCAN_MEDIA):
        return True
    if str(text or "")[:5].startswith("%PDF"):
        return True
    # A Commons File: page is HTML ABOUT a scan. Its body carries the licence
    # and the file name, never the speeches.
    head = str(text or "")[:4000]
    return bool(_SCAN_PAGE.search(head)) and "wgCanonicalNamespace\":\"File\"" in text


def assess_lead(lead: dict, *, cache_dir: str, do_fetch: bool,
                refresh: bool = False) -> CORPUS.LeadReport:
    report = CORPUS.LeadReport(
        iso=str(lead.get("iso") or ""),
        play=str(lead.get("play") or ""),
        scene=str(lead.get("scene") or ""),
        url=CORPUS.strip_tracking(lead.get("url") or ""),
        licence=str(lead.get("transcription_license") or ""),
        translator=str(lead.get("translator") or ""),
        translator_died=lead.get("translator_death_date", ""),
        first_published=lead.get("translation_first_published", ""),
        revision_id=str(lead.get("revision_id") or ""),
        excluded=str(lead.get("excluded") or ""),
    )
    if report.excluded:
        # A recorded dead end. Never fetched, in either mode: re-opening it is
        # the hunt this field exists to stop from happening twice.
        return CORPUS.assess(report)
    if not do_fetch:
        # An unopened page cannot be READY whatever else is true of it, so the
        # lead is reported PARTIAL with the reason said out loud.
        report = CORPUS.assess(report)
        if report.verdict != CORPUS.BLOCKED:
            report.verdict = CORPUS.PARTIAL
            report.reasons = ["not fetched -- no page was opened"]
        return report
    if not report.url:
        # A lead with no page yet. The verdict is the WORK ITEM -- locate the
        # page -- because that is the only actionable fact about this row.
        # Rights decide nothing here (operator 2026-09-18); only a recorded
        # `excluded` dead end still outranks the missing page, and it should,
        # since there is no point locating a page for a lead a human already
        # ruled out.
        report = CORPUS.assess(report)
        if report.verdict != CORPUS.BLOCKED:
            report.verdict = CORPUS.EMPTY
            report.reasons = ["no url recorded -- locate the page first"]
        return report
    got = fetch(report.url, cache_dir, refresh=refresh)
    text = got.get("text") or ""
    report.http_status = int(got.get("status") or 0)
    report.final_url = str(got.get("final_url") or report.url)
    report.byte_length = int(got.get("bytes") or 0)
    report.encoding = str(got.get("encoding") or "")
    report.is_scan = _looks_like_a_scan(text, got.get("headers"))
    if not report.byte_length:
        report.transport_error = str(got.get("error") or "")
    report.headings_present = CORPUS.headings_present(text, report.scene)
    report.act_headings = CORPUS.act_headings_found(text)
    report.speaker_labels, report.distinct_speakers = \
        CORPUS.speaker_label_stats(text)
    report.dialogue_ratio = CORPUS.dialogue_ratio(text)
    report.pending_markers = CORPUS.find_pending_markers(text)
    # The lead may pin a revision itself; otherwise take what the fetch proves.
    report.revision_id = report.revision_id or revision_id(
        report.final_url, text, got.get("headers"))
    return CORPUS.assess(report)


def load_leads(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    leads = data.get("leads") if isinstance(data, dict) else data
    if not isinstance(leads, list):
        raise SystemExit("%s: expected a list of leads" % path)
    return leads


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--leads", required=True)
    ap.add_argument("--iso", action="append", default=[],
                    help="limit to these language rows (repeatable)")
    ap.add_argument("--fetch", action="store_true",
                    help="open every lead; without it only rights are checked")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cache and re-download")
    ap.add_argument("--cache", default=os.path.join(_REPO, "tmp", "corpus_cache"))
    ap.add_argument("--report", default=os.path.join(
        _REPO, "config", "source_banks", "shakespeare", "translations",
        "gate_report.json"))
    args = ap.parse_args(argv)

    leads = load_leads(args.leads)
    if args.iso:
        want = {s.strip().lower() for s in args.iso}
        leads = [l for l in leads if str(l.get("iso") or "").lower() in want]
    if not leads:
        print("no leads selected")
        return 1

    reports = [assess_lead(l, cache_dir=args.cache, do_fetch=args.fetch,
                           refresh=args.refresh) for l in leads]
    rows = [r.as_row() for r in reports]
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"schema_version": CORPUS.SCHEMA_VERSION,
                   "fetched": bool(args.fetch), "scenes": rows},
                  fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    tally: dict = {}
    for r in reports:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
    width = max((len(f"{r.iso} {r.play} {r.scene}") for r in reports), default=10)
    for r in reports:
        head = f"{r.iso} {r.play} {r.scene}".ljust(width)
        why = ("; ".join(r.reasons))[:96]
        print(f"{r.verdict:<8} {head}  {why}")
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print("report: %s" % args.report)
    if not args.fetch:
        print("NOT FETCHED -- no page was opened. Re-run with --fetch before "
              "vendoring anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
