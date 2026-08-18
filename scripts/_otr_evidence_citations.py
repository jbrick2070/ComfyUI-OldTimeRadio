"""The ledger is the seal: refuse to overwrite bytes a record cites by sha256.

WHY THIS EXISTS. An audition script is written to be RE-RUN; a qualification or
provisional record is written to be PERMANENT. The moment a record cites the
script's output by hash, that output became an append-only archive and the
script's default output path became a destructive one. Re-running destroys the
bytes the record names, and nothing reports it: the re-run is green, the files
are all present, and the citation is silently unverifiable. That is Bug Bible
`12.111`, and its verify step 3 says the class travels -- so this guard is shared
rather than written into whichever instrument was noticed first.

WHY IT WALKS THE LIVE POLICY INSTEAD OF KNOWING THE SCHEMA. `config/cast_pools.py`
records citations in at least four different shapes -- flat
`artifact_path`/`artifact_sha256`, flat `audition_manifest_path`/
`audition_manifest_sha256`, and a nested `audition_manifest: {path, sha256}` dict
that appears under two different container keys. A guard taught those four field
names would silently fail to cover a fifth, which is precisely the
"partial guard reads as a complete one" defect this module exists to prevent. So
it does not read field names at all: it walks the LOADED policy object and
collects every value that looks like a sha256. That covers the approved,
provisional and superseded tiers alike, plus whatever tier is invented next.

Walking the loaded object rather than the source text also means a hash sitting in
a comment or in a docstring cannot arm the guard against a file nobody cites.

WHY THE UNIT IS THE FILE AND NOT THE DIRECTORY. A blanket "refuse to write into a
non-empty directory" is right for a one-shot instrument and wrong here: these
campaign directories are shared workspaces. `scripts/otr_lemmy_listen_page.py`
writes `LISTEN.html` into the same directory on every run, and the cross-engine
audition is deliberately resumable -- its own failure path tells the operator to
re-run one engine into the directory that already holds the others. Sealing the
directory would fight both. The thing that must not change is the bytes a record
names, so that is what is guarded.

NO OVERRIDE FLAG, DELIBERATELY. `12.111` forbids a `--force` that defaults on;
this goes one step further and provides no override at all. A flag that lets a
human assert "this one is not cited" asks them to hold a fact the program can
simply check. If a citation is genuinely stale the honest ceremony is to retire
the record first, after which this guard passes on its own -- which keeps
`cast_pools` the single source of truth rather than making a record lie.
"""
from __future__ import annotations

import hashlib
import os
import re

#: A sha256 rendered as hex, anchored so a longer hex run cannot match.
#:
#: BOTH CASES, DELIBERATELY. Matching only `[0-9a-f]` reads as tidy and is a
#: FALSE NEGATIVE -- an uppercase citation would fail to match, never enter the
#: digest set, and the guard would report the file it names as uncited and permit
#: overwriting it. That is the exact outcome this module exists to prevent, and
#: it is reachable rather than theoretical: `scripts/otr_g1_lemmy_audition.py`
#: computes `hexdigest().upper()`, so this codebase already produces uppercase
#: digests, and `config/cast_pools.py` says repeatedly that qualification records
#: are written BY HAND. A guard must never fail open on presentation.
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CitationGuardUnavailable(RuntimeError):
    """The ledger could not be read, so no file can be proven safe to write.

    This is raised rather than returning "nothing is cited", because a guard that
    degrades to permissive when its evidence source is missing is worse than no
    guard: it reports safety it did not establish.
    """


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_for_digests(node, found: set) -> None:
    """Collect every sha256-shaped string anywhere in a nested structure."""
    if isinstance(node, str):
        if _SHA256_RE.match(node):
            found.add(node.lower())
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_for_digests(key, found)
            _walk_for_digests(value, found)
        return
    if isinstance(node, (list, tuple, set, frozenset)):
        for value in node:
            _walk_for_digests(value, found)


def cited_digests(policy=None) -> set:
    """Every sha256 the ledger cites, read from the LIVE policy object.

    Pass ``policy`` to test the walk against a fixture; production callers leave
    it None so the real `config.cast_pools.LEMMY_VOICE_POLICY` is read.
    """
    if policy is None:
        try:
            from config.cast_pools import LEMMY_VOICE_POLICY as policy
        except Exception as exc:                        # noqa: BLE001
            raise CitationGuardUnavailable(
                "cannot read the voice policy, so no file can be proven "
                "uncited: %s" % (exc,))
    # A LEDGER THIS WALK CANNOT READ MUST NOT LOOK LIKE A LEDGER WITH NO
    # CITATIONS. `_walk_for_digests` skips leaf types it does not recognize,
    # which is right for ints and None and wrong for the whole policy: handed
    # some object it cannot descend, it would return an empty set, every file
    # would look uncited, and the guard would wave through exactly the writes it
    # exists to stop. Fail closed on the container instead.
    if not isinstance(policy, (dict, list, tuple)):
        raise CitationGuardUnavailable(
            "the voice policy is a %s, which this walk cannot descend -- "
            "refusing to report that nothing is cited"
            % type(policy).__name__)
    found = set()
    _walk_for_digests(policy, found)
    return found


def cited_among(paths, policy=None) -> list:
    """``[(path, digest)]`` for each EXISTING path whose bytes the ledger cites.

    A path that does not exist yet cannot be destroyed, so it is not a finding.
    Ordering follows ``paths`` so the caller's message is stable.
    """
    digests = cited_digests(policy)
    hits = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        # Compared lowercase on both sides: `cited_digests` lowercases on the way
        # in, and a caller may hand this a digest produced by a sibling that
        # renders hex in upper case.
        digest = sha256_of(path).lower()
        if digest in digests:
            hits.append((path, digest))
    return hits


def refuse_if_cited(paths, instrument, policy=None) -> int:
    """Print a refusal and return 2 when any path is cited; return 0 otherwise.

    Returns the process exit code rather than raising, because every caller is a
    CLI ``main()`` whose contract is an exit status. `2` matches the refusal code
    the sibling audition instruments already use.
    """
    hits = cited_among(paths, policy)
    if not hits:
        return 0
    print("\nREFUSING: %s would overwrite evidence a record cites by sha256."
          % instrument)
    for path, digest in hits:
        print("  %s\n    sha256 %s" % (path, digest))
    print("\n  A record in config/cast_pools.py names those bytes, so replacing\n"
          "  them would leave it citing a hash nothing on disk matches -- and\n"
          "  nothing would report it.\n"
          "  Render into a NEW directory with --out-dir <name>. A new run is a\n"
          "  new artifact; refreshing evidence in place is never the operation\n"
          "  you want. If a citation is genuinely stale, retire the record in\n"
          "  config/cast_pools.py first and this guard will pass on its own.")
    return 2
