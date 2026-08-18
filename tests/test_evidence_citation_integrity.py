"""Every hash the ledger cites must still resolve to bytes on this box.

WHY THIS EXISTS SEPARATELY FROM THE GUARDS. A writer-side guard stops the
instrument that bit us. It cannot see any of the other ways cited evidence goes
missing: a manual delete, a directory move, a different script writing there, a
half-finished copy. Detection converts silent rot into loud rot, and it costs no
GPU.

THE ASYMMETRY THIS CLOSES, which is the reason the file was written. Before this,
detection existed exactly where the stakes were LOWEST and was absent where they
were HIGHEST:

  * the PROVISIONAL routes -- pending-listen, nothing selects them -- had a real
    byte-level check in `test_lemmy_provisional_tier.py`;
  * the QUALIFIED route's manifest, the evidence behind the route production
    actually selects, had NO on-disk check anywhere in `tests/`;
  * the SUPERSEDED G1 record had only a config-literal assertion -- a test that
    the number is still typed in `cast_pools.py` -- while its own docstring
    claimed the manifest "still hashes to the value it claims". A comment doing a
    test's job.

SKIP AND FAIL ARE DIFFERENT ANSWERS, AND THE OLD PROBE CONFLATED THEM. The
existing helper resolves its root by looking for the artifact itself, so a
deleted archive and a box that never had one both return None and both skip. That
is the one state this file must not be silent about: if there IS an output root
here, a cited artifact that is missing is a FAILURE, not a shrug.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The reference WAV the routes hash to prove the voice identity did not change.
#: It lives in the MODELS tree, not under `otr/episodes/`, so it is not an
#: audition artifact and is deliberately not swept by the path checks below. It
#: is named here rather than filtered silently, so the allowlist is a statement
#: someone can disagree with.
_NON_EPISODE_DIGESTS = {
    "47e733d51ea58773142f934f3484cf3633cada5fe603b672cdfc47c712a60db2":
        "the Lemmy reference WAV in the models tree (source_ref/bank_ref), "
        "hashed by the routes to pin identity -- not an episode artifact",
}


def _load_citation_helper():
    """Reach the guard helper the way this repo already reaches scripts."""
    path = os.path.join(REPO_ROOT, "scripts", "_otr_evidence_citations.py")
    spec = importlib.util.spec_from_file_location("otr_evidence_citations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The real ComfyUI output tree, derived from the repo's own location. Used to
#: decide SKIP-versus-FAIL, deliberately without consulting `folder_paths`: under
#: pytest that module is a conftest STUB whose `get_output_directory()` returns
#: the current working directory, and this repo carries its own untracked
#: `otr/episodes`. Believing the stub is how a check concludes the evidence is
#: gone while it sits safely one directory away.
_COMFY_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(REPO_ROOT))), "ComfyUI", "output")


def _episodes_root(probe_relative):
    """``(root, resolved)`` -- where the evidence is, and whether it was found.

    ARTIFACT-FIRST, AND THAT ORDERING IS LOAD-BEARING. There is more than one
    plausible output root on this box, so the root is chosen by probing for the
    ARTIFACT rather than for a folder. Picking the first candidate that merely
    has an `otr/episodes` directory is how this check reported every cited file
    missing while all eleven were present -- the pytest stub named the repo root,
    and the repo has an episodes directory of its own.

    When no candidate holds the artifact, the answer still is not automatically
    "stay quiet". If the REAL ComfyUI output tree exists, the evidence is gone
    and that is a failure; only a box with no such tree earns a skip.
    """
    candidates = []
    try:
        import folder_paths                              # ComfyUI runtime
        candidates.append(folder_paths.get_output_directory())
    except Exception:                                    # noqa: BLE001
        pass
    candidates.append(_COMFY_OUTPUT)
    for base in candidates:
        if base and os.path.isfile(os.path.join(base, probe_relative)):
            return base, True
    if os.path.isdir(os.path.join(_COMFY_OUTPUT, "otr", "episodes")):
        return _COMFY_OUTPUT, False
    return None, False


def _cited_artifacts():
    """``[(label, relative_path, sha256)]`` for every artifact a record cites.

    The three record shapes are enumerated EXPLICITLY rather than sniffed, so a
    reader can see exactly what is covered. The coverage test below then proves
    this enumeration did not miss a shape.
    """
    from config import cast_pools as POOLS

    policy = POOLS.LEMMY_VOICE_POLICY
    found = []

    # Shape 1 -- the QUALIFIED route: nested `audition_manifest: {path, sha256}`
    # inside `qualification_record`, plus the legacy flat receipt beside it.
    for engine, route in (policy.get("approved_native_routes") or {}).items():
        record = route.get("qualification_record") or {}
        manifest = record.get("audition_manifest") or {}
        if manifest.get("path") and manifest.get("sha256"):
            found.append(("qualified/%s audition_manifest" % engine,
                          manifest["path"], manifest["sha256"]))

    # Shape 2 -- the SUPERSEDED record: same nested shape, different container.
    for engine, records in (policy.get("superseded_native_routes") or {}).items():
        for record in records:
            manifest = record.get("audition_manifest") or {}
            if manifest.get("path") and manifest.get("sha256"):
                found.append(("superseded/%s audition_manifest" % engine,
                              manifest["path"], manifest["sha256"]))

    # Shape 3 -- the PROVISIONAL receipts: flat `<thing>_path` / `<thing>_sha256`.
    for engine, record in (policy.get("provisional_native_routes") or {}).items():
        receipt = record.get("provisional_receipt") or {}
        for prefix in ("audition_manifest", "neutral_clip", "emotional_clip"):
            path = receipt.get("%s_path" % prefix)
            digest = receipt.get("%s_sha256" % prefix)
            if path and digest:
                found.append(("provisional/%s %s" % (engine, prefix),
                              path, digest))
    return found


def test_every_cited_artifact_still_hashes_to_what_the_record_claims():
    """THE STANDING ROT CHECK, across every tier -- Bible 12.111 verify step 4.

    This is the cheap check that catches a citation which has already gone bad,
    however it went bad. It covers the qualified and superseded records for the
    first time.
    """
    artifacts = _cited_artifacts()
    assert artifacts, "the policy cites nothing -- the enumerator is broken"

    root, resolved = _episodes_root(artifacts[0][1])
    if root is None:
        pytest.skip("no ComfyUI output tree on this box")
    if not resolved:
        pytest.fail(
            "the ComfyUI episodes tree exists at %s but the cited evidence is "
            "not in it -- %s is missing. Evidence a record names by sha256 has "
            "gone from disk; that is the failure this check exists for, not a "
            "reason to stay quiet." % (root, artifacts[0][1]))

    missing, mismatched = [], []
    for label, relative, claimed in artifacts:
        path = os.path.join(root, relative)
        if not os.path.isfile(path):
            # NOT a skip. The episodes tree resolved, so a cited artifact that is
            # absent is evidence that went missing, which is the whole point.
            missing.append("%s -> %s" % (label, relative))
            continue
        with open(path, "rb") as handle:
            actual = hashlib.sha256(handle.read()).hexdigest()
        if actual.lower() != str(claimed).lower():
            mismatched.append("%s\n    claims %s\n    actual %s"
                              % (label, claimed, actual))

    assert not missing, (
        "cited evidence is MISSING from disk:\n  " + "\n  ".join(missing))
    assert not mismatched, (
        "cited evidence no longer matches its record -- something re-rendered or "
        "replaced it:\n  " + "\n  ".join(mismatched))


def test_the_enumeration_covers_every_digest_the_policy_carries():
    """A NEW CITATION SHAPE MUST NOT SLIP IN UNCHECKED.

    The test above enumerates three record shapes by hand. If a fourth is added
    -- and this policy has already grown from one shape to three -- that hand
    enumeration would silently stop being complete, and the rot check would
    quietly cover less than it appears to. So: walk the live policy for every
    sha256-shaped value and prove each one is either checked above or explicitly
    allowlisted as a non-episode artifact.
    """
    helper = _load_citation_helper()
    from config import cast_pools as POOLS

    every = helper.cited_digests(POOLS.LEMMY_VOICE_POLICY)
    enumerated = {str(digest).lower() for _label, _path, digest in _cited_artifacts()}
    unaccounted = every - enumerated - set(_NON_EPISODE_DIGESTS)

    assert not unaccounted, (
        "the policy cites %d digest(s) that no test verifies against disk:\n  %s\n"
        "Add the new record shape to `_cited_artifacts`, or allowlist it in "
        "`_NON_EPISODE_DIGESTS` with the reason it is not an episode artifact."
        % (len(unaccounted), "\n  ".join(sorted(unaccounted))))


def test_the_allowlist_only_excuses_digests_that_are_really_there():
    """An allowlist entry for a digest the policy no longer carries is a stale
    excuse, and a stale excuse is how a real gap gets hidden later."""
    helper = _load_citation_helper()
    from config import cast_pools as POOLS

    every = helper.cited_digests(POOLS.LEMMY_VOICE_POLICY)
    stale = set(_NON_EPISODE_DIGESTS) - every
    assert not stale, (
        "allowlisted digests are no longer cited anywhere -- remove them:\n  %s"
        % "\n  ".join(sorted(stale)))
