"""The audition instruments must refuse to overwrite hash-cited evidence.

Bug Bible `12.111` verify steps 1-3. Step 4 (the standing re-hash of every cited
artifact) lives in `test_evidence_citation_integrity.py`; step 5
(byte-identical-under-changed-code) is not provable without re-rendering, and
re-rendering to test a guard would destroy the evidence under discussion, so it is
deliberately not claimed here.

WHY SUBPROCESSES AND NOT A DIRECT `main()` CALL. Both instruments rebind a module
-level `_OUT_DIR` from their CLI arguments. Calling `main()` twice in one pytest
process leaves that global pointing at the previous test's directory, so a later
test that omits `--out-dir` would silently inherit it -- and the failure would look
like a guard bug rather than a fixture bug. This repo already reaches for a
subprocess for exactly this class of shared-state hazard (see
`tests/test_ltx_av_env_import_safety.py`). A subprocess also proves the real
command line an operator types, argument parsing included.

NOTHING HERE RENDERS. Every case is a refusal, and every refusal is asserted to
happen BEFORE the engines load -- which is what makes these tests headless.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CROSS_ENGINE = os.path.join(REPO_ROOT, "scripts",
                            "otr_lemmy_cross_engine_audition.py")
G1 = os.path.join(REPO_ROOT, "scripts", "otr_g1_lemmy_audition.py")

#: Every instrument that writes an artifact a record could cite by sha256, with
#: the reason it is on the list. THE ROSTER IS THE TRIPWIRE: a fourth evidence
#: writer added later trips `test_the_guarded_roster_is_complete` and forces a
#: decision, which is what stops this defect class travelling again.
GUARDED_INSTRUMENTS = {
    "otr_lemmy_cross_engine_audition.py":
        "writes MANIFEST.json + eight clips; its manifest and six clips are "
        "cited by sha256 in three provisional records",
    "otr_g1_lemmy_audition.py":
        "writes MANIFEST.json + six arm clips + KEY.json; its manifest is cited "
        "by sha256 in the superseded qualification record",
    "otr_lemmy_production_audition.py":
        "writes MANIFEST.json + nine clips + KEY.json; its manifest is cited by "
        "sha256 in the QUALIFIED route production selects",
}

#: Writers into the same directories that are deliberately NOT guarded, each with
#: the reason. Listed rather than ignored so the exemption is arguable.
UNGUARDED_BY_DESIGN = {
    "otr_lemmy_listen_page.py":
        "derived view -- writes LISTEN.html (cited by nothing) and refuses to "
        "clobber an existing DECISIONS.json",
    "otr_g1_listen_page.py":
        "derived view -- writes G1-LISTEN.html, cited by nothing",
    "bark_preset_audition.py":
        "writes a manifest under docs/, cited by no record",
}


def _run(script, args, cwd=None):
    return subprocess.run(
        [sys.executable, script] + args,
        cwd=cwd or REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})


def _snapshot(directory):
    """``{name: (mtime_ns, sha256)}`` -- what a refusal must leave untouched."""
    state = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                state[name] = (os.stat(path).st_mtime_ns,
                               hashlib.sha256(handle.read()).hexdigest())
    return state


@pytest.fixture()
def cited_campaign(tmp_path, monkeypatch):
    """A directory whose bytes a stand-in policy cites, plus that policy.

    Built in a temp directory from invented bytes rather than pointed at the real
    archive: a test that needs the real evidence to prove a guard is a test that
    can destroy it.
    """
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    manifest = campaign / "MANIFEST.json"
    manifest.write_text(json.dumps({"campaign": "stand-in"}), encoding="utf-8")
    clip = campaign / "kokoro_neutral.wav"
    clip.write_bytes(b"not really audio, but it hashes like a file")

    digests = {}
    for path in (manifest, clip):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return campaign, digests


def test_a_second_run_into_a_cited_directory_refuses_and_writes_nothing():
    """VERIFY STEP 1, against the REAL cited campaign -- exit code and bytes.

    Asserted on the exit code and on every file's mtime and hash, never on the
    wording of the message.
    """
    from tests.test_evidence_citation_integrity import (_cited_artifacts,
                                                        _episodes_root)

    artifacts = _cited_artifacts()
    root, resolved = _episodes_root(artifacts[0][1])
    if root is None or not resolved:
        pytest.skip("the cited artifacts are not on this box")
    campaign = os.path.join(root, "otr", "episodes", "lemmy_cross_engine")
    if not os.path.isdir(campaign):
        pytest.skip("the cross-engine campaign is not on this box")

    before = _snapshot(campaign)
    result = _run(CROSS_ENGINE, ["--render"])
    after = _snapshot(campaign)

    assert result.returncode == 2, result.stdout + result.stderr
    assert after == before, "a refusal must not touch a single byte"


def test_the_refusal_happens_before_any_engine_loads():
    """A guard that fires after preflight costs four engine loads to say no.

    Preflight spawns sidecar worker processes and reads weights. Proving the
    refusal precedes it is both an ergonomics guarantee and the reason these
    tests can run headless.
    """
    result = _run(CROSS_ENGINE, ["--render"])
    assert result.returncode == 2
    assert "REFUSING" in result.stdout
    assert "preflight" not in result.stdout.lower(), (
        "the preflight banner printed, so engines were loaded before the guard "
        "refused:\n" + result.stdout)


def test_rendering_only_an_uncited_engine_still_refuses_on_the_shared_manifest():
    """THE CASE THAT LOOKS HARMLESS AND IS NOT.

    `bark` has no route row at all, so re-cutting its two comparison clips seems
    like the one guaranteed-safe act available. But the manifest is SHARED and is
    rewritten on every save, so the run would rot `audition_manifest_sha256` on
    all three records while all six clip hashes kept verifying -- partial rot,
    which is the hardest kind to diagnose later.
    """
    result = _run(CROSS_ENGINE, ["--render", "--engine", "bark"])
    assert result.returncode == 2
    assert "MANIFEST.json" in result.stdout
    assert "bark_neutral.wav" not in result.stdout, (
        "bark's own clips are not cited and should not be named as if they were")


def test_g1_refuses_on_its_cited_manifest():
    """The sibling that kept the defect. Its manifest backs the superseded
    qualification, whose record promises it stays re-verifiable byte for byte."""
    result = _run(G1, ["--render"])
    assert result.returncode == 2
    assert "REFUSING" in result.stdout


def test_g1_no_longer_accepts_the_overwrite_escape():
    """`--overwrite` had zero callers and its only function was to overwrite
    cited evidence. An old command line must die loudly rather than proceed."""
    result = _run(G1, ["--render", "--overwrite"])
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_the_guard_refuses_a_directory_holding_only_clips(cited_campaign,
                                                          tmp_path):
    """VERIFY STEP 2, first half: a directory with siblings but NO manifest.

    A guard that keys on `MANIFEST.json` lets this through, which is exactly the
    partial guard 12.111 names. The manifest is deleted so only the clip remains.
    """
    campaign, digests = cited_campaign
    os.remove(campaign / "MANIFEST.json")
    guard = _load_guard()
    policy = {"receipt": {"clip_sha256": digests["kokoro_neutral.wav"]}}

    hits = guard.cited_among([str(campaign / "kokoro_neutral.wav")], policy)
    assert len(hits) == 1, "a clip is evidence even with no manifest beside it"


def test_the_guard_refuses_a_key_directory_the_primary_does_not_cover(tmp_path):
    """VERIFY STEP 2, second half: the KEY directory exists on its own.

    The blinding key says which arm a human approved. G1's old guard never looked
    at it at all, so it is checked here explicitly rather than assumed.
    """
    key_dir = tmp_path / "campaign_KEY"
    key_dir.mkdir()
    key = key_dir / "KEY.json"
    key.write_text(json.dumps({"mapping": "stand-in"}), encoding="utf-8")
    digest = hashlib.sha256(key.read_bytes()).hexdigest()

    guard = _load_guard()
    hits = guard.cited_among([str(key)], {"anything": {"nested": digest}})
    assert hits == [(str(key), digest)]


def test_a_path_that_does_not_exist_yet_is_not_a_finding(tmp_path):
    """Nothing can be destroyed that was never written; a fresh campaign must
    proceed. This is what keeps the guard from becoming a blanket refusal."""
    guard = _load_guard()
    assert guard.cited_among([str(tmp_path / "nothing.wav")],
                             {"r": {"sha256": "a" * 64}}) == []


def test_the_guard_fails_closed_on_a_ledger_it_cannot_descend():
    """A LEDGER THE WALK CANNOT READ MUST NOT LOOK LIKE ONE WITH NO CITATIONS.

    This is the fail-open path that matters: the walk skips leaf types it does
    not recognize, so handed an object it cannot descend it would return an empty
    set, every file would test as uncited, and the guard would permit exactly the
    writes it exists to refuse. Silence and safety must not be the same answer.
    """
    guard = _load_guard()
    with pytest.raises(guard.CitationGuardUnavailable):
        guard.cited_digests(_UnwalkablePolicy())


class _UnwalkablePolicy:
    """A policy-shaped object this walk has no way to descend."""


def test_the_guard_fails_closed_when_the_ledger_import_fails():
    """The production path takes no argument and imports the policy itself. If
    that import ever breaks, the guard must raise rather than permit."""
    probe = (
        "import importlib.util, sys, types;"
        "sys.modules['config'] = types.ModuleType('config');"
        "sys.modules['config.cast_pools'] = None;"
        "spec = importlib.util.spec_from_file_location('g', r'%s');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "\ntry:\n    m.cited_digests()\n    print('PERMITTED')\n"
        "except m.CitationGuardUnavailable:\n    print('REFUSED')\n"
        % os.path.join(REPO_ROOT, "scripts", "_otr_evidence_citations.py")
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert "REFUSED" in result.stdout, result.stdout + result.stderr


def test_the_walk_ignores_hex_that_is_not_a_sha256():
    """A short hex run and a 64-char non-hex string must not widen the guard."""
    guard = _load_guard()
    found = guard.cited_digests({
        "short": "abc123",
        "long_but_not_hex": "z" * 64,
        "real": "b" * 64,
    })
    assert "b" * 64 in found
    assert "abc123" not in found
    assert "z" * 64 not in found


def test_an_uppercase_citation_is_still_a_citation():
    """A FALSE NEGATIVE HERE DESTROYS EVIDENCE, so it gets its own test.

    The first version of this guard matched `[0-9a-f]` only, and the first
    version of the test above declared an uppercase fixture, promised in its
    docstring that case must not narrow the guard, and then never asserted on it.
    The fixture and the claim shipped; the assertion did not, which is why a green
    suite proved nothing about the case that mattered.

    It is not a hypothetical: `scripts/otr_g1_lemmy_audition.py` computes
    `hexdigest().upper()`, so this repo already produces uppercase digests, and
    qualification records are written by hand.
    """
    guard = _load_guard()
    upper = "C" * 64
    found = guard.cited_digests({"receipt": {"artifact_sha256": upper}})
    assert upper.lower() in found, (
        "an uppercase citation was dropped -- the guard would report the file it "
        "names as uncited and permit overwriting it")


def test_an_uppercase_citation_actually_refuses_a_real_file(tmp_path):
    """End to end, not just the walk: a file whose ledger entry is uppercase must
    still be refused."""
    guard = _load_guard()
    artifact = tmp_path / "MANIFEST.json"
    artifact.write_text(json.dumps({"campaign": "stand-in"}), encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    policy = {"route": {"audition_manifest": {"sha256": digest.upper()}}}
    assert guard.cited_among([str(artifact)], policy) == [(str(artifact), digest)]
    assert guard.refuse_if_cited([str(artifact)], "a test", policy) == 2


def test_the_guarded_roster_is_complete():
    """THE TRIPWIRE. A new instrument that writes into the episodes tree must be
    classified -- guarded, or exempt with a stated reason -- rather than quietly
    inheriting neither.
    """
    scripts_dir = os.path.join(REPO_ROOT, "scripts")
    classified = set(GUARDED_INSTRUMENTS) | set(UNGUARDED_BY_DESIGN)
    writers = set()
    for name in os.listdir(scripts_dir):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(scripts_dir, name), "r", encoding="utf-8",
                  errors="replace") as handle:
            body = handle.read()
        if "MANIFEST.json" in body or "KEY.json" in body:
            writers.add(name)

    unclassified = writers - classified
    assert not unclassified, (
        "these scripts touch a manifest or key but are neither guarded nor "
        "listed as exempt:\n  %s\nAdd each to GUARDED_INSTRUMENTS (and give it "
        "the citation guard) or to UNGUARDED_BY_DESIGN with the reason."
        % "\n  ".join(sorted(unclassified)))


def test_every_guarded_instrument_actually_calls_the_guard():
    """A roster entry claiming a guard that is not wired is worse than no entry.
    """
    for name in GUARDED_INSTRUMENTS:
        path = os.path.join(REPO_ROOT, "scripts", name)
        with open(path, "r", encoding="utf-8") as handle:
            body = handle.read()
        assert ("refuse_if_cited" in body) or ("is not empty" in body), (
            "%s is on the guarded roster but calls no refusal" % name)


def _load_guard():
    import importlib.util
    path = os.path.join(REPO_ROOT, "scripts", "_otr_evidence_citations.py")
    spec = importlib.util.spec_from_file_location("otr_evidence_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
