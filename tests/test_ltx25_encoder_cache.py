"""Episode-scoped text-encoder residency on the ``ltx25_video`` lane.

WHY THIS EXISTS. A live canonical leg on 2026-08-20 rendered 15 shots and read
the 8.86 GiB Gemma-4 12B Q5 GGUF text encoder off disk **13 times** -- a 1:1
ratio, ~63 s a shot, on top of a 54.2 s CPU encode. The lane rebuilt its whole
graph per shot, including the ``te`` loader, and nothing owned the loaded CLIP
between beats.

EVERY TEST BELOW IS A DEFECT THAT WAS REAL DURING THE ARC, not a checklist row:

* the cache key could collide two DIFFERENT broken states into a false HIT,
  because ``_weight_receipt`` answers ``(basename, -1, -1)`` for both an
  unresolvable path and an ``OSError``;
* the drift digest a review lane proposed indexed ``out[0][0]`` and would have
  raised ``AttributeError: 'list' object has no attribute 'sum'`` on EVERY
  cache hit, because a conditioning entry is ``[tensor, dict]`` and the tensor
  is one subscript deeper;
* the kill switch could be flipped off and leave 8.86 GiB pinned by a scope
  opened before the flip;
* and the whole thing is worthless if no scope is open, which must mean
  "behave exactly as before", not "cache anyway".

CPU-safe and pure: no renders, no CUDA, no model loads.
"""

from __future__ import annotations

import os

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ltx25


@pytest.fixture()
def eng():
    return eng_ltx25.Ltx25VideoEngine()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The switch is read from the real environment; never inherit one."""
    monkeypatch.delenv(eng_ltx25._ENCODER_CACHE_ENV, raising=False)


class _FakeTensor:
    """The smallest thing that answers the digest's duck-type questions."""

    def __init__(self, total=1.5, shape=(1, 4, 8), dtype="torch.float16",
                 numel=32):
        self.shape = shape
        self.dtype = dtype
        self._total = total
        self._numel = numel

    def numel(self):
        return self._numel

    def sum(self):
        return self

    def item(self):
        return self._total


def _conditioning(total=1.5, meta=None):
    """A CONDITIONING output shaped exactly as ``run_graph`` normalises it:
    a 1-tuple holding a LIST of ``[tensor, dict]`` entries."""
    return ([[_FakeTensor(total=total), dict(meta or {"pooled": 1})]],)


# --- the kill switch: opt-OUT, so unset and empty both mean ON -------------- #
@pytest.mark.parametrize("raw,expected", [
    (None, True), ("", True), ("   ", True),
    ("1", True), ("true", True), ("on", True), ("yes", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("OFF", False), ("  False  ", False),
])
def test_the_switch_is_opt_out_not_opt_in(monkeypatch, raw, expected):
    """The MIRROR of the voice-cast opt-IN parse, and deliberately so.

    That one had to reject ``""`` and ``"false"`` as ENABLED once its default
    flipped. This one defaults ON, so an unset or empty value must keep the
    default and only an explicit disable token may turn it off.
    """
    if raw is None:
        monkeypatch.delenv(eng_ltx25._ENCODER_CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(eng_ltx25._ENCODER_CACHE_ENV, raw)
    assert eng_ltx25._encoder_cache_enabled() is expected


# --- the cache key: a broken stat is not an identity ----------------------- #
def test_an_unresolvable_encoder_path_is_a_MISS_not_a_KEY(eng, monkeypatch):
    """``_weight_receipt`` returns ``("<unresolved>", -1, -1)`` for a missing
    path and ``(basename, -1, -1)`` on OSError -- two DIFFERENT broken states
    that can compare EQUAL. Keying on that would produce a false HIT against a
    cache entry with nothing to do with the weight now on disk."""
    monkeypatch.setattr(eng_ltx25, "_resolve", lambda folder, name: "")
    assert eng._encoder_cache_key() is None


def test_an_unstattable_encoder_path_is_a_MISS_not_a_KEY(eng, monkeypatch):
    monkeypatch.setattr(eng_ltx25, "_resolve",
                        lambda folder, name: "/nope/does/not/exist.gguf")

    def _boom(_path):
        raise OSError("gone")

    monkeypatch.setattr(eng_ltx25.os, "stat", _boom)
    assert eng._encoder_cache_key() is None


def test_the_key_carries_inode_identity_not_just_size_and_mtime(eng, tmp_path):
    """This key spans a whole EPISODE, not one segment. A swapped symlink
    target at the same size and mtime is exactly what a longer-lived cache has
    to notice and what the per-segment receipt never had to."""
    real = tmp_path / "encoder.gguf"
    real.write_bytes(b"x" * 64)
    eng_ltx25_resolve = eng_ltx25._resolve
    try:
        eng_ltx25._resolve = lambda folder, name: str(real)
        key = eng._encoder_cache_key()
    finally:
        eng_ltx25._resolve = eng_ltx25_resolve
    assert key is not None
    st = os.stat(str(real))
    assert st.st_dev in key and st.st_ino in key and st.st_size in key
    assert ("cpu", "cpu") in key, "placement belongs to the identity"


# --- the conditioning SHAPE, pinned because a near-miss depended on it ----- #
def test_a_conditioning_entry_is_a_PAIR_not_a_tensor():
    """The shape ``_copy_conditioning`` is built on, pinned deliberately.

    THIS PIN EXISTS BECAUSE OF A NEAR-MISS. Two reviewers in the 2026-08-20 arc
    proposed guarding the cached negative with ``neg[0][0].sum()``. That reaches
    the ``[tensor, dict]`` ENTRY -- a list -- so it would have raised
    ``AttributeError: 'list' object has no attribute 'sum'`` on EVERY cache hit.
    The tensor is one subscript deeper, at ``out[0][0][0]``.

    The digest itself was CUT on the panel's own later advice (a whole-tensor
    reduction cannot reliably catch cancelling mutations, and it re-ran twice a
    shot to guard a hazard three independent code reads had already ruled out).
    What survives is this: the defensive copy still relies on an entry being a
    ``[tensor, dict]`` pair.

    SCOPE, STATED HONESTLY (a review lane asked for exactly this): the fixture
    is a LOCAL FAKE, so this pins the SHAPE THE HELPER IS BUILT AGAINST -- it
    is a harness-integrity check, not a probe of the installed ComfyUI. It
    cannot notice an upstream change on its own. The upstream shape was
    established by READING it: ``CLIPTextEncode`` returns
    ``(clip.encode_from_tokens_scheduled(tokens), )`` (``nodes.py:77``) and
    ``node_helpers.conditioning_set_values`` builds entries as
    ``n = [t[0], t[1].copy()]`` (``node_helpers.py:9-23``). If ComfyUI ever
    changes that, this test stays green and the copier breaks -- so the real
    guard is that the fixture and those two citations move together.
    """
    out = _conditioning()
    entry = out[0][0]
    assert isinstance(entry, (list, tuple)) and len(entry) == 2
    assert not hasattr(entry, "sum"), "the ENTRY is not the tensor"
    assert hasattr(entry[0], "sum"), "the tensor is at out[0][0][0]"
    assert isinstance(entry[1], dict), "the metadata is the second element"


# --- the defensive copy ---------------------------------------------------- #
def test_the_copy_is_private_in_the_list_and_the_dict():
    out = _conditioning()
    copied = eng_ltx25._copy_conditioning(out)
    assert copied[0] is not out[0], "the outer list must be private"
    assert copied[0][0] is not out[0][0], "the entry must be private"
    assert copied[0][0][1] is not out[0][0][1], "the metadata must be private"
    copied[0][0][1]["frame_rate"] = 25.0
    assert "frame_rate" not in out[0][0][1], "the cached entry was mutated"


def test_the_copy_SHARES_the_tensor_on_purpose():
    """Copying the tensor would defeat the point -- it is the expensive half
    and nothing on this graph writes into it."""
    out = _conditioning()
    assert eng_ltx25._copy_conditioning(out)[0][0][0] is out[0][0][0]


def test_the_copy_still_guards_when_the_conditioning_is_a_TUPLE():
    """A narrower ``isinstance(cond, list)`` check would silently SKIP the copy
    for a tuple -- a guard that quietly stops guarding while the receipts still
    say it is on. The container type is preserved on the way back out."""
    out = ((( _FakeTensor(), {"pooled": 1}),),)
    copied = eng_ltx25._copy_conditioning(out)
    assert isinstance(copied[0], tuple), "the container type must survive"
    assert copied[0][0][1] is not out[0][0][1], "the metadata must be private"
    assert copied[0][0][0] is out[0][0][0], "the tensor is still shared"


@pytest.mark.parametrize("bad", [None, (), ("scalar",), 7])
def test_the_copy_passes_unrecognised_shapes_through_untouched(bad):
    assert eng_ltx25._copy_conditioning(bad) is bad


# --- the liveness check on a cached handle --------------------------------- #
class _Patcher:
    def __init__(self, load="cpu", offload="cpu", model=object()):
        self.load_device = load
        self.offload_device = offload
        self.model = model


class _Clip:
    def __init__(self, patcher):
        self.patcher = patcher


def test_a_cpu_pinned_cached_clip_is_live():
    assert eng_ltx25.Ltx25VideoEngine._cached_clip_is_live((_Clip(_Patcher()),))


@pytest.mark.parametrize("patcher", [
    _Patcher(load="cuda:0"),
    _Patcher(offload="cuda:0"),
    _Patcher(load=None, offload=None),
    _Patcher(model=None),
])
def test_a_clip_that_drifted_off_the_cpu_is_NOT_live(patcher):
    """Falling through to a full load is the correct response, not a raise:
    that load runs the pinned loader, which owns the loud refusal already."""
    assert not eng_ltx25.Ltx25VideoEngine._cached_clip_is_live((_Clip(patcher),))


@pytest.mark.parametrize("bad", [None, (), (None,), ("no patcher",)])
def test_a_malformed_cache_entry_is_NOT_live(bad):
    assert not eng_ltx25.Ltx25VideoEngine._cached_clip_is_live(bad)


# --- ownership ------------------------------------------------------------- #
def test_no_scope_open_means_exactly_todays_behaviour(eng):
    """Caching is a property of running an EPISODE, not of being this engine.
    A direct render_single diagnostic or a unit test must load per shot."""
    assert eng._encoder_scope is None


def test_begin_claims_the_scope_and_end_releases_it(eng):
    eng.begin_encoder_scope()
    assert eng._encoder_scope == {}
    eng.end_encoder_scope()
    assert eng._encoder_scope is None


def test_begin_is_idempotent_and_never_discards_a_live_cache(eng):
    """The driver resolves this engine on every shot and must not have to
    remember which call was the first."""
    eng.begin_encoder_scope()
    eng._encoder_scope["key"] = "sentinel"
    eng.begin_encoder_scope()
    assert eng._encoder_scope.get("key") == "sentinel"
    eng.end_encoder_scope()


def test_end_is_idempotent_because_it_runs_from_a_finally(eng):
    eng.end_encoder_scope()
    eng.end_encoder_scope()
    assert eng._encoder_scope is None


def test_the_kill_switch_refuses_to_open_a_scope(eng, monkeypatch):
    monkeypatch.setenv(eng_ltx25._ENCODER_CACHE_ENV, "0")
    eng.begin_encoder_scope()
    assert eng._encoder_scope is None


def test_two_overlapping_owners_both_have_to_let_go(eng):
    """THE CONCURRENCY BUG THE r3 PANEL CAUGHT.

    The engine is a process-wide singleton, but each ``run_episode`` keeps its
    OWN private ownership map. The POST harness routes that used to start
    UNGUARDED DAEMON THREADS were removed 2026-09-05, but overlap is still
    reachable -- `scripts/` drives the same `render_driver` entry points
    in-process -- so the ownership contract still has to hold.

    With a bare flag, whichever run finished FIRST set the scope to ``None`` --
    and the survivor's remaining beats ran cold while its private map still
    said "opened", so it never reopened. Silent, and it restored one 8.86 GiB
    reload per beat for the rest of that episode.
    """
    eng.begin_encoder_scope()
    eng.begin_encoder_scope()          # a second, overlapping owner
    eng._encoder_scope["clip"] = "sentinel"

    eng.end_encoder_scope()            # the first run finishes
    assert eng._encoder_scope is not None, (
        "the first run to finish released a cache the second was still using")
    assert eng._encoder_scope.get("clip") == "sentinel"

    eng.end_encoder_scope()            # the last owner lets go
    assert eng._encoder_scope is None


def test_the_owner_count_never_goes_negative(eng):
    """``end`` runs from a ``finally`` and may be called without a matching
    ``begin``; an unbalanced close must not push the count below zero and
    strand the NEXT episode's scope open forever."""
    eng.end_encoder_scope()
    eng.end_encoder_scope()
    assert eng._encoder_scope_owners == 0
    eng.begin_encoder_scope()
    assert eng._encoder_scope is not None
    eng.end_encoder_scope()
    assert eng._encoder_scope is None


def test_the_kill_switch_RELEASES_rather_than_decrements(eng):
    """"Off" means the memory goes, not that one of several owners lets go.

    A refcount decrement here would leave 8.86 GiB resident whenever more than
    one owner held the scope -- which is exactly when it matters.
    """
    eng.begin_encoder_scope()
    eng.begin_encoder_scope()
    eng.release_encoder_cache()
    assert eng._encoder_scope is None
    assert eng._encoder_scope_owners == 0


def test_a_stale_close_after_a_kill_switch_release_cannot_shut_the_NEXT_scope(eng):
    """THE ABA FAILURE THE r4 PANEL FOUND, and it needs NO thread race::

        begin(A)                  # owners 1
        release_encoder_cache()   # kill switch: scope gone, owners 0
        begin(B)                  # a brand-new scope
        end(A)                    # ...used to close B's scope, not A's

    Nothing distinguished A's outstanding close from B's live ownership, so the
    documented mid-run escape hatch was itself a way to strand the next
    episode cold. A release now bumps the GENERATION, so A's token is stale and
    its close is a logged no-op.
    """
    token_a = eng.begin_encoder_scope()
    eng.release_encoder_cache()
    token_b = eng.begin_encoder_scope()
    assert token_b != token_a, "a new scope must not reuse the old generation"
    eng._encoder_scope["clip"] = "B's encoder"

    eng.end_encoder_scope(token_a)          # A's late, stale close
    assert eng._encoder_scope is not None, "a stale close shut the new scope"
    assert eng._encoder_scope.get("clip") == "B's encoder"

    eng.end_encoder_scope(token_b)          # B's own close still works
    assert eng._encoder_scope is None


def test_a_token_from_a_still_valid_generation_closes_normally(eng):
    """The guard must not break the ordinary path it is protecting."""
    token = eng.begin_encoder_scope()
    eng.end_encoder_scope(token)
    assert eng._encoder_scope is None


def test_a_tokenless_close_keeps_the_old_unconditional_behaviour(eng):
    """``token=None`` is the compatibility path for any caller that does not
    carry one; it must still decrement."""
    eng.begin_encoder_scope()
    eng.end_encoder_scope()
    assert eng._encoder_scope is None


def test_the_scope_is_per_instance_not_shared_by_the_class(eng):
    """The class-level default keeps the registry's zero-arg construction
    cheap, but a write must land on the INSTANCE or every engine in the
    process would share one cache."""
    other = eng_ltx25.Ltx25VideoEngine()
    eng.begin_encoder_scope()
    try:
        assert other._encoder_scope is None
        assert eng_ltx25.Ltx25VideoEngine._encoder_scope is None
    finally:
        eng.end_encoder_scope()
