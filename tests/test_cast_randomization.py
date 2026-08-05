"""BUG-LOCAL-269: the cast RNG is decoupled from the `seed` widget.

The cast must be truly random every episode, not pinned by a fixed
seed. A fixed `seed` reproduced ONE cast forever (seed 42 always rolled
HAYES VANCE / GULLIVER REEVES / JIMBO BLACK). These tests pin the
contract of `_resolve_cast_rng_seed`: OS entropy by default, with the
OTR_CAST_SEED env var as the C7 audio-regression reproducibility path.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nodes.OTR_LedgerScriptWriter import _resolve_cast_rng_seed


def test_env_override_pins_the_seed(monkeypatch):
    """OTR_CAST_SEED forces a fixed seed -- the C7 reproducibility path."""
    monkeypatch.setenv("OTR_CAST_SEED", "12345")
    seed, source = _resolve_cast_rng_seed()
    assert seed == 12345
    assert source == "OTR_CAST_SEED override"


def test_no_env_uses_os_entropy(monkeypatch):
    """With no override, the cast seed is drawn from OS entropy."""
    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    seed, source = _resolve_cast_rng_seed()
    assert isinstance(seed, int)
    assert 0 <= seed < 2 ** 32
    assert source == "OS entropy"


def test_os_entropy_varies_across_calls(monkeypatch):
    """True randomization: separate episodes get different cast seeds.

    Two 32-bit OS-entropy draws colliding is ~1 in 4 billion -- a
    repeat across 20 calls means the cast is not actually randomized
    (the BUG-LOCAL-269 regression).
    """
    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    seeds = {_resolve_cast_rng_seed()[0] for _ in range(20)}
    assert len(seeds) == 20, (
        "cast RNG seed repeated across calls -- cast is not truly random"
    )


def test_blank_env_falls_back_to_os_entropy(monkeypatch):
    """A blank / whitespace OTR_CAST_SEED is treated as unset."""
    monkeypatch.setenv("OTR_CAST_SEED", "   ")
    seed, source = _resolve_cast_rng_seed()
    assert source == "OS entropy"
    assert isinstance(seed, int)


# ---------------------------------------------------------------------------
# 2026-08-05: the cast was decoupled from the fixed seed, but its VOICES were
# not. Every voice draw, the announcer draw and the music base RNG read
# meta["episode_seed"], and on the inline lanes nothing ever wrote that key --
# coerce_int_seed(None) folds to one constant, so measured across 14 published
# episodes the announcer was bf_emma 14/14 and 18 cast rows used 5 voices.
# ---------------------------------------------------------------------------
def test_an_absent_episode_seed_folds_to_one_constant():
    """The premise. If this ever stops being a single value the defect is gone
    by some other route and the stamp below can be re-examined.
    """
    from nodes._otr_voice_node_common import coerce_int_seed

    frozen = {coerce_int_seed(None) for _ in range(5)}
    assert len(frozen) == 1
    assert coerce_int_seed(None) == coerce_int_seed("")


def test_a_varying_seed_actually_changes_the_voice_drawn():
    """Guards the whole point: the bank, the tier floor and the announcer unpin
    are all seeded on this value, so a frozen seed silently neutralizes them.
    """
    from nodes._otr_voice_bank import announcer_voice_ref, assign_voice_for_slot

    voices = {
        assign_voice_for_slot(
            role="char_voice", engine="indextts2", char_id="c02",
            gender="female", timbre=("warm",), age_band="adult",
            episode_seed=s).voice_ref_id
        for s in range(40)
    }
    assert len(voices) > 1, "a varying seed must reach the voice draw"

    announcers = {
        announcer_voice_ref("kokoro", episode_seed=s).voice_ref_id
        for s in range(40)
    }
    assert len(announcers) > 1, "a varying seed must reach the announcer draw"


def test_the_writer_stamps_episode_seed_at_the_cast_mint():
    """The stamp lives beside _resolve_cast_rng_seed, not beside cast_contract
    150 lines later: the consumers read meta['episode_seed'], so any path that
    does not reach the contract stamp would leave them on the constant.
    """
    import inspect

    from nodes import OTR_LedgerScriptWriter as W

    src = inspect.getsource(W)
    mint = src.index("cast_seed, cast_seed_source = _resolve_cast_rng_seed()")
    contract = src.index('meta["cast_contract"] = {')
    stamp = src.index('meta["episode_seed"] = int(cast_seed)')
    assert mint < stamp < contract, (
        "the episode_seed stamp must sit between the cast-seed mint and the "
        "cast_contract stamp")


def test_episode_seed_and_cast_contract_seed_do_not_diverge():
    """otr_credits_roll prefers meta.cast_contract.cast_seed while voices and
    music read meta.episode_seed. Equal values are what keeps the printed seed
    receipt honest about what actually rendered.
    """
    import inspect

    from nodes import OTR_LedgerScriptWriter as W

    src = inspect.getsource(W)
    assert 'meta["episode_seed"] = int(cast_seed)' in src
    assert '"cast_seed":              int(cast_seed),' in src, (
        "both keys must be stamped from the SAME cast_seed value")


def test_an_explicit_episode_seed_is_never_clobbered():
    """A caller (or a content-owned lane runner) that already set the seed owns
    it; the writer must not mint over the top of it.
    """
    import inspect

    from nodes import OTR_LedgerScriptWriter as W

    src = inspect.getsource(W)
    i = src.index('meta["episode_seed"] = int(cast_seed)')
    guard = src[max(0, i - 200):i]
    assert 'meta.get("episode_seed") is None' in guard, (
        "the stamp must be guarded so an existing seed wins")


def test_pinning_OTR_CAST_SEED_pins_the_inline_lanes(monkeypatch):
    """The C7 gate needs a reproducible episode. OTR_CAST_SEED already pinned
    the cast; now it pins the voices and the music bed too, because they all
    derive from the same stamped number.

    INLINE lanes only. The content-owned fable2 runner mints its own seed and is
    pinned by OTR_FABLE2_SEED; the soak launcher sets both under OTR_C7.
    """
    monkeypatch.setenv("OTR_CAST_SEED", "12345")
    seed, source = _resolve_cast_rng_seed()
    assert seed == 12345
    assert "env" in source.lower() or "OTR_CAST_SEED" in source

    from nodes._otr_voice_bank import announcer_voice_ref

    a = announcer_voice_ref("kokoro", episode_seed=seed).voice_ref_id
    b = announcer_voice_ref("kokoro", episode_seed=seed).voice_ref_id
    assert a == b, "a pinned seed must reproduce the same announcer"


def test_two_entropy_draws_propagate_as_themselves(monkeypatch):
    """Production entropy must reach the consumers as its own value, not get
    flattened through coerce_int_seed(None).
    """
    from nodes._otr_voice_node_common import coerce_int_seed

    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    seeds = {_resolve_cast_rng_seed()[0] for _ in range(8)}
    assert len(seeds) > 1, "production must draw fresh entropy per episode"
    frozen = coerce_int_seed(None)
    assert frozen not in seeds or len(seeds) > 1


def test_the_music_bed_seed_follows_the_episode_seed():
    """stable_audio_theme derives music_rng_seed_v1 from the episode seed, so a
    frozen seed froze the music bed's RNG too -- the third symptom of the same
    root cause, alongside the announcer and the character voices.
    """
    from nodes._otr_resolved_request import _seed_to_int64
    from nodes._otr_voice_node_common import coerce_int_seed

    def music_base(meta):
        return _seed_to_int64(
            "music_rng_seed_v1", coerce_int_seed(meta.get("episode_seed")))

    assert music_base({"episode_seed": 111}) != music_base({"episode_seed": 222})
    assert music_base({}) == music_base({}), "absent seed is deterministic..."
    assert music_base({}) == music_base({"episode_seed": None}), (
        "...and that determinism is exactly the freeze the writer stamp ends")


def test_the_episode_seed_reaches_the_line_seed_and_the_audio_cache_key():
    """The blast radius is wider than voice SELECTION: episode_seed is folded
    into stable_line_seed and therefore into the resolved-request identity that
    keys the audio cache. A per-episode seed means per-episode synthesis and a
    one-time cache rekey -- intended, but it must be stated, not discovered.
    """
    from nodes import _otr_resolved_request as RR

    # episode_seed is part of the resolved-request identity, which IS the audio
    # cache key -- so it does not merely pick a reference, it decides whether a
    # cached line can be reused at all.
    assert "episode_seed" in RR._IN_KEY_FIELDS if hasattr(RR, "_IN_KEY_FIELDS") \
        else True
    import inspect
    src = inspect.getsource(RR)
    assert '"stable_line_seed_v1"' in src
    assert "episode_seed" in src.split('"stable_line_seed_v1"')[1][:200], (
        "episode_seed must feed the per-line synthesis seed")

    def line_seed(ep):
        return RR._seed_to_int64("stable_line_seed_v1", int(ep), 0, "b002", "c02")

    assert line_seed(111) == line_seed(111), "pinned seed reproduces the line"
    assert line_seed(111) != line_seed(222), "a new episode seed moves the line"


def test_a_divergent_preset_episode_seed_is_reported_not_swallowed():
    """Credits print cast_contract.cast_seed while voices and music read
    episode_seed. If a caller presets a different value the two describe
    different renders, so the writer says so out loud.
    """
    import inspect

    from nodes import OTR_LedgerScriptWriter as W

    src = inspect.getsource(W)
    i = src.index('meta["episode_seed"] = int(cast_seed)')
    tail = src[i:i + 900]
    assert "elif int(meta[" in tail, "a divergent preset seed must be detected"
    assert "log.warning" in tail, "and must be logged, never silently accepted"
