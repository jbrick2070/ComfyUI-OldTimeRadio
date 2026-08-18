"""VOICE IDENTITY -- a character must not change voice between his own lines.

PBUG-20260817-09. The operator heard it on a published episode: *"Nag 1 sounded
good, Nag beat 2 was another voice."* NAG (`c03`) in
`signal_lost_mongooses_stand_20260817_234050`, same reference WAV both times:

    b003  ref=vz_donor_glenn  alpha=1.0  delivery=v2:nonzero(derived)  seed=532084266468738542
    b005  ref=vz_donor_glenn  alpha=1.0  delivery=v2:nonzero(derived)  seed=5038394939402288039

TWO CAUSES, AND THEY COMPOUND.

1. A DIFFERENT SEED PER LINE. The engine seed was reduced from
   `stable_line_seed`, which includes `line_id`, so every line one character
   speaks drew its own seed. IndexTTS2 samples autoregressively under that seed;
   the further the conditioning sits from the speaker's own embedding, the more
   audible the draw becomes.

2. THE EMOTION BLEND SPENT THE WHOLE SPEAKER. Read from the installed
   `indextts/infer_v2.py`, the vendor combines
   `emovec = emovec_mat + (1 - sum(weight_vector)) * emovec`, where the
   right-hand term is the speaker's OWN emotional embedding. The vector's SUM is
   therefore a budget spent AGAINST him. At `alpha=1.0` a neutral line stamped
   `calm=1.0` leaves `1 - 1.0 == 0.0` of him -- the quiet beat is the WORST case,
   not the safest one.

   Carry the operator's caveat verbatim: that is the EMOTION-LATENT BLEND, not
   literally "26% of his vocal tract".

WHAT THESE TESTS ASSERT, AND WHY IT IS THE PRODUCT AND NOT THE ABSENCE OF A
CRASH. This repo has shipped a fix wrapped in a guard that raised on every
episode while all 10,905 tests passed, because the tests asserted that nothing
raised. So the seed tests here drive the REAL dispatch (`BatchCharacterVoices.
generate` -> `_render_per_line`) and read the seeds the adapter was ACTUALLY
called with, and the payload tests read the JSON line actually written to the
worker's stdin. A helper that computes the right number and is never reached
fails these.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import re

import pytest
import torch

from nodes import _otr_voice_node_common as VNC
from nodes._otr_audio_engines import get_engine
from nodes._otr_audio_engines.eng_indextts2 import (
    EFFECTIVE_EMOTION_MASS_CAP, EMO_ALPHA_DEFAULT,
)
from nodes._otr_resolved_request import _seed_to_int64, build_resolved_request

EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic",
            "surprised", "calm")


# --------------------------------------------------------------------------- #
# Fixtures: a two-character ledger and a reference WAV that resolves on disk
# --------------------------------------------------------------------------- #
def _reference_wav(tmp_path, name="vz_donor_glenn.wav"):
    """An absolute path the shared resolver passes through untouched."""
    path = tmp_path / name
    path.write_bytes(b"RIFF" + b"\0" * 40)
    return str(path)


def _ledger(ref_path, tension=0.0):
    """NAG speaks twice (b003 emotional, b005 neutral); GLENN speaks once."""
    return {
        "meta": {"episode_seed": 20260817, "cast_lock_revision": "1"},
        "cast": [
            {"char_id": "c03", "name": "NAG", "gender": "male",
             "voice_ref_id": "vz_donor_glenn", "voice_ref_path": ref_path},
            {"char_id": "c01", "name": "GLENN", "gender": "male",
             "voice_ref_id": "vz_donor_glenn", "voice_ref_path": ref_path},
        ],
        "lines": [
            {"line_id": "b003", "char_id": "c03", "speaker_role": "character",
             "text": "Help me, the danger is behind you!",
             "scene_tension": tension},
            {"line_id": "b005", "char_id": "c03", "speaker_role": "character",
             "text": "The kettle is on the stove.", "scene_tension": tension},
            {"line_id": "b004", "char_id": "c01", "speaker_role": "character",
             "text": "The kettle is on the stove.", "scene_tension": tension},
        ],
    }


def _stub_indextts2(monkeypatch):
    """Record every (line-order) call the dispatch makes to the adapter.

    Only `generate_voice` is replaced -- the seed, the reference and the
    delivery vector all arrive exactly as the real dispatch computed them.
    """
    from nodes import _otr_engine_profiles as ep

    adapter = get_engine("indextts2")
    calls = []

    def _gen(text, ref_clip_path, delivery_vector, seed):
        calls.append({"text": text, "ref": ref_clip_path, "seed": seed,
                      "delivery": delivery_vector,
                      "emotion": adapter.emotion_payload(delivery_vector)})
        return {"waveform": torch.zeros(1, 8, dtype=torch.float32),
                "sample_rate": 22050}

    monkeypatch.setattr(adapter, "generate_voice", _gen)
    monkeypatch.setattr(ep, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(ep, "assert_model_available", lambda p: None)
    return adapter, calls


def _render(monkeypatch, ledger):
    """Drive the shipped node end to end; return (calls, log lines)."""
    from nodes.batch_character_voices import BatchCharacterVoices

    _adapter, calls = _stub_indextts2(monkeypatch)
    _audio, log, _done = BatchCharacterVoices().generate(
        script_json=json.dumps(ledger), engine="indextts2")
    return calls, log.splitlines()


def _seeds_by_line(calls, ledger):
    """Map line_id -> the seed the adapter was called with, by dispatch order."""
    ordered = [ln["line_id"] for ln in ledger["lines"]]
    assert len(calls) == len(ordered), (
        "every line must have rendered -- got %d calls for %d lines"
        % (len(calls), len(ordered)))
    return {lid: c["seed"] for lid, c in zip(ordered, calls)}


# --------------------------------------------------------------------------- #
# THE DEFECT, AT THE REAL DISPATCH SITE
# --------------------------------------------------------------------------- #
def test_one_character_two_lines_now_draw_ONE_engine_seed(tmp_path, monkeypatch):
    """The whole bug, in one assertion, measured where it actually happened.

    Pre-fix these two seeds were 532084266468738542 and 5038394939402288039.
    """
    ledger = _ledger(_reference_wav(tmp_path))
    calls, _log = _render(monkeypatch, ledger)
    seeds = _seeds_by_line(calls, ledger)

    assert seeds["b003"] == seeds["b005"], (
        "NAG drew a different engine seed for his own two lines -- this is the "
        "defect the operator heard")


def test_two_different_characters_still_draw_different_seeds(tmp_path, monkeypatch):
    """A fix that gave the whole cast one voice would be worse than the bug.

    b005 and b004 are the SAME TEXT on purpose, so only the character differs.
    """
    ledger = _ledger(_reference_wav(tmp_path))
    calls, _log = _render(monkeypatch, ledger)
    seeds = _seeds_by_line(calls, ledger)

    assert seeds["b005"] != seeds["b004"], (
        "two characters speaking identical text collapsed onto one seed")


def test_every_line_actually_rendered(tmp_path, monkeypatch):
    """THE PRODUCT EXISTS. A guarded fix that silently rendered nothing, or
    swallowed its own NameError, would pass a 'nothing raised' test and fail
    this one."""
    ledger = _ledger(_reference_wav(tmp_path))
    calls, log = _render(monkeypatch, ledger)

    assert len(calls) == 3
    assert all(c["ref"] for c in calls), "a line rendered with no reference"
    assert sum(1 for line in log if "->" in line and "seed=" in line) == 3


def test_the_receipt_reports_the_policy_the_mass_and_the_seed(tmp_path, monkeypatch):
    """P-OBS is the evidence the 2x2 proof is read from, so it must carry the
    new facts: which seed policy ran, and how much emotion mass was spent."""
    ledger = _ledger(_reference_wav(tmp_path))
    _calls, log = _render(monkeypatch, ledger)
    pobs = [line for line in log if "seed=" in line and "policy=" in line]

    assert len(pobs) == 3, log
    for line in pobs:
        assert "policy=char_v1" in line, line
        mass = re.search(r"emo_mass=([0-9.]+)", line)
        assert mass and float(mass.group(1)) <= EFFECTIVE_EMOTION_MASS_CAP, line
        assert "seed_ref=vz_donor_glenn" in line, line


def test_the_seed_survives_a_malformed_stamped_delivery_vector(tmp_path, monkeypatch):
    """THE LAW: a render degrades, never raises [QA-4].

    A delivery vector is hand-editable JSON. P-OBS used to call `float()` on the
    RAW stamped values before any sanitation, so a ledger carrying a string
    where a number belongs raised ValueError out of the OBSERVABILITY line and
    killed the episode. The assertion is that the audio still exists, not merely
    that nothing raised.
    """
    ledger = _ledger(_reference_wav(tmp_path))
    for line in ledger["lines"]:
        line["delivery"] = {
            "version": "v2",
            "emotion_vector": {"happy": "very", "angry": 99, "sad": -4,
                               "calm": None, "surprised": float("nan")},
        }
    calls, log = _render(monkeypatch, ledger)

    assert len(calls) == 3, "a malformed stamped vector killed the render"
    for call in calls:
        assert call["emotion"]["effective_mass"] <= EFFECTIVE_EMOTION_MASS_CAP
    assert all(("delivery=v2:" in line) for line in log if "seed=" in line)


# --------------------------------------------------------------------------- #
# THE LEGACY FORMULA IS PRESERVED EXACTLY [QA-1]
# --------------------------------------------------------------------------- #
class _Profile:
    def __init__(self, character_stable_seed=False):
        self.character_stable_seed = character_stable_seed
        self.default_params = {}


class _PlainAdapter:
    """An engine with no emotion payload and no alpha -- the boring majority."""

    def render_time_params(self):
        return {}


def _request(char_id="c03", line_id="b003", voice_ref_id="vz_donor_glenn"):
    return build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="2",
        char_id=char_id, line_id=line_id, prepared_text="Evening.",
        voice_ref_id=voice_ref_id, episode_seed=20260817, cast_lock_revision=1)


def _seed(profile, *, char_id="c03", line_id="b003",
          voice_ref_id="vz_donor_glenn", voice_ref="/refs/glenn.wav",
          episode_seed=20260817, cast_lock_revision=1, engine="indextts2"):
    """Both phases, in production order.

    Phase one resolves the params and the seed-policy decision; phase two
    derives the seed once the reference is known. Calling only phase two would
    test a helper the dispatch might not reach -- and reaching it is half of
    what these tests are for.
    """
    runtime = VNC._begin_line_runtime(_PlainAdapter(), profile, None)
    VNC._resolve_engine_seed(
        runtime, _seed_to_int64, profile, engine,
        _request(char_id=char_id, line_id=line_id, voice_ref_id=voice_ref_id),
        char_id, episode_seed, cast_lock_revision, voice_ref_id, voice_ref)
    return runtime


def test_a_profile_that_did_not_opt_in_keeps_the_exact_legacy_seed():
    """Not `stable_line_seed` on its own -- the WHOLE legacy expression [QA-1].

    Every engine that has not opted in must render byte-identically, and the
    only way to know that is to compute the old formula here and compare.
    """
    request = _request()
    runtime = _seed(_Profile(character_stable_seed=False))

    assert runtime.seed_policy == VNC.SEED_POLICY_LINE
    assert runtime.engine_seed == _seed_to_int64(
        "indextts2", request.stable_line_seed)


def test_a_blank_char_id_falls_back_to_the_legacy_formula():
    """An anonymous line has no character to be stable about, and seeding them
    all alike would collapse them onto one voice."""
    request = _request(char_id="")
    runtime = _seed(_Profile(character_stable_seed=True), char_id="")

    assert runtime.seed_policy == VNC.SEED_POLICY_LINE
    assert runtime.engine_seed == _seed_to_int64(
        "indextts2", request.stable_line_seed)


def test_a_whitespace_char_id_is_treated_as_blank():
    runtime = _seed(_Profile(character_stable_seed=True), char_id="   ")
    assert runtime.seed_policy == VNC.SEED_POLICY_LINE


# --------------------------------------------------------------------------- #
# WHAT THE CHARACTER SEED KEYS ON
# --------------------------------------------------------------------------- #
def test_the_character_seed_ignores_the_line_and_keys_the_character():
    opted_in = _Profile(character_stable_seed=True)
    b003 = _seed(opted_in, line_id="b003")
    b005 = _seed(opted_in, line_id="b005")

    assert b003.seed_policy == VNC.SEED_POLICY_CHARACTER
    assert b003.engine_seed == b005.engine_seed


@pytest.mark.parametrize("field,value", [
    ("char_id", "c07"),
    ("episode_seed", 20260818),
    ("cast_lock_revision", 2),
    ("voice_ref_id", "vz_donor_other"),
    ("engine", "chatterbox"),
])
def test_the_character_seed_still_moves_with_real_identity_changes(field, value):
    """Stable is not frozen. A re-cast, a new episode, a bumped cast revision or
    a different engine must all draw a different voice."""
    opted_in = _Profile(character_stable_seed=True)
    base = _seed(opted_in)
    moved = _seed(opted_in, **{field: value})

    assert base.engine_seed != moved.engine_seed, field


def test_the_seed_uses_the_resolved_reference_not_a_blank(tmp_path):
    """[QA-5] On the non-cache path the request can be built BEFORE the fallback
    reference is resolved, so the seed must read the resolved value."""
    opted_in = _Profile(character_stable_seed=True)
    resolved = _seed(opted_in, voice_ref_id="",
                     voice_ref=str(tmp_path / "vz_fallback.wav"))

    assert resolved.seed_policy == VNC.SEED_POLICY_CHARACTER
    assert resolved.ref_identity == "vz_fallback.wav"
    assert resolved.engine_seed != _seed(
        opted_in, voice_ref_id="", voice_ref="").engine_seed


def test_the_seed_does_not_depend_on_where_the_models_root_lives():
    """The models root is `C:\\ComfyUI-Models` on this box and something else on
    the next one. A seed keyed on an absolute path would render one character
    two ways on two machines -- the exact defect this seed exists to end."""
    opted_in = _Profile(character_stable_seed=True)
    here = _seed(opted_in, voice_ref_id="",
                 voice_ref=r"C:\ComfyUI-Models\TTS\refs\indextts2\glenn.wav")
    there = _seed(opted_in, voice_ref_id="",
                  voice_ref="/mnt/models/TTS/refs/indextts2/glenn.wav")

    assert here.engine_seed == there.engine_seed


# --------------------------------------------------------------------------- #
# THE EMOTION-MASS CEILING [B / C / D, QA-3]
# --------------------------------------------------------------------------- #
def _vector(**weights):
    vec = {e: 0.0 for e in EMOTIONS}
    vec.update(weights)
    return vec


def test_the_neutral_line_now_keeps_most_of_the_character():
    """b005's shape: `calm=1.0`, which at alpha 1.0 left NOTHING of the speaker.

    `1 - effective_mass` is the share of his own emotional embedding that
    survives, so this asserts the fix in the units the defect was measured in.
    """
    payload = get_engine("indextts2").emotion_payload(_vector(calm=1.0))

    assert payload["effective_mass"] == pytest.approx(0.4)
    assert 1.0 - payload["effective_mass"] == pytest.approx(0.6)


def test_the_cap_is_applied_AFTER_alpha_never_before():
    """[QA-3 / D] Capping the raw vector first and letting alpha scale the
    capped result would soften the delivery TWICE -- `0.4 * 0.4 == 0.16` -- and
    quietly flatten every performance to make the arithmetic tidy."""
    payload = get_engine("indextts2").emotion_payload(
        _vector(calm=1.0), alpha=EMO_ALPHA_DEFAULT)

    assert payload["effective_mass"] == pytest.approx(0.4)
    assert payload["effective_mass"] != pytest.approx(0.16)
    assert payload["mass_capped"] is False, (
        "alpha alone already landed on the ceiling -- the cap must not fire "
        "again on top of it")


def test_a_hand_stamped_overweight_vector_is_capped_even_at_alpha_one():
    """WHY THE CEILING IS NOT JUST A SMALLER ALPHA. Alpha is a multiplier, so it
    can promise nothing about a vector it did not derive: a hand-edited ledger
    summing to 8.0 still spends 3.2 at alpha 0.4 -- eight times the speaker."""
    payload = get_engine("indextts2").emotion_payload(
        {e: 1.0 for e in EMOTIONS}, alpha=1.0)

    assert payload["mass_capped"] is True
    assert payload["effective_mass"] <= EFFECTIVE_EMOTION_MASS_CAP


@pytest.mark.parametrize("alpha", [0.0, 0.001, 0.1, 0.4, 0.401, 0.5, 0.999, 1.0])
@pytest.mark.parametrize("weight", [0.0, 0.001, 0.266, 0.734, 1.0])
def test_the_ceiling_holds_across_the_grid(alpha, weight):
    """A ceiling with a gap is not a ceiling. The rescale FLOORS to three
    decimals rather than rounding, because rounding a rescaled weight up is
    exactly how an enforced 0.4 comes back as 0.401."""
    payload = get_engine("indextts2").emotion_payload(
        {e: weight for e in EMOTIONS}, alpha=alpha)

    assert payload["effective_mass"] <= EFFECTIVE_EMOTION_MASS_CAP


def test_the_mass_is_measured_on_the_list_that_survives_serialization():
    """[QA-3] The worker receives JSON. Measuring an idealized in-memory vector
    would describe a blend the vendor never sees."""
    payload = get_engine("indextts2").emotion_payload(
        {e: 0.9 for e in EMOTIONS}, alpha=1.0)
    round_tripped = json.loads(json.dumps(payload["emo_vector"]))

    assert round_tripped == payload["emo_vector"]
    assert sum(round_tripped) <= EFFECTIVE_EMOTION_MASS_CAP


def test_the_mass_mirrors_the_vendors_own_post_alpha_truncation():
    """The vendor pre-scales the vector by alpha and TRUNCATES to four decimals
    (`int(x * scale * 10000) / 10000`) -- read from the installed
    `indextts/infer_v2.py`, not assumed. An idealized `alpha * sum` would
    disagree with what actually gets summed into `1 - sum(weight_vector)`."""
    engine = get_engine("indextts2")

    assert engine._apply_vendor_alpha([0.12345], 0.5) == [0.0617]   # not 0.061725
    assert engine._apply_vendor_alpha([0.12345], 1.0) == [0.12345]  # untouched


# --------------------------------------------------------------------------- #
# THE WIRE. What the worker is actually handed.
# --------------------------------------------------------------------------- #
class _FakeWorkerStream:
    """Captures the request line and replies with one canned success."""

    def __init__(self):
        self.written = []

    def write(self, payload):
        self.written.append(payload)

    def flush(self):
        pass


class _FakeWorker:
    def __init__(self, out_path):
        self.stdin = _FakeWorkerStream()
        self._out_path = out_path

    def poll(self):
        return None

    @property
    def stdout(self):
        outer = self

        class _Reader:
            def readline(self):
                return json.dumps({"ok": True, "out_path": outer._out_path,
                                   "sample_rate": 22050}) + "\n"
        return _Reader()


def test_the_capped_vector_is_what_reaches_the_worker(tmp_path, monkeypatch):
    """THE PRODUCT, ON THE WIRE. A cap computed for the receipt and not applied
    to the outbound payload would pass every test above and change nothing the
    operator can hear.
    """
    engine = get_engine("indextts2")
    worker = _FakeWorker(str(tmp_path / "out.wav"))

    monkeypatch.setattr(engine, "load", lambda: None)
    monkeypatch.setattr(engine, "_proc", worker, raising=False)
    monkeypatch.setattr(type(engine), "_load_wav",
                        staticmethod(lambda path, sr: {"waveform": torch.zeros(1, 4),
                                                       "sample_rate": sr}))
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "1.0")

    engine.generate_voice("Evening.", str(tmp_path / "ref.wav"),
                          {e: 1.0 for e in EMOTIONS}, 12345)

    sent = json.loads(worker.stdin.written[0])
    assert sum(sent["emo_vector"]) <= EFFECTIVE_EMOTION_MASS_CAP, sent
    assert sent["emo_alpha"] == 1.0
    assert sent["seed"] == 12345


# --------------------------------------------------------------------------- #
# SCOPE. Seeding = three clone profiles. Cap = IndexTTS2 only. [QA-6]
# --------------------------------------------------------------------------- #
def test_exactly_the_three_char_clone_profiles_opt_into_character_seeding():
    """Announcer profiles are deliberately untouched: an announcer is not a
    cloned character and has no identity to hold steady across beats."""
    from nodes._otr_engine_profiles import load_resolver

    resolver = load_resolver()
    opted_in = sorted(
        pid for pid in resolver.profile_ids()
        if getattr(resolver.get(pid), "character_stable_seed", False))

    assert opted_in == ["char_chatterbox_v1", "char_dia_v1",
                        "char_indextts2_v1"], opted_in


def test_those_three_profiles_bumped_their_impl_version():
    """[E] Stale audio is invalidated honestly: `engine_impl_version` keys the
    cache, so audio rendered under the per-line seed can never be replayed as
    if this build had made it."""
    from nodes._otr_engine_profiles import load_resolver

    resolver = load_resolver()
    for pid in ("char_indextts2_v1", "char_chatterbox_v1", "char_dia_v1"):
        assert resolver.get(pid).engine_impl_version == "2", pid


def test_the_bumped_version_moves_the_cache_key():
    """A version bump that did not move the key would invalidate nothing."""
    v1 = build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="1",
        char_id="c03", line_id="b003", prepared_text="Evening.")
    v2 = build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="2",
        char_id="c03", line_id="b003", prepared_text="Evening.")

    assert v1.cache_key != v2.cache_key


def test_the_emotion_cap_is_indextts2_only():
    """Chatterbox and dia get the character SEED, not the emotion ceiling --
    they have no emotion vector to cap."""
    from nodes._otr_audio_engines import registry as areg

    declaring = []
    for name in sorted(areg._REGISTRY):
        try:
            adapter = get_engine(name)
        except Exception:  # noqa: BLE001 -- an engine that cannot be built
            continue
        if callable(getattr(adapter, "emotion_payload", None)):
            declaring.append(name)

    assert declaring == ["indextts2"], declaring


def test_no_announcer_profile_gained_the_flag():
    from nodes._otr_engine_profiles import load_resolver

    resolver = load_resolver()
    for pid in resolver.profile_ids():
        profile = resolver.get(pid)
        if profile.role == "announcer_voice":
            assert profile.character_stable_seed is False, pid


# --------------------------------------------------------------------------- #
# LEMMY IS UNQUALIFIED AFTER THIS CHANGE [QA-7]
#
# A qualification record stores the runtime it was proved under, and
# `config/cast_pools.py` has described that field as "the sha256 of the adapter
# plus its worker script" since the day it was written -- the whole point being
# that changing the rendering code changes it. Nothing ever COMPUTED the live
# value, so nothing ever compared them: the field was stored, described, and
# never read. This fix changes IndexTTS2's seed handling and its emotion blend,
# which are precisely the two things a listener judges, so the August audition
# no longer describes what this engine does.
#
# THE DEMOTION IS A DEGRADE, NOT A FAILURE, and that distinction is the whole
# safety argument. `resolve_policy_route_claim` RAISES `VoiceRouteError` on a
# SELECTED route that cannot prove itself -- so gating inside the validator
# would have killed every episode that casts Lemmy. Gating at SELECTION reaches
# the module's own established outcome instead: nothing was selected, so nothing
# failed, the row takes the ordinary draw, and the episode still publishes.
# --------------------------------------------------------------------------- #
def _live_indextts2_fingerprint():
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    return ROUTE.live_engine_impl_version("indextts2")


def test_the_live_fingerprint_is_computable_and_reproducible():
    """A gate that cannot compute a live value would silently never fire."""
    first = _live_indextts2_fingerprint()

    assert re.fullmatch(r"[0-9a-f]{16}", first), first
    assert first == _live_indextts2_fingerprint(), "the recipe is not stable"


def test_an_engine_with_no_recipe_is_never_demoted_on_this_ground():
    """Silence, not a guess: an unknown fingerprint is not evidence of a
    changed one."""
    from nodes import _otr_voice_route as ROUTE

    assert ROUTE.live_engine_impl_version("bark") == ""
    record = {"route_id": "r",
              "qualification_record": {"runtime": {"engine_impl_version": "zz"}}}
    assert ROUTE.stale_runtime_fingerprint(record, "bark") is None


def test_the_shipped_lemmy_route_is_no_longer_selected():
    """The release gate, asserted: no new qualification, no selected route."""
    from config import cast_pools as POOLS
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    policy = POOLS.LEMMY_VOICE_POLICY

    assert ROUTE.select_policy_route(policy, "indextts2") is None


def test_the_lemmy_record_itself_is_preserved_unedited():
    """PRESERVE THE OLD RECORD [QA-7]. The claim is withdrawn by the gate, not
    by deleting the evidence -- a re-audition needs something to compare to."""
    from config import cast_pools as POOLS

    record = POOLS.LEMMY_VOICE_POLICY["approved_native_routes"]["indextts2"]
    qual = record["qualification_record"]

    assert record["route_id"] == "lemmy-indextts2-algenib-cockney-v1"
    assert qual["record_id"] == "g1-test-a-2026-08-10"
    assert qual["status"] == "qualified"
    assert qual["rights"]["status"] == "approved"
    assert qual["runtime"]["engine_impl_version"] == "b965453f355661a3"


def test_a_re_qualified_record_selects_again():
    """The route is recoverable, and the ONLY thing that recovers it is a
    record whose runtime matches the build that will render it."""
    from config import cast_pools as POOLS
    from nodes import _otr_voice_route as ROUTE

    live = _live_indextts2_fingerprint()
    requalified = copy.deepcopy(
        POOLS.LEMMY_VOICE_POLICY["approved_native_routes"]["indextts2"])
    requalified["qualification_record"]["runtime"]["engine_impl_version"] = live
    policy = {"policy_version": "test",
              "approved_native_routes": {"indextts2": requalified}}

    selected = ROUTE.select_policy_route(policy, "indextts2")
    assert selected is not None
    assert selected["route_id"] == "lemmy-indextts2-algenib-cockney-v1"


def test_the_demotion_degrades_and_never_raises():
    """THE LAW: an audit may never FAIL an episode; a render degrades.

    `resolve_policy_route_claim` is the call CastLock makes, and it raises on a
    SELECTED route that fails its checks. A demoted route must come back as
    None from here -- not as an exception that aborts the episode.
    """
    from datetime import datetime, timezone

    from config import cast_pools as POOLS
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    claim = ROUTE.resolve_policy_route_claim(
        POOLS.LEMMY_VOICE_POLICY, "indextts2", datetime.now(timezone.utc),
        bank_entries=[])

    assert claim is None


def test_an_unknown_active_engine_is_still_loud():
    """The demotion must not have widened into a general "say nothing" -- the
    one case this module exists to shout about still shouts."""
    from config import cast_pools as POOLS
    from nodes import _otr_voice_route as ROUTE

    with pytest.raises(ROUTE.VoiceRouteError):
        ROUTE.select_policy_route(POOLS.LEMMY_VOICE_POLICY, "")


def test_a_record_that_claims_no_fingerprint_is_not_contradicted():
    """Only a record that CLAIMS a runtime can contradict one. This keeps the
    gate off records the validator judges by its own rules."""
    from nodes import _otr_voice_route as ROUTE

    for runtime in ({}, {"engine_impl_version": ""}, {"engine_impl_version": None}):
        record = {"route_id": "r",
                  "qualification_record": {"runtime": runtime}}
        assert ROUTE.stale_runtime_fingerprint(record, "indextts2") is None


# --------------------------------------------------------------------------- #
# THE CONTROL ARM. `OTR_VOICE_CHARACTER_SEED=0` restores the pre-fix seed.
#
# QA-8 requires a 2x2 -- line-vs-character seed against alpha 1.0 vs 0.4, one
# fresh server boot per environment arm -- and that axis is unrunnable without a
# selector. This mirrors `OTR_DELIVERY_VECTOR=0`, which the dispatch already
# documents as "a TRUE old path", and it doubles as the operator's one-line
# rollback if he dislikes what the character seed sounds like.
# --------------------------------------------------------------------------- #
def test_the_kill_switch_restores_the_exact_legacy_seed(monkeypatch):
    """The control arm has to be the REAL old path, not an approximation of it,
    or the 2x2 compares the fix against something that never shipped."""
    monkeypatch.setenv("OTR_VOICE_CHARACTER_SEED", "0")
    opted_in = _Profile(character_stable_seed=True)
    runtime = _seed(opted_in)

    assert runtime.seed_policy == VNC.SEED_POLICY_LINE
    assert runtime.engine_seed == _seed_to_int64(
        "indextts2", _request().stable_line_seed)


def test_the_kill_switch_reproduces_the_defect_for_the_control_arm(
        tmp_path, monkeypatch):
    """The control arm must actually SPLIT a character's seeds again -- an arm
    that quietly ran the fix would make the comparison meaningless."""
    monkeypatch.setenv("OTR_VOICE_CHARACTER_SEED", "0")
    ledger = _ledger(_reference_wav(tmp_path))
    calls, log = _render(monkeypatch, ledger)
    seeds = _seeds_by_line(calls, ledger)

    assert seeds["b003"] != seeds["b005"], (
        "the control arm did not reproduce the pre-fix behaviour")
    assert all("policy=line_v1" in line for line in log if "policy=" in line)


def test_the_seed_policy_is_recorded_in_the_cache_key(monkeypatch):
    """A KNOB THAT MOVES THE RENDER MUST MOVE THE KEY (Lemmy chunk A1).

    Otherwise the two arms of the 2x2 share a cache key while producing
    different audio, and the second arm replays the first one's voice under a
    receipt describing its own.
    """
    opted_in = _Profile(character_stable_seed=True)

    monkeypatch.setenv("OTR_VOICE_CHARACTER_SEED", "1")
    on = VNC._begin_line_runtime(_PlainAdapter(), opted_in, None)
    monkeypatch.setenv("OTR_VOICE_CHARACTER_SEED", "0")
    off = VNC._begin_line_runtime(_PlainAdapter(), opted_in, None)

    assert on.params["voice_seed_policy"] == VNC.SEED_POLICY_CHARACTER
    assert off.params["voice_seed_policy"] == VNC.SEED_POLICY_LINE

    keyed_on = build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="2",
        char_id="c03", line_id="b003", prepared_text="Evening.",
        params=on.params)
    keyed_off = build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="2",
        char_id="c03", line_id="b003", prepared_text="Evening.",
        params=off.params)

    assert keyed_on.cache_key != keyed_off.cache_key


def test_a_profile_that_never_opted_in_contributes_no_new_key_material():
    """Every other engine must key byte-identically to before the knob existed."""
    plain = VNC._begin_line_runtime(_PlainAdapter(), _Profile(False), None)

    assert "voice_seed_policy" not in plain.params
    assert plain.params == {}


def test_the_receipt_describes_THE_PAYLOAD_THAT_REACHED_THE_WORKER(
        tmp_path, monkeypatch):
    """QA-2's most important seam, closed by a test instead of by convention.

    The dispatch resolves the emotion payload once for the RECEIPT
    (`_begin_line_runtime`) and the adapter resolves it again for the WIRE
    (`generate_voice`). They agree because `emotion_payload` is pure and both
    read the same env through the same function -- but "agree by construction"
    is a claim, and an unguarded claim is how the alpha once keyed one value
    while the forward rendered another (Lemmy chunk A1).

    So this stubs ONLY the worker process. `generate_voice` runs for real, the
    dispatch runs for real, and the JSON on the worker's stdin is compared
    against the P-OBS line the operator will read. If the two resolutions ever
    drift, this fails and nothing else does.
    """
    from nodes import _otr_engine_profiles as ep
    from nodes.batch_character_voices import BatchCharacterVoices

    engine = get_engine("indextts2")
    worker = _FakeWorker(str(tmp_path / "out.wav"))
    monkeypatch.setattr(engine, "load", lambda: None)
    monkeypatch.setattr(engine, "_proc", worker, raising=False)
    monkeypatch.setattr(
        type(engine), "_load_wav",
        staticmethod(lambda path, sr: {"waveform": torch.zeros(1, 8),
                                       "sample_rate": 22050}))
    monkeypatch.setattr(ep, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(ep, "assert_model_available", lambda p: None)
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "0.4")

    ledger = _ledger(_reference_wav(tmp_path))
    _audio, log, _done = BatchCharacterVoices().generate(
        script_json=json.dumps(ledger), engine="indextts2")

    written = [json.loads(payload) for payload in worker.stdin.written]
    # The last write is the teardown handshake, not a render -- its presence
    # also proves I-7 (the engine is released in `finally`, before `done`).
    assert written and written[-1] == {"stop": True}, written
    wire = [w for w in written if "emo_vector" in w]
    receipts = [line for line in log.splitlines()
                if "seed=" in line and "policy=" in line]
    assert len(wire) == 3, "generate_voice did not reach the worker for every line"
    assert len(receipts) == len(wire)

    for sent, receipt in zip(wire, receipts):
        # The seed the receipt names is the seed the worker was given.
        assert "seed=%d " % sent["seed"] in receipt, (sent, receipt)
        # The alpha the receipt names is the alpha the worker was given.
        assert "alpha=%s " % sent["emo_alpha"] in receipt, (sent, receipt)
        # The mass the receipt reports is the mass of the WIRE's own vector,
        # under the vendor's own post-alpha transform -- not a recomputation
        # from some other copy of the delivery vector.
        stated = float(re.search(r"emo_mass=([0-9.]+)", receipt).group(1))
        actual = sum(engine._apply_vendor_alpha(
            sent["emo_vector"], sent["emo_alpha"]))
        assert stated == pytest.approx(round(actual, 4)), (sent, receipt)
        assert actual <= EFFECTIVE_EMOTION_MASS_CAP, sent

    # And the defect itself, on the real path: NAG's two lines, one seed.
    assert wire[0]["seed"] == wire[1]["seed"]
    assert wire[1]["seed"] != wire[2]["seed"]


# --------------------------------------------------------------------------- #
# THE EMOTION CEILING'S CONTROL ARM.
#
# The ceiling was unconditional in the first cut, and the final structural gate
# caught what that cost. On a NEUTRAL line -- calm=1.0, sum exactly 1.0, the
# shape of the very beat the operator reported -- alpha 1.0 is rescaled to 0.4
# by the cap and alpha 0.4 lands on 0.4 by the vendor's own scaling. Identical
# effective mass. The 2x2's alpha axis was DEGENERATE on the defect's own case,
# no arm reproduced pre-fix emotion behaviour, and the release gate's evidence
# could not attribute the fix between its two causes.
# --------------------------------------------------------------------------- #
def test_without_the_override_the_two_alpha_arms_are_the_SAME_arm():
    """The degeneracy itself, pinned. If someone removes the override, this
    fails and explains why it mattered."""
    engine = get_engine("indextts2")
    neutral = _vector(calm=1.0)

    hot = engine.emotion_payload(neutral, alpha=1.0, mass_cap=0.4)
    soft = engine.emotion_payload(neutral, alpha=0.4, mass_cap=0.4)

    assert hot["effective_mass"] == soft["effective_mass"] == pytest.approx(0.4)


def test_the_override_restores_pre_fix_emotion_behaviour():
    """The control arm has to be the REAL old path: the whole vector spent, and
    nothing of the speaker's own embedding left. That is the defect."""
    from nodes._otr_audio_engines.eng_indextts2 import EMOTION_MASS_CAP_DISABLED

    payload = get_engine("indextts2").emotion_payload(
        _vector(calm=1.0), alpha=1.0, mass_cap=EMOTION_MASS_CAP_DISABLED)

    assert payload["effective_mass"] == pytest.approx(1.0)
    assert payload["mass_capped"] is False
    assert 1.0 - payload["effective_mass"] == pytest.approx(0.0)


def test_the_ceiling_still_binds_when_nobody_sets_the_env(monkeypatch):
    """The override is opt-in. The SHIPPED default must still be the ceiling,
    or the fix silently stops being applied."""
    monkeypatch.delenv("OTR_INDEXTTS2_EMO_MASS_CAP", raising=False)
    engine = get_engine("indextts2")

    assert engine.current_emo_mass_cap() == EFFECTIVE_EMOTION_MASS_CAP
    payload = engine.emotion_payload({e: 1.0 for e in EMOTIONS}, alpha=1.0)
    assert payload["mass_capped"] is True
    assert payload["effective_mass"] <= EFFECTIVE_EMOTION_MASS_CAP


def test_the_ceiling_knob_keys(monkeypatch):
    """A KNOB THAT MOVES THE RENDER MUST MOVE THE KEY (Lemmy chunk A1), and this
    one moves the render more audibly than alpha does."""
    engine = get_engine("indextts2")

    monkeypatch.setenv("OTR_INDEXTTS2_EMO_MASS_CAP", "0.4")
    tight = engine.render_time_params()
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_MASS_CAP", "8")
    loose = engine.render_time_params()

    assert tight["emo_mass_cap"] == 0.4 and loose["emo_mass_cap"] == 8.0
    keyed_tight = build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="2",
        char_id="c03", line_id="b003", prepared_text="Evening.", params=tight)
    keyed_loose = build_resolved_request(
        role="char_voice", engine_name="indextts2",
        engine_profile_id="char_indextts2_v1", engine_impl_version="2",
        char_id="c03", line_id="b003", prepared_text="Evening.", params=loose)

    assert keyed_tight.cache_key != keyed_loose.cache_key


@pytest.mark.parametrize("raw,expected", [
    ("junk", EFFECTIVE_EMOTION_MASS_CAP),
    ("", EFFECTIVE_EMOTION_MASS_CAP),
    ("nan", EFFECTIVE_EMOTION_MASS_CAP),
    ("-5", 0.0),
    ("999", 8.0),
])
def test_a_malformed_ceiling_falls_back_to_the_shipped_one(raw, expected, monkeypatch):
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_MASS_CAP", raw)
    assert get_engine("indextts2").current_emo_mass_cap() == expected


# --------------------------------------------------------------------------- #
# A WITHDRAWN CLAIM MUST NOT COME BACK THROUGH AN OLD LEDGER.
#
# `select_policy_route` only runs at CastLock time, so a ledger frozen while the
# route was still qualified carries the route dict on its own cast row. The
# render path used to trust it outright -- so re-rendering that ledger under
# changed engine code would stamp the old qualification_record_id into per-line
# receipts describing audio the audition never heard.
# --------------------------------------------------------------------------- #
def _routed_cast_row(fingerprint):
    return {
        "char_id": "c02", "name": "LEMMY",
        "voice_ref_id": "idx_lemmy_algenib_cockney_v1",
        "voice_route": {
            "route_id": "lemmy-indextts2-algenib-cockney-v1",
            "route_contract_version": 1,
            "status": "qualified",
            "engine": "indextts2",
            "voice_ref_id": "idx_lemmy_algenib_cockney_v1",
            "reference_kind": "local_wav",
            "ref_path": "models/TTS/refs/indextts2/lemmy_algenib_cockney_v1.wav",
            "source_ref_sha256": "47e733d51ea58773142f934f3484cf3633cada5f"
                                 "e603b672cdfc47c712a60db2",
            "qualification_record_id": "g1-test-a-2026-08-10",
            "runtime": {"model_id": "IndexTTS-2",
                        "engine_impl_version": fingerprint,
                        "weight_revision": "6238972345f704ef"},
        },
    }


def test_an_old_locked_ledger_stops_re_asserting_the_withdrawn_claim():
    """It DEGRADES to an ordinary bank reference -- it does not raise, and the
    row still renders. THE LAW."""
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    # verify_bytes=False + an injected bank: the point here is the STALENESS
    # gate, and it deliberately sits AFTER every structural proof, so the row
    # has to get past those first.
    resolved = ROUTE.resolve_and_verify_reference(
        _routed_cast_row("b965453f355661a3"), "indextts2", verify_bytes=False,
        bank_lookup=lambda vid: {"engine": "indextts2", "voice_ref_id": vid})

    assert resolved.is_policy_route is False
    assert resolved.route_id == ""
    assert resolved.qualification_record_id == "", (
        "a stale route still handed its record id to the per-line receipt")


def test_a_current_ledger_route_is_still_honoured(tmp_path, monkeypatch):
    """The demotion must be about STALENESS, not a blanket refusal to trust a
    stamped route -- a re-qualified episode has to keep working."""
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    row = _routed_cast_row(ROUTE.live_engine_impl_version("indextts2"))
    resolved = ROUTE.resolve_and_verify_reference(
        row, "indextts2", verify_bytes=False,
        bank_lookup=lambda vid: {"engine": "indextts2", "voice_ref_id": vid})

    assert resolved.is_policy_route is True
    assert resolved.route_id == "lemmy-indextts2-algenib-cockney-v1"
    assert resolved.qualification_record_id == "g1-test-a-2026-08-10"
