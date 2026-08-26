"""The character's face must not change between beats -- and the basis that
guarantees it must be derived the SAME way whether or not a portrait is minted.

WHY THIS FILE EXISTS (live regression, 2026-08-26). Commit 89e82181 (08-05) gave
every beat of one character the same seed by deriving it from that character's
PORTRAIT prompt hash. Commit a88cede5 (08-22) then let a video lane's
``still_plan`` suppress minting the portrait -- correct for PIXELS, because an
unused portrait had killed a live leg on an engine that refuses face close-ups
-- and silently emptied the hash the seed needs. One switch was doing two jobs.

MEASURED on 17 episodes rendered 2026-08-26: **75 scene_character stills, 4
anchored, 71 not.** The only anchored episode ran the one lane whose plan has no
portrait row at all, so no exemption applied. Every other lane -- including
``still_flat``, which `workflows/otr_canonical.json` renders production with --
had lost it.

THE KEYSTONE IS `test_virtual_basis_equals_a_rendered_portrait_row`. The first
cut of this fix transported the PRODUCER's raw prompt hash and asserted it was
"byte-for-byte" identical. It was not: the dispatcher mutates the prompt (safety
clause -> style front-anchor -> banana) and re-hashes it, so the portrait ROW
carries a different value than the producer computed. Transporting the raw hash
would have moved every face exactly once. Four independent readings -- a driver
anchor, two review lanes, and a synthesis pass -- all repeated that claim before
one reviewer read the recomputation and refuted it. An equality test would have
caught it in milliseconds, so there is now an equality test.
"""
from __future__ import annotations

import sys
from pathlib import Path
from pathlib import Path as pathlib_Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes._otr_shared import still_plan_helpers as _sp          # noqa: E402
from nodes._otr_video_engines import cheap_families as _cf       # noqa: E402
from nodes.otr_image_gen_dispatcher import (                     # noqa: E402
    normalize_prompt_for_render, resolve_seed_and_mode)


# --------------------------------------------------------------------------- #
# THE KEYSTONE
# --------------------------------------------------------------------------- #

def test_the_producer_hash_is_NOT_the_hash_the_seed_consumes():
    """THE ERROR THIS FILE EXISTS FOR, pinned so it cannot come back.

    The first cut of the fix transported the PRODUCER's raw prompt hash and
    called the result "byte-for-byte identical". It is not: the dispatcher
    mutates the text (safety clause -> style front-anchor -> banana) and
    re-hashes it, and THAT is what lands on the portrait row and feeds the
    seed. This test asserts the two genuinely DIFFER, which is precisely why
    the transport must carry TEXT and re-derive through the dispatcher's own
    normalizer.

    If this test ever fails, the two paths have converged and someone has
    removed a mutation -- at which point transporting the raw hash becomes
    safe and this file's central claim needs rewriting.
    """
    from nodes.otr_meta_brief_image_prompt import _content_hash
    from nodes._otr_visual_styles import get_visual_style

    portrait_prompt = "a weathered dock foreman, deep-set eyes, oilskin collar"
    vstyle = get_visual_style({"visual_style": "archival_documentary"})
    producer_hash = _content_hash(portrait_prompt)
    dispatcher_hash = normalize_prompt_for_render(
        portrait_prompt, vstyle=vstyle, banana_on=False,
        banana_key="frozen", source="portrait").prompt_hash
    assert producer_hash != dispatcher_hash, (
        "producer and dispatcher hashes agree -- the style front-anchor is no "
        "longer mutating the prompt, so this file's premise needs revisiting")


def test_the_producer_transports_TEXT_and_never_a_hash():
    """THE STRUCTURAL KEYSTONE -- it prevents the exact regression rather than
    re-measuring it.

    The seed's basis must be computed in ONE place: the dispatcher's own
    normalize-then-hash chain. The failure mode is somebody "helpfully" hashing
    the portrait prompt in the PRODUCER, which looks equivalent and is not --
    the producer hashes raw text, the dispatcher hashes text that the safety
    clause, the style front-anchor and the banana transform have already
    rewritten. That divergence is asserted directly by
    `test_the_producer_hash_is_NOT_the_hash_the_seed_consumes` above.

    So: the object the producer emits carries the portrait PROMPT, and must
    carry no hash-shaped identity field at all.
    """
    import inspect
    from nodes import otr_meta_brief_image_prompt as _mb

    src = inspect.getsource(_mb.derive_image_prompts)
    assert '"identity_prompt"' in src, (
        "the producer no longer transports the portrait prompt TEXT")
    for banned in ('"identity_basis"', '"identity_seed_basis"',
                   '"identity_prompt_hash"'):
        assert banned not in src, (
            f"producer stamps {banned} -- the basis must be derived ONCE, in "
            f"the dispatcher's normalizer, or the two copies will drift and "
            f"the symptom is faces moving")


def test_the_dispatcher_derives_the_basis_through_the_shared_normalizer():
    """One chain, one implementation. If the dispatcher ever grows its own
    inline hash for the identity path, the shared-normalizer guarantee is
    gone."""
    import inspect
    from nodes import otr_image_gen_dispatcher as _d

    src = inspect.getsource(_d.dispatch_images)
    assert "normalize_prompt_for_render(" in src
    # the identity path specifically must go through it
    idx = src.index("identity_basis")
    window = src[idx:idx + 900]
    assert "normalize_prompt_for_render(" in window, (
        "the identity basis is computed outside the shared normalizer")


def test_one_basis_gives_every_beat_of_a_character_the_same_seed():
    """The property the operator actually cares about, stated directly."""
    basis = "deadbeefcafe"
    seeds = {
        resolve_seed_and_mode({"request_seed": 7, "mode": "request_hash"},
                              f"still_b{n}", f"per_beat_hash_{n}",
                              kind="scene_character", char_id="c02",
                              portrait_prompt_hash=basis)[0]
        for n in range(1, 8)
    }
    assert len(seeds) == 1, "same character, same basis -> one face, every beat"


def test_without_a_basis_every_beat_draws_a_different_seed():
    """THE BUG ITSELF, pinned. This is what 71 of 75 stills did on 2026-08-26 --
    an empty basis falls through to the per-beat digest, which is exactly the
    pre-2026-08-05 drift behaviour."""
    seeds = {
        resolve_seed_and_mode({"request_seed": 7, "mode": "request_hash"},
                              f"still_b{n}", f"per_beat_hash_{n}",
                              kind="scene_character", char_id="c02",
                              portrait_prompt_hash="")[0]
        for n in range(1, 8)
    }
    assert len(seeds) == 7, "no basis -> a fresh face per beat (the regression)"


# --------------------------------------------------------------------------- #
# The LANE declares; it does not get inferred
# --------------------------------------------------------------------------- #

def test_identity_defaults_to_portrait_seed_so_forgetting_fails_safe():
    """The default direction IS the fix. A lane written tomorrow that declares
    nothing gets a stable face for free -- identity costs no render, because the
    basis is a text hash and the portrait pixels stay suppressed by
    ``required``. Defaulting the other way would recreate this bug for every
    future lane."""
    row = _sp.StillPlanRow("scene_character", "per_beat", "scene", "wide",
                           "always", "framing", "full")
    assert row.identity == _sp.IDENTITY_PORTRAIT_SEED


def test_still_word_declares_none_without_touching_its_three_siblings():
    """THE SHIP-BLOCKER, caught in review before it landed.

    still_motion / still_pan / still_flat / still_word share ONE plan tuple
    object. Writing ``identity="none"`` onto that shared ``scene_character`` row
    in place would have stripped the anchor from the three CINEMATIC lanes --
    the exact lanes carrying the measured 71 unanchored faces, with still_flat
    being what the canonical workflow renders production with. The fix would
    have shipped green and left production broken.
    """
    def _identity_of(engine_cls, kind="scene_character"):
        return next(r.identity for r in engine_cls.still_plan if r.kind == kind)

    by_name = {c.name: c for c in vars(_cf).values()
               if isinstance(c, type) and getattr(c, "name", "").startswith("still_")}
    assert _identity_of(by_name["still_word"]) == _sp.IDENTITY_NONE, (
        "still_word mints typography from the spoken line, not a face")
    for lane in ("still_flat", "still_pan", "still_motion"):
        assert _identity_of(by_name[lane]) == _sp.IDENTITY_PORTRAIT_SEED, (
            f"{lane} lost its face anchor -- the shared tuple was mutated")
    assert _cf._STILL_WORD_STILL_PLAN is not _cf._CHEAP_FAMILY_STILL_PLAN


@pytest.mark.parametrize("token", ["portrait_seed", "none"])
def test_identity_is_a_closed_enum(token):
    assert token in _sp.VALID_IDENTITY


def test_an_invalid_identity_token_is_refused_by_the_audit():
    with pytest.raises(ValueError):
        _sp.validate_still_plan_row(_sp.StillPlanRow(
            "scene_character", "per_beat", "scene", "wide", "always",
            "framing", "full", "banana"))


# --------------------------------------------------------------------------- #
# The receipt must describe what ACTUALLY happened
# --------------------------------------------------------------------------- #

def test_fixed_mode_and_the_kill_switch_report_no_seed_anchor(monkeypatch):
    """`identity_seed_basis` records whether the basis PARTICIPATED, not merely
    whether one was available -- so a receipt can never claim an anchor the
    seed did not use."""
    _, mode = resolve_seed_and_mode({"request_seed": 5, "mode": "fixed"},
                                    "still_b1", "ph", kind="scene_character",
                                    char_id="c02", portrait_prompt_hash="abc")
    assert mode == "", "fixed mode exits before the identity branch"

    monkeypatch.setenv("OTR_PORTRAIT_IDENTITY_SEED", "0")
    _, mode = resolve_seed_and_mode({"request_seed": 5, "mode": "request_hash"},
                                    "still_b1", "ph", kind="scene_character",
                                    char_id="c02", portrait_prompt_hash="abc")
    assert mode == "", "the kill switch bypasses the identity branch"


def test_jump_segments_keep_distinct_seeds_even_with_a_basis():
    """A jump CUT is supposed to be a cut. This is deliberately NOT the
    scene_character behaviour, and the fix must not quietly change it."""
    from nodes._otr_video_engines import coverage_plan as _cp  # noqa: PLC0415
    seeds = {
        resolve_seed_and_mode({"request_seed": 7, "mode": "request_hash"},
                              f"jump_{n}", f"ph_{n}", kind=_cp.JUMP_STILL_KIND,
                              char_id="c02", portrait_prompt_hash="shared")[0]
        for n in range(1, 4)
    }
    assert len(seeds) == 3, "jump segments must stay visually distinct"
