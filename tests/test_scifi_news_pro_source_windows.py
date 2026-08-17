"""Complete-source coverage contracts for the scifi_news_pro lane."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_scifi_news_pro as F2  # noqa: E402
from nodes import production_ledger as PL  # noqa: E402


PACK_PATH = (
    REPO_ROOT
    / "nodes"
    / "story_packs"
    / "scifi_news_pro"
    / "scifi_news_pro.json"
)


@pytest.fixture(autouse=True)
def _preserve_current_ledger():
    saved = PL._CURRENT
    yield
    PL._CURRENT = saved


def _pack() -> SimpleNamespace:
    raw = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    return SimpleNamespace(
        story_model_id=raw["story_model_id"],
        prompt_stages=raw["prompt_stages"],
    )


def _payload(*, body: str) -> dict[str, str]:
    return {
        "headline": "A complete source reaches the radio desk",
        "summary": "A test wire item with exact source coverage.",
        "full_text": body,
        "source": "Test Science Wire",
        "date": "2026-07-30",
        "link": "https://example.invalid/complete-source",
        "seed_text": "A complete source reaches the radio desk",
    }


def _long_body() -> str:
    return (
        "HEAD_SENTINEL "
        + ("alpha beta gamma " * 530)
        + "TailEntity 909 TAIL_SENTINEL"
    )


def _dossier(
    fact: str,
    *,
    number: str = "",
    thing: str = "",
) -> F2.DossierLLM:
    return F2.DossierLLM(
        facts_to_keep=[fact],
        allowed_numbers=[number] if number else [],
        named_entities=F2.NamedEntities(
            things=[thing] if thing else [],
        ),
        dramatizable_vectors=[f"dramatize {fact}"],
    )


def _treatment() -> F2.Treatment:
    return F2.Treatment.model_validate({
        "title": "The Complete Signal",
        "dramatic_question": "Will Sela hear the final fragment?",
        "setting": "a mountain relay station at night",
        "cast_shapes": [{
            "name": "SELA",
            "role": "relay keeper",
            "want": "to hear the complete transmission",
            "pressure": "the reserve battery is failing",
            "register": "plain, quick answers",
        }],
        "turn": "The final fragment changes the signal's meaning.",
        "priced_ending": {
            "choice": "Sela sends the complete record onward.",
            "cost_paid": "She spends the relay's last reserve charge.",
        },
        "news_thread": "The source article was read in full.",
        "news_close_read": (
            "The complete source described TailEntity and the number 909."
        ),
    })


def _valid_markup() -> str:
    return "\n".join([
        "TITLE: The Complete Signal",
        "MUSIC: low glass tones",
        "ANNOUNCER: A relay waits above the sleeping valley.",
        "SCENE 1: the mountain relay room",
        "SELA: The final fragment is here.",
        "ANNOUNCER: The complete signal travels before dawn.",
        "CODA: The factual source beneath this fiction is:",
        "MUSIC: one low sustained note",
        "END.",
    ])


def _coverage_for(
    payload: dict[str, str],
    windows: tuple[F2._DossierWindow, ...],
) -> dict:
    dossiers = tuple(
        _dossier(f"window fact {window.index}") for window in windows
    )
    merged, counts = F2._merge_dossiers(dossiers)
    allowance = F2._DIGEST_CHAR_CAP - len(F2._digest_header(payload))
    overlap = min(
        F2._DOSSIER_WINDOW_OVERLAP_CHARS,
        max(0, allowance - 1),
    )
    body = payload["full_text"]
    return F2._source_coverage_receipt(
        payload,
        windows,
        overlap_chars=overlap,
        attempts_by_window=[1] * len(windows),
        local_dossiers=dossiers,
        merged_dossier=merged,
        merge_counts=counts,
        news_seed_receipt={
            "body_chars": len(body),
            "body_bytes_utf8": len(body.encode("utf-8")),
            "body_sha256": hashlib.sha256(
                body.encode("utf-8")
            ).hexdigest(),
            "body_source": "rss_full",
            "rss_content_index": 2,
            "rss_content_count": 4,
        },
    )


def test_digest_windows_cover_complete_source():
    payload = _payload(body=_long_body() + " Ω")
    body = payload["full_text"]
    windows = F2._build_digest_windows(payload)

    assert len(windows) > 2
    assert windows[0].body_start == 0
    assert windows[-1].body_end == len(body)
    for window in windows:
        assert len(window.digest) <= F2._DIGEST_CHAR_CAP
        assert window.digest == (
            F2._digest_header(payload)
            + body[window.body_start:window.body_end]
        )
    for previous, current in zip(windows, windows[1:]):
        assert previous.body_end - current.body_start == 239
        assert current.body_start <= previous.body_end

    rebuilt = body[
        windows[0].body_start:windows[0].body_end
    ]
    for previous, current in zip(windows, windows[1:]):
        rebuilt += body[previous.body_end:current.body_end]
    assert rebuilt == body

    receipt = _coverage_for(payload, windows)
    assert receipt["coverage_complete"] is True
    assert receipt["body_chars"] == len(body)
    assert receipt["body_bytes_utf8"] == len(body.encode("utf-8"))
    assert receipt["body_sha256"] == hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()
    assert receipt["news_seed_receipt_match"] is True
    assert receipt["body_source"] == "rss_full"
    assert receipt["rss_content_index"] == 2
    assert receipt["rss_content_count"] == 4


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("body_source", "not_a_route"),
        ("rss_content_index", 4),
        ("rss_content_count", -1),
    ],
)
def test_news_seed_route_coordinates_must_match_selected_article(
    field,
    wrong_value,
):
    payload = _payload(body=_long_body())
    windows = F2._build_digest_windows(payload)
    dossiers = tuple(
        _dossier(f"window fact {window.index}") for window in windows
    )
    merged, counts = F2._merge_dossiers(dossiers)
    allowance = F2._DIGEST_CHAR_CAP - len(F2._digest_header(payload))
    overlap = min(
        F2._DOSSIER_WINDOW_OVERLAP_CHARS,
        max(0, allowance - 1),
    )
    body = payload["full_text"]
    news_seed = {
        "body_chars": len(body),
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "body_source": "rss_full",
        "rss_content_index": 2,
        "rss_content_count": 4,
    }
    news_seed[field] = wrong_value

    with pytest.raises(F2.NewsProDossierError, match="meta.news_seed"):
        F2._source_coverage_receipt(
            payload,
            windows,
            overlap_chars=overlap,
            attempts_by_window=[1] * len(windows),
            local_dossiers=dossiers,
            merged_dossier=merged,
            merge_counts=counts,
            news_seed_receipt=news_seed,
        )


def test_short_source_digest_is_byte_identical_to_former_input():
    payload = _payload(body="Short complete source.")
    expected = (
        F2._digest_header(payload) + payload["full_text"]
    )[:F2._DIGEST_CHAR_CAP]

    windows = F2._build_digest_windows(payload)

    assert len(windows) == 1
    assert windows[0].digest == expected
    assert F2._build_source_preview(payload) == expected


def test_short_source_keeps_long_summary_when_whole_input_fits():
    payload = _payload(body="A short article body.")
    payload["summary"] = "s" * 800
    former_input = F2._raw_digest_header(payload) + payload["full_text"]
    assert len(former_input) < F2._DIGEST_CHAR_CAP

    windows = F2._build_digest_windows(payload)
    receipt = _coverage_for(payload, windows)

    assert len(windows) == 1
    assert windows[0].digest == former_input
    assert F2._build_source_preview(payload) == former_input
    assert receipt["framing_truncated"] is False


def test_oversized_summary_is_bounded_without_losing_article_body():
    payload = _payload(body=_long_body())
    payload["summary"] = "long summary framing " * 500
    assert len(F2._raw_digest_header(payload)) > F2._DIGEST_CHAR_CAP

    windows = F2._build_digest_windows(payload)
    receipt = _coverage_for(payload, windows)

    assert windows[0].body_start == 0
    assert windows[-1].body_end == len(payload["full_text"])
    assert all(
        len(window.digest) <= F2._DIGEST_CHAR_CAP for window in windows
    )
    assert receipt["coverage_complete"] is True
    assert receipt["framing_truncated"] is True
    assert receipt["framing_used_chars"] < receipt[
        "framing_original_chars"
    ]
    assert "TAIL_SENTINEL" in windows[-1].digest


def test_merge_is_bounded_balanced_deduplicated_and_keeps_tail():
    dossiers = tuple(
        F2.DossierLLM(
            facts_to_keep=[f"fact-{index}", "shared overlap fact"],
            allowed_numbers=[str(index)],
            named_entities=F2.NamedEntities(
                people=[f"Person {index}"],
                places=[f"Place {index}"],
                things=[f"Thing {index}"],
            ),
            dramatizable_vectors=[f"vector-{index}"],
        )
        for index in range(25)
    )

    merged, counts = F2._merge_dossiers(dossiers)
    repeated, repeated_counts = F2._merge_dossiers(dossiers)

    assert merged == repeated
    assert counts == repeated_counts
    assert len(merged.facts_to_keep) == F2._DOSSIER_FACTS_MAX
    assert merged.facts_to_keep[0] == "fact-0"
    assert "fact-24" in merged.facts_to_keep
    assert counts["facts_to_keep"] == {
        "seen": 50,
        "unique": 26,
        "retained": F2._DOSSIER_FACTS_MAX,
    }
    assert len(merged.allowed_numbers) == F2._DOSSIER_NUMBERS_MAX
    assert "0" in merged.allowed_numbers
    assert "24" in merged.allowed_numbers
    for field_name in ("people", "places", "things"):
        values = getattr(merged.named_entities, field_name)
        assert len(values) == F2._DOSSIER_ENTITIES_PER_BUCKET_MAX
        assert values[0].endswith("0")
        assert any(value.endswith("24") for value in values)
    assert len(merged.dramatizable_vectors) == F2._DOSSIER_VECTORS_MAX
    assert "vector-24" in merged.dramatizable_vectors


def test_a_source_richer_than_the_bucket_limit_still_parses():
    """A cap that refuses a legitimate source is a defect, not a safeguard.

    Measured 2026-08-14: a UCLA Health story names 11 places and 14
    institutions. The model extracted them correctly and the pydantic cap --
    set to the same number the MERGE already trims to -- rejected the reply
    three times and killed the episode with `NewsProDossierError`. The refusal
    fired in front of the trim that exists to handle exactly this.

    The schema ceiling is now a backstop against a degenerate decode; the
    per-bucket limit is enforced where it always was, at the merge.
    """
    entities = F2.NamedEntities(
        people=[f"person {i}" for i in range(11)],
        places=[f"place {i}" for i in range(11)],
        things=[f"thing {i}" for i in range(14)],
    )
    assert len(entities.places) == 11
    assert len(entities.things) == 14

    values, _count = F2._balanced_window_values(
        [entities.things], limit=F2._DOSSIER_ENTITIES_PER_BUCKET_MAX,
    )
    assert len(values) == F2._DOSSIER_ENTITIES_PER_BUCKET_MAX

    # The backstop is still a backstop.
    with pytest.raises(Exception):
        F2.NamedEntities(
            things=[f"t{i}"
                    for i in range(F2._DOSSIER_ENTITIES_PER_BUCKET_CEILING + 1)],
        )


@pytest.mark.parametrize("field,keep_limit", [
    ("facts_to_keep", "_DOSSIER_FACTS_MAX"),
    ("allowed_numbers", "_DOSSIER_NUMBERS_MAX"),
    ("dramatizable_vectors", "_DOSSIER_VECTORS_MAX"),
])
def test_no_dossier_list_refuses_what_the_merge_would_have_trimmed(
    field, keep_limit,
):
    """THE DEFECT CLASS, swept across the whole model.

    Each of these carried a pydantic cap equal to the number the merge
    already trims them to, so a source richer than the keep limit was
    REFUSED before the trim could run. One of them killed two live episodes;
    the other three were the same failure waiting for a richer source. A
    schema ceiling may exist to bound a degenerate decode, but it must never
    sit at the same number as a downstream trim.
    """
    keep = getattr(F2, keep_limit)
    payload = {
        "facts_to_keep": ["a fact"],
        "allowed_numbers": [],
        "named_entities": {},
        "dramatizable_vectors": ["dramatize it"],
    }
    payload[field] = [f"{field} {i}" for i in range(keep + 1)]

    parsed = F2.DossierLLM.model_validate(payload)
    assert len(getattr(parsed, field)) == keep + 1

    values, _count = F2._balanced_window_values(
        [getattr(parsed, field)], limit=keep,
    )
    assert len(values) == keep


def test_blank_dossier_fact_opens_structural_repair_before_merge():
    responses = iter([
        json.dumps({
            "facts_to_keep": ["   "],
            "allowed_numbers": [],
            "named_entities": {
                "people": [],
                "places": [],
                "things": [],
            },
            "dramatizable_vectors": [],
        }),
        json.dumps({
            "facts_to_keep": ["The source contains a complete signal."],
            "allowed_numbers": [],
            "named_entities": {
                "people": [],
                "places": [],
                "things": [],
            },
            "dramatizable_vectors": [],
        }),
    ])
    calls = {"count": 0}

    def technical_fn(_messages, *, temperature, max_new_tokens):
        del temperature, max_new_tokens
        calls["count"] += 1
        return next(responses)

    dossier = F2._pass_dossier(
        technical_fn,
        _pack(),
        "A bounded complete-source window.",
    )

    assert calls["count"] == 2
    assert dossier.facts_to_keep == [
        "The source contains a complete signal."
    ]


class _StopAfterPitch(RuntimeError):
    pass


def _run_until_treatment(
    monkeypatch,
    *,
    led,
    payload: dict[str, str],
) -> tuple[list[str], list[list[dict]], F2.DossierLLM]:
    technical_digests: list[str] = []
    creative_prompts: list[list[dict]] = []
    captured: dict[str, F2.DossierLLM] = {}

    def technical_fn(messages, *, temperature, max_new_tokens):
        del temperature, max_new_tokens
        user = next(
            row["content"] for row in reversed(messages)
            if row["role"] == "user"
        )
        digest = user.split("SCIENCE STORY:\n", 1)[1]
        technical_digests.append(digest)
        is_tail = "TAIL_SENTINEL" in digest
        return json.dumps({
            "facts_to_keep": [
                "tail fact from final source window"
                if is_tail
                else f"window fact {len(technical_digests)}"
            ],
            "allowed_numbers": ["909"] if is_tail else [],
            "named_entities": {
                "people": [],
                "places": [],
                "things": ["TailEntity"] if is_tail else [],
            },
            "dramatizable_vectors": [
                "the final source window changes the fictional stakes"
                if is_tail
                else "a bounded source window shapes the fiction"
            ],
        })

    def creative_fn(messages, *, temperature, max_new_tokens):
        del temperature, max_new_tokens
        creative_prompts.append(list(messages))
        return json.dumps({
            "frame_card": "restored by validator",
            "logline": "A radio keeper follows the complete signal.",
            "hook": "The last fragment changes everything.",
            "scifi_device": "an impossible relay",
            "cast_size": 1,
            "ending_shape": "open_question",
        })

    def stop_at_treatment(
        _creative_fn,
        _pack_value,
        dossier,
        _pitch,
        _stance,
        **_kwargs,
    ):
        captured["dossier"] = dossier
        raise _StopAfterPitch

    monkeypatch.setattr(F2, "_pass_treatment", stop_at_treatment)
    meta = led.data.setdefault("meta", {})
    with pytest.raises(_StopAfterPitch):
        F2.run_scifi_news_pro_episode(
            payload=payload,
            pack=_pack(),
            resolved={
                "act_count": 3,
                "num_characters": 1,
                "creative_writing_model": "test/creative",
                "technical_model": "test/technical",
                # Pin the cameo OFF: the runner now decides it at entry, and an
                # unpinned fixture would take the ~11% OS-entropy roll and make
                # this test flaky one run in nine.
                "lemmy_force": False,
            },
            led=led,
            meta=meta,
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            slot_scheduler=None,
            source_bank_row=SimpleNamespace(
                source_bank_id="scifi_news_pro"
            ),
            episode_root=Path("."),
            episode_id="complete-source-test",
        )
    return technical_digests, creative_prompts, captured["dossier"]


def test_real_runner_invokes_every_window_and_pitch_sees_tail(monkeypatch):
    payload = _payload(body=_long_body())
    windows = F2._build_digest_windows(payload)
    meta: dict = {}
    # `save` is part of the real Ledger interface, and the pipeline now
    # persists the frame-card/stance deal before the passes that can die
    # on it. A double that omits a method the real object has is an
    # incomplete double, not a reason to make production defensive.
    led = SimpleNamespace(data={"meta": meta}, save=lambda: True)

    technical_digests, creative_prompts, dossier = _run_until_treatment(
        monkeypatch,
        led=led,
        payload=payload,
    )

    assert len(technical_digests) == len(windows)
    assert technical_digests == [window.digest for window in windows]
    assert "tail fact from final source window" in dossier.facts_to_keep
    assert "TailEntity" in dossier.named_entities.things
    assert "909" in dossier.allowed_numbers
    pitch_user = creative_prompts[0][-1]["content"]
    assert "tail fact from final source window" in pitch_user
    assert "TailEntity" in pitch_user
    assert '"909"' in pitch_user
    assert meta["scifi_news_pro"]["source_coverage"]["coverage_complete"] is True


def test_partial_window_coverage_fails_before_any_model_call(monkeypatch):
    payload = _payload(body=_long_body())
    complete = F2._build_digest_windows(payload)
    assert len(complete) > 1
    monkeypatch.setattr(
        F2,
        "_build_digest_windows",
        lambda _payload_value: complete[:-1],
    )
    model_calls = {"count": 0}

    def trap(*_args, **_kwargs):
        model_calls["count"] += 1
        raise AssertionError("coverage must fail before a model call")

    meta: dict = {}
    # `save` is part of the real Ledger interface, and the pipeline now
    # persists the frame-card/stance deal before the passes that can die
    # on it. A double that omits a method the real object has is an
    # incomplete double, not a reason to make production defensive.
    led = SimpleNamespace(data={"meta": meta}, save=lambda: True)
    with pytest.raises(F2.NewsProDossierError, match="coverage ends"):
        F2.run_scifi_news_pro_episode(
            payload=payload,
            pack=_pack(),
            resolved={
                "act_count": 3,
                "num_characters": 1,
                "creative_writing_model": "test/creative",
                "technical_model": "test/technical",
                # Pin the cameo OFF: the runner now decides it at entry, and an
                # unpinned fixture would take the ~11% OS-entropy roll and make
                # this test flaky one run in nine.
                "lemmy_force": False,
            },
            led=led,
            meta=meta,
            creative_fn=trap,
            technical_fn=trap,
            slot_scheduler=None,
            source_bank_row=SimpleNamespace(
                source_bank_id="scifi_news_pro"
            ),
            episode_root=Path("."),
            episode_id="partial-source-test",
        )
    assert model_calls["count"] == 0
    assert "source_coverage" not in meta["scifi_news_pro"]


def test_source_coverage_survives_real_ledger_save(
    tmp_path,
    monkeypatch,
):
    payload = _payload(body=_long_body() + " Ω")
    body = payload["full_text"]
    led = PL.new_ledger(
        episode_id="scifi_news_pro_source_coverage",
        out_dir=str(tmp_path / "episode"),
    )
    meta = led.data.setdefault("meta", {})
    meta["news_seed"] = {
        "body_chars": len(body),
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "body_source": "rss_full",
        "rss_content_index": 2,
        "rss_content_count": 4,
    }

    _technical, _creative, _dossier_value = _run_until_treatment(
        monkeypatch,
        led=led,
        payload=payload,
    )
    before = dict(led.data["meta"]["scifi_news_pro"]["source_coverage"])

    saved_path = Path(led.save())
    after = led.data["meta"]["scifi_news_pro"]["source_coverage"]
    reopened = json.loads(saved_path.read_text(encoding="utf-8"))
    on_disk = reopened["meta"]["scifi_news_pro"]["source_coverage"]

    assert after == before
    assert on_disk == before
    assert on_disk["news_seed_receipt_match"] is True
    assert on_disk["body_chars"] == meta["news_seed"]["body_chars"]
    assert on_disk["body_bytes_utf8"] == meta["news_seed"][
        "body_bytes_utf8"
    ]
    assert on_disk["body_sha256"] == meta["news_seed"]["body_sha256"]


def test_runner_restamps_coverage_after_assembly_rebind(
    tmp_path,
    monkeypatch,
):
    payload = _payload(body=_long_body())
    body = payload["full_text"]
    led = PL.new_ledger(
        episode_id="scifi_news_pro_assembly_rebind",
        out_dir=str(tmp_path / "assembly-rebind"),
    )
    meta = led.data.setdefault("meta", {})
    meta["news_seed"] = {
        "body_chars": len(body),
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "body_source": "rss_full",
        "rss_content_index": 2,
        "rss_content_count": 4,
    }

    dossier_calls = {"count": 0}

    def technical_fn(messages, *, temperature, max_new_tokens):
        del temperature, max_new_tokens
        dossier_calls["count"] += 1
        return json.dumps({
            "facts_to_keep": [f"fact {dossier_calls['count']}"],
            "allowed_numbers": [],
            "named_entities": {
                "people": [],
                "places": [],
                "things": [],
            },
            "dramatizable_vectors": ["a relay receives a signal"],
        })

    def creative_fn(messages, *, temperature, max_new_tokens):
        del messages, temperature, max_new_tokens
        return ""

    monkeypatch.setattr(
        F2,
        "_pass_pitch",
        lambda *_args, **_kwargs: F2.Pitch(
            frame_card="restored",
            logline="A relay keeper hears the complete signal.",
            hook="The final fragment changes the message.",
            scifi_device="an impossible relay",
            cast_size=1,
            ending_shape="open_question",
        ),
    )
    monkeypatch.setattr(
        F2,
        "_pass_treatment",
        lambda *_args, **_kwargs: _treatment(),
    )
    monkeypatch.setattr(
        F2,
        "_pass_news_read",
        lambda *_args, **_kwargs: F2.NewsCloseRead(
            news_close_read=_treatment().news_close_read
        ),
    )

    def pass_script(
        counting_fn,
        _pack_value,
        _treatment_value,
        _digest,
        _envelope,
        cast_names,
    ):
        counting_fn([], temperature=0.75, max_new_tokens=None)
        raw = _valid_markup()
        parsed, defects = F2.parse_scifi_news_pro_markup(raw, cast_names)
        assert parsed is not None, defects
        trace = F2.PassAttemptTrace(
            attempt=1,
            temperature=F2._TEMP["script"],
            outcome="accepted",
            selected=True,
            character_words=parsed.character_word_count,
            scene_character_words=F2._scene_character_word_counts(parsed),
        )
        return raw, parsed, {
            "attempt_trace": [trace],
            "defects_by_attempt": [],
            "structural_retries": 0,
        }

    monkeypatch.setattr(F2, "_pass_script", pass_script)
    monkeypatch.setattr(
        F2,
        "_apply_fable_safety_cleanup",
        lambda raw, parsed, treatment, **_kwargs: (
            raw,
            parsed,
            treatment,
            {"status": "clean", "hits": [], "patch_count": 0},
        ),
    )
    monkeypatch.setattr(
        F2,
        "_pass_casting",
        lambda *_args, **_kwargs: F2.CastingVoices(cast=[
            F2.CastVoice(
                name="SELA",
                role="relay keeper",
                character_description="A steady relay keeper.",
                gender="female",
                timbre="m01",
            )
        ]),
    )
    monkeypatch.setattr(F2, "_assign_voices", lambda *_args: [])

    def assemble_and_drop_live_receipt(
        ledger,
        *_args,
        **_kwargs,
    ):
        assert ledger.save()
        # Simulate the exact nested-meta loss the post-assembly reacquisition
        # guards. A shallow disk merge will not restore a missing nested key.
        ledger.data["meta"]["scifi_news_pro"].pop("source_coverage")

    monkeypatch.setattr(F2, "_assemble", assemble_and_drop_live_receipt)
    monkeypatch.setattr(
        F2._OTRWD,
        "stamp_actual",
        lambda *_args, **_kwargs: {"actual_words": 5},
    )
    # (The `scan_spoken_ledger` stub that used to sit here is gone: the scifi_news_pro
    # lane's terminal spoken-safety scan was removed 2026-08-05 by operator
    # directive, so there is no longer a symbol on F2 to neutralize.)

    parts = F2.run_scifi_news_pro_episode(
        payload=payload,
        pack=_pack(),
        resolved={
            "act_count": 3,
            "num_characters": 1,
            "creative_writing_model": "test/creative",
            "technical_model": "test/technical",
            # Pin the cameo OFF -- see the note above; an unpinned fixture takes
            # the ~11% roll and goes flaky.
            "lemmy_force": False,
        },
        led=led,
        meta=meta,
        creative_fn=creative_fn,
        technical_fn=technical_fn,
        slot_scheduler=None,
        source_bank_row=SimpleNamespace(
            source_bank_id="scifi_news_pro"
        ),
        episode_root=tmp_path,
        episode_id="scifi_news_pro_assembly_rebind",
    )

    receipt = parts.scifi_news_pro_meta["source_coverage"]
    assert receipt["coverage_complete"] is True
    assert receipt == led.data["meta"]["scifi_news_pro"]["source_coverage"]
    reopened = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert receipt == reopened["meta"]["scifi_news_pro"]["source_coverage"]


def test_old_prefix_counterfactual_loses_tail():
    payload = _payload(body=_long_body())
    old_prefix = (
        F2._digest_header(payload) + payload["full_text"]
    )[:F2._DIGEST_CHAR_CAP]
    windows = F2._build_digest_windows(payload)

    assert "TAIL_SENTINEL" not in old_prefix
    assert "TAIL_SENTINEL" in windows[-1].digest
