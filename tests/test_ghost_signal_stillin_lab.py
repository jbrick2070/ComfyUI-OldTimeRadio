"""``animatediff15_v3_stillin_lab_video`` -- the haunted v3 lane started from a plate.

Executes the graph with recorded fake node classes (the haunted file's method:
a source grep cannot tell a conditional branch that runs from one that does
not), and pins the contract of docs/2026-09-02-animatediff-ledger-experiments/
still-in-peer/driver_anchor.md sections 10 + 12: eleven render-time instances on
the parent's seven classes, the plate sampled on the plain checkpoint MODEL, the
video sampler fed the repeated plate at the resolved denoise, the plate PNG on
disk and its sha OUTSIDE the causal hash, the strict denoise dial, the plate
prompt composer over every registered pack, the derived bundle, the whole-plan
replay override and the verifier's plate rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import pytest
import torch

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import eng_ghost_signal_stillin_lab as lab
from nodes._otr_video_engines import ghost_plate_prompt as gpp
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.registry import EngineUnusable

LAB = "animatediff15_v3_stillin_lab_video"
HAUNTED = "animatediff15_v3_haunted_video"
PLATE_PROMPT = "anime style, expressive linework, painterly anime background set, a harbour town"


class _Patcher:
    def __init__(self, tag):
        self.tag = tag
        self.detached = False

    def detach(self, unpatch_all=True):
        self.detached = True
        return self


class _Recorder:
    """Fake node classes recording every call; the sampler returns a real LATENT."""

    def __init__(self, source_request=16):
        self.calls = []
        self.source_request = source_request
        self.base_model = _Patcher("base")
        self.lora_model = _Patcher("lora")
        self.ade_model = _Patcher("ade")
        self.clip = object()
        self.vae = object()

    def classes(self):
        rec = self

        def _node(tag, result):
            class _N:
                FUNCTION = "go"

                def go(self, **kw):
                    rec.calls.append((tag, dict(kw)))
                    return result(kw) if callable(result) else result
            return _N

        def _sample(kw):
            n = int(kw["latent_image"]["samples"].shape[0])
            return ({"samples": torch.zeros((n, 4, 36, 64))},)

        def _decode(kw):
            n = int(kw["samples"]["samples"].shape[0])
            frames = np.zeros((n, 288, 512, 3), dtype=np.uint8)
            for i in range(n):
                frames[i, 0, 0, 0] = i % 256
            return (frames,)

        return {
            "checkpoint": _node("ckpt", (rec.base_model, rec.clip, rec.vae)),
            "text_encode": _node("text_encode", (("cond",),)),
            "context": _node("context", ("CONTEXT_OPTS",)),
            "lora": _node("lora", (rec.lora_model,)),
            "ade": _node("ade", (rec.ade_model,)),
            "latent": _node("latent", lambda kw: ({"samples": torch.zeros(
                (int(kw["batch_size"]), 4, 36, 64))},)),
            "sampler": _node("sampler", _sample),
            "decode": _node("decode", _decode),
        }


def _request(target=32, seed=4242, shot_id="shot_b001", plate_prompt=PLATE_PROMPT,
             plate_path=""):
    return {
        "shot_id": shot_id,
        "request_id": shot_id,
        "text_prompt": "a tall stooped figure, mid-shot or wider, turns",
        "negative_prompt": "text, watermark, caption, lettering, subtitles",
        "timing": {"target_frame_count": target},
        "seed_bundle": {"request_seed": seed},
        "plate_prompt": plate_prompt,
        "plate_path": plate_path,
    }


def _engine():
    return vreg.get_engine(LAB)


def _render(monkeypatch, tmp_path, target=32, **req_kw):
    eng = _engine()
    rec = _Recorder(source_request=gs.ghost_source_request(target))
    monkeypatch.setattr(eng, "_classes", rec.classes())
    monkeypatch.setattr(eng, "_loaded", True)
    monkeypatch.setattr(eng, "_patchers", [rec.base_model])
    prepared = {"engine_id": eng.name, "lease": None,
                "patchers": eng._patchers, "session_ctx": {},
                "base_model": (rec.base_model,), "clip": (rec.clip,),
                "vae": (rec.vae,), "recipe": eng._recipe_receipt()}
    monkeypatch.setattr(
        wb, "encode_frames_to_silent_mp4",
        lambda frames, out_path, fps, **kw: (
            out_path, int(np.asarray(frames).shape[0])))
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    req = _request(target=target, plate_path=str(tmp_path / "ghost_plates"), **req_kw)
    raw = eng.render_clip(req, prepared)
    return rec, eng, raw, req


# --------------------------------------------------------------------------- #
# THE GRAPH, PROVEN BY EXECUTION
# --------------------------------------------------------------------------- #

def test_the_lab_lane_is_registered_as_a_haunted_sibling():
    eng = _engine()
    parent = vreg.get_engine(HAUNTED)
    assert isinstance(eng, type(parent))
    assert eng.name == LAB and eng.recipe_receipt_id != parent.recipe_receipt_id
    assert eng.wants_plate_prompt is True
    for attr in ("family", "roles", "prompt_profile", "frame_contract",
                 "accepts_still", "required_inputs", "still_plan",
                 "subject_ownership", "lora_name", "render_canvas"):
        assert getattr(eng, attr) == getattr(parent, attr), attr
    assert sorted(eng._node_candidates()) == sorted(parent._node_candidates())
    # THE G2.2 CANVAS PIN: the lane declares (512, 288) and this file names it.
    assert eng.render_canvas == (512, 288)
    assert eng.target_fps == 25 and eng.frame_contract.max_frames == 0


def test_eleven_render_time_instances_on_the_parents_classes(monkeypatch, tmp_path):
    rec, _eng, _raw, _req = _render(monkeypatch, tmp_path)
    tags = [name for name, _ in rec.calls]
    assert len(tags) == 11, tags
    assert tags.count("text_encode") == 3
    assert tags.count("latent") == 1          # the PLATE's; the video sampler has none
    assert tags.count("sampler") == 2
    assert tags.count("decode") == 2
    assert tags.count("lora") == 1 and tags.count("ade") == 1 and tags.count("context") == 1


def test_the_plate_samples_on_the_base_model_and_the_video_samples_the_repeated_plate(
        monkeypatch, tmp_path):
    rec, eng, _raw, _req = _render(monkeypatch, tmp_path, target=32)
    samplers = [kw for name, kw in rec.calls if name == "sampler"]
    plate, video = samplers
    assert plate["model"] is rec.base_model
    assert plate["denoise"] == 1.0
    assert int(plate["latent_image"]["samples"].shape[0]) == 1
    assert video["model"] is rec.ade_model
    assert video["denoise"] == pytest.approx(lab.STILLIN_LAB_DENOISE_DEFAULT)
    assert int(video["latent_image"]["samples"].shape[0]) == gs.ghost_source_request(32)
    # the plate decode saw the batch-1 latent, the beat decode the full batch
    decodes = [kw for name, kw in rec.calls if name == "decode"]
    assert int(decodes[0]["samples"]["samples"].shape[0]) == gs.ghost_source_request(32)
    assert int(decodes[1]["samples"]["samples"].shape[0]) == 1
    assert rec.base_model.detached and rec.ade_model.detached and rec.lora_model.detached


def test_the_plate_png_is_written_and_its_sha_recorded_outside_the_hash(monkeypatch, tmp_path):
    _rec, _eng, raw, _req = _render(monkeypatch, tmp_path)
    path = pathlib.Path(raw["plate_path"])
    assert path.is_file() and path.parent == tmp_path / "ghost_plates"
    assert raw["plate_name"] == path.name and path.name.endswith(".png")
    assert raw["plate_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert raw["plate_source"] == "minted"
    assert len(raw["plate_identity_sha256"]) == 64
    execs = [r["node_id"] for r in raw["graph_exec"]]
    assert execs == [lab.NODE_PLATE_SAMPLER, lab.NODE_PLATE_DECODE]
    assert raw["recipe"] == _engine().recipe_receipt_id


def test_a_blank_plate_prompt_is_refused_by_name(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="plate_prompt"):
        _render(monkeypatch, tmp_path, plate_prompt="")


def test_the_denoise_dial_is_strict(monkeypatch):
    eng = _engine()
    monkeypatch.delenv(lab.STILLIN_LAB_DENOISE_ENV, raising=False)
    assert eng.resolve_denoise() == pytest.approx(lab.STILLIN_LAB_DENOISE_DEFAULT)
    monkeypatch.setenv(lab.STILLIN_LAB_DENOISE_ENV, "0.35")
    assert eng.resolve_denoise() == pytest.approx(0.35)
    for bad in ("abc", "1.5", "-0.1", "nan", "inf"):
        monkeypatch.setenv(lab.STILLIN_LAB_DENOISE_ENV, bad)
        with pytest.raises(EngineUnusable):
            eng.resolve_denoise()


def test_the_video_sampler_uses_the_resolved_denoise(monkeypatch, tmp_path):
    monkeypatch.setenv(lab.STILLIN_LAB_DENOISE_ENV, "0.5")
    rec, _eng, raw, _req = _render(monkeypatch, tmp_path)
    video = [kw for name, kw in rec.calls if name == "sampler"][1]
    assert video["denoise"] == pytest.approx(0.5)
    assert raw["plate_denoise"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# RECEIPTS AND IDENTITY
# --------------------------------------------------------------------------- #

def test_sampler_inputs_carry_plate_inputs_and_never_an_output_hash(monkeypatch):
    monkeypatch.delenv(lab.STILLIN_LAB_DENOISE_ENV, raising=False)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    si = _engine().sampler_inputs_for(_request())
    assert si["latent"] == "ghost_plate_init" and si["init_image"] is None
    assert si["denoise"] == pytest.approx(lab.STILLIN_LAB_DENOISE_DEFAULT)
    assert si["plate_prompt"] == PLATE_PROMPT and si["plate_seed"] == 4242
    assert si["plate_adapter_strength"] == 0.0
    assert si["init_repeat_method"] == "torch_repeat"
    assert si["init_repeat_count"] == gs.ghost_source_request(32)
    assert not [k for k in si if "sha" in k.lower()], "an output hash in the causal set"
    assert si["adapter"] == "v3_sd15_adapter.ckpt"


def test_shot_cache_identity_moves_with_plate_inputs_and_denoise(monkeypatch):
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.delenv(lab.STILLIN_LAB_DENOISE_ENV, raising=False)
    eng = _engine()
    base = eng.shot_cache_identity(_request())
    assert eng.shot_cache_identity(_request()) == base
    assert eng.shot_cache_identity(_request(plate_prompt=PLATE_PROMPT + ", dusk")) != base
    monkeypatch.setenv(lab.STILLIN_LAB_DENOISE_ENV, "0.8")
    assert eng.shot_cache_identity(_request()) != base
    parent_len = len(vreg.get_engine(HAUNTED).shot_cache_identity(_request()))
    assert len(base) == parent_len + 2


def test_the_receipt_hashes_plate_inputs_and_projects_the_output_beside_the_hash(monkeypatch):
    monkeypatch.delenv(lab.STILLIN_LAB_DENOISE_ENV, raising=False)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    eng = _engine()
    shot = {"shot_id": "shot_b001", "role": "character_video", "engine_id": LAB,
            "render_request_hash": "abc"}

    def clip(sha):
        return {"clip_id": "shot_b001", "path": "x.mp4", "frame_count": 32,
                "recipe": eng.recipe_receipt_id, "engine_id": LAB,
                "family": eng.family,
                "qc": {"plate_sha256": sha, "plate_name": "p.png",
                       "plate_source": "minted", "plate_identity_sha256": "i" * 64}}

    r1 = rd.build_actual_receipt(eng, shot, _request(), clip("a" * 64))
    r2 = rd.build_actual_receipt(eng, shot, _request(), clip("b" * 64))
    r3 = rd.build_actual_receipt(eng, shot, _request(plate_prompt=PLATE_PROMPT + ", dusk"),
                                 clip("a" * 64))
    assert r1["plate_sha256"] == "a" * 64 and r1["plate_source"] == "minted"
    assert r1["actual_request_sha"] == r2["actual_request_sha"], "an OUTPUT moved the hash"
    assert r1["actual_request_sha"] != r3["actual_request_sha"], "a plate INPUT did not move it"
    assert "plate_sha256" not in rd._RECEIPT_CAUSAL_KEYS


def test_canonicalize_carries_the_plate_record_in_qc(monkeypatch, tmp_path):
    _rec, eng, raw, req = _render(monkeypatch, tmp_path)
    probed = []
    monkeypatch.setattr(lab, "ffprobe_clip_fields",
                        lambda path: probed.append(path) or {
                            "width": 512, "height": 288, "fps": 25,
                            "frame_count": 32, "has_audio": False})
    monkeypatch.setattr(lab, "validate_silent_clip_contract", lambda fields, fps: None)
    clip = eng.canonicalize(raw, req, {})
    assert probed == [raw["out_path"]], "one probe, on the emitted clip"
    assert clip["engine_id"] == LAB and clip["recipe"] == eng.recipe_receipt_id
    assert clip["qc"]["plate_sha256"] == raw["plate_sha256"]
    assert clip["qc"]["plate_source"] == "minted"
    assert [r["node_id"] for r in clip["qc"]["graph_exec"]] == [
        lab.NODE_PLATE_SAMPLER, lab.NODE_PLATE_DECODE]


# --------------------------------------------------------------------------- #
# THE REPEAT RULE
# --------------------------------------------------------------------------- #

def test_repeat_latent_follows_the_live_repeat_latent_batch_rule():
    latent = {"samples": torch.arange(8.0).reshape(1, 2, 2, 2),
              "noise_mask": torch.ones((1, 1, 2, 2)),
              "batch_index": [0]}
    out = lab.repeat_latent(latent, 3)
    assert tuple(out["samples"].shape) == (3, 2, 2, 2)
    assert torch.equal(out["samples"][2], latent["samples"][0])
    assert "noise_mask" in out                      # a batch-1 mask is left alone
    assert out["batch_index"] == [0, 1, 2]
    assert latent["samples"].shape[0] == 1          # pure: the original is untouched
    with pytest.raises(RuntimeError, match="batch 1"):
        lab.repeat_latent({"samples": torch.zeros((2, 4, 1, 1))}, 2)
    with pytest.raises(RuntimeError):
        lab.repeat_latent({"samples": torch.zeros((1, 4, 1, 1))}, 0)
    with pytest.raises(RuntimeError, match="LATENT"):
        lab.repeat_latent("SAMPLED", 2)


# --------------------------------------------------------------------------- #
# THE PLATE PROMPT COMPOSER, OVER EVERY REGISTERED PACK
# --------------------------------------------------------------------------- #

_CAMERA_WORDS = ("camera", "lens", "zoom", "dolly", "tilt")
_META = {"story_brief_terms": {"setting": ["a rain-soaked harbour town", "1938"],
                               "palette": ["rust", "sodium orange"],
                               "lighting": ["low tungsten"], "atmosphere": ["fog"]}}


def _words_measure(text):
    n = len(text.split()) + 2
    return n, 1 if n <= 77 else 2


@pytest.mark.parametrize("style_id", __import__(
    "nodes._otr_visual_styles", fromlist=["list_style_ids"]).list_style_ids())
def test_the_plate_prompt_leads_with_the_packs_full_positive_tail(style_id):
    from nodes import _otr_visual_styles as vs
    st = vs.resolve_visual_style(style_id)
    out = gpp.compose_plate_prompt(st, dict(_META, visual_style=style_id),
                                   token_measure_fn=_words_measure)
    positive = out["positive"]
    assert out["head"] == st.positive_tail.strip().rstrip(",.").strip()
    assert positive.startswith(out["head"])
    assert positive != vs.compact_style_cue(st), "the two-word cue is the defect"
    assert out["clip_windows"] == 1 and out["clip_tokens"] <= 77
    assert len(positive) <= 320
    low = " %s " % positive.lower().replace(",", " ")
    # The pack's OWN language is carried as authored (sci_fi_radio's grade tail
    # says "anamorphic lens"); what may never ride the plate is a MOTION
    # register (camera moves, some with damping words), a subject, or lettering.
    for reg in (getattr(st, "motion_registers", None) or {}).values():
        for clause in ([reg] if isinstance(reg, str) else list(reg or [])):
            clause = str(clause).strip().lower()
            assert clause and clause not in low, (style_id, clause)
    for word in ("subtitle", "caption", "watermark", "lettering"):
        assert word not in low, (style_id, word)
    assert st.positive_tail.strip().rstrip(",.").strip() in positive
    assert out["sha8"] == hashlib.sha256(positive.encode("utf-8")).hexdigest()[:8]


def test_the_composer_drops_last_first_and_never_the_positive_tail():
    from nodes import _otr_visual_styles as vs
    st = vs.resolve_visual_style("storybook_engraving")

    def tight(text):     # every prompt over ~30 words spills to a second window
        n = len(text.split()) + 2
        return n, 1 if n <= 32 else 2

    out = gpp.compose_plate_prompt(st, dict(_META, visual_style="storybook_engraving"),
                                   token_measure_fn=tight)
    # dropped LAST FIRST: an in-order subsequence of the drop order (a clause the
    # dedupe already emptied is simply skipped), and the positive_tail survives
    assert out["dropped"]
    order = list(gpp.PLATE_DROP_ORDER)
    positions = [order.index(k) for k in out["dropped"]]
    assert positions == sorted(positions)
    assert "plate_look" not in out["dropped"] or out["dropped"][-1] == "plate_look"
    assert out["positive"].startswith(st.positive_tail.strip().rstrip(",.").strip())

    from nodes._otr_video_engines.ghost_signal_author import GhostBudgetError

    def impossible(text):
        return 90, 2
    with pytest.raises(GhostBudgetError):
        gpp.compose_plate_prompt(st, _META, token_measure_fn=impossible)


# --------------------------------------------------------------------------- #
# THE DERIVED BUNDLE, THE IMPORT STAMP AND THE WHOLE-PLAN OVERRIDE
# --------------------------------------------------------------------------- #

def _bundle(tmp_path, name="ep_src"):
    from nodes import production_ledger as PL
    b = tmp_path / name
    (b / "audio").mkdir(parents=True)
    ledger = {"episode_id": name, "lines": [{"line_id": "l1"}],
              "meta": {"freeze_timestamp": "t"},
              "video": {"shots": [{"shot_id": "s1", "engine_id": HAUNTED}]}}
    lp = b / "audio" / ("%s_ledger.json" % name)
    lp.write_text(json.dumps(ledger), encoding="utf-8")
    mp = b / "audio" / ("%s_master.wav" % name)
    mp.write_bytes(b"RIFF0000WAVE")
    files = []
    for rel, p in (("audio/%s_ledger.json" % name, lp), ("audio/%s_master.wav" % name, mp)):
        files.append({"path": rel, "bytes": p.stat().st_size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {"schema_version": PL.REPLAY_MANIFEST_SCHEMA, "source_episode_id": name,
                "source_episode_root": str(tmp_path / "episodes" / name), "source_commit": "abc",
                "ledger": files[0]["path"], "master_audio": files[1]["path"],
                "planned_shots": 1, "files": files}
    (b / PL.REPLAY_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return b


def test_derive_engine_bundle_is_bundle_to_bundle_and_immutable(tmp_path):
    from nodes import production_ledger as PL
    import scripts.otr_freeze_replay_bundle as fz
    src = _bundle(tmp_path)
    derived = fz.derive_engine_bundle(src, LAB)
    assert derived == tmp_path / ("ep_src__engine_%s" % LAB)
    man = PL.load_replay_manifest(str(derived))
    assert man["engine_override"] == LAB and man["derived_from"] == str(src.resolve())
    assert [f["path"] for f in man["files"]] == [f["path"] for f in
                                                 PL.load_replay_manifest(str(src))["files"]]
    assert not (src / PL.REPLAY_MANIFEST_NAME).read_text(encoding="utf-8").count("engine_override")
    with pytest.raises(SystemExit, match="already exists"):
        fz.derive_engine_bundle(src, LAB)
    with pytest.raises(SystemExit, match="ORIGINAL"):
        fz.derive_engine_bundle(derived, HAUNTED)


def test_apply_replay_engine_override_rewrites_the_whole_plan_atomically():
    from nodes.otr_shot_lock import _apply_replay_engine_override
    planned = {
        "roles": {"character_video": HAUNTED, "announcer_visual": HAUNTED},
        "roles_effective": {"character_video": HAUNTED, "announcer_visual": HAUNTED},
        "execution_groups": [{"engine_id": HAUNTED, "role": "character_video"}],
        "shots": [{"shot_id": "s1", "engine_id": HAUNTED, "family": "text_to_video"},
                  {"shot_id": "s2", "engine_id": HAUNTED, "family": "text_to_video"}],
    }
    _apply_replay_engine_override(planned, LAB, {})
    assert set(planned["roles_effective"].values()) == {LAB}
    assert set(planned["roles"].values()) == {LAB}
    assert planned["execution_groups"][0]["engine_id"] == LAB
    assert all(s["engine_id"] == LAB and s["family"] == "text_to_video" for s in planned["shots"])
    # the reverse direction (the baseline of every A/B) works the same way
    _apply_replay_engine_override(planned, HAUNTED, {})
    assert all(s["engine_id"] == HAUNTED for s in planned["shots"])
    # a non-sibling is refused by name, and nothing moved
    with pytest.raises(ValueError, match="not a sibling"):
        _apply_replay_engine_override(planned, "ltx_8gb", {})
    assert all(s["engine_id"] == HAUNTED for s in planned["shots"])
    with pytest.raises(ValueError, match="not a registered"):
        _apply_replay_engine_override(planned, "no_such_engine", {})
    # a plan without its effective route is refused whole, never rewritten in part
    no_route = {"roles": {"character_video": HAUNTED}, "execution_groups": [],
                "shots": [{"shot_id": "s1", "engine_id": HAUNTED, "family": "text_to_video"}]}
    with pytest.raises(ValueError, match="roles_effective"):
        _apply_replay_engine_override(no_route, LAB, {})
    assert no_route["shots"][0]["engine_id"] == HAUNTED


def test_import_replay_bundle_stamps_the_override_raw(tmp_path, monkeypatch):
    """Through the replay suite's own episode builder and freezer, so the
    imported ledger is a REAL one (lines, audio, stills) and the only new thing
    under test is the raw override stamp."""
    import importlib.util
    from nodes import production_ledger as PL
    import scripts.otr_freeze_replay_bundle as fz
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    # every ledger the test creates lands under tmp (the replay suite's redirect)
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    ep, _ledger = tcr._make_episode(tmp_path / "episodes")
    src = tcr._freeze(ep, tmp_path / "bundles")
    derived = fz.derive_engine_bundle(src, LAB)
    led = PL.import_replay_bundle(str(derived))
    meta = led.data["meta"]
    assert meta["replay_engine_override"] == LAB
    assert meta["replay_derived_from"] == str(pathlib.Path(src).resolve())
    assert meta["replay_of_episode"] == ep.name
    # an ORIGINAL bundle stamps an empty override -- the pure A/A, byte for byte
    led2 = PL.import_replay_bundle(str(src))
    assert led2.data["meta"]["replay_engine_override"] == ""


# --------------------------------------------------------------------------- #
# THE VERIFIER'S PLATE RULE
# --------------------------------------------------------------------------- #

def _ledger_with_trace(tmp_path, name, plate, *, seed=7, sha="deadbeef" * 8):
    from scripts import otr_verify_replay as vr
    row = {"shot_id": "s1", "segment_index": 0, "seed": seed, "text_prompt": "x",
           "negative_prompt": "y", "engine_id": LAB}
    for k in vr._RECEIPT_CAUSAL_KEYS:
        row.setdefault(k, None)
    row["actual_request_sha"] = vr.recompute_sha(row)
    if plate:
        row["plate_sha256"] = plate
    led = {"episode_id": name,
           "meta": {"replay_of_episode": "src", "replay_workspace_id": "w" + name,
                    "freeze_timestamp": "t", "render_trace": [row]},
           "video": {"shots": [{"shot_id": "s1", "render_request_hash": "h"}]},
           "audio": {"master_audio_sha256": "m"}}
    p = tmp_path / ("%s.json" % name)
    p.write_text(json.dumps(led), encoding="utf-8")
    return p


def test_the_verifier_requires_equal_plate_hashes_across_the_aa(tmp_path, capsys):
    from scripts import otr_verify_replay as vr
    src = tmp_path / "src.json"
    src.write_text(json.dumps({"episode_id": "src", "meta": {"freeze_timestamp": "t"},
                               "video": {"shots": [{"shot_id": "s1", "render_request_hash": "h"}]},
                               "audio": {"master_audio_sha256": "m"}}), encoding="utf-8")
    a = _ledger_with_trace(tmp_path, "r1", "a" * 64)
    b = _ledger_with_trace(tmp_path, "r2", "a" * 64)
    assert vr.main([str(src), str(a), str(b)]) == 0
    c = _ledger_with_trace(tmp_path, "r3", "b" * 64)
    assert vr.main([str(src), str(a), str(c)]) == 1
    out = capsys.readouterr().out
    assert "plate hashes present and equal" in out
