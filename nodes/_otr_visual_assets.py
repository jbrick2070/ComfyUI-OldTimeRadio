"""Pre-writer native visual-weight readiness for the shipped canonical graph.

No model imports or network at module import. Only the TWENTY-SEVEN allowlisted
files below can be fetched (three z_image_turbo, two ltx_8gb, three
stable_audio_3, one sd15, two lumina_image -- the Flux ae VAE is already the
z_image row -- seven for the native LTX 2.5 lanes, four for the
AnimateDiff lanes, whose SD 1.5 checkpoint is the sd15 row, and five for the
two MiniMax H3 lanes, which are fetched at an exact pinned revision).
Existing native loader choices are preserved, not rehash-qualified, and
readiness is NOT a claim of GPU/render compatibility. Other engines keep
their existing adapter checks with explicit uncovered logs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time

log = logging.getLogger(__name__)

#: SOURCES FETCHED AT AN EXACT PIN: ``(category, repo, filename, revision,
#: bytes, sha256)``. Every other row in ``_SOURCES`` pins whatever commit the
#: Hub serves at queue time and verifies against that commit's own LFS SHA-256;
#: a row here is requested at THIS revision and refused unless the Hub still
#: serves exactly these bytes (``_pin_metadata``). The rows are appended to
#: ``_SOURCES`` below, so this table is the only place their names live.
#: ``scripts/otr_fetch_lane_weights.py`` builds its H3 lanes from it, so the
#: dev-tree fetcher and the queue-time preflight cannot pin different bytes.
#:
#: MINIMAX H3, added 2026-09-26 (operator: "keep them as long as they are auto
#: download"; "MiniMax 3 is popular, so people have it"). The five files the two
#: H3 lanes load: FL2VA and REF2VA DiTs, the NVFP4 Qwen3-VL encoder and video
#: VAE they share, and the audio VAE only REF2VA loads. Comfy-Org/MiniMax-H3 is
#: public and ungated (anonymous 302 on this revision, measured 2026-09-25).
#: Revision, sizes and hashes are the ones the lane-19/20 receipts were
#: measured on. Each lane asks for its own files through ``_weight_rows()``, so
#: h3_low_video never pulls the REF2VA DiT or the audio VAE.
_PINNED_SOURCES = (
    ("diffusion_models", "Comfy-Org/MiniMax-H3",
     "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
     "4cc1d817b6184899b41293954329f576cb5ae86b", 20_970_379_616,
     "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"),
    ("diffusion_models", "Comfy-Org/MiniMax-H3",
     "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
     "4cc1d817b6184899b41293954329f576cb5ae86b", 20_970_379_616,
     "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"),
    ("text_encoders", "Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
     "4cc1d817b6184899b41293954329f576cb5ae86b", 15_687_142_551,
     "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"),
    ("vae", "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_video_vae_fp16.safetensors",
     "4cc1d817b6184899b41293954329f576cb5ae86b", 5_207_808_496,
     "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"),
    ("vae", "Comfy-Org/MiniMax-H3",
     "vae/minimax_h3_audio_vae_fp32.safetensors",
     "4cc1d817b6184899b41293954329f576cb5ae86b", 605_254_808,
     "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"),
)

_SOURCES = (
    ("diffusion_models", "Comfy-Org/z_image_turbo",
     "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/z_image_turbo",
     "split_files/text_encoders/qwen_3_4b.safetensors"),
    ("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
    ("checkpoints", "Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled.safetensors"),
    ("text_encoders", "comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    # MUSIC, added 2026-09-06. stable_audio_3 is ungated and commercially
    # clean, and the engine already declares ["cuda", "mps"] -- but its ONLY
    # fetcher lived in scripts/, which .comfyignore strips from the published
    # bundle. So a registry install selecting it hit EngineUnusable "fetch
    # Comfy-Org/stable-audio-3 (ungated) first" with nothing able to do the
    # fetching. That left musicgen as the only music engine that self-supplies,
    # and musicgen is CC-BY-NC -- so every published episode carried a
    # non-commercial music bed by default. Filenames verified against the live
    # Hub listing.
    #
    # TWO CHECKPOINT ROWS FOR ONE ENGINE, AND THE PREFERENCE IS THE ENGINE'S,
    # NOT THIS TABLE'S (2026-09-12). `native_requests` downloads whatever
    # name `StableAudio3Engine.resolve_ckpt()` returns; this table only says
    # what it is ALLOWED to download. With nothing on disk the engine names
    # its fetch default -- the BASE checkpoint, the only one whose cfg and
    # negative prompt are live (PBUG-20260912-03) -- and a test pins that the
    # fetch default is a key here, because until 2026-09-12 the engine fell
    # through to the post-trained name and every fresh install fetched the
    # wrong file. The post-trained row stays so an explicit OTR_SA3_CKPT pin
    # (the A/B harness's control arm) can still be fetched on a box that
    # lacks it; it is never chosen by default.
    ("checkpoints", "Comfy-Org/stable-audio-3",
     "checkpoints/stable_audio_3_small_music_base.safetensors"),  # 2,270,384,940 B
    ("checkpoints", "Comfy-Org/stable-audio-3",
     "checkpoints/stable_audio_3_small_music.safetensors"),   # 2,270,384,940 B
    ("text_encoders", "Comfy-Org/stable-audio-3",
     "text_encoders/t5gemma_b_b_ul2.safetensors"),            # 1,187,264,003 B
    # SD 1.5, added 2026-09-12, and this one file gates more lanes than any
    # other row here. `sd15` mints the still that the four `still_*` lanes and
    # `ltx098_low_video` all consume -- LTX 0.9.8 is image-to-video, so its own
    # two weights self-fetching was never enough to make that lane one click.
    # Until now the ONLY route to this checkpoint was a `hf_hub_download` line
    # the adapter prints inside its refusal, plus a manual copy into
    # models/checkpoints/; `scripts/otr_fetch_lane_weights.py` can also do it,
    # and .comfyignore strips `scripts/*` from the published bundle, so a
    # registry install had no automated route at all.
    #
    # MEASURED COST OF NOT HAVING THIS: the 4060 clean-room drill
    # (CR-20260912-04) followed the README's 8 GB row by hand, produced a valid
    # ledger, six Kokoro clips, a music master and a 78-second 1,950-frame
    # intermediate video, then died on the first shot with DEPENDENCY_MISSING
    # naming this exact file. Nothing reached otr/obs.
    #
    # Ungated and public (verified against the Hub API, gated:false), 2.0 GB.
    # The engine still decides WHICH checkpoint it loads -- a visual pack may
    # name its own, and `_resolve_ckpt_name` only returns a pack name that is
    # already installed -- so this table permits the default and never overrides
    # a choice, exactly as the stable_audio_3 note above describes.
    ("checkpoints", "Comfy-Org/stable-diffusion-v1-5-archive",
     "v1-5-pruned-emaonly-fp16.safetensors"),                 # 2,132,696,762 B
    # LUMINA-IMAGE 2.0, added 2026-09-16. The 16 GB shipping graphs stamp
    # lumina_image on all three still slots. Until this row existed, a
    # registry install that picked Lumina (or loaded a 16 GB still/video/
    # foley/mime/animatediff graph) hit EngineUnusable "set OTR_LUMINA_CKPT"
    # with no fetch path -- the same hole sd15 and stable_audio_3 had.
    # Ungated Apache-2.0 (Hub gated:false, verified 2026-09-17). The VAE is
    # Flux ae.safetensors, already allowlisted on the z_image row above;
    # do not add a second (vae, ae.safetensors) key -- MANIFEST is a dict.
    ("diffusion_models", "Comfy-Org/Lumina_Image_2.0_Repackaged",
     "split_files/diffusion_models/lumina_2_model_bf16.safetensors"),  # 5.22 GB
    ("text_encoders", "Comfy-Org/Lumina_Image_2.0_Repackaged",
     "split_files/text_encoders/gemma_2_2b_fp16.safetensors"),        # 5.23 GB
    # THE NATIVE LTX 2.5 STACK, added 2026-09-25. Operator: the point was
    # "less friction for the end user, auto download things to work". Every file below is UNGATED and loads
    # through stock ComfyUI loaders, so a lane that selects LTX 2.5 now fetches
    # its own weights at queue time exactly like Z-Image and SD 1.5 do.
    # Lightricks' own LTX-2.5 repo is gated (401 without a token,
    # PBUG-20260923-03); the VAEs and the upscaler come from a byte-identical
    # ungated mirror (same SHA-256 as Lightricks' copies) that stores them at
    # the REPO ROOT -- chosen for that: the HF cache path of the upscaler under
    # a `latent_upscale_models/` subfolder is 169 characters, past the 162 that
    # prestartup_script.py budgets for Windows MAX_PATH, and would fail to
    # materialise on a stock ComfyUI Desktop root. At the root it is 138.
    # One DiT per card class; each lane names its own through `_dit_name()`, so a 16 GB lane
    # never pulls the 24 GB weight.
    ("diffusion_models", "joeygambino/LTX-2.5-Quantized",
     "LTX25-distilled-DiT-comfy-mix4x8-13.8GB.safetensors"),  # 13,810,250,240 B
    ("diffusion_models", "joeygambino/LTX-2.5-Quantized",
     "LTX25-distilled-DiT-comfy-int8.safetensors"),           # 21,504,050,168 B
    ("diffusion_models", "joeygambino/LTX-2.5-Quantized",
     "LTX25-distilled-DiT-comfy-nvfp4.safetensors"),          # 12,499,335,336 B
    ("text_encoders", "joeygambino/LTX-2.5-Quantized",
     "gemma4-12b-ltx25-comfy-w4a8.safetensors"),              # 10,604,342,914 B
    ("vae", "yuvraj108c/LTX-2.5",
     "ltx-2.5-video-vae-bf16.safetensors"),                   #  1,472,223,346 B
    ("vae", "yuvraj108c/LTX-2.5",
     "ltx-2.5-audio-vae-bf16.safetensors"),                   #    364,866,540 B
    ("latent_upscale_models", "yuvraj108c/LTX-2.5",
     "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),  # 995,778,752 B
    # THE ANIMATEDIFF LANES, added 2026-09-25 for the same reason as every row
    # above: a new user presses Run and the weights arrive. The SD 1.5
    # checkpoint was already the sd15 row; these are the rest of what
    # `_weight_tokens()` names. Until now only `scripts/otr_fetch_lane_weights.py`
    # fetched them, and .comfyignore strips scripts/ from the published
    # bundle, so both AnimateDiff graphs refused on a fresh install. All three
    # repos are ungated (Hub API gated:false, 2026-09-25): guoyww Apache-2.0,
    # ByteDance CreativeML Open RAIL-M, stabilityai MIT. The Lightning file is
    # one of four step-count variants of identical size; the pinned commit's
    # LFS SHA-256 is taken for THIS filename, so the 8-step file is the one
    # verified. `animatediff_models` is registered by ComfyUI-AnimateDiff-
    # Evolved, which these lanes need to run at all.
    ("animatediff_models", "guoyww/animatediff", "v3_sd15_mm.ckpt"),  # 1,673,262,583 B
    ("loras", "guoyww/animatediff", "v3_sd15_adapter.ckpt"),          #   102,134,097 B
    ("animatediff_models", "ByteDance/AnimateDiff-Lightning",
     "animatediff_lightning_8step_comfyui.safetensors"),              #   908,929,664 B
    ("vae", "stabilityai/sd-vae-ft-mse-original",
     "vae-ft-mse-840000-ema-pruned.safetensors"),                     #   334,641,190 B
    # THE MINIMAX H3 LANES: the rows of `_PINNED_SOURCES` above, which carry
    # their exact revision, size and SHA-256.
) + tuple(row[:3] for row in _PINNED_SOURCES)
#: Windows MAX_PATH is 260 including the terminating NUL, so 259 is what a
#: path may actually occupy. Named here because _scrub_transfer_error reports
#: against it and prestartup_script.py decides the HF_HOME pin by it.
_WINDOWS_PATH_BUDGET = 259

#: ``(repo, filename) -> {"revision", "size", "sha256"}`` for the pinned rows.
_PINS = {(repo, filename): {"revision": revision, "size": size, "sha256": sha256}
         for _category, repo, filename, revision, size, sha256 in _PINNED_SOURCES}

#: ``(category, basename) -> spec``. A pinned row's spec also carries its
#: ``revision``, ``size`` and ``sha256``, which ``_pin_metadata`` enforces.
MANIFEST = {(category, filename.rsplit("/", 1)[-1]):
            dict({"repo_id": repo, "filename": filename},
                 **_PINS.get((repo, filename), {}))
            for category, repo, filename in _SOURCES}
_VIDEO_SLOTS = ("announcer_video_model", "music_video_model", "character_video_model")
_IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
_LTX_8GB_WEIGHT_ENGINES = frozenset({"ltx_8gb", "razzle_ltx_8gb"})
#: Every registered native LTX 2.5 lane. Each one is asked for its own file
#: names, so this set only says WHICH engines are covered, never what they load.
_LTX25_WEIGHT_ENGINES = frozenset({
    "ltx25_video",
    "ltx25_foley_16gb", "ltx25_foley_24gb",
    "ltx25_foley_blackwell",
    "ltx25_mime_16gb", "ltx25_mime_24gb",
    "ltx25_audio_in_16gb", "ltx25_audio_in_24gb",
})
#: Every registered AnimateDiff lane. They load SD 1.5 INSIDE their own graph,
#: not through an image slot (``accepts_still = False``), so each lane is asked
#: for its own files through ``_weight_tokens()``.
_ANIMATEDIFF_WEIGHT_ENGINES = frozenset({
    "animatediff15_lightning_video",
    "animatediff15_v3_haunted_video",
    "animatediff15_v3_stillin_lab_video",
})
#: Every registered MiniMax H3 lane. Each one is asked for its own rows through
#: ``_weight_rows()``, the table its assert_usable, byte floors and graph read.
_MINIMAX_H3_WEIGHT_ENGINES = frozenset({
    "minimax_h3_video",
    "minimax_h3_audio_in",
})
_COVERED = frozenset({"z_image_turbo", "stable_audio_3", "sd15", "lumina_image"}
                     | _LTX_8GB_WEIGHT_ENGINES | _LTX25_WEIGHT_ENGINES
                     | _ANIMATEDIFF_WEIGHT_ENGINES | _MINIMAX_H3_WEIGHT_ENGINES)
#: The music node is scanned alongside OTR_VideoDirector. It is a DIFFERENT
#: class with a single ``engine`` widget rather than per-role slots, so it gets
#: its own pass; an absent node is a skip, not a refusal, because a graph
#: without theme music is legitimate.
_MUSIC_NODE = "OTR_StableAudioTheme"


class VisualAssetError(RuntimeError):
    """Early readiness refusal, before the writer or any visual model load."""


def _literal(inputs, name, default=None):
    value = inputs.get(name, default)
    if not isinstance(value, str):
        raise VisualAssetError(
            "visual asset preflight cannot inspect linked/dynamic %s before execution; "
            "no asset download or model substitution was performed" % name)
    return value.strip()


#: The dropdown sentinel that means "I will name my own engine". The directors
#: resolve it through their ``custom_models_json`` widget.
_ADD_CUSTOM = "+ Add Custom"


def _custom_models(inputs):
    """The director's ``custom_models_json`` as a slot -> engine_id mapping.

    Unparseable or absent JSON is an empty mapping, not a refusal: the caller
    below still refuses a sentinel with no entry, and that refusal names the
    slot, which is a far better message than a JSON error.
    """
    import json

    raw = inputs.get("custom_models_json")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v).strip() for k, v in parsed.items()
            if isinstance(v, (str, int, float)) and str(v).strip()}


def _resolve_slot(inputs, slot, custom, kind):
    """The engine a slot selects, resolving the custom-model escape hatch.

    THE SENTINEL USED TO BE AN UNCONDITIONAL REFUSAL HERE, and that was a real
    defect: `otr_video_director` and `otr_image_director` BOTH support
    ``+ Add Custom Model`` and resolve it through their ``custom_models_json``
    widget, but this preflight rejected the run before either got the chance.
    So the documented escape hatch worked in one half of the pipeline and was
    hard-refused by the other -- an operator who declared a perfectly valid
    custom engine could not start a render at all.

    A custom engine is simply not in `_COVERED`, so it flows on to the existing
    "automatic visual-weight coverage unavailable; existing adapter checks
    remain" note. That is the correct outcome: we cannot auto-download an engine
    we do not have an allowlisted manifest row for, and we must not pretend to.
    Refusing is still right when the sentinel is chosen and NOTHING declares it
    -- that is an incomplete graph, and the message now says which slot.
    """
    picked = _literal(inputs, slot)
    if picked and not picked.startswith(_ADD_CUSTOM):
        return picked
    if picked.startswith(_ADD_CUSTOM):
        declared = custom.get(slot, "")
        if declared:
            return declared
        raise VisualAssetError(
            "visual asset preflight: %s is '+ Add Custom Model' but "
            "custom_models_json declares no '%s' entry; name the engine there "
            "or pick one from the dropdown" % (slot, slot))
    raise VisualAssetError("visual asset preflight requires an explicit %s "
                           "engine selection for %s" % (kind, slot))


def _default_role_video_slots():
    """``{role: video_slot}`` from the shared authority, or ``{}`` if it cannot
    be reached.

    THIS MODULE IS COLD-IMPORT CLEAN AND ISOLATION-TESTABLE BY CONTRACT -- its
    own docstring promises no model imports at module import, and
    ``tests/test_visual_assets_stdlib.py`` loads it with NO parent package to
    hold it to that. A hard ``from ._otr_shared.role_slots import ...`` inside
    ``plan_prompt`` broke that: run alone the file failed 19 tests, while a full
    suite run PASSED because an earlier test had already put ``_otr_shared`` in
    ``sys.modules``. An order-dependent green is worse than a red one.

    ``{}`` is the FAIL-SAFE value: no role pairs to a slot, so nothing is
    skipped and every image engine is required -- the same direction as
    :func:`_proven_no_still`. Production never takes that path;
    :func:`ensure_prompt_visual_assets` injects the real map explicitly.
    """
    try:
        from ._otr_shared.role_slots import ROLE_TO_VIDEO_SLOT
    except Exception:  # noqa: BLE001 -- isolation harness / flat import context
        try:
            from _otr_shared.role_slots import ROLE_TO_VIDEO_SLOT  # type: ignore
        except Exception:  # noqa: BLE001
            return {}
    return dict(ROLE_TO_VIDEO_SLOT)


def _image_slot_for(video_slot) -> str:
    """The image slot paired with a per-role VIDEO slot. Derived from the slot
    name rather than hand-listed, so a new role cannot pair itself wrongly."""
    return str(video_slot).replace("_video_model", "_image_model")


def _registry_consumes_still(engine_id) -> bool:
    """Does ``engine_id`` consume the role's still? THE ONE AUTHORITY, borrowed.

    Delegates to :func:`otr_image_gen_dispatcher.engine_consumes_still` -- the
    same predicate the still dispatcher itself keys on -- so the preflight's
    DEMAND and the dispatcher's MINTING are one decision, never two that can
    drift. Raises on an unknown/unregistered id; the caller fails safe.

    Lazy imports keep this module cold-import clean (no torch, no ComfyUI at
    import time), which the module docstring promises.
    """
    try:
        from .otr_image_gen_dispatcher import engine_consumes_still
        from . import _otr_video_engines  # noqa: F401 -- registers built-ins
        from ._otr_video_engines import registry as _vreg
    except ImportError:  # pragma: no cover -- flat test imports
        from otr_image_gen_dispatcher import engine_consumes_still
        import _otr_video_engines  # noqa: F401
        from _otr_video_engines import registry as _vreg
    eid = str(engine_id)
    if not _vreg.is_registered(eid):
        raise KeyError(eid)
    return bool(engine_consumes_still(_vreg.get_engine(eid)))


def _proven_no_still(engine_id, consumes_still) -> bool:
    """``True`` ONLY when the registry affirmatively proves ``engine_id`` mints
    no still, so that role's image weights are provably unused.

    **FAIL-SAFE IN EXACTLY ONE DIRECTION, deliberately.** An unknown engine, an
    unregistered id, or any registry/import failure returns ``False`` -- require
    the weights. Skipping a download that render then needs is a broken episode;
    requiring one it does not need costs only the bytes this function exists to
    stop spending. Absence of proof is never taken as proof of absence.
    """
    if not engine_id:
        return False
    probe = consumes_still if consumes_still is not None else _registry_consumes_still
    try:
        return probe(engine_id) is False
    except Exception:  # noqa: BLE001 -- any doubt at all -> require the weights
        return False


def walk_gate_consumers(prompt, unique_id):
    """Transitive ``gate_in`` walk from this validator (PBUG-20260907-02).

    Reachability starts at THIS validator and follows gate edges only, so
    another validator's subgraph stays unreachable. No source-slot constraint:
    audio hops pass the gate on output slot 2.
    """
    scoped = []
    if unique_id is None or not str(unique_id).strip():
        return scoped
    if not isinstance(prompt, dict) or not prompt:
        return scoped
    seen_ids = set()
    reachable = {str(unique_id)}
    pending = True
    while pending:
        pending = False
        for node_id, node in prompt.items():
            if not isinstance(node, dict) or node_id in seen_ids:
                continue
            gate = (node.get("inputs") or {}).get("gate_in")
            if (isinstance(gate, (list, tuple)) and len(gate) == 2
                    and str(gate[0]) in reachable):
                seen_ids.add(node_id)
                reachable.add(str(node_id))
                scoped.append(node)
                pending = True
    return scoped


def plan_prompt(prompt, unique_id, *, resolve_video, freeze_video,
                consumes_still=None, role_video_slots=None):
    """Inspect only this validator's direct gate consumers in the LIVE prompt.

    No saved-JSON reads or traversal of unrelated workflow branches. Replay
    bundles bind frozen selections, so their live widgets must not trigger
    downloads. Mixed replay/live writers behind one validator are ambiguous.
    """
    result = {"engines": set(), "skipped": [], "replay": False}
    if unique_id is None or not str(unique_id).strip():
        result["skipped"].append("legacy call without live prompt context")
        return result
    if not isinstance(prompt, dict) or not prompt:
        raise VisualAssetError("visual asset preflight requires the live queued prompt")
    scoped = walk_gate_consumers(prompt, unique_id)
    writers = [n for n in scoped if n.get("class_type") == "OTR_LedgerScriptWriter"]
    replay = [bool(_literal(n.get("inputs") or {}, "replay_from", "")) for n in writers]
    if any(replay):
        if not all(replay):
            raise VisualAssetError("visual asset preflight refuses mixed replay/live writers "
                                   "behind one validator; frozen and live selections differ")
        result["replay"] = True
        result["skipped"].append("replay uses frozen bundle assets, not live dropdowns")
        return result
    directors = [n for n in scoped if n.get("class_type") == "OTR_VideoDirector"]
    if not directors:
        result["skipped"].append("no directly gated VideoDirector; native adapter checks remain")
    for node in directors:
        inputs = node.get("inputs") or {}
        custom = _custom_models(inputs)
        videos = {}
        for slot in _VIDEO_SLOTS:
            picked = _resolve_slot(inputs, slot, custom, "video")
            videos[slot] = resolve_video(picked)
        effective = freeze_video(videos)
        result["engines"].update(resolve_video(v) for v in effective.values() if v)
        # AN IMAGE ENGINE IS ONLY REQUIRED IF ITS ROLE'S VIDEO LANE CONSUMES THE
        # STILL (PBUG-20260907-03). The four `viz_*` visualizers are procedural
        # and audio-reactive: they declare `accepts_still = False` and an
        # explicit `still_plan = ()`, the image dispatcher honours that and mints
        # nothing, and the VIDEO dropdown already tells the operator so in words
        # -- "(audio-reactive, no scene image)". Only this preflight disagreed,
        # and it demanded the full image set anyway.
        #
        # THE COST WAS NOT THEORETICAL. A 2026-09-07 4060 episode published as
        # `..._vmcp__none__...` -- image field `none`, because NOT ONE still was
        # minted -- after the preflight had downloaded 20.6 GB of z_image_turbo
        # weights to reach it. Both AMD profiles pair `viz_mxc_cpu` with
        # `z_image_turbo`, so the configuration that exists to be the LOW-friction
        # one carried the largest unused download in the pack.
        #
        # NOTHING IS HIDDEN FROM ANY DROPDOWN, and that is an operator rule, not
        # a preference: all 12 image engines stay listed and selectable, the pick
        # is still resolved here (so an empty or unresolvable slot refuses
        # exactly as loudly as before), and the ONLY thing that changes is
        # whether its weights are fetched. The skip is logged per role rather
        # than inferred silently.
        no_still = {}
        pairing = (role_video_slots if role_video_slots is not None
                   else _default_role_video_slots())
        for role, vslot in dict(pairing).items():
            lane = effective.get(role)
            lane = resolve_video(lane) if lane else ""
            if _proven_no_still(lane, consumes_still):
                no_still[_image_slot_for(vslot)] = lane
        for slot in _IMAGE_SLOTS:
            picked = _resolve_slot(inputs, slot, custom, "image")
            lane = no_still.get(slot)
            if lane:
                result["skipped"].append(
                    "%s=%s not downloaded: this role's video lane %s mints no still "
                    "(accepts_still=False), so its image weights are provably unused"
                    % (slot, picked, lane))
                continue
            result["engines"].add(picked)
    # MUSIC ENGINE, same prompt, its own node class. Read only; an absent node
    # or an unset widget is a skip so a graph without theme music still plans.
    for node in [n for n in scoped if n.get("class_type") == _MUSIC_NODE]:
        picked = _literal(node.get("inputs") or {}, "engine")
        if picked and not picked.startswith("+ Add Custom"):
            result["engines"].add(picked)
    for engine in sorted(result["engines"] - _COVERED):
        result["skipped"].append("%s: automatic visual-weight coverage unavailable; "
                                 "existing adapter checks remain" % engine)
    return result


def _native_path(folder_paths, category, token):
    path = folder_paths.get_full_path(category, token)
    if path is None:
        return None
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise VisualAssetError("native loader file is absent/empty: %s/%s; preserved, "
                               "not overwritten" % (category, token))
    return path


def _same_file(left, right):
    try:
        return left is not None and right is not None and os.path.samefile(left, right)
    except OSError:
        return False


def native_requests(engines, *, folder_paths, zimage=None, ltx=None, sa3=None,
                    sd15=None, lumina=None, ltx25=None, animatediff=None,
                    minimax_h3=None, env=None):
    """Bind the adapters' exact tokens to native folders; no writes/network.

    A missing nondefault choice is a refusal, never a default-weight fallback.
    Explicit paths must name the same file the native basename loader will use.
    """
    env = env or {}
    requests = []
    seen = set()

    def add(category, token, *, explicit="", authority=None):
        token = str(token)
        loader_token = os.path.basename(token)
        # Two lanes of one family share files (every 16 GB LTX 2.5 lane loads
        # the same DiT), and a duplicate request would download it twice.
        if (category, loader_token) in seen:
            return
        seen.add((category, loader_token))
        path = _native_path(folder_paths, category, loader_token)
        if authority is not None and not _same_file(authority, path):
            raise VisualAssetError("adapter/native loader disagree for %s/%s; "
                                   "no download or path repair" % (category, loader_token))
        if explicit and os.path.dirname(explicit) and not _same_file(explicit, path):
            raise VisualAssetError("explicit weight path disagrees with native loader for "
                                   "%s/%s; no download or path repair" % (category, loader_token))
        spec = MANIFEST.get((category, loader_token))
        if path is None:
            if token != loader_token or spec is None:
                raise VisualAssetError("selected weight %s/%s is missing and has no "
                                       "allowlisted download; no substitution" % (category, token))
            try:
                roots = folder_paths.get_folder_paths(category)
            except KeyError:
                # `animatediff_models` exists only once the AnimateDiff-Evolved
                # pack has registered it.
                roots = None
            if not roots:
                raise VisualAssetError("no native model folder registered for " + category)
            # Do not hide a stale symlink or directory by downloading elsewhere.
            for root in roots:
                if os.path.lexists(Path(root) / loader_token):
                    raise VisualAssetError("unusable existing native entry for %s/%s; "
                                           "preserved, no path repair" % (category, loader_token))
        requests.append({"category": category, "token": loader_token,
                         "spec": dict(spec) if spec else None, "path": path})

    if "z_image_turbo" in engines:
        if zimage is None:
            raise VisualAssetError("Z-Image adapter resolution is unavailable")
        token, verified = zimage._resolve_unet_name()
        if verified and _native_path(folder_paths, "diffusion_models", token) is None:
            # The older adapter strips nested filenames to basenames. Never
            # turn its existing-but-unloadable selection into a new download.
            raise VisualAssetError("Z-Image adapter reports an installed selection that the "
                                   "native loader cannot resolve; no replacement/path repair")
        add("diffusion_models", token, explicit=str(env.get(zimage.MODEL_ENV) or ""))
        for category, key, default in (
            ("text_encoders", zimage.CLIP_ENV, zimage._DEFAULT_CLIP),
            ("vae", zimage.VAE_ENV, zimage._DEFAULT_VAE),
        ):
            explicit = str(env.get(key) or "")
            add(category, os.path.basename(explicit or default), explicit=explicit)
    if engines & _LTX_8GB_WEIGHT_ENGINES:
        if ltx is None:
            raise VisualAssetError("LTX098 adapter resolution is unavailable")
        add("checkpoints", ltx._ckpt_name(), authority=ltx._ckpt_path())
        add("text_encoders", ltx._t5_name(), authority=ltx._t5_path())
    if "stable_audio_3" in engines:
        # PBUG-20260907-02, third layer. The MANIFEST rows and `_COVERED` for
        # stable_audio_3 landed on 2026-09-06, but NOTHING TURNED THE ENGINE
        # INTO A REQUEST -- this function only ever had branches for the image
        # and video engines. So once the planner could finally see the music
        # engine it logged
        #     READY engines=ltx_8gb,stable_audio_3,z_image_turbo files=5
        # -- the engine named, and still only the five visual files requested.
        # Being in `_COVERED` even suppressed the "coverage unavailable" note
        # that would otherwise have said so out loud.
        #
        # `resolve_ckpt()` / `_TENC` already honour OTR_SA3_CKPT /
        # OTR_SA3_TEXT_ENCODER, so an operator pin is passed through as
        # `explicit` exactly as the Z-Image branch does with its own env keys.
        if sa3 is None:
            raise VisualAssetError("stable_audio_3 adapter resolution is unavailable")
        # ASK THE ADAPTER, DO NOT READ ITS CONSTANT (2026-09-12). `_CKPT` used
        # to BE the filename; it is now the operator's override and is EMPTY by
        # default, because the engine picks between the base and post-trained
        # checkpoints at load time. Reading the raw constant here asked the
        # preflight to find a weight called "" and killed every render that did
        # not set OTR_SA3_CKPT -- found by the first canonical leg after the
        # change, which is exactly what legs are for.
        add("checkpoints", sa3.StableAudio3Engine.resolve_ckpt()[0],
            explicit=str((env or {}).get("OTR_SA3_CKPT") or ""))
        add("text_encoders", sa3._TENC,
            explicit=str((env or {}).get("OTR_SA3_TEXT_ENCODER") or ""))
    if "sd15" in engines:
        # ASK THE ADAPTER, for the same reason the stable_audio_3 branch above
        # does: `_resolve_ckpt_name()` is the ONE resolver both `assert_usable`
        # and the render path already share, so the preflight can never fetch a
        # different file from the one that will load.
        #
        # IT IS CALLED WITH NO STYLE ON PURPOSE. With a style it would return a
        # visual pack's own checkpoint -- but `_style_ckpt_name` only ever
        # returns a name that is ALREADY INSTALLED, so a pack can never become a
        # download here, and the file this fetches is the one the pack path
        # falls back to anyway. Same shape as the note in the _SOURCES table:
        # this permits the default and never overrides an operator's choice.
        if sd15 is None:
            raise VisualAssetError("sd15 adapter resolution is unavailable")
        add("checkpoints", sd15._resolve_ckpt_name(),
            explicit=str((env or {}).get(sd15.CKPT_ENV) or ""))
    if "lumina_image" in engines:
        # ASK THE ADAPTER, same as z_image / sd15: resolve_ckpt_name is the
        # one answer assert_usable and the loader share. CLIP + VAE follow
        # the env-or-default basename rule; the Flux ae token is already in
        # MANIFEST from the z_image row.
        if lumina is None:
            raise VisualAssetError("lumina_image adapter resolution is unavailable")
        token, verified = lumina.resolve_ckpt_name()
        if verified and _native_path(folder_paths, "diffusion_models", token) is None:
            raise VisualAssetError("Lumina adapter reports an installed selection that the "
                                   "native loader cannot resolve; no replacement/path repair")
        add("diffusion_models", token, explicit=str(env.get(lumina.MODEL_ENV) or ""))
        for category, key, default in (
            ("text_encoders", lumina.CLIP_ENV, lumina._DEFAULT_CLIP),
            ("vae", lumina.VAE_ENV, lumina._DEFAULT_VAE),
        ):
            explicit = str(env.get(key) or "")
            add(category, os.path.basename(explicit or default), explicit=explicit)
    selected_animatediff = sorted(engines & _ANIMATEDIFF_WEIGHT_ENGINES)
    if selected_animatediff:
        # ASK EACH LANE: `_weight_tokens()` names the files its own
        # assert_usable checks. When an sd15 image slot is also selected it
        # names the same checkpoint, and `add` requests it once.
        if not animatediff:
            raise VisualAssetError("AnimateDiff adapter resolution is unavailable")
        for eid in selected_animatediff:
            lane = animatediff.get(eid)
            if lane is None:
                raise VisualAssetError("AnimateDiff adapter for %s is unavailable" % eid)
            for category, token in lane._weight_tokens():
                add(category, token)
    selected_ltx25 = sorted(engines & _LTX25_WEIGHT_ENGINES)
    if selected_ltx25:
        # ASK EACH LANE, same rule as every branch above: `_dit_name()` and its
        # siblings are what `_weight_paths()` -- the lane's own assert_usable
        # list -- resolves, so the preflight fetches exactly the files the lane
        # will open, operator env overrides included. The upscaler is requested
        # only by a lane that builds its loader, for the reason `_weight_paths`
        # gives.
        if not ltx25:
            raise VisualAssetError("LTX 2.5 adapter resolution is unavailable")
        for eid in selected_ltx25:
            lane = ltx25.get(eid)
            if lane is None:
                raise VisualAssetError("LTX 2.5 adapter for %s is unavailable" % eid)
            add("diffusion_models", lane._dit_name(),
                explicit=str(env.get("OTR_LTX25_NATIVE_DIT") or ""))
            add("text_encoders", lane._text_encoder_name(),
                explicit=str(env.get("OTR_LTX25_NATIVE_TE") or ""))
            add("vae", lane._video_vae_name(),
                explicit=str(env.get("OTR_LTX25_VIDEO_VAE") or ""))
            add("vae", lane._audio_vae_name(),
                explicit=str(env.get("OTR_LTX25_AUDIO_VAE") or ""))
            if lane._ingraph_upscale:
                add("latent_upscale_models", lane._upscaler_name(),
                    explicit=str(env.get("OTR_LTX25_UPSCALER") or ""))
    selected_h3 = sorted(engines & _MINIMAX_H3_WEIGHT_ENGINES)
    if selected_h3:
        # ASK EACH LANE: `_weight_rows()` is the one table the lane's own
        # assert_usable, byte floors, session identity and graph read, and
        # `_token_for` applies the same OTR_MINIMAX_H3_*_NAME override the
        # loader node is handed. A row's FIRST category is the folder its
        # loader reads (UNETLoader diffusion_models, CLIPLoader text_encoders,
        # VAELoader vae). Both lanes load the encoder and the video VAE, and
        # `add` requests each once.
        if not minimax_h3:
            raise VisualAssetError("MiniMax H3 adapter resolution is unavailable")
        for eid in selected_h3:
            lane = minimax_h3.get(eid)
            if lane is None:
                raise VisualAssetError("MiniMax H3 adapter for %s is unavailable" % eid)
            for label, categories, default, _floor in lane._weight_rows():
                add(categories[0], lane._token_for(label, default),
                    explicit=str(env.get("OTR_MINIMAX_H3_%s_NAME" % label) or ""))
    return requests


def _pin_metadata(spec, *, hf_hub_url, get_hf_file_metadata):
    """Pin public source HEAD, then verify metadata again at that commit.

    The server's 64-hex LFS etag is the expected content SHA-256. A git blob
    etag, missing size, gated source or changing metadata is a hard refusal.
    No credential is requested/read. The GET uses the pinned Hub URL so a CDN
    URL obtained before other large transfers cannot expire in our queue.

    A spec from ``_PINNED_SOURCES`` carries its own ``revision``, ``size`` and
    ``sha256``: HEAD is never asked, one metadata call is made at that
    revision, and anything but the recorded commit, size and LFS SHA-256 is a
    refusal -- never a fall back to another revision.
    """
    if spec not in MANIFEST.values():
        raise VisualAssetError("visual weight source is not allowlisted")
    revision = spec.get("revision")
    if revision:
        pinned_url = hf_hub_url(spec["repo_id"], spec["filename"], revision=revision,
                                endpoint="https://huggingface.co")
        pinned = get_hf_file_metadata(pinned_url, token=False, timeout=30)
        size = getattr(pinned, "size", None)
        if (str(getattr(pinned, "commit_hash", None) or "").lower() != revision.lower()
                or str(getattr(pinned, "etag", None) or "").lower() != spec["sha256"].lower()
                or type(size) is not int or size != spec["size"]):
            raise VisualAssetError(
                "visual weight source no longer serves the pinned revision %s of %s/%s "
                "(expected %d bytes, sha256 %s); no fallback to another revision"
                % (revision, spec["repo_id"], spec["filename"], spec["size"],
                   spec["sha256"]))
        return {"commit": revision.lower(), "sha256": spec["sha256"].lower(),
                "size": spec["size"], "url": pinned_url}
    url = hf_hub_url(spec["repo_id"], spec["filename"], endpoint="https://huggingface.co")
    first = get_hf_file_metadata(url, token=False, timeout=30)
    commit = first.commit_hash
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise VisualAssetError("visual weight source did not provide a pinned commit")
    pinned_url = hf_hub_url(spec["repo_id"], spec["filename"], revision=commit,
                           endpoint="https://huggingface.co")
    pinned = get_hf_file_metadata(pinned_url, token=False, timeout=30)
    if (pinned.commit_hash != commit or pinned.etag != first.etag or pinned.size != first.size
            or not isinstance(pinned.etag, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", pinned.etag)
            or type(pinned.size) is not int or pinned.size <= 0):
        raise VisualAssetError("visual weight metadata lacks stable exact size/content SHA-256")
    return {"commit": commit, "sha256": pinned.etag, "size": pinned.size, "url": pinned_url}


def _hf_fetch(spec, metadata, progress=None):
    """Fetch one allowlisted file through huggingface_hub; return its local path.

    REPLACES A HAND-ROLLED urllib DOWNLOADER (2026-09-07), and the reason is not
    tidiness. Diffing four published zips showed the Comfy Registry scanner
    Flags precisely the versions that ship a bespoke fetcher: alpha.25 added
    this module and was Flagged where alpha.24 was Active; alpha.23 REMOVED an
    indextts2 weight-downloader and a PowerShell installer and went Active where
    alpha.22 was Flagged. Nine of fourteen versions are Flagged, and a Flagged
    version never resolves as `latest_version`, so Manager's default install
    button does not offer it -- which is the single largest piece of friction in
    the whole zero-friction campaign. The LLM lane moves just as many bytes and
    has never been a differing file, because it goes through this same library.

    IT ALSO FIXES THREE REAL DEFECTS the loop could not:
      * RESUME and RETRY. The planner used to print "no resume/retry" about
        itself; a drop 11 GB into a 12 GB file restarted that file at zero, and
        a real 50-second stall was measured mid-way through a 36.8 GB fetch.
      * THE OPERATOR'S TOKEN. `_pin_metadata` deliberately passes `token=False`
        for METADATA (pinning must not depend on a credential), which is why
        startup logs a resolved HF_TOKEN and the next line still warns
        "unauthenticated". The TRANSFER may legitimately use it, so a token is
        passed when one exists and omitted when it does not -- an anonymous
        install keeps working unchanged.
      * HTTPS and redirect handling are the library's, not a hand-written
        `HTTPRedirectHandler` subclass that had to be reasoned about here.

    The revision is PINNED to the commit `_pin_metadata` already verified, so
    this cannot follow a branch that moved. Verification is unchanged and still
    ours: `fetch_verified` hashes the returned bytes against the pinned sha256
    before publishing, so a poisoned or stale cache entry is still refused.
    """
    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": spec["repo_id"],
        "filename": spec["filename"],
        "revision": metadata["commit"],
    }
    token = _resolve_transfer_token()
    if token:
        kwargs["token"] = token
    if progress is not None:
        kwargs["tqdm_class"] = _progress_tqdm(progress, metadata["size"])
    try:
        return hf_hub_download(**kwargs)
    except VisualAssetError:
        raise
    except Exception as exc:  # noqa: BLE001 -- never leak a signed CDN URL
        raise VisualAssetError(
            "visual weight transfer failed (%s) for %s/%s: %s"
            % (type(exc).__name__, spec["repo_id"], spec["filename"],
               _scrub_transfer_error(exc))) from None


#: Parameter and header names that carry a credential. Matched case-insensitively
#: against a token's KEY, which is what makes the redaction robust: every bypass
#: found in review had a recognisable key and an unrecognisable shape.
_CREDENTIAL_KEYS = (
    "signature", "sig", "token", "secret", "credential", "password", "passwd",
    "apikey", "api_key", "api-key", "auth", "access_key", "accesskey",
    "sessionid", "session_id", "x-amz", "bearer", "hf_token",
)


def _redact_credentials(message):
    r"""Every credential-bearing run replaced, whatever shape it arrived in.

    WHY THIS IS KEY-BASED AND NOT URL-BASED. The first cut redacted only
    ``scheme://<non-whitespace>`` runs, and an adversarial review broke it five
    ways out of six -- every one of which a real Hugging Face error can produce:

      * ``cdn-lfs.hf.co/f?X-Amz-Signature=...``  -- a bare host, no scheme
      * ``Authorization: Bearer ...``            -- a header echoed into the text
      * ``https%3A%2F%2F...%3FX-Amz-Signature%3D...`` -- percent-encoded
      * a URL split across a newline, signature on the second line
      * ``hf_token=...``                         -- not in a URL at all

    A shape-based rule has to anticipate the shape. A KEY-based rule only has to
    recognise the name, and the names are few and stable. So each whitespace
    token is percent-decoded and checked for a credential key or a scheme; a hit
    redacts the WHOLE token rather than trying to excise part of it, because a
    partial redaction of a credential is not a redaction.

    Header forms are handled first because they span two tokens
    (``Authorization:`` then the value), which a per-token rule cannot see.
    """
    import re
    import urllib.parse

    # Two-token header forms, before tokenising.
    message = re.sub(r"(?i)\bauthorization\s*:\s*\S+", "<credential redacted>",
                     message)
    message = re.sub(r"(?i)\bbearer\s+\S+", "<credential redacted>", message)

    out = []
    for token in re.split(r"(\s+)", message):
        if not token.strip():
            out.append(token)
            continue
        try:
            probe = urllib.parse.unquote(token)
        except Exception:                    # noqa: BLE001 -- treat as raw
            probe = token
        low = probe.lower()
        if "://" in probe or any(k in low for k in _CREDENTIAL_KEYS):
            out.append("<credential redacted>")
        else:
            out.append(token)
    return "".join(out)


def _scrub_transfer_error(exc):
    r"""The exception's own message, credentials removed, plus path lengths.

    WHY THIS EXISTS. ``_hf_fetch`` used to report only ``type(exc).__name__``, so
    a Windows MAX_PATH failure arrived as a bare "visual weight transfer failed
    (FileNotFoundError)" with no path and no cause. That cost a cross-machine
    investigation to identify as a 261-character path, and it would cost it again
    -- because Windows does not say "too long" for this. The underlying
    ``os.rename`` raises WinError 3, "cannot find the path specified", while the
    5.22 GB blob it was moving sits happily on disk at 220 characters.

    WHY IT SCRUBS RATHER THAN OMITS. The caller's ``from None`` exists so a signed
    CDN URL can never reach a log, and that property is preserved -- see
    ``_redact_credentials``, which was rewritten after a review broke the first
    attempt five ways. A path is not a credential, and a path is what a reader
    needs.

    WHY IT APPENDS LENGTHS. The one fact that diagnoses this failure class is
    invisible in the text -- a 261-character path and a 238-character one look
    identical in a log line. So each path-shaped token is annotated with its own
    length, and anything past the Windows budget is named as such.

    IT MAY NOT RAISE, and the fallback is written so that it cannot either: an
    earlier version called ``exc.__class__.__name__`` unguarded, which an
    exception overriding attribute access can defeat. This runs inside an
    exception handler, where a formatter that throws replaces a useful error with
    a confusing one.
    """
    try:
        import re
        try:
            raw = str(exc)
        except Exception:                    # noqa: BLE001 -- __str__ may raise
            raw = ""
        message = _redact_credentials(raw or type(exc).__name__)

        notes = []
        # Paths may contain spaces, so a quoted run is taken whole first and the
        # bare form only picks up what is left.
        candidates = re.findall(r"'((?:[A-Za-z]:\\|/)[^']{8,})'", message)
        candidates += re.findall(r"(?:[A-Za-z]:\\|/)[^\s'\"]{8,}", message)
        seen = set()
        for candidate in candidates:
            trimmed = candidate.rstrip(".,;:)").replace("\\\\", "\\")
            if trimmed in seen:
                continue
            seen.add(trimmed)
            note = "%d chars" % len(trimmed)
            if len(trimmed) > _WINDOWS_PATH_BUDGET:
                note += " -- OVER the %d-char Windows path budget" % (
                    _WINDOWS_PATH_BUDGET,)
            notes.append("%s (%s)" % (trimmed, note))
        if notes:
            message += " [paths: %s]" % "; ".join(notes)
        return message
    except Exception:                        # noqa: BLE001 -- formatter only
        try:
            return type(exc).__name__
        except Exception:                    # noqa: BLE001 -- nothing is safe
            return "unprintable exception"
def _resolve_transfer_token():
    """The operator's HF token if one is set, else None. Never raises."""
    try:
        from ._otr_hf_auth import resolve_hf_token_runtime
    except ImportError:  # pragma: no cover -- flat (sys.path) import
        try:
            from _otr_hf_auth import resolve_hf_token_runtime  # type: ignore
        except ImportError:
            return None
    try:
        return resolve_hf_token_runtime()
    except Exception:  # noqa: BLE001 -- an anonymous install must still work
        return None


def _progress_tqdm(progress, total_bytes):
    """A real tqdm subclass that forwards byte counts to ``progress``.

    Subclassed rather than imitated, for the reason PBUG-20260906-08 cost a
    whole run: huggingface_hub treats `tqdm_class` as the tqdm PROTOCOL -- it
    reads and WRITES `.total` and calls `.refresh()` -- so a hand-rolled
    look-alike raises AttributeError mid-transfer. Inheriting gives the entire
    surface for free and cannot drift when the library touches a new attribute.
    """
    import io

    from tqdm.std import tqdm as _tqdm_base

    class _ProgressTqdm(_tqdm_base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs.setdefault("file", io.StringIO())  # keep the console clean
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            # MUST NOT RAISE: a progress bar may never fail a 12 GB transfer,
            # and tqdm's refresh() leaks its class-level lock on any exception.
            try:
                progress(int(self.n or 0), int(self.total or total_bytes))
            except BaseException:  # noqa: BLE001
                pass
            return True

    return _ProgressTqdm


def _load_adapters(engines):
    """The adapter objects :func:`native_requests` asks, loaded for exactly
    ``engines`` -- no adapter module is imported for a lane nobody selected."""
    adapters = dict.fromkeys(
        ("zimage", "ltx", "sa3", "sd15", "lumina", "ltx25", "animatediff",
         "minimax_h3"))
    if "z_image_turbo" in engines:
        from ._otr_image_engines import z_image_turbo
        adapters["zimage"] = z_image_turbo
    if engines & _LTX_8GB_WEIGHT_ENGINES:
        from ._otr_video_engines.eng_ltx_8gb import Ltx8gbEngine
        adapters["ltx"] = Ltx8gbEngine()
    if "stable_audio_3" in engines:
        from ._otr_audio_engines import eng_stable_audio_3
        adapters["sa3"] = eng_stable_audio_3
    if "sd15" in engines:
        from ._otr_image_engines import sd15
        adapters["sd15"] = sd15
    if "lumina_image" in engines:
        from ._otr_image_engines import lumina_image
        adapters["lumina"] = lumina_image
    for key, family in (("animatediff", _ANIMATEDIFF_WEIGHT_ENGINES),
                        ("ltx25", _LTX25_WEIGHT_ENGINES),
                        ("minimax_h3", _MINIMAX_H3_WEIGHT_ENGINES)):
        if engines & family:
            from . import _otr_video_engines  # noqa: F401 -- registers built-ins
            from ._otr_video_engines import registry as _vreg
            lanes = {}
            for eid in engines & family:
                engine = _vreg.get_engine(eid)
                lanes[eid] = engine() if isinstance(engine, type) else engine
            adapters[key] = lanes
    return adapters


class _NothingInstalled:
    """A ``folder_paths`` stand-in for planning: no weight is installed, and
    every category has one native root that does not exist."""

    _ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "__otr_nothing_installed__")

    @staticmethod
    def get_full_path(category, token):
        return None

    @classmethod
    def get_folder_paths(cls, category):
        return [cls._ROOT]


def planned_downloads(engines, env=None):
    """``{(category, basename)}`` the queue-time preflight would fetch for
    ``engines`` on a box that holds none of them.

    The same adapters and the same :func:`native_requests` the validator runs,
    so the answer cannot drift from the download it describes; no disk is
    read and nothing is fetched. An engine outside ``_COVERED`` contributes
    nothing, and a selection the preflight would refuse raises
    :class:`VisualAssetError` exactly as it would at queue time.
    """
    engines = {str(e) for e in engines if e} & _COVERED
    if not engines:
        return set()
    adapters = _load_adapters(engines)
    if adapters["ltx"] is not None:
        adapters["ltx"] = _NothingInstalledLtx(adapters["ltx"])
    requests = native_requests(engines, folder_paths=_NothingInstalled,
                               env=dict(env or {}), **adapters)
    return {(r["category"], r["token"]) for r in requests}


class _NothingInstalledLtx:
    """The LTX 0.9.8 adapter's names, with its on-disk lookups answering
    "not installed" to match :class:`_NothingInstalled` -- its path methods
    read the real models root, which a planning call must not touch."""

    def __init__(self, inner):
        self._inner = inner

    def _ckpt_name(self):
        return self._inner._ckpt_name()

    def _t5_name(self):
        return self._inner._t5_name()

    @staticmethod
    def _ckpt_path():
        return None

    @staticmethod
    def _t5_path():
        return None


def _refuse_missing_node_packs(engines):
    """Refuse at QUEUE time a video engine whose ComfyUI node classes are not
    registered -- before any weight download, writer pass or render.

    PBUG-20260925-02: a fresh Manager install queued otr_8gb_animatediff on a
    box without ComfyUI-AnimateDiff-Evolved. The gate fetched 3.9 GB of
    weights, the writer wrote three acts, the base video encoded 4,664
    frames, and the run failed 18 minutes in on ADE_AnimateDiffLoaderGen1 --
    the engine's own ``assert_usable`` knew, but only the render driver ever
    asked it. The same table it reads (``_node_candidates``) is read here, at
    the one point a refusal is still free.

    NODE CLASSES ONLY, on purpose. Each engine's ``assert_usable`` also checks
    its weight files, and this gate is about to fetch exactly those, so calling
    it here would refuse every fresh box for weights it was about to download.
    Engines without a node table (the procedural visualizers) pass through; an
    unregistered id is the registry's refusal, not this one's. The message
    leads with what to do; the class names come after.
    """
    # GUARDED, like every other import in this gate: the runtime-bridge tests
    # fake the package tree and stub only the modules this function needs, so
    # a bare import here is the isolation break the docstring below warns
    # about. No engine registry means nothing to check -- the render driver's
    # own `assert_usable` still stands behind this.
    try:
        try:
            from ._otr_video_engines import registry as _vreg
            from ._otr_video_engines import wrapper_bridge as _wb
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_video_engines import registry as _vreg  # type: ignore
            from _otr_video_engines import wrapper_bridge as _wb  # type: ignore
    except ImportError as exc:
        log.warning("[OTR.assets] node-class check skipped (no engine registry): %s", exc)
        return
    # `node_class_mappings` never raises: with no ComfyUI registry it returns
    # {} (wrapper_bridge.py:79-83). An empty map is "nothing to check", not
    # "every class is missing" -- refusing the whole graph with an install
    # message for CheckpointLoaderSimple would be the wrong failure (agy QA).
    # Inside the server the map always holds the core nodes.
    mapping = _wb.node_class_mappings()
    if not mapping:
        log.warning("[OTR.assets] node-class check skipped: no ComfyUI node registry")
        return
    problems = []
    for name in sorted(engines):
        try:
            eng = _vreg.get_engine(name)
        except Exception:  # noqa: BLE001
            continue
        table = getattr(eng, "_node_candidates", None)
        if not callable(table):
            continue
        try:
            candidates = dict(table())
        except Exception:  # noqa: BLE001 -- a table that cannot be read is not a miss
            continue
        absent = []
        for _logical, names in candidates.items():
            try:
                _wb.resolve_node_class(tuple(names), mapping)
            except Exception:  # noqa: BLE001 -- collect EVERY miss before raising
                absent.append("/".join(names))
        if absent:
            # THE FIX FIRST, AND THE RIGHT ONE. Measured 2026-09-25 from the
            # live server's /object_info `python_module`: every class any
            # shipped engine asks for is ComfyUI CORE (`nodes`, `comfy_extras`)
            # except AnimateDiff-Evolved's ADE_* pair. So a missing class is
            # a pack to install only when a known pack provides it
            # (`wrapper_bridge._PACK_FOR_PREFIX`); otherwise it is a ComfyUI
            # too old to have the node, and "install a pack" would send the
            # user hunting for something that does not exist.
            hint = getattr(eng, "NODE_PACK_HINT", None)
            if not hint:
                packs = {}
                for entry in absent:
                    for candidate in entry.split("/"):
                        for prefix, pack, _url in getattr(_wb, "_PACK_FOR_PREFIX", ()):
                            if candidate.startswith(prefix):
                                packs[pack] = True
                if packs:
                    hint = ("Install %s from ComfyUI Manager, then restart ComfyUI"
                            % " and ".join(sorted(packs)))
                else:
                    hint = ("Update ComfyUI: these are built-in ComfyUI nodes that "
                            "this version does not have")
            problems.append("%s -- the video engine '%s' needs node classes "
                            "that are not registered on this server: %s"
                            % (hint, name, ", ".join(absent)))
    if problems:
        raise VisualAssetError(
            "%s. Nothing was downloaded or rendered; fix this and press Run "
            "again." % "; ".join(problems))


def _boot_fix(bc, known, state):
    """The exact change that makes this boot acceptable, and what is unmet.

    Judged against the CHEAPEST accepted contract -- the one this boot misses
    by the fewest knobs -- so a card is never sent to a boot it does not need
    (Cursor review, 2026-09-26). Each knob names its own correction: Sage on,
    ``--cpu`` on, or pinned memory left on for a contract that needs it off.
    A Sage state the probe could not read is reported as exactly that, never
    blamed on a Sage that may already be off; the probe's reason follows in
    ``unmet``.
    """
    best = min(known, key=lambda c: len(bc.check_running_server(c, state=state)))
    unmet = bc.check_running_server(best, state=state)
    spec = bc.contract_spec(best)
    clauses = []
    if spec["sage_attention"] is False and state.get("sage_attention"):
        clauses.append("without SageAttention")
    if spec["cpu"] is False and state.get("cpu"):
        clauses.append("without --cpu")
    if spec["disable_pinned_memory"] and not state.get("disable_pinned_memory"):
        clauses.append("with --disable-pinned-memory")
    if clauses:
        return "Restart ComfyUI " + " and ".join(clauses), unmet
    if spec["sage_attention"] is False and state.get("sage_attention") is None:
        return ("Could not confirm ComfyUI is running without SageAttention",
                unmet)
    return "Restart ComfyUI to match the %r boot" % best, unmet


def _refuse_unmet_boot_contracts(engines, state=None):
    """Refuse at QUEUE time a video engine this server was not BOOTED for --
    before any weight download, writer pass or render.

    THE SECOND HALF OF THE SAME LESSON (PBUG-20260925-02, and the 2026-09-25
    rule: a check the render asks that is knowable at t=0 is asked at t=0).
    An engine that declares boot contracts WITHOUT ``default`` -- MiniMax H3
    today, which needs SageAttention off -- cannot run on a boot that breaks
    its contract, and its own ``assert_usable`` says so only at the first
    video beat. Once its weights auto-download (2026-09-26) that
    would be ~39 GB fetched, a script written and every voice rendered before
    the refusal.

    ONE SOURCE OF TRUTH: the same ``boot_contracts`` checks the adapter's
    render-time ``_assert_boot_contract`` runs. A production policy carries no
    ``launch``, so -- as at render time -- the boot is matched against the
    engine's OWN compatible contracts: met if any one of them is satisfied,
    otherwise refused naming the engine's FIRST declared contract, whose argv
    is the fix. An engine that runs on ``default`` passes untouched, and an
    unreadable boot state (no ComfyUI -- tests, CLI) skips this gate: the
    render-time check still stands behind it.
    """
    try:
        try:
            from ._otr_video_engines import registry as _vreg
            from ._otr_shared import boot_contracts as _bc
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_video_engines import registry as _vreg  # type: ignore
            from _otr_shared import boot_contracts as _bc  # type: ignore
    except ImportError as exc:
        log.warning("[OTR.assets] boot-contract check skipped (no engine registry): %s", exc)
        return
    state = _bc.running_server_boot_state() if state is None else dict(state)
    if not state.get("available"):
        log.warning("[OTR.assets] boot-contract check skipped: %s",
                    state.get("error") or "boot state unavailable")
        return
    problems = []
    for name in sorted(engines):
        try:
            eng = _vreg.get_engine(name)
            eng = eng() if isinstance(eng, type) else eng
            allowed = tuple(_bc.compatible_contracts_for_engine(eng))
        except Exception:  # noqa: BLE001 -- an unregistered id is the registry's refusal
            continue
        if not allowed or _bc.DEFAULT in allowed:
            continue
        # Unknown contract names are skipped, as the render-time
        # identification skips them (boot_contracts.contract_from_running_server
        # filters its candidates the same way); only an engine that declares NO
        # known contract is refused, by name. Then: met when ANY known contract
        # is satisfied by this boot.
        known = tuple(c for c in allowed if _bc.known_contract(c))
        if not known:
            problems.append("the video engine '%s' declares no boot contract "
                            "this pack can check (%s)" % (name, ", ".join(allowed)))
            continue
        if any(not _bc.check_running_server(c, state=state) for c in known):
            continue
        fix, unmet = _boot_fix(_bc, known, state)
        problems.append(
            "%s -- the video engine '%s' cannot run on the boot this server "
            "has: %s" % (fix, name, "; ".join(unmet) or "no accepted boot matches"))
    if problems:
        raise VisualAssetError(
            "%s. Nothing was downloaded or rendered; fix this and press Run "
            "again." % "; ".join(problems))


def ensure_prompt_visual_assets(prompt, unique_id):
    """Called by the already-wired validator before the writer may execute."""
    from ._otr_shared.public_engines import resolve_engine_id
    from ._otr_shared import route_freeze, env as otr_env
    # Resolved through the GUARDED helper, not a hard import: the runtime-bridge
    # tests fake a package tree in sys.modules and stub only the submodules this
    # function needs, so a bare import here is an isolation break that a full-suite
    # run hides (an earlier test leaves `_otr_shared` in sys.modules) and a
    # single-file run exposes. An empty map is fail-safe -- nothing is skipped and
    # every image engine is required -- but it must never be SILENT, because that
    # is the fix quietly not applying.
    role_video_slots = _default_role_video_slots()
    if not role_video_slots:
        log.warning("[OTR.assets] role->video-slot map unavailable; requiring every "
                    "selected image engine (no no-still skipping this run)")
    plan = plan_prompt(prompt, unique_id, resolve_video=resolve_engine_id,
                       freeze_video=route_freeze.freeze_role_engines,
                       role_video_slots=role_video_slots)
    for note in plan["skipped"]:
        log.warning("[OTR.assets] %s", note)
    # The node-pack and boot checks come FIRST: a graph that cannot run must
    # not cost a download (PBUG-20260925-02).
    _refuse_missing_node_packs(plan["engines"])
    _refuse_unmet_boot_contracts(plan["engines"])
    engines = plan["engines"] & _COVERED
    if not engines:
        return {"status": "not-covered", "notes": plan["skipped"], "receipts": []}
    import folder_paths
    from comfy import model_management
    adapters = _load_adapters(engines)
    cancel = model_management.throw_exception_if_processing_interrupted
    cancel()
    requests = native_requests(engines, folder_paths=folder_paths,
                               env=otr_env.snapshot(), **adapters)
    missing = [r for r in requests if r["path"] is None]
    receipts = []
    gui_progress = None
    for item in requests:
        if item["path"] is not None:
            log.info("[OTR.assets] EXISTING %s/%s bytes=%d native=%s "
                     "(preserved; content hash/GPU compatibility not qualified)",
                     item["category"], item["token"], item["path"].stat().st_size, item["path"])
    if missing:
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        from ._otr_visual_asset_download import fetch_verified
        # Finish source validation and report total bytes before transferring.
        for item in missing:
            cancel()
            try:
                item["metadata"] = _pin_metadata(item["spec"], hf_hub_url=hf_hub_url,
                                                get_hf_file_metadata=get_hf_file_metadata)
            except VisualAssetError:
                raise
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                raise VisualAssetError("visual weight metadata failed (%s, HTTP %s) for %s/%s; "
                                       "no token acquisition or fallback" %
                                       (type(exc).__name__, status, item["spec"]["repo_id"],
                                        item["spec"]["filename"])) from None
            meta = item["metadata"]
            log.info("[OTR.assets] PLAN %s/%s revision=%s bytes=%d sha256=%s",
                     item["spec"]["repo_id"], item["spec"]["filename"],
                     meta["commit"], meta["size"], meta["sha256"])
        # "no resume/retry" was true of the hand-rolled loop and is FALSE
        # now: the transfer goes through huggingface_hub, which resumes and
        # retries. A log line that still claimed otherwise would be read as
        # a live warning by the next operator staring at a 36.8 GB fetch.
        log.info("[OTR.assets] missing files=%d total_download_bytes=%d; "
                 "no packs, no substitution; resume/retry via huggingface_hub",
                 len(missing), sum(item["metadata"]["size"] for item in missing))
        # Use Comfy's native execution-context hook, not a server/API call or
        # worker thread. The total-only constructor also supports older Comfy.
        from comfy.utils import ProgressBar
        from ._otr_models_root import model_type_dir as _model_type_dir
        total_download_bytes = sum(item["metadata"]["size"] for item in missing)
        completed_bytes = 0
        gui_progress = ProgressBar(1000)
        gui_progress.update_absolute(0)
        for item in missing:
            cancel()
            # Native order wins; never silently switch to a writable alternate.
            # The owner answers with the folder ComfyUI reads first for this
            # type (under an env pin when one of them sits there).
            destination = _model_type_dir(item["category"],
                                          folder_paths=folder_paths) / item["token"]
            started = time.monotonic()
            last_report = [0.0]

            def progress(done, total):
                # A final chunk is not yet hash-verified/published. Reserve the
                # last 1% until every receipt and native-loader recheck passes.
                # Do not swallow interruption raised by Comfy's progress hook.
                gui_progress.update_absolute(min(
                    990, 990 * (completed_bytes + done) // total_download_bytes))
                now = time.monotonic()
                if done in (0, total) or now - last_report[0] >= 5:
                    log.info("[OTR.assets] DOWNLOAD %s bytes=%d/%d elapsed_s=%.1f",
                             item["token"], done, total, now - started)
                    last_report[0] = now

            receipt = fetch_verified(item["spec"], destination, item["metadata"],
                                     fetch=_hf_fetch, cancel=cancel, progress=progress)
            receipts.append(receipt)
            native = _native_path(folder_paths, item["category"], item["token"])
            if not _same_file(native, destination):
                raise VisualAssetError("download completed but native loader does not resolve "
                                       "the destination; no path repair")
            completed_bytes += item["metadata"]["size"]
            log.info("[OTR.assets] %s %s bytes_verified=%d elapsed_s=%.1f native=%s",
                     receipt["status"].upper(), item["token"], receipt["bytes_verified"],
                     time.monotonic() - started, native)
        # Re-resolve adapter picks as well as native token identity after writes.
        after = native_requests(engines, folder_paths=folder_paths,
                                env=otr_env.snapshot(), **adapters)
        if ([(r["category"], r["token"]) for r in after]
                != [(r["category"], r["token"]) for r in requests]
                or any(r["path"] is None for r in after)):
            raise VisualAssetError("visual asset selection changed or remains missing after "
                                   "download; stopping before writer")
    cancel()
    if gui_progress is not None:
        gui_progress.update_absolute(1000)
    log.info("[OTR.assets] READY engines=%s files=%d (availability only; "
             "render/GPU/publish success not qualified)", ",".join(sorted(engines)), len(requests))
    return {"status": "ready", "engines": sorted(engines), "receipts": receipts,
            "notes": plan["skipped"]}
