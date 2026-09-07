"""Pre-writer native visual-weight readiness for the shipped canonical graph.

No model imports or network at module import. Only the SEVEN allowlisted default
files below can be fetched (three z_image_turbo, two ltx_8gb, two stable_audio_3 --
it said five until the stable_audio_3 rows landed 2026-09-06). Existing native loader choices are preserved, not
rehash-qualified, and readiness is NOT a claim of GPU/render compatibility.
Other engines keep their existing adapter checks with explicit uncovered logs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time

log = logging.getLogger(__name__)

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
    # Hub listing and against eng_stable_audio_3._CKPT / ._TENC.
    ("checkpoints", "Comfy-Org/stable-audio-3",
     "checkpoints/stable_audio_3_small_music.safetensors"),   # 2,270,384,940 B
    ("text_encoders", "Comfy-Org/stable-audio-3",
     "text_encoders/t5gemma_b_b_ul2.safetensors"),            # 1,187,264,003 B
)
MANIFEST = {(category, filename.rsplit("/", 1)[-1]):
            {"repo_id": repo, "filename": filename}
            for category, repo, filename in _SOURCES}
_VIDEO_SLOTS = ("announcer_video_model", "music_video_model", "character_video_model")
_IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
_COVERED = frozenset({"z_image_turbo", "ltx_8gb", "stable_audio_3"})
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
    # THE GATE CHAIN IS WALKED TRANSITIVELY, not one hop (PBUG-20260907-02).
    #
    # This used to collect only nodes whose `gate_in` named THIS validator
    # directly, and the audio branch is two hops away:
    #
    #     validator -> BatchCharacterVoices -> AnnouncerVoice -> StableAudioTheme
    #
    # so the music engine was never read and `stable_audio_3` was never planned.
    # Selecting it therefore killed the render 5m40s in, after the whole script
    # had been written, with "SA3 checkpoint not found ... fetch it first" -- the
    # licence-clean music engine was unusable on every card while musicgen
    # (CC-BY-NC) was the only thing that worked.
    #
    # THE REPLAY ISOLATION THE DOCSTRING PROTECTS IS UNCHANGED. Reachability
    # still starts at THIS validator and follows gate edges only, so another
    # validator's subgraph remains unreachable -- a frozen replay bundle's live
    # widgets still cannot trigger a download. Widening from one hop to N hops
    # along the same edges does not cross that boundary; it just stops losing
    # the far end of our own chain.
    scoped = []
    seen_ids = set()
    reachable = {str(unique_id)}
    pending = True
    while pending:
        pending = False
        for node_id, node in prompt.items():
            if not isinstance(node, dict) or node_id in seen_ids:
                continue
            gate = (node.get("inputs") or {}).get("gate_in")
            # NO SOURCE-SLOT CONSTRAINT, and that is deliberate. The old code
            # required `gate[1] == 0`, which happens to be true for the
            # validator's single output and is FALSE for every chain hop: the
            # audio nodes pass their gate on from output slot 2, so
            # `gate_in` reads ['81', 2] and ['82', 2]. Requiring slot 0 rejected
            # exactly the hops this walk exists to follow, and the first cut of
            # this fix still logged "READY engines=ltx_8gb,z_image_turbo" with
            # the graph correctly wired. The API prompt keys inputs by NAME, so
            # `gate_in` is already unambiguous -- which output slot happens to
            # carry the gate is the source node's business, not ours.
            if (isinstance(gate, (list, tuple)) and len(gate) == 2
                    and str(gate[0]) in reachable):
                seen_ids.add(node_id)
                reachable.add(str(node_id))
                scoped.append(node)
                pending = True  # its own consumers may now be reachable
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
                    env=None):
    """Bind the adapters' exact tokens to native folders; no writes/network.

    A missing nondefault choice is a refusal, never a default-weight fallback.
    Explicit paths must name the same file the native basename loader will use.
    """
    env = env or {}
    requests = []

    def add(category, token, *, explicit="", authority=None):
        token = str(token)
        loader_token = os.path.basename(token)
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
            roots = folder_paths.get_folder_paths(category)
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
    if "ltx_8gb" in engines:
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
        # `_CKPT` / `_TENC` are the adapter's own resolved names and already
        # honour OTR_SA3_CKPT / OTR_SA3_TEXT_ENCODER, so an operator pin is
        # passed through as `explicit` exactly as the Z-Image branch does with
        # its own env keys.
        if sa3 is None:
            raise VisualAssetError("stable_audio_3 adapter resolution is unavailable")
        add("checkpoints", sa3._CKPT,
            explicit=str((env or {}).get("OTR_SA3_CKPT") or ""))
        add("text_encoders", sa3._TENC,
            explicit=str((env or {}).get("OTR_SA3_TEXT_ENCODER") or ""))
    return requests


def _pin_metadata(spec, *, hf_hub_url, get_hf_file_metadata):
    """Pin public source HEAD, then verify metadata again at that commit.

    The server's 64-hex LFS etag is the expected content SHA-256. A git blob
    etag, missing size, gated source or changing metadata is a hard refusal.
    No credential is requested/read. The GET uses the pinned Hub URL so a CDN
    URL obtained before other large transfers cannot expire in our queue.
    """
    if spec not in MANIFEST.values():
        raise VisualAssetError("visual weight source is not allowlisted")
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
            "visual weight transfer failed (%s) for %s/%s"
            % (type(exc).__name__, spec["repo_id"], spec["filename"])) from None


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
    engines = plan["engines"] & _COVERED
    if not engines:
        return {"status": "not-covered", "notes": plan["skipped"], "receipts": []}
    import folder_paths
    from comfy import model_management
    zimage = ltx = sa3 = None
    if "z_image_turbo" in engines:
        from ._otr_image_engines import z_image_turbo as zimage
    if "ltx_8gb" in engines:
        from ._otr_video_engines.eng_ltx_8gb import Ltx8gbEngine
        ltx = Ltx8gbEngine()
    if "stable_audio_3" in engines:
        from ._otr_audio_engines import eng_stable_audio_3 as sa3
    cancel = model_management.throw_exception_if_processing_interrupted
    cancel()
    requests = native_requests(engines, folder_paths=folder_paths, zimage=zimage,
                               ltx=ltx, sa3=sa3, env=otr_env.snapshot())
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
        total_download_bytes = sum(item["metadata"]["size"] for item in missing)
        completed_bytes = 0
        gui_progress = ProgressBar(1000)
        gui_progress.update_absolute(0)
        for item in missing:
            cancel()
            # Native order wins; never silently switch to a writable alternate.
            destination = Path(folder_paths.get_folder_paths(item["category"])[0]) / item["token"]
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
        after = native_requests(engines, folder_paths=folder_paths, zimage=zimage,
                                ltx=ltx, sa3=sa3, env=otr_env.snapshot())
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
