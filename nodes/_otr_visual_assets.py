"""Pre-writer native visual-weight readiness for the shipped canonical graph.

No model imports or network at module import. Only the five allowlisted default
files below can be fetched. Existing native loader choices are preserved, not
rehash-qualified, and readiness is NOT a claim of GPU/render compatibility.
Other engines keep their existing adapter checks with explicit uncovered logs.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

log = logging.getLogger(__name__)

_SOURCES = (
    ("diffusion_models", "Comfy-Org/z_image_turbo",
     "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ("text_encoders", "Comfy-Org/z_image_turbo",
     "split_files/text_encoders/qwen_3_4b.safetensors"),
    ("vae", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
    ("checkpoints", "Lightricks/LTX-Video", "ltxv-2b-0.9.8-distilled.safetensors"),
    ("text_encoders", "comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
)
MANIFEST = {(category, filename.rsplit("/", 1)[-1]):
            {"repo_id": repo, "filename": filename}
            for category, repo, filename in _SOURCES}
_VIDEO_SLOTS = ("announcer_video_model", "music_video_model", "character_video_model")
_IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
_COVERED = frozenset({"z_image_turbo", "ltx_8gb"})


class VisualAssetError(RuntimeError):
    """Early readiness refusal, before the writer or any visual model load."""


def _literal(inputs, name, default=None):
    value = inputs.get(name, default)
    if not isinstance(value, str):
        raise VisualAssetError(
            "visual asset preflight cannot inspect linked/dynamic %s before execution; "
            "no asset download or model substitution was performed" % name)
    return value.strip()


def plan_prompt(prompt, unique_id, *, resolve_video, freeze_video):
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
    scoped = []
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        gate = inputs.get("gate_in")
        if (isinstance(gate, (list, tuple)) and len(gate) == 2
                and str(gate[0]) == str(unique_id) and gate[1] == 0):
            scoped.append(node)
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
        videos = {}
        for slot in _VIDEO_SLOTS:
            picked = _literal(inputs, slot)
            if not picked or picked.startswith("+ Add Custom"):
                raise VisualAssetError("visual asset preflight requires an explicit video "
                                       "engine selection for " + slot)
            videos[slot] = resolve_video(picked)
        effective = freeze_video(videos)
        result["engines"].update(resolve_video(v) for v in effective.values() if v)
        for slot in _IMAGE_SLOTS:
            picked = _literal(inputs, slot)
            if not picked or picked.startswith("+ Add Custom"):
                raise VisualAssetError("visual asset preflight requires an explicit image "
                                       "engine selection for " + slot)
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


def native_requests(engines, *, folder_paths, zimage=None, ltx=None, env=None):
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


class _HTTPSOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise VisualAssetError("visual weight transport refused non-HTTPS redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@contextmanager
def _open_stream(url):
    try:
        request = Request(url, headers={"Accept-Encoding": "identity",
                                       "User-Agent": "OTR-visual-weight-readiness"})
        with build_opener(_HTTPSOnlyRedirect()).open(request, timeout=30) as response:
            yield response
    except HTTPError as exc:
        try:
            exc.close()  # open() failed before the response context was entered.
        finally:
            raise VisualAssetError("visual weight download HTTP %d; no retry/token acquisition" %
                                   exc.code) from None
    except (URLError, TimeoutError) as exc:
        # URLs in external exceptions can contain signed CDN query parameters.
        raise VisualAssetError("visual weight network failure (%s); no retry" %
                               type(exc).__name__) from None


def ensure_prompt_visual_assets(prompt, unique_id):
    """Called by the already-wired validator before the writer may execute."""
    from ._otr_shared.public_engines import resolve_engine_id
    from ._otr_shared import route_freeze, env as otr_env
    plan = plan_prompt(prompt, unique_id, resolve_video=resolve_engine_id,
                       freeze_video=route_freeze.freeze_role_engines)
    for note in plan["skipped"]:
        log.warning("[OTR.assets] %s", note)
    engines = plan["engines"] & _COVERED
    if not engines:
        return {"status": "not-covered", "notes": plan["skipped"], "receipts": []}
    import folder_paths
    from comfy import model_management
    zimage = ltx = None
    if "z_image_turbo" in engines:
        from ._otr_image_engines import z_image_turbo as zimage
    if "ltx_8gb" in engines:
        from ._otr_video_engines.eng_ltx_8gb import Ltx8gbEngine
        ltx = Ltx8gbEngine()
    cancel = model_management.throw_exception_if_processing_interrupted
    cancel()
    requests = native_requests(engines, folder_paths=folder_paths, zimage=zimage,
                               ltx=ltx, env=otr_env.snapshot())
    missing = [r for r in requests if r["path"] is None]
    receipts = []
    for item in requests:
        if item["path"] is not None:
            log.info("[OTR.assets] EXISTING %s/%s bytes=%d native=%s "
                     "(preserved; content hash/GPU compatibility not qualified)",
                     item["category"], item["token"], item["path"].stat().st_size, item["path"])
    if missing:
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        from ._otr_visual_asset_download import download_verified
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
        log.info("[OTR.assets] missing files=%d total_download_bytes=%d; "
                 "no packs, no substitution, no resume/retry",
                 len(missing), sum(item["metadata"]["size"] for item in missing))
        for item in missing:
            cancel()
            # Native order wins; never silently switch to a writable alternate.
            destination = Path(folder_paths.get_folder_paths(item["category"])[0]) / item["token"]
            started = time.monotonic()
            last_report = [0.0]

            def progress(done, total):
                now = time.monotonic()
                if done in (0, total) or now - last_report[0] >= 5:
                    log.info("[OTR.assets] DOWNLOAD %s bytes=%d/%d elapsed_s=%.1f",
                             item["token"], done, total, now - started)
                    last_report[0] = now

            receipt = download_verified(item["spec"], destination, item["metadata"],
                                        open_stream=_open_stream, cancel=cancel, progress=progress)
            receipts.append(receipt)
            native = _native_path(folder_paths, item["category"], item["token"])
            if not _same_file(native, destination):
                raise VisualAssetError("download completed but native loader does not resolve "
                                       "the destination; no path repair")
            log.info("[OTR.assets] %s %s bytes_verified=%d elapsed_s=%.1f",
                     receipt["status"].upper(), item["token"], receipt["bytes_verified"],
                     time.monotonic() - started)
        # Re-resolve adapter picks as well as native token identity after writes.
        after = native_requests(engines, folder_paths=folder_paths, zimage=zimage,
                                ltx=ltx, env=otr_env.snapshot())
        if ([(r["category"], r["token"]) for r in after]
                != [(r["category"], r["token"]) for r in requests]
                or any(r["path"] is None for r in after)):
            raise VisualAssetError("visual asset selection changed or remains missing after "
                                   "download; stopping before writer")
    cancel()
    log.info("[OTR.assets] READY engines=%s files=%d (availability only; "
             "render/GPU/publish success not qualified)", ",".join(sorted(engines)), len(requests))
    return {"status": "ready", "engines": sorted(engines), "receipts": receipts,
            "notes": plan["skipped"]}
