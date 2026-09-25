r"""Where this box keeps its model weights, and nothing else.

THE ONE OWNER of ``_models_root()``. Every part of the pack that needs to know
where weights live -- audio engines, video engines, the provisioner, the lane
weight fetcher, the asset index -- asks here.

WHY THIS MODULE EXISTS SEPARATELY. This resolution used to live inside a writer
backend module. That made a question every subsystem asks depend on one
optional backend, so the backend could not be retired without taking the shared
answer with it. Nothing here knows or cares which writer or which loader is in
use; it resolves a directory and raises when it honestly cannot.

IT IS DELIBERATELY DEPENDENCY-LIGHT: stdlib plus the pack's own env shim, no
torch, no transformers, no model library, and the one runtime-only import
(``folder_paths``) is made lazily inside the function and guarded. A caller must
be able to ask where the models are without loading a model, and a cold import
must stay cold.

IT SITS IN ``nodes/`` ON PURPOSE. Step 4 below walks up from ``__file__`` to
find ComfyUI's ``models/`` beside ``custom_nodes/``, so moving this file one
level in or out silently changes which directory that step returns.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

#: ``_models_root`` keeps its leading underscore. Sixteen call sites already
#: import it under that name, and renaming it inside a removal would be churn
#: rather than a fix -- a second public spelling is exactly the kind of drift
#: this extraction exists to end.
__all__ = ["ModelsRootUnresolved", "_models_root", "model_type_dir"]


class ModelsRootUnresolved(RuntimeError):
    r"""No models root could be resolved, and guessing would create garbage.

    Raised only off Windows, and only after the env vars, an existing
    ``C:\ComfyUI-Models``, ComfyUI's ``folder_paths`` and the sibling
    ``models/`` directory have all been tried. Callers that can carry on
    without weights should catch it; a fetcher should let it out, because
    downloading gigabytes into a guessed path is worse than stopping.
    """


def _configured_tree() -> "Path | None":
    """The folder holding the first configured ``checkpoints`` path, or None
    outside a running ComfyUI (no ``folder_paths``, or one without
    ``get_folder_paths``, such as the test stub)."""
    try:
        import folder_paths  # ComfyUI runtime only
        first = folder_paths.get_folder_paths("checkpoints")[0]
    except Exception:  # noqa: BLE001 -- outside ComfyUI, fall through
        return None
    return Path(first).expanduser().parent


def _has_entries(folder: Path) -> bool:
    try:
        return any(folder.iterdir())
    except OSError:
        return False


def _models_root() -> Path:
    r"""Where the weights live: env, then this box's tree, then ComfyUI's own.

    The order matters and each step is here for a reason.

    1. The env vars win outright. That is how anyone puts weights wherever
       they like, and how every pod run is pinned.
    1b. Inside a running ComfyUI, the tree the USER configured: the folder
       holding the first ``checkpoints`` path, which is where
       extra_model_paths.yaml (and ComfyUI Desktop's own config) puts a
       relocated tree first. ADDED 2026-09-25 after the 4060 measured the
       defect: a leftover, EMPTY ``C:\ComfyUI-Models`` won step 2 on bare
       existence and sent the writer's LLM folder out of Desktop's
       ``ComfyUI-Shared\models`` while every other category resolved there.
       A directory existing is not a configuration. The reference machine is
       unchanged by this step: its yaml lists ``C:\ComfyUI-Models\checkpoints``
       first (read from its live ``/internal/folder_paths``), so the answer is
       the same tree for a better reason.
    2. Then the legacy literal, BUT ONLY IF IT EXISTS AND HOLDS SOMETHING.
       Reached outside ComfyUI (fetch scripts, the provisioner), where no
       configuration can be read. It is the reference machine's real 55-entry
       tree and neither env var is set there, so dropping it would relocate
       that machine's script-side root. An empty one is a leftover, not a
       tree.
    3. Then ComfyUI's own folder_paths -- the USER'S configuration, including
       extra_model_paths.yaml, which is the documented way to relocate
       models. This is the friendly default a fresh install should get, on
       any OS, and on POSIX it also avoids creating a directory literally
       named "C:\\ComfyUI-Models" under the working directory.

    WHAT MUST NEVER HAPPEN: preferring <comfy>/models over an existing tree at
    step 2. That machine has BOTH, the second holds a plausible 34 entries,
    and an earlier version guessed it and returned the wrong root while
    looking verified.
    """
    raw = (
        otr_env.get("OTR_COMFYUI_MODELS_ROOT")
        or otr_env.get("COMFYUI_MODELS_ROOT")
    )
    if raw:
        return Path(raw).expanduser()
    configured = _configured_tree()
    if configured is not None:
        return configured
    legacy = Path(r"C:\ComfyUI-Models")
    if legacy.is_dir() and _has_entries(legacy):
        return legacy
    try:
        import folder_paths  # ComfyUI runtime only
        base = getattr(folder_paths, "models_dir", None)
        if base:
            return Path(base).expanduser()
    except Exception:  # noqa: BLE001 -- outside ComfyUI, fall through
        pass
    # 4. The ComfyUI checkout puts custom_nodes/ beside models/, so when this
    #    pack sits where it is installed the sibling directory IS the answer.
    #    This is the step that makes a Linux or Docker box work with no
    #    environment variable, and it has to come before the Windows literal
    #    because step 3 is unavailable outside a running ComfyUI -- which is
    #    exactly when a fetch script calls this.
    #    os.path rather than pathlib on purpose: Path() binds to the host
    #    flavour, so a test that simulates POSIX cannot construct one here.
    #
    #    FOUR dirname() CALLS, COUNTED FROM THIS FILE. It was THREE until
    #    2026-09-23, which landed on <comfy>/custom_nodes/models -- a directory
    #    INSIDE custom_nodes rather than the models/ dir beside it. The comment
    #    above has always described the fourth; the code only ever walked three.
    #
    #    WHERE THE OFF-BY-ONE CAME FROM, stated correctly after a review found
    #    an earlier version of this comment had the story backwards. Commit
    #    fa87d1c4 (2026-09-21) INTRODUCED this step, together with the
    #    off-Windows guard below, to fix a different and more basic defect:
    #    before it there was no sibling lookup at all and the function ended in
    #    an unconditional `return legacy`, which is what put 3.7 GB of weights
    #    into a directory literally named "C:\ComfyUI-Models" inside the repo on
    #    a Linux pod. So this step did not fail during that incident -- it did
    #    not exist yet. It arrived with the wrong depth and has never once
    #    resolved on a normal install, which is why nothing noticed for two
    #    days: the guard below caught the fallout the step was meant to
    #    prevent.
    #
    #    THE DEPTH IS COUNTED FROM THIS FILE'S LOCATION, so moving this module
    #    changes the answer silently. tests/test_models_root_is_one_owner.py
    #    pins both the depth and the resolved path for that reason.
    sibling = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "models")
    if os.path.isdir(sibling):
        return Path(sibling)
    # 5. The literal, last. ADDED A GUARD 2026-09-21: returning it
    #    unconditionally is what put 3.7 GB of AnimateDiff lane weights into a
    #    directory literally named "C:\ComfyUI-Models" INSIDE the repo on a
    #    Linux pod. scripts/otr_fetch_lane_weights.py has its own sibling
    #    fallback for precisely that case and never reached it, because this
    #    function returned a value instead of declining. On POSIX the literal
    #    is not an absolute path at all, so it can only ever create garbage.
    if os.name != "nt":
        raise ModelsRootUnresolved(
            "no models root: set OTR_COMFYUI_MODELS_ROOT (or COMFYUI_MODELS_ROOT), "
            "or run inside ComfyUI so folder_paths is importable. Refusing to "
            "fall back to the Windows path %r off Windows, because it is "
            "relative here and would create a directory of that name." % str(legacy)
        )
    return legacy


#: ComfyUI's own legacy folder names (``folder_paths.map_legacy``), mirrored so
#: the answer off the runtime matches the answer on it.
_LEGACY_CATEGORY = {"unet": "diffusion_models", "clip": "text_encoders"}

_PIN_VARS = ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT")


def _registered_dirs(category: str, folder_paths=None) -> list:
    """The folders ComfyUI scans for ``category``, first one first; empty
    outside a running ComfyUI, for an unknown category, or for a stub without
    ``get_folder_paths``."""
    try:
        if folder_paths is None:
            import folder_paths  # ComfyUI runtime only
        return [Path(p).expanduser()
                for p in folder_paths.get_folder_paths(category)]
    except Exception:  # noqa: BLE001 -- outside ComfyUI, or KeyError
        return []


def _is_under(folder: Path, root: Path) -> bool:
    here = os.path.normcase(os.path.abspath(os.fspath(folder)))
    top = os.path.normcase(os.path.abspath(os.fspath(root)))
    try:
        return os.path.commonpath([here, top]) == top
    except ValueError:  # different drives
        return False


def model_type_dir(category: str, *, folder_paths=None) -> Path:
    r"""The folder for one model TYPE: where ComfyUI looks for it first, which
    is where a new download of that type belongs.

    WHY PER TYPE (2026-09-25). ``_models_root()`` is one folder and callers
    joined a type name onto it. But extra_model_paths.yaml sets every type
    separately -- ``is_default`` applies per yaml block, to the keys that block
    declares -- so on the reference machine ``checkpoints`` resolves first to
    ``C:\ComfyUI-Models\checkpoints`` while ``upscale_models`` resolves first
    to the Documents tree. "Root plus type" can name a folder ComfyUI does not
    read first for that type. Inside ComfyUI, ask ComfyUI.

    1. ``unet`` and ``clip`` become ``diffusion_models`` and ``text_encoders``.
    2. Inside a running ComfyUI, the registered folders for the type. With an
       env pin set, the first of them that sits UNDER the pin, else the first
       -- a pin naming a folder ComfyUI does not scan would fetch files the
       loader cannot load (the visual-assets rule: native order wins).
    3. Otherwise ``_models_root() / category``, where an env pin is exclusive.

    ``folder_paths`` may be passed in (the visual-assets gate is handed one);
    otherwise the runtime module is imported lazily, so this module stays
    cold-import clean. Never raises inside ComfyUI: ``checkpoints`` is always
    registered, so step 3 resolves there. Off Windows with nothing resolvable
    it raises ``ModelsRootUnresolved``, as ``_models_root()`` does.
    """
    category = _LEGACY_CATEGORY.get(category, category)
    registered = _registered_dirs(category, folder_paths)
    if registered:
        raw = next((otr_env.get(v) for v in _PIN_VARS if otr_env.get(v)), None)
        if raw:
            pin = Path(raw).expanduser()
            for folder in registered:
                if _is_under(folder, pin):
                    return folder
        return registered[0]
    return _models_root() / category
