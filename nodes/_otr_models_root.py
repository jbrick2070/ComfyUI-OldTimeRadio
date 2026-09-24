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

IT SITS IN ``nodes/`` ON PURPOSE, at the same directory depth as its previous
home. Step 4 below walks up from ``__file__`` to find ComfyUI's ``models/``
beside ``custom_nodes/``, so moving this file one level in or out would silently
change which directory that step returns. Same depth, same answer -- verified,
not assumed.
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
__all__ = ["ModelsRootUnresolved", "_models_root"]


class ModelsRootUnresolved(RuntimeError):
    r"""No models root could be resolved, and guessing would create garbage.

    Raised only off Windows, and only after the env vars, an existing
    ``C:\ComfyUI-Models``, ComfyUI's ``folder_paths`` and the sibling
    ``models/`` directory have all been tried. Callers that can carry on
    without weights should catch it; a fetcher should let it out, because
    downloading gigabytes into a guessed path is worse than stopping.
    """


def _models_root() -> Path:
    r"""Where the weights live: env, then this box's tree, then ComfyUI's own.

    The order matters and each step is here for a reason.

    1. The env vars win outright. That is how anyone puts weights wherever
       they like, and how every pod run is pinned.
    2. Then the legacy literal, BUT ONLY IF IT EXISTS. It is the reference
       machine's real 55-entry tree and neither env var is set there, so
       dropping it would relocate that machine's entire model root. Guarding
       it on existence keeps that box working while making the literal
       invisible to everybody else.
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
    legacy = Path(r"C:\ComfyUI-Models")
    if legacy.is_dir():
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
    #    THE THREE dirname() CALLS ARE COUNTED FROM THIS FILE'S LOCATION. This
    #    module lives in nodes/, exactly where the previous owner did, so the
    #    walk lands on the same directory it always did. Moving this file
    #    changes that answer silently; tests/test_models_root_is_one_owner.py
    #    pins the depth for that reason.
    sibling = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "models")
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
