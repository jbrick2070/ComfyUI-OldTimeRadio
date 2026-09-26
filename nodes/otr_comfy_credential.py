"""OTR_ComfyCredential -- the ONE node that receives the queue's Comfy API key.

WHY THIS NODE EXISTS (plan 0k, 2026-09-26). ComfyUI hands a V1 node its
hidden ``API_KEY_COMFY_ORG`` input as an ordinary input value, and when that
node RAISES, the executor formats every input value -- the key included --
into the error record it writes to /history, /api/jobs and the websocket
error event (ComfyUI v0.36.0 execution.py: get_input_data -> format_value ->
current_inputs). Nine OTR nodes used to declare the key, and several refuse
ON PURPOSE (the Workflow Validator's balance, asset and story gates), so an
ordinary refusal published the key to anyone who can read the history -- on
a pod bound to 0.0.0.0, anyone the proxy admits.

NOW ONLY THIS NODE DECLARES IT, AND IT CANNOT RAISE: all of its work sits in
one try/except, and its only output is a fixed token that never carries the
key. It binds the key to the running prompt in both places the pack spends
from -- the per-prompt stash the partner media sessions open with, and the
Comfy Credits writer backend, whose auth is now tied to the same prompt id so
a queue without this node can never spend the previous queue's key -- and
every host reads it from there. Its one link feeds the Workflow Validator,
the only root of every shipped workflow, so it runs before any node that
could spend.

WHY NOT A V3 NODE, which is how Comfy's own partner nodes avoid the leak (a
V3 node's hidden values never enter an error record): the pack's tests and
its offline schema and graph tools run outside ComfyUI, where ``comfy_api``
cannot be imported, and would lose sight of the node. A V1 node that cannot
raise closes the same leak and loads everywhere.

NOT AN API NODE (no ``api_node`` flag): that flag makes ComfyUI inject the
signed-in session bearer as well, which the Comfy Registry security scan
reads as credential access (PBUG-20260902-04), and paints the node as a paid
partner node.
"""
from __future__ import annotations

import logging

log = logging.getLogger("OTR.comfy_credential")

#: The node's output. NEVER the key: an output is held in the execution cache
#: and handed to whatever the node is wired to.
KEY_PRESENT = "comfy_key:present"
KEY_ABSENT = "comfy_key:absent"


def bind_queue_credential(api_key) -> bool:
    """Bind this queue's Comfy API key to the running prompt. True only when a
    key was given AND bound -- a key that could not be bound is reported as
    absent, so the node's token never claims a credential the spend paths do
    not have. Never raises, and never logs the key or an exception message
    that might quote it.

    Always SETS the writer backend's auth, even to nothing: a queue that
    carries no key must clear the previous queue's (the 2026-09-19 codex
    rule), and the prompt id it is bound to makes a stale one unusable even
    if this node is missing from a later graph.
    """
    key = api_key.strip() if isinstance(api_key, str) else ""
    bound = False
    try:
        from ._otr_shared.cloud_media_invoke import (
            current_prompt_id, stash_comfy_api_key)
        try:
            prompt_id = current_prompt_id()
        except Exception:  # noqa: BLE001 -- no executing prompt (tests, CLI)
            prompt_id = ""
        from . import _otr_comfy_backend
        _otr_comfy_backend.set_auth(api_key=key or None,
                                    prompt_id=prompt_id or None)
        if key:
            stash_comfy_api_key(key)
            bound = True
    except Exception as exc:  # noqa: BLE001 -- this node must never raise
        log.warning("[OTR credential] the queue's Comfy key could not be "
                    "bound (%s); cloud lanes will refuse with the reason",
                    type(exc).__name__)
    return bound


class OTR_ComfyCredential:
    DESCRIPTION = (
        "Receives this queue's Comfy API key (from the app when you are "
        "signed in with an API key, or from a headless submitter) and hands "
        "it to the nodes that spend Comfy credits. It is the only node that "
        "sees the key, and it never fails, so the key can never appear in an "
        "error report. Keep it wired into the Workflow Validator; a local-only "
        "run needs no key and loses nothing."
    )
    CATEGORY = "OldTimeRadio"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("credential",)
    OUTPUT_TOOLTIPS = (
        "comfy_key:present or comfy_key:absent -- never the key itself. Wire "
        "it into the Workflow Validator's credential input so this node runs "
        "first.",
    )
    FUNCTION = "bind"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {"api_key_comfy_org": "API_KEY_COMFY_ORG"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every queue brings its own key under its own prompt id; a cached
        # result would bind nothing for this one.
        return float("nan")

    def bind(self, api_key_comfy_org=None):
        try:
            present = bind_queue_credential(api_key_comfy_org)
        except Exception:  # noqa: BLE001 -- belt: the helper never raises
            present = False
        return (KEY_PRESENT if present else KEY_ABSENT,)


NODE_CLASS_MAPPINGS = {"OTR_ComfyCredential": OTR_ComfyCredential}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_ComfyCredential": "0 - Comfy Credential"}
