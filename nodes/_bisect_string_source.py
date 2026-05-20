"""nodes/_bisect_string_source.py -- minimal STRING-emit node for the
BUG-LOCAL-231 step 6 re-add bisect.

The deferred-loader bisect variants need a STRING source to feed
`gate_signal` (forceInput=True). Stock ComfyUI has no primitive
STRING-output node, and we don't want to introduce community pack
dependencies. This one-class file is a temporary investigation
artifact -- Step 9 cleanup deletes it along with the
`workflows/_bisect_*.json` variant files.

Architectural notes:
  * No model load. No GPU touch. No side effects.
  * Pure passthrough: widget text -> STRING output.
  * forceInput=False on the widget so it's settable from the UI.
  * Registered in nodes/__init__.py under class name
    `OTR_BisectStringSource` and display name "Bisect String
    Source (delete after BUG-LOCAL-231 closes)" to make the
    temporary nature visible.
"""
from __future__ import annotations


class BisectStringSource:
    """STRING emit node for BUG-LOCAL-231 bisect variants only.
    Delete after Step 9 cleanup.
    """

    FUNCTION = "emit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = False
    CATEGORY = "OTR/bisect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "bisect-string-source: gate-signal-stub",
                    "tooltip": (
                        "Static STRING value emitted to downstream "
                        "consumers. Used as a stub gate_signal source "
                        "in BUG-LOCAL-231 bisect variants. "
                        "Temporary -- delete after bisect closes."
                    ),
                }),
            },
        }

    def emit(self, text: str):
        return (text,)


__all__ = ["BisectStringSource"]
