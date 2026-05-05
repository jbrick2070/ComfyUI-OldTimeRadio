"""
otr_save_copy.py  --  per-stage video save tee for OTR smoke / QA workflows
============================================================================

A tiny pass-through node that COPIES the mp4 from any upstream OTR
node's STRING video-path output to a sibling file with a configurable
suffix, so each stage of a smoke run produces a distinct on-disk
deliverable that downstream stages cannot clobber.

Use case: BUG-LOCAL-031 smoke workflow re-runs Composite -> Upscale ->
Blend on pre-rendered assets. The composite output overwrites
``<ep>/composited/<ep>.mp4`` and the upscale output overwrites
``obs/<ep>.mp4`` -- so the ORIGINAL canonical-named files from the
prior live run get destroyed before you can compare. Inserting an
``OTR_SaveCopy`` after each stage with a distinct ``out_suffix``
copies that stage's output to ``<source>_<suffix>.mp4`` BEFORE the
next stage runs. The next stage still operates on the original path
(passed through unchanged), and you get a side-by-side QA archive on
disk with one file per stage.

Behavior:
  * Input:  ``video_path`` (STRING, forced input from upstream node)
            ``out_suffix`` (STRING widget, default ``_smoke_stage``)
  * Side effect: copies ``<src_dir>/<src_stem>.<src_ext>`` ->
                 ``<src_dir>/<src_stem><out_suffix>.<src_ext>``.
                 If the destination already exists it is overwritten
                 (safe for repeated smoke iterations).
  * Output: ``video_path`` (STRING) -- the ORIGINAL upstream path
            unchanged. Downstream nodes wired to this output operate
            on the upstream's canonical file, NOT on the copy.

Non-invasive: production nodes are unchanged; this is a sidecar tee.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

# folder_paths is the ComfyUI canonical path resolver. This node copies
# from a caller-supplied source path to a sibling file, so it doesn't
# need to resolve the output_dir itself -- but the import documents the
# Bug Bible BUG-01.02 contract that every OUTPUT_NODE module references
# the canonical resolver, even when the node operates on caller paths.
import folder_paths  # noqa: F401

log = logging.getLogger("OTR")


class SaveCopy:
    """Copy a video mp4 to a sibling file with a configurable suffix.

    Use as a tee in QA / smoke workflows so each pipeline stage's
    output is preserved on disk before the next stage runs and
    potentially clobbers the upstream's canonical filename.
    """

    CATEGORY = "OTR/v2/Visual"
    OUTPUT_NODE = True
    FUNCTION = "save_copy"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Path to an mp4 (or other file) to tee. Wire "
                        "from any upstream OTR node's STRING video-"
                        "path output. The path passes through "
                        "unchanged on the output side; the side "
                        "effect is the on-disk copy at "
                        "<src_dir>/<src_stem><out_suffix>.<src_ext>."
                    ),
                }),
                "out_suffix": ("STRING", {
                    "multiline": False,
                    "default": "_smoke_stage",
                    "tooltip": (
                        "Suffix appended to the source filename "
                        "stem before the extension. e.g. source = "
                        "'<ep>.mp4' + suffix = '_smoke_composite' "
                        "yields '<ep>_smoke_composite.mp4' next to "
                        "the source. Use distinct suffixes per "
                        "pipeline stage so each stage's output is "
                        "uniquely named on disk."
                    ),
                }),
            },
        }

    def save_copy(self, video_path: str, out_suffix: str = "_smoke_stage"):
        src = Path((video_path or "").strip())
        report_lines: list[str] = []
        if not src.exists():
            msg = (
                f"OTR_SaveCopy: video_path does not exist: {src} "
                f"(suffix={out_suffix!r}); passing through unchanged"
            )
            log.warning("[OTR_SaveCopy] %s", msg)
            report_lines.append(msg)
            return {
                "ui": {"text": [msg]},
                "result": (str(src),),
            }

        suffix = (out_suffix or "_smoke_stage").strip()
        if not suffix:
            suffix = "_smoke_stage"
        # Always anchor the suffix BEFORE the extension. e.g.
        # foo.mp4 + "_x" -> foo_x.mp4; foo.bar.mp4 + "_x" -> foo.bar_x.mp4.
        dst = src.with_name(f"{src.stem}{suffix}{src.suffix}")
        try:
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / (1024 * 1024)
            msg = (
                f"OTR_SaveCopy: {src.name} -> {dst.name} "
                f"({size_mb:.2f} MB) [suffix={suffix!r}]"
            )
            log.info("[OTR_SaveCopy] %s", msg)
            report_lines.append(msg)
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"OTR_SaveCopy: copy {src.name} -> {dst.name} "
                f"FAILED: {exc}; passing through original path"
            )
            log.warning("[OTR_SaveCopy] %s", msg)
            report_lines.append(msg)

        # ALWAYS pass the original upstream path through. Downstream
        # operates on the canonical file, not on the copy. The copy
        # is a side-effect QA archive only.
        return {
            "ui": {"text": report_lines},
            "result": (str(src),),
        }


NODE_CLASS_MAPPINGS = {"OTR_SaveCopy": SaveCopy}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_SaveCopy": "[QA] Save Copy (per-stage tee)",
}
