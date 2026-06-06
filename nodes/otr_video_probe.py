"""OTR_VideoProbe -- detect usable video engines + host capabilities (A-S1/W0).

The probe is the platform's "what can actually run here" node. It enumerates the
registered video adapters, runs each one's registry-level ``assert_usable`` for
the roles it claims, and emits (1) a ``host_caps`` JSON (ffmpeg / torch-cuda /
NVML VRAM, all probed LAZILY + fail-soft), (2) a usable-engine list per role, and
(3) a plain validator report. It runs BEFORE render and NEVER mutates a widget
(V-6); the Director consumes its usable list as static data, not as a dynamic
COMBO rewrite.

In CW-1 no concrete adapters are registered yet (they land CW-4+), so the probe
reports an empty usable set + the host floor -- exactly what a box-fresh platform
should say. It is additive and cold-import-clean (V-12): module scope imports
only stdlib + the dep-free registry; torch / pynvml are imported lazily inside
``probe`` and every probe is wrapped so a missing dep degrades to "unknown",
never an exception.
"""
from __future__ import annotations

import json
import logging
import shutil

log = logging.getLogger("OTR")

from ._otr_video_engines import registry as _vreg
from ._otr_shared.role_compat import ROLES


class OTRVideoProbe:
    """Registered as ``OTR_VideoProbe``. Host + usable-engine probe."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "probe"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("host_caps_json", "usable_engines_json", "validator_report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Optional ordering signal (wire an upstream done). "
                        "Treated as an opaque STRING -- not parsed."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Local-disk / host probe only; nothing to reject at queue time.
        return True

    # ------------------------------------------------------------------ #
    def probe(self, gate_in=""):
        host_caps = self._host_caps()
        usable = self._usable_engines()
        report = self._report(host_caps, usable)
        return (
            json.dumps(host_caps, ensure_ascii=True, separators=(",", ":")),
            json.dumps(usable, ensure_ascii=True, separators=(",", ":")),
            report,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _host_caps() -> dict:
        """Lazily probe the host; every field fail-soft to 'unknown'.

        No heavy import at module scope (V-12). torch / pynvml are imported
        here inside guards so the probe never crashes a box-fresh graph.
        """
        caps: dict = {
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "ffprobe": bool(shutil.which("ffprobe")),
            "cuda_available": "unknown",
            "torch_version": "unknown",
            "total_vram_mb": "unknown",
        }
        try:  # torch is optional in the bare test env
            import torch  # noqa: F401

            caps["torch_version"] = str(getattr(torch, "__version__", "unknown"))
            try:
                caps["cuda_available"] = bool(torch.cuda.is_available())
            except Exception:  # noqa: BLE001
                caps["cuda_available"] = "unknown"
        except Exception:  # noqa: BLE001
            pass
        try:  # machine-wide NVML (the real cross-process VRAM truth, AS-3/CW-2)
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            caps["total_vram_mb"] = int(info.total // (1024 * 1024))
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
        return caps

    @staticmethod
    def _usable_engines() -> dict:
        """Per-role list of registered engines that pass registry assert_usable.

        Fail-closed per engine: an engine that raises ``EngineUnusable`` for a
        role is simply absent from that role's list (availability, never a
        hierarchy). With no adapters registered (CW-1) every list is empty.
        """
        out: dict = {role: [] for role in ROLES}
        for role in ROLES:
            for name in _vreg.engines_for_role(role):
                try:
                    _vreg.assert_usable(name, role)
                    out[role].append(name)
                except _vreg.EngineUnusable:
                    continue
        out["_all_registered"] = _vreg.all_engine_names()
        return out

    @staticmethod
    def _report(host_caps, usable) -> str:
        lines = ["[OTR_VideoProbe] host + usable-engine report"]
        lines.append(
            f"  ffmpeg={host_caps.get('ffmpeg')} "
            f"cuda={host_caps.get('cuda_available')} "
            f"vram_mb={host_caps.get('total_vram_mb')}"
        )
        all_reg = usable.get("_all_registered", [])
        lines.append(f"  registered video engines: {len(all_reg)} {all_reg}")
        for role in ROLES:
            lines.append(f"  role {role}: usable={usable.get(role, [])}")
        if not all_reg:
            lines.append(
                "  (no concrete video adapters registered yet -- radio-floor "
                "+ engine families land in later windows; the platform shell "
                "is in place)"
            )
        return "\n".join(lines)
