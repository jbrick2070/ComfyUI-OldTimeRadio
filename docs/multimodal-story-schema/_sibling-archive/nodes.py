"""Live ComfyUI nodes for the OTR upstream story lab v2 (transplant workspace).

Every dropdown choice is REGISTRY-DISCOVERED from fixtures JSON - no
module-level content lists (kibitz r1). Unknown/unrunnable selections raise;
there are no hidden compatibility fallbacks. This package is not imported by
production and does not touch the production workflow.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent
LAB_SRC = LAB_ROOT / "src"


def _ensure_lab_importable() -> None:
    if not LAB_SRC.exists():
        raise RuntimeError(f"Upstream Story Lab source folder missing: {LAB_SRC}")
    lab_src = str(LAB_SRC)
    if lab_src not in sys.path:
        sys.path.insert(0, lab_src)


_REGISTRY_CACHE: tuple[str, object] | None = None


def _registry():
    """Registry cached behind the state digest (kibitz r3, Claude Code S5):
    ComfyUI calls INPUT_TYPES repeatedly; rebuilding+revalidating every
    fixture each time is waste. A digest change rebuilds."""

    global _REGISTRY_CACHE
    _ensure_lab_importable()
    from upstream_story_lab.registry import Registry

    stamp = _lab_state_digest()
    if _REGISTRY_CACHE is not None and _REGISTRY_CACHE[0] == stamp:
        return _REGISTRY_CACHE[1]
    registry = Registry(LAB_ROOT)
    _REGISTRY_CACHE = (stamp, registry)
    return registry


def _lab_state_digest() -> str:
    digest = hashlib.sha256()
    paths = [
        LAB_ROOT / "__init__.py",
        LAB_ROOT / "nodes.py",
        LAB_ROOT / "PRODUCTION_MIRROR_MANIFEST.md",
    ]
    # production_mirror included (kibitz r3 anchor + Codex r3 M4): the
    # validator reads the mirror for drift checks, so a re-mirror must
    # invalidate ComfyUI's cache.
    for folder in (LAB_SRC, LAB_ROOT / "fixtures", LAB_ROOT / "production_mirror"):
        if folder.exists():
            paths.extend(
                p for p in folder.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            )
    for path in sorted(paths):
        try:
            stat = path.stat()
            digest.update(path.relative_to(LAB_ROOT).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        except OSError as exc:
            digest.update(f"{path}:ERROR:{exc}".encode("utf-8"))
    return digest.hexdigest()


def _bank_choices() -> list[str]:
    return sorted(_registry().banks)


def _model_choices() -> list[str]:
    registry = _registry()
    models = {
        model_id
        for (_bank, model_id, _pipe), (pack, _p) in registry.packs.items()
        if pack.status != "experimental"
    }
    return sorted(models) + ["auto"]


def _pipeline_choices() -> list[str]:
    return sorted(_registry().pipelines) + ["auto"]


def _style_choices() -> list[str]:
    return sorted(_registry().styles) + ["auto"]


class OTR_UpstreamStoryLabValidator:
    """Run the registry + matrix + leakage + drift checks and report."""

    CATEGORY = "OTR/Upstream Story Lab"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "validate"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        return _lab_state_digest()

    def validate(self) -> tuple[str]:
        _ensure_lab_importable()
        from upstream_story_lab.bridge import build_spec
        from upstream_story_lab.compat import (
            MOTION_ROLE_KEYS,
            NEWS_BRIEFS_FIELDS,
            PRODUCTION_VISUAL_TAILS,
            extract_motion_role_keys,
            extract_news_briefs_fields,
            extract_visual_tails,
        )
        from upstream_story_lab.preview import (
            render_prompt_preview,
            scan_story_leakage,
            scan_visual_leakage,
        )

        registry = _registry()
        mirror = LAB_ROOT / "production_mirror" / "nodes"

        drift: list[str] = []
        if tuple(extract_news_briefs_fields(mirror / "news_interpreter.py")) != NEWS_BRIEFS_FIELDS:
            drift.append("NewsBriefs fields drifted")
        if tuple(extract_motion_role_keys(
            mirror / "_otr_video_engines" / "render_driver.py"
        )) != MOTION_ROLE_KEYS:
            drift.append("motion role keys drifted")
        if extract_visual_tails(mirror / "_otr_story_brief_helpers.py") != PRODUCTION_VISUAL_TAILS:
            drift.append("visual tail constants drifted")
        if drift:
            raise RuntimeError(
                "Production mirror drift detected: " + "; ".join(drift)
                + " - re-verify and re-pin compat.py before transplant work."
            )

        leaks: list[str] = []
        combos = 0
        for (bank_id, model_id, pipeline_id), (pack, _p) in registry.packs.items():
            if bank_id == "custom_source_bank":
                continue
            spec = build_spec(
                registry, source_bank_id=bank_id, story_model_id=model_id,
                story_pipeline_id=pipeline_id,
            )
            combos += 1
            if bank_id != "science_news":
                found = scan_story_leakage(registry, spec)
                if found:
                    leaks.append(f"{bank_id}/{model_id}: {found}")
            visual_found = scan_visual_leakage(spec)
            if visual_found:
                leaks.append(f"{bank_id}/{model_id} visual: {visual_found}")
            assert render_prompt_preview(spec)
        if leaks:
            raise RuntimeError(f"Leakage detected: {leaks}")

        manifests = registry.public_domain_manifests()
        lines = [
            "OK Upstream Story Lab v2 (transplant workspace)",
            f"lab_root={LAB_ROOT}",
            f"banks={sorted(registry.banks)}",
            f"story_packs={len(registry.packs)}",
            f"visual_styles={sorted(registry.styles)}",
            f"pipelines={sorted(registry.pipelines)}",
            f"validated_specs={combos}",
            f"public_domain_manifests={len(manifests)}",
            "mirror_drift=none",
            "production_workflow=not touched by this lab node",
        ]
        return ("\n".join(lines),)


class OTR_StoryPackPreview:
    """Preview one explicit selection; every choice is registry-discovered."""

    CATEGORY = "OTR/Upstream Story Lab"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preview_json",)
    FUNCTION = "preview"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "source_bank_id": (_bank_choices(), {"default": "media_archive"}),
                "story_model_id": (_model_choices(), {"default": "auto"}),
                "story_pipeline_id": (_pipeline_choices(), {"default": "auto"}),
                "visual_style_id": (_style_choices(), {"default": "auto"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        return _lab_state_digest()

    def preview(self, source_bank_id: str, story_model_id: str,
                story_pipeline_id: str, visual_style_id: str) -> tuple[str]:
        _ensure_lab_importable()
        from upstream_story_lab.bridge import build_bridge_artifact, build_spec
        from upstream_story_lab.preview import (
            render_prompt_preview,
            render_visual_preview,
        )

        registry = _registry()
        spec = build_spec(
            registry,
            source_bank_id=source_bank_id,
            story_model_id=story_model_id,
            story_pipeline_id=story_pipeline_id,
            visual_style_id=visual_style_id,
        )
        artifact = build_bridge_artifact(spec)
        result = {
            "status": "ok",
            "lab_root": str(LAB_ROOT),
            "resolution": spec.resolution.model_dump(mode="json"),
            "provenance": spec.provenance.model_dump(mode="json"),
            "prompt_preview": render_prompt_preview(spec),
            "visual_preview": render_visual_preview(spec),
            "bridge_artifact": artifact.model_dump(mode="json"),
            "notes": [
                "Live upstream lab preview only; the production workflow is untouched.",
                "Invalid source/story/style/pipeline combinations fail loudly.",
            ],
        }
        return (json.dumps(result, indent=2, ensure_ascii=False),)


class OTR_BridgeArtifactEmit:
    """Explicitly write the frozen bridge artifact JSON to bridge_out/."""

    CATEGORY = "OTR/Upstream Story Lab"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artifact_path",)
    FUNCTION = "emit"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "source_bank_id": (_bank_choices(), {"default": "media_archive"}),
                "story_model_id": (_model_choices(), {"default": "auto"}),
                "story_pipeline_id": (_pipeline_choices(), {"default": "auto"}),
                "visual_style_id": (_style_choices(), {"default": "auto"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> str:
        return _lab_state_digest()

    def emit(self, source_bank_id: str, story_model_id: str,
             story_pipeline_id: str, visual_style_id: str) -> tuple[str]:
        _ensure_lab_importable()
        from upstream_story_lab.bridge import (
            build_bridge_artifact,
            build_spec,
            emit_bridge_artifact,
        )

        registry = _registry()
        spec = build_spec(
            registry,
            source_bank_id=source_bank_id,
            story_model_id=story_model_id,
            story_pipeline_id=story_pipeline_id,
            visual_style_id=visual_style_id,
        )
        artifact = build_bridge_artifact(spec)
        # Filename carries ALL resolved axes + a provenance hash so two
        # different selections can never silently overwrite each other
        # (kibitz r3, Codex M2).
        from upstream_story_lab.compat import canonical_json_hash

        content_hash = canonical_json_hash(
            artifact.ledger_writing_spec.model_dump(mode="json")
        )[:8]
        out = LAB_ROOT / "bridge_out" / (
            f"bridge_{spec.source_bank_id}_{spec.story_model_id}_"
            f"{spec.story_pipeline_id}_{spec.visual_style_id}_{content_hash}.json"
        )
        path = emit_bridge_artifact(artifact, out)
        return (str(path),)


NODE_CLASS_MAPPINGS = {
    "OTR_UpstreamStoryLabValidator": OTR_UpstreamStoryLabValidator,
    "OTR_StoryPackPreview": OTR_StoryPackPreview,
    "OTR_BridgeArtifactEmit": OTR_BridgeArtifactEmit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_UpstreamStoryLabValidator": "OTR Upstream Story Lab Validator (v2)",
    "OTR_StoryPackPreview": "OTR Story Pack Preview (v2)",
    "OTR_BridgeArtifactEmit": "OTR Bridge Artifact Emit (v2)",
}
