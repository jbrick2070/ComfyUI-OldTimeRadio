"""Fixture registry: load, validate, route. Zero content literals.

Everything here is behavior. All story/visual/config content lives under
fixtures/. Missing, duplicate, unknown, or malformed content is a hard error
naming the offending file - never a fallback.
"""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from pathlib import PurePosixPath

from .compat import file_sha256
from .contracts import (
    PipelineSpec,
    PublicDomainSourceManifest,
    Resolution,
    ResolutionDecision,
    SourceBankSpec,
    SourceMaterialPacket,
    StoryPack,
    VisualStylePolicy,
    allowed_seam_variables,
)


class RegistryError(ValueError):
    """Fail-loud registry problem (missing/duplicate/unknown/malformed)."""


class UnknownIdError(RegistryError):
    """An id was requested that no fixture declares."""


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise RegistryError(f"required fixture missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryError(f"malformed JSON in {path}: {exc}") from exc


def _validate_template(owner: str, seam: str, template: str) -> None:
    """Per-seam variable allowlist (kibitz r2: labels everywhere + declared
    runtime variables for the seams whose production templates carry them)."""

    allowed = allowed_seam_variables(seam)
    try:
        fields = [
            f for _, f, _, _ in string.Formatter().parse(template) if f is not None
        ]
    except ValueError as exc:
        raise RegistryError(f"{owner}: seam {seam!r} template unparsable: {exc}") from exc
    unknown = sorted({f for f in fields if f.split(".")[0].split("[")[0] not in allowed})
    if unknown:
        raise RegistryError(
            f"{owner}: seam {seam!r} references undeclared variables {unknown}; "
            f"allowed: {sorted(allowed)}"
        )


class Registry:
    """One loaded, validated view of fixtures/. Constructable uncached for
    tests; nodes.py may cache an instance behind the state digest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.fixtures = self.root / "fixtures"
        if not self.fixtures.exists():
            raise RegistryError(f"fixtures folder missing: {self.fixtures}")
        self.banks = self._load_banks()
        self.pipelines = self._load_pipelines()
        self.packs = self._load_packs()
        self.styles = self._load_styles()
        self._cross_validate()

    # -- loading ---------------------------------------------------------

    def _load_banks(self) -> dict[str, SourceBankSpec]:
        data = _load_json(self.fixtures / "banks.json")
        entries = data.get("banks")
        if not isinstance(entries, list) or not entries:
            raise RegistryError("banks.json must contain a non-empty 'banks' list")
        banks: dict[str, SourceBankSpec] = {}
        for entry in entries:
            bank = SourceBankSpec(**entry)
            if bank.source_bank_id in banks:
                raise RegistryError(f"duplicate bank id {bank.source_bank_id!r}")
            banks[bank.source_bank_id] = bank
        return banks

    def _load_pipelines(self) -> dict[str, PipelineSpec]:
        data = _load_json(self.fixtures / "pipelines.json")
        entries = data.get("pipelines")
        if not isinstance(entries, list) or not entries:
            raise RegistryError("pipelines.json must contain a non-empty 'pipelines' list")
        pipelines: dict[str, PipelineSpec] = {}
        for entry in entries:
            spec = PipelineSpec(**entry)
            if spec.story_pipeline_id in pipelines:
                raise RegistryError(f"duplicate pipeline id {spec.story_pipeline_id!r}")
            pipelines[spec.story_pipeline_id] = spec
        return pipelines

    def _load_packs(self) -> dict[tuple[str, str, str], tuple[StoryPack, Path]]:
        pack_dir = self.fixtures / "story_packs"
        if not pack_dir.exists():
            raise RegistryError(f"story pack folder missing: {pack_dir}")
        paths = sorted(pack_dir.rglob("*.json"))
        if not paths:
            raise RegistryError(f"no story packs found under {pack_dir}")
        packs: dict[tuple[str, str, str], tuple[StoryPack, Path]] = {}
        for path in paths:
            pack = StoryPack(**_load_json(path))
            key = (pack.source_bank_id, pack.story_model_id, pack.story_pipeline_id)
            if key in packs:
                raise RegistryError(f"duplicate story pack key {key!r}: {path}")
            for seam, template in pack.prompt_stages.items():
                _validate_template(str(path), seam, template)
            packs[key] = (pack, path)
        return packs

    def _load_styles(self) -> dict[str, tuple[VisualStylePolicy, Path]]:
        style_dir = self.fixtures / "visual_styles"
        if not style_dir.exists():
            raise RegistryError(f"visual style folder missing: {style_dir}")
        paths = sorted(style_dir.glob("*.json"))
        if not paths:
            raise RegistryError(f"no visual styles found under {style_dir}")
        styles: dict[str, tuple[VisualStylePolicy, Path]] = {}
        for path in paths:
            policy = VisualStylePolicy(**_load_json(path))
            if policy.style_id in styles:
                raise RegistryError(f"duplicate visual style id {policy.style_id!r}: {path}")
            styles[policy.style_id] = (policy, path)
        return styles

    def _cross_validate(self) -> None:
        for bank in self.banks.values():
            if bank.default_story_model != "" and bank.runnable:
                key_prefix = (bank.source_bank_id, bank.default_story_model)
                if not any(k[:2] == key_prefix for k in self.packs):
                    raise RegistryError(
                        f"bank {bank.source_bank_id!r} default_story_model "
                        f"{bank.default_story_model!r} has no story pack"
                    )
            if bank.default_visual_style and bank.default_visual_style not in self.styles:
                raise RegistryError(
                    f"bank {bank.source_bank_id!r} default_visual_style "
                    f"{bank.default_visual_style!r} not a known style"
                )
            if bank.default_story_pipeline not in self.pipelines:
                raise RegistryError(
                    f"bank {bank.source_bank_id!r} default_story_pipeline "
                    f"{bank.default_story_pipeline!r} not a known pipeline"
                )
        for (bank_id, model_id, pipeline_id), (pack, path) in self.packs.items():
            if bank_id not in self.banks:
                raise RegistryError(f"pack {path} names unknown bank {bank_id!r}")
            if pipeline_id not in self.pipelines:
                raise RegistryError(f"pack {path} names unknown pipeline {pipeline_id!r}")
            bank = self.banks[bank_id]
            if pack.status != "experimental":
                missing = [
                    seam for seam in bank.required_seams
                    if not pack.prompt_stages.get(seam, "").strip()
                ]
                if missing:
                    raise RegistryError(
                        f"pack {path} missing required seams for bank "
                        f"{bank_id!r}: {missing} (no default prose is invented)"
                    )
            # Executable pipelines must have FULL seam coverage in the pack
            # (roundtable pass01, DeepSeek: pipelines and packs were validated
            # atomically, never in combination - a missing seam only surfaced
            # inside the runner). Descriptive pipelines are exempt: an absent
            # seam there means "production constant stays", by contract.
            pipeline = self.pipelines[pipeline_id]
            if pipeline.executable_in_lab:
                needed = {
                    seam for p in pipeline.passes for seam in p.seam_refs
                }
                uncovered = sorted(
                    seam for seam in needed
                    if not pack.prompt_stages.get(seam, "").strip()
                )
                if uncovered:
                    raise RegistryError(
                        f"pack {path} does not cover executable pipeline "
                        f"{pipeline_id!r} seams: {uncovered}"
                    )

    # -- routing ---------------------------------------------------------

    def bank(self, source_bank_id: str) -> SourceBankSpec:
        try:
            return self.banks[(source_bank_id or "").strip()]
        except KeyError as exc:
            raise UnknownIdError(
                f"unknown source_bank_id {source_bank_id!r}; known: {sorted(self.banks)}"
            ) from exc

    def pack(self, source_bank_id: str, story_model_id: str,
             story_pipeline_id: str) -> StoryPack:
        key = (source_bank_id, story_model_id, story_pipeline_id)
        try:
            return self.packs[key][0]
        except KeyError as exc:
            raise UnknownIdError(
                f"no story pack for {key!r}; known: {sorted(self.packs)}"
            ) from exc

    def pack_path(self, source_bank_id: str, story_model_id: str,
                  story_pipeline_id: str) -> Path:
        key = (source_bank_id, story_model_id, story_pipeline_id)
        try:
            return self.packs[key][1]
        except KeyError as exc:
            raise UnknownIdError(f"no story pack for {key!r}") from exc

    def style(self, style_id: str) -> VisualStylePolicy:
        try:
            return self.styles[(style_id or "").strip()][0]
        except KeyError as exc:
            raise UnknownIdError(
                f"unknown visual style id {style_id!r}; known: {sorted(self.styles)}"
            ) from exc

    def style_path(self, style_id: str) -> Path:
        return self.styles[style_id][1]

    def pipeline(self, story_pipeline_id: str) -> PipelineSpec:
        try:
            return self.pipelines[(story_pipeline_id or "").strip()]
        except KeyError as exc:
            raise UnknownIdError(
                f"unknown story_pipeline_id {story_pipeline_id!r}; "
                f"known: {sorted(self.pipelines)}"
            ) from exc

    def resolve(self, *, source_bank_id: str, story_model_id: str = "auto",
                story_pipeline_id: str = "auto",
                visual_style_id: str = "auto") -> Resolution:
        """Resolve every axis; each decision is recorded, never invisible."""

        bank = self.bank(source_bank_id)
        decisions = [
            ResolutionDecision(
                axis="source_bank", requested=source_bank_id,
                resolved=bank.source_bank_id, default_applied=False,
                decided_by="explicit",
            )
        ]

        requested_model = (story_model_id or "auto").strip()
        if requested_model == "auto":
            resolved_model = bank.default_story_model
            decided = "banks.json:default_story_model"
            if not resolved_model:
                raise UnknownIdError(
                    f"bank {bank.source_bank_id!r} declares no default_story_model "
                    "and story_model_id was 'auto'"
                )
        else:
            resolved_model, decided = requested_model, "explicit"
        decisions.append(ResolutionDecision(
            axis="story_model", requested=requested_model,
            resolved=resolved_model, default_applied=requested_model == "auto",
            decided_by=decided,
        ))

        requested_pipeline = (story_pipeline_id or "auto").strip()
        if requested_pipeline == "auto":
            resolved_pipeline, decided = bank.default_story_pipeline, "banks.json:default_story_pipeline"
        else:
            resolved_pipeline, decided = requested_pipeline, "explicit"
        decisions.append(ResolutionDecision(
            axis="story_pipeline", requested=requested_pipeline,
            resolved=resolved_pipeline,
            default_applied=requested_pipeline == "auto", decided_by=decided,
        ))

        requested_style = (visual_style_id or "auto").strip()
        if requested_style == "auto":
            resolved_style, decided = bank.default_visual_style, "banks.json:default_visual_style"
            if not resolved_style:
                raise UnknownIdError(
                    f"bank {bank.source_bank_id!r} declares no default_visual_style "
                    "and visual_style_id was 'auto'"
                )
        else:
            resolved_style, decided = requested_style, "explicit"
        decisions.append(ResolutionDecision(
            axis="visual_style", requested=requested_style,
            resolved=resolved_style, default_applied=requested_style == "auto",
            decided_by=decided,
        ))

        # Existence checks: resolution never returns ids nothing declares.
        self.pack(bank.source_bank_id, resolved_model, resolved_pipeline)
        self.style(resolved_style)
        self.pipeline(resolved_pipeline)
        return Resolution(decisions=decisions)

    # -- source packets / PD manifests ------------------------------------

    def source_packet(self, source_bank_id: str) -> tuple[SourceMaterialPacket, Path]:
        bank = self.bank(source_bank_id)
        if not bank.runnable:
            raise RegistryError(
                f"source bank {bank.source_bank_id!r} is visible but not "
                f"runnable yet. {bank.guide_ref}".strip()
            )
        packet_dir = self.fixtures / "source_packets"
        path = packet_dir / f"{source_bank_id}.json"
        if not path.exists():
            raise RegistryError(
                f"no source packet fixture for bank {source_bank_id!r}: {path}"
            )
        packet = SourceMaterialPacket(**_load_json(path))
        if packet.source_bank_id != source_bank_id:
            raise RegistryError(
                f"source packet {path} declares bank {packet.source_bank_id!r}, "
                f"expected {source_bank_id!r}"
            )
        return packet, path

    def public_domain_manifests(self) -> list[tuple[PublicDomainSourceManifest, Path]]:
        root = self.fixtures / "public_domain_sources"
        if not root.exists():
            raise RegistryError(f"public-domain source folder missing: {root}")
        out: list[tuple[PublicDomainSourceManifest, Path]] = []
        for path in sorted(root.glob("*/manifest.json")):
            manifest = PublicDomainSourceManifest(**_load_json(path))
            base = path.parent
            base_resolved = base.resolve()
            for rel in manifest.text_files + manifest.image_files:
                rel_path = PurePosixPath(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts:
                    raise RegistryError(
                        f"public-domain manifest {path} uses unsafe path {rel!r}"
                    )
                target = (base / str(rel_path))
                if not target.exists():
                    raise RegistryError(
                        f"public-domain manifest {path} references missing file {rel!r}"
                    )
                # kibitz r2 (Codex): resolve() containment - a symlink may not
                # escape the manifest folder.
                resolved = target.resolve()
                if base_resolved not in resolved.parents and resolved != base_resolved:
                    raise RegistryError(
                        f"public-domain manifest {path}: {rel!r} resolves outside "
                        f"the manifest folder ({resolved})"
                    )
            out.append((manifest, path))
        if not out:
            raise RegistryError(f"no public-domain manifests found under {root}")
        return out

    # -- state digest ------------------------------------------------------

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.fixtures.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            stat = path.stat()
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        return digest.hexdigest()

    def fixture_hashes(self) -> dict[str, str]:
        return {
            "banks_sha256": file_sha256(self.fixtures / "banks.json"),
            "pipelines_sha256": file_sha256(self.fixtures / "pipelines.json"),
        }
