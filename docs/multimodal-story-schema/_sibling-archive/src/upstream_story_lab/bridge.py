"""Translator head: assemble a validated LedgerWritingSpec and emit the one
frozen bridge artifact production will consume at transplant time.

Refuses to emit anything partial. Mirror shapes are pinned in compat.py and
drift-tested; the bridge maps StoryInputPacket.close_brief ->
meta.news.news_close_brief explicitly (kibitz r2: production field name wins
at the mirror boundary).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .compat import (
    NEWS_ARTICLE_KEYS,
    NEWS_BRIEFS_FIELDS,
    NEWS_SEED_KEYS,
    canonical_json_hash,
    file_sha256,
)
from .contracts import (
    BridgeArtifact,
    LedgerWritingSpec,
    MetaMirrors,
    Provenance,
    SourceMaterialPacket,
)
from .interpreters import resolve_interpreter
from .profiles import resolve_profile
from .registry import Registry


class BridgeError(ValueError):
    """Fail-loud bridge assembly/emission problem."""


def _read_production_baseline(root: Path) -> str:
    """Read the mirrored production commit hash from
    PRODUCTION_MIRROR_MANIFEST.md so provenance.production_baseline is LIVE
    data, never hand-typed (kibitz r3: makes the field real; note the
    provenance covers fixture state + this baseline hash - mirror FILE
    contents are a validation-time concern covered by the drift tests and
    the ComfyUI digest, intentionally not re-hashed here)."""

    manifest = Path(root) / "PRODUCTION_MIRROR_MANIFEST.md"
    if not manifest.exists():
        raise BridgeError(
            f"PRODUCTION_MIRROR_MANIFEST.md missing at {manifest} - cannot "
            "stamp production_baseline provenance"
        )
    match = re.search(r"commit\s+([0-9a-f]{40})", manifest.read_text(encoding="utf-8"))
    if not match:
        raise BridgeError(
            "no production commit hash found in PRODUCTION_MIRROR_MANIFEST.md"
        )
    return match.group(1)


def build_spec(registry: Registry, *, source_bank_id: str,
               story_model_id: str = "auto", story_pipeline_id: str = "auto",
               visual_style_id: str = "auto",
               packet: SourceMaterialPacket | None = None,
               production_baseline: str = "") -> LedgerWritingSpec:
    """Resolve ids, interpret source, and assemble a cross-validated spec."""

    resolution = registry.resolve(
        source_bank_id=source_bank_id,
        story_model_id=story_model_id,
        story_pipeline_id=story_pipeline_id,
        visual_style_id=visual_style_id,
    )
    bank_id = resolution.resolved("source_bank")
    model_id = resolution.resolved("story_model")
    pipeline_id = resolution.resolved("story_pipeline")
    style_id = resolution.resolved("visual_style")

    bank = registry.bank(bank_id)
    if packet is None:
        packet, packet_path = registry.source_packet(bank_id)
    else:
        packet_path = None
    interpreter = resolve_interpreter(bank)
    story_input = interpreter(packet, bank, None, story_model_id=model_id)
    profile = resolve_profile(registry, bank_id, model_id, pipeline_id)
    policy = registry.style(style_id)
    pipeline = registry.pipeline(pipeline_id)

    provenance = Provenance(
        production_baseline=(
            production_baseline or _read_production_baseline(registry.root)
        ),
        lab_state_digest=registry.state_digest(),
        pack_sha256=file_sha256(registry.pack_path(bank_id, model_id, pipeline_id)),
        style_sha256=file_sha256(registry.style_path(style_id)),
        packet_sha256=(
            file_sha256(packet_path) if packet_path is not None
            else canonical_json_hash(packet.model_dump(mode="json"))
        ),
        **registry.fixture_hashes(),
    )

    return LedgerWritingSpec(
        source_bank_id=bank_id,
        story_model_id=model_id,
        story_pipeline_id=pipeline_id,
        visual_style_id=policy.style_id,
        source_material=packet,
        story_input=story_input,
        prompt_profile=profile,
        visual_policy=policy,
        slot_plan=list(pipeline.passes),
        resolution=resolution,
        provenance=provenance,
    )


def build_meta_mirrors(spec: LedgerWritingSpec, *, created_utc: str) -> MetaMirrors:
    """Emit the two production meta shapes, complete (all keys, right types).

    meta.news carries ALL 13 NewsBriefs fields; python-stamped fields get
    bridge-provenanced values. meta.news_seed carries all 7 documented keys
    (required consumer subset is headline+source; we emit everything)."""

    material = spec.source_material
    story = spec.story_input
    news = {
        "casting_brief": story.casting_brief,
        "script_brief": story.script_brief,
        "news_close_brief": story.close_brief,  # explicit mapping (kibitz r2)
        "key_terms": list(story.key_terms),
        "source_hash": material.source_hash,
        "source_chars": len(material.raw_text or material.source_summary),
        "prompt_version": "bridge-v2",
        "schema_version": "bridge-v2",
        "model_id": "",
        "decoder_profile": "",
        "seed": 0,
        "attempts": 0,
        "attempt_failures": [],
    }
    if set(news) != set(NEWS_BRIEFS_FIELDS):
        raise BridgeError(
            "meta.news mirror keys drifted from pinned NewsBriefs fields: "
            f"{sorted(set(news) ^ set(NEWS_BRIEFS_FIELDS))}"
        )
    if not isinstance(news["key_terms"], list):
        raise BridgeError("meta.news.key_terms must be a list (freeze invariant)")

    news_seed = {
        "headline": material.source_title,
        "source": material.source_label or material.source_author or "bridge",
        "url": material.source_url,
        "date": "",
        "body_chars": len(material.raw_text),
        "style": "",
        "selected_at": created_utc,
    }
    if set(news_seed) != set(NEWS_SEED_KEYS):
        raise BridgeError(
            "meta.news_seed mirror keys drifted from pinned shape: "
            f"{sorted(set(news_seed) ^ set(NEWS_SEED_KEYS))}"
        )
    return MetaMirrors(news=news, news_seed=news_seed)


def build_adapter_news_article(spec: LedgerWritingSpec) -> dict:
    """The writer-internal article dict (custom_premise-branch shape,
    writer :1338-1346): how packet-driven lanes enter production."""

    material = spec.source_material
    seed_text = " ".join(
        p for p in (material.source_title, material.source_summary) if p
    ).strip() or material.raw_text.strip()
    if not seed_text:
        raise BridgeError(
            "cannot build adapter article: packet has no title/summary/raw_text"
        )
    article = {
        "headline": material.source_title,
        "summary": material.source_summary,
        "full_text": material.raw_text or material.source_summary,
        "source": material.source_label or "Bridge Packet",
        "date": "",
        "link": material.source_url,
        "seed_text": seed_text,
    }
    if set(article) != set(NEWS_ARTICLE_KEYS):
        raise BridgeError("adapter article keys drifted from pinned shape")
    return article


def build_bridge_artifact(spec: LedgerWritingSpec) -> BridgeArtifact:
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return BridgeArtifact(
        created_utc=created,
        ledger_writing_spec=spec,
        meta_mirrors=build_meta_mirrors(spec, created_utc=created),
        adapter_news_article=build_adapter_news_article(spec),
    )


def emit_bridge_artifact(artifact: BridgeArtifact, path: Path) -> Path:
    """Write the artifact JSON: UTF-8, no BOM, LF, trailing newline."""

    payload = json.dumps(
        artifact.model_dump(mode="json"), indent=2, ensure_ascii=False
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload + "\n")
    # Round-trip validation: the emitted file must re-validate or be deleted.
    try:
        BridgeArtifact(**json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise BridgeError(f"emitted bridge artifact failed round-trip: {exc}") from exc
    return path
