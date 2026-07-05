"""Pinned production compatibility shapes + AST drift-proofing.

Every pin cites its production_mirror source. Drift tests EXTRACT the same
shape from the mirrored file by AST (never by importing it - the mirror is
deliberately dependency-incomplete) and compare against the pin. When
production moves and the mirror is refreshed, any shape change fails a test
with a pointed re-pin message instead of silently rotting.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: meta.news == NewsBriefs.model_dump() - all 13 fields, exactly.
#: Source: production_mirror/nodes/news_interpreter.py (class NewsBriefs).
NEWS_BRIEFS_FIELDS = (
    "casting_brief",
    "script_brief",
    "news_close_brief",
    "key_terms",
    "source_hash",
    "source_chars",
    "prompt_version",
    "schema_version",
    "model_id",
    "decoder_profile",
    "seed",
    "attempts",
    "attempt_failures",
)

#: meta.news_seed dict keys (current ledger shape).
#: Source: production_mirror/nodes/_otr_legacy_to_stage1_adapter.py
#: (_coerce_news_seed_text docstring). REQUIRED subset = the keys the
#: adapter actually reads; the rest are optional for mirror emission.
NEWS_SEED_KEYS = (
    "headline",
    "source",
    "url",
    "date",
    "body_chars",
    "style",
    "selected_at",
)
NEWS_SEED_REQUIRED_KEYS = ("headline", "source")

#: news_article dict keys (writer fetch return / news_used socket shape).
#: Source: production_mirror/nodes/OTR_LedgerScriptWriter.py
#: (_fetch_rss_seed_or_die docstring return-shape contract).
NEWS_ARTICLE_KEYS = (
    "headline",
    "summary",
    "full_text",
    "source",
    "date",
    "link",
    "seed_text",
)

#: Production motion-prompt role vocabulary.
#: Source: production_mirror/nodes/_otr_video_engines/render_driver.py
#: (_LTX_MOTION_PROMPT_BY_ROLE keys).
MOTION_ROLE_KEYS = ("announcer", "music_open", "music_close", "music_inter")

#: Genre-neutral production visual tail constants (NOT sci-fi-specific; the
#: sci_fi_radio policy must REPRODUCE them byte-identically).
#: Source: production_mirror/nodes/_otr_story_brief_helpers.py.
PRODUCTION_VISUAL_TAILS = {
    "ERA_TAIL_DEFAULT": "timeless cinematic aesthetic",
    "STYLE_TAIL_DEFAULT": (
        "cinematic, 35mm film look, subtle film grain, volumetric lighting"
    ),
    "IMAGE_GRADE_TAIL": (
        "anamorphic lens, heavy vignette, muted color grade, sharp focus"
    ),
    "RADIO_BROADCAST_TAIL": (
        "35mm film grain, broadcast-distressed cinematic aesthetic, "
        "centered composition"
    ),
}


class CompatDriftError(AssertionError):
    """A pinned production shape no longer matches the mirror. Re-verify the
    shape against the refreshed production_mirror and re-pin compat.py."""


def canonical_json_hash(data: Any) -> str:
    """sha256 over canonical JSON (sort_keys, tight separators). Whitespace
    and key order never change the hash (kibitz r1 optional, accepted)."""

    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(path: Path) -> ast.Module:
    if not path.exists():
        raise CompatDriftError(
            f"mirror file missing: {path} - refresh production_mirror before "
            "running drift checks"
        )
    return ast.parse(path.read_text(encoding="utf-8"))


def extract_news_briefs_fields(news_interpreter_py: Path) -> list[str]:
    """AST-extract NewsBriefs annotated field names (order preserved)."""

    tree = _parse(news_interpreter_py)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NewsBriefs":
            fields = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
            if not fields:
                raise CompatDriftError(
                    "NewsBriefs class found but no annotated fields extracted - "
                    "re-pin NEWS_BRIEFS_FIELDS against the mirror"
                )
            return fields
    raise CompatDriftError(
        f"class NewsBriefs not found in {news_interpreter_py} - re-pin against "
        "the refreshed mirror"
    )


def extract_news_seed_keys(adapter_py: Path) -> list[str]:
    """Extract the documented news_seed dict keys from the adapter docstring.

    The shape lives in prose (a docstring contract), so this is a regex over
    the documented dict literal, not code AST - still read-only, never an
    import."""

    text = adapter_py.read_text(encoding="utf-8") if adapter_py.exists() else ""
    if not text:
        raise CompatDriftError(f"mirror file missing or empty: {adapter_py}")
    match = re.search(r"meta\.news_seed\s*=\s*\{(.*?)\}", text, flags=re.DOTALL)
    if not match:
        raise CompatDriftError(
            "documented news_seed dict not found in adapter docstring - re-pin "
            "NEWS_SEED_KEYS against the refreshed mirror"
        )
    keys = re.findall(r'"([a-z_]+)"\s*:', match.group(1))
    if not keys:
        raise CompatDriftError("news_seed docstring matched but no keys parsed")
    return keys


def extract_motion_role_keys(render_driver_py: Path) -> list[str]:
    """AST-extract _LTX_MOTION_PROMPT_BY_ROLE key strings."""

    tree = _parse(render_driver_py)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_LTX_MOTION_PROMPT_BY_ROLE" in targets and isinstance(
                node.value, ast.Dict
            ):
                keys = [
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                ]
                if not keys:
                    raise CompatDriftError(
                        "_LTX_MOTION_PROMPT_BY_ROLE found but empty - re-pin"
                    )
                return keys
    raise CompatDriftError(
        f"_LTX_MOTION_PROMPT_BY_ROLE not found in {render_driver_py} - re-pin "
        "MOTION_ROLE_KEYS against the refreshed mirror"
    )


def extract_visual_tails(brief_helpers_py: Path) -> dict[str, str]:
    """AST-extract the four production tail constants (string concatenations
    folded)."""

    tree = _parse(brief_helpers_py)
    wanted = set(PRODUCTION_VISUAL_TAILS)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError) as exc:
                        raise CompatDriftError(
                            f"{target.id} is no longer a literal string - re-pin"
                        ) from exc
                    if isinstance(value, str):
                        out[target.id] = value
    missing = sorted(wanted - set(out))
    if missing:
        raise CompatDriftError(
            f"tail constants not found in mirror: {missing} - re-pin "
            "PRODUCTION_VISUAL_TAILS against the refreshed mirror"
        )
    return out
