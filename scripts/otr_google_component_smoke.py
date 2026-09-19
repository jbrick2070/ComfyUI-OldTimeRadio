"""Catalog-only Google component smoke. No generate. Writes a text report.

    python scripts/otr_google_component_smoke.py

Lists every Google BYO component the validator will check (writer pointers,
Veo, Omni, image, TTS, Lyria), pings models.list once, and writes
docs/2026-09-18-google-component-smoke.txt. Use that file to see which
slugs are live BEFORE a paid episode run.

Exit 0 = every concrete slug was in the catalog.
Exit 2 = at least one shipped default is missing.
Exit 3 = no key.
Exit 4 = catalog transport/auth failed.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "docs" / "2026-09-18-google-component-smoke.txt"

sys.path.insert(0, str(REPO))

COMPONENTS = (
    ("writer.creative_pointer", "gemini-flash-latest"),
    ("writer.technical_pointer", "gemini-flash-lite-latest"),
    ("writer.pro_pointer", "gemini-pro-latest"),
    ("google_veo_video", "veo-3.1-lite-generate-preview"),
    ("google_omni_video", "gemini-omni-flash-preview"),
    ("google_image", "gemini-3.1-flash-image"),
    ("google_tts", "gemini-2.5-flash-preview-tts"),
    ("google_lyria", "lyria-3-clip-preview"),
)


def _normalize(slug: str) -> str:
    from nodes._otr_shared.google_slug_verifier import normalize_model_name
    return normalize_model_name(slug) or str(slug).strip()


def main() -> int:
    lines = [
        "OTR Google component smoke",
        "catalog only -- no generate, no episode",
        "when: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "",
    ]
    from nodes._otr_google_api.client import get_json, resolve_api_key
    from nodes._otr_shared.google_slug_verifier import fetch_catalog

    try:
        key = resolve_api_key()
    except Exception as exc:  # noqa: BLE001 -- report, do not hide
        lines.extend([
            "KEY: missing (%s)" % type(exc).__name__,
            str(exc),
            "",
            "Put the key in google.secret or google_api_key.location,",
            "or set OTR_GOOGLE_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY.",
            "Then re-run this script. Do not start a full Google episode",
            "until KEY is present and the rows below say LIVE.",
            "",
        ])
        for name, slug in COMPONENTS:
            lines.append("%-28s  %s  NOT CHECKED" % (name, slug))
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(REPORT)
        return 3

    lines.append("KEY: present (not printed)")
    try:
        fetched = fetch_catalog(lambda path: get_json(path, _api_key=key))
    except Exception as exc:  # noqa: BLE001
        lines.extend([
            "CATALOG: failed (%s)" % type(exc).__name__,
            str(exc),
            "",
        ])
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(REPORT)
        return 4

    if not fetched.complete:
        lines.extend([
            "CATALOG: incomplete -- %s" % (fetched.error or "no terminal page"),
            "pages=%s" % fetched.pages,
            "",
        ])
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(REPORT)
        return 4

    ids = fetched.ids
    lines.append("CATALOG: %d ids, %d pages, host=generativelanguage.googleapis.com"
                 % (len(ids), fetched.pages))
    lines.append("validator treats a miss here as a stale pin (same as Comfy T1).")
    lines.append("")

    missing = []
    for name, slug in COMPONENTS:
        bare = _normalize(slug)
        if bare in ids:
            lines.append("%-28s  %s  LIVE" % (name, bare))
        else:
            missing.append(bare)
            lines.append("%-28s  %s  MISSING -- stale vs live catalog" % (name, bare))

    lines.append("")
    if missing:
        lines.append("RESULT: FAIL -- %d stale/missing slug(s)" % len(missing))
        lines.append("Queue-time validator will refuse these before spend.")
        code = 2
    else:
        lines.append("RESULT: PASS -- every listed component slug is in the catalog.")
        lines.append("Model-to-model congruity holds for the shipped defaults.")
        code = 0

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    print(lines[-2])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
