"""Sprint D D0b license audit framework.

Schema-positive gate over per-row markdown audit files.

Inputs:
    docs/model-license-audit-targets.txt -- one repo_id per line,
        comments (#) and blank lines ignored.

For each repo_id in the targets file:
    sanitize the repo_id (lowercase, `/` -> `--`, nothing else),
    look up docs/model-license-<sanitized>.md,
    parse the YAML-style frontmatter (--- delimited),
    validate the 4 required keys plus matching repo_id.

Stdlib-only on purpose. Mistral / Gemma / Qwen / future variants must
all flow through the same gate without growing a yaml dependency.

CLI:
    python tools/audit_model_license.py                 # validate all rows
    python tools/audit_model_license.py --check         # validate, exit 2 on fail
    python tools/audit_model_license.py --list-targets  # echo target list
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS_FILE = REPO_ROOT / "docs" / "model-license-audit-targets.txt"
AUDIT_DIR = REPO_ROOT / "docs"
AUDIT_FILENAME_TEMPLATE = "model-license-{sanitized}.md"

REQUIRED_KEYS: tuple[str, ...] = (
    "repo_id",
    "license",
    "license_audit_status",
    "verdict_date",
)
ALLOWED_LICENSE_VALUES: frozenset[str] = frozenset({
    "mit",
    "apache_2_0",
    "non_commercial",
    "community",
    "gated_terms",
})
ALLOWED_STATUS_VALUES: frozenset[str] = frozenset({
    "mit_equivalent",
    "research_lane",
    "pending",
})


@dataclass(frozen=True)
class AuditRecord:
    repo_id: str
    sanitized: str
    path: Path
    frontmatter: dict[str, str]


class LicenseAuditError(Exception):
    """Raised when an audit row fails validation."""


def sanitize_repo_id(repo_id: str) -> str:
    """Lowercase the repo_id and replace '/' with '--'.

    Strips nothing else. The original repo_id is recoverable by
    reversing the '--' split on the first occurrence and re-capitalizing
    via lookup against the targets list (not via heuristic).
    """
    return repo_id.strip().lower().replace("/", "--")


def read_targets(targets_file: Path = TARGETS_FILE) -> list[str]:
    """Read non-comment non-blank lines from the targets file."""
    if not targets_file.is_file():
        raise LicenseAuditError(
            f"audit targets file not found: {targets_file}"
        )
    out: list[str] = []
    for raw in targets_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the YAML-style frontmatter at the head of a markdown file.

    Expected shape:
        ---
        key: value
        key2: value2
        ---
        <body>

    Returns the parsed key/value dict. Values are stripped strings.
    Raises LicenseAuditError if the frontmatter is missing or malformed.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise LicenseAuditError("missing opening --- frontmatter delimiter")
    fm: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped == "---":
            return fm
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise LicenseAuditError(
                f"frontmatter line {i} has no ':' separator: {stripped!r}"
            )
        key, _, value = stripped.partition(":")
        fm[key.strip()] = value.strip()
    raise LicenseAuditError("missing closing --- frontmatter delimiter")


def load_audit_record(repo_id: str) -> AuditRecord:
    """Load and validate a single audit file for one repo_id."""
    sanitized = sanitize_repo_id(repo_id)
    path = AUDIT_DIR / AUDIT_FILENAME_TEMPLATE.format(sanitized=sanitized)
    if not path.is_file():
        raise LicenseAuditError(
            f"audit file missing for {repo_id!r}: expected {path}"
        )
    fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    _validate_frontmatter(repo_id=repo_id, fm=fm, path=path)
    return AuditRecord(
        repo_id=repo_id, sanitized=sanitized, path=path, frontmatter=fm,
    )


def _validate_frontmatter(
    *, repo_id: str, fm: dict[str, str], path: Path,
) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in fm]
    if missing:
        raise LicenseAuditError(
            f"{path.name}: missing required keys {missing}"
        )
    if fm["repo_id"] != repo_id:
        raise LicenseAuditError(
            f"{path.name}: frontmatter repo_id {fm['repo_id']!r} "
            f"does not match target {repo_id!r}"
        )
    if fm["license"] not in ALLOWED_LICENSE_VALUES:
        raise LicenseAuditError(
            f"{path.name}: license {fm['license']!r} not in "
            f"{sorted(ALLOWED_LICENSE_VALUES)}"
        )
    if fm["license_audit_status"] not in ALLOWED_STATUS_VALUES:
        raise LicenseAuditError(
            f"{path.name}: license_audit_status "
            f"{fm['license_audit_status']!r} not in "
            f"{sorted(ALLOWED_STATUS_VALUES)}"
        )


def audit_all(
    targets_file: Path = TARGETS_FILE,
) -> tuple[list[AuditRecord], list[LicenseAuditError]]:
    """Audit every target row. Returns (passing_records, errors)."""
    targets = read_targets(targets_file)
    passing: list[AuditRecord] = []
    errors: list[LicenseAuditError] = []
    for repo_id in targets:
        try:
            passing.append(load_audit_record(repo_id))
        except LicenseAuditError as exc:
            errors.append(exc)
    return passing, errors


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 2 on any audit failure (default exit 0)",
    )
    parser.add_argument(
        "--list-targets", action="store_true",
        help="echo the targets list and exit",
    )
    args = parser.parse_args()
    if args.list_targets:
        for t in read_targets():
            print(t)
        return 0
    passing, errors = audit_all()
    for r in passing:
        print(
            f"OK  {r.repo_id} -> {r.path.name} "
            f"({r.frontmatter['license']} / "
            f"{r.frontmatter['license_audit_status']})"
        )
    for e in errors:
        print(f"FAIL {e}", file=sys.stderr)
    if errors and args.check:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
