"""Client-authored source-bank bundles -- discovery, integrity, quarantine.

Independent source banks v1 (plan of record:
docs/2026-07-24-independent-source-banks-v1-plan.md; requirements contract:
docs/EXTENDING_OTR.md). A client adds a 7th+ bank EQUAL to the shipped six by
dropping a self-contained bundle under `user_packs/source_banks/<bank_id>/`
and activating it with `otr_check bank <path> --activate`.

Law (differs from the shipped seed ON PURPOSE):
  * A broken SHIPPED seed (nodes/story_packs/banks.json) fails node
    registration LOUD -- that is _otr_story_routing's job and stays unchanged.
  * A broken CLIENT bundle QUARANTINES: this module NEVER raises for a
    bundle-level problem. It returns (admitted, issues); a quarantined bank is
    absent from every dropdown, boot survives, and the stored ValidationIssue
    names the bank, the path, and what to fix.

This module owns bundle INTEGRITY only (layout, id safety, path containment,
activation receipt, content digest). Row SEMANTICS are parsed by the ONE
authority via the injected `parse_row` callable, so there is exactly one bank
schema parser in the tree and client rows can never drift from shipped ones.

LAZY: importing this module performs ZERO file I/O, the same posture as the
routing authority and the pack loader. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# Bundle layout (docs/EXTENDING_OTR.md section 2).
USER_PACKS_DIRNAME = "user_packs"
SOURCE_BANKS_DIRNAME = "source_banks"
SNAPSHOTS_DIRNAME = ".snapshots"
BANK_JSON_FILENAME = "bank.json"
RECEIPT_FILENAME = ".otr_receipt.json"
STORY_PACKS_DIRNAME = "story_packs"
FIXTURES_DIRNAME = "fixtures"

# Never part of the authoring bytes: the receipt is written BY activation, and
# bytecode caches are machine-local noise. Excluding them keeps the digest a
# pure function of what the client actually authored.
_DIGEST_EXCLUDED_NAMES = frozenset({RECEIPT_FILENAME})
_DIGEST_EXCLUDED_DIRS = frozenset({"__pycache__", SNAPSHOTS_DIRNAME})

RECEIPT_KEYS = frozenset({"schema_version", "source_bank_id", "digest", "snapshot"})
RECEIPT_SCHEMA_VERSION = "v2.0"


@dataclass(frozen=True)
class ValidationIssue:
    """One quarantined client bundle. Never raised -- stored and reported.

    `code` is a stable machine id (tests and `otr_check` match on it);
    `detail` is the human sentence that names what to fix."""
    bank_id: str
    path: str
    code: str
    detail: str

    def render(self) -> str:
        return f"[OTR] source bank {self.bank_id!r} QUARANTINED ({self.code}): {self.detail} [{self.path}]"


@dataclass(frozen=True)
class UserBankBundle:
    """An ADMITTED client bundle: integrity proven, row parsed by the one
    authority. Everything downstream routes by this owner record."""
    bank_id: str
    root: Path
    module_path: Path
    story_packs_dir: Path
    fixtures_dir: Path
    digest: str
    row: object  # the SourceBank the injected parse_row returned


class UserBankLayoutError(Exception):
    """A programming error in THIS module's callers (never a bundle problem).

    Bundle problems become ValidationIssue rows; this exists so a genuine
    contract violation (e.g. a non-callable parse_row) still fails loud."""


class _Quarantine(Exception):
    """Internal control flow: carries the (code, detail) of one bad bundle.

    Never escapes `discover()` -- it is converted to a ValidationIssue there."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def repo_root() -> Path:
    """The custom-node repo root (this file lives in nodes/)."""
    return Path(__file__).resolve().parent.parent


def user_banks_root(root: "Path | None" = None) -> Path:
    """`<repo>/user_packs/source_banks` unless a root is injected (tests)."""
    base = repo_root() if root is None else Path(root)
    return base / USER_PACKS_DIRNAME / SOURCE_BANKS_DIRNAME


def snapshots_root(root: "Path | None" = None) -> Path:
    """`<repo>/user_packs/.snapshots` -- content-addressed activation copies."""
    base = repo_root() if root is None else Path(root)
    return base / USER_PACKS_DIRNAME / SNAPSHOTS_DIRNAME


def _contained(base: Path, target: Path) -> bool:
    """True iff `target` resolves INSIDE `base` (symlink escapes included)."""
    try:
        target.resolve(strict=True).relative_to(base.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _authoring_files(root: Path) -> "list[tuple[str, Path]]":
    """Every authored file in the bundle as (posix relpath, path), sorted.

    Excludes the activation receipt and bytecode caches. Refuses symlinks and
    anything that resolves outside the bundle -- a bundle that reaches out of
    its own folder is quarantined, not digested."""
    out: "list[tuple[str, Path]]" = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError as exc:
            raise _Quarantine("unreadable_bundle", f"cannot list {current}: {exc}")
        for entry in entries:
            if entry.is_symlink():
                raise _Quarantine(
                    "symlink_rejected",
                    f"{entry.name} is a symlink; bundles must be plain files",
                )
            if entry.is_dir():
                if entry.name in _DIGEST_EXCLUDED_DIRS:
                    continue
                if not _contained(root, entry):
                    raise _Quarantine(
                        "path_escape", f"{entry.name} resolves outside the bundle")
                stack.append(entry)
                continue
            if entry.name in _DIGEST_EXCLUDED_NAMES:
                continue
            if not _contained(root, entry):
                raise _Quarantine(
                    "path_escape", f"{entry.name} resolves outside the bundle")
            out.append((entry.relative_to(root).as_posix(), entry))
    out.sort(key=lambda row: row[0])
    return out


def bundle_digest(root: Path) -> str:
    """Timestamp-free canonical digest of the bundle's AUTHORING BYTES.

    Same bytes -> same digest on any machine and any clock, so the activation
    receipt proves "these are the bytes the client validated" and nothing
    softer. Path, length, and content all feed the hash, so a rename or a
    truncation is as visible as an edit."""
    digest = hashlib.sha256()
    digest.update(b"otr-user-bank-v1\0")
    for relpath, path in _authoring_files(root):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise _Quarantine("unreadable_bundle", f"cannot read {relpath}: {exc}")
        digest.update(relpath.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _Quarantine(f"missing_{label}", f"{path.name} not found")
    except OSError as exc:
        raise _Quarantine(f"unreadable_{label}", f"cannot read {path.name}: {exc}")
    except UnicodeDecodeError as exc:
        raise _Quarantine(
            f"malformed_{label}", f"{path.name} is not valid UTF-8: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Quarantine(f"malformed_{label}", f"{path.name}: {exc}")
    if not isinstance(data, dict):
        raise _Quarantine(
            f"malformed_{label}", f"{path.name}: top level must be an object")
    return data


# bank.json mirrors nodes/story_packs/banks.json: a versioned envelope around
# the bank row, so the row itself stays byte-comparable with a shipped row and
# the ONE parser validates both.
_BANK_JSON_KEYS = frozenset({"schema_version", "bank"})
_ID_FIRST_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz")
_ID_CHARS = _ID_FIRST_CHARS | frozenset("0123456789_")


def _check_bank_id(bank_id: str) -> None:
    """Folder name = bank id = dropdown value. Keep it boring on purpose."""
    if not bank_id or bank_id[0] not in _ID_FIRST_CHARS:
        raise _Quarantine(
            "bad_bank_id",
            f"{bank_id!r} must start with a lowercase ascii letter",
        )
    bad = sorted(set(bank_id) - _ID_CHARS)
    if bad:
        raise _Quarantine(
            "bad_bank_id",
            f"{bank_id!r} contains {bad}; use lowercase letters, digits, underscore",
        )


def read_receipt(root: Path) -> dict:
    """The activation receipt written by `otr_check bank <path> --activate`."""
    data = _read_json(root / RECEIPT_FILENAME, "receipt")
    unknown = sorted(set(data) - RECEIPT_KEYS)
    if unknown or sorted(RECEIPT_KEYS - set(data)):
        raise _Quarantine(
            "malformed_receipt",
            f"{RECEIPT_FILENAME} keys must be exactly {sorted(RECEIPT_KEYS)}",
        )
    if data.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise _Quarantine(
            "malformed_receipt",
            f"receipt schema_version must be {RECEIPT_SCHEMA_VERSION!r}",
        )
    for key in ("source_bank_id", "digest", "snapshot"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise _Quarantine(
                "malformed_receipt", f"receipt {key} must be a non-empty string")
    return data


def snapshot_dirname(bank_id: str, digest: str) -> str:
    """Content-addressed snapshot folder name (stable, timestamp-free)."""
    return f"{bank_id}-{digest[:16]}"


def _validate_bundle(root: Path, *, parse_row, protected_ids, snapshots: Path
                     ) -> UserBankBundle:
    """One bundle, fully checked. Raises _Quarantine on ANY problem."""
    bank_id = root.name
    _check_bank_id(bank_id)
    if bank_id in protected_ids:
        raise _Quarantine(
            "protected_id",
            f"{bank_id!r} is a shipped bank id; pick a different folder name",
        )
    envelope = _read_json(root / BANK_JSON_FILENAME, "bank_json")
    unknown = sorted(set(envelope) - _BANK_JSON_KEYS)
    if unknown:
        raise _Quarantine(
            "malformed_bank_json",
            f"{BANK_JSON_FILENAME}: unknown key(s) {unknown}; "
            f"known: {sorted(_BANK_JSON_KEYS)}",
        )
    if envelope.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise _Quarantine(
            "malformed_bank_json",
            f"{BANK_JSON_FILENAME}: schema_version must be "
            f"{RECEIPT_SCHEMA_VERSION!r}",
        )
    row_data = envelope.get("bank")
    if not isinstance(row_data, dict):
        raise _Quarantine(
            "malformed_bank_json", f"{BANK_JSON_FILENAME}: 'bank' must be an object")
    if row_data.get("source_bank_id") != bank_id:
        raise _Quarantine(
            "id_mismatch",
            f"bank.source_bank_id {row_data.get('source_bank_id')!r} does not "
            f"match the folder name {bank_id!r}",
        )

    module_path = root / f"{bank_id}.py"
    if not module_path.is_file():
        raise _Quarantine(
            "missing_module",
            f"{bank_id}.py not found; the bundle must ship one module with "
            f"fetch_source + interpret_source + check_compatibility",
        )
    story_packs_dir = root / STORY_PACKS_DIRNAME
    if not story_packs_dir.is_dir() or not sorted(story_packs_dir.glob("*.json")):
        raise _Quarantine(
            "missing_story_pack",
            f"{STORY_PACKS_DIRNAME}/ must contain at least one story pack JSON",
        )
    # Integrity BEFORE semantics: prove the bytes are the activated bytes,
    # then let the one authority judge the row.
    digest = bundle_digest(root)
    receipt = read_receipt(root)
    if receipt["source_bank_id"] != bank_id:
        raise _Quarantine(
            "receipt_id_mismatch",
            f"receipt names {receipt['source_bank_id']!r}, folder is {bank_id!r}",
        )
    if receipt["digest"] != digest:
        raise _Quarantine(
            "stale_receipt",
            "the bundle changed since activation; re-run "
            f"`otr_check bank {root} --activate`",
        )
    snapshot_dir = snapshots / receipt["snapshot"]
    if not snapshot_dir.is_dir() or not _contained(snapshots, snapshot_dir):
        raise _Quarantine(
            "missing_snapshot",
            f"activation snapshot {receipt['snapshot']!r} is absent; re-run "
            f"`otr_check bank {root} --activate`",
        )
    try:
        row = parse_row(row_data, f"{BANK_JSON_FILENAME} ({bank_id})")
    except Exception as exc:  # the authority's own validation error
        raise _Quarantine("bad_row", f"{type(exc).__name__}: {exc}")
    return UserBankBundle(
        bank_id=bank_id,
        root=root,
        module_path=module_path,
        story_packs_dir=story_packs_dir,
        fixtures_dir=root / FIXTURES_DIRNAME,
        digest=digest,
        row=row,
    )


def discover(*, parse_row, protected_ids, root: "Path | None" = None
             ) -> "tuple[tuple[UserBankBundle, ...], tuple[ValidationIssue, ...]]":
    """Every client bundle under `user_packs/source_banks/`, partitioned.

    Returns (admitted, issues). NEVER raises for a bundle problem -- a broken
    bundle is quarantined alone and every healthy sibling still admits.
    `parse_row(row_dict, origin)` is the ONE bank-row parser, injected by the
    routing authority so client and shipped rows can never diverge."""
    if not callable(parse_row):
        raise UserBankLayoutError("parse_row must be callable")
    banks_root = user_banks_root(root)
    snapshots = snapshots_root(root)
    if not banks_root.is_dir():
        return (), ()
    protected = frozenset(protected_ids)
    admitted: "list[UserBankBundle]" = []
    issues: "list[ValidationIssue]" = []
    try:
        entries = sorted(banks_root.iterdir())
    except OSError as exc:
        return (), (ValidationIssue(
            bank_id="*", path=str(banks_root), code="unreadable_root",
            detail=f"cannot list the client bank root: {exc}"),)
    for entry in entries:
        if not entry.is_dir() or entry.is_symlink():
            issues.append(ValidationIssue(
                bank_id=entry.name, path=str(entry), code="not_a_bundle_dir",
                detail="only plain bundle directories may live here"))
            continue
        try:
            admitted.append(_validate_bundle(
                entry, parse_row=parse_row, protected_ids=protected,
                snapshots=snapshots))
        except _Quarantine as exc:
            issues.append(ValidationIssue(
                bank_id=entry.name, path=str(entry), code=exc.code,
                detail=exc.detail))
    return tuple(admitted), tuple(issues)


__all__ = [
    "BANK_JSON_FILENAME",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_KEYS",
    "UserBankBundle",
    "UserBankLayoutError",
    "ValidationIssue",
    "bundle_digest",
    "discover",
    "read_receipt",
    "repo_root",
    "snapshot_dirname",
    "snapshots_root",
    "user_banks_root",
]
