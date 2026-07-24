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

It also owns the EXECUTION seam for an already-admitted bundle (wave 3):
loading the bundle's single module and handing back one declared entry point.
That half is loud by design -- see `UserBankExecutionError`. Integrity is judged
before anything is dropped in a dropdown; execution happens only after the
operator selected the bank, so a failure there is a failed run, never a
substitution.

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

# Client-owned execution (wave 3). A bank row routes an entry point to its own
# bundle by naming SELF_ENTRY_POINT where a shipped row names a registry id --
# "my bundle owns this". Only a CLIENT row may say it: the shipped registries
# in _otr_source_payload never learn this value, so a shipped row that declares
# it still dies at the registry sweep as an unregistered typo.
SELF_ENTRY_POINT = "self"
FETCH_ENTRY_ATTR = "fetch_source"
INTERPRET_ENTRY_ATTR = "interpret_source"
# RESERVED, wired to nothing (wave 4 decision). No request type, no decision
# type, no runtime consumer exists, so `otr_check bank --activate` does not
# inspect it -- not even for callability. Giving it a shape now would freeze an
# interface with nobody to keep it true. The wave that gives it a real consumer
# defines its types and its checks in the same change.
COMPAT_ENTRY_ATTR = "check_compatibility"
BUNDLE_ENTRY_ATTRS = frozenset({
    FETCH_ENTRY_ATTR, INTERPRET_ENTRY_ATTR, COMPAT_ENTRY_ATTR,
})

# sys.modules namespace for loaded bundles. The activated digest rides in the
# name so edited bytes can never be served from a stale cache entry.
_MODULE_NAME_PREFIX = "otr_user_bank_"


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


class UserBankExecutionError(Exception):
    """An admitted bundle's Python could not be loaded, or does not expose an
    entry point its own bank row routed to it.

    DELIBERATELY LOUD -- the one failure class in this module that raises.
    Discovery quarantines a broken bundle long BEFORE it can reach a dropdown;
    by the time anything executes, the operator has selected this bank and
    pressed go. Substituting another bank, another entry point, or a canned
    payload here would be exactly the silent routing fallback the source-payload
    contract forbids. The run fails, named, with the path to fix."""


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


@dataclass(frozen=True)
class _Authoring:
    """What a bundle's AUTHORED bytes say, judged before any receipt exists.

    This is the half of validation `otr_check bank --activate` can run on a
    bundle it is about to activate -- and the identical half boot runs before
    it looks at the receipt. One implementation, so the CLI can never approve a
    layout the authority would refuse, or refuse one it would admit."""
    bank_id: str
    row_data: dict
    module_path: Path
    story_packs_dir: Path
    digest: str


def _validate_authoring(root: Path, *, protected_ids) -> _Authoring:
    """Layout + id + bank.json + digest. Raises _Quarantine on ANY problem.

    Deliberately stops BEFORE the receipt: at activation time there is no
    receipt yet, and at boot the receipt checks must still run in exactly the
    position they always did, so a doubly-broken bundle reports the same code
    it always reported."""
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
    digest = bundle_digest(root)
    return _Authoring(bank_id=bank_id, row_data=row_data,
                      module_path=module_path,
                      story_packs_dir=story_packs_dir, digest=digest)


def _validate_bundle(root: Path, *, parse_row, protected_ids, snapshots: Path
                     ) -> UserBankBundle:
    """One bundle, fully checked. Raises _Quarantine on ANY problem."""
    # Integrity BEFORE semantics: prove the bytes are the activated bytes,
    # then let the one authority judge the row.
    authoring = _validate_authoring(root, protected_ids=protected_ids)
    bank_id = authoring.bank_id
    row_data = authoring.row_data
    module_path = authoring.module_path
    story_packs_dir = authoring.story_packs_dir
    digest = authoring.digest
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


# ---------------------------------------------------------------------------
# activation (wave 4)
#
# `otr_check bank <path> --activate` is the CONSENT ACT -- the moment a
# client's Python becomes something this repo will import. Everything that act
# writes is defined HERE so there is exactly one owner of the receipt and
# snapshot format. The checker never hand-rolls a second copy, because a second
# copy is a second thing that can drift from what boot reads.
# ---------------------------------------------------------------------------

UNCHECKED = "UNCHECKED"
ACTIVATED = "ACTIVATED"
STALE = "STALE"
BROKEN = "BROKEN"


class UserBankActivationError(Exception):
    """Activation could not be WRITTEN (I/O, or the bundle changed mid-flight).

    Distinct from a quarantine: nothing is wrong with the client's bundle as
    such, the publish itself failed. The operator asked for a write and did not
    get one, so this is loud -- a half-published activation reported as success
    is how a receipt ends up pointing at a snapshot that never existed."""


@dataclass(frozen=True)
class BundlePreflight:
    """A bundle judged WITHOUT its receipt: what the CLI knows before it writes
    one. `issue` is None iff the bundle's authored bytes are sound."""
    root: Path
    bank_id: str
    ok: bool
    digest: "str | None" = None
    module_path: "Path | None" = None
    story_packs_dir: "Path | None" = None
    fixtures_dir: "Path | None" = None
    row: object = None
    issue: "ValidationIssue | None" = None


def _preflight_issue(root: Path, bank_id: str, exc: "_Quarantine"
                     ) -> BundlePreflight:
    return BundlePreflight(
        root=root, bank_id=bank_id, ok=False,
        issue=ValidationIssue(bank_id=bank_id, path=str(root),
                              code=exc.code, detail=exc.detail))


def preflight_bundle(bundle_root: Path, *, parse_row, protected_ids,
                     base: "Path | None" = None) -> BundlePreflight:
    """Judge a bundle's authored bytes exactly the way boot will, minus the
    receipt that activation is about to write.

    Same law as `discover`: this NEVER raises for a bundle problem. The CLI
    turns the returned issue into an exit code and a sentence the client can
    act on, and a refusal here costs that bundle alone."""
    if not callable(parse_row):
        raise UserBankLayoutError("parse_row must be callable")
    bundle_root = Path(bundle_root)
    bank_id = bundle_root.name
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        return _preflight_issue(bundle_root, bank_id, _Quarantine(
            "not_a_bundle_dir",
            "only a plain directory can be activated as a source bank"))
    # A bundle boot will never look at is not activatable. Saying so here beats
    # handing back a receipt for a folder that can never reach a dropdown.
    banks_root = user_banks_root(base)
    if bundle_root.parent.resolve() != banks_root.resolve():
        return _preflight_issue(bundle_root, bank_id, _Quarantine(
            "outside_bank_root",
            f"a client bank must live directly under {banks_root}; "
            f"move the bundle there and re-run"))
    try:
        authoring = _validate_authoring(
            bundle_root, protected_ids=frozenset(protected_ids))
    except _Quarantine as exc:
        return _preflight_issue(bundle_root, bank_id, exc)
    try:
        row = parse_row(authoring.row_data,
                        f"{BANK_JSON_FILENAME} ({authoring.bank_id})")
    except Exception as exc:  # the authority's own validation error
        return _preflight_issue(bundle_root, authoring.bank_id, _Quarantine(
            "bad_row", f"{type(exc).__name__}: {exc}"))
    return BundlePreflight(
        root=bundle_root, bank_id=authoring.bank_id, ok=True,
        digest=authoring.digest, module_path=authoring.module_path,
        story_packs_dir=authoring.story_packs_dir,
        fixtures_dir=bundle_root / FIXTURES_DIRNAME, row=row)


def preflight_bundle_record(pre: BundlePreflight) -> UserBankBundle:
    """The admitted-shape record for a bundle that PASSED preflight but is not
    activated yet.

    Exists so `--activate` can probe a bundle through the EXACT loader the
    writer will later use (`load_bundle_module` / `bundle_entry_point`) instead
    of a second import path that could quietly disagree with it."""
    if not isinstance(pre, BundlePreflight) or not pre.ok:
        raise UserBankLayoutError(
            "preflight_bundle_record requires a passing BundlePreflight")
    return UserBankBundle(
        bank_id=pre.bank_id, root=pre.root, module_path=pre.module_path,
        story_packs_dir=pre.story_packs_dir, fixtures_dir=pre.fixtures_dir,
        digest=pre.digest, row=pre.row)


def write_activation(bundle_root: Path, *, digest: str,
                     base: "Path | None" = None) -> dict:
    """Publish an activation: content-addressed snapshot FIRST, receipt SECOND.

    The ORDER is the whole safety property. A crash between the two leaves the
    bundle UNCHECKED -- honest, and one more `--activate` fixes it. The reverse
    order would leave a receipt naming a snapshot that never existed, which
    reads at boot as a bundle the client broke.

    Re-activating unchanged bytes is a real no-op: the snapshot NAME is the
    content, so an existing directory already holds exactly these bytes."""
    import os
    import shutil

    bundle_root = Path(bundle_root)
    bank_id = bundle_root.name
    snapshots = snapshots_root(base)
    name = snapshot_dirname(bank_id, digest)
    target = snapshots / name
    if not target.is_dir():
        staging = snapshots / f".staging-{name}-{os.getpid()}"
        try:
            snapshots.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
            for relpath, path in _authoring_files(bundle_root):
                dest = staging / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            # The COPY is hashed, not the original. A snapshot published under
            # a name derived from bytes it does not actually contain is a lie
            # the receipt would then certify -- and the copy is exactly where a
            # truncated write or a file changing mid-copy would show up.
            staged_digest = bundle_digest(staging)
            if staged_digest != digest:
                raise UserBankActivationError(
                    f"source bank {bank_id!r}: the snapshot copy hashes to "
                    f"{staged_digest[:16]} but the bundle validated as "
                    f"{digest[:16]}; the bundle changed while it was being "
                    f"copied -- re-run --activate")
            os.replace(staging, target)
        except (_Quarantine, OSError) as exc:
            raise UserBankActivationError(
                f"source bank {bank_id!r}: could not write the activation "
                f"snapshot ({type(exc).__name__}: {exc})") from exc
        finally:
            # Covers EVERY exit from the block, including the digest mismatch
            # raised above -- a leaked `.staging-*` directory would otherwise
            # sit in the snapshots root looking like a real snapshot.
            shutil.rmtree(staging, ignore_errors=True)
    receipt = {"schema_version": RECEIPT_SCHEMA_VERSION,
               "source_bank_id": bank_id, "digest": digest, "snapshot": name}
    # Staged OUTSIDE the bundle: a temp file inside it would join the authoring
    # bytes and change the very digest this receipt is recording.
    staged_receipt = snapshots / f".receipt-{name}-{os.getpid()}.tmp"
    try:
        staged_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(staged_receipt, bundle_root / RECEIPT_FILENAME)
    except OSError as exc:
        try:
            staged_receipt.unlink()
        except OSError:
            pass
        raise UserBankActivationError(
            f"source bank {bank_id!r}: snapshot published but the receipt "
            f"could not be written ({type(exc).__name__}: {exc}); the bundle "
            f"stays UNCHECKED until --activate succeeds") from exc
    return receipt


def activation_status(bundle_root: Path, *, base: "Path | None" = None
                      ) -> "tuple[str, str]":
    """(status, detail) for the RECEIPT half of admission: digest freshness,
    receipt well-formedness, and the snapshot's presence.

    Scope is deliberately narrow and the caller must not widen it by wishing:
    this says nothing about layout, ids, row semantics, packs, or
    cross-references. `otr_check` pairs it with `preflight_bundle` and the
    authority's `validate_client_bundle_contract` precisely because ACTIVATED
    from this function alone would not mean a bank boot will admit."""
    bundle_root = Path(bundle_root)
    try:
        digest = bundle_digest(bundle_root)
    except _Quarantine as exc:
        return BROKEN, exc.detail
    if not (bundle_root / RECEIPT_FILENAME).exists():
        return UNCHECKED, (
            "no activation receipt; run "
            f"`otr_check bank {bundle_root} --activate`")
    try:
        receipt = read_receipt(bundle_root)
    except _Quarantine as exc:
        return BROKEN, exc.detail
    if receipt["source_bank_id"] != bundle_root.name:
        return BROKEN, (f"receipt names {receipt['source_bank_id']!r}, "
                        f"folder is {bundle_root.name!r}")
    if receipt["digest"] != digest:
        return STALE, "the bundle changed since activation; re-activate it"
    if not (snapshots_root(base) / receipt["snapshot"]).is_dir():
        return STALE, (f"activation snapshot {receipt['snapshot']!r} is "
                       f"absent; re-activate it")
    return ACTIVATED, f"digest {digest[:16]}, snapshot {receipt['snapshot']}"


# ---------------------------------------------------------------------------
# client-owned execution (wave 3)
#
# Everything above judges BYTES; everything below RUNS them. The boundary is
# deliberate: discovery never raises for a bundle problem, execution never
# swallows one.
# ---------------------------------------------------------------------------


def bundle_module_name(bank_id: str, digest: str) -> str:
    """`sys.modules` key for a bundle's module: bank id + activated digest.

    The digest is part of the name ON PURPOSE. Re-activating changed bytes
    yields a different name, so an edited bundle can never be served from a
    cache entry built out of its old source."""
    return f"{_MODULE_NAME_PREFIX}{bank_id}_{digest[:16]}"


def load_bundle_module(bundle: UserBankBundle):
    """Import an ADMITTED bundle's single module. LOUD on any failure.

    `importlib` is imported function-locally so this module keeps its zero-I/O,
    stdlib-leaf import posture. The module executes in-process, user-trusted,
    like any custom node -- `otr_check bank <path> --activate` was the consent
    act (docs/EXTENDING_OTR.md section 2), and integrity/staleness were already
    proven at discovery. One run imports it once."""
    import importlib.util
    import sys

    if not isinstance(bundle, UserBankBundle):
        raise UserBankLayoutError(
            f"load_bundle_module expects an admitted UserBankBundle, got "
            f"{type(bundle).__name__}"
        )
    name = bundle_module_name(bundle.bank_id, bundle.digest)
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, bundle.module_path)
    if spec is None or spec.loader is None:
        raise UserBankExecutionError(
            f"source bank {bundle.bank_id!r}: cannot build an import spec for "
            f"{bundle.module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec so the module can import itself by name; popped
    # again on failure so a half-executed module is never reused.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        sys.modules.pop(name, None)
        raise UserBankExecutionError(
            f"source bank {bundle.bank_id!r}: importing "
            f"{bundle.module_path.name} failed ({type(exc).__name__}: {exc}). "
            f"Fix the bundle and re-run `otr_check bank {bundle.root} "
            f"--activate`."
        ) from exc
    return module


def bundle_entry_point(bundle: UserBankBundle, attr: str):
    """One declared entry point off an admitted bundle's module. LOUD.

    A row that routed `attr` to its own bundle and then does not define it is a
    broken contract, not a reason to reach for someone else's implementation."""
    if attr not in BUNDLE_ENTRY_ATTRS:
        raise UserBankLayoutError(
            f"{attr!r} is not a bundle entry point; known: "
            f"{sorted(BUNDLE_ENTRY_ATTRS)}"
        )
    module = load_bundle_module(bundle)
    fn = getattr(module, attr, None)
    if fn is None:
        raise UserBankExecutionError(
            f"source bank {bundle.bank_id!r}: {BANK_JSON_FILENAME} routes this "
            f"lane to the bundle ({SELF_ENTRY_POINT!r}) but "
            f"{bundle.module_path.name} defines no {attr!r} function"
        )
    if not callable(fn):
        raise UserBankExecutionError(
            f"source bank {bundle.bank_id!r}: "
            f"{bundle.module_path.name}.{attr} is {type(fn).__name__}, "
            f"not callable"
        )
    return fn


__all__ = [
    "ACTIVATED",
    "BANK_JSON_FILENAME",
    "BROKEN",
    "BUNDLE_ENTRY_ATTRS",
    "COMPAT_ENTRY_ATTR",
    "FETCH_ENTRY_ATTR",
    "FIXTURES_DIRNAME",
    "INTERPRET_ENTRY_ATTR",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_KEYS",
    "SELF_ENTRY_POINT",
    "STALE",
    "STORY_PACKS_DIRNAME",
    "UNCHECKED",
    "BundlePreflight",
    "UserBankActivationError",
    "UserBankBundle",
    "UserBankExecutionError",
    "UserBankLayoutError",
    "ValidationIssue",
    "activation_status",
    "bundle_digest",
    "bundle_entry_point",
    "bundle_module_name",
    "discover",
    "load_bundle_module",
    "preflight_bundle",
    "preflight_bundle_record",
    "read_receipt",
    "repo_root",
    "snapshot_dirname",
    "snapshots_root",
    "user_banks_root",
    "write_activation",
]
