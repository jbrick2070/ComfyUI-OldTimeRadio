"""otr_check -- the client source-bank checker (independent source banks v1, wave 4).

    otr_check.py bank <path> [--activate] [--json] [--timeout S]
    otr_check.py bank --all [--json]

`--activate` is the CONSENT ACT: it is the moment a client's Python becomes
something this repo will import at boot. It validates the bundle's authored
bytes, imports the module in a BOUNDED CHILD PROCESS, checks the recorded
fixtures, and only then publishes a content-addressed snapshot plus
`.otr_receipt.json`.

This file owns the CLI -- argument parsing, the child-process bound, exit
codes, and the sentences a client reads. It owns NO format: every digest,
receipt, snapshot-name and quarantine decision comes from
`nodes/_otr_user_banks.py`, and every bank-row judgement comes from the one
routing authority via `shipped_bank_seed()`. A second copy of either is a
second thing that can drift from what boot actually enforces.

Exit codes: 0 the bank is activated (or already was); 1 the bundle was
REFUSED, with a named issue; 2 the checker itself could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nodes import _otr_story_routing as _routing  # noqa: E402
from nodes import _otr_user_banks as _ub  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2

# A client's module-level code gets one minute to import. Anything slower is a
# bug in the bundle, not a slow machine: heavy imports belong inside the
# functions that need them (docs/EXTENDING_OTR.md section 2).
DEFAULT_PROBE_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# the probe -- CHILD SIDE
#
# Runs in a separate interpreter on purpose. Importing a client's module is the
# first moment their code executes, and a module that hangs, calls sys.exit, or
# segfaults must cost the checker a named refusal, never the checker itself.
# ---------------------------------------------------------------------------


class _ProbeRefusal(Exception):
    """A named, client-actionable refusal raised inside the probe process."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _self_entry_attrs(row) -> "list[str]":
    """The entry points this row routed to its OWN bundle.

    A client row names the reserved id `"self"` where a shipped row names a
    registry id. Only those lanes must exist in the module; a row that reuses a
    shipped fetcher/interpreter id is not promising any local function."""
    attrs = []
    if getattr(row, "fetcher", "") == _ub.SELF_ENTRY_POINT:
        attrs.append(_ub.FETCH_ENTRY_ATTR)
    if getattr(row, "interpreter", "") == _ub.SELF_ENTRY_POINT:
        attrs.append(_ub.INTERPRET_ENTRY_ATTR)
    return attrs


def _check_fixtures(fixtures_dir: Path) -> int:
    """Validate recorded fetch payloads with the runtime's OWN validator.

    `fixtures/` is optional. When it is present, every JSON in it is a recorded
    fetch payload and is judged by `normalize_fetch_result` -- the same
    function that will judge the bundle's live `fetch_source` output. Checking
    fixtures with a checker-local copy of the envelope rules would let a
    fixture pass here and fail at run time, which is the one thing activation
    exists to prevent."""
    from nodes import _otr_source_payload as _osp

    if not fixtures_dir.is_dir():
        return 0
    count = 0
    for path in sorted(fixtures_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _ProbeRefusal(
                "bad_fixture",
                f"fixtures/{path.name} is not readable UTF-8 JSON: {exc}")
        try:
            _osp.normalize_fetch_result(data, f"fixtures/{path.name}")
        except Exception as exc:
            raise _ProbeRefusal(
                "bad_fixture",
                f"fixtures/{path.name} is not a valid fetch payload "
                f"({type(exc).__name__}: {exc})")
        count += 1
    return count


# The keyword sets the TRUSTED WRITER actually passes, re-derived from the live
# call sites -- `OTR_LedgerScriptWriter._resolve_inputs` (the fetch call) and
# `_run_source_interpreter` (the interpreter call) -- and cross-checked against
# docs/EXTENDING_OTR.md section 2, which agrees. Derived from the CALLER on
# purpose: the caller is what a client's function actually meets, so if the two
# ever drift, this list must follow the code and the doc gets fixed.
_WRITER_KEYWORDS = {
    _ub.FETCH_ENTRY_ATTR: ("bank", "technical_model", "source_ref",
                           "load_config", "policy"),
    _ub.INTERPRET_ENTRY_ATTR: ("bank", "payload", "technical_fn", "model_id"),
}


def _bind_writer_signature(attr: str, fn) -> None:
    """Prove the callable can ACCEPT what the writer will pass -- without
    calling it.

    Importability alone is not enough: a function that imports fine but takes
    the wrong keywords is a render that dies minutes in, after the operator
    committed a GPU. Binding is free, needs no network, and no client code
    runs. A `**kwargs` catch-all satisfies this, as it should."""
    import inspect

    keywords = _WRITER_KEYWORDS[attr]
    try:
        inspect.signature(fn).bind(**{name: None for name in keywords})
    except TypeError as exc:
        raise _ProbeRefusal(
            "bad_entry_signature",
            f"{attr} cannot accept the keywords the writer passes "
            f"({', '.join(keywords)}): {exc}. Accept them by name, or take "
            f"**kwargs.")


def _probe_verdict(bundle_root: Path, base: "Path | None") -> dict:
    """Everything that must survive executing the client's module. One dict."""
    parse_row, shipped_ids = _routing.shipped_bank_seed()
    pre = _ub.preflight_bundle(bundle_root, parse_row=parse_row,
                               protected_ids=shipped_ids, base=base)
    if not pre.ok:
        return {"ok": False, "code": pre.issue.code, "detail": pre.issue.detail,
                "bank_id": pre.bank_id, "digest": None}
    record = _ub.preflight_bundle_record(pre)
    entry_points = _self_entry_attrs(pre.row)
    for attr in entry_points:
        # The SAME loader the writer uses at run time -- so a bundle that
        # activates cleanly cannot fail its first render on an import.
        _bind_writer_signature(attr, _ub.bundle_entry_point(record, attr))
    # `check_compatibility` is deliberately NOT examined. It is a reserved
    # name with no runtime consumer, so there is nothing true for activation
    # to assert about it; enforcing callability would reject a bundle over an
    # interface production never touches. docs/EXTENDING_OTR.md says so too.
    return {"ok": True, "code": "", "detail": "", "bank_id": pre.bank_id,
            "digest": pre.digest, "entry_points": entry_points,
            "fixtures": _check_fixtures(pre.fixtures_dir)}


def _run_probe(bundle_root: Path, result_path: Path, base: "Path | None") -> int:
    """Child entry point. ALWAYS writes a verdict file, never a traceback.

    A crash with no verdict is itself a verdict -- the parent reports
    `probe_crashed` with the tail of stderr -- but a refusal we can name is
    worth far more to the client than a stack trace."""
    try:
        verdict = _probe_verdict(bundle_root, base)
    except _ProbeRefusal as exc:
        verdict = {"ok": False, "code": exc.code, "detail": exc.detail,
                   "bank_id": bundle_root.name, "digest": None}
    except _ub.UserBankExecutionError as exc:
        verdict = {"ok": False, "code": "bundle_execution",
                   "detail": str(exc), "bank_id": bundle_root.name,
                   "digest": None}
    except BaseException as exc:  # a client module may raise anything at all
        verdict = {"ok": False, "code": "bundle_import_failed",
                   "detail": f"{type(exc).__name__}: {exc}",
                   "bank_id": bundle_root.name, "digest": None}
    result_path.write_text(json.dumps(verdict), encoding="utf-8")
    return EXIT_OK


# ---------------------------------------------------------------------------
# the probe -- PARENT SIDE
# ---------------------------------------------------------------------------


def _kill_probe_tree(proc) -> None:
    """Kill the probe AND whatever it spawned.

    `Popen.kill()` reaps only the direct child, so a bundle that started a
    helper at import time would leave it running on the operator's machine
    after the checker returned. On Windows that is `taskkill /F /T` -- the same
    tool the render harness uses for exactly this reason."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.kill()


def _probe_bundle(bundle_root: Path, *, timeout: float,
                  base: "Path | None" = None) -> dict:
    """Run the probe in a bounded child and return its verdict dict.

    A timeout or a dead child is a REFUSAL with a name, never an exception out
    of the checker: the client gets a sentence describing what their bundle
    did, and no other bank is affected. The bound is WALL TIME plus the process
    tree -- a bundle that prints without end is not bounded here, and saying so
    is better than implying a guarantee that does not hold."""
    with tempfile.TemporaryDirectory(prefix="otr_check_") as tmp:
        result_path = Path(tmp) / "verdict.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "_probe",
               "--bundle", str(bundle_root), "--result", str(result_path)]
        if base is not None:
            cmd += ["--base", str(base)]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env,
                                cwd=str(_REPO_ROOT))
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_probe_tree(proc)
            proc.communicate()
            return {"ok": False, "code": "probe_timeout", "digest": None,
                    "bank_id": bundle_root.name,
                    "detail": (f"importing the bundle did not finish within "
                               f"{timeout:.0f}s; module-level work must be "
                               f"cheap -- keep heavy imports inside the "
                               f"functions that need them")}
        if not result_path.is_file():
            tail = (err or out or "").strip().splitlines()[-6:]
            return {"ok": False, "code": "probe_crashed", "digest": None,
                    "bank_id": bundle_root.name,
                    "detail": (f"the probe process exited {proc.returncode} "
                               f"without a verdict: {' | '.join(tail)}")}
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "code": "probe_crashed", "digest": None,
                    "bank_id": bundle_root.name,
                    "detail": f"unreadable probe verdict: {exc}"}


# ---------------------------------------------------------------------------
# the `bank` command
# ---------------------------------------------------------------------------


def _refused(bundle_root: Path, code: str, detail: str, bank_id: str) -> dict:
    return {"bank_id": bank_id, "path": str(bundle_root), "status": _ub.BROKEN,
            "code": code, "detail": detail, "activated": False}


def _contract_refusal(pre) -> "dict | None":
    """Run the admission contracts `discover` does NOT run, or None.

    `_admit_user_banks` validates a bundle's story packs and cross-references
    AFTER discovery. Skipping them here is how a checker publishes a receipt
    for a bank that quarantines at boot -- so this runs before any write, under
    boot's own stable code."""
    try:
        _routing.validate_client_bundle_contract(_ub.preflight_bundle_record(pre))
    except Exception as exc:
        return _refused(pre.root, "bad_bundle_contract",
                        f"{type(exc).__name__}: {exc}", pre.bank_id)
    return None


def _inspect(bundle_root: Path, *, base: "Path | None") -> dict:
    """What boot would say about this bundle -- WITHOUT executing a line of it.

    Consent has a direction: reading a client's files to describe them is not
    the same act as importing their Python. Only `--activate` does the second."""
    parse_row, shipped_ids = _routing.shipped_bank_seed()
    pre = _ub.preflight_bundle(bundle_root, parse_row=parse_row,
                               protected_ids=shipped_ids, base=base)
    if not pre.ok:
        return _refused(bundle_root, pre.issue.code, pre.issue.detail,
                        pre.issue.bank_id)
    contract = _contract_refusal(pre)
    if contract is not None:
        return contract
    status, detail = _ub.activation_status(bundle_root, base=base)
    return {"bank_id": pre.bank_id, "path": str(bundle_root), "status": status,
            "code": "", "detail": detail, "digest": pre.digest,
            "activated": status == _ub.ACTIVATED}


def _activate(bundle_root: Path, *, timeout: float, base: "Path | None") -> dict:
    """Validate -> probe -> publish -> PROVE. The last step is not ceremony.

    Activation reports success only when a fresh `activation_status` says
    ACTIVATED, i.e. when the same conditions boot enforces are satisfied by
    what we just wrote. Anything less is a checker that congratulates itself."""
    parse_row, shipped_ids = _routing.shipped_bank_seed()
    pre = _ub.preflight_bundle(bundle_root, parse_row=parse_row,
                               protected_ids=shipped_ids, base=base)
    if not pre.ok:
        return _refused(bundle_root, pre.issue.code, pre.issue.detail,
                        pre.issue.bank_id)
    contract = _contract_refusal(pre)
    if contract is not None:
        return contract
    verdict = _probe_bundle(bundle_root, timeout=timeout, base=base)
    if not verdict.get("ok"):
        return _refused(bundle_root, verdict.get("code", "probe_failed"),
                        verdict.get("detail", ""), verdict.get(
                            "bank_id", bundle_root.name))
    # The probe re-derived the digest AFTER importing the module. If it differs
    # from the digest we validated, the bundle changed underneath us and the
    # receipt would record bytes nobody checked.
    if verdict.get("digest") != pre.digest:
        return _refused(bundle_root, "changed_during_activation",
                        "the bundle changed while it was being checked; "
                        "re-run --activate", pre.bank_id)
    receipt = _ub.write_activation(bundle_root, digest=pre.digest, base=base)
    status, detail = _ub.activation_status(bundle_root, base=base)
    return {"bank_id": pre.bank_id, "path": str(bundle_root), "status": status,
            "code": "" if status == _ub.ACTIVATED else "activation_unproven",
            "detail": detail, "digest": pre.digest,
            "snapshot": receipt["snapshot"],
            "entry_points": verdict.get("entry_points", []),
            "fixtures": verdict.get("fixtures", 0),
            "activated": status == _ub.ACTIVATED}


def _render(report: dict) -> str:
    """One bundle, one human paragraph. The code is always visible: it is the
    stable handle a client quotes when they ask for help."""
    head = f"{report['bank_id']}: {report['status']}"
    if report.get("code"):
        head += f" ({report['code']})"
    lines = [head, f"  path      {report['path']}"]
    if report.get("detail"):
        lines.append(f"  detail    {report['detail']}")
    if report.get("snapshot"):
        lines.append(f"  snapshot  {report['snapshot']}")
    if report.get("entry_points"):
        lines.append(f"  own lanes {', '.join(report['entry_points'])}")
    if report.get("fixtures"):
        lines.append(f"  fixtures  {report['fixtures']} recorded payload(s) "
                     f"validated")
    return "\n".join(lines)


def _emit(reports: "list[dict]", as_json: bool) -> None:
    if as_json:
        print(json.dumps(reports, indent=2, sort_keys=True))
        return
    print("\n\n".join(_render(report) for report in reports))


def _cmd_bank(args) -> int:
    base = Path(args.base) if args.base else None
    if args.all:
        if args.activate:
            print("otr_check: --all never activates; consent is per bundle. "
                  "Activate each bank by path.", file=sys.stderr)
            return EXIT_USAGE
        banks_root = _ub.user_banks_root(base)
        if not banks_root.is_dir():
            _emit([], args.as_json)
            print(f"otr_check: no client banks ({banks_root} does not exist)",
                  file=sys.stderr)
            return EXIT_OK
        roots = [p for p in sorted(banks_root.iterdir()) if p.is_dir()]
        reports = [_inspect(root, base=base) for root in roots]
        _emit(reports, args.as_json)
        return EXIT_OK if all(r["activated"] for r in reports) else EXIT_REFUSED
    if not args.path:
        print("otr_check: bank needs a path, or --all", file=sys.stderr)
        return EXIT_USAGE
    bundle_root = Path(args.path).expanduser()
    if args.activate:
        report = _activate(bundle_root, timeout=args.timeout, base=base)
    else:
        report = _inspect(bundle_root, base=base)
    _emit([report], args.as_json)
    return EXIT_OK if report["activated"] else EXIT_REFUSED


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="otr_check",
        description="Validate and activate a client-authored OTR source bank.")
    sub = ap.add_subparsers(dest="command", required=True)

    bank = sub.add_parser(
        "bank", help="validate, activate, or report a client source bank")
    bank.add_argument("path", nargs="?",
                      help="the bundle directory under user_packs/source_banks")
    bank.add_argument("--all", action="store_true",
                      help="report every client bank (never activates)")
    bank.add_argument("--activate", action="store_true",
                      help="import the bundle and publish snapshot + receipt")
    bank.add_argument("--json", action="store_true", dest="as_json",
                      help="machine-readable report")
    bank.add_argument("--timeout", type=float, default=DEFAULT_PROBE_TIMEOUT_S,
                      help=f"seconds the bundle import may take "
                           f"(default {DEFAULT_PROBE_TIMEOUT_S:.0f})")
    # Test hook: point the checker at a throwaway user_packs tree.
    bank.add_argument("--base", default=None, help=argparse.SUPPRESS)

    probe = sub.add_parser("_probe", help=argparse.SUPPRESS)
    probe.add_argument("--bundle", required=True)
    probe.add_argument("--result", required=True)
    probe.add_argument("--base", default=None)
    return ap


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "_probe":
        return _run_probe(Path(args.bundle), Path(args.result),
                          Path(args.base) if args.base else None)
    try:
        return _cmd_bank(args)
    except _ub.UserBankActivationError as exc:
        # The publish itself failed. Loud, named, and NOT reported as success --
        # the bundle stays UNCHECKED, which is the honest state.
        print(f"otr_check: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
