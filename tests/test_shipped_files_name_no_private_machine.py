"""The published zip names no machine on the author's own network.

Every file the registry zip carries is read by a policy reviewer and by anyone
who installs the pack. A private LAN address, a share name or a workstation's
nickname in a comment tells them nothing about the code and something about
the author's network.

THE DEFECT THIS PINS (shipped through 2.3.64, found by sweeping the published
zip): nodes/_otr_paths.py quoted a mapped drive's real UNC target -- a private
10.x address and its share name -- in two docstrings, config/workflow_matrix.json
carried a drive-letter path to one render in an evidence note, and three
comments named a workstation.

Scans the PACKED set (git-tracked minus .comfyignore), the same list the scan
replica reads, so a file that does not ship is not policed here. CPU only.
UTF-8 no BOM.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: RFC 1918 addresses. A CIDR block ("10.0.0.0/8") is documentation of a rule,
#: not a machine, so an address followed by "/<digits>" is not matched.
_PRIVATE_IPV4 = re.compile(
    r"(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.]|/\d)"
)
#: The workstations this pack is authored on, by nickname.
_WORKSTATIONS = re.compile(r"\b(?:MRKT|IDREAM)\b")
#: A drive-letter path into a named output tree, as an evidence note would
#: quote it. Drive C: examples in help text are ordinary documentation.
_OTHER_DRIVE_OUTPUT = re.compile(r"(?<![\w])[ABD-Zabd-z]:\\{1,2}output\\", re.IGNORECASE)

_TEXT_SUFFIXES = frozenset((
    ".py", ".json", ".md", ".yaml", ".yml", ".toml", ".txt", ".js", ".cfg",
    ".ini", ".example", ".html", ".css",
))


def _shipped_text_files() -> list:
    spec = importlib.util.spec_from_file_location(
        "otr_registry_scan_oracle", REPO / "scripts" / "otr_registry_scan_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    return [p for p in oracle._shipped_files(False)
            if Path(p).suffix.lower() in _TEXT_SUFFIXES]


def _hits(pattern: re.Pattern) -> list:
    found = []
    for rel in _shipped_text_files():
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            match = pattern.search(line)
            if match:
                found.append("%s:%d: %s" % (rel, number, match.group(0)))
    return found


def test_the_shipped_set_is_not_empty():
    assert len(_shipped_text_files()) > 300


def test_no_shipped_file_carries_a_private_lan_address():
    assert _hits(_PRIVATE_IPV4) == []


def test_no_shipped_file_names_an_authoring_workstation():
    assert _hits(_WORKSTATIONS) == []


def test_no_shipped_file_quotes_a_path_on_another_drive():
    assert _hits(_OTHER_DRIVE_OUTPUT) == []


def test_the_patterns_match_what_they_claim():
    assert _PRIVATE_IPV4.search(r"-> \\10.55.0.2\share")
    assert _PRIVATE_IPV4.search("host 192.168.1.20:8188")
    assert not _PRIVATE_IPV4.search("blocks 10.0.0.0/8 and 192.168.0.0/16")
    assert not _PRIVATE_IPV4.search("Windows 10.0.26200.1 build")
    assert not _PRIVATE_IPV4.search("listen on 127.0.0.1:8188")
    assert _OTHER_DRIVE_OUTPUT.search(r"verified as D:\\output\\otr\\obs\\x.mp4")
    assert not _OTHER_DRIVE_OUTPUT.search(r"C:\Users\you\ComfyUI\output\otr")
