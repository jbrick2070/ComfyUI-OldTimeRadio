"""The pack reaches the network from FIVE named files, and nowhere else.

Unlike the environment and the process spawn, this one is a REGISTER, not a
collapse. The r1 panel's finding, kept verbatim in
``kibitz-runs/2026-09-04-registry-findings-collapse/r1/antigravity.md``: a
unified HTTP owner would save four `info` findings while merging three
genuinely incompatible layers -- ``requests`` (the two LLM backends, one of
which STREAMS), ``urllib.request`` (the Google API client and the cloud-media
invoke, each with its own error handling), and a raw ``socket`` plus
``http.client`` (the feed fetcher, which is doing its own connection control on
purpose). Collapsing those would put real risk on the streaming path to shorten
a report. So the collapse was CUT and this guard took its place.

WHAT IT BUYS, and it is the honest version: not a smaller finding count, but a
list a reviewer can read. Each file says WHY it talks to the network, and both
directions are asserted -- a sixth file that starts calling out fails here, and
a named file that stops calling out must leave the list rather than sit as
decoration.

WHAT COUNTS is an outbound CALL, never an import. ``urllib.parse`` and
``urllib.error`` are string and exception modules; ``eng_google_tts.py`` imports
``urllib.error`` to catch one and reaches nothing, and a rule keyed on imports
would have named it here falsely.
"""
from __future__ import annotations

import ast

from tests.fixtures.ratchet import REPO, assert_ratchet, scan

NODES = REPO / "nodes"
ROOTS = (NODES,)

#: The calls that open an outbound connection.
_URLLIB_CALLS = frozenset({"urlopen", "urlretrieve"})
_REQUESTS_VERBS = frozenset({"get", "post", "put", "patch", "delete", "head",
                             "request", "Session"})
_CONNECTIONS = frozenset({"HTTPConnection", "HTTPSConnection"})
#: The client libraries whose verbs are outbound. Matched on the ROOT of the
#: attribute chain, so ``requests.post`` counts and ``self.session.post`` --
#: which is one of these, reached through an object -- is left to the file that
#: owns it. Every site in the tree today is spelled through the module.
_CLIENT_MODULES = frozenset({"requests", "httpx"})


#: The five files that talk to the network, each with the reason it does.
NETWORK_CALLERS = {
    "nodes/_otr_comfy_backend.py": (
        "posts a prompt to a local or remote ComfyUI over `requests`; the pack "
        "cannot drive a server it may not call"),
    "nodes/_otr_feed_fetch.py": (
        "the news feed reader, and the one site doing its own connection "
        "control -- a raw socket plus http.client, so it can bound a hostile "
        "or slow feed itself rather than inherit a library's timeout policy"),
    "nodes/_otr_google_api/client.py": (
        "the Google API transport: two `urllib.request.urlopen` sites with "
        "their own error mapping, which a shared helper would flatten"),
    "nodes/_otr_openrouter_backend.py": (
        "the OpenRouter transport -- a catalog GET and a completion POST, the "
        "second of which STREAMS; wrapping it is risk taken for a report line"),
    "nodes/_otr_shared/cloud_media_invoke.py": (
        "invokes a cloud media provider and reads the result back over "
        "`urllib.request.urlopen`"),
}


def _offenders(tree, rel):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _URLLIB_CALLS:      # `from urllib.request import urlopen`
                out.append(f"{rel}:{node.lineno} calls {func.id}()")
            continue
        if not isinstance(func, ast.Attribute):
            continue
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        root_id = root.id if isinstance(root, ast.Name) else None
        if func.attr in _URLLIB_CALLS:
            out.append(f"{rel}:{node.lineno} calls {func.attr}()")
        elif func.attr == "socket" and root_id == "socket":
            out.append(f"{rel}:{node.lineno} opens a raw socket")
        elif func.attr in _CONNECTIONS:
            out.append(f"{rel}:{node.lineno} opens an {func.attr}")
        elif func.attr in _REQUESTS_VERBS and root_id in _CLIENT_MODULES:
            out.append(f"{rel}:{node.lineno} calls {root_id}.{func.attr}()")
    return out


_HINT = ("a sixth file now reaches the network. That is a decision, not a "
         "detail: add it to NETWORK_CALLERS with the reason it cannot go "
         "through an existing caller, or route it through one.")


def test_every_named_caller_says_why():
    for rel, reason in NETWORK_CALLERS.items():
        assert (REPO / rel).is_file(), rel
        assert reason.strip(), rel


def test_the_network_is_reached_only_from_the_named_files():
    assert_ratchet(scan(ROOTS, _offenders), set(NETWORK_CALLERS),
                   owner_hint=_HINT)


# --------------------------------------------------------------------------- #
# the finder itself
# --------------------------------------------------------------------------- #
def _find(src):
    return _offenders(ast.parse(src), "probe.py")


def test_the_finder_catches_every_outbound_spelling():
    for src in (
            "import urllib.request\nurllib.request.urlopen(r)\n",
            "from urllib.request import urlopen\nurlopen(r)\n",
            "import requests\nrequests.get(u)\n",
            "import requests\nrequests.post(u)\n",
            "import requests\nrequests.Session()\n",
            "import socket\nsocket.socket(f, s, p)\n",
            "import http.client\nhttp.client.HTTPSConnection(h)\n",
            "import http.client\nhttp.client.HTTPConnection(h)\n",
    ):
        assert _find(src), src


def test_the_finder_does_not_name_a_file_that_only_parses_a_url():
    """`urllib.parse` and `urllib.error` reach nothing. Keying this rule on
    imports would have put eng_google_tts.py and google_slug_verifier.py on the
    list for importing an exception class and a query-string builder."""
    for src in (
            "import urllib.error\ntry:\n    pass\n"
            "except urllib.error.HTTPError:\n    pass\n",
            "import urllib.parse\nurllib.parse.urlencode(d)\n",
            "d = {}\nd.get('a')\nd.post = 1\n",       # a mapping, not a client
            "s = 'requests.get(url)'\n",
    ):
        assert _find(src) == [], src
