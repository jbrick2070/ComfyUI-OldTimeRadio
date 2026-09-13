"""Generate the per-dropdown matrix: what each choice costs, and what runs it.

Two questions get asked about every dropdown in the shipped workflow, and they
are NOT the same question:

  1. **Will OTR even offer me this on my machine?**  Answered by the engine's
     own ``CAPABILITIES`` row against the shipped capability profiles. This is
     a fact about the code and it is DERIVED here -- never typed, never stale.
  2. **Will it fit, and has anyone run it?**  Answered by receipts and by
     memory measurement. Nothing can derive this, so it is curated in
     ``docs/dropdown_matrix.json``.

WHAT IS CURATED, EXACTLY -- stated because an earlier draft of this docstring
claimed the memory verdicts were the ONLY curated input, and that was false. The
curated file also carries ``size_gb``, and it has to: a lane with a fetch
manifest gets its size summed from real artifact bytes and the curated figure is
ignored, but an engine with NO fetch lane (``sd15``, the sidecar voices, the
documented-manual tiers) has no manifest to sum, and its size is typed. Those
rows are the ones a size drift can still reach.

Collapsing the two is exactly how README came to tell Mac users that
``flux2_klein`` was "proven" there: it HAD rendered on an M4, and its
declaration is ``device_backends: ["cuda"]``, so the code refuses it on every
Mac profile. Both halves were true; the single word was wrong. So the rendered
table always carries both, and this generator FAILS when a curated receipt
lands on a cell the code refuses.

The download half comes from the real fetch manifests -- ``MANUAL_TIERS`` and
``LANES`` in the provisioner and the fetcher -- so sizes are artifact bytes and
"is this a free auto-download?" is read out of the same table that performs the
download. That column is the one the shipped JSON's whole design turns on, and
it cannot drift from what actually happens on a first run.

Usage:
    python scripts/otr_dropdown_matrix.py            # write doc + README block
    python scripts/otr_dropdown_matrix.py --check    # fail if either is stale
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DOC = os.path.join(_REPO, "docs", "DROPDOWN_MATRIX.md")
_CURATED = os.path.join(_REPO, "docs", "dropdown_matrix.json")
_README = os.path.join(_REPO, "README.md")
#: The SHIPPING copy. `docs/` is excluded from the registry bundle
#: (`.comfyignore` line 178) and `apple/` is not, so this is the only machine
#: table an installed user can read. It is generated for the same reason the
#: other two are: a hand-kept copy of a derived table is a place for the same
#: fact to be wrong in.
_APPLE = os.path.join(_REPO, "apple", "MACHINES.md")

_BEGIN = "<!-- BEGIN GENERATED: dropdown-matrix -->"
_END = "<!-- END GENERATED: dropdown-matrix -->"

#: The machine classes the table has columns for, in reading order. Each names
#: a REAL shipped profile -- the availability answer is that profile's, not an
#: invented one, so a reader can reproduce any cell with `--profile <id>`.
MACHINES = (
    {"key": "nv8", "label": "8 GB NVIDIA", "profile": "8gb_lite",
     "blurb": "RTX 4060 / 3070 / 2080 class"},
    {"key": "nv16", "label": "16 GB+ NVIDIA", "profile": "16gb_full",
     "blurb": "RTX 5080 / 4080 / 3090 class"},
    {"key": "mac16", "label": "Mac 16 GB", "profile": "otr_mac_mps",
     "blurb": "Apple Silicon, unified memory"},
    {"key": "amd", "label": "AMD ROCm", "profile": "otr_amd16_rocm",
     "blurb": "Linux only -- and read the caveat under the table"},
    {"key": "cpu", "label": "CPU only", "profile": "cpu_floor",
     "blurb": "no GPU at all"},
)

_REGISTRIES = (
    ("video", "nodes/_otr_video_engines/registry.py"),
    ("image", "nodes/_otr_image_engines/registry.py"),
    ("audio", "nodes/_otr_audio_engines/registry.py"),
    ("upscale", "nodes/_otr_upscale_engines/registry.py"),
)

#: Why the code refuses an engine on a machine -> what to say in a cell. The
#: wording matters: none of these mean "your hardware cannot do this", and a
#: table that implies they do sends people to buy a machine they already own.
#: The machine classes `_otr_model_catalog.fit_tags_for` actually evaluates.
#: Anything outside this set gets no derived verdict, only "?".
_WRITER_FIT_KEYS = frozenset({"mac16", "nv8", "nv16", "nv24"})

_REFUSAL = {
    "requires_cuda": "not offered",
    "requires_vendor": "not offered",
    "missing_toolchain": "not offered",
    "sidecars_disabled": "not offered",
    "impractical_on_cpu": "too slow",
}


def _load(path: str, name: str):
    """Import a repo module by path -- scripts/ and nodes/ are not packages."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def registry_capabilities() -> dict:
    """``{namespace: {engine: declaration}}``, read with ``ast``.

    Deliberately parsed rather than imported: the registries pull in adapters
    that want torch, and this generator has to run in a docs check with no GPU
    stack at all.
    """
    out = {}
    for namespace, rel in _REGISTRIES:
        tree = ast.parse(io.open(os.path.join(_REPO, rel), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "CAPABILITIES" for t in node.targets):
                out[namespace] = ast.literal_eval(node.value)
                break
        else:                                    # pragma: no cover - structural
            raise SystemExit("no CAPABILITIES table in %s" % rel)
    return out


def availability_grid(caps: dict) -> dict:
    """``{(namespace, engine): {machine_key: reason}}`` -- fully derived."""
    profiles = _load("nodes/_otr_shared/capability_profiles.py", "_odm_profiles")
    grid = {}
    for machine in MACHINES:
        path = os.path.join(_REPO, "config/profiles/%s.json" % machine["profile"])
        doc = json.load(io.open(path, encoding="utf-8"))
        profile = doc.get("capability_profile") or doc
        for namespace, table in caps.items():
            for engine, reason in profiles.availability(profile, table).items():
                grid.setdefault((namespace, engine), {})[machine["key"]] = reason
    return grid


def download_facts() -> dict:
    """``{lane: {"gb": float, "gated": bool, "manual": bool}}``, from the real
    manifests -- never a typed-in size.

    The two lane tables carry their sizes differently. ``MANUAL_TIERS`` states
    ``bytes`` per artifact, so its total is summed. ``LANES`` predates that on
    several rows -- the legacy entries are bare 3-tuples whose size lives only
    in a trailing comment -- so the authority for a fetcher lane is
    ``LANE_INFO``, the pick list a person actually reads before spending the
    bandwidth. Where every spec in a lane DOES carry ``expected_bytes``, the
    two are cross-checked: ``LANE_INFO``'s own docstring says to keep them in
    step, and until now nothing enforced it.
    """
    provision = _load("scripts/otr_provision.py", "_odm_provision")
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_odm_fetcher")
    info = getattr(fetcher, "LANE_INFO", {})
    facts, drift = {}, []
    for lane, specs in getattr(fetcher, "LANES", {}).items():
        stated = float(info.get(lane, (0.0, ""))[0] or 0.0)
        byte_totals = [getattr(spec, "expected_bytes", None) for spec in specs]
        if byte_totals and all(b for b in byte_totals):
            # GiB, not GB. LANE_INFO is powers of two -- humo and minimax_h3
            # both match their manifests to three decimals that way and to
            # nothing at all in powers of ten.
            summed = round(sum(byte_totals) / 2 ** 30, 2)
            if stated and abs(summed - stated) > 0.02:
                drift.append("%s: LANE_INFO says %.2f GiB, its manifest sums to "
                             "%.2f GiB" % (lane, stated, summed))
            stated = stated or summed
        # Every lane in the fetcher is by definition a no-account, no-manual-step
        # public install -- that is what the fetcher IS -- so none is gated.
        facts[lane] = {"gb": stated or None, "gated": False, "manual": False}
    for tier, specs in getattr(provision, "MANUAL_TIERS", {}).items():
        total = sum(int(s.get("bytes", 0) or 0) for s in specs)
        facts[tier] = {"gb": round(total / 2 ** 30, 1) or None,
                       "gated": any(s.get("gated") for s in specs),
                       "manual": True}
    if drift:
        raise SystemExit("fetch manifests disagree with LANE_INFO:\n  "
                         + "\n  ".join(drift))
    return facts


#: How an engine with no provisioning lane still gets its weights.
_NO_LANE_WORD = {"hf_cache": "auto", "sidecar": "sidecar",
                 "remote": "none", "manual_doc": "manual", "builtin": "nothing",
                 "remote_unprovisioned": "none*"}

#: One-slot memo for :func:`graph_fetched_engines`.
_GRAPH_FETCHED_CACHE: list = []


def graph_fetched_engines() -> set:
    """Engines the GRAPH downloads for you, read from the one list that decides.

    HAVING A FETCHER LANE IS NOT THE SAME CLAIM AS "auto", and conflating them
    printed the wrong word into two shipped documents (2026-09-12). The README's
    own legend defines auto as "fetched on first use ... just pick it and run".
    But `scripts/` is a development-tree tool that is NOT in the registry
    bundle, and every local video adapter is fail-closed by design -- read
    ``eng_ltx_8gb`` ("the offline invariant -- no runtime fetch") and
    ``eng_ghost_signal`` ("Fail CLOSED ... and NEVER a download"). Six engines
    were being advertised as costing nothing on the strength of a script the
    reader does not have.

    WHAT ACTUALLY FETCHES FOR A LANE is `OTR_WorkflowValidator`, a node inside
    the canonical workflow, which calls
    ``_otr_visual_assets.ensure_prompt_visual_assets`` at queue time. That
    function reduces the graph's selections with ``& _COVERED`` and requests
    weights for exactly what survives -- so ``_COVERED`` IS the answer, by
    construction, and this returns it rather than re-deriving it.

    An earlier cut joined each fetcher lane's filenames against the MANIFEST.
    It got the same three engines and was correct, but it was the wrong
    primitive: it could only ever describe engines that HAVE a lane, and `sd15`
    -- added to the manifest 2026-09-12 precisely because it had none -- would
    have kept reading "manual" while the graph downloaded it.

    **THE SCOPE OF "fetches for a lane" MATTERS.** This is not a claim that the
    manifest is the only download in the pack. ``eng_musicgen`` calls
    ``from_pretrained`` and ``_otr_kokoro_voice_prefetch`` calls
    ``hf_hub_download``; both are real and legitimate. They are also not lanes,
    so they never reach this function -- they resolve through
    ``NO_LANE_REASON["hf_cache"]``, which already says "auto" for that reason.

    **ALSO NOT A CONTRADICTION:** ``otr_provision.profile_lanes`` still reports
    ``humo`` and ``wan_ti2v_gguf`` as "automatic". That answers a DIFFERENT
    question -- can the dev-tree provisioner fetch it during setup -- and it is
    right. Do not "reconcile" ``tests/test_otr_provision_humo.py`` with this
    function; you would break a correct test.

    Memoized because ``friction_for`` asks once per engine and ``_load``
    re-executes a module every call. The memo has no invalidation on purpose --
    every caller today is a fresh process or a fresh ``_generator()`` module --
    so a test that loads this module ONCE and then mutates ``_COVERED`` between
    two scenarios must call ``_GRAPH_FETCHED_CACHE.clear()`` itself, or it will
    read the first answer twice with no error.
    """
    if _GRAPH_FETCHED_CACHE:
        return _GRAPH_FETCHED_CACHE[0]
    assets = _load("nodes/_otr_visual_assets.py", "_odm_visual_assets")
    covered = {str(e) for e in getattr(assets, "_COVERED", ())}
    _GRAPH_FETCHED_CACHE.append(covered)
    return covered


def friction_for(engine: str, namespace: str, facts: dict) -> tuple:
    """``(word, size_gb_or_None, lane_or_None)`` -- how you get the weights."""
    provision = _load("scripts/otr_provision.py", "_odm_provision")
    lane = provision.lane_for_engine(engine, namespace)
    fetched = graph_fetched_engines()
    if lane is None:
        # "Needs nothing" is TWO different things and the columns must not
        # merge them: a procedural visualizer is pure code that runs anywhere,
        # a hosted lane runs nowhere without a credential. Both download zero
        # bytes, which is the only thing they have in common.
        if engine in getattr(provision, "_NO_WEIGHT_VIDEO_ENGINES", set()):
            return ("nothing", None, None)
        return ("none", None, None)
    if lane is provision.UNROUTED:
        # No fetcher lane -- but that is three different situations, and the
        # provisioner declares which. Only an engine it says nothing about is
        # a real gap.
        reason = provision.NO_LANE_REASON.get(engine)
        # THE GRAPH OUTRANKS THE PROVISIONER'S CLASSIFICATION, and `sd15` is why
        # (2026-09-12). It has no fetcher lane and is filed "manual_doc", which
        # was exactly right until the visual-asset manifest started carrying its
        # checkpoint -- at which point the truth a reader needs became "queue it
        # and the graph downloads it", whatever the provisioner calls it. The
        # word has to follow what actually happens on a first run, and no
        # size comes from this path: an engine with no lane has no manifest to
        # sum, so its figure is the curated one, as the module docstring says.
        if engine in fetched:
            return ("auto", None, None)
        return (_NO_LANE_WORD.get(reason, "unrouted"), None, None)
    fact = facts.get(lane.lane, {})
    if lane.manual or engine not in fetched:
        # A lane the render path will not fetch for you is a manual step, even
        # when `scripts/otr_fetch_lane_weights.py` can do it -- see
        # graph_fetched_engines(). The gate is what the graph downloads, not the
        # existence of a lane.
        word = "GATED+manual" if fact.get("gated") else "manual"
    else:
        word = "GATED" if fact.get("gated") else "auto"
    return (word, fact.get("gb"), lane.lane)


def hf_token_engines() -> frozenset:
    """Engines whose weights are HF-GATED, read from the audio profiles.

    `config/audio_engine_profiles.yaml` declares `requires_hf_token` per
    profile, and it is the only structured statement of that fact in the repo.
    Before this was read, the matrix derived `stable_audio_music` as **auto**
    -- whose legend promises "no account and no token; just pick it and run" --
    for a lane whose weights are gated behind a licence click. That is the
    single worst thing this table can say, because a reader picks a music
    engine on exactly that promise.

    Parsed with a regex rather than a YAML library on purpose: this generator
    runs in a docs check with no third-party imports, and the block is a flat
    list of `engine:` / `requires_hf_token:` pairs.
    """
    path = os.path.join(_REPO, "config", "audio_engine_profiles.yaml")
    if not os.path.exists(path):
        return frozenset()
    gated, engine = set(), None
    for line in io.open(path, encoding="utf-8"):
        stripped = line.strip()
        if stripped.startswith("engine:"):
            engine = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("requires_hf_token:") and engine:
            if stripped.split(":", 1)[1].split("#")[0].strip() == "true":
                gated.add(engine)
    return frozenset(gated)


def os_friction() -> dict:
    """``{engine: "windows-only installer"}`` -- derived from what is on disk.

    The capability profiles are about hardware, not operating systems, so an
    8 GB NVIDIA card answers identically on Linux and Windows and a second set
    of columns would be a duplicate. Exactly one thing in the pack IS
    OS-specific: three voice engines install through their own script, and
    those scripts are PowerShell. No `.sh` twin exists, so on Linux and macOS
    the friction is the same one -- and it is friction, not a hardware limit.
    Derived by looking for the file rather than asserted, so writing the
    installer is all it takes to clear the note.
    """
    import glob
    out = {}
    for path in glob.glob(os.path.join(_REPO, "scripts", "_otr_*_install.*")):
        base = os.path.basename(path)
        engine = base[len("_otr_"):base.rindex("_install.")]
        out.setdefault(engine, set()).add(os.path.splitext(base)[1])
    return {engine: "windows-only installer"
            for engine, kinds in out.items() if kinds == {".ps1"}}


def writer_rows() -> list:
    """The WRITER dimension, which this table was silent about entirely.

    THE GAP THIS CLOSES. Every shipped graph needs a writer LLM, and every one
    of the seven episodes published on the M4 used the same one -- yet neither
    this document nor its curated file mentioned writers at all. A per-dropdown
    matrix that omits the dimension carrying the largest single download, and
    the one measured at 14 GB resident on a 16 GB machine, is not a per-dropdown
    matrix. Found by an audit of the published episodes against the table.

    Writers do not live in an engine registry, so they cannot come through
    `registry_capabilities`: they come from the catalog's curated list, and
    their machine fit is the same `fit_tags_for` the dropdown badge uses -- one
    derivation, so the table and the picker cannot disagree.
    """
    cat = _load("nodes/_otr_model_catalog.py", "_odm_catalog")
    curated = load_curated().get("writers", {})
    rows = []
    for m in cat._active_curated_models():
        if getattr(m, "provider", "local") != "local":
            continue          # hosted lanes: a credential, not a download
        repo = m.repo_id
        tags = cat.fit_tags_for(repo)
        gated = (repo in cat.GATED_CURATED_MODELS
                 or bool(getattr(m, "requires_auth", False)))
        hand = curated.get(repo, {})
        rows.append({
            "engine": repo,
            "public": repo,
            "namespace": "writer",
            "friction": "GATED" if gated else "auto",
            "lane": None,
            "size_gb": float(getattr(m, "approx_safetensors_gb", 0.0) or 0.0),
            "size_measured": True,
            # Fit is DERIVED; only the receipts are curated.
            "availability": {mch["key"]: "ok" for mch in MACHINES},
            "memory": hand.get("memory", {}),
            "note": hand.get("note", ""),
            "remote": False,
            "os_note": "",
            "fit_tags": tags,
        })
    return rows


def load_curated() -> dict:
    return json.load(io.open(_CURATED, encoding="utf-8"))


def public_name(engine: str) -> str:
    """The id the OTR_VideoDirector dropdown actually shows, when it differs."""
    pe = _load("nodes/_otr_shared/public_engines.py", "_odm_public")
    return pe._INTERNAL_TO_PUBLIC.get(engine, engine)


def build_rows() -> list:
    """One row per registered engine, derived facts joined to curated ones."""
    caps = registry_capabilities()
    grid = availability_grid(caps)
    facts = download_facts()
    curated = load_curated().get("engines", {})
    os_only = os_friction()
    gated = hf_token_engines()
    rows = []
    for namespace, table in caps.items():
        for engine in sorted(table):
            word, gb, lane = friction_for(engine, namespace, facts)
            # A declared HF gate outranks a derived "auto": the weights may
            # well fetch themselves, but only after a licence click and a
            # token, and that is the thing the reader needs to know first.
            if engine in gated and word in ("auto", "manual"):
                word = "GATED" if word == "auto" else "GATED+manual"
            remote = word in ("none", "none*")
            hand = curated.get(engine, {})
            rows.append({
                "engine": engine,
                "public": public_name(engine),
                "namespace": namespace,
                "friction": word,
                "lane": lane,
                "remote": remote,
                "os_note": os_only.get(engine, ""),
                # A manifest figure beats a curated one every time; the curated
                # size only fills in for engines with no fetch lane at all.
                "size_gb": gb if gb is not None else hand.get("size_gb"),
                "size_measured": gb is not None,
                "availability": grid[(namespace, engine)],
                "memory": hand.get("memory", {}),
                "note": hand.get("note", ""),
            })
    rows.extend(writer_rows())
    return rows


def conflicts(rows: list) -> list:
    """Curated receipts that sit on a cell the code refuses.

    This is the flux2_klein trap, generalized: a table claiming an engine is
    proven somewhere OTR will not offer it. Either the receipt is wrong or the
    declaration is stale, and both are worth stopping a docs build over.
    """
    bad = []
    for row in rows:
        for machine in MACHINES:
            key = machine["key"]
            if row["memory"].get(key) != "proven":
                continue
            reason = row["availability"].get(key)
            if reason != "ok":
                bad.append("%s: curated PROVEN on %s, but the code answers %r "
                           "there. Either the receipt is wrong, or the "
                           "declaration is stale." % (row["engine"], key, reason))
    return bad


def _cell(row: dict, key: str) -> str:
    """One machine cell: selectability first, then the memory verdict."""
    reason = row["availability"].get(key, "??")
    if reason != "ok":
        return _REFUSAL.get(reason, reason)
    if row["remote"]:
        # No local weights and no local compute. What gates a hosted lane is
        # the credential, and it gates it identically on every machine -- five
        # unmeasured question marks would imply a hardware question that is
        # not being asked here.
        return "key"
    if row.get("fit_tags") is not None and not row["memory"].get(key):
        # WRITERS: the arithmetic half is derived, so an uncurated cell is not
        # unknown -- it is "the size says yes/no and nobody has run it". Only
        # a receipt is curated, and a receipt always wins over this.
        #
        # ONLY FOR MACHINES fit_tags_for ACTUALLY EVALUATES. It computes
        # mac16/nv8/nv16/nv24 and says nothing about AMD or CPU, so treating a
        # missing tag as "no" there announced that a 5.2 GB writer cannot run
        # on an AMD card -- inventing a verdict out of a column that was never
        # calculated. Absence of a tag is only evidence where a tag was on
        # offer.
        if key in _WRITER_FIT_KEYS:
            tags = row["fit_tags"]
            if key in tags:
                return "fits"
            if key + "-tight" in tags:
                return "**tight**"
            return "**no**"
        return "?"
    verdict = row["memory"].get(key, "unknown")
    return {"proven": "**proven**", "measured": "measured", "fits": "fits",
            "oom": "**OOM**", "no": "**no**", "unknown": "?"}.get(verdict, verdict)


def _size(row: dict) -> str:
    if row["size_gb"] is None:
        return "--" if row["friction"] in ("none", "none*", "nothing") else "?"
    return "%.1f GiB" % row["size_gb"]


_FRICTION_CELL = {
    "auto": "**auto**", "GATED": "GATED", "manual": "manual",
    "GATED+manual": "GATED + manual", "none": "none", "bundled": "bundled",
    "unrouted": "*no lane*", "sidecar": "own installer", "nothing": "nothing",
    "none*": "none, **but see below**",
}

#: Every friction word that means "a hosted service, not a download". ``none*``
#: is the SAME KIND of thing as ``none`` -- a cloud lane no shipping profile
#: selects yet -- and testing only for ``none`` sent three cloud engines
#: (`cloud_kling_avatar`, `cloud_seedance_2`, `cloud_vidu_q2_pro_fast_720p`)
#: into the catch-all below, so they printed under the heading "local
#: diffusion" while their own cells read "key". The asterisk still marks the
#: gap; it no longer moves them to the wrong table.
_HOSTED_WORDS = ("none", "none*")

_GROUPS = (
    ("Video -- procedural, nothing to download",
     lambda r: r["namespace"] == "video" and r["friction"] == "nothing"),
    ("Video -- hosted, no weights but you supply the key",
     lambda r: r["namespace"] == "video" and r["friction"] in _HOSTED_WORDS),
    ("Video -- local diffusion",
     lambda r: r["namespace"] == "video"
     and r["friction"] not in _HOSTED_WORDS + ("nothing",)),
    ("Image -- local",
     lambda r: r["namespace"] == "image" and r["friction"] not in _HOSTED_WORDS),
    ("Image -- hosted",
     lambda r: r["namespace"] == "image" and r["friction"] in _HOSTED_WORDS),
    ("Voice and music -- local",
     lambda r: r["namespace"] == "audio" and r["friction"] not in _HOSTED_WORDS),
    ("Voice and music -- hosted",
     lambda r: r["namespace"] == "audio" and r["friction"] in _HOSTED_WORDS),
    ("Upscale", lambda r: r["namespace"] == "upscale"),
    ("Writer (the LLM that writes the script)",
     lambda r: r["namespace"] == "writer"),
)


def render_table(rows: list, machines=MACHINES) -> str:
    """The joined table: friction and size, then one column per machine."""
    keys = [m["key"] for m in machines]
    head = ("| dropdown | how you get it | size | "
            + " | ".join(m["label"] for m in machines) + " |\n")
    head += "|---|---|---|" + "---|" * len(machines) + "\n"
    out = []
    for title, keep in _GROUPS:
        group = [r for r in rows if keep(r)]
        if not group:
            continue
        out.append("\n**%s**\n\n" % title)
        out.append(head)
        for row in sorted(group, key=lambda r: (r["size_gb"] or 0, r["public"])):
            out.append("| `%s` | %s | %s | %s |\n" % (
                row["public"],
                _FRICTION_CELL.get(row["friction"], row["friction"])
                + (" (Windows)" if row["os_note"] else ""),
                _size(row),
                " | ".join(_cell(row, k) for k in keys)))
    return "".join(out)


_LEGEND = """
**How you get the weights.** Two things do the fetching for an **auto** row, and
neither of them is a script you have to run: the engine's own library pulls it
through the Hugging Face cache, or `OTR_WorkflowValidator` -- a node inside the
graph -- downloads it at queue time. A **manual** row may still have a helper in
`scripts/`, but `scripts/` is not in the registry bundle, so from a normal
install it is a step you take by hand and it is labelled as one.

**auto** -- fetched on first use, no account and no
token; just pick it and run. **GATED** -- fetches itself, but only after you
accept a licence on the model page and set `HF_TOKEN`. **manual** -- you fetch
it yourself; `docs/MODEL_ASSET_INDEX.md` names the files and where they go.
**none** -- no weights at all. *no lane* -- the engine is registered but no
provisioning lane is declared for it, so nothing will fetch it for you.

**own installer** -- installs through its own script rather than the model
provisioner; **(Windows)** marks the three whose installer is PowerShell with no
`.sh` twin, so on Linux and macOS there is no install path today. That is
packaging, not hardware -- writing the shell installer is what clears it. **nothing** -- pure code; there is nothing to obtain.

Sizes are GiB, summed from the real artifact bytes in the fetch manifests where
a lane carries them, otherwise the figure the fetcher's own pick list states.

**What a machine cell means, and read this before you read one.** Each cell
answers TWO questions in order.

* **not offered** -- OTR will not put this engine in your dropdown on that
  machine, because its declaration does not list that backend. This is a
  statement about the code, **not about your hardware**: several of these have
  run on that hardware, and the declaration is a record of what has been
  PROVEN, not of what is possible. Making one available is a code change plus a
  receipt, not a purchase.
* **too slow** -- offered on a CPU-only box in principle, kept off it because it
  is not practical there.
* Otherwise the engine IS offered, and the word is the memory verdict:
  **proven** (a PUBLISHED EPISODE used it), measured (it ran on that hardware
  in a lab test and worked, but no episode has ever used it), fits (nothing
  blocks it and the arithmetic says it fits -- nobody has run it at all),
  **OOM** (expect to exhaust memory), **?** (offered, nobody has measured it).

**The proven/measured split IS the test plan.** "measured" is precisely the list
of engines to close next, and the distinction was earned: a first pass called
both states "proven", which put engines in the same column as ones that had
carried a whole episode. Note also that an episode's FILENAME records only its
dominant video lane, so counting receipts from filenames under-reports -- one
published episode here ran viz_camera, viz_mxc_cpu and viz_green together.

**The AMD column is the weakest one here, and it is weak by construction.** The
ROCm profiles declare `device_backend: "cuda"`, because that is how ROCm
presents itself to torch -- so every CUDA lane reads as offered there, and the
column is really answering "is this vendor-locked or sidecar-locked?" rather
than "has this been run on AMD?". Nothing in this repo has an AMD receipt. Treat
an AMD cell as the absence of a hard blocker, nothing more.

**On a Mac, OOM is a HARD MACHINE REBOOT, not a failed render** -- unified
memory has no separate pool to exhaust. That is why the Mac column is worth
reading before you pick, and why an unmeasured **?** there deserves more caution
than the same mark on a discrete card.
"""


def render_doc(rows: list) -> str:
    L = ["# Dropdown matrix -- what each choice costs, and what runs it\n\n",
         "**GENERATED by `scripts/otr_dropdown_matrix.py`. Do not hand-edit; "
         "regenerate.** Selectability is derived from each engine's "
         "`CAPABILITIES` row against the shipped profile named in its column; "
         "sizes and gating are read out of the fetch manifests. Only the "
         "memory verdicts and receipts are curated, in "
         "`docs/dropdown_matrix.json`.\n\n",
         "You never need all the weights in this workflow. One graph ships; "
         "the dropdowns decide what it loads, and therefore what you have to "
         "fetch.\n\n",
         "## The machines\n\n",
         "| column | machine | reproduce with |\n|---|---|---|\n"]
    for machine in MACHINES:
        L.append("| %s | %s | `config/profiles/%s.json` |\n" % (
            machine["label"], machine["blurb"], machine["profile"]))
    L.append("\n## Every dropdown\n")
    L.append(render_table(rows))
    L.append(_LEGEND)
    unprov = [r for r in rows if r["friction"] == "none*"]
    if unprov:
        L.append("\n## Hosted lanes no shipped profile provisions\n\n")
        L.append("These need no weights and no VRAM -- only a credential -- and "
                 "the engines themselves are registered and selectable. But no "
                 "shipping profile selects one, so they were never added to the "
                 "provisioner's remote route, and `scripts/otr_provision.py` "
                 "REFUSES a profile that names one rather than guessing. Using "
                 "one today means selecting it in the graph, not through a "
                 "profile. Worth closing before shipping.\n\n")
        for row in sorted(unprov, key=lambda r: r["public"]):
            L.append("* `%s` (%s)\n" % (row["public"], row["namespace"]))

    L.append("\n## Engines with no provisioning lane\n\n")
    orphans = [r for r in rows if r["friction"] == "unrouted"]
    if orphans:
        L.append("These are registered and selectable, but "
                 "`scripts/otr_provision.py` has no lane that fetches their "
                 "weights -- so a profile that selects one fails to "
                 "provision. Worth closing before shipping.\n\n")
        for row in sorted(orphans, key=lambda r: (r["namespace"], r["public"])):
            L.append("* `%s` (%s)\n" % (row["public"], row["namespace"]))
    else:
        L.append("None -- every registered engine routes to a lane or "
                 "declares that it needs no weights.\n")
    return "".join(L)


#: Loader-class prefix -> the pack that provides it, read from the ONE table
#: the runtime error message already uses, so a document and an error can never
#: name different packs. Source: `_PACK_FOR_PREFIX`,
#: `nodes/_otr_video_engines/wrapper_bridge.py`.
def _pack_prefixes() -> tuple:
    bridge = os.path.join(_REPO, "nodes", "_otr_video_engines", "wrapper_bridge.py")
    tree = ast.parse(io.open(bridge, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_PACK_FOR_PREFIX" for t in node.targets):
            return tuple((p, pack) for p, pack, _url in ast.literal_eval(node.value))
    raise SystemExit("wrapper_bridge.py no longer defines _PACK_FOR_PREFIX")


def packs_by_engine() -> dict:
    """``{engine: "ComfyUI-GGUF"}`` -- what to install BEYOND this pack.

    DERIVED, because the hand-typed equivalent has already drifted. Two passes:

    1. A module NEEDS a pack when its own source emits a class whose name
       starts with one of the bridge's prefixes (``ADE_``, ``VHS_``,
       ``UnetLoaderGGUF``, ``CLIPLoaderGGUF``).
    2. A module INHERITS that need when it imports a module which has it --
       ``eng_ghost_signal_official`` subclasses ``GhostSignalEngine`` and names
       no ``ADE_`` class of its own, so a source scan alone would report it as
       needing nothing. This is exactly the case the hand table's own comment
       says to follow, and the case a naive scan gets wrong.

    An engine is attributed to the module whose source contains its id as a
    string literal. A module registering several engines gives all of them the
    same answer, which is correct: they share the graph it emits.
    """
    prefixes = _pack_prefixes()
    # `wrapper_bridge` DEFINES the prefix table, and every registry names every
    # engine id; scanning either credits the whole roster with every pack. That
    # is not a hypothetical -- it is what the first run of this function did.
    skip = {"wrapper_bridge", "registry", "__init__"}
    declares, emits, imports = {}, {}, {}
    for _ns, rel in _REGISTRIES:
        folder = os.path.join(_REPO, os.path.dirname(rel))
        siblings = {f[:-3] for f in os.listdir(folder) if f.endswith(".py")}
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".py") or fn[:-3] in skip:
                continue
            mod = fn[:-3]
            text = io.open(os.path.join(folder, fn), encoding="utf-8").read()
            found = {pack for prefix, pack in prefixes if prefix in text}
            if found:
                emits[mod] = found
            edges = set()
            try:
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        # `from .X import Y` puts X in `module`; `from . import
                        # X` puts it in `names`; `import a.b.X` puts it in the
                        # dotted alias. Collect all three spellings.
                        mod_name = getattr(node, "module", None)
                        if mod_name:
                            edges.add(mod_name.rsplit(".", 1)[-1])
                        for alias in node.names:
                            edges.add(alias.name.rsplit(".", 1)[-1])
            except SyntaxError:
                pass
            imports[mod] = {m for m in siblings
                            if m != mod and m not in skip and m in edges}
            # An adapter DECLARES its id: `name = "z_image_turbo"`. Identity,
            # not "the file mentions this string somewhere".
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for stmt in node.body:
                        if (isinstance(stmt, ast.Assign)
                                and any(getattr(t, "id", "") == "name"
                                        for t in stmt.targets)
                                and isinstance(stmt.value, ast.Constant)
                                and isinstance(stmt.value.value, str)):
                            declares.setdefault(mod, set()).add(stmt.value.value)

    # Propagate along import edges to a fixpoint: a subclass inherits the graph
    # its base emits (eng_ghost_signal_official names no ADE_ class of its own).
    changed = True
    while changed:
        changed = False
        for mod, deps in imports.items():
            inherited = set()
            for dep in deps:
                inherited |= emits.get(dep, set())
            if inherited - emits.get(mod, set()):
                emits.setdefault(mod, set()).update(inherited)
                changed = True

    known = {e for ns in registry_capabilities().values() for e in ns}
    out = {}
    for mod, packs in emits.items():
        for engine in declares.get(mod, set()):
            if engine in known:
                out.setdefault(engine, set()).update(packs)
    return {e: " + ".join(sorted(p)) for e, p in out.items()}


def _variant_for(profile_id: str) -> str:
    """The saved graph a profile emits, by `build_variants.py`'s own rule.

    `build_variants.py:135-148` globs `config/profiles/*.json`, takes each
    stem, and refuses outright when a bare `X` would collide with an existing
    `otr_X`. So the mapping is total and unambiguous: prefix unless already
    prefixed.
    """
    stem = profile_id if profile_id.startswith("otr_") else "otr_%s" % profile_id
    return "workflows/variants/%s.json" % stem


def manual_artifacts(lane: str, provision, fetcher) -> list:
    """Every artifact a manual lane needs, from whichever manifest owns it.

    THREE MANIFESTS, NOT ONE, and reading only the first is what made the
    first cut of section 3 silently omit six of nine manual lanes:

    * ``MANUAL_TIERS`` in the provisioner -- dicts, receipts complete.
    * ``LANES`` in the fetcher -- ``WeightSpec`` namedtuples, except the lanes
      that predate receipts, which are still bare 3-tuples of
      ``(repo, path_in_repo, destination)``. Both shapes are read; a legacy
      tuple simply has no size to report.
    * The OPERATOR-ONLY ALIAS: ``h3_operator_only`` is the provisioner's name
      for artifacts the fetcher keeps under ``minimax_h3``. The mapping is
      declared in ``OPERATOR_ONLY_FETCH_LANES`` and is read from there, so a
      future alias needs no change here.

    Returns ``[]`` when no manifest owns the lane -- the caller must SAY so
    rather than print nothing.
    """
    tiers = getattr(provision, "MANUAL_TIERS", {})
    if lane in tiers:
        return [dict(spec) for spec in tiers[lane]]
    alias = getattr(provision, "OPERATOR_ONLY_FETCH_LANES", {}).get(lane, lane)
    out = []
    for spec in getattr(fetcher, "LANES", {}).get(alias, []) or []:
        if hasattr(spec, "repo"):                 # a WeightSpec
            out.append({"repo": spec.repo, "path": spec.path_in_repo,
                        "destination": spec.destination,
                        "bytes": getattr(spec, "expected_bytes", None) or 0,
                        "gated": False})
        else:                                     # a legacy bare 3-tuple
            parts = list(spec) + ["", "", ""]
            out.append({"repo": parts[0], "path": parts[1],
                        "destination": parts[2], "bytes": 0, "gated": False})
    return out


def render_apple(rows: list) -> str:
    """`apple/MACHINES.md` -- the three questions, in the order people ask them.

    1. Which graph do I open?  2. Will this engine run here?  3. Where do the
    manual weights come from? The third has never been answered anywhere: the
    document the matrix points at for manual rows lists filenames with no
    repository and no destination directory, while `MANUAL_TIERS` has carried
    `repo`, `revision`, `path`, `destination`, `bytes` and `gated` per artifact
    the whole time.
    """
    provision = _load("scripts/otr_provision.py", "_odm_provision")
    packs = packs_by_engine()
    by_engine = {row["engine"]: row for row in rows}
    L = ["# Which machine runs what\n\n",
         "Generated by `scripts/otr_dropdown_matrix.py` -- do not hand-edit; "
         "run `python scripts/otr_dropdown_matrix.py` and the change lands in "
         "every copy at once.\n\n",
         "Three questions, in the order people ask them.\n\n"]

    # ---------------------------------------------------------------- 1. graph
    L.append("## 1. Which graph do I open?\n\n")
    L.append("Every row below is a real saved graph in this repository. Drag it "
             "onto the ComfyUI canvas, or use **Browse Templates** for the "
             "canonical. If your machine is not listed, open the canonical: it "
             "names no vendor anywhere and resolves your device at run time.\n\n")
    L.append("| Your machine | Open this | Also install |\n")
    L.append("|---|---|---|\n")
    L.append("| Anything, to start | `workflows/otr_canonical.json` "
             "(Browse Templates &rarr; OTR) | nothing |\n")
    for machine in MACHINES:
        graph = _variant_for(machine["profile"])
        if not os.path.exists(os.path.join(_REPO, graph)):
            continue
        needed = sorted({packs[e] for e in _profile_engines(machine["profile"])
                         if e in packs})
        L.append("| %s -- %s | `%s` | %s |\n" % (
            machine["label"], machine["blurb"], graph,
            ", ".join(needed) if needed else "nothing"))
    L.append("\nEvery machine needs **ffmpeg and ffprobe** on PATH, and Linux "
             "needs one monospace TTF installed for burned captions.\n\n")
    if packs:
        L.append("If you change a dropdown yourself, these are the only "
                 "picks that need anything beyond this pack. Everything "
                 "not listed here runs on what you already have.\n\n")
        L.append("| If you select | Install |\n|---|---|\n")
        def _label(engine):
            row = by_engine.get(engine)
            return row["public"] if row else engine
        for engine in sorted(packs, key=lambda e: (packs[e], _label(e))):
            L.append("| `%s` | %s |\n" % (_label(engine), packs[engine]))
        L.append("\n")

    # --------------------------------------------------------------- 2. table
    L.append("## 2. Will this engine run on my machine?\n\n")
    L.append(render_table(rows).strip() + "\n\n")

    # ------------------------------------------------------------- 3. weights
    L.append("## 3. Where do the manual weights come from?\n\n")
    L.append("Every file a **manual** row needs: the repository to download it "
             "from, and the folder under your ComfyUI `models/` directory to "
             "put it in. `gated` means you must accept the model's licence on "
             "Hugging Face first, while signed in.\n\n")
    L.append("Two engines can share one group and still download different "
             "amounts, because they draw different files from it. **The size "
             "in section 2 is what YOUR pick costs**; the total on a heading "
             "here is the whole group. A heading with no total means that "
             "group's manifest predates byte receipts -- section 2 still has "
             "the figure.\n\n")
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_odm_fetcher")
    lane_to_engines, unsourced = {}, []
    for row in rows:
        if "manual" not in str(row["friction"]).lower():
            continue
        lane = str(row["lane"]) if row["lane"] else ""
        specs = manual_artifacts(lane, provision, fetcher) if lane else []
        if specs:
            lane_to_engines.setdefault(lane, []).append(row["public"])
        else:
            unsourced.append(row["public"])

    for lane in sorted(lane_to_engines):
        specs = manual_artifacts(lane, provision, fetcher)
        users = sorted(set(lane_to_engines[lane]))
        total = sum(int(s.get("bytes", 0) or 0) for s in specs)
        head = "### %s" % lane
        if total:
            head += " &mdash; %.1f GiB total" % (total / 2 ** 30)
        L.append(head + "\n\n")
        L.append("Selected by: %s\n\n" % ", ".join("`%s`" % u for u in users))
        L.append("| File | From | Put it in | Size | Gated |\n")
        L.append("|---|---|---|---|---|\n")
        for spec in specs:
            dest = str(spec.get("destination") or "")
            folder = dest.rsplit("/", 1)[0] if "/" in dest else "(models root)"
            size = int(spec.get("bytes", 0) or 0)
            L.append("| `%s` | [`%s`](https://huggingface.co/%s) | `models/%s/` "
                     "| %s | %s |\n" % (
                         str(spec.get("path", "")).rsplit("/", 1)[-1],
                         spec.get("repo", "?"), spec.get("repo", ""),
                         folder,
                         "%.2f GiB" % (size / 2 ** 30) if size else "--",
                         "yes" if spec.get("gated") else "no"))
        L.append("\n")

    if unsourced:
        L.append("### No manifest ships for these\n\n")
        L.append("This pack cannot fetch them, and no table here can tell you "
                 "the filename, because the engine chooses it. Select one "
                 "anyway and it refuses by name before anything else runs -- "
                 "**that refusal is the install instruction**: it prints the "
                 "exact file it wants and the folder it expects. It never "
                 "quietly substitutes another.\n\n")
        for public in sorted(set(unsourced)):
            L.append("* `%s`\n" % public)
        L.append("\n")

    L.append("## What the words mean\n\n")
    L.append(_LEGEND.strip() + "\n")
    return "".join(L)


def _profile_engines(profile_id: str) -> set:
    """Every engine id a profile selects, public labels resolved to real ids."""
    path = os.path.join(_REPO, "config", "profiles", "%s.json" % profile_id)
    if not os.path.exists(path):
        return set()
    data = json.load(io.open(path, encoding="utf-8"))
    picked = set()
    for block in ("role_overrides", "slot_overrides"):
        for value in (data.get(block) or {}).values():
            if isinstance(value, str):
                picked.add(value)
    known = {e for ns in registry_capabilities().values() for e in ns}
    public = _load("nodes/_otr_shared/public_engines.py", "_odm_public")
    resolved = set()
    for name in picked:
        if name in known:
            resolved.add(name)
            continue
        # A recipe legitimately names the DROPDOWN LABEL a human sees
        # (`h3_low_video`) rather than the internal id (`minimax_h3_video`).
        # There is exactly one function that maps one to the other; a substring
        # test is not a second one, and using it credited every profile with
        # every engine.
        try:
            engine = public.resolve_engine_id(name)
        except Exception:
            continue
        if engine in known:
            resolved.add(engine)
    return resolved


def render_readme_block(rows: list) -> str:
    """The README injection -- the three machines a stranger is likely on."""
    machines = [m for m in MACHINES if m["key"] in ("nv8", "nv16", "mac16")]
    return ("%s\n\n%s\n%s\n%s\n" % (
        _BEGIN, render_table(rows, machines).strip(), _LEGEND.strip(), _END))


def inject_readme(block: str, write: bool) -> bool:
    text = io.open(_README, encoding="utf-8").read()
    if _BEGIN not in text or _END not in text:
        raise SystemExit("README has no dropdown-matrix markers to inject into")
    head = text[:text.index(_BEGIN)]
    tail = text[text.index(_END) + len(_END):]
    fresh = head + block.rstrip("\n") + tail
    if fresh == text:
        return True
    if write:
        io.open(_README, "w", encoding="utf-8").write(fresh)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit non-zero if anything is stale")
    args = ap.parse_args(argv)

    rows = build_rows()
    bad = conflicts(rows)
    if bad:
        for line in bad:
            print("CONFLICT: " + line)
        return 2

    doc = render_doc(rows)
    block = render_readme_block(rows)
    apple = render_apple(rows)
    current = io.open(_DOC, encoding="utf-8").read() if os.path.exists(_DOC) else None
    current_apple = (io.open(_APPLE, encoding="utf-8").read()
                     if os.path.exists(_APPLE) else None)

    if args.check:
        stale = []
        if current != doc:
            stale.append("docs/DROPDOWN_MATRIX.md")
        if current_apple != apple:
            stale.append("apple/MACHINES.md")
        if not inject_readme(block, write=False):
            stale.append("README's dropdown-matrix block")
        if stale:
            print("STALE: " + ", ".join(stale))
            print("Regenerate: python scripts/otr_dropdown_matrix.py")
            return 1
        print("dropdown matrix is in sync (%d engines)" % len(rows))
        return 0

    io.open(_DOC, "w", encoding="utf-8").write(doc)
    os.makedirs(os.path.dirname(_APPLE), exist_ok=True)
    io.open(_APPLE, "w", encoding="utf-8").write(apple)
    inject_readme(block, write=True)
    print("wrote %s (%d bytes) and %s (%d bytes), injected the README block; "
          "%d engines" % (os.path.relpath(_DOC, _REPO), len(doc),
                          os.path.relpath(_APPLE, _REPO), len(apple), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
