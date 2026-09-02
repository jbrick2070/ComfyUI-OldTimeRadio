#!/usr/bin/env python
"""Generate `docs/MACHINE_MATRIX.md` -- what runs on which machine.

    python scripts/otr_machine_matrix.py            # write the doc
    python scripts/otr_machine_matrix.py --stdout   # print it

WHY THIS IS A MATRIX AND NOT PROSE. Operator, 2026-08-31, after reading the
restructured pod guide: *"i feel its more of a history and not a true guide to
all computers -- they don't need a story, just a matrix of what we think will
work and what has been tested."* He is right. A reader arriving with a card in
their machine wants one row, not a narrative of how the row was discovered.

WHY IT IS GENERATED. The stranger-facing machine answer, proof receipts, and
measurements come from `config/machine_classes.json`; experimental profile
detail comes from `config/profiles/*.json`. A hand-written compatibility table
is the single most rot-prone document a project can own.

THE CONFIDENCE LEVELS ARE NOT THE SAME CLAIM, and the table says which:
  * `shipping` / `draft` come from the profile's own `status` field -- the
    project's standing judgement about whether a combination is ready.
  * PROVEN means an episode actually rendered and published, with the evidence
    named in the notes below the table. That is a much stronger claim than
    `shipping`, and only a handful of rows have it.
  * LAB-PROVEN means an isolated recipe produced receipt-bearing media on the
    named physical hardware. It does not promote a full OTR profile.
Nothing here is inferred from "it looks like it should fit". A blank is an
honest unknown.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: Machine classes are DECLARED in config/machine_classes.json, not here.
#: Facts in a generator rot invisibly -- nobody greps a script for truth -- so
#: the judgement lives in data and this module only validates and renders it.
_CLASS_FILE = os.path.join(_REPO, "config", "machine_classes.json")

#: Not a profile: a mapping config that lives in the same directory and would
#: otherwise render as a "draft profile not vouched for".
_NOT_PROFILES = {"widget_mapping"}


class ClassValidationError(Exception):
    """A declared machine class is incomplete, duplicated, or impossible.

    Raised rather than warned. A compatibility table that quietly disagrees
    with the profiles is worse than no table: README told users an 8 GB card
    had "rendered nothing" after six documented episodes published from one.
    """


class EvidenceValidationError(Exception):
    """An engine proof row is incomplete, duplicated, or overclaims its scope."""


_EVIDENCE_LEVELS = {"PROVEN", "LAB-PROVEN"}


def load_engine_evidence() -> list:
    """Load and validate receipt-backed engine/hardware proof rows.

    This is deliberately separate from machine-class proof. A class receipt
    says its recommended full profile published; an engine evidence row can
    also preserve a narrower isolated lab result without promoting that profile.
    """
    try:
        doc = json.load(io.open(_CLASS_FILE, encoding="utf-8"))
    except OSError:
        return []
    out, problems, seen = [], [], set()
    for index, row in enumerate(doc.get("engine_evidence", [])):
        if not isinstance(row, dict):
            problems.append("engine_evidence[%d] is not an object" % index)
            continue
        missing = [key for key in (
            "engine", "label", "level", "hardware", "date", "scope", "evidence"
        ) if not row.get(key)]
        if missing:
            problems.append("engine_evidence[%d] is missing %s"
                            % (index, ", ".join(missing)))
            continue
        level = str(row["level"]).upper()
        if level not in _EVIDENCE_LEVELS:
            problems.append("engine_evidence[%d] has unsupported level %r"
                            % (index, row["level"]))
            continue
        count_key = "episodes" if level == "PROVEN" else "artifacts"
        try:
            count = int(row.get(count_key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            problems.append("engine_evidence[%d] %s needs a positive %s count"
                            % (index, level, count_key))
            continue
        identity = (row["engine"], level, row["hardware"], row["date"])
        if identity in seen:
            problems.append("duplicate engine evidence row %r" % (identity,))
            continue
        seen.add(identity)
        item = dict(row)
        item["level"] = level
        out.append(item)
    if problems:
        raise EvidenceValidationError(
            "config/machine_classes.json has invalid engine evidence:\n  - "
            + "\n  - ".join(problems))
    return out


def _md_cell(value) -> str:
    """Keep config prose inside one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def load_classes(_profiles=None):
    """Load complete, self-contained machine rows; no profile indirection."""
    try:
        doc = json.load(io.open(_CLASS_FILE, encoding="utf-8"))
    except OSError:
        return []
    out, problems, seen = [], [], set()
    defaults = doc.get("defaults") or {}
    required = (
        "key", "label", "gpu_vendor", "vram_min_gb",
        "writer", "writer_model", "writer_ceiling_gb", "video", "image",
        "char_voice",
    )
    for index, row in enumerate(doc.get("classes", [])):
        if not isinstance(row, dict):
            problems.append("classes[%d] is not an object" % index)
            continue
        merged = dict(defaults)
        merged.update(row)
        missing = [name for name in required if merged.get(name) in (None, "")]
        if missing:
            problems.append("classes[%d] is missing %s"
                            % (index, ", ".join(missing)))
            continue
        if "vram_max_gb" not in row:
            problems.append(
                "classes[%d] is missing vram_max_gb (use null for open-ended)"
                % index)
            continue
        key = str(row["key"])
        if key in seen:
            problems.append("duplicate machine key %r" % key)
            continue
        seen.add(key)
        try:
            lo = float(row["vram_min_gb"])
            hi = (None if row["vram_max_gb"] is None
                  else float(row["vram_max_gb"]))
        except (TypeError, ValueError):
            problems.append("machine %r has a non-numeric VRAM range" % key)
            continue
        if lo <= 0 or (hi is not None and hi < lo):
            problems.append("machine %r has impossible VRAM range %s-%s"
                            % (key, row["vram_min_gb"], row["vram_max_gb"]))
            continue
        receipts = row.get("proven") or []
        if receipts and not row.get("proof_summary"):
            problems.append("machine %r has receipts but no proof_summary" % key)
            continue
        bad_receipt = False
        for receipt_index, receipt in enumerate(receipts):
            if not isinstance(receipt, dict):
                problems.append("machine %r receipt[%d] is not an object"
                                % (key, receipt_index))
                bad_receipt = True
                continue
            receipt_missing = [name for name in (
                "hardware", "episodes", "date", "scope", "evidence"
            ) if not receipt.get(name)]
            try:
                episode_count = int(receipt.get("episodes") or 0)
            except (TypeError, ValueError):
                episode_count = 0
            if receipt_missing or episode_count <= 0:
                detail = ("missing %s" % ", ".join(receipt_missing)
                          if receipt_missing else "non-positive episode count")
                problems.append("machine %r receipt[%d] has %s"
                                % (key, receipt_index, detail))
                bad_receipt = True
        if bad_receipt:
            continue
        out.append(row)
    if problems:
        raise ClassValidationError(
            "config/machine_classes.json has invalid machine rows:\n  - "
            + "\n  - ".join(problems))
    return out

#: Hardware episode receipts live on the declared class row, in its `proven`
#: list. Each one carries an exact scope; their presence does NOT silently
#: certify the complete current tuple. Engine-level evidence remains separate.
#:
#: Keeping it honest matters more than keeping it full: PROVEN is the difference
#: between "we think" and "we know", and a padded list destroys the only reason
#: the table is worth reading.
def merged_row(row):
    """defaults <- row, through the ONE merge the profile builder uses.

    Reused from otr_machine_profile rather than repeated here: a second
    merge would be a second definition of what a machine class means, and
    the doc and the applied profile could then disagree without either
    being wrong on its own terms.
    """
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from otr_machine_profile import _merged, load_matrix
    return _merged(load_matrix(), row)


def proven_receipts(obj) -> list:
    """Proof receipts, read from a MATRIX ROW.

    Deliberately not from a profile file. Operator, 2026-08-31: "no profiles,
    no proven in the code, they just live there" -- there being
    config/machine_classes.json. Putting receipts on profiles also broke
    `build_variants --check` on an unknown-key guard, which was that guard
    correctly refusing a shape nobody had declared.
    """
    return [r for r in ((obj or {}).get("proven") or []) if isinstance(r, dict)]


def load_measurements() -> list:
    """Load exact measured values and their conditions from matrix data."""
    try:
        doc = json.load(io.open(_CLASS_FILE, encoding="utf-8"))
    except OSError:
        return []
    out, problems = [], []
    for index, row in enumerate(doc.get("measurements", [])):
        if not isinstance(row, dict):
            problems.append("measurements[%d] is not an object" % index)
            continue
        missing = [name for name in ("engine", "conditions", "measured")
                   if not row.get(name)]
        if missing:
            problems.append("measurements[%d] is missing %s"
                            % (index, ", ".join(missing)))
            continue
        out.append(dict(row))
    if problems:
        raise EvidenceValidationError(
            "config/machine_classes.json has invalid measurements:\n  - "
            + "\n  - ".join(problems))
    return out


def load_profiles() -> list:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import otr_provision as provision

    out = []
    for path in sorted(glob.glob(os.path.join(_REPO, "config/profiles/*.json"))):
        try:
            d = json.load(io.open(path, encoding="utf-8"))
        except Exception:
            continue
        pid = d.get("id") or os.path.basename(path)[:-5]
        if pid in _NOT_PROFILES:
            continue
        ro = d.get("role_overrides", {}) or {}
        so = d.get("slot_overrides", {}) or {}
        try:
            routes = provision.profile_lanes(d)
        except provision.ProvisionFailure:
            install_recipe = "missing exact owner"
        else:
            manual = set(routes.get("manual") or [])
            if "h3_operator_only" in manual:
                install_recipe = "operator-only files"
            elif manual:
                install_recipe = "complete; manual tier"
            else:
                install_recipe = "complete"
            if provision.profile_python_issue(d, (3, 14)):
                install_recipe += "; Python <=3.13"
        out.append({
            "id": pid,
            "status": d.get("status", "?"),
            "vram": (d.get("llm", {}) or {}).get("vram_ceiling_gb"),
            "vendor": d.get("gpu_vendor", "?"),
            "backend": d.get("device_backend", "?"),
            "platform": d.get("platform", "any"),
            "video": ro.get("character_visual") or so.get("video_render_engine") or "-",
            "image": ro.get("character_image") or "-",
            "voice": so.get("char_voice_engine") or "-",
            "music": so.get("music_engine") or "-",
            "proven": d.get("proven") or [],
            "install_recipe": install_recipe,
        })
    return out


def _tier(vram):
    """Group experimental profiles by their declared writer VRAM ceiling."""
    if not vram:
        return "unstated"
    ceiling = float(vram)
    if ceiling <= 9:
        return "8 GB"
    # A 14.5 GB writer ceiling is the established 16 GB-card class: the
    # process still needs allocator/headroom outside that declared writer
    # budget. The middle machine row deliberately caps its writer at 10 GB;
    # experimental 10-12 GB ceilings stay in that detail band.
    if ceiling <= 12:
        return "10-15 GB"
    return "16 GB+"


_ORDER = ["8 GB", "10-15 GB", "16 GB+", "unstated"]


def render() -> str:
    profs = load_profiles()
    L = []
    A = L.append
    A("# Machine matrix -- what runs where\n")
    A("**GENERATED by `scripts/otr_machine_matrix.py` from "
      "`config/machine_classes.json` and `config/profiles/*.json`. "
      "Do not hand-edit; regenerate.**\n")
    A("Find your listed machine in **What works on what machine**. Apple "
      "Silicon and CPU-only systems still have experimental profiles rather "
      "than a stranger-facing machine key; find those in the tier details.\n")

    # ---- evidence first, then the front-door machine answer ----------------
    classes = load_classes()
    A("## Engine proof by hardware\n")
    A("Only receipt-backed rows appear here. **PROVEN** means a full OTR "
      "episode published on the named hardware. **LAB-PROVEN** means an "
      "isolated recipe produced valid media there; it does not promote the "
      "full OTR profile. A setup, model load, queued prompt, reserve clamp, or "
      "run still in progress is unqualified and does not appear.\n")
    A("| engine | proof | hardware | exact scope | receipt |")
    A("|---|---|---|---|---|")
    level_order = {"PROVEN": 0, "LAB-PROVEN": 1}
    for row in sorted(load_engine_evidence(),
                      key=lambda r: (level_order[r["level"]], r["engine"],
                                     r["hardware"], r["date"])):
        A("| `%s` (%s) | **%s** | %s | %s | %s |" % (
            _md_cell(row["engine"]), _md_cell(row["label"]), row["level"],
            _md_cell(row["hardware"]), _md_cell(row["scope"]),
            _md_cell(row["evidence"])))
    A("\nPROVEN episode counts come from `delivered_engine` in the episode "
      "ledger for episodes with a final mp4. LAB-PROVEN rows instead cite "
      "their immutable runner receipt and exact artifact scope. Recency "
      "matters because the code moves -- an engine proven in June is proven "
      "against June.\n")
    A("Do not grep `episode_canon.json` for engine names: it records none, "
      "and matches land in PROSE -- searching it for `humo` finds the word "
      "`humorous` and invents a receipt.\n")
    A("## What works on what machine\n")
    A("| your machine | writer | video | voice | music | image | status |")
    A("|---|---|---|---|---|---|---|")
    for row in classes:
        label = row.get("label", "?")
        # Read the ROW, never a profile it may not name. Every machine value
        # lives in config/machine_classes.json now (operator, 2026-08-31: "no
        # profiles in the code"), and no row carries `recommended`, so the old
        # profile-linked branch printed "no profile yet" for ALL FOUR classes
        # while the matrix held real, proven values. The front-door table of
        # the guide told every newcomer that nothing runs anywhere.
        cells = merged_row(row)
        conf = (row.get("proof_summary")
                or "`%s`, unproven" % cells.get("status", "draft"))
        A("| **%s** | %s | %s | %s | %s | %s | %s |" % (
            label, cells.get("writer", "--"), cells.get("video", "--"),
            cells.get("char_voice", "--"), cells.get("music", "--"),
            cells.get("image", "--"), conf))
    A("")
    A("**Use the machine key, not an experimental profile name.** Run these "
      "with the exact Python executable that launches ComfyUI (shown as "
      "`<ComfyUI Python>`). Preview the install plan first, then run the same "
      "command without `--list` to install it.\n")
    for row in classes:
        A("* **%s** -> `<ComfyUI Python> scripts/otr_provision.py --machine %s --list`"
          % (row.get("label"), row.get("key")))
    A("\nProvisioning installs and verifies artifacts; it does not rewrite the "
      "saved graph. To apply one row atomically to the real canonical workflow "
      "on a normal port-8188 ComfyUI server, run `<ComfyUI Python> scripts/"
      "otr_canonical_api_run.py --comfyui-url http://127.0.0.1:8188 "
      "--machine 8gb --act-count 1 --source-bank original --visual-style "
      "sci_fi_radio --timeout 0`, replacing only the exact machine key. To use "
      "an explicit profile instead, replace `--machine 8gb` with `--profile "
      "<exact-profile-id>`; the two selectors are intentionally exclusive. "
      "Every machine row selects the Kokoro voice. On the Python 3.13 that "
      "ComfyUI Desktop and the portable build ship it runs through kokoro-onnx "
      "on the CPU (the same voices, about six times faster than realtime); on "
      "Python 3.12 through the torch kokoro package. Python 3.14 has no kokoro "
      "backend packaged yet; there, run `--profile otr_4060_floor` for the bark "
      "route or switch the OTR_CastLock voice dropdowns to bark.")
    A("\nApple Silicon is still the unproven experimental `otr_mac_mps` profile; "
      "CPU-only is `cpu_floor`. Neither is promoted to a machine key or "
      "PROVEN until a named physical system publishes an episode.\n")
    A("")

    A("## How to read the confidence column\n")
    A("| value | means |")
    A("|---|---|")
    A("| **PROVEN** | an episode actually rendered and published. Evidence named below. |")
    A("| **LAB-PROVEN** | an isolated recipe produced receipt-bearing media on named physical hardware; not a full OTR episode. |")
    A("| **EPISODE PATH PROVEN** | the components actually invoked by the named episode path published; an unused configured lane is not included. |")
    A("| **COMPONENTS PROVEN** | named components published on named hardware, but this exact row as one tuple is not certified. |")
    A("| `shipping` | the profile is considered runtime-ready on a preloaded machine. It is neither hardware proof nor a complete clean-install claim. |")
    A("| `draft` | exists, not vouched for. Try it; expect to debug. |")
    A("")
    A("Nothing here is inferred from \"it looks like it should fit\". A blank is "
      "an unknown, recorded as one.\n")

    by = {}
    for p in profs:
        by.setdefault(_tier(p["vram"]), []).append(p)

    for tier in _ORDER:
        rows = by.get(tier) or []
        if not rows:
            continue
        ship = sum(1 for r in rows if r["status"] == "shipping")
        A("## %s  --  %d experimental profile(s), %d shipping\n"
          % (tier, len(rows), ship))
        # Only PROVEN and shipping rows are tabled. A tier holding 76 drafts is
        # a dump, not a guide: a reader picking a row cannot tell which of 76 to
        # trust. Drafts are counted and folded away, listed by the engine they
        # select, which is the only thing anyone scans them for.
        headline = [r for r in rows if r["status"] == "shipping"]
        drafts = [r for r in rows if r not in headline]
        if headline:
            A("| profile | video | voice | music | image | confidence | install recipe |")
            A("|---|---|---|---|---|---|---|")
            for r in sorted(headline, key=lambda x: x["id"]):
                conf = "`%s`" % r["status"]
                A("| `%s` | %s | %s | %s | %s | %s | %s |" % (
                    r["id"], r["video"], r["voice"], r["music"], r["image"],
                    conf, r["install_recipe"]))
            A("")
        else:
            A("**No shipping experimental profile at this tier.**\n")
        if drafts:
            eng = sorted({r["video"] for r in drafts})
            A("<details><summary>%d draft profile(s) here -- not vouched for"
              "</summary>\n" % len(drafts))
            A("Video engines they select: %s\n"
              % ", ".join("`%s`" % e for e in eng))
            A("| profile | video | voice |")
            A("|---|---|---|")
            for r in sorted(drafts, key=lambda x: x["id"]):
                A("| `%s` | %s | %s |" % (r["id"], r["video"], r["voice"]))
            A("\n</details>\n")

    _voice_engines_table(A)

    A("## A bigger card does not currently get you more\n")
    A("The tier is `16 GB+` because that is the truth: nothing in "
      "`config/profiles/` declares a VRAM ceiling above 16, so a 24 GB or "
      "32 GB card runs exactly what a 16 GB one runs.\n")
    A("There is currently no separate 24/32 GB machine key or heavy-rental "
      "profile. More memory gives headroom, but the install planner still "
      "selects the 16 GB+ row. A future larger recipe belongs here only after "
      "its config and reproducible receipt both exist.\n")
    A("**That matters when you are paying by the hour.** A rented 24 GB card "
      "ran the 16 GB haunted profile and peaked at 15,990 MB. Rented Ampere "
      "has since published both Wan 2.2 TI2V and LTX-2b, proving useful reach "
      "beyond the floor lane. A bigger card still does not auto-select HuMo or "
      "LTX 2.5: choose an explicit qualification profile and preserve its exact "
      "hardware/software/RAM receipt.\n")

    A("## Hardware episode receipts, with their exact scope\n")
    for row in classes:
        for r in proven_receipts(row):
            A("* **%s** on %s -- %s episode(s), %s. Scope: %s. Evidence: %s"
              % (row.get("label", row.get("key", "?")),
                 r.get("hardware", "?"), r.get("episodes", "?"),
                 r.get("date", "?"), r.get("scope", "?"),
                 r.get("evidence", "")))
    A("")
    A("## Measured peaks, with their conditions\n")
    A("A VRAM number without its conditions is how somebody buys the wrong card.\n")
    A("| engine | conditions | measured |")
    A("|---|---|---|")
    for row in load_measurements():
        A("| `%s` | %s | %s |" % (
            _md_cell(row["engine"]), _md_cell(row["conditions"]),
            _md_cell(row["measured"])))
    A("")
    A("### Read every VRAM number with suspicion\n")
    A("**A peak is mostly what the allocator GRABBED, not what the model "
      "NEEDED.** Engines often take available VRAM and recover under real "
      "pressure; Torch's caching allocator may then retain it. A reserve clamp "
      "on a larger card is not a physical smaller-card receipt. Prefer actual "
      "card runs such as the retained RTX 4060 H3 artifacts when making a "
      "compatibility claim.\n")
    A("The honest use of these numbers is COMPARATIVE -- which lane is "
      "heavier than which -- not a shopping threshold. A lane marked "
      "PROVEN on 8 GB is worth more than any peak figure, because a card "
      "actually did it.\n")
    A("**Host RAM is the limit people miss.** The HuMo 14B receipt used "
      "27.53 GiB of system RAM while using 13.06 GiB VRAM. Its public recipe "
      "therefore asks for at least 32 GiB host RAM. The legal-length H3 receipt "
      "did not capture host RAM, so it provides no H3 host-memory minimum.\n")
    # KNOWN LIMITS ARE DECLARED IN config/machine_classes.json AND WERE A DEAD
    # CHANNEL: the field was read by nothing and rendered nowhere, so a limit
    # someone took the trouble to write down never reached the person it was
    # written for. A limit nobody sees is not a limit.
    try:
        with open(_CLASS_FILE, encoding="utf-8") as fh:
            _limits = (json.load(fh) or {}).get("known_limits") or []
    except Exception:  # noqa: BLE001 -- the doc still renders without them
        _limits = []
    if _limits:
        A("## Known limits, written down when they were found\n")
        for _lim in _limits:
            A("* %s\n" % _lim)

    A("## What is NOT here\n")
    A("* No row means an episode will look good -- only that the combination "
      "loads and runs. Output quality is a separate judgement.\n")
    A("* A `draft` row is not a promise. Most have never been run end to end.\n")
    A("* `install recipe` describes whether every selected artifact has one "
      "exact automatic or manual owner. It is not a hardware result.\n")
    A("* Non-NVIDIA is largely unexplored. Profiles exist "
      "(`otr_amd8_rocm`); none is proven.\n")
    A("* Where a lane needs weights, see "
      "[MODEL_ASSET_INDEX.md](MODEL_ASSET_INDEX.md) for the exact files and "
      "where to get them. That generated index is the cross-reference owner; "
      "the single RunPod playbook may repeat exact manual recipes where a "
      "stranger needs them at install time.\n")
    return "\n".join(L) + "\n"


_BEGIN = "<!-- BEGIN GENERATED: machine-matrix -->"
_END = "<!-- END GENERATED: machine-matrix -->"


def headline_block() -> str:
    """Just the class table -- what gets injected into README.

    README is the universal front door and already answers "what do I run on my
    machine". Two hand-eyed surfaces answering one question is how it came to
    claim an 8 GB card had "rendered nothing" while six documented episodes had published
    from one. So the answer is generated ONCE and injected, rather than told
    twice.
    """
    full = render()
    start = full.index("## What works on what machine")
    end = full.index("## How to read the confidence column")
    return full[start:end].rstrip() + "\n"


def inject_readme(check_only: bool = False) -> bool:
    """Put the class table into README between markers. True if in sync."""
    path = os.path.join(_REPO, "README.md")
    s = io.open(path, encoding="utf-8").read()
    block = _BEGIN + "\n\n" + headline_block() + "\n" + _END
    if _BEGIN in s and _END in s:
        head, rest = s.split(_BEGIN, 1)
        _stale, tail = rest.split(_END, 1)
        out = head + block + tail
    else:
        hook = "## Which video models fit your card"
        if hook not in s:
            print("  README hook not found; nothing injected")
            return True
        out = s.replace(hook, block + "\n\n" + hook, 1)
    if out == s:
        return True
    if check_only:
        return False
    io.open(path, "w", encoding="utf-8", newline="\n").write(out)
    return False



# Which voice engines a fresh install can use without doing anything, and what
# the others need. GENERATED from the audio registry so the table cannot drift
# from the code; the ships column is the one hand-kept fact per engine, and it
# is a fact about packaging, not about capability.
_VOICE_SHIP_NOTES = {
    "kokoro": "ships: `kokoro` (torch) on Python 3.12, `kokoro-onnx` (CPU) on 3.13; "
              "voices and the ONNX model fetch once at boot",
    "bark": "ships: weights download on first use (about 4 GB)",
    "indextts2": "install it yourself: `scripts/_otr_indextts2_install.ps1` plus your "
                 "own reference WAVs (voice cloning)",
    "chatterbox": "install it yourself: isolated sidecar venv, reference WAVs",
    "dia": "install it yourself: isolated sidecar venv, reference WAVs",
    "elevenlabs": "your own API key (cloud)",
    "google_tts": "your own API key (cloud)",
}


def _voice_engines_table(A) -> None:
    """Every registered voice engine, from `nodes/_otr_audio_engines/registry.py`."""
    try:
        import os as _os
        import sys as _sys

        _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from nodes._otr_audio_engines.registry import CAPABILITIES, _REGISTRY
    except Exception as exc:  # noqa: BLE001 -- the matrix must still generate
        A("## Voice engines\n")
        A("(registry unavailable in this interpreter: %s)\n" % exc)
        return
    A("## Voice engines\n")
    A("What each voice engine needs, read from the audio registry. Kokoro is the "
      "shipped default for both voice slots on every machine row; the others stay in "
      "the `OTR_CastLock` dropdowns as install-it-yourself upgrades.\n")
    A("| engine | roles | runs on | usable without a GPU | sidecar / vendor | ships with the pack |")
    A("|---|---|---|---|---|---|")
    names = sorted(
        n for n, e in _REGISTRY.items()
        if {"char_voice", "announcer_voice"} & set(getattr(e, "roles", ()) or ()))
    for name in names:
        cap = CAPABILITIES.get(name) or {}
        roles = ", ".join(r.replace("_voice", "") for r in getattr(_REGISTRY[name], "roles", ()))
        backends = ", ".join(cap.get("device_backends") or []) or "-"
        cpu_ok = "yes" if cap.get("practical_without_gpu") else "no"
        side = "sidecar" if cap.get("requires_sidecar") else "in-process"
        if cap.get("requires_vendor"):
            side += ", %s only" % cap["requires_vendor"]
        A("| `%s` | %s | %s | %s | %s | %s |" % (
            name, roles, backends, cpu_ok, side,
            _VOICE_SHIP_NOTES.get(name, "install it yourself")))
    A("")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="fail if the doc or README block is out of date; "
                         "writes nothing. For CI and the test suite.")
    args = ap.parse_args(argv)
    text = render()
    dest = os.path.join(_REPO, "docs", "MACHINE_MATRIX.md")

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        try:
            on_disk = io.open(dest, encoding="utf-8").read()
        except OSError:
            on_disk = ""
        doc_ok = on_disk == text
        readme_ok = inject_readme(check_only=True)
        if doc_ok and readme_ok:
            print("  in sync")
            return 0
        if not doc_ok:
            print("  STALE: docs/MACHINE_MATRIX.md differs from the profiles")
        if not readme_ok:
            print("  STALE: README's machine-matrix block differs")
        print("  run: python scripts/otr_machine_matrix.py")
        return 1

    io.open(dest, "w", encoding="utf-8", newline="\n").write(text)
    inject_readme()
    print("wrote %s (%d bytes) and injected the README block" % (dest, len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
