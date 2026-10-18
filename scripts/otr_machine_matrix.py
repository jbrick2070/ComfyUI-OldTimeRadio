#!/usr/bin/env python
"""Generate `apple/MACHINE_MATRIX.md` -- what runs on which machine.

    python scripts/otr_machine_matrix.py            # write the doc
    python scripts/otr_machine_matrix.py --stdout   # print it

WHY THIS IS A MATRIX AND NOT PROSE. Operator, 2026-08-31, after reading the
restructured pod guide: *"i feel its more of a history and not a true guide to
all computers -- they don't need a story, just a matrix of what we think will
work and what has been tested."* He is right. A reader arriving with a card in
their machine wants one row, not a narrative of how the row was discovered.

WHY IT IS GENERATED. Everything comes from `config/workflow_matrix.json`: the
workflows themselves and the receipts beside them (engine evidence, hardware
episode receipts, measured peaks, known limits). There are no machine classes
and no profiles (operator, 2026-10-18: "no machine classes, no profiles, just
JSONs"). A hand-written compatibility table is the single most rot-prone
document a project can own.

THE CONFIDENCE LEVELS ARE NOT THE SAME CLAIM, and the table says which:
  * `shipping` / `draft` come from the workflow's own `status` field -- the
    project's standing judgement about whether a combination is ready.
  * PROVEN means an episode actually rendered and published, with the evidence
    named in the notes below the table. That is a much stronger claim than
    `shipping`, and only a handful of rows have it.
  * LAB-PROVEN means an isolated recipe produced receipt-bearing media on the
    named physical hardware. It does not promote a full OTR workflow.
Nothing here is inferred from "it looks like it should fit". A blank is an
honest unknown.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: The receipts live in config/workflow_matrix.json beside the workflows, not
#: here. Facts in a generator rot invisibly -- nobody greps a script for truth --
#: so the judgement lives in data and this module only validates and renders it.
_MATRIX_FILE = os.path.join(_REPO, "config", "workflow_matrix.json")

#: Not a profile: a mapping config that lives in the same directory and would
#: otherwise render as a "draft workflow not vouched for".
_NOT_PROFILES = {"widget_mapping"}


class EvidenceValidationError(Exception):
    """An engine proof row is incomplete, duplicated, or overclaims its scope."""


_EVIDENCE_LEVELS = {"PROVEN", "LAB-PROVEN"}


def load_engine_evidence() -> list:
    """Load and validate receipt-backed engine/hardware proof rows.

    This is deliberately separate from the hardware episode receipts. A
    hardware receipt says full workflows published; an engine evidence row can
    also preserve a narrower isolated lab result without promoting a workflow.
    """
    try:
        doc = json.load(io.open(_MATRIX_FILE, encoding="utf-8"))
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
            "config/workflow_matrix.json has invalid engine evidence:\n  - "
            + "\n  - ".join(problems))
    return out


def _md_cell(value) -> str:
    """Keep config prose inside one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def load_hardware_receipts() -> list:
    """Hardware episode receipts: which workflows published on which card.

    Each one carries an exact scope; its presence does NOT silently certify the
    complete current tuple. Engine-level evidence stays separate. Keeping it
    honest matters more than keeping it full: PROVEN is the difference between
    "we think" and "we know".
    """
    try:
        doc = json.load(io.open(_MATRIX_FILE, encoding="utf-8"))
    except OSError:
        return []
    out, problems = [], []
    for index, row in enumerate(doc.get("hardware_receipts", [])):
        if not isinstance(row, dict):
            problems.append("hardware_receipts[%d] is not an object" % index)
            continue
        missing = [name for name in (
            "workflows", "hardware", "episodes", "date", "scope", "evidence"
        ) if not row.get(name)]
        try:
            episodes = int(row.get("episodes") or 0)
        except (TypeError, ValueError):
            episodes = 0
        if missing or episodes <= 0:
            problems.append("hardware_receipts[%d] has %s" % (
                index, "missing %s" % ", ".join(missing) if missing
                else "a non-positive episode count"))
            continue
        out.append(dict(row))
    if problems:
        raise EvidenceValidationError(
            "config/workflow_matrix.json has invalid hardware receipts:\n  - "
            + "\n  - ".join(problems))
    return out


def load_known_limits() -> list:
    """Failures written down when they were found -- a record, not a gate."""
    try:
        doc = json.load(io.open(_MATRIX_FILE, encoding="utf-8"))
    except OSError:
        return []
    return [str(item) for item in (doc.get("known_limits") or []) if item]


def load_measurements() -> list:
    """Load exact measured values and their conditions from matrix data."""
    try:
        doc = json.load(io.open(_MATRIX_FILE, encoding="utf-8"))
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
            "config/workflow_matrix.json has invalid measurements:\n  - "
            + "\n  - ".join(problems))
    return out


def load_workflow_rows() -> list:
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import otr_provision as provision

    # EVERY WORKFLOW THE MATRIX KNOWS -- not a folder glob.
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from nodes._otr_shared.workflow_rows import (
        WorkflowError, known_workflow_ids)
    from nodes._otr_workflow_apply import resolved_workflow

    canonical = json.load(io.open(
        os.path.join(_REPO, "workflows", "otr_canonical.json"), encoding="utf-8"))

    out = []
    for pid in known_workflow_ids():
        if pid in _NOT_PROFILES:
            continue
        # RESOLVED, not raw. This table prints a workflow's video lane, image lane,
        # voice and ceiling; a matrix row omits whichever of those it inherits, so a
        # raw read would print "-" for values the workflow really has.
        # ONLY "no such workflow" is tolerated: `known_workflow_ids` can name an id
        # whose file disappears between the listing and the read. Anything else is
        # a fault and must not be swallowed -- a bare `except Exception: continue`
        # here would silently drop rows from this table, and the doc's own count
        # would change without a word.
        try:
            d = resolved_workflow(pid, canonical)
        except WorkflowError:
            continue
        ro = d.get("role_overrides", {}) or {}
        so = d.get("slot_overrides", {}) or {}
        try:
            routes = provision.workflow_lanes(d)
        except provision.ProvisionFailure:
            install_recipe = "missing exact owner"
        else:
            manual = set(routes.get("manual") or [])
            if manual:
                install_recipe = "complete; manual download"
            else:
                install_recipe = "complete"
            if provision.workflow_python_issue(d, (3, 14)):
                install_recipe += "; Python <=3.13"
        out.append({
            "id": pid,
            "status": d.get("status", "?"),
            "vram": (d.get("llm", {}) or {}).get("vram_ceiling_gb"),
            "vendor": d.get("gpu_vendor", "?"),
            "backend": d.get("device_backend", "?"),
            "platform": d.get("platform", "any"),
            "video": ro.get("character_visual") or "-",
            "image": ro.get("character_image") or "-",
            "voice": so.get("char_voice_engine") or "-",
            "music": so.get("music_engine") or "-",
            "proven": d.get("proven") or [],
            "install_recipe": install_recipe,
        })
    return out


#: Workflows are grouped by the card their NAME says they are for -- the JSON
#: is the machine answer, nothing derives a class from a VRAM number.
_GROUPS = (
    ("otr_8gb_", "8 GB NVIDIA"),
    ("otr_16gb_", "16 GB NVIDIA"),
    ("otr_24gb_", "24 GB NVIDIA"),
    ("otr_mac16_", "Apple Silicon (16 GB)"),
    ("otr_amd_", "AMD / ROCm"),
    ("otr_cloud_", "Cloud"),
    ("otr_google_", "Cloud"),
)


def _group(profile_id: str) -> str:
    for prefix, label in _GROUPS:
        if str(profile_id).startswith(prefix):
            return label
    return "Any machine"


_ORDER = ["8 GB NVIDIA", "16 GB NVIDIA", "24 GB NVIDIA", "Apple Silicon (16 GB)",
          "AMD / ROCm", "Cloud", "Any machine"]


#: ENGINE -> the third-party ComfyUI node pack it needs, and whether its weights
#: are Hugging Face gated. NEITHER fact can live in requirements.txt or
#: pyproject.toml: a node pack is not a pip distribution, and gating is a
#: property of the weights repo rather than of any package we declare. So the
#: guide is the only place a reader can learn it before a render fails.
#:
#: DERIVED, not guessed:
#:   packs   -- nodes/_otr_video_engines/wrapper_bridge.py `_PACK_FOR_PREFIX`
#:              mapped onto the class prefixes each engine module emits,
#:              FOLLOWING INHERITANCE (eng_ghost_signal_official subclasses
#:              GhostSignalEngine, so the haunted lane inherits its ADE_ graph).
#:   gating  -- the `gated` flags in scripts/otr_provision.py's artifact groups.
#: Re-derive both when an engine changes its graph or its weights repo.
_EXTRA_INSTALL = {
    "animatediff15_v3_haunted_video": "ComfyUI-AnimateDiff-Evolved",
    "animatediff15_v3_stillin_lab_video": "ComfyUI-AnimateDiff-Evolved",
    "animatediff15_lightning_video": "ComfyUI-AnimateDiff-Evolved",
}


def extra_install_for(cells) -> str:
    """What a user must install BEYOND this pack for that row to render."""
    needs = []
    for key in ("video", "image", "writer", "music", "char_voice"):
        val = str(cells.get(key) or "")
        for engine, pack in _EXTRA_INSTALL.items():
            if engine and engine in val and pack not in needs:
                needs.append(pack)
    return ", ".join(needs) if needs else "none"


def render() -> str:
    # Cleared per render: the flag is module-global, so a second call in the
    # same process would otherwise inherit the first one's verdict.
    global _DEGRADED
    _DEGRADED = ""
    profs = load_workflow_rows()
    L = []
    A = L.append
    A("# Machine matrix -- what runs where\n")
    A("**GENERATED by `scripts/otr_machine_matrix.py` from "
      "`config/workflow_matrix.json` (the workflows and their receipts). "
      "Do not hand-edit; regenerate.**\n")
    A("Open the workflow named for your card. There are no machine classes and "
      "no profiles: the workflow JSONs are the whole machine answer.\n")

    # ---- evidence first, then the front-door machine answer ----------------
    A("## Engine proof by hardware\n")
    A("Only receipt-backed rows appear here. **PROVEN** means a full OTR "
      "episode published on the named hardware. **LAB-PROVEN** means an "
      "isolated recipe produced valid media there; it does not promote the "
      "full OTR workflow. A setup, model load, queued prompt, reserve clamp, or "
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
    A("## Which workflow to open\n")
    A("| your card | open |")
    A("|---|---|")
    A("| 8 GB NVIDIA (RTX 4060, 3070, 2080) | an `otr_8gb_*` workflow |")
    A("| 10-15 GB NVIDIA (RTX 4070, 3080) | an `otr_8gb_*` workflow -- or try an "
      "`otr_16gb_*` one at your own risk |")
    A("| 16 GB NVIDIA and up (RTX 5080, 3090, 4090) | an `otr_16gb_*` workflow; "
      "`otr_24gb_foley` targets a 24 GB card |")
    A("| Apple Silicon, 16 GB | an `otr_mac16_*` workflow |")
    A("| AMD / ROCm | `otr_amd_still` -- PROVEN end to end on a Radeon AI PRO R9700 "
      "under ROCm 7.2 (see apple/ROCM.md) |")
    A("| no GPU, or not sure | `otr_canonical` -- it resolves the device at run time |")
    A("")
    A("**Read the `extra install` column in the tables below before you pick a "
      "row.** A lane that needs a third-party ComfyUI node pack cannot say so in "
      "`requirements.txt` or `pyproject.toml`, because a node pack is not a pip "
      "distribution -- so the requirement is invisible until the render fails "
      "with a missing class (PBUG-20260829-09).\n")
    A("Run these with the exact Python executable that launches ComfyUI (shown "
      "as `<ComfyUI Python>`). Preview a workflow's install plan first, then run "
      "the same command without `--list` to install it:\n")
    A("* `<ComfyUI Python> scripts/otr_provision.py --workflow-id otr_8gb_low --list` "
      "-- replace `otr_8gb_low` with the workflow you picked.\n")
    A("Provisioning installs and verifies artifacts; it does not rewrite the "
      "saved workflow. To apply one workflow to the real canonical graph on a "
      "normal port-8188 ComfyUI server, run `<ComfyUI Python> scripts/"
      "otr_canonical_api_run.py --comfyui-url http://127.0.0.1:8188 "
      "--workflow-id otr_8gb_low --act-count 1 --source-bank original "
      "--visual-style sci_fi_radio --timeout 0`. Every NVIDIA and Mac workflow "
      "selects the Kokoro voice. On the Python 3.13 that ComfyUI Desktop and the "
      "portable build ship it runs through kokoro-onnx on the CPU (the same "
      "voices, about six times faster than realtime), or on CUDA by itself when "
      "onnxruntime-gpu is installed; on Python 3.12 through the torch kokoro "
      "package. Python 3.14 has no kokoro backend packaged yet; there, switch "
      "the OTR_CastLock voice dropdowns to bark.")
    A("\nApple Silicon runs the `otr_mac16_*` workflows, PROVEN on a named "
      "physical system -- a Mac mini M4 / 16 GB published episodes to "
      "`otr/obs/` on 2026-09-07 and 2026-09-08, including local `sd15` stills "
      "and local `ltx_8gb` video diffusion. Read `apple/MAC.md` before "
      "starting. CPU-ONLY IS PROVEN ON THE CANONICAL ITSELF, 2026-09-13: "
      "ComfyUI launched with `--cpu` on an x86 laptop ran `otr_canonical` with "
      "no overrides and published a 2m07s episode to `otr/obs/` in 19 minutes "
      "54 seconds. The writer is nearly all of that: Qwen3.5-4B generates at "
      "about 3 tokens a second on a CPU, while Kokoro runs at 0.12x realtime "
      "and the procedural video lanes draw their own frames.\n")
    A("")

    A("## How to read the confidence column\n")
    A("| value | means |")
    A("|---|---|")
    A("| **PROVEN** | an episode actually rendered and published. Evidence named below. |")
    A("| **LAB-PROVEN** | an isolated recipe produced receipt-bearing media on named physical hardware; not a full OTR episode. |")
    A("| **EPISODE PATH PROVEN** | the components actually invoked by the named episode path published; an unused configured lane is not included. |")
    A("| **COMPONENTS PROVEN** | named components published on named hardware, but this exact row as one tuple is not certified. |")
    A("| `shipping` | the workflow is considered runtime-ready on a preloaded machine. It is neither hardware proof nor a complete clean-install claim. |")
    A("| `draft` | exists, not vouched for. Try it; expect to debug. |")
    A("")
    A("Nothing here is inferred from \"it looks like it should fit\". A blank is "
      "an unknown, recorded as one.\n")

    by = {}
    for p in profs:
        by.setdefault(_group(p["id"]), []).append(p)

    for group in _ORDER:
        rows = by.get(group) or []
        if not rows:
            continue
        ship = sum(1 for r in rows if r["status"] == "shipping")
        A("## %s  --  %d workflow(s), %d shipping\n"
          % (group, len(rows), ship))
        # Only PROVEN and shipping rows are tabled. A tier holding 76 drafts is
        # a dump, not a guide: a reader picking a row cannot tell which of 76 to
        # trust. Drafts are counted and folded away, listed by the engine they
        # select, which is the only thing anyone scans them for.
        headline = [r for r in rows if r["status"] == "shipping"]
        drafts = [r for r in rows if r not in headline]
        if headline:
            A("| row | video | voice | music | image | extra install | confidence | install recipe |")
            A("|---|---|---|---|---|---|---|---|")
            for r in sorted(headline, key=lambda x: x["id"]):
                conf = "`%s`" % r["status"]
                extra = extra_install_for({
                    "video": r["video"], "image": r["image"],
                    "music": r["music"], "char_voice": r["voice"]})
                A("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["id"], r["video"], r["voice"], r["music"], r["image"],
                    extra, conf, r["install_recipe"]))
            A("")
        else:
            A("**No shipping workflow in this group.**\n")
        if drafts:
            eng = sorted({r["video"] for r in drafts})
            A("<details><summary>%d draft workflow(s) here -- not vouched for"
              "</summary>\n" % len(drafts))
            A("Video engines they select: %s\n"
              % ", ".join("`%s`" % e for e in eng))
            A("| row | video | voice |")
            A("|---|---|---|")
            for r in sorted(drafts, key=lambda x: x["id"]):
                A("| `%s` | %s | %s |" % (r["id"], r["video"], r["voice"]))
            A("\n</details>\n")

    _voice_engines_table(A)

    A("## A bigger card does not currently get you more\n")
    A("A 24 or 32 GB card runs the `otr_16gb_*` workflows. One shipped "
      "workflow targets a bigger card on purpose -- `otr_24gb_foley`, LTX 2.5 "
      "on the int8 DiT with a 22 GB writer ceiling -- and you open it by hand.\n")
    A("**That matters when you are paying by the hour.** A rented 24 GB card "
      "ran the 16 GB haunted workflow and peaked at 15,990 MB, and rented "
      "Ampere has published LTX-2b, so the smaller lanes reach well beyond the "
      "floor. A bigger card still does not auto-select HuMo or the 24 GB LTX "
      "2.5 lane: pick the workflow explicitly and keep its exact "
      "hardware/software/RAM receipt.\n")

    A("## Hardware episode receipts, with their exact scope\n")
    for r in load_hardware_receipts():
        A("* **`%s`** on %s -- %s episode(s), %s. Scope: %s. Evidence: %s"
          % (r["workflows"], r["hardware"], r["episodes"], r["date"],
             r["scope"], r.get("evidence", "")))
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
    # KNOWN LIMITS WERE ONCE A DEAD CHANNEL: the field was read by nothing and
    # rendered nowhere, so a limit someone took the trouble to write down never
    # reached the person it was written for. A limit nobody sees is not a limit.
    _limits = load_known_limits()
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
    A("* AMD/ROCm has one proven workflow, `otr_amd_still`, on one card (a "
      "Radeon AI PRO R9700 under ROCm 7.2); other AMD cards and lanes are "
      "untested. Apple Silicon has published episodes on a Mac mini M4 / 16 GB, "
      "and the engine-proof table above carries its receipts (`viz_camera`, "
      "`sd15`, `ltx_8gb`, and all four `still_*` lanes).\n")
    A("* Where a lane needs weights, see "
      "[MODEL_ASSET_INDEX.md](MODEL_ASSET_INDEX.md) for the exact files and "
      "where to get them. That generated index is the cross-reference owner; "
      "the single RunPod playbook may repeat exact manual recipes where a "
      "stranger needs them at install time.\n")
    return "\n".join(L) + "\n"


_BEGIN = "<!-- BEGIN GENERATED: machine-matrix -->"
_END = "<!-- END GENERATED: machine-matrix -->"


def inject_readme(check_only: bool = False) -> bool:
    """REMOVE the old machine table from README. True if README has none.

    The name is kept because `--check` and `main` both call it and the job is
    still "make README agree with this generator" -- the agreement is now that
    README carries no copy (it links apple/MACHINES.md). A leftover block from an
    older checkout is stripped, markers and all; there is no re-injection path,
    so running this generator can no longer put the table back.
    """
    path = os.path.join(_REPO, "README.md")
    s = io.open(path, encoding="utf-8").read()
    if _BEGIN not in s or _END not in s:
        return True
    head, rest = s.split(_BEGIN, 1)
    _stale, tail = rest.split(_END, 1)
    out = (head.rstrip("\n") + "\n" + tail.lstrip("\n"))
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
    "kokoro": "ships: `kokoro` (torch) on Python 3.12, `kokoro-onnx` on 3.13 (CPU; CUDA with onnxruntime-gpu); "
              "voices and the ONNX model fetch once at boot",
    "bark": "ships: weights download on first use (about 4 GB)",
    "indextts2": "install it yourself: `scripts/_otr_indextts2_install.ps1` plus your "
                 "own reference WAVs (voice cloning)",
    "chatterbox": "install it yourself: isolated sidecar venv, reference WAVs",
    "elevenlabs": "your own API key (cloud)",
    "google_tts": "your own API key (cloud)",
}


#: Set when the VOICE-ENGINE section could not read the live audio registry --
#: and ONLY that section, which is narrower than it sounds and is said out loud
#: here because a reader who assumed "any degraded section" would trust this
#: flag further than it deserves. Two other soft failures do NOT set it: a
#: known-limits load that falls back to an empty section, and a corrupt profile
#: JSON that load_workflow_rows skips. Widening it means setting it there too.
#:
#: A degraded render is fine to LOOK at and fatal to SAVE, so it gates writing
#: and --check, not rendering. --stdout deliberately still prints: nothing is
#: overwritten and a human reading the output can see the placeholder.
_DEGRADED = ""


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
        # ...but it must NEVER be WRITTEN in this state. Proven 2026-09-09 by
        # doing exactly that: run under a torch-less interpreter, this branch
        # replaced the whole voice table in the shipped doc with one
        # parenthetical, and the write reported success. `render()` may still
        # degrade -- callers that only want the front tables depend on it -- so
        # the flag is what main() refuses on.
        global _DEGRADED
        _DEGRADED = "voice-engine table (%s)" % exc
        A("## Voice engines\n")
        A("(registry unavailable in this interpreter: %s)\n" % exc)
        return
    A("## Voice engines\n")
    A("What each voice engine needs, read from the audio registry. Kokoro is the "
      "shipped default for both voice slots on every NVIDIA and Mac workflow; the others stay in "
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
    dest = os.path.join(_REPO, "apple", "MACHINE_MATRIX.md")

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if _DEGRADED:
        # Refusing is the whole point: this generator OVERWRITES a shipped doc,
        # and a run that cannot read the registry produces a doc that is worse
        # than the one already on disk. Run it under the ComfyUI interpreter.
        # --check gets the same refusal, and for the same reason: a degraded
        # render differs from the good doc on disk, so checking it would report
        # DRIFT that does not exist and send somebody to "fix" a correct file.
        print("CANNOT %s: this interpreter could not read the %s, so the "
              "render is incomplete. It would %s. Re-run under the ComfyUI "
              "venv (the one with torch)."
              % ("CHECK" if args.check else "WRITE", _DEGRADED,
                 "report drift against a correct doc" if args.check
                 else "overwrite a good doc with a placeholder"))
        return 3

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
            print("  STALE: apple/MACHINE_MATRIX.md differs from the profiles")
        if not readme_ok:
            print("  STALE: README still carries a machine-matrix block; "
                  "this generator no longer puts one there")
        print("  run: python scripts/otr_machine_matrix.py")
        return 1

    io.open(dest, "w", encoding="utf-8", newline="\n").write(text)
    stripped = not inject_readme()
    print("wrote %s (%d bytes)%s" % (dest, len(text),
          "; removed the stale README block" if stripped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
