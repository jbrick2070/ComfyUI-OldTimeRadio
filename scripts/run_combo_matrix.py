"""run_combo_matrix.py -- risky-but-valid 30-60 word end-to-end combo matrix.

Full-render bug hunt for v2.0-alpha. Each combo is schema-valid against the
live /object_info (it SHOULD work), but isolates exactly one code- or
memory-grounded failure mode (it is the MOST LIKELY to break). Every combo
runs the FULL workflow -- script -> cast -> voice -> music -> HuMo -> upscale
-> procgen -> final mp4.

Runs SEQUENTIALLY (one render at a time, polled to completion before the next)
so the heavy stages never contend for VRAM. The chatterbox probe (C9) is last
and isolated because its dep-pins can brick the torch2.10 venv.

Usage (from repo root, with the ComfyUI venv python):
    python scripts\\run_combo_matrix.py --list            # print matrix, run nothing
    python scripts\\run_combo_matrix.py                   # run all 9, in order
    python scripts\\run_combo_matrix.py --only C3         # run a single combo
    python scripts\\run_combo_matrix.py --skip-cloud      # skip C7/C8 (no credits)
    python scripts\\run_combo_matrix.py --skip-dangerous  # skip C9 (chatterbox)
    python scripts\\run_combo_matrix.py --timeout 3000    # per-combo seconds (default 2700)

Results stream to scripts\\combo_matrix_results.json after every combo, so a
mid-run abort still leaves a partial record. A summary table prints at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import requests  # noqa: E402  (provided by the ComfyUI venv)

from otr_api import (  # noqa: E402
    COMFYUI_URL,
    apply_profile_to_workflow,
    fetch_schemas,
    load_workflow,
    patch_creative,
    patch_widget_by_name,
    poll_history,
    submit_prompt,
    workflow_to_api_prompt,
)

WORKFLOW_PATH = os.path.join(os.path.dirname(_HERE), "workflows", "otr_scifi_16gb_full.json")
RESULTS_PATH = os.path.join(_HERE, "combo_matrix_results.json")

# Node ids in the canonical 29-node workflow (from /object_info probe 2026-06-05).
N_WRITER = 1          # OTR_LedgerScriptWriter
N_CASTLOCK = 80       # OTR_CastLock
N_CHARVOICE = 81      # OTR_BatchCharacterVoices
N_MUSIC = 83          # OTR_StableAudioTheme

# Model slugs (exact /object_info choice strings).
GEMMA = "google/gemma-4-12b-it"
MISTRAL = "mistralai/Mistral-Nemo-Instruct-2407"
OR_A = "openrouter:slot-a"
COMFY_A = "comfy:slot-a"
OR_A_MODEL = "anthropic/claude-opus-4.8"      # only real model cached on slot-a
COMFY_A_MODEL = "anthropic/claude-opus-4.7"   # comfy slot-a recommended default

# ---------------------------------------------------------------------------
# The matrix. Each combo: schema-valid inputs + the ONE failure mode it probes.
# `expect` is the PASS criterion (a clean fail-closed counts as a pass for the
# coupling/scarcity probes -- a silent wrong result or uncaught crash does not).
# `prereq` flags operator setup; `danger` runs it last and isolated.
# ---------------------------------------------------------------------------
COMBOS = [
    dict(id="C1", words=60, chars=6, acts="auto", creative=GEMMA, technical=GEMMA,
         voice_bank="default", char_engine="indextts2", reuse=False, music="stable_audio_3",
         hyp="indextts2 bank has 1 CC0 female ref; 6 unique speakers exhaust distinct female refs",
         expect="renders, OR fails closed on ref scarcity -- NOT a silent same-voice collision"),
    dict(id="C2", words=55, chars=4, acts="auto", creative=GEMMA, technical=GEMMA,
         voice_bank="default", char_engine="indextts2", reuse=False, music="stable_audio_3",
         hyp="4 chars + gemma raises odds it abbreviates a cast name mid-script -> BUG-305 freeze-halt (acts=auto: the writer floor is ~50 words/act, so 55w supports 1 act)",
         expect="cast resolves via _cast_first_last_aliases; no freeze-halt"),
    dict(id="C3", words=40, chars=2, acts="auto", creative=GEMMA, technical=MISTRAL,
         voice_bank="default", char_engine="indextts2", reuse=False, music="stable_audio_3",
         hyp="crosses Ollama(gemma) <-> transformers(mistral) loader seam between narrative and JSON passes",
         expect="both backends co-resident; no VRAM-evict deadlock; clean slot handoff"),
    dict(id="C4", words=45, chars=3, acts="auto", creative=MISTRAL, technical=MISTRAL,
         voice_bank="default", char_engine="bark", reuse=False, music="stable_audio_3",
         hyp="voice_bank=default (clip-ref bank) vs engine=bark (preset) is an inconsistent pairing",
         expect="CastLock guards the mismatch (clear error) OR coherent bark render -- NOT silent wrong-voice"),
    dict(id="C5", words=50, chars=2, acts="auto", creative=GEMMA, technical=GEMMA,
         voice_bank="default", char_engine="indextts2", reuse=False, music="musicgen",
         hyp="2nd music engine (musicgen) fed a gemma-written Meta-brief is an untested cross",
         expect="musicgen loads + renders a theme from the brief; no prompt-route mismatch"),
    dict(id="C6", words=30, chars=2, acts="auto", creative=MISTRAL, technical=MISTRAL,
         voice_bank="default", char_engine="indextts2", reuse=False, music="stable_audio_3",
         hyp="30 words = the schema floor (target_words min); minimal 2-char episode -> does budget/assembly hold at the lower bound (act-count stress is out of band: ~50 words/act floor)",
         expect="renders a coherent minimal episode; no empty-scene crash at EpisodeAssembler"),
    dict(id="C7", words=40, chars=2, acts="auto", creative=OR_A, technical=OR_A,
         or_a_model=OR_A_MODEL, voice_bank="default", char_engine="indextts2", reuse=False,
         music="stable_audio_3", cloud="openrouter",
         hyp="full OpenRouter cloud writer (claude-opus-4.8) -> remote fail-closed JSON, cost-guard, no-evict",
         expect="remote passes return valid JSON; cost-guard respected; clean cloud->local handoff to voice",
         prereq="OPENROUTER_API_KEY + OTR_ENABLE_OPENROUTER=1 (restart Comfy). Spends OpenRouter credits (~cents @ 40w)"),
    dict(id="C8", words=40, chars=2, acts="auto", creative=COMFY_A, technical=COMFY_A,
         comfy_a_model=COMFY_A_MODEL, voice_bank="default", char_engine="indextts2", reuse=False,
         music="stable_audio_3", cloud="comfy",
         hyp="full Comfy Credits cloud writer (claude-opus-4.7) -> auth/slot resolution, cost-guard, pinned catalog",
         expect="comfy slug resolves via logged-in account; credits billed; valid JSON",
         prereq="OTR_ENABLE_COMFY_CREDITS=1 + logged-in Comfy Desktop w/ credits. Spends Comfy credits (~cents @ 40w)"),
    dict(id="C9", words=45, chars=2, acts="auto", creative=MISTRAL, technical=MISTRAL,
         voice_bank="default", char_engine="chatterbox", reuse=False, music="stable_audio_3",
         danger=True,
         hyp="chatterbox dep-pins (torch2.6/numpy1.26) can brick the torch2.10 venv; only bark is GPU-viable",
         expect="chatterbox sidecar renders OR fails closed WITHOUT corrupting the main venv -- RUN LAST",
         prereq="DANGEROUS: run last; if it hangs, kill ComfyUI + verify venv before re-running others"),
]


def build_patches(combo: dict) -> list[tuple[int, str, object]]:
    """Translate a combo dict into (node_id, widget_name, value) patches.

    Seeds are deliberately NOT patched -- the per-episode creative RNGs draw OS
    entropy (see memory: true-randomization-over-seed-pinning), so each combo
    gets a fresh cast/style draw."""
    p: list[tuple[int, str, object]] = [
        (N_WRITER, "target_words", int(combo["words"])),
        (N_WRITER, "num_characters", int(combo["chars"])),
        (N_WRITER, "act_count", str(combo["acts"])),
        (N_WRITER, "creative_writing_model", combo["creative"]),
        (N_WRITER, "technical_model", combo["technical"]),
        # GATE B S2: the ENGINE selections (char_engine / music) moved OUT of
        # this hand-coded patch list into combo_profile() -- the ONE applier
        # patches managed widgets. CastLock policy knobs are neither managed
        # nor whitelisted-creative; they stay direct (the regression test
        # bans MANAGED names only).
        (N_CASTLOCK, "voice_bank", combo["voice_bank"]),
        (N_CASTLOCK, "allow_voice_reuse", bool(combo["reuse"])),
    ]
    # Bind the cloud slot picker only when a cloud handle is the writer model.
    if combo.get("or_a_model"):
        p.append((N_WRITER, "openrouter_slot_a_model", combo["or_a_model"]))
    if combo.get("comfy_a_model"):
        p.append((N_WRITER, "comfy_slot_a_model", combo["comfy_a_model"]))
    return p


def combo_profile(combo: dict) -> dict:
    """An EPHEMERAL capability profile for one combo: the committed 16gb_full
    base with this combo's engine slots overridden. Managed widgets reach the
    graph ONLY via the applier (GATE B S2); cross-validation still applies
    when the profile rides apply_profile_to_workflow."""
    import copy as _copy
    repo_root = os.path.dirname(_HERE)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from nodes._otr_shared.capability_profiles import load_profile

    profile = _copy.deepcopy(load_profile("16gb_full"))
    profile["slot_overrides"]["char_voice_engine"] = combo["char_engine"]
    profile["slot_overrides"]["music_engine"] = combo["music"]
    return profile


def ledger_summary() -> dict:
    """Best-effort snapshot of /otr/latest_ledger after a run (non-fatal)."""
    try:
        r = requests.get(f"{COMFYUI_URL}/otr/latest_ledger", timeout=10).json()
    except Exception as e:
        return {"ledger": f"unreadable: {e!r}"}
    out = {}
    for k in ("episode_title", "episode_dir", "status", "audio_done", "error",
              "final_mp4", "output_path"):
        if isinstance(r, dict) and k in r:
            out[k] = r[k]
    if not out and isinstance(r, dict):
        out["ledger_keys"] = sorted(r.keys())[:12]
    return out


def cloud_env_warning(combo: dict) -> str | None:
    """Warn (do not block) if this runner's own env is missing a cloud opt-in.
    The ComfyUI process may have its own env; a missing flag here usually means
    the lane will fail closed with a clear message -- which is still useful."""
    if combo.get("cloud") == "openrouter":
        miss = [v for v in ("OPENROUTER_API_KEY", "OTR_ENABLE_OPENROUTER") if not os.environ.get(v)]
        return f"openrouter env missing here: {miss}" if miss else None
    if combo.get("cloud") == "comfy":
        if not os.environ.get("OTR_ENABLE_COMFY_CREDITS"):
            return "comfy env missing here: ['OTR_ENABLE_COMFY_CREDITS']"
    return None


def run_one(combo: dict, schemas: dict, timeout_s: int) -> dict:
    """Patch + submit + poll a single combo. Returns a result record. Never
    raises -- every failure mode is captured as a status string so the matrix
    keeps going."""
    rec = {"id": combo["id"], "hyp": combo["hyp"], "expect": combo["expect"],
           "status": "PENDING", "elapsed_s": 0, "prompt_id": "", "error": "",
           "warn": cloud_env_warning(combo) or ""}
    # 1) Load a fresh copy + apply patches (a patch error is a SETUP fault).
    try:
        wf = load_workflow(WORKFLOW_PATH)
        # Managed engine widgets ride the ONE applier (GATE B S2)...
        wf = apply_profile_to_workflow(wf, combo_profile(combo), schemas)
        applied = [f"profile:char_voice_engine={combo['char_engine']}",
                   f"profile:music_engine={combo['music']}"]
        # ...creative widgets ride the whitelist; CastLock policy knobs are
        # unmanaged + stay direct.
        for nid, widget, value in build_patches(combo):
            if widget in ("voice_bank", "allow_voice_reuse"):
                patch_widget_by_name(wf, nid, widget, value, schemas)
            else:
                patch_creative(wf, nid, widget, value, schemas)
            applied.append(f"{nid}.{widget}={value}")
        rec["patches"] = applied
        api = workflow_to_api_prompt(wf, schemas)
    except Exception as e:
        rec["status"] = "SETUP_ERROR"
        rec["error"] = repr(e)
        return rec
    # 2) Submit (a node_errors/validation reject surfaces here).
    try:
        pid = submit_prompt(api)
        rec["prompt_id"] = pid
    except Exception as e:
        rec["status"] = "SUBMIT_REJECT"
        rec["error"] = repr(e)
        return rec
    # 3) Poll to completion.
    t0 = time.time()
    status, err = poll_history(pid, timeout_s=timeout_s)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["status"] = status            # SUCCESS | FAIL | TIMEOUT
    rec["error"] = err
    rec["ledger"] = ledger_summary()
    return rec


def print_matrix() -> None:
    print("RISKY-BUT-VALID 30-60w COMBO MATRIX (full render to mp4)\n")
    for c in COMBOS:
        tag = "  [DANGER, run last]" if c.get("danger") else ("  [cloud:%s]" % c["cloud"] if c.get("cloud") else "")
        print(f"{c['id']}  {c['words']}w / {c['chars']}ch / acts={c['acts']}{tag}")
        print(f"     writer: {c['creative']}  +  {c['technical']}")
        print(f"     voice : bank={c['voice_bank']} engine={c['char_engine']} reuse={c['reuse']}   music: {c['music']}")
        print(f"     break : {c['hyp']}")
        print(f"     pass  : {c['expect']}")
        if c.get("prereq"):
            print(f"     PRE   : {c['prereq']}")
        print()


def write_results(results: list[dict]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"comfyui_url": COMFYUI_URL, "results": results}, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--only", default="", help="run a single combo id, e.g. C3")
    ap.add_argument("--skip-cloud", action="store_true", help="skip C7/C8 (no credit spend)")
    ap.add_argument("--skip-dangerous", action="store_true", help="skip C9 (chatterbox)")
    ap.add_argument("--timeout", type=int, default=2700, help="per-combo seconds (default 2700)")
    args = ap.parse_args()

    if args.list:
        print_matrix()
        return 0

    queue = [c for c in COMBOS]
    if args.only:
        queue = [c for c in queue if c["id"].lower() == args.only.lower()]
        if not queue:
            print(f"no combo with id {args.only!r}; valid: {[c['id'] for c in COMBOS]}")
            return 2
    if args.skip_cloud:
        queue = [c for c in queue if not c.get("cloud")]
    if args.skip_dangerous:
        queue = [c for c in queue if not c.get("danger")]

    print(f"COMFYUI_URL = {COMFYUI_URL}")
    print(f"workflow    = {WORKFLOW_PATH}")
    print("Fetching /object_info schemas...")
    try:
        schemas = fetch_schemas()
    except Exception as e:
        print(f"ABORT: ComfyUI not reachable at {COMFYUI_URL}: {e!r}")
        return 1
    print(f"Running {len(queue)} combo(s) sequentially, {args.timeout}s timeout each.\n")

    results: list[dict] = []
    for c in queue:
        w = cloud_env_warning(c)
        if w:
            print(f"[{c['id']}] WARN {w}")
        print(f"[{c['id']}] submitting ({c['words']}w/{c['chars']}ch, "
              f"{c['creative'].split('/')[-1]} + {c['char_engine']} + {c['music']})...")
        rec = run_one(c, schemas, args.timeout)
        results.append(rec)
        write_results(results)  # stream after every combo
        print(f"[{c['id']}] {rec['status']}  {rec['elapsed_s']}s  "
              f"{('-- ' + rec['error'][:160]) if rec['error'] else ''}")
        if c.get("danger") and rec["status"] in ("FAIL", "TIMEOUT"):
            print(f"[{c['id']}] danger combo did not pass cleanly -- "
                  f"verify the venv before trusting later runs.")

    print("\n==== SUMMARY ====")
    for r in results:
        print(f"{r['id']:>3}  {r['status']:<13} {r['elapsed_s']:>7}s  {r['hyp'][:64]}")
    print(f"\nfull records -> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
