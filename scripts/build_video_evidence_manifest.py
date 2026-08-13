"""GENERATOR for `docs/evidence/video_evidence_manifest.json` (spec S9).

The manifest is the answer to lesson L7: a number without its evidence key is
not evidence, and a DIGEST OF A FILE NOBODY SHIPS proves nothing to a reader who
does not have it. So for every receipt the corpus cites this records the sha256
of the bytes on disk, whether the path is actually CONTAINED in the named lab
evidence commit (`git cat-file -e <commit>:<path>` -- the only check that
distinguishes a baseline from a claim), and the QA verdict already ruled
against it.

It never re-measures and never invents a number. A receipt that is absent is
recorded absent, and a row whose corpus wording overstated its receipt keeps the
QA's correction in `note` rather than the flattering version.

BUILT INCREMENTALLY, APPEND-ONLY (r4). Each lane that produces a qualification
receipt appends its rows to :data:`ROWS`, bumps ``manifest_version``, and
regenerates -- rows are never rewritten in place. Run:

    python scripts/build_video_evidence_manifest.py

Read by `tests/test_lane_preflight_matrix.py` gate G4, which is why
``admission_unenforced`` lives here: a lane is either QUALIFIED or it says
"admission NOT enforced" in words, on disk, reachable in the manifest.
"""
import hashlib
import json
import os
import subprocess

LAB = os.environ.get(
    "OTR_VRAM_LAB_ROOT", r"C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab")
LAB_COMMIT = "4d87cfa3278c39cbdde6f3cb8b16f241aeb58c02"
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_HERE, "docs", "evidence", "video_evidence_manifest.json")

# (lane, key, receipt relative paths, narrative, qa_verdict, note)
ROWS = [
    ("ltx_audio_in", "ltx_av_gguf_q3_832x480_f97",
     ["results/ltx_audio_gguf_run4.json", "results/ltx_audio_gguf_run5.json"],
     "docs/VIDEO_RECIPE_ATTEMPTS.md", "PARTIAL",
     "Exact cold/warm are 7.47/7.41 GiB; no receipt supports the 7.2 lower "
     "endpoint the corpus table printed."),
    ("ltx_audio_in", "ltx_av_hq_1024x576_f193",
     ["results/ltx_audio_hq_h3_1024x576_193f_run2.json"],
     "docs/VIDEO_RECIPE_ATTEMPTS.md", "SUPPORTED",
     "7.36 GiB warm / 585.3 s at the S3 HQ canvas."),
    ("wan_ti2v", "wan_ti2v_5b_q5_832x480_f193",
     ["results/wan_ti2v_5b_cmp_832x480_f193_run4.json"],
     "docs/ENVELOPE_LADDERS.md", "UNSUPPORTED_AS_WRITTEN",
     "Exact 832x480x193 Q5 warm receipt is 12.1 GiB, not the 12.5-13.2 GiB the "
     "corpus table printed; the row omitted rung and measurement surface."),
    ("wan_i2v", "wan_i2v_14b_832x480_f33_exoneration",
     ["results/wan_i2v_14b_exoneration_832x480_f33_run1.json",
      "results/wan_i2v_14b_exoneration_832x480_f33_run2.json"],
     "docs/PROMOTION_BRIEF.md", "PARTIAL",
     "13.93 warm / 14.05 cold is supported ONLY at 832x480x33. The rung f33 "
     "must be stated with the number; f177 is unqualified."),
    ("humo_1.7B", "humo_1p7b_default_480x832_f129",
     ["results/otr_side/humo_1_7b_bakeoff_take1.json",
      "results/otr_side/humo_1_7b_bakeoff_take2.json"],
     "docs/HUMO_BAKEOFF.md", "SUPPORTED_WITH_QUALIFIER",
     "OTR-side portrait f129 only; 15.12-15.23 GiB is not a lab-gated number."),
    ("humo_1.7B", "humo_1p7b_diet_480x832_f129",
     ["results/humo_1p7b_diet_run2.json"],
     "docs/HUMO_DIET.md", "SUPPORTED",
     "12.84 GiB warm, portrait 480x832x129, diet boot."),
    ("humo", "humo_14b_default_480x832_f97",
     ["results/otr_side/humo_14b_fp8_bakeoff_take1.json"],
     "docs/HUMO_BAKEOFF.md", "SUPPORTED_WITH_QUALIFIER",
     "OTR-side portrait 480x832x97; 14.98 GiB is OVER the 14.5 GiB gate."),
    ("humo_14B_169", "humo_14b_diet_landscape_832x480_f97",
     ["results/humo_14b_diet_landscape_832x480_f97_run1.json",
      "results/humo_14b_diet_landscape_832x480_f97_run2.json"],
     "docs/ENVELOPE_LADDERS.md", "NUMERICALLY_SUPPORTED",
     "13.06 GiB warm / 13.17 cold, diet boot, LANDSCAPE -- the ruled hero cast. "
     "Human parity RULED 2026-08-10."),
    ("humo", "humo_14b_diet_portrait_480x832_f97",
     ["results/humo_14b_diet_portrait_480x832_f97_run1.json",
      "results/humo_14b_diet_portrait_480x832_f97_run2.json"],
     "docs/ENVELOPE_LADDERS.md", "NUMERICALLY_SUPPORTED",
     "13.22 GiB warm / 13.14 cold, diet boot, portrait."),
    ("minimax_h3_audio_in", "h3_ref2va_864x480_f124_cold",
     ["results/h3_r2v_refaudio_tts_lipsync_exact_seed42_run1.json"],
     "docs/HUMO_BAKEOFF.md", "PARTIAL_WRONG_SURFACE",
     "COLD 864x480x124 only. There is no warm pass and no 832x480 "
     "qualification. Ref2VA CANNOT classify H3 I2V or score/mime."),
    ("minimax_h3_video", "h3_i2v_canonical_832x480_f107_FAILED",
     ["results/h3_i2v_canonical_832x480_f107_run1.json"],
     "docs/ENVELOPE_LADDERS.md", "MEASURED_BELOW_RANGE_FAILURE",
     "15.390 GiB at canvas 832x480, model f107 -- BELOW the trained 124..362 "
     "range. Evidence of a failure below range, never evidence for a minimum."),
    ("minimax_h3_music", "h3_music_score_f192_cold",
     ["results/h3_music_followup_score_seed42_f192_run1.json"],
     "docs/H3_MUSIC_FOLLOWUP.md", "SUPPORTED_COLD",
     "11.063 GiB cold at model f192. f277 reached 14.722 GiB and FAILS the "
     "14.5 GiB gate -- lengths above f192 are not machine-qualified."),
    ("wan_ti2v", "wan_ti2v_chained_177_plus_25_retention",
     ["results/otr_side/wan_retention/phase1_wan_ti2v_long_first.json"],
     "docs/WAN_RETENTION_FINDINGS.md", "DIAGNOSTIC_ONLY",
     "12.43 peak / +5.11 GiB retained, WHOLE-CHILD chained surface. Not a "
     "single-render envelope and never quotable as one."),
    ("fastwan_8gb", "fastwan_chained_177_plus_25_retention",
     ["results/otr_side/wan_retention/phase3_fastwan_8gb_long_first.json"],
     "docs/WAN_RETENTION_FINDINGS.md", "DIAGNOSTIC_ONLY",
     "12.57 peak / +5.33 GiB retained, whole-child chained surface."),
    ("ltx_video", "ltx_video_chained_169_plus_169_retention",
     ["results/otr_side/wan_retention/phase3_ltx_video_long_first.json"],
     "docs/WAN_RETENTION_FINDINGS.md", "FAILED_DIAGNOSTIC",
     "14.59 peak / +3.06 retained at 832x448 -- NOT the 832x480 the lane "
     "declares, and a failed run. Cannot decide this lane's low/high marker."),
]

#: OTR-SIDE LEGS -- measured through OTR's real prepare() + render_clip()
#: lifecycle on this box, as opposed to the lab receipts above. Kept in their
#: own list because they are a DIFFERENT MEASUREMENT SURFACE and the corpus's
#: original defect was putting two surfaces in one column (lesson L7).
#:
#: PROVENANCE RULE (operator, 2026-08-11, binding): a cost row may be seeded
#: ONLY from a true VramPeakProbe MAXIMUM. A single nvidia-smi reading is a
#: lower bound on the peak and never seeds a row -- it is recorded here with
#: seeds_cost_row False so the number stays readable without becoming usable.
#:
#: NET, NOT ABSOLUTE (operator, 2026-08-11, binding). `free_vram_mb()` returns
#: torch.cuda.mem_get_info() FREE bytes, and `compute_real_frame_budget`
#: compares overhead + per_frame*frames against free * 0.85. FREE already
#: excludes the resident desktop baseline, so an overhead derived from an
#: ABSOLUTE peak double-charges that baseline on every prediction -- which is
#: exactly how the shipped WAN row ended up refusing every segment length the
#: coverage planner produces. net_mb = absolute_peak_mb - the leg's own
#: pre-queue baseline.
#:
#: SUPERSEDED BY S7.1 WHEN IT LANDS: baseline subtraction is a first-order
#: correction and the baseline is not constant across a render (ComfyUI evicts
#: and reloads). S7.1 records free_vram_mb() at render start and its MINIMUM
#: during the window; that difference IS the demand in the units admission
#: compares against. Re-derive then, and treat any disagreement with these
#: numbers as a FINDING rather than a correction.
OTR_SIDE_LEGS = [
    {
        "lane": "humo", "public_id": "humo14_high_audio_in_portrait",
        "canvas": "480x832", "model_frames": 97, "delivered_frames": 97,
        "boot_lane": "humo_diet", "cache_state": "cold",
        "surface": "absolute device total (VramPeakProbe max)",
        "absolute_peak_mb": 13800, "baseline_mb": 1889, "net_mb": 11911,
        "wall_time_s": 271.3, "seeds_cost_row": True,
        "receipt": "docs/evidence/lane_receipts/lane04-humo14_high_audio_in_portrait.md",
        "note": "First render after a fresh boot, so the model load sits inside "
                "the measured window.",
    },
    {
        "lane": "humo_14B_169", "public_id": "humo14_high_audio_in_wide",
        "canvas": "832x480", "model_frames": 97, "delivered_frames": 97,
        "boot_lane": "humo_diet", "cache_state": "cold",
        "surface": "absolute device total (VramPeakProbe max)",
        "absolute_peak_mb": 14604, "baseline_mb": 1940, "net_mb": 12664,
        "wall_time_s": 249.2, "seeds_cost_row": True,
        "receipt": "docs/evidence/lane_receipts/lane02-humo14_high_audio_in_wide.md",
        "note": "The ruled hero cast. Net sits BELOW the lab's 13.06 GiB warm "
                "figure, which is the expected direction once the baseline is "
                "removed from a cold total.",
    },
    {
        "lane": "humo_1.7B", "public_id": "humo17_high_audio_in_portrait",
        "canvas": "480x832", "model_frames": 129, "delivered_frames": 129,
        "boot_lane": "humo_diet", "cache_state": "cold",
        "surface": "absolute device total (VramPeakProbe max)",
        "absolute_peak_mb": 15261, "baseline_mb": 1940, "net_mb": 13321,
        "wall_time_s": 210.9, "seeds_cost_row": True,
        "receipt": "docs/evidence/lane_receipts/lane03-humo17_high_audio_in_portrait.md",
        "note": "The most expensive of the three HuMo legs because it renders a "
                "third more frames at the same pixel area, NOT because it is a "
                "heavier model. Frame count dominates on this family.",
    },
    {
        "lane": "wan_i2v", "public_id": "wan22_high_i2v",
        "canvas": "832x480", "model_frames": 33, "delivered_frames": 33,
        "boot_lane": "default", "cache_state": "cold",
        "surface": "nvidia-smi SAMPLE -- lower bound, NOT a probe maximum",
        "absolute_peak_mb": None, "baseline_mb": 1925, "net_mb": None,
        "observed_sample_mb": 13751, "wall_time_s": 217.9,
        "seeds_cost_row": False,
        "receipt": "docs/evidence/lane_receipts/lane01-wan22_high_i2v.md",
        "note": "13,751 MB was read by hand mid-sampling, so it is a lower "
                "bound on the peak. Under the binding provenance rule it does "
                "NOT seed a cost row; net_mb is deliberately null rather than "
                "11,826 so the number cannot be picked up by accident.",
    },
    {
        "lane": "wan_ti2v", "public_id": "wan22_high_video",
        "canvas": "832x480", "model_frames": 81, "delivered_frames": 81,
        "boot_lane": "default", "cache_state": "cold",
        "surface": "NOT CAPTURED", "absolute_peak_mb": None,
        "baseline_mb": 1883, "net_mb": None, "wall_time_s": 171.2,
        "seeds_cost_row": False,
        "receipt": "docs/evidence/lane_receipts/lane05-wan22_high_video.md",
        "note": "The engine DOES run VramPeakProbe and threads the max into its "
                "clip dict, but render_driver._clip_summary dropped the field, "
                "so it never reached disk. Recoverable by the passthrough fix, "
                "then one re-smoke -- no re-measurement campaign needed.",
    },
    {
        "lane": "fastwan_8gb", "public_id": "wan22_high_fast",
        "canvas": "832x480", "model_frames": 81, "delivered_frames": 81,
        "boot_lane": "default", "cache_state": "cold",
        "surface": "NOT CAPTURED", "absolute_peak_mb": None,
        "baseline_mb": 1883, "net_mb": None, "wall_time_s": 70.5,
        "seeds_cost_row": False,
        "receipt": "docs/evidence/lane_receipts/lane06-wan22_high_fast.md",
        "note": "Same dropped-field cause as wan_ti2v. Wall time is the "
                "comparable datum that DID survive: 70.5 s against wan_ti2v's "
                "171.2 s for the identical render, 2.43x.",
    },
]

NARRATIVES = [
    "docs/PROMOTION_BRIEF.md", "docs/HUMO_BAKEOFF.md", "docs/HUMO_DIET.md",
    "docs/ENVELOPE_LADDERS.md", "docs/WAN_RETENTION_FINDINGS.md",
    "docs/H3_MUSIC_FOLLOWUP.md", "docs/VIDEO_RECIPE_ATTEMPTS.md",
    "docs/H3_LICENSE_GRANT.md",
]


def sha256(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def in_commit(rel):
    r = subprocess.run(["git", "cat-file", "-e", "%s:%s" % (LAB_COMMIT, rel)],
                       cwd=LAB, capture_output=True)
    return r.returncode == 0


def artifact(rel):
    full = os.path.join(LAB, rel.replace("/", os.sep))
    return {
        "path": "vram-recipe-lab/" + rel,
        "sha256": sha256(full),
        "present_on_disk": os.path.isfile(full),
        "contained_in_evidence_commit": in_commit(rel),
    }


entries = []
for lane, key, receipts, narrative, verdict, note in ROWS:
    entries.append({
        "lane": lane,
        "envelope_key": key,
        "qa_verdict": verdict,
        "note": note,
        "narrative": artifact(narrative),
        "receipts": [artifact(r) for r in receipts],
    })

manifest = {
    "schema": "otr.video_evidence_manifest/1",
    # STALE, AND THE GUARD AT THE BOTTOM OF THIS FILE IS WHY (2026-08-13).
    # This generator still produces version 1. The SHIPPED manifest is at 5:
    # lanes 7b, 8, 9 and the H3 pair appended their rows to the JSON directly
    # and never came back to this script, so running it verbatim DELETES 123
    # lines of real evidence. Found the only way it could be -- by running it.
    # Do not bump this number to silence the guard; that just re-arms the trap.
    # Reconciling the two is its own piece of work: every row the JSON has and
    # this file does not must be re-derived from its receipt, not copied.
    "manifest_version": 1,
    "built": "2026-08-11",
    "lab_repo": "vram-recipe-lab",
    "lab_evidence_commit": LAB_COMMIT,
    "doctrine": [
        "Lab receipts are EVIDENCE: never re-measured here, never invented.",
        "A digest of a file that is not shipped proves nothing to a reader "
        "without it -- every row records contained_in_evidence_commit.",
        "Three separate columns, never inferred from one another: "
        "model-legal window, machine-qualified window, episode-policy cap.",
        "APPEND-ONLY. Every receipt-producing commit bumps manifest_version "
        "and appends its rows; rows are never rewritten in place.",
    ],
    # G4.1's honest escape hatch, IN WORDS, on disk, reachable in the manifest.
    # QUALIFIED_COST_ROWS is empty today (motion_common.py:367), so NO local
    # lane is guarded; saying so here is the difference between an unguarded
    # lane and an unguarded lane that looks guarded. A lane leaves this table
    # in the same commit that qualifies its cost row through OTR's real
    # prepare() + render_clip() lifecycle -- never from lab numbers alone.
    "admission_unenforced": {
        "wan_ti2v": (
            "admission NOT enforced: BOTH paths are inert because "
            "QUALIFIED_COST_ROWS is empty. Corrected 2026-08-13 -- the STATIC "
            "path (compute_real_frame_budget -> MotionBudgetError) used to "
            "fire on this lane while the coverage-plan path did not, because "
            "it never asked cost_row_may_refuse. It refused two live 45-word "
            "render-gate legs on the row this repo disqualifies before the "
            "gate was moved to cover it. Recalibration + qualification ship "
            "together in lane 5."),
        "fastwan_8gb": (
            "admission NOT enforced: this lane mirrors the same disqualified "
            "cost row from its own module -- byte-identical to the fallback, "
            "so deleting it would change nothing -- and would go silently "
            "stale if recalibrated apart from wan_ti2v. It was the second leg "
            "the ungated STATIC path refused on 2026-08-13. Owned by lane 6."),
        "wan_i2v": (
            "admission NOT enforced: single-clip renders on this lane get no "
            "check before or after. Only f33 at 832x480 has warm evidence, "
            "so no envelope covers the f177 ceiling it declares."),
        "humo": (
            "admission NOT enforced: no cost row and no envelope key. "
            "Evidence is portrait 480x832x97 only, on two boot lanes with a "
            "2.16 GiB gap between them (14.98 default, 13.22 diet)."),
        "humo_1.7B": (
            "admission NOT enforced: no cost row, and safe_render_frames is "
            "None so the exact-fit guard is skipped entirely -- an "
            "over-ladder beat emits 177 frames stamped as if honest. "
            "Owned by lane 3."),
        "humo_1.7B_169": (
            "admission NOT enforced: inherits the 1.7B tier's missing cost "
            "row and has no landscape receipt of its own at any rung."),
        "humo_14B_169": (
            "admission NOT enforced: the ruled hero cast has a measured "
            "13.06 GiB warm landscape f97 receipt under the diet boot, but a "
            "receipt is not a qualified cost row and nothing refuses an "
            "over-budget plan on this lane today."),
        "ltx_audio_in": (
            "admission NOT enforced: no cost row. The affine FIT is CUT for "
            "LTX because the HQ ladder is non-monotonic against the model's "
            "pixel scaling, so this lane needs conservative ABSOLUTE "
            "envelopes rather than a smoothed line."),
        "ltx_video": (
            "admission NOT enforced: no cost row, and its only datapoint is "
            "a FAILED chained diagnostic at 832x448 -- not the 832x480 it "
            "declares, and not a single-render number at all."),
        "ltx_8gb": (
            "admission NOT enforced: no cost row and NO measurement of any "
            "kind on this box, which is also why its low/high public marker "
            "is still provisional."),
        "mesh_stage": (
            "admission NOT enforced: no cost row; the lane's real preflight "
            "gap is the ungated hy3d graph, not a VRAM envelope."),
    },
    "narratives": [artifact(n) for n in NARRATIVES],
    "entries": entries,
    "otr_side_legs": OTR_SIDE_LEGS,
}

# ---- NEVER CLOBBER A NEWER MANIFEST (2026-08-13). The append-only doctrine
# above assumes every appending lane appends HERE. Several did not: they edited
# the JSON directly, so this generator's ROWS are a strict subset of what ships
# and a plain re-run is a silent 123-line deletion of evidence. A generator that
# destroys the artifact it is named for is worse than no generator, so it fails
# closed and says exactly what it would have thrown away.
_existing_version = 0
if os.path.exists(OUT):
    try:
        with open(OUT, encoding="utf-8") as fh:
            _existing_version = int(json.load(fh).get("manifest_version") or 0)
    except (OSError, ValueError, TypeError):
        _existing_version = 0
if _existing_version > manifest["manifest_version"]:
    raise SystemExit(
        "REFUSING to overwrite %s: it is at manifest_version %d and this "
        "generator only produces %d. Rows that lanes appended to the JSON "
        "directly are NOT in this script, so writing would delete them. "
        "Reconcile the missing rows into ROWS / admission_unenforced -- "
        "re-derived from their receipts, never copied out of the JSON -- "
        "before running this again."
        % (OUT, _existing_version, manifest["manifest_version"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=False)
    fh.write("\n")

miss = [a["path"] for e in entries for a in e["receipts"] if not a["present_on_disk"]]
notin = [a["path"] for e in entries for a in e["receipts"]
         if a["present_on_disk"] and not a["contained_in_evidence_commit"]]
print("entries:", len(entries))
print("absent on disk:", miss)
print("present but NOT in %s:" % LAB_COMMIT[:7], len(notin))
for p in notin:
    print("   ", p)
print("narratives not in commit:",
      [a["path"] for a in manifest["narratives"] if not a["contained_in_evidence_commit"]])
