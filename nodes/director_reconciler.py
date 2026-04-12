"""
DirectorReconciler — Merges Segmenter scenes with Director visual hints.

Algorithm: for each Segmenter scene, pick the Director scene with max
Jaccard overlap on line_indices. Tie -> earlier Director scene.

Segmenter count wins. Always.
"""

import json
import logging
import os
from datetime import datetime

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reconcile(director_scenes: list, seg_scenes: list) -> list:
    """Reconcile Director's visual plan with Segmenter's scene breaks.

    Args:
        director_scenes: List of dicts from Director's visual_plan.scenes.
            Each should have 'scene_id', 'visual_prompt', 'shot_description',
            'motion', and optionally 'line_indices'.
        seg_scenes: List of Scene dataclass instances from SceneSegmenter.

    Returns:
        List of dicts, one per Segmenter scene, enriched with Director hints.
        Each dict has:
            scene_id: from Segmenter (authoritative)
            line_indices: from Segmenter (authoritative)
            dialogue: from Segmenter
            visual_prompt: from best-matching Director scene (advisory)
            motion: from best-matching Director scene (advisory)
            director_scene_id: which Director scene was matched
            jaccard_score: overlap quality (0.0 - 1.0)
    """
    if not seg_scenes:
        return []

    # Extract Director line_indices for Jaccard computation
    director_index_sets = []
    for ds in director_scenes:
        indices = ds.get("line_indices", [])
        director_index_sets.append(set(indices))

    divergences = []
    result = []

    for seg in seg_scenes:
        seg_indices = set(getattr(seg, "line_indices", []))
        seg_id = getattr(seg, "scene_id", "s??")
        seg_dialogue = getattr(seg, "dialogue", [])

        best_idx = -1
        best_jaccard = -1.0

        for d_idx, d_set in enumerate(director_index_sets):
            if not seg_indices and not d_set:
                score = 0.0
            elif not seg_indices or not d_set:
                score = 0.0
            else:
                intersection = len(seg_indices & d_set)
                union = len(seg_indices | d_set)
                score = intersection / union if union > 0 else 0.0

            # Tie-break: earlier Director scene wins
            if score > best_jaccard:
                best_jaccard = score
                best_idx = d_idx

        # Build enriched scene dict
        enriched = {
            "scene_id": seg_id,
            "line_indices": list(seg_indices),
            "dialogue": seg_dialogue,
            "visual_prompt": "",
            "motion": "medium",
            "director_scene_id": None,
            "jaccard_score": round(best_jaccard, 3),
        }

        if best_idx >= 0 and best_idx < len(director_scenes):
            ds = director_scenes[best_idx]
            enriched["visual_prompt"] = ds.get(
                "visual_prompt", ds.get("shot_description", ""))
            enriched["motion"] = ds.get("motion", "medium")
            enriched["director_scene_id"] = ds.get("scene_id", f"d{best_idx + 1:02d}")

        if best_jaccard < 0.3:
            divergences.append({
                "seg_scene": seg_id,
                "best_director": enriched["director_scene_id"],
                "jaccard": best_jaccard,
            })

        result.append(enriched)

    # Log divergences
    if divergences:
        log.warning(
            "[Reconciler] %d scenes with low overlap (<0.3): %s",
            len(divergences),
            [d["seg_scene"] for d in divergences]
        )
        _log_divergence(divergences)

    log.info(
        "[Reconciler] Reconciled %d Segmenter scenes with %d Director scenes",
        len(result), len(director_scenes)
    )
    return result


def _log_divergence(divergences: list) -> None:
    """Append divergence info to the current run log."""
    try:
        log_dir = os.path.join(_REPO_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_path = os.path.join(log_dir, f"run_{ts}.json")

        entry = {
            "ts": datetime.now().isoformat(),
            "type": "reconciler_divergence",
            "divergences": divergences,
        }

        # Append to run log (may already exist from other phases)
        existing = []
        if os.path.isfile(run_log_path):
            with open(run_log_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.append(entry)
        with open(run_log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass
