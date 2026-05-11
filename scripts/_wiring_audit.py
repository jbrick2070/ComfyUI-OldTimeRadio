"""scripts/_wiring_audit.py

Post-cleanup wiring audit. Verifies:

1. No legacy widget references (cleanup_model_id / self_critique /
   target_length / arc_enhancer / their constants) remain in any .py
   or .json under the repo (except docs comments).
2. Beat-by-beat ledger writes happen end-to-end:
   - writer calls init_lines_from_outline + save
   - writer calls update_line_text + save per beat
   - no leftover line_rows accumulator references
3. Downstream consumers iterate the ledger via iter_lines (which
   honors skip):
   - batch_bark_generator
   - kokoro_announcer
   - scene_sequencer (timing math is allowed to iterate directly,
     but should still honor skip for clip duration)
4. act_count threads correctly: widget -> run() -> resolved
   -> compute_episode_budget -> OutlineRequest.budget -> outline
   beats with arc_phase -> LineRequest.arc_phase

Pure read-only audit. Exits 0 if everything's clean; 1 if any
problem found.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODES_DIR = ROOT / "nodes"
TESTS_DIR = ROOT / "tests"
WORKFLOWS_DIR = ROOT / "workflows"

LEGACY_NAMES = [
    "cleanup_model_id",
    "self_critique",
    "target_length",
    "arc_enhancer",
    "_TARGET_LENGTH_CHOICES",
    "_TARGET_LENGTH_FORCE_WORDS",
    "_CLEANUP_MODEL_CHOICES",
]

# A reference is OK if it's clearly in a docstring / comment about
# the REMOVAL. We allow lines matching any of these markers.
OK_CONTEXT_PATTERNS = [
    r"removed 20\d\d",
    r"deprecated",
    r"dropped 20\d\d",
    r"v2\.0 MVP no-op",
    r"cleanup pass",
    r"# .* dropped",
    r"# .* removed",
    r'""".*"""',
    r"docs?/",
    r"^\s*#",
    r"^\s*['\"]?[A-Z]",   # docstring-ish starter
]

ok_re = re.compile("|".join(OK_CONTEXT_PATTERNS), re.IGNORECASE)


def audit_legacy_refs():
    print("=== AUDIT 1: legacy widget references in non-doc context ===")
    bad = 0
    for base in (NODES_DIR, TESTS_DIR, WORKFLOWS_DIR):
        for p in base.rglob("*"):
            if "__pycache__" in p.parts or p.suffix not in (".py", ".json"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                for name in LEGACY_NAMES:
                    if name in line:
                        # Check if line looks like comment / docstring / removal note
                        stripped = line.strip()
                        is_comment = stripped.startswith("#") or stripped.startswith('"""')
                        is_string_assignment = bool(re.match(r'^\s*["\']', stripped))
                        looks_legacy_ok = (
                            is_comment
                            or "removed" in line.lower()
                            or "dropped" in line.lower()
                            or "deprecated" in line.lower()
                            or "legacy" in line.lower()
                            or "cleanup" in line.lower()
                            or "back-compat" in line.lower()
                            or "no-op" in line.lower()
                            or "(was " in line
                            or "field removed" in line.lower()
                        )
                        if looks_legacy_ok:
                            continue
                        rel = p.relative_to(ROOT)
                        print(f"  BAD: {rel}:{i}: {line.strip()[:120]}")
                        bad += 1
                        break
    if bad == 0:
        print("  PASS: no non-doc legacy references found")
    return bad


def audit_beat_by_beat_writes():
    print("\n=== AUDIT 2: beat-by-beat ledger writes wired ===")
    writer = (NODES_DIR / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    issues = 0
    checks = [
        ("led.init_lines_from_outline", "init_lines_from_outline call"),
        ("led.update_line_text", "update_line_text call"),
        ("led.save()", "led.save() call (at least one)"),
        ("LineLedgerMismatchError", "update_line_text return check"),
    ]
    for needle, label in checks:
        n = writer.count(needle)
        if n == 0:
            print(f"  BAD: writer missing {label} ({needle!r})")
            issues += 1
        else:
            print(f"  PASS: writer has {label} ({n}x)")
    # Detect stale line_rows accumulator pattern.
    stale_patterns = [
        r"\bline_rows\s*=\s*\[\]",
        r"\bline_rows\.append\b",
        r"\bfor r in line_rows\b",
        r"\bled\.set_lines\(",
    ]
    for pat in stale_patterns:
        if re.search(pat, writer):
            print(f"  BAD: writer still contains stale pattern {pat!r}")
            issues += 1
    if issues == 0:
        print("  PASS: no stale line_rows accumulator patterns")
    return issues


def audit_consumer_iter_lines():
    print("\n=== AUDIT 3: downstream consumers use iter_lines ===")
    issues = 0
    consumers = [
        "batch_bark_generator.py",
        "kokoro_announcer.py",
        "scene_sequencer.py",
        "batch_kokoro_generator.py",
        "batch_audiogen_generator.py",
    ]
    for name in consumers:
        p = NODES_DIR / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        uses_iter_lines = bool(re.search(r"\biter_lines\(", text))
        # Look for direct iteration of led["lines"] / data["lines"]
        # that bypasses iter_lines. Allowable if it's iter_lines itself.
        direct_iters = []
        for i, line in enumerate(text.splitlines(), 1):
            m = re.search(
                r"for\s+\w+\s+in\s+.*\.get\(['\"]lines['\"]|"
                r"for\s+\w+\s+in\s+.*\[['\"]lines['\"]\]",
                line,
            )
            if m and "iter_lines" not in line:
                direct_iters.append((i, line.strip()))
        if not uses_iter_lines:
            print(f"  BAD: {name} does NOT use iter_lines")
            issues += 1
        else:
            print(f"  PASS: {name} uses iter_lines")
        if direct_iters:
            for ln, content in direct_iters[:3]:
                print(f"    DIRECT-ITER: {name}:{ln}: {content[:100]}")
    return issues


def audit_act_count_thread():
    print("\n=== AUDIT 4: act_count threads through pipeline ===")
    issues = 0
    writer = (NODES_DIR / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    outline = (NODES_DIR / "_otr_outline.py").read_text(encoding="utf-8")
    composer = (NODES_DIR / "_otr_line_composer.py").read_text(encoding="utf-8")
    budget = (NODES_DIR / "_otr_episode_budget.py").read_text(encoding="utf-8")

    checks = [
        (writer, "act_count: int = 0", "act_count kwarg in _resolve_inputs"),
        (writer, '"act_count":', "act_count in resolved dict / meta"),
        (writer, "compute_episode_budget", "compute_episode_budget call"),
        (writer, "episode_budget", "episode_budget local"),
        (writer, "budget=episode_budget", "budget passed to OutlineRequest"),
        (writer, "arc_phase=(beat.arc_phase", "arc_phase threaded to LineRequest"),
        (outline, "budget: object = None", "OutlineRequest.budget field"),
        (outline, "validate_outline_against_budget", "budget validator"),
        (outline, 'arc_phase: str = Field(', "Beat.arc_phase required-with-default"),
        (composer, "arc_phase: str", "LineRequest.arc_phase field"),
        (composer, "ARC_PHASE_GUIDANCE", "ARC_PHASE_GUIDANCE rendered in prompt"),
        (budget, "def compute_episode_budget", "compute_episode_budget defined"),
        (budget, "ACT_COUNT_CONFIG", "ACT_COUNT_CONFIG table"),
    ]
    for source, needle, label in checks:
        if needle in source:
            print(f"  PASS: {label}")
        else:
            print(f"  BAD: missing {label} ({needle!r})")
            issues += 1
    return issues


def audit_phase_completeness():
    print("\n=== AUDIT 5: phase-completeness markers ===")
    issues = 0
    composer = (NODES_DIR / "_otr_line_composer.py").read_text(encoding="utf-8")
    reviewer = (NODES_DIR / "_otr_ledger_reviewer.py").read_text(encoding="utf-8")
    ledger = (NODES_DIR / "production_ledger.py").read_text(encoding="utf-8")
    consumers = (NODES_DIR / "_otr_ledger_consumers.py").read_text(encoding="utf-8")

    checks = [
        (composer, "def build_allowed_roster", "Phase 0: build_allowed_roster"),
        (composer, "def detect_phantom_names", "Phase 0: detect_phantom_names"),
        (composer, "def aggregate_compose_flags", "Phase 0: aggregate_compose_flags"),
        (composer, "class LineResult", "Phase 0: LineResult dataclass"),
        (composer, "def render_outline_spine", "Phase 1: render_outline_spine"),
        (composer, "def build_voice_card", "Phase 1: build_voice_card"),
        (ledger, "def init_lines_from_outline", "Phase 2B: init_lines_from_outline"),
        (ledger, "def update_line_text", "Phase 2B: update_line_text"),
        (ledger, "def stamp_word_counts", "Phase 2A §6.G: stamp_word_counts"),
        (ledger, "def peek_ledger", "Phase 3: peek_ledger"),
        (ledger, "def has_current_ledger", "Phase 3: has_current_ledger"),
        (ledger, "self.data = merged", "Wiring-review #5: save self.data sync"),
        (reviewer, "def review_ledger", "Phase 3: review_ledger orchestrator"),
        (reviewer, "def audit_cast_contract", "Phase 3: audit_cast_contract"),
        (reviewer, "def auto_remap_phantom", "Phase 3: Levenshtein auto-remap"),
        (reviewer, "def apply_phantom_skip_fallback", "Phase 3 Step 2.5"),
        (reviewer, "_ALLOWED_DOCTOR_ROLES", "Phase 3 doctor scope guard"),
        (reviewer, "audit_failed: bool", "Wiring-review #8: audit fail-loud"),
        (consumers, "def iter_lines", "Rec 5: iter_lines"),
        (consumers, '"skip"', "Rec 5: skip honor"),
    ]
    for source, needle, label in checks:
        if needle in source:
            print(f"  PASS: {label}")
        else:
            print(f"  BAD: missing {label} ({needle!r})")
            issues += 1
    return issues


def main():
    bad = 0
    bad += audit_legacy_refs()
    bad += audit_beat_by_beat_writes()
    bad += audit_consumer_iter_lines()
    bad += audit_act_count_thread()
    bad += audit_phase_completeness()
    print("\n" + "=" * 60)
    if bad == 0:
        print("AUDIT PASS: all phases complete + clean")
        return 0
    print(f"AUDIT FAIL: {bad} issue(s) found")
    return 1


if __name__ == "__main__":
    sys.exit(main())
