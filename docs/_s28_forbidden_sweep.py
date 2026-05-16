# S28 forbidden-pattern sweep (Phase 5 hand-off artifact).
#
# Walks the s27-cleanbreak-tail..HEAD diff of every *.py file and
# pulls every ADDED line ('+') that matches one of the S28 forbidden
# patterns. Then opens the actual source file and checks whether
# that line is inside a docstring (via tokenize), because git diff
# context isn't wide enough to track docstring state across a
# typical hunk.
from __future__ import annotations

import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root

DIFF_PATH = ROOT / "docs" / "s28_diff_tmp.txt"
OUT_PATH = ROOT / "docs" / "2026-05-13-S28-new-forbidden-hits.txt"

forbidden = re.compile(
    # S28 original markers
    r"DeprecationWarning"
    r"|back-compat"
    r"|legacy fallback"
    r"|legacy shape"
    r"|\bshim\b"
    r"|\balias\b"
    r"|\botr_legacy_audio_dir\b"
    r"|budget is None"
    # S29 Phase 6.2 extinction markers (per S29 plan §Phase 6.2).
    # Anything that was extincted in S28 + S29 STAYS extincted.
    r"|req\.budget is None"
    r"|polish_generate_fn is not None"
    r"|hasattr\(self\.budget"
    r"|DEPRECATED_manifest"
    r"|C:/Users/jeffr"
    r"|\bOTR_LedgerScriptReviewer\b"
    r"|\bGemma4\b"
    r"|\breviewer_verdict\b"
    # S30 B2c: cleanup_model_id was a legacy normalizer + widget that
    # the two-model selector replaces. The widget was already gone
    # pre-S30; this marker locks the SYMBOL out so any future
    # contributor cannot reintroduce a function / parameter / variable
    # by that name. Forensic mentions (string literals in legacy guard
    # lists, doc references) are suppressed by the tokenize classifier.
    r"|\bcleanup_model_id\b"
    # S30 B7 (2026-05-14): extinction markers from the two-model
    # selector sprint. Each name appears verbatim in some deleted
    # symbol; reintroduction trips the sweep loud.
    r"|\bOTR_VisualLLMSelector\b"           # B5: deleted node class
    r"|\bVisualLLMSelector\b"               # B5: deleted node class (no OTR_ prefix form)
    r"|\b_LLM_MODEL_CHOICES\b"              # B5: deleted dropdown literal (visual selector)
    r"|\b_MODEL_CHOICES\b"                  # B2a: deleted writer dropdown literal
    r"|\bDEFAULT_MODEL_ID\b"                # B3: deleted DEFAULT_MODEL_ID at cascade scope
    r"|\b_POLISH_CACHE\b"                   # B5: collapsed into LLM_CACHE
    r"|\bMODEL_CONTEXT_CAPS\b"              # B1b: deleted static context-cap dict
    r"|\bDEFAULT_CONTEXT_CAP\b"             # B1b: deleted default-cap fallback
    r"|\bOTR_LFCPhase4Scene\b"              # B4: deleted standalone node class
    r"|\bOTR_LFCPhase5Voice\b"              # B4: deleted standalone node class
    r"|\bOTR_LFCPhase6Arc\b"                # B4: deleted standalone node class
    r"|\benable_phase_3_polish\b"           # B3/B4: deleted cascade-toggle widget
    r"|\bpolish_announcer_beats\b"          # B3/B4: deleted cascade-toggle widget
    r"|\benable_phase_4_scene_coherence\b"  # B3/B4: deleted cascade-toggle widget
    r"|\benable_phase_4_5_smart_suggestion\b"  # B3/B4: deleted cascade-toggle widget
    r"|\benable_phase_5_voice_drift\b"      # B3/B4: deleted cascade-toggle widget
    r"|\benable_phase_6_episode_arc\b"      # B3/B4: deleted cascade-toggle widget
    r"|\bPhase3PolishReport\b"              # B4: deleted dataclass
    # S31 B5 (2026-05-14): extinction markers from the legacy LLM
    # stack clean break sprint. Each name was a runtime symbol in
    # `nodes/story_orchestrator.py` deleted at S31 B4 (Hard rule #1A
    # non-deferrable). Reintroduction trips the sweep loud.
    r"|\b_load_llm\b"                       # S31 B4: deleted legacy load entrypoint
    r"|\b_unload_llm\b"                     # S31 B4: deleted legacy teardown
    r"|\b_LLM_CACHE\b"                      # S31 B4: deleted legacy cache dict
    r"|\b_generate_with_llm\b"              # S31 B4: deleted legacy generate wrapper
    # S31 B5: preemptive lock -- one generate surface (Hard rule #5).
    # Any wrapper-by-another-name on top of `make_generate_fn` is
    # forbidden. These two literal names cover the "rename to drop the
    # underscore prefix" and "name it without the verb suffix"
    # reintroduction paths.
    r"|\bgenerate_text\b"                   # S31 B5: preemptive lock on Hard rule #5
    r"|\bgenerate_with_llm\b"               # S31 B5: preemptive lock (no-underscore variant)
    # S31.5 B3 (2026-05-14): wrapper eliminated when the
    # `register_vram_cleanup` contract audit confirmed the caller
    # already wraps each callback in try/except. Adding the marker
    # locks against reintroduction even though the wrapper name is
    # specific enough that accidental restoration is unlikely.
    r"|\b_vram_cleanup_via_loader\b"        # S31.5 B3: eliminated wrapper
    # S32 B4 (2026-05-14): widget rejected; critic always routes
    # creative. Per-beat T-dispatch in differing-slots adds ~3.3 hr
    # of VRAM transition overhead per episode. No-widget rule
    # (Jeffrey: features useful are on; default-OFF gates on
    # architecturally-rejected paths are dead-code maintenance debt).
    r"|\buse_technical_critic\b"            # S32 B4: widget rejected
    # S33 B2 (2026-05-15): cascade rollback-gate retirements per the
    # refined no-auditors rule. The two verdict literals are the
    # surface symptoms of the rollback gates; locking the names
    # prevents accidental reintroduction.
    r"|\bcast_unrecoverable\b"              # S33 B2: retired rollback-gate verdict
    r"|\bpost_audit_failed\b"               # S33 B2: retired rollback-gate verdict
    # S33 B4 (2026-05-15): pipeline-cutting phantom handlers retired
    # per B1.5 classification.
    r"|\bapply_phantom_skip_fallback\b"     # S33 B4: pipeline-cut mute helper
    r"|\b_final_phantom_check\b"            # S33 B4: report-only helper
    # S33 B5 (2026-05-15): polish design lock. The two prompts must
    # stay split (character vs announcer). Lock against accidental
    # collapse into a unified prompt under any common bad name.
    r"|\b_POLISH_SYSTEM_PROMPT_UNIFIED\b"   # S33 B5: unify-attempt name #1
    r"|\b_UNIFIED_POLISH_PROMPT\b"          # S33 B5: unify-attempt name #2
    # Sprint C C2a (2026-05-15): era literal cleanbreak -- visual layer.
    # Production code in visual/ and workflows/ no longer carries era
    # anchors (1940s noir radio drama style / 1980s broadcast aesthetic).
    # Tokenize suppression handles forensic mentions in docstrings and
    # comments. Test fixture at tests/test_musicgen_cache_keys.py:23
    # retains its noir-literal cache-key payload by design (cache-key
    # stability) -- the diff-based sweep only flags NEW additions, so
    # the pre-existing fixture is naturally exempt.
    r"|\b1940s\b"                           # C2a: visual-layer era literal
    r"|1980s broadcast"                     # C2a: video-plan era literal
    # Sprint C C2b (2026-05-15): era literal cleanbreak -- orchestrator.
    # The rerank prompt era anchor at line 1596 was rewritten to be
    # style-neutral. SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT
    # constants (which carried OMNI-RETRO 5-pillar / golden-age-radio
    # era flavor) were orphans from the pre-LPL pipeline and were
    # deleted entirely (no consumers; no shim). These markers prevent
    # accidental reintroduction of period-locked language inside any
    # new orchestrator-layer LLM prompt. Tokenize suppression handles
    # forensic comment / docstring mentions (e.g. the C2b deletion
    # forensic note itself, and the still-orphan _load_canon_for_writer
    # docstring that still names SCAFFOLDING_PREAMBLE as historical
    # context until the Sprint G orphan sweep).
    r"|1950s Americana"                     # C2b: 5-pillar era anchor
    r"|golden.age radio"                    # C2b: dramaturg era anchor
    r"|\bOmni.Retro\b"                      # C2b: 5-pillar branding
    r"|\bOrson Welles\b"                    # C2b: dramaturg name-drop
    r"|\bNorman Corwin\b"                   # C2b: dramaturg name-drop
    r"|\bLucille Fletcher\b"                # C2b: dramaturg name-drop
)
diff_file_re = re.compile(r"^\+\+\+ b/(.+)$")
hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def classify_lines(path: Path) -> dict[int, str]:
    """Return a map of 1-indexed line number to a classification:
    "code", "comment", "string" (docstring or triple-quoted)."""
    src = path.read_text(encoding="utf-8", errors="replace")
    classifications: dict[int, str] = {}
    try:
        toks = list(tokenize.generate_tokens(iter(src.splitlines(keepends=True)).__next__))
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        # Fall back to "code" for every line if tokenize chokes.
        for i in range(1, src.count("\n") + 2):
            classifications[i] = "code"
        return classifications
    string_lines: set[int] = set()
    comment_lines: set[int] = set()
    # S30 B7: also classify Python 3.12 f-string tokens
    # (FSTRING_START, FSTRING_MIDDLE, FSTRING_END) as string context.
    # Without this, an f-string embedded in code (e.g. `raise
    # AssertionError(f"... _POLISH_CACHE ...")`) would land on a line
    # that has both STRING and code tokens; the classifier picks
    # the dominant non-string token type and the line gets marked as
    # "code", producing a spurious runtime hit on a forensic mention.
    _STRING_TOKEN_TYPES = (
        tokenize.STRING,
        getattr(tokenize, "FSTRING_START", -1),
        getattr(tokenize, "FSTRING_MIDDLE", -1),
        getattr(tokenize, "FSTRING_END", -1),
    )
    for tok in toks:
        if tok.type in _STRING_TOKEN_TYPES and "\n" in tok.string:
            start_l, _ = tok.start
            end_l, _ = tok.end
            for L in range(start_l, end_l + 1):
                string_lines.add(L)
        elif tok.type in _STRING_TOKEN_TYPES:
            # Single-line string — these don't span lines so just
            # mark the one.
            start_l, _ = tok.start
            string_lines.add(start_l)
        elif tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
    total = src.count("\n") + 1
    for i in range(1, total + 1):
        if i in string_lines:
            classifications[i] = "string"
        elif i in comment_lines:
            classifications[i] = "comment"
        else:
            classifications[i] = "code"
    return classifications


# Walk the diff, collecting (file, new_line_number, line_text) for
# every `+` line that hits a forbidden pattern.
hits_raw: list[tuple[str, int, str]] = []
current_file: str | None = None
current_line: int | None = None
for raw in DIFF_PATH.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
    m = diff_file_re.match(raw.rstrip("\n"))
    if m:
        current_file = m.group(1)
        current_line = None
        continue
    if raw.startswith("---"):
        continue
    mh = hunk_re.match(raw.rstrip("\n"))
    if mh:
        current_line = int(mh.group(1))
        continue
    if current_line is None or current_file is None:
        continue
    side = raw[:1]
    if side == "+":
        if forbidden.search(raw):
            hits_raw.append((current_file, current_line, raw))
        current_line += 1
    elif side == " ":
        current_line += 1
    # `-` lines do not advance the new-side counter

# For each hit, look at the actual file and classify.
classified_cache: dict[str, dict[int, str]] = {}
forensic: list[str] = []
runtime: list[str] = []
for fpath, lineno, raw in hits_raw:
    abs_path = ROOT / fpath
    if not abs_path.exists():
        runtime.append(f"{fpath}:{lineno} [missing-file] {raw}")
        continue
    cls = classified_cache.get(fpath)
    if cls is None:
        cls = classify_lines(abs_path)
        classified_cache[fpath] = cls
    kind = cls.get(lineno, "code")
    line_str = f"{fpath}:{lineno} [{kind}] {raw}"
    if kind in ("string", "comment"):
        forensic.append(line_str)
    else:
        runtime.append(line_str)

OUT_PATH.write_text("".join(runtime), encoding="utf-8")
print(f"HITS: {len(hits_raw)}  forensic: {len(forensic)}  runtime: {len(runtime)}")
print(f"OUT:  {OUT_PATH}")
if forensic:
    print("--- forensic (suppressed from output) ---")
    for l in forensic:
        print("  " + l.rstrip("\n"))
if runtime:
    print("--- runtime (in output file) ---")
    for l in runtime:
        print("  " + l.rstrip("\n"))
