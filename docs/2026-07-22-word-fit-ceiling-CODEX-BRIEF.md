# CODEX BRIEF -- 2026-07-22

> **HISTORICAL / SUPERSEDED 2026-07-30.** Do not implement this brief's fixed
> twelve-candidate ceiling, `OTR_MAX_WORD_FIT_CANDIDATES`, or
> `WordFitCeilingExceeded` proposal on the canonical Sci-Fi candidate route.
> The current operator ruling requires fresh complete model-authored
> candidates until a downstream-valid ledger can be accepted or the operator
> cancels. Deterministic configuration, security, provider, I/O, and invariant
> failures remain loud. The current plan of record is
> `docs/2026-07-30-story-never-fails/FINAL_PLAN.md`.

**Word-fit delivery campaign hangs forever (all 6 banks) + kibitz agy model.**
Author: Claude (Cowork). Branch: v2.0-alpha. Priority: P0 (Task 1), P1 (Task 2).

FIX IS NOT YET APPLIED. This brief is the full spec; Codex implements + verifies.

---

## TASK 1 (P0): Bound the unbounded word-fit campaign

### Symptom (live-verified, 2026-07-22)
- The 6-bank sweep (`tmp/sixbank_sweep_20260722/`) HUNG at LEG 1
  (media_archive @120w) for 90 min. Client polled `status=pending` to
  t=5396s, then was killed. Zero episodes produced; the whole 12-leg sweep
  was blocked behind leg 1.
- `server.log` tail = thousands of alternating
  `[Selector] slot=creative reuse cache` / `slot=technical reuse cache`
  and NOTHING else (no generation, no pass markers).

### Root cause
1. Story writes fine (episode "Ink on Transport Papers", 152 words).
2. `normalize_length` fails PostValidation twice
   ("Guard1: duplicate edit on beat_index 3, action SPLIT_LINE") ->
   StructuredCallFailedError, CAUGHT as a WARNING by StorySpine
   (best-effort, non-fatal -- a red herring, not the hang itself).
3. The FINAL word-count fitter then runs:
   `fit_final_word_delivery_campaign` in `nodes/_otr_radio_editor.py`.
   It is a bare `while True:` -- author a fresh candidate (creative slot),
   fit it (technical slot), stamp the word band -- that exits ONLY on
   (a) an in-band accept, or (b) an `insufficient_active_rows_capacity` raise.
4. `_OTRWD.accept_word_fit_candidate` HARD-REFUSES any out-of-band candidate.
   So when capacity is sufficient but the model never lands in the band,
   there is NO exit -> infinite re-authoring. `_outer_liveness_state`
   docstring says so: "deliberately no outer model-output ceiling ... until
   the ledger stamp accepts one (or the operator cancels the run)".
   policy = "unbounded_model_output_retries_until_ledger_legal".

### All-banks audit (the same bug in 3 copies -- ONE fix covers all)
Three writer families each own an unbounded `while True` delivery campaign;
ALL retire through the SAME `_OTRWD.retire_word_fit_candidate` and accept
through `_OTRWD.accept_word_fit_candidate`:

- `nodes/_otr_radio_editor.py` -> `fit_final_word_delivery_campaign`
  (outer while True ~2811 + inner `_author_next_candidate` while True ~2803;
  retire @2821, 2849, 2891; accept @2902). Banks: **media_archive, original,
  public_domain, shakespeare, science_news** (ledger/news path; entry
  `OTR_LedgerScriptWriter.py:7630`). THIS is what hung.
- `nodes/_otr_scifi_fable2.py` -> delivery loop (while True @5249; retire
  @5263, 5382; accept @5475), `owner="scifi_news_pro"`. Bank: **scifi_news_pro**.
- `nodes/_otr_scifi_codex.py` -> delivery loop (while True @6361; retire
  @6383, 6466, 6511, 6544; accept @6790). Bank: **scifi_news**.

=> A single fail-closed ceiling inside `retire_word_fit_candidate`
(`nodes/_otr_word_delivery.py`) bounds all three at once -- every bank.

### Why raising from retire propagates cleanly (verified, not swallowed)
In every loop `retire_word_fit_candidate` is called INSIDE an `except`
handler for a DIFFERENT exception type (`_InlineCandidateGenerationError` /
`FinalWordDeliveryError` / `WordDeliveryError` / `Fable2ScriptError` /
`CodexSpokenTextError`) -- so a raise from retire is never re-caught by the
same handler. fable2/codex wrap delivery in an outer try that converts
`WordDeliveryError` -> `Fable2WordDeliveryError` / `CodexWordDeliveryError`
(fail-closed at the node boundary); radio_editor propagates up to
OTR_LedgerScriptWriter. All fail closed; none swallow it.

### The fix (root cause, no shim) -- nodes/_otr_word_delivery.py

(1) Add `import os` to the top import block (next to hashlib/json/math).

(2) After `MAX_CONSECUTIVE_REPAIR_STALLS = 4`, add:

    DEFAULT_MAX_OUTER_WORD_FIT_CANDIDATES = 12


    def _resolve_outer_candidate_ceiling() -> int:
        raw = os.environ.get("OTR_MAX_WORD_FIT_CANDIDATES", "")
        try:
            n = int(str(raw).strip())
        except (TypeError, ValueError):
            return DEFAULT_MAX_OUTER_WORD_FIT_CANDIDATES
        return n if n >= 1 else DEFAULT_MAX_OUTER_WORD_FIT_CANDIDATES

(3) After `class WordDeliveryError(RuntimeError): ...`, add:

    class WordFitCeilingExceeded(WordDeliveryError):
        """Outer producer-candidate campaign hit its fail-closed ceiling.

        Subclasses WordDeliveryError so every existing `except
        WordDeliveryError` boundary still fails closed; the distinct type
        lets callers/tests identify a runaway campaign specifically.
        """

(4) In `retire_word_fit_candidate`, the tail currently is:

        for key in (
            "freeze_verdict", "freeze_report", "freeze_audit",
            "audio_ready", "video_ready", "media_ready", "obs_ready",
        ):
            meta.pop(key, None)
        return dict(row)

Insert the ceiling check just BEFORE `return dict(row)`:

        ceiling = _resolve_outer_candidate_ceiling()
        if int(state["active_candidate_index"]) >= ceiling:
            raise WordFitCeilingExceeded(
                f"{str(owner)!r} outer word-fit campaign retired "
                f"{int(state['active_candidate_index'])} candidates without "
                f"an in-band accept (ceiling {ceiling}); "
                f"boundary={str(boundary)!r}, last discard reason="
                f"{row['discard_reason']!r}. Failing closed instead of "
                f"spinning forever (OTR_MAX_WORD_FIT_CANDIDATES tunes this)."
            )
        return dict(row)

(5) Update `_outer_liveness_state` docstring + the `policy` default string in
    _otr_word_delivery.py to say BOUNDED, not "no outer ceiling".
    GREP FIRST: `git grep -n unbounded_model_output_retries_until_ledger_legal`
    and `git grep -n "no outer"`. If a test asserts the literal policy string,
    update that test in the SAME change (the unbounded policy WAS the bug).

(6) Update the `fit_final_word_delivery_campaign` docstring in
    _otr_radio_editor.py ("intentionally no outer model-output ceiling" ->
    "bounded by the shared fail-closed candidate ceiling").

### NO workflow JSON change
Pure internal safety bound -- no node / widget / wiring change.
`workflows/otr_canonical.json` stays untouched (confirm, per CLAUDE.md section 0).

### Tests (add, then run the full gate)
- Unit (near existing _otr_word_delivery tests): a campaign that never
  accepts raises `WordFitCeilingExceeded` at the ceiling; env
  `OTR_MAX_WORD_FIT_CANDIDATES` override respected; a run that accepts on
  candidate k < ceiling still succeeds (no false trip).
- Suite: `cd <repo>; $env:PYTHONUTF8=1;
  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
- Bug Bible: `cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide;
  <venv python> tests\bug_bible_regression.py`

### Live proof (REQUIRED before commit-green, per CLAUDE.md sections 4-6)
- Reset the box: SELECTIVE CIM kill of the ComfyUI server + sweep pythons by
  CommandLine (NEVER a blanket `Stop-Process -Name python` -- it severs the
  Claude MCP pythons). Confirm :8000 empty and nvidia-smi VRAM at desktop
  baseline before booting.
- Run `tmp/_six_bank_sweep_20260722.ps1` (6 banks x 120/320 = 12 legs).
- PASS: every leg reaches RESULT SUCCESS + obs_publish OK + asset on disk,
  OR fail-closed FAST (minutes, WordFitCeilingExceeded). NO leg may hang.

SCOPE NOTE: the ceiling stops the HANG; it does NOT by itself make a
non-converging episode GREEN -- such a leg now FAILS FAST instead of hanging.
If a leg raises WordFitCeilingExceeded, that is a SEPARATE convergence bug --
start with (a) the `normalize_length` "duplicate edit on beat_index N" repair
prompt/guard (the model re-emits two edits on the same beat_index; the repair
prompt should forbid a duplicate beat_index), and (b) word-band reachability
for short news episodes. Report which banks are green vs which fail-closed.

### Commit / push (CLAUDE.md section 7)
v2.0-alpha; commit code + tests together; push same session; verify
HEAD==origin, no BOM, AST parse on touched .py. After live proof, log the
PBUG in `PROD_BUG_LOG.md` + a Bug Bible entry (admission rule: a live-verified
production hang qualifies).

---

## TASK 2 (P1): kibitz agy model -> Gemini 3.6 Flash (High)
- Operator's NEW preferred agy (Antigravity) fan-out/QA model is
  **Gemini 3.6 Flash (High)** (supersedes the old 3.5 Flash High note).
- agy needs the EXACT DISPLAY NAME, not a slug -- a bad slug makes the whole
  kibitz arc silently run codex-only. Set env
  `KIBITZ_AGY_MODEL="Gemini 3.6 Flash (High)"`.
- Read by `scripts/kibitz.py` (env `KIBITZ_AGY_MODEL`). The skills-cache copy
  is READ-ONLY -- pass via env / update the default in `scripts/kibitz.py`;
  do not edit the cache. Verify `antigravity.log` is non-empty per round
  (not just codex.md).
- Picker (2026-07-22): Gemini 3.6 Flash (Low/Med/High), 3.5 Flash
  (Low/Med/High), 3.1 Pro (Low/High), Claude Sonnet/Opus 4.6 (Thinking),
  GPT-OSS 120B. Keep agy on a Gemini for family diversity.
