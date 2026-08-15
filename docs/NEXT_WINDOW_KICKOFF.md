# Next-window kickoff (paste this whole thing)

Written 2026-08-14 at wrap-up. Replaces itself every session -- if the HEAD or
the baselines below do not match reality, say so before building on them.

---

Resume the OTR build. CODER window. Repo:
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch v2.0-alpha, HEAD ea6939ac, equal to origin.

BOX: a ComfyUI server is RESIDENT on port 8000 holding ~12.4 GB. That is
finished-run behaviour, not a crash. RESET IT before any headless run -- kill
SELECTIVELY by CommandLine via CIM, never a blanket python kill, it severs the
MCP tools.

DO NOT ASK ME ANYTHING. Every decision below is settled. Where something is
genuinely open, a DEFAULT is given -- take it, record it in your receipt, and
keep going. Work until the item is done or you hit a wall that needs me.

READ FIRST, then do not re-derive any of it:
  docs/GO_FORWARD_PLAN.md  -- PRIORITY 1 at the top, rewritten and forward-only
  docs/HANDOFF_LOG.md      -- the newest entry (2026-08-14 CODER). It carries
                              the measurement tables. They cost hours. Reuse.
  scripts/otr_ledger_view.py -- the module docstring IS the contract

BASELINES -- detect drift against these, measured 2026-08-14:
  full suite             10445 passed / 110 skipped / 1 xfailed, 0 failures
  Bug Bible              20 passed / 26 skipped / 3 xfailed, 280 entries
  build_variants --check 50 variants, 0 failures
If your first run differs, explain the delta before building on it.

=== ALREADY DONE TODAY. DO NOT REDO, DO NOT REOPEN. ===
PBUG-20260814-01/02/03 are CLOSED and each is proven on a generated episode,
not on the suite:
  * `speaker` on every ledger line row. Proven live on ALL THREE code paths --
    codex lane, writer lane (covers original/shakespeare/public_domain/
    media_archive), and scifi_news_pro. Bible 12.101.
  * The coda is P6, its own pass, and it NAMES THE SOURCE. Verified on air:
    AstroRad, StemRad, Artemis I, 60 percent, 26 -> 16 kilograms.
  * Per-beat dialogue + a per-scene review, with per-job decode budgets.
  * Four schema caps that refused legitimate sources. Bible 12.102.
  * scripts/otr_ledger_view.py -- grade / --watch / --ladder / --html.
Do NOT write a second ledger grader. One exists and is corpus-graded.
Do NOT delete gemma-2-2b-it -- it is the fastest writer-lane model on the box
(5.1 min for a clean 3-act `original`). It simply cannot drive the two news
lanes' evidence passes, and never could.

=== THE WORK, IN THIS ORDER ===

1. THE CLEAN STAGE. This is the only thing between five banks and "passable".
   F1 (action inside a spoken row) is 11-40% of rows on every bank except
   scifi_news -- media_archive 40%, scifi_news_pro 32%, original 30%. Those
   lanes ALREADY write one model call per beat, so job size is not their
   problem and today's per-beat fix cannot help them.
   Shape (already specified in GO_FORWARD): code DETECTS, a MODEL repairs,
   every pass DETECTOR-GATED so a clean episode costs zero cleaning calls.
   The detector half EXISTS -- `otr_ledger_view.grade()` is the same F1/F2
   grader for every bank. Reuse it; do not grow a second opinion about what
   "action" means. Repair model = creative_writing_model. Bounded retries,
   each told what was still wrong, then flag loudly in the ledger and
   CONTINUE. Never a silent pass, never a hard stop.
   The repair THINKS, it does not strip: the model gets the line, the speaker
   and the acts/beats so far, and returns the best edit -- usually removing
   the stage business AND adjusting the dialogue so the moment still plays.
   GRADE THE FIDELITY LANES WITH CARE: on shakespeare and public_domain the
   author's own language is carried as written, so a third-person construction
   from the source is NOT a defect. The detector over-reports there.

2. THE GRADUATED EXTRACTION CONTRACT (operator ruling, end of session):
   "if it fails once on extraction, we relax the extraction requirements on
   the second pass -- it just has to get the gist of the story and populate
   the coda." Attempt 1 strict; attempt 2 drops `source_spans` (the literal
   quote transcription a small model cannot do) and keeps fact claims, entity
   names and numbers -- exactly what the coda anchors need, nothing more.
   THE PART THE RULING DOES NOT SAY, AND IT IS LOAD-BEARING: stamp WHICH
   CONTRACT produced the index. A relaxed extraction is not span-proven.
   Enumerate every span reader (_span_ok, _span_mismatch, the citation audit,
   _otr_scifi_source_repair) and give each a defined behaviour BEFORE writing
   the relaxed pass -- the standing "a ripped pass may not leave a ledger
   field unowned" rule applied to a field that becomes conditionally absent.

3. THE MODEL FLOOR SHOULD REFUSE, NOT GRIND. A model that cannot satisfy a
   lane's contract should be refused at the top of the run WITH THE REASON,
   not discovered after 34 minutes of bounded retries. That grind is the real
   defect, not the floor.

=== SETTLED RULINGS -- do not ask, these are decided ===
* Only TWO things are failure: action in the ledger, and a character speaking
  another character's lines. NOTHING ELSE. Story quality is NOT a defect
  (2026-08-04 directive); character consistency IS.
* Code may DETECT and explain. Only a MODEL pass may rewrite prose. No Python
  stripper, no shim, no regex surgery on a line.
* No word-count authority anywhere. act_count 1..8 is the only knob that
  shapes an episode; length is an observation.
* Runaway guards are code-side and STAY. Right-size the JOB; never raise the
  guard.
* One prompt per job for every model tier. Vary the JOB SIZE, not the text.
* The ledger holds spoken lines and music cues only. music_* rows are
  load-bearing (audio slicing, still keys, video tail window) and stay.
* Rerolls split by POSITION: upstream casting rerolls STAY; rerolls that
  redraw written story GO.
* Shakespeare and public_domain: FIDELITY OUTRANKS ARC. A faithful scene that
  ends unresolved is a PASS.
* A schema cap must never sit at the number a downstream trim already
  enforces (Bible 12.102). Ceilings are backstops; limits live at the trim.

=== GENUINELY OPEN -- TAKE THE DEFAULT, RECORD IT, MOVE ON ===
* The codex lane carries the same cap-equals-trim shape on MAX_FACT_ROWS (6)
  and MAX_ENTITY_ROWS (4). That lane decodes under a grammar, so it TRUNCATES
  rather than refuses -- silent evidence thinning, unproven on an artifact.
  DEFAULT: leave it. Prove it on an artifact before touching it.
* P3 abandons 11 of 31 ladders on cast_coverage (the score forgetting the
  announcer a beat). DEFAULT: leave it; it is the strongest candidate for a
  schema fix later, not now.
* The creativity knob does nothing on scifi_news (the codex lane hard-codes
  .72). Three options are written up in
  docs/2026-08-14-temperature-problem-statement.md. DEFAULT: leave it, it is
  the operator's call.
* Lane rename (codex/fable2 -> scifi news pro/non-pro) is QUEUED and APPROVED,
  sequenced AFTER the above. DEFAULT: do not start it. When you do, build the
  seam-resolution check FIRST -- the 12 seam ids are prompt ROUTING KEYS and
  an inconsistent rename silently kills a seam.

=== GATES, EVERY TURN, NON-NEGOTIABLE ===
full suite; Bug Bible (cd to
C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide, sync to
origin/main, run the RELATIVE path tests\bug_bible_regression.py);
scripts/build_variants.py --check; AST parse on touched .py; BOM check (no
EF BB BF) on every touched file; HEAD == origin after every push.
Commit AND push together to v2.0-alpha per green chunk -- local-only commits
are the failure mode. Stage BY PATHSPEC or `git add -u`; NEVER `git add .` --
there are untracked OPERATOR files (config/profiles/otr_sbcov_*.json,
config/source_banks/_corpus/, docs/H3_LICENSE_ATTESTATION*.md, kibitz/,
scratch_check_server.py, uv.lock). Do not touch, move or commit them.

Review routing per the 2026-08-11 directive: NO full kibitz r1-r4 arc. Codex
CLI is the consult for a genuine quandary or a third failed fix; Sonnet 5 runs
post-coding QA on the finished diff before the push. A scoped tail is NEVER
reported as a full arc. Two-strikes floor stands: a bug surviving two of your
fixes gets a consult before the third swing.

=== HOW TO PROVE IT. THIS IS THE POINT. ===
"The tests pass" is NOT success. Only a story proves a story fix.
  interpreter  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
  boot         scripts/_otr_soak_server_launch.cmd, with
               OTR_WRITER_HEARTBEAT_EVERY=16 for a readable live stream
  run          python scripts/otr_canonical_api_run.py --workflow
               workflows/otr_story_only.json --source-bank <bank>
               --act-count 3 --creative-model '<exact dropdown string>'
  MODEL FLOOR  scifi_news / scifi_news_pro need gemma-4-12b-it or
               Mistral-Nemo. gemma-2-2b-it is fine and FAST on the other four.
  grade        python scripts/otr_ledger_view.py            (newest episode)
               python scripts/otr_ledger_view.py --ladder 60
  watch live   python scripts/otr_ledger_view.py --watch 3 --html tmp/otr_live.html
               then open that file -- it repaints itself and turns red,
               naming the phrase, when the model repeats.

Report what the STORY did, not what the suite did.
