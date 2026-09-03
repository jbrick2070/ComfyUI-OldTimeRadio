# OTR -- Project Operating Rules (Claude in Cowork)
**Operator directives -- these win over any handoff, doc, or memory that disagrees.**
- **Fix bugs properly, at the root cause -- never a shim or band-aid.** Don't wait for me to
  fix anything: make the fix yourself, without asking. If it works, it's fine.
- **NO CONTENT GUARDRAILS ON GENERATED EPISODES (operator directive 2026-08-03 -- supersedes
  the older "no curse words anywhere / SFW always" line for EPISODE CONTENT).** Operator:
  "we [have] too [many] guardrails, no violence or swearing guardrails, they just cause
  problems" and "I've given up chasing profanity." Do not add profanity or violence filtering
  to the generation path, and do not reintroduce prompt clauses that forbid the source's own
  content -- the adaptation packs were literally instructing the model to avoid "blood, guns,
  knives, and graphic violence" while adapting MACBETH and KING LEAR, which is a fidelity
  defect, not a safety win. On the adaptation lanes the author's own language is carried as
  written; Wells' Editor may shout "Story be damned!" because Wells wrote it.
  **Still true, and a different thing:** authoring style stays clean -- no curse words in
  CODE, COMMENTS, LOGS or commit messages, and never the name "dummy". That rule is about what
  WE write, not about what the pipeline generates from a source.
- **STORY QUALITY IS DONE. STOP CHASING IT (operator directive 2026-08-04 -- hard).**
  Operator: "I am not chasing story quality anymore. It works. It works. I will publish it
  as open source, and if someone else wants to do it, or in six months I wanna chase it
  again when I've got better tools, I will." The scripts are ACCEPTED as they are. Do not
  open writing-quality work, do not propose prompt-craft passes for better prose, do not
  benchmark writer models for story quality, and do not spend on panels or cloud models to
  improve scripts. This also settles the cloud-writer question: local stays the default and
  no paid writer is adopted to raise prose quality.
  **What this does NOT cover** -- these are CORRECTNESS defects, not quality chasing, and
  remain open: a character's gender/voice contradicting the source, a character's face
  changing between beats, voice-pool staleness, and any structural or ledger fault. Fixing
  "Malvolio speaks with a woman's voice" is a bug fix; rewriting Malvolio's dialogue to be
  better is not.
- **NO WORD-COUNT CHASING (operator directive 2026-08-03).** "We never chase word count."
  The target words value is a REQUEST, not a gate: no refusals, no hard caps, no shunts. The
  manifest `recommended_word_budget` upper bound is removed for exactly this reason. The only
  real ceiling is structural (the beat topology tops out near 1,520 spoken words), and a
  request beyond it simply delivers the closest performable episode.
- **EVERY CODING ITEM GETS A FULL KIBITZ (operator directive 2026-08-04 -- hard, stands until
  the operator withdraws it).** Operator: "for the next coding sprint be sure any coding work
  has a full `/kibitz-plugin:kibitz` review." **FULL means the default four-round arc** --
  r1 arc -> r2 coding -> r3 wiring -> r4 convergence, 8 external calls (two reviewers x four
  rounds). Not a scoped tail, not one round, not a continuation receipt; a partial campaign
  may never be reported as a full arc. **Invoke the PLUGIN skill by name:
  `kibitz-plugin:kibitz`** -- `anthropic-skills:kibitz` is the older duplicate and is not what
  the operator asked for. Panel is driver-aware: Claude drives from Cowork, so **Codex +
  Antigravity** review and you do NOT launch a second `claude -p` lane against your own family.
  Use the ComfyUI profile overlay already in the repo, `.kibitz/comfyui.local.md` (written
  2026-07-11 -- regenerate with `kibitz/scripts/comfyui_profile.py` if the tree has moved past
  it). You still write the code-grounded `driver_anchor.md` FIRST and remain the sole judge:
  the panel proposes, Claude disposes, and every panel claim is checked against the real
  Windows files before it is folded in. This RAISES the two-strikes rule below -- a first-try
  root fix now gets a panel too. It is local, it is $0, and it costs a wait, not a budget.
- **AMENDED 2026-08-17 (operator): MATCH THE REVIEW TO THE TASK -- a full arc is not the
  answer to everything.** Operator: *"we aren't running a full kibitz on everything, it needs
  to choose the best path for the right task."* The 08-04 directive above stands for what it
  was aimed at; it was being applied to work that has no design in it, which wastes a wait and
  teaches nobody anything.
  **THE TEST, and it is one question: is there a design choice with more than one defensible
  answer?**
  * **YES -- new capability, architecture, a schema or ledger change, anything touching the
    canonical workflow, anything where reasonable people would disagree** -> FULL four-round
    arc BEFORE code, exactly as above. This is where the panel earns its keep: on the
    2026-08-17 style build the panel killed a build-breaker (a required pack key would have
    bricked every frozen embedded pack via its sha256 receipt), deleted the riskiest mechanism
    outright, and corrected the driver's own execution-order claim.
  * **NO -- one verifiable right answer: wiring conformance, a grep-and-fix, a stale comment,
    a rename, an `/object_info` check, a deterministic edit** -> NO arc. An arc pressure-tests
    DESIGN; there is nothing to pressure-test. Sonnet 5 QA on the finished diff before the
    push is the correct and sufficient gate.
  * **ONE CLEAN FINISHED-DIFF REVIEW IS ENOUGH (operator directive 2026-08-20).** Run one
    independent reviewer after code. If it returns clean and the driver grounds that result
    against the real files, STOP reviewing and move to tests/live proof. Add another reviewer
    only when the first reports a blocker, two reviewers disagree, a material claim remains
    unverifiable, or the two-strikes rule below fires. An available internal subagent is a
    valid reviewer; do not wait on or multiply CLI/cloud lanes merely to increase the count.
    Save the real roster and result in the handoff receipt.
  * **UNSURE** -> treat it as YES. The arc is $0 and a missed design flaw is not.
- **`otr/obs/` IS THE SUCCESS SIGNAL. ALWAYS PUBLISH TO IT (operator directive 2026-08-17 --
  hard, and it OVERRIDES the tidiness instinct).** Operator, in his words: *"always publish to
  obs -- a test is not complete unless published to obs (or it's just testing one part). If I
  see it in obs then it's somewhat a success."* And the failure half: *"if I don't see it in
  obs and it took more than 5 minutes, it's a fail."*
  * **A leg that does not reach `otr/obs/` did not pass**, however green its logs are. The
    published artifact IS the receipt -- that is how HE reads success without reading a log.
  * **NEVER move, hide, sort or clean harness runs out of `otr/obs/`.** Soak, banksweep,
    probe, acceptance and bank-gate legs BELONG there; seeing them is the point. **This rule
    exists because I moved 17 of them into a `_diagnostics/` subfolder the same day, reading
    them as pollution. They were his proof that the full path worked. Restored within
    minutes; nothing was deleted.** The narrow, real complaint was only that the harness RUN
    LABEL becomes the on-screen TITLE CARD -- cosmetic, and not a reason to touch the folder.
  * **The `_bench_4arm` carve-out is NOT a precedent for this.** That exemption is scoped to
    isolated stock-node BENCH graphs that never run the canonical workflow. A soak or bank-gate
    leg DOES run the real canonical path end to end, which is exactly why its publication
    proves something.
  * **The 5-minute rule is a real gate:** if a leg has run longer than 5 minutes with nothing
    in `otr/obs/`, treat it as failing and go read the leg log rather than waiting it out.
  * **THE PRIORITY ORDER, in his words:** *"long term yeah I want the title right, but for my
    dailies it keeps me going to see episodes."* So the title-card label IS a real want and it
    is a LONG-TERM one; the daily stream of published episodes is the thing that must never be
    interrupted to get it. **Fix the title at the source (stop reusing `episode_title` as the
    harness's scratch field) -- never by suppressing, relocating or gating the publish.** If a
    proposed fix would reduce how many episodes he sees in `otr/obs/` tomorrow, it is the wrong
    fix no matter how tidy it is.
- **A MISSING REVIEWER NEVER BLOCKS THE ARC -- SUBSTITUTE AND KEEP GOING (operator directive
  2026-08-17 -- hard).** Operator: *"if out of budget then a kibitz reviewer is not needed,
  ask Fable, Sonnet, or Opus, or `/anthropic-skills:roundtable` -- ANY model in lieu."*
  **A quota-held, timed-out, or unavailable lane is NOT a reason to stop, to defer the item,
  or to park a finished diff waiting for a lane to come back.** Fill the seat and run the
  round.
  * **Eligible substitutes, freely interchangeable:** Fable, Sonnet, Opus (as Cowork
    subagents), the other `agy` model lanes (Gemini 3.1 Pro / 3.7 Flash are different
    reviewers, not one), and `/anthropic-skills:roundtable` for a cloud panel. Any model in
    lieu of any other.
  * **WHAT STILL DOES NOT CHANGE:** you write the code-grounded `driver_anchor.md` FIRST and
    you remain the sole judge; every panel claim is verified against the real Windows files
    before it is folded in; and **the PROVENANCE is stated exactly** -- name which lanes
    actually reviewed which rounds. A campaign a reviewer short is described as a campaign a
    reviewer short, never as a full arc. Substituting is honest; misreporting the roster is
    not.
  * **This was written because it cost real time on 2026-08-17 (item F).** Codex was
    quota-held to 08-19 and the driver treated that as a hard gate on the whole item --
    "F is not secretly cheap ... it wants the whole arc once Codex is back". The operator's
    correction: run it with who is available. Item F then ran r1-r4 on two agy lanes plus a
    Fable gate and a Sonnet QA pass, caught three driver errors (two of them build-breakers:
    an `UnboundLocalError` on four of six banks, and a lane collision that would have
    announced an invented play on 57% of `media_archive`), and shipped proven on a live leg
    -- **with Codex never participating in a single round.** The seat mattered; which model
    sat in it did not.
  * **A TIMEOUT IS NOT A QUOTA BLOCK.** `agy` failing with `Error: timeout waiting for
    response` means the print timeout, not exhaustion -- `kibitz.py --timeout` does NOT reach
    agy; the knob is **`KIBITZ_AGY_PRINT_TIMEOUT`** (default `5m`; `15m` fixed it first try).
    Check `agy models` returns rc=0 before calling a lane dead.
  **THE TWO-STRIKES FLOOR BELOW IS UNTOUCHED AND NEVER LAPSES.** A bug that survived two of
  your fixes gets a panel on the third attempt no matter how mechanical it looks -- two failed
  fixes IS the evidence that your model of the problem is wrong, which is a design problem
  wearing mechanical clothes.
  **Fable routing (section 9) is unchanged and is a separate axis:** never spend Fable on
  mechanical work. Reserve it for narrative judgment and the final gate on a high-stakes,
  hard-to-unwind structural change. Do not burn a scalpel on a screw.
- **TWO STRIKES, THEN THE PANEL (operator directive 2026-07-14 -- hard).** You get **two**
  attempts to fix a given problem on your own. If the **third** attempt is about to begin --
  i.e. the same bug/failure survived two of your fixes -- you MUST `/kibitz` (local, $0,
  file-grounded) BEFORE writing more code. No third solo swing, ever. Two failed fixes means
  the model of the problem is wrong, and a third guess from the same wrong model just burns a
  live roll. The panel's job is to break your framing, not to bless your patch. You remain the
  judge: ground every panel claim against the real files and discard what does not survive.
  (This is the FLOOR and it never lapses -- a third try ALWAYS gets a panel. The 2026-08-04
  directive above is stricter while it stands: coding work gets a full panel on the first
  swing too. The older "kibitz on EVERY live failure" wording is what this replaced.)
- **RIPPING AN LLM IS ALLOWED. A HOLE IN THE LEDGER IS NOT (operator directive 2026-07-14 --
  hard, non-negotiable).** Removing or repurposing an LLM pass is a legitimate and sometimes
  necessary hard decision. But the LEDGER MUST STILL BE FILLED COMPLETELY for downstream
  consumers -- TTS, per-beat audio slicing, video/shot direction, captions, credits,
  `obs_publish`. They read FIELDS, not intentions. Before you remove a pass: (1) enumerate
  EVERY ledger field it wrote; (2) give each field a new owner -- deterministic Python,
  another pass, or an explicit default -- exactly one owner each; (3) only then delete the
  call; (4) prove it on a LIVE leg (RESULT SUCCESS + obs_publish OK + the asset on disk).
  A green unit suite does NOT prove the ledger is complete. A ripped pass with an unowned
  field is a broken render, not a simplification.
- **When genuinely torn between approaches, run the roundtable LIVE for convergence
  (ChatGPT + Gemini + DeepSeek).** Skip the dry-run / cost estimate -- just run it, pronto.
  You are the judge. Escalate to me only if the panel still leaves it unresolved.
- `/kibitz` and `/roundtable` are also welcome for IDEAS at any time -- pressure-testing a
  plan, hunting a defect class, hardening a doc -- not only as the two-strikes backstop.
- This file is hard rules + the real Cowork operating model for this repo + hard-won gotchas.
## 0. WORKFLOW SOURCE OF TRUTH (hard)
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_canonical.json`
IS my workflow.
- ANY json / node / wiring / widget change MUST be made IN that file, in the SAME change as the code.
  Code that is not wired into this JSON is DEAD -- "your updates are for naught" (the 2026-06-13 §4D
  miss: a node + a new blend input shipped + tested but UNWIRED -> ran dormant in production).
- EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot.
- Schema: litegraph. Top level = `nodes[]` + `links[]` + `last_node_id` + `last_link_id`; a link is
  `[link_id, src_node, src_slot, dst_node, dst_slot, type]`; one output fans out via its `links` list.
  `widgets_values` is POSITIONAL -- only ever APPEND a new optional widget at the END (inserting mid-list
  shifts every saved value -> silent drift, BUG-LOCAL-097). A widget converted to an input keeps its
  value slot AND gains an input with `"widget": {"name": ...}`.
- **REMOVING A WIDGET TOUCHES THREE THINGS, NOT ONE (learned the hard way 2026-08-28).** An inert widget
  SHOULD be removed -- operator: *"that's being lazy not to remove an inert widget"* -- and the positional
  problem is WORK, not a veto. But the work is bigger than it looks, and I corrupted all 63 workflows by
  doing only the first two:
  1. **`widgets_values`** -- drop the value at the widget's index in every saved graph.
  2. **The `inputs` DESCRIPTOR array** -- drop the `{"widget": {"name": ...}}` entry.
  3. **EVERY LINK TARGETING A LATER SLOT.** `dst_slot` in `[link_id, src_node, src_slot, dst_node,
     dst_slot, type]` is an INDEX INTO THAT SAME `inputs` ARRAY, which holds link sockets and widget
     descriptors together. Remove a descriptor and every link past it is off by one. This is invisible to
     a widget-count check and to `build_variants --check`; `tests/test_workflow_link_target_indexes.py`
     is what catches it. **Repair by IDENTITY, not arithmetic:** set each `dst_slot` to the index whose
     `inputs[i].link` equals that link's id -- self-correcting and impossible to double-apply.
  **A TRAILING widget is nearly free; a mid-list one costs the re-index everywhere.** Check which you have
  before estimating.
  **VARIANTS ARE GENERATED, NEVER HAND-EDITED.** Fix `otr_canonical.json`, then run
  `python scripts/build_variants.py --all` and confirm with `--check`. Editing the variants directly
  passes a naive diff and then fails regeneration.
  **VERIFY WITH ALL FOUR:** `build_variants.py --check`, `tests/test_widget_value_alignment.py` (one node
  type declares one widget ORDER everywhere -- catches a migration that updated canonical and missed a
  variant), `tests/test_canonical_widget_input_parity.py` (count parity), and
  `tests/test_workflow_link_target_indexes.py` (link integrity). The first three all passed while the
  links were broken.
- After editing it, re-validate: `OTR_WorkflowValidator` + a JSON round-trip + a link/widget audit
  (widget-count vs live INPUT_TYPES, every wired input-name in INPUT_TYPES, link referential integrity).
### 0A. THE ISOLATED-BENCH CARVE-OUT IS STRUCK -- THERE IS NO EXEMPTION (operator, 2026-08-23)
Operator: *"I think I am done with all bakeoffs."* Every bake-off runner, builder,
pinned graph and the vendored bench helper were retired that day, so the ONE exemption
to the rule above no longer has any machinery to license.

**THE RULE IN SECTION 0 IS NOW ABSOLUTE: every API / headless / soak run loads
`workflows/otr_canonical.json`. There is no second path.** Do not reconstruct a bench
harness to get around it -- if a measurement genuinely needs a stock-node graph, that is
a new operator decision, not a revival of this paragraph.

What the carve-out used to permit, kept only so a reader of the git history understands
what was removed: `run_video_arm_bakeoff.py` and `run_wan_ti2v_bakeoff.py` submitted
SHA-256-pinned API graphs from `scripts/bench_graphs/`, wrote only to
`otr/episodes/_bench_4arm/<arm>/`, and were forbidden from qualifying anything -- a bench
result could never be worded as qualification. That last clause is the part worth
remembering even now: a measurement is not a proof, and only the canonical path ships.

### 0B. START EVERY SESSION BY PULLING, AND NEVER FIX ONE BOX BY BREAKING THE OTHER
**Operator directive 2026-08-29. Two rules, one cause: there are now TWO boxes writing this
repo (section 1's split-by-area), so YOUR CHECKOUT IS NOT AUTHORITATIVE AND YOUR HARDWARE IS
NOT THE ONLY HARDWARE.**

**FIRST ACTION OF EVERY SESSION, BEFORE READING OR EDITING ANYTHING:**
```
git fetch origin v2.0-alpha
git log --oneline HEAD..origin/v2.0-alpha     # what the other box did while you were away
git pull --rebase origin v2.0-alpha
```
Then SAY what came down. A session that opens `workflows/otr_canonical.json` without doing this
is reading a file that may be hours stale, and the moment it edits and pushes, **the other box's
work is reverted by a change that looks like a clean commit.** That is the single most expensive
way to lose work here and it leaves no conflict and no error -- just a quiet revert with a
plausible message on top of it. The `merge=union` guard protects the append-only LOGS; it does
NOT protect the workflow JSON, `nodes/`, or the profiles, and nothing can.

**THE SECOND RULE, AND IT IS THE ONE WITH TEETH: A FIX FOR ONE MACHINE MUST PROVE THE OTHER
MACHINE IS UNCHANGED -- MEASURED, NOT ASSERTED.**
The 4060 exists to find what the 5080 cannot see about itself. That is its whole value, and it
is also the hazard: a change that makes an 8 GB card work can silently degrade the 16 GB card
that renders the actual episodes, and the 5080 window will not notice because its tests still
pass. "Passes the suite" is not the same claim as "the 5080's numbers did not move."
* **Confine it if you can.** Per-machine choices belong in PROFILES and VARIANTS, not in
  `otr_canonical.json` and not in shared code. A change that lives entirely inside a 4060
  profile CANNOT reach the 5080, and that is the preferred shape of every portability fix.
* **When shared code genuinely must change, run the OTHER box's path and show the number
  before and after.** The worked example is PBUG-20260829-07 (2026-08-29): the 12B writer was
  being handed the 2B VRAM budget on 8 GB cards. The fix touched `_plan_max_memory`, which every
  machine calls. Before pushing, the 5080's own path was executed and printed both sides --
  `gemma-4-12b-it @ 15.99 GB -> {0: '13.5GiB'}` before AND after, byte-identical -- because the
  `>= 12 GiB` branch returns before any size tag is read. That single printed line is what made
  the fix safe to push, and it took under a minute.
* **State the blast radius in the commit message.** Name which machines' behaviour changes and
  which is provably untouched. A commit that silently alters both boxes while claiming to fix one
  is the defect this rule exists to prevent.
* **Unsure whether it reaches the other box? It reaches the other box.** Treat it as shared and
  measure.

## 1. HOW COWORK ACTUALLY WORKS HERE (read this first)
- **Two separate filesystems.** The file tools (Read / Write / Edit) operate on the REAL Windows files --
  that is your primary editor. **Desktop Commander** (`mcp__Desktop_Commander__*`) runs PowerShell on the
  same real Windows box -- use it for git, the venv python, tests, and process control. The
  `mcp__workspace__bash` Linux sandbox is a DIFFERENT machine: its mount LAGS the file-tool writes (shows
  stale/truncated copies -> phantom "corruption") and has NO torch. Use bash only for quick greps of
  UNCHANGED files; never trust it for current state and never run the suite there.
- **The loop:** edit with the file tools -> verify/test with the Windows venv via Desktop Commander ->
  commit AND push via DC git -> verify HEAD == origin.
- **Test runner:** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (torch 2.10). Run with
  `$env:PYTHONUTF8=1`; `pytest -q -p no:cacheprovider`. The conftest sets `OTR_TEST_MODE`/`CUDA_VISIBLE_DEVICES=''`.
  Bug Bible lives in a SEPARATE repo: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`
  -- `cd` to its root and use the RELATIVE path `tests\bug_bible_regression.py` (an absolute forward-slash
  path fails to collect).
- **Knowledge gate before implementation or diagnosis:** read `docs\PRODUCTION_SPRINT_LESSONS.md`, then the
  relevant entries in `docs\PROD_BUG_LOG.md`, and the matching portable rules in
  `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\BUG_BIBLE.yaml`. The lessons are
  mandatory project context, the production log is the staging record, and the Bible is the reusable
  cross-project contract. A newly fixed, repeatable production failure must be recorded in the log and
  promoted with a Bible entry plus executable coverage whenever its verify condition is automatable.
- **Admission rule:** only a bug verified by a live production artifact, headless run, smoke, soak, or
  published episode may enter `PROD_BUG_LOG.md` or be promoted to the Bug Bible. A review observation,
  static-audit finding, or invented test fixture may verify a known production bug, but never creates a
  new PBUG or Bible rule on its own.
- **Bible delta-scrape discipline (2026-08-07):** the Bible repo carries `otr_coverage_index.yaml` --
  all 369 OTR bug records through 2026-08-07 (BUG_LOG.md, BUG_LOG_2026-06.md, docs/handoffs, kibitz
  runs, loose logs, git fix history) mapped to Bible ids against the 261-entry Bible (HEAD 3759ae5).
  NEVER re-scrape indexed history -- the full scrape cost ~4M tokens once and the index exists so it
  is never paid again. At session wrap-up / handoff, if the session recorded a NEW bug (admission rule
  above): check it against the index + Bible, promote a genuinely uncovered one with a Bible entry,
  and append its row to the index in the same change. Any cached Bible copy, entry count, or vendored
  test snapshot re-syncs to the Bible repo's origin/main -- do not pin a stale local copy of the
  Bible or its tests.
- **PowerShell reality (DC runs powershell.exe):** use `;` to chain, NOT `&&`. Do NOT use
  `python -c "..."` with nested quotes -- PowerShell mangles them; instead WRITE A TEMP `.py` file, run
  it, then delete it. `2>&1` makes stderr render as scary red text -- that is NOT a failure; check the
  exit code / output. Pipe noisy output through `Select-Object -Last N`.
- **QUOTING / `$`-INTERPOLATION RULE (hard -- this bites EVERY session that ignores it):** any command
  that would need NESTED quotes, backtick-escaped quotes (`` `" ``), a `$` variable inside a quoted
  argument, a here-string fed to another program, or `cmd.exe /c "<...>"` -- DO NOT attempt it inline
  and DO NOT iterate on escaping ("one more tweak" never works; PowerShell mangles `$var` and eats
  quote layers). IMMEDIATELY write a temp LAUNCHER SCRIPT instead -- a `.ps1` (or `.py`/`.cmd`) written
  via the FILE TOOLS (Write, not echo/Set-Content with its own quoting problem), with all variables and
  quoting inside the script where they are literal and safe -- run it via
  `powershell -ExecutionPolicy Bypass -File <tmp.ps1>` (or `Start-Process -FilePath`), then delete it.
  First quoting error = STOP escaping, switch to the script. Zero exceptions.
- **The ~60s MCP ceiling:** any single DC command that blocks longer (a `Start-Sleep` > ~45s, a big render
  loop, the full ~4200-test suite, a slow boot wait) TIMES OUT and orphans the process. Background it to a
  log and poll the log, or shrink the job (a test subset, fewer frames). DC itself is NOT flaky -- the
  command was too long.
- **Subagents (Agent tool)** are excellent for read-only fan-out audits, but TELL them to use Desktop
  Commander + the Windows venv/path -- left to default they read the lagging Linux mount and report
  phantom truncation/corruption (happened 2026-06-13; the third agent that used the Windows path was right).
- **Stale git `index.lock`:** if `git add`/`commit` fails with "index.lock: File exists" AND
  `Get-Process git` is empty, the lock is STALE (a real git op finishes in seconds) -- remove
  `.git\index.lock` and retry. Do NOT remove it while a git process is actually running.
- **TWO WINDOWS, SPLIT BY AREA, BOTH PUSH (operator decision 2026-08-29 -- supersedes the older
  "one coder window at a time" line).** The 5080 and the 4060 both run coder windows and both push
  to `v2.0-alpha` directly. Serialization by turn-taking was the old answer and it cost more than it
  bought: on 2026-08-29 the two boxes landed ~25 commits with zero lost work, and the 4060 found
  three defects in the 5080's own files (a fetcher pulling the wrong motion module, an h3 profile
  declaring a canvas its engine overruled, a re-run that could not have tested what it claimed)
  while the 5080 fixed a loader bug that was killing the 4060's legs. Neither would have happened
  quickly behind a single-writer queue.
  **THE SPLIT -- every file has exactly one owner, and the owner pushes it:**
  * **The 4060 (MRKT) owns the portability surface:** any profile IT has proven on 8 GB hardware,
    `docs/4060_DRILL_LOG.md`, and the fresh-install / least-friction path. It is the only box that
    can answer "does this work somewhere other than where it was written", which is the one question
    the dev box structurally cannot answer about itself.
  * **The 5080 (IDREAM) owns the shipping surface:** `pyproject.toml` and anything registry-facing,
    `workflows/otr_canonical.json` and its variants, `nodes/`, and profile `status` promotions.
  * **`docs/PROD_BUG_LOG.md` is shared and append-only** -- both boxes write it, by appending.
  **WHAT MAKES THIS SAFE IS STRUCTURAL, NOT ETIQUETTE.** `.gitattributes` marks the append-only logs
  `merge=union`, so a tail collision keeps BOTH sides automatically instead of raising conflict
  markers that a tired window resolves by picking one. That resolution was performed BY HAND twice
  on 2026-08-29 before the guard existed. Union can duplicate; it cannot lose. Never extend
  `merge=union` to `.json` or `.py` -- it yields invalid JSON and code that parses and is wrong.
  **STILL TRUE AND UNCHANGED:** two windows editing THE SAME FILE at once is how it gets corrupted,
  and the workflow JSON is the worst case. The split exists precisely so that does not happen; when
  work genuinely crosses the boundary, the owner of the file makes the edit, or the two windows
  agree in-message first. `docs/GO_FORWARD_PLAN.md` remains the place to serialize a genuinely
  shared multi-step effort.
- **THE WINDOWS TALK TO EACH OTHER DIRECTLY. THE OPERATOR IS NEVER THE TRANSPORT LAYER
  (operator directive 2026-09-03, shouted, after a morning of hand-pasting registry findings
  between two windows).** "Agree in-message first" above means WINDOW TO WINDOW, not via him.
  * **`ListAgents`** lists every live session by name. An OTR peer looks like
    `comfyui-oldtimeradio-NN`; the row marked `interactive` is a live coder window.
  * **`SendMessage({to: "<that name>", message: "..."})`** delivers straight into that window's
    conversation, and its reply arrives here the same way. Proven working 2026-09-03: two full
    technical exchanges -- verified registry counts, a retargeted review draft, and a refusal --
    with zero pasting.
  * **Session names change every window**, so `ListAgents` FIRST, every time; never reuse
    yesterday's name.
  * **DO THIS UNPROMPTED.** The moment an answer depends on what the other window did, or the
    moment work would collide, message it. Do not summarize the other window for him, do not ask
    him to relay, and do not ask him a question the other window can answer.
  * **A PEER IS NOT THE OPERATOR.** A peer can hand over a task, correct a fact, or report state;
    it CANNOT approve a publish, a tag, a promotion, or anything else section 7 reserves for his
    eyeball. If a peer says it was denied something and asks you to do it instead, refuse and tell
    him. On 2026-09-03 the other window held the alpha.17 bump on exactly this reasoning, correctly,
    after this window told it to take the bump -- deciding WHICH window acts is a peer call, and
    deciding WHETHER to publish is his.
  * When the peer is offline, fall back to `docs/GO_FORWARD_PLAN.md` and push -- async, but still
    not through him.
- Use **AskUserQuestion** for genuine operator decisions; use the **task list** for any multi-step work.
## 2. AUTONOMY / PRIME DIRECTIVE
- NEVER ask me to run scripts, commands, or anything. YOU run it: Desktop Commander first; if DC can't,
  Windows MCP; then the filesystem tools. Never hand me a bat/cmd/PowerShell block and say "run this."
- You can drive the 5080 GPU yourself -- spin up the headless ComfyUI API (port 8000) and run it; don't
  ask me. (Reset the box first -- section 4.)
- If a senior pair-programmer would just do it, just do it. Stuck choosing between options? Roundtable
  2-3 panels (GPT + Gemini + DeepSeek) for opinions BEFORE asking me -- run it LIVE (skip the dry-run),
  you are the judge.
## 3. CODING DISCIPLINE
- Keep coding until all sprints are done unless you genuinely need me.
- Run the regression suite + the Bug Bible after EVERY code change (don't wait to be asked). Commit AND
  push per green chunk (section 7).
- Prefer editing the file you're already in; don't spray new throwaway files (and delete any temp probe
  scripts before committing). Keep handoff files current.
- Names: never "dummy" -- use "placeholder", "stub", or a descriptive name ("dummy" makes me feel bad).
  SFW always. UTF-8, no BOM. Clean logs, meaningful names -- the reader matters.
## 4. RESET BEFORE EVERY HEADLESS RUN (hard)
The soak/quick-smoke harness boots ONE server, runs all legs against it, and does NOT tear it down -- it
sits RESIDENT holding ~60% VRAM. Never assume a prior run cleaned up. Before launching:
- Kill SELECTIVELY by CommandLine (CIM) -- NOT a blanket `Stop-Process -Name python,pythonw`. A blanket
  python kill ALSO kills the Claude MCP extension pythons (Desktop Commander / computer-use) and severs
  your own tools mid-run. Use `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and kill the
  ones whose CommandLine matches the ComfyUI server + the soak/sweep harness; plus the port kill via
  `Get-NetTCPConnection -LocalPort 8000`. (`.Path` is BLANK for half-booted servers, so filter on
  CommandLine, not Path.)
- Confirm `Get-NetTCPConnection -LocalPort 8000 -State Listen` is EMPTY and
  `nvidia-smi --query-gpu=memory.used --format=csv,noheader` dropped to the desktop baseline (~1.5 GB)
  before booting fresh.
## 5. HEADLESS BOOT + MONITORING GOTCHAS (2026-06-12 -- do not relose)
- **A render that FINISHED leaves the server RESIDENT (~9-10 GB, 1% util) -- that is NOT a crash.** Before
  declaring a run dead, read the server log: `Prompt executed in HH:MM:SS` + `obs_publish OK` = it
  COMPLETED. The idle resident VRAM is the no-teardown behavior. (Misread twice 2026-06-12.)
- **Use the watchdog for long renders** (`scripts/otr_render_watchdog.ps1 -LegLog <leg.log>`): declares the
  run DEAD on a 5-min heartbeat stall OR a down :8000/queue endpoint (exit 2), exits 0 with the verdict
  when the leg finishes. It REPORTS only; reset per section 4.
- **Headless boot needs UTF-8.** A detached cmd inherits the Windows cp1252 codec, so OTR
  `prestartup_script.py` crashes the instant it prints an emoji (UnicodeEncodeError on U+2705/U+2713) ->
  boot dies ~13s, exit 1 ("SERVER DID NOT COME UP"). The launcher (`scripts/_otr_soak_server_launch.cmd`)
  sets `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`; any new boot path MUST too.
- **Boot is ~20s, NOT 7-8 min.** If a boot "hangs", read the log -- it has already died.
- **Launch the server via the .cmd as `-FilePath`, never `cmd.exe /c "<cmd>" "<log>"`.** The `/c`
  two-quoted-token rule eats the outer quotes -> mangled path -> ZERO log output. Use
  `Start-Process -FilePath $LAUNCHCMD -ArgumentList "`"$LOG`""`.
## 6. OUTPUT / ASSET PATHS (hard)
Rendered episode assets are deliverables -- they do NOT live in tmp/scratch, and they are NEVER left in a
swept dir to be moved later.
- Every rendered asset (audio, frames, intermediate video) -> `otr\episodes\<ep>\`. The final published
  episode -> `otr\obs\` (what `obs_publish` targets; `obs_publish OK` in the log = it landed there).
- Point the render at its canonical path the FIRST time. Never stage an asset in tmp "to move later" --
  the move-later step is where work dies.
- A temp probe/script in tmp is fine (throwaway, per section 3) -- but the ASSET it produces is written
  STRAIGHT to its canonical path, never parked in tmp.
- Output destination is workflow config: if a node writes to the wrong place, fix the path IN the
  workflow JSON (section 0), in the SAME change as any code -- not with a post-hoc move.
- After any render leg, confirm the asset exists at its canonical path (`Test-Path otr\episodes\<ep>\<file>`)
  before continuing or declaring success. The file check -- NOT just VRAM -- is what distinguishes the
  "finished but resident" server (section 5) from a real miss. Missing = STOP and report; do not continue.
- Paths are relative to the repo root (`...\ComfyUI-OldTimeRadio\`). If `otr\` actually resolves to
  ComfyUI's real `output\` base on disk, use that base -- but the episodes/obs split holds either way.
### 6A. THE MODELS ROOT IS `C:\ComfyUI-Models` -- NOT a folder under the repo or the ComfyUI tree
**Written 2026-08-22 because this window searched the wrong tree and reported a model MISSING
that was on disk all along.** The weights do NOT live under
`C:\Users\jeffr\Documents\ComfyUI\models\` (that tree exists and holds *some* things, which is
exactly what makes the mistake convincing) and they do NOT live under the ComfyUI-Installs tree.
- **The authority is `nodes/_otr_gguf_backend.py::_models_root()`** -- read it rather than
  guessing: `OTR_COMFYUI_MODELS_ROOT` -> `COMFYUI_MODELS_ROOT` -> default **`C:\ComfyUI-Models`**.
- **GGUF writers resolve to `<models_root>\LLM\converted\<subdir>\<file>`**, e.g.
  `C:\ComfyUI-Models\LLM\converted\gemma-4-12b-it\gemma-4-12b-it-Q4_K_M.gguf` (7.12 GB, present).
  `GEMMA4_12B_GGUF_PATH` is a whole-path escape hatch for the GEMMA row ONLY.
- **A `find` under the repo or under `Documents\ComfyUI` proves NOTHING about model presence.**
  Before telling the operator a model is missing -- or proposing a multi-GB download -- resolve
  the path through the code above and `Test-Path` THAT. A false "missing" costs him a download
  he does not need and costs you the profile leg you were about to skip.
## 7. GIT POLICY (operator directive 2026-06-10 -- never lose work)
- ONE branch: `v2.0-alpha`. COMMIT AND PUSH TOGETHER: every green commit gets pushed to origin
  immediately, same session, no exceptions. Local-only commits are the failure mode we guard against.
- **`v2.0-alpha` IS THE GITHUB DEFAULT BRANCH as of 2026-08-22.** It was switched from `main`
  because `main` sits 3,923 commits behind and still advertises `version = "1.0.0"` -- so every
  bare repo link, every fresh `git clone`, and every ComfyUI-Manager NIGHTLY install was serving
  stale v1 code while looking like it had worked. A fresh clone now lands on `v2.0-alpha` with the
  real v2 tree (verified by actual clone, not assumption). Do NOT "helpfully" switch it back, and
  do not treat `main` as current for ANY purpose -- it is a stale v1.7 release merge, nothing more.
- The operator eyeball gates TAGS and PROMOTIONS (`v2.0-alpha-stable`, prod, main, v2 release) -- NEVER
  pushes. Pushing to `v2.0-alpha` is always safe, expected, and required.
- This SUPERSEDES any "do not push until the eyeball passes" line written before 2026-06-10 evening.
- A stable branch only exists if the operator explicitly declares one.
- After every push verify: HEAD == origin, no 0-byte files, no BOM, AST parse on touched .py files.
### 7A. COMFY REGISTRY PUBLISHING (live since 2026-08-22 -- read before touching pyproject.toml)
The pack is published to registry.comfy.org as **`comfyui-old-time-radio`** under publisher
**`fluxus`** (the operator's account). `.github/workflows/publish_action.yml` publishes via
`Comfy-Org/publish-node-action`, keyed on the repo secret `REGISTRY_ACCESS_TOKEN`.
- **EDITING `pyproject.toml` AUTO-FIRES A PUBLISH.** The workflow triggers on any push to
  `v2.0-alpha` whose diff touches `pyproject.toml`. Treat that file as a release trigger, not a
  config file: never edit it "just to tidy" mid-session, and never edit it while a version is
  already pending. A push that does NOT touch it never publishes.
- **Every publish needs a NEW version string.** `(node_id, version)` is uniquely indexed
  server-side; re-publishing the same version is refused.
- **`.comfyignore` decides what SHIPS** (gitignore syntax, layered on top of git tracking --
  untracked files are excluded already). It currently strips `tests/`, `kibitz-runs/`,
  `.github/`, `.claude/`, plus the exec()-using probe/smoke scripts that comfy-cli's security
  scanner flags. Verify by downloading the published zip, never by assuming.
- **PENDING IS A QUEUE, NOT A REJECTION.** A new version lands `NodeVersionStatusPending` and is
  promoted to `Active` only by Comfy-Org's own Cloud Scheduler cron hitting their `/security-scan`
  endpoint, which ONLY considers versions older than **30 minutes** (`registry.go:938`). Clean
  scan -> Active; issues found -> `Flagged` (their private Discord, not us); missing zip ->
  `Deleted`. The scanner itself is a PRIVATE repo and its schedule is not in any public config --
  possibly nightly. **While a version is Pending, `latest_version` resolves to null, which is
  exactly why ComfyUI Manager reports "not a CNR node" / "Cannot resolve install target".**
  That error is NOT a local install fault -- do not send anyone chasing torch/dependency ghosts.
- **There is NO publisher self-service path to Active.** Confirmed by reading
  `Comfy-Org/registry-backend`. Waiting, or asking Comfy-Org, are the only moves.
- **DELETE ASYMMETRY, and it is a trap:** deleting the NODE is a HARD delete (row removed,
  versions cascade, every version string freed for reuse). Deleting a VERSION is a SOFT delete
  (status flipped to `deleted`, ROW REMAINS) -- which BURNS that version string permanently.
  Prefer node-delete for a clean slate; never version-delete a string you want back.
- **The registry DELETE API needs the operator's browser (Firebase) session -- the publish token
  returns 401 "user not found".** Claude cannot delete a listing; the operator clicks it. Deletion
  is also EVENTUALLY consistent: the version list can serve stale reads from different replicas
  for minutes afterward. Read it 2-3 times before concluding anything.
- **`[project] dependencies` MUST BE A STATIC LIST, and it must be kept in sync with
  requirements.txt BY HAND.** The registry reads that literal field; it does NOT evaluate
  setuptools' `dynamic = ["dependencies"]` + `[tool.setuptools.dynamic]`. Proven the hard way on
  2026-08-22: alpha.3 shipped with the dynamic form and the registry still recorded
  `dependencies: []`; alpha.4 with a static list recorded all 12. A pack published with `[]`
  installs its code with NONE of its libraries. Verify after every publish:
  `curl https://api.comfy.org/nodes/comfyui-old-time-radio/versions` and check the count.
- **"No nodes found" on a registry node PAGE is NOT a signal that the pack is broken.**
  Established 2026-08-22 by comparing against working packs: ComfyUI-DramaBox -- Active, correct
  static deps, a real shipped pack -- displays no nodes either, and `/nodes/<id>/comfy-nodes`
  returns 404 for EVERY pack sampled (propost, kjnodes, rgthree, dramabox, cache-cleaner). That
  panel is fed by a separate extraction service that evidently does not populate for most packs.
  **The only trustworthy test is the LOCAL one: install, restart ComfyUI, and look for the OTR
  nodes in the node menu / the ComfyUI console for `[OldTimeRadio]` lines.** Do not diagnose from
  the registry page, and do not send the operator chasing a phantom because of it.
- **`__init__.py` loads each node in its OWN try/except, and that is deliberate** (partial-install
  resilience). Consequence for debugging: a missing dependency SKIPS the affected node and prints
  `[OldTimeRadio] Skipped '<name>': <reason>` -- it does NOT zero out the pack. Proven by loading
  the real published zip with every requirements.txt dep blocked: 32/34 nodes still registered.
  So **a TOTAL zero-node outcome is NOT explained by missing dependencies** -- if you ever see a
  true zero, look for the pack not being loaded at all (wrong dir, prestartup death, ComfyUI never
  scanning it), not for a missing library.
## 8. ROUNDTABLE DEFAULTS (operator directive 2026-06-22)
Standing shape for EVERY `/roundtable` in this repo. These OVERRIDE the skill's stock
"Claude is judge-only / panel only critiques" and "dry-run estimate first" defaults.
- **Panel = 2-3 FRONTIER models per round** (GPT + Gemini + DeepSeek/Grok class; `~latest`
  aliases, record the resolved model in the manifest). Lean panel of genuinely different
  families beats many near-duplicates -- diversity is the point, correctness comes from the
  grounding step.
- **Cowork (Claude) is ALWAYS a code-aware grounded PANELIST *and* the sole judge.** Write your
  own grounded review FIRST (every claim checked against the real Windows files via Desktop
  Commander, never the lagging Linux mount), THEN ground the panel's reviews, discard the
  misreads, and synthesize. The panel proposes; Claude disposes -- never outsource synthesis.
- **Four-round campaign, in order:** R1 high-level arc / creative approach -> R2 coding plan ->
  R3 wiring (workflow JSON / nodes / widgets + any re-baseline procedure) -> R4 final
  convergence (confirm no new must-fix). Re-loop a round only if it surfaces new material; stop
  at convergence (don't grind passes to hear "looks good" in more accents).
- **Never dry-run, never pre-compute the cost -- just spend and run it LIVE,** then state the
  actual spend after. (Only backstop = the global >= $20-or-irreversible gate, which roundtable
  passes never reach in practice.)
- **ARC ROUTING (operator directive 2026-07-09): R1 (ideas/high-level) = cloud `/roundtable`
  on the highest frontier models (that's where paid diversity earns its keep); R2-R4 (coding /
  wiring / convergence) = `/kibitz` (local Codex + Antigravity) ALWAYS PREFERRED -- $0, file-
  grounded. R1 via kibitz is also fine when economy matters. OpenRouter is paid, so default mechanical review to the local panel.
- Artifacts under `docs/<YYYY-MM-DD>-<topic>/roundtable/` (pass00..passNN_plan.md +
  passNN_judgment.md), UTF-8 no BOM, ASCII where practical.
## 9. MODEL ROUTING -- when to spend Fable (operator directive 2026-07-03)
Fable is a scalpel, not a default. Spawn it ONLY when output quality depends on narrative
judgment; keep everything mechanical off it. This is about WHICH MODEL a subagent runs --
orthogonal to the filesystem rule in section 1 and the frontier roundtable panel in section 8
(that panel is external cloud models via OpenRouter; this is the Fable model as a Cowork subagent).
- **Use Fable for:** generative creative work where voice/taste matter -- story spine, character
  interiority, dialogue passes, pitch-room ideation, style/tone calls. Divergent fanout when you
  want real variance, not one answer (e.g. 3 subagents each pitching a different take on a scene --
  only fan out on a genuine fork you'll actually select between). Judgment on already-generated
  narrative -- "which brief holds together," "where does this arc sag."
- **Do NOT use Fable for:** mechanical/deterministic work -- repo grep, mapping references, editing
  JSON, validation, wiring checks, git. Route those to general-purpose / Explore, or just do them in
  the main window. Also skip Fable for one-shot factual/structural questions with a single right answer.
- **Keep the spawn count down:** batch context so each Fable call does a meaningful chunk (a whole
  scene, not one line); gate the spawn on a real fork -- if you'd just take the first answer, use one
  call or none.
- **REALITY EXCEPTION -- Fable's grounded FAN-OUT audit DOES catch real code build-breakers the
  mechanical panels miss (proven 2026-07-03).** On the VRAM-tier rip, a Fable end-to-end fan-out found
  a CHUNK-ORDER KeyError (the workflow validator reads a profile key that was being removed mid-rip ->
  every production render would KeyError) that BOTH codex and general-purpose review missed; it also did
  the frame-budget arithmetic (29->33 frames) that broke two unlisted tests, and spotted a grep-invisible
  hyphenated `--vram-ceiling` flag. So Fable is NOT only-narrative. Reserve it as the FINAL GATE on a
  HIGH-STAKES, hard-to-unwind, production-touching STRUCTURAL change (a big rip/refactor about to be
  executed / merged) -- ONE grounded pass, AFTER codex + general-purpose have gone first, when a missed
  thread would break the build and cost a debugging-and-revert loop. This is the "insight" worth the
  spend. It does NOT reopen Fable for ROUTINE review: everyday grep / wiring / validation / JSON still
  defaults to general-purpose/Explore/codex. Fable = the expensive last set of eyes on the make-or-break,
  not the default reviewer.
