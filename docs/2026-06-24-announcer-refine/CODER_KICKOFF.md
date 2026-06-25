# CODER KICKOFF -- announcer redesign + KILL 2 + KILL 4 (paste as message #1 of a fresh CODER window)

Run the otr-handoff skill to resume. (Manual fallback: read docs/GO_FORWARD_PLAN.md
IN FULL + git log/status on v2.0-alpha.) ACTIVE step: BUILD the announcer redesign +
KILL 2 + KILL 4 in this CODER window. origin/v2.0-alpha HEAD 1ffed0e4 == local; no
.py changed this session (docs only), so all code-map line numbers are current.

YOU ARE THE CODER. Build it yourself end to end (Desktop Commander for git + the
Windows venv + tests; file tools for edits). Do NOT hand me commands. Fix at the
root cause. Tell me the current step + a 5-line summary to prove you've got it, then
GO -- you don't need to wait for me unless you're genuinely blocked.

## READ FIRST (the build is fully specified -- edit by ANCHOR, don't reason about location)
1. docs/2026-06-24-announcer-refine/roundtable/pass04_plan.md  -- the FINAL ticket (C1-C4, exact dataclasses/signatures, STEP A-I).
2. docs/2026-06-24-announcer-refine/roundtable/CODE_MAP.md     -- every edit point: file:line + GREP ANCHOR + reuse/new-symbol inventory. Grep the anchor if a line drifted.
3. docs/2026-06-24-announcer-refine/coda-segue/roundtable/pass03_plan.md -- the FINAL coda design (compose_news_coda). THIS SUPERSEDES pass04 STEP F (the fixed lead-in) and DROPS the climax-line decoupling.

## THE BUILD -- 4 commits, all behind the existing `story_scaffold` widget, byte-identical when off, NO workflow-JSON change
- **C1 = KILL 2 (StoryContract).** Hoist `_style_grammar_on` to run() top; build `StoryContract` pre-outline (cast_seed-keyed) via a new `build_story_contract`; inject the style GRAMMAR at the MACRO prompt (this gives `render_style_grammar` a caller -- the literal KILL-2 fix) + `story_engine` at phase/beat (explicit param, the prompts take `macro` not the request); add `OutlineRequest.style_grammar`/`story_engine` (default ""); REPLACE (don't delete) the late `select_style` @ :3224 with `contract.slug` under flag; `meta["story_contract"]` dict. sound_world stays OUT of dialogue line prompts.
- **C2 = announcer OPEN.** Input-starvation: new `SafeOpenBrief`; new `compose_announcer_intro(..., story_scaffold=False, safe_open_brief=None)` (backward-compatible); capture `opening_status_quo` from the first character beat's intent AFTER generate_outline + BEFORE build_sq_data mutates it; sever `script_brief` under flag; `fallback_safe_open` never reads script_brief.
- **C3 = NEWS CODA (from coda-segue/pass03_plan.md).** New `compose_news_coda`: the LLM writes ONLY a dynamic BRIDGE clause (from `outline.premise` + the safe `intro_text`, NEVER the outcome/news facts); append cleaned `news_close_brief` deterministically; sha256(cast_seed) rotating-pool fallback; coda-specific `validate_news_coda_bridge` (length cap + generic-opener blacklist). Call-site EARLY BRANCH: `if _style_grammar_on and nc_brief -> compose_news_coda(...)` else the UNCHANGED `compose_announcer_outro` (build its `_outro_*` locals inside the else). LineResult is frozen -> `dataclasses.replace` for the `news_coda_no_brief` flag. Use `hashlib.sha256`, NOT builtin `hash()`.
- **C4 = KILL 4.** Role-keyed enrichment map (setup/pressure/personal_stake + every CLIMAX_CLASS_ROLES member; consequence omitted, not stubbed) + the truncation reserve/clamp (`max(0, ...)` -- the spec has the exact code).

## PER-CHUNK DISCIPLINE (do every chunk, don't wait to be asked)
- Edit the file you're in; new symbols go where CODE_MAP says.
- Full suite green vs the 5 pre-existing 267a53e workflow-pin fails (verify pre-existing by stash+rerun if unsure -- this build touches ZERO workflow JSON). Bug Bible (cd the survival-guide repo, relative `tests\bug_bible_regression.py`). `$env:PYTHONUTF8=1; pytest -q -p no:cacheprovider`.
- Add run()-level OFF-flag GOLDEN tests (open line, outro line, ledger meta JSON) + keep `test_audio_byte_identical` green (audio spine FROZEN, mux-LAST).
- Commit AND push to v2.0-alpha per green chunk (one push; verify HEAD==origin, no BOM, AST-parse the touched .py). Surgical `git add` of the files you changed -- do NOT `git add -A` (the repo has a large untracked scratch pile; leave it).

## AFTER C1-C4
LIVE re-soak (gemma + mistral) via the `story_scaffold` toggle (on vs off) -- confirm the body no longer collapses to machinery, the OPEN has no spoilers, and the CODA reads as the real fact after the character climax. Then report to me.

## DEFERRED / GATED
KILL 3 (climax POSITION = spine-driven) is DEFERRED -- its own later build; only the (now-unneeded) outro decoupling was folded away. prod/main + tags GATED (don't tag/promote). The `NEWS_CODA_POOL` fallback phrases are mine to tweak; the happy-path coda is the dynamic bridge.

## HARD RULES
100% local for the pipeline; determinism (seed-keyed); LOUD fallbacks (log + ledger restamp, never silent); UTF-8 no BOM; SFW; never "dummy" (use "placeholder"/"stub"); the workflow JSON is the source of truth IF you must touch a widget (you shouldn't here). If genuinely torn between approaches, run the roundtable LIVE before escalating to me.
