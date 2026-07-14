# 720-Word Bakeoff -- Pre-Flight Hardening (problem statement for the panel)

**Date:** 2026-07-14 (overnight run, operator asleep)
**Branch:** `v2.0-alpha`, HEAD == origin == `d07e6a75`
**Suite:** 7898 passed / 31 skipped / 1 xfailed. Bug Bible 17/16/3.

## What is running right now

The 720-word all-banks bakeoff. The **bank is the only variable**; the model pair
is pinned identically across every leg:

| Slot | Model | Why |
|---|---|---|
| creative | `aion-labs/aion-3.0-mini` (OpenRouter, ctx 131,072) | the story writer under test |
| technical | `mistralai/Mistral-Nemo-Instruct-2407` (local, ctx 8,192) | deterministic, free, structured passes |

Ten banks: `science_news`, `media_archive`, `public_domain_story`, `shakespeare`,
`original_radio`, `scifi_fable2`, `scifi_codex`, `scifi_gemini`, `scifi_sonnet`,
`original_codex56sol`.

A 30-word smoke on the exact bakeoff pair is in flight to find cheap failures
before the expensive ones. Status at time of writing: 6 green, 1 root-fixed
failure (below), 3 pending.

## THE LAW (operator, 2026-07-13 -- governs everything below)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE.**
> Only DETERMINISTIC validators may end an episode. An LLM verdict may trigger a
> bounded rewrite; it may never raise.

On 2026-07-13 every LLM veto was supposedly ripped. **One survived**, and it cost
a live roll tonight.

## The failure that just happened (live, prompt `030f73e6`)

`original_radio`, 30 words, Aion creative. Every writer pass succeeded. The
ledger froze clean. Then:

```
original_qa: still dirty after the bounded repair (['epilogue_moralizes'])
```

`epilogue_moralizes` is what the module's own comments call a **subjective**
epilogue class. The flow is: judge -> if the outro "moralizes", run ONE bounded
`compose_announcer_outro` recompose -> re-judge -> **if the judge still dislikes
it, raise and kill the episode.**

So an aesthetic opinion about an outro *the model itself had just rewritten at
that audit's own request* destroyed a complete, otherwise-clean episode.

Note the module had already learned half this lesson: `epilogue_missing` is
refuted **deterministically**, because the re-judge twice "killed a survivable
episode" by claiming the non-empty outro row it had just been shown did not
exist.

**Root fix shipped at `d07e6a75`:** the subjective epilogue classes
(`moralizes`, `contradicts`) no longer raise. The bounded recompose is the
improvement the audit is entitled to; the episode then ships with the recomposed
outro, the objection logged LOUDLY and stamped in meta as
`shipped_over_subjective_objection`.

**Deliberately not changed:** the evidence-gated hard classes
(`weapons_smoking`, `news_source_framing`, `machine_attribution`,
`anachronism_dependency`) still end the episode -- but ONLY when corroborated by
a lexicon hit or a grounded verbatim quote from the script itself
(`triage_hard_findings`). Plus G9, the deterministic SFW ship-stop every lane
crosses at Phase 10. The law frees subjective verdicts, not proven ship-stops.

## The other root fix shipped tonight (`32e680b2`)

A remote model's context window was read from the **static virtual catalog row**,
which hard-coded `DEFAULT_CONTEXT_WINDOW = 8192`. Live logs
(`final2_42_server.log` and friends) show:

```
[OpenRouter] load slot=A slug=aion-labs/aion-3.0-mini ctx=8192 (remote, 0 VRAM)
```

Aion's real window is **131,072**; `tencent/hy3:free` is **262,144**. Every remote
call in every prior run was budgeted against a window 16x-32x too small.

Latent and scheduled to detonate at exactly this event:
`original_codex56sol` P6 budgets `240 + 160*beats + 4*target_words`. At 720 words
the beat ceiling is 40, so it asks for **9,520** output tokens. Against the
fictitious 8,192 window, `fit_output_tokens` would silently reduce that to
whatever was left after the prompt, the performance script would come back cut
off mid-JSON, and the ladder would report a bare `JSONDecodeError` three times --
blaming the frontier model for a constant in our own catalog row. Logged as
PBUG-20260713-20. Live-reverified: the leg now logs `ctx=131072`.

## WHAT I WANT FROM THE PANEL

Ground every claim in the real files. Do not propose a refactor; propose
**specific defects with file:line evidence**, ranked by whether they can kill a
720-word leg.

### Q1. Are there any REMAINING paths where an LLM verdict can end an episode?

The law says there must be none. I found one that survived a rip that was
declared complete. Sweep **all ten lanes** and the shared writer for any place a
model's opinion -- a judge verdict, an audit finding, a `severity`, a
`*_pass: false`, an "exhausted" rewrite ladder, a critic score -- can `raise`,
abort, or otherwise prevent a complete episode from shipping.

For each: file:line, the class of verdict, whether an evidence bar or a
deterministic check stands behind it, and whether it is lawful. Distinguish
carefully between:
- a DETERMINISTIC validator (lawful ship-stop),
- an LLM verdict gated by deterministic corroboration (lawful),
- a bare LLM opinion that raises (**unlawful -- this is what I am hunting**).

### Q2. What else breaks specifically at 720 words?

The P6 token budget was one length-scaling landmine, found by arithmetic rather
than by a live roll. Find the others. Concretely:

- Every budget, cap, ceiling, floor, window, or truncation that is a function of
  `target_words`, beat count, line count, scene count, or manifest size --
  evaluate each at **720 words** and tell me which one crosses a limit.
- The beat ceiling saturates at 40 for any `target_words >= 160`
  (`max(8, min(40, target_words // 4))`). What ELSE saturates, overflows, or
  silently clamps between 420 (proven green) and 720?
- The **technical** slot is local Nemo at an honest 8,192. Any technical pass
  whose prompt or artifact grows with episode length is a candidate. Which
  technical prompts get bigger at 720 words, and do they still fit?
- A repair/rewrite prompt is often the LARGEST call (it carries the failed
  artifact PLUS the contract). Which repair envelope is biggest at 720 words,
  and does it fit its slot?

### Q3. Where can a 720-word leg silently DEGRADE rather than fail?

Worse than a crash is an episode that ships but is quietly worse -- a truncated
artifact that still parses, a clamped budget that still returns valid JSON, a
scene list cut short, an advisory word target that silently becomes a hard cap.
The bakeoff's whole purpose is to compare STORY QUALITY across banks, so a
silent degradation that hits one bank harder than another **corrupts the
verdict**. Name any such path.

## Anti-goals

- Do not propose new features, new lanes, or architectural rewrites.
- Do not suggest raising a cap "to be safe" without arithmetic showing it is
  crossed.
- Do not invent a defect. A claim I cannot verify in the file is worse than
  silence -- I will ground every one of these against the real code and discard
  what does not survive.
