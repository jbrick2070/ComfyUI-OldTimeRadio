# Question -- 2026-05-22

# Design consult: dedicated announcer intro/outro LLM passes (OTR "SIGNAL LOST")

## System under discussion
OTR is a ComfyUI radio-drama generator. A "writer" node (`OTR_LedgerScriptWriter`) builds a JSON "ledger" with a `lines[]` array. Each episode has an ANNOUNCER (the radio host who frames the story: an opening line and a closing line) plus one or more characters. Per-line dialogue text is produced by `compose_line` in `nodes/_otr_line_composer.py`.

## Current state (verified by a code map)
- The outline (`nodes/_otr_outline.py`) deterministically stamps the FIRST beat (b001) and the LAST beat as `speaker_role="announcer"` -- so the announcer intro/outro line SLOTS always exist.
- INTRO text: routes through the shared `compose_line` with `speaker=ANNOUNCER`. There is no announcer-specific prompt branch in `_build_user_prompt` -- the intro gets the same "you are now ANNOUNCER, produce one line of dialogue" prompt as a character line. The only role-aware divergence today is an optional `polish_line` pass with an announcer-specific system prompt.
- OUTRO text: also composed by the shared `compose_line`, then a post-loop helper `override_announcer_close` is supposed to overwrite it verbatim with the news interpreter's `news_close_brief` (a journalistic closing read derived from the source news article). That overwrite is currently BROKEN by a key-name contract bug, so the outro is just generic composer output.
- A code comment in the writer states the "'ANNOUNCER bookends' technical pass" is "a hypothetical refactor that doesn't exist."

## The proposal
Promote the announcer intro and outro into their own dedicated LLM pass(es), separate from the character-dialogue `compose_line`:
- `compose_announcer_intro()` -- fed `script_brief` (a <=350-char premise/arc brief from the news interpreter); produces a purpose-written framing intro.
- `compose_announcer_outro()` -- fed `news_close_brief`; rewrites it into in-voice announcer narration, replacing the broken verbatim-stamp overlay.
Both write their `text` into the existing `ledger.lines[]` rows. The ledger is the interface to the audio stage, so nothing downstream changes.

## Hard constraints
- Audio output must stay byte-identical to baseline. The ledger row SHAPE must not change -- only how the announcer's `text` is generated.
- Project rule: every LLM call is tagged `creative` or `technical` and routed through the writer's `creative_writing_model` / `technical_model` slots -- no new model widget. Announcer narration is a creative/narrative pass, so it would be the `creative` slot.
- Single RTX 5080 / 16 GB VRAM, 14.5 GB ceiling. Adding LLM calls is acceptable but they should reuse the already-loaded creative model.
- Each pass must have a deterministic fallback line if its LLM call fails -- the narrative frame (a beginning and an end) must never be missing.

## Questions
1. One combined "bookend" pass that emits BOTH the intro and the outro in a single LLM call, or two separate passes? Trade-offs -- open/close coherence from one call, vs. the outro needing post-script content (the finished script + news_close_brief) that the intro does not.
2. Should the outro pass run inside the per-beat composer loop, or stay a post-loop pass (it needs the finished script and `news_close_brief`)?
3. Deterministic fallback design -- what should the fallback intro/outro text be, and how should the code detect "the LLM pass failed or drifted" cleanly?
4. Any risk in retiring `override_announcer_close` entirely, versus keeping a fixed verbatim-stamp as the fallback?
5. Anything missing -- failure modes, ordering hazards, or a simpler approach than two new LLM calls?

Cite the OTR file names above where relevant. Prefer the smallest change with the largest payoff.
