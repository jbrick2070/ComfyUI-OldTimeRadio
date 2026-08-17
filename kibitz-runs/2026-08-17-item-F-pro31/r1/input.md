# Driver anchor -- item F, the Shakespeare wrong-play announcer frame

**Driver:** Claude (Cowork), sole judge. **Date:** 2026-08-17.
**Repo:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, branch
`v2.0-alpha`, HEAD `d85f4d79`.
**Panel this round:** Antigravity `Gemini 3.1 Pro (High)` + `Gemini 3.7 Flash (High)`
(operator-named 2026-08-17). Codex is quota-held until 2026-08-19 20:31 and is
excluded; **this is therefore not yet a full four-round arc and must not be
reported as one.**

**CITE SYMBOLS, NEVER LINE NUMBERS.** 8 of 21 citations rotted within an hour on
2026-08-17 because a diff shifted them. Every claim below names a function,
constant or class and quotes it. Reviewers: do the same, or your claim cannot be
grounded.

---

## 1. The measured defect

From the 2026-08-16 blind narrative read
(`docs/2026-08-16-blind-bank-narrative-ranking.md`), two current-era `shakespeare`
episodes:

* a **Twelfth Night** scene announced as *"Verona ... Capulets and Montagues"*
* a **Tempest** scene framed as **Romeo and Juliet**

Fidelity defect on the lane where fidelity outranks arc.

## 2. WHAT THE PLAN SAYS, AND WHY IT IS WRONG

`docs/GO_FORWARD_PLAN.md` section D states the shape as:

> *"The announcer FRAME is sampled independently of the selected excerpt instead
> of being generated from the same metadata record."*

**Both halves of that sentence are misleading, and a fix built on it would be
built against a ghost.**

**(a) Nothing samples a play or a setting.** The only draws anywhere on this path
are `_otr_shakespeare_sources.select_shakespeare_scene_ref` (which chooses the
SCENE, correctly, via `chooser.choice(scenes)`) and
`_otr_style_catalog.select_style` (which chooses a style slug by
`sha256(f"{cast_seed}:style:adaptation")`, deterministically). **No constant
anywhere in the tree contains "Verona"** on this path. The wrong place is a
free-text LLM field -- `_otr_outline._MacroShape.setting`, declared
`setting: str = Field(..., min_length=1)`. "Sampled" is true only in the
token-sampling sense, and reading it as a pool draw sends you hunting a pool that
does not exist.

**(b) It is not "generated from a different record" -- the record is NEVER
HANDED TO IT.** `source_meta_from_scene` produces a complete record
(`play_code`, `play_title`, `act`, `scene`, `scene_label`, `source_ref`, `year`,
`cast_hints`) and `OTR_LedgerScriptWriter` stamps it: `meta["source_meta"] =
dict(resolved["source_meta"])`. It then dies in three places:

1. `_run_source_interpreter` **takes** a `source_meta` parameter but the
   happy-path call is `briefs = interpreter(bank=..., payload=..., technical_fn=...,
   model_id=...)` -- `source_meta` is passed **only** inside the
   `except SourceInterpretError` fallback. `_otr_source_payload._interpret_shakespeare`
   has no `source_meta` parameter at all.
2. `_otr_outline.OutlineRequest` has **no** `source_meta`, `play_title` or
   `source_ref` field, and `_build_macro_user_prompt` renders only the brief,
   style, cast block, style grammar, diversity hint and prior macro. The play is
   not a fact in the macro prompt.
3. `_otr_line_composer.SafeOpenBrief` has **exactly five fields** --
   `setting, time_of_day, opening_status_quo, cast, era` -- and
   `compose_announcer_intro` builds its entire user message from those five.

Downstream, `play_title` is consumed by exactly two symbols,
`_otr_source_identity.identity_from_meta` (`prov["work_title"]`) and
`_otr_roster_gender.gender_map_for_names(..., play_code=...)`. **Neither is on the
announcer-frame path.** So this is a THREADING defect, not a consumption defect,
and "make the frame generator read the record it already has" is not a fix that
can be written -- it has no record.

**The routed system prompt asks for the very thing the body withholds.**
`nodes/story_packs/shakespeare/folger_scene_adaptation.json`,
`prompt_stages.announcer_intro_safe_system`, says *"Sentence 1 orients the
listener: the play-world place and who is there."* The model is instructed to
name a play-world place and is never told which play. Inventing Verona is the
most cooperative thing it can do.

## 3. THE THING THAT MAKES THIS A REAL DESIGN FORK, and it is not in the plan

**The five-field starvation is DELIBERATE and load-bearing.** `SafeOpenBrief`'s
own docstring:

> *"No-spoiler inputs for the announcer OPEN (KILL 2, 2026-06-24). Captured right
> after the outline is generated and BEFORE build_sq_data mutates the setup beat,
> so the open is composed by INPUT STARVATION: the script_brief (which can carry
> the outcome) is never passed -- only these setup-framed fields reach the prompt.
> ``cast`` is the LOCKED cast: the only proper names the announcer may use."*

So the announcer is starved ON PURPOSE, to stop it spoiling the ending, and
`cast` is explicitly the ONLY proper-name allowance. **A sixth field carrying
`play_title` is not a free addition: it widens a surface that was narrowed to
close a different defect, and it hands the announcer a proper name the KILL 2
design says it may not have.**

That is the fork the panel exists to break. Three shapes, and I am not asking
which is prettiest -- I am asking which cannot reopen KILL 2:

| # | Shape | What it costs | What I fear |
|---|---|---|---|
| **A** | Add `work_title` to `SafeOpenBrief` + `OutlineRequest`, thread it, and name it in the system prompt as a FACT the announcer must use | Two dataclasses, two prompt builders, one pack seam | Reopens the starvation surface. Does a title leak plot? "The Tempest" does not; "The Tragedy of Romeo and Juliet" arguably telegraphs an ending |
| **B** | Do not thread anything; make the system prompt FORBID naming any place or house not present in the locked cast / setting | Prompt-only, no schema change | It is persuasion, not structure -- the same objection Bible `12.103` records. And it makes the frame vaguer, not righter |
| **C** | Wire the already-built `_otr_passage_selector.select_passage`, whose own docstring says *"this is what stops a Forest-of-Arden scene being narrated as if it were Verona"* | Unknown -- it is **dead code with no production caller** (`nodes/`+`scripts/` grep returns only its own module, its test, and two comments in `_otr_episode_budget.py`) | A module written for this exact defect that was never wired is a fact worth knowing BEFORE we design a fourth thing. It may also be abandoned for a reason nobody wrote down |

## 4. A SECOND FRAME PRODUCER OVERWRITES THE FIRST

Fixing only the first producer fixes nothing. `OTR_LedgerScriptWriter` section
I.4.9 runs a post-composition **announcer-intro rewrite** via
`_otr_story_brief.derive_produced_open_brief` (prompt `_PRODUCED_OPEN_PROMPT`,
input builder `_build_produced_open_input`), which reads **only scene-1 spoken
ledger rows plus the cast roster** and then calls the SAME
`compose_announcer_intro` with a fresh `SafeOpenBrief`. It is even further from
the metadata record -- it never touches `meta` except `meta["period"]`. **Any fix
must land on BOTH producers or the rewrite silently restores the defect.**

## 5. IT IS THE FAMILY'S BUG, NOT ONE LANE'S

`compose_announcer_intro`, `SafeOpenBrief`, the `safe_open_brief` construction,
the I.4.9 rewrite and `OutlineRequest` are all bank-agnostic; only
`source_bank_id` differs, and it selects a system prompt through
`_otr_creative_prompt_router._PHASE_TO_PACK_SEAM["announcer_intro_safe_system"]`.
`public_domain` has the identical seam
(`faithful_radio_adaptation.json`: *"Sentence 1 orients the listener: the time
and place, and who is there."*) and the identical metadata drop
(`_otr_public_domain_sources.source_meta_from_unit` yields `title`/`author`/
`unit_label`; `_interpret_public_domain` takes no `source_meta`, with the comment
*"Mirrors the shakespeare lane"*).

**shakespeare is merely where it is FALSIFIABLE** -- a wrong play name can be
checked against a 14-row manifest. Whether `public_domain` has SHIPPED a wrong
title is **not proven** and I am not claiming it.

## 6. What I am asking the panel

1. **Break shape A on the spoiler question.** Is threading `work_title` into a
   deliberately starved prompt safe, and how would we know? What is the smallest
   addition that names the play without widening the surface?
2. **Is C alive or dead?** Read `_otr_passage_selector` and tell me whether
   wiring it is a fix or an archaeology project.
3. **The acceptance test.** This lane's proof cannot be a green unit test --
   Bible `12.103` and this repo's own 2026-08-17 receipts both record that a
   green gate is not a working fix. What is the cheapest test that would have
   caught *"Twelfth Night announced as Verona"*, and does it need a render?
4. **The second producer.** Does any fix that lands only on the first producer
   survive I.4.9? Say so explicitly.
5. **Tell me what I got wrong.** The panel corrected this driver on execution
   order twice on 2026-08-17. Assume there is a third.

## 7. Standing constraints -- a proposal that breaks one is dead on arrival

* **THE LAW:** an audit may improve a story, never fail one for length, language,
  style or quality. A frame gate that REJECTS an episode is forbidden.
* **No content guardrails on generated episodes** (operator 2026-08-03). This is
  a FIDELITY fix, not a filter.
* **Fidelity lanes invent nothing:** `shakespeare` and `public_domain` must be
  true to source. That is what makes this a correctness bug rather than the story
  quality the operator has closed.
* **Story quality is DONE** (operator 2026-08-04). Do not propose prose
  improvements to the announcer. Naming the right play is correctness; writing a
  better opening is not on the table.
* **A render must not die.** Degrade, never raise.
