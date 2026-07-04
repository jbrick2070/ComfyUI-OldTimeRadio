# R1 Final: Architecture, Story Models, And Visual Style Separation

Status: grounded synthesis for R2 coding-plan review.

## Verdict

Build-ready for planning, not for coding yet.

The top-level architecture is correct:

```
source_bank -> source material -> source brain -> story_model -> ledger_writing_spec
ledger_writing_spec -> existing multi-stage writer/adapter -> production ledger
visual_style_policy -> meta.visual_style -> MetaBrief / ShotLock / prompt seams
```

There is still one production ledger. The new layer is a ledger-writing control
plane that tells the writer how to fill that ledger.

The key correction from the latest user requirement is accepted: `media_archive`
must not be sci-fi anthology plotting with archive nouns pasted in. The plan now
needs three independent concepts:

- `source_bank`: where the material comes from.
- `story_model`: what kind of dramatic/tonal shape the story takes.
- `visual_style`: how still/video prompts should look.

## Must-Fix Architecture Decisions

1. Add `story_model` as a first-class planning concept.
   - Confirmed by user correction.
   - Initial media-archive models:
     - `media_restoration_adventure`
     - `cinematic_humorous`
     - `happy_archive_mystery`
     - `gentle_thriller`
     - `broadcast_history_comedy`
   - These are writing profiles, not visual styles.
   - They must carry forbidden plot-pattern guidance, especially "no Star-Trek
     / Amazing-Stories-style speculative anthology default."

2. Split raw source material from interpreted story material.
   - Accept Antigravity's concern, but use clearer names:
     - `SourceMaterialPacket`: raw title/author/url/file/hash/rights/source text
       reference/source summary.
     - `StoryInputPacket`: interpreted briefs, close/source note, key terms,
       story model, adaptation trace, source-fidelity expectations.
   - `LedgerWritingSpec` consumes `StoryInputPacket`, not raw feed/text alone.
   - This avoids circularity between ingest and LLM-authored briefs.

3. Keep the existing writer/quality machinery, but make the adapter explicit.
   - New source brains produce packets/specs.
   - `_otr_ledger_input_adapter.py` maps the spec into existing outline,
     casting, dramatic-state, title, line-composer, coda, and ledger fields.
   - Do not create a second complete story pipeline unless a stage proves
     impossible to parameterize.

4. Add runtime prompt-profile selection.
   - R2 must define `get_profile(source_bank_id, story_model_id)`.
   - The profile carries source labels, story form labels, grounding labels,
     coda mode, tone guardrails, forbidden plot patterns, and outline extras.
   - Confirmed prompt surfaces include `_otr_outline.py`, `_otr_pitch_room.py`,
     `_otr_story_select.py`, `_otr_dramatic_state_llm.py`,
     `_otr_line_composer.py`, `_otr_casting.py`,
     `OTR_LedgerScriptWriter.py`, and `_otr_style_picker.py`.

5. Rename the visual style id currently called `media_archive`.
   - Claude is right: using `media_archive` as both source bank and visual style
     repeats the `meta.news` confusion.
   - Keep source bank id: `media_archive`.
   - Use visual style id: `archival_documentary` or `broadcast_archival`.
   - UI label can still read "Media Archive" if desired.

6. Add an explicit visual ledger policy seam.
   - `finish_visual_prompt` currently ignores `meta.visual_style`.
   - `compose_still_prompt` appends hardcoded cinematic/radio tails.
   - `render_driver.py` has hardcoded LTX motion prompts.
   - R2 must define a pure helper that stamps/parses `VisualStylePolicy` into a
     ledger dict and a shared reader used by prompt finishers.

7. Preserve role safety separately from style.
   - Keep cast anchoring, no text, face/headroom framing, mesh isolation, and
     prompt hashing.
   - Move only aesthetic language like cinematic, 35mm, film grain, 1940s radio,
     and dramatic film lighting into style policy.

8. Keep `meta.news` as compatibility mirror, not meaning.
   - Mirror from active source packet:
     - `script_brief -> meta.news.script_brief`
     - `close_brief -> meta.news.news_close_brief`
     - `casting_brief -> meta.news.casting_brief`
     - `key_terms -> meta.news.key_terms`
   - Do not rename the legacy ledger key in R1/R2. Conceptual names can become
     source-neutral while compatibility keys stay stable.

9. Do not touch the canonical workflow until transplant.
   - `workflows/otr_scifi_16gb_full.json` remains untouched during upstream
     design and pure-module implementation.
   - Later transplant must be append-only for widgets, with validator,
     JSON round-trip, link audit, and widget/input audit.

## Rejected Claims

1. Reject adding `source_text_path` to the workflow/writer now.
   - It violates the deferred-transplant rule.
   - Add it only when the public-domain adapter is real.

2. Reject stamping visual style through a process-global ledger singleton as the
   architectural plan.
   - The repo has a production ledger singleton, but this feature should flow
     through explicit policy JSON / ledger dict stamping at MetaBrief, ShotLock,
     and prompt seams.

3. Reject cutting `anime`, `cartoon`, and `paper_origami`.
   - The user explicitly wants them as equal visual-style partners.
   - Keep them, with tests that prevent cinematic/radio tail leakage.

4. Reject cutting `rights_status` and `adaptation_trace`.
   - Public-domain and archive provenance/fidelity need them.

5. Reject Pydantic v1 fallback as a new-contract requirement.
   - Local venv confirms Pydantic `2.12.5`.
   - New strict contracts can use v2 APIs.

## R2 Coding Targets

R2 should convert this into chunks, in this order:

1. Contract modules:
   - `_otr_source_material_packet.py`
   - `_otr_source_packet.py` or `_otr_story_input_packet.py`
   - `_otr_ledger_writing_spec.py`
   - `_otr_story_prompt_profile.py`
   - `_otr_visual_style_policy.py`
   - `_otr_visual_style_catalog.py`

2. Runtime registries/factories:
   - `get_source_brain(source_bank_id)`
   - `get_profile(source_bank_id, story_model_id)`
   - `get_visual_style_policy(style_id)`

3. Adapter:
   - `_otr_ledger_input_adapter.py`
   - maps source/story/profile into existing prompt request shapes and
     compatibility `meta.news` mirror fields.

4. Prompt-profile integration plan:
   - outline/macro/phase/beat prompts
   - pitch room
   - story select/refine grading
   - dramatic state
   - line grounding/coda/title
   - `_otr_style_picker.py`

5. Visual-style integration plan:
   - policy parser/stamper
   - `finish_visual_prompt`
   - `compose_still_prompt`
   - MetaBrief
   - ShotLock
   - render-driver motion/fallback prompts

6. No-fallback tests:
   - unknown source/style raises
   - media archive does not call science RSS
   - media archive prompts do not include science/news/sci-fi anthology phrases
   - anime/cartoon/origami do not inherit 35mm/film-grain/1940s radio tails
     unless explicitly allowed by the selected style policy

## R1 Judgment Log

Accepted:

- Codex anchor: same ledger, fresh upstream control plane, explicit adapter.
- User correction: media archive needs its own story-model set.
- Antigravity: source ingest and interpreted story briefs need a clearer phase
  split.
- Antigravity: LTX/render-driver visual prompt hardcoding is a real style leak.
- Antigravity: `_resolve_inputs()` currently falls back to science RSS.
- Claude: `meta.visual_style` has no current writer/reader seam.
- Claude: `story_model` must be specified at contract level.
- Claude: source bank `media_archive` and visual style `media_archive` should
  not share the same id.
- Claude: `_otr_style_catalog.py` already owns narrative grammar naming; visual
  catalog naming must be clearly distinct.

Rejected or modified:

- Antigravity: `source_text_path` widget now.
- Antigravity: process-global ledger singleton as visual-style architecture.
- Antigravity: Pydantic v1 fallback for new contracts.
- Antigravity: cut anime/cartoon/origami.
- Antigravity: cut provenance/fidelity fields.
- Claude: cut `public_domain_story` entirely. Better: reserve it in pure
  registries or mark explicit not-implemented, but do not expose a workflow
  selector until the adapter is real.
- Claude: treat `news_close_brief` as something to rename now. Keep it as a
  compatibility key while moving conceptual prompt labels behind profiles.

