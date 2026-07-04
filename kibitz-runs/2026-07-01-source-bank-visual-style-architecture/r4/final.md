# R4 Final Convergence

Status: converged. This round found implementation guardrails, not a new
architecture.

## Verdict

Proceed with the two-axis redesign:

- Source-bank fork before story/ledger generation.
- Visual-style fork after the story ledger exists, feeding prompt composition.

The first implementation remains conservative: pure schemas, existing writer
integration, prompt-profile fixes, and a visual-style policy fan-out. No new
source-director node in V1.

## Accepted Final Fixes

1. Both writer widget guardrails must be updated.
   - `OTR_LedgerScriptWriter` inline optional count: 16 -> 17.
   - `tests/test_workflow_json_guardrails.py` writer `widgets_values` length:
     25 -> 26 after appending `"science_news"` to the workflow.

2. Add `source_bank` to the headless creative whitelist.
   - `nodes/_otr_workflow_apply.py`
   - `scripts/otr_api.py`
   - keep `tests/test_workflow_apply.py` parity green.

3. The visual-style meta injection seam must be explicit.
   - `OTR_VisualStyleDirector` emits JSON.
   - `OTR_MetaBriefImagePromptGen.generate` and `OTR_ShotLock.lock` parse it.
   - Each injects parsed policy as `meta["visual_style"]` into a local meta
     dict before any prompt finishing.
   - `finish_visual_prompt` reads `_meta(meta).get("visual_style")`.

4. `finish_visual_prompt` remains the style application seam.
   - No policy means byte-identical current behavior.
   - Malformed policy falls back visibly with a warning/report.
   - `style_tail=False` suppresses only the default tail, not the policy
     positive tail.

5. C2 must cover all active outline system prompts.
   - `_SYSTEM_PROMPT`
   - `_MACRO_SYSTEM_PROMPT`
   - `_PHASE_SYSTEM_PROMPT`
   - `_BEAT_SYSTEM_PROMPT`
   - Builders return constants unchanged for `science_news` and only substitute
     wording for alternate source banks.

6. V1 chooses bypass over parameterizing pitch-room/story-select.
   - For `source_bank != science_news`, bypass `_otr_pitch_room.py`,
     `_otr_story_select.py`, and story-grade refine loops.
   - Parameterizing those paths is later work.

7. `OTR_VisualStyleDirector` file and registration are fixed.
   - File: `nodes/otr_visual_style_director.py`
   - Class: `OTRVisualStyleDirector` or repo-consistent class name.
   - Public node key: `OTR_VisualStyleDirector`.
   - Register through the repo's node-registration path, including
     `_otr_class_registry.py` and the top-level loader/mappings.
   - Add a registration test.

8. Use `Field(default_factory=...)` for mutable Pydantic defaults.
   - `StoryInputPacket.key_terms`
   - `StoryInputPacket.adaptation_trace`
   - `VisualStylePolicy.forbidden_terms`

9. Media archive V1 needs concrete sources and offline fixtures.
   - Runtime source table can start with curated LOC/NFPF/ACE-style sources.
   - Tests use local fixture items and network-free interpreters.
   - A dead source fails closed with a clear message.
   - Do not make production depend only on a toy static list.

10. C7 public-domain input is `source_text_path` first.
    - Avoid giant pasted story text in a ComfyUI widget.
    - Public-domain flow remains:
      `source_text_path -> StoryInputPacket -> StoryBlueprint -> OutlineRequest -> ledger`.

## Rejected Or Modified

- Rejected: replacing the full `StoryInputPacket` schema with only
  `title/raw_text/key_terms`. Keep the richer provenance schema from R2/R3.
- Rejected: treating Antigravity's "optional widget count is 22" as a
  replacement for the inline writer assertion. It identified a different
  workflow guardrail; both guardrails must be updated.
- Rejected: adding `story_scaffold` to the whitelist as part of this redesign.
  That is an existing policy question, not required for `source_bank`.
- Modified: the media archive source table should be concrete, but production
  should not be a static-only demo database.
- Modified: the render-batch ledger-sync issue remains a verify/fix bug watch,
  not a source/style architecture blocker.

## Final Build Shape

1. C0: add pure contracts for `StoryInputPacket`, `StoryBlueprint`, and
   `VisualStylePolicy`.
2. C1: append `source_bank` to the writer, workflow widget vector, guardrail
   tests, and headless whitelists.
3. C2: add source prompt profiles and fix all active outline system prompts.
4. C3: implement `media_archive` inside the writer, with raw fetch in
   `_resolve_inputs` and LLM interpretation after model load.
5. C4: implement visual style in `finish_visual_prompt`.
6. C5: add and register `OTR_VisualStyleDirector`.
7. C6: wire visual style to MetaBrief and ShotLock in the canonical workflow.
8. C7: add public-domain `source_text_path` adapter after C0-C6 are green.

## Final Verification Gates

- Targeted tests per chunk.
- Workflow validator on `workflows/otr_scifi_16gb_full.json`.
- JSON round-trip, link integrity, widget/input audit.
- Headless whitelist parity test.
- Visual-style registration test.
- Default science-news prompt and visual prompt stability tests.
- Media archive fixture test with no science wording leak.
- `public_domain_story` clear not-implemented test until C7 lands.
- Full regression suite and Bug Bible after code changes.

