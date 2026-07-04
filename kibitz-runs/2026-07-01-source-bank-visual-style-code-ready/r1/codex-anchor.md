VERDICT: strong architecture, buildable if the plan keeps "same ledger" as a
hard contract and treats source bank + visual style as independent axes.

CONFIRMED ARCHITECTURE FIT:

1. [CONFIRMED] `OTR_LedgerScriptWriter` is the right V1 owner for the source
   bank selector. The canonical workflow has writer node id 1 before visual
   policy/prompt nodes. Adding an upstream standalone source node first would
   create unnecessary wiring churn.

2. [CONFIRMED] The ledger can remain the bible. Source banks should change the
   prompt pack and data used to fill the ledger, not create alternate downstream
   schemas. This matches the current graph: writer -> freeze/cast/audio ->
   MetaBrief/ShotLock -> image/video.

3. [CONFIRMED] Visual style must be ledger-level direction, not a prompt suffix
   only. Current graph has `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock`
   as the obvious consumers. `finish_visual_prompt` is the shared prompt seam.

4. [CONFIRMED] The top-level schema should reuse the existing multi-stage story
   structure. The new abstraction is source intent -> prompt profile -> existing
   story/ledger stages -> same ledger. Science/news, media RSS/archive, and
   public-domain adaptation alter the prompt profile and source packet, not the
   downstream ledger contract.

MUST-FIX BEFORE CODE:

1. [C2: prompt surface] The active science assumptions are broader than one
   prompt. Confirmed strings exist in:
   - `nodes/_otr_outline.py`
   - `nodes/_otr_pitch_room.py`
   - `nodes/_otr_story_select.py`
   - writer self-test/sample text
   The code-ready plan must require all active generation prompts to route
   through `source_bank` prompt profiles or be explicitly proven test-only/dead.

2. [C1: widget guardrails] Adding `source_bank` must update both current
   guardrails:
   - writer inline optional count `assert n_optional == 16`
   - workflow JSON guardrail `assert len(wv) == 25`
   and append `"science_news"` to the canonical workflow writer widgets.

3. [C1: headless support] `source_bank` must be added to the creative whitelist
   in both `nodes/_otr_workflow_apply.py` and `scripts/otr_api.py`, with parity
   tests. Otherwise headless/API runs cannot select the new bank.

4. [C5/C6: method contract] Adding `visual_style_policy_json` sockets requires
   appending method parameters to:
   - `OTR_MetaBriefImagePromptGen.generate`
   - `OTR_ShotLock.lock`
   and wiring the canonical workflow in the same change.

5. [No silent fallback] The plan must distinguish explicit current mode from
   fallback. `science_news` is a valid selected mode; `media_archive` failing
   into science prompts is forbidden.

SHOULD-FIX:

1. [Prompt profile shape] Name the source profile fields before coding:
   system label, source label, develop verb, story form label, close/coda mode,
   and ledger intent labels. Otherwise implementors will keep passing raw
   strings through ad hoc call sites.

2. [Visual style shape] `VisualStylePolicy` should carry both prompt-tail
   language and ledger directives, because anime/noir/cartoon/origami must
   influence visual ledger sections, not just final text tails.

3. [Implementation order] Do not start with public domain. Build contracts,
   source selector default, visual-style stamp, then media archive. Public
   domain is the first true adaptation branch and should be later.

CUT:

1. Do not create `OTR_SourceBankDirector` in V1.
2. Do not add arbitrary public-domain search in V1.
3. Do not keep a source-bank path that quietly reuses the science prompt pack.

R1 JUDGMENT:

The user's concept is now clear: one ledger bible, multiple source brains,
multiple visual bibles. The plan should advance to coding review with that as
the invariant.
