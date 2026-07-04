VERDICT: code-ready direction is good, but the implementation must introduce a
fresh upstream ledger-writing layer and only translate into the current writer/
ledger boundary when the source packet is ready.

CONFIRMED CODING SEAMS:

1. [CONFIRMED] The writer execution method is `run`, not `execute`.
   `OTR_LedgerScriptWriter.run(...)` currently accepts all optional widgets by
   keyword and calls `_resolve_inputs(...)`. `source_bank` must be appended to
   `INPUT_TYPES`, `run`, and `_resolve_inputs` together.

2. [CONFIRMED] `_resolve_inputs` is the right early selector for source mode,
   but it must not run source LLM interpretation before models/generate
   functions are available. It should select/fetch/normalize raw source inputs
   and create the ledger-writing spec shell.

3. [CONFIRMED] Root `__init__.py` owns `_NODE_MODULES` for video/image nodes.
   `OTR_VisualStyleDirector` should be added there directly unless the audio
   class registry is generalized. Do not use the audio-only registry by default.

4. [CONFIRMED] MetaBrief and ShotLock already use `forceInput` strings for
   wirable policy/gate sockets. `visual_style_policy_json` must follow the same
   pattern and append method kwargs.

MUST-FIX IN R2 PLAN:

1. Define fresh upstream modules:
   - `_otr_source_packet.py`
   - `_otr_ledger_writing_spec.py`
   - `_otr_story_prompt_profile.py`
   - `_otr_source_brains.py` or per-bank source brain modules
   - `_otr_ledger_input_adapter.py`
   These produce a normalized packet/spec that the existing writer can consume.

2. Define a source brain boundary:
   - science_news brain wraps current news/RSS interpreter.
   - media_archive brain reads archive item and runs archive prompt profile.
   - public_domain_story brain is not implemented until C7 and must raise
     clearly.
   All brains output `StoryInputPacket` + `LedgerWritingSpec`, not ledger rows.

3. Define the translator boundary:
   - active packet/spec -> current writer inputs (`script_brief`,
     `casting_brief`, `close_brief`, `key_terms`, source labels)
   - compatibility mirror -> `meta.news`
   - canonical source stamp -> `meta.source`

4. Define prompt profile variables from the audit artifact, not ad hoc strings.
   The first implementation can parameterize only known active prompt sites,
   but the audit artifact becomes the backlog/test oracle.

5. Decide story-quality path:
   The user's intent favors reusing the existing multi-stage structure. So R2
   should prefer parameterizing pitch-room/story-select/refine/dramatic-state
   prompts instead of bypassing them, unless a module is too tangled for C1.

6. Define visual-style catalog:
   `_otr_visual_style_catalog.py` provides style IDs and policies.
   `otr_visual_style_director.py` is thin UI/serialization.

7. Define visual policy behavior:
   - exact default style that preserves current output
   - base-tail strategy semantics
   - forbidden-term scrub using word-boundary regex + cleanup
   - ledger directives shape

8. Define workflow delta:
   - C1: writer `source_bank` widget append + workflow widgets append
   - C5/C6: visual style node + two forceInput links
   No source_text_path until public-domain adapter exists.

SHOULD-FIX:

1. Make the prompt audit artifact a required input to tests:
   archive/PD prompt rendering should fail if science-only phrases remain in
   active prompts.

2. Add explicit fixture shape for media archive items.

3. Keep `ConfigDict(extra="forbid")` for canonical contracts, but provide
   parse helpers with clear errors so node reports are operator-readable.

CUT:

1. Hidden offline fallback.
2. Unused public-domain widget in C1.
3. Inline visual style definitions inside the node class.

