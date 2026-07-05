# r1 Anchor Review (Claude driver, pre-fan-out)

Target: docs/R1_ARCHITECTURE_AND_CODING_PLAN_V2.md
Focus: high-level arc / creative coherence.

VERDICT: architecture is sound and correctly law-driven (JSON content, Python
behavior, no fallbacks, declared models/engines). Four self-identified risks
must fold into the plan before coding starts.

MUST-FIX:

1. pipelines.json fiction risk. `legacy_many_pass` is NOT declaratively
   executable - the many-pass structure is hardwired inside
   production_mirror/nodes/OTR_LedgerScriptWriter.py (6155 lines; pass order
   baked into the writer). CONFIRMED. The plan must mark the
   `legacy_many_pass` entry as descriptive metadata (stamping/audit only),
   with only `simple_4_prompt_experimental` lab-executable via the FakeLLM
   runner. Otherwise the JSON claims an execution semantic production will
   not honor tomorrow.

2. Seam template variables vs production plumbing. Production's line
   grounding is a static instruction appended at
   production_mirror/nodes/_otr_line_composer.py:1642 - there is no
   {scene_premise}-style variable plumbing at that call site. CONFIRMED.
   v2 seam templates must be zero-variable prose or label-substitution only
   ({source_grounding_label} class), with the declared-variable mechanism
   present but conservative. Variable plumbing beyond labels is transplant
   scope creep.

3. Pack duplication inside one bank. Five media_archive packs would repeat
   labels/coda_mode/story_form - drift-in-JSON replaces drift-in-Python.
   Fix: bank-level seam/label defaults in banks.json with SINGLE-LEVEL pack
   override (explicitly not an inheritance tree).

4. Byte-pin needs exact constants. The sci_fi_radio policy pin must equal
   the four production constants at
   production_mirror/nodes/_otr_story_brief_helpers.py:228-251
   (ERA_TAIL_DEFAULT, STYLE_TAIL_DEFAULT, IMAGE_GRADE_TAIL,
   RADIO_BROADCAST_TAIL) byte-identically, comma and space placement
   included. CONFIRMED strings available in the mirror.

SHOULD-FIX:

- provenance.production_baseline should be read from
  PRODUCTION_MIRROR_MANIFEST.md programmatically, not hand-typed.
- Cross-product tests must keep fixture contexts tiny so the 12x5 matrix
  stays sub-second.
- The plan should state explicitly that nodes.py dropdown choices come from
  the registry (JSON-discovered), never from module-level Python lists,
  including the source-bank list itself.

UNVERIFIABLE (verify at build): none material; production behavior claims
above were verified against production_mirror copies, not the live repo -
acceptable because the mirror is the pinned baseline for this work.
