VERDICT: yes-with-fixes. The plan is structurally sound but will fail build/test verification due to brittle AST extraction of inline strings, dataclass validation TypeError bugs, and redundant disk I/O.

MUST-FIX BEFORE BUILD:
1. [Section 4 / Section 5] AST Extraction Failure for Outro Tail:
   - Defect: The plan maps `announcer_outro_resolved_tail` to an inline string literal inside the function body at [nodes/_otr_line_composer.py:3517](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_line_composer.py#L3517). Standard AST-extraction tests (which parse module-level `Assign` nodes) cannot verify inline literals nested inside conditional blocks in function bodies without complex, fragile syntax traversal. [ASSUMPTION]
   - Concrete Fix: Refactor the inline string literal in `nodes/_otr_line_composer.py` to a module-level constant `_ANNOUNCER_OUTRO_RESOLVED_TAIL` and reference it in both the function and the AST verification test.
2. [Section 3] Dataclass Instantiation Failure on Missing Optional Fields:
   - Defect: The `StoryPack` dataclass lists optional/inert fields (e.g. `label`, `status`, `examples`, `tone_guardrails`, `forbidden_plot_patterns`, `forbidden_leakage_terms`, `source_requirements`, `ledger_validation_notes`). If instantiated via `StoryPack(**data)` where the loaded JSON only contains required fields, Python will raise a `TypeError` due to missing arguments. [ASSUMPTION]
   - Concrete Fix: Declare explicit default values (such as `None` or `field(default_factory=...)`) for all optional fields in the `StoryPack` dataclass definition in `nodes/_otr_story_pack.py`.
3. [Section 3 / Section 5] Redundant Disk I/O Performance Bottleneck:
   - Defect: Resolving prompts occurs during dialogue line composition. If `resolve_creative_system_prompt` repeatedly reads and parses the JSON story pack file from disk inside the dialogue generation loops, it will hit performance and latency ceilings.
   - Concrete Fix: Implement a simple dictionary-based in-memory cache (e.g. `_PACK_CACHE`) inside `nodes/_otr_story_pack.py` to ensure each story pack JSON is read and parsed from disk at most once per execution.

SHOULD-FIX:
1. [Section 4] Allowlist Naming Inconsistency for Style Chooser:
   - Defect: The granular key authored in Stage 1 is `style_pick_chooser_user` (mapping to `_CHOOSER_USER_TEMPLATE`), but the reserved name listed in line 113 of the allowlist is `style_pick_chooser_user_template`.
   - Concrete Fix: Change the reserved name in `PRODUCTION_SEAM_ALLOWLIST` to `style_pick_chooser_user` to align with the granular key.
2. [Section 3] Undefined Exception Hierarchy:
   - Defect: The plan lists several loader errors (malformed JSON, duplicate keys, missing file, etc.) that should raise a typed `StoryPackError` naming the path, but does not specify distinct subclasses.
   - Concrete Fix: Define distinct subclasses (e.g. `StoryPackNotFoundError`, `StoryPackParseError`, `StoryPackValidationError`, `UnknownSeamError`) inheriting from `StoryPackError` to allow callers to catch specific failure modes.

OPTIONAL / NICE-TO-HAVE:
- [Section 9] Seam Selection for Stage 1b Pilot:
  - Recommendation: Use `line_composer_system` as the pilot seam instead of `outline_system`. `line_composer_system` is passed directly to the LLM message content without identity comparison logic, whereas `outline_system` is only used to evaluate a complex overlay check (`is` vs `==`) and is not actually sent to the LLM as a direct outline prompt (since outline uses three stage-specific prompts).

CUT THESE (over-engineering):
1. [Section 4 / Section 9] Split Keys for `coda` Prompts:
   - Why it is safe to cut: The plan suggests splitting the coda prompt into `coda_system` and `coda_examples` and joining them at runtime. Unlike `announcer_outro` (where the outro tail is conditionally appended), the examples for `coda` are appended unconditionally. Splitting them introduces redundant JSON structures, loader complexity, and whitespace validation bugs. Storing a single, pre-joined `coda_system` prompt in JSON is safer and simpler.
