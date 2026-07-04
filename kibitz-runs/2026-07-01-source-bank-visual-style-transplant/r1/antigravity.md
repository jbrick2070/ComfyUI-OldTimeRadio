VERDICT: build-ready as-is? no. The plan lacks typing for Pydantic contracts, misses the user-facing source input widget for public-domain adaptation, and contains a structural divergence where style-agnostic script writing is forced to handle visual directives.

MUST-FIX BEFORE BUILD:
1. [Expected Contracts] Defect: The Pydantic model contract for `VisualStylePolicy` (lines 265-280) is declared with field names only and lacks type annotations and default values. Since the plan requires Pydantic v2 and `ConfigDict(extra="forbid")` (line 282), unannotated attributes will be ignored or raise immediate validation/compilation errors.
   Concrete Fix: Explicitly type and default annotate all `VisualStylePolicy` fields (e.g., `style_id: str`, `label: str = ""`, `positive_tail: str = ""`, `negative_or_forbidden_terms: list[str] = Field(default_factory=list)`, `base_tail_strategy: Literal["keep", "replace", "suppress"] = "keep"`, `image_grade_tail: str = ""`, `radio_broadcast_tail_replacement: str = ""`, `announcer_visual_subject: str = ""`, `music_visual_subject: str = ""`, `scene_open_subject: str = ""`, `character_portrait_style: str = ""`, `character_scene_style: str = ""`, `motion_prompt_profile: str = ""`, `ledger_directives: dict[str, Any] = Field(default_factory=dict)`).

2. [Source Banks] Defect: The plan introduces the `public_domain_story` bank for "faithful adaptation... for public-domain source text" (line 56) but does not define any node input widget or socket (such as `source_text_path` or `source_text`) on `OTR_LedgerScriptWriter` (or any other node) to specify the source text path. Without it, the bank is non-functional as it lacks a data input pathway.
   Concrete Fix: Declare `source_text_path` (STRING, default="") as a new optional widget input for `OTR_LedgerScriptWriter.INPUT_TYPES`, appended at the end of the optional inputs dictionary to prevent positional LiteGraph shifts in saved workflows.

3. [Expected Contracts] Defect: The `StoryInputPacket` contract (lines 243-260) mixes raw ingestion metadata (e.g., `source_title`, `source_hash`) with LLM-interpreted briefs (`casting_brief`, `script_brief`, `close_brief`, `key_terms`). If this packet is the input to the prompt profile and is used to write the ledger spec (line 23), this forms a circular dependency because the interpreted fields do not exist before the news interpreter / source brain LLM runs.
   Concrete Fix: Split the contract into `SourceIngestPacket` (representing raw inputs before LLM interpretation) and `StoryBlueprintSpec` (representing the interpreted briefs and terms generated during the run).

4. [Visual Style Architecture] Defect: In `nodes/_otr_video_engines/render_driver.py` (lines 546-564), LTX motion prompts are hardcoded inside the module-level dictionary `_LTX_MOTION_PROMPT_BY_ROLE` and contain science-fiction / radio-console-specific language ("Vacuum tubes pulse", "Tuning dial needle"). If a non-radio style (e.g., `anime` or `paper_origami`) is selected, the render engine will still produce radio-console-centric motion prompts.
   Concrete Fix: Require `nodes/_otr_video_engines/render_driver.py` to extract `motion_prompt_profile` from `ledger["meta"]["visual_style"]` and select motion prompt strings dynamically based on the policy, falling back to `_LTX_MOTION_PROMPT_BY_ROLE` only when the visual style is `sci_fi_radio`.

5. [Operator Intent / Ledger-Writing Spec] Defect: The plan states that the writer fills the ledger using `ledger_writing_spec` which includes "selected `visual_style`" and "visual ledger directives for still/video fields" (lines 124-125). However, the writer (`OTR_LedgerScriptWriter`) composer is completely decoupled from visual style ("Style must not rewrite source facts... or dialogue", line 197) and has no `visual_style` widget input. The writer cannot resolve the visual style policy during script composition.
   Concrete Fix: Decouple the writer from visual style resolution. Have `OTR_VisualStyleDirector` stamp the visual style policy and visual directives directly onto the process-global ledger singleton (`production_ledger.py`) downstream of the writer.

SHOULD-FIX:
1. [No Silent Fallbacks] Defect: In `OTR_LedgerScriptWriter.py`'s `_resolve_inputs()` (lines 1367-1371), the code currently falls back to `_fetch_rss_seed_or_die()` which hardcodes the science-news RSS fetcher. If a user selects `media_archive`, it will execute the science RSS fetcher, violating the non-science isolation rule.
   Concrete Fix: Rewrite `_resolve_inputs()` to choose the RSS fetcher dynamically based on `source_bank` (e.g., executing the archive feed fetcher for `media_archive` and raising not-implemented for `public_domain_story`).

2. [Story Model / Tone Layer] Defect: The plan defines `story_model` (lines 93-104) but does not specify how these models are injected into the outline generator in `nodes/_otr_outline.py` to enforce the forbidden plot patterns and tone guardrails.
   Concrete Fix: Parameterize outline prompts in `nodes/_otr_outline.py` to append the prompt profile's `outline_rules_extra` and `forbidden_plot_patterns` variables dynamically based on the active `story_model`.

3. [Expected Contracts] Defect: [ASSUMPTION] The plan requires strict Pydantic v2 models but does not check for environment portability. While the local venv uses v2, other developer environments might run Pydantic v1.
   Concrete Fix: Include a v1 compatibility fallback check in the imports of the new contracts modules, similar to the try/except block in `nodes/news_interpreter.py` (lines 65-70).

OPTIONAL / NICE-TO-HAVE:
1. Provide a template JSON file under `config/source_banks/` showing a mocked fixture for `media_archive` to allow developers to test the parser offline.

CUT THESE:
1. Cut `anime`, `cartoon`, and `paper_origami` from the initial visual style catalog. These styles represent major visual paradigm shifts requiring complex negation and prompt-finisher exclusions (e.g., scrubs for film grain/35mm). Restricting the initial catalog to `sci_fi_radio`, `cinematic_35mm`, and `media_archive` isolates the policy injection logic without prompt contamination risk.
2. Cut `adaptation_trace: dict[str, Any]` and `rights_status: str` fields from `StoryInputPacket`. These fields are completely unused by the downstream audio/video/cast layers in the current codebase. Cutting them simplifies the initial Pydantic contract and prevents unnecessary serialization bloat.
