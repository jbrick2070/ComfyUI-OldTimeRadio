# Sprint D Cowork Action Plan v3 -- 2026-05-16

## §0 What changed from v2

A new commit **D0d** lands before any catalog work begins to fix a MAJOR Sprint C wiring bug: three downstream consumer `ledger_json` inputs (HuMo, VideoComposite, LTX) were sourced from `SignalLostVideo.video_path` -- a video file path string -- instead of the actual `FreezeCascade.script_json`. This is a silent STRING-to-STRING semantic miswire pytest cannot catch but runtime JSON parsers will reject. D0d also decouples `audio_gate` from `SignalLostVideo.video_path` so FLUX planning no longer waits on 1080p procgen video completion, wires HuMo `portraits_dir` to the FLUX portrait render output so HuMo actually consumes generated portraits (pending G1 operator confirm intent), and confirms `OTR_WorkflowValidator` carries `OUTPUT_NODE=True` so it actually executes. Finding "creative model not carried into FreezeCascade" is **already resolved** by D2b's `meta.creative_model` and `meta.creative_prompt_profile` stamping introduced in v2; row added to §5 to make the alignment explicit. Operator gates renumbered: G1 workflow intent (new, before D0d), G2 license verdict, G3 HF login, G4 audio C7, G5 context window, G6 release readiness. Total day estimate moves from 6.0d to 6.5d, at the cap. All other v2 decisions hold: `prompt_profile="otr_1940s_v1"`, duck-typed loader, AST-node-scoped sweep carve-out, `.py`-only file extension scope, hard-fail workflow validator on non-`mit_equivalent` creative bindings, phase-name caller inventory at D2b, reframed reflection boundary, both allocated and reserved VRAM peaks under 14.5 GB, determinism `xfail(strict=False)`, no-news fixture for diction guard plus modern-news warning advisory, context-window precondition strict by default.

## §1 Final commit chain

| #  | Commit  | What lands                                                                                                  | Day est | Pytest count target |
|---:|---------|-------------------------------------------------------------------------------------------------------------|--------:|--------------------:|
| 1  | D0a     | Branch cut from `sprint-c-story-brief-v2 @ a125a35`. Plan lands as `docs/closed-sprints/2026-05-16-sprint-d-period-llm.md`. | 0.25 | 0 new |
| 2  | D0b     | License-audit framework over explicit `AUDIT_TARGETS.txt`. Per-row markdown audit files plus schema-positive gate.            | 0.5  | +4 |
| 3  | D0c     | SA-101 carryover: interpolated `log.info` clamp-visibility at `_otr_story_brief.py` repair pass + 2 tests.                    | 0.25 | +2 |
| 4  | **D0d** | **MAJOR Sprint C miswire fix** plus 3 medium rewires. Three `ledger_json` inputs rewired to `FreezeCascade.script_json`. `audio_gate` decoupled. HuMo `portraits_dir` wired. `OTR_WorkflowValidator` `OUTPUT_NODE=True` confirmed. | 0.5 | +5 |
| 5  | D1a     | `CuratedModel` extension: 6 new fields, backfill 6 existing rows, append talkie row reading from D0b audit. `prompt_profile="otr_1940s_v1"`. | 0.5 | +5 |
| 6  | D1b     | Loader-backend protocol + `transformers_gptq_int4` adapter scaffold + `compute_effective_context_limit` helper.                | 0.75 | +6 |
| 7  | D1c     | Talkie loader smoke runtime-gated + tokenizer chat-template guard + effective-context-limit clamp test + dropdown coverage.    | 0.5 | +6 |
| 8  | D2a     | `nodes/_otr_creative_prompt_router.py` resolver defined; caller-count 0; few-shot OMIT documented as zero-callers test.        | 0.5 | +5 |
| 9  | D2b     | Wire resolver to 4 sites by phase-name inventory. Writer stamps `meta.creative_model` + `meta.creative_prompt_profile`. Audio C7 proxy + clamp counter. | 0.75 | +10 |
| 10 | D2c     | Chat-template + stop-token dispatch on `chat_template_kind` only.                                                              | 0.5  | +5 |
| 11 | D3      | Reframed reflection boundary, AST-node-scoped sweep carve-outs, `.py`-only sweep scope, default-workflow validator hard-fail, news_interpreter unrouted test. | 0.75 | +8 |
| 12 | D4      | Runtime-gated VRAM peak (allocated + reserved), determinism xfail, no-news diction guard, news-warning advisory, context-window precondition. | 0.5 | +6 |
| 13 | D-final | Archive `SPRINT.md`. Rename workflow JSON FreezeCascade title `Phase 0..10` -> `Phase 0.10`. Sprint A backlog drops SA-101.    | 0.25 | +1 |

**Total:** 6.5 days (at cap). **Calendar window to v1.9 Memorial Day 2026-05-24/25:** 8-9 days. **Slack:** 1.5 - 2.5 days.

---

## §2 Per-commit Cowork brief

### D0a -- Branch cut and plan landing

- **Goal:** Cut `sprint-d-period-llm` from `sprint-c-story-brief-v2 @ a125a35`; land this plan.
- **Files touched:** `docs/closed-sprints/2026-05-16-sprint-d-period-llm.md` (new); `SPRINT.md` (overwrite header).
- **Code changes:** Paste this plan. Set `SPRINT.md` first line to `# Sprint D -- period-LLM CATEGORY -- in progress`.
- **Wire-in:** None.
- **Pytest to add:** None.
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy against `tests/fixtures/baseline_v1.5.wav` + `.sha256`.
- **Commit message:**
  ```
  Sprint D D0a cut sprint-d-period-llm and land v3 plan seed

  Cut from sprint-c-story-brief-v2 a125a35. v3 plan lands as
  docs/closed-sprints/2026-05-16-sprint-d-period-llm.md and
  SPRINT.md is reset to the Sprint D skeleton header.
  ```
- **Exit condition:** Branch pushed to origin; plan visible; Bug Bible green.

### D0b -- License-audit framework over explicit repo_id list

- **Goal:** Land generic license-audit framework keyed off `docs/model-licenses/AUDIT_TARGETS.txt`. Audit files for 6 existing rows plus talkie exist before D1a reads them.
- **Files touched:** `docs/model-licenses/AUDIT_TARGETS.txt` (new, 7 lines); `tools/audit_model_license.py` (new); `docs/model-licenses/<sanitized repo_id>.md` x 7 (new); `tests/test_license_audit_schema.py` (new).
- **Code changes:** YAML frontmatter per audit file:
  ```
  ---
  repo_id: talkie-lm/talkie-1930-13b-it
  license: non_commercial
  license_audit_status: research_lane
  verdict_date: 2026-05-XX
  audit_method: hf_model_card_read_plus_license_file
  reviewer: operator
  ---
  Forensic notes here.
  ```
  `tools/audit_model_license.py` reads `AUDIT_TARGETS.txt`, validates each audit file has 4 required keys plus matching repo_id.
- **Wire-in:** None at this commit. D1a consumes the verdicts.
- **Pytest to add:**
  - `test_every_audit_target_has_a_corresponding_markdown_file`
  - `test_audit_yaml_frontmatter_has_required_keys`
  - `test_audit_repo_id_matches_filename`
  - `test_talkie_audit_status_is_research_lane`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D0b license audit framework keyed off explicit repo id list

  Lands tools/audit_model_license.py plus per repo audit files
  under docs/model-licenses for the 6 existing catalog rows and
  the talkie row. Framework reads AUDIT_TARGETS.txt so audit
  files exist before D1a constructs the catalog rows. Talkie
  verdict is research_lane non_commercial. Generic across Mistral
  Gemma Qwen variants.
  ```
- **Exit condition:** 4 new pytest pass. G2 (license verdict) clears before D1a opens.

### D0c -- SA-101 silent clamp visibility carryover

- **Goal:** Pull SA-101 forward from Sprint A. Interpolate `str(e)` so exception context survives the clamp.
- **Files touched:** `nodes/_otr_story_brief.py` (one line between current 490 and 491); `tests/test_story_brief_clamp_logging.py` (new).
- **Code changes:**
  ```python
  log.info(
      "[OTR_StoryBrief] repair pass clamped temperature to 0.55 "
      "after exception in initial reflection pass: %s",
      str(e),
  )
  ```
- **Wire-in:** None. Pure additive emission on exception arm.
- **Pytest to add:**
  - `test_repair_pass_emits_clamp_log_when_clamp_fires`
  - `test_repair_pass_clamp_log_does_not_break_no_change_logs_rule_for_other_strings`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical (happy path emits no new log).
- **Commit message:**
  ```
  Sprint D D0c surface reflection repair clamp via log.info

  Pulls SA-101 forward from Sprint A backlog. The repair pass
  temperature clamp around line 490 emits a single log.info line
  on the exception arm with str(e) interpolated so the original
  structural JSON failure is captured before the temperature drop
  hides it. Additive only no existing log string modified.
  ```
- **Exit condition:** Both tests green; existing log strings byte-stable.

### D0d -- Workflow JSON rewires and validator activation

- **Goal:** Fix MAJOR Sprint C miswire: three downstream `ledger_json` inputs were sourced from `SignalLostVideo.video_path` (a video file path string) instead of `FreezeCascade.script_json`. Decouple `audio_gate` from procgen video completion. Wire HuMo `portraits_dir`. Confirm `OTR_WorkflowValidator.OUTPUT_NODE=True`. **G1 must resolve operator intent on portraits and audio_gate replacement source before this commit opens.**
- **Files touched:**
  - `workflows/otr_scifi_16gb_full.json`
  - `nodes/_otr_workflow_validator.py` (only if `OUTPUT_NODE` attribute is missing or False)
  - `tests/test_workflow_json_wiring_invariants.py` (new)
- **Code changes:** JSON link edits:
  - HuMo `ledger_json` input: source -> `FreezeCascade.script_json`
  - VideoComposite `ledger_json` input: source -> `FreezeCascade.script_json`
  - LTX `ledger_json` input: source -> `FreezeCascade.script_json`
  - VideoPlan `audio_gate` input: source -> `FreezeCascade.script_json` (or `EpisodeAssembler.completion_token` per G1)
  - HuMo `portraits_dir` input: source -> FLUX portrait batch render output path (per G1)
  - `OTR_WorkflowValidator` class: ensure `OUTPUT_NODE = True` at module level
- **Wire-in:** Workflow JSON link IDs renumbered consistently; ComfyUI loads cleanly. The default workflow must still load and the audio C7 proxy must still hold at default config (the rewire corrects WHERE the JSON comes from, not WHAT it contains, so the byte-identical baseline at default config is preserved).
- **Pytest to add:**
  - `test_no_ledger_json_input_sources_from_signal_lost_video_path_output`
  - `test_all_ledger_json_inputs_source_from_freeze_cascade_script_json`
  - `test_audio_gate_does_not_source_from_signal_lost_video_path`
  - `test_humo_portraits_dir_is_linked_to_portrait_render_output`
  - `test_otr_workflow_validator_class_has_output_node_true`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D0d workflow JSON rewires and validator activation

  MAJOR Sprint C miswire fix. Three ledger_json inputs HuMo
  VideoComposite LTX were sourced from SignalLostVideo video_path
  a video file path string instead of the FreezeCascade script_json
  the frozen ledger they actually need. Silent STRING to STRING
  semantic miswire pytest cannot catch but runtime JSON parsers
  reject. Rewired all three to FreezeCascade script_json.

  audio_gate decoupled from SignalLostVideo video_path which had
  forced FLUX planning to wait for 1080p procgen video completion.
  Now sources from FreezeCascade script_json a cheap completion
  signal per G1 operator decision.

  HuMo portraits_dir wired to FLUX portrait batch render output
  so HuMo consumes generated portraits as face reference per G1
  operator confirmation that portraits are intended content not
  just an execution gate.

  OTR_WorkflowValidator class confirmed OUTPUT_NODE True so the
  validator actually executes at runtime rather than sitting idle
  in the graph.
  ```
- **Exit condition:** 5 new pytest pass; default workflow loads in ComfyUI; audio C7 proxy byte-identical against v1.5 fixture; validator runs.

### D1a -- Catalog metadata schema extension

- **Goal:** Extend `CuratedModel` with 6 new fields. Backfill 6 existing rows. Append talkie row reading license fields from the D0b audit file. `prompt_profile="otr_1940s_v1"` makes target-era binding explicit.
- **Files touched:** `nodes/_otr_model_catalog.py`; `tests/test_model_catalog_schema.py` (new); `tests/test_catalog_matches_audit_files.py` (new); `tests/fixtures/workflow_schema_canonical.py` (inline note).
- **Code changes:**
  ```python
  @dataclass(frozen=True)
  class CuratedModel:
      repo_id: str
      requires_auth: bool
      loader_backend: Literal[
          "transformers_safetensors",
          "transformers_multimodal_text_only",
          "transformers_gptq_int4",
      ]
      vram_fit_tier: Literal["PASS", "WARN", "UNKNOWN", "FAIL"]
      approx_safetensors_gb: float
      notes: str = ""
      prompt_profile: Literal["modern", "otr_1940s_v1"] = "modern"
      chat_template_kind: Literal[
          "transformers_default", "manual", "raw_completion"
      ] = "transformers_default"
      stop_tokens: tuple[str, ...] = ()
      context_window: int = 8192
      license: Literal[
          "mit", "apache_2_0", "non_commercial", "community", "gated_terms"
      ] = "mit"
      license_audit_status: Literal[
          "mit_equivalent", "research_lane", "pending"
      ] = "pending"
  ```
  Talkie row:
  ```python
  CuratedModel(
      repo_id="talkie-lm/talkie-1930-13b-it",
      requires_auth=True,
      loader_backend="transformers_gptq_int4",
      vram_fit_tier="UNKNOWN",
      approx_safetensors_gb=7.5,
      notes=(
          "Period trained 13B at GPTQ int4. Training corpus pre 1930 "
          "may mismatch OTR_PERIOD_SYSTEM_PROMPT 1938-1952 target era. "
          "Modern news with post 1952 references produces era "
          "anachronistic dialogue. Research lane not eligible for "
          "default workflow JSON until license_audit_status flips "
          "to mit_equivalent."
      ),
      prompt_profile="otr_1940s_v1",
      chat_template_kind="transformers_default",
      stop_tokens=("</s>",),
      context_window=4096,
      license="non_commercial",
      license_audit_status="research_lane",
  ),
  ```
  Workflow fixture note:
  ```python
  # tests/fixtures/workflow_schema_canonical.py
  CANONICAL_PRESERVED_PLACEHOLDERS = ("", "[]", "{}")
  # These appear in widgets_values arrays of the default workflow
  # JSON. They are the BUG-LOCAL-032 preserved mode fix (commit
  # dabcebd 2026-04-14) for widget drift bug class BUG-LOCAL-027
  # 029 030 031. NOT placeholder data. Removing them re-introduces
  # widget drift.
  ```
- **Wire-in:** Forbidden-sweep allow-list adds `prompt_profile`, `otr_1940s_v1`, `transformers_gptq_int4`, `license_audit_status`.
- **Pytest to add:**
  - `test_curated_model_has_extended_fields`
  - `test_existing_rows_default_to_prompt_profile_modern`
  - `test_talkie_row_uses_otr_1940s_v1_profile`
  - `test_catalog_license_fields_match_audit_files_for_every_row`
  - `test_default_workflow_only_binds_mit_equivalent_rows_to_creative_slot`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D1a extend CuratedModel with 6 fields and add talkie row

  Adds prompt_profile chat_template_kind stop_tokens context_window
  license license_audit_status fields. Existing 6 rows backfilled.
  Talkie appended with prompt_profile otr_1940s_v1 explicit target
  era binding so the 1938-1952 OTR convention does not silently
  conflate with the pre 1930 training corpus. License fields
  mirror the D0b audit file. BUG-LOCAL-032 preserved placeholder
  note added to the workflow schema test fixture so future
  reviewers do not relitigate the empty string array fix.
  ```
- **Exit condition:** 5 new pytest pass; catalog matches audit files; default workflow still Mistral-Nemo only.

### D1b -- Loader-backend protocol, GPTQ adapter, context-limit helper

- **Goal:** Land duck-typed loader-backend protocol. Refactor 2 existing backends to fit (no behavior change). Scaffold `transformers_gptq_int4` adapter. Add `compute_effective_context_limit(row)` helper.
- **Files touched:** `nodes/_otr_loader_backends.py` (new); `nodes/_otr_model_runtime.py` (refactor to dispatch); `tests/test_loader_backend_protocol.py` (new); `tests/test_effective_context_limit.py` (new).
- **Code changes:**
  ```python
  from typing import Protocol

  HARD_VRAM_CONTEXT_LIMIT = 8192

  class LoaderBackend(Protocol):
      def load(self, repo_id: str, row) -> object: ...
      def generate(self, model, messages, **kwargs) -> str: ...
      def unload(self, model) -> None: ...

  def compute_effective_context_limit(row) -> int:
      return min(HARD_VRAM_CONTEXT_LIMIT, row.context_window)
  ```
  Existing 2 backends become thin wrappers conforming to the protocol; no body change. GPTQ int4 adapter scaffolds `AutoGPTQForCausalLM.from_quantized` without runtime execution. Dispatch table in `_otr_model_runtime.py` keys on `row.loader_backend`.
- **Wire-in:** Central dispatch. AST signature snapshot tests confirm existing public function signatures unchanged.
- **Pytest to add:**
  - `test_protocol_three_callables_present`
  - `test_existing_safetensors_backend_signatures_unchanged`
  - `test_existing_multimodal_text_only_backend_signatures_unchanged`
  - `test_gptq_int4_adapter_constructable`
  - `test_dispatch_table_routes_loader_backend_literal_to_correct_adapter`
  - `test_compute_effective_context_limit_clamps_to_hard_limit_when_window_larger`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D1b duck typed loader backend protocol GPTQ adapter and context limit helper

  Lands nodes/_otr_loader_backends.py with a duck typed protocol
  load generate unload. Existing transformers_safetensors and
  transformers_multimodal_text_only refactored to thin wrappers
  with no behavior change confirmed by AST signature snapshots.
  New transformers_gptq_int4 adapter scaffolded for talkie does
  not execute under non runtime pytest. Adds
  compute_effective_context_limit helper returning
  min HARD_VRAM_CONTEXT_LIMIT row.context_window for downstream
  prompt assembly cap.
  ```
- **Exit condition:** 6 new pytest pass; AST diff clean.

### D1c -- Loader smoke, tokenizer chat-template guard, dropdown coverage

- **Goal:** Runtime-gated loader smoke for talkie. Structural tokenizer chat-template guard. Structural effective-context-limit clamp test. Dropdown enum coverage.
- **Files touched:** `tests/test_talkie_loader_smoke_runtime.py` (new); `tests/test_tokenizer_chat_template_guard.py` (new); `tests/test_talkie_catalog_dropdown_surface.py` (new).
- **Code changes:** Runtime tests under `@pytest.mark.skipif(not OTR_REGRESSION_RUNTIME, reason="runtime gate")`. Tokenizer guard:
  ```python
  def test_talkie_tokenizer_has_chat_template_when_kind_is_transformers_default():
      row = lookup_curated_row("talkie-lm/talkie-1930-13b-it")
      tok = AutoTokenizer.from_pretrained(row.repo_id)
      if row.chat_template_kind == "transformers_default":
          assert tok.chat_template is not None, (
              f"{row.repo_id} declares chat_template_kind="
              f"transformers_default but tokenizer.chat_template is None. "
              f"Switch the row to chat_template_kind=manual."
          )
  ```
- **Wire-in:** Writer's `creative_writing_model` widget enum auto-rebuilds from `CURATED_LLM_MODELS`.
- **Pytest to add:**
  - `test_talkie_load_smoke_runtime_gated`
  - `test_talkie_unload_returns_vram_clean_runtime_gated`
  - `test_talkie_in_creative_writing_model_dropdown_enum`
  - `test_writer_widget_routes_talkie_to_gptq_int4_adapter`
  - `test_talkie_tokenizer_has_chat_template_when_kind_is_transformers_default`
  - `test_effective_context_limit_for_talkie_is_4096_not_8192`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D1c talkie loader smoke tokenizer chat template guard and dropdown coverage

  Two runtime gated tests verify load 1 token warmup unload cycle
  leaves VRAM clean. One structural tokenizer guard fails loud if
  the row declares chat_template_kind transformers_default but
  the tokenizer has no chat_template instructing the operator to
  switch the row to chat_template_kind manual. Two structural
  tests verify talkie dropdown enum presence and dispatch
  routing. One structural test verifies effective context limit
  for talkie clamps to 4096 not the 8192 hard limit. Hugging
  Face login required before runtime tests due to requires_auth
  True on the talkie row.
  ```
- **Exit condition:** 4 structural tests green; 2 runtime tests staged-skipped; G3 covers HF login before runtime execution.

### D2a -- Prompt-profile resolver helper

- **Goal:** Land `_otr_creative_prompt_router.py` resolver. Helper defined, not wired. Few-shot OMIT decision encoded as a zero-callers test.
- **Files touched:** `nodes/_otr_creative_prompt_router.py` (new); `tests/test_creative_prompt_router.py` (new).
- **Code changes:**
  ```python
  from typing import Literal
  from nodes._otr_model_catalog import lookup_curated_row
  from nodes._otr_outline import _SYSTEM_PROMPT as _MODERN_OUTLINE_PROMPT
  from nodes._otr_line_composer import (
      _SYSTEM_PROMPT as _MODERN_LC_SYSTEM,
      _POLISH_SYSTEM_PROMPT_CHARACTER as _MODERN_POLISH_CHAR,
      _POLISH_SYSTEM_PROMPT_ANNOUNCER as _MODERN_POLISH_ANN,
  )
  from nodes._otr_period_prompts import OTR_PERIOD_SYSTEM_PROMPT

  Phase = Literal[
      "outline",
      "line_composer_system",
      "polish_character",
      "polish_announcer",
  ]

  _MODERN_BY_PHASE = {
      "outline": _MODERN_OUTLINE_PROMPT,
      "line_composer_system": _MODERN_LC_SYSTEM,
      "polish_character": _MODERN_POLISH_CHAR,
      "polish_announcer": _MODERN_POLISH_ANN,
  }

  def resolve_creative_system_prompt(repo_id: str, phase: Phase) -> str:
      if phase not in _MODERN_BY_PHASE:
          raise ValueError(f"unknown phase {phase!r}")
      row = lookup_curated_row(repo_id)
      if row.prompt_profile == "otr_1940s_v1":
          return OTR_PERIOD_SYSTEM_PROMPT
      return _MODERN_BY_PHASE[phase]
  ```
- **Wire-in:** None. Caller-count enforced at 0.
- **Pytest to add:**
  - `test_router_returns_modern_for_default_mistral_nemo`
  - `test_router_returns_period_for_talkie_otr_1940s_v1`
  - `test_router_zero_production_callers_at_d2a_boundary`
  - `test_router_raises_on_unknown_phase`
  - `test_render_few_shot_block_has_zero_production_callers`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D2a creative prompt router resolver helper defined not wired

  Lands nodes/_otr_creative_prompt_router.py with
  resolve_creative_system_prompt repo_id phase returning the
  modern phase prompt by default or OTR_PERIOD_SYSTEM_PROMPT when
  the catalog row prompt_profile equals otr_1940s_v1. Caller
  count test asserts 0 production callers at this boundary. Few
  shot exemplar splice decision is OMIT for v1 encoded as a test
  asserting render_few_shot_block has 0 production callers
  saving about 600 tokens of context budget. Can reintroduce
  post v1.9 if quality requires.
  ```
- **Exit condition:** 5 new pytest pass; 0 production callers of both helpers.

### D2b -- Wire resolver to 4 sites and stamp creative meta

- **Goal:** Replace static system-prompt references at 4 phase sites with resolver calls. Writer stamps `meta.creative_model` and `meta.creative_prompt_profile` into script_json so FreezeCascade preserves creative slot identity (Sprint C wiring bottleneck #3 from review). Default config stays byte-stable.
- **Files touched:**
  - `nodes/_otr_outline.py` (1 site near current 411)
  - `nodes/_otr_line_composer.py` (3 sites near current 790, 1077, 1099)
  - `nodes/_otr_ledger_script_writer.py` (meta stamping)
  - `tests/test_creative_prompt_routing_wired.py` (new)
  - `tests/test_audio_c7_clamp_counter.py` (new)
  - `tests/test_writer_stamps_creative_meta.py` (new)
- **Code changes:** At each of the 4 sites:
  ```python
  system_prompt = resolve_creative_system_prompt(
      repo_id=self.creative_repo_id,
      phase="outline",  # or line_composer_system / polish_character / polish_announcer
  )
  ```
  Writer's script_json builder appends two meta keys at the same point it stamps `meta.story_brief`:
  ```python
  script_json["meta"]["creative_model"] = self.creative_repo_id
  script_json["meta"]["creative_prompt_profile"] = (
      lookup_curated_row(self.creative_repo_id).prompt_profile
  )
  ```
- **Wire-in:** Writer node passes `creative_repo_id` into outline and line_composer phase calls. No new log line; no existing log string modified. Two new `meta.*` keys are additive; audio path reads only `meta.story_brief` so byte identity holds.
- **Pytest to add:**
  - `test_router_has_exactly_4_production_callers_at_d2b_boundary`
  - `test_outline_phase_wired_to_router_by_name_inventory`
  - `test_line_composer_system_phase_wired_to_router_by_name_inventory`
  - `test_polish_character_phase_wired_to_router_by_name_inventory`
  - `test_polish_announcer_phase_wired_to_router_by_name_inventory`
  - `test_default_config_both_slots_mistral_nemo_returns_modern_prompts_byte_stable`
  - `test_audio_c7_proxy_holds_at_d2b_against_v1_5_fixture`
  - `test_clamp_counter_zero_during_audio_c7_proxy_run`
  - `test_writer_stamps_meta_creative_model_with_creative_repo_id`
  - `test_writer_stamps_meta_creative_prompt_profile_from_catalog_row`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; **audio C7 byte-identical proxy MUST hold -- this is the critical commit for Prime Directive 1**.
- **Commit message:**
  ```
  Sprint D D2b wire creative prompt router at 4 sites and stamp creative meta

  Replaces local _SYSTEM_PROMPT references at outline
  line_composer system polish_character polish_announcer with
  resolve_creative_system_prompt calls. Inventory test checks
  each phase name is wired not just total caller count.

  Writer stamps meta.creative_model and meta.creative_prompt_profile
  into script_json so FreezeCascade preserves creative slot
  identity. Sprint C closed with creative_writing_model unlinked
  from FreezeCascade so post freeze diagnostics were blind to
  which creative model produced the script. This stamping fixes
  that visibility gap without rewiring the workflow JSON.

  Default config both slots Mistral Nemo returns identical modern
  prompt strings preserving byte stable audio C7 proxy against
  v1.5 fixture. Clamp counter test using D0c log surface confirms
  reflection clamp did not fire during the audio C7 proxy run.
  Two new meta keys are additive and audio path reads only
  meta.story_brief so byte identity holds.
  ```
- **Exit condition:** All 10 new pytest pass; audio C7 byte-identical; clamp counter 0; phase-name inventory matches 4; meta stamping visible in script_json output. G4 before D2c.

### D2c -- Chat-template and stop-token dispatch

- **Goal:** Adapter dispatches on `chat_template_kind` and passes `stop_tokens` to generation.
- **Files touched:** `nodes/_otr_loader_backends.py` (extend with `_encode_messages` and stop-tokens passthrough); `tests/test_chat_template_kind_dispatch.py` (new).
- **Code changes:**
  ```python
  def _encode_messages(tokenizer, messages, row):
      kind = row.chat_template_kind
      if kind == "transformers_default":
          return tokenizer.apply_chat_template(messages, return_tensors="pt")
      if kind == "raw_completion":
          return tokenizer(
              "\n".join(m["content"] for m in messages),
              return_tensors="pt",
          )
      if kind == "manual":
          raise NotImplementedError(
              "manual chat_template_kind requires a row level template "
              "field not present in v1 schema. Deferred to D-future."
          )
      raise ValueError(f"unknown chat_template_kind {kind!r}")
  ```
  Stop tokens via `generate(..., stop_strings=list(row.stop_tokens))`.
- **Wire-in:** All adapter `generate` paths thread through `_encode_messages` and stop-tokens kwarg.
- **Pytest to add:**
  - `test_transformers_default_kind_uses_apply_chat_template`
  - `test_raw_completion_kind_concatenates_content_only`
  - `test_manual_kind_raises_not_implemented_with_clear_message`
  - `test_stop_tokens_threaded_to_generate_kwargs`
  - `test_dispatch_path_uses_compute_effective_context_limit`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D2c chat template kind dispatch and stop tokens passthrough

  Adapter encodes messages via apply_chat_template for
  transformers_default kind via concatenation for raw_completion
  and raises NotImplementedError with clear deferral message for
  manual. Stop tokens threaded to generate kwargs. Dispatch path
  uses compute_effective_context_limit helper from D1b to cap
  prompt budget at min HARD_VRAM_CONTEXT_LIMIT row.context_window.
  ```
- **Exit condition:** 5 new pytest pass; manual kind raises clearly.

### D3 -- Reflection boundary, sweep carve-outs, validator hard-fail

- **Goal:** Reframed period-prose reflection boundary. AST-node-scoped sweep carve-out plus `.py`-only file scope. Default-workflow validator hard-fails on non-`mit_equivalent` creative bindings. News_interpreter unrouted test.
- **Files touched:**
  - `tests/test_period_prose_reflection_boundary.py` (new)
  - `scripts/_schema_sweep.py` (AST-node carve-out + `.py` filter)
  - `tools/audit_workflow_schema.py` (hard-fail validator)
  - `tests/test_default_workflow_validator.py` (new)
  - `tests/test_news_interpreter_unrouted.py` (new)
  - `tests/test_forbidden_sweep_scope.py` (new)
- **Code changes:** Reflection boundary:
  ```python
  def test_reflection_boundary_no_system_prompt_literal_bleed_when_lines_are_period():
      led = make_led_with_period_diction_lines()
      brief = run_reflection(led, technical_model=DEFAULT_LLM)
      assert "1940s tube radio" not in brief["text"]
      assert "Suspense, Lights Out, Inner Sanctum" not in brief["text"]
      assert "[SOUND:" not in brief["text"]

  def test_reflection_brief_records_technical_slot_model():
      brief = run_reflection(led, technical_model=DEFAULT_LLM)
      assert brief["story_brief_model"] == DEFAULT_LLM

  PERIOD_SLANG_NEGATIVE_LOOKAHEAD = re.compile(
      r"\b(swell|fellows)\b|"
      r"^say,\s|"
      r"\ball right\b(?=[^.]*\b(fine|swell|grand)\b)"
  )
  def test_reflection_brief_no_period_slang_bleed():
      brief = run_reflection(led, technical_model=DEFAULT_LLM)
      assert not PERIOD_SLANG_NEGATIVE_LOOKAHEAD.search(brief["text"])
  ```
  AST-scoped carve-out:
  ```python
  CARVEOUT_CONSTANTS = {
      "nodes/_otr_period_prompts.py": (
          "OTR_PERIOD_SYSTEM_PROMPT",
          "PERIOD_EXEMPLARS",
      ),
  }
  TEXT_FILE_EXTENSIONS = (".py",)
  def _is_in_carveout(file_path, ast_node):
      names = CARVEOUT_CONSTANTS.get(file_path, ())
      if not names:
          return False
      target = _enclosing_assign_target_name(ast_node)
      return target in names
  ```
  Workflow validator hard-fail:
  ```python
  def check_default_workflow_creative_binding(workflow_json, catalog):
      for node in _writer_nodes(workflow_json):
          row = catalog.lookup(node.creative_writing_model)
          if row.license_audit_status != "mit_equivalent":
              raise WorkflowSchemaError(
                  f"default workflow binds {row.repo_id} to creative "
                  f"slot but license_audit_status is "
                  f"{row.license_audit_status} not mit_equivalent"
              )
          if row.prompt_profile != "modern":
              raise WorkflowSchemaError(
                  f"default workflow binds {row.repo_id} to creative "
                  f"slot but prompt_profile is {row.prompt_profile} "
                  f"not modern"
              )
  ```
- **Wire-in:** Sweep AST analysis with carve-out walking Assign target names. Workflow validator runs on default shipped workflow in the test suite.
- **Pytest to add:**
  - `test_reflection_boundary_no_system_prompt_literal_bleed_when_lines_are_period`
  - `test_reflection_brief_records_technical_slot_model_in_story_brief_model_key`
  - `test_reflection_brief_no_period_slang_bleed`
  - `test_forbidden_sweep_carveout_only_covers_named_constants_not_whole_file`
  - `test_forbidden_sweep_catches_planted_era_literal_in_a_different_constant_in_period_prompts_file`
  - `test_forbidden_sweep_ignores_non_py_extensions`
  - `test_default_workflow_validator_hard_fails_on_non_mit_equivalent_creative_binding`
  - `test_news_interpreter_prompts_come_from_technical_slot_never_routed_through_creative_resolver`
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0 runtime hits including the negative-control planted literal; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D3 reframed reflection boundary tightened sweep carve outs and validator hard fail

  Reflection boundary test reframed to assert no system prompt
  literal bleed plus story_brief_model equals technical slot
  model plus a negative lookahead regex on specific period slang.
  Replaces the brittle zero period diction token framing which
  would false fail when a modern brief legitimately mentions
  period setting like 1940s diner.

  Forbidden sweep carve out is AST node scoped to the
  OTR_PERIOD_SYSTEM_PROMPT constant and the PERIOD_EXEMPLARS list
  literal only not the whole _otr_period_prompts.py file. Sweep
  also restricted to .py extensions so .safetensors .wav .pt
  fixture directories do not false positive on the 0x97 byte.
  Negative control plants an era literal in a different constant
  in the same file and verifies the sweep catches it.

  Workflow validator hard fails when any default shipped workflow
  binds a row to the creative slot with license_audit_status not
  equal to mit_equivalent or prompt_profile not equal to modern.
  This blocks research lane models from accidentally shipping
  bound to the default config.

  News interpreter test asserts its prompts come from the
  technical slot and are never routed through the creative
  resolver so modern news grounding stays unaffected by period
  selection on the creative slot.
  ```
- **Exit condition:** 8 new pytest pass including the negative control.

### D4 -- Runtime VRAM, determinism xfail, no-news diction, context precondition

- **Goal:** Stage 3 runtime-gated tests. Land 2 structural context-window precondition tests.
- **Files touched:** `tests/test_period_creative_runtime.py` (new); `tests/test_context_window_precondition.py` (new); `nodes/_otr_loader_backends.py` (precondition in adapter `load`).
- **Code changes:**
  ```python
  def _check_context_window(row):
      if row.context_window < HARD_VRAM_CONTEXT_LIMIT:
          raise RuntimeError(
              f"context_window {row.context_window} for {row.repo_id} "
              f"is below HARD_VRAM_CONTEXT_LIMIT {HARD_VRAM_CONTEXT_LIMIT}. "
              f"Pick a larger window variant or land a compact mode "
              f"binding in D-future."
          )
  ```
  Runtime tests:
  ```python
  @pytest.mark.skipif(not OTR_REGRESSION_RUNTIME, reason="runtime gate")
  def test_period_creative_modern_technical_vram_peak_under_14_5gb():
      run_full_pipeline(creative="talkie-lm/talkie-1930-13b-it",
                       technical=DEFAULT_LLM)
      peak_allocated = torch.cuda.max_memory_allocated() / 1e9
      peak_reserved = torch.cuda.max_memory_reserved() / 1e9
      assert peak_allocated < 14.5
      assert peak_reserved < 14.5

  @pytest.mark.xfail(strict=False, reason="GPTQ split K nondeterminism")
  @pytest.mark.skipif(not OTR_REGRESSION_RUNTIME, reason="runtime gate")
  def test_period_creative_stable_across_two_runs_advisory():
      out_a = run_fixed_seed_period_creative(seed=42)
      out_b = run_fixed_seed_period_creative(seed=42)
      assert out_a == out_b

  @pytest.mark.skipif(not OTR_REGRESSION_RUNTIME, reason="runtime gate")
  def test_period_creative_diction_guard_no_modernisms_on_no_news_fixture():
      out = run_period_creative_against_fixture(
          "tests/fixtures/no_news_period_seed_v1.json"
      )
      for tok in (r"\bokay\b", r"\bguys\b", r"\bcool\b",
                  r"\bhey\b", r"\bsmartphone\b", r"\binternet\b"):
          assert not re.search(tok, out, re.IGNORECASE)

  @pytest.mark.skipif(not OTR_REGRESSION_RUNTIME, reason="runtime gate")
  def test_period_creative_modern_news_warning_emit():
      out = run_period_creative_against_fixture(
          "tests/fixtures/modern_news_seed_v1.json"
      )
      anachronisms = _find_post_1952_tokens(out)
      if anachronisms:
          warnings.warn(
              f"period creative produced post 1952 tokens "
              f"{anachronisms} when given modern news input",
              UserWarning,
          )
  ```
- **Wire-in:** Each adapter `load` invokes `_check_context_window(row)`. Talkie at 4096 trips this by design (G5 operator decision).
- **Pytest to add:**
  - `test_period_creative_modern_technical_vram_peak_under_14_5gb` (runtime, allocated + reserved)
  - `test_period_creative_stable_across_two_runs_advisory` (runtime, xfail strict=False)
  - `test_period_creative_diction_guard_no_modernisms_on_no_news_fixture` (runtime)
  - `test_period_creative_modern_news_warning_emit` (runtime, warning-only)
  - `test_context_window_precondition_below_hard_limit_raises` (structural)
  - `test_context_window_precondition_at_or_above_hard_limit_passes` (structural)
- **Regression gates:** Bug Bible 23/1/2; forbidden-sweep 0; audio C7 byte-identical proxy.
- **Commit message:**
  ```
  Sprint D D4 runtime gated period creative tests and context window precondition

  VRAM peak test asserts both max_memory_allocated and
  max_memory_reserved stay under 14.5 GB during talkie creative
  plus Mistral Nemo technical slot swap catching caching
  allocator bloat the allocated only check would miss.

  Determinism test is xfail strict False rather than advisory only
  because GPTQ split K nondeterminism is known cannot become a
  false sprint blocker.

  Diction guard runs against a fixed no news fixture so the test
  exercises the diction layer in isolation. A separate news
  warning test runs the period creative path against modern news
  and emits a UserWarning if post 1952 tokens appear documenting
  the known limitation without failing.

  Context window precondition raises clear error at adapter load
  when row.context_window is below HARD_VRAM_CONTEXT_LIMIT 8192.
  Talkie at 4096 trips this by design research lane disposition
  G5 operator decision covers whether to relax or accept.
  ```
- **Exit condition:** 2 structural tests green; 4 runtime tests staged-skipped.

### D-final -- Sprint close

- **Goal:** Archive plan. Rename workflow JSON FreezeCascade title `Phase 0..10` -> `Phase 0.10`. Sprint A backlog drops SA-101. Prepare v1.9 tag.
- **Files touched:** `docs/closed-sprints/2026-05-16-sprint-d-period-llm.md` (final state); `SPRINT.md` (reset); `docs/sprint-a-backlog.md` (drop SA-101); `workflows/otr_scifi_16gb_full.json` (title rename).
- **Code changes:** Around the FreezeCascade node title near line 1938, rename `Phase 0..10` to `Phase 0.10`.
- **Wire-in:** None.
- **Pytest to add:**
  - `test_freeze_cascade_node_title_is_phase_0_10_not_double_dot`
- **Regression gates:** Final Bug Bible 23/1/2; final forbidden-sweep 0; final audio C7 byte-identical proxy against v1.5 fixture.
- **Commit message:**
  ```
  Sprint D close archive plan rename FreezeCascade title and update Sprint A backlog

  Archives the Sprint D plan to docs/closed-sprints and resets
  SPRINT.md. Renames the workflow JSON FreezeCascade node title
  from Phase 0..10 to Phase 0.10 fixing the stale double dot
  label that drifts grep and runtime screenshot review. Sprint A
  backlog row SA-101 silent reflection clamp log is removed
  citing the D0c carryover commit hash. Sprint A retains SA-100
  SA-102 SA-103 plus the audio C7 baseline reset captures
  originally scoped. v1.9 tag prepared on this commit.
  ```
- **Exit condition:** All gates green; v1.9 tag ready.

---

## §3 Sprint C gotcha appendix

Future-self pre-mortem. Minor and medium only; major wiring carryovers landed in D0d and D2b.

| # | Severity | File:line | Gotcha | Mitigation |
|--:|---|---|---|---|
| 1 | **MAJOR** | `workflows/otr_scifi_16gb_full.json` HuMo / VideoComposite / LTX `ledger_json` inputs | Three downstream consumer `ledger_json` inputs were sourced from `SignalLostVideo.video_path` (video file path string), not `FreezeCascade.script_json`. Silent STRING-to-STRING semantic miswire pytest cannot catch but runtime JSON parsers reject. | **Pulled forward into Sprint D D0d**; all three rewired to `FreezeCascade.script_json` with structural invariant test. |
| 2 | MEDIUM | `workflows/otr_scifi_16gb_full.json` `VideoPlan.audio_gate` input | `audio_gate` was sourced from `SignalLostVideo.video_path`, forcing FLUX planning to wait on 1080p procgen video completion. | **Resolved in Sprint D D0d**; decoupled to a cheap completion signal from `FreezeCascade.script_json` (G1 confirms source). |
| 3 | MEDIUM | `workflows/otr_scifi_16gb_full.json` HuMo `portraits_dir` | Unlinked; portrait render only acted as execution gate not content. | **Resolved in Sprint D D0d** (G1 confirms intent: portraits feed); wired to FLUX portrait render output. |
| 4 | MEDIUM | `workflows/otr_scifi_16gb_full.json` FreezeCascade -- creative model | Creative model identity was not carried into FreezeCascade; only technical model. Post-freeze diagnostics blind to which creative slot produced the script. | **Resolved in Sprint D D2b** via `meta.creative_model` and `meta.creative_prompt_profile` stamping into script_json before freeze. Cleaner than rewiring `creative_writing_model` into FreezeCascade because the writer is the source of truth. |
| 5 | MEDIUM | `nodes/_otr_workflow_validator.py` | `OTR_WorkflowValidator` was unconnected in the graph; ComfyUI does not execute a node unless `OUTPUT_NODE=True` or wired into a downstream gate. May have been sitting idle. | **Resolved in Sprint D D0d** by confirming `OUTPUT_NODE=True` on the class. |
| 6 | MEDIUM | `nodes/_otr_story_brief.py:487-494` | Silent temperature clamp on reflection exception arm emitted no log. | **Pulled forward into Sprint D D0c**; `log.info` with `str(e)` interpolated. D2b clamp counter catches drift at default config. |
| 7 | MEDIUM | `tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum` | b3sum cross-hardware floating-point drift may not byte-match across sm_120 driver or precision mode. | Sprint A SA-102 captures `hardware_snapshot.json`; same-hardware time-axis drift bounded by snapshot pinning. |
| 8 | MEDIUM | `nodes/_otr_model_runtime.py` slot-swap path | Implicit single-slot VRAM eviction; GPTQ int4 has a different allocator profile. | Sprint D D4 (a) asserts both allocated and reserved peaks under 14.5 GB; Sprint A SA-103 layers per-cycle `torch.cuda.memory_summary()`. |
| 9 | MEDIUM | downstream consumers `OTR_SignalLostVideo`, `OTR_BatchHumoRender` | Widget null-state parsing for BUG-LOCAL-032 placeholders could raise `IndexError` if consumers index without checking. | Sprint A precondition: audit each consumer's widget-array indexing path for empty-tolerant access. Not on Sprint D path. |
| 10 | LOW | `nodes/story_orchestrator.py` `_STYLE_WORLD_BLOCK` | Dormant orphan logic retained after C2b skipped cleanup as moot. | Deferred to Sprint G broad cleanup sweep. |
| 11 | LOW | `tests/test_b7_forbidden_sweep.py` | Em-dash byte 0x97 sweep would false-positive on `.safetensors`, `.wav`, `.pt` binaries if extension scope is wrong. | **Resolved in Sprint D D3** via explicit `.py`-only filter. |
| 12 | LOW | `workflows/otr_scifi_16gb_full.json:56-75` | Preserved-mode `""`, `'[]'`, `'{}'` may be misread by future reviewers as bad empty data. | **Resolved in Sprint D D1a** via inline note in workflow-schema test fixture. |
| 13 | LOW | `nodes/_otr_period_prompts.py:37-84` | `OTR_PERIOD_SYSTEM_PROMPT` body contains intentional era literals. | **Resolved in Sprint D D3** via AST-node-scoped carve-out. |
| 14 | MEDIUM | `nodes/_otr_model_catalog.py` talkie row | `requires_auth=True` HF gated repo; runtime smoke needs HF login. | Operator gate G3 before D1c runtime. |
| 15 | LOW | GPTQ int4 quantization | Talkie output not byte-deterministic at fixed seed due to split-K kernel nondeterminism. | D4 (b) is `@pytest.mark.xfail(strict=False)`; documented in commit body and row notes. |
| 16 | LOW | talkie row notes | Era mismatch: training pre-1930 vs system prompt 1938-1952. | Research-lane caveat in row notes; explicit `prompt_profile="otr_1940s_v1"` makes target-era binding visible. |
| 17 | LOW | `news_interpreter` (technical slot) | Modern news with post-1952 references produces era-anachronistic period dialogue downstream. | Research-lane caveat; D4 news-warning advisory surfaces anachronisms without failing. |

---

## §4 Operator gates

Mandatory pauses. Cowork halts and surfaces the gate question before opening the next commit.

- **G1 -- Before D0d opens.** Operator confirms (a) **HuMo `portraits_dir` intent**: consume portraits (wire FLUX portrait render output) or gate-only (revert that hunk of D0d); (b) **`audio_gate` replacement source**: `FreezeCascade.script_json` (default recommendation: cheap and already in scope) or `EpisodeAssembler.completion_token` if that node exists with a suitable output.

- **G2 -- After D0b, before D1a opens.** Operator reviews `docs/model-licenses/talkie-lm--talkie-1930-13b-it.md` and confirms `research_lane` (proceed) or `mit_equivalent` (revisit catalog defaults). Spot-check 2 existing audit files (Mistral-Nemo `mit`, Gemma-4 `gated_terms`).

- **G3 -- After D1b, before D1c runtime tests run.** HF login confirmed. Operator runs `hf auth login` and verifies `HF_TOKEN` in shell environment. Structural pytest does NOT require this.

- **G4 -- After D2b, before D2c opens.** Audio C7 byte-identical pytest proxy MUST be green at default config AND clamp counter MUST be 0. This is the critical commit for Prime Directive 1. Likely failure modes: resolver emits a new log line (fix: confirm no logs); Python string-interning breaks byte identity (fix: resolver returns the literal `_SYSTEM_PROMPT` object directly); meta-stamping changes a hash the audio path consumes (fix: confirm audio path reads only `meta.story_brief`).

- **G5 -- After D4, before D-final.** Operator decides whether talkie's `context_window=4096` precondition error is acceptable for v1.9 ship (research-lane catalog-visible but load-blocked until larger-window variant ships or D-future compact-mode binding lands) or whether to raise the row to 8192 with documented quality risk. Default: keep strict.

- **G6 -- Before D-final commit.** Operator confirms v1.9 release-tag readiness: period catalog rows present, none default-bound; Sprint A backlog row SA-101 deleted citing D0c commit hash; v1.5 audio fixture pair still byte-matched; FreezeCascade title rename present.

---

## §5 Final plan delta vs v1 packet

| Item | v1 packet | v3 plan | Why |
|---|---|---|---|
| Total commits | 12 | 13 | New D0d for MAJOR Sprint C miswire fix and three medium workflow rewires |
| Total day estimate | 6.25 d | 6.5 d | At cap; D0d adds 0.5 d, D0c shrunk from 0.5 d to 0.25 d |
| **Three `ledger_json` inputs sourced from `SignalLostVideo.video_path`** | Not addressed | **D0d rewires HuMo / VideoComposite / LTX `ledger_json` to `FreezeCascade.script_json`** | MAJOR Sprint C semantic miswire; pytest blind; runtime JSON parsers reject the video path string |
| **`VideoPlan.audio_gate` sourced from procgen video completion** | Not addressed | **D0d decouples to `FreezeCascade.script_json` cheap completion signal** | Eliminates FLUX planning wait on 1080p procgen video |
| **HuMo `portraits_dir` unlinked** | Not addressed | **D0d wires to FLUX portrait render output (G1 confirms intent)** | Portraits become content not just execution gate |
| **`OTR_WorkflowValidator` `OUTPUT_NODE` status** | Not addressed | **D0d confirms `OUTPUT_NODE=True`** | Validator actually executes rather than sitting idle |
| **Creative model carried into FreezeCascade** | Not addressed | **Already resolved in D2b** via `meta.creative_model` and `meta.creative_prompt_profile` stamping | Writer is source of truth; meta stamping is cleaner than rewiring `creative_writing_model` into FreezeCascade |
| `prompt_profile` value name | `"period_v1"` | `"otr_1940s_v1"` | Explicit target-era binding; decouples 1938-1952 OTR convention from talkie's pre-1930 training corpus |
| D0b audit ordering | "6 rows plus talkie" implied talkie exists at D0b | Audit operates on `AUDIT_TARGETS.txt`; D1a reads pre-existing files | Removed circular dependency |
| Sprint C SA-101 silent clamp log | Queued for Sprint A | Pulled forward into D0c with `str(e)` interpolation | Major Sprint C bug; captures exception context before clamp masks it; protects D2b audio C7 proxy |
| D2b caller-count assertion | `caller_count == 4` exact | Phase-name inventory: each of 4 phases independently wired | Catches silent coverage reduction if a phase is removed and a different one is added |
| D2c scope | Tokenizer + stop-tokens + few-shot decision | Tokenizer + stop-tokens only | Few-shot OMIT in D2a as zero-callers test; D2c single-concern |
| D2c context-budget test | None | `compute_effective_context_limit` helper at D1b + dispatch test at D2c | Clamps prompt budget at `min(HARD_VRAM_CONTEXT_LIMIT, row.context_window)` |
| D3 reflection boundary framing | "zero period-diction tokens in story_brief" | No system-prompt literal bleed AND `meta.story_brief_model == technical_slot_model_id` AND negative-lookahead on period slang | Original framing would false-fail on legitimate brief mentioning "1940s diner" as setting |
| D3 sweep carve-out scope | Line ranges in `_otr_period_prompts.py` | AST-node scoped to `OTR_PERIOD_SYSTEM_PROMPT` and `PERIOD_EXEMPLARS` only; planted-literal negative control | Line-range scope could mask new era literals inside the carved range |
| D3 sweep file scope | All files | `.py` text extension only | Prevents 0x97 byte false-positives on `.safetensors`, `.wav`, `.pt` |
| D3 workflow validator | Warning on `prompt_profile != "modern"` | Hard fail on `license_audit_status != "mit_equivalent"` OR `prompt_profile != "modern"` creative binding | Stronger guarantee research-lane models cannot ship default-bound |
| D3 news_interpreter test | None | Asserts news_interpreter prompts come from technical slot never routed through creative resolver | Closes modern-news-in-period-frame ambiguity at the test layer |
| D4 VRAM assertion | `memory_allocated` peak only | Both `memory_allocated` AND `memory_reserved` peaks under 14.5 GB | Catches caching allocator bloat |
| D4 determinism advisory | "Advisory" comment | `@pytest.mark.xfail(strict=False)` | Cannot become false sprint blocker |
| D4 diction guard | Single test over generated dialogue | Split: no-news fixture test (fail) plus modern-news warning (advisory) | Diction layer tested in isolation; modern-news limitation surfaced as warning |
| D1c tokenizer guard | None | Structural test fails loud if `tokenizer.chat_template is None` while row declares `chat_template_kind="transformers_default"` | Catches schema/tokenizer mismatch before runtime |
| D-final FreezeCascade title | Unchanged | Renamed `Phase 0..10` to `Phase 0.10` | Fixes stale double-dot label drift in grep and screenshots |
| D1a workflow fixture note | None | Inline note naming `""`, `'[]'`, `'{}'` as BUG-LOCAL-032 preserved-mode placeholders | Prevents future reviewers from relitigating the canonical-shape fix |
| Sprint G `_STYLE_WORLD_BLOCK` cleanup | Out of scope | Out of scope | Confirmed deferred |
| Manual `chat_template_kind` | Implicit | Explicit `NotImplementedError` with clear deferral message in D2c | No row currently needs manual templates |
| Audio C7 v1.5 fixture | Holds through Sprint D | Holds through Sprint D asserted at every commit including D2b critical boundary | Prime Directive 1 unchanged |

---

**End of Sprint D Cowork action plan v3 -- 2026-05-16.**

---

## §D-final shipped state -- 2026-05-16

Sprint D closed at commit `<this-commit>` on branch `sprint-d-period-llm`.
12 substantive commits + this D-final close. All gates green at every
commit boundary. Audio C7 byte-identical pytest proxy HELD against
`tests/fixtures/baseline_v1.5.wav` end-to-end. Bug Bible regression
23/1/2 held end-to-end. Forbidden-pattern sweep 0 runtime hits at every
commit boundary.

### Commit chain as actually shipped

| # | Commit | Subject |
|---|---|---|
| D0a | `f7dfe5d` | branch cut + plan landing |
| D0b | `92a69eb` | license audit framework + 7 flat audit files |
| D0c | `d00450f` | SA-101 silent clamp log carryover |
| D0d | `19a99d2` | workflow rewires + portraits_dir output + validator activation |
| D1a | `78adf58` | CuratedModel extended (6 new fields) + talkie row |
| D1b | `87d7b93` | loader-backend protocol + GPTQ scaffold + context helper |
| D1c | `56dc6d9` | talkie dropdown + dispatch + chat-template guard |
| D2a | `747a376` | creative prompt router (defined, not wired) |
| D2b | `0e6d50b` | wire resolver into 4 phase sites + writer meta stamping |
| D2c | `bc11788` | chat-template + stop-token dispatch helpers |
| D3 | `2cf2333` | reflection boundary + sweep scope + validator + news_interpreter |
| D4 | `c876714` | runtime-gated period creative + context-window precondition |
| D-final | `<this-commit>` | sprint close + workflow title rename + Sprint G handoff |

### Deviations from v3 plan -- documented

Five deviations from the v3 plan as written. All endorsed by the
operator at the corresponding gates during execution. Recorded
here so future maintainers see immediately what shipped vs what
the planning artifact says.

1. **D0b flat docs structure.** Per operator G1 Q1 directive, the
   plan's `docs/model-licenses/` subfolder was flattened into 7
   files at `docs/model-license-<sanitized>.md` plus
   `docs/model-license-audit-targets.txt` in the docs root.
   Sanitization: lowercase repo_id, replace `/` with `--`, strip
   nothing else. Framework adapted to read the flat layout.

2. **D0c triage-doc canonical log line.** The v3 plan D0c snippet
   showed `str(e)` interpolation inside `_repair_pass`, but `e` is
   not in scope inside that function. The shipped log line is the
   canonical SA-101 patch from
   `docs/retrospectives/2026-05-15-sprint-c-triage-findings.md`
   which is in-scope and mathematically complete:
   `log.info("[OTR_StoryBrief] repair pass clamped: base=%.3f ...")`.
   Same effect (Sprint A inspectors can distinguish a
   0.55-ceilinged retry from a 0.35 retry from a pre-clamp
   failure), correct semantics. v3 snippet was illustrative not
   literal.

3. **D0d scope expansion.** The v3 plan called D0d "JSON edits
   only." The HuMo `portraits_dir` wire could not land cleanly
   because `BatchFluxPortraitRender` shipped with
   `RETURN_TYPES = ("IMAGE", "STRING")` -- no `portraits_dir`
   output existed. Per operator G1 (a) "wire it" decision, D0d
   added a 3rd STRING output to BatchFluxPortraitRender (small
   source change: extended RETURN_TYPES + RETURN_NAMES, hoisted
   portraits_dir computation from line 386 to line 308 so both
   early-return and normal-return surface the directory string,
   updated both return sites). 13/13 Sprint C portrait tests
   stayed green; 130/130 wider workflow tests stayed green.

4. **D3 forbidden-sweep carve-out simplified.** The v3 plan
   called for an explicit AST-node-scoped carve-out for
   `OTR_PERIOD_SYSTEM_PROMPT` + `PERIOD_EXEMPLARS`. Discovery
   during D3 execution: the existing tokenize-based suppression
   in `docs/_s28_forbidden_sweep.py` already covers period
   prompts (triple-quoted string body classified as "string" ->
   forensic-suppressed). The `.py`-only file extension scope is
   already enforced by `tests/test_b7_forbidden_sweep.py`'s git
   diff `-- "*.py"` filter. D3 shipped PIN tests for both
   properties rather than re-architecting the sweep.

5. **D4 runtime tests use double-skip.** The 4 D4 runtime-gated
   tests use `@pytest.mark.skipif(not OTR_REGRESSION_RUNTIME)`
   PLUS an internal `pytest.skip(...)` because the writer
   runtime harness for end-to-end fixture-based generation does
   not yet exist (Sprint A precondition). The test surfaces ship
   now so Sprint A unblocks them by wiring the harness, not by
   writing new tests from scratch.

### Sprint A backlog adjustment

`docs/sprint-a-backlog.md` does not exist at Sprint D close. The
v3 plan D-final "drop SA-101" instruction is captured here
forensically: **SA-101 is subsumed by D0c commit `d00450f`** and
should be omitted from any future Sprint A backlog that lands.
SA-100 (schema-positive gate), SA-102 (hardware snapshot), SA-103
(VRAM telemetry) remain Sprint A's scope per the triage doc.

### Sprint G handoff

Added per operator B + 1 tweak at D-final.

**`_LegacyTransformersBackendBase` orphan-candidate review (Sprint G).**
The base class at `nodes/_otr_model_runtime.py:34-58` is currently
the shared implementation for `TransformersSafetensorsBackend` and
`TransformersMultimodalTextOnlyBackend`, both of which delegate
to the legacy monolithic `nodes/_otr_model_loader.load_llm`. If a
future sprint splits `load_llm` per-backend (separating the
multimodal-text-only branch from the safetensors branch into their
own concrete implementations), `_LegacyTransformersBackendBase`
becomes an unnecessary intermediate -- the two concrete adapters
would each carry their own `load`/`generate`/`unload`. Sprint G's
orphan-constant sweep should include this case-by-case judgment
call after Sprint A's empirical pass validates the per-backend
split is needed.

### Sprint D test counts

Approximately 70 new active pytest tests across the 12 substantive
commits plus 1 at D-final, totaling ~71 new active tests.
Approximately 7 runtime-staged-skip tests waiting for Sprint A
unblock (D1c: 3 HF-required tokenizer/loader smokes; D4: 4
fixture-harness-required period-creative pipeline tests).

### v1.9 tag readiness

Sprint D is ready for v1.9 release-tag preparation. Per
CLAUDE.md "Only Jeffrey merges to main and tags releases" -- the
operator runs `git tag` directly. Suggested label per the v3 plan
intent: `v1.9.0-rc1` or `v1.9.0` at operator discretion.
Memorial Day weekend 2026-05-24/25 target preserved (Sprint D
closed 2026-05-16 with comfortable slack).

---

**End of Sprint D shipped-state record -- 2026-05-16.**
