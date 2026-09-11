VERDICT: build-ready as-is? no.
The plan lacks an executable, non-contradictory design for its primary blocker (F1/F2 semantic fidelity), misdiagnoses the Mac 19 GB OS kill as an inactive LMFE cache defect (M1) while leaving Mac testing deadlocked, and burdens the critical path with non-blocking logging cleanups (O1-O3).

MUST-FIX BEFORE BUILD:
1. [## Proposed order §3 / ## Critical constraints §3, §4 / ## Review questions]
Defect: The semantic preservation mechanism for F1/F2 is a design vacuum wrapped in contradictory negative constraints. Critical constraints §3 and §4 forbid both model-extracted fact lists and subjective LLM judges, while ruling out literal quote/byte-hash matching as inadequate for semantic entailment. The plan promises to "Close F1/F2 across actual authoring owners" in step 3, but review questions (lines 113-115) admit that no design exists for evaluating whether P0, P1, P2, or visual prompts contradict natural prose listener facts without using model judgments.
Fix: Define a two-tier fact contract before code begins:
  (a) At admission in `_otr_story_input.py`, compile an immutable listener fact struct directly from `StoryInputBundle` capturing explicit entities, stated living/relationship status, and required co-presence settings.
  (b) Thread this struct directly into P1 (`_pass_treatment`), P2 (`_pass_act`), and MetaBrief (`_build_char_scene_request`) prompt templates so generation is conditioned on source facts rather than derived summaries.
  (c) Add a targeted, fail-closed contradiction validator to `_make_treatment_validator` and `_make_act_validator` inside the existing `structured_call` post-validator ladder. The validator checks only explicit negations against the admission struct (e.g. living entity marked deceased/absent, required co-present entity isolated), raising `PostValidationError` to engage the existing `_full_artifact_repair`. [ASSUMPTION: Bounded contradiction checks against an explicit fact struct can fit within existing slot budgets without introducing a secondary retry engine.]

2. [## Grounded triage M1, D2 / ## Evidence and boundaries / ## Proposed order §1, §6]
Defect: Triage M1 claims that LMFE token cache accumulation in `nodes/_otr_constrained_generate.py:123-142` is relevant to the Mac OS process kill at 19,163 MB (`MAC_LESSONS_LEARNED.md:337`). However, inspection of `nodes/_otr_my_story.py` (lines 421, 470, 560, 598) and `nodes/_otr_ledger_clean.py` (lines 614, 906, 1228, 1459) shows that neither module calls `_otr_bind_schema` or `make_constrained_generate_fn`; both execute unbound closures. M1 was completely inert during the Mac trial. The Mac process kill was driven by resident PyTorch MPS memory/KV-cache growth across dozens of sequential unconstrained model calls (`MAC_LESSONS_LEARNED.md:341-347`). Fixing M1 will not prevent Qwen3.5-4B from being killed on a 16 GB Mac. Furthermore, D2 leaves the Mac deadlocked: Qwen OOMs, Llama-3.2 cannot download due to the 5 GB margin on 8.5 GB free disk, and cache cleanup is banned.
Fix:
  (a) Decouple M1 (a real latent bug in `_otr_constrained_generate.py`) from the Mac OS memory kill.
  (b) To address actual Mac memory pressure, insert explicit PyTorch MPS cache reclamation (`torch.mps.empty_cache()`) in `OTR_LedgerScriptWriter.py` between major passes and after `run_ledger_clean`. [ASSUMPTION: Reclaiming MPS cached allocations between passes prevents resident process growth beyond the 16 GB unified memory limit.]
  (c) Resolve the D2 storage deadlock operationally: either grant explicit operator authorization to prune ephemeral test artifacts in `Documents/` to yield >12 GB free space for `Llama-3.2-3B-Instruct`, or test a resident 4-bit quantized GGUF model via `gguf_native`.

3. [## Critical constraints §8, §9 / ## Proposed order §1 / ## Grounded triage S1]
Defect: Connecting `_otr_bind_schema` to P1 without fixing LMFE lifecycle isolation will cause immediate generation failures or cross-attempt token corruption. In `nodes/_otr_constrained_generate.py:133-140`, `(parser, prefix_fn)` is cached globally by schema on `cache_entry["_otr_lmfe_constraint_cache"]`. LMFE's `build_transformers_prefix_allowed_tokens_fn` encapsulates a `TokenEnforcer` instance that mutates internal `prefix_states` during generation. Reusing `prefix_fn` across calls or retries contaminates the token prefix tree. Additionally, LMFE's default `max_json_array_length=20` and 2048 open-string decode guard risk truncating treatment acts and cast lists.
Fix: In `_otr_constrained_generate.py`:
  (a) Keep only the expensive `tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)` cached on `cache_entry`.
  (b) Instantiate a fresh `JsonSchemaParser` and `prefix_allowed_tokens_fn` per generation call so `TokenEnforcer` state is discarded after each attempt.
  (c) Pass `max_json_array_length=0` to disable implicit array caps, and size the open-string decode guard to accommodate full treatment output room.

4. [## Critical constraints §5 / ## Grounded triage C1 / ## Proposed order §2]
Defect: The proposed fix for C1 ("validated complaint coordinates and the existing authorized transaction") cannot protect short lines or valid codas. In `nodes/_otr_ledger_clean.py:1806-1900`, `_repair_row` passes the entire line to `_call_repair` to rewrite the whole string. On short lines like the Step 120 coda ("Until next time,"), the judge flags the entire line. With no uncomplained coordinates present, the repair model generates an arbitrary substitute ("That's a wrap"), which `_repair_row` accepts because it contains no stage directions.
Fix: In `nodes/_otr_ledger_clean.py`:
  (a) In `_repair_row`, check if the line text is an exact match for an authored frame asset (`frame.coda`, `frame.announcer_outro`) or raw user input. If matched and the line contains no bracketed/parenthetical stage directions or speaker prefixes, suppress whole-line repair and retain original text.
  (b) For multi-segment lines, require that `_repair_row` validates that candidate text retains uncomplained word spans above a minimum token overlap threshold (e.g. >= 80% word retention of unflagged text) before accepting the edit.

5. [## Grounded triage F2 / ## Critical constraints §6 / ## Proposed order §3]
Defect: F2 (visual scenes inverting shared presence) cannot be resolved inside `_compose_char_scene_prompt` without modifying the underlying request builder. In `nodes/otr_meta_brief_image_prompt.py:1614-1641`, `_build_char_scene_request` is hardcoded to mandate a solo portrait: "The image MUST show the CHARACTER THEMSELVES ... as the subject ... character_appearance: {appearance}". It passes no information about co-present characters. Downstream image generation faithfully generates isolated single-person stills, inverting shared scenes like a family dinner into solitary isolation.
Fix: In `nodes/otr_meta_brief_image_prompt.py`:
  (a) Extend `_build_char_scene_request` to extract co-present cast members for the scene/beat from the ledger.
  (b) When co-present characters are specified, relax the solo-character framing instruction to require a two-shot or ensemble composition showing the characters together in `story_setting`.

SHOULD-FIX:
1. [## Grounded triage S2 / ## Proposed order §1]
Defect: In `nodes/_otr_my_story.py:433, 484`, `_pass_interpret` and `_pass_treatment` render `"music between acts: %s"` directly from `include_act_breaks`, telling the LLM `"yes"` even when `act_count=1`.
Fix: In `run_my_story_episode`, calculate `interstitial_count = max(0, act_count - 1) if include_act_breaks else 0` and pass `"yes" if interstitial_count > 0 else "no"` (and the exact count) to P0 and P1.

2. [## Grounded triage C2 / ## Evidence and boundaries line 58]
Defect: C2 ("Five dirty rows and six edits") is carried as an unverified defect. Code inspection of `nodes/_otr_ledger_clean.py:1690-1708` confirms it is an accounting asymmetry: `receipt["judged_dirty"]` increments only for model-judged rows, while `receipt["pattern_only"]` increments for pattern matches. Both invoke `_repair_row`, so total edits (`repaired + improved`) can exceed `judged_dirty`.
Fix: Close C2 as a non-defect and update `_otr_ledger_clean.py` receipt logging to report `total_flagged_rows = judged_dirty + pattern_only`.

3. [## Proposed order §6 / ## Evidence and boundaries lines 10-24]
Defect: Restricting 4060 testing strictly to 1 act while running 6 acts on 5080 leaves multi-act interstitial logic and scene progression unverified on intermediate configurations.
Fix: Mandate at least one 3-act canonical run on 5080 or RunPod before concluding the qualification campaign, confirming that `interstitial_count=2` functions end-to-end without music-parent warnings or leaked interstitial cues.

OPTIONAL / NICE-TO-HAVE:
1. [## Proposed order §4 / ## Grounded triage O1, O2, O3]
Observability items O1 (relabeling pre-image still warning), O2 (scoping LTX-open health checks), and O3 (downgrading same-file rename logging) are non-failing diagnostic messages. Move them to a separate maintenance PR after canonical qualification.

CUT THESE (scope / over-engineering):
1. [## Proposed order §4 / ## Grounded triage O1, O2, O3]
Safe to cut because they never caused a pipeline stop or corrupted an artifact (`docs/2026-09-10-my-story-4060/README.md:137-153`). Editing `nodes/_otr_video_engines/render_driver.py` (7,016 lines) for diagnostic log text during this sprint introduces unnecessary regression risk.
2. [## Grounded triage C2]
Safe to cut because it is verified from code as an expected telemetry divergence between `judged_dirty` and `pattern_only`, not an unowned edit.
3. [## Proposed order §6 line 107 ("repeat on a second compatible installed model family")]
Safe to cut from this defect-closure sprint. Multi-model stress qualification is explicitly defined in `docs/GO_FORWARD_PLAN.md:216` as post-A1R coverage.
