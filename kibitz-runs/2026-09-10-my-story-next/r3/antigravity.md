# R3 Adversarial Review: Wiring, Integration & Sequencing (Sprints A, B, C)

VERDICT: build-ready as-is? no. Critical interface mismatches in frame/cue sequencing, attribution receipt stamping, act-number shot ID collisions, and validator gates must be closed before build.

---

## MUST-FIX BEFORE BUILD

### 1. [Sprint A1: Frame Assembly & Interstitial Cue Ordering / Invariant Mismatch]
- **Defect:** 
  In [`nodes/_otr_my_story.py:1118`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L1118), `inter_wanted` is computed as `max(0, act_count - 1)` using the initial requested `act_count`, NOT the actual number of treatment acts `len(treatment.acts) - 1`. When a treatment authors 2 acts for requested 1 (the exact live failure in prompt `e11e5d86`), `inter_wanted = 0`, causing `_pass_frame` to generate 0 interstitial cues.
  Subsequently, in [`_assemble`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L851-L910) (lines 897–909):
  1. `_assemble` does not take `include_act_breaks` as an argument and unconditionally loops through `len(acts) - 1`. If `include_act_breaks` was False in graph controls, interstitials are still emitted if cues exist.
  2. If `include_act_breaks` is True and `len(frame.music_inter) < len(acts) - 1`, indexing or iterating drops the boundary cues or raises an out-of-bounds error.
  3. If `len(frame.music_inter) > len(acts) - 1`, surplus cues are silently dropped without disclosing their disposition in metadata.
- **Concrete Fix:**
  1. In `nodes/_otr_my_story.py:1118`, compute `inter_wanted = max(0, len(treatment.acts) - 1)` before invoking `_pass_frame`.
  2. Update `_assemble(..., include_act_breaks: bool)` to accept the widget boolean.
  3. In `_assemble`:
     - If `include_act_breaks` is False: emit zero interstitial cues. Record all `frame.music_inter` proposals as `"unused_act_breaks_disabled"` in `meta.my_story.music_cue_disposition`.
     - If `include_act_breaks` is True: for index `i` in `range(len(acts) - 1)`, if `i < len(frame.music_inter)`, emit `frame.music_inter[i]`; otherwise emit an empty string cue `""` for downstream composer synthesis (`StableAudioTheme`).
     - Record any surplus cues (`i >= len(acts) - 1`) as `"surplus_boundary_dropped"` in `meta.my_story.music_cue_disposition`.

### 2. [Sprint A1: Attribution Injection vs Frame Serialization & Receipt Verification Race]
- **Defect:**
  In [`nodes/_otr_my_story.py:1120-1122`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L1120-L1122), `story["frame"] = frame.model_dump(mode="json")` is serialized immediately after `_pass_frame`.
  In [`_assemble`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L946-L954), `stamp_receipt` records `accepted_artifacts["frame"] = story["frame"]` and hashes the artifact dictionary.
  If the mandatory deterministic `attribution_sentence` is injected into the announcer outro *during* assembly or after `story["frame"]` has been dumped, the spoken line text in `lines` will contain the attribution sentence while `accepted_artifacts["frame"]["announcer_outro"]` does NOT. Downstream freeze verification ([`nodes/_otr_content_authorship.py:validate_receipt`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_content_authorship.py#L270-L310)) validates exact correspondence between accepted artifacts and line text. This mismatch will fail freeze verification with a `needs-rerun` verdict or corrupted receipt exception.
- **Concrete Fix:**
  Enforce strict chronological sequencing in `run_my_story_episode`:
  1. Receive validated `frame = _pass_frame(...)`.
  2. Check if the attribution sentence exists in `frame.announcer_intro` or `frame.announcer_outro`. If missing, append `attribution_sentence` directly to `frame.announcer_outro`.
  3. Strip whitespace from frame text fields; if any optional field is empty, leave it empty (do not create empty spoken rows).
  4. ONLY THEN serialize `story["frame"] = frame.model_dump(mode="json")`.
  5. Pass the mutated `frame` into `_assemble(...)` and downstream receipt generators.

### 3. [Sprint A1: Act Numbering Normalization Order and Shot ID Collision]
- **Defect:**
  In [`_assemble:871-872`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L871-L872):
  `scene_id = "s%02d" % act.n`
  `shot_id = "shot_%03d" % act.n`
  And at line 911 (postamble/outro):
  `post_shot = "shot_%03d" % (len(acts) + 1)`
  If the LLM returns acts with non-sequential or offset numbers (e.g. Model authors Acts 2 and 3 for a 2-act story), Act 3 generates `shot_003`. But `post_shot` for `len(acts) == 2` is `shot_%03d % (2 + 1) = shot_003`. This causes a direct Shot ID collision between the final act and the outro shot, corrupting shot ordering, beat scheduling, and line foreign keys.
  Furthermore, in [`_make_act_validator:604`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L604):
  `if int(model.n) != n:` raises `ValueError`, which actively rejects any act whose authored number doesn't match the prompt slot before normalization can occur.
- **Concrete Fix:**
  1. Remove `if int(model.n) != n:` from `_make_act_validator`.
  2. Immediately upon accepting the treatment, store raw model act numbers in `meta.my_story.act_number_normalization` and normalize the treatment's act slot indices to `1..N`.
  3. In the act generation loop for slot index `i` (from `1..N`), enforce `act.n = i` on the accepted Pydantic model before passing it to `_assemble`. This guarantees `act.n` is strictly sequential `1..N`, ensuring `shot_%03d` and `post_shot = shot_%03d % (len(acts) + 1)` never collide.

### 4. [Sprint A1: Rejection Gates Remaining in Validators Violate Operator Contract]
- **Defect:**
  Multiple legacy rejection gates remain active in `nodes/_otr_my_story.py` that will abort valid generations on cosmetic or planning differences:
  - [`_make_treatment_validator:536-538`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L536-L538): Rejects if `len(names) != planned` (`"characters count mismatch"`).
  - [`_make_treatment_validator:547-550`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L547-L550): Rejects if `c.gender not in ("male", "female")`.
  - [`_make_treatment_validator:551-554`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L551-L554): Rejects if `c.gender != want`.
  - [`_make_treatment_validator:555-557`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L555-L557): Rejects if `len(acts) != act_count`.
  - [`_make_treatment_validator:559-560`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L559-L560): Rejects if `act.n != i`.
  - [`_make_act_validator:608-611`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L608-L611): Rejects if `line.speaker not in allowed` without handling case/whitespace variations.
  - [`_make_act_validator:615-616`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L615-L616): Rejects if `len(heard) < 2` for non-solo casts, forbidding monologues.
  - [`_make_frame_validator:690-693`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L690-L693): Rejects blank music descriptions.
  - [`_make_frame_validator:700-702`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_my_story.py#L700-L702): Rejects if `len(model.music_inter) != inter_wanted`.
- **Concrete Fix:**
  Prune all artificial gates in `_make_treatment_validator`, `_make_act_validator`, and `_make_frame_validator`:
  - Retain only: nonempty cast names, unique cast names, and `len(acts) >= 1`.
  - In `_make_act_validator`, canonicalize speaker names by casefold/trim matching against accepted cast names before membership validation (`speaker_map = {c.strip().casefold(): c for c in allowed}`). Remove the `len(heard) < 2` monologue check.
  - In `_make_frame_validator`, allow blank music strings and remove the exact interstitial count check.

### 5. [Sprint B: OpenRouter Sidecar Leakage Across Retries & Cloud Runaway Lifecycle Invariant]
- **Defect:**
  In [`nodes/_otr_openrouter_backend.py:1379-1400`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_openrouter_backend.py#L1379-L1400), model-gone fallback updates `payload["model"] = fb_clean` inside `_post_with_retries`.
  If a shared dictionary `receipt_out` is passed into `generate` -> `_post_with_retries` -> `_extract_text:1409`, `_extract_text` parses JSON and populates `receipt_out["executed"]` and `receipt_out["reported"]`.
  However, in [`make_openrouter_generate_fn`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_openrouter_backend.py#L1440-L1485), `_extract_text` executes *before* `assert_no_verbatim_cycle(text)` (line 1481). If the cloud runaway repetition check trips and raises an exception:
  1. The generation call has failed.
  2. If `receipt_out` was mutated in-place or attached to the closure before this check, the caller or scheduler could log a failed or aborted call as successful.
  3. If retried across multiple attempts within `_post_with_retries`, stale metadata from prior failed attempts could contaminate `receipt_out`.
- **Concrete Fix:**
  In `make_openrouter_generate_fn`:
  1. Allocate a fresh invocation-local dictionary `call_receipt = {}` inside the wrapper `_call(messages, **kwargs)`.
  2. Pass `receipt_out=call_receipt` to `backend.generate(...)`. Clear `receipt_out.clear()` at the start of each HTTP retry attempt inside `_post_with_retries`.
  3. In `make_openrouter_generate_fn`, only attach the receipt to the closure (`_call._last_receipt = dict(call_receipt)`) AFTER `text = backend.generate(...)` returns AND `assert_no_verbatim_cycle(text)` completes successfully.
  4. Wrap in `try...except`: if an exception occurs anywhere during generation, validation, or runaway checks, set `_call._last_receipt = None` so `_SlotScheduler` never records a failed attempt in `successful_model_calls`.

### 6. [Sprint B: Telemetry Stamping Sequencing & Bank ID Mutation Invariant]
- **Defect:**
  In [`nodes/_otr_writer_tail.py:_run_writer_tail:1525-1530`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_writer_tail.py#L1525-L1530), `_stamp_final_slot_telemetry(led.data)` executes right before `tail_finalizer.before_save` and `script_json = json.dumps(led.data)`.
  If `meta["model_call_provenance"]` and `meta["credits_source_line"]` are stamped out of order or after `tail_finalizer`, finalizer schema verification will fail or strip unwhitelisted fields.
  Furthermore, the bank ID in `banks.json` is `"original"`, NOT `"original_radio"`.
  If `_stamp_final_slot_telemetry` uses `meta.setdefault("credits_source_line", ...)`, it will fail to overwrite the early generic machine credit placed during template initialization. Conversely, if it unconditionally overwrites `credits_source_line` for all banks, it will destroy human bylines in My Story and true author credits in adaptation banks.
- **Concrete Fix:**
  In `_stamp_final_slot_telemetry` (called strictly before `tail_finalizer.before_save`):
  1. Stamp `meta["model_call_provenance"] = {"version": 1, "calls": self.successful_model_calls, "generation_models": gen_models, "finishing_models": fin_models, "unclassified_helpers": unclass_helpers}`.
  2. Check `if meta.get("source_bank") == "original":`. Only when this condition is True, unconditionally overwrite `meta["credits_source_line"] = f"Story generation models used: {', '.join(gen_models)}"`.
  3. Ensure helper classification strictly maps scheduler helper names: creative = (`build_news_briefs`, `lock_cast`, `generate_outline`, `compose_line`, `generate_title`, `compose_news_coda`, `compose_announcer_intro`, `compose_announcer_outro`, `announcer_intro_rewrite`); finishing = (`ledger_clean`, `ledger_cleanup`, `cast_coverage_repair`). All other helpers map to `unclassified_helpers`.

### 7. [Sprint C: Hero Title Token Splitting & Vertical Advance Mismatch in `_flow_col1`]
- **Defect:**
  In [`nodes/otr_credits_roll.py:_flow_col1:943-950`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_credits_roll.py#L943-L950), when `_autoshrink_pt` reaches `_PT_HERO_MIN`, the code issues a single `d.text((x, y), hero, ...)` and increments `y += _fh(fh)`.
  There are two fatal layout bugs:
  1. If wrapping is introduced using the existing [`_wrap:697-708`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_credits_roll.py#L697-L708), `_wrap` appends any single token that exceeds `max_w` onto the line without splitting (`if _fw(...) <= max_w or not cur: cur = t`). An unbroken word or token longer than Column 1 width will spill directly across into Column 2 (`CAST & VOICES`).
  2. When an overlong title is wrapped onto multiple lines, advancing `y` only once by `_fh(fh)` causes subsequent metadata (subtitle, tagline, date, metadata rows) to render directly on top of lines 2+ of the wrapped hero title.
  3. If `_scratch_draw` (used by `_draw_col1` to determine spacing tiers and trigger `_draw_col1_abridged`) uses a different wrapping or line-advance calculation than the real draw pass, measurement and drawing will diverge, corrupting vertical budget checks.
- **Concrete Fix:**
  In `nodes/otr_credits_roll.py:_flow_col1`:
  1. When `hero_pt == _sc(_PT_HERO_MIN, h)` and measured title width exceeds `_COL1_W * sx`: wrap lines by measuring tokens against `_COL1_W * sx`. If an individual token exceeds `_COL1_W * sx`, split the token character-by-character so no line exceeds the column boundary.
  2. For every line of the wrapped title emitted, render the text (when not in scratch mode) and advance `y += _fh(fh)` per line.
  3. Ensure `_flow_col1` is the single shared function executed by both `_scratch_draw` and real paint passes so that total measured height is identical across measurement, abridgment, and rendering.

---

## SHOULD-FIX

### 1. [Sprint A2: Nested `text_config` & Canonical Cache Root Order in Catalog]
- **Defect:**
  [`nodes/_otr_model_catalog.py:_read_advertised_context:701-710`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_catalog.py#L701-L710) reads context limit fields only from the root of `config.json`. Modern composite and multimodal architectures (such as Qwen2.5/Qwen3.5) nest sequence length fields inside `text_config`.
  In [`_hf_hub_root:658`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_catalog.py#L658), cache discovery should strictly respect `HF_HUB_CACHE` before `HF_HOME`.
- **Concrete Fix:**
  In `_read_advertised_context`, inspect `config.get("text_config", {})` before falling back to root keys. In `_hf_hub_root`, check `os.environ.get("HF_HUB_CACHE")` before `os.environ.get("HF_HOME")`.

### 2. [Sprint A2: Remove Architecture Mutation in Model Loader]
- **Defect:**
  [`nodes/_otr_model_loader.py:1229`](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_loader.py#L1229) mutates `model_config.max_position_embeddings = _cap` in place.
  Mutating the loaded Hugging Face model configuration object corrupts cached config metadata across subsequent model reloads or inspection queries in the same process.
- **Concrete Fix:**
  Delete line 1229 (`model_config.max_position_embeddings = _cap`). Pass `_cap` strictly as a generation budget parameter without mutating the underlying model architecture config.

### 3. [Sprint B: Explicit Unreported Identity Mapping for Provenance Stamping]
- **Defect:**
  If a backend returns an empty string or None for `executed_model`, naive joining of `generation_models` can produce `"Story generation models used: "` or omit entries, violating provenance transparency.
- **Concrete Fix:**
  In `_SlotScheduler`, if `call["reported"]` is False or `call["executed"]` is empty/None, map the recorded identity to `"model identity unreported"` before deduplication and joining.

---

## OPTIONAL / NICE-TO-HAVE

1. **Authorship Receipt Unit Test:** Add an explicit test in `test_otr_content_authorship.py` verifying that injecting `attribution_sentence` into `frame` before `story["frame"]` serialization validates successfully under `validate_receipt`, whereas mutating `lines` post-serialization fails.
2. **OpenRouter Fallback Telemetry Log:** Emit a one-line `logging.info` in `_SlotScheduler` when `call["executed"] != call["requested"]`, indicating that model fallback occurred during generation.

---

## CUT THESE (over-engineering)

1. **Replay System & Metadata Reconstruction:** The operator ruling (2026-09-10) explicitly prohibited replay-system work, saved-episode reconstruction, and old bundle migration. Fresh canonical generation only.
2. **Generic Title Card Refactoring & Font Resolver Overhaul:** Do not build a general-purpose font layout engine or refactor title cards across all nodes. The defect is strictly contained within `otr_credits_roll._flow_col1`.
3. **Local Schema Binder / Grammar Enforcement Engine:** The installed LMFE parser enforces an unconfigurable 20-element list ceiling (`default20` in `consts.py`), which would truncate dialogue lines. Do not integrate the local schema binder in this sprint.
4. **VRAM Estimator Overhaul:** Native HF context is an upper capability limit, not a static allocation instruction. Do not rewrite VRAM estimation math or introduce speculative minimum window rejections.
