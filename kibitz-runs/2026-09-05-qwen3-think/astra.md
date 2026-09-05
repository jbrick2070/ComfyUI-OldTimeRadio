**REFUTED as the cause of this measured failure. The GGUF-only suppression gap is confirmed—but Gemma, not Qwen, generated the failed dossier.**

1. **The production artifact identifies the failing model.**

   [tmp/otr_headless_62693.log:358](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tmp/otr_headless_62693.log:358) records:

   > `creative_model='Qwen/Qwen3-8B', technical_model='google/gemma-2-2b-it'`

   Immediately after dossier attempt 1, [line 363](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tmp/otr_headless_62693.log:363) says:

   > `[Selector] slot=technical reuse cache for google/gemma-2-2b-it`

   Lines 376 and 392 repeat that selection for attempts 2 and 3. [Line 370](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tmp/otr_headless_62693.log:370) reports a “92-token run verbatim 3 times”; [line 403](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tmp/otr_headless_62693.log:403) captures repeated:

   > `The storms in the Pacific were Lowell, Karina, and Marie.`

   This is Gemma dossier repetition. Qwen suppression cannot fix this particular failure.

2. **Dossier is unconstrained labelled text, then schema-validated.**

   [_otr_scifi_news_pro.py:1878](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_news_pro.py:1878) supplies `schema=DossierLLM` and `slot_fn=technical_fn`; [line 1892](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_scifi_news_pro.py:1892) supplies:

   > `text_parser=parse_dossier_sections`

   [_otr_structured_call.py:920](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py:920) skips JSON-contract injection when that parser exists. [Line 954](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_structured_call.py:954) passes `force_json_object=text_parser is None`, therefore **False**. No `response_format` or schema-bound decoder is requested.

   The scheduler builds `_build_truncating_generate_fn` at [OTR_LedgerScriptWriter.py:714](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:714). Its [line 1123](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:1123) contains the exact reported “in-decode liveness guard” exception. **This is the writer’s plain closure, not `_otr_constrained_generate.py`.**

3. **Your suppression finding is correct. There is no active transformers suppression mechanism to name.**

   [_otr_gguf_backend.py:1502](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_gguf_backend.py:1502) applies `_apply_qwen3_no_think` under `qwen3_no_think`. Conversely, the writer at [line 869](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py:869) and loader at [_otr_model_loader.py:1899](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_model_loader.py:1899) call:

   > `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`

   Neither supplies `enable_thinking=False`. The installed [Qwen tokenizer_config.json:230](C:/ComfyUI-Models/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/tokenizer_config.json:230) explicitly supports:

   > `if enable_thinking is defined and enable_thinking is false`

   Also, Leg 0’s acceptance is merely `out["ok"] = bool(text)` at [otr_llm_preflight_leg0.py:175](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_llm_preflight_leg0.py:175). A nonempty reasoning fragment passes that transport check.

4. **Fix shape: one shared owner of model-row generation policy.**

   Proposed sole owner: `nodes/_otr_shared/llm_think_policy.py`. It owns exact-model-ID policy assignments, template controls, the single `/no_think` literal, and envelope handling. Catalog/GGUF/cache fields project that policy; transports consume it. Include discovered model IDs, not just curated rows.

   For HF Qwen3-8B, select native **`enable_thinking=False` before tokenization and budget measurement**. The installed template supports it, and [Qwen documents it as the hard switch](https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode).

   Wire the shared policy into **every actual transformers entry point**: writer closure, loader generation, polish, and constrained generation. **Patching only `make_generate_fn` fixes the probe while missing canonical dossier generation.**

   Move GGUF’s existing policy implementation into that owner without changing its behavior. No second writer-local Qwen rule.

5. **Preservation and acceptance.**

   GGUF suppression applies to **both** structured and plain calls; only stripping is conditional. [_otr_gguf_backend.py:1584](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_gguf_backend.py:1584) says:

   > `if rf is None and _think_policy == "qwen3_no_think":`

   Preserve that distinction, including equivalent protection for HF schema-bound generation. Preserve GGUF whitespace-stop filtering, later literal `<think>` text, empty-answer failure, request bytes and existing hash baselines. [test_gguf_registry.py:414](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_gguf_registry.py:414) explicitly asserts:

   > `assert out == "<think>x</think>Y"  # rf present -> no strip`

   Leave `eng_indextts2.py`, `_otr_indextts2_worker.py`, and `_otr_resolved_request.py` byte-identical. Default policy remains identity; no family-wide Qwen matching or stray directives.

**Keep Qwen pending proper qualification.** This leg never established creative-slot failure. Correct the contrary claim at [GO_FORWARD_PLAN.md:537](C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/GO_FORWARD_PLAN.md:537), then qualify Qwen creative with a proven technical partner and test Qwen technical separately. Non-thinking mode makes it a legitimate candidate; only canonical story completion and publication establish that it works.

No files changed or generation runs performed.