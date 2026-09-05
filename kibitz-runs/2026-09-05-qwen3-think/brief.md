# Qwen/Qwen3-8B fails the scifi dossier pass. Confirm or refute my diagnosis, then give me the fix shape.

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
READ-ONLY. Change nothing. Cite `file:line` and quote what you actually read.
A grep hit is a lead, not a finding.

## THE MEASURED FAILURE (2026-09-05, a real canonical leg, not a theory)

A canonical episode leg with the writer's `creative_writing_model` pinned to
`Qwen/Qwen3-8B` and `source_bank=scifi_news_pro` died in 1.8 minutes:

```
NewsProDossierError: [scifi_news_pro] pass 'dossier' failed after 3 attempt(s):
generation was halted by the in-decode liveness guard: the output repeated a run
of tokens verbatim, which is a decode that is cycling rather than a long
artifact (no fallback to legacy_many_pass)
```

The same row PASSES a 40-token transport probe: it loads, generates and unloads
in both slots (`scripts/otr_llm_preflight_leg0.py`, measured minutes earlier).
Its probe output was 179 characters and began:
`<think> Okay, the user is asking for one thing a radio needs to work. Let me think. A radio requires a power s`

## MY DIAGNOSIS -- attack it

The Qwen3 reasoning suppression exists in this repo but is **GGUF-ONLY**, and
`Qwen/Qwen3-8B` is a **transformers-lane** row, so it never receives it:

* `nodes/_otr_gguf_backend.py:832` `_QWEN3_NO_THINK_DIRECTIVE = "/no_think"`,
  applied by `_apply_qwen3_no_think` (`:837`) and stripped by
  `_strip_leading_think_envelope` (`:809`).
* The ONLY call sites are `_otr_gguf_backend.py:1503` and `:1585`.
  (`grep -rn "_apply_qwen3_no_think\|_strip_leading_think_envelope" nodes/`
  returns nothing outside that one file.)
* `think_policy="qwen3_no_think"` is a field on `GGUFRow` (`:226`, set at `:376`
  for `unsloth/Qwen3-8B-GGUF`). There is no equivalent notion on the
  transformers lane.
* `nodes/_otr_shared/llm_policy.lane_for_row("Qwen/Qwen3-8B")` returns
  `transformers` (measured).
* The comment at `_otr_gguf_backend.py:826-831` says exactly why it exists:
  *"Without it Qwen3 opens a long `<think>` reasoning block, so a short-output
  call ... exhausts its whole budget mid-thought and never emits the answer."*

So on the transformers lane Qwen3-8B reasons instead of answering, and on the
structured multipass dossier pass the decode cycles until the liveness guard in
`nodes/OTR_LedgerScriptWriter.py:1123` (and/or
`nodes/_otr_constrained_generate.py:346`) halts it.

## WHAT I NEED FROM YOU

1. **CONFIRM or REFUTE** the above against the real files. If you refute it,
   name the mechanism that actually applies think-suppression on the
   transformers lane, with `file:line`.
2. **Is the dossier pass CONSTRAINED generation** (a JSON/response_format
   schema) or free prose? Read `nodes/_otr_scifi_news_pro.py` for the `dossier`
   pass and follow which generate path it takes
   (`_otr_constrained_generate.py` vs the writer's plain generate closure).
   This matters: the GGUF policy deliberately does NOT strip the think envelope
   when a `response_format` is in force (`_otr_gguf_backend.py:190`), so the fix
   must respect the same rule.
3. **THE FIX SHAPE.** Where does the think policy belong so BOTH lanes honour
   it -- on the row (a catalog field both lanes read), in
   `make_generate_fn`/the transformers transport
   (`nodes/_otr_model_loader.py:1833`), or at the writer? Name the ONE owner.
   The pack's rule is one owner per machine fact; a second copy of `/no_think`
   living in a different file is the failure mode to avoid.
4. **What must NOT break:** the GGUF lane's existing behaviour is byte-hashed in
   places and already correct; the three byte-hashed files
   (`nodes/_otr_audio_engines/eng_indextts2.py`,
   `scripts/_otr_indextts2_worker.py`, `nodes/_otr_resolved_request.py`) must
   not change; and no other row's prompts may gain a stray `/no_think`.
5. Is `Qwen/Qwen3-8B` worth keeping at all, or is the honest answer that a
   reasoning model does not belong in a structured multipass writer slot?
   The operator's instruction is: **"if it can make a story, add it"** -- so the
   question is whether the fix makes it write, not whether it is elegant.

Answer in markdown, terse, evidence first. Lead with CONFIRMED or REFUTED.
