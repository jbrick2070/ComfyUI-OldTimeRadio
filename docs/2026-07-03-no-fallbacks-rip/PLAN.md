# No-Fallbacks Rip — stack-wide "if it fails, it fails hard"

**Operator directive (2026-07-03):** no fallbacks for ANY model — all LLMs, all
video, all image, all audio, all TTS. A model failure is a hard stop (fail loud),
never a silent swap to another model or a degraded output.

Grounded inventory: three read-only fan-out audits (audio/tts, video/image,
llm/cloud), 2026-07-03. Cloud voice (S1 @ 925438e2) + cloud music (S5 @ c7da53b1)
were BUILT no-fallback from the start — this rip is about the LOCAL model lanes.

---

## A. RIP — true model fallbacks (silent swap / soft-fail that hides a failure)

### Audio / TTS
1. **Bark missing-ref net** — `nodes/_otr_voice_node_common.py:27-44, 462-552`.
   A cloning engine (indextts2/chatterbox/dia) with no usable voice_ref renders
   that line on **bark** instead. RIP → raise `EngineUnusable(MISSING_MODEL/REF)`.
   Drop `missing_ref_fallback` from adapter metadata + the whole `_bark_fb` branch.
2. **`_resolve_clone_ref_path` gender/any-ref best-effort** — `_otr_voice_node_common.py:76-135`.
   Never raises; picks ANY ref as last resort. RIP → raise when no matching ref.
3. **`_resolve_character_voices_fail_soft`** — `nodes/cast_lock.py:387-513`.
   Never raises; orphan lines fall to "node-81 engine fallback" (:511). RIP →
   fail loud on an unvoiceable character line.
4. **`_fallback_voice_identity`** — `cast_lock.py:351-369`. Deterministic
   `v2/en_speaker_N` synthesis for a missing preset. RIP → raise.
5. **Kokoro voice-id swap** — `nodes/_otr_audio_engines/eng_kokoro.py:158-174`.
   Missing `.pt` → swap to the seeded episode voice. RIP → raise `EngineUnusable`.
6. **Stage-direction silence** — `_otr_voice_node_common.py:430-442`. Empty
   prepared text → 0.3s silence, skip engine. JUDGMENT (see C): a beat with no
   dialogue is arguably legit, but it IS a silent substitution.

### Image
7. **Per-role slot → other_beats fallback** — `nodes/otr_image_gen_dispatcher.py:158-159`.
   Empty named slot silently uses the global other_beats image model. RIP → raise
   on an unresolved explicitly-named slot.
8. **Scene-still-missing soft-degrade** — `nodes/_otr_video_engines/render_driver.py:1025-1029`.
   image_to_video/static_motion with no scene still → "pre-spine init" fallback.
   RIP → hard raise (the rest of render_driver is already no-fallback).

### LLM / Writer
9. **Voice-preset healthcheck swap + pool exhaustion** — `OTR_LedgerScriptWriter.py:682-759`.
   Disabled preset → same-gender sibling; exhaustion logs, no raise. RIP → raise.
10. **body-score-never-fails** — `OTR_LedgerScriptWriter.py:1603-1659`. Every
    feature error → score 0, biasing the reroll decision silently. RIP → raise
    (or at minimum log ERROR + surface) so a scoring break can't ship unnoticed.
11. **Contract / pitch / grammar soft-fails** — `OTR_LedgerScriptWriter.py:3169,
    3209, 3409, 3513, 3666, 3715, 3900, 3932, 4210, 4264, 4308`. Bare
    `except: continue  # never break the writer`. RIP → raise ValidationError.
12. **News degrade** — `story_orchestrator.py:2856-2915`. Retry budget exhausted
    → `meta["news"]=None`. RIP → raise when news is required (toggle-gated).
13. **Title / announcer-outro / news-coda template fallbacks** —
    `story_orchestrator.py:4100-4182, 4765, 4924-4928`. LLM fail → deterministic
    template. RIP → raise (these are model-output fallbacks).
14. **Character portrait 3-tier fallback** — `story_orchestrator.py:5393`. RIP →
    raise when all tiers exhausted.
15. **OpenRouter model-gone remote fallback** — `_otr_openrouter_backend.py:1045-1060`.
    404 → one remote fallback slug. JUDGMENT (see C): already one-shot + loud, but
    it IS a model swap.

---

## B. KEEP — NOT model fallbacks (ripping these breaks correctness, not a swap)

- **INPUT_TYPES / `build_engine_combo` / `load_resolver` C-5 safety** —
  `_otr_voice_node_common.py:162-180`, `_otr_engine_profiles.py:342-354`. A widget
  list must NEVER crash or ComfyUI can't load the node pack. Dispatch path already
  fail-loud via `require_resolver()`. KEEP (add a debug log only).
- **Transient network retry ladders** — `_otr_openrouter_backend.py:1037-1143`,
  `_otr_ollama_backend.py:239-266`. Retrying the SAME model on 429/503 is not a
  fallback. KEEP (already fail-closed after budget).
- **Teardown `except: pass`** — `_otr_voice_node_common.py:587-603` etc. Cleanup
  must not mask the render result. KEEP.
- **`empty_audio_batch` for a zero-dialogue scene** — `_otr_resolved_request.py:158`.
  No model involved; silence is the correct output for a scene with no lines. KEEP
  (make the log LOUD).
- **Engine import-time `except: pass`** — `_otr_video_engines/__init__.py`,
  `_otr_audio_engines/__init__.py`. A missing optional dep must not break the pack
  import. KEEP (add a warning log so the drop is visible).
- **Observability best-effort** — heartbeat/provenance/settlement swallow
  (`cloud_media_invoke.py:238, 530, 577`). KEEP (does not hide a model failure).

---

## C. JUDGMENT CALLS — RESOLVED by operator 2026-07-03

1. **Stage-direction-only silence (#6): RIP the silence.** Operator was surprised
   the ledger even carries stage-direction-only beats. Decision: it must NOT emit
   silence. Fail LOUD if an empty-after-clean (stage-direction-only) line reaches
   the voice gate, so the writer never silently ships silence and such lines can't
   creep into dialogue. Future idea PARKED in `docs/ROADMAP_IDEAS.md`: route a
   stage-direction beat to a NEW media engine (overlay video / procgen / 3D /
   still) instead of a voice — re-add to the ledger then.
2. **OpenRouter model-gone (#15): keep a CONSTRAINED backstop tied to "latest".**
   The dropdown should offer dynamic **"latest"** aliases as the DEFAULT plus a
   few standard version pins (~last 3) expected to stay available; the model-gone
   path may fall back only to those REAL valid pins (never an invented slug). This
   is why the operator wanted "latest": it resolves dynamically so a dead pin is
   rare. Folds into the dropdown-validity workstream below.
3. **rank-chain / local auto-select: KEEP.** It only fires when no explicit engine
   is chosen and already picks only valid registered engines (cloud excluded). Not
   a mid-render model swap. The operator's "latest / only-valid-models" directive
   applies to the CLOUD model dropdowns, not the local engine rank-chain.

## C2. NEW workstream (operator 2026-07-03) — valid-models-only dropdowns
The OpenRouter model list AND the Comfy cloud model dropdowns must expose ONLY
real, currently-valid models, and DEFAULT to a dynamic **"latest"** alias. Dead /
stale model ids must not appear. Tracked separately from the fallback rip (R5).

---

## D. Sprint sequencing (each chunk green + committed+pushed to v2.0-alpha)

- **R1 — audio voice rip:** #1-#5 (+#6 per C1). One coherent change to
  `_otr_voice_node_common.py` + `cast_lock.py` + `eng_kokoro.py` + the adapter
  `missing_ref_fallback` metadata. Retire the bark-fallback tests, add fail-loud
  tests. Full suite + Bug Bible.
- **R2 — image rip:** #7-#8.
- **R3 — LLM/writer rip:** #9-#14 (largest; many tests pin the soft-fail).
- **R4 — convergence:** kibitz r2/r4 + Fable final grounded gate (CLAUDE.md §9,
  high-stakes structural), workflow-JSON audit, no new must-fix.

Every RIP replaces a silent swap with a NAMED loud raise (EngineUnusable /
ValueError / RenderError) — never a bare `raise`. UTF-8, no BOM, SFW.
