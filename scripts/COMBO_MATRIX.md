# Risky-but-valid 30-60w combo matrix (full render)

Bug-hunt for v2.0-alpha. 9 combos that are schema-valid against the live
`/object_info` (each **should** work) but each isolates one code- or
memory-grounded failure mode (each is **most likely to break**). Every combo
runs the FULL workflow to a final mp4: script -> cast -> voice -> music ->
HuMo -> upscale -> procgen. Runs sequentially, chatterbox last and isolated.

Driver: `scripts\run_combo_matrix.py`. Dry-run 2026-06-05: all 9 patch +
convert clean against live schemas (29-node prompt, values verified). Nothing
submitted yet -- that is the run below.

## Run it (ComfyUI must be up on :8000)

```
cd 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
& 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe' scripts\run_combo_matrix.py --list
& 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe' scripts\run_combo_matrix.py
```

Flags: `--only C3` (one combo) | `--skip-cloud` (no C7/C8 credit spend) |
`--skip-dangerous` (no C9 chatterbox) | `--timeout 3000` (per-combo seconds,
default 2700).

Results stream to `scripts\combo_matrix_results.json` after every combo (a
mid-run abort still leaves a partial record). A summary table prints at the end.

## Prerequisites

- **All combos:** ComfyUI live; use the browser tab at :8000 for heavy renders
  (the Electron window goes black under VRAM pressure -- cosmetic, backend
  completes via API).
- **Gemma combos (C1, C2, C3, C5):** Ollama up at 127.0.0.1:11434; gemma model
  pulled. C3 also needs the Mistral-Nemo transformers weights.
- **indextts2 (most combos):** the Path-B sidecar venv + `OTR_INDEXTTS2_*` env
  vars set (default char voice).
- **C7 OpenRouter:** `OPENROUTER_API_KEY` + `OTR_ENABLE_OPENROUTER=1` (User env,
  restart Comfy). Spends OpenRouter credits (~cents at 40w).
- **C8 Comfy Credits:** `OTR_ENABLE_COMFY_CREDITS=1` + logged-in Comfy Desktop
  with prepaid credits. Spends Comfy credits (~cents at 40w).
- **C9 chatterbox (DANGER):** runs last. Memory: chatterbox dep-pins can brick
  the torch2.10 venv. If it hangs, kill ComfyUI and verify the venv before
  trusting later runs (it is already last, so nothing follows it).

Runtime: each render is bounded by speech length (~13-26s of video for 30-60
words), roughly a few-to-15 min on the RTX 5080. All 9 ~ 1-2 hrs.

## The matrix

| # | Words | Cast | Writer (creative / technical) | Voice | Music | Why it should work | Why it's most likely to break |
|---|------|------|-------------------------------|-------|-------|--------------------|-------------------------------|
| C1 | 60 | 6 | gemma-4-12b / gemma-4-12b | indextts2, reuse=off | stable_audio_3 | 6 chars allowed; indextts2 is default | Only **1 CC0 female ref** in the bank -> 6 unique speakers exhaust distinct female refs |
| C2 | 55 | 4 | gemma-4-12b / gemma-4-12b | indextts2 | stable_audio_3 | BUG-305 name-abbrev fix shipped | More chars + gemma -> higher odds it abbreviates a name and re-trips the freeze-halt |
| C3 | 40 | 2 | **gemma-4-12b / mistral-nemo** | indextts2 | stable_audio_3 | each model proven solo | Crosses the **Ollama <-> transformers** loader seam mid-episode (narrative vs JSON passes) |
| C4 | 45 | 3 | mistral-nemo / mistral-nemo | **bank=default + engine=bark** | stable_audio_3 | both values valid | bank/engine **mismatch**: clip-ref bank vs preset engine -- does CastLock guard it? |
| C5 | 50 | 2 | gemma-4-12b / gemma-4-12b | indextts2 | **musicgen** | musicgen shipped (clean-break 1c) | 2nd music engine fed a gemma-written Meta-brief is an untested cross |
| C6 | 30 | 2 | mistral-nemo / mistral-nemo | indextts2 | stable_audio_3 | 30=min words, **act_count=7** allowed | ~4 words/act -> beat-budget band underflow / empty acts at the assembler |
| C7 | 40 | 2 | **openrouter:slot-a** (claude-opus-4.8) | indextts2 | stable_audio_3 | OpenRouter lane shipped, cached | Full cloud writer: fail-closed remote JSON, cost-guard floor, no-evict, cloud->local handoff |
| C8 | 40 | 2 | **comfy:slot-a** (claude-opus-4.7) | indextts2 | stable_audio_3 | Comfy Credits lane shipped | Cloud writer via Comfy auth: slot resolution, credit billing, cost-guard |
| C9 | 45 | 2 | mistral-nemo / mistral-nemo | **engine=chatterbox** | stable_audio_3 | chatterbox offered in the UI | dep-pins (torch2.6/numpy1.26) can brick the torch2.10 venv -- **run last, isolated** |

**Pass criterion** is per-combo, not just "it rendered." For the coupling and
scarcity probes (C1, C4, C6) a clean **fail-closed with a clear message** is a
PASS; a silent wrong-voice render, a same-voice collision, or an uncaught crash
is a FAIL. Full pass criteria are in each combo's `expect` field
(`--list` or `combo_matrix_results.json`).

## Reading results

`scripts\combo_matrix_results.json` -> per combo: `status`
(`SUCCESS` / `FAIL` / `TIMEOUT` / `SUBMIT_REJECT` / `SETUP_ERROR`),
`elapsed_s`, `error`, `prompt_id`, the applied `patches`, and a
`/otr/latest_ledger` snapshot (episode dir + final mp4 when present).
