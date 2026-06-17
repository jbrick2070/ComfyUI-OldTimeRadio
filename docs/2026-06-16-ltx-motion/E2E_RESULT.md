# motion_clause -- live E2E result (2026-06-16)

## Run
- Driver: `scripts/_otr_ltx_allroles_soak.py` (loads the REAL
  `workflows/otr_scifi_16gb_full.json` -- source of truth).
- Config: 30 words, 1 act, 3 characters; ALL video beats forced to `ltx_video` via the
  sanctioned 16gb_full profile override (announcer + music + character); char voice = bark.
- Flag: `OTR_LTX_MOTION_CLAUSE=1` set in the SERVER env before boot (the generation runs
  in the server process, not the client -- important).

## Result: SUCCESS
- Episode status **success**, 1201s (~20 min). Deliverable = the obs final (the full
  workflow ALWAYS lands here): `output/otr/obs/signal_lost_fistful_of_wind_20260616_165708_
  silent_procgen_blended_final.mp4` -- h264 1472x832 + AAC audio ~65s ("silent" = the
  pre-mux stage name; the final IS muxed with audio). The episode-dir /audio/*.mp4 is NOT
  the deliverable.
- `[OTR_VideoRenderBatch] motion_clause: {'generated': 4, 'reused': 0, 'fallback': 2, 'invalid': 1}`
- LTX fired on all 6 beats (6 GGUF loads), VRAM fit (≈10.5 GB resident, partial unloads).

## The clauses landed in the prompts (per-beat, dialogue-driven, subject-named)
- b001 announcer  -> "An announcer, eyes gleaming, leans in, whispering urgently." (generated)
- b002 character  -> "...Sunan Spender ti[lts]..." (generated)
- b003 character  -> "...Quinn Wells squi[nts]..." (generated)
- b004 character  -> "...Kevin Reeves lea[ns]..." (generated)
- b000 music_open -> static console motion (no dialogue -> fallback, byte-identical)
- b005 announcer  -> static console motion (fallback; 1 invalid generation fell back safely)

Each generated clause names the subject and uses a subtle verb -- exactly the design. The
invalid:1 proves the fail-closed path (a bad LLM output was rejected -> static fallback,
episode never broke).

## Known refinement (NOT a correctness bug) -- ledger persistence
The clauses are mutated into the IN-MEMORY ledger and used by the render (confirmed in the
prompts + logged), but they are NOT written back to the saved episode
`..._ledger.json` (0 files carry motion_clause). Consequence: re-renders REGENERATE
(reused:0) rather than reuse, and you can't audit the exact clause from the ledger file
(only from the render log/trace). To close it, the render node would persist the mutated
ledger to the episode ledger path. Deferred (needs the ledger-save-back point; the feature
works without it).

## Writer-lane architecture sanity
- Default lane (this run) proven E2E.
- `make_writer_generate_fn` reuses `_build_truncating_generate_fn`, which has explicit
  per-provider branches: HF (in-process), Ollama (gemma4/mistral local daemon),
  OpenRouter, Comfy Credits -- so the clause generation is lane-agnostic by construction.
- Per-lane concrete spot-check = set `creative_writing_model` (node 1) to the lane's model
  and re-run the soak; no code change needed.

## Status
SHIPPED + pushed (v2.0-alpha, HEAD 3f4cc0e): module + render read-hook + live generation
wiring + tests (25) + spec. Full suite 4437/0, Bug Bible 16/7/3. Opt-in
(`OTR_LTX_MOTION_CLAUSE=1`); default OFF = byte-identical.
