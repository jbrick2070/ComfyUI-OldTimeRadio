# TTS-engine sidecars (chatterbox + Dia) -- operator handoff (2026-06-05)

The project is becoming a collection of swappable TTS engines. This session added
TWO commercial-clean clone engines on the proven IndexTTS2 Path-B pattern, and the
foundation refactor so future engines slot in with zero dispatch surgery.

## What shipped (code complete, fully tested headless)
- **Refactor (MUST-FIX #6):** deleted the hard-coded `_OTR_CLONE_ENGINES` tuple;
  the dispatch now branches on adapter metadata (`requires_voice_ref`,
  `voice_ref_kind`, `missing_ref_fallback`). indextts2 + bark behavior unchanged.
- **chatterbox** (MIT): `eng_chatterbox.py` rewritten from the venv-bricking
  in-process import to a subprocess sidecar + `scripts/_otr_chatterbox_worker.py`.
- **Dia** (Apache-2.0 -> commercial-clean): new `eng_dia.py` +
  `scripts/_otr_dia_worker.py`. char_voice only this pass.
- **Shared sidecar lifecycle** (`nodes/_otr_audio_engines/_otr_sidecar.py`):
  bounded read (no infinite hang) + idempotent teardown (no handle/zombie leak).
- **Bank:** 36 chatterbox (`cb_*`) + 36 Dia (`dia_*`) char refs mirror the 36 CC0
  indextts2 WAVs (no new files), + 1 chatterbox announcer ref. Old 5 placeholder
  rows (0 WAVs on disk) removed. Regenerate with `scripts/_otr_mirror_clone_refs.py`.
- **Profiles:** `char_dia_v1` added; chatterbox profiles flipped to `oop_venv`.
- **Tests:** `tests/test_tts_engine_sidecars.py` (23 tests). Full suite **3781
  passed / 0 failed**; Bug Bible green. 3 review rounds (1 design + 2 polish, all
  grounded) folded; see `roundtable/` and `polish/`.

## Operator steps (heavy / GPU / needs RESTART -- not doable headless)
### chatterbox
1. Run `scripts\_otr_chatterbox_install.ps1` (creates the isolated venv +
   `pip install chatterbox-tts`; first render downloads the model).
2. `setx OTR_ENABLE_CHATTERBOX 1`
3. RESTART ComfyUI (loads the new env + the .py via module cache).
4. Canonical workflow: node 80 `voice_bank=default`, node 81 `engine=chatterbox`.
   Queue a small cast.
5. If the install smoke printed `cuda False` or a render errors with an sm_120 /
   kernel message: reinstall a cu128 torch INTO the chatterbox venv only (the
   commented line in the install script).

### Dia
1. Run `scripts\_otr_dia_install.ps1` (isolated venv + torch 2.8 nightly cu128 +
   `pip install git+https://github.com/nari-labs/dia.git`).
2. `setx OTR_ENABLE_DIA 1`
3. RESTART ComfyUI.
4. node 80 `voice_bank=default`, node 81 `engine=dia`. Queue a small cast.
5. (optional quality upgrade) add `config/dia_ref_transcripts.json` keyed by
   reference WAV basename to enable transcript-conditioned cloning.

Full per-engine detail: `docs/chatterbox_pathb_setup.md`, `docs/dia_pathb_setup.md`.

## Verify-at-build (only the GPU box can answer these)
1. chatterbox-tts pinned torch on Blackwell sm_120 -- runs, or needs the cu128
   override in its venv?
2. chatterbox `generate()` external `torch.Generator` -- flip
   `supports_external_generator` True for bit_exact only after the dep pilot
   confirms (`scripts/otr_audio_dep_pilot.py --engines chatterbox,dia`).
3. Dia audio_prompt-only clone quality -- acceptable, or add ref transcripts?
4. Dia 1.6B-0626 vs **Dia2** (released 2025-11-19) -- target 0626 now; evaluate Dia2 later.
5. 16 GB VRAM headroom: one clone worker resident + later HuMo video.

## Git (state is clean: branch even with origin/v2.0-alpha)
This session's work is UNCOMMITTED (14 modified + 6 new files). Nothing else is
pending (the prior 7 commits are now pushed; no stray workflow deletions). To
checkpoint via Desktop Commander (cmd) -- after deleting the scratch `_run*.out.txt`
/ `_fullsuite*.out.txt` capture logs under `docs/2026-06-05-tts-engine-sidecars/`:
```
git -C <repo> add -A
git -C <repo> commit -m "feat(audio): chatterbox + Dia Path-B sidecars on adapter metadata (MUST-FIX #6); 3781 tests green"
```
Push is a separate decision (one attempt max, then verify HEAD match / no BOM / AST parse).
