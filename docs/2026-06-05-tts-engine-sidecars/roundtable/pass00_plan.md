# OTR TTS-engine sidecars -- chatterbox + Dia on adapter metadata (pass00)

## Context & goal
OTR is becoming a collection of swappable, "greatest-hits" TTS engines (a TTS
experiment). IndexTTS2 ships today as the default character voice via a Path-B
isolated-venv subprocess worker -- it runs out of process because its deps
hard-pin torch 2.8 / numpy 1.26 that would brick the main torch-2.10 / cu130
Blackwell (RTX 5080, sm_120) ComfyUI venv. This plan adds TWO more clone engines
on the SAME proven pattern, plus the foundation refactor that lets them (and
future engines) slot in cleanly:

- **chatterbox** (Resemble AI, MIT) -- already half-wired but UNUSABLE (in-process
  import = venv-bricker; 0 refs on disk). Engine is commercial-clean (MIT).
- **Dia 1.6B-0626** (Nari Labs, Apache-2.0) -- NEW. Commercial-clean engine, which
  fixes IndexTTS2's bilibili NON-COMMERCIAL license liability for Jeffrey's films.
  Dialogue-native, zero-shot clone -> reuses the existing CC0 reference bank.

The casting MUST-FIX #6 refactor (replace the hard-coded clone-engine name tuple
with adapter metadata) lands FIRST, so chatterbox + Dia register with zero
dispatch surgery.

## Invariants (reject any panel critique that breaks one)
- C-5: import-time side-effect-free (no model load / IO / CUDA at import).
- C-7: fail CLOSED with a NAMED error; never auto-fetch a default-on model; never
  silently swap engines.
- PD1: the episode ALWAYS renders (audio is king) -- missing ref -> bark fallback.
- Byte-identical legacy bark path when bark is selected.
- Model-agnostic dispatch: engines self-describe; NO per-engine name ladders.
- ZERO shared torch between the main venv and any sidecar venv.
- 16 GB VRAM ceiling; only one clone engine resident per render (I-7 teardown).
- UTF-8, no BOM, ASCII-only source.

## Proven template (ground every claim against these real files)
- `nodes/_otr_audio_engines/eng_indextts2.py` -- adapter: subprocess.Popen +
  line-delimited JSON + readiness line + fail-closed named errors + soundfile WAV
  load (NOT torchaudio: main venv's torchaudio routes through uninstalled
  torchcodec). env: OTR_INDEXTTS2_VENV / _DIR / _WORKER / _FP16.
- `scripts/_otr_indextts2_worker.py` -- worker: dup(1) saved aside, dup2(2,1) so
  all model/torch/tqdm prints land on stderr and can NEVER corrupt the JSON
  channel; protocol JSON written to the saved real-stdout fd.
- `nodes/_otr_voice_node_common.py` -- `_render_per_line` dispatch;
  `_OTR_CLONE_ENGINES`; `_resolve_clone_ref_path` (bank lookup by gender, keyed on
  char_id); the lazy bark fallback; I-7 teardown-in-finally.
- `nodes/_otr_audio_engines/registry.py` + `base.py` -- AudioEngine Protocol,
  `assert_usable` (6-class fail-closed), `supported_kwargs` blind-call guard,
  `pack_audio_batch` (single-rate batch contract).
- `config/voice_reference_bank.json` -- 36 CC0 indextts2 char refs ON DISK (real
  sha256); 5 placeholder chatterbox rows pointing at /refs/chatterbox/*.wav that
  do NOT exist; 1 kokoro announcer (bm_george).

## Part A -- Foundation refactor (casting MUST-FIX #6)
Replace `_OTR_CLONE_ENGINES = ("indextts2","chatterbox")` and its two membership
branches in `_render_per_line` with adapter metadata.
- Add to each CLONE adapter (indextts2, chatterbox, dia):
  `requires_voice_ref = True`, `voice_ref_kind = "wav_path"`,
  `missing_ref_fallback = "bark"`.
- Defaults on `AudioEngineAdapter` (and via getattr for duck-typed legacy
  adapters): `requires_voice_ref = False`, `voice_ref_kind = None`,
  `missing_ref_fallback = None`.
- Dispatch edits (the `adapter` object is already in scope):
  `engine in _OTR_CLONE_ENGINES` -> `getattr(adapter,"requires_voice_ref",False)`;
  hard-coded `get_engine("bark")` -> `get_engine(getattr(adapter,
  "missing_ref_fallback", None))` (skip fallback if None).
- Grounded: `_OTR_CLONE_ENGINES` is referenced ONLY in
  `_otr_voice_node_common.py`; NO test depends on it; `voice_ref_field`
  (bark="voice_preset", kokoro="voice_ref_id") already proves the metadata
  pattern works. Behavior stays byte-identical for indextts2 + bark.

## Part B -- chatterbox Path-B sidecar (MIT)
- Isolated venv: python 3.11, `pip install chatterbox-tts`. It pins its own torch;
  the main venv is NEVER touched. VERIFY-AT-BUILD: whether the pinned torch runs
  on Blackwell sm_120 or the venv needs a cu128 torch override (same class of fix
  IndexTTS2 needed). Model cache via HF (auto-download on first from_pretrained).
- Worker `scripts/_otr_chatterbox_worker.py` (mirror the indextts2 worker fd
  dance). Launch: `python _otr_chatterbox_worker.py [--variant base|turbo]`.
  request: `{text, ref_clip, exaggeration, cfg_weight, seed, out_path, verbose}`.
  Body: `from chatterbox.tts import ChatterboxTTS`;
  `m = ChatterboxTTS.from_pretrained(device="cuda")`;
  `wav = m.generate(text, audio_prompt_path=ref, exaggeration=ex, cfg_weight=cw)`;
  save with soundfile at `m.sr`; reply `{ok, out_path, sample_rate: m.sr}`.
- Adapter rewrite `eng_chatterbox.py`: in-process import -> Popen worker. env
  OTR_CHATTERBOX_VENV / _WORKER (and optional _MODEL for a pinned cache dir).
  `commercial_clean=True`, `requires_flag="OTR_ENABLE_CHATTERBOX"`,
  `interface="per_line"`. Keep `_project(delivery_vector)->exaggeration`.
  Return `m.sr` dynamically (do not hardcode 24000).
- NOTE: every chatterbox output is PerTh-watermarked (imperceptible). Document it;
  not a blocker for OTR audio.

## Part C -- Dia Path-B sidecar (Apache-2.0, commercial-clean)
- Isolated venv: python 3.10/3.11 + torch 2.8 nightly cu128 (Blackwell, Nari
  issue #26) + `pip install git+https://github.com/nari-labs/dia.git`. ~4.4 GB
  fp16. Output sample rate 44100 (Descript Audio Codec).
- Worker `scripts/_otr_dia_worker.py`: `from dia.model import Dia`;
  `m = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")`.
  per line: build `[S1] <target text>`; if a clone transcript is supplied prepend
  it (see wrinkle); `out = m.generate(text, audio_prompt=ref_clip, ...)`;
  `m.save_audio(out_path, out)`; reply `{ok, out_path, sample_rate: 44100}`.
- Adapter `eng_dia.py`: same Popen template. env OTR_DIA_VENV / _WORKER / _MODEL.
  `commercial_clean=True`, `requires_flag="OTR_ENABLE_DIA"`,
  `roles=("char_voice","announcer_voice")`, `interface="per_line"`,
  `sample_rate=44100`. Each per-line render is a single `[S1]` turn.

### Dia voice-clone wrinkle (KEY open question)
Dia's zero-shot clone REQUIRES the transcript of the reference clip prepended:
`[S1] <ref transcript> [S1] <target>` with `audio_prompt=ref.wav`. Our 36 CC0
refs have NO transcripts. Proposed: `scripts/_otr_dia_transcribe_refs.py`
(faster-whisper, one-time, writes `config/dia_ref_transcripts.json` keyed by
voice_ref_id). Adapter resolves the transcript; if absent -> log a WARNING and
fall back to audio_prompt-only (degraded clone) rather than bark, since Dia can
still condition on the clip. DECISION: audio_prompt-only vs treat-as-missing-ref
(-> bark). Quality of audio_prompt-only is empirical (verify-at-build).

## Part D -- Reference bank wiring (no new files, no downloads)
The 36 CC0 indextts2 WAVs are clone-engine-agnostic. Mirror the 36 indextts2
`char_voice` rows to `engine="chatterbox"` and `engine="dia"` (same `ref_path` +
`ref_sha256` + gender/timbre/roles/age_band/commercial_clean), with new unique
`voice_ref_id` prefixes (`cb_*`, `dia_*`). Drop the 5 placeholder chatterbox rows
that point at nonexistent `/refs/chatterbox/*.wav`. Tool: a small idempotent
`scripts/_otr_mirror_refs.py` (read bank -> emit mirrored rows) OR extend
`scripts/otr_dl_indextts2_refs.py` with `--engine`. Bank JSON hot-reloads (no
restart). Caster filters by `engine`, so each engine gets a full 36-voice pool.
- Effective commercial_clean = engine AND ref (MUST-FIX #2, separate work):
  chatterbox/dia + CC0 ref = clean; indextts2 + CC0 ref = NOT clean. Bank entry
  `commercial_clean` stays the REF's value (CC0=true); the engine carries its own.

## Part E -- Tests (run full suite after every .py change; was 3758/0)
- registry: chatterbox + dia register; correct roles; `requires_flag` gates them
  off by default; `assert_usable` fail-closed when flag unset.
- adapter load fail-closed: missing venv / worker / weights -> NAMED RuntimeError
  (no crash, no silent swap), mirroring the indextts2 load() tests.
- refactor: metadata replaces the tuple; a clone engine with no ref still triggers
  bark fallback; bark/indextts2 byte-identical; non-clone engines never fall back.
- worker protocol: unit-test the JSON request/response contract with a stub model
  (no GPU) -- readiness line, bad-json handling, stop.
- bank: chatterbox + dia rows load + validate; caster assigns a gender-correct
  voice; no on-disk-claimed ref is missing.
- Bug Bible regression + core + dropdown tests (Three-File Contract if any Bug
  Bible YAML changes).

## Part F -- Operator-gated (heavy; on the box; needs a ComfyUI RESTART)
Code + bank + tests land this session. The operator runs: each sidecar install
ps1 (isolated venv + first-run model download), set USER env vars, RESTART
ComfyUI, transcribe Dia refs, set node 81 `engine` + node 80 `voice_bank`, live
smoke render, confirm the main venv is untouched. Documented in
`docs/chatterbox_pathb_setup.md` and `docs/dia_pathb_setup.md`.

## Open questions for the panel (each flagged verify-at-build where empirical)
1. chatterbox-tts pinned torch on Blackwell sm_120 -- runs as-is, or needs a cu128
   torch override in its venv? (empirical)
2. chatterbox `generate()` -- does it bind an external `torch.Generator`? Keep
   `supports_external_generator=False` until the GPU pilot confirms (G1/bit_exact).
3. Dia clone WITHOUT transcript -- acceptable, or hard-require transcripts? Pick
   the missing-transcript policy (audio_prompt-only vs bark).
4. Dia 1.6B-0626 vs **Dia2** (released 2025-11-19) -- target the proven 0626 now,
   or jump to Dia2? Confirm Dia2 license + API before switching.
5. Process model -- one lazily-spawned worker per engine, torn down in `finally`
   (I-7); is keeping the worker warm across lines within a render safe for 16 GB
   alongside later HuMo video? (empirical)
6. Sidecar venv torch flavor drift -- pin exact torch builds per sidecar so a
   future `pip install` upgrade cannot silently break sm_120.

## What ships this session vs deferred
SHIP: refactor (A), chatterbox worker+adapter (B), Dia worker+adapter (C), bank
wiring (D), install ps1 + setup docs, tests (E), full suite green. DEFER to
operator: the two isolated-venv installs + model downloads + Dia ref
transcription + RESTART + live GPU smoke render (F). The other casting MUST-FIX
items (#1-#5) are tracked separately in the casting roundtable plan.
