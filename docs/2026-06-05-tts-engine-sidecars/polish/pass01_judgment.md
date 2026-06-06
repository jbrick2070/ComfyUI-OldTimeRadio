# Polish pass01 judgment (Claude = sole judge, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Spend ~$0.26. Verdicts:
GPT "no", Grok "no", Gemini "yes-with-fixes", DeepSeek "yes-with-fixes". All four
converged on the SAME real defects in the new sidecar lifecycle. Every claim
checked against the actual files. Fixes folded + full suite re-run (3778 passed,
0 failed) + Bug Bible green.

## FOLDED (grounded, fixed)
- **stderr file-handle leaks on every error path** (all 4). New
  `_otr_sidecar.close_worker(proc, stderr)`: always closes stderr (even when
  proc is None -- Gemini's early-return bug), reaps with kill + wait. Used by
  both adapters' load/generate_voice/unload. Popen failure now closes stderr.
- **`readline()` with no timeout -> render thread hangs forever** (GPT, Grok).
  New `_otr_sidecar.read_protocol_line(proc, timeout, what)` (daemon reader
  thread; `select` is not usable on Windows pipes). Generous + configurable
  timeouts (`OTR_SIDECAR_STARTUP_TIMEOUT` 1800s for first-run downloads,
  `OTR_SIDECAR_REQUEST_TIMEOUT` 600s). On timeout/EOF -> kill + named RuntimeError.
- **Unguarded `json.loads(resp_line)`** (GPT). Both adapters now catch
  Timeout/EOF/ValueError/OSError around write+read+parse, kill the worker (stream
  no longer trustworthy), clear `_proc`/`_stderr`, remove the temp, raise named.
- **Dia worker never enforced 44100** (Grok, GPT). New `_ensure_rate(out_path)`
  resamples to 44100 after save (torchaudio, numpy-interp fallback), matching the
  chatterbox worker -- protects the single-rate pack contract.
- **`_resolve_clone_ref_path` hard-coded role="char_voice"** (GPT). Now takes a
  `role` param; `_render_per_line` passes `self.ROLE`, so an announcer render
  selects announcer refs. (indextts2 char path unchanged: role defaults char_voice.)
- **`_build_prompt` could emit "[S1] [S1]"** if a transcript already carried a
  tag (GPT). Added `_strip_lead_tag`.
- **`_load_wav` leaked the temp on a corrupt-WAV read** (GPT). `os.remove` moved
  into `finally` (via `remove_quietly`).

## ACCEPTED partially / noted
- GPT#4 "chatterbox announcer predictably fails when no ref": MITIGATED, not a
  hard fail -- `_resolve_clone_ref_path`'s gender-agnostic last resort returns an
  on-disk chatterbox ref (37 exist), so announcer renders (PD1). The role-aware
  fix improves ref SELECTION. Full announcer-via-clone casting stays future work.
- Determinism (GPT, Grok): sidecar output is seeded (per-request `_seed_everything`)
  but NOT guaranteed bit-exact (no external generator). Already reflected by
  `supports_external_generator=False`; bit_exact stays gated on the GPU pilot.

## REJECTED (MISREAD / not applicable)
- Grok#4 "fallback fires even when fb_name is None": the bark-fallback block
  already carries `and fb_name`; verified in the source. No change.
- Grok SHOULD#2 "chatterbox save uses a possibly-None src_sr": the worker saves
  at the constant `_TARGET_SR` (24000) regardless and only uses `src_sr` (already
  `or _TARGET_SR`-guarded) to decide whether to resample. No bug.

## CUTS considered / deferred
- `tempfile.mktemp` -> `mkstemp` (GPT SHOULD#1): NOT applied -- `mkstemp`
  pre-creates the file, which would defeat the worker's "produced no file" check.
  Kept `mktemp` (matches the proven indextts2 adapter). Minor, noted.
- Drop `_resolve_transcript` / `dia_ref_transcripts.json` (Grok, DeepSeek CUT):
  KEPT -- it is the documented zero-cost-when-absent quality-upgrade hook the
  design roundtable endorsed (basename-keyed). Disagreement noted.
- `verbose` request field + `base.supported_kwargs` consolidation (GPT CUT): left
  as-is (harmless; removing exported utils risks unrelated tests).
- Per-request readline timeout retrofit to the proven IndexTTS2 adapter: deferred
  (keep the shipped path byte-identical; the shared helper makes it a 1-line
  retrofit later).

## Verify-at-build (unchanged; GPU box only)
chatterbox torch on sm_120; chatterbox external-Generator; Dia audio_prompt-only
quality; Dia 0626 vs Dia2; exact library `generate()` signatures.
