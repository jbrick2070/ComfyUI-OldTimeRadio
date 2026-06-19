# Clip-Fill / Dynamic-VRAM -- GPU VERIFICATION RESULTS (2026-06-18)

Headless small-episode smoke on the 5080. Box reset clean before + after (GPU
baseline ~1.5 GB). Server booted via `scripts/_otr_soak_server_launch.cmd <log> WAN`
(enables wan_ti2v + the Wan2.2 VAE). Episode driven by `scripts/_otr_combo_soak.py`
with all three video slots forced to wan_ti2v, bark voice (dodges the headless
indextts2 gap).

Invocation: `OTR_COMBO_ANNOUNCER/MUSIC/BEATS=wan_ti2v OTR_SOAK_TARGET_WORDS=60
OTR_SOAK_ACT_COUNT=1 OTR_COMBO_NCHARS=2 OTR_SOAK_CHAR_VOICE=bark`.

## RESULT: SUCCESS (status=success, 1654s; Prompt executed 00:27:20)

All five clip-fill pieces verified end-to-end on real GPU renders:

### The freeze fix (Pieces 1+2+3) -- 6/6 beats clip-filled, adaptive
Per-beat log (`[OTR video] wan_ti2v CLIP-FILL ... VRAM render-phase peak`):

| beat | role               | native frames | extended to (target) | VRAM peak |
|------|--------------------|---------------|----------------------|-----------|
| b001 | announcer_visual   | 29            | 238                  | 8610 MB   |
| b00x | (music/abstract)   | 25            | 290                  | 9988 MB   |
| ...  | (4 more beats)     | 25-29         | audio-derived target | <=9988 MB |

- The native render length is PREDICTED from live free VRAM (the budget shrank
  29 -> 25 across beats as residue varied -- proof the predictor reads live
  `mem_get_info`, not a static value), then ping-pong-extended to each beat's
  audio-derived target. BEFORE the fix every clip was hard-clamped to 17 frames
  (0.68s) then frozen.
- Every render-phase NVML peak (8610-9988 MB) stayed well under the 14500 MB
  ceiling. NO OOM, no `ceiling breached`, no react-to-OOM.

### Motion proof (acceptance #1)
`clips/shot_b001_announcer_visual_wan_ti2v.mp4` = 290 frames @ 25fps = **11.6s**
(the full beat, not 0.68s). Frames pulled at t=2.0 / 3.5 / 5.0s have THREE
DISTINCT md5 hashes -> continuous motion fills the beat (no hold-last-frame
freeze). FCA26AE5..., 551F3691..., 6991A6C1...

### Persistence (Piece 4, acceptance #2)
`[OTR video] persisted 6 clip(s) to episodes/pending_20260618_182026/clips/`.
Durable folder holds all 6 named clips
(`shot_<beat>_<role>_wan_ti2v.mp4`) -- NOT the janitor-swept `_shared/tmp`.

### Underrun guard (Piece 5)
No `CLIP UNDERRUN` warnings fired -- every clip filled its beat, which is the
correct outcome (the guard is insurance for a future short-clip engine).

### Frozen audio spine intact
`[OTR_MasterAudioMux] audio_byte_identical OK (9762fbbabfcb)` -- the clip-fill
changes never touched audio (V-1 holds).

### Final deliverable
`obs_publish OK -> output/otr/obs/signal_lost_humming_dilemma_20260618_182418_silent_procgen_blended_final.mp4`

## Offline suite (pre-GPU)
4547 passed / 33 skipped / 0 failed; Bug Bible 16/7/3. LTX path untouched ->
byte-identical (unit-covered).
