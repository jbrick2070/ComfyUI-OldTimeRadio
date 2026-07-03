# Proof #9 / night-queue verdict -- S4x full stack (2026-07-02 night, scored same evening)

Night queue: proof9c (768x416) -> 120w soak -> 30w soak, chained on the :8000 headless
server. Scored ~20:30 the same evening (soak2 still in flight at scoring time).

## proof9c: FAILED -- ops failure, NOT a model verdict

Episode `signal_lost_shadows_of_diplomacy_20260702_182914` (pending_20260702_182451)
died at shot_b002 with the LOUD no-fallback RenderError:

    VRAM ceiling breached across ltx_audio_in-render window: 14775 MB > 14500 MB

even at the stepped-down 768x416 canvas (the desktop-session squatter again; baseline
had crept ~2.9GB at queue time). No clips were produced; the driver's
"history output #92 has no text payload" crash is this failure surfacing (node 92 emits
no report when the render batch raises). NOTHING to score. proof9c must be RE-RUN on a
clean baseline (<=~2.5GB) -- ideally back at 832x448 so scores compare to proof7's bar.

## 120w soak (soak1): COMPLETED end-to-end -- scored as the interim S4x verdict

Episode `signal_lost_frostbitten_exodus_20260702_193614` (pending_20260702_184401):
6/6 clips, obs_publish OK (final 48.3MB), Prompt executed in 01:18:23. Rendered clip
canvas: 768x384. S4 portrait init FIRED on all character beats (log-proven:
"IA2V PORTRAIT INIT (S4): beat b002/b003/b004 conditions on portrait c0*.png 832x480");
S4c radio-face fired on b000/b001/b005; talking register on all speech beats.

Scores (raw clip + slice audio muxed; eval = otr_talking_radio_probe_eval same-file-twice):

| beat | role       | init                     | motion | r-onset | lag |
|------|------------|--------------------------|--------|---------|-----|
| b000 | music_open | radio-face (wide)        | 1.00   | 0.074   | -1  |
| b001 | announcer  | radio-face (wide)        | 2.22   | 0.135   | -1  |
| b002 | character  | portrait c02 (S4)        | 1.13   | 0.156   |  1  |
| b003 | character  | portrait c03 (S4)        | 1.49   | 0.098   | -2  |
| b004 | character  | portrait c02 (S4)        | 0.79   | 0.270   |  3  |
| b005 | announcer  | radio-face (wide)        | 2.09   | 0.203   |  3  |

## Honest read: the >=2.0 bar does NOT transfer across canvases

The 2.0 bar was set at 832x448 (proof7), where announcers scored 4.62/5.51. At 768x384
the SAME announcer stack scores 2.22/2.09 -- a ~55% metric drop from canvas alone, so
mouth-motion energy is NOT scale-invariant (the session_handoff "roughly scale-invariant"
assumption is falsified). Canvas-relative (char/announcer ratio) is the honest comparison:

- proof7 (832x448, scene-still init): chars 0.13 / 0.06 / 0.27 of the announcer mean.
- soak1  (768x384, S4 portrait init): chars 0.52 / 0.69 / 0.37 of the announcer mean.

S4 roughly TRIPLED character articulation relative to the in-episode announcer anchor --
directionally consistent with the decisive portrait A/B (0.57 -> 2.86). But characters
are still ~half the announcer level, and an absolute >=2.0 verdict at the proof7 canvas
is UNPROVEN until a clean 832x448 re-run.

## VERDICT: PROVISIONAL PASS on direction; final GO/NO-GO needs the 832x448 re-run

1. Re-run proof9 at 832x448 on a clean baseline (kill the desktop squatter first;
   confirm <=~2.5GB before boot). Score against the original bar + operator eyeball.
2. Consider a baseline-aware canvas step-down guard (the squatter keeps costing runs),
   and record the canvas in every future scorecard -- bars are canvas-specific.

## 30w soak (soak2): in flight at scoring time (~23 min elapsed, GPU 15.9GB active).
QA it on land: final in otr\obs + 6/6 clips + no ceiling breach + audio byte-identical.
