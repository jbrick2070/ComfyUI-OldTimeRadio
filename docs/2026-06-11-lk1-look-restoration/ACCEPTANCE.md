# LK-1 LTX look restoration -- acceptance record (2026-06-11 night)

Same-seed A/B pairs for every LTX shot of the reference episode
`signal_lost_plasmas_embrace_20260611_201117`, rendered on the live
headless server (RTX 5080, LTX lane) with the episode's own ST-3 stills,
driver-composed prompts and request seeds (`scripts/_otr_lk1_dump_b000.py`
-> `scripts/_otr_lk1_ab_probe.py`).

* **Leg A (pre-LK-1, the murk):** text-only EmptyLTXVLatentVideo +
  KSampler 30 steps / cfg 3.0 -- what shipped after the platform refactor.
* **Leg B (LK-1 restoration, the shipped default @ babb7d1):**
  scene-still init via LTXVImgToVideo strength 0.75 + the legacy
  distilled chain (LTX_DISTILLED_SIGMAS 9 vals / 8 steps, euler,
  CFGGuider cfg 1.0) -- the v0_9 production config of the 6/5 look
  (User env OTR_LTX_ENGINE=v0_9 + the legacy node's default).

| beat | seed | A (text+ksampler) | B (still+distilled) |
|------|------|-------------------|---------------------|
| b000_music_open | 607339423 | 235.8 s | 50.1 s |
| b001 (announcer open) | 944021193 | 245.7 s | 50.2 s |
| b005 (announcer close) | 1794688454 | 235.6 s | 50.1 s |

B is ~4.7x faster (matches the legacy v0_9-vs-v2_3 4.4x note) and
still-conditioned: composition pinned to the episode's FLUX scene still,
no murk. 3-frame contact sheets (first/mid/last) per clip are beside this
file; the webm clips live in the episode at `qa/lk1_ab/`.

Code: LK-1a/b/c shipped @ `babb7d1` (still-conditioning default ON,
distilled sampler default, palette diet verified as Part-A era_profile).
OPERATOR EYEBALL gates the look; the post-OH-4 fresh 30w render is the
end-to-end proof on the production path.
