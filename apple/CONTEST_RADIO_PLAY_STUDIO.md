# Radio Play Studio -- Comfy Open Source Higgsfield contest plan

The resume point for any session working the contest. Read this, then `git log` on the
branch, then do the first unchecked step. Update this file in the same commit as the work.

## The contest

- Submission closes **Oct 30, 9:00 am PT**, through the Typeform. Comfy Build Night LA is Oct 29.
- One submission per person. SFW. No unlicensed IP, no real-person likenesses.
- Open-source license plus a runnable setup. MIT is already in `contest/radio-play-studio/LICENSE`.
- Built on the Comfy Developer Platform: here, the Comfy SDK (`comfy-sdk` 0.5.0) through
  `comfy-api-proxy` 0.1.6.
- The demo video must name every model and tool used.
- **Open weights only** (Open Weights prize): no Comfy Router, no partner nodes, no cloud writer.
- LTX as the core video model also qualifies for the **$2,000 LTX VFX bonus**.

## Where things are

- **Branch:** `claude/sharp-heisenberg-no3hzt` (draft PR #8, jbrick2070/ComfyUI-OldTimeRadio).
  Push here. Never merge to `main` until the operator says so.
- **Pack fixes** (anything under `nodes/`) go to `main` first, then `git merge main` into the branch.
- **Never edit `pyproject.toml`** (it auto-publishes). Never tag or release.
- **App:** `contest/radio-play-studio/`. Its venv is `C:\Users\jeffr\Documents\ComfyUI\.venv-rps`
  (Python 3.12, comfy-sdk 0.5.0, comfy-api-proxy 0.1.6, pytest).
- **App tests:** from `contest/radio-play-studio`:
  `C:\Users\jeffr\Documents\ComfyUI\.venv-rps\Scripts\python.exe -m pytest -q -p no:cacheprovider tests`
  (95 pass as of 9411772d). Repo side: `tests/test_contest_app_converter_parity.py` with the ComfyUI
  venv (30 pass).
- **Default workflow:** `workflows/otr_16gb_foley.json` (operator: "i like foley better", and
  "always 16 GB foley"). It ships 3 acts; the test runs use `--acts 1`.

## How to run it live on the 5080

1. Reset (CLAUDE.md section 4): nothing on :8000, VRAM at about 1.5 GB.
2. Boot: `Start-Process -FilePath scripts\_otr_soak_server_launch.cmd -ArgumentList "<log>"` (port 8000).
   ComfyUI core must be **v0.41.0+** (FLUX 3 Klein); the 5080 is on **v0.42.1** since 2026-10-19.
3. Bridge: `C:\Users\jeffr\Documents\ComfyUI\.venv-rps\Scripts\comfy-api-proxy.exe run --comfyui http://127.0.0.1:8000 --port 8189`
4. `python -m radio_play_studio doctor --comfyui http://127.0.0.1:8000` (every line ok), then
   `run --comfyui http://127.0.0.1:8000 --bank shakespeare --acts 1`, or `serve --port 8766 --comfyui ...`
   (`.claude/launch.json` has `radio-play-studio` on 8766 and `radio-play-studio-ui` on 8767).
5. Success = `RESULT`/`DONE`, `obs_publish OK` in the server log, the mp4 in `output\otr\obs\`
   AND in `contest\radio-play-studio\episodes\<id>\`, byte-identical.

## Phase 1 -- the live loop (target Oct 22): DONE 2026-10-19

- [x] App venv, requirements, doctor all ok.
- [x] ComfyUI 0.39.2 -> 0.42.1 (Klein needed v0.41.0; the 16 GB graphs now use MOSS-Image, which
      needs v0.42.0).
- [x] Pack bug: Gemma 4 12B wrote `",""presence"` on every casting retry. JSON repair on main:
      3e8affa1 + 9e81e879 (PBUG-20261019-01).
- [x] Live, video lane: As You Like It, 1 act, 65 min, RESULT SUCCESS, obs_publish OK,
      `bark_etchings_20261019_143717_..._final.mp4` (74 MB, 1920x1080, 1m52s), sha d713cf92.
- [x] Live, foley lane: 1 act, 63 min 37 s, obs_publish OK,
      `boxtree_secrets_20261019_154309_..._final.mp4` (81 MB, 1920x1080, 1m48s), sha 20be0dc4.
- [x] Settled live: SDK events + progress through the proxy; mp4 from OTR_MasterAudioMux ("gifs"
      -> video); a 65-minute job survives the SDK's timeouts; file in otr/obs and ./episodes.
      The `__value__` wrap: no shipped graph produces one; the proxy forwards the graph untouched
      and core unwraps it (execution.py:996), by reading.
- [x] App fixes, each Sonnet-QA'd: progress bar (4812a65b, 06d0a475, e71c8664, 3544cfd7, c84e790a:
      clock-based bar with minutes left, learned per machine), loading notice, foley default
      (3ea1eda0), "Tell my own story" (84f34439, be39d460, 4aa044ca, 74e662c8, 9411772d).
- [x] Browser run (pressed on the page, shakespeare, 1 act, foley): 75 min 53 s, obs_publish OK,
      `two_berries_20261019_164732_..._final.mp4` (139 MB, 1920x1080, 2m18s), sha f09e2c20,
      byte-identical in otr/obs and episodes/2f74674b184c/.
- [x] It saved WRONG timings (all 76 min on the writer): a curl recorder on the same job took
      ComfyUI's one-socket-per-client-id (server.py `sockets[sid] = ws`) and starved the app's
      stream. Guard: a stream that never saw half the graph finish saves nothing. The history was
      rebuilt from that recorder's complete log: writer 64 s, stills ~2 min, video 4,209 s of 4,400.
- [x] **Clock bar, measured:** proof run (CLI, shakespeare, 1 act, foley) 63 min 41 s,
      obs_publish OK, `carved_bark_20261019_180624_..._final.mp4` (73 MB, 1920x1080, 1m54s),
      sha 9a7027e8, byte-identical in both places; timings saved for every stage.
      Bar vs clock (5-minute rows): -3, -4, -5, -5, -7, -8, -9, -10, -11, -12, -13, -14 points;
      worst 15, "min left" over by up to 11. The SHAPE is right (node count sat near 80% for the
      whole video stage); the SCALE trails because this story's video took 3,501 s against the
      4,209 s it was calibrated on (one-act foley ran 64-76 min today, beats follow the story).
      Fix: saved times are now a running average (half old, half new), and the history is the
      mean of the two clean runs (video 3,855 s, total 4,050 s).
- [x] **Second check (Midsummer, averaged history):** 57 min 06 s, obs_publish OK,
      `bird_song_20261019_191143_..._final.mp4` (71 MB, 1920x1080, 1m45s), sha 0c8953bd, both
      places. Bar trailed by -3 at 5 min to -17 at 55; min-left over by up to 12. **+/-10 NOT
      met**, and averaging cannot meet it: one-act foley ran 57, 64, 64 and 76 min today because
      the beat count follows the story, and the app cannot see beats during the video node. The
      only real fix is a beats-done signal from the pack (OTR_VideoRenderBatch), which reopens the
      refuted hook-swap design -- operator's call whether it is worth it before Oct 30.
- [x] **Beat-level bar, PROVEN (operator chose "fix bar with beats").** Pack: the video node draws
      render frames done of the episode, inner sampler bars quieted on the render thread only
      (main ec55add4 + 94b0d32e; v1 and v2 refuted by Opus, Codex findings fixed). App: a node's
      own forward-only bar drives its share and its pace drives minutes left (e53c6cdf, c5194eac).
      Live: Twelfth Night (Malvolio's letter), 1 act foley, 67 min 34 s, obs_publish OK,
      `shadows_box_20261019_204048_..._final.mp4` (82 MB, 1920x1080, 1m45s), sha 66376b14, both
      places. Bar vs clock every 5 min: 0, -3, -3, -3, -2, -1, 0, 0, 0, 0, 0, 0, +1, -3; **worst 4
      points** (was 15 and 17). Minutes left within 2 from minute 20 on; 7 at the very start. **Never attach a second listener to the job's SSE** -- read the
      app's own output or its `/api/episodes/<id>/events`.

## Phase 2 -- quality budget (target Oct 25)

Story and music are FROZEN (CLAUDE.md 2026-08-04, 2026-09-17). Do not chase prose or score.

- [x] Comedy demo candidate rendered: A Midsummer Night's Dream, Bottom and Titania
      (`--source-ref folger-midsummer:act3-scene1-bottom-titania`), 1 act foley, above. Awaits the
      operator's eye.
- [x] **Demo episode, classic half: OPERATOR PICKED MIDSUMMER (2026-10-19).**
      `output/otr/obs/bird_song_20261019_191143__cart__n16f__fk3i__koko__sspr__cg412__sa3_final.mp4`
      (A Midsummer Night's Dream, Bottom and Titania; 1 act foley; 71 MB, 1m45s; sha 0c8953bd).
      Keep it: do not re-render or overwrite. The demo shows a highlight cut of it (ffmpeg, no render).
      CUT MADE: output/otr/episodes/demo_cuts/midsummer_highlight.mp4 (39.5-66 s of the episode,
      26.6 s, Titania wakes -> Bottom's song -> "I love thee"; 0.5 s fades). Its stills are FLUX 3
      Klein (rendered before the MOSS switch). Eye check: the title cards show glitchy lettering and
      two frames (0:36, 0:52) flash as negatives -- intended Signal Lost glitches, or a defect?
      Form + demo drafts: apple/CONTEST_SUBMISSION_DRAFT.md, apple/CONTEST_DEMO_SCRIPT.md.
- [ ] **Demo episode, My Story half: RENDERING (restarted 23:46 on 10-19, 3 acts, foley, MOSS).**
      The 22:53 run was canceled at the stills stage (log kept as
      tmp/rps_live/mystory_magnolia_generic_canceled.log) so Jeffrey could be drawn from the
      operator's own photos -- he is the author and the character is him, so this is his own
      likeness, not a third party's.
      Operator: "make my story the fictional tale of Jeffrey and MagnoliaRex, the later chapters
      ... it's my published book, I own the rights" (his book "Mime Lung Music Festival", Part 2;
      the book's own disclaimer lists the fiction chapters). He approved the four boxes as drafted.
      CONTEST CARE: MagnoliaRex is a thin disguise for a real musician, so the boxes carry NO real
      names (no artists, no Nam June Paik, no real artworks) and describe her only by the text's
      personality. The exact boxes are in the launcher scratchpad script mystory_magnolia.sh and
      the run log tmp/rps_live/mystory_magnolia.log. When it lands: verify, then check the stills
      and script for any real name before it goes in the demo.
- [ ] Compare by eye on the same shot and seed: foley (default) vs `otr_16gb_video.json` vs a
      procedural lane. LTX stays core if it holds up (LTX bonus).
- [ ] Stills: FLUX 3 Klein is the shipped NVIDIA engine; confirm it looks right on the episode.
- [x] **MOSS-Image A/B vs Klein -- DECIDED: "MOSS everywhere" (operator, 2026-10-19, by eye).**
      Same scene (Midsummer, Bottom and Titania), same style (Shakespeare stage realism), 8 stills
      each, sheet tmp/rps_live/ab/klein_vs_moss_midsummer_shakespeare_stage_realism.jpg. His reason:
      Klein's faces carried big black smudges he did not like (Titania, b003 and b005). Stills
      episode 7 min on MOSS vs 4.7 on Klein. The six otr_16gb_* graphs switched on main 7a8bdce8;
      8 GB and 24 GB stay Klein. FOLLOW-UP: check whether Klein's smudges come from the model or
      the prompt (a dirt/mud appearance detail) on the lanes still using it.
      Engine built and pushed opt-in (main 49fc8eb0, Sonnet holds); the app's "Still pictures"
      picker sets it (e0b2d329). Next: stills-only 16 GB workflow, same scene and pinned seeds,
      klein vs moss.
      Apache-2.0, native in ComfyUI core since v0.42.0 (comfy/ldm/moss_image), repackaged at
      hf.co/Comfy-Org/MOSS-Image: int8-convrot DiT 5.4 GB, Qwen3.5-9B text encoder w4a8 7.3 GB,
      flux2-vae_small; 8-step DMD LoRA (lodestones) for ~3-4x speed at CFG 1.0. NOT smaller than
      Klein on 8 GB (Klein proven 7.5 GB peak on the 4060); "4 GB" claims mean GGUF, which stays
      out (auto-download ruling). Needs a new engine in nodes/_otr_image_engines/ (pinned tokenless
      fetch, registry, fixtures), so it is a build, not a setting. Caution: it knows celebrities and
      brands, and the contest forbids likenesses/unlicensed IP. Source: operator-shared AI Search
      video cUeZMUGCxrw, 2026-10-19.
- [ ] 1080p upscale on cuda and CaptionBurn on: check what the foley workflow ships
      (OTR_SilentComposite upscale_engine, OTR_CaptionBurn) and decide with the operator.
- [x] **Face consistency -- ANSWERED 2026-10-19 23:45: option (2), by text.** The operator sent
      ten photos of himself and chose "restart now" with Jeffrey described from them: a mime, lean,
      long straight nose, wide grin, wavy light-brown hair, white face paint with small red cheek
      marks, black knit beanie, blue-and-white checkered satin costume. No photo is fed to any
      model; the words go in the Characters box.
      THEN (23:50, operator): "all lanes should generate ref images for char stills so we have
      consistency" -- reference images become a build for every lane (row below). Original
      question: Peek at the
      MagnoliaRex stills: she holds steady, Jeffrey's face drifts between beats, and "her dark gray
      cat" was drawn as a big cat. Today's consistency is text + a portrait-derived identity SEED; the
      dispatcher can pass the portrait as a REFERENCE but every local engine has it off (z_image:
      grid artefact, measured 2026-08-20; flux3_klein: supports 10 refs, unwired/unmeasured;
      moss_image: no reference mode). Options: (1) wire Klein reference mode for character stills
      only + A/B (~half a day; Klein faces again); (2) keep MOSS and sharpen the weak character's
      locked appearance text (a correctness fix per CLAUDE.md). Default until he answers: judge the
      finished video first; build nothing.
- [ ] **REFERENCE STILLS FOR EVERY LANE (operator, 2026-10-19 23:50 -- his main point).** "All
      lanes should generate ref images for char stills so we have consistency." Not a one-off for
      the demo: "we can use the photos if we give an architecture for other people to use their own
      photos too ... it's not a sleight-of-hand hack."
      WHAT EXISTS: every lane already mints a canonical PORTRAIT per character (the 3-act run made
      c01-c04 + announcer on MOSS), and the dispatcher already hands that portrait as
      `reference_image` to any engine advertising `accepts_reference_image`
      (otr_image_gen_dispatcher.py ~1954). No local engine advertised it, so every still was text +
      identity seed only -> Jeffrey's face drifted.
      ENGINES: MOSS-Image has NO reference path in core (MOSSImage.extra_conds is text only; no
      open edit/ref MOSS weights on HF as of 10-19). FLUX 3 Klein takes reference latents natively.
      DONE: main 86e821bd -- Klein reference chain (LoadImage -> ImageScale 768h -> VAEEncode ->
      ReferenceLatent on both conds), opt-in `OTR_FLUX3_REFERENCE=1`, default off = byte-identical
      t2i graph. Sonnet QA HOLDS (minor: ref height not in the cache key; staged otr_ref_* files
      accumulate in ComfyUI/input). Test hygiene fix main 8c511b8c.
      BUILT OVERNIGHT 10-20 (all pushed; Sonnet QA on every commit):
      * main e815e37d + d4d994bc -- CHARACTER PHOTOS: OTR_ImageDirector trailing widget
        `character_photos` ("Name = file" in ComfyUI/input) -> image policy -> MetaBrief matches the
        name to the cast ONCE and tags the portrait `identity_photo` -> the dispatcher uses the photo
        as the PORTRAIT's reference (portrait painted in the episode style; raw photo never in the
        episode; photo sha in the cache key). Unusable photo = LOUD report. Design: Codex contrarian
        (BROKEN on draft 1, every grounded point folded in).
      * same commit -- PORTRAITS ON EVERY LANE where the still engine takes a reference: the
        portrait-free exemption (still_flat/pan/motion/word, ltx_8gb, cloud) no longer applies to a
        reference-capable still engine. Measured live first: a still_motion run minted 23 stills, all
        reference=none, before this.
      * main c41dbe25 -- CO-REFERENCES: each still also takes the portraits of the other cast members
        its prompt names (up to 3); Klein chains one ReferenceLatent per portrait.
      * main d1414a7d + 9b0ed9ea -- NON-HUMAN characters keep their body: _ensure_gender_anchor put
        "adult woman, " in front of "an Egyptian-type adult cat" (the cat-woman / cat-as-a-man bug).
        Head-noun rule after a Sonnet BLOCKER ("a cat burglar" stays a person).
      * branch e5a18331 + bd4ce4ae + 861a2b5e -- APP CAST CARDS: one card per character (name, voice
        he/she/any, who they are, how they look, optional photo) after Veo-Moovie-Maker's Cast Bank;
        "Load the example" = Jeffrey, MagnoliaRex, Cat with the operator's own Veo descriptions;
        POST /api/photos stores the photo in ComfyUI/input by content hash. README updated.
      LIVE PROOF 1 (00:14-00:30, 1-act still_motion, flux3_klein + OTR_FLUX3_REFERENCE=1, the example
      cast, a STAND-IN photo = the generic MOSS Jeffrey portrait, never the operator's pictures):
      obs_publish OK. Sheet tmp/rps_live/ab/photo_reference_proof.jpg (sent to the operator). Photo ->
      portrait kept the face; Jeffrey held in 7/7 of his stills, MagnoliaRex in 7/7 of hers; the
      OTHER person in a two-shot drifted (-> co-references) and the Cat was humanoid (-> non-human fix).
      NOW: LIVE PROOF 2 rendering (tmp/rps_live/photo_proof2_1act.log, same inputs, with both fixes).
      THEN: if it holds, re-render the 3-act MagnoliaRex demo with the example cast on flux3_klein +
      references (that IS the demo of the feature).
      OPERATOR DECISION (morning): how references switch on for everyone. Today it is the opt-in
      server env OTR_FLUX3_REFERENCE=1 (default off = both boxes byte-identical). Options: (a) flip
      Klein's default ON -- reaches the 8 GB lane, needs a 4060 VRAM measurement first (no 4060
      window was live overnight); (b) a separate dropdown choice "flux3_klein with references"
      (~10 registry/fixture edits, zero blast radius); (c) leave it opt-in. MOSS (the 16 GB default,
      his pick by eye) has no reference path at all, so a photo there is reported unused.
      CONTEST RISK: rules say "no real-person likenesses" -- the feature is fine; a demo built on his
      own face is his call. The demo uses written descriptions, not his photos.
- [ ] Voice engine per role (OTR_CastLock rows in `config/app_mode.json`): operator's ear decides.
- [ ] Any winning change to a shipped workflow: edit `workflows/otr_canonical.json` on main, then
      `python scripts/build_variants.py --all` and `--check`, plus the four section-0 tests.
      Pickers the app lacks go in `FIELDS` in `contest/radio-play-studio/radio_play_studio/form.py`.

## Phase 3 -- demo and submission (Oct 26-29)

- [ ] Record the demo: the app (both cards: a classic, and Tell my own story), the progress bar
      with minutes left, then the finished episode.
- [ ] Name every model and tool in the video: writer Gemma 4 12B (comfy_native int8); FLUX 3 Klein
      stills; LTX 2.5 foley (mix4x8 DiT) video; Kokoro voices; Stable Audio 3 music; the upscaler
      the workflow uses; comfy-sdk 0.5.0; comfy-api-proxy 0.1.6; ComfyUI 0.42.1; the OTR pack.
- [ ] Why it is new: one press yields a cast-locked, shot-locked radio play on local open weights.
- [ ] README final pass (`contest/radio-play-studio/README.md`).
- [ ] Submit through the Typeform. **The operator submits** (it is his entry and his account).

## Rules for any session here

- Every code change: scoped suite green, push to the branch, then Sonnet QA briefed to REFUTE
  the pushed diff; a finding is the next commit.
- Shared pack code changes must show the 4060 path is unchanged (CLAUDE.md section 0B).
- Never two renders at once on the 5080. Before starting one, check `:8000/queue` is empty.
- Report: commit hashes, RESULT + obs_publish OK + file path, render time, what the app got wrong.
