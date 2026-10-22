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
  Push here. MERGED INTO MAIN 2026-10-20 (operator: "merge the contest branch into main now so
  the code link does not move"); keep pushing app work here and merge again when it is green.
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
- [x] **Demo episode, My Story half: DONE 10-20 05:18 -- WITH REFERENCE STILLS.** "The Last Loop of
      the Afterlife", 3 acts, 5:29, 16 GB foley, flux3_klein + OTR_FLUX3_REFERENCE=1 (main 2b30170e
      loaded), C7 seeds, the app's example cast cards (the operator's own Veo-Moovie-Maker
      descriptions of Jeffrey, MagnoliaRex and the Cat), NO photo. Render 4:19:00 (01:00-05:18; the
      reference stills add little -- the LTX video stage is the hours). Verified: RESULT succeeded,
      obs_publish OK, the app's copy (contest/radio-play-studio/episodes/c869a6266614/) and
      otr/obs/afterlife_loop_20261020_010758__cart__n16f__fk3i__koko__myst__cg412__sa3_final.mp4
      byte-identical (cmp), timing saved (episodes/timing/otr_16gb_foley.acts-3.f16920c1.json).
      Stills: 31 mints, 21 drawn with co-references, no LOUD reports. By eye (frame sheet
      tmp/rps_live/ab/demo3_frames.jpg): Jeffrey (stocky, round face, pink shirt) and MagnoliaRex
      (star-speckled hair, purple/gold jacket) hold in EVERY frame; the Cat is a real small black
      cat in Athens but a humanoid cat figure in act 2 (the open KIND row). Real-name check: captions
      and credits name only Jeffrey, MagnoliaRex, Cat; the boxes carried no real names (the ledger
      and stills were removed by the workflow's Space saver after publish).
      Highlights (output/otr/episodes/demo_cuts/): magnolia_highlight.mp4 = 0:30-0:56, Athens, act 1
      (both faces, no Cat; sent to the operator) -- the demo-script cut; magnolia_highlight_ending.mp4
      = 3:25-3:51 (the duet and the countdown; shows the humanoid Cat at ~0:20).
      History: the 22:53 (generic) and 23:46 (mime look from his photos) MOSS runs were canceled
      when the operator made reference stills his main point; operator: "make my story the
      fictional tale of Jeffrey and MagnoliaRex ... it's my published book, I own the rights".
- [ ] Compare by eye on the same shot and seed: foley (default) vs `otr_16gb_video.json` vs a
      procedural lane. LTX stays core if it holds up (LTX bonus).
      10-20 05:30-07:23: the video-lane side RENDERED and verified -- act 1 of the same example story
      (cast cards, flux3_klein + references, C7 seeds) on otr_16gb_video: 2:38, render 1h49m,
      obs_publish OK, both copies byte-identical, timing saved (otr_16gb_video.acts-1.f16920c1.json).
      otr/obs/last_circuit_20261020_053349__shst__l25v__...; frame sheet
      tmp/rps_live/ab/compare_video_lane_frames.jpg (sent). It rolled stage-realism (the foley demo
      rolled comic art), so compare the motion and sound, not the look. By eye: MagnoliaRex holds, a
      real gray cat at 0:30, but the duet frame (~1:30) shows two strangers. OPERATOR'S EYE DECIDES
      foley vs video; the procedural third arm is not rendered (say if wanted).
- [x] Stills: superseded -- MOSS-Image is the 16 GB default (operator, 10-19), and FLUX 3 Klein is
      the reference-stills engine; both looked right by eye on their episodes (Midsummer, MagnoliaRex).
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
- [x] 1080p upscale on cuda and CaptionBurn on: check what the foley workflow ships
      (OTR_SilentComposite upscale_engine, OTR_CaptionBurn) and decide with the operator.
      CHECKED 10-20 16:30: BOTH ALREADY ON in workflows/otr_16gb_foley.json -- node 84
      OTR_SilentComposite [1920, 1080, 25, '', 'rtx_vsr', 'cuda'] and node 86 OTR_CaptionBurn
      [True, 'sdh_standard', 25, ...]. The verified 15:58 demo measures h264 1920x1080 25 fps with
      captions burned in (visible in every highlight he has watched). Nothing to change; if he
      wants a caption-free cut for the video, that is one widget (OTR_CaptionBurn enabled=False).
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
- [x] **REFERENCE STILLS FOR EVERY LANE (operator, 2026-10-19 23:50 -- his main point).** "All
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
      * NON-HUMAN characters -- BUILT 10-20 (operator: "do it here"; "person? how about human?"):
        main 2da9297e + f77dca36. kind = human | animal | machine | creature on the cast row
        ("person" reads as human), written by My Story's treatment, kept by the ledger only for a
        non-human, read by every picture prompt; every row without it is byte-identical (Sonnet
        HOLDS; Codex r2 BROKEN on the first plan, all six points folded in). My Story's act-writer
        JSON now shows "kind": "human" per cast member (My Story only). The generic casting and
        scifi_news_pro writers do not write kind yet -- an animal there is still drawn as a person.
        LIVE PROOF PASSED 07:29-07:41 (1-act still_motion, flux3_klein + references, the example
        cast): obs_publish OK, copies byte-identical, timing saved. Sheet
        tmp/rps_live/ab/kind_proof.jpg (sent). The Cat is a real four-legged gray cat in every beat
        (its portrait is a cat's face in a period robe -- the stage-realism style's costume look);
        Jeffrey and MagnoliaRex hold everywhere, and the man beside MagnoliaRex is now Jeffrey in
        all six of her stills (the script-partner co-references, 2b30170e, proven live).
      * main 807e8e07 + 089e48e6 -- SEVERAL PHOTOS PER CHARACTER (operator: "feel free to make an
        amalgamation"): Name = a.png, b.png (up to 3); the first is the portrait's reference, the
        others ride along as further references; each hash joins the cache anchor (Sonnet HOLDS).
      HIS PHOTOS (operator 10-20 08:2x: "please use my images as a ref image"; he is the author and
        the character is him, and he chose this knowing the "no real-person likenesses" rule):
        six photos in C:\Users\jeffr\Pictures\otr_jeffrey_ref\ (kept OUT of the public repo).
        Codex (gpt-5.6-sol, its image_generation tool -- a CLOSED model; the reference is an input
        picture, the pipeline stays open weights; say so if entering Open Weights Only) blended
        them into jeffrey_reference.png (front-facing, mime paint, plain black shirt, grey
        background, no logos). Jeffrey's references = that + two clean crops (Barcelona mime,
        barbershop close-up); his "How they look" describes the photos.
      * main b6408675 + ed9789f9 -- CHARACTER PHOTO NODE (operator: "is that all best practice for
        comfy workflows?"; then "build it ... design review first"). Photos already landed where
        ComfyUI puts user images (input/, via its own /upload/image; core's opt-in asset system would
        tag them input). The canvas side is now Comfy-native: OTR_CharacterPhoto = name + ONE image
        picker declared like core's LoadImage (upload, paste, preview), chained one node per photo
        (up to 3 per name) into Image Director's appended character_photos_in socket; the typed
        lines win for a name both give. Canonical ships one BLANK node (98) wired in. IS_CHANGED
        hashes the picture. Design: Codex contrarian (REFUTED draft 1; one picker per node, core's
        annotated paths, IS_CHANGED, typed chain, wired blank node). Live canvas check on a CPU
        ComfyUI (port 8001): a blank node loads blank and sends {name:"", image:""}; the frontend
        SAVES the upload button as one more slot ("image") -- every converter and guard now reads it
        (ed9789f9; app f138851c). Sonnet HOLDS on b6408675.
      HIS PHOTOS MAY SHIP PUBLIC (operator 10-20: "my photos can be public and ship with it ... it's
        my story"). NEXT, if the demo is good: make his true Jeffrey/MagnoliaRex story My Story's
        built-in default (the cast cards + the three boxes) with Jeffrey's photos bundled.
      MAGNOLIAREX REFERENCE -- FINAL LOOK (operator 10-20 10:35: "this is the final magnolia rex";
        "remove the title"; then "make the outfit whitish like a magnolia, white silver not yellow").
        Codex made three references from his final art (long wavy honey-brown hair, pastel pink/teal
        streaks, curtain bangs, high collar, glowing petal shoulders), title text and DJ gear left out,
        then re-coloured the outfit magnolia white + silver, no gold. Set used:
        C:\Users\jeffr\Pictures\otr_magnoliarex_ref_c\ (front, three-quarter, dark-violet portrait;
        sheet sent, operator: "perfect"). Her look text in the launcher matches it. The earlier
        platinum and gold sets (otr_magnoliarex_ref, _ref_b) are superseded.
      [x] VERIFIED 15:58 -- 3-act demo with BOTH leads' references: RESULT succeeded, obs_publish OK,
        5:15:25 render, 5:48 episode, sha f5524300, app + obs copies byte-identical (cmp), timing
        saved (episodes/timing/otr_16gb_foley.acts-3.5ab87708.json; video 15,766 s of 18,558 s).
        Faces hold in motion as in the stills. Highlight: output/otr/episodes/demo_cuts/
        magnolia_both_refs_seoul.mp4 (238-298 s, the Seoul ending; sent). Supersedes the 05:18
        demo.
      [x] HIS STORY IS MY STORY'S BUILT-IN DEFAULT (operator 10-20: "Yes, make my story the default").
        main 20f8c34b: a blank My Story run (blank node, cleared field, rolled pick) writes
        DEFAULT_STORY -- his Jeffrey/MagnoliaRex/Cat cast, plot and setting, the demo's exact
        words, the plot without act numbers so a 1-act roll is not handed three labelled acts --
        credited to Jeffrey A. Brick even when only an author is typed; held to its source like
        typed fields (house_source no longer switches fidelity off); licence label "the author's
        own story, shipped with his permission". His six photos ship at
        nodes/story_packs/my_story/photos (0.9 MB, 768 px); "name.jpg [bundled]" resolves there,
        confined like input/output/temp; the house story uses them when no photos are given
        (given photos win; a typed story never gets his). App (this branch): "Load the example"
        fills the new text and both leads' three photos, thumbnails from a confined
        /bundled-photos/ route. Codex contrarian on the design: BROKEN first, every point folded
        in. Scoped suite 5533 passed; app 124 passed. Sonnet QA HOLDS (no blocker; its docstring
        nit fixed on main).
      [x] REFERENCE STILLS AS A SHIPPED CHOICE (operator 10-20: "use flux 3 ... we want ref image for
        my story as an option for this contest"; "it's our 16gb foley flux3 ref in my story json";
        "same basic json but these new dropdowns selected by default"; "leave the prior 16gb foley
        with no ref"; must not touch the non-reference lanes). MOSS cannot take references (core
        has no path; ostris's MOSS edit node needs an edit LoRA nobody has published -- only a
        1-of-6 proof of concept). main d1b3586d:
        * flux3_klein_ref -- FLUX 3 Klein with references always on (same files and recipe;
          at most 3 references a still, the measured count; never drawn by a roll).
        * workflows/otr_16gb_foley_flux3_ref.json -- foley with character stills on
          flux3_klein_ref, announcer/music on flux3_klein; diff from foley = those 3 dropdowns.
          otr_16gb_foley unchanged; every graph lists the new engine and keeps its own pick.
        * App (this branch): a My Story run uses the workflow's "_flux3_ref" twin when it ships
          beside it; every other story runs the workflow as named.
        Strategy readers before code: Codex + agy Gemini 4 Argon (both NEEDS CHANGES, all folded;
        pinning source_bank=my_story declined -- unmanaged by design, the app selects it).
        Full suite 18011 passed; app 126 passed. Sonnet QA HOLDS (minors only: the twin's first run
        has no ETA history; a ComfyUI started before d1b3586d rejects the twin until restarted).
      LIVE NOW: blank 1-act My Story on the new code (started 17:28, references on via the boot
        switch; tmp/rps_live/mystory_default_1act.log): "house DEFAULT_STORY ... by 'Jeffrey A.
        Brick'", "house My Story: the author's shipped photos of Jeffrey, MagnoliaRex", character
        stills painted from references. The new co-reference log shows the WRITER names the Cat
        in its scene prompts (so the Cat-in-Athens is the writer's, not a stray word).
        STILLS MEASURED 18:57: 22 stills, nothing typed -- Jeffrey (mime) and MagnoliaRex (white-
        silver) drawn from the SHIPPED photos and steady in every character frame, the Cat steady
        too; only the announcer (no reference) is a stranger. Sheet sent.
      [x] VERIFIED 19:31 -- the blank 1-act default-story run: RESULT succeeded, obs_publish OK,
        2:02:57 render, 2:26 episode at 1920x1080, sha 2796acb3, app + obs copies byte-identical,
        timing saved (episodes/timing/otr_16gb_foley.acts-1.5ab87708.json). The act-neutral plot
        fit one act. It ran on otr_16gb_foley with the boot switch (it started before the twin
        shipped); the twin otr_16gb_foley_flux3_ref is what My Story uses from now on. Highlight:
        output/otr/episodes/demo_cuts/default_story_1act_highlight.mp4 (sent).
      TWIN PROOF (19:32, PLAIN boot -- no OTR_FLUX3_REFERENCE, no pinned seeds): the app, given only
        --bank my_story --acts 1, picked workflows/otr_16gb_foley_flux3_ref.json by itself
        (validator stamp OK, widget drift 0); the house story ran; 16 character stills painted
        from references, 9 announcer/music cards without (by design). Video rendering
        (tmp/rps_live/mystory_twin_1act.log, server_plain.log).
      [x] TWIN VERIFIED 22:07 -- otr_16gb_foley_flux3_ref on a plain boot: RESULT succeeded,
        obs_publish OK, 2:34:32 render, 3:00 episode, sha a66c37ff, copies byte-identical, timing
        saved (otr_16gb_foley_flux3_ref.acts-1.5ab87708.json).
      CONTEST STACK DECIDED -- FULL OPEN WEIGHTS (operator 10-20: "we skip VSR, we stay open
        weight"; "flux 3 and esrgan"; "we skip the ref images"). main 4c43ea10 adds
        workflows/otr_16gb_foley_open.json = otr_16gb_foley with FLUX 3 Klein stills from text for
        every role and the open Real-ESRGAN x2 upscale (spandrel_esrgan) instead of NVIDIA RTX VSR;
        diff from foley is exactly those four widgets + validator stamps. The app defaults to it
        (f3a20a1a); it has no reference twin, so My Story runs it too. The reference workflow
        (otr_16gb_foley_flux3_ref) still ships as an option, outside the contest. Submission draft
        ticks Open Weights Only.
      SUPERSEDED 10-21 (operator: "yes, switch to off"; "I want to be 100% open"): main b0f9a6e4
        drops Real-ESRGAN from otr_16gb_foley_open -- the 1080p step is now "off" (ffmpeg Lanczos +
        unsharp, no model), the most faithful of the three in apple/UPSCALERS.md. Real-ESRGAN was
        running both times the machine went down. The tiled/fp16 ESRGAN fix is shelved on
        wip/esrgan-tiled-fp16 (Codex NEEDS CHANGES), post-contest. Sonnet QA on b0f9a6e4: HOLDS.
      REFERENCE STILLS BACK, STILL OPEN (operator 10-21: "use reference photos. Make an open-weights
        reference twin (FLUX 3 ref with the ffmpeg 1080p step, no VSR) and render the demo on it"):
        main a9244e8a adds workflows/otr_16gb_foley_open_flux3_ref.json -- otr_16gb_foley_open with
        character stills on flux3_klein_ref; diff is that one VideoDirector widget + validator
        stamps. The app's reference_twin() picks it for My Story by itself. He chose this knowing
        the "no real-person likenesses" rule (the photos are his). Scoped suite 2454 passed,
        build_variants --check 28 variants 0 failures, app tests 126 passed.
      [x] DEMO VERIFIED 14:36 (10-21) -- the twin, 3 acts, blank My Story (built-in story + its
        bundled photos), plain boot, new NVIDIA driver: RESULT succeeded, obs_publish OK, Prompt
        executed in 5:25:57, episode 6:17 at 1920x1080,
        seoul_duet_20261021_091839__vstb__n16f__fk3r__koko__myst__cg412__sa3_final.mp4 (sha
        4f6a9a92), episodes/4a38607f7f0d copy byte-identical, timing saved
        (otr_16gb_foley_open_flux3_ref.acts-3.5ab87708.json). No crash on the new driver. Stills
        ~33 min (vs ~10 from text), filming ~4.6 h for 64 beats; the ffmpeg 1080p step took minutes.
      CLOSED 10-22 (check-in): built, shipped and proven. Portraits on every reference-capable
        lane, co-references, the general own-photo path (Name = a.png, b.png in the cast card),
        the shipped flux3_klein_ref choice, the open-weights twin the app picks for My Story,
        and the 3-act MagnoliaRex demo re-rendered on it (seoul_duet_20261021_091839, above).
        The demo video and range reel are cut from it. No re-render is owed.
      [x] FLATLAND 1-ACT (10-21 17:11) -- My Story with Abbott's Flatland (public domain, 1884) as
        plot/characters/setting, rolled video_art: RESULT succeeded, obs_publish OK, 2:09:59, 2:53
        at 1920x1080, three_dimensions_20261021_150516__vart__...final.mp4 (sha 9ea1d189), app copy
        byte-identical, timing saved (open_flux3_ref.acts-1). No digital glitch: the glitch seen on
        My Story is that story's own digital setting, not a style leak (0 hits of the video_art /
        recur_frac pack words in the demo run's prompts; "digital" 176, "glitch" 24, all story text).
      STYLE SWEEP, 3 ACTS EACH (operator 10-21: "run the 3 acts, my story and flatland, a few runs
        for a few days and see what we get"; "not a fan of origami"): Flatland on
        storybook_engraving (started 17:13), then tmp/rps_live/style_queue.py (detached, log
        style_queue.log) runs one at a time: My Story anime, Flatland archival_documentary, My Story
        sci_fi_radio, Flatland shakespeare_stage_realism, My Story cartoon, Flatland recur_frac,
        My Story video_art, Flatland visual_storybased. Styles are set, not rolled.
      [x] FLATLAND 3-ACT, storybook_engraving (10-21 22:27, laptop): RESULT succeeded, obs_publish
        OK, 5:14:05, 6:49 at 1920x1080,
        three_dimensions_20261021_171919__sbke__n16f__fk3r__koko__myst__cg412__sa3_final.mp4 (sha
        7ad47109), app copy byte-identical, timing saved (open_flux3_ref.acts-3).
        MOVED TO RUNPOD (operator 10-21: "spin up a volume and the runpod, I give approval"; "you
        can run my story jeffrey and magnolia on runpod too in parallel"): all eight style runs
        (Flatland archival_documentary / shakespeare_stage_realism / recur_frac / visual_storybased;
        My Story anime / sci_fi_radio / cartoon / video_art) run on pods in parallel from one
        network volume, driven by the session "RunPod: Flatland 3-act style sweep in parallel";
        episodes come back to otr/obs and every pod is stopped after. The laptop queue is stopped;
        the laptop finishes Flatland storybook_engraving, then re-renders the demo video's scenes
        in the new palette (black/purple, yellow, aquamarine; operator: "orange isn't good") and
        re-cuts it (tmp/rps_live/after_flatland3_demo.sh).
      [x] MERGED INTO MAIN 6364c3df (operator: "merge the contest branch into main now so the code
        link does not move"): https://github.com/jbrick2070/ComfyUI-OldTimeRadio/tree/main/contest/radio-play-studio
        resolves. No node code, no pyproject.toml, contest/ excluded from the published pack.
      NOW: THE DEMO RENDER on that exact workflow (operator: "we need another run with ESRGAN for
        our demo to be precise"; "3 acts, same as the approved demo"): started 22:33, blank My Story
        (his default story), --acts 3, tmp/rps_live/demo_open_3act.log, validator stamp OK
        (profile otr_16gb_foley_open, drift 0). ~5 h. Then cut the under-2-minute video from it.
      DEMO VIDEO BUILD (operator: "make a video in the same style as their intro video ... 3D intro
        into walking through the nodes and videos appear"): storyboard apple/CONTEST_DEMO_STORYBOARD.md
        (agy Gemini 4 Argon reviewed: pacing, honest app shot, LTX hero beat, a flat name grid -- all
        folded); Blender 5.2 scenes, Kokoro narration (146 words) and headless-Chrome app shots in
        contest/radio-play-studio/demo_video/. CPU test frames OK. Full EEVEE render is queued to
        start the moment the demo render ends (never two GPU jobs); then the live progress-bar shot,
        the excerpt, and the ffmpeg cut to output/otr/episodes/demo_cuts/radio_play_studio_demo.mp4.
      !! DEMO RENDER LOST 10-21 ~07:45 -- GPU RESET (operator: "I think I had to reset my GPU"):
        server_plain.log ends "comfy-aimdo WDDM VRAM query failed" -> "CUDA error: unknown error" ->
        "Fatal Python error: Aborted" at upscale seg_0034 of ~60 (video stage was complete). Not a
        pack bug. Operator is installing new NVIDIA drivers and rebooting.
      RESUME AFTER THE REBOOT, in this order (one GPU job at a time):
        1. Boot ComfyUI plain: powershell -File tmp/rps_live/restart_plain.ps1 (port 8000).
        2. Profile the upscaler (minutes): python tmp/rps_live/profile_esrgan.py
           <a clips/*character_video*.mp4 of signal_lost_the_last_circuit_20261020_224235> 48
           -> tmp/rps_live/profile_esrgan.log. If fp16 or batching is a clean win (same pixels
           within a tight PSNR, lower time), fix eng_spandrel_esrgan on main FIRST (tests, push,
           Sonnet QA) so the rerun's composite takes minutes, not hours.
        3. Rerun the demo: from contest/radio-play-studio,
           python -m radio_play_studio run --comfyui http://127.0.0.1:8000 --bank my_story --acts 3
           (default workflow otr_16gb_foley_open) -> tmp/rps_live/demo_open_3act.log.
        4. Then the motion graphics: output/otr/episodes/demo_cuts/build/ (scenes.py per scene,
           blender -b --factory-startup --python scenes.py -- <type|title|cards|nodes|ring|end>),
           the live progress-bar shot (shoot_app.py progress, during a run), the Seoul excerpt,
           Kokoro narration (narration/*.wav), and the ffmpeg cut.
      [-] (MOOT 10-21, Real-ESRGAN left the contest graph) REAL-ESRGAN WAS THE BOTTLENECK (measured 10-21): the video
        stage ended ~04:15 (56 beats); the composite then upscales at ~0.5 frames/s (seg_0003: 183
        frames in 372 s; seg_0004: 249 in 493 s) with the GPU ~12% busy -- about 6-7 h for a 3-act
        episode, publish ~11:00-11:30. Read of the code: spandrel runs fp32, one frame per call
        (eng_spandrel_esrgan.upscale_frames), x2 to 3328x1920 then a GPU bicubic fit to 1920x1080
        (_pipeline._fit_and_pad_bhwc) and a host copy per frame. GPU is not saturated, so the model
        is not the limit. NEXT, once the GPU is free: profile one segment (decode / to-device / model
        / fit / host copy / encode), then fix the root -- candidates: half precision where spandrel
        reports supports_half, batching frames through the module, overlapping decode/encode with
        compute -- same pixels out (A/B by PSNR on a held segment), Codex + Sonnet QA.
        Sonnet QA on 4c43ea10 + the app switch: diff holds, card holds; its one finding -- My Story
        no longer uses reference pictures on the default -- is the operator's decision, so the fix
        is wording: README and draft Q9 now call references an optional workflow. Real-ESRGAN
        rehearsed on CPU with a 1664x960 foley frame: engine, output check and 1920x1080 fit OK.
      4060 (MRKT) -- ONE INSTALL, ON MAIN, REGRESSING (operator 10-20: "we will still need to regress
        other workflows, the 8gb machine is ready"; "push the latest for sure"; "you are free to
        clean it up so there's only one").
        * Kept: Comfy Desktop (C:\Users\jeffr\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI,
          core 0.41.3) with OTR as a git checkout of main d1b3586d (was registry 2.3.28); models in
          Comfy-Desktop\ComfyUI-Shared\models, which OTR_COMFYUI_MODELS_ROOT now points at.
        * Removed after moving every model file not already in that store (17 files, 158.7 GB:
          MiniMax H3 ref2va/fl2va, LTX 2.5 int8/nvfp4, Ideogram 4, HuMo, FLUX.1-dev, AnimateDiff):
          C:\OTR-Human-2.3.10 (portable 2.3.11, core 0.37), C:\OTRCold (registry 2.3.22),
          C:\OTR-CleanRoom, D:\OTR-ColdDrill-aside-20261008 (658 GB of nested drills), and the
          10-13/10-18 QA installs under D:\otr-4060-testing; 51 stale scheduled tasks. Free: C 285 GB,
          D 904 GB. Kept D:\otr-4060-testing\ComfyUI-OldTimeRadio (the 4060 window's checkout,
          clean, fast-forwarded to main) and its reports.
        * Boot: C:\Temp\otr_one_server.cmd (task OTR_one_server, port 8188; Desktop's own
          model-paths yaml + C:\Temp\otr_one_paths.yaml for the wrapper packs and AnimateDiff).
          ComfyUI-GGUF fails to import there -- irrelevant (GGUF is out).
        * RUNNING 20:27: all seven shipped 8 GB graphs, 1 act each, one at a time, fastest first
          (low, still, animatediff, video, ltx25_foley, ltx25_mime, ltx25_audio_in), driver
          C:\Temp\otr_regress_8gb.py (task OTR_regress_8gb), log C:\Temp\otr_regress.log. They all
          use plain flux3_klein, so they prove today's Klein refactor left the 8 GB path unchanged.
          [x] otr_8gb_low -- SUCCESS 514 s, obs_publish OK (shadow_pouch_..._vcam, an original story).
          [x] otr_8gb_still -- SUCCESS 931 s, obs_publish OK (afterlife_echoes_..._stmo_fk3i): it rolled
              My Story, so the new house default ran on 8 GB too (photos matched; plain Klein paints
              from text, by design).
          [x] otr_8gb_animatediff -- SUCCESS 2825 s, obs_publish OK (last_duet_..._adhv, My Story).
          [x] otr_8gb_video -- SUCCESS 624 s, obs_publish OK (wrestling_system_..._lx8g_fk3i).
          [x] otr_8gb_ltx25_foley -- SUCCESS 3611 s (published, see obs on the 4060).
          [x] otr_8gb_ltx25_mime -- SUCCESS 3031 s. All 8 GB graphs upscale with RTX VSR, so VSR is
              proven on the 4060 too.
          [x] otr_8gb_ltx25_audio_in -- SUCCESS 3504 s.
          ALL SEVEN 8 GB GRAPHS PASS on main d1b3586d+ (00:37): today's Klein refactor, the house
          default story and the roll changes leave the 8 GB path working; FLUX 3 Klein stills and RTX
          VSR proven on the 4060.
      WAS: the 3-act demo RENDERING with BOTH leads' references -- Jeffrey's three photos and the
        white-silver MagnoliaRex set (started 10:42 on main 45b58a73, ComfyUI restarted with the
        Character Photo node; tmp/rps_live/mystory_magnolia_both.log; launcher scratchpad
        demo_photos2.py). The Jeffrey-only render (08:32) was stopped at the operator's word ("u can
        kill the current renders"), as was a 10:39 start with the gold outfit. The 05:18 demo and its
        Athens highlight stay until the new one is verified.
      MEASURED (11:30, stills done, video still rendering): 54 stills, every character still carries a
        reference (server_ref_on.log: Jeffrey's portrait 618a, MagnoliaRex's d564, the Cat's 22d0;
        reference=none only on the radio/music cards). FACES HOLD across all three acts -- his mime
        face and black shirt, her face, streaked hair and white-silver petal outfit, Athens to the
        desert to frozen Seoul. Two small drifts: a few frames draw a second Jeffrey, and two desert
        frames soften her into an illustrated look.
      THE CAT IN ATHENS AND SEOUL is the per-beat still WRITER's choice, not the reference code: the
        script gives the Cat lines in shot_002 only, but each beat's still prompt is rewritten by the
        writer LLM (my_story_source_rewrite_scene_*), which sees the whole cast as
        candidate_companions and puts the Cat into Athens/Seoul frames; the co-reference matcher then
        correctly attaches the Cat's portrait because the prompt names it. Operator 10-20: "the llm
        writes each beat, we can't necessarily change that ... unless we go to a full beat-by-beat
        writer" -- ACCEPTED AS IS; a beat-by-beat writer is a separate, larger decision, not taken.
        Unverified: the still prompts are not persisted (no ledger row, no PNG text), so a stray
        "cat-eye" alone could also trip the whole-word match. DONE main bd752dd3: each scene still
        now logs "co-references for <still> (<speaker>): <id> <- named: ...<prompt words>... |
        script partner in this shot", so the next render answers it from the server log (Sonnet
        HOLDS; its one minor -- a casefold offset shifting the quote -- fixed in main 28d56993; no
        behaviour change; 1315 tests green across every module test file).
        History of the bug: _ensure_gender_anchor puts "adult woman, " in front
        of "an Egyptian-type adult cat" (the cat-woman / cat-as-a-man bug). Two word-list fixes
        (d1414a7d, 9b0ed9ea) both failed Sonnet as BLOCKERs -- people like "Sailor with a parrot" lost
        their anchor -- and were REVERTED in main 83375f0f (prompts byte-identical to before; titles-
        not-first-names and the LOUD co-portrait report kept). Third attempt goes through a panel
        first (CLAUDE.md). PANEL DONE (Codex gpt-5.6-sol, one lane, 10-20 00:50): PICK = design A,
        the character's KIND as data on the cast row (person | animal | machine | creature; missing or
        invalid = person, so frozen/replayed ledgers are unchanged). Refuted: a My-Story-only card
        choice (other lanes stay broken), an LLM classification at MetaBrief (no stable owner; replay
        makes no call), any free-text reading. Change list, all grounded: _otr_casting.py (casting
        LLM writes `kind`; CastingResponse; generic/adaptation rows ~2301), _otr_my_story.py
        (CastMember.kind; _assign_voices ~1154), _otr_scifi_news_pro.py (~4297; repair rows =
        person), production_ledger.py set_cast allowlist (~1518 -- it DROPS unknown keys),
        otr_meta_brief_image_prompt.py (anchor only for person; drop the hard-coded "a person with a
        clearly visible face" at ~1541/~1601 for non-person), otr_shot_lock.py ~175. CAUTION: the
        casting prompts insist appearance names "a woman or a man" (_otr_casting.py ~441,
        story_packs/my_story/my_story.json:10, scifi_news_pro.json:13) and story packs are sha256-pinned
        -- no REQUIRED pack key. This is a schema/ledger change: per the 08-17 rule it gets a design
        arc before code; parked for a daytime session, not built unattended.
      * branch e5a18331 + bd4ce4ae + 861a2b5e -- APP CAST CARDS: one card per character (name, voice
        he/she/any, who they are, how they look, optional photo) after Veo-Moovie-Maker's Cast Bank;
        "Load the example" = Jeffrey, MagnoliaRex, Cat with the operator's own Veo descriptions;
        POST /api/photos stores the photo in ComfyUI/input by content hash. README updated.
      LIVE PROOF 1 (00:14-00:30, 1-act still_motion, flux3_klein + OTR_FLUX3_REFERENCE=1, the example
      cast, a STAND-IN photo = the generic MOSS Jeffrey portrait, never the operator's pictures):
      obs_publish OK. Sheet tmp/rps_live/ab/photo_reference_proof.jpg (sent to the operator). Photo ->
      portrait kept the face; Jeffrey held in 7/7 of his stills, MagnoliaRex in 7/7 of hers; the
      OTHER person in a two-shot drifted (-> co-references) and the Cat was humanoid (-> non-human fix).
      LIVE PROOF 2 (00:44-00:56, same inputs + co-references; server still on d1414a7d): published.
      Sheet tmp/rps_live/ab/photo_reference_proof2.jpg (sent). Jeffrey's 7 stills: he, MagnoliaRex
      AND a real gray cat all hold. MagnoliaRex's stills: she holds, the man beside her drifted in
      4 of 6 -- her prompts DESCRIBED him instead of naming him -> main 2b30170e adds the beat's
      script partners (who speaks just before/after in the same shot) to the co-references.
      NOTE: the cat was a real cat in proof 2 because that server still ran the (since reverted)
      word-list rule; until the KIND field lands the Cat portrait is a cat-woman again.
      DONE 05:18 -- THE 3-ACT DEMO (see the My Story demo row above; started 00:59,
      tmp/rps_live/mystory_magnolia_ref.log):
      16 GB foley, flux3_klein + OTR_FLUX3_REFERENCE=1, C7 seeds, the example cast cards as the
      app composes them, NO photo (written descriptions only). When it lands: verify, cut the
      highlight, check the stills for real names, and update the demo script (shot 5).
      01:27 check-in: stills stage done -- 31 mints, 21 drawn with co-references (12 with one other
      portrait, 9 with two), no LOUD reports; video stage at 76%. Since the demo started: main
      c2301ff3 (bodiless speakers and shot-less lines lend no face; Sonnet HOLDS) and fd3019e3 (the
      end-to-end test Sonnet asked for). The demo run predates c2301ff3; its cast has no voice-only
      character, so it is unaffected.
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

- [x] DRAFT DEMO VIDEO (10-21 22:45): output/otr/episodes/demo_cuts/radio_play_studio_demo.mp4,
      1:53, phone copy beside it. Black/purple, yellow and aquamarine palette (operator: "orange
      isn't good"), Montserrat, lit letters; app shots (cards, cast with photos, live progress bar);
      the unedited 35 s Seoul ending of the reference-stills demo; the 3D node walk naming every
      model and tool (Gemma 4 12B, FLUX 3 Klein, LTX 2.5, Kokoro, Stable Audio 3, ffmpeg, ComfyUI,
      the OTR pack, Comfy SDK, comfy-api-proxy); "whole story, not a clip" + open weights card.
      Scripts: contest/radio-play-studio/demo_video/ (scenes.py, narrate.py, shoot_app.py, cut.py).
      WAITING ON: the operator's watch-through with sound.
- [x] RANGE REEL (10-21 night, operator: "take the best of the videos and make a conglomerate
      showing range of languages 1,2,3,4 and story types 1,2,3 and something else 1,2 ... classic 3D
      bold blocky lettering"; "showing my great taste in literature, Shakespeare, openness to
      languages, but yeah some good dramatic moments too"): demo_cuts/radio_play_studio_range.mp4,
      Languages 1-5 (English Twelfth Night, Spanish Much Ado, French film archive, Hindi Hamlet,
      Japanese sci-fi news), Stories 1-3 (Shakespeare, his own story, sci-fi from the news), Looks 1-5
      (Flatland engraving "I am falling" vs video art "I am rising", recursive fractal, cartoon My
      Story, the 4060's AnimateDiff GUSTO). Builder: demo_video/build_reel.py + reel_cards.py.
      10-22 update: Languages 6 (Mandarin Kipling, clean CJK), Looks 6 (Flatland archival), no
      numbers on the badges ("it clutters space"), a 100% open weights card, and a closing YOUR
      STORY section of four GUSTO beats from the 16 GB pod render (side_b_20261022_045001):
      turntables, Nina's "leave the silence", "the next selection belongs to Rook", "keep the
      performance". 2:12, 25 parts.
- [~] README final pass (`contest/radio-play-studio/README.md`), 10-22 00:30: still model is FLUX 3
      Klein (it said MOSS-Image), My Story reference stills, the ffmpeg 1080p row, licence note.
      Left: the demo video link once the operator posts it.
- [ ] Submit through the Typeform. **The operator submits** (it is his entry and his account).

## Rules for any session here

- Every code change: scoped suite green, push to the branch, then Sonnet QA briefed to REFUTE
  the pushed diff; a finding is the next commit.
- Shared pack code changes must show the 4060 path is unchanged (CLAUDE.md section 0B).
- Never two renders at once on the 5080. Before starting one, check `:8000/queue` is empty.
- Report: commit hashes, RESULT + obs_publish OK + file path, render time, what the app got wrong.

## After the contest (ideas, not commitments)

- Voices: KEEP KOKORO (operator 2026-10-21, after the Gemini TTS survey
  apple/RESEARCH_PROMPT_TTS_KILLER.md). Worth a measured bake-off later on our own hardware
  (4060, 5080, CPU, Mac; one scene in all 8 languages; his ear decides): Supertonic 3 (MIT, ~99M,
  ONNX, ungated, very fast, but only 10 shared voices and an archived repo) and Qwen3-TTS 0.6B/1.7B
  (Apache-2.0, ungated, Voice Design from a text description -- a fit for casting from the writer's
  notes -- but slow on CPU, no Hindi). Every number in the survey is unverified until we measure.
