## 2026-09-26 -- Foley speech duck implementation (Codex, isolated otr-duck)

Settled operator design wired into `_compile_foley_master` before the real
foley mix, with no new node/widget or workflow change. The existing canonical
foley receipt connector reaches this call. Only resolved foley video lanes
participate; mime and audio-in keep their existing audio behaviour.

Silero VAD stays on CPU. Whisper's ungated multilingual base auto-downloads
tokenless through huggingface_hub into the resolved models root, then runs
CUDA float16 when available or CPU int8 otherwise. Silero's bundled JIT
weight is copied to that same root. Imports are lazy. Whisper releases before
the episode technical slot is acquired; that slot releases in finally.

One strict, complete per-beat JSON batch; empty VAD-positive transcripts duck
without an LLM. Speech-only transcription suppresses non-speech tokens, and
the judge is explicitly told effects/music descriptions are not utterances.
Descriptions are never stripped into empty-transcript babble. Any stage
failure rolls back every duck; mix/master envelope otherwise unchanged.
The fresh ledger merge records VAD, transcript, verdict, reason and actually
mixed duck status; the report counts it. No live claims or new PBUG entries.

Code commit: `56dcceb6`, pushed to main after rebasing on origin/main.
Validation: baseline 128 scoped tests passed; final scoped suite 163 passed
(35 new speech tests), and the final rerun after merging concurrent main
updates also passed all 163. Bug Bible: 48 passed, 14 skipped, 3 xfailed.
Additional schema/policy/registry/dependency checks: 78 passed; the one failing
dependency-sync assertion names exactly the two deliberately deferred entries
below (its known-failure guard exits 2). The assertion remains intact.
Today used CPU mocks only, without loading weights, touching the GPU, booting
ComfyUI, or editing the main checkout. File integrity checks: touched Python
parses; all 7 changed files are nonempty, UTF-8 without BOM. Canonical workflow
and pyproject are byte-unchanged by this commit.

Whisper option provenance: its tokenizer documents non-speech suppression as
blocking tokens used for speaker labels and non-speech annotations; faster-
whisper maps `suppress_tokens=[-1]` to that default list. This is explicitly
enabled and tested. It is not a guarantee against invented ordinary words;
Whisper's model card documents that limitation. Source references:
https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
https://github.com/openai/whisper/blob/main/model-card.md

Post-push QA: Composer 2.5 via cursor-agent, then Sonnet (Claude Code alias
`sonnet`, high effort), two external calls. Both read the actual Windows
worktree. Driver grounded every finding: no functional code defect survived.
The wordless branch and deferred dependency entries were already explicit;
repo text grep cannot prove a git SHA absent; actual counts are 35 new tests.
Sonnet's test-mode stage-label nit has no production effect and the complete
reason already says models were disabled. Real-model accuracy, hardware and
offline-cache replay remain unproven, as requested. No new code patch from
QA. Full local artifacts: kibitz-runs/2026-09-26-foley-speech/r4 (ignored);
this is a scoped post-push review, not a four-round design campaign.

Registry publish is deliberately deferred: requirements adds
`silero-vad>=6.0` and `faster-whisper>=1.1.0`; pyproject.toml is untouched.
The dependency-sync test will flag that known, requested release gap. Live
proof and the operator's ear remain in GO_FORWARD_PLAN.

---

## 2026-09-26 -- HEAD (main) -- LATE EVENING: native Gemma everywhere it is proven, T5 on the GPU

Driver: the 5080 Claude window (Opus), the operator on and off the phone.

WRITERS SHIPPED (plan row 0n has the table): c1d071c4 then 79923170 -- native
  Gemma 4 12B on 16/24 GB NVIDIA, native E2B on 8 GB NVIDIA and the canonical,
  Qwen on Mac and AMD (QWEN_LLM split out of DEFAULT_LLM; default_writer_for_host
  keeps a fresh node on Apple/AMD on Qwen -- Composer's blocker on c1d071c4).
  40b998e2 added the E4B/12B rows; 1e7ef149 primes the 12B's closed thought
  block the way ComfyUI does (Composer's find on 40b998e2). Full suite 17085
  passed, 0 failed at 79923170.
LTX 0.9.8 RECIPE v3 (0f588cf0): the T5 encodes on the GPU; Apple keeps the CPU.
  4060 A/B on the same replay: T5 stage ~3.6 s vs ~40 s, no OOM, frames
  approved by eye (output/otr/ab_ltx8_t5/).
PROOF, shipped as-is: the 4060 ran otr_8gb_video untouched at 0f588cf0 --
  ink_secrets_20260926_190737 in obs, 11:30 whole prompt (native E2B writer +
  v3 T5). The 5080 published E4B (tar_line_20260926_183241) and 12B
  (dead_wire_20260926_184325) episodes.
SPEED HUNT (Cursor Composer, read-only, grounded by measurement): the encode
  findings are real and small -- libx264 vs NVENC 1.1 s vs 0.84 s a clip, 7.7 s
  vs 5.5 s for a whole 81 s episode burn -- not worth a quality change; the
  master-audio hash decode is about a second. One real waste: LTX 2.5's
  post-evict settle spends 3 s every clip ("never moved", both boxes) on a log
  line; trimming it is the operator's call (the 4060 asked for that telemetry).
  No wrong-device or slow-library case left on a shipped path.
QA: Composer on every code push (A3, fixes, family rows, shortcodes, switch +
  v3); findings folded as the next commits.
OWED: Mac/AMD native proof; the 14k-prompt cap probe; dropdown-matrix and fit
  tags for the native rows; receipts for 0n in TEST_WAVE if a wave is opened.

## 2026-09-26 -- HEAD (main) -- EVENING: the Comfy-native Gemma 4 writer is wired and live on both boxes

Driver: the 5080 Claude window (Opus), both boxes, the operator mostly away
("the 4060 is all yours and 5080"; "just wire, focus on getting the new
Gemma 4 and then loading it all up on the 4060 so you can test everything at
once").

GEMMA WRITER (plan row 0n; GO_FORWARD_PLAN has the table):
  A2 3687e592, A1 34e5e3a2 + 5699e232, A3 810d3a26 + 576103d3. The dropdown
    offers comfy_native:gemma4-e2b-it-int8-convrot on every workflow; no
    workflow selects it. Composer QA on A1 and A3 (A1 blocker folded as
    5699e232; A3 follow-ups folded as 576103d3; the 8192-cap finding stays
    open as 0n item 2).
  The live legs found two real defects the fakes could not: 3b1aced6 (0.34's
    CLIP.generate has no mtp, its sample_token no penalty_mask --
    PBUG-20260926-03) and f488f4d7 + 67da20b9 (a second generate() inside one
    node replayed ComfyUI's captured decode graph -- device-side assert,
    server abort -- PBUG-20260926-04, Bible 12.178, e0bed43 in the Bible
    repo). Composer HOLDS on both.
  44877ade: filename codes cg4e2 (native writer) and q3827 (Qwen3.8-27B,
    missing since 09-21), plus an in-process test over every writer row.
  Episodes in obs: 4060 clink_bone_20260926_175812 (otr_8gb_video, 1 act,
    15:24), 5080 knot_midnight_20260926_180140 (otr_16gb_low, 1 act, 217 s).
    4060 writer 44.9 tok/s overall, ~65 steady, vs Qwen3.5-4B NF4 at 12.2 on
    the same card; writer span 1.9 min vs 6.5 min.
  OWED: Slice B (8 GB default) is the operator's call; the 14k-prompt cap
    probe on the 4060.

LTX 2.5 TEXT ENCODER ON THE GPU (67d67fb7, 008fa041, accepted by eye on the
  5080 A/B: "perfect pencil sketches, I can't tell the difference"): every
  LTX 2.5 lane inherits it. 4060, ltx25_foley_16gb replay, 97-frame clips:
  421 s and 479 s against 707 s CPU-pinned; decode peak ~4.46 GB, no OOM.
  The 8 GB mime and audio-in display names now carry that number; the RAM
  figures there are still the CPU-pinned measurement.
  STILL CPU: only LTX 0.9.8's T5 (ltx_8gb, frozen recipe). A GPU-T5 timing
  on the 4060 is owed; changing the recipe is the operator's call.

4060 COLD AUTO-DOWNLOAD DRILL: done, every leg READY from empty folders with
  no HF token (full_8gb_video published; the fetch legs stop at READY by
  design). The Gemma weight later fetched itself the same way.

Suite: full run 17,074 passed, 0 failed at 810d3a26 (before the docs); the
  scoped suites were green at every later push. Bible regression vs OTR:
  48 passed.

## 2026-09-26 -- HEAD 5df92ba4 (main) -- MORNING: no VRAM reserves, unload after use, --video-lane

Driver: the 5080 Claude window (Opus), 09:00-11:00, the operator napping from
~10:15 ("go boldly"). The rulings that drove it, his words:
  "I don't like messing with reserves"; "we need to unload models after they
  are used and if it OOMs we record it, not artificially create a scenario";
  "there is no profiles -- video lanes and workflows"; hand-roll the video
  dropdown for a headless run rather than add a workflow.

CODE (each pushed green, then a Sonnet contrarian; findings became the next
commit):
  84c98900 + 6f4cdd95 H3's contract asks only for Sage off (and not CPU): the
    12 GiB reserve and pinned-memory switch are gone. Sonnet REFUTED the
    first cut on stale docs and a Sage-probe message; folded.
  9e677825 + c5111ff2 scripts/otr_canonical_api_run.py --video-lane <lane>:
    sets the announcer/character/music video dropdowns after the workflow
    row, through the dropdown's own label lookup -- what picking it in the
    app does, and ComfyUI's own headless practice (set the widget, POST
    /prompt). Sonnet: the runner now says plainly when nothing checks a
    lane's weights before the run (HuMo), and the test pins the literal
    label. Cursor agreed with the shape in the operator's window.
  205e96ad + 7c6e9cb2 CHUNK A of Cursor's remove-VRAM-gates plan, with the
    Opus review's amendment: the reserve knob is gone from boot_contracts
    (every contract, argv, env, live check, identification); humo_diet is
    DELETED (no workflow row selected it) rather than kept as an alias; HuMo
    declares no contract; the headless launcher passes no --reserve-vram;
    the queue-time refusal names the exact fix from the cheapest accepted
    boot (Sage, --cpu, both, pinned for a lab-only engine, or "could not
    confirm" Sage). Sonnet: one real gap (unread Sage beside --cpu) folded;
    "HuMo is now accepted on h3 boots" REJECTED -- those are plain Sage-free
    GPU boots now, and every undeclared lane already accepts them.
  64d1b256 + 5df92ba4 CHUNK B, one call instead of Cursor's counter/lock/
    lease coordinator (ComfyUI runs one prompt at a time; render code starts
    no threads). The validator, first node of every episode, sets ComfyUI's
    own free_memory queue flag -- the mechanism behind its unload-models
    control and POST /free -- so ComfyUI unloads every model and resets its
    node cache after the prompt, success or failure. Sonnet verified the
    flag is read only after the prompt returns; its real finding (the writer
    LLM and Bark sit outside ComfyUI's manager and can outlive a FAILED
    prompt) is folded as a prompt-start free_otr_pipeline_residue. Its LTX
    2.5 claim was refuted: that encoder cache is episode-scoped.

REVIEWS: Sonnet on every commit (above). The operator pasted Cursor
  (otr_reviews/cursor_vram_gates_review.md) and agy
  (otr_reviews/agy_vram_unload_review.md) prompts before his nap.
SUITE: 17006 passed / 0 failed at 6f4cdd95 (full, 13:24). Scoped at each
  later chunk: 1809 (boot contracts, HuMo, launcher, gate), 863 (validator,
  levers), 43 (runner); build_variants --check 25/0; Bug Bible 46.
APP FORM: all 26 shipped workflows carry the 33 rows, every row resolves to a
  live input on the running server (the frontend drops a row that does not),
  only otr_app opens as the app. The visual look still wants his eye.

OWED, in order:
  1. The 16 GB 1-act video leg at a676fa33 finishing and publishing.
  2. A fresh STOCK boot (no reserve exists; the launcher is Sage-free), then
     `otr_canonical_api_run.py --profile otr_16gb_video --video-lane
     h3_low_video --act-count 1`, VRAM sampled throughout. An out-of-memory
     is recorded as a PBUG, not answered with a reserve.
  3. After it ends: nvidia-smi should fall to the desktop baseline instead of
     the ~9-10 GB a finished render used to leave resident -- the live proof
     of chunk B.
  4. The 4060: B5/B6 from the background agent, then the B4 free-RAM
     re-measure.
2.3.7 note: "a server booted wrong for H3 is refused at once" in the list
  below is now "H3 runs on a stock boot; only a Sage boot is refused".

## 2026-09-26 -- HEAD 9d50e7fe (main) -- EARLY MORNING: Sprint 2 code complete

Driver: the 5080 Claude window (Opus), 01:30-05:00. Operator answered the
three questions ("RAM fix first", Bible fixes yes, clean-up yes), then ran
Cursor, agy Pro and agy Flash lanes from his own windows until ~04:35; the
last three wrote their reports to Documents\ComfyUI\otr_reviews\.

CODE (each pushed green, then reviewed by another family):
  f011bb79 RAM FIX (PBUG-20260926-01): ltx_8gb caches the T5 CONDITIONING
    (key = T5 file identity + exact text), so a beat loads the 9.8 GB T5 once
    instead of once per segment. The first design -- keep the T5 resident for
    the episode, the eng_ltx25 pattern -- was refuted by an Opus contrarian
    (on a Mac `cpu` IS GPU memory; on CUDA it competes with mapped weights).
    Reviews: Sonnet HOLD, Cursor HOLD. LIVE (5080, 1-act otr_8gb_video):
    PASS 12:52, unstable_ink_20260926_034958, 13 segments / 6 T5 loads.
  47e0ab05 app form: the premise row is "Story idea" / "Optional; every bank
    uses it" -- it said "My Story only", false on every bank (found by the
    Start-here contrarian). Cursor noted snapshot replay ignores the premise:
    a harness-only env path, left.
  f9552204 0d Start-here note (MarkdownNote id 96) + NOTE_NODE_TYPES skipped by
    both converters, the test baseline, the identity stamp and the parity
    checks. 2b3c0ec8 note text names Extensions (Cursor).
  2e78e2b6 + 9d50e7fe 0k CREDENTIAL: OTR_ComfyCredential (id 97, "0 - Comfy
    Credential") is the only node declaring the hidden Comfy key and cannot
    raise; nine hosts dropped it; one link into the validator (the only root)
    orders it first; the writer auth is bound to its prompt id; the stash is
    never swept while its prompt runs. V1-that-cannot-raise, not V3: the
    pack's tests run outside ComfyUI. Design: Opus contrarian (CHANGE, taken),
    agy Pro (REFUTE -- both MUSTs refuted on the files). Reviews: Cursor HOLD,
    agy Pro HOLD, Sonnet HOLD. LIVE on a fresh server, ComfyUI 0.37.4, "All 25
    nodes loaded": canonical queued with a fake probe key, validator forced to
    refuse -- the key is absent from current_inputs, /history, /api/jobs and
    the server log.
  2b3c0ec8 docs: "Run", and "Browse Templates > Extensions > Old-Time Radio"
    (agy Flash audit, each claim checked live). Its Manager renames were NOT
    taken -- this box shows a Manager button; a fresh Desktop needs a look.
  af526cd7 0d palette, the operator's call ("standard ... Lakers purple and
    gold"): ComfyUI's stock purple and yellow pairs alternating by stage;
    Story Writer stays expanded ("whatever is most standard").
  Plan: 0l was already done (3451eb7b); 0c item 5 Kokoro PARKED with the
    grounded reason (3be8ff0f).
  Tests restored, not code: 267e9f48 the portrait identity-seed tests the
    09-04 rip deleted while the mechanism stayed live.
BUG BIBLE (separate repo): 85cb0de2 729cf9c7 9eaab8a5 -- the regression is
  46 passed / 0 failed against OTR; two Sonnet passes found real holes in my
  first two cuts (aliases, prose matches, subscript calls, a broken helper);
  final HOLD. 61b44d9d promotes 12.175 (PBUG-20260925-03) and 12.176
  (PBUG-20260926-01), drafted by agy Flash and rewritten.
Suite: 16928 passed, 0 failed at f011bb79 (full); every later chunk ran its
  affected tests (up to 5064 at once) and --check 25/0. FINAL HEAD b545d1f2:
  16973 passed, 0 failed (full, 11:43); a fresh 0.37.4 server loads all 25
  nodes; 1-act otr_8gb_low PASSED 9:49, crack_repair_20260926_045130 in obs.

  88211ed2 ROW 0, voice-route deletion (Opus worker on the hardened spec;
    driver review + set-diff): 16965 passed / 0 failed = 16973 - 10 + 2. LIVE
    at 88211ed2, Lemmy forced in: PASS 9:26, cold_iron_hum_20260926_053814,
    Lemmy = kokoro bm_george (his recurring voice). README Known failures now
    says the suite is green.
  32c1d7f9 0b2 MINIMAX H3 AUTO-DOWNLOAD (Opus worker; the driver added the
    queue-time boot gate the worker flagged -- without it a stock boot would
    fetch ~39 GB, write and voice, then refuse at the first video beat). LIVE
    on a stock 5080 boot: h3_low_video refused in 2 s, restart flags first,
    nothing downloaded. For his eye: H3_LICENSE_ATTESTATION section 3 still
    says his own hardware only.
STILL RUNNING: the 4060 re-run of B7/B5/B6 at 0d44385c (background agent).
OWED (Sprint 3, the final regression): the 4060 B4 free-RAM re-measure on
  f011bb79+; B5/B6/B7 results; 2.3.7 on his word (the registry Banner line
  rides it). The full suite on the final HEAD is DONE (above).

2.3.7 -- READY ON HIS WORD (not published; 105 commits since 2.3.6 at 96ea436a).
  What a user would notice:
  - Every per-machine workflow opens as an app form; otr_app has every picker.
  - A Google-only workflow (otr_google_still); Google TTS speaks every
    episode language.
  - Progress bars on the writer, the voices and the scene sequencer.
  - The 8 GB LTX lane loads its T5 once per beat, not per clip.
  - ComfyUI-LTXVideo is no longer needed; MiniMax H3 downloads its own
    weights, and a server booted wrong for H3 is refused at once.
  - The Comfy API key can no longer appear in the queue history.
  - A Start-here note, the stock purple-and-gold palette, "Run" and
    "Extensions > Old-Time Radio" in every doc; a gallery thumbnail per card.
  - The talking-face still mode, the creativity dial and the Kling Avatar
    engine are gone (his rulings of 09-25).
  pyproject change for the bump: version 2.3.7 and the Banner line (plan 0f
  item 6). requires-comfyui stays out until the oldest working core is
  MEASURED (0f item 3). Suite green at every step; the 4060 legs below.

FOR HIM:
  - Look at the canvas in purple and gold (any workflow, graph view).
  - Desktop instances: keep "ComfyUI" (it holds the junctions to this repo);
    "ComfyUI (1)" is the 09-14 registry test bed -- safe to delete after 2.3.7
    is checked from the registry.
  - Still his eye: stone_key_20260925_230558 lip sync.

## 2026-09-26 -- HEAD 09fea0ed (main) -- NIGHT: app mode, Google lane live, 4060 fresh-start regression

Driver: the 5080 Claude window (Opus). Operator asleep from ~22:00; standing
orders "code everything we can, then test GPU", "use the 4060 strategically",
2.3.7 only on his word. Workers he ran in parallel: Cursor windows (0a HF_HOME,
0c items 6/8, 0c-7 pack writes, Gemma-on-Mac) and agy (0f-1 descriptions).

CODE (every chunk suite-green and pushed, then reviewed by another family):
  0e APP MODE (operator: "Both") -- fad90bf7 2d5af945 63ba1f16 a2c36590:
    every per-machine workflow opens as a story-only app form; NEW generated
    workflows/otr_app.json opens with every picker in his order; plain-English
    labels (Acts, Story bank, Space saver) on the generated files only; notes
    under source_ref / premise / title / video picker; each generated workflow
    has its own uuid5 id; the mux returns the published mp4 as ui.video so the
    app pane plays it. Design: one Sonnet contrarian round (CHANGE x3, taken).
    Reviews: Cursor x3 (HOLD). SEEN LIVE in the browser pane: gallery category
    "Old-Time Radio", otr_app and otr_8gb_low open straight into app view.
  0h Google stills workflow otr_google_still -- c69687e3; 1585d38a dropped the
    copied `cpu` boot contract (it refused every GPU-booted ComfyUI); the
    writer's Google rows are listed with or without a key (Queue stops on the
    named missing-key message).
  0j follow-up 610bda1e + 37b39dc2: a writer-rolled `other` (or neutral /
    non-binary) takes the seeded draw on google_tts -- FOUND LIVE, the first
    English leg died on it.
  0f-2 progress bars f3d4f4e5 + 9e8d29f5 (Google TTS fan-out too, Cursor catch).
  0c-1 third pass c4fcae80 (second-GPU writer current device; Sonnet HOLD).
  0c-7 follow-up 969241d1 (telemetry reads the live log only; Sonnet catch on
    Cursor's 2c952de1). 0f-1 follow-up 594a6652 (tooltip guard test).
  pathbudget 96466ef4 (the one invalid escape, seen on the 4060's Python 3.13).
  f111293b + 09fea0ed: ComfyUI-LTXVideo RIPPED per TEST_WAVE's B4 rule.
  0d44385c: the 8 GB LTX 2.5 rows stop pinning Stable Audio 3's files in
    preflight (the runner refused them on a box with the post-trained file).
  Worker commits reviewed (Sonnet unless noted), all HOLD: a0875708 89a95212
    (HF_HOME; Spark stood in for Composer, plus Sonnet), a70d22b4 3e1fd8be
    ed4cd47b e1ed8798 (Cursor portability), c04aad46 2c952de1 (pack writes),
    cf75a95e (agy descriptions), a06ee44d (Gemma on Mac; bnb gfx1201 confirmed).
Suite: 16906 passed, 0 failed at a2c36590 (full run, 11:09).

LIVE LEGS -- 5080 (HEAD at boot noted; obs is <output>/otr/obs):
  otr_google_still 1-act EN  FAIL at 1585d38a, 52 s, CastLock `other` (c04)
  otr_google_still 1-act EN  PASS 3:45  refined_metal_20260925_223823 (after 610bda1e; SEAN BOUVIER `other` -> gt_alnilam gender_unservable)
  otr_google_still 1-act ES  PASS 4:00  canvas_firewood_es_20260925_224243 ("La lona bajo la lena")
  otr_google_still my_story  PASS 11:39 grey_reef_20260925_224927 (Sam, Robin unstated -> gender_unspecified)
  otr_8gb_ltx25_audio_in 1-act PASS 2:00:29 stone_key_20260925_230558 (server booted 37b39dc2; the checkout moved to 84640918 mid-leg -- already-imported modules, no effect). LIPS vs LINES: his eye.
  App mode: RUN pressed in the browser pane on otr_app (09fea0ed) at ~01:05 ->
    PASS 9:13 final_bee_list_20260926_010709 ("The Last Bee on the List"),
    and the episode PLAYED in the app pane (node 85 ui.video). The progress
    bars moved in ComfyUI's stream: writer 6/6, Character Voices 1/4..4/4.
LIVE LEGS -- 4060 (TEST_WAVE Part B, fresh start at 1585d38a; receipts in the
  4060 driver's log, every PASS published on the 4060's own obs):
  B3a refusal PASS 7 s; B3 animatediff PASS 60:07; B4 ltx_8gb WITHOUT
  LTXVideo PASS 33:39; B1 low PASS 10:57; B5/B6/B7 refused by the runner
  preflight (fixed 0d44385c) -- re-run at 0d44385c in progress at write time.
BUG BIBLE (Part C): 11 red / 38 green; Sonnet triage: 0 real OTR regressions
  (4 heuristic false positives, 7 stale pins on ripped mechanisms). FIXED on
  his word: Bible 85cb0de2 -> 46 passed, 0 failed against OTR. One pin was
  right: the portrait identity-seed test the 09-04 rip deleted is restored
  (OTR 267e9f48) -- the mechanism was live and had no positive test.

FOR HIM IN THE MORNING:
  - Open otr_app from Browse Templates -> Old-Time Radio; press Run.
  - Watch stone_key_20260925_230558: do the character lips follow their lines?
  - (Answered: the worktree cleanup is DONE -- `main` is the only worktree
    and the only local branch; the Bible fixes are DONE, above.)
  - 2.3.7: on his word.

## 2026-09-25 -- HEAD e6e396b1 (main) -- CODE + PLAN HARDENED (evening)

Did (5080, the only window; the 4060 is wiped and holding):
  Published 2.3.5 (7646ae36) and 2.3.6 (96ea436a); 2.3.6 is Pending, 2.3.3
  still the Manager's Active default. pyproject stays 2.3.6 until the next
  bump, which carries the registry Banner line (plan 0f item 6).
  Code, each pushed green and Sonnet-QA'd (all HOLD after their follow-ups):
    2cb05851 models root asks ComfyUI's configured tree before a folder that
      merely exists (PBUG-20260925-03); e4d0ec19 per-type model_type_dir.
    1af4acee dc93ff05 5952c678 2ec73ef5 -- 11 inherited reds fixed at the
      root; the suite has been fully green since a1e5548f.
    a1e5548f Kling Avatar engine removed.
    ba0a0e87 creativity dial ripped: each writer model samples at its
      maker's own baseline (_otr_model_catalog.SAMPLING_BASELINES; cloud
      slots send no sampling keys). a898360a: the title step's float(None)
      for a cloud writer (QA catch).
    b118c377 talking-face still mode ripped; a0341900: frozen ledgers from
      before the rip replay again (the retired pack key stays known).
    7cf82cda b8edeeef gallery thumbnails: the operator's art copied beside
      all 25 workflows by build_variants, --check guards it.
    021b7c0c registry banner (21:9) and GitHub social preview (uploaded by
      him, confirmed live) from his art; tools/make_registry_icon.py now
      refuses to overwrite the hand-finished registry GIF without
      --registry-gif (an unguarded run overwrote it once; restored from git
      before any commit).
  Plan hardened for a Sonnet window (628afe20 .. e6e396b1): 0b2 has the exact
  64-file "native" rename table and order; 0c's three crash fixes are
  file:line specs with their tests; 0d's cnr_id/ver stamp is a 4-step spec;
  the five section-1 forks are closed by his word into rows 0h-0k (Google
  stills workflow, Google TTS wherever Kokoro is, my_story seeded gender,
  the V3 credential node); the pre-push hook is cut.
  Official language, his rule: the 25 shipped JSONs are WORKFLOWS -- one
  canonical workflow and 24 per-machine workflows ("no graph, no profile, no
  variant, just workflow"); the script-submitted form is the "API-format
  workflow". "graph" only where it means the canvas itself.
Suite: full run green (0 failures) at 7cf82cda and again after every code
  commit since a1e5548f.
4060: wiped (pack and models) and holding. Briefed to wait through the code
  cycle, then run the fresh-start regression with the clean break on the
  exact registry version the 5080 names (sequence in TEST_WAVE Part B).
Server: the 5080 server on :8000 is resident and idle; nothing queued.
Next, in plan order: 0b2 step 1 (the audio-in workflow's character lane),
  then the rename; 0c items 1-3; 0h-0k; 0d; then publish 2.3.7 on his word.
  Proving legs owed: the audio-in workflow's character lip-sync, the three
  kept LTX 2.5 engines on RunPod (Blackwell and 24 GB), and MiniMax H3 after
  its auto-download move.

## 2026-09-25 -- HEAD a841b98d (main) -- LIVE PROOF + GALLERY + PUBLISH 2.3.4

Did (5080, the only window; the 4060 runs the fresh-user walk):
  Live asset_cleanup regression, operator's call: canonical + otr_16gb_still,
  act_count 1, server restarted first (it predated the widget -- /object_info
  lacked asset_cleanup and apply_profile refused 36 vs 35). All three RESULT
  SUCCESS and verified on disk, not by log:
    off     prompt 300f33b6, 8 beats: folder kept (31 files), obs 55.3 MB,
            no asset_cleanup key, no receipt.
    partial prompt 0a80024e, 35 beats: "removed 81 files (560.9 MB), kept 6,
            skipped 0"; ledger, treatment, canon, QA json, captions .ass and
            stills manifest kept; emptied dirs gone; receipt state done;
            obs 81.1 MB.
    full    prompt eb487964: "removed 31 files (550.4 MB), kept 0"; folder
            gone; obs 99.8 MB; the canvas still got its poster frame (the
            preview is taken before the delete); both sibling test episodes
            and the other 2,437 episode dirs untouched.
  c0c286be: the 24 variants moved from workflows/variants/ into workflows/
  (operator: "we can't store the variants in a subfolder"; "All 24"). The
  gallery globs one level, so they shipped and never listed. Reverses the
  09-02 one-JSON ruling (recorded in the standing rulings). One template
  folder still; tests/_support/shipped_graphs.py tells canonical from
  variants for 18 tests; --check skips the canonical and fails if
  variants/ returns. `build_variants --all` hit WinError-22 "Invalid
  argument" writing a random freshly moved file twice (Cursor's ~25 watcher
  processes and Defender were on the tree); --check was 24/0 throughout, so
  nothing was left half-written. Generated docs rebuilt by their own scripts.
  a841b98d: 2.3.4 published on his word. Registry: Pending, 23 deps. The
  downloaded zip carries 25 top-level graphs, no launch recipes, no
  variants/, nodes/_otr_asset_cleanup.py.
Suite: full run on the gallery move, the same 12 inherited reds as
  fe17f426, nothing new.
4060: re-cloned main, found PBUG-20260925-01 (Desktop Manager install
  showed success and installed nothing; a later reboot then loaded the pack
  -- it is confirming which). Briefed to relaunch and install 2.3.4 by
  picking it explicitly in the version picker, confirm the 25-entry gallery,
  and keep logging every step in apple/FRESH_INSTALL_4060_2026-09-25.md.
Server: the 5080 server on :8000 is resident and idle (PID from
  scripts/_otr_soak_server_launch.cmd); nothing queued.

## 2026-09-25 -- HEAD 6ee02d79 (main) -- REVIEW (TEST_WAVE Part A: the overnight chain)

Did (5080, read-only from `output/otr/yt_server.log`): Part A reviewed and
  recorded in apple/TEST_WAVE.md. 4 prompts queued, 4 published, no failure;
  ended by the 08:00 clock after the mime leg (04:48:18). AnimateDiff never
  queued, so A3 (first live otr_16gb_animatediff on the per-engine gate) and
  A4 (Ghost Half B counts) are owed: one 16 GB AnimateDiff leg on this box.
  The log's trailing tracebacks were this window's own /object_info probes
  during the custom_nodes cleanup, not the chain's.
  Also today, earlier: the plan pass (6ee02d79) -- live box rewritten,
  registry section reduced to his items, row 0a's clause, row 0b (name the
  node pack per engine via wrapper_bridge._pack_hint), the V3 row's graph
  count 63 -> 25, section 3's standing.
Next: the one owed leg above; row 0b; the kibitz lane on the plan once the
  CLIs are free (cursor+agy are on a553ac3b); 2.3.5 on his word.

## 2026-09-25 -- HEAD a553ac3b (main) -- CODE (writer LLM folder; queue-time node-pack gate; reviews folded; Bible 12.174)

Did (5080, the only window; Fable driving from the writer-folder build on):
  8f8ccebb: the writer LLM downloads as real files into ComfyUI's `LLM`
  model folder (operator: "real folder", "best practice", no Developer Mode
  assumption). New module nodes/_otr_llm_folder.py; both resolvers check the
  folder first, then the hub cache unchanged; nothing migrated or deleted.
  5080 measured before/after in a HEAD worktree: gemma-4-12b-it resolved to
  the same hub snapshot, same split-revision template, download
  short-circuited -- byte-identical. Docs: README, INSTALL, LLM_PREFLIGHT,
  the standing ruling, a companion note on row 0a (unchanged).
  02758478: PBUG-20260925-02 fixed -- `_refuse_missing_node_packs` at queue
  time, before any download, node classes only, fix-first message; Ghost
  Signal's own refusal leads with the same hint. Plus the cursor/agy
  findings on 8f8ccebb: receipt follows a structural check (hf_hub 1.32
  returns a non-empty local_dir when the Hub is down), provisioner warmer
  routed through the catalog, complete beats partial across roots, stale
  .tmp receipts ignored, config.json part of complete, unsharded hand-placed
  models count, LLM_PREFLIGHT rewritten, hf_download_driver's dead kwarg
  (two sites), `_otr_paths.resolve_hf_model_path` ripped (65 lines, zero
  callers).
  a553ac3b: Sonnet + agy BLOCK on 02758478, both grounded: the
  provisioner loads the catalog BY PATH, and three bare relative imports
  (auto_download_if_missing, validate_model_id, estimate_model_size_gb's
  no-Hub branch) raised ImportError on entry there -- the warmer swallowed it
  as FAILED, so a provisioned box downloaded nothing. My provision test had
  mocked the catalog and proved routing, not the call; it now loads the real
  standalone catalog and runs the seam through it plus both refusal branches.
  cursor: transfers in progress now block completeness on the sharded branch
  too. agy: an empty node registry is "nothing to check", not "everything is
  missing". PROD_BUG_LOG PBUG-02 gets its fix line.
  Bible ed7064e (comfyui-custom-node-survival-guide main): 12.174 "a
  queue-time gate that checks weights but not node classes burns a full
  render before failing"; index row promoted from PBUG-20260925-02; README
  count 358 -> 359; 24 passed.
Suite: full run on each push, the same 12 inherited reds as fe17f426 every
  time, nothing new. Scoped: 190 passed on the touched files at the last
  push. Registry scan replica clean.
Models: writer folder design -- five Composer lanes (grounded), Grok as the
  contrarian (BLOCK on two must-fixes, both built in: completeness rule,
  independent metadata lookup). 8f8ccebb: Sonnet HOLD (executed), cursor
  and agy yes-with-fixes (all grounded and folded in 02758478). 02758478:
  Sonnet BLOCK (the standalone-import regression, real), agy BLOCK (same
  finding), cursor yes-with-fixes (the in-progress-transfer hole, real; the
  rest brief critique). a553ac3b: Sonnet re-review running at push time (the BLOCK finding is fixed and tested against the real standalone load); result follows as its own commit. Reviewers left HEAD
  and the tree untouched every round (checked).
Leftovers for him: C:\ComfyUI-Models\LLM\converted (6.7 GB, the removed
  GGUF backend's) -- the new scan ignores it; delete or keep. Local branch
  worktree-agent-aaf3612b6248a9d9e is a dead pointer.
Next: 2.3.5 waits on his word (2.3.4 is Pending and the scan replica says it
  will Flag over an internal doc that shipped; the tree is clean now). 4060:
  B3 retry on 02758478+ proves the gate live (refusal at t=0 without ADE),
  then B4. Row 0a (short Windows HF_HOME) still open, writer no longer on
  that path.

## 2026-09-25 -- HEAD e7a4806a (main) -- CODE + DOCS (row 0b asset_cleanup built; stale pointers; one local copy)

Did (5080 window, now the only window; the server on :8000 was not touched):
  3e01b1c6: row 0b built as written. `asset_cleanup` on OTR_LedgerScriptWriter,
  three full-text labels, default off, the trailing widget (canonical node 1
  widgets_values[35], descriptor last in inputs[], no dst_slot moved; 24
  variants regenerated, --check 24/0). The writer stamps the first word beside
  delivery_intent, and inside the replay branch with led.save() before the
  wire is built; `asset_cleanup` is run-volatile on a replay. New pure module
  `nodes/_otr_asset_cleanup.py` (plan + execute, stdlib only), called ONLY
  from `OTRMasterAudioMux._asset_cleanup`, the last step of mux() after the
  preview frame. Linked folders inside the episode are a REFUSAL (the row's
  test list and the standing ruling; the row's prose said unlink-as-entry --
  refusal is the stricter reading and is what shipped). Paths compare by
  realpath so a junctioned output tree still binds. Docs: RUN.md "Saving disk
  space", the standing ruling, the janitor header.
  c292539c: two stale pointers -- `__init__.py` named `otr_4060_floor` as live
  and said workflows/variants/ was deleted; the rulings named
  docs/GO_FORWARD_ARCHIVE.md as a live file (it is `a0ff6c8c~1`).
  e7a4806a: README widget row + "Where things land" paragraph; INSTALL.md's
  "one thing the pack deletes" is now two.
  Local cleanup (operator): one OTR copy on disk. Removed the stray baseline
  worktree and the 2026-09-24 agent worktree (its WIP commit 40aef8ab stays on
  local branch worktree-agent-aaf3612b6248a9d9e; main already supersedes all
  of it except a client.py rename and three EXPECTED_RED G2 entries, which
  are a matrix-row call for him). Six retired projects moved out of
  custom_nodes and deleted by the operator; their six GitHub repos deleted by
  him too (confirmed gone).
Suite: full run on the tree, 16 reds; the 16 re-run at fe17f426 in a temp
  worktree outside custom_nodes: 12 inherited. The 4 new ones were mine and
  are fixed in 3e01b1c6: the one-act writer count pin, and a ledger-singleton
  leak from the new replay tests into test_audio_cache_wiring (alphabetically
  next) -- the fixture now restores `production_ledger._CURRENT`. New file 40
  passed, incl. a real junction and two real Windows locks. build_variants
  --check 24/0. Touched .py AST-clean, no BOM, LF. Bug Bible not run.
Models: QA on 3e01b1c6 -- Sonnet subagent (REFUTE, executed the planner and
  executor on temp trees: real junction refused and target untouched, locked
  file skipped and named with the done receipt still written, replay of a
  `full` source with the widget off leaves no key on wire or disk): HOLD, no
  must-fix. Cursor lane cursor-grok-4.6-high in ask mode via kibitz: critiqued
  the brief, not the code; its three code claims checked -- linked-dir
  policy (shipped as refusal, above), `peek_ledger` DOES exist
  (production_ledger.py:808, so row 0b's review record was wrong about
  ChatGPT), and the stem-prefix identity (held: a replay's silent video sits
  in its own folder, so the inside-the-folder check refuses). Antigravity Gemini 3.8 Flash (High) via
  kibitz (r4): ran its OWN junction and locked-file probes independently
  of Sonnet's, got the same results (refused/target untouched; skipped,
  named, receipt still written), and read the mux ordering, the replay
  drop test, and build_variants --check the same way. HOLD, no must-fix;
  one nice-to-have (a debug log on an empty-dir full run) not taken.
  Codex out of credits until 2026-09-29 14:59 PDT. Reviewers left HEAD and
  the tree untouched (checked).
Next risk (not a defect): a replay keeps its source's delivery_token, so the
  token alone cannot tell a replay from its source; the stem check and the
  inside-the-folder check are what do. A fresh token per replay import would
  close it.
Next: 2.3.4 carries asset_cleanup, when he says (plan, registry section).
  4060: re-cloned main at c292539c, confirmed node 1 is 36 wide, has the GO for
  Part B once his ComfyUI Desktop reinstall is done. The plan's "Live box"
  note describes the 1-act chain that was due to end at 08:00.

## 2026-09-25 -- HEAD eaf07017 (main) -- CODE + PLAN (orphan cleanup d245cc27, asset-cleanup design row 0b)

Did (5080 window; the 1-act chain on port 8000 was not touched):
  d245cc27: the orphan/stale-reference cleanup. `_validate_scene_envelope`
  wired at `_build_envelope`'s one call site (standing ruling #3 satisfied);
  the VRAM sentinel chain ripped whole (`vram_sentinel`, `force_vram_offload`,
  `_CLEANUP_CALLBACKS`, `register_vram_cleanup` and the one registration in
  story_orchestrator) on the operator's word that the VRAM measures say
  little (ruling #4 records it); the Bark squeal metric,
  `slot_matrix.eligible_engines_for_role`, `SCIFI_ORCHESTRA` and
  `MAX_PROVENANCE_NOTE_CHARS` gone with their test mentions; about twenty
  stale references repointed (docs/ -> apple/ and the sibling
  vram-recipe-lab/docs/, config/profiles/ -> the matrix row). Two review
  recommendations were wrong and were not applied: the `otr_soak` reset
  marker (its launcher exists) and `animatediff15_v2_video` (a RETIRED_ENGINE_IDS
  tombstone). Only story_orchestrator's own two hunks were staged; the other
  window's unfinished patch at ~774 and ~901 stays in the working tree.
  eaf07017: GO_FORWARD_PLAN row 0b, the asset-cleanup design (`asset_cleanup`
  off / partial / full on the writer; the delete is the last step inside the
  mux after publish, ledger stamp and canvas preview; identity guard from
  BUG-LOCAL-014; partial is a delete-list plus a receipt in the surviving
  ledger). Measured: 90.5 GB in 135 media-bearing episode dirs, partial keeps
  0.3%. A ChatGPT design-review prompt was handed to the operator.
  Output-tree facts for the next window: otr/ top level carries more than
  episodes/ + obs/ (audio, replay_bundles, _gemini_judge, _probe_clips,
  _quarantine_model_translated_20260920, a smoke still, the chain scripts and
  logs); the ComfyUI-Installs output/otr tree is 8.3 GB and last written
  2026-06-13; 2,291 episode dirs hold only ledger and text (his hand cleanup).
Suite: scoped set 1062 passed / 3 skipped before d245cc27; build_variants
  --check 24 / 0; touched .py AST-clean, no BOM, no CRLF. Bug Bible not run.
Next: fold whatever survives of ChatGPT's lettered answers into row 0b, then
  build 0b. 5080: TEST_WAVE Part A once the chain ends (08:00 local). The
  10-file red-test patch stays uncommitted and separate.
Models: QA on d245cc27, all three no must-fix -- Sonnet subagent (executed
  the envelope guard for act_count 1..7, grep for every ripped name), Cursor
  lane cursor-grok-4.6-high in ask mode via kibitz (no defect in the commit;
  it mostly critiqued the brief's scoping), Antigravity Gemini 3.8 Flash
  (High) via kibitz. Codex is out of credits until 2026-09-29 14:59 PDT.
  Reviewers left HEAD and the tree untouched (checked).

## 2026-09-25 -- HEAD d167f0ab +branch claude/practical-hamilton-ftvd2j -- CODE + PLAN (AnimateDiff auto-download, required_models, test wave)

Did (cloud window, no hardware touched; the 5080 chain was not reached):
  AnimateDiff weights now download at queue time. `_SOURCES` in
  `nodes/_otr_visual_assets.py` gained v3_sd15_mm.ckpt and
  v3_sd15_adapter.ckpt (guoyww/animatediff, Apache-2.0),
  animatediff_lightning_8step_comfyui.safetensors (ByteDance, OpenRAIL-M) and
  vae-ft-mse-840000-ema-pruned.safetensors (stabilityai, MIT) -- all ungated
  on the Hub API. Each lane names its own files through the new
  `GhostSignalEngine._weight_tokens()`; the three lanes joined `_COVERED`, so
  the dropdown matrix reads them "auto". Measured before the change: with an
  empty server the runner gate refused otr_8gb_animatediff and
  otr_16gb_animatediff over the motion module and adapter, and every other
  shipping row passed. After: no shipping row is refused
  (`test_no_shipping_row_is_refused_on_an_empty_server`).
  `preflight.required_models` now accepts weight FILENAMES only (schema in
  `capability_profiles.py`). Removed the writer repo ids from both
  AnimateDiff rows (the 8 GB row named gemma-4-E2B-it; its writer is
  Qwen3.5-4B) and replaced otr_8gb_video's logical ids with
  ltxv-2b-0.9.8-distilled.safetensors and t5xxl_fp16.safetensors. The
  runner's report-only branch for ids is gone. Variant JSONs unchanged; three
  launch recipes regenerated.
  Plan cleaned: dead links fixed, the Google row cut to its fork, the
  resolved word-counter row removed (WORD_RE is already Unicode), Shakespeare
  moved to Parked with a git pointer. Row 0 and 0a specs restored as
  apple/ROUTE_DELETION_PLAN.md and apple/HF_HOME_WINDOWS_PIN.md and
  re-grounded. The wave is apple/TEST_WAVE.md.
  Registry, read 2026-09-25: 2.3.3 Active, 2.3.2 Active; 2.3.0, 2.3.1, 2.1.5,
  2.1.6 Flagged.
Suite (Linux sandbox, not the Windows venv): scoped set 1,631 passed; two
  red, both already red on d167f0ab before this change --
  test_lane_preflight_matrix.py::test_g2_canvas_truth and
  ::test_g4_admission_honesty. Full suite result is on the PR. Bug Bible not
  run.
Next: 5080 -- TEST_WAVE Part A once the chain ends. 4060 -- Part B on a head
  that carries this branch. Then Part C. Code rows 0 and 0a stay open.
Models: Sonnet QA on the pushed diff (see the PR).

## 2026-09-25 -- HEAD 9b9e0766 +handoff (main) -- RENDER (16 GB 1-act rotation, registry 2.3.3)

Did: Stills descriptions now say z_image_turbo. Lumina is not a default in any
  profile JSON. Mac stays sd15. Cloud and Google rows were not touched.
  Commits 248bf651 (descriptions) and 9b9e0766 (pyproject 2.3.2 -> 2.3.3).
  Registry publish Action 36103232501 succeeded. GET
  /nodes/comfyui-old-time-radio/versions shows version 2.3.3 status
  NodeVersionStatusPending (created 2026-09-25T06:31:46Z). Zip contents were
  not checked. Active waits on Comfy-Org's scan.
  Live 1-acts on this 5080, canonical graph, act_count forced to 1. Published
  to C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\:
  covenant_ink_20260924_232810__arch__stmo__zimg__koko__pubd__g412__sa3_final.mp4
  (otr_16gb_still),
  notched_key_20260924_234023__scif__l25v__zimg__koko__orig__g412__sa3_final.mp4
  (otr_16gb_video),
  black_fog_20260925_011016__shst__n16f__zimg__koko__sspr__g412__sa3_final.mp4
  (otr_16gb_foley). All three filenames say zimg.
  Mime prompt ea876b5c-3646-4b37-8603-a92fe79a4657 was still running at
  handoff (server log on beat shot_001_b15, ltx25_native_mime_16gb).
  AnimateDiff (otr_16gb_animatediff) is next. Chain script
  C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_chain.ps1 repeats stills,
  video, foley, mime, animatediff until 2026-09-25 08:00 local or the first
  failure. Server log: C:\Users\jeffr\Documents\ComfyUI\output\otr\yt_server.log.
  Do not kill python and do not free port 8000.
  Full pytest suite and Bug Bible were not run. A render occupied the box.
  Dirty and uncommitted, do not git add with other work: a partial red-test
  patch (worktree commit 40aef8ab, branch
  worktree-agent-aaf3612b6248a9d9e). The EXPECTED_RED ledger edit and the
  google client quota_state rename were rejected and are not in the tree.
  The rest is unstaged: apple/evidence/video_evidence_manifest.json,
  nodes/story_orchestrator.py (LLM-slot comments only), and the test files
  named in git status. apple/MACHINES.md shows modified from CRLF only.
  Operator asked for a Stable Audio 3 house/techno check of prompt, inputs,
  temperature, and seed. Not done. Rendering was the priority. The
  music-is-done ruling still covers cue-wording chases.
Current step: mime 1-act still on the GPU. AnimateDiff has not started.
Next: leave the chain alone until 08:00 or a RESULT FAIL. Then take the
  next GO_FORWARD row. Do not reset the box while :8000 is serving this chain.
Models: no panel. Description strings plus the version bump. No Composer or
  Sonnet QA on that diff.
Commits: 248bf651, 9b9e0766. The sha above is the second-to-last on the
  branch; the last is this handoff commit.

## 2026-09-25 -- No quant pack, no Wan, no harnesses; LTX 2.5 downloads itself

Branch `claude/practical-hamilton-ftvd2j` (draft PR #5). Operator directives:
no quant-format mentions outside the Bug Bible, no harnesses, no rigs, no saved
fixtures, delete all history (every dated `docs/2026-*` folder, `kibitz-runs/`,
this log's older entries and `GO_FORWARD_ARCHIVE.md` -- all recoverable from git
history), and "less friction for the end user, auto download things to work".

What the pack is now: the canonical graph plus 24 variants generated from
`config/workflow_matrix.json` rows -- the only workflow definitions. The LTX 2.5
lanes are native safetensors on stock loaders and fetch their own weights at
queue time from ungated mirrors (about 25 GB for the 16 GB lanes). Removed:
`wan_ti2v`, `fastwan_8gb`, `ltx_video`, `ltx_audio_in`, the quant LTX 2.5 lanes,
the DMD sampler node (24 nodes now), 98 harness scripts, 72 rigs,
`tests/fixtures/`. Needed reference docs moved to `apple/`.

Verification: full suite on the Linux cloud box (no GPU, no Windows paths) --
zero new failures against the 8ffb09d baseline (289 baseline reds, 225 now).
`build_variants --check`: 24 variants, 0 failures. Sonnet refute-QA ran on the
pushed diffs; findings folded in. NOT yet proven on hardware: the LTX 2.5
queue-time download and the 16 GB silent lane on its new weights. Bug Bible not
run from here -- its repo is not reachable in the cloud session.

## 2026-09-27 -- Overnight: rolls, the foley speech duck, Gemini 3.8 TTS, 2.3.8 to 2.3.11

Branch `main` (built in the `_worktrees/otr-rolls` worktree, pushed with
`git push origin HEAD:main`). About 80 commits since 2026-09-26 noon.

Shipped:
- Selective rolls. Style and language rolls take a typed pool
  (`style_roll_pool`, `language_roll_pool`, STRING widgets appended after
  `gate_in`; empty = all). A multi-select COMBO drew as a blank box in the app
  view (2.3.8), so 2.3.9 made them typed lists. `parse_roll_pool` also reads the
  saved-list repr that core's `str()` produces. The language dropdown gained
  "roll (any language)"; the pick is stamped in `meta["language_roll"]`.
- The foley speech duck (`_otr_video_engines/foley_speech.py`): voice gate,
  faster-whisper base, one batched judge call on the writer's technical model,
  then the foley stem is halved on a beat with words. Operator rule: any words
  duck, unsure ducks, a failed judge ducks every beat with a transcript.
- Gemini 3.8 Flash TTS is the Google TTS default (retry on the Flash Lite TTS).
- Registry: 2.3.8, 2.3.9, 2.3.10 published (2.3.9 Active, 2.3.10 Pending when
  this was written), then 2.3.11 carrying the judge fix and the structured-call
  alias contract fix.

Live proof (RunPod RTX PRO 4000, 24 GB):
- Episodes delivered to the 5080's obs: `crown_ass_20260927_040716`,
  `dying_breaths_20260927_081511`, and the duck test
  `crown_ass_20260927_123721`.
- Whisper test: ten hard-coded LTX 2.5 foley prompts in a replayed ledger.
  Detection 10 of 10 -- every scripted line word for word, nothing invented on
  music, clapping, applause, crying, laughing or the silent mime. The duck
  itself did not fire on that run (judge JSONDecodeError rolled it back); fixed
  in 0df248d5 / cd0ff845 / 0d46b48d. Receipt:
  `C:\Users\jeffr\OTR-pod\duck_test\results_20260927\WHISPER_RESULTS.md`.
- PBUG-20260927-01 (replay paths with backslashes on Linux) and -02 (Whisper
  dead on CUDA 13 stacks) both fixed and live-verified on the pod. Bug Bible
  promotion for both is still owed.

Open:
- 4060 human-install check: the official ComfyUI portable is unpacked at
  `C:\OTR-Human-2.3.10\ComfyUI_windows_portable` on the 4060, served on port
  8190 with `--enable-manager` (the portable needs it for Node Manager; noted in
  the README), reached from the 5080 through an SSH tunnel. Next: install
  "old time radio" from Node Manager once 2.3.11 is Active, then run each 8 GB
  workflow from Browse Templates one at a time and check obs after each.
- The duck needs one more live leg on the fixed judge to prove a halved stem.
- Astra's Whisper harness collector fails on its ledger lookup ("Need exactly
  one replay ledger carrying this test id"); the numbers above come from the
  episode's own ledger receipt.
Models: DeepSeek, Sonnet and Composer QA on the pushed diffs, as each commit
  names.

## 2026-09-27 afternoon -- 2.3.11 and 2.3.12, the human-install check on the 4060, AnimateDiff friction

Published: 2.3.11 (Active) and 2.3.12 (0149ce33, Pending at writing). 2.3.12
carries the story bank's roll pool ("Banks to roll", so all three rolls look
alike in the app view) and the AnimateDiff missing-pack hint.

4060 human install (official ComfyUI portable 0.37.0, `--enable-manager`, pack
from Node Manager, each workflow from Browse Templates > Old-Time Radio, Run in
the app view; every episode copied to the 5080's obs):

| workflow | result | time |
|---|---|---|
| otr_8gb_low | published (sugar_light_20260927_130357) | 24:43 |
| otr_8gb_still | published (dry_warning_20260927_133513) | 37:21 |
| otr_8gb_animatediff | published (candy_box_20260927_141350) after installing AnimateDiff-Evolved from Node Manager | 1:40:31 |
| otr_8gb_video | published (rhymes_rotten_20260927_155406) | 46:20 |
| otr_8gb_ltx25_foley | rendering at writing (~3.9 min per 3.88 s clip) | |
| otr_8gb_ltx25_mime, otr_8gb_ltx25_audio_in, otr_canonical | not started | |

Findings from the human path:
- A fresh Manager install gave 2.3.9 while 2.3.11 was the Active latest: the
  Manager's registry cache was stale. The pack card's version picker,
  "Latest (2.3.11)", then Apply Changes (restart) fixed it.
- The AnimateDiff workflow failed at Run in 1 s with the right message, but
  Manager's Missing Nodes could not see the dependency (no ADE node in the
  graph). Fixed in d40338cb: js/lane_node_packs.js adds the lane's ADE classes
  to the frontend's missing-node list on load (kibitz r1: Claude + Cursor +
  Fable; a bypassed placeholder node was measured and rejected, the frontend
  skips mode 2/4). Proven on a CPU sandbox with no ADE; the Install button
  itself (Manager on) is not yet seen live.
- Text encoders: every NVIDIA lane runs them on the GPU; ltx_8gb keeps the CPU
  T5 on Apple only (operator: until a Mac proves otherwise).

Open:
- Finish the 4060 set (foley, mime, audio_in, canonical), then a second
  still and video run for fair timings (the first runs included downloads).
- Pod AnimateDiff regression: not run. The migrated pod (ckoq6477osfqev, a NEW
  id) refused SSH until authorized_keys is re-added; stopped, disk kept. Needs
  the operator at a desktop for the one-line key paste.
- Subgraph blueprint: dropped by the operator.
Models: Sonnet QA on each pushed diff (bank pool HOLDS; hint HOLDS with one
  real finding fixed in the next commit), Cursor + Fable on the design round.

## 2026-09-27 evening -- 2.3.12 Active, the foley leg, the LLM pet project

- 2.3.12 went Active at 19:04 PDT and is the registry's latest. Nothing else
  was published tonight.
- The subgraph blueprint row is out of GO_FORWARD_PLAN.md (operator: OTR is a
  full pipeline, not a piece to splice; not worth it).
- A public-facing field guide of vibe-coding gotchas was built from the commit
  history and the Bug Bible (about 40 items, each with its commit or PBUG). It
  is a private artifact the operator shares when he chooses; the link is in
  the driver's memory notes, not here.
- The operator posted on r/comfyui recruiting an ML trainer for his next pet
  project, an LLM trained on real ComfyUI node-dev lessons. Not OTR work.
- 4060 human-install check, continued: otr_8gb_ltx25_foley was at 17 of ~21
  beats at 22:09 (about 15 min a beat on the 8 GB card; ETA ~23:15). Plan when
  it publishes: run otr_canonical, then a second still and video run for fair
  timings (the first runs included downloads). The 8 GB mime and audio_in legs
  are skipped unless asked: same LTX 2.5 engine, 6-7 h each on this card.
- Text encoders: confirmed every NVIDIA lane runs them on the GPU. ltx_8gb
  keeps the CPU T5 on Apple only, until a Mac proves the GPU version (web
  research the same evening supports waiting: fp16 T5 NaNs on MPS, the 9.9 GB
  unified-memory footprint).
- Lemmy cameo ships on its ~11% roll in all 27 workflows; the operator was
  asked whether it should stay and did not answer. Unchanged.

## 2026-09-28 00:30 -- 4060 human-install check COMPLETE (8 legs, all published)

Every leg ran from the official ComfyUI portable with the pack installed from
Node Manager (2.3.11), each workflow opened from Browse Templates > Old-Time
Radio and started with Run in the app view. Every episode copied to the 5080's
obs as it published. Operator's Option A: skip otr_8gb_ltx25_mime and
_audio_in (same LTX 2.5 engine as foley, 6+ h each on this card).

| leg | episode | time | note |
|---|---|---|---|
| otr_8gb_low | sugar_light_20260927_130357 | 24:43 | first run, Kokoro fetched at boot |
| otr_8gb_still | dry_warning_20260927_133513 | 37:21 | 18 stills, includes the Z-Image download |
| otr_8gb_animatediff | candy_box_20260927_141350 | 1:40:31 | after installing AnimateDiff-Evolved from Node Manager |
| otr_8gb_video | rhymes_rotten_20260927_155406 | 46:20 | 15 beats / 3420 frames, includes LTX 0.9.8 + T5 downloads |
| otr_8gb_ltx25_foley | fap_echo_20260927_164552 | 6:26:58 | 21 beats, ~8.5 min per 3.88 s segment; first LTX 2.5 run |
| otr_canonical | receptor_lock_20260927_230128 | 21:12 | resolves to the low lane on this card |
| otr_8gb_still (2nd) | rainbow_boxes_20260927_233214 | 57:56 | 47 stills: ~1.2 min/still vs ~2.1 on run 1 |
| otr_8gb_video (2nd) | misplaced_flicker_20260928_002836 | 41:16 | 17 beats / 3037 frames, no downloads |

Timings are per episode and each story differs in length; compare per still
or per beat, not per run. No OOM, no traceback, no hang on any leg.
Portable is left at C:\OTR-Human-2.3.10 on the 4060 (server stopped by the
next session if wanted; it costs nothing running).

## 2026-09-28 01:00 -- the 4060 human-install check is complete

Fresh official ComfyUI portable 0.37.0 on the 8 GB RTX 4060, `--enable-manager`,
the pack installed from Node Manager (2.3.11 via the version picker), every
workflow opened from Browse Templates > Old-Time Radio and run from the app
view with its shipped settings. Every episode published on the 4060 and was
copied to the 5080's obs.

| workflow | time | episode |
|---|---|---|
| otr_8gb_low | 24:43 | sugar_light_20260927_130357 |
| otr_8gb_still | 37:21 | dry_warning_20260927_133513 (18 stills) |
| otr_8gb_animatediff | 1:40:31 | candy_box_20260927_141350, after installing AnimateDiff-Evolved from Node Manager |
| otr_8gb_video | 46:20 | rhymes_rotten_20260927_155406 |
| otr_8gb_ltx25_foley | 6:26:58 | fap_echo_20260927_164552 (19 foley beats, ~15 min a beat; duck: 0 of 19, VAD found no voice) |
| otr_canonical | 21:12 | receptor_lock_20260927_230128 |
| otr_8gb_still, second run | 57:56 | rainbow_boxes_20260927_233214 (47 stills, Shakespeare; the story, not the download, set the time) |
| otr_8gb_video, second run | 41:16 | misplaced_flicker_20260928_002836 |

Skipped on purpose: otr_8gb_ltx25_mime and otr_8gb_ltx25_audio_in (same LTX 2.5
engine as foley, 6-7 h each on this card). The first-run timings include the
one-time model downloads; the reruns show the spread between stories is
larger than the download cost.

Two observations, neither a code change:
- The app view's Run button queued the job TWICE on each of my emulated clicks
  (still and video reruns); the duplicate was removed with `POST /queue
  {"delete": [id]}` both times. Not seen with a real mouse; check before
  calling it a bug.
- The 4060's portable server is left resident, idle, on port 8190.

## 2026-09-28 12:40 -- model randomizers with checklists, proven live; H3 on 8 GB

**What shipped.** Three Yes/No switches, off in every workflow, local engines
only, each noted "Suggested for 16 GB+" in the app view: Randomize video
(`roll_video_lanes`) and Randomize stills (`roll_still_models`) on
OTR_VideoDirector, Randomize audio (`roll_audio_engines`) on OTR_CastLock.
Video and stills each have a checklist, Video lanes to roll and Still models to
roll; none ticked draws from every local model that runs here. Every checklist
(language, bank, style, video, still) now sits at the foot of the app form,
after Space saver (operator idea, same day). Commits: 0088f5fd, 893f3bea,
f2db297e, 38faa134, 933bc52f, 46a55415.

How it works: the validator's gate draws the video lane, still model and music
first, before the cloud checks, pack and boot refusals and downloads, and writes
the drawn ids into the queued prompt. The voice is drawn in CastLock, where the
episode language is known. CastLock saves every receipt to the durable ledger
(video_lane_roll, still_model_roll, music_engine_roll, voice_engine_roll) and
the credits name them under Rolled. Seeds replay through OTR_VIDEO_LANE_SEED,
OTR_STILL_MODEL_SEED, OTR_VOICE_ENGINE_SEED and OTR_MUSIC_ENGINE_SEED in the
environment ComfyUI starts with.

**Live proof on the 5080 (both published to otr/obs):**

| leg | draw | time | episode |
|---|---|---|---|
| switches on, seeds 9 and 5 | MiniMax H3 (from 24 local lanes), Z-Image-Turbo, MusicGen, Kokoro | 1:36:26 | signal_time_20260928_105643, 1:54 |
| switches on, still_pan and sd15 ticked | still_pan, sd15, MusicGen, Kokoro | 4:10 | ass_noll_20260928_123415, 2:19 |

The first leg found a real defect: CastLock's durable save copies a fixed key
list, so the receipts reached the wire ledger but not the saved one the credits
read (fixed in 893f3bea). The second leg's saved ledger carries all six roll
receipts, and its credits read "Rolled: source bank, visual style, video lane,
stills, voices, music".

**Review roster.** Design r1: agy on Gemini 3.8 Flash (Codex is out of credits
until 2026-10-03 16:04 PDT). Folded: voice language admission, preserve_ledger,
clone reference clips. Post-push: Sonnet (0 defects on 0088f5fd; one stale-doc
defect on 38faa134, and a full suite on 38faa134 of 17,245 passed, 0 failed) and
Composer 2.5 on the Cursor lane (a derived replay's video receipt, test gaps,
one stale plan line). All folded; judgments under kibitz-runs/2026-09-28-lane-*.

**H3 on the 8 GB 4060 (7db8095b).** otr_8gb_still with minimax_h3_video, one
act: three beats rendered, then beat 4 died out of memory loading the text
encoder, 2:54:27 in, with CUDA reporting 0 bytes free while PyTorch held 213 MiB.
No episode. The dropdown matrix keeps oom, now as a measured note; a follow-up
task to find what holds the memory between beats was offered to the operator.

**Registry.** 2.3.12 is Active. Main now also carries the clickable roll pools,
the forgiving pool spelling and everything above; a 2.3.13 publish is the
operator's call.

## 2026-09-28 14:00 -- 2.3.13 published: no token caps, no doubled dialogue

**Doubled dialogue (90902989, PBUG-20260928-01, Bible 12.181).** The operator
heard every line twice in signal_time_20260928_105643. A stage-direction repair
in ledger_clean spliced the rest of the line back in; the splice now refuses a
replacement that repeats the kept speech. Live verify: hidden_sequence_20260928_125321.

**No token caps on the cloud writers (505d7d51, PBUG-20260928-02, Bible
12.182).** The operator's My Story run on ComfyUI Desktop (3 acts, OpenRouter
Sonnet) died inside ledger_clean at 299,911 of a 300,000 per-run ceiling after
the script was written and paid for; the same death is in the log for 09-17.
The ceiling counted each call as prompt plus its whole 16,384 output allowance,
never the bill. Operator: "no caps ... remove that whole feature". OpenRouter,
Comfy Credits and Google writers now send no output number and have no per-run
or per-call ceiling; the queue-time wallet check is the money guard, and the
log shows the provider's real tokens and dollars per call. Live-proven before
relying on it: Sonnet 5.5 wrote 37,533 tokens uncapped and stopped on its own
($0.38); OpenRouter kept a 36 s reply alive past a 20 s read timeout; Gemini
Flash wrote 8,915 tokens to a natural end; ComfyUI's own OpenRouter node posts
to the Comfy proxy with no max_tokens. Comfy and Google writer timeouts went to
600 s (replies arrive only when finished).

**Follow-up (82c007da).** Composer on 505d7d51 found the Google writer's context
check skipped calls with no requested size; grounding it showed the real defect
was the Google rows' placeholder 8,192 window on models that take 1,048,576
(measured from Google's model list). Fixed, with the check now on every call.
Sonnet on 505d7d51: two stale comments, folded into the same commit.

**The operator's "tiptoe" question.** No baked-in text leaked into My Story. The
queued prompt had the music description in "My Story - by" and "Jeffrey A Brick"
in "My Story - music in words" -- the two boxes were swapped, so the announcer
said "Tonight's story is by Playful, gently spooky jazz...". His 13:13 re-run
(local 12B writer) carries the same swap. The labels were left as they are.

**Registry.** 2.3.13 published at d4328211 (full suite before the bump: 17,246
passed, 0 failed); Pending at 20:57 UTC with 25 dependencies and its zip on the
CDN. 82c007da (the Google window fix) is on main, not in 2.3.13.

**Open.** The operator's ComfyUI Desktop must be restarted to load the no-caps
code (it loaded at 12:57). Post-push QA running on 90902989 (Sonnet) and
82c007da (Composer).

## 2026-09-28 16:25 -- LTX 8 GB recipe v5: the pulse gone, v4's cut-aways reverted, joins no longer darken

**The complaint.** costume_masquerade_20260928_132721 (Desktop, `otr_8gb_video`):
"it really skips", then "much of it is blurry and you can feel it stepping".
Three defects, all fixed and live-verified on paired replays of that episode --
the same stills, prompts and seeds, only the code moved -- scored with the new
`scripts/otr_replay_ab.py` (2131cab5).

**1. Joins stepped darker (PBUG-20260928-03, Bible 12.183).** The terminal frame
that seeds each chained segment came out of ffmpeg's default yuv420p-to-RGB
conversion 1.3 luma dark; the compositor's upscaler decode, the credits backdrop
and the canvas poster had the same conversion. Fixed ae7b4515, 1f457e75,
625f40de. Live: median join brightness step -2.08 -> -0.13.

**2. A 3 Hz pulse of softness (PBUG-20260928-04, Bible 12.184).** The 16-frame
temporal decode tile blended every 8th frame. d12f5b61 (64-frame tile), kept in
v5. Live: pulse 0.72 -> 1.03; detail on the frames v3 blurred +62% and +51%. On
the 8 GB 4060: peak 7,579 of 8,188 MiB through 161-frame segments.

**3. Recipe v4 cut away from its stills (PBUG-20260928-05, Bible 12.185).** v4
took Lightricks' canonical distilled schedule and LTXVPreprocess(38): 62 of 68
beats sat further from their still by frame 24, 58 left it entirely, many as a
hard cut. A 2x2 on the 4060 blamed the schedule (frame 24: v3 17.0, compression
alone 20.5, schedule alone 31.2, v4 30.1); the preprocess softened the frames.
v5 (39bb1f18) is v3's sampling with the 64-frame tile -- its production graph
differs from v3's only in temporal_size (diffed node by node) -- and 6c18ba84
requires LTXVPreprocess only when the graph builds it. Live: v5 holds its still
like v3 (frame 12: further in 37 of 70 beats, median +0.07). v4 was on main for
42 minutes and never published.

**In otr/obs, for his eye:**
- costume_masquerade_20260928_155329 -- v5, the one to judge.
- costume_masquerade_20260928_150631 -- v4 (the cut-aways; reference only).
- costume_masquerade_20260928_153132 -- measurement arm: canonical schedule, no compression.
- clink_bone_20260928_154809 -- v5 on the 8 GB 4060 (copied over).
The 4060's own obs also holds its clink arms (152154 v4, 153604, 154102).

**Also on main since 2.3.13:** e18a855e (a cloud-writer piece that must arrive
whole gets room for itself) and fbdffba5 (ledger_clean refuses a reworded or
split repeat of the kept line).

**Review roster (post-push).** Sonnet and Composer 2.5 on d12f5b61 (both hold);
ae7b4515 (both hold; Sonnet's gap became 625f40de); 1f457e75 (Composer said the
scale flags cannot govern `format=rgb24` -- measured -1.27 -> +0.03, refuted);
625f40de (Composer's "may not ride the same path" -- measured -1.83 -> +0.03,
refuted); 39bb1f18 (both hold; Sonnet's note became 6c18ba84); 6c18ba84
(Composer holds). Codex is out of credits until 2026-10-03.

**Registry.** 2.3.13 is Active. Main now carries user-visible fixes beyond it
(v5 and the pulse, the join brightness, the two writer fixes): a 2.3.14 is the
operator's call, with the full suite first.

**Open.**
- Guided multi-frame joins (designed, r1 at kibitz-runs/2026-09-28-ltx-chain-guides,
  not built): a chained segment still restarts from ONE still, so motion
  restarts at each join -- joins change 2.5x their neighbourhood. It must win a
  paired replay; the design's LTXVPreprocess(29) on the guide frames is now
  suspect after -05.
- AnimateDiff blur: the 8 GB Lightning lane already decodes with vae-ft-mse; it
  renders 512x288 and holds each frame 3x (8.33 fps, by operator ruling). The
  levers are his: the compositor's AI upscaler, a larger canvas, the cadence.
- LTX 2.5 checked: its stage-2 decode tiles 64/16 (no pulse) and the join fix
  covers it (shared extract_terminal_frame). `LTX25_DECODE_TEMPORAL_SIZE` /
  `_OVERLAP` (33/4) in ltx25_recipe.py have no consumer.
- The 4060 clean-room clone sits detached at 39bb1f18; both headless servers
  are stopped.

## 2026-09-28 18:30 -- app form, Start here note, Space saver full, My Story on the roll

Operator requests after the v5 wrap-up, all on main:

- **App form (295a7a09, 9573e5c4).** Story idea (`custom_premise`) and Episode
  title (`episode_title`) are off the app form of every workflow: "I don't want
  people using My Story to get confused." Both stay on the canvas, blank.
  README and RUN.md say where to set them.
- **Start here note (01abedc1, 26d6ee94).** Fable's redesign, operator:
  "perfect". Stock pale_blue (#2a363b / #3f5159), no backtick code spans (they
  drew blue-on-grey chips), all six banks with the house three first, Visual
  style, one link to apple/MACHINES.md, no custom_premise, "your ComfyUI output
  folder under otr/obs". Checked live on a ComfyUI canvas.
  tests/test_workflow_notes_are_never_sent.py pins names, labels, no backticks.
- **Space saver ships full (26d6ee94).** Operator: "full for everyone". All 27
  workflows carry "full (keep only the published video)" and so does the node
  default; a graph saved before the widget opens full too, and only an API
  prompt that omits the key reads as off. A paired-replay measurement leg that
  needs the working files passes
  `--set OTR_LedgerScriptWriter.asset_cleanup="off (keep everything)"`.
- **My Story on the roll (3dec8a5b).** Four of his randomized runs each stopped
  in a second: leftover My Story text with Story bank on roll. Operator rule:
  the roll uses the boxes if it lands on my_story (DEFAULT_IDEA when empty) and
  ignores them anywhere else, logged; a bank picked by hand with filled boxes
  still refuses, now in the form's words. StoryInputPolicy.rolled, set at the
  validator, the writer's pre-roll check and _resolve_inputs. BANKS.md and
  README updated. **Follow-up a53865cd** (Sonnet 5.5, 3 confirmed): the
  pre-roll check had reached `_ROLLS` before the replay shortcut (3 replay
  tests red -- my scoped run grepped by topic and missed files importing the
  writer as W); the story module now keeps its own BANK_ROLL_LABEL, held equal
  to the roll module's by a test. Wiring tests added at the gate and the
  writer; each hookup switched off in turn fails its own test. And the Space
  saver docs claimed an older saved graph "still reads as off" -- false: the
  frontend fills it from the widget default, so it opens full; corrected with
  every stale "ships off" line and an amendment to the standing ruling.
  Composer on a53865cd doubted that; measured on a live ComfyUI (:8000): the
  canonical cut back to a pre-Space-saver writer (34 widget names and values,
  links re-pointed by identity) loads with "full (keep only the published
  video)" and queues full. OTR's own loader refuses a names/values mismatch
  outright ("Nothing was loaded"), so a probe must drop both.
- **Saved workflows moved, not deleted.** Operator: the repo's workflows folder
  is the only home for workflows. All 8 saved workflows in the two ComfyUI user
  folders moved to `user\default\_retired_workflows_20260928\` beside them
  (ComfyUI (1): the BARK test; Documents\ComfyUI: seven, incl. the v1 radio
  drama). Both installs junction the pack to the repo.

**Review roster.** Composer 2.5 on every commit; Sonnet 5.5 on 3dec8a5b +
26d6ee94 (3 confirmed, fixed in a53865cd). Composer's findings that held became 9573e5c4 (docs) and the obs
wording in 26d6ee94; its "free text is not tested" note was declined (no
doc-wording tests).

**Open.** His Desktop ComfyUI must be restarted to load the roll rule and the
Space saver default. 2.3.14 (v5, the brightness fixes, these changes) is his
call; the full suite is 17,273 passed / 0 failed at a53865cd (12 min).

## 2026-09-28 23:00 -- Kokoro speaks five more languages on Python 3.13; the language settles at Run

- **The failure (PBUG-20260928-06, Bible 12.186 at 4ddba6c).** Three of his
  randomized Desktop runs (ComfyUI Desktop, Python 3.13) wrote their scripts
  and died at the first Spanish line: the ONNX Kokoro was pinned to en-gb and
  refused every other lang_code. Operator: "we standardize everything to ONNX
  for Mac compat, but if ONNX doesn't handle languages that's a problem."
- **4f973e98 -- ONNX speaks Spanish, French, Italian, Portuguese, Hindi.**
  EspeakPhonemizer repeats misaki 0.9.4's EspeakG2P (phonemizer and espeak-ng
  already come with kokoro-onnx) and the ONNX backend hands kokoro-onnx the
  phonemes. Measured: phonemes identical to misaki (Sonnet: 821 hand-written
  and 7,500 random strings, 0 mismatches; torch KPipeline feeds its model the
  same phonemes; ONNX and torch waveforms correlate 0.988-0.996 with identical
  Whisper transcripts); all five spoken on the Desktop's own Kokoro, clips
  sent to him.
- **2326632b -- the language settles at the queue-time gate.**
  `settle_prompt_language` runs after the model rolls and before any
  download: a language picked by hand that the run's voices cannot speak
  refuses when he presses Run; a rolled one loses those from its pool, logged
  with the reason. It asks the same questions CastLock and the engine do
  (`cast_voice_engines`, Kokoro's `language_gap` / `lang_code_gap`,
  `voice_pool(host, language=row)`). 0.47 s cold, 0.01 s warm.
- **85a1d304 -- Sonnet QA of 4f973e98 (six findings).** A chunk with no
  vocabulary symbol (a lone inverted question mark) is skipped instead of
  killing the line; the ONNX load is atomic and an espeak row with no
  phonemizer refuses rather than reading as English; golden phonemes let the
  parity test run on 3.13 boxes; the LANG_CODES pin reads kokoro's source
  (importing it left `kokoro` in sys.modules and the node import tests failed
  whenever they ran after it -- the full suite passed only by file order); the
  ONNX log names its lang; AGENT_INSTALL.md and VOICES.md corrected.
- **e4ecb822 -- Sonnet QA of 2326632b (six findings).** misaki's Japanese and
  Chinese extras are Kokoro's own, now required only when Kokoro voices the
  cast: Google TTS voices Japanese and Mandarin on 3.13 (the misaki rule used
  to refuse it); the refusal says what works in the form's words (google_tts
  with his Google API key, or turn 'Randomize audio (non-cloud)' off); one
  language left is logged as the episode's language; the English row never
  narrows the voice pool; README, MULTILINGUAL.md and the pool tooltip.
- **Declined, with reasons.** The app form's "none = all" note is shared by
  five pools and stays true as "all that can run here" (a label change
  restamps 26 workflows); a language seed lands differently on a 6-language
  and an 8-language box, as the model rolls do (documented); wired voice
  inputs and missing weights are not language questions.
- **Behaviour change to know:** `OTR_KOKORO_BACKEND=onnx` on a box with the
  torch build now keeps the five espeak rows on ONNX (pinned by a test);
  Japanese and Mandarin still go to torch.
- **Open.** Japanese and Mandarin with Kokoro on Python 3.13: their libraries
  install there (pyopenjtalk-plus and fugashi have cp313 wheels; jieba,
  pypinyin, cn2an are pure Python), but misaki 0.9.4 and misaki-fork 0.9.6 pin
  Python below 3.13. Copying misaki's ja/zh modules (about 1,400 lines,
  Apache-2.0) would close it; he has the problem statement and it is his
  call. His Desktop needs a restart to load all of this; no Desktop Spanish
  episode has run yet.

**Review roster.** Composer 2.5 on 4f973e98 (two stale docs, fixed in
4491c738) and 2326632b (nothing held); Sonnet 5.5 on both (six findings each,
fixed in 85a1d304 and e4ecb822). Full suite at 2326632b: 17,298 passed, 0
failed; at e4ecb822: 17,312 passed, 0 failed.
