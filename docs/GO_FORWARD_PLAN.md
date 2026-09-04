# OTR Go-Forward Plan

**Forward-only.** Open work and nothing else. Receipts go to `docs/PROD_BUG_LOG.md`,
`docs/GO_FORWARD_ARCHIVE.md` and the dated receipt docs; rulings go to
`docs/OTR_STANDING_RULINGS.md`; only what is still TO DO belongs here.

Layout: THE QUEUE (ordered; item 1 is the next thing to do) -> Section 1 the build rows
behind the queue, in queue order -> Section 2 render work batched by the leg that proves
it -> Section 3 rulings owed by the operator, each with its default -> Section 4 parked
-> Section 5 Bug Bible promotion field -> Section 6 open risks. Standing traps, recorded
limits and lifted rulings live in `docs/OTR_STANDING_RULINGS.md` ("KNOWN OPEN" and the
2026-09-02 lifted-rulings section), not here. `docs/2026-*/` and `kibitz-runs/` are
gitignored: pointers to them are 5080-local.

* **Standing rulings, laws, review routing and the credit ladder:**
  `docs/OTR_STANDING_RULINGS.md` -- read it, it is not optional. The plan says what to
  do; that file says what you may not do while doing it.
* **Closed receipts:** `docs/GO_FORWARD_ARCHIVE.md` (not read to resume).
* **Machine guide:** `docs/MACHINE_MATRIX.md` (generated; edit `config/machine_classes.json`).
  **Pod manual:** `docs/RUNPOD_INSTALL.md` (Codex owns it). **Ship-readiness receipts:**
  `docs/ship-audit-2026-09-01/`. **The highest authority is still `CLAUDE.md`.**

## STATUS -- WHAT THE 2026-09-03/04 SESSION SETTLED (read before re-deriving any of it)

**2026-09-04, second half -- the runpod-found fixes landed in the commit that
carries this paragraph, as ONE invariant: a machine fact has ONE owner, and a
test proves the copies agree.** Full four-round `kibitz-plugin:kibitz` arc
(Codex gpt-5.6-sol, Antigravity Gemini 3.8 Flash, Cursor grok-4.6 -- r4's
Cursor seat by a Sonnet substitute plus the operator's manual paste), a Sonnet 5
QA pass on the named functions (clean; verified live that with nothing pinned
every answer on the 5080 resolves to the same binary and the same tree -- the
only string-level change is a resolved path where a bare literal used to be), a
Fable gate (ship-with-fixes, applied), then the suite -- **13483 passed, 0
failed** -- and the live publish matrix (three split roots, a pinned ffmpeg
copy, the shipping 8 GB profile; its receipt is in the commit that follows
this one). What landed, in 1.4a's rows:
ffmpeg has one resolver (`_otr_shared/ffmpeg.py`; measured live, 9 of 10 sites
had ignored `OTR_FFMPEG` on a box with ffmpeg on PATH -- PBUG-20260904-02); the
output root has one owner and `OTR_OBS_DIR` is a declared publication root
(R-A); the seven remote video adapters DECLARE `session_residency = "remote"`
and BeatSession no longer asks them for handles they do not have (R-B, the
word_razzle pod failure -- PBUG-20260904-03); the models root has one spelling;
the widget-drift gate has no silent exemption; the nvenc probe is cached per
binary. Three AST guards (`test_ffmpeg_single_resolution`,
`test_output_root_single_owner`, `test_models_root_single_spelling`) make the
claim a test, not a docstring. Bible 12.151-12.154 promoted. The pod
(`k0ph3m9491yp6s`) was stopped at 7:15 AM after its chain went idle; its seven
overnight episodes are in `otr/obs`. Side find: the 5080's own Q4_K_M writer
copy predates the 2026-08-29 pin (PBUG-20260904-04) -- re-downloaded. Open
threads live in 1.4a row G.

**The big one: character directives were being thrown away, 100% of the time.**
`derive_creative_directives` batches up to 15 beats and asks for one JSON row
each, but reached the model through a wrapper hardcoding `max_new_tokens=300`.
A row costs ~73 tokens, so a full batch needed ~1,100 and got 300: the reply
stopped mid-object, the parser returned `{}`, and the reseed loop retried the
SAME prompt under the SAME ceiling. Every character beat fell through to the
deterministic template -- a character PORTRAIT with no action verb in it -- which
is why the picture never matched the story. Fixed (PBUG-20260903-07) and proven
live: 14 of 14 beats from the writer, zero warnings, where it had been 0 of 14.

**Also fixed and pushed:** `"camera locked"` as the default when no camera was
given (3 lanes); the compactor silently dropping the CAMERA clause over a
12-word cap (2 of 32 registers, both in packs the operator watches); `Voice:`
reaching picture models in 90% of joint-AV prompts; the one-sentence camera ask,
now requiring a STRATEGIC move and banning framing-only answers; the wan/fastwan
bookends, which previously shipped a static seed with ZERO verbs.

**A portability bug only a second machine could find.** `scope_draw._has_nvenc`
asked whether ffmpeg was COMPILED with nvenc, not whether it can ENCODE -- the
third copy of a mistake already fixed twice, and `encode_sink`'s docstring
claimed to be the only such decision, which is why it survived. On a rented 4090
that lists the encoder and cannot open a session, `viz_camera` went FAIL -> PASS
on the fix. Any AMD / Mac / container host hit the same wall.

**`ideogram4_local` was never Blackwell-only** -- only its default was. It now
walks a precision ladder (nvfp4 -> fp8_scaled -> int8_convrot), all three in the
same UNGATED Comfy-Org/Ideogram-4 repo. NOT an 8 GB lane: fp8 is ~29 GB all-in.

**Published filenames now say what made the episode** --
`<title>_<ts>__<style>__<video>__<image>__<tts>__<bank>_final.mp4`. The old tail
was compositing noise and actively misleading: this session read `procgen` as a
render engine and built a wrong diagnosis on it. Obs copy only; archival
untouched.

**Rented-4090 lane matrix (1 act each), and read the failures correctly:**

| outcome | lanes |
|---|---|
| PASS | viz_camera, viz_green, still_flat, still_motion, still_pan, still_word, animatediff15_v3_haunted, ltx_8gb |
| TIMEOUT (driver's 40 min, render survived -- NOT a defect) | wan_ti2v, fastwan_8gb |
| OOM at decode (known 24 GB Ada negative) | ltx25_foley_plus, ltx25_mime, ltx25_video |
| dependency absent on that pod | ltx_video, mesh_stage (portable Blender) |
| REAL defect -- FIXED 2026-09-04 | word_razzle -- refused every beat long enough to split, because `BeatSession` demanded a handle identity from an adapter that holds no handles. The seven remote adapters now DECLARE `session_residency = "remote"` and the demand is skipped for exactly them, fail-closed (kibitz runpod-found-fixes R-B; PBUG-20260904-03) |

A 3-act AnimateDiff episode also passed unattended in 33.7 min.

**STILL OPEN, and today's audit CONFIRMED it rather than fixing it:** item 3d
below -- ten of eleven lanes lead the prompt with the cast's face paragraph via
`motion_common.compose_parts`, 83 words of a hard 100-word cap on `wan_ti2v`, so
the camera clause falls off the end. That is a SHARED-composer change touching
eleven lanes on both machines and is deliberately not a drive-by.

Eight pod-driving lessons landed in `docs/RUNPOD_INSTALL.md` section 7A and its
failure atlas.

---

## THE QUEUE -- READ THIS FIRST (ordered by the operator, 2026-09-02)

**Standing constraints:** the 5080 loop is untouched -- nothing ships that reduces
tomorrow's `obs` count. The pod stays STOPPED until item 9. One coder window per file
owner (CLAUDE.md section 1); every chunk = focused tests + full suite + Bug Bible + commit
AND push + `HEAD == origin/v2.0-alpha`.

**1. THE REGISTRY -- alpha.17 IS PUBLISHED AND VERIFIED (2026-09-03). Two SEPARATE goals
live here and conflating them is what made this feel stuck: (a) an INSTALLABLE listing,
which needs an admin approval we cannot self-serve, and (b) the "N Nodes" count on the
card, which is a different pipeline entirely -- and which, MEASURED 2026-09-03, is
STALLED ON COMFY-ORG'S SIDE and is not ours to fix at all (see the ANSWERED block
below; the newest successful extract in a 360-pack sample is 2026-04-28).** The old
control experiment (republish alpha.8 byte-identical) is CANCELLED -- the cause was found
directly instead.

* **alpha.17 published 2026-09-03 17:56Z, commit `524426ee`, Action green, and VERIFIED
  AGAINST THE ARTIFACT rather than the commit.** The CDN zip
  (`.../2.0.0-alpha.17/node.zip`, 6,397,276 bytes, 814 files) was downloaded and grepped:
  `OTR_ENABLE_HTTP_RENDER_ROUTES` present in `__init__.py`, the pycairo marker and the
  `kokoro-onnx` line present in the shipped `requirements.txt`, and none of `tests/`,
  `.claude/`, `.github/`, `kibitz-runs/` leaked. The registry recorded **19 dependencies**
  (alpha.16 recorded 18) -- the static-list check that alpha.3 taught us to run.
  Status is `Pending`, which is the 30-minute scan queue, not a verdict.
* **alpha.16 scanned 0 critical / 157 info (2026-09-03), the first zero-critical scan
  since alpha.8.** The two `critical` `prohibited-string` findings that flagged alpha.9
  through alpha.15 were the writer declaring the session-bearer hidden input beside
  `api_key_comfy_org` (PBUG-20260902-04); removing it cleared both.
  `tests/test_registry_prohibited_strings.py` guards every shipped file. Confirmed still
  true of alpha.17: the four remaining `auth_token_comfy_org` occurrences in `nodes/` are
  byte-identical to alpha.16 and are a consumer-side NAME CHECK, not a declaration.
* **THE NODE COUNT WAS A DEPENDENCY BUG, NOT AN APPROVAL PROBLEM -- FIXED IN `aff1f9c4`.**
  The "N Nodes" badge is fed by Algolia's `comfy_nodes` array, which is populated by
  Comfy-Org's `node-pack-extract`: a Cloud Build container that boots a headless CPU
  ComfyUI on Linux, runs a bare `pip install -r requirements.txt` under `set -e`, and
  reads `/object_info`. **It is independent of Flagged/Active, so no approval would ever
  have populated it.** Measured: `/versions/2.0.0-alpha.16/comfy-nodes` returns
  `{"comfy_nodes": null, "totalNumberOfPages": 0}` where
  `comfyui-videohelpersuite/versions/1.7.9` returns real rows. Cause: `pycairo` publishes
  21 Windows wheels, 1 sdist and **zero Linux wheels**, so that install had to build from
  source against `libcairo2-dev` and died. `requirements.txt` now marks it
  `sys_platform == 'win32'`, mirrored into `pyproject.toml`.
  **Cost, measured not asserted:** one function. The only `import cairo` in
  `scope_draw.py` is inside `paint_mandala`, so importing the module never imports cairo;
  `freq_bars_green`, `cfr_flags`, `find_ffmpeg`, `encode_silent_mp4` are cairo-free, so
  the scope overlays, caption burn, silent composite and encode sink are untouched.
  `viz_mxc_mandala` guards `load()` and `assert_usable()` and appears zero times in
  `otr_canonical.json`. The Linux pod keeps the engine because
  `scripts/otr_pod_provision.sh` apt-installs the headers and now pip-installs pycairo
  explicitly -- do NOT "simplify" that line away.
  **ANSWERED 2026-09-03 18:30Z, AND THE ANSWER CLOSES (b) AS NOT-OURS. DO NOT CHASE
  KOKORO.** The re-check ran: `/versions/2.0.0-alpha.17/comfy-nodes` is still `null`.
  The paragraph that stood here read that as "something else in the install also fails
  on Linux -- next suspects are `kokoro`'s spacy/thinc/blis chain against the 600 s
  extract timeout". **That is wrong, and acting on it would have burned a coder window
  on infrastructure we do not own.** It offered a binary -- our fix worked, or our
  install still fails -- and omitted the third option, which is the true one.
  `?include_status_reason=true` exposes a field the theory was built without:
  **`comfy_node_extract_status`**, which has TERMINAL values (`success` and
  `invalid_format` both observed). Measured across **360 registry packs**: only 64 have
  ever recorded a `success`, and **the most recent successful extract ANYWHERE in that
  sample is 2026-04-28**. rgthree (Active, 47 prior successes) has versions sitting
  `pending` since 2026-05-09; kjnodes (Active, 24 successes) since 2026-05-05;
  florence2 since 2026-05-06. **Comfy-Org's `node-pack-extract` has produced no
  successful extract for any pack in over four months.** OTR records zero successes
  because OTR's first publish was 2026-08-22 -- four months AFTER the pipeline last
  worked. It never had a chance, and no `requirements.txt` change reaches it.
  **`aff1f9c4` STAYS.** The pycairo marker is correct on its own merits -- a package
  publishing 21 Windows wheels and zero Linux wheels genuinely does not install in that
  container -- and it is what makes a Linux `pip install -r requirements.txt` work at
  all, extract pipeline or no. It simply was not the node-count blocker.
  **Reproduce before re-opening this:** compare the newest `success` timestamp across a
  broad pack sample against OTR's first publish date. If a pack ever extracts
  successfully again, the question becomes live and OUR deps become worth re-testing;
  until then the card's node count is not a work item.
* **STILL FLAGGED IS THE EXPECTED OUTCOME for (a), not a failure.** Active needs ZERO
  findings of any severity or an admin batch approval; 157 info-level YARA hits (env
  reads, ffmpeg subprocess, opt-in cloud lanes) keep it Flagged forever on the automated
  path. The gate is a literal `if issues == ""` on the scanner body -- one info finding
  and 157 are identical. There is no publisher self-service route to Active, confirmed by
  reading `Comfy-Org/registry-backend`. Corroborated independently: rgthree's two newest
  publishes are Flagged too, so the ruleset tightened in late August and comparing against
  older Active versions of popular packs proves nothing.
* **ALPHA.17 SCANNED 2026-09-03 19:06Z: 158 findings, ALL `info`, ZERO `critical`, status
  `Flagged`.** The credential fix held across the bump. The delta from alpha.16's 157 is
  exactly +1 `python_environment_manipulation` (102 -> 103), which is the route gate's own
  `OTR_ENABLE_HTTP_RENDER_ROUTES` env read that alpha.17 added -- i.e. the one new finding
  is the feature we shipped, behaving as designed. Flagged with zero criticals is the
  expected outcome, not a failure.
* **THE NODE-COUNT HALF IS ALREADY FILED, AND HAS BEEN SINCE 2026-08-24: the operator
  opened `Comfy-Org/registry-backend#203`.** It is still OPEN with no comments, labels or
  assignee. It names a cause this window's 480-pack survey could not have found on its own
  -- **a backfill cron scheduled for February 29th**, a date that does not occur in 2026 and
  next occurs in 2028. The operator added the survey as a comment on 2026-09-03 and
  corrected the issue's date window: extraction did not stop in February, it stopped
  **2026-04-28**, and 19 packs' newest success falls in April with nothing after. Do NOT
  open a second issue on this; comment on #203.
* **PARKED BY THE OPERATOR 2026-09-03: THE REVIEW REQUEST IS THE LAST THING WE DO,
  AFTER WE SHIP AND DROP THE ALPHA TAG.** Not a deferral -- a sequencing decision,
  and the measurement behind it is that **admin approval is PER VERSION.** Verified
  on `comfyui-video-xy-plot`: all five of its versions carry their own
  `{"message": "subprocess: ffprobe", "by": "dr.lt.data@gmail.com"}` note, four of
  them published on the same day. So approving alpha.17 buys nothing for
  alpha.18, and this branch shipped five versions in five days. Asking now would
  spend a reviewer's time on a version superseded within the week -- which is the
  behaviour that actually annoys a maintainer, as distinct from asking at all.
  **File it on the first NON-ALPHA version, the same day it publishes**, so the
  version under review is the version a stranger would install. Include the
  question about node-level review; it costs them less than five separate ones.
  Everything is ready and needs no further work:
  `docs/2026-09-03-registry-review-request-READY.md` carries the title, the body,
  alpha.17's version id, counts read from its own scan, the zero-is-unreachable
  measurement, and the video-xy-plot precedent. **Re-read it before posting** --
  the version id, the counts and the "0 critical" claim all move with each new
  publish, and a request quoting a superseded version is worse than none.
* **The old NEXT line, kept so the parking is legible:** file the manual review
  request. Draft ready and retargeted at alpha.17 in
  `docs/2026-09-02-registry-manual-review-request.md`. **Precondition 2 (grep the
  published zip for the route gate) IS NOW SATISFIED** -- see the verification above, so
  strike that step. **BOTH REMAINING PRECONDITIONS ARE NOW SATISFIED
  (2026-09-03):** alpha.17's scan landed at 0 critical, and every finding count in the
  draft has been re-read from alpha.17's own scan (158, not alpha.16's 157). The draft is
  ready to post as-is. Filing is a public post on Comfy-Org's tracker and needs the
  operator's explicit go -- it is the only thing still holding it.
* **A REGISTRY-TESTER REPO (operator idea, 2026-09-03): worth it for INSTALL
  verification, NOT for scanner probing.** Publishing a small pack under his own publisher
  to confirm a dependency set actually installs in their Linux container is ordinary use
  and would have caught the pycairo bug for free, before it cost four version strings.
  Iterating publishes to map which YARA patterns trip is the part to avoid: it is noise in
  a real security queue, and it is **unnecessary** -- `?include_status_reason=true` already
  returns all 157 findings itemized by rule, so there is nothing left to discover.
* DONE WHEN a version reads `Active` and a Manager install on the 4060 lands the OTR nodes.
  Never version-delete: a soft delete burns the string permanently.

**2. THE KOKORO-ONNX BACKEND -- DONE 2026-09-02, receipts archived.** Both proofs
published and eyeballed ("this episode is perfect, voices great"). Its one leftover, the
kokoro-onnx dependency line reaching registry installs, rides the alpha.17 bump in item 1.
Detail in `docs/GO_FORWARD_ARCHIVE.md` and `docs/2026-09-02-kokoro-onnx/`; row in
Section 1.1.

**3. THE CORRECTNESS BUGS -- in this order.** Story quality is done; these are
correctness defects (a gender or voice contradicting the source, a beat that renders the
wrong picture, a leg that dies late).
* **3a. Character gender ladder -- DONE 2026-09-02.** Receipt archived; row in Section 1.2.
* **3b. PROMPT v3 "DRAW THE CRUX" -- HALF A SHIPPED AND PROVEN 2026-09-03. HALF B IS OPEN
  AND IS THE NEXT CODER WINDOW ON THIS LANE.** The AnimateDiff lane was composing an
  invented costume plus a leaf written from that costume plus a framing law -- none of the
  three knew the story, which is why it drew a charcoal coat and a satchel for an episode
  about film canisters. Half A (`b198026a`'s parent merge `b97cce6d`) composes pack cue +
  crux kernel (the episode's own `key_objects` in its own setting) + light + world motion +
  a vantage mapped from the stored mode, reading neither the stored motif nor the leaf nor
  the law. Render-time only, so a frozen episode replays with a byte-identical seed --
  `render_request_hash` never included the prompt. **Proven:** same-seed A/B in `otr/obs/`
  (21:08 old vs 23:59 new, seeds identical 8/8, prompts differ 8/8), every prompt composed
  live at exactly the token count predicted on CPU, 32.9 tokens against v2's 40.2.
  Operator verdict on the video_art leg: "surprisingly good... it's about Huckleberry Finn
  and there's actually a river!"
  **HALF B -- THE OPERATOR RULED ITS SHAPE 2026-09-03 AND r1 INVERTED ITS
  MOTIVATING CASE. Read `docs/OTR_STANDING_RULINGS.md` "THE BEAT'S SUBJECT IS A
  PHYSICAL ARTIFACT" before touching this.** In short: extend the EXISTING batched
  Ghost author with the beat's dialogue (never a second pass), and have it name a
  PHYSICAL ARTIFACT of the story, preferring one the beat refers to -- never an
  abstraction and never a noun the dialogue mentions only to say it is absent.
  **The truck must NOT be drawn:** "this isn't just some dusty list of truck
  routes" is a rhetorical negative, Ellie is holding a ledger in an archive, and
  the r1 panel caught that the item's own worked example had been read backwards.
  The residual defect is that `resolve_crux_kernel` picks
  `objects[ordinal % len(objects)]`, so the beat about the ledger drew `pen`
  because it was the third row. r1 also corrected three inherited claims: the
  render batch DOES have `IS_CHANGED` now, replay returns before the author runs
  (so a migration needs an explicit seam at the replay boundary, not a version
  check in the author), and the beat's own `text` projection is the accessor --
  not a `_ghost_line_index` join, which silently misses synthesized beats. r2 is
  running; records in `kibitz-runs/2026-09-03-prompt-v3-half-b/`.
  **The old framing, kept only so a reader of the history understands what
  changed:** subject-appropriate motion needs the beat's own dialogue in front
  of the writer -- Half A buys the story's object, place and light on every beat and cannot
  buy its motion. He named the gap himself on the ledger episode: "I don't see any trucks
  though, it does mention a truck once" -- `truck` is in one spoken line and not in
  `key_objects`, so Half A had no path to it. Half B changes the STORED object, so it needs
  the version-dispatch discipline r2 specified and a re-author path for replay. Full arc
  before code. Anchor and the four-round records:
  `docs/2026-09-02-animatediff-ledger-experiments/prompt-rule/` and
  `kibitz-runs/2026-09-02-prompt-v3-crux/`.
  **Cheap and unblocked:** the kernel joins subject and place with a fixed `"in the"`,
  which reads wrong on a few settings ("a spinning turntable **in the** riverbank"). One
  small preposition fix; waiting only on his eye, not on Half B.
* **3c. The open defect list** -- the P0 / source-span cluster, the orphan-occupancy
  registry (full arc before code), the coverage and routing rows. Section 1.4.
* **3d. Item 3b-of-the-old-numbering, the OTHER video lanes -- measured, not started.**
  Ten of eleven lanes lead their prompt with the cast's face paragraph (83 words on one
  real episode) via `motion_common.compose_parts`. On `wan_ti2v` that is 83 of a hard
  100-word cap that truncates mid-sentence, so the camera clause silently falls off.
  **AMENDED by the r3 reviewer:** ADD a crux clause beside `appearance` on the silent
  image-to-video lanes; do NOT drop the face on redundancy alone -- the foley/mime lanes
  drop it because their joint latent SPEAKS the prompt, which is not a general I2V rule.
  Audio-in lanes (HuMo, the h3 audio lane, `ltx_audio_in`) and the two Google
  text-to-video lanes keep the face unconditionally. Runs after his eye on Half A.
  Measurements: `docs/2026-09-02-animatediff-ledger-experiments/prompt-rule/other_lanes_audit.md`.

**4. THE 8 GB SHIP SET -- what the clean room proved becomes a saved dropdown set.**
Klein stills run on 8 GB under stock launch flags (Leg C5: ~21 s a still) and LTX 2.5
renders there at ~14 min a clip; the Leg C5 episode PUBLISHED 2026-09-02 12:10
(`signal_lost_rationed_breath_20260902_060027`, 24 clips, 6 h 35 min; receipt in
`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`). (a) DONE 2026-09-02: the 8 GB row is PROVEN with its image lane on the
all-kokoro clean-room episode `signal_lost_the_ledger_of_shadows_20260902_134447` (the
operator's eyeball), receipt merged into the row's 4060 entry (7 episodes), matrix
regenerated;
(b) a shipped `otr_8gb_klein_ltx25` profile and variant (the clean-room profile is
untracked on the 4060 -- promote it, do not rewrite it); (c) the eight profiles that still
pair an 8 GB ceiling with the 12B writer get the E2B writer the 8gb class ships. Section
1.5.

**5. THE 4060 TEMPLATE TEST -- the frictionless capstone.** Runs once, after items 1 and
2, on a tree with the bugs above closed: install from the registry on the 4060, load the
8 GB saved-dropdown variant, click run, one published episode with zero hand steps. ONE
JSON ships for now (`otr_canonical`, kokoro on both voice slots -- operator 2026-09-02); a
4060 dropdown-friendly JSON is saved only after this testing is done, and that save is
what the test promotes. What is still open on the set is in Section 1.6.

**6. THE LOCAL-LLM ACCEPTANCE SWEEP** -- Leg 0 in-process preflight (Section 1.7,
~15-20 min, idle GPU), then the four canonical legs of Batch R3.

**7. DOCS DELETION PASS -- DONE 2026-09-02.** `docs/` went 644 tracked files to 237 across
three deletion-only commits (45895801, 2bf15784, 607a5ee7). Kept set in Section 1.8.

**8. DESIGN ROWS -- each gets a full arc before code, and none is the next coder
window:** the one-manifest auto-download (1.9), the ship-audit survivors (1.10), the
adaptation design (1.11), the style / identity decisions (1.12).

**9. THE NEXT POD SESSION -- one rental.** The provisioner's acid test (saved template ->
provision -> one published episode, zero hand steps) AND a looped lane sweep on the same
dollar. Receipts file under the 16gb class; there is no 24 GB class or profile by ruling
(95feac86), and Section 3 L asks whether one is wanted. Section 2, Batch R8.

**DEFERRED BY THE OPERATOR (2026-09-02), not scheduled:**
* **Token rotation** (Codex's security note, relayed in-session 2026-09-02 and recorded
  nowhere else: an earlier temporary diagnostic briefly captured inherited HF / OpenRouter /
  provider credentials; the file was removed and the leak path hardened). Rotate when
  convenient; nothing in the queue waits on it.
* The Section 3 question list (A)-(L), each with its default if unruled.

**WATCH -- recorded, not scheduled:**
* **`obs_publish OK` is not proof of an episode (2026-09-03, cost two GPU legs).** Three
  replay bugs shipped that morning; TWO of them published green with a broken file --
  one second of picture in an 85-second episode, because a placeholder wire set the
  procgen overlay's length and `PostUpscaleProcgenBlend` takes the shorter input. What
  caught it was measuring the published file's DURATION at every pipeline stage (render /
  blend / caption / credits / mux), never the log. A stage that changes the duration is
  the defect. PBUG-20260903-01/02/03; two 7.5 MB casualties are still in `otr/obs/` and are
  deliberately not swept -- `..._231401` and `..._233738`, both "The Faded Ledger".
* **An unrelated rotation loop owns the box's spare cycles.** `video_rotation.sh` has run
  since 2026-08-31 out of session `8a385813`, cycling 16 engine/image lane combos one act
  at a time, forever, and it deliberately skips AnimateDiff. It is Jeffrey's daily obs
  proof; leave it running. It does not block a registry publish (that runs in GitHub
  Actions against the repo, not this box) and holds the resident server on :8000, which
  only matters for a local boot check.
* Zero-frame beat: the root fix landed (`415b1ba0`, PBUG-20260831-01: an untimed
  `music_open`/`music_close` with no timed mirror got no duration; it was also the ghost
  pool r1's "roughly 70 minutes" death). Owed: one canonical leg of that shape reaches
  `otr/obs/`, recorded as a "verified live" line under the PBUG; and confirm `shot_b006`
  (`mode=object source=deterministic_fallback`) was the ghost shot for those music rows,
  or it is a second defect. No coder time.
* Whether a PRUNED P0 index is ever accepted has not been measured live (the deterministic
  rung became reachable at `47c554fa` after the campaign stopped); the next P0 campaign's
  instrumentation answers it.
* `OTR_LedgerFreezeCascade` failed twice and the message was never captured -- the
  runner's eight-frame traceback truncates it. Next occurrence, read the SERVER log.
* `OTR_VideoRenderBatch` `RenderError` cluster -- triage after items 2 and 3.
* The two eyeball re-observations (announcer framing, name-splice #2) ride any real
  render leg -- Batch R5.

---

## SECTION 1 -- THE BUILD ROWS BEHIND THE QUEUE (in queue order; everything here ships without a GPU leg)

All of this is provable by the suite, a scoped review, or an offline read. Two windows,
split by area, both push (CLAUDE.md section 1: the 5080 owns `nodes/`, the canonical and
its variants and `pyproject.toml`; the 4060 owns the profiles it has proven and the
fresh-install path). Every chunk = focused tests + full suite + Bug Bible + commit AND
push + `HEAD == origin/v2.0-alpha`. A row with a design choice in it gets a full kibitz
arc BEFORE code (substitute seats allowed, roster stated); a grep-and-fix gets one Sonnet
QA on the diff.

**Running beside the order -- the dead-code campaign and the knob census** (operator
standing instruction 2026-08-28: keep hunting "until there are no more dead code
candidates"; STOP RULE = two independent blind deep sweeps returning zero CONFIRMED
findings). Open: the V5 sweep (18 findings, `docs/2026-08-28-dead-code-hunt-v5/`,
5080-local) is under adjudication; the live hunt prompt is `docs/DEAD_CODE_HUNT_PROMPT_V5.md`.
The knob census (`docs/KNOB_CENSUS_PROMPT.md`) is a separate pre-ship pass: the operator
rules per row and the census informs what the 4060 template pins.

### 1.1 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (design item, kibitz arc BEFORE code)

**STATUS 2026-09-02: BUILT, proofs pending (see queue item 2).** Shape as shipped:
`nodes/_otr_audio_engines/_kokoro_backends.py` (torch path moved verbatim; ONNX path
CPU by design, explicit provider, 4-thread cap, voices from the existing `.pt` files
through a digest-named npz); `eng_kokoro.py` selects per `load()` via
`OTR_KOKORO_BACKEND=auto|torch|onnx`; the prefetch fetches `onnx/model.onnx` at boot only
when the ONNX backend will be used; canonical nodes 80/81 default both voice slots to
kokoro on `kokoro_builtin`; a character-side engine-agreement guard; five profiles fixed;
complementary `kokoro` / `kokoro-onnx` markers in `requirements.txt`; the generated "Voice
engines" table in the matrix. Design record: `docs/2026-09-02-kokoro-onnx/`.

Measured 2026-09-01: `kokoro-onnx>=0.6.1` + `onnxruntime` pip-install clean on Python 3.13
(kokoro-onnx pins `>=3.10,<3.14`; espeak-ng ships bundled for win / mac / linux; the torch
`kokoro` package stays `<3.13`). Receipts: PBUG-20260901-04 and `docs/OTR_STANDING_RULINGS.md`
"ALL AUDIO LANES SHIP ON KOKORO".

Shape (settle the details in the arc, do not re-derive these):
* A second backend inside `nodes/_otr_audio_engines/eng_kokoro.py` (247 lines; today it loads
  `KPipeline`). ONNX path: load model + voices, phonemize with espeak-ng, run onnxruntime,
  return 24 kHz audio. Prefer the torch backend only when `kokoro` is importable (3.12 boxes);
  ONNX everywhere else; bark stays the zero-dependency fallback.
* Weights: kokoro-onnx expects `kokoro-v1.0.onnx` + one `voices-v1.0.bin` (an npz keyed by
  voice, GitHub release `model-files-v1.1`); the HF mirror `onnx-community/Kokoro-82M-v1.0-ONNX`
  (ungated; model 86 MB q8f16 to 326 MB full, 55 per-voice `.bin` files at 0.5 MB) uses a
  different per-voice format. Pick ONE source, fetch it with the pack's existing
  `_otr_kokoro_voice_prefetch` machinery, and make the announcer/character voice ids map onto
  the same names the torch path uses so the cast ledger does not change.
* Registry deps: `kokoro-onnx>=0.6.1` and `onnxruntime>=1.20.1` in both manifests (plain
  PyPI wheels); `onnxruntime-gpu` optional. Keep the model on CPU by default when a video
  engine holds the GPU; it is an 82M model and faster than realtime on CPU.
* SHIP SCOPE (operator ruling 2026-09-01: "we can ship all audio lanes with kokoro"): in
  the same change, `workflows/otr_canonical.json` and every generated variant default BOTH
  voice slots to kokoro (`char_voice_engine`, `announcer_voice_engine`, `voice_bank`
  `kokoro_builtin`); indextts2, chatterbox, dia and bark stay in the dropdowns as
  install-it-yourself upgrades. That dissolves the indextts2-default half of the ship-audit finding (the
  reference-WAV preflight is the bullet below) and the Section 4 voice-bank item stays parked.
* THE COMPATIBILITY TABLE IS GENERATED, NOT HAND-KEPT: extend
  `scripts/otr_machine_matrix.py` to emit a "Voice engines" table into
  `docs/MACHINE_MATRIX.md` from `nodes/_otr_audio_engines/registry.py` (device_backends,
  requires_sidecar, requires_vendor, practical_without_gpu, model_requirements) plus a
  per-engine "ships with the pack / install on your own" column, so README can say "want a
  better TTS, install it yourself, here is what runs where" and point at one table.
* DONE WHEN: a clean 3.13 portable install renders a 1-act episode with kokoro voices for
  announcer and characters through `workflows/otr_canonical.json` and publishes to `otr/obs/`,
  and the same commit passes on the 5080's 3.12 venv with the torch path still selected.
* **Cloning engines preflight their reference WAVs** (the surviving half of the ship-audit
  indextts2 finding): preflight the resolved `ref_path` files in `OTR_CastLock` BEFORE the
  writer call whenever a cloning engine IS selected, so an opt-in indextts2 / chatterbox /
  dia user fails in seconds, not after a whole script.
* **The `pyproject.toml` dependency edit auto-fires a registry publish.** It rides the bump
  AFTER alpha.15 (and alpha.16 if the control runs) resolves -- never while a version is
  Pending, and never reusing a string (CLAUDE.md 7A).

### 1.2 CHARACTER GENDER LADDER (queue item 3a) -- the SPEC REWRITE is written; next is ONE review round, then code

**DONE 2026-09-02.** Round: Antigravity r2 on the driver anchor
(`docs/2026-09-02-gender-ladder/driver_anchor.md`, sections 8 and 9 carry the fold and the
proof) -- 7 must-fixes, all grounded and taken. Code: tiers 3-4 in
`scripts/otr_stamp_character_genders.py` (recall into a committed per-bank
`character_gender_index.json`, the first-name pool, the body-hash-anchored merge, the
Shakespeare scene stamper), the verdict fields and the alias-aware join in
`nodes/_otr_roster_gender.py`, greedy decoding at temperature 0 in the constrained closure.
Proof: `signal_lost_intensity_in_the_drawingroom_20260902_152901` -- ELIZABETH BENNET female,
MR. DARCY male, COLONEL FITZWILLIAM male, all `llm_recall` / `recalled` in
`meta.cast_source_contract.evidence`; the leg before it (`unhand_me_sir_20260902_151527`) is
the before-picture where Darcy rolled female because the join ignored the aliases. Operator
rulings folded: ARIEL / PUCK / ROBIN stay on the roll (locked index entries); Dr. Lira Kell is
female (locked). Left open, recorded in PBUG-20260815-04's follow-up: a given-name alias can
match a different character with that surname (COLONEL FITZWILLIAM via "fitzwilliam"; right
here by coincidence) -- a surname-only alias for the short_form tier is the next fork.

**TWO OPERATOR RULINGS, 2026-08-28, and they reshape the spec:**

* **Shakespeare: fill ONLY the 32 `unknown` roster rows.** KNOWN rows from the
  parsed dramatis personae stay untouchable; the ladder's lower tiers may fill
  the blanks.
* **THE WEB-SEARCH TIER IS REPLACED, not plumbed.** Operator's design, his
  words: *"just have the LLM decide -- ask what the likely gender of this
  person name is, have the LLM decide, and keep that in an index of names."*
  So tier 3 becomes an LLM VERDICT ON THE NAME (the model already knows
  Scrooge and Marley from training -- no live search needed), cached in a
  PERSISTENT name index so each name is asked once, ever. Tier 4
  name-frequency stays as the deterministic floor beneath it, keeping the
  ladder TOTAL when the LLM call fails. This dissolves both review rounds'
  biggest must-fix (the silent no-op web call): there is no web call. On his
  "is it not easy to query a search engine?" -- keyless search-engine querying
  is the fragile part (scraping is blocked/ToS; keyless APIs are thin); the
  RSS precedent covers feeds because feeds are MEANT to be fetched. The
  LLM-ask design avoids the whole problem, offline-first.
  The invented lanes (original, scifi_news_pro, media_archive) KEEP ROLLING
  by the standing ruling -- their characters do not exist, so no lookup of
  any kind applies.

Evidence and the v1 spec: `docs/2026-08-05-character-gender-ladder-SPEC.md` and
`docs/GO_FORWARD_ARCHIVE.md` (the prose lane stamps `characters: None`; the Shakespeare
sidecars carry genders; the consumer `_otr_roster_gender.py` is lane-neutral and correct).
v2 folds v1; the rulings above are baked into v2.

### 1.3 GHOST POOL -- uniqueness on the finalized prompt (queue item 3b; r1 is in, build)

r1 verdict (2026-08-31, Codex `no` / Antigravity `yes-with-fixes` / Cursor `no`; artifacts
`kibitz-runs/2026-08-31-ghost-clause-pool/r1/`, 5080-local; narrative in
`docs/GO_FORWARD_ARCHIVE.md`): **the pool is not too small, the duplicate check is.** The
picture is four slots (`GHOST_V2_SLOTS`) and the check reads one (`key = leaf.casefold()`
in `otr_shot_lock.py`), so two beats with the same leaf and different characters are
rejected although they render different pictures. That is why 6 -> 18 failed and why 50
would fail too.

**BUILD:** uniqueness on the FINALIZED POSITIVE PROMPT, applied identically to writer
output, replay and the deterministic path (capacity becomes clauses x motifs). Then a
bounded progression, total by construction: unused finalized prompt -> reuse a leaf where a
different motif keeps the prompt new -> reuse the least-recent signature, deterministic on
`episode_seed + beat_id`, never adjacent. The allocator appends a PER-BEAT reuse
disposition to that beat's existing `fallback_reason` (ShotLock stamps one batch-wide
reason today, which would erase the original model-failure reason). Only pool exhaustion
becomes recoverable: the ten `GhostAuthorError` raise sites (unknown mode, missing bookend
motif, invalid role, empty `motif_cue`) are structural corruption and stay loud. Shared
code: measure both boxes before pushing (CLAUDE.md 0B).

**CUT, so nobody rebuilds them:** the combinatorial generator, act-scoped uniqueness (no
authoritative act field exists), and "loud handover" (controlled reuse under a second name).

**DONE WHEN:** >18 same-mode beats complete; mixed replay plus fresh authoring completes;
all three paths share the invariant; adjacent finalized prompts never repeat; same seed
gives identical output AND receipts; every beat keeps a valid `ghost_prompt`; then the
failing five-act topology through `workflows/otr_canonical.json` with `obs_publish OK` and
the file on disk. The tests that encode the obsolete absolute-leaf rule
(`test_ghost_prompt_v2_lane.py:399-405, 437-451`; `test_ghost_signal_author.py:925-931`)
are REPLACED with the new invariant, not deleted.

*Open from r1:* whether "no adjacent repeat" is the right viewer threshold -- check it
against frames rather than more reasoning. (The r1 anchor's "roughly 70 minutes" failure
was the zero-frame beat, PBUG-20260831-01; not this row.)

### 1.4 OPEN DEFECTS THAT ARE CODING WORK (queue item 3c; a leg may prove some later, none needs a leg to FIX)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not.

**Line cites in this section drift; re-pin a row's cite when you touch it.** Engine adapters
live under `nodes/_otr_video_engines/`, `_otr_audio_engines/` and `_otr_image_engines/`;
bare `eng_*.py` cites mean those paths, and `render_driver.py` is
`nodes/_otr_video_engines/render_driver.py`.

#### The P0 / source-span cluster (2026-07-30)

- **`full_text` HTML block joins fuse tokens** (`...PolygonsNASA/JPL-`, `...School
  ofEngine`, `...doing.Let's s`, `...(AMR).The resea`): the dominant P0 span-rejection
  cause on live evidence. The RSS adapter strips tags without inserting whitespace, and
  `_normalize_span_source_text` can collapse runs but cannot insert a space that was never
  there. Owed, after Section 3 question B is ruled: name the adapter that builds
  `full_text`, insert the separator at admission without breaking any accepted ledger's
  `source_digest`, and pin a fixture from those four strings. Belongs in the source
  adapter, not the codex normalizer. History: `docs/GO_FORWARD_ARCHIVE.md`.
- **Re-pin before spending time on the two rows below:** `repair_literal_source_metadata`,
  `_validate_fact_index` and `a0_payload` no longer exist under `nodes/` (only
  `allowed_source_fields` survives, in `_otr_scifi_p0_contract.py`). Locate the deterministic
  P0 repair rung by behaviour, or tombstone both rows.
- **The deterministic P0 rung PRUNES SILENTLY, which violates the plan's own
  Invariant 3.** `repair_literal_source_metadata` drops an unsupported span, then its
  evidence row, then the fact -- and emits no receipt. An accepted P0 index simply
  has fewer facts than the model wrote, and nothing says which were dropped or why.
  Under "fail loud, not fatal" the degrade is the right direction and the silence is
  not.
- **The deterministic P0 rung is ALL-OR-NOTHING across an artifact, and can poison
  its own good work.** It is handed `a0_payload` (all seven keys) while
  `_validate_fact_index` restricts spans to `allowed_source_fields` (the projection).
  A quote rehomed into a field the projection omitted makes `post_validator` reject
  the WHOLE repaired artifact -- "cites source field ... outside the supplied P0
  evidence" -- so one unlucky rehome discards every correct prune in the same pass.
  Either give the repairer the allowlist or prune per row.
- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0 after
  two attempts on non-literal fact source spans; provider/model convergence, extends
  BUG-11.35. NOT a word/length gate. Blocks the last 120w receipt and the
  `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed in tree, reverify
  still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap
  `512`; the whole-artifact retry contracts LANDED @ `314dd481` are the base; the
  residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do not
  raise the minimum word target as a capacity workaround.

#### The orphan-occupancy registry (design item -- full arc BEFORE code)

`has_local_resident_llm()` (`nodes/_otr_model_loader.py`) reports "nothing resident" the
moment a timeout clears the cache dict, even while the orphan worker is still running CUDA
kernels on the model that entry described; `nodes/otr_shot_lock.py` and
`nodes/otr_video_render_batch.py` both trust that signal before starting visual or video
work. Shape: a process-global, lock-protected registry of in-flight generations,
registered before invalidation and cleared via `Future.add_done_callback`, with fail-fast
admission on `request_slot` and the two visual-entry guards reading real occupancy instead
of the dict's cleared-or-not state. Deferred three times as correctly out of scope for the
cache-bookkeeping fixes (PBUG-20260825-04, `fb67d059`; arc history in the archive), and
each cut of that fix found a new race, so this is a genuine design choice: full arc first.
`da2b7a36` (the dispatcher frees the writer before the first local still) narrows the
image-stage exposure but does not build the registry.

#### Coverage, canvas and clip-contract

- **The route lock is ONE NODE TOO LATE for the image phase** (found 2026-07-25, node
  order confirmed against the canonical JSON: `87 VideoDirector -> 88 ImageDirector ->
  89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher -> 92 VideoRenderBatch`).
  `resolve_final_shot_engines` runs at node 92, but stills are minted at 91 and image
  PROMPTS at 89. The landed fix closed the spine-validation gap; the image phase still
  relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`,
  whose docstring says it "mirrors the image dispatcher's effective-engine seam").
  **Chunk 1 of the coverage block is the fix.** Note node 89 precedes node 90, so
  hoisting to ShotLock still does not put MetaBrief downstream of the authority --
  that needs a VideoDirector-time freeze and is NOT in scope. (This is also the
  "image-phase still ownership" item from the campaign queue.)
- **The ShotLock WRITE-side canvas validation is still owed** (O1 judgment item 1).
  `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated from a possibly-empty
  policy. B5 made this non-load-bearing for the render (the engine declares its own
  canvas now), so it is no longer urgent -- the drift guard in
  `tests/test_ltx_8gb_canonical_canvas.py` covers the disagreement that matters. Close
  it when the general canvas resolver lands.
- **`ltx_av` underruns long beats** (found 2026-07-25, codex; confirmed). It caps at
  `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py`, default 497, env-overridable) and clamps to it
  before render. It is NOT "renders to target natively" as three earlier docs claimed.
- **`docs/ENGINE_MATRIX.md` prints the DECLARED contract only.** Once a profile pins an
  `ltx_8gb` ceiling the matrix keeps printing `9-161 step 8` for a tier whose real window
  is narrower, and the `--check` drift gate cannot notice because it diffs the registry.
  Owed at the prequalification step, not before.

#### Voice engines

- **bark rolls non-speech (a steady tone or a noise floor) on some lines and returns
  success** (PBUG-20260902-03; ROOT-PINNED 2026-09-02 14:20). The 4060 Leg C5 episode
  lost both announcer lines to a 1.4 kHz tone; a direct engine probe on the same box
  then rendered the exact announcer line as speech and a 9-word character line as
  seven seconds of noise plus a 2.5 kHz tone. So it is the engine's roll, not the
  announcer graph, not the knobs, not line length (the 180-character chunker already
  exists). Fix: an OUTPUT GUARD in `eng_bark.py` -- score each take for speech shape
  (dominant frequency 70-400 Hz with low flatness in most one-second windows), re-roll
  with `seed + 1` up to twice on a failing score, WARNING-log every re-roll, keep the
  best take if all fail (the ledger field is always filled). Design choice (threshold,
  retry count) -> full arc before code; test = a stub engine that returns a tone first
  and speech second yields speech in one retry. bark is opt-in now (kokoro is the
  shipped default on both slots), so this is a dropdown-fallback fix, not a ship gate.
- **Remember, do not build (operator 2026-09-02): the 5080's overnight runs ROTATE voice
  engines on purpose** ("for overnight runs I like to rotate voice models, to be honest,
  since the 5080 can handle them all"; "there's probably no rotation machinery, just
  something you need to remember"). The `config/profiles/otr_rot_tts_*` profiles are the
  instrument (bark / chatterbox / dia characters, chatterbox / dia announcers, and
  `otr_rot_tts_kokoro`, kokoro on both slots -- the stranger default belongs in the
  rotation too); whoever queues overnight legs cycles them by hand. Never fix his
  overnight profile to kokoro-only in the name of the ruling; the stranger-facing default
  stays kokoro on both slots.

#### Routing, env-capture and the credits card

- **`wants_talking_prompt()` escapes any routing freeze.** It calls
  `_recipe_config(self._recipe())`, and `_recipe()` (`eng_ltx_av.py`) re-reads
  `OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / the UNET name on EVERY call by documented
  design ("Read fresh every call"). So a `required="when_engine_talking"` row evaluated
  through the hook re-reads the environment after capture. S0b-core needs ONE shared
  `row_is_active(...)` evaluator over captured state, with the talking result inside
  `ltx_resolved`.
- **`provider_side` is a THREE-part rule, not an attribute.** `_is_cloud_video_engine`
  (`render_driver.py`) accepts a `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`.
  `cloud_kling_avatar` has no `provider_side` attribute and is caught by the id prefix
  alone, so an `engine_facts` builder using a bare `getattr` would classify it local
  and let the radio-host redirect send a cloud avatar to local LTX. Needs a regression
  on picked AND forced `cloud_kling_avatar`.
- **Env-read sites missing from the S0b inventory** (was four; the `OTR_ENABLE_LTX_I2V`
  site was DELETED by the 2026-08-28 retirement, so two remain): the `OTR_ENABLE_HUMO_HOSTS`
  reads in `render_driver.py` and `otr_meta_brief_image_prompt.py`, and the recipe / UNET
  re-read in `eng_ltx_av.py` (`wants_talking_prompt` / `_recipe`) outside `assert_usable`.
- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At
  512x288 (the ltx_8gb tier) col1 is 65px past its footer even with every ledger row
  this policy may drop already dropped; at 640x360 it is 12px over. Both are drawn
  anyway (a terminal node never destroys a finished episode) and LOGGED at ERROR
  naming the canvas -- the old behaviour was drawn, clipped by PIL, silent. At 288
  lines the three-column console is already a polite fiction: col3's scrolling
  transcript is as unreadable as anything col1 clips. This is a DESIGN job -- a card
  laid out for a small canvas -- not more ladder heroics.

#### 1.4a DEAD CODE / DUPLICATED-DECISION AUDIT (2026-09-04) -- three Sonnet lanes, driver-verified

Every row below was re-checked against the real files by the driver before it was
written down; the REJECTED list is part of the record precisely so a discarded
claim is not re-raised as a fresh finding. Line cites drift -- re-pin when touched.

**A. VERIFIED, HIGH -- the output tree is decided TWICE, and the copies disagree.**
`nodes/_otr_paths.py` is the documented owner (`OTR_OUTPUT_DIR` is "highest
precedence", :74-106). `nodes/otr_master_audio_mux.py` imports NOTHING from it and
carries its own `_episodes_root()` (:695) and `_obs_dir()` (:709), neither of which
reads `OTR_OUTPUT_DIR`. Proven by EXECUTION, not by reading:

```
OTR_OUTPUT_DIR=D:\RenderOutput , folder_paths=C:\ComfyRoot\output
_otr_paths.otr_episodes_root()  -> D:\RenderOutput\otr\episodes
mux._episodes_root()            -> C:\ComfyRoot\output\otr\episodes   AGREE? False
_otr_paths.otr_obs_dir()        -> D:\RenderOutput\otr\obs
mux._obs_dir()                  -> C:\ComfyRoot\output\otr\obs        AGREE? False
```

**Blast radius is the terminal publisher.** `_episodes_root()` gates
`_inflight_episode_for_stem()` (:797) -- a mismatch silently rejects a valid
in-flight ledger and falls back to filename-suffix peeling the module itself calls
fragile. `_obs_dir()` is where `_publish_to_obs()` writes the deliverable. The same
bare-`folder_paths`-or-`"."` idiom is copy-pasted again in `otr_caption_burn.py`
(:349) and `otr_silent_composite.py` (:1692).

**Why it has not bitten:** both launchers hand-pin `OTR_OUTPUT_DIR` =
`--output-directory` = `OTR_OBS_DIR` in lockstep -- by human discipline in two
separate shell scripts, not by any code invariant.

**CORRECTED BY KIBITZ r1 (2026-09-04) -- the first draft of this row was wrong
in three ways, and it matters because the first fix proposed would have broken a
contract test.** Full record in
`kibitz-runs/2026-09-04-runpod-found-fixes/r1/{judgment,final}.md` (local, gitignored):

* **Four owners, not two.** `_otr_ledger.py:182` reads `OTR_OBS_DIR` too, and
  `__init__.py:97-108` WRITES `OTR_OUTPUT_DIR` at import inside ComfyUI -- that
  pin, not "the launcher was skipped", is the mechanism behind the Desktop /
  bare-`main.py` split.
* **Two concepts, deliberately named apart.** The *episode workspace root*
  (`_episodes_root`) IS a true duplicate that ignores `OTR_OUTPUT_DIR` -- delegate
  it. The *publication root* (`_obs_dir`) is NOT: `_otr_paths.otr_obs_dir()`
  (:401-429) never reads `OTR_OBS_DIR`, which is a first-class override read by
  seven files including five launchers and exists to point OUTSIDE the tree (the
  documented two-tree split). The mux is right to honour it; `_otr_paths` is the
  incomplete one. "Delegate the mux to `_otr_paths`" for obs would have broken the
  split AND tripped `_validate_contract` (:220-236 raises on an out-of-tree path).
* **A contract test pins it.** `tests/test_output_tree_contract.py:48-50` lists
  `otr_obs_dir` in `_NOARG_HELPERS` and asserts in-tree. Any change ships WITH
  that test updated, or does not ship.
* **Open ruling, deciding in r2:** an explicit pin SKIPS the in-tree assert
  (driver's first lean) or registers as a *declared second authorized root* the
  way `_otr_ledger.py:168-191` already does (panel 2-1, and now the driver's lean
  -- it keeps the validator meaningful instead of holing it).
* Both docstrings -- mux "ONE owner", `_otr_paths` "THE obs dir" -- are false and
  must become true or go; this is the nvenc shape.
* **Acceptance:** Path equality is necessary and NOT sufficient. A live headless
  publish with `OTR_OUTPUT_DIR != folder_paths.get_output_directory()` landing in
  the watched tree. Caption-burn / silent-composite fallbacks are a rostered
  follow-up, not equals of `_publish_to_obs`.

**RESOLVED 2026-09-04 (kibitz r2-r3, built the same day).** Ruling R-A: the
pin is honoured INSIDE `_otr_paths.otr_obs_dir()` and skips the in-tree assert
for that path only (returned as typed, no `resolve()`); `_validate_contract`
untouched; the ledger already authorized both roots. The mux's two functions
are thin delegates (names kept -- the publication tests monkeypatch them).
r3 grew the scope: the canonical graph's node 84 (`OTR_SilentComposite`, empty
`output_path`) writes via `_default_out`, and node 93 writes beside it, so the
"rostered follow-up" above would have left the canonical chain split -- silent
composite, caption burn AND `portrait_ledger._output_base` now delegate too.
Enforced by `tests/test_output_root_single_owner.py` (AST, named allowlist:
`eng_mesh_stage` because ComfyUI's SaveGLB refuses any path outside
`folder_paths`' own dir -- NOT a leftover twin, do not "retire" it -- and
`vram_context_test`, a diagnostic outside the contract). conftest now strips
`OTR_OBS_DIR` at import (the launch recipe sets it on this box). The `__init__`
pin is preserved deliberately; removing it is its own design item.
**Still owed: the live publish matrix** (`--output-directory A`,
`OTR_OUTPUT_DIR=B`, `OTR_OBS_DIR=<watched>`; everything under B, the mp4 in
the watched dir, nothing new under A).

**B. VERIFIED, LATENT -- ffmpeg is resolved nine ways.** Three are BYTE-IDENTICAL
copy-paste (`otr_caption_burn.py:53`, `otr_master_audio_mux.py:49`,
`otr_silent_composite.py:123`) and each self-documents that this defect class has
already hit the pack twice. The same three correctly delegate `_ffprobe_bin()` to
`_otr_shared/ffprobe.py`, so the pack knows the right shape and did not use it here.
Two have live gaps, dormant only because nothing calls them:
`_otr_shared/encode_sink.py::find_ffmpeg` (:20) never checks `OTR_FFMPEG`, and
`_otr_shared/content_oracle.py::_ffmpeg` (:88) has no PATH fallback at all (that
module has zero importers). The rest disagree on what to return when nothing
resolves (`None` / `""` / bare `"ffmpeg"`).

**RESOLVED 2026-09-04 -- and it was not latent (PBUG-20260904-02).** The count
was twelve, not nine, plus three sites that ran a literal `"ffmpeg"` with no
resolver at all (`otr_post_upscale_procgen_blend` -- canonical node 93 -- ,
`eng_google_lyria`, `foley_stems`). Six copies read the caller's own bare
default as an operator choice, so on any box with ffmpeg on PATH the pin was
never consulted: measured live on the 5080, 9 of 10 resolvers ignored
`OTR_FFMPEG`. The cloud preflights refused env-only installs. ONE owner now:
`nodes/_otr_shared/ffmpeg.py::resolve_ffmpeg` (explicit choice -> pin -> PATH
-> the Windows install dirs -> `None`; reuses the probe's bare-name rule);
every runtime site a thin adapter with its own refusal; `resolve_ffprobe`'s
sibling steps go through it. Guard: `tests/test_ffmpeg_single_resolution.py`,
three AST rules, NO allowlist. Matrix: `tests/test_ffmpeg_resolution_precedence.py`,
PATH left available in every case. The nvenc cache is keyed per binary and the
second cache in front of it (`video_engine._check_nvenc`) is gone.
Out of scope, named: `scripts/otr_ingest_pd_voices.py:101`,
`scripts/otr_macbeth_probe.py:603/620/1196` (scripts cannot import the owner
without the package; convert when a script is next touched).

**C. VERIFIED, LATENT -- the models root is decided twice.**
`_otr_gguf_backend.py::_models_root()` (:891) existence-gates the legacy
`C:\ComfyUI-Models` before falling back to `folder_paths.models_dir`.
`_otr_video_engines/wan_shared.py::configured_models_root()` (:35) returns that
literal UNCONDITIONALLY with no existence check and no `folder_paths` fallback --
while its docstring claims "folder_paths is the authority and is probed first",
which is true of its callers and false of the function. Harmless today because
every caller probes `folder_paths` itself first, but the docstring invites
standalone use, and a direct caller would break on exactly the fresh-install box
this repo's notes keep warning about.

**RESOLVED 2026-09-04 -- delegated, not reworded.** `configured_models_root()`
now returns `str(_models_root())` through a LAZY import (the module's V-12
cold-import statement holds, and `tests/test_models_root_single_spelling.py`
pins both the equality under every env state and the absence of a top-level
import). The "ONE spelling" sentence is true instead of admitting it was false.
The third convention, `_otr_paths.comfy_models_dir()` / `OTR_MODELS_DIR`, stays
parked (cursor r3: do not open a third env in this diff). A FOURTH spelling
belongs to the same parked item (agy r4): `_otr_image_engines/flux2_klein.py:209-215`
probes `folder_paths` first, then the two env vars, and never the legacy tree --
the inverse of `_models_root`'s order. When the merge happens it has four
owners to retire, not three.

**D. VERIFIED, RISKY -- the widget-drift HARD GATE has a silent escape hatch.**
`_otr_workflow_validator.py:172` -- `try: expected = _expected_slot_count(...)
except Exception: continue`. This backs the gate at :497 that exists to catch the
BUG-210/253/281 silent-widget-shift class. Any node whose `INPUT_TYPES()` raises is
silently exempted with no log, and the gate can report `widget_vector_drift=0`
having never checked it. Owed: log the skip at minimum, and decide whether an
un-introspectable node should fail the gate rather than pass it.

**RESOLVED 2026-09-04 (`ad89a45b`).** A raising `INPUT_TYPES()` is a named drift
FINDING (node id, type, exception class, repr'd message), so the hard gate
fails; every class reaching that line is ours (third-party types are skipped
above it), so there was no foreign-node cost to weigh. Precise blast radius,
found while writing the public-path test: `validate_workflow_contract` already
raised on a raising `INPUT_TYPES()` for `OTR_`-prefixed types BEFORE the drift
gate, so the node's own `validate()` was fail-closed all along -- the hatch
mattered for the STANDALONE callers of `widget_vector_drift` (the canonical-graph
integrity test could report zero drift on a class it never checked) and for any
mapped class without the prefix. The `validate_anyway` tooltip no longer claims
the switch "never blocks the episode". 7 tests in
`tests/test_widget_drift_gate_no_silent_exemption.py`.

**E. VERIFIED, DEAD -- 12 symbols with exactly one repo-wide reference (their own
definition), confirmed by `git grep -n -w -F`:** `MAX_HEADLINE_CLEAN_CHARS`
(`_otr_scifi_p0_contract.py:23`), `PLATE_DIRNAME`
(`eng_ghost_signal_stillin_lab.py:72`), `RADIO_HOST_FACE_NEG`
(`otr_meta_brief_image_prompt.py:322`), `CRT_RED`/`CRT_WHITE`/`CRT_MAGENTA`
(`video_engine.py:68-71`), `HARNESS_VERSION`/`MACHINE_CEILING_MB`/`NVML_FLOOR_MB`
(`scripts/_otr_b_spikes/_b_harness.py`), `ffprobe_timebase`
(`scripts/audit_otr_full_run.py:101`), `corpus_ledgers`
(`scripts/otr_clean_stage_lab.py:529`), `_negative_phrases`
(`scripts/otr_style_traceroute.py:213` -- and its logic is duplicated inline in the
function directly below it, which is the slop shape exactly).
`MAX_HEADLINE_CLEAN_CHARS` was queried as a possible UNWIRED cap rather than dead
code; the file answers it at :50 -- "headline/summary are short and bounded by the
fetcher" -- so the bound exists upstream and the constant is vestigial. Clean delete.

**F. VERIFIED, SMALL -- vestigial rows.** `_otr_structured_call.py:110` calls
`REPAIR_TEMPERATURE` the shared public export and `_REPAIR_TEMPERATURE` a compat
alias; it is backwards (the underscore name is the one every call site and test
uses, the public one has zero consumers). `otr_shot_lock.py:3396` re-imports
`_stamp_durable` into a scope that already bound it at :3207. `scripts/otr_asset_index.py:264`
has an `elif engine.startswith("humo")` that cannot fire (only `eng_humo.py` exists,
so the preceding `== "humo"` always wins). `scripts/otr_machine_matrix.py:124`
`load_classes(_profiles=None)` never reads the parameter and both call sites pass
nothing. `_otr_paths.py:449` `episodes_for_obs_dir()` claims to be kept "for
back-compat with existing imports" and has zero callers.

**G. FOLLOW-UPS THE 2026-09-04 ARC SURFACED AND DID NOT BUILD -- each with the
reason it waits.**
* `otr_post_upscale_procgen_blend.py` keeps a `sys.path` insert of `nodes/` for
  ComfyUI's flat loading. Both tool owners it uses (`_otr_shared.ffmpeg`,
  `_otr_shared.ffprobe`) now import package-relative first with the flat
  fallback, and its unused flat `_otr_paths` import is gone -- the flat-first
  spelling had made a SECOND module instance of the owner that no fixture could
  reach (it kept its own Windows-dir tuple and answered a test the fixture
  thought it had muted). Only the insert itself remains; it is what the flat
  fallbacks need.
* `tests/test_openrouter_slug_curation.py::test_routers_appear_in_both_slot_dropdowns_and_auto_leads`
  passes on this box only because of an UNTRACKED catalog cache ("no OpenRouter
  models cached -- run refresh_catalog_cache" in a clean worktree). A machine-local
  lie; the test should build the cache it needs or skip by name.
* The google omni / veo / image adapter tests resolve `comfy_output_dir()` without
  pinning `OTR_OUTPUT_DIR`, so on the real tree they write provider bytes under the
  live `output/otr/episodes/_shared/tmp/`; in a worktree the Tier-3 walk-up lands
  on `C:\Users\output` and they fail. Pin in their fixtures.
* **UNFOUND EXPORTER (2026-09-04):** inside a pytest session `OTR_OUTPUT_DIR` is
  set to `C:\Users\jeffr\Documents\ComfyUI\output` -- exactly the value the package
  `__init__.py:97-108` pin would compute -- yet it is NOT set by `tests/conftest.py`
  (import, `pytest_configure`, `pytest_sessionstart`), `tests/__init__.py`,
  `pyproject.toml`, any registered plugin (anyio, hydra, langsmith, pytest_asyncio)
  or by importing the mux in plain Python. Consequence: a test that injects an
  output root ONLY through a `folder_paths` stub is overridden once the reader it
  drives delegates to the owner (three `test_video_render_path_cw4` tests broke this
  way and now pin the env too). Find the exporter; then decide whether conftest
  should strip `OTR_OUTPUT_DIR` at import the way it now strips `OTR_OBS_DIR`.
* `otr_credits_roll.py:149` reads `.git/HEAD` for the production ledger's commit
  rev; a git WORKTREE has a `.git` pointer FILE, so 44 credits tests fail there.
  Harmless in production; the reader should follow a gitdir pointer.
* A RECIPE identity for remote video lanes (see R-B above): logged, not built.
* Scripts' own ffmpeg readers (`otr_ingest_pd_voices.py:101`,
  `otr_macbeth_probe.py:603/620/1196`): convert when a script is next touched.
* `.kibitz/comfyui.local.md:26` records 23 nodes / 60 links / 132 widget slots;
  the canonical graph is 23 / 61 / 133. Regenerate with `--force` before the next
  arc.
* The `__init__.py:97-108` output pin: preserved deliberately (it is what makes
  every helper agree inside ComfyUI); removing it changes where Desktop installs
  render and is its own design item.
* **The draft 8 GB profiles cannot write at all (PBUG-20260904-05):**
  `8gb_lite`, `otr_4060_floor` and `otr_4060_viz_12b` (all `status: draft`) set
  `gguf_n_ctx: 2048`, and the writer prompt alone is 2,741 tokens on
  `science_news` and 3,338 on `original` -- the budget refuses loud before
  writing a word, on any bank. The row that SHIPS, `otr_4060_12b_gguf_offload`,
  runs the same 12B at `gguf_n_ctx: 4096` (measured 7.8 of 8.2 GB) and fits.
  Design call, owed a kibitz arc: retire or re-context the drafts, pair a
  smaller pinned writer (Qwen3-4B / gemma E4B are on disk), or make the plan
  stage refuse a profile whose context cannot hold its own prompt -- in the
  profile/variant layer, never a silent truncation. The 4060 owns the 8 GB rows.

**FIXED IN THIS SESSION -- the audit's best catch was the driver's own slop from
the night before.** Four rows, all introduced 2026-09-03 and corrected 09-04:
`_obs_basename`'s broad except returned silently while all eight other fallbacks in
that file log -- the exact shape that had already hidden a missing `re` import for
one dev cycle; `_resolve_writer_llm`'s docstring justified `max_new_tokens` by "the
one caller whose reply length scales with a batch" after the same commit taught that
caller to bypass the wrapper; the ideogram ladder comment claimed "all four slots in
three precisions" when the real shape is 3/3/2/1, which also hides a genuine
mixed-precision requirement (no int8 text encoder exists); and four `_DEFAULT_*`
aliases were added with a comment justifying them as the error message's lead name,
which the literal refusal text never reads.

**REJECTED, with reasons -- do not re-raise these.**
* `_DEFAULT_CLIP` / `_DEFAULT_VAE` "have 7 references": a DRIVER false positive.
  `git grep -w` matched `flux2_klein.py` and `lumina_image.py`, which define their
  own same-named constants. Within ideogram4_local all four were definition-only.
* `MAX_PROVENANCE_NOTE_CHARS`: looked dead in-file, has 2 repo-wide refs. ALIVE.
* The ~90 `OTR_*` env knobs: a deliberate, pervasive escape-hatch pattern, every
  sampled one inside live code. Not slop.
* The standalone HuMo render chain (`render_episode_concat.py` and peers, ~5,000
  lines with tests): zero live callers, but `docs/LEAN_MEAN_CLEANUP.md` sec 2.4
  explicitly protects active render tools pending a per-file re-ground, and the old
  bulk kill list is marked SUPERSEDED -- DO NOT EXECUTE. A deferred decision, not
  an oversight. Belongs to whoever runs that pass.
* Long historical comments explaining past bugs: deliberate project practice.
* Duplicated engine PROMPT COMPOSERS: ruled 2026-08-23, lanes stay independently
  re-wordable. Only duplicated FACTS and DECISIONS count.


### 1.5 THE 8 GB SHIP SET -- promote what the clean room proved (queue item 4)

Measured 2026-09-02 on the physical RTX 4060 under plain stock launch flags (receipts:
`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`, PBUG-20260902-01 / -02, Bible 12.145 /
12.146): Klein 4B Q4 GGUF stills at ~21 s a still after three residency fixes (`9b90189a`,
`ad6a635f`, `da2b7a36`, each measured on the 5080 first and byte-identical there); LTX 2.5
(Q3_K_M DiT, 12B encoder pinned to CPU) at ~14 min a clip, so a ~5-hour episode. It WORKS
on 8 GB; it is not a daily driver at that pace. Klein is already the image default in all
19 low-VRAM profiles and the 8gb / 12gb / amd classes (ruling 2026-09-01, `c0ebe31f`;
`docs/OTR_STANDING_RULINGS.md` "IMAGE ENGINE DEFAULTS BY MACHINE CLASS").

Owed, in order:
1. **The PROVEN flip.** The Leg C5 episode is in the clean room's obs
   (`signal_lost_rationed_breath_20260902_060027`, published 2026-09-02 12:10, 24 LTX 2.5
   clips, 12 Klein stills, 6 h 35 min). When the operator has watched it: add the receipt to `config/machine_classes.json` (`proven[]`
   on the 8gb class, image column included; `known_limits` keeps the pace), regenerate
   `docs/MACHINE_MATRIX.md` and the README block (`scripts/otr_machine_matrix.py`), and
   the 8gb `proof_summary` stops saying "image lane unexercised". Record only what
   published.
2. **Ship the profile.** Promote the 4060's untracked `otr_cleanroom_8gb_klein_ltx25`
   profile as `otr_8gb_klein_ltx25` (writer `google/gemma-4-E2B-it`, image `flux2_klein`,
   video `ltx25_high_video`, bark voices until Section 1.1 lands so it is 3.13-safe),
   build its variant (`scripts/build_variants.py --all` then `--check`), add the matrix
   row, and point README's "8 GB card, Klein stills and LTX 2.5 video" row at it instead
   of "not a shipped graph yet". `status: draft` until item 1 flips.
3. **Eight profiles still pair an 8 GB ceiling with the 12B writer and die in the writer
   preflight** (`Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)` under a 6.8 GB
   ceiling; the smallest prompt needs 2064 input tokens and P0 reserves 2800 output, so
   4096 ctx is ~9.5 GB -- ctx is the symptom, the writer MODEL is the cause). Draft set:
   `otr_8gb_ltx`, `otr_8gb_wan`, `otr_8gb_fastwan`, `8gb_lite`, `cpu_floor`,
   `otr_amd8_rocm`; SHIPPING pair: `otr_g4_ltx_8gb`, `otr_w45_ltx_8gb` (14.5 GB ceiling
   next to the only genuinely 8 GB video engine). Repoint all eight to the E2B writer the
   8gb class already ships, or delete them. Not a one-line edit: the GGUF registry has only
   the 12B and Qwen3-8B rows and `gemma-2-2b-it` is a TRANSFORMERS row (agy proposed it
   and was wrong). "Finish retiring the profiles" is no longer an option: `95feac86` keeps
   `config/profiles` as lab presets and makes the matrix the record.
4. **The cfg 1.0 promotion A/B** on three real episode prompts (announcer portrait,
   character portrait, scene beat), the same four cells (nvfp4 / bf16 x cfg 1.0 / 2.0),
   rendered to a NEW dir under `docs/ship-audit-2026-09-01/image-jury/`, then the
   operator's eyeball. One seed was a strong lead, not a proof. Section 2, Batch R5.

### 1.6 THE 4060 TEMPLATE TEST SET -- what is still open before the capstone (queue item 5)

* **One JSON for now (operator 2026-09-02).** `workflows/otr_4060_floor.json` is
  removed from the gallery (it shipped 2026-08-29 to 2026-09-02 as a bark-voiced
  zero-download floor, before kokoro ran on 3.13); `config/profiles/otr_4060_floor.json`
  and its generated variant stay as lab presets. The 4060 dropdown-friendly JSON is
  saved AFTER the testing below, with kokoro on both voice slots (never bark: bark
  renders a long announcer line as a tone, PBUG-20260902-03).
* **The test itself:** a clean portable (Python 3.13) on the 4060 -> Manager install of
  the Active registry version -> the 8 GB saved-dropdown variant (`workflows/variants/
  otr_nvidia_8gb_haunted.json`, kokoro on both slots, or the Klein + LTX 2.5 profile from
  Section 1.5 once it ships) -> run -> `obs_publish OK`, zero hand steps. Any hand step
  is a bug: file it in `docs/PROD_BUG_LOG.md`, fix it at the root, retry. A pass is what
  earns the saved 4060 JSON. Section 2, Batch R7.
* **README model table** from the compatibility workbook's Baseline Combos tab
  (`outputs/20260828-ungated-models/`, the LIVING fact sheet: edit cells in place, never
  add a changelog tab) -- or the operator rules that README's injected class table and
  per-lane facts (66da15da) already cover it and the bullet goes.
* The knob census (Section 1 header) informs what the template pins.

### 1.7 LOCAL-LLM SWEEP LEG 0 -- the in-process preflight (queue item 6)

Leg 0 = one in-process command (`request_slot` -> ~40-token generate -> `_self_unload`
per row, with `reset_peak_memory_stats()` around each), ~15-20 min, IDLE GPU, no ComfyUI.
It fails loudly on a dead row. The four canonical legs that are the real proof, and the
whole sweep design, are Section 2 Batch R3.

### 1.8 DOCS (queue item 7) -- a DELETION pass, not a review (operator ruling 2026-09-01)

Operator: most of `docs/` is stale and should be deleted unless it carries a useful
recipe about a video model. The ownership map that survives the pass: README is the
front door; `docs/MACHINE_MATRIX.md` chooses the profile; `scripts/` own automation;
`docs/RUNPOD_INSTALL.md` is the sole RunPod manual and recovery guide (Codex owns it);
bugs and history live only in `docs/PROD_BUG_LOG.md`, the archive and
`docs/OTR_STANDING_RULINGS.md` (never in this plan). Working rule for the pass: a dated spec, plan, handoff,
kickoff, brief or log under `docs/` goes unless it is cited by a test, by the Bible
coverage index, or by a shipping doc, or it records a video-model recipe (measured
settings, VRAM, canvas, frame counts) that no profile or engine adapter carries yet;
those recipes move INTO the adapter comment or `docs/MACHINE_MATRIX.md` first. No new
guide gets written. One commit per batch, deletions only, with the list in the message.
Runs AFTER this plan rewrite has landed, so no pointer the plan keeps is orphaned.

**DONE 2026-09-02 (45895801, 2bf15784, 607a5ee7; 644 -> 237 tracked docs).** How it
was run: a scripted citation scan over tests, the Bible coverage index, `nodes/`,
`scripts/`, README, CLAUDE.md, this plan, the rulings and the ship audit produced 378
uncited dated docs plus 98 uncited named ones; a 13-agent read of every dated candidate
returned 373 DELETE / 5 KEEP; the driver then grounded the list against the real tree.
Citations from the append-only history logs (`HANDOFF_LOG`, `PROD_BUG_LOG`,
`GO_FORWARD_ARCHIVE`) did not keep a doc -- git history holds all of them. Kept on
purpose: the five recipe carriers (`2026-07-31-PROBLEM-STATEMENT-under-8gb-still-to-video`
bench table, `2026-08-02-MEASUREMENT-M2-humo-vram-ladder`,
`2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames`, `2026-08-16-video-sprint-PLAN` LTX 2.3
distilled recipe, `2026-08-01-fastwan-8gb-MODEL-MANIFEST` with the LoRA sha256 and the
licence gap -- these still owe a move into the adapter comments or the matrix); the
08-26 foley-bed operator rulings; every doc a test, the code, this plan, the ship audit,
`WRITER_INPUT_MATRIX` or `LANE_BUILD_LESSONS` cites (the 09-02 kokoro-onnx and
encoder-eviction receipts among them); `COMFY_TEMPLATE_DIFF_PROTOCOL`, `WIDGET_OWNERSHIP_LEGEND`, `SPEC_haunted_image_to_video`
and its design review, `DEAD_CODE_EXECUTION_PLAN`, the licence attestations, the lane
receipts, the ship-audit folder, the schema examples and the skills backup. Still
open from this pass: `workflows/otr_story_only.json` (operator to say whether the
one-JSON ruling removes it too) and the ~170 dated docs that survived only because a
history log cites them (a second pass, same rule, if he wants `docs/` smaller still).

### 1.9 ONE MANIFEST, PREFLIGHT AUTO-DOWNLOAD (queue item 8; design row; operator asked "and auto download for all?" 2026-09-01 -- confirm to schedule)

Today only the writer LLMs, bark, musicgen and the kokoro voices fetch themselves; every
image engine, every video engine, Stable Audio 3, the cloning TTS engines and the reference
WAVs are manual placement behind two fetchers under `scripts/` that the registry bundle does
not ship, and README's "auto-fetch" wording for the 8 GB lane is what the drill measured as
manual. Keep the rule that nothing downloads DURING a render; move the fetch to the
queue-time preflight that already refuses with "PREFLIGHT FAIL: the running server cannot
see: <files>":
1. One manifest, `config/model_manifest.json` (repo, revision, path, destination, bytes,
   sha256, gated), merging the provisioner's pinned tiers and the fetcher's lanes (12 rows
   each today), read by the preflight, the pod provisioner and the matrix generator.
2. Preflight resolves the selected dropdowns to artifacts, prints the total, refuses early
   if disk is short, downloads into the running ComfyUI's models tree through `folder_paths`
   (never `C:\ComfyUI-Models`) with `.part` files, hash verification and resume; the fetcher
   already has that code and moves under `nodes/` so it ships.
3. Gated rows (LTX 2.5) refuse BEFORE downloading, naming the terms URL and the token step.
4. A `download_policy` widget: auto (default), ask (list sizes only), never (air-gapped).
The manifest carries the kokoro-onnx weights, so this lands after Section 1.1. Design item:
kibitz arc before code. (The first-run download bill is README's number, not this file's.)

### 1.10 SHIP-AUDIT SURVIVORS (2026-09-01) -- each needs a design decision, not a grep (queue item 8)

Receipts and every file:line: `docs/ship-audit-2026-09-01/SHIP_LIST.md` (71 confirmed,
51 disputed for the operator to rule, section 8). The mechanical items are landed; the voice item rides Section 1.1, the 8 GB writer item
Section 1.5, Mac / AMD Section 1.13. These are not mechanical and each wants a kibitz arc
before code:
1. **Runtime writes inside the pack directory** (cloud-media billing ledger,
   OpenRouter catalog cache): a registry update wipes them. Route through
   `nodes/_otr_paths.otr_shared_cache_dir()`; needs a migration note for existing
   ledgers.
   Bites exactly the registry-install users the queue's first items create.
2. **`_fit_reason` never consults `needs_fp8_te` / `needs_fp4_te`**
   (`nodes/_otr_shared/capability_profiles.py`), so fp8 and NVFP4 engines qualify on the
   ROCm tiers whose `dtype_policy` forbids them. Two clauses keyed on `dtype_policy`.
3. **The janitor cannot sweep `tmp/audio_slices`** (`nodes/_otr_janitor.py`: directory
   granularity, newest-child mtime): 9.3 GB measured, and the boot sweep stats 21,440
   files (6.7 s) every ComfyUI start. Three lines, but it widens what gets auto-deleted,
   so it lands with a test.
4. **Cloud spend with no ceiling**: `cpu_floor` and `otr_mac_mps` route every image role
   to the paid Google API on the mere presence of a key, and the BYO-key lane has no
   reserve/bill/ledger path (`eng_google_image.py`). Ledger it or make it an explicit
   opt-in.
5. **`eng_ltx_video` / `eng_ltx_av` reload ~14 GiB of weights per beat**; the
   `prepare()` + `external_results` pattern the sibling lanes use is the fix.

### 1.11 THE ADAPTATION DESIGN (queue item 8; hardened, NOT yet built; multi-session -- start only with room to finish step 1)

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md` (5080-local).
Keystone: compile source speech from an authenticated segmented artifact, never generate
it; "summarize into X words" means SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not
paraphrasing (which also removes the VRAM hazard: no model sits in the source-speech path).
Ceiling by arithmetic: an episode cannot exceed **1,520 words** (19 voiced beats at
act_count 7, `BEAT_WORD_HARD_MAX` 80), so full-scene performance needs a beat-topology
redesign; build target is the 300-word unit. Scope per the 2026-08-23 ruling: the verbatim
lane is Shakespeare only; public_domain may paraphrase.

**NEXT, IN ORDER:**

1. **The segmented source artifact** (schema, spans, hashes,
   `body[start:end] == segment.text`, omission receipts) and the pass-to-field
   ownership table -- **nothing else codes until that table exists.**
2. **Cast from the selected cut.** Real scenes carry 3-12 speakers against a
   6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so which
   speakers appear must follow from the cut that fits the word budget. Coupled hard
   to the capacity guard: at act_count 1 there are exactly THREE voiced beats, so a
   4-person cast is a mathematically guaranteed `CastVoiceCoverageError` -- the
   failure that killed `scifi_news` in the six-bank run. `compute_episode_budget`
   must also receive the TRUE locked cast.
3. **Loosen the count-match invariant** (`OTR_LedgerScriptWriter.py:4061-4067` hard-
   raises on any locked != requested) and change the pack text that tells the model
   to drop figures.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner -- and
   bind its output to the verified body hash.
5. **Schema migration** to retire `cast_hints`; still required by the validators and
   by `public_domain_manifest_schema.json`, so manifests and tests migrate in the
   same change. (`visual_style_policy`, the other half of this item, was ripped
   2026-08-04.)

**KNOWN AND NOT FIXED:** `canonicalize_shakespeare_text` truncates at 12,000 chars
and the interpreter sees only the first 5,000, so a 3,445-word scene reaches the
brief as ~880 words, silently. Belongs with the artifact work, where each beat is
fed its own segment rather than a blind prefix.

### 1.12 STYLE / IDENTITY DECISION WORK (queue item 8; backlog; not the next coder window)

Grounded by the 2026-08-03 four-agent forensics; every line has a file:line in the
session traces.

1. **"Invent one and tag it"**: add a derived style/genre field to
   `run_story_brief_reflection` (`_otr_story_brief.py:513` -- proven content-loyal on
   both specimens), stamp beside `story_brief`, repoint the treatment `Style:` line
   (`video_engine.py:1762`) and the HUD (`video_engine.py:1336` -> `_build_left`
   `:1592`) at it. Highest-leverage item here: it fixes the credits line for all six
   banks uniformly.
2. **Rename `meta.style` -> `meta.story_scaffold`** (operator: too many metas; the
   field is neither scifi nor a description). Consumers move in ONE atomic change:
   writer stamps, credits `_story_style_receipt`, `visual_plan.style`,
   `video_engine.py:1336`, tests -- AND the ledger validators (r3):
   `_otr_ledger_consistency.py` pins the field in its matrix
   (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`
   reads it too; missing them fails ledger validation on the first episode.
3. **Ghost-name reconciliation fork**: pitch cast never reaches `lock_cast` (names
   are a pure pool draw; `source_character_names` deliberately None for invention
   lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence:
   Evelyn/Leonard as offscreen lore; Fogbound Rails bio still opens "Lizzie Gray".
   (Cross-listed as Section 3, question C.)
4. **Dead fields found**: `ending_template` computed but zero LineRequest call sites
   pass it; `seed_policy.style_seed_env` validated but unconsumed; `dramatic_state`
   derived PRE-dialogue goes stale in the treatment.
5. **`meta` is a 120-key drawer** -- the cleanup the operator keeps asking for. Scope
   as its own rip with the ledger law (every field one owner).

### 1.13 MAC / AMD -- images only, later (queue item 8 tail)

Operator: Mac and AMD ship images only (ruling 2026-09-01), and he is "not hopeful".
Landed: the credits font, the llama-cpp hint and four platform guards. Owed, in order:
(1) one measured Klein render on Apple Silicon -> `nodes/_otr_image_engines/registry.py`
gains `mps` on the `flux2_klein` row (cuda-only today) -> `otr_mac_mps` flips off
`google_image` (README's Mac row says so); (2) the upscale stage accepting `mps`
(`_otr_upscale_engines/__init__.py`, deliberately deferred); (3) a measured ROCm boot for
`otr_amd8_rocm` / `otr_amd16_rocm`. ROCm already qualifies for Klein (presents as cuda).
Needs hardware neither NVIDIA box has.

### 1.14 ANIMATEDIFF V3 + THE LEDGER + STILLS -- the experiment campaign (operator ask 2026-09-02; awaits his pick)

Operator, 2026-09-02: a fresh set of eyes on whether the AnimateDiff v3 (+ adapter) lane takes
advantage of the ledger and the per-beat stills, with the VISUAL STYLE obeyed ("I think we lost
the visual style" -- confirmed: the style reaches that lane as a two-word cue on a fixed base
checkpoint under a photographic-grime adapter, no still) and stills-in among the first one to
three arms. The self-contained statement is
`docs/2026-09-02-animatediff-ledger-experiments/PROBLEM_STATEMENT.md` (11,000 words, grounded by
a five-reader workflow); four fresh reads (Fable 5.1 cold, Codex, Cursor, Antigravity) and the
driver's grounded judgment are in `fresh-eyes/`. The judged order: (0) the instrument -- durable
prompt + seed + request hash on the ledger, a canonical REPLAY mode (today the writer node always
mints a new ledger, so a same-ledger A/A cannot run), a blinded two-null scorecard; (1) the
adapter strength swept PER STYLE including 0.0 (zero code; env override exists); (2) a still-in
LAB PEER engine with the lane's own in-family 512x288 plate (never a flip of the shipping
engine's contract, never a Klein gate on the 4060), one subject-free scene plate per beat with the
plan untouched (the earlier "cycle frozen to figure" was superseded in the item's own arc,
`still-in-peer/driver_anchor.md` section 7 D4); (3) the pack's
language back into the prompt under a 77-token budget; (4) one timeline per shot (a
FrameContract design arc). Deferred: speech-energy scaling, SparseCtrl, per-style checkpoints,
FreeInit. Cut: the style-aware roll, CameraCtrl, IP-Adapter / PIA / Lightning. Nothing in the
shipping recipe moves until the operator picks the first arm; the still-in idea stays parked
(Section 4) until then.

**Status 2026-09-02 (evening).** Operator: "I leave it up to you to synthesize and code maybe
1-3 of the best options." Item 0, the instrument, is CODED after a full four-round arc
(`instrument/driver_anchor.md`, sections 1-13): every rendered clip carries a versioned ACTUAL
receipt hashed into `actual_request_sha` and stamped durably as `meta.render_trace`; the writer's
trailing `replay_from` widget (and `otr_canonical_api_run.py --replay-from`) re-renders a frozen
bundle (`scripts/otr_freeze_replay_bundle.py`) through the whole canonical graph with no writer,
TTS, music or stills, node 7 byte-copying the SHA-verified master; `scripts/otr_verify_replay.py`
is the offline A/A verifier. Item 1, the adapter sweep, is RUNNING on the 5080 overnight profile
(two styles x 1.0 / 0.5 / 0.25 / 0.0, titles "Adapter <k> <style>", published to `otr/obs/` for
the operator's eye). Next: the live replay proof (render, freeze, replay twice, verify) once the
sweep releases the GPU, then item 2, the still-in lab peer -- registered only after the
`docs/VIDEO_LANE_PREFLIGHT.md` gates 1-8 and `tests/test_lane_preflight_matrix.py`, as the
operator's standing rule for any new video pack requires.

**Title-card leak, 2026-09-02 17:45 (operator: "your attempt at a video plan bled into my
title").** The first three sweep legs were launched with `--title "Adapter <k> anime"` so the
leg could be told apart in `otr/obs/`; that flag rides the writer's `episode_title` widget and
so became the on-screen TITLE CARD of three published episodes (the known harness-label path,
CLAUDE.md "fix the title at the source"). Standing rule from the operator, same minute: every
canonical leg runs the CLEAN runner, title generated from the story (public_domain: the
source's own title). Legs 4-8 run without `--title`; a leg is identified by its log, its
publish timestamp and, now that item 0 is merged, durably by
`meta.render_trace[*].sampler_inputs.adapter_strength`. The three already-published episodes
stay in `otr/obs/` (nothing is ever tidied out of it). SOURCE FIX, still open: give the harness a
`run_label` of its own that reaches the log and the ledger meta but never `episode_title` -- a
small design item (touches the API runner, the whitelist and the mux's filename), arc before code.

**Every style, not anime (operator, 2026-09-02 evening): "anime is not the only target; all
visual styles need to craft the episode as well when selected."** The registry carries nine
(`nodes/_otr_visual_styles.list_style_ids()`: anime, archival_documentary, cartoon,
paper_origami, recur_frac, sci_fi_radio, shakespeare_stage_realism, storybook_engraving,
video_art). The two-word-cue defect is style-blind, so every arm of this campaign -- the
adapter sweep (already anime + storybook_engraving), the still-in lab peer, the pack language
under the 77-token budget -- is designed and proven PER STYLE, with a non-anime style in every
proof leg, and judged by one question: is the SELECTED style visibly the episode's style on the
stills and the video alike.

**First read of the sweep (operator, 2026-09-02 evening, on the three anime legs 1.0 / 0.5 /
0.25):** "the anime looks improved; which one is better, not sure at the moment." He reviews
the full candidate set when he is back; the driver's standing instruction meanwhile is "keep
coding, do your best" and to decide whether a blinded A/B is needed. The instrument's
render-trace rows on legs 4-8 (adapter_strength per shot) are what makes any later A/B
attributable; the first three legs pre-date the merge and carry no trace.

**Item 1 CLOSED by the operator's eye (2026-09-02, ~20:30), six of eight legs watched.**
Storybook engraving: 0.5 (`the_last_reading_190630`) "very storybook and did the best of
maintaining the style"; 1.0 (`the_trembling_ground_182739`) "cool rendered 3D realistic
storybook stuff but drifted into the army men in green coats". Anime: 1.0 "more anime than
0.0 but drifted into real-life stuff for a while"; 0.5 "some anime but some people in blue
coats"; 0.25 "maybe the most anime"; 0.0 "definitely animation, not strikingly anime, but
cool". His delegation: "if you think any of them are an improvement to the haunted we have,
take your best take." TAKEN: **`ADAPTER_V3_STRENGTH` 1.0 -> 0.5** (the shipped default on
both boxes; the env override stays the sweep knob; every receipt stamps what ran). 0.25 is
the standing contender on anime; if storybook 0.25 (leg 7) also beats 0.5 by his eye the
default moves to 0.25 -- one constant. **The coats finding:** across the traced episodes the
only prompts that say "coat" say "cream coat" and "charcoal coat" (the cast's own looks, both
characters), never green or blue; the green army coats and blue uniforms are the base model's
prior for "coat" under an engraving or anime medium, amplified by the adapter. The GARMENT
WORD is a style lever -- an input to item 3 (pack language): the motif composer should map
garments through the pack rather than pass the writer's word raw, and this is the first
measured prompt-to-picture pair the instrument's trace made citable.

**Item 2 CODED (2026-09-02, later evening), after a full four-round arc (Fable cold +
Antigravity r1, Codex r2, Cursor r3, Sonnet r4 -- converged; Sonnet QA on the finished diff).**
`animatediff15_v3_stillin_lab_video`: the haunted lane started from an in-family 512x288
plate minted in-graph from the pack's full language and the ledger's world, repeated in Python
to the sampler batch, sampled at a strict `OTR_STILLIN_LAB_DENOISE` (default 0.65); a lab id
on `otr_stillin_lab_5080` (`draft`, never a default); the replay instrument gained a derived
bundle (`--derive-engine`) so ONE ledger renders on both Ghost siblings; the verifier gained
the plate-hash rule; `scripts/otr_stillin_probe_report.py` measures motion energy against the
lane's own A/A band. The design and receipts are `still-in-peer/driver_anchor.md` sections
1-14. NEXT: merge, the full suite in the main checkout, then the live probe on the 5080 --
one style per invocation, anime and storybook_engraving first, every leg to `otr/obs/` with
the story's own title -- and the operator's eye on the triptychs. Stop rules stand (section 7
D9): plates that do not read as the style end E1 (next arm E11); no denoise that both moves
and keeps the plate ends E1 (next arm the E2 probe).

**Item 3 REDEFINED and IN ARC -- Prompt v3, "draw the crux" (operator, 2026-09-02 21:00-22:00).**
He read every beat of the storybook episode beside the prompt it got, rewrote ten by hand,
and ruled: draw what the line is ABOUT in the story's one setting, mix the visual style with
the CRUX of the story on every beat, the world's own motion, no costume or prop the dialogue
never names, fewer variables ("too many variables -> humans and bags"), get creative, and
"yes, I want to see more STORY, not just the characters." Mechanically confirmed: the lane's
character motif is a costume template with fallback pools (coat; lantern / key / ledger /
satchel), the object beats draw the carried prop, the leaf author is handed no dialogue and
its one hint field was empty on all 27 beats, and the story brief plus `key_objects` sat unused
by this lane. Source: `prompt-rule/operator_rewrites.md`; anchor: `prompt-rule/driver_anchor.md`
(sections 1-7; r1 = Fable cold + Antigravity, five reversals, R1-R10 govern; r2 Codex running).
**GPU FREEZE (operator, 21:50): "don't waste any GPU runs until the prompting is fixed."** The
adapter sweep's last leg finished on its own (8/8 published); the overnight chain was NOT
launched; nothing boots a server until v3 is merged and green. The first GPU run after is ONE
episode, same seeds, v2 beside v3 (R9). The still-in peer (item 2) and the 0.5 adapter default
(item 1) merged at 7c231e24 with the full suite queued on CPU; their probes wait behind v3.
Item 3b (the other video lanes get the same SHAPE, per-lane budgets, talking-face lanes
exempt) follows once v3 is judged on this lane.

**Operator, 2026-09-02 20:45: "of course I'm really more interested in video art or
recursive fractal, or both if they are separate."** They are separate packs (`video_art`,
`recur_frac`), and they are the two LONGEST packs in the registry -- the ones the shipping
lane's two-word cue serves worst and the ones the peer's full-language plate is built for
(recur_frac's plate reduces to its positive_tail under the 320-char ceiling; the receipt shows
what survived). Tonight's GPU chain leads with them on BOTH lanes: after the instrument's
replay proof, four adapter legs (video_art and recur_frac at the shipped 0.5 and the
contender 0.25), then the still-in probe on video_art, then on recur_frac. Anime and
storybook_engraving probes follow on a later night. His eye on those two styles is the
verdict that matters most to him; the per-style rule stands for the rest.

---

## SECTION 2 -- RENDER WORK, BATCHED BY THE LEG THAT PROVES IT (test the least)

A leg is 1-3 hours on the one GPU. Run batches SERIALLY -- *"two windows
resetting one GPU is how each kills the other's leg"* (from the archived
scheduling note; still true). Reset per `CLAUDE.md` section 4 before every
leg. A leg that does not reach `otr/obs/` did not pass. **Every canonical leg
below is also a free chance to** (a) re-observe the two parked eyeball items
(Batch R5) and (b) watch the two ledger-cleanup behaviour changes named in
Open risks that have no live receipt yet.

### Batch R1 -- THE H3 PROMPT-POLICY VERDICT: read the receipts first, render only if they do not answer

The fix (`e923a9f3`, 2026-08-27) already has two post-fix `minimax_h3_video` episodes in
the matrix's PROVEN row: `signal_lost_the_poise_of_stone_20260827_143538` and
`signal_lost_reel_of_resistance_20260828_121427` (the operator watched the latter). Owed
is the VERDICT, still marked "STILL OWED" under PBUG-20260827-01 in `docs/PROD_BUG_LOG.md`:
the positive video prompt shows nonverbal action and camera PRESENT, and the beat's exact
dialogue and any speaking / lip-sync / mouth anchor ABSENT. The ledger on disk stores only
`prompt_sha8` for the positives, so read the render receipt or the server log. Clean
receipts close the row. Render a fresh leg ONLY if neither episode's receipt answers it:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_w45_minimax_h3_video -Acts 1
```

Engine routing is a PROFILE, never `-Set` (`patch_creative` whitelists creative widgets
only; writers may ride as `-Set`); the profile carries the h3 boot contract
(`--reserve-vram 12`, `--disable-pinned-memory`); a FRESH episode id is mandatory because
`request_hash` excludes prompt bytes, so a cached SPEAKING clip would be a false pass. The
BEFORE sample `signal_lost_the_caretakers_clause_20260826_155835` (every beat on H3,
pre-fix, under `output/otr/episodes/`) stays untouched: it is half of the A/B.

### Batch R2 -- ONE image-mode sweep on `otr_soak_llmsweep_02` settles SCENE + PORTRAIT `elements: []`

**SO THE NEXT STEP IS A MEASUREMENT, NOT A PANEL. It is a RENDER item.**
Re-run the image sweep's profile `otr_soak_llmsweep_02` against post-`ae7e7b6a`
HEAD -- `scripts/otr_bank_engine_sweep.py`, image mode, which walks every bank
against both engine profiles -- and read whether SCENE and PORTRAIT beats refuse
at all now.
* **If they refuse:** the fork below is real and gets the full arc, now with
  numbers instead of an inference.
* **If they do not:** this row collapses to a documentation correction and costs
  nothing further. That is the likelier outcome and it is why no arc runs first.

If SCENE / PORTRAIT beats still refuse, the fork is real and gets the full arc, now with
numbers: **(a)** derive a subject noun from the prose -- already written off in
`ideogram4_local.py` (`_wrapped_caption`: the five-layer string is a convention, not a
grammar), and a wrong noun INVENTS CONTENT, which the source-fidelity rule forbids; or
**(b)** a metadata channel that hands the lens a real subject -- more wiring, but the
anchor is derived rather than guessed. Full argument:
`docs/2026-08-26-ideogram-music-card-PROBLEM-STATEMENT.md` (5080-local) and
PBUG-20260826-01.

### Batch R3 -- FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator directive 2026-08-25)

After Leg 0 (Section 1.7), four canonical legs prove all 7 surviving local LLM rows in
BOTH slots plus the gemma Q8_0 negative probe -- and each leg doubles as a ledger-cleanup
live-watch and an eyeball re-observation chance. Charter: `docs/OTR_STANDING_RULINGS.md`
"ONLY EASY-TO-LOAD LLMs SHIP" -- every surviving row does creative AND technical, or it
is ripped on a MEASURED failure, never on assumption. All 7 rows are on disk;
`docs/LLM_PREFLIGHT_GUIDE.md` is the preflight guide.

**THE SWEEP ITSELF is what is open. Design is done (11-agent fan-out, 2026-08-25), build
is not.** The honest shape, which is NOT one render:

* **7 local rows / 2 model slots = 4 canonical legs MINIMUM**, plus a **Leg 0**
  in-process preflight (no ComfyUI; `request_slot` -> ~40-token generate ->
  `_self_unload` per row, with `reset_peak_memory_stats()` around each). Leg 0
  is one command, ~15-20 min, and is what fails loudly on a dead row -- but a
  leg that never reaches `otr/obs/` did not pass, so the 4 canonical legs are
  the real proof.
* **Every leg must PIN `--source-bank` to the scifi lane.** Canonical ships
  `'roll (any eligible bank)'`, and `_otr_scifi_news_pro.py` is the only runner
  code-verified to drive BOTH slots. Unpinned, a leg can land on a lane that
  never touches the technical slot and the sweep proves nothing about that row.
* **`gguf_quant` is ONE per-run widget**, and `unsloth/Qwen3-8B-GGUF` ships
  only `Q4_K_M` -- so any leg carrying it runs Q4_K_M.
* **A KNOWN FALSE-GREEN TO DESIGN AROUND:** `meta.slot_calls_by_slot` is incremented ONLY
  inside `_SlotScheduler._account_and_get_entry`; SIX `request_slot` sites live outside it
  (`story_orchestrator.py`, `otr_shot_lock.py`, `OTR_LedgerFreezeCascade.py`, the
  SlotContract path in `OTR_LedgerScriptWriter.py`, the registered `nodes/vram_context_test.py`).
  The counter proves IN-WRITER generation only; reading it as full-row exercise is a false
  green.
* **The operator's creative/technical parity rule is the sweep's acceptance
  criterion.** Structurally both slots already build from the IDENTICAL
  `dropdown_choices()` list, so no row is slot-restricted; what is unproven is
  whether each row can actually do the TECHNICAL job (constrained JSON / GBNF).
  A row that cannot do both, and was never tested or implemented, is a RIP
  candidate under his rule -- but rip only on a measured failure, never on
  assumption.
* **A negative probe worth running deliberately:** the gemma GGUF row at
  `Q8_0` / `n_ctx=4096` needs ~14.70 GiB FREE against a 15.92 GiB card with
  ComfyUI resident, and `_otr_gguf_backend.py` compares against
  `mem_get_info()` FREE with "NO silent context downgrade". Either outcome is
  informative; record both.

**Full design (coverage matrix, per-row assertions, skip-reporting rules, risks)
is in the 2026-08-25 workflow result; re-derive from this row if it is lost.**

### Batch R4 -- ONE canonical `fastwan_8gb` leg with 60-second opening AND closing cues proves PBUG-20260811-02

The only row of the 2026-08-25 re-triage still open. Root cause established, the repair is
WRITTEN, and it is not a coding item: it needs a canonical `fastwan_8gb` leg whose cues are
long enough to chunk at `_MUSIC_MAX_CHUNK_DUR_S = 22.0`. A render window, not a coder
slot. Detail: `docs/PROD_BUG_LOG.md`; the closed trio is in the archive.

### Batch R5 -- OPPORTUNISTIC: the cfg 1.0 promotion cells, D2 fail-hunting still legs, and the eyeball re-observations

Run when a render window is free and nothing above needs the box.

* **The cfg 1.0 promotion A/B** (Section 1.5 item 4): four cells x three real episode
  prompts to a NEW dir under `docs/ship-audit-2026-09-01/image-jury/`, then the operator's
  eyeball. Confirms cfg 1.0 on real content or sends it back.
* **D2:** reset per CLAUDE.md section 4, boot headless, run 320-word `public_domain` or
  `shakespeare` still legs until one fails (~1 in 6). A publish is a clean leg; a
  fail-closed with the compact JSON `MISSING_TARGET` record in the SERVER log (arm, token,
  index, canonical `prompt_hash`, repr-escaped excerpt; the canonical runner truncates the
  exception at 500 chars) is the PROOF D1 WORKS -- D3 then fixes THAT branch at its root and
  `docs/PROD_BUG_LOG.md` gets a mechanism, not a guess. Do NOT weaken the completion gate,
  revive the portrait-init fallback, or rebuild the withdrawn "give the collapse guard a
  still owner" fix (the 08-04 postmortem disproved that chain). Receipts:
  `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`,
  `docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.
* **Two eyeball items ride ANY real render leg, at zero extra legs:** the announcer
  framing defect (`docs/2026-07-11-announcer-framing-defect.md`: episodes START a story
  instead of admitting you into one) and name-splice defect #2. Both predate THE LAW and
  have no reproduction at HEAD, so no coder time: still there -> re-admit as a FRESH dated
  row with that leg as evidence (the framing fix stays seam + score contract + fail-closed
  validator, never Python authorship); gone -> tombstone.

### Batch R6 -- RETIRE OR RE-DERIVE the seven 45-word engine proofs

Cross-check the seven public engine IDs of the archived 2026-08-13 runway table (row 3;
its pointer was recorded BROKEN on 2026-08-16) against `config/machine_classes.json`
`engine_evidence` (PROVEN rows dated 08-23 .. 09-01). Render a 45-word proof (`COVERS`,
`RESULT SUCCESS`, `obs_publish OK`, the file on disk) ONLY for an engine with no
post-08-13 receipt; otherwise retire the row. Do not spend seven legs on an unverified
list.

### Batch R7 -- THE 4060 CLEAN ROOM: the legs still owed

The clean room stays on disk: `C:\OTR-CleanRoom` on the 4060 (portable v0.34.0, Python
3.13, OTR clone at `da2b7a36`, ComfyUI-GGUF pinned + patched, pinned weights, bark
voices). Server and legs start through Task Scheduler, never from an SSH session. The
clean-room profiles are untracked stand-ins on the 4060; the shipped equivalent is
Section 1.5 item 2. Friction log and every leg receipt (R1, R2, C through C6):
`docs/ship-audit-2026-09-01/4060_CLEANROOM.md`.

* **Leg C5** (Klein + LTX 2.5, stock flags): PUBLISHED 2026-09-02 12:10, 24 clips,
  `RESULT SUCCESS` + `obs_publish OK`; the operator eyeball is Section 1.5 item 1. Leg A's question is
  answered by this leg (LTX 2.5 works on 8 GB, not a daily driver); Leg C6 (fp8 encoder)
  is not needed.
* **Leg B -- OPEN:** `otr_cleanroom_8gb_humo17` (HuMo 1.7B, 13.6 GB of Comfy-Org weights,
  no extra node pack) -- the faster 8 GB video candidate if LTX 2.5's ~14 min a clip is
  too slow to ship as the 8 GB video default. Pull the clone to HEAD first (later commits
  are docs and the provisioner).
* **Z-Image from the dropdown on 8 GB:** one stock-flag `z_image_turbo` still
  post-`da2b7a36`. The R1 abort (`Fatal Python error: Aborted` at sampler step 5/8 under
  DynamicVRAM) was never re-tested after the residency fixes, which the clean-room doc
  calls only its "likely root". If it still aborts: document the
  `--disable-dynamic-vram --lowvram` pair in README's 8 GB row for that dropdown choice
  (README carries no such text today) and file the faulthandler report with ComfyUI. The
  shipped 8 GB set runs Klein by ruling and is unaffected either way.
* **The registry-install template test** (Section 1.6) runs here once a version is
  Active.
* Record ONLY what publishes (`RESULT SUCCESS` + `obs_publish OK` + the file) into
  `config/machine_classes.json`, regenerate `docs/MACHINE_MATRIX.md`. Do not advertise a
  lane the clean room did not finish.

### Batch R8 -- THE NEXT POD SESSION: the acid test and a looped lane sweep on one rental (queue item 9)

The volume stays (it holds the warm cache, the expensive thing to recreate); the pod stays
STOPPED until this batch runs. Codex owns it and `docs/RUNPOD_INSTALL.md`. Sequence on the
one rental: saved template -> `scripts/otr_provision.py` -> one published episode with
ZERO hand steps (the acid test the DOER was built for: Stable Audio 3, index-tts in its own
`uv` venv at `INDEXTTS2_PYTHON = "3.10"`, the reference WAVs -- all installed by the
provisioner, none by hand), THEN the looped lane sweep (`scripts/otr_pod_lane_soak.sh`,
`scripts/otr_pod_overnight_sweep.sh`) for the HuMo 14B / LTX 2.5 / indextts2
second-machine receipts. Receipts file under the 16gb class in `config/machine_classes.json`
`engine_evidence`: there is no 24 GB class or profile by ruling (`95feac86`; Section 3 L
asks whether a row is wanted).

### Deferred render items (each blocked, or waiting on something else first)

- **Capped-14B HuMo leg** -- live proof of the ping-pong lip-sync reversal fix
  (`a1d810f1`); see the coverage cluster row in Section 1.4.
- **`scifi_news` live reverify** (PBUGs 20260712-22/23/24/25, fixed in tree) --
  blocked by the `scifi_news` P0 convergence defect (Section 1.4), then fan-out.
- **The WAN physical 8 GB proof** -- a render on a PHYSICAL 8 GB card is still owed;
  behind Section 3 question D, detail there.

---

## SECTION 3 -- RULINGS OWED BY THE OPERATOR (each with its default if unruled)

One operator pass clears every row here. Nothing in the queue's first five items waits on
this section.

### J. THE REGISTRY -- the control experiment, only if alpha.15 flags

Two versions exist, both Flagged (a private secret scanner, not an exec linter; CLAUDE.md
7A), no Active, no rollback target (the node hard-delete freed alpha.8's string). The
alpha.15 push is queue item 1; the README's token-shaped literal, the only shipped string
matching a published secret rule, is already gone (`64d81ca7`).

* **If alpha.15 flags, run the control:** republish the alpha.8 tree (`e44235f5`)
  byte-identical as alpha.16. Active means the trigger is in the alpha.9+ delta and can be
  bisected; Flagged means the ruleset moved and that result is the evidence to hand
  Comfy-Org. Never version-delete (a soft delete burns the string).
* **Version sequencing:** alpha.15 = the marker patch; alpha.16 = the control, only if
  needed; the kokoro-onnx dependencies (Section 1.1) ride the NEXT bump after both, never
  while a version is Pending.

### The question list

* **(A) Arm `defaults.scene_coherence_check` on any bank?** The G15 vacuity fix
  (`e2807dcc`) is live code with zero callers. Decide whether any bank arms it; if yes,
  measure OFFLINE over the published corpus first, then arm in ONE change (no-render work
  once ruled). Default if unruled: nothing arms it.
* **(B) The `full_text` HTML block-join separator.** Inserting separators is a WIDER
  change to the coordinate system `source_digest` pins -- it belongs in the source
  adapter, and it is the DOMINANT P0 failure cause on live evidence. Detail: the first
  row of Section 1.4's P0 cluster. Default if unruled: not touched.
* **(C) Ghost-name reconciliation fork:** scrub briefs after cast lock, or propagate
  pitch names. Detail: Section 1.12, item 3.
* **(D) After profile retirement, who owns a tier's native render ceiling?** Since
  `95feac86` the machine matrix, not a profile, is the stranger-facing channel; profiles
  are lab presets. Candidates: a `video.max_render_frames` field on the class row, or the
  adapter's own capability row with the widget as an override (0 = adapter contract).
  Blocks the WAN 8 GB proof and the A2 echo fix. Detail block below.
* **(E) The three works that refuse to vendor** (`ghost_ship` gid 11045, `purple_cloud`
  11229, `beleaguered_city` 11521 -- `scripts/otr_vendor_public_domain_library.py` against
  its parser) **need one Gutenberg fetch, so it is operator-opt-in only** -- not
  schedulable inside an offline sprint.
* **(F) The Bible fan-out batch** -- one operator pass clears every row marked "awaiting
  fan-out" in Section 5, the PBUG-20260710-07 retirement ratification, the duplicate-id
  cleanup, and the PBUG-20260901-04 promotion.
* **(G) Name the first H3 video-path sprint** -- standing context below.
* **(H) Keep the research_only behaviour?** Since 08-15 a research_only source WITHHOLDS
  the OBS copy instead of killing the finished render. Say so if the old kill-the-render
  behaviour is wanted back (a one-line revert). Default if unruled: keep.
* **(I) Does `media_archive` want the catalog premise at all**, or the same scaffold-off
  treatment as `original`? Found by the five-bank beat test: a
  `pirate_radio_resistance_drama` premise was drawn over a film-reel standoff seeded by a
  real Library of Congress item on 'Midnight' (1939) -- the operator caught it on screen.
  Second specimen of the content-blind-draw class; the scaffold-off rule so far was stated
  only for `original`.
* **(J) `style_tail_policy` needs a third token, or the `ltx_radio_face` path is EXEMPT.**
  `build_radio_host_prompt`'s `ltx_radio_mouth` branch (`otr_meta_brief_image_prompt.py`)
  returns early with `"%s, warm dramatic lighting"`, skipping `finish_visual_prompt` and
  the `image_grade_tail` append by the 2026-07-02 look direction, while the `ltx_audio_in`
  bookend row declares `style_tail_policy="full"`. Default if unruled: the exemption,
  because it changes no behaviour.
* **(K) `check_compatibility`: ratify the inert constant, or schedule the rip?** The name
  reserves nothing (`tests/test_otr_check_cli.py` activates a bundle whose value is a plain
  integer); rip blast radius ~5 code sites, 2 test files, 3 docs; codex and Fable both said
  RIP, Claude grounded both. Default if unruled: leave it, and add the doctrine line to
  `EXTENDING_OTR.md`: a name published to clients before its consumer exists is "reserved,
  no contract, ignored if defined" and lives in no executable code. Argument in the
  archive.
* **(L) Does a `24gb` class row exist at all?** The matrix says 16 GB+ is the top tier by
  design (`95feac86`) and LTX 2.5 at 1664x960 OOMed on a rented 24 GB 4090
  (`docs/RUNPOD_INSTALL.md`). Default if unruled: no row; rental receipts file under the
  16gb class's `engine_evidence`.

### Standing context for question (G): MiniMax H3 is a sprint series

Operator, 2026-08-09: H3 is "a series of sprints all to refine the video paths"; scope
TBA -- do NOT invent the sprint list. It is not a dropdown-admission question any more,
and nothing here is blocked on the operator; the 4 s floor is an INPUT the video paths
accommodate or route around. When the operator names the first sprint it becomes its own
row in Section 1 or 2. Grounding: `docs/2026-08-03-PROBLEM-STATEMENT-minimax-h3.md`
(untracked, another window's working file -- read it, never stage, edit or delete it) and
`docs/2026-08-06-SPEC-subsystem-matrix-pattern.md` section 5. The recipes are never on
the table: a video-path sprint refines PATHS (routing, canvas negotiation, admission,
extension), never the shipped render recipe.

### Detail for question (D): the 8 GB / profile cluster -- one decision underneath

- **WAN 8 GB ceiling: CODE-COMPLETE, PROOF-INCOMPLETE, and ONE decision blocks it.** The
  17-frame ceiling reaches a leg only through a variant workflow or a hand-set widget:
  `otr_canonical.json` node 87 ships `max_render_frames=0`, so a plain canonical WAN run
  is UNPINNED and inherits `_TI2V_MAX_FRAMES = 177` -- exactly the 2026-07-23 failure
  shape. Pinning 17 in the canonical is WRONG (it would cap the 16 GB LTX / HuMo legs).
  Proposed shape: `eng_wan_ti2v` DECLARES its own tier ceiling as a capability-row field,
  the widget becomes an operator OVERRIDE with 0 meaning "use the adapter's contract", and
  the profile channel stops mattering -- a real design change with a live blast radius on
  any card with headroom, so ratify before code. The wired chain is pinned by
  `tests/test_remaining_video_contracts.py` (nine hop-by-hop tests) and
  `tests/test_multiclip_effective_contract.py`; commit history in the archive. Also owed
  after the ruling: a render on a PHYSICAL 8 GB card (the four-arm bench prequalified on a
  16 GB card told to reserve 8 GiB, which is not the same claim; the 18-engine campaign is
  coverage, not an 8 GB qualification). One untested edge, cheap to close when this
  reopens: WAN is out of `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan
  CAN contradict by design and `_planned_length` hard-refuses mid-episode -- no test asserts
  a 17-frame tier survives a multi-segment beat.
- **A2 -- the applied-overrides echo hides the profile's `llm.*` override.**
  `nodes/_otr_workflow_apply.py` already flattens `llm`; `scripts/otr_api.py` echoes only
  role / slot / features plus two seed keys, so a run reports "16 overrides" while also
  having replaced the entire LLM configuration. Fix: generate the echo FROM the applier's
  flattened map (never add keys by hand). HELD on question (D), because its whole subject
  is the profile channel.
- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by
  coincidence** (found 2026-07-27, B6 panel, two lenses independently). The coverage
  PLANNER reads `config/profiles/otr_8gb_ltx.json` `video.max_render_frames`, and
  `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The ADAPTER's own
  pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Today both land on 161 (profile
  unpinned, env unset), so nothing breaks. But `workflows/variants/otr_8gb_ltx.env.json`
  ships `OTR_LTX_8GB_MAX_FRAMES=97` and NOTHING currently reads that file. The day a
  launcher honours it without also pinning the profile, the planner emits a 98-161
  frame segment and the adapter refuses it MID-EPISODE -- after the stills are minted
  and, on a multi-segment beat, after the 6.34 GiB checkpoint is hoisted.
  **Deliberately NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame
  beat partitions, which is a production planning decision, not a cleanup. The preset
  carries a `_ceiling_note` saying do not export it alone. Compare WAN, which B3
  wired correctly: `otr_8gb_wan.json` sets BOTH `launch.env.OTR_WAN_TI2V_MAX_FRAMES`
  and `video.max_render_frames`.

---

## SECTION 4 -- PARKED / DEFERRED (out of the working queue)

### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected** (operator: *"park it on go forward"*). The writer casts
from 10 Bark presets (6M/4F, `config/cast_pools.py`) while the cloning engines draw from a
204-entry reference bank (153 resolvable on disk, 97M/106F/1N,
`config/voice_reference_bank.json`); `MAX_SPEAKING_CAST = 10` is a Bark artifact, and
raising it alone does nothing because `_deal_voice_menu` builds the menu from
`VOICE_PROFILES`. Not a constant change: `voice_preset` / `tts_model` are ledger JOIN KEYS
(`cast[].name` / `char_id` / `voice_preset` / `voice_ref_id` / `voice_engine`, joined from
`lines[].speaker` and `beats[].char_id`), so every field's owner is enumerated BEFORE the
menu moves (the ledger law). Under Section 1.1 the shipped default becomes kokoro, so this
item's trigger is "a cloning engine is SELECTED", not "the shipped default". Measured
table and the full argument: `docs/GO_FORWARD_ARCHIVE.md`.

#### What the work is, when it is taken up

1. Enumerate every consumer of a cast row's voice fields -- casting, TTS
   dispatch, per-beat audio slicing, credits, portraits, captions, `obs_publish`
   -- and name the new owner of each field. Exactly one owner each.
2. Make the casting menu engine-aware: Bark presets when the character engine is
   Bark, reference-bank entries when it is a cloning engine. Gender and
   `commercial_clean` already exist on bank rows.
3. Replace `_assert_unique_bark_voices` with an engine-agnostic
   one-voice-per-character invariant. The rule itself is right and must survive:
   two characters sharing a voice is a correctness defect.
4. Derive `MAX_SPEAKING_CAST` from the ACTIVE engine's pool instead of a
   constant. `tests/test_cast_size_is_a_request.py` already asserts the constant
   matches the live stock, so it will report the drift rather than hide it.
5. Prove on `scifi_news_pro` (the only bank on the fable2 writer) with a cast
   larger than 10 and complete speaker-to-`char_id` equality in the ledger.

Related and already shipped: `num_characters` is now a REQUEST rather than a cap
(operator directive, all banks) -- see `tests/test_cast_size_is_a_request.py`.

### The Shakespeare verbatim executor -- do NOT start it in a single session

**Do NOT start the Shakespeare verbatim executor in this session.** It is a
multi-session structural change gated on the ownership table
(`docs/2026-08-03-fidelity-pass-ownership.md`) with four overwrite paths to close
first, and starting it half-way is worse than not starting it.

### PARKED (operator idea, 2026-09-01): image input for the AnimateDiff haunted lane

An i2v anchor for the 8 GB floor lane. Not started; ship-readiness first.

### Carried administrative rows

- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema `.v4`
  literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

---

### 4.X OTR-LITE -- a second, frictionless pack AFTER v2 ships (operator idea, 2026-09-03)

**Not work yet, and explicitly not before v2 ships.** The operator, thinking it
over away from the desk: *"once we ship v2 we ship an OTR-Lite, similar
architecture but only the most frictionless auto-download non-gated models, and
maybe just maybe we can figure an ffmpeg-less solution for a truly streamlined
workflow."*

**Two halves, and the second one pays a debt nobody connected to it.**

* **Non-gated auto-download only.** Already half-mapped:
  `scripts/otr_fetch_lane_weights.py` offers UNGATED sources by design and
  deliberately refuses to paper over the one gated repo (Lightricks/LTX-2.5), so
  its lane list is effectively the candidate set. Measured on the 4090 pod
  2026-09-03, the fully-frictionless bundle is `haunted + one z_image precision
  + stable_audio_3` -- about 20 GB for one complete episode.
* **ffmpeg-less.** This is not only an install-friction win. `subprocess` calls
  to ffmpeg/ffprobe are **35 of the pack's 158 Comfy Registry scan findings**
  (`python_command_injection_risk`), plus most of the 12
  `python_url_command_execution` ones. An in-process encode path shrinks the
  largest non-`os.environ` finding class at the same time as removing the binary
  dependency -- the two goals are the same work. PyAV (`av>=16.0.0`) is already a
  ComfyUI CORE dependency, so it ships on every install.

**THE ONE KNOWN BLOCKER, and it is where this starts.** This PyAV build has no
`libass` and no `drawtext`, so CAPTION BURN and CREDITS cannot move in-process
as-is. The mux, the silent composite and the probe already have PyAV routes. So
the first question of an OTR-Lite effort is captions and credits without
libass/drawtext -- not the mux, which is the part that looks hard and is not.

**Related evidence already on disk:** `docs/RUNPOD_INSTALL.md` section 7A (what a
second machine actually trips over), and the registry finding counts in
`docs/2026-09-03-registry-review-request-READY.md`.

### 4.Y BRING `word_razzle` HOME -- it is the one cloud lane whose NAME hides it (operator, 2026-09-03)

**Operator:** *"word_razzle shouldn't be cloud anymore"*, and the reason behind it:
*"I don't want to mislead my audience -- make cloudy lanes transparent."*

**The transparency problem, stated exactly.** 8 of the 30 registered video
engines render provider-side:

```
cloud_kling_avatar  cloud_seedance_2  cloud_wan_i2v  cloud_wan_i2v_audio
cloud_vidu_q2_pro_fast_720p          google_omni_video  google_veo_video
word_razzle   <- the outlier: nothing in the name says it phones out
```

Seven of the eight self-label via a `cloud_`/`google_` prefix. `word_razzle`
reads like a local text-effect lane and is not one. That is the misleading row,
and it is also the one with a plausible local replacement.

**WHY IT IS NOW FEASIBLE, AND IT MAY NOT EVEN BE A NEW ENGINE.** word_razzle is
"animate a word-card still into a living period poster" -- a cloud i2v taking
(init image + prompt + seed + duration + motion_mode). Both halves already exist
locally:

* **The card.** `ideogram4_local` is the pack's spelling champion -- its own
  recipe note records that every card in the campaign with perfect spelling
  rendered at mu 0.5. It did NOT exist when word_razzle was built as a cloud
  lane in 2026-07-03, which is the whole reason that lane is cloud.
* **The motion.** `still_motion`, `still_pan` and `ltx_video` already animate a
  still. A word card is a still.

**So the first question of the arc is whether this is an ENGINE at all, or just
a PROFILE** -- `character_image: ideogram4_local` + an existing local i2v in
`character_visual`. A profile costs nothing; a new engine id trips five
generated fixtures, two literal rosters, the terminal-frame proof rule and
`docs/VIDEO_LANE_PREFLIGHT.md` gates 1-8. Those are very different prices for
the same outcome, which is exactly why this is a design item and not a
drive-by.

**Owed before code:** a kibitz arc on that question. Also decide what happens to
the cloud `word_razzle` row -- retire it, or rename it `cloud_word_razzle` so the
roster is honest either way. Renaming alone would fix the transparency defect
even if the local lane never lands.

**Separately noted -- FIXED 2026-09-04 (kibitz runpod-found-fixes, ruling R-B;
PBUG-20260904-03).** `word_razzle` failed the 2026-09-03 pod matrix with
`SessionIdentityUnavailable` -- it would render 2 segments from one set of
handles but declares no `session_identity()`. NOT a credentials problem and NOT
pod-specific: the same refusal fires on the 5080 for any beat long enough to
need two segments, on every one of the seven remote adapters. (This paragraph
said "already fixed" for a day while the refusal was untouched; the roster
test's tripwire was the only honest record.)

The fix is a DECLARATION, not an inference: `_CloudVideoBase` (five rows),
`google_omni_video` and `google_veo_video` set `session_residency = "remote"`,
and `beat_session.holds_local_handles()` skips the identity demand for exactly
them -- fail-closed, so an absent or misspelt attribute still reads as local and
the demand stands. No `session_identity()` was added to any cloud adapter: an
identity of handles that do not exist would be theater. The roster test is keyed
on the declaration and a control pins that no engine declares remote AND a local
isolation. Eight local engines still declare the identity (ghost_signal, humo,
ltx25, ltx_8gb, ltx_av, ltx_video, minimax_h3, wan_ti2v).

**Not done, and the predicate question if anyone wants it:** a RECIPE identity
for remote lanes (model id + resolution + params), so a mid-beat env flip is
caught for a cloud lane the way weight drift is caught for a local one. Nobody
has asked for long cloud beats; log it, do not build it.

## SECTION 5 -- Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval; Section 3, question F) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |
| **Seedance softener mangles authored prompts (2026-08-17)** | CANDIDATE only: fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever produces the artifact; nearest coverage `12.108` does not cover blind-regex rewriting of authored text. Detail: `docs/GO_FORWARD_ARCHIVE.md` |
| `PBUG-20260901-04` (kokoro on Python 3.13) | Bible CANDIDATE (a Requires-Python marker rule); promote at the fan-out (Section 3, question F) |

(The 12.139 / 12.140 promotions completed 2026-08-28 and the 2026-08-25 /
2026-08-18 / 2026-08-17 promotion receipts are in the archive.)

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; promotion to the Bible is
tracked in the Bible repo's `otr_coverage_index.yaml` (CLAUDE.md, delta-scrape discipline);
no plan review or invented fixture creates a row.

## SECTION 6 -- Open risks

- **NO CLIENT BANK HAS EVER RUN LIVE.** Every extensibility wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven path end
  to end (fetch -> interpret -> writer -> cleanup -> tail -> publish). Treat the first
  live client-bank leg as a qualification, not a formality. Deferred power-user tiers
  (client own-runner + staging, dependency manifest, standalone story_rules) are
  explicitly OUT of v1 and are a NEW block if the operator ever wants them.
- **CLIENT-AUTHORED PYTHON executes in-process** (wave 3). The posture that must hold in
  every future change: `--activate` is the consent act; the seam fails LOUD
  (`UserBankExecutionError`) and never substitutes; client code never touches the
  canonical ledger; owner IDENTITY is verified so a bank can only run its OWN bundle; the
  shipped fetcher/interpreter registries are never widened to admit a client id. Do not
  relax any of these for convenience.
- **The client-facing surface is LIVE TEXT, not just docs:** the `custom_source_bank`
  row's `guide_ref` is raised to the operator by `require_runnable_bank`, and the
  `source_bank` tooltip repeats it. Any future change to the activation path (folder
  name, CLI verb, restart behaviour) must update `nodes/story_packs/banks.json`, that
  tooltip and `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **The ledger-cleanup pass runs on EVERY bank, not just client banks** (`3d97a130`). It
  is a no-op on a complete ledger and costs no LLM call there, but two shipped-lane
  behaviours did change and are worth watching on the next live legs: (a) unsafe spoken
  language on a `content_owned_readonly` bank is now REPAIRED at the writer tail instead
  of reaching G9 untouched, so a leg that used to die at freeze may now ship a sanitized
  line; (b) a blank `meta.episode_title` is now filled at the tail instead of exploding
  later in `otr_credits_roll`. Both are the intended direction under THE LAW; neither has
  a live receipt yet.
- No code lands mid-sweep of an active qualification campaign (the 420-rung
  uniform-code-confound lesson).
- There is no standalone SFX provider layer to rebuild. Current video clips are
  silent and the terminal mux uses the frozen upstream master audio. The future
  direction in `ROADMAP.md` is to retain and mix selected video-generation audio
  as inexpensive ambience; do not revive the fast-moving provider/bed stack or
  claim that future path is already wired.
- Lean-mean has one current ordered campaign in `docs/LEAN_MEAN_CLEANUP.md`.
  The retired FRONT/TAIL and SW-1 execution model must not be revived.

## After this queue

One owner per file (CLAUDE.md section 1); every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

When the sections above are exhausted, continue with `ROADMAP.md`: lean-mean ->
RunPod/AMD/Mac -> install -> product docs/v2 release. That is a pointer, not work that
precedes lean-mean. Lean-mean is not an item in this queue: `docs/LEAN_MEAN_CLEANUP.md`
is its sole current scope, blast-radius, coding-order, and verification authority.
