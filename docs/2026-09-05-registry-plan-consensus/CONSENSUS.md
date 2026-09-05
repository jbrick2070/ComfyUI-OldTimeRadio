# THE REGISTRY PLAN FROM HERE -- panel consensus (2026-09-05)

**Driver and sole judge:** Claude (Fable 5.1). **Panel and provenance, exactly:**
GPT-6 Astra at `ultra` (`astra_r1.md`, one pass, read-only); a 7-tracer / 21-skeptic
Workflow (Fable 5.1 subagents; 73 grounded findings, 10 BAN_RISK confirmed by two
skeptics each, 0 refuted, 11 UNSURE; journal `wf_3f8af9e0-3a4`); a 10-skeptic
Workflow over Astra's five extra items (all five STAND; `wf_97f8a5e4-d90`); Sonnet 5
byte-level provenance diff of the published alpha.19 zip; Sonnet 5 read of five
packs APPROVED under policy-v0.2 (positive controls); and a live survey of 578
packs / 3,707 versions (`scratchpad/survey_verify_deep.json`). Every panel claim
cited below was re-read by the driver at the real file:line before it was kept.
Anchor: `driver_anchor.md` (two of its claims were wrong and are corrected in §3).

## 1. The verdict in four lines

1. **The published alpha.19 will be BANNED if the reviewer reaches it.** It carries
   an arbitrary-file-read the panel confirmed end to end, plus nine unconfined
   write sinks of the reviewer's #3 ban class. HEAD is better but NOT clean.
2. **Do not post the SHORT request.** It invites the review onto that artifact and
   claims more than is true. Plan B is dead; the panel ranked **D > C > A > B**.
3. **The channel that decides is the code review, not the GitHub issue.** Of ~20
   manual-review issues since 08-02, zero got a maintainer reply; verdicts landed
   on the code regardless. Same-day through 09-01, then a 4-day gap that is still
   open (127 versions from 09-02 on, ours included, untouched). That gap is the
   window to publish alpha.20 with the punch list closed.
4. **The 12 `info` findings are not the punch list and never were.** 0 of 102
   versions approved under policy-v0.2 had a clean scan; 31 of them shell out via
   subprocess, 72 do network I/O, 39 touch `os.environ`. ffmpeg stays.

## 2. What the reviewer is (measured, not inferred)

* Signed `drltdata@comfy.org`, worded `policy-v0.2: <CLASS> -- ... confirmed by
  code-level verify-deep` (ban) or `reviewed SAFE (GOAL2 verify-deep, policy-v0.2)`
  (Active). Astra is right that identical wording does not PROVE automation; the
  survey shows it processed every Flagged version the same day for 18 straight days,
  which is the fact that matters for timing either way.
* Since 08-15: **102 approved, ~350 banned.** Bans by class: RCE 203, command
  injection 59, path traversal 35, unauthenticated side effect ~18, SSRF/egress ~16,
  arbitrary file read 10, deserialization 2, launch 1.
* Rules quoted in verdicts: **Rule 1b cl.1** widget taint keeps full severity;
  **Rule 11** free STRING -> filesystem write with no realpath+commonpath containment
  and no `..` reject = PATH_TRAVERSAL, INSECURE; **Rule 12** free STRING -> network
  with no host allowlist = SSRF (banned even with a `127.0.0.1` default); **s.0.8**
  defaults are irrelevant, `forceInput` is irrelevant -- `/prompt` sets any literal.
  Env vars and on-disk config are operator-trusted (`llmlink` approved with
  `os.environ.get(KEY)`; `openrouter-simple` approved with env-only host/key).
* **Positive controls (five approved packs, read in full):** none writes to a
  widget-named absolute path unconfined. Two idioms pass: (a) never expose a free
  path (combo of registered files; env for hosts/keys); (b) literal containment --
  `LlamaServe-Doc` `backend.py:263-268`: `target = (dest / name).resolve(); if target
  != dest and dest not in target.parents: raise`. An unauthenticated POST with a
  self-scoped side effect (stop its own child process, no params) was approved.
  Our UNC-only refusal is a strict subset of (b) and meets neither idiom.

## 3. The punch list, confirmed (each: two skeptics failed to refute; artifact-identical unless noted)

**A. Arbitrary file read (LFI) -- ban class, 10 bans in the sample**
| id | chain | HEAD | alpha.19 |
|---|---|---|---|
| LFI-00 | `script_json` -> `pool_path`/`path` -> `_materialize_episode_copy` `shutil.copyfile` -> PNG served by `/view` (`otr_image_gen_dispatcher.py`) | closed `9d3f56a7` | **OPEN** (`:1557` existence-only) |
| **LFI-01** | `OTR_VideoRenderBatch.patched_ledger_json` -> `render_driver._still_spine_materialize_row` `:909-929` `shutil.copyfile(source, destination)` into `otr_stills_dir` -> `/view`. **The sibling copier `9d3f56a7` never looked at.** No GPU, no engine needed. | **OPEN** | **OPEN** |

**B. Write-side path traversal (Rule 11) -- ban class, 35 bans in the sample. NINE sinks, not three.**
| id | node / widget | sink | guard today |
|---|---|---|---|
| W1 | `OTR_CaptionBurn` `output_path` :350 -- AND `video_path` :317, because `_default_out` :376-385 writes `<dirname(video_path)>/<stem>_captioned.mp4` beside whatever the caller names | ffmpeg `-y` :280-291 + `.ass` sidecar written BEFORE ffmpeg (`_otr_captions.py:505-507`) | `reject_remote_paths` (UNC/URL only) |
| W2 | `OTR_MasterAudioMux` `output_path` :1064 | ffmpeg `-y` :399-406 (+ obs copy) | same + an obs deny-list, not containment |
| W3 | `OTR_SilentComposite` `output_path` :1536 | ffmpeg `-y` :1458 + `json.dump` `.qa.json` :1482 | same |
| W4 | `OTR_SceneSequencer` `output_dir` :796 (value otherwise DEAD) | `os.makedirs` :910 | none. One skeptic refuted the CLASS (makedirs-only is never banned alone); the sink is real -- delete it |
| W5 | `OTR_PostUpscaleProcgenBlend` `source_mp4_path` :736 decides `src.parent` | `shutil.copy2(src, output_path)` :963/:980 on bypass or empty procgen -- **no ffmpeg, no env, no state needed** | remote reject + suffix reject (filename, not directory) |
| W6 | `OTR_CreditsRoll` `video_path` :1517 decides the directory | six sibling writes :1322-1484 | HEAD: remote reject `:1531-1534`; alpha.19: **nothing** |
| W7 | `OTR_SceneAwareScopes` manifest `episode_id` | filename join :557 -> ffmpeg `-y` :564 (Windows lexical `..` escape) | none on `key` |
| W8 | `OTR_ImageGenDispatcher` `episode_id` :2251 | `makedirs` + PNG copy + JSON | HEAD closed (`portrait_ledger.py:83-92`); alpha.19 **OPEN** |
| W9 | voice nodes `ledger_json` :354 -> `meta.paths.ledger_path` :643 | `save_ledger_safe` -> `_otr_ledger.py:519` `os.replace` over ANY existing JSON | none (the erpk config-poison class) |
| W10 | voice `meta.paths.audio_dir` -> `_otr_audio_cache.py:258-281` | makedirs + writes | UNSURE: needs `use_cache=True` profile + credentials; grep `config/` for `use_cache` before deciding |
| P10 | `OTR_VideoRenderBatch` `engine` :465 | `"node_single_%s.json" % engine` :543 -> `open(...,"w")` :560 (Windows) | none on the join |

**C. Remaining UNC/SMB egress (the class we were ALREADY dinged for; `79dc9828` was incomplete)**
voice `cast[].voice_route` -> `ref_path` -> `_otr_voice_route.py:102 open()` inside fingerprinting;
blend `scopes_mp4_path` :858 omitted from the guard at :925; validator `profile_id` :263 ->
`capability_profiles.py:324` join + open (also a `..`-capable open, discloses nothing);
composite manifest clip paths :1415-1417; scopes :261-266; `scene_sequencer.py:85`;
`otr_video_render_batch.py` has no remote reject at all; and `otr_silent_composite.IS_CHANGED`
step 3 (`os.listdir(dirname(base_video_path))`) survived `79dc9828` on the very file it names.
Graded UNSURE as a ban class (no verdict text names a stat-only UNC sink) -- one guard line
each, and HEAD's own commit message claims them closed, so they ship in alpha.20.

**D. Not a ban risk (refuted or ruled SAFE, with the reason)**
* The 12 scanner sites: every value is env/constant/provider-response; `proc.py` refuses
  shell/string-argv/executable; feeds are https-only + public-address-only + bounded.
* `GET /otr/latest_ledger`: no request input, no side effect, contract-confined path;
  info disclosure (`fullpath`, whole ledger) to an audience that already has `/history`.
  Optional: drop `fullpath`. **Two stale docs say it is env-gated; it is not** (only the
  POSTs are) -- `.comfyignore` viewer paragraph and `PROD_BUG_LOG.md:10690`. Do not
  repeat that sentence in any post.
* The two POST render routes: dead unless `OTR_ENABLE_HTTP_RENDER_ROUTES=1`; in their
  enabled shape they are textbook UNAUTHENTICATED_SIDE_EFFECT with an unconfined
  `engine` filename. Nothing shipped calls them. **Delete them from the shipped
  `__init__.py`** (retire `tests/test_http_render_route_gate.py` in the same commit)
  rather than argue a gate no verdict has ruled on.
* Filtergraphs: the only workflow string that reaches one is the `ass=` basename, guarded
  by the full syntax-character reject `:99-119`; every other token is typed numeric.
  No `drawtext`, `movie=`, `amovie=` anywhere. Do not invent a filter fix.
* Env pins: names are literals; values are `"1"/"0"`, a profile id that LOADED, or a
  hex digest. Deserialization: `torch.load(weights_only=True)` only; no pickle,
  marshal, yaml.load, startfile, webbrowser, extractall, eval, exec.
* Media reads through ffmpeg `-i <widget path>` (mux/composite/credits/caption): the
  panel's ruling is these are NOT what "LFI" means (VideoHelperSuite's identical
  LoadVideoPath is Active on every version), BUT the bytes-copied-verbatim cases above
  ARE, and containing the input paths to the output tree closes both questions at once.

**Anchor corrections (driver was wrong):** `_otr_paths.py:302 _validate_contract` DOES
exist (resolve+relative_to) -- it guards the pack's own helpers only; explicit
widget paths bypass it. `widget_ffmpeg_is_ignored` is in 6 shipped files, not 12.
CaptionBurn's default is NOT inside episodes; it is beside the input.

## 4. THE PLAN (ranked D > C > A > B; unanimous on B last)

**Step 1 -- coder, one change set, before anything is posted or published: close the
inventory, not the scanner's twelve lines.**
* One helper in `nodes/_otr_paths.py`: `confine_to_output_tree(value, field, *, roots=None)`
  -- after `reject_remote_path`, `realpath` both sides, `os.path.commonpath([real, root]) == root`,
  textual `..` reject, root = `comfy_output_dir()` (so the 4060's `OTR_OUTPUT_DIR` keeps
  working) plus an operator env allowlist `OTR_EXTRA_OUTPUT_ROOTS` (env is operator-trusted
  under the policy; this is how the operator keeps any out-of-tree destination). Compare
  resolved-to-resolved so a mapped-drive root matches itself (`U:` resolves to UNC).
* Call it at the EXECUTE METHODS (never in shared resolvers or the spawn gateway --
  `OTR_STANDING_RULINGS`): W1/W2/W3 on `out` INCLUDING the `_default_out` result and on the
  input paths (`video_path`, `silent_video_path`, `master_audio_path`, `base_video_path`);
  W5 on `source_mp4_path`/`procgen_mp4_path`/`scopes_mp4_path`; W6 on `video_path`; W7
  whitelist `key` through `_validate_episode_id`; P10 whitelist `engine` against the
  registry before it becomes a filename. W4: delete the makedirs (and the inert widget,
  three-things rule, trailing so it is nearly free). W9/W10: take the ledger path and cache
  dir from `in_flight_ledger_path()` / the pack's own resolvers, never from wire `meta.paths`.
* LFI-01: gate `render_driver._still_spine_materialize_row` with the same
  `_trusted_pool_source` HEAD already has for the dispatcher (and note Astra's point: it uses
  `abspath`, not `realpath` -- a junction case is one line away). LFI-00 is already at HEAD.
* Section C: `is_remote_path` before every stat/open listed; add `reject_remote_paths` to
  `otr_video_render_batch.py` and the blend's `scopes_mp4_path`; finish
  `otr_silent_composite.IS_CHANGED` step 3.
* Delete the two POST routes from `__init__.py`; drop `fullpath` from the GET (optional).
* Untouched: the three byte-hashed files; ffmpeg via subprocess; every episode-content path.
* Proof: full suite + Bug Bible; `build_variants --check` + the four widget/link tests if any
  widget is removed; then ONE canonical leg that reaches `otr/obs/` (the containment must not
  move where a real episode lands -- every default already sits under `otr/episodes/<ep>/`,
  verified, and the canonical JSON stores no absolute path, verified).
* Review: this document is the r1 arc. r2/r3 on the diff per the matched-review rule --
  Sonnet 5 QA on the finished diff; a Fable gate only because a missed thread here costs a
  fourth ban and a burned version string.

**Step 2 -- operator: bump `pyproject.toml` to `2.0.0-alpha.20` and push** (auto-publishes).
Coder then downloads the CDN zip and byte-diffs it against the commit (the Sonnet method,
`scratchpad/a19/compare.py`) and reads the scan record before anyone says anything publicly.

**Step 3 -- operator, AFTER alpha.20 is up: post ONE short factual note** naming alpha.20 as
the version to look at, the two banned surfaces and how they closed, and the four classes
closed since (file read, write containment, UNC stat, routes removed) -- asking nothing.
Rewrite `docs/2026-09-04-registry-review-request-SHORT.md` for that; as written it is
DO-NOT-SEND (wrong version, claims every surface closed, cites a pre-v0.2 precedent).
Expect Flagged first (12 info), then the verify-deep verdict; if it is SAFE, alpha.20 goes
Active and `latest_version` resolves for the first time since alpha.8.

**Why not A (fix, publish, say nothing):** it is Step 1+2 without Step 3, and Step 3 costs
nothing once the artifact is clean. **Why not B (post now):** it points the review at an
artifact with a confirmed LFI and nine Rule-11 sinks -- a third ban, on a version we asked
them to read.

## 5. Residuals the panel could not settle (byte facts named)

* Whether verify-deep credits an env gate around ROUTE REGISTRATION (no sample verdict either
  way) -- mooted by deleting the routes.
* Whether a stat-only UNC touch inside `IS_CHANGED` is a ban class -- mooted by closing them.
* W10 prerequisites (`use_cache` profiles) -- one grep.
* `feihou-toolbox` 2.9.5 (published 09-04, still Flagged) is the cleanest external test of
  whether "ffmpeg `-i <widget path>`" alone is treated as LFI; read its verdict when it lands.
* Coverage gap recorded by the P1 tracer: per-line wav / foley stem FILENAMES inside validated
  dirs were not individually audited (a `..` in a `line_id` is the residual question).
