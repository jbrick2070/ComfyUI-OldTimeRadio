# OTR Go-Forward Plan

**THIS FILE IS WORK THAT IS NOT DONE YET.** Operator, 2026-09-04, twice: *"GO_FORWARD
should be clean with the coming work, not done work"* and *"GO_FORWARD should NOT have
done stuff -- only stuff that needs to be done."* Finished work lives in
`docs/HANDOFF_LOG.md` (what happened, with receipts) and `docs/GO_FORWARD_ARCHIVE.md`
(the receipts themselves, verbatim, never summarized). If you find a paragraph here that
only reports something already shipped, move it to the archive -- do not delete it, because
roughly a third of these paragraphs carry an operator RULING and losing one costs far more
than the length does.

**THE ORDER BELOW IS THE OPERATOR'S, set 2026-09-04 evening:** finish the CODING (the
correctness bugs, then the design rows that need an arc first), then the SCAN COLLAPSE,
then prove it on the SECOND MACHINE, then the POD rental, and the REGISTRY LAST -- because
the manual review request is filed on the first NON-ALPHA version, the same day it
publishes, so the version under review is the version a stranger would install. In his
words: *"I want to get all those design and coding done so we can finally try our registry
fix and see our improvements."*

**Standing constraints, every chunk, every window:** the 5080 loop is untouched -- nothing
ships that reduces tomorrow's `obs` count. The pod stays STOPPED until its own item. One
coder window per file owner (CLAUDE.md section 1). Every chunk = focused tests + full suite
+ Bug Bible + commit AND push + `HEAD == origin/v2.0-alpha`. Reviews are ONE OR TWO lanes
per decision (operator, 2026-09-04), not three; a design choice with more than one
defensible answer still gets its arc BEFORE code.

---

**The `1.x` and `Batch Rn` labels are STABLE ROW IDS, not an order.** Other docs, the
handoff log and the bug log cite them by number, so they keep their names even though the
operator's order above moves them around. Read the numbered sections for sequence and the
row ids for identity.

---

## WHERE TO PICK UP

**Tree:** `v2.0-alpha`, HEAD == origin, clean. Suite baseline **13492 passed / 126 skipped
/ 1 xfailed**. Bible **335 entries** (repo `81a8a9e`). No resident server; VRAM at the
desktop baseline. Codex CLI standard credits are OUT until **2026-09-07 08:34 PDT** and
Spark overflows its context on a 12 KB plan -- fill that seat with Sonnet/Opus or a second
agy model (Gemini 3.1 Pro (High) and 3.8 Flash are different reviewers) and name the roster
honestly.

**THE NEXT COMMIT is the ratchet commit of the scan collapse** (item 3 below). Its plan is
CLOSED and is `kibitz-runs/2026-09-04-registry-findings-collapse/r4/final.md` -- read it
whole before typing. The owners `nodes/_otr_shared/env.py` and `nodes/_otr_shared/proc.py`
and the shared ratchet `tests/fixtures/ratchet.py` are already on disk and INERT: nothing
imports them, so the tree's behaviour is unchanged until the migration starts.

**Decisions owed by the operator (skip, do not guess):** the four in item 6's registry row,
the Section 3 question list, and PBUG-20260904-05's profile design (the 4060 proves, the
5080 promotes).

---

## 1. THE CORRECTNESS BUGS -- coding, now, no GPU leg needed to FIX

Story quality is DONE and is not reopened (operator 2026-08-04). These are CORRECTNESS
defects: a gender or voice contradicting the source, a beat that renders the wrong
picture, a leg that dies late.

### 1.2 CHARACTER GENDER LADDER (queue item 3a) -- the SPEC REWRITE is written; next is ONE review round, then code

Left open, recorded in PBUG-20260815-04's follow-up: a given-name alias can
match a different character with that surname (COLONEL FITZWILLIAM via "fitzwilliam"; right
here by coincidence) -- a surname-only alias for the short_form tier is the next fork.

Operator rulings folded: ARIEL / PUCK / ROBIN stay on the roll (locked index entries); Dr. Lira Kell is
female (locked).

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

**STANDING RULINGS FROM THE AUDIT (rows A-F are built; the receipts are in the archive,
these sentences are not).**
* Ruling R-A: the pin is honoured INSIDE `_otr_paths.otr_obs_dir()` and skips the in-tree
  assert for that path only (returned as typed, no `resolve()`); `_validate_contract`
  untouched; the ledger already authorized both roots.
* Enforced by `tests/test_output_root_single_owner.py` (AST, named allowlist:
  `eng_mesh_stage` because ComfyUI's SaveGLB refuses any path outside `folder_paths`' own
  dir -- NOT a leftover twin, do not "retire" it -- and `vram_context_test`, a diagnostic
  outside the contract).
* The `__init__` pin is preserved deliberately; removing it is its own design item.
* The third convention, `_otr_paths.comfy_models_dir()` / `OTR_MODELS_DIR`, stays parked
  (cursor r3: do not open a third env in this diff). A FOURTH spelling belongs to the same
  parked item (agy r4): `_otr_image_engines/flux2_klein.py:209-215` probes `folder_paths`
  first, then the two env vars, and never the legacy tree -- the inverse of `_models_root`'s
  order. When the merge happens it has four owners to retire, not three.
* Out of scope, named: `scripts/otr_ingest_pd_voices.py:101`,
  `scripts/otr_macbeth_probe.py:603/620/1196` (scripts cannot import the owner without the
  package; convert when a script is next touched).
* Operator directive: an orphan is ripped 100% or wired back in.
* Operator: "even though there may be dependency B, what does dependency B lead to,
  dependency C -- follow the dependency chain to find truly rippable code."
* Fifteen swept symbols are protected by `docs/OTR_STANDING_RULINGS.md`; the sweep now READS
  those rulings and reports such candidates under a separate heading naming the doc that
  speaks for them. They stay, each with its ruling. Among them, `p0_source_char_budget` is
  the diagnosed-but-unwired fix for OPEN `PBUG-20260729-03` -- still owed.

**G. FOLLOW-UPS THE 2026-09-04 ARC SURFACED AND DID NOT BUILD -- each with the
reason it waits.**
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
* **NEW BUILD ITEM, arc opened 2026-09-04 -- collapse the registry scan from
  158 findings to about five by the one-owner rule** (plan and driver anchor
  in `docs/2026-09-04-registry-findings-collapse/`, 5080-local). Measured from
  alpha.17's real payload: the env rule fires ONCE PER FILE (103 findings, 103
  files), so one `os.environ` owner takes it to 1; the subprocess rule fires per
  site (35 in 20 files), so one process runner takes it to ~2; six of the
  twelve "url command" hits are the words `ffprobe -count_frames` inside error
  strings; the three singletons (OpenProcess, `Path.read_bytes` for a sha256,
  `__import__("sys")`) each have a clean replacement. It does NOT reach Active
  -- that needs zero findings or the manual review -- but it turns the human
  review from a ledger into five lines and drops every `credential-access` tag.
  Semantics-neutral by construction (no env name, default or precedence moves;
  the 4060's numbers do not move). Design questions for the arc: typed getters
  vs casts at the site, a declared knob catalog, the guard's shipping order
  across batches, whether the sidecar-venv Popen streams share the runner.
* **Handed over by the shipping window when it stood down (2026-09-04):**
  (a) re-check `GET /nodes/comfyui-old-time-radio/versions/2.0.0-alpha.17/comfy-nodes`
  periodically -- non-null confirms the pycairo-marker theory and the card
  should show ~34 nodes; still-null means something else fails in the Linux
  extract container (residual suspect: kokoro pulls torch, and a multi-GB
  download would blow the 600 s extract timeout). (b) `viewer/index.html`
  SHIPS in the alpha.17 zip and calls three unregistered endpoints; one
  `.comfyignore` line fixes it, and `.comfyignore` decides what ships -- the
  operator authorizes that line, neither window just does it. (c) The registry
  manual-review request is ready to file and is a PUBLIC post: it needs his
  own explicit go, not a peer relay. (d) From the `-02` window:
  `nodes/_otr_shared/partner_nodes.yaml` carries the literal
  `AUTH_TOKEN_COMFY_ORG` fourteen times as pinned data about Comfy's own
  partner nodes -- the YARA scanner never read it (YAML), but a human reviewer
  who greps the zip after reading our request finds the string we said we
  removed. Scrub to a placeholder BEFORE a reviewer engages, after checking the
  partner-row parser does not key on the literal. (e) Optional draft polish for
  the review request: one line that 47 of the 158 findings are the
  subprocess/ffmpeg family and 103 are `os.environ` reads, so no single
  subsystem's removal clears the version -- the argument for a review over
  another patch.
* **The draft 8 GB profiles cannot write on two of the three banks
  (PBUG-20260904-05):** `8gb_lite`, `otr_4060_floor` and `otr_4060_viz_12b`
  (all `status: draft`) set `gguf_n_ctx: 2048`, and the writer prompt is 2,741
  tokens on `science_news` and 3,338 on `original` -- the budget refuses loud
  before writing a word. `media_archive` FITS: attempt 5 of the publish matrix
  above rendered and published on `8gb_lite` with that bank, so the fault is
  the bank-plus-context pairing, not the profile alone. The row that SHIPS,
  `otr_4060_12b_gguf_offload`,
  runs the same 12B at `gguf_n_ctx: 4096` (measured 7.8 of 8.2 GB) and fits.
  Design call, owed a kibitz arc: retire or re-context the drafts, pair a
  smaller pinned writer (Qwen3-4B / gemma E4B are on disk), or make the plan
  stage refuse a profile whose context cannot hold its own prompt -- in the
  profile/variant layer, never a silent truncation. The 4060 owns the 8 GB rows.

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

## 2. THE DESIGN ROWS -- each gets its arc BEFORE code

Operator 2026-09-04: *"design rows -- designing, what, yeah let's do this."* Each row
below has more than one defensible answer, so each takes its arc first and then the
code. None of them is a drive-by.

### 1.1 KOKORO-ONNX BACKEND (queue item 2) -- the default voice that installs everywhere (design item, kibitz arc BEFORE code)

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

## 3. THE SCAN COLLAPSE -- the coding that makes the registry fix worth filing

**The plan is CLOSED after a full r1-r4 arc and is the next coder chunk:**
`kibitz-runs/2026-09-04-registry-findings-collapse/r4/final.md`, with its round judgments
beside it and the two measured receipts in
`docs/2026-09-04-registry-findings-collapse/` (`argv0_receipt.txt` resolves all 35 spawn
sites; `env_drift_receipt.txt` lists the 14 knobs read with more than one default).

**The invariant:** a machine fact has ONE owner, and a test proves the copies agree. From
158 `info` findings to about 9, and the `credential-access` tag on one file instead of
eleven.

**Sequence, each commit full-suite green:**

1. **The ratchet commit** -- both AST guards as named-set ratchets calling
   `tests/fixtures/ratchet.py` in BOTH directions, the network guard with its five named
   files, `tests/test_terminal_frame.py` taught `otr_proc` at every import depth with
   lowercase `popen` added to `_SPAWN_CALLS` (`:495` AND `:498` are the receipt), and the
   mux knob test's predicate resolved from the mux's own import.
2. **Batch (a)** `nodes/_otr_shared/**` including `gpu_residency`'s Windows-liveness fix.
3. **Batch (b)** the engine subpackages, one commit each: audio, image, upscale (the
   spandrel chunked sha256 rides here), video (`eng_ltx25`'s `import sys` and the six
   `wan_shared` strings ride here), google_api.
4. **Batch (c)** `nodes/_otr_*.py`.
5. **Batch (d)** EVERY remaining `nodes/*.py` -- eleven carry no prefix, so a glob would
   miss them -- plus the root `__init__.py`, whose env import goes ABOVE line 51 and
   OUTSIDE the swallowing try/excepts.
6. **Acceptance** both guards green with empty pending sets, then a canonical leg that
   PUBLISHES to `otr/obs/`, its launcher's profile and roots READ BACK from the leg log
   before any receipt is written.

**Every migrated module imports the owners ALIASED** -- `otr_env` / `otr_proc` -- at its own
depth, because `env` and `proc` collide with live parameter and local names. Test seams
patch `M.otr_proc.run` in the module's own batch.

**After the batches, re-run `python scripts/dead_code_closure.py`:** migrating a hundred
files strands helpers, and the sweep now reports anything a standing ruling protects under
its own heading rather than as a clean candidate.

## 4. PROVE IT ON THE SECOND MACHINE -- the 8 GB set and the 4060 capstone

The 4060 is the only box that can answer whether the pack works somewhere other than
where it was written. Its profiles and its fresh-install path are ITS to own; the
5080 owns the workflow JSON, `nodes/`, and profile status promotions.

### 1.5 THE 8 GB SHIP SET -- promote what the clean room proved (queue item 4)

Klein is already the image default in all
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

### 1.13 MAC / AMD -- images only, later (queue item 8 tail)

Operator: Mac and AMD ship images only (ruling 2026-09-01), and he is "not hopeful".
Landed: the credits font, the llama-cpp hint and four platform guards. Owed, in order:
(1) one measured Klein render on Apple Silicon -> `nodes/_otr_image_engines/registry.py`
gains `mps` on the `flux2_klein` row (cuda-only today) -> `otr_mac_mps` flips off
`google_image` (README's Mac row says so); (2) the upscale stage accepting `mps`
(`_otr_upscale_engines/__init__.py`, deliberately deferred); (3) a measured ROCm boot for
`otr_amd8_rocm` / `otr_amd16_rocm`. ROCm already qualifies for Klein (presents as cuda).
Needs hardware neither NVIDIA box has.

## 5. THE RENDER PROOFS AND THE POD RENTAL

Batched by the leg that proves them -- test the least. The pod is ONE rental with two
jobs on the same dollar: the provisioner's acid test (saved template -> provision -> one
published episode, ZERO hand steps) and then the looped lane sweep for the HuMo 14B /
LTX 2.5 / indextts2 second-machine receipts. The overnight runners publish and shut the
GPU off by themselves, so it is one evening, not a campaign.

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

* **Leg C5** (Klein + LTX 2.5, stock flags): the operator eyeball is Section 1.5 item 1. Leg A's question is
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

## 6. THE REGISTRY -- LAST, and that is a ruling not an accident

### The registry rows, carried forward (the 2026-09-02 queue's item 1)

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

* So the credential fix held across the bump, Flagged-on-info-only is the expected outcome,
  and **every precondition on the manual review request is now satisfied** -- it is ready
  to file on the operator's go and nothing else gates it.
* **THE SCAN COLLAPSE IS THE CODER'S NEXT CHUNK** -- see WHERE TO PICK UP at the
  top: plan closed at r4, orphan rips shipped, owners on disk; the ratchet commit is
  next. Nothing in it needs the operator except the review filing above.
* **One residual risk that cannot be settled from here:** kokoro pulls `torch`,
  and if the container's preinstalled CPU torch does not satisfy the constraint, pip would
  download multiple GB inside the 600 s extract timeout. Unknowable without the image.
* **ANSWERED 2026-09-03 18:30Z, AND THE ANSWER CLOSES (b) AS NOT-OURS. DO NOT CHASE
  KOKORO.**
  **`aff1f9c4` STAYS.** The pycairo marker is correct on its own merits -- a package
  publishing 21 Windows wheels and zero Linux wheels genuinely does not install in that
  container -- and it is what makes a Linux `pip install -r requirements.txt` work at
  all, extract pipeline or no. It simply was not the node-count blocker.
  The Linux pod keeps the engine because
  `scripts/otr_pod_provision.sh` apt-installs the headers and now pip-installs pycairo
  explicitly -- do NOT "simplify" that line away.
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

**3. THE CORRECTNESS BUGS -- in this order.** Story quality is done; these are
correctness defects (a gender or voice contradicting the source, a beat that renders the
wrong picture, a leg that dies late).
* **3b. PROMPT v3 "DRAW THE CRUX" -- HALF A SHIPPED AND PROVEN 2026-09-03. HALF B IS OPEN
  AND IS THE NEXT CODER WINDOW ON THIS LANE.**
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

## 7. RULINGS OWED BY THE OPERATOR -- each with its default if unruled

Skipped by every coder window until he rules. A window that guesses one of these is
doing work that may be thrown away.

### How to read these rulings

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

## 8. PARKED / DEFERRED -- out of the working queue, kept for the ruling each carries

Nothing here is scheduled. Each row is here because a ruling parked it, and the ruling
is the reason it must not be quietly re-opened.

### The parked rows

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

**Not done, and the predicate question if anyone wants it:** a RECIPE identity
for remote lanes (model id + resolution + params), so a mid-beat env flip is
caught for a cloud lane the way weight drift is caught for a local one. Nobody
has asked for long cloud beats; log it, do not build it.

## 9. THE BUG BIBLE FIELD AND THE OPEN RISKS

### Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval; Section 3, question F) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk; never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |
| **Seedance softener mangles authored prompts (2026-08-17)** | CANDIDATE only: fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever produces the artifact; nearest coverage `12.108` does not cover blind-regex rewriting of authored text. Detail: `docs/GO_FORWARD_ARCHIVE.md` |
| `PBUG-20260904-05` (draft 8 GB profiles: 2048 ctx holds `media_archive`, refuses the other two banks) | CANDIDATE: a live refusal, but the fix is a profile-design call not yet made; promote when the verify condition is automatable |
| `PBUG-20260901-04` (kokoro on Python 3.13) | Bible CANDIDATE (a Requires-Python marker rule); promote at the fan-out (Section 3, question F) |

(12.151-12.156 promoted 2026-09-04 -- PBUG-20260904-01..04 and -06; receipts in
HANDOFF_LOG. The 12.139 / 12.140 promotions completed 2026-08-28 and the 2026-08-25 /
2026-08-18 / 2026-08-17 promotion receipts are in the archive.)

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; promotion to the Bible is
tracked in the Bible repo's `otr_coverage_index.yaml` (CLAUDE.md, delta-scrape discipline);
no plan review or invented fixture creates a row.

### Open risks

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

### After all of the above

One owner per file (CLAUDE.md section 1); every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

When the sections above are exhausted, continue with `ROADMAP.md`: lean-mean ->
RunPod/AMD/Mac -> install -> product docs/v2 release. That is a pointer, not work that
precedes lean-mean. Lean-mean is not an item in this queue: `docs/LEAN_MEAN_CLEANUP.md`
is its sole current scope, blast-radius, coding-order, and verification authority.
