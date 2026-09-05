# OTR Go-Forward Plan

**THIS FILE IS WORK THAT IS NOT DONE YET.** Operator, 2026-09-04, twice: *"GO_FORWARD
should be clean with the coming work, not done work"* and *"GO_FORWARD should NOT have
done stuff -- only stuff that needs to be done."* Finished work lives in
`docs/HANDOFF_LOG.md` (what happened, with receipts) and `docs/GO_FORWARD_ARCHIVE.md`
(the receipts themselves, verbatim, never summarized). If you find a paragraph here that
only reports something already shipped, move it to the archive -- do not delete it, because
roughly a third of these paragraphs carry an operator RULING and losing one costs far more
than the length does.

**THE ORDER BELOW IS BY DEPENDENCY, and it is the operator's (2026-09-04, late):** *"I really
want to get the registry thing fixed, but no sense in fixing it if bug fixes will break it --
get all the easy code items done first, then the registry, then testing"* -- and, on second
look, *"shouldn't we move [the design rows] before the registry?"* Yes: a design row is a code
change too, and a code change after the publish is a NEW VERSION with its own per-version
review. So: the SCAN COLLAPSE first, because its guard tests make every later change that adds
an env read or a process spawn FAIL THE SUITE -- once they ship, nothing after can push the
findings back up. Then the EASY code (closed specs). Then the DESIGN rows, each behind its arc,
because they edit shipped code and must be in the version that gets reviewed. THEN the
REGISTRY: publish the first non-alpha and file the review the same day, so the version under
review is the version a stranger installs. Then TESTING, which needs the published pack. The
cost of this order is real and chosen: the registry waits for the design arcs.

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

**Row ids were renumbered 2026-09-04 (twice; this is the final order).** Other docs, the
handoff log and the bug log cite the ORIGINAL ids, so here is the map.

| now | cited elsewhere as |
|---|---|
| `2.1` | `1.1` |
| `2.2` | `1.2` |
| `2.3` | `(new row)` |
| `2.4` | `1.3` |
| `2.5` | `(new row)` |
| `3.1` | `(new row)` |
| `3.2` | `(new row)` |
| `3.3` | `(new row)` |
| `3.4` | `1.3` |
| `3.5` | `(new row)` |
| `3.6` | `1.4` |
| `3.7` | `2.5` |
| `5.1` | `1.5` |
| `5.2` | `1.6` |
| `5.3` | `1.7` |
| `5.4` | `(new row)` |
| `5.5` | `(new row)` |
| `5.6` | `(new row)` |
| `5.7` | `1.1` |
| `5.8` | `1.9` |
| `5.9` | `1.10` |
| `5.10` | `1.11` |
| `5.11` | `1.12` |
| `5.12` | `(new row)` |

---

## WHERE TO PICK UP

**State (2026-09-05, the registry punch list is CLOSED):** `v2.0-alpha`, HEAD ==
origin at `cc90b261`, clean tree. Suite **13459 passed / 126 skipped / 1 xfailed,
RC=0**; Bug Bible 22. No resident server; VRAM at the desktop baseline. GPT-6 Astra is available
at `ultra` reasoning -- the older "Codex credits out until 09-07" note is dead.

**`2.0.0-alpha.19` IS PUBLISHED AND FLAGGED -- its scan landed 09-05** (19 deps
recorded). The record is BYTE-IDENTICAL to alpha.18's: **12 findings, all `info`,
zero critical** (4 env-var reads, 5 network, 3 subprocess). The collapse
**158 -> 12** is now measured on the version a stranger would install.

**NEXT UP: the alpha.20 bump, and it is the operator's.** Item 2.9 is CLOSED
(`cc90b261`): the nine unconfined write sinks, the arbitrary file read the earlier fix
missed, the remaining UNC stats, and both POST routes. Suite **13459 passed / 126 skipped / 1 xfailed**.
The coder's next move comes AFTER the bump: download the CDN zip, byte-diff it
against the commit, read its scan record. Only then does the short note get
posted, naming alpha.20.

**ONLY THE OPERATOR DOES THESE -- stop and ask:**
* **The alpha.20 bump. 2.9 is green and pushed (`cc90b261`), so this is the next
  move** -- editing `pyproject.toml` IS the publish. Then the short note (item 4).
  **All three review drafts are DO-NOT-SEND until alpha.20 exists** -- the SHORT one
  names alpha.19 and claims every surface is closed; neither is true.
* Any further `pyproject.toml` bump (each one auto-fires a publish).
* The rulings in item 6.

**Keep this file clean:** when a row is finished, DELETE it from here and write the
receipt in `docs/HANDOFF_LOG.md`. Completed narrative goes to
`docs/GO_FORWARD_ARCHIVE.md`. This file is the to-do list, not the log.

---

## 1A. THE SECURITY WORK -- SIX SURFACES CLOSED, NONE OPEN

**Receipts in `docs/HANDOFF_LOG.md`; the narrative is in GO_FORWARD_ARCHIVE.md.**
Reading the registry's own `status_reason` -- not the finding counts -- showed
alpha.13/.14 were BANNED by a human for *"RCE (code execution) -- attacker-reachable
via unauthenticated /prompt (node widget) or no-auth route"*. Both clauses were real.

**CLOSED:**
| surface | commit |
|---|---|
| the `ffmpeg` widget reaching `argv[0]` on five nodes | `a9e0383e` |
| UNC/SMB coercion via `replay_from`, `workflow_json_path`, media paths | `a9e0383e` / `843b79d4` |
| the same coercion through `IS_CHANGED`, which runs BEFORE the execute guard | `79dc9828` |
| forged image-cache entries -> arbitrary local FILE READ, served by `/view` | `9d3f56a7` |
| the no-auth route half (`POST /otr/video_render_*`, unconditional in alpha.13) | `b198026a`, 09-03 |
| the pending sweep deleting what it could not READ (Fable finding 2) | `31dc6861` |
| replay import trusting a ledger the manifest never verified (Fable finding 3) | `14c6a6db` |

**Nothing from the security reviews remains open.**

**THE METHOD WORTH KEEPING:** read `status_reason`, never infer from counts. Every
prior session optimised a number that was never the blocker.

## 1. THE SCAN COLLAPSE AND THE DEAD-CODE RIP -- DONE

**Nothing owed. Receipts in `docs/HANDOFF_LOG.md`.** The env/proc single-owner
migration, the acceptance leg (`obs_publish OK`, 00:10:17, asset verified on disk),
and two rounds of dead-code rip all shipped:

* **37 top-level symbols** ripped (`47bf95d6`) -- `-859` lines from `nodes/`, 220
  tests removed. Verified by Astra + Gemini 3.8 + Sonnet, unanimous, against a brief
  that ruled **"it has tests" an INVALID objection** -- that circular justification
  is what had kept them alive, because `dead_code_closure.py` treats `tests/` as ROOTS.
* **11 more** (`e5a9fd0f`) that the first rip orphaned -- the chain terminates at
  round 3.
* **Two files are now UNTOUCHABLE alongside `eng_indextts2.py`:**
  `nodes/_otr_resolved_request.py` (byte-hashed by `RUNTIME_FINGERPRINT_SOURCES`;
  ripping one symbol from it DEMOTED the Lemmy voice route and had to be reverted)
  and `nodes/_otr_source_grounding.py` (its siblings are ruling-protected).

**THE LESSON, because it cost three false-positive rounds:** an AST scan sees CALLS,
not registrations or type positions. Aliased imports, `@register` decorators, pydantic
field annotations, and overrides of a base in ANOTHER installed pack are all invisible
to it. Grep for the PATTERN, never the name.

## 2. THE EASY CODE -- closed specs, no arc; lands BEFORE the version that gets reviewed

Every row here edits shipped `nodes/`. With the collapse's guards in place, a row that adds
an env read or a spawn fails the suite, so this is exactly the work that must land before
the publish and cannot move the findings once it does. Story quality is DONE and is not
reopened (operator 2026-08-04); these are CORRECTNESS defects.

### 2.1 CHARACTER GENDER LADDER (queue item 3a) -- the SPEC REWRITE is written; next is ONE review round, then code

- **Run ONE review round on the rewritten spec, then write the code.** DONE WHEN: the round is recorded and the ladder is implemented and green.
- **Decide the surname-only alias tier (open fork).** A given-name alias can match a different character with that surname (COLONEL FITZWILLIAM via "fitzwilliam"), so short_form needs a surname-only alias rule. DONE WHEN: the rule is specified and a wrong-character match is impossible. Recorded in PBUG-20260815-04's follow-up.
- **Implement tier 3 as an LLM verdict on the name**, cached in a PERSISTENT name index so each name is asked once, ever. DONE WHEN: a repeat name costs zero LLM calls.
- **Keep tier 4 name-frequency as the deterministic floor**, so the ladder stays TOTAL when the LLM call fails. DONE WHEN: an LLM failure still yields a gender.

RULINGS (constraints, not history):
- ARIEL / PUCK / ROBIN stay on the roll (locked index entries); Dr. Lira Kell is female (locked).
- **Shakespeare: fill ONLY the 32 `unknown` roster rows.** KNOWN rows from the parsed dramatis personae stay untouchable; the ladder's lower tiers may fill the blanks.
- **THE WEB-SEARCH TIER IS REPLACED, not plumbed.** There is no web call.
- Operator's design, his words: *"just have the LLM decide -- ask what the likely gender of this person name is, have the LLM decide, and keep that in an index of names."*
- The invented lanes (original, scifi_news_pro, media_archive) KEEP ROLLING by the standing ruling -- their characters do not exist, so no lookup of any kind applies.

### 2.2 GHOST POOL -- uniqueness on the finalized prompt (queue item 3b; r1 is in, build)

**Root cause (why this matters):** the pool is not too small, the duplicate check is. Four slots (`GHOST_V2_SLOTS`) make the picture; the check reads one leaf (`key = leaf.casefold()` in `nodes/otr_shot_lock.py`), so two beats with the same leaf and different characters are rejected although they render different pictures. Growing the pool cannot fix this.

- **Build:** key uniqueness on the FINALIZED POSITIVE PROMPT, applied identically to writer output, replay and the deterministic path (capacity becomes clauses x motifs).
- **Build:** a bounded progression, total by construction -- unused finalized prompt -> reuse a leaf where a different motif keeps the prompt new -> reuse the least-recent signature, deterministic on `episode_seed + beat_id`, never adjacent.
- **Build:** the allocator appends a PER-BEAT reuse disposition to that beat's existing `fallback_reason`. ShotLock stamps one batch-wide reason today, which would erase the original model-failure reason.
- **Build:** only pool exhaustion becomes recoverable. The ten `GhostAuthorError` raise sites (unknown mode, missing bookend motif, invalid role, empty `motif_cue`) are structural corruption and stay loud.
- **Shared code:** measure both boxes before pushing (CLAUDE.md 0B).

**CUT, so nobody rebuilds them:** the combinatorial generator, act-scoped uniqueness (no authoritative act field exists), and "loud handover" (controlled reuse under a second name).

**DONE WHEN:** >18 same-mode beats complete; mixed replay plus fresh authoring completes; all three paths share the invariant; adjacent finalized prompts never repeat; same seed gives identical output AND receipts; every beat keeps a valid `ghost_prompt`; then the failing five-act topology through `workflows/otr_canonical.json` with `obs_publish OK` and the file on disk.

The tests that encode the obsolete absolute-leaf rule (`test_ghost_prompt_v2_lane.py:399-405, 437-451`; `test_ghost_signal_author.py:925-931`) are REPLACED with the new invariant, not deleted.

**Open question:** whether "no adjacent repeat" is the right viewer threshold -- check it against frames rather than more reasoning.

### 2.3 PROMPT v3 HALF B -- the one cheap, unblocked piece
**Cheap and unblocked:** the kernel joins subject and place with a fixed `"in the"` ("a spinning turntable **in the** riverbank"). One small preposition fix; waits only on his eye.

### 2.4 OPEN DEFECTS THAT ARE CODING WORK (queue item 3c; a leg may prove some later, none needs a leg to FIX)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not.

**Line cites in this section drift; re-pin a row's cite when you touch it.** Engine adapters live under `nodes/_otr_video_engines/`, `_otr_audio_engines/` and `_otr_image_engines/`; bare `eng_*.py` cites mean those paths, and `render_driver.py` is `nodes/_otr_video_engines/render_driver.py`.

#### P0 / source-span cluster

- **`full_text` HTML block joins fuse tokens** (`...PolygonsNASA/JPL-`, `...School ofEngine`, `...doing.Let's s`, `...(AMR).The resea`) -- dominant P0 span-rejection cause. Do: name the adapter that builds `full_text`, insert the separator at admission without breaking any accepted ledger's `source_digest`, pin a fixture from those four strings. Belongs in the source adapter, not the codex normalizer. DONE WHEN: fixture passes and no accepted digest changes. BLOCKED ON: Section 3 question B being ruled.
- **TOMBSTONED 2026-09-04 -- the two deterministic-P0-rung rows below are GONE, and the re-pin they were waiting on is DONE.** The row asked to "locate the deterministic P0 repair rung by behaviour, or tombstone both rows". Located by behaviour: it does not exist. `repair_literal_source_metadata`, `_validate_fact_index` and `a0_payload` are absent from `nodes/` entirely (0 files each); only `allowed_source_fields` survives, in `_otr_scifi_p0_contract.py`. No function anywhere under `nodes/` prunes a span, an evidence row or a fact -- the only `*_repair*` callables left are LLM retry-NOTE builders in `_otr_scifi_news_pro.py`, which is a different mechanism. AND THE SURVIVING CODE ALREADY DOES THE OPPOSITE OF BOTH COMPLAINTS: `compact_p0_repair_context` (`_otr_scifi_p0_contract.py:334`) trims a repair CONTEXT to a byte budget longest-field-first, never trims `rejection` / `source_digest` / `allowed_source_fields`, populates a `trim_receipt`, and its own docstring says "NO SILENT ANYTHING". The silent-prune defect and the all-or-nothing defect were carried out by the rewrite that removed those symbols. Nothing to build; if a prune rung is ever reintroduced, these two rows are the record of what it must not do.
- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0 after two attempts on non-literal fact source spans; provider/model convergence, extends BUG-11.35. NOT a word/length gate. Blocks the last 120w receipt and the `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed in tree, reverify still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs provider cap `512`; residual fix now unblocked. Related independent items: P9 8K structured-capacity follow-up, GGUF structured-enforcement NEWBUG. Do not raise the minimum word target as a capacity workaround.

#### Coverage, canvas and clip-contract

- **Route lock is ONE NODE TOO LATE for the image phase** (node order in canonical JSON: `87 VideoDirector -> 88 ImageDirector -> 89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher -> 92 VideoRenderBatch`). `resolve_final_shot_engines` runs at 92; stills are minted at 91 and image PROMPTS at 89, so the image phase relies on its own MIRROR (`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`). **Chunk 1 of the coverage block is the fix.** Note node 89 precedes node 90, so hoisting to ShotLock still does not put MetaBrief downstream of the authority -- that needs a VideoDirector-time freeze and is NOT in scope. (Also the "image-phase still ownership" item from the campaign queue.)
- **ShotLock WRITE-side canvas validation still owed** (O1 judgment item 1): `otr_shot_lock.py` stamps `video.canonical_canvas` unvalidated from a possibly-empty policy. No longer urgent -- the engine declares its own canvas and `tests/test_ltx_8gb_canonical_canvas.py` guards the disagreement that matters. Close it when the general canvas resolver lands.
- **`ltx_av` underruns long beats.** It caps at `_LTX_AV_MAX_FRAMES` (`eng_ltx_av.py`, default 497, env-overridable) and clamps before render. It is NOT "renders to target natively" as three earlier docs claimed.
- **`docs/ENGINE_MATRIX.md` prints the DECLARED contract only.** Once a profile pins an `ltx_8gb` ceiling the matrix keeps printing `9-161 step 8` for a tier whose real window is narrower, and the `--check` drift gate cannot notice because it diffs the registry. Owed at the prequalification step, not before.

#### Voice engines

- **bark rolls non-speech (steady tone or noise floor) on some lines and returns success** (PBUG-20260902-03). Not the announcer graph, not the knobs, not line length. Fix: an OUTPUT GUARD in `eng_bark.py` -- score each take for speech shape (dominant frequency 70-400 Hz, low flatness in most one-second windows), re-roll with `seed + 1` up to twice on a failing score, WARNING-log every re-roll, keep the best take if all fail (the ledger field is always filled). Design choice (threshold, retry count) -> full arc before code. DONE WHEN: a stub engine returning a tone first and speech second yields speech in one retry. bark is opt-in (kokoro is the shipped default on both slots), so this is a dropdown-fallback fix, not a ship gate.
- **Remember, do not build (operator 2026-09-02): the 5080's overnight runs ROTATE voice engines on purpose** ("for overnight runs I like to rotate voice models, to be honest, since the 5080 can handle them all"; "there's probably no rotation machinery, just something you need to remember"). The `config/profiles/otr_rot_tts_*` profiles are the instrument (bark / chatterbox / dia characters, chatterbox / dia announcers, and `otr_rot_tts_kokoro`, kokoro on both slots -- the stranger default belongs in the rotation too); whoever queues overnight legs cycles them by hand. Never fix his overnight profile to kokoro-only in the name of the ruling; the stranger-facing default stays kokoro on both slots.

#### Routing, env-capture and the credits card

- **`wants_talking_prompt()` escapes any routing freeze -- REAL, but it is a DESIGN
  row, not a closed spec (re-pinned 2026-09-04).** Grounded: `wants_talking_prompt`
  (`eng_ltx_av.py:640`) returns `_recipe_config(self._recipe())["two_stage"]`, and
  `_recipe()` (`:652`) documents its fresh read as DELIBERATE -- "an operator flips
  daily<->hero per beat by swapping OTR_LTX_AV_UNET / OTR_LTX_AV_RECIPE". THREE live
  consumers call the hook at three different graph times: `otr_video_director.py:603`
  (node 87), `otr_meta_brief_image_prompt.py:604` (node 89), and
  `render_driver.py:1760` (node 92). Nothing captures the answer between them, and
  `route_freeze.py` captures only `OTR_FORCE_ENGINE_MAP` / `OTR_ENABLE_HUMO_HOSTS`
  -- not the recipe knobs. So the director can plan a talking still, the prompt
  generator write a talking prompt, and the renderer render the other register.
  **The DONE WHEN names code that does not exist** (`row_is_active`, `ltx_resolved`
  -- 0 hits each), i.e. it describes a new shared evaluator over captured state.
  That is a design choice with more than one defensible answer -- what to capture,
  where the capture lives, and whether the deliberate per-beat flip survives it --
  so it takes an arc BEFORE code. Not scheduled.
- **`provider_side` is a THREE-part rule, not an attribute.** `_is_cloud_video_engine` (`render_driver.py`) accepts a `cloud_` id prefix OR the attribute OR `node_key.startswith("cloud_")`. `cloud_kling_avatar` has no `provider_side` attribute, so an `engine_facts` builder using a bare `getattr` would classify it local and send a cloud avatar to local LTX. DONE WHEN: a regression covers picked AND forced `cloud_kling_avatar`.
- **Env-read sites missing from the S0b inventory** (two remain): the `OTR_ENABLE_HUMO_HOSTS` reads in `render_driver.py` and `otr_meta_brief_image_prompt.py`, and the recipe / UNET re-read in `eng_ltx_av.py` (`wants_talking_prompt` / `_recipe`) outside `assert_usable`.
- **The credits card needs a SMALL-CANVAS VARIANT, and the ladder is not it.** At 512x288 (ltx_8gb) col1 is 65px past its footer even with every droppable ledger row dropped; at 640x360 it is 12px over. Both are drawn anyway and LOGGED at ERROR naming the canvas. At 288 lines the three-column console is already a polite fiction. This is a DESIGN job -- a card laid out for a small canvas -- not more ladder heroics.

#### 1.4a DEAD CODE / DUPLICATED-DECISION AUDIT (2026-09-04) -- three Sonnet lanes, driver-verified

The REJECTED list below is part of the record precisely so a discarded claim is not re-raised as a fresh finding. Line cites drift -- re-pin when touched.

**STANDING RULINGS FROM THE AUDIT.**
* Ruling R-A: the pin is honoured INSIDE `_otr_paths.otr_obs_dir()` and skips the in-tree assert for that path only (returned as typed, no `resolve()`); `_validate_contract` untouched; the ledger already authorized both roots.
* Enforced by `tests/test_output_root_single_owner.py` (AST, named allowlist: `eng_mesh_stage` because ComfyUI's SaveGLB refuses any path outside `folder_paths`' own dir -- NOT a leftover twin, do not "retire" it -- and `vram_context_test`, a diagnostic outside the contract).
* The `__init__` pin is preserved deliberately; removing it is its own design item.
* The third convention, `_otr_paths.comfy_models_dir()` / `OTR_MODELS_DIR`, stays parked (cursor r3: do not open a third env in this diff). A FOURTH spelling belongs to the same parked item (agy r4): `_otr_image_engines/flux2_klein.py:209-215` probes `folder_paths` first, then the two env vars, and never the legacy tree -- the inverse of `_models_root`'s order. When the merge happens it has four owners to retire, not three.
* Out of scope, named: `scripts/otr_ingest_pd_voices.py:101`, `scripts/otr_macbeth_probe.py:603/620/1196` (scripts cannot import the owner without the package; convert when a script is next touched).
* Operator directive: an orphan is ripped 100% or wired back in.
* Operator: "even though there may be dependency B, what does dependency B lead to, dependency C -- follow the dependency chain to find truly rippable code."
* Fifteen swept symbols are protected by `docs/OTR_STANDING_RULINGS.md`; the sweep now READS those rulings and reports such candidates under a separate heading naming the doc that speaks for them. They stay, each with its ruling. Among them, `p0_source_char_budget` is the diagnosed-but-unwired fix for OPEN `PBUG-20260729-03` -- still owed.

**G. OPEN FOLLOW-UPS FROM THE 2026-09-04 ARC.**
* `tests/test_openrouter_slug_curation.py::test_routers_appear_in_both_slot_dropdowns_and_auto_leads` passes on this box only because of an UNTRACKED catalog cache. A machine-local lie; the test should build the cache it needs or skip by name.
* The google omni / veo / image adapter tests resolve `comfy_output_dir()` without pinning `OTR_OUTPUT_DIR`, so they write provider bytes under the live `output/otr/episodes/_shared/tmp/` and fail in a worktree (Tier-3 walk-up lands on `C:\Users\output`). Pin in their fixtures.
* **UNFOUND EXPORTER:** inside a pytest session `OTR_OUTPUT_DIR` is set to `C:\Users\jeffr\Documents\ComfyUI\output` -- the value `__init__.py:97-108` would compute -- yet it is NOT set by `tests/conftest.py`, `tests/__init__.py`, `pyproject.toml`, any registered plugin, or by importing the mux in plain Python. Consequence: a test injecting an output root only through a `folder_paths` stub is overridden. DONE WHEN: the exporter is found and it is decided whether conftest should strip `OTR_OUTPUT_DIR` at import the way it now strips `OTR_OBS_DIR`.
* `otr_credits_roll.py:149` reads `.git/HEAD` for the production ledger's commit rev; a git WORKTREE has a `.git` pointer FILE, so 44 credits tests fail there. Harmless in production; the reader should follow a gitdir pointer.
* A RECIPE identity for remote video lanes (see R-B above): logged, not built.
* Scripts' own ffmpeg readers (`otr_ingest_pd_voices.py:101`, `otr_macbeth_probe.py:603/620/1196`): convert when a script is next touched.
* `.kibitz/comfyui.local.md:26` records 23 nodes / 60 links / 132 widget slots; the canonical graph is 23 / 61 / 133. Regenerate with `--force` before the next arc.
* The `__init__.py:97-108` output pin: preserved deliberately (it is what makes every helper agree inside ComfyUI); removing it changes where Desktop installs render and is its own design item.
* **BUILD ITEM, arc opened 2026-09-04 -- collapse the registry scan from 158 findings to about five by the one-owner rule** (plan and driver anchor in `docs/2026-09-04-registry-findings-collapse/`, 5080-local). One `os.environ` owner takes the env rule from 103 findings to 1; one process runner takes the subprocess rule (35 sites in 20 files) to ~2; six of twelve "url command" hits are the words `ffprobe -count_frames` inside error strings; the three singletons (OpenProcess, `Path.read_bytes` for a sha256, `__import__("sys")`) each have a clean replacement. It does NOT reach Active -- that needs zero findings or the manual review -- but it turns the human review from a ledger into five lines and drops every `credential-access` tag. Semantics-neutral by construction (no env name, default or precedence moves; the 4060's numbers do not move). Design questions for the arc: typed getters vs casts at the site, a declared knob catalog, the guard's shipping order across batches, whether the sidecar-venv Popen streams share the runner.
* **Handed over by the shipping window when it stood down (2026-09-04):** (a) re-check `GET /nodes/comfyui-old-time-radio/versions/2.0.0-alpha.17/comfy-nodes` periodically -- non-null confirms the pycairo-marker theory and the card should show ~34 nodes; still-null means something else fails in the Linux extract container (residual suspect: kokoro pulls torch, and a multi-GB download would blow the 600 s extract timeout). (b) `viewer/index.html` SHIPS in the alpha.17 zip and calls three unregistered endpoints; one `.comfyignore` line fixes it, and `.comfyignore` decides what ships -- the operator authorizes that line, neither window just does it. (c) The registry manual-review request is ready to file and is a PUBLIC post: it needs his own explicit go, not a peer relay. (d) `nodes/_otr_shared/partner_nodes.yaml` carries the literal `AUTH_TOKEN_COMFY_ORG` fourteen times as pinned data about Comfy's own partner nodes -- the YARA scanner never read it (YAML), but a human reviewer who greps the zip after reading our request finds the string we said we removed. Scrub to a placeholder BEFORE a reviewer engages, after checking the partner-row parser does not key on the literal. (e) Optional draft polish for the review request: one line that 47 of the 158 findings are the subprocess/ffmpeg family and 103 are `os.environ` reads, so no single subsystem's removal clears the version -- the argument for a review over another patch.
* **The draft 8 GB profiles cannot write on two of the three banks (PBUG-20260904-05):** `8gb_lite`, `otr_4060_floor` and `otr_4060_viz_12b` (all `status: draft`) set `gguf_n_ctx: 2048`, and the writer prompt is 2,741 tokens on `science_news` and 3,338 on `original` -- the budget refuses loud before writing a word. `media_archive` FITS, so the fault is the bank-plus-context pairing, not the profile alone. The row that SHIPS, `otr_4060_12b_gguf_offload`, runs the same 12B at `gguf_n_ctx: 4096` and fits. Design call, owed a kibitz arc: retire or re-context the drafts, pair a smaller pinned writer (Qwen3-4B / gemma E4B are on disk), or make the plan stage refuse a profile whose context cannot hold its own prompt -- in the profile/variant layer, never a silent truncation. The 4060 owns the 8 GB rows.

**REJECTED, with reasons -- do not re-raise these.**
* `_DEFAULT_CLIP` / `_DEFAULT_VAE` "have 7 references": a DRIVER false positive. `git grep -w` matched `flux2_klein.py` and `lumina_image.py`, which define their own same-named constants. Within ideogram4_local all four were definition-only.
* `MAX_PROVENANCE_NOTE_CHARS`: looked dead in-file, has 2 repo-wide refs. ALIVE.
* The ~90 `OTR_*` env knobs: a deliberate, pervasive escape-hatch pattern, every sampled one inside live code. Not slop.
* The standalone HuMo render chain (`render_episode_concat.py` and peers, ~5,000 lines with tests): zero live callers, but `docs/LEAN_MEAN_CLEANUP.md` sec 2.4 explicitly protects active render tools pending a per-file re-ground, and the old bulk kill list is marked SUPERSEDED -- DO NOT EXECUTE. A deferred decision, not an oversight. Belongs to whoever runs that pass.
* Long historical comments explaining past bugs: deliberate project practice.
* Duplicated engine PROMPT COMPOSERS: ruled 2026-08-23, lanes stay independently re-wordable. Only duplicated FACTS and DECISIONS count.

### 2.5 PREFLIGHT THE CLONING ENGINES' REFERENCE WAVs -- THE ROW AS WRITTEN WAS WRONG; it is a DESIGN item now

**ATTEMPTED AND REVERTED 2026-09-04. Read this before rebuilding it.** The row used
to say "preflight the resolved `ref_path` files in `OTR_CastLock` BEFORE the writer
call". It was implemented that way, with 9 passing tests, and a gpt-6-astra kibitz
lane killed it on three grounded counts. All three were verified against the real
files:

1. **CASTLOCK RUNS AFTER THE WRITER, so the stated benefit is impossible.**
   `workflows/otr_canonical.json` link 230 carries node 1 `OTR_LedgerScriptWriter`
   into node 62 `OTR_LedgerFreezeCascade`, and link 234 carries node 62 into node 80
   `OTR_CastLock`. A check at CastLock cannot save the writer's call, because the
   writer has already run.
2. **IT WOULD HAVE REFUSED CASTS THAT RENDER FINE TODAY -- the worse direction.**
   `nodes/_otr_voice_node_common.py:413-418` DELIBERATELY clears a stale
   `voice_ref_path` whose file is absent and resolves another through
   `_resolve_clone_ref_path`. A missing DECLARED path is a recoverable state by
   design, not a failure. Refusing on it blocks working episodes.
3. **It would rarely fire on a real cast anyway.** The ordinary CastLock stamp
   writes `voice_ref_id` + `voice_engine`, NOT the path fields; the dispatch
   resolves that ID through the bank. A check keyed on a declared path skips the
   common case entirely.

**WHAT A CORRECT VERSION WOULD NEED, if this is still wanted:**
* it must run BEFORE node 1 (the writer), which is a different node or a check
  inside the writer's own preflight -- not CastLock;
* it must reuse the dispatch's FULL effective-reference resolution, bank fallback
  and policy-route precedence included, and refuse only when THAT fails -- anything
  narrower re-creates defect 2;
* it must handle ID-only rows, which is the normal shape.

That is no longer a closed spec with one right answer, so it is a DESIGN item and
takes an arc before code. Not scheduled. DONE WHEN: an owner is named for the
pre-writer check and it refuses only what the dispatch could not have resolved.

## 3. THE DESIGN ROWS -- each gets its arc BEFORE code; ALSO before the registry, because a code change after the publish is a new version and a new review

None of these is the next coder window (that is item 1), but every one of them lands
BEFORE the registry publish: a design row edits shipped code, and shipped code changed after
the publish is a new version and a new review. Each row has more than one defensible
answer, so each takes its arc first (one or two lanes per decision -- operator 2026-09-04).
The guards from item 1 keep every one of these from moving the findings.

### 3.1 PROMPT v3 HALF B -- the arc (r2 in flight), then the code
* **3b. PROMPT v3 HALF B -- the next coder window on this lane.** Read `docs/OTR_STANDING_RULINGS.md` "THE BEAT'S SUBJECT IS A PHYSICAL ARTIFACT" before touching this.
  Shape (operator-ruled 2026-09-03): extend the EXISTING batched Ghost author with the beat's dialogue (never a second pass), and have it name a PHYSICAL ARTIFACT of the story, preferring one the beat refers to -- never an abstraction and never a noun the dialogue mentions only to say it is absent. **The truck must NOT be drawn** ("this isn't just some dusty list of truck routes" is a rhetorical negative; Ellie is holding a ledger in an archive).
  Root defect: `resolve_crux_kernel` picks `objects[ordinal % len(objects)]`, so the ledger beat drew `pen`.
  Constraints carried from r1: the render batch DOES have `IS_CHANGED`; replay returns before the author runs, so a migration needs an explicit seam at the replay boundary, not a version check in the author; the beat's own `text` projection is the accessor -- not a `_ghost_line_index` join, which silently misses synthesized beats. Half B changes the STORED object, so it needs version-dispatch discipline and a re-author path for replay.
  Full arc before code. r2 in flight; records in `kibitz-runs/2026-09-03-prompt-v3-half-b/`, anchor in `docs/2026-09-02-animatediff-ledger-experiments/prompt-rule/` and `kibitz-runs/2026-09-02-prompt-v3-crux/`.

### 3.2 THE OTHER VIDEO LANES -- the shared composer's face paragraph (measured, not started)
* **3d. The OTHER video lanes -- measured, not started.** Ten of eleven lanes lead their prompt with the cast's face paragraph via `motion_common.compose_parts`; on `wan_ti2v` that is 83 of a hard 100-word cap, so the camera clause silently falls off. **AMENDED by the r3 reviewer:** ADD a crux clause beside `appearance` on the silent image-to-video lanes; do NOT drop the face on redundancy alone -- the foley/mime lanes drop it because their joint latent SPEAKS the prompt, which is not a general I2V rule. Audio-in lanes (HuMo, the h3 audio lane, `ltx_audio_in`) and the two Google text-to-video lanes keep the face unconditionally. Runs after his eye on Half A. Measurements: `docs/2026-09-02-animatediff-ledger-experiments/prompt-rule/other_lanes_audit.md`.

### 3.3 ORPHAN-OCCUPANCY REGISTRY -- design item, full arc BEFORE code

`has_local_resident_llm()` (`nodes/_otr_model_loader.py`) reports "nothing resident" the moment a timeout clears the cache dict, even while the orphan worker still runs CUDA kernels; `nodes/otr_shot_lock.py` and `nodes/otr_video_render_batch.py` both trust that signal before visual or video work. Shape: process-global lock-protected registry of in-flight generations, registered before invalidation, cleared via `Future.add_done_callback`, fail-fast admission on `request_slot`, the two visual-entry guards reading real occupancy. Deferred three times as correctly out of scope for the cache-bookkeeping fixes (PBUG-20260825-04), and each cut of that fix found a new race, so this is a genuine design choice: full arc first.

### 3.4 ONE MANIFEST, PREFLIGHT AUTO-DOWNLOAD (queue item 8; design row; operator asked "and auto download for all?" 2026-09-01 -- confirm to schedule)

RULING: Keep the rule that nothing downloads DURING a render.
RULING: Design item: kibitz arc before code.

Problem: only the writer LLMs, bark, musicgen and the kokoro voices fetch themselves. Every image engine, every video engine, Stable Audio 3, the cloning TTS engines and the reference WAVs are manual placement behind two fetchers under `scripts/` that the registry bundle does not ship. Move the fetch to the queue-time preflight that already refuses with "PREFLIGHT FAIL: the running server cannot see: <files>".

- **Manifest** -- write one `config/model_manifest.json` (repo, revision, path, destination, bytes, sha256, gated) merging the provisioner's pinned tiers and the fetcher's lanes (12 rows each today). DONE WHEN it is the single source read by the preflight, the pod provisioner and the matrix generator.
- **Preflight fetch** -- resolve the selected dropdowns to artifacts, print the total, refuse early if disk is short, download into the running ComfyUI's models tree through `folder_paths` (never `C:\ComfyUI-Models`) with `.part` files, hash verification and resume. Move the fetcher's existing download code under `nodes/` so it ships in the registry bundle. DONE WHEN a clean box renders with zero manual model placement.
- **Gated rows** -- LTX 2.5 and any other gated row refuses BEFORE downloading, naming the terms URL and the token step.
- **`download_policy` widget** -- auto (default), ask (list sizes only), never (air-gapped).
- BLOCKED ON: Section 1.1 -- the manifest carries the kokoro-onnx weights, so this lands after it.

### 3.5 SHIP-AUDIT SURVIVORS (2026-09-01) -- each needs a design decision, not a grep (queue item 8)

Receipts and every file:line: `docs/ship-audit-2026-09-01/SHIP_LIST.md` (section 8 holds 51 disputed items still awaiting an operator ruling).
Related survivors live elsewhere: the voice item rides Section 1.1, the 8 GB writer item Section 1.5, Mac / AMD Section 1.13.
These are not mechanical and each wants a kibitz arc before code.

- **Runtime writes inside the pack directory** (cloud-media billing ledger, OpenRouter catalog cache) -- a registry update wipes them, and it bites exactly the registry-install users. Route through `nodes/_otr_paths.otr_shared_cache_dir()`. DONE WHEN: no runtime file is written under the pack dir, plus a migration note for existing ledgers.
- **`_fit_reason` never consults `needs_fp8_te` / `needs_fp4_te`** (`nodes/_otr_shared/capability_profiles.py`) -- fp8 and NVFP4 engines qualify on ROCm tiers whose `dtype_policy` forbids them. DONE WHEN: two clauses keyed on `dtype_policy` reject them, with coverage.
- **The janitor cannot sweep `tmp/audio_slices`** (`nodes/_otr_janitor.py`: directory granularity, newest-child mtime) -- the unswept tree costs disk and slows every ComfyUI boot sweep. DONE WHEN: slices are swept and a test pins the widened auto-delete scope (three lines of code, but it widens what gets auto-deleted).
- **Cloud spend with no ceiling** -- `cpu_floor` and `otr_mac_mps` route every image role to the paid Google API on the mere presence of a key, and the BYO-key lane has no reserve/bill/ledger path (`eng_google_image.py`). DONE WHEN: spend is ledgered or the lane is an explicit opt-in.
- **`eng_ltx_video` / `eng_ltx_av` reload ~14 GiB of weights per beat** -- adopt the `prepare()` + `external_results` pattern the sibling lanes use. DONE WHEN: weights load once per run, not per beat.

### 3.6 THE ADAPTATION DESIGN (queue item 8; hardened, NOT yet built; multi-session -- start only with room to finish step 1)

Plan of record: `kibitz-runs/2026-08-03-adaptation-fidelity/r2/final.md` (5080-local).

RULING (2026-08-23): the verbatim lane is Shakespeare only; public_domain may paraphrase.
RULING (keystone): compile source speech from an authenticated segmented artifact, never generate it; "summarize into X words" means SELECTING WHICH REAL SEGMENTS FIT THE BUDGET, not paraphrasing.
CONSTRAINT: ceiling by arithmetic is 1,520 words (19 voiced beats at act_count 7, `BEAT_WORD_HARD_MAX` 80); full-scene performance would need a beat-topology redesign. Build target is the 300-word unit.

**NEXT, IN ORDER:**

1. **Segmented source artifact** -- schema, spans, hashes, `body[start:end] == segment.text`, omission receipts, plus the pass-to-field ownership table. **BLOCKS EVERYTHING: nothing else codes until that table exists.** DONE WHEN: artifact schema + ownership table exist and the span/hash identity holds.
2. **Cast from the selected cut** -- real scenes carry 3-12 speakers against a 6-character ceiling (`_otr_casting.py` 1-6, `OutlineRequest` rejects >6), so which speakers appear must follow from the cut that fits the word budget. Coupled to the capacity guard: at act_count 1 there are exactly THREE voiced beats, so a 4-person cast is a guaranteed `CastVoiceCoverageError`. DONE WHEN: cast derives from the cut and `compute_episode_budget` receives the TRUE locked cast. Blocked on step 1.
3. **Loosen the count-match invariant** at `OTR_LedgerScriptWriter.py:4061-4067` (hard-raises on any locked != requested), and change the pack text that tells the model to drop figures. DONE WHEN: a locked count differing from requested no longer raises.
4. **Extend `_otr_provenance.py`** -- do not add a second attribution owner -- and bind its output to the verified body hash. DONE WHEN: attribution has exactly one owner and is hash-bound.
5. **Schema migration to retire `cast_hints`** -- still required by the validators and by `public_domain_manifest_schema.json`, so manifests and tests migrate in the same change. DONE WHEN: no validator or manifest references `cast_hints`.

**KNOWN AND NOT FIXED:** `canonicalize_shakespeare_text` truncates at 12,000 chars and the interpreter sees only the first 5,000, so a 3,445-word scene reaches the brief as ~880 words, silently. Fix belongs with the step-1 artifact work, where each beat is fed its own segment rather than a blind prefix.

### 3.7 STYLE / IDENTITY DECISION WORK (queue item 8; backlog; not the next coder window)

- **Derived style/genre field** -- add to `run_story_brief_reflection` (`_otr_story_brief.py:513`), stamp beside `story_brief`, repoint the treatment `Style:` line (`video_engine.py:1762`) and the HUD (`video_engine.py:1336` -> `_build_left` `:1592`) at it. Why: fixes the credits line uniformly for all six banks. DONE WHEN: all six banks show a content-derived style in credits + HUD.
- **Rename `meta.style` -> `meta.story_scaffold`** in ONE atomic change across: writer stamps, credits `_story_style_receipt`, `visual_plan.style`, `video_engine.py:1336`, tests, AND the ledger validators -- `_otr_ledger_consistency.py` (`MatrixRow("style", ...)` at `:68`, `:177`) and `_otr_ledger_cleanup.py`. DONE WHEN: no `meta.style` references remain and ledger validation passes on a live episode. Miss the validators and the first episode fails ledger validation.
  - Operator ruling: too many metas; the field is neither scifi nor a description.
- **Ghost-name reconciliation fork (DECISION NEEDED)** -- pitch cast never reaches `lock_cast` (names are a pure pool draw; `source_character_names` deliberately None for invention lanes). Decide: scrub briefs after cast lock, or propagate pitch names. Evidence: Evelyn/Leonard as offscreen lore; Fogbound Rails bio still opens "Lizzie Gray". Cross-listed as Section 3, question C. BLOCKED ON: operator/decision.
- **Dead fields to resolve** -- `ending_template` computed but no LineRequest call site passes it; `seed_policy.style_seed_env` validated but unconsumed; `dramatic_state` derived PRE-dialogue and goes stale in the treatment. DONE WHEN: each is either wired to a consumer or removed.
- **`meta` is a 120-key drawer** -- scope as its own rip under the ledger law (every field exactly one owner).
  - Operator ruling: this is the cleanup the operator keeps asking for.

## 4. THE REGISTRY -- ONE THING LEFT, AND IT IS THE OPERATOR'S

**`2.0.0-alpha.19` is published and Flagged -- the expected outcome.** Its scan
landed 09-05 with a record byte-identical to alpha.18's (12 `info`, zero critical).
Nothing here is a coder task.

**THE ORDER CHANGED ON 2026-09-05 -- the post comes LAST, not first.** The consensus
panel found the published alpha.19 still carries a confirmed arbitrary-file-read and
nine unconfined write sinks (item 2.9), and that the reviewer is a code review that
bans within the day, not a queue that reads issues (zero maintainer replies on ~20
manual-review issues since 08-02). Sequence: **2.9 green and pushed -> operator bumps
alpha.20 -> coder byte-diffs the CDN zip against the commit and reads its scan ->
operator posts ONE short note naming alpha.20** (the two banned surfaces and how they
closed; the four classes closed since; asks nothing).

**REWRITE BEFORE POSTING:** `docs/2026-09-04-registry-review-request-SHORT.md` is
DO-NOT-SEND as written -- it names alpha.19, claims every surface is closed, says
`widget_ffmpeg_is_ignored` covers "each node" (it is 6 files), and cites
`comfyui-video-xy-plot`, a pre-v0.2 human approval that proves nothing about the
current reviewer. Keep its ban-quote opening and the measured table; change the
version, drop the xy-plot line, add the 2.9 classes.

**DO NOT POST EITHER OLDER DRAFT.** `...-READY-v2.md` and the 2026-09-03 draft are
kept only as the alpha.17 record. They are not merely stale, they are WRONG:
* neither mentions the BAN -- they ask for a review of finding COUNTS, as though
  flagging were the problem. Asking for a re-review while the thing they banned the
  pack for is unfixed is the one way to burn the request;
* they claim credit for fixing 2 `critical` findings that were rule
  `prohibited-string` and were **already gone as of alpha.16**;
* they credit the `python_url_command_execution` drop to reworded error strings. That
  rule DID go 12 -> 0, but the SPAWN-OWNER collapse did it, not the wording.

**MEASURED, from `GET /nodes/comfyui-old-time-radio/versions?include_status_reason=true`:**

| rule | alpha.17 | alpha.18 |
|---|---:|---:|
| `python_environment_manipulation` | 103 | 4 |
| `python_command_injection_risk` | 35 | 3 |
| `python_url_command_execution` | 12 | 0 |
| `python_network_operations` | 5 | 5 |
| `windows_process_manipulation` / `bytecode` / `sensitive_file_access` | 1 each | 0 |
| **total** | **158** | **12** (all `info`, zero critical) |

**alpha.19's own scan is READ (09-05): identical to alpha.18's, so the table above
is the number the post quotes** -- for the version a stranger would install.

**UNCHANGED AND STILL TRUE:** there is no publisher self-service route to Active;
`Flagged` is the expected outcome, not a failure; the card's "N Nodes" count is a
different, stalled pipeline and is not a work item; and `aff1f9c4` (the pycairo
marker) STAYS -- do not "simplify" it away.

## 5. TESTING -- needs the published pack or a GPU leg

The 4060 template test installs FROM THE REGISTRY, which is why it cannot precede item 4.
The render proofs are batched by the leg that proves them -- test the least. The pod is
ONE rental with two jobs on the same dollar, and its runners shut the GPU off themselves.

### 5.1 THE 8 GB SHIP SET -- promote what the clean room proved (queue item 4)

RULING (2026-09-01): Klein is the image default in all 19 low-VRAM profiles and the 8gb / 12gb / amd classes -- see `docs/OTR_STANDING_RULINGS.md` "IMAGE ENGINE DEFAULTS BY MACHINE CLASS".
RULING: "Finish retiring the profiles" is no longer an option -- `config/profiles` stays as lab presets and the matrix is the record.
RULING: Record only what published.

Owed, in order:

1. **Record the Leg C5 receipt** -- BLOCKED on the operator watching the published episode. Then add the receipt to `config/machine_classes.json` (`proven[]` on the 8gb class, image column included; `known_limits` keeps the pace) and regenerate `docs/MACHINE_MATRIX.md` + the README block via `scripts/otr_machine_matrix.py`. DONE WHEN the 8gb `proof_summary` no longer says "image lane unexercised".

2. **Ship the 8 GB Klein/LTX profile** -- the 4060's untracked `otr_cleanroom_8gb_klein_ltx25` is not a shipped graph. Promote it as `otr_8gb_klein_ltx25` (writer `google/gemma-4-E2B-it`, image `flux2_klein`, video `ltx25_high_video`, bark voices until Section 1.1 lands so it is 3.13-safe), build its variant (`scripts/build_variants.py --all` then `--check`), add the matrix row, and repoint README's "8 GB card, Klein stills and LTX 2.5 video" row at it. `status: draft` until item 1 flips. DONE WHEN `--check` is clean and README no longer says "not a shipped graph yet".

3. **Eight profiles pair an 8 GB ceiling with the 12B writer and die in the writer preflight** (`Needed=8.13 GB (weights=6.63, kv=1.40 @ n_ctx=2048)` under a 6.8 GB ceiling; smallest prompt needs 2064 input tokens and P0 reserves 2800 output, so 4096 ctx is ~9.5 GB -- ctx is the symptom, the writer MODEL is the cause). Draft set: `otr_8gb_ltx`, `otr_8gb_wan`, `otr_8gb_fastwan`, `8gb_lite`, `cpu_floor`, `otr_amd8_rocm`; SHIPPING pair: `otr_g4_ltx_8gb`, `otr_w45_ltx_8gb`. Repoint all eight to the E2B writer the 8gb class already ships, or delete them. Not a one-line edit: the GGUF registry has only the 12B and Qwen3-8B rows, and `gemma-2-2b-it` is a TRANSFORMERS row -- do not re-propose it. DONE WHEN all eight clear the writer preflight on an 8 GB ceiling.

4. **cfg 1.0 promotion A/B** on three real episode prompts (announcer portrait, character portrait, scene beat), the same four cells (nvfp4 / bf16 x cfg 1.0 / 2.0), rendered to a NEW dir under `docs/ship-audit-2026-09-01/image-jury/`, then the operator's eyeball. One seed was a strong lead, not a proof. DONE WHEN the operator rules. Scheduled: Section 2, Batch R5.

### 5.2 THE 4060 TEMPLATE TEST SET -- what is still open before the capstone (queue item 5)

RULINGS (verbatim, do not re-open):
* **One JSON for now (operator 2026-09-02).**
* The 4060 dropdown-friendly JSON is saved AFTER the testing below, with kokoro on both voice slots (never bark: bark renders a long announcer line as a tone, PBUG-20260902-03).
* Any hand step is a bug: file it in `docs/PROD_BUG_LOG.md`, fix it at the root, retry.

OPEN WORK:
* **Run the 4060 zero-friction test.** Clean portable (Python 3.13) on the 4060 -> Manager install of the Active registry version -> the 8 GB saved-dropdown variant (`workflows/variants/otr_nvidia_8gb_haunted.json`, kokoro on both slots, or the Klein + LTX 2.5 profile from Section 1.5 once it ships) -> run. Why: proves the pack works on hardware other than the dev box. DONE WHEN: `obs_publish OK` with zero hand steps. Blocked on: an Active registry version; optionally the Section 1.5 Klein + LTX 2.5 profile. Runs as Section 2, Batch R7.
* **Save the 4060 dropdown-friendly JSON.** DONE WHEN: the JSON is committed with kokoro on both voice slots. Blocked on: a passing test leg above -- a pass is what earns it.
* **README model table** from the compatibility workbook's Baseline Combos tab (`outputs/20260828-ungated-models/`, the LIVING fact sheet: edit cells in place, never add a changelog tab). DONE WHEN: the table is in README, or the operator rules that README's injected class table and per-lane facts already cover it and this bullet goes. Blocked on: that operator call.
* The knob census (Section 1 header) informs what the template pins.

### 5.3 LOCAL-LLM SWEEP LEG 0 -- the in-process preflight (queue item 6)

Leg 0 = one in-process command (`request_slot` -> ~40-token generate -> `_self_unload`
per row, with `reset_peak_memory_stats()` around each), ~15-20 min, IDLE GPU, no ComfyUI.
It fails loudly on a dead row. The four canonical legs that are the real proof, and the
whole sweep design, are Section 2 Batch R3.

### 5.4 THE H3 PROMPT-POLICY VERDICT: read the receipts first, render only if they do not answer

- **Close the h3 prompt-policy verdict** (PBUG-20260827-01 in `docs/PROD_BUG_LOG.md` is still marked "STILL OWED") -- without it the prompt-policy fix has no receipt.
  - **What to check:** in the positive video prompt, nonverbal action and camera PRESENT; the beat's exact dialogue and any speaking / lip-sync / mouth anchor ABSENT.
  - **Where:** the ledger on disk stores only `prompt_sha8` for the positives, so read the render receipt or the server log for the two post-fix `minimax_h3_video` episodes -- `signal_lost_the_poise_of_stone_20260827_143538` and `signal_lost_reel_of_resistance_20260828_121427`.
  - **DONE WHEN:** clean receipts close the row. Render a fresh leg ONLY if neither episode's receipt answers it:
    ```powershell
    cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 -Profile otr_w45_minimax_h3_video -Acts 1
    ```
  - Engine routing is a PROFILE, never `-Set` (`patch_creative` whitelists creative widgets only; writers may ride as `-Set`).
  - The profile carries the h3 boot contract (`--reserve-vram 12`, `--disable-pinned-memory`).
  - A FRESH episode id is mandatory because `request_hash` excludes prompt bytes, so a cached SPEAKING clip would be a false pass.
  - The BEFORE sample `signal_lost_the_caretakers_clause_20260826_155835` (every beat on H3, pre-fix, under `output/otr/episodes/`) stays untouched: it is half of the A/B.

### 5.5 ONE image-mode sweep on `otr_soak_llmsweep_02` settles SCENE + PORTRAIT `elements: []`

**This is a RENDER/MEASUREMENT item, not a panel item. No arc runs before the numbers exist.**

* **DO:** re-run profile `otr_soak_llmsweep_02` in image mode via `scripts/otr_bank_engine_sweep.py` (it walks every bank against both engine profiles) against current HEAD, and read whether SCENE and PORTRAIT beats refuse at all now.
* **WHY:** the SCENE/PORTRAIT `elements: []` fork is currently an inference; the sweep replaces it with numbers.
* **DONE WHEN:** the sweep has run and the refusal question is answered either way.
  * No refusals -> this row collapses to a documentation correction and costs nothing further (the likelier outcome).
  * Still refusing -> the fork is real and gets the FULL four-round arc, now with numbers, between:
    * **(a)** derive a subject noun from the prose -- **already written off** in `ideogram4_local.py` (`_wrapped_caption`: the five-layer string is a convention, not a grammar), and a wrong noun INVENTS CONTENT, which the source-fidelity rule forbids;
    * **(b)** a metadata channel that hands the lens a real subject -- more wiring, but the anchor is derived rather than guessed.
* **CONTEXT:** `docs/2026-08-26-ideogram-music-card-PROBLEM-STATEMENT.md` (5080-local) and PBUG-20260826-01.
* **BLOCKED ON:** nothing -- needs GPU time for the sweep.

### 5.6 FOUR canonical legs prove the WHOLE local-LLM acceptance sweep (operator directive 2026-08-25)

Prove all 7 surviving local-LLM rows in BOTH model slots (creative + technical) plus the gemma `Q8_0` negative probe. Runs after Leg 0 (Section 1.7). All 7 rows are on disk; preflight guide: `docs/LLM_PREFLIGHT_GUIDE.md`; charter: `docs/OTR_STANDING_RULINGS.md`.

**RULINGS (verbatim, do not soften):**
- Charter: `docs/OTR_STANDING_RULINGS.md` "ONLY EASY-TO-LOAD LLMs SHIP" -- every surviving row does creative AND technical, or it is ripped on a MEASURED failure, never on assumption.
- A row that cannot do both, and was never tested or implemented, is a RIP candidate under his rule -- but rip only on a measured failure, never on assumption.
- A leg that never reaches `otr/obs/` did not pass, so the 4 canonical legs are the real proof.

**OPEN WORK:**
- **Leg 0 -- in-process preflight.** No ComfyUI: `request_slot` -> ~40-token generate -> `_self_unload` per row, with `reset_peak_memory_stats()` around each. One command, ~15-20 min. Fails loudly on a dead row. DONE WHEN: every row generates and unloads cleanly, or a failure is recorded as measured.
- **4 canonical legs (MINIMUM) -- 7 rows / 2 slots.** Each leg doubles as a ledger-cleanup live-watch and an eyeball re-observation chance. DONE WHEN: all 4 legs publish to `otr/obs/`. Blocked on Leg 0.
- **Every leg must PIN `--source-bank` to the scifi lane.** Canonical ships `'roll (any eligible bank)'`, and `_otr_scifi_news_pro.py` is the only runner code-verified to drive BOTH slots. Unpinned, a leg can land on a lane that never touches the technical slot and the sweep proves nothing about that row.
- **`gguf_quant` is ONE per-run widget**, and `unsloth/Qwen3-8B-GGUF` ships only `Q4_K_M` -- so any leg carrying it runs Q4_K_M.
- **gemma GGUF negative probe.** `Q8_0` / `n_ctx=4096` needs ~14.70 GiB FREE against a 15.92 GiB card with ComfyUI resident; `_otr_gguf_backend.py` compares against `mem_get_info()` FREE with "NO silent context downgrade". Either outcome is informative; record both. DONE WHEN: the outcome is recorded.

**KNOWN FALSE-GREEN TO DESIGN AROUND:** `meta.slot_calls_by_slot` is incremented ONLY inside `_SlotScheduler._account_and_get_entry`; SIX `request_slot` sites live outside it (`story_orchestrator.py`, `otr_shot_lock.py`, `OTR_LedgerFreezeCascade.py`, the SlotContract path in `OTR_LedgerScriptWriter.py`, the registered `nodes/vram_context_test.py`). The counter proves IN-WRITER generation only; reading it as full-row exercise is a false green. Do not use it as the acceptance signal.

**Reference:** coverage matrix, per-row assertions, skip-reporting rules and risks live in the 2026-08-25 workflow result; re-derive from this row if it is lost.

### 5.7 ONE canonical `fastwan_8gb` leg with 60-second opening AND closing cues proves PBUG-20260811-02

The only row of the 2026-08-25 re-triage still open. Root cause established, the repair is
WRITTEN, and it is not a coding item: it needs a canonical `fastwan_8gb` leg whose cues are
long enough to chunk at `_MUSIC_MAX_CHUNK_DUR_S = 22.0`. A render window, not a coder
slot. Detail: `docs/PROD_BUG_LOG.md`; the closed trio is in the archive.

### 5.8 OPPORTUNISTIC: the cfg 1.0 promotion cells, D2 fail-hunting still legs, and the eyeball re-observations

Run when a render window is free and nothing above needs the box.

* **cfg 1.0 promotion A/B** (Section 1.5 item 4) -- confirms cfg 1.0 on real content or sends it back. Four cells x three real episode prompts, into a NEW dir under `docs/ship-audit-2026-09-01/image-jury/`. DONE WHEN: the operator's eyeball has ruled. BLOCKED ON: a free render window + his eyeball.
* **D2 -- hunt the still failure** so D3 can fix that branch at its root and `docs/PROD_BUG_LOG.md` gets a mechanism, not a guess. Reset per CLAUDE.md section 4, boot headless, run 320-word `public_domain` or `shakespeare` still legs until one fails (~1 in 6). DONE WHEN: a fail-closed leg writes the compact JSON `MISSING_TARGET` record in the SERVER log (arm, token, index, canonical `prompt_hash`, repr-escaped excerpt); a publish is just a clean leg, keep going.
  * RULING: Do NOT weaken the completion gate, revive the portrait-init fallback, or rebuild the withdrawn "give the collapse guard a still owner" fix (the 08-04 postmortem disproved that chain).
  * Receipts: `docs/2026-08-04-POSTMORTEM-still-unmaterialized-320w.md`, `docs/2026-08-04-D1-SHIPPED-still-skip-evidence.md`.
* **Two eyeball items ride ANY real render leg, at zero extra legs** -- no coder time, just look: the announcer framing defect (`docs/2026-07-11-announcer-framing-defect.md`: episodes START a story instead of admitting you into one) and name-splice defect #2. Both predate THE LAW and have no reproduction at HEAD. DONE WHEN: each is either re-admitted as a FRESH dated row with that leg as evidence, or tombstoned.
  * RULING: the framing fix stays seam + score contract + fail-closed validator, never Python authorship.

### 5.9 RETIRE OR RE-DERIVE the seven 45-word engine proofs

Cross-check the seven public engine IDs of the archived 2026-08-13 runway table (row 3;
its pointer was recorded BROKEN on 2026-08-16) against `config/machine_classes.json`
`engine_evidence` (PROVEN rows dated 08-23 .. 09-01). Render a 45-word proof (`COVERS`,
`RESULT SUCCESS`, `obs_publish OK`, the file on disk) ONLY for an engine with no
post-08-13 receipt; otherwise retire the row. Do not spend seven legs on an unverified
list.

### 5.10 THE 4060 CLEAN ROOM: the legs still owed

Context to act: clean room is `C:\OTR-CleanRoom` on the 4060 (portable v0.34.0, Python 3.13, OTR clone at `da2b7a36`, ComfyUI-GGUF pinned + patched, pinned weights, bark voices). Server and legs start through Task Scheduler, never from an SSH session. The clean-room profiles are untracked stand-ins on the 4060; the shipped equivalent is Section 1.5 item 2. Friction log and leg receipts: `docs/ship-audit-2026-09-01/4060_CLEANROOM.md`.

* **Leg B -- OPEN:** run `otr_cleanroom_8gb_humo17` (HuMo 1.7B, 13.6 GB of Comfy-Org weights, no extra node pack) -- the faster 8 GB video candidate if LTX 2.5's ~14 min a clip is too slow to ship as the 8 GB video default. Pull the clone to HEAD first (later commits are docs and the provisioner). DONE WHEN: leg publishes (`RESULT SUCCESS` + `obs_publish OK` + the file) and per-clip time is recorded.
* **Z-Image on 8 GB -- OPEN:** run one stock-flag `z_image_turbo` leg on a build post-`da2b7a36`; the R1 abort (`Fatal Python error: Aborted` at sampler step 5/8 under DynamicVRAM) was never re-tested after the residency fixes, which the clean-room doc calls only its "likely root". DONE WHEN: it publishes, or -- if it still aborts -- the `--disable-dynamic-vram --lowvram` pair is documented in README's 8 GB row for that dropdown choice (README carries no such text today) and the faulthandler report is filed with ComfyUI.
* **Registry-install template test** (Section 1.6): runs here, BLOCKED until a published registry version is Active.
* Record ONLY what publishes (`RESULT SUCCESS` + `obs_publish OK` + the file) into `config/machine_classes.json`, regenerate `docs/MACHINE_MATRIX.md`. Do not advertise a lane the clean room did not finish.

Rulings -- do not re-open:
* Leg A's question is answered by Leg C5 (LTX 2.5 works on 8 GB, not a daily driver); Leg C6 (fp8 encoder) is not needed.
* The shipped 8 GB set runs Klein by ruling and is unaffected by the Z-Image outcome either way.
* Operator eyeball on Leg C5 (Klein + LTX 2.5, stock flags) is tracked as Section 1.5 item 1.

### 5.11 THE NEXT POD SESSION: the acid test and a looped lane sweep on one rental (queue item 9)

- **Owner:** Codex, who also owns `docs/RUNPOD_INSTALL.md`. Pod stays STOPPED until this batch runs; the volume stays (it holds the warm cache, the expensive thing to recreate).
- **Run both legs on ONE rental, in this order:** saved template -> `scripts/otr_provision.py` -> acid test -> looped lane sweep.
- **Leg 1 -- the acid test (why: it is what the DOER was built for):** one published episode with ZERO hand steps. Stable Audio 3, index-tts in its own `uv` venv at `INDEXTTS2_PYTHON = "3.10"`, and the reference WAVs must ALL be installed by the provisioner, none by hand. DONE WHEN: an episode publishes with no manual install step.
- **Leg 2 -- the looped lane sweep:** `scripts/otr_pod_lane_soak.sh` + `scripts/otr_pod_overnight_sweep.sh` for the HuMo 14B / LTX 2.5 / indextts2 second-machine receipts. DONE WHEN: receipts are filed under the 16gb class in `config/machine_classes.json` `engine_evidence`.
- RULING: there is no 24 GB class or profile by ruling (Section 3 L asks whether a row is wanted).

### 5.12 Deferred render items (each blocked, or waiting on something else first)

- **Capped-14B HuMo leg** -- live proof of the ping-pong lip-sync reversal fix
  (`a1d810f1`); see the coverage cluster row in Section 1.4.
- **`scifi_news` live reverify** (PBUGs 20260712-22/23/24/25, fixed in tree) --
  blocked by the `scifi_news` P0 convergence defect (Section 1.4), then fan-out.
- **The WAN physical 8 GB proof** -- a render on a PHYSICAL 8 GB card is still owed;
  behind Section 3 question D, detail there.

---

## 6. RULINGS OWED BY THE OPERATOR -- each with its default if unruled

Skipped by every coder window until he rules. A window that guesses one of these is
doing work that may be thrown away.

### How to read these rulings

One operator pass clears every row here. Nothing in the queue's first five items waits on
this section.

### J. THE REGISTRY -- the control experiment, only if alpha.15 flags

* **Blocked on:** the alpha.15 push (queue item 1). Do nothing here until its scan verdict lands.
* **If alpha.15 comes back Active:** nothing to do; close this row.
* **If alpha.15 flags -- run the control:** republish the alpha.8 tree byte-identical as alpha.16. DONE WHEN a verdict exists: Active means the trigger lives in the alpha.9+ delta and can be bisected; Flagged means the ruleset moved, and that result is the evidence to hand Comfy-Org.
* **Version sequencing:** alpha.15 = the marker patch; alpha.16 = the control, only if needed; the kokoro-onnx dependencies (Section 1.1) ride the NEXT bump after both.

Never version-delete (a soft delete burns the string).
Never bump a version while another version is Pending.

### The question list

Open operator decisions. Each row: the call to make, why it matters, DONE WHEN, blockers. No row is coded until ruled.

* **(A) Arm `defaults.scene_coherence_check` on any bank?** The vacuity-fix code is live with zero callers. If yes: measure OFFLINE over the published corpus first, then arm in ONE change (no-render work once ruled). DONE WHEN: a ruling exists and, if armed, exactly one bank change lands.
  Default if unruled: nothing arms it.
* **(B) Insert a `full_text` HTML block-join separator?** It is the dominant P0 failure cause on live evidence, but separators widen the coordinate system `source_digest` pins, so the change belongs in the SOURCE ADAPTER, not downstream. Detail: first row of Section 1.4's P0 cluster. DONE WHEN: ruled, and if yes the adapter change lands with `source_digest` re-baselined.
  Default if unruled: not touched.
* **(C) Ghost-name reconciliation fork:** scrub briefs after cast lock, OR propagate pitch names. Pick one. Detail: Section 1.12, item 3. DONE WHEN: one branch is chosen and implemented.
* **(D) Who owns a tier's native render ceiling now that profiles are lab presets and the machine matrix is the stranger-facing channel?** Candidates: a `video.max_render_frames` field on the class row, OR the adapter's own capability row with the widget as an override (0 = adapter contract). BLOCKS the WAN 8 GB proof and the A2 echo fix. DONE WHEN: one owner is named and the ceiling reads from it. Detail block below.
* **(E) Three works refuse to vendor** (`ghost_ship` gid 11045, `purple_cloud` 11229, `beleaguered_city` 11521 -- `scripts/otr_vendor_public_domain_library.py` against its parser). DONE WHEN: the three vendor cleanly. Blocked on one Gutenberg fetch.
  **need one Gutenberg fetch, so it is operator-opt-in only** -- not schedulable inside an offline sprint.
* **(F) Bible fan-out batch.** One operator pass clears: every row marked "awaiting fan-out" in Section 5, the PBUG-20260710-07 retirement ratification, the duplicate-id cleanup, and the PBUG-20260901-04 promotion. DONE WHEN: no "awaiting fan-out" rows remain in Section 5. Blocked on the operator pass.
* **(G) Name the first H3 video-path sprint** -- standing context below. DONE WHEN: a named sprint exists.
* **(H) Keep the current research_only behaviour?** A research_only source withholds the OBS copy rather than killing the finished render. Say so if the old kill-the-render behaviour is wanted back (a one-line revert).
  Default if unruled: keep.
* **(I) Does `media_archive` want the catalog premise at all**, or the same scaffold-off treatment as `original`? Second specimen of the content-blind-draw class; the scaffold-off rule so far was stated only for `original`. DONE WHEN: ruled, and if scaffold-off then `media_archive` matches `original`.
* **(J) `style_tail_policy` needs a third token, or the `ltx_radio_face` path is EXEMPT.** `build_radio_host_prompt`'s `ltx_radio_mouth` branch (`otr_meta_brief_image_prompt.py`) returns early with `"%s, warm dramatic lighting"`, skipping `finish_visual_prompt` and the `image_grade_tail` append required by the 2026-07-02 look direction, while the `ltx_audio_in` bookend row declares `style_tail_policy="full"`. DONE WHEN: either a third token exists or the exemption is written down.
  Default if unruled: the exemption, because it changes no behaviour.
* **(K) `check_compatibility`: ratify the inert constant, or schedule the rip?** The name reserves nothing (`tests/test_otr_check_cli.py` activates a bundle whose value is a plain integer); rip blast radius ~5 code sites, 2 test files, 3 docs. DONE WHEN: ruled, and either the rip lands or the doctrine line is in `EXTENDING_OTR.md`.
  Default if unruled: leave it, and add the doctrine line to `EXTENDING_OTR.md`: a name published to clients before its consumer exists is "reserved, no contract, ignored if defined" and lives in no executable code.
* **(L) Does a `24gb` class row exist at all?** The matrix treats 16 GB+ as the top tier by design, and LTX 2.5 at 1664x960 OOMed on a rented 24 GB 4090 (`docs/RUNPOD_INSTALL.md`). DONE WHEN: ruled.
  Default if unruled: no row; rental receipts file under the 16gb class's `engine_evidence`.

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

- **D1 -- RATIFY the WAN 8 GB frame-ceiling ownership before any code.** Blocks D2/D3 and A2.
  - Problem: `otr_canonical.json` node 87 ships `max_render_frames=0`, so a plain canonical WAN run is UNPINNED and inherits `_TI2V_MAX_FRAMES = 177` -- the 2026-07-23 failure shape. The 17-frame ceiling reaches a leg only via a variant workflow or a hand-set widget.
  - **RULING: Pinning 17 in the canonical is WRONG (it would cap the 16 GB LTX / HuMo legs).**
  - Proposed shape to ratify: `eng_wan_ti2v` DECLARES its own tier ceiling as a capability-row field; the widget becomes an operator OVERRIDE with 0 meaning "use the adapter's contract"; the profile channel stops mattering.
  - **RULING: a real design change with a live blast radius on any card with headroom, so ratify before code.**
  - DONE WHEN: the operator has ratified an ownership shape.
- **D2 -- Prove the WAN 8 GB ceiling on a PHYSICAL 8 GB card.** Blocked on D1.
  - A 16 GB card told to reserve 8 GiB is not the same claim; the 18-engine campaign is coverage, not an 8 GB qualification.
  - DONE WHEN: a render completes on real 8 GB hardware.
- **D3 -- Close the untested WAN tier-vs-multiclip edge (cheap; do when D1 reopens).**
  - WAN is out of `PLANNING_CAP_ENGINES`, so a tier ceiling and a multi-clip plan CAN contradict by design and `_planned_length` hard-refuses mid-episode.
  - DONE WHEN: a test asserts a 17-frame tier survives a multi-segment beat.
- **A2 -- the applied-overrides echo hides the profile's `llm.*` override.** HELD on question (D), because its whole subject is the profile channel.
  - `nodes/_otr_workflow_apply.py` already flattens `llm`; `scripts/otr_api.py` echoes only role / slot / features plus two seed keys, so a run reports "16 overrides" while also having replaced the entire LLM configuration.
  - Fix: generate the echo FROM the applier's flattened map (never add keys by hand).
  - DONE WHEN: the echo is derived from the flattened map and reports the `llm.*` keys.
- **The `ltx_8gb` render-length ceiling has TWO owners that only agree by coincidence.**
  - The coverage PLANNER reads `config/profiles/otr_8gb_ltx.json` `video.max_render_frames`, and `ltx_8gb` is the sole member of `PLANNING_CAP_ENGINES`. The ADAPTER's own pre-render refusal reads `OTR_LTX_8GB_MAX_FRAMES`. Both land on 161 today (profile unpinned, env unset), so nothing breaks. But `workflows/variants/otr_8gb_ltx.env.json` ships `OTR_LTX_8GB_MAX_FRAMES=97` and NOTHING currently reads that file. The day a launcher honours it without also pinning the profile, the planner emits a 98-161 frame segment and the adapter refuses it MID-EPISODE -- after the stills are minted and, on a multi-segment beat, after the 6.34 GiB checkpoint is hoisted.
  - **RULING -- Deliberately NOT fixed in B6:** pinning the profile to 97 changes how a 237-frame beat partitions, which is a production planning decision, not a cleanup.
  - **The preset carries a `_ceiling_note` saying do not export it alone.**
  - DONE WHEN: any launcher that starts honouring `otr_8gb_ltx.env.json` pins the profile in the same change.

---

## 7. PARKED / DEFERRED -- out of the working queue, kept for the ruling each carries

Nothing here is scheduled. Each row is here because a ruling parked it, and the ruling
is the reason it must not be quietly re-opened.

### The parked rows

### MAC / AMD -- images only, later (deferred; needs its own design row when picked up)

Operator: Mac and AMD ship images only (ruling 2026-09-01), and he is "not hopeful".
Landed: the credits font, the llama-cpp hint and four platform guards. Owed, in order:
(1) one measured Klein render on Apple Silicon -> `nodes/_otr_image_engines/registry.py`
gains `mps` on the `flux2_klein` row (cuda-only today) -> `otr_mac_mps` flips off
`google_image` (README's Mac row says so); (2) the upscale stage accepting `mps`
(`_otr_upscale_engines/__init__.py`, deliberately deferred); (3) a measured ROCm boot for
`otr_amd8_rocm` / `otr_amd16_rocm`. ROCm already qualifies for Klein (presents as cuda).
Needs hardware neither NVIDIA box has.

### PARKED (operator ruling 2026-08-12): wire character casting to the VOICE REFERENCE BANK

**Status: PARKED, not rejected** (operator: *"park it on go forward"*).
Trigger to take it up: "a cloning engine is SELECTED" -- NOT "the shipped default" (Section 1.1 makes kokoro the shipped default).
Background/measured table: `docs/GO_FORWARD_ARCHIVE.md`.

Why it matters: the writer casts from 10 Bark presets (`config/cast_pools.py`) while cloning engines draw from the 204-entry `config/voice_reference_bank.json`.
Constraint: `voice_preset` / `tts_model` are ledger JOIN KEYS (`cast[].name` / `char_id` / `voice_preset` / `voice_ref_id` / `voice_engine`, joined from `lines[].speaker` and `beats[].char_id`) -- enumerate every field's owner BEFORE the menu moves (the ledger law). Raising `MAX_SPEAKING_CAST` alone does nothing: `_deal_voice_menu` builds the menu from `VOICE_PROFILES`.

- [ ] Enumerate every consumer of a cast row's voice fields (casting, TTS dispatch, per-beat audio slicing, credits, portraits, captions, `obs_publish`) and name exactly one new owner per field.
- [ ] Make the casting menu engine-aware: Bark presets for Bark, reference-bank entries for a cloning engine (gender and `commercial_clean` already exist on bank rows).
- [ ] Replace `_assert_unique_bark_voices` with an engine-agnostic one-voice-per-character invariant -- the rule must survive: two characters sharing a voice is a correctness defect.
- [ ] Derive `MAX_SPEAKING_CAST` from the ACTIVE engine's pool instead of a constant; `tests/test_cast_size_is_a_request.py` will report the drift.
- DONE WHEN: proven on `scifi_news_pro` (the only bank on the fable2 writer) with a cast larger than 10 and complete speaker-to-`char_id` equality in the ledger.

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

**Not work yet, and explicitly not before v2 ships.**
Operator: *"once we ship v2 we ship an OTR-Lite, similar architecture but only the most frictionless auto-download non-gated models, and maybe just maybe we can figure an ffmpeg-less solution for a truly streamlined workflow."*

* **BLOCKED ON v2 SHIPPING.** Do not open any of the rows below until then.
* **First question, and it is where this starts: captions + credits without libass/drawtext.** This PyAV build has neither, so caption burn and credits cannot go in-process as-is; the mux, silent composite and probe already have PyAV routes. DONE WHEN a caption/credits render path exists that needs no ffmpeg binary. Not the mux -- that is the part that looks hard and is not.
* **ffmpeg-less encode path.** Removes the binary dependency and shrinks the largest non-`os.environ` Comfy Registry finding class (`python_command_injection_risk` from ffmpeg/ffprobe `subprocess` calls) in the same work. PyAV (`av>=16.0.0`) is already a ComfyUI CORE dependency, so it ships everywhere. DONE WHEN no `subprocess` call to ffmpeg/ffprobe remains on the Lite path.
* **Non-gated auto-download lane set.** `scripts/otr_fetch_lane_weights.py` offers UNGATED sources by design and deliberately refuses to paper over the one gated repo (Lightricks/LTX-2.5), so its lane list is the candidate set. DONE WHEN a Lite lane list is pinned to ungated sources only.

**Evidence to read first:** `docs/RUNPOD_INSTALL.md` section 7A; `docs/2026-09-03-registry-review-request-READY.md`.

### 4.Y BRING `word_razzle` HOME -- it is the one cloud lane whose NAME hides it (operator, 2026-09-03)

**Operator:** *"word_razzle shouldn't be cloud anymore"*, and the reason behind it:
*"I don't want to mislead my audience -- make cloudy lanes transparent."*

- **Problem.** 8 of 30 registered video engines render provider-side; seven self-label with a `cloud_`/`google_` prefix (`cloud_kling_avatar`, `cloud_seedance_2`, `cloud_wan_i2v`, `cloud_wan_i2v_audio`, `cloud_vidu_q2_pro_fast_720p`, `google_omni_video`, `google_veo_video`). `word_razzle` does not -- it reads like a local text-effect lane and phones out. DONE WHEN: no registered engine hides that it is remote.
- **Design question, decide FIRST: engine or profile?** word_razzle is "animate a word-card still into a living period poster" -- init image + prompt + seed + duration + motion_mode. Both halves now exist locally: `ideogram4_local` for the card (the pack's spelling champion; it did not exist when word_razzle was built cloud on 2026-07-03, which is why that lane is cloud), and `still_motion` / `still_pan` / `ltx_video` for the motion. So it may be only a PROFILE -- `character_image: ideogram4_local` + an existing local i2v in `character_visual`. A profile costs nothing; a new engine id trips five generated fixtures, two literal rosters, the terminal-frame proof rule and `docs/VIDEO_LANE_PREFLIGHT.md` gates 1-8. DONE WHEN: the arc has picked one and the price is accepted.
- **Blocked on:** a kibitz arc on that question, owed before any code (this is a design item, not a drive-by).
- **Also decide in the same arc:** what happens to the cloud `word_razzle` row -- retire it, or rename it `cloud_word_razzle`. Renaming alone fixes the transparency defect even if the local lane never lands. DONE WHEN: the roster is honest either way.
- **PARKED -- log it, do not build it:** a RECIPE identity for remote lanes (model id + resolution + params) so a mid-beat env flip is caught for a cloud lane the way weight drift is caught for a local one. *Nobody has asked for long cloud beats; log it, do not build it.*

**DEFERRED BY THE OPERATOR (2026-09-02), not scheduled:**
* **Token rotation** -- an earlier temporary diagnostic briefly captured inherited HF / OpenRouter / provider credentials; the file was removed and the leak path hardened. Rotate when convenient; nothing in the queue waits on it.
* The Section 3 question list (A)-(L), each with its default if unruled.

## 8. THE BUG BIBLE FIELD AND THE OPEN RISKS

### Bug Bible promotion field -- pending actions only

| Record | Pending action | DONE WHEN | Blocked on |
|---|---|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify, then fan-out | live artifact reverifies all four | the `scifi_news` P0 convergence defect |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Bible fan-out: overlap check + operator approval | Bible rows exist + `otr_coverage_index.yaml` updated | next operator Bible fan-out (Section 3, question F) |
| `PBUG-20260713-19` | Live requalification (promoted BUG-05.11) | a live leg requalifies it | -- |
| duplicate-id cleanup | Set BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) | each Bible row cites one distinct legacy_id | same fan-out |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out | retirement recorded in the Bible/index | next fan-out |
| Seedance softener mangles authored prompts (2026-08-17) | CANDIDATE only. Detail: `docs/GO_FORWARD_ARCHIVE.md` | a cloud leg produces the artifact | a cloud render this repo cannot observe |
| `PBUG-20260904-05` (draft 8 GB profiles: 2048 ctx holds `media_archive`, refuses the other two banks) | CANDIDATE: a live refusal | verify condition is automatable | a profile-design call not yet made |
| `PBUG-20260901-04` (kokoro on Python 3.13) | Bible CANDIDATE (a Requires-Python marker rule) | promoted at the fan-out | fan-out (Section 3, question F) |

Rulings -- do not re-open:
- historical `PBUG-20260711-18`: Keep as a standing context/cap engineering risk; never eligible from static evidence.
- Seedance softener: fixed pack-side, but it conditions a CLOUD render this repo cannot observe, so it fails the admission rule. Promote only if a cloud leg ever produces the artifact; nearest coverage `12.108` does not cover blind-regex rewriting of authored text.
- The active production-fix owner updates `docs/PROD_BUG_LOG.md`; promotion to the Bible is tracked in the Bible repo's `otr_coverage_index.yaml` (CLAUDE.md, delta-scrape discipline); no plan review or invented fixture creates a row.

### Open risks

- **First live client-bank leg is unproven end to end** (fetch -> interpret -> writer -> cleanup -> tail -> publish); no client bank has ever run live, so a wrong bundle path ships silently. DONE WHEN: one client bundle completes a canonical leg and publishes to `otr/obs/`. Treat that first leg as a qualification, not a formality.
  - RULING: Deferred power-user tiers (client own-runner + staging, dependency manifest, standalone story_rules) are explicitly OUT of v1 and are a NEW block if the operator ever wants them.
- **Client-authored Python executes in-process — keep the wave-3 posture in every future change**, or a client bundle gets ledger or registry reach it must never have.
  - RULING (verbatim): `--activate` is the consent act; the seam fails LOUD (`UserBankExecutionError`) and never substitutes; client code never touches the canonical ledger; owner IDENTITY is verified so a bank can only run its OWN bundle; the shipped fetcher/interpreter registries are never widened to admit a client id. Do not relax any of these for convenience.
- **Activation-path text must change in three places at once** — `nodes/story_packs/banks.json` (`custom_source_bank` row's `guide_ref`, raised by `require_runnable_bank`), the `source_bank` tooltip, and `docs/EXTENDING_OTR.md`. Any future change to folder name, CLI verb or restart behaviour that updates fewer than all three makes the product confidently instruct clients to do the wrong thing. DONE WHEN: the three agree on the shipped activation steps.
- **Two changed shipped-lane behaviours still need a live receipt** from the ledger-cleanup tail (it runs on EVERY bank): (a) unsafe spoken language on a `content_owned_readonly` bank is repaired at the writer tail rather than reaching G9, so a leg that used to die at freeze may ship a sanitized line; (b) a blank `meta.episode_title` is filled at the tail rather than exploding in `otr_credits_roll`. DONE WHEN: a live leg shows each path taken and published.
- RULING: No code lands mid-sweep of an active qualification campaign (the 420-rung uniform-code-confound lesson).
- RULING: There is no standalone SFX provider layer to rebuild. Current video clips are silent and the terminal mux uses the frozen upstream master audio. The future direction in `ROADMAP.md` is to retain and mix selected video-generation audio as inexpensive ambience; do not revive the fast-moving provider/bed stack or claim that future path is already wired.
- **Lean-mean: work the one ordered campaign in `docs/LEAN_MEAN_CLEANUP.md`.**
  - RULING: The retired FRONT/TAIL and SW-1 execution model must not be revived.

### After all of the above

One owner per file (CLAUDE.md section 1); every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

When the sections above are exhausted, continue with `ROADMAP.md`: lean-mean ->
RunPod/AMD/Mac -> install -> product docs/v2 release. That is a pointer, not work that
precedes lean-mean. Lean-mean is not an item in this queue: `docs/LEAN_MEAN_CLEANUP.md`
is its sole current scope, blast-radius, coding-order, and verification authority.

**WATCH -- recorded, not scheduled:**
* **`obs_publish OK` is not proof of an episode.** Measure the published file's DURATION at every pipeline stage (render / blend / caption / credits / mux), never the log. A stage that changes the duration is the defect. Two 7.5 MB casualties are still in `otr/obs/` and are deliberately not swept -- `..._231401` and `..._233738`, both "The Faded Ledger".
* **An unrelated rotation loop owns the box's spare cycles.** `video_rotation.sh` (session `8a385813`) cycles 16 engine/image lane combos one act at a time, forever, skipping AnimateDiff. It is Jeffrey's daily obs proof; leave it running. It does not block a registry publish and holds the resident server on :8000, which only matters for a local boot check.
* Zero-frame beat -- owed: one canonical leg of that shape reaches `otr/obs/`, recorded as a "verified live" line under PBUG-20260831-01; and confirm `shot_b006` (`mode=object source=deterministic_fallback`) was the ghost shot for those music rows, or it is a second defect. No coder time.
* Whether a PRUNED P0 index is ever accepted has not been measured live; the next P0 campaign's instrumentation answers it.
* `OTR_LedgerFreezeCascade` failed twice and the message was never captured -- the runner's eight-frame traceback truncates it. Next occurrence, read the SERVER log.
* `OTR_VideoRenderBatch` `RenderError` cluster -- triage after items 2 and 3.
* The two eyeball re-observations (announcer framing, name-splice #2) ride any real render leg -- Batch R5.
