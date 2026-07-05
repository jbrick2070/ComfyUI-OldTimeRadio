# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> Last updated 2026-07-05 (TOTAL-COVERAGE COMPLETE -- CHUNK C SHIPPED) | branch v2.0-alpha HEAD 3eced145 + docs, ALL PUSHED (HEAD==origin) | prod/main + tags operator-GATED.
> CHUNK C @3eced145: still_word typography/backdrop/title-mood are pack-owned now (composer + derive provenance read the resolved VisualStyle; genre SELECTOR + per-episode lettering LOCK stay Python -- operator 2026-07-04); sci_fi byte-identical (extraction constants survive as fixtures + AST-guarded out of production); 4 packs authored their still_word voices (all 3 fields changed from sci_fi). test_visual_styles_c.py (+59 tests: deltas, sci_fi byte-identity, provenance stamps still_word_typography:<genre>/still_word_title_mood_style, lettering-lock determinism, negative-vocab, AST re-route guard); A1 TestDormantDefaults RETIRED, A2 provenance pin + still_word signature pin re-pointed. Suite 6614/0, Bug Bible 16, workflow JSON untouched. ALL FOUR chunks (A1+A2+B+C) done -> visual_style TOTAL COVERAGE is CODE-COMPLETE. NEXT = operator acceptance: an ANIME episode's announcer/radio + music stills AND still_word cards provably carry the anime fields (ledger prompt-metadata stamps); operator eyeball gates the look. Documented limitation unchanged: procedural viz_* engines are promptless -> full coverage needs promptable engines selected for announcer/music video roles.
> DONE + PROVEN LIVE this session: (1) credits col-3 scroll @b89a30ca; (2) ElevenLabs cast-voice @fae7081f
> (voices render on elevenlabs, real provider ids); (3) Sonilo music @8f146394 (min-duration floor+trim, episode
> completed to obs). cv1 word_razzle x ideo = obs PASS. NEXT: cv2 kling_avatar FAIL root-cause; image-engine
> sweep (recraft/flux_pro/nano_banana_2/seedream_2/ideogram); 800w all-visualizer credits run. Live cloud legs
> use scripts/_otr_cloud_audio_babysit.py (direct submit; voice_bank/node-83 music engine via patch_widget_by_name).
> SHIPPED this session: credits col-3 scroll @b89a30ca (loop/fps-time still inputs); ElevenLabs cast-voice
> POOL @fae7081f (21 premade voices w/ provider_voice_id -> CastLock stamps gender-matched ids under the
> elevenlabs_cloud bank; verified MARGOT->Laura / DOLPH->Charlie; suite 6146/0). IN-FLIGHT: cv1 word_razzle
> x ideo 30w (ideo stills minted OK/auth OK; word_razzle->cloud_pixverse_i2v per shot, healthy). IMMEDIATE
> NEXT (gated on the server freeing): LIVE ElevenLabs VOICE leg (bank=elevenlabs_cloud, char_voice=elevenlabs)
> + Sonilo MUSIC leg -- Sonilo 422 likely a min-duration floor vs 12/8/4s cues (adapter floor+trim per
> docs/2026-07-04-elevenlabs-cast-voice/PLAN.md Part B). THEN image-engine sweep + 800w all-visualizer credits runs.
> ACTIVE (operator overnight mission): root-cause+fix all Comfy Cloud engine hard-fails + the scrolling-
> script credits bug, babysit-iterate to GREEN on 30w full-pipeline episodes; finals in otr/obs. NO
> fallbacks, no hidden promotion; root-cause fixes; /kibitz (codex) if hung up; box reset per headless run.
> CREDITS COL-3 = DONE @b89a30ca: base/scroll stills were single-frame `-i png` -> scroll crop y-expr froze
> at t=0 (blank top pad); fixed with `-loop 1 -framerate <fps>` so `t` advances. Suite 6143/0 + Bug Bible 16.
> NEXT = (a) live 800w all-local all-visualizer render to confirm the roll on a real otr/obs final; (b) the
> cloud-engine e2e sweep (ideo image -> others; word_razzle/kling/seedance video; ElevenLabs voice; Sonilo
> music) with OTR_COMFY_API_KEY per-command. Sonilo NOTE (proven this session): a headless port / the MCP
> /prompt get 401 on cloud calls (no login injected) -- only the Desktop UI or OTR_COMFY_API_KEY authenticates;
> the operator's live Sonilo 422 is a request rejection (not auth), likely a duration floor vs the 12/8/4s cues.
> ACTIVE (operator, overnight): CODE EVERYTHING FIRST (still_word + word_razzle + pending build
> items) -> regress -> push, THEN the 30-45w model-matrix SOAK -- see section 1.
> still_word SHIPPED @ 097f44ad. word_razzle SHIPPED: Phase 0 audit @ 3843bbd0
> (CANDIDATE_FOUND, 36 rows) + Phase 1 engine @ c914321e (Pixverse cloud i2v, dark/selectable).
> Suite 6142/0, Bug Bible 16, B7. CODE PHASE DONE. (4) SOAK LAUNCHED + RUNNING DETACHED:
> scripts/_otr_night_matrix_soak.py (10 newest-model legs, force-map video + flux2 image +
> indextts2/bark, 35w). m01 ltx_audio_in verified rendering. MONITOR:
> scripts/_otr_soak_capstone_results/nightmatrix_20260703_012410/results.jsonl + otr/obs.
> NOTE: OTR_COMFY_API_KEY UNSET -> cloud legs (m10 word_razzle) FAIL LOUD at invoke (expected);
> the live word_razzle spike + soak cloud legs await the key.
> SPRINT A + Sprint B S1 stills core + `ideo` SHIPPED.
> KIBITZ ARC CONVERGED on the remaining-sprints plan (r2/r3/r4 judged; BUILD-READY).
> soak2 QA PASS (6/6 clips, obs final, no breach). proof9d 832x448 FAILED on a CLEAN
> baseline -- MARGINAL breach 14506 > 14500 MB at shot_b002 (6MB over; zero headroom at
> this canvas). S4x GO/NO-GO = OPERATOR DECISION (options in PROOF9_VERDICT.md).
> Kibitz panel change (operator): claude CLI DROPPED -- panel = codex + antigravity;
> Cowork Claude is anchor + judge.
>
> **LEAN + FORWARD-ONLY.** This doc holds the PLAN: current step, forward order, open items,
> hard rules, and POINTERS to sprint specs. It does NOT record what got done -- that lives in
> `docs/HANDOFF_LOG.md` (recent sessions) + `docs/GO_FORWARD_ARCHIVE.md` (deep history). If this
> doc starts growing a change-log of shipped work or inlining sprint detail, TRIM it: move history
> to the log, replace detail with a pointer. Keep it short.

---

## 1. CURRENT STEP

**MULTI-MODAL STORY SCHEMA -- IN-REPO BUILD (operator GO 2026-07-04). Plan of record:
`docs/multimodal-story-schema/BUILD_PLAN.md`.** Sprint-3 item-1 is now the full multi-modal vision,
built HERE (one repo), clean-break, NO fallbacks/tracebacks (breakage-in-progress is accepted; but
tests move with the schema so the suite stays GREEN per chunk). The two-repo "transplant" shape was
rejected (unanimous kibitz, HIGH -- `docs/2026-07-04-json-prompt-transplant/kibitz/arch_decision_JUDGMENT.md`);
the sibling lab's DESIGN came over (`docs/multimodal-story-schema/design-reference/` +
`schema-examples/`), its parallel package/registry/bridge scaffolding did NOT. Build into existing
`nodes/` -- NO new top-level package.

VISION: multiple STORY PATHS (public-domain / media-archive RSS / simple-4-LLM / existing sci-fi) all
filling the SAME ledger with different content+logic, PLUS a VISUAL STYLE switch (scifi/anime/origami)
that rewrites ONLY downstream visual prompts. Law: JSON owns content, Python owns validation/routing/
execution, fail-loud on unknown id.

STAGES (see BUILD_PLAN): 1 Content->JSON foundation (loader on existing nodes + extract current sci-fi
prompts to the first pack, byte-identical start) -> 2 story-path routing + packs -> 3 visual_style
schema -> 4 asserts->JSON (LAST; needs a declarative-rule enforcer node built first).

**OPERATOR DIRECTIVE 2026-07-05 (visual style = TOTAL):** when a visual_style is
selected it must impact ALL downstream prompts -- stills, video, 3D, every promptable
surface (announcer_visual + music_visual included, not just character beats). The
2026-07-05 anime episode proved the v1 tails-only slice is NOT enough (announcer/radio
stayed classic visualizer). The Stage-3 section-8 checklist (subject overrides,
portrait/scene style phrases, motion/talking registers, geometry-vs-look split) is
PROMOTED from "deferred" to an ORDERED build item, queued immediately after
lane-enablement. NOTE: procedural viz_* engines have no prompts -- full coverage also
needs promptable engines selectable/selected for announcer/music video roles.

**OPERATOR DIRECTIVE 2026-07-05 (at yoga -- CODE AUTONOMOUSLY, DO NOT WAIT):**
the operator is out; the next window must NOT stop for a go -- start coding
IMMEDIATELY and keep coding as much as possible until it genuinely can't (needs
the operator / GPU / cloud creds / a real fork). The anime ACCEPTANCE render
below is OPERATOR/GPU-gated, so DON'T block on it -- skip straight to the
CODEABLE, non-gated items and drive them in forward order: (1) the credits Fable-
gate non-blocking follow-ups N1/N2/N3 (section 1 "NON-BLOCKING follow-ups": restore
`from typing import Optional` in otr_post_upscale_procgen_blend.py; node-93 restamp
final_video_path truthfully at 86/85; strip the stale node-93 caption log strings);
(1)+(2) DONE @2c1d22aa (credits nits: Optional import, node-85 truthful final_video_path
restamp fail-soft, node-93 stale caption strings) + @7d7005c1 (conformance debt:
`_engines_by_node_key` node_key->LIST, not last-wins). (3) S-C C1 = COMPLETE: extraction
core @d60bf371 + producer FOLDED IN @4308d663 (Option B, ZERO JSON, operator "just build it
now" -- run_episode collects each shot's conditioning-WAV; OTR_VideoRenderBatch durably stamps
the per-beat audio_motion_profiles fail-soft, read-only, byte-identity held; +6 tests, 6634/0).
Option A dedicated node NOT needed (kibitz-r1 spec kept only for a future C2-on-procgen need).
(`nodes/_otr_audio_motion.py`, 8-field read-only analyzer). SUPERSEDED note: the PRODUCER WIRING was a fork that
edits the FROZEN production JSON = operator-gated -> HANDED BACK with a kibitz-r1-hardened
build-ready spec (`docs/2026-07-05-audio-motion-c1/`): Option A = new OTR_AudioMotionProfile
node inserted **91 ImageGenDispatcher -> [96] -> 92 VideoRenderBatch** (codex fixed my
ShotLock insertion error), per-video-shot rows, resolver = render_driver._slice_master_audio
(read-only), durable save_ledger_safe, NO custom IS_CHANGED v1. C2 consumers deferred ->
no urgency to touch the frozen graph while operator away. Regress + commit + PUSH per green
chunk (rule #1: always push, don't ask); update GO_FORWARD + HANDOFF_LOG each chunk;
/kibitz if torn. Only escalate for GPU/cloud/operator-eyeball work.

**CURRENT STEP = VISUAL-STYLE TOTAL-COVERAGE BUILD COMPLETE (all 4 chunks
A1+A2+B+C shipped). NEXT = OPERATOR ACCEPTANCE (not code):** render an ANIME
episode and confirm its announcer/radio + music stills AND still_word cards
provably carry the anime fields via the ledger prompt-metadata stamps
(`visual_style` + `prompt_field_source`), then eyeball the look. If the
operator wants the remaining promptless-engine gap closed, that is a config
choice (select promptable engines for the announcer/music video roles -- the
procedural viz_* engines take no prompt; documented limitation, no code owed).
Plan of record: `docs/multimodal-story-schema/STAGE3_TOTAL_COVERAGE_SUBPLAN.md`
(v5 FINAL -- kibitz r1-r4 CONVERGED 2026-07-05, codex+antigravity panel, Claude
anchor+judge; artifacts `kibitz-runs/2026-07-05-style-total-coverage/`). The r4
header block in the sub-plan is the authoritative schema inventory.

**CHUNK B SHIPPED @be42dc47 (2026-07-05, pushed HEAD==origin):** the 4
non-default packs AUTHORED in their own voice -- 10 str fields (the 1a nine +
scene_instruction_look, now non-empty on all 4) + open_subjects +
motion_registers (every key re-voiced; every authored field CHANGED from the
sci-fi defaults, r4 requirement). Voices: anime = linework/cel/speed-lines;
cartoon = rubber-hose/squash-and-stretch; paper_origami = card-stock/
crease-and-pleat; archival = restored-photograph/natural-light. +48 tests
(test_visual_styles_b.py): raw-field deltas, per-surface forced-meta deltas
(incl. motion through the REAL build_request_from_shot), per-pack
negative-vocab smokes. still_word fields stay sci-fi defaults (dormant pin
narrowed to those 3). Suite 6567/0, Bug Bible 16.

**CHUNK A1 SHIPPED @2b4a481a + CHUNK A2 SHIPPED @c265d48a (2026-07-05, pushed
HEAD==origin):** A1 = schema v2 loader (11 str + 4 dict fields, full lint set,
v1 pack fails LOUD "upgrade to v2") + ALL 5 packs v2 (sci_fi extraction
byte-identical; non-defaults dormant sci-fi defaults) + image-lane re-routes
(anchors geometry-vs-look, radio_host_style rename + pack subjects,
instruction looks, open-subject templates, resolve-once vstyle threading).
A2 = motion registers pack-routed in build_request_from_shot (exact-key, the
silent or-announcer fallback RETIRED, static env key set, probe-locked ia2v
talking prompt outranks every pack) + radio_object/plate look splits + emblem
{base} template + ADDITIVE provenance (visual_style + prompt_field_source on
the 4 scene-family image objects + request observability + trace allowlist).
Suites 6494/0 -> 6519/0, Bug Bible 16, workflow JSON untouched both chunks.

LANE-ENABLEMENT CHUNK 3 SHIPPED @d7ea448c (2026-07-05): the banks'
fetcher/interpreter ids are LIVE routing coordinates (`nodes/_otr_source_payload.py`;
science byte-identical incl. halt re-raise identity; sweep forbids a runnable flip
without a real lane; pipelines.json `requires_source_contract`; 41 tests; suite 6440/0;
kibitz r1-r4 `kibitz-runs/2026-07-05-multimodal-chunk3/`). 4b item 3 DONE -- a
non-science runnable flip now needs only per-lane curation + the item-4 seam audit.
CHUNK 2 SHIPPED @9809e36f (exchange seam: new `exchange_system` production seam --
allowlist + science pack byte-identical to the extracted EXCHANGE_SYSTEM_PROMPT
fixture + exchange_compose pass row; system_prompt threaded run_exchange_prepass ->
compose_exchange -> build_exchange_prompt, dynamic craft bullets stay Python-owned;
the writer resolves via the router repo=None lane OUTSIDE the prepass PD1 swallow so
a bank without the seam fails the episode LOUD, media_archive pinned; 13 tests incl.
a resolve-outside-try AST pin; suite 6400/0, Bug Bible 16). CHUNK 1 SHIPPED @69afbd83 (outline stage prompts
pack-routed via the router repo=None lane; science byte-identical; bank-without-seams
fails loud; 13 tests; suite 6387/0). Remaining outline-side seam coverage (announcer
intro/coda/style-pick) rides the same pattern when a lane needs it. Overnight 2026-07-05/06 (operator "EVERYTHING" directive; full detail in
docs/2026-07-06-overnight/MORNING_REPORT.md):
- 3C @c24dc0fa: STAGE 3 COMPLETE (visual_style widget slot 26; all 5 styles LIVE).
- LIVE SMOKE PASS x2 on the new code: sci-fi default (byte-identical stamps) + ANIME
  episode in otr/obs (signal_lost_neons_flicker_20260705_044421; portrait visually
  cel-shaded -- widget->meta->composers->pixels proven).
- STAGE 4 CORE @8da76394: rules packs (nodes/story_rules/science_news.json generated
  from the constants) + _otr_story_rules.py loader + all 7 hygiene wrappers pack-routed
  (155 pinned tests unchanged) + the stage3 seed producer (was DEAD) + scan third resolve
  site. Kibitz r1-r4 + 3-lens Sonnet fan-out. BUILD_PLAN amended: module enforcer, no node.
- BUG-LOCAL-416 (refine TypeError) + BUG-LOCAL-417 (reroll bank mismatch + repo-None
  no-op trap) root-caused, fixed, regression-tested, logged.
- Suite 6374/0, Bug Bible 16, box reset. Remaining Stage-4 tail: the 3 non-science rules
  packs ride each bank's lane-enablement (deliberate; no fake curation).
Sub-plan of record: `docs/multimodal-story-schema/STAGE3_SUBPLAN.md` (v5 FINAL -- kibitz
r1-r4 CONVERGED 2026-07-05; r3 = codex + a 3-lens Sonnet grounded fan-out per operator).
**3A SHIPPED @4f611cb3** (suite 6294/0, Bug Bible 16): `nodes/_otr_visual_styles.py`
lazy fail-loud loader + `nodes/visual_styles/sci_fi_radio.json` byte-identical; ALL tail
reads pack-routed (incl. mesh_fodder + still_word); SIX composer seams de-swallowed;
47 tests incl. 3 AST guards. Then 3C (widget slot 26, GATED, the 2C playbook). v1 slice
= tails + allow_radio_tails ONLY (section 8 checklist for the rest). STAGE 2 COMPLETE.

**2C SHIPPED 2026-07-05 @78bee5d5** (kibitz r1-r4 CONVERGED, codex panel --
antigravity credit-bugged/dropped -- Claude anchor+judge; artifacts + plan of
record `kibitz-runs/2026-07-05-multimodal-2c/`): source_bank widget at slot 25
(default science_news, registry-live choices, fail-loud INPUT_TYPES);
require_runnable_bank is the FIRST statement of run() (zero side effects on a
non-runnable pick); selection threads run() -> _resolve_inputs -> the 3 writer
compose_line sites -> compose_line/draft (incl. all 3 recursive repair calls)
-> resolve_creative_system_prompt(source_bank_id); meta.source_bank stamped;
source_bank on both CREATIVE_WHITELISTs. BONUS root-cause fix BUG-LOCAL-416
(refine _core locals() leak -> TypeError on every refine-enabled run since
2026-06-24; kibitz-found). Suite 6247/0 (+15 2C tests), Bug Bible 16.
NOTE: only `line_composer_system` is pack-routed today -- the LANE-ENABLEMENT
CHECKLIST (STAGE2_SUBPLAN.md section 4b) gates any future runnable:true flip
(outline seams, exchange seam, source payload, remaining seams).

Prior context (2A+2B, shipped @1d06f5c3):
STAGE 2 CHUNKS 2A+2B SHIPPED 2026-07-05 (@1d06f5c3; precondition @843ced43; sub-plan v3 @cda2076a),
science lane byte-identical, zero episode change:
- Precondition: `_otr_outline.py` outline-resolver `except Exception -> overlay=None` swallow REMOVED
  (Fable forward-note) + AST pin test (no resolver call may sit in a try/except).
- 2A routing: `nodes/story_packs/banks.json` + `pipelines.json` registries (LIST rows, element-level
  id uniqueness) + `nodes/_otr_story_routing.py` (stdlib, LAZY -- zero import-time I/O, test-pinned;
  sweep: every story_packs/ subdir must be a registered bank, every pack header triple must match its
  path; cross-refs incl. pipeline-precedence equality + required_seams presence; typed StoryRoutingError
  hierarchy). RUN GATE = `bank.runnable` ONLY (`pipeline.executable` is metadata, never consulted).
  Cache-safe loader split: `load_pack` stays strict; `load_pack_with_seams(path, extra_seams)` has its
  OWN cache keyed (path, seams) -- no strict-cache poisoning; sole caller = routing (test-pinned).
  Router drops `_SCIENCE_PACK_PATH` -> `resolve_story_pack("science_news")` (bank binding transitional
  until 2C threads the widget).
- 2B lanes: `media_archive/media_restoration_adventure` + `public_domain_story/faithful_radio_adaptation`
  (exact keys {line_composer_system, coda_system}) + `custom_source_bank/simple_4_prompt_experimental`
  (exactly its 4 pipeline-declared pass seams; strict load_pack rejects it by design). All 3 banks
  `runnable:false` -- run-intent raises StoryBankNotRunnableError, NEVER falls through to science.
Suite 6232/0, Bug Bible 16. GATES: kibitz r1+r2 (codex+antigravity, Claude anchor+judge) CONVERGED on
the sub-plan; artifacts `kibitz-runs/2026-07-05-multimodal-stage2/`. Sub-plan of record:
`docs/multimodal-story-schema/STAGE2_SUBPLAN.md` (v3 FINAL -- 2C spec lives there: widget appended at
END slot 25 default science_news, guardrail test :673-733 updated SAME commit, run() threads selection
explicitly, require_runnable_bank before story execution, NO fallback choice list at INPUT_TYPES).
NEXT = Stage 3 (visual_style schema; see BUILD_PLAN) -> Stage 4 (asserts->JSON enforcer).

Prev step: WIDGET-SURFACE CLEANUP BUILD -- COMPLETE @82f39a23 (history in `docs/HANDOFF_LOG.md`).
- Batch 1 @364a9278 -- surface-only removal (node 80 delivery_profile + 81/82/83 stereo_policy);
  kwargs KEPT byte-identical (neutral/mono_safe); test_audio_byte_identical stayed green.
- Batch 2 @f18746ce -- tooltips only (protagonist_only supersede note + VideoRenderBatch mode-
  conditional docs); no renames.
- Standalone @268e7352 -- strict-types CLI unions the dynamic registry keys (nodes 80-83 false-flag
  fixed) + _otr_workflow_validator hyphen dotted-path fallback replaced by a sys.modules scan.
- Batch 3 @82f39a23 -- caption single-owner migration to node 86 (chain 84->93->86->95->85); node 93
  caption path stripped (13->11 widgets, scopes L273 dst 11->9); env-only enablement CUT; mapping
  retargeted. Grounded general-purpose review + Fable §9 gate BOTH SHIP.
NON-BLOCKING follow-ups from the Fable gate (fold into a later pass, none gate anything): N1 restore
`from typing import Optional` in otr_post_upscale_procgen_blend.py; N2 node 93 stamps
ledger.final_video_path with the now-UNcaptioned blend (restamp at 86/85 for truthfulness); N3 stale
node-93 caption log strings.

**NEXT (operator-gated) = resume the PARKED credits-enrichment LIVE frame-level smoke** (S3+S1 shipped
@5f510ebe) on a SHORT + LONG episode: last frame is a credit frame, body audio byte-identical, no mux
ValueError on the long roll, final in otr/obs carries the credits tail; then S0 -> S4. Source of truth:
`docs/2026-07-03-credits-enrichment/GO_FORWARD_CREDITS.md` (v4). NOTE: the operator is handling the
Comfy Cloud model hard-fails himself -- the model-matrix SOAK is DEFERRED by operator directive
(2026-07-04). Reset the box per CLAUDE.md section 4 before any headless render.

**Previously active = STACK-WIDE NO-FALLBACK RIP (COMPLETE).** Operator directive: every model
failure FAILS THE EPISODE LOUD (named raise), never a silent swap/canned-template. Source of truth +
full grounded site map + kibitz/agy/Sonnet/Fable reconciliation: `docs/2026-07-03-no-fallbacks-rip/PLAN.md`.
DONE + pushed to v2.0-alpha this arc (all suite 6138-6142 + Bug Bible green, 0 regressions):
- Cloud voice (ElevenLabs @925438e2) + cloud music (Sonilo @c7da53b1) -- built fail-loud (cloud
  excluded from rank_chain auto-select). writer_fallback backup-LLM dropdown ROADMAPPED (ROADMAP_IDEAS.md).
- R1 @822cb0c9 (audio-voice: bark missing-ref net + cast_lock fail_soft->fail_loud KEEP announcer
  reroute + kokoro voice-id swap + stage-direction silence) -- Fable-gated.
- R1c @2d4cd864 (scene_sequencer inline-Bark clip-shortfall -> raise; Fable-caught).
- R2 @f07b837d (image: empty named-slot raise [E8-precise] + scene-still-missing raise).
- RENAME @31c2a473 (other_beats_image_model -> character_image_model; widget+JSON+profiles+tests;
  validator green -- the slot is character-only after scene_broll/background were ripped 2026-07-01).
- R3-chunk-1 @de6af8c2 (dramatic-state import-fail + LLM call-fail -> raise; kept no-news-input path).
- R3-chunk-2 @1d8f7e2b (announcer intro/outro/coda LLM->template swaps -> raise; F3-hedge KEEPS the
  AI line not a template; retired the NEWS_CODA_POOL floor; 16 tests inverted across 5 files).
- **R3-chunk-3 COMPLETE + Fable SHIP** @74433163 (image lane) + @4ac329f2 (refine-loop) + @a11d8605
  (stale-comment fix): `otr_meta_brief_image_prompt.py` portrait 4 tiers (tier-2 empty / consistency
  gate / person-guard / gear-scrub-empty) + `_compose_char_scene_prompt` now RAISE when the prompt
  came from the LLM (`source=="llm"` / `llm_fn` attempted); `llm_fn=None` keeps the legit local
  template lane; `consistency_gate_warn_only=True` keeps. `_otr_casting.py _apply_llm_slot_fill`
  (opt-in `llm_slot_fill`) raises CastValidationLLMError on both fail paths. Made effective through
  the writer `_refine_loop` (was swallowing casting errors on refine pass>=1). Tests inverted +
  char-scene fail-loud test added. Suite 6139/0, Bug Bible 16. Fable §9 = SHIP (all 4 portrait tiers
  RAISE confirmed correct; happy-path dormant; no surviving swallow).

**NO-FALLBACK RIP COMPLETE -- ALL 10 SITES RIPPED + GREEN + Fable SHIP.** chunks 1-3 (audio/image/
casting/writer) + chunk-4 (`otr_shot_lock.derive_creative_directives`, the 10th site operator ordered
ripped). Every attempted-model failure across the stack now fails LOUD with a named raise carrying
"no-fallback rip"; the deterministic template survives ONLY as the `llm_fn=None` 100%-local lane +
`consistency_gate_warn_only` keep. Suite 6141/0, Bug Bible 16. Chunk-4 @26b236e6 + @432cb576.
KEPT (NOT fallbacks, unchanged): title->outline.title (primary AI title), news degrade (gated by
news_briefs_required toggle), J2 outline speaker round-robin (structural), the "no news input" degrade,
the shared `_resolve_writer_llm` None-on-load-fail seam (same in both lanes; out of scope, flag if reopened).
- Nit (fold into any future edit of that block): `otr_meta_brief_image_prompt.py` gear-scrub template
  rebuild ~:1219 omits `_aspect` (portrait-aspect rebuild); pre-existing, practically unreachable.
**NEXT = resume the PARKED credits-enrichment (below): the LIVE frame-level smoke (S3+S1 shipped).**

**Previously active (still open, resume after the rip): CREDITS ENRICHMENT --
S3+S1 ATOMIC SHIPPED @ 5f510ebe (+ fade23c3 JSON
recompact, + 20a669de Fable-gate fix). NEXT: LIVE frame-level smoke -> S0 -> S4.** Source of truth:
`docs/2026-07-03-credits-enrichment/GO_FORWARD_CREDITS.md` (v4) + `KICKOFF.md`.

**TAIL-CHAIN ORDER (closeout-verified 2026-07-04 against the REAL workflows/otr_scifi_16gb_full.json --
round-trip OK, no BOM, link-validator exit 0, suite 6141/0 + Bug Bible 16):** the video tail is
`12 SignalLostVideo -> 84 OTR_SilentComposite -> 86 OTR_CaptionBurn -> 93 OTR_PostUpscaleProcgenBlend
-> 95 OTR_CreditsRoll -> 85 OTR_MasterAudioMux` (terminal mux-LAST). Credits (node 95) sits AFTER the
upscale/procgen blend (93) and is the LAST video stage before the master mux (85). Wiring: L266
86->93[0]; L250 93->95[0]; L275 92 VideoRenderBatch[1]->95[1]; L274 95[0]->85[0] (video); L276 95[1]->85[6]
(FLOAT declared credits tail -> the credits-aware mux guard). Node 95 present in the real JSON (not a copy).
NOTE: the link-validator's 4 `--strict-types` reports (node types 80-83 legacy audio: CastLock/
BatchCharacterVoices/AnnouncerVoice/StableAudioTheme not in NODE_CLASS_MAPPINGS) are PRE-EXISTING baseline,
unrelated to credits; link integrity is clean (exit 0) and the workflow-validator suite is green.

DONE (pushed, HEAD==origin):
- S2 durable singleton stamps @ 3e0003e8; OTR_CreditsRoll SCAFFOLD @ f00a8e8e.
- **S3+S1 ATOMIC @ 5f510ebe:** registered OTR_CreditsRoll; ripped node-12 RENDER-ENGINES dossier
  section + HUD credits-music loop (->silence pad) + too-early treatment engine-enrich; ripped node-84
  BUG-410 floor-extend-past-master ONLY (kept the master-mix A/V-sync cap + looped-last-clip tail);
  wired 93->95->85 (node 95; link 250 rewired, new 274/275/276; declared-tail input node 85 slot 6);
  credits-AWARE mux guard (v <= a + declared_tail + tol); retired the 4 moved tests + updated the
  93->95->85 visual pin. Suite 6108/0, Bug Bible 16, B7 green.
- **GATE (CLAUDE.md §9):** grounded general-purpose review (0 breakers) + Fable FINAL gate -- Fable
  caught a real deliverable-path bug (mux _default_out didn't peel "_with_credits"); FIXED @ 20a669de.
  NOTE: the formal codex+antigravity kibitz was NOT run (general-purpose+Fable used) -- operator may
  run it before promotion. Non-blocker flagged: cast_voice_slots isn't durably stamped, so the credits
  CAST speech_signature quote is silently omitted (cosmetic; fold into S0/a follow-up).
NEXT = (a) LIVE frame-level smoke on a SHORT + LONG episode: last frame is a credit frame (not black),
video ends with the roll, body audio byte-identical, no mux ValueError on the long roll, final in
otr/obs carries the credits tail. Also the operator's 30-45w e2e matrix (local flux+ltx_audio_in +
cloud legs; OTR_COMFY_API_KEY SET len=72; cloud voice TTS w/ indextts2/chatterbox fallback). Then
(b) S0 (font +50% + credits-aware duration budget; fold in the cast_voice_slots signature), then
(c) S4 (footer :598 only).

**Previously active (PARKED behind credits enrichment; resume after):
CLOUD SOAK + Bug B (motion) + 1080p, below.**

**ACTIVE (2026-07-03 day) = CLOUD SOAK + fix Bug B (motion) + 1080p, on the LOGGED-IN Desktop.**
Cloud auth is UNBLOCKED: operator set `OTR_COMFY_API_KEY` at USER scope (len=72) -> headless
runs inherit it (load per-command: `$env:OTR_COMFY_API_KEY=[Environment]::GetEnvironmentVariable(
'OTR_COMFY_API_KEY','User')`). Smoke leg1 proved auth resolves (old "no credentials" gone; the
remaining `nodes.MAX_RESOLUTION` error is a STANDALONE-harness artifact, not production).
CLOUD-VIDEO soak RUNNING detached: `scripts/_otr_cloud_video_soak.py` (word_razzle/kling_avatar/
seedance_2/kling_lipsync x ideo image, 30w, indextts2) -> results in
`scripts/_otr_soak_capstone_results/cloudvid_<stamp>/results.jsonl`, obs -> otr/obs. cv1
word_razzle x ideo verified rendering (auth OK).
OPEN, do next window:
- **1080p (operator):** cloud video + stills at 1920x1080 (humo PORTRAIT excepted). Wire canvas_w/
  canvas_h=1920/1080 on OTR_VideoDirector (VERIFY it's patch-safe / whitelist) + OTR_CLOUD_PIXVERSE_
  QUALITY=1080p; kling has its own res tier. Current run is DEFAULT 832x480 (proves path, not 1080p).
- **Bug B (motion not in final):** heavy-engine finals ship the LEGACY procgen video (video_engine
  HUD/credits over the scene STILL), NOT the per-beat motion clips (OTR_VideoRenderBatch ltx/humo).
  Trace episode dir manifest -> compositor blend -> final mux; make motion the BASE. See
  docs/2026-07-03-cloud-video-fixes/PLAN.md.
- **Bug A (auth code wiring):** now OPTIONAL (env key solves it) -- only for the pure-login-no-env
  case: declare hidden api_key_comfy_org/auth_token_comfy_org on OTR_ImageGenDispatcher +
  OTR_VideoRenderBatch. Plan + kibitz r1 (codex+agy, launched) in docs/2026-07-03-cloud-video-fixes/.
- wan_i2v PARKED (ckpt missing).



**ACTIVE = CODE EVERYTHING FIRST, THEN SOAK (operator directive 2026-07-03 night).**
Order is HARD: (1) CODE -> (2) REGRESS -> (3) PUSH -> (4) SOAK. Do NOT soak until the code is
built, green, and pushed.

**GOLDEN RULES (operator, restated -- these govern the whole mission):**
- NO fallbacks. NO hidden promotion of models. Every model/engine must work END-TO-END or FAIL
  LOUD and BE FIXED (root cause, no shims). A silent degrade or an auto-swap is a bug.
- If you get HUNG UP on an approach, run `/kibitz` (codex panel; Cowork Claude anchor+judge) for
  convergence BEFORE escalating -- you are the judge.

### (1) CODE -- build the pending items
- **still_word** per `docs/2026-07-03-sprintb-remainder/BUILD_PLAN.md` (model-agnostic
  still_flat-sibling VIDEO engine; kibitz r2 + roundtable converged; exact grounded sites listed
  there -- render_driver ENGINE_FAMILY + :1044 tuple; composer via image_policy[video_models] +
  _still_word_roles_from_policy; pure compose_still_word_prompt fail-LOUD; register in 5 sites).
- **word_razzle** -- the ANIMATED word-card variant (operator wants it BUILT now, not just a name
  constant). Ref: `docs/GO_FORWARD_NEXT/2026-07-02-ideo-word-razzle-vid.md` (Phase 0 audit +
  Phase 1). Golden rule applies: if a promptable cloud i2v path is missing, FAIL LOUD + fix or
  /kibitz -- never a hidden fallback.
- Any other build-ready pending items the operator has queued.
- Each engine wired IN `workflows/otr_scifi_16gb_full.json` in the SAME change (hard rule 0);
  validator + widget audit after.

### (2) REGRESS + (3) PUSH
- Full suite + Bug Bible + B7 green after every code change; commit AND push per green chunk to
  v2.0-alpha; verify HEAD==origin / no BOM / AST parse.

### (4) SOAK -- only after the code is green + pushed
Render a WHOLE BUNCH of 30-45 word FULL-PIPELINE episodes sweeping the model matrix, autonomously.
Budget = ~2000 Comfy Cloud credits, ALL usable.
- **Pass 1 = COHERENT same-model combos, NEWEST models first** (same video model across video
  roles + same image model across image roles per episode, NOT mismatched -- including still_word /
  word_razzle where coherent). Voice = indextts2 (default) or bark. target_words 30-45.
- **Later passes = deliberate MIXES** once the coherent baselines pass.
- Watch CLOSELY; on any leg failure ROOT-CAUSE fix (golden rules), re-regress, push green per chunk.
- **Discipline (CLAUDE.md):** RESET the box before EVERY headless run (SELECTIVE CIM kill, never a
  blanket python kill; confirm :8000 empty + VRAM at baseline). LOAD the REAL
  `workflows/otr_scifi_16gb_full.json`. Assets -> `otr/episodes/<ep>/`, final -> `otr/obs/`;
  Test-Path the asset before declaring success. Single resident heavy <= 14.5 GB; audio byte-identical.
- Harness: `scripts/queue_smoke.py` + `scripts/otr_api.py` (live full run) + the matrix/slot drivers
  under `scripts/`. Cloud rows need `OTR_COMFY_API_KEY`. Log per-leg verdicts + a HANDOFF_LOG entry.

**SPRINT A DONE @ 8de5862d** (E1/E2 no-fallback rip; details in HANDOFF_LOG/ARCHIVE).

**SPRINT B S1 STILLS CORE SHIPPED @ b5ef58bc (2026-07-03).** B1-B5 landed:
canonicalize_image (real, cover+crop to exact role canvas, sRGB PNG, sha256) + 4 cloud
image adapters (cloud_recraft/cloud_flux_pro/cloud_nano_banana_2/cloud_seedream_2) on the
reduced ImageEngine protocol (render_image -> invoke_partner_node -> canonicalize_image ->
str(png_path)); B4 cloud_model_ids.py single-source V3 model-id resolver (never forwards the
placeholder); B2 one cpu CAPABILITIES row each (consistency invariant green); B5
profile->schema conformance test over the billed yaml rows (image+video covered;
elevenlabs/sonilo/stability xfail pending their sprints). B7 = NO JSON CHANGE NEEDED (the
ImageDirector combo is dynamic from all_engine_names(); the 4 engines auto-appear
selectable, defaults stay flux_gen1). Suite 6089/0 + Bug Bible 16-pass + B7 in-suite.
EMPTY default_roles (never automatic); no enable flag; per-row env-overridable estimated_usd.

**S1+1 `ideo` SHIPPED @ 1bf2a2d2** (plain cloud Ideogram scene-still; node_key
cloud_ideogram_v4; rendering_speed price map; off the conformance xfail list). Suite 6093/0.

**NEXT CODE = `still_word` (BUILD-READY, kibitz r2 + roundtable converged 2026-07-03).**
Operator RE-ARCHITECTED the words feature away from a cloud-Ideogram `ideo_word` IMAGE engine
to a MODEL-AGNOSTIC VIDEO engine `still_word` (a still_flat sibling in cheap_families.py):
selected per-role in the VIDEO dropdown; the base still is minted by ANY chosen image model
(decoupled -- "don't fix the image model into the video options"); the delta vs still_flat is
the PROMPT the base still is generated from -- char/announcer = word-driven from the beat
script line, music = abstract episode-title picture (no words), pooled-char = DEFERRED
(pooling removed 2026-07-01). `word_razzle` = animated variant (NAME CONSTANT ONLY in v1, no
dark registered engine). FULL BUILD-READY SPEC + exact grounded sites (render_driver
ENGINE_FAMILY + the :1044 still-init tuple; composer via image_policy["video_models"] +
_still_word_roles_from_policy like mesh_fodder_roles; pure compose_still_word_prompt fail-LOUD;
fail-LOUD no-floor; register in all 5 sites): `docs/2026-07-03-sprintb-remainder/BUILD_PLAN.md`.
Panel = codex (local) + Grok + Gemini (GPT empty-reasoned; antigravity credit-bug dropped);
Cowork Claude anchor+judge; ~$0.20 roundtable spend. THEN (b) **B6** portrait-mint 3D
pre-selection gate (3D lanes PARKED, lower-urgency). Then order C(S3 full) -> D(TTS) ->
E(C1) -> F. S5 GPU exit gate + S4x GO/NO-GO still await the operator (unchanged).

Conformance debt (fold into still_word or a follow-up): the B5
`_engine_by_node_key()` map is node_key->ONE engine (last wins); if any two engines ever share
a node_key, make it iterate ALL engines per node_key.

**QUEUED (operator pickups 2026-07-03, ride BEHIND the forward order):** ideo_word family --
(1) `docs/GO_FORWARD_NEXT/2026-07-02-ideogram-lyric-stills.md` = the MAIN build, pulled in as
S1+1 (`ideo` plain scene-still + `ideo_word` words-specialist IMAGE engine in
nodes/_otr_image_engines/, lyric_text vs title_mood modes, worded cards NEVER pooled);
(2) `docs/GO_FORWARD_NEXT/2026-07-02-ideo-word-razzle-vid.md` Phase 0 (--audit-i2v report-only
on otr_pin_partner_nodes.py) = safe filler ANY time; Phase 1 GATED on S1 + ideo_word landing.
Parked siblings (do NOT build): 2026-07-02-ideo-word-vid.md (NEEDS-DECISION),
2026-07-02-ideo-word-3d-razzle.md (rejected, reference only).

**OPERATOR DIRECTIVE 2026-07-02 late evening: NO FALLBACKS, NO AUTO-DEFAULTS, ANYWHERE.**
The dropdown values SAVED in `workflows/otr_scifi_16gb_full.json` are the ONLY defaults.
Consequences: (a) S3 FULL is RESCOPED -- "reactive auto-defaults + fallback chains" are
CUT from its scope (cloud rows stay selectable-only; remaining S3-full scope = ShotLock
audit stamps + seedance/wan V3-expansion pins + live provider proof); (b) E1 (rip the
fallback scaffolding out of render_driver: make_fallback_of/UNIVERSAL_FLOOR/SYNTH_FALLBACKS/
EXPECTED_OOM_TRAIL) + E2 (kill allow_auto_fallback) are PROMOTED from tech debt to
directive compliance -- engine failure = LOUD stop, never a swap; (c) no code-side
default_engine_for_role may override the shipped JSON's widget values.

**OPERATOR DIRECTIVE 2026-07-02 evening (SHIPPED @ cc349c1d): NO hidden cloud enable
switch.** `OTR_ENABLE_COMFY_CLOUD_MEDIA` is REMOVED (same clean break as OpenRouter C6):
the dropdown pick IS the enable; missing credentials fail LOUD at invoke-time auth
(naming OTR_COMFY_API_KEY / logged-in hidden inputs); budget unset = $10 DEFAULT_BUDGET_USD
safety cap (explicit 0 = deliberate spend-off). This SUPERSEDES every "flag OFF/ON" test/
acceptance line in pass04 sec 8 -- S1 builds on the no-flag reality. The S0 live smokes
now need only auth (+ OTR_RUN_CLOUD_SMOKE=1, the paid-run gate for the script).

**S5 SHIPPED @ fb23d82d (2026-07-02 late evening):** silent two-stage HQ recipe in
eng_ltx_video (OTR_LTX_VIDEO_RECIPE auto|hq_two_stage|single_pass; dev-unet auto default;
init still required fail-LOUD; single_pass frozen byte-identical; 2 LTX rows, NO
ltx_lowvram). REMAINING = the GPU A/B exit gate above.

**CLOUD ENGINE LANES S0 (context; operator "build" 2026-07-02).** Build doc =
`docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md` @ 29b11e77 (4-round roundtable converged
+ operator amendment: audio reactivity DEFAULT-ON all video roles, mute = audited opt-down).
S0 PROGRESS (2026-07-02): chunks 1-3 SHIPPED + PUSHED, suite 5988/0 + Bug Bible green each chunk:
- c1 @ 5a79a926 `nodes/_otr_shared/cloud_media_backend.py` (error taxonomy, auth broker, prompt_id
  session table w/ leak sweep, budget state machine, provider semaphores, billing JSONL, mute-opt-
  down knob) + 27 tests.
- c2 @ 7d8c490f `cloud_media_cache.py` (RequestCacheKey pre-submit-only, atomic store, quarantine,
  per-key locks) + `cloud_media_canonical.py` (PartnerResult/CanonicalAsset + S1/S2/S3 contracts)
  + 14 tests.
- c3 @ 44f36fdc+f3a97ea6 `scripts/otr_pin_partner_nodes.py` + CHECKED-IN
  `nodes/_otr_shared/partner_nodes.yaml` -- 11/11 rows pinned from the LIVE core
  (`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI`, override `OTR_COMFY_CORE_ROOT`); all async
  EXECUTE_NORMALIZED_ASYNC, all expose both auth hidden inputs; drift test = SUBPROCESS `--check`
  (in-process comfy import corrupts pytest teardown -- do not inline it) + 6 tests.
- c4 @ f9eed360 `nodes/_otr_shared/cloud_media_invoke.py` -- `invoke_partner_node(node_key,
  inputs, *, timeout_s, estimated_usd) -> PartnerResult` (backend-owned loop thread; session from
  prompt context / `bind_prompt_id()` headless; hidden-auth per row; 5s-tick watchdog w/ interrupt
  cancel + 20s ProgressBar heartbeat; streamed temp downloads; release-vs-bill settlement) + 23
  tests + GATED smoke harness `scripts/otr_cloud_s0_smoke.py` (leg1 recraft auth proof, leg2 Kling
  avatar conditioned by `tests/fixtures/baseline_v1.5.wav`).
S0 REMAINING: ONLY the live smoke RUNS -- operator must set `OTR_RUN_CLOUD_SMOKE=1` and
`OTR_COMFY_API_KEY` (budget optional -- unset = the $10 default cap; the enable flag is
REMOVED per the 2026-07-02 directive above), then
`python scripts/otr_cloud_s0_smoke.py --leg 1` / `--leg 2`. NO existing render-path changes in S0;
workflow JSON untouched until S4 (operator-gated). NEXT CODE = S1 (stills lane: canonicalize_image
+ recraft/flux/nano adapters + portrait-mint gates).

**CODE BATON 2026-07-02 (later):** back with the LTX-fixes window; cloud S0 remainder BUILT there
(c4 above). c5 (docs window, 2026-07-02 pm): roster expanded to 14 pinned rows
(+cloud_ideogram_v4, +cloud_seedream_2, +cloud_elevenlabs_voice_selector AUX -- the TTS row
requires its ELEVENLABS_VOICE output) + versioned pricing stamp `docs/2026-07-02-cloud-engines/
PRICING.md` (211cr=$1; voice ~$1.10/ep FLAT across tiers; per-line Kling lipsync ~$0.25-1.00 = the
dominant cost) + prompt/param profiles doc (see below).

**VIDEO TEAM warning (2026-07-02 pm, codex-verified) -- RESOLVED at a9440980 (same day, night):**
the audited eng_cloud_video.py draft emitted unpinned kwargs; the SHIPPED commit folds every catch:
cloud_wan_i2v sends EXACTLY `first_frame`/`model`/`prompt_extend`/`seed`/`watermark` (no prompt;
audio not sent -- mute row); cloud_seedance_2 is an HONEST DARK ROW (raises loud) until the S1
V3-expansion pin names its dynamic inputs; the kling pair's kwargs are all pinned; per-row
conformance is test-locked in tests/test_cloud_video_adapters.py. STILL OWED AT S1: the generic
profile->schema CONFORMANCE TEST (every emitted kwarg declared in the yaml) as the permanent guard.
`canonicalize_video` is REAL now (strip + post-strip proof) -- see HANDOFF_LOG.

**S3 hold SUPERSEDED (operator, 2026-07-02 evening, voice: "code the cloud video plan... tagged to
go forward"):** S3 CORE shipped @ a9440980 -- 4 rows registered dark + fail-closed, EMPTY
default_roles (selectable picks only; never automatic). S3 FULL (reactive auto-defaults + ShotLock
audit stamps + fallback chains + seedance/wan V3 expansion + live provider proof) still rides the
original order: live smokes (operator env) -> S1 STILLS -> S3 full. One window at a time,
coordinated HERE.

QUEUED BEHIND cloud S1+S3: creative formats F1 Living Evidence Board + F2
Tin-Toy Theatre -- plan `docs/2026-07-02-creative-formats/CREATIVE_FORMATS_PLAN.md`
(kibitz-hardened; ideation record in docs/2026-07-02-cloud-engines/roundtable/ideas_synthesis.md).

**Previously current (unchanged, operator-gated, NOT code):** All-engines x all-slots soak RUN +
talking-radio (C) morning-eyeball GO/NO-GO -- see below.

**All-engines x all-slots: CODE SHIPPED (slot-audit C0-C5) -- remaining = the live-GPU soak RUN.**
Boot headless ComfyUI, load `otr_scifi_16gb_full.json`, apply the all-role profile
(`slot_matrix.build_all_role_profile` -- 3 roles post rip-sfx-broll), render a leg per engine, run
`content_oracle.check_manifest` on the per-beat manifest. GPU-operator-gated (not code).
Sprint spec: `docs/2026-06-30-slot-audit/SPRINT_PLAN.md`. Acceptance met in code; the RUN proves
it empirically. Accelerator = S-F visual smoke fixture (shipped).

**rip-sfx-broll SHIPPED 2026-07-01 (see HANDOFF_LOG.md):** the role model is now
speaker = {character, announcer, music_open, music_close, music_inter} and
video = {announcer_visual, music_visual, character_video}; NO FALLBACKS -- an unmapped role or an
old `speaker_role:"sfx"` ledger FAILS LOUD everywhere. Build plan + kibitz judgments:
`docs/2026-07-01-rip-sfx-broll/`. Old on-disk episode ledgers predating the rip must be
regenerated before reuse.

**Opt-in feature SHIPPED (not part of the forward order):** brief-driven HuMo radio-host
+ `OTR_LTX_RADIO_FACE` A/B (default OFF, byte-identical). See HANDOFF_LOG.md.

**TALKING-RADIO: RECIPE_IA2V is the dev-default lane; S4/S4b/S4c SHIPPED (2026-07-02).**
The canonical comfy.org IA2V transplant reversed the C NO-GO (see HANDOFF_LOG + PROOF7_VERDICT.md
in docs/2026-07-02-canonical-ia2v). Shipped since: talking prompt register, S4 portrait init, S4b
face-forward portrait mint, S4c radio-face DEFAULT-ON for ia2v bookends (env A/B = single-pass
only). VERDICT (2026-07-02 late evening, PROOF9_VERDICT.md): PROVISIONAL PASS on direction --
the 120w soak (S4 fired, log-proven) lifted characters ~3x relative to the announcer anchor,
but the 2.0 bar is CANVAS-SPECIFIC; final GO/NO-GO = proof9d clean 832x448 re-run. **S5 SHIPPED
@ fb23d82d** (silent HQ two-stage in eng_ltx_video; GPU A/B exit gate owed). Story-writer
fixes are PARKED in the transplant repo (UpstreamStoryLab GO_FORWARD "DEFERRED STORY-LLM FIXES")
-- NO production story-LLM changes until the refactor. Source-bank visual-style transplant stays
OUT (research mode; docs in `ComfyUI-OTR-UpstreamStoryLab\docs`).

---

## 1A. OPEN ITEMS (post-soak, priority order -- detail in the sprint specs, do not inline here)

Coverage-soak sprint spec (kibitz r1-r4 converged): `docs/2026-06-29-coverage-soak/SPRINT_PLAN.md`.
The load-bearing OPEN items (7 of 11 sub-items already shipped -- see HANDOFF_LOG.md):

- **E1/E2 = Sprint A -- SHIPPED @ 8de5862d** (no-fallback rip; see section 1 + HANDOFF_LOG).
- **E3-doc -- edit THIS doc only.** station_card + abstract retired (C0). `still_motion` is NOT retired
  (it is UNIVERSAL_FLOOR + mesh_stage's fallback target) -- do not unregister it.
- **E4 -- "which model" dropdown spell-out (DEFERRED, low priority).** Audio-reactive + VRAM-tier
  suffixes shipped; spelling out LTX/Wan/image recipe per label needs a no-drift design decision.
- **S-C C1 -- shared `audio_motion_profile` (NOT STARTED).** Per-beat rms/peak/onset/silence/brightness/
  dynamic-range/speech-vs-music/duration driving every engine. C2 (per-engine consumers + HuMo
  phrase-chunking, the real clip-underrun fix) deferred per the plan.
- **Writer (non-blocking):** long-target freeze-gate flakiness -- target_words=800 tripped
  BUG-LOCAL-276 on 2/3 attempts + under-delivered length. Writer-side look. See
  `docs/2026-07-01-overnight/MORNING_REPORT.md`.
- **Force-map optimization (non-blocking):** `OTR_FORCE_ENGINE_MAP=*=<engine>` still mints Flux stills
  for the pre-override plan; short-circuit still-gen when a role is forced to an `accepts_still=False`
  engine.

Invariants for all: single resident heavy <= 14.5 GB; audio byte-identical; no-fallback (hard-fail
LOUD); UTF-8 no BOM; SFW; workflow-JSON edited in the SAME change as code; suite + Bug Bible + B7
green + push per green chunk.

---

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (hard):** `workflows/otr_scifi_16gb_full.json` IS production. ANY node/
  wiring/widget change goes IN that file in the SAME change as the code (unwired code is dead). Every
  API/headless/soak run LOADS this real JSON. After editing, re-validate: `OTR_WorkflowValidator` +
  JSON round-trip + link/widget audit. `widgets_values` is POSITIONAL (BUG-LOCAL-097) -- append at
  END; a mid-list removal shifts every later value (re-audit by name).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8).
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream "whiny" voice fix.
- Invariants: single resident heavy <= 14.5 GB (host NVML); 100% local/offline; determinism seed-keyed;
  every in-render fallback LOUD; UTF-8 no BOM; SFW; V-12 dep isolation; no new widgets in the static
  shell (V-11).
- GIT: ONE branch `v2.0-alpha`; commit AND push together per green chunk; operator eyeball gates TAGS/
  promotions only; after a push verify HEAD==origin / no 0-byte / no BOM / AST parse on touched .py.
  prod/`main` GATED (a `v2.0-alpha-stable` tag on `v2.0-alpha` is fine).
- EVERY session updates THIS doc (lean, forward-only) + appends one entry to `docs/HANDOFF_LOG.md`.
  (The old otr-build-tracker dashboard is retired -- the small log replaces it.)
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs log
  `cast RNG seed=... (OS entropy)`.

---

## 3. FORWARD ORDER (do in sequence within a track)

> Two tracks. Item 1 (punch-list) is OPERATOR-GATED look-QA; the ENGINE track (items 3-4) proceeds.

1. **Punch list (GATE A) -- operator-approved.** Captions DONE. REMAINING: node-audit that LTX
   radio-open + procgen rolling credits are in the SAVED JSON (not just the headless path); prove a
   render FROM the JSON has them, then operator look-QA.
2. **latentsync -- REMOVED** (not a live lane; dropped from the order).
3. **Wan 2.2 video -- operator-approved.** Both engines BUILT + validated (wan_i2v 14B + wan_ti2v 5B).
   REMAINING = operator WEBM eyeball (14B vs 5B) + optional formal `--acceptance` GREEN (slow
   wan-music-bed leg, attended) + M9 CS-3 proof. Detail: section 4.
4. **Coverage sweep GREEN (GATE-A acceptance).** Re-run the permutation matrix after the soak fixes;
   RED until Wan lands (Wan is core/blocking). Visual-engine set is wired; writer-LLM + voice leg-sets
   still need a runnable harness. Hardening M1-M9 shipped (see HANDOFF_LOG.md); exact `--acceptance`
   invocation in `docs/2026-06-13-goforward-wan-hardening/`.
5. **3D sprints.** S-3D-0 spike + T1 template + T2a wrap smoke -> `character_3d` family. Detail:
   `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing).
   Detail: `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.

**0-E parallel track:** CPU side shipped, GPU-green; Phase B (E-1..E-7) HELD on
`scripts/_otr_0e_gpu_go.txt`. **Audio parallel track:** the "whiny" voice fix (upstream TTS only;
may have self-resolved -- verify first).

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 engines, eyeball-gated, b-roll/camera motion only (lip-sync stays on HuMo).
Core Comfy Wan nodes (NOT the KJ wrapper). Phase 1 + the 5 code-gap fixes DONE (`2fbc2f3`).

- **Phase 2 -- 16GB leg.** Drive `eng_wan_i2v.render_clip` via the real path; ASSERT wan_i2v is the
  final_engine (FAIL LOUD on fallback) + render-phase NVML <= 14.5 GB + byte-identical mux + silent
  mp4. Reset the box first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF + wan2.2 VAE (record repo/sha/
  license, fail-closed); own flag/model/VAE env + registration + tests. Do NOT alias WanI2VEngine.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B) in `docs/2026-06-12-ltx23-motion/wan_clips/`.
  Bar = real camera motion, still preserved, no warp. If motion too subtle, the Path B two-expert
  HIGH/LOW handoff is the mitigation (not a knob tweak).
- **CS-3:** sequential residency (Wan ~14GB + HuMo ~7GB cannot co-reside) -- prove per-beat NVML
  <= 14.5 GB + inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`) drains the prior engine. A
  mixed Wan+HuMo episode is the test.

---

## 5. OPEN THREADS / BACK-BURNER (pointers, not plans)

- **LTX motion amount (recommended next opener thread).** LTX holds composition; open = MOTION amount.
  Env-testable A/B first (no code): at 832x480 set `OTR_LTX_SAMPLER=distilled` +
  `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`, A/B vs the good 5/09 `l001` /
  5/28 `b001` bookends; if it matches, bake those + boomerang + audio-length into `eng_ltx_video.py`.
  Forensic in `BUG_LOG_2026-06.md` (BUG-LOCAL-412).
- **CS-2** phase attribution (~16 GB machine pin vs 14.5 render-phase). **CS-4-open** (deprioritized)
  14B HuMo umt5-TE detach; default char tier is `humo_1.7B`.
- **README "what to expect per video model"** (newbie audience; folds into S6) once the opener bake-off
  settles.
- **Ship defaults (release):** announcer + character = flux_still, music = viz_green; HuMo/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish:** output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR` (fail LOUD
  on mismatch) -- NOTE: bit the 2026-07-01 visual soak (the capstone default tree disagreed with the
  launcher's `--output-directory`; the driver now pins `OTR_SOAK_SERVER_OUTPUT`). OH-3 janitor at boot.
- **OH-4** live->attic migration STAGED (`docs/2026-06-11-output-tree-consolidation/`), awaits "go OH-4".
- **0-E Phase B** tickets E-1..E-7, gated on the sweep GO file.
- **Operator gates:** ComfyUI Desktop relaunch, fresh-render acceptance, whiny-voice reel, S-3D-0 green
  light, `v2.0-alpha-stable` tag decision.

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9: S-3D-0 spike ->
T2b keystone GO/NO-GO -> T4 driver + LOOK gate -> W7 production wiring + soak ("v1-usable") -> S3-S6
distribution. SHORTCUT FORK: keystone NO-GO -> `character_3d` defers (HuMo-2D stays) -> ~2-3 sprints.

---

## 7. POINTERS (evidence + tooling)

- Done history: `docs/HANDOFF_LOG.md` (recent) + `docs/GO_FORWARD_ARCHIVE.md` (deep).
- 3D spec (item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Wan/sweep hardening: `docs/2026-06-13-goforward-wan-hardening/`.
- Bug logs: ACTIVE `BUG_LOG_2026-06.md` (BUG-LOCAL-400+); ARCHIVE `BUG_LOG.md` (001..~305).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; cd-to-root + venv python + RELATIVE path).
- Smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`. Overnight sweep launch +
  GO file: `scripts/_otr_0e_gpu_go.txt`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale; switchable S3-S6 (closing,
after 3D); 3D GPU lanes until S-3D-0 + operator green light; the STORY-ENGINE quality roundtable
side-campaign (`docs/2026-06-21-allnight-864-frontier/SPRINT_READY_PLAN.md` -- resume only on explicit
operator go).
