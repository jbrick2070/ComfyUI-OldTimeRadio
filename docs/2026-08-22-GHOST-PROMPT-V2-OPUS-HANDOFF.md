# OPUS CODER HANDOFF -- Ghost Prompt v2

**Date:** 2026-08-22
**Repo:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
**Branch:** `v2.0-alpha`
**Start HEAD:** `667a3283e1529d1bed35aaa1f7b1000fd8234441` = `origin/v2.0-alpha`
**Model rung:** Opus / coder (rung 5). Design is settled; do not spend another
panel on it.

## Paste this to Opus

Resume OTR as the sole CODER for Ghost Prompt v2. Work on the real Windows repo
above, branch `v2.0-alpha`. Read `AGENTS.md`, `CLAUDE.md`,
`docs/GO_FORWARD_PLAN.md`, `docs/PRODUCTION_SPRINT_LESSONS.md`, the relevant
Ghost/prompt entries in `docs/PROD_BUG_LOG.md`, and the matching Bug Bible rules
before code. Then implement the exact contract in
`docs/2026-08-22-GHOST-PROMPT-V2-CONTROLLED-ABSTRACTION-PLAN.md`.

The operator explicitly stopped the remaining review panel and said to hand the
settled plan to Opus for coding. Do not restart the panel and do not use Cursor.
R1-R3 were completed and grounded; R4 has the Codex driver anchor plus one
completed Antigravity residual review whose surviving fixes are already folded
into the plan. Make the root fix, run the required tests and live proof, commit
and push each green chunk to `v2.0-alpha`, and verify HEAD equals origin.

## Operator decisions -- do not reopen

- The target is the official AnimateDiff v3 / SD1.5 Ghost family.
- One current beat = one current clip = one prompt. Do not introduce arbitrary
  32/64-frame caps. The 16-frame value is the sliding context, not clip length.
- Keep cadence, frame coverage, style cue, and negative prompt unchanged. The
  current anime/archive/material looks are accepted and must survive byte-for-
  byte through this content-prompt change.
- Prefer controlled abstraction over rigid face continuity. Recur a compact
  color/prop/silhouette motif across `figure`, `object`, and `signal` modes; do
  not force the same mediocre person into every clip.
- The LLM owns only one short drawable leaf per beat. Python owns style,
  negative, motif, mode, framing law, identities, hashes, retry, and fallback.
- The LLM never sees dialogue, title, M4/story wall, raw cast prose, or names.
- No new widget, socket, node, link, or workflow setting. The canonical workflow
  must remain byte-identical.

## Current status

- **Production code:** untouched.
- **Canonical workflow:** untouched; current SHA-256
  `89B8DF59438A1644416D376FFED66BF9D44AB5B6DC181156DA7D3436DF3A4912`.
- **Hardened plan:**
  `docs/2026-08-22-GHOST-PROMPT-V2-CONTROLLED-ABSTRACTION-PLAN.md`, SHA-256
  `D066B6FC6AFAF1A87B1ACBE067F414FBAD37B7C2BC9539E771A565967398C766`
  at handoff creation.
- **Review artifacts:**
  `kibitz-runs/2026-08-22-ghost-prompt-v2-controlled-abstraction/`.
- **No tests were run** because this window stopped before code.
- A complete pre-change v1 v3-peer episode is available as supplemental proof:
  `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\signal_lost_after_hours_encounter_20260822_152909`.
  It published eight `animatediff15_v3_haunted_video` clips and the OBS final;
  its volatile input/report/manifest are preserved under
  `evidence\ghost_prompt_v1_baseline\` with hashes:
  - input `29643B08238F5863A814ADCDE8AFCACB41EC7BC78EC6989D8BC636662AC048C3`
  - manifest `DD7409C8A0A7E3770F2CD98B9276986FD0AEAE0CB1B0F5516849322E5E052ACF`
  - report `475E0F7FC5C045AA62C0C1468A9162668DA31445C089920DDBE58D1D63D8B976`
  This is useful v1 evidence but is **not** the formal official-v3 same-seed A
  arm because its engine is the haunted v3 peer.
- A completed headless server is still resident on port `51684`. Do not reuse it
  for proof; reset selectively and boot the formal A/B on port 8000.

## Formal A arm -- run before production code

Use `config/profiles/otr_ghost_signal_v3.json` (official engine
`animatediff15_v3_video`, profile SHA-256
`9C6EFE09AE6E63652AC9E2C738B70BC22969F9654C865FCC6460F021564CDE55`),
the real `workflows/otr_canonical.json`, port 8000, and a fresh selective reset.
Pin:

- `OTR_C7=1`
- `OTR_WRITER_SEED=42`
- `source_bank=media_archive`
- `visual_style=archival_documentary`
- `custom_premise=At midnight in a shuttered radio archive, a conservator discovers that a damaged lacquer disc answers questions with tomorrow's emergency broadcasts.`
- `lemmy_cameo=never include`
- `num_characters=2`
- `Acts=1`

Use `scripts/otr_reset_gpu.ps1`, then
`scripts/otr_headless_canonical.ps1 -NoReset -Port 8000 -Profile otr_ghost_signal_v3`
with the pinned values above. Do not pass `-Workflow`; the wrapper must load the
canonical graph. Archive the API prompt dump, `/history/<prompt_id>`, leg/server
logs, and a receipt by copying them into the episode's
`evidence\ghost_prompt_v1_a\` directory. Leave episode and OBS media where the
pipeline publishes them. Recent official one-act runs cost roughly 25-35 minutes.

The B arm uses the same pins on a separate fresh boot. A new freeze timestamp
and episode path are expected, so compare the semantic control vector rather
than whole-ledger bytes: voiced-text semantic SHA, episode seed, roles/shot IDs,
delivered frame counts, video seeds, style, negative hashes, cadence, and
delivery settings. The premise deliberately avoids weapon nouns, so the
freeze-keyed banana variety cannot confound the visual content comparison.

## Exact implementation contract

### 1. New pure author module

Add `nodes/_otr_video_engines/ghost_signal_author.py`. It owns:

- safe projections and opaque `g000...` IDs;
- deterministic mode scheduling and compact motif reduction;
- strict batch JSON parsing with duplicate/nonfinite/extra/trailing rejection;
- 5-14-word, <=96-character leaf validation (tell the model 6-10 words);
- exact stored-object validation, request/output hashes, replay, whole-batch
  retry once, and complete deterministic fallback;
- lazy installed SD1 token measurement; and
- the shared `finalize_ghost_prompt_v2(...)` used by author and render.

It may import stdlib, sibling `ghost_signal_prompt`, and `_otr_banana_route`.
It must not import ShotLock, render driver, registry, or model loader. It does
not load the LLM; ShotLock owns orchestration.

Tokenizer math is exact:

```python
rows = SD1Tokenizer().tokenize_with_weights(
    text, return_word_ids=True)["l"]
payload = sum(word_id != 0 for row in rows for _, _, word_id in row)
windows = len(rows)
total = payload + 2 * windows
```

Installed contract: max length 77, BOS 49406, EOS/pad 49407. A 75-payload-token
string is 77 tokens/one window; 76 payload tokens spills to two windows. Never
count the padded row length. Production fails closed if the tokenizer is
unavailable. Under `OTR_TEST_MODE == "1"`, only the unavailable installed-
tokenizer gate may skip; all safety/shape/character checks remain. An injected
measurer must still gate.

The request hash uses compact sorted JSON with exactly:
`author_version`, `beat_id`, `mapped_arc`, `mode`, `model_id`, `motif_cue`,
`motif_sha256`, `normalized_emotion`, `ordinal`, `role`, `sanitized_intent`,
`schema_version`, `template_sha256`. `motif_sha256` hashes exact motif UTF-8;
`ordinal` is the zero-based Ghost batch position. `template_sha256` covers the
system/user template, envelope, temperature `0.1`, and output budget
`64 + 48 * len(specs)`.

### 2. Pure composer changes

In `ghost_signal_prompt.py`:

- retain `GHOST_PROMPT_PROFILE="ghost_signal_v1"`;
- add `GHOST_PROMPT_VERSION_V2="ghost_signal_v2"`;
- preserve existing v1 composer and all existing sigil bytes;
- refactor internal structured bucket choices only enough to derive v2 motifs;
- change only the unknown v1 `resolve_action()` branch so it returns a complete
  checked-in neutral action instead of copying six free-text words;
- add `compose_ghost_prompt_v2(role, style, mode, motif_cue, drawable_beat)`.

V2 ordered pieces are the existing `_prefix_pack_cue()` result, motif, leaf,
and affirmative mode law. It calls the existing negative composer verbatim and
never calls v1 action/register/trim logic.

### 3. ShotLock transaction

In `otr_shot_lock.py`, author one atomic map after effective route + legacy
sigils and before cast-time preflight. Select every registered engine whose
`prompt_profile == GHOST_PROMPT_PROFILE`, across all roles and peers. Require
`meta.episode_seed` even for bookend-only Ghost episodes. Attach a deep copy to
the temporary preflight shot, require exact coverage, and stamp the same object
on the durable row without changing `render_request_hash`.

Add `ghost_prompt: Optional[dict] = None` to `ShotRow`; do not change
`VideoRequest` or a workflow-facing schema.

Normalize the requested model ID with pure `_otr_model_catalog.validate_model_id()`
before replay lookup. Do not call `request_slot()` unless at least one row needs
authoring; after a load, assert the cache-entry model ID agrees. Preserve the
existing M4 resolver's TEST_MODE/empty-model/fail-loud behavior and its old
prompt-only `0.1/300` call surface. Ghost uses the raw message generator.

One invalid batch gets one fresh whole-batch retry. After a second semantic
failure, production may stamp a complete deterministic batch. A reused writer
row becomes `source=replay`; a reused deterministic fallback stays
`source=deterministic_fallback` with its nonempty reason, so replay cannot
launder proof eligibility. Unload the writer once in an episode-level `finally`
and assert no local resident model before preflight/video.

### 4. Render and banana/token finalization

In `render_driver.py`, validate present `ghost_prompt` before the legacy sigil
guard. Valid presence uses v2 only; absence is explicit v1 compatibility;
malformed presence fails closed. V2 reads no raw intent/traits/arc, pack motion
register, open subject, or later motion clause.

The shared finalizer composes, applies the exact video banana gate with ledger
`freeze_timestamp` and `shield_quoted_card_text=False`, validates the
post-transform style/motif/leaf/law components, and measures literal final
positive and unchanged negative. Banana substitutions inside the leaf are
legal. Author candidates target <=69 tokens/one window. Final positive and
negative each require <=77 tokens/one window; positive also remains <=320
characters. V2 never trims or repairs a protected field.

Install the finalizer's banana receipt and skip the generic banana funnel only
for this already-finalized local request, or a second idempotent transform will
overwrite the real substitution count with zero. Assert every Ghost peer is
local so the cloud visual-safety hook remains inert. Stamp v2
`prompt_version=ghost_signal_v2`; leave the engine capability token v1.

### 5. Later LLM pass and engine

Extend `_otr_motion_clause.generate_motion_clauses()` backward-compatibly with
an optional lazy factory and skip predicate. Node 92 keeps `_mc_on()`, skips
Ghost by registered prompt profile without writing `motion_clause`, and invokes
the factory only for the first eligible non-Ghost character/dialogue row. Wrap
the enabled motion block in `try/finally`, unload, and assert no local writer
before `run_real_episode()`.

Change Ghost `motion_source` to `ledger_ghost_drawable_beat`. Keep authored
content out of `render_request_hash` and video-seed identity; final literal
positive/negative already invalidate the engine cache. Extend request
observability and the hard-coded trace allowlist with every declared Ghost/
token receipt. Any pre-encode defense must call the same author-module measurer,
not a second loaded-CLIP counting algorithm.

## Expected production files

- new `nodes/_otr_video_engines/ghost_signal_author.py`
- `nodes/_otr_video_engines/ghost_signal_prompt.py`
- `nodes/otr_shot_lock.py`
- `nodes/_otr_video_engines/schemas.py`
- `nodes/_otr_video_engines/render_driver.py`
- `nodes/_otr_motion_clause.py`
- `nodes/otr_video_render_batch.py`
- `nodes/_otr_video_engines/eng_ghost_signal.py`

`workflows/otr_canonical.json` should have zero diff. If implementation reveals
that a node/socket/widget change is actually necessary, stop and obey the
same-change workflow rule instead of shipping dead code.

## Tests and acceptance

Add `tests/test_ghost_signal_author.py`; extend
`test_ghost_signal_prompt.py`, `test_ghost_signal_lane.py`,
`test_motion_clause.py`, ShotLock/preflight, node-92 unload, trace, cache/seed,
and route-freeze tests. Pin:

- strict JSON and exact coverage;
- safe input projection and no name/dialogue/title/M4 leakage;
- deterministic modes/fallback and no partial salvage;
- v1 sigil goldens and explicit absence compatibility;
- style prefix and negative byte equality across every shipped pack;
- banana on/off/fidelity cases and transformed leaf behavior;
- actual 75/76-token boundary plus TEST_MODE behavior;
- temporary/durable object equality and missing-map refusal;
- replay/no-spend, configured-load fail-loud, and unload-before-video;
- all-Ghost zero motion-factory calls and mixed lazy single load;
- same `render_request_hash`/video seed but changed final prompt/cache identity;
- every receipt reaches the durable trace; and
- one beat/clip/prompt with unchanged cadence/frame coverage.

After every code chunk: focused tests, full Windows suite with
`C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`, Bug Bible from
its separate repo, AST/JSON/BOM/zero-byte checks, canonical workflow validator,
JSON round-trip, live INPUT_TYPES/widget and link audits, then commit + push and
verify HEAD equals origin. After the finished diff, one clean independent
review is sufficient unless it reports a blocker.

The formal B proof requires `RESULT SUCCESS`, `Prompt executed`,
`obs_publish OK`, final media in the live OBS path, full real-writer coverage
(`source=writer_llm`, empty fallback reason on every Ghost row), unchanged
control vector, v2 prompt/token receipts, and operator eyeball comparison. Any
fallback or replay row disqualifies the LLM treatment; do not present it as B.

## Git / workspace state

At handoff creation, HEAD equals origin. Several unrelated untracked files are
present (4060 profiles/scripts, lofi briefs, H3 attestations, `uv.lock`, and
earlier Ghost briefs). They belong to the operator/other work. Do not edit,
delete, or stage them. Stage only this sprint's plan/handoff and the production/
test files Opus actually changes.

## Open blockers

None. The next concrete action is the official-v3 v1 A capture, then code.
