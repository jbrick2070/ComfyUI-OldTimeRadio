VERDICT: no — the root cause is real, but the document leaves the ownership decision open, misclassifies lanes, and overstates the credits/music blast radius.

MUST-FIX BEFORE BUILD:

1. [Why the existing stamp does not fire / Q1] The affected-lane model is wrong. `scifi_news` is dispatched through `scifi_news_circuit`, resolves as content-owned, and is explicitly tested to receive `meta.episode_seed`; the actual inline families also include `original` and `media_archive`, which the document omits. Replace the prose with an explicit matrix: inline `original`, `media_archive`, `public_domain`, and `shakespeare` inherit the cast seed; content-owned `scifi_news` and `scifi_news_pro` retain their existing episode-seed path. Evidence: `nodes/_otr_lane_specs.py:97-123`, `nodes/_otr_freeze_cascade.py:171-199`, `tests/test_fable2_tail_context.py:225-272`, `nodes/story_packs/*/*.json`.

2. [Q1 / Q2] Resolve the design instead of presenting two candidates. Choose (a): after the inline cast succeeds, stamp `meta["episode_seed"] = int(cast_seed)` beside `meta.cast_contract` in `nodes/OTR_LedgerScriptWriter.py:4083-4099`. Reject (b): the tail-wide call at `nodes/OTR_LedgerScriptWriter.py:6055-6062` would mint a second independent entropy value on inline lanes, separating the voice/music seed from the displayed/replayable cast seed and creating two owners. Copying the same number is safe because replay is triggered only by `meta.cast_contract.cast_seed`, not by numeric equality or `meta.episode_seed` (`nodes/cast_lock.py:340-352`).

3. [Every consumer of the frozen value] The credits claim is false. Credits prefer `meta.cast_contract.cast_seed` and consult `meta.episode_seed` only as a fallback (`nodes/otr_credits_roll.py:313-326`). The cited published ledger already has a fresh OS-entropy cast seed, so its displayed credits seed is not the frozen `coerce_int_seed(None)` value: `output/otr/episodes/signal_lost_lute_strings_fools_tongue_20260805_021040/audio/signal_lost_lute_strings_fools_tongue_20260805_021040_ledger.json`. Remove credits from the behavioral blast radius and do not use the false “identical receipt” premise to justify broader changes.

4. [Q3 / Q4] Define the acceptance contract. Require: inline lanes stamp identical `episode_seed` and `cast_contract.cast_seed`; content-owned lanes retain an integer episode seed without fabricating a replay key; `OTR_CAST_SEED=N` pins the inline episode seed and downstream voice/music draws; different explicit seeds change downstream derived seeds; the cast replay golden remains unchanged. The existing tests cover only fragments: resolver entropy/override (`tests/test_cast_randomization.py:20-55`), replay parity (`tests/test_cast_voice_replay_parity.py:65-123`), content-owned stamping (`tests/test_fable2_tail_context.py:225-272`), and C7’s environment contract (`tests/test_audio_byte_identical.py:43-62`). None proves the missing inline writer-to-consumer handoff.

5. [Q1 / Q4] Disambiguate the second object already named `episode_seed`. `OTR_LedgerFreezeCascade` exposes an unwired output derived from the complete frozen ledger, not from the writer’s entropy (`nodes/OTR_LedgerFreezeCascade.py:62-70`, `tests/test_freeze_cascade_v2_ports.py:76-81`). State explicitly that this output is not the fix, remains unwired, and is not authoritative for audio RNG. Otherwise the plan invites a workflow-level patch that would create two incompatible episode identities.

6. [Verification] Add a canonical live acceptance leg: load `workflows/otr_canonical.json`, render one inline episode with an explicit seed and two unpinned inline episodes, verify the ledger keys, cast/announcer selections, music manifest seeds, `obs_publish OK`, and the canonical episode/OBS assets. Unit-only proof is insufficient under `docs/PRODUCTION_SPRINT_LESSONS.md:78-91`.

SHOULD-FIX:

1. [Q5] Narrow the music claim. Missing `episode_seed` does freeze the base RNG and therefore the opening/closing engine seeds because their seed keys are fixed (`nodes/stable_audio_theme.py:264-277`, `nodes/stable_audio_theme.py:336-378`). It does not prove identical music output or identical complete parameters: `compose_music_prompt` varies mood, setting, and period from episode metadata (`nodes/_otr_music_prompt.py:76-143`). Test seed variation separately from prompt variation.

2. [The defect, measured on published episodes] “Every voice draw” is too broad. Bark replay uses `cast_contract.cast_seed`; accepted hybrid voice decisions bypass the fallback draw; some announcer engines intentionally use a fixed sorted reference (`nodes/cast_lock.py:305-352`, `nodes/cast_lock.py:584-607`, `nodes/_otr_voice_bank.py:816-839`). Limit the claim to the seeded auto-registry and render-time fallback paths actually exercised by the published configuration.

3. [Q4] Document that `OTR_CAST_SEED` now acts as the inline episode-wide audio seed, not merely a cast control. It will govern cast, auto-registry voices, line synthesis, Kokoro fallback, and music through their domain-separated derivations (`nodes/OTR_LedgerScriptWriter.py:1347-1367`, `nodes/_otr_voice_bank.py:316-331`, `nodes/stable_audio_theme.py:264-277`).

4. [The defect, measured on published episodes] Add the production incident to `docs/PROD_BUG_LOG.md` after the live fix and promote the reusable frozen-RNG/receipt-ownership rule with executable coverage. The incident already meets the published-artifact admission bar and aligns with Bug Bible `10.07` and `12.51` (`comfyui-custom-node-survival-guide/BUG_BIBLE.yaml:746-756`, `:4122-4146`).

5. [Assumption surface] [ASSUMPTION] The voice pool and slot scoring will yield useful diversity once the seed varies. Replace that assumption with a deterministic multi-seed distribution check per stable slot shape; do not require every episode to differ, but prohibit the current one-value collapse.

OPTIONAL / NICE-TO-HAVE:

- Stamp or log an explicit episode-seed owner/source such as `legacy_cast_seed` versus `content_owned_seed`; avoid duplicating the numeric receipt unless a consumer needs it.
- Add a forensic assertion that downstream derived voice/music seeds can be recomputed from the persisted episode seed.

CUT THESE (scope / over-engineering):

1. [Q1 candidate (b)] Cut the tail-wide second entropy draw. It violates the stated single-owner rule and makes one published seed insufficient to replay cast plus audio.

2. [Q5] Cut any music-quality rerender or prompt retuning from this fix. Only seed propagation and deterministic seed receipts are in scope.

3. [Every consumer] Cut changes to `coerce_int_seed` or consumer-side missing-seed fallbacks. Old/minimal ledgers still need deterministic compatibility; the root defect is the writer failing to stamp new inline ledgers.

4. [Workflow] Cut any new node, widget, socket, or wiring change, including wiring `OTR_LedgerFreezeCascade.episode_seed`. The existing canonical ledger path already carries the metadata through Writer → FreezeCascade → CastLock → voice/music nodes (`workflows/otr_canonical.json`; `tests/test_freeze_cascade_v2_ports.py:76-86`).
