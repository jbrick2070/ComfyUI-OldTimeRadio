<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. S0/S1/S3 dependencies are ordered wrong, registration/gating contracts conflict, and video compatibility will allow invalid cloud engines into talking-beat slots unless extra validation is added.

MUST-FIX BEFORE BUILD:
1. [2 + 5] Registration contract contradicts itself: [2] says cloud adapters “always register” because the registry is the menu and saved COMBO values must remain valid; [5] says “a missing class DROPS the row at registration.” Dropping a row breaks the [2] saved-workflow invariant. Concrete fix: always register the row, but mark it unusable at resolve/prepare time with a named error when the pinned class is absent or schema-mismatched; do not remove it from the registry after release. If pre-release S0 pinning fails, block promotion instead of shipping the row.

2. [8 S0 + 7 VIDEO + 5 + 6] S0 smoke #2 requires a working Kling audio-driven clip before S1/S3, but the Kling adapter, upload semantics, video canonicalizer, reactivity matrix, and audio-ref request wiring are otherwise placed in S3. That smoke cannot run from “control plane skeleton” alone. Concrete fix: either move Kling smoke #2 to the S3 promotion gate, or explicitly pull a minimal `cloud$ kling_avatar` adapter, audio upload, polling, canonicalization, and cache write path into S0.

3. [8 S0 + 7 STILLS] S0 smoke #1 says “headless auth injection end-to-end on one cheap image node,” but still-image row implementation is scheduled for S1. Concrete fix: either make the S0 smoke use a backend-only mock/live partner node that is not an image-registry row, or pull one minimal cheap image adapter into S0.

4. [7 VIDEO + role_compat.py] The planned reactivity matrix is not enforceable through the current video registry contract. Grounding `role_compat.engine_fits_role()` only checks `required_inputs <= ROLE_AVAILABLE_INPUTS`, and all three roles currently expose `text_prompt`, `init_image`, `audio_ref`, and `base_clip_ref`. Therefore a `mute_only` engine such as `cloud$ wan_i2v` with `required_inputs=("text_prompt","init_image")` will capability-fit announcer/talking roles unless another validator blocks it. Concrete fix: add a beat/role reactivity validator in `OTR_ShotLock` or `VideoRenderBatch` that rejects `mute_only` / `optional_audio_ref` for talking/announcer beats requiring `required_audio_ref` or `lipsync_overlay`; include `reactivity` and `must_strip_audio` in `descriptor_for_engine()` or a parallel single-source descriptor, and test it.

5. [7 VIDEO + role_compat.py] `lipsync_overlay` creates an unstated render dependency on `base_clip_ref`. Grounding says `base_clip_ref` is an input token, but the plan does not sequence creation of the base clip before invoking a lip-sync overlay node. Concrete fix: split video execution into two paths: direct audio-driven avatar from `init_image + audio_ref`, and overlay path that first generates or retrieves a mute base clip, then invokes lip-sync with `base_clip_ref + audio_ref`; include the base-clip content hash in the overlay cache key.

6. [3 + 8 S4] [3] says fallback behavior depends on “under `--profile cloud`” and the active profile enable-set, but the `cloud` profile and default-override map are not created until S4. That makes S0/S1/S2/S3 fallback semantics depend on a later sprint. Concrete fix: move the minimal cloud profile enable-set needed by the fallback resolver into S0, or remove `--profile cloud` behavior from [3] until S4 and make earlier sprints explicitly manual-engine-only.

7. [2 + registry.py audio/image/video] Gating is specified “at the profile resolver,” but the shown registries do not enforce it uniformly: audio `assert_usable()` explicitly has “NO GATED_BY_FLAG case”; image/video registry grounding shows dispatcher/ShotLock validation via registry compatibility, not profile resolution. [ASSUMPTION] If dispatch can call adapters after only `assert_usable()`, cloud rows will bypass `OTR_ENABLE_COMFY_CLOUD_MEDIA`. Concrete fix: name and wire the exact profile-resolver call before every cloud adapter execution path: audio voice/music dispatch, `OTR_ImageGenDispatcher`, and `OTR_VideoRenderBatch`/`OTR_ShotLock`. Add tests proving flag-off selected cloud rows raise `EngineUnusable(GATED_BY_FLAG)` for audio, image, and video.

8. [4 + 8] Ledger is append-only JSONL while adapters can run concurrently under per-provider semaphores. Concurrent JSONL appends from multiple worker threads can interleave or reorder status transitions. Concrete fix: put ledger writes behind a single writer queue or a process/thread lock; include monotonic sequence numbers per `request_id` so ESTIMATED/ACTUAL/FALLBACK records are reconstructable.

9. [4 + 5] Cache-key contract requires “cloud requests always carry an explicit seed,” but [5] admits node schemas are discovered live and candidate partner nodes may not expose a seed input. If a provider ignores or lacks seed, the cache key implies reproducibility the invocation contract cannot provide. Concrete fix: S0 schema pinning must record whether each row supports seed; rows without seed support either fail registration/promotion, use a provider-supported deterministic parameter, or mark `determinism="provider_nondeterministic"` and remove seed from the reproducibility claim.

10. [3 + _otr_comfy_backend.py] Media budget is separate from the existing Comfy Credits text backend. Grounding `_otr_comfy_backend.py` has its own token ceilings and module-global `_run_token_total`; [3] introduces a USD media budget only inside `CloudMediaSession`. [ASSUMPTION] A run can use Comfy Credits for both writer/chat and media, drawing from the same Comfy account credits. Concrete fix: either explicitly scope `OTR_CLOUD_MEDIA_BUDGET_USD` to media-only and report combined Comfy spend separately, or integrate Comfy text/backend estimated spend into the same run ledger/budget ceiling.

11. [5 + 8 TESTS] `partner_nodes.yaml` is generated “FROM THE LIVE INSTALL,” but no build artifact ordering is specified. Fresh installs/imports need the pinned YAML before adapters decide schema/class availability. Concrete fix: make `partner_nodes.yaml` a committed/generated artifact produced before adapter import in S0, with a clear failure mode if absent; CI should fail schema-drift, not runtime registration.

SHOULD-FIX:
1. [3 RATE LIMITS] `OTR_CLOUD_MAX_CONCURRENCY_<PROVIDER>` does not define provider-name normalization. Providers contain mixed casing/brands (`ElevenLabs`, `BFL`, `ByteDance`, etc.), so env vars will drift. Concrete fix: define canonical provider keys, e.g. `ELEVENLABS`, `STABILITY`, `SONILO`, `KLING`, `BYTEDANCE`, and use one normalization function for semaphores, pricing, ledger, and env lookup.

2. [3 FALLBACK POLICY] Retry handling does not mention HTTP `Retry-After` or provider quota/rate-limit payloads. Concrete fix: map 429/quota responses to `retryable_transport` only when retryable, honor `Retry-After`, and classify hard quota/insufficient-credit as `provider_rejected` or `budget` without burning retries.

3. [6 VIDEO + 10 verify #8] ffmpeg is only a verify item, but video canonicalization depends on it for `must_strip_audio`. Concrete fix: make ffmpeg availability part of adapter/profile `assert_usable()` for any row with `must_strip_audio=True`, not a late canonicalizer failure after credits are spent.

4. [6 voice/music] “Loudness normalized to the SAME reference the local lane produces” is still unresolved in verify #11. Concrete fix: block S2 implementation until the exact local loudness function/stage is identified and reused; do not create a separate cloud-only LUFS target.

5. [3 AUTH] Auth precedence says `explicit env > server config > logged-in token`, but the hidden-input names are injected into several nodes. Concrete fix: centralize this in one auth broker used by headless runner and UI nodes, and test that no node logs or serializes `auth_token_comfy_org` / `api_key_comfy_org`.

6. [4] Global cache location is Windows-style `otr\cache\cloud_media\`. Concrete fix: define it via `Path("otr") / "cache" / "cloud_media"` and document whether it is repo-relative, user-cache-relative, or Comfy output-relative.

7. [4] Hardlink/copy into episode paths needs cross-volume fallback. Concrete fix: attempt hardlink, verify same-device constraints, then copy atomically with SHA-256 validation.

8. [7 VIDEO] The document uses non-grounded role labels like “music_video + other_beats b-roll.” Grounding `role_compat.py` only recognizes `announcer_visual`, `music_visual`, and `character_video`; `scene_broll`/`background_abstract` were ripped. Concrete fix: express the video matrix only in terms of the three current role tokens and beat categories mapped to them.

9. [5 WATCHDOG] “progress heartbeat every poll tick” depends on an unspecified concrete progress API and is verify item #9. Concrete fix: S0 must choose the actual progress callback/event surface before implementing job-backed polling; otherwise long partner jobs risk the 5-minute stall detector.

OPTIONAL / NICE-TO-HAVE:
- [8] `otr-cloud-doctor` is useful but should remain non-blocking until the core auth/budget/cache path is stable.
- [8] Dry-run manifest mode is useful after cost quoting is stable; do not let it block S0 smoke fixes.

CUT THESE (over-engineering):
1. [3 CANCELLATION] Keep cancellation cut for v1. Do not add provider-specific cancel hooks until at least one provider documents a reliable cancel endpoint and refund behavior; logging `ORPHANED_JOB` is sufficient for the first build.

2. [9 Appendix B] Keep Surface B entirely out of this build. It introduces a second lifecycle, flag, polling model, and workflow-submission auth path before Surface A is wired.

3. [8 Nice-to-have dry-run manifest] Safe to cut from S0 because the required safety controls are pre-submit budget reserve, fail-closed auth, cache validation, and live smokes. Dry-run can follow once pricing/schema versions stabilize.