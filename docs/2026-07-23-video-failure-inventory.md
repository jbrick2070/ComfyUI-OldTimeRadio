# 2026-07-23 Video Qualification Failure Inventory and Fix Plan

**Status:** grounded inventory; GPU campaign aborted on operator direction;
receipt truth is fixed in the working tree and focused-green; live
requalification remains pending. The attempted KIBITZ panel was incomplete
because AGY was quota-held and the requested Claude lane was omitted.

**Scope:** live canonical OTR runs on 2026-07-23, using local `google/gemma-4-12b-it`
in both writer slots. The 120-word six-bank viz campaign completed its six legs;
the 45-word local video-model matrix has nine terminal failures recorded; the
Wan lane is terminal and the later LTX-style sweep was aborted at case 6/54.

## Evidence

- Canonical graph: `workflows/otr_canonical.json`.
- Six-bank 120-word run: `tmp/six_bank_viz_120_20260723_20260723_011138/`.
- 45-word matrix: `tmp/overnight_comfy_campaign_20260723_012809/` (the GPU
  campaign was aborted after the Wan lane).
- Matrix definition: `tmp/_run_canonical_engine_matrix_20260723.py`.
- Coordinator: `tmp/_overnight_comfy_campaign_20260723.ps1`.
- The 120-word campaign used `scripts/otr_canonical_api_run.py` through the
  temporary PowerShell launcher and recorded stdout, stderr, exit code, and queue
  state per leg.

## Terminal failure inventory

| Case | Observed failure | Classification | Existing coverage/status |
|---|---|---|---|
| `still_flat__qwen_image__scifi_news` | Sci-Fi Codex P3 exhausted the remaining provider/context capacity: 5,693 output tokens after a 2,499-token prompt. | Open provider-capacity/liveness defect at a short target; distinguish from the existing P5 transport and P0 fact-span defects. | Related open capacity records exist, but this exact P2/P3 short-leg surface is not closed by a live proof. |
| `still_motion__z_image_turbo__scifi_news_pro` | `scifi_fable2` script markup ladder exhausted after four attempts; residual `BAD_LINE_SHAPE` rows contained parenthesized sound directions. | Open structural authored-output repair/liveness defect in the `scifi_news_pro` route. | No matching current open entry found for this exact live failure. |
| `still_word__flux2_klein__original` | `still_word` refused `shot_music_opening_001`: required base still/init image was absent. | Open still-spine/image-ledger integration defect; fail-closed video behavior is correct. | Existing PBUG-20260712-19 covers upstream still-consumer policy, but does not close this missing per-beat mint. |
| `mesh_stage__lumina_image__public_domain` | `mesh_stage` had no mesh fodder for the opening beat, then the request lacked required `init_image`. | Open mesh-fodder/image-phase integration defect; no scene-still substitution is correct. | Source contains loud missing-fodder protection and tests, but no live closure for the opening-object mint. |
| `word_razzle__qwen_image__shakespeare` | Historical run selected the unprovisioned `qwen_image` adapter and failed its asset gate. | Closed by removing the optional Qwen-Image engine from the registry and future matrix; retained as historical evidence, not a renderer regression. | Engine-specific adapter/smoke tests and selection surface removed; no fallback or replacement claim. |
| `ltx_8gb__z_image_turbo__scifi_news` | Sci-Fi Codex P2 exhausted provider/context capacity: 7,128 output tokens after a 1,064-token prompt. | Same provider-capacity family as the first case, at P2 and a different source payload. | Existing capacity follow-ups are not verified against this 45-word surface. |
| `ltx23_16gb_video__flux_gen1__scifi_news_pro` | `ltx_video` HQ two-stage recipe required an on-disk init image for every beat and received `''`. | Same upstream still-spine/image-ledger defect as `still_word`; LTX’s strict input gate is correct. | Existing render-driver tests cover missing still behavior, but the live image phase still emitted a deficient ledger. |
| `humo_1.7B__qwen_image__public_domain` | Historical run selected the unprovisioned `qwen_image` adapter and failed its asset gate. | Closed by removing the optional Qwen-Image engine from the registry and future matrix; retained as historical evidence. | Engine-specific adapter/smoke tests and selection surface removed. |
| `wan_i2v__flux2_klein__scifi_news_pro` | Freeze rejected `shot_004_b2`: voiced line cleaned to empty spoken text; `freeze_verdict=needs_full_rerun`; CastLock correctly refused render. | Open residual spoken-safety/repair-liveness defect, not a Wan renderer defect. | Existing BUG-LOCAL-276 gate is working; this live case needs route-level repair triage. |
| `wan_8gb__lumina_image__media_archive` | Terminal `FAIL` at `OTR_VideoRenderBatch`: `wan_ti2v` received a 177-frame request and the cost model allowed 30 frames at the observed free VRAM; no silent resize was performed. | Open low-VRAM profile/launch-contract defect. The requested 832x480/17-frame lane was not applied to this matrix leg, so this is a failed configuration qualification, not evidence against the Wan weights. | `model_coverage_wan/receipts.json` and `server_wan.log`; rerun only after the profile pins canvas and frame budget. |

## Receipt-integrity defect

The six-bank 120-word campaign log says `6/6 PASS`, but
`leg05_scifi_news_120w.stdout.log` contains:

```text
[canonical-api] RESULT FAIL prompt_id=cde10c6d-3b70-4732-8179-4b18c8bcd933
```

The same leg’s `receipts.json` records `status=PASS`, `exit_code=0`, and the
campaign completes `6/6 PASS`. The failure detail is a live Sci-Fi Codex P0
fact-span validation error. The campaign wrapper therefore could not use its
current receipt as qualification evidence. The wrapper is now fixed in the
working tree: it requires an explicit terminal `RESULT SUCCESS` sentinel, a zero
exit code, and an empty queue, and records contradictory evidence. The helper
regression feeds the captured `RESULT FAIL` stdout with exit code zero and
requires a failed receipt. A live six-bank requalification is still required.

## Non-terminal warning inventory

The successful LTX/HuMo logs still contain repeated `MISSING-STILL (LOUD)` and
`LTX-I2V MISSING-STILL` warnings for bookend and scene beats. These are not benign
coverage noise: the code explicitly says the condition should be unreachable after
the still-spine coverage fix. A successful composite can therefore hide a degraded
text-only or dark-floor shot. The fix must assert the image ledger’s required
per-beat targets before video dispatch and attach a complete target/minted-path
receipt to the image phase.

## Proposed fix sequence for KIBITZ

1. **Receipt truth first — fixed offline.** Harden the reusable headless campaign observer/launcher
   so explicit API terminal status is authoritative and a contradictory stdout,
   history, or exit-code combination is recorded as a harness failure. Re-run the
   120-word six-bank campaign after the fix; require six terminal SUCCESS results,
   six `obs_publish OK` lines, six episode-root finals, and six OBS finals.
2. **Still-spine ownership.** Trace one failing opening beat from
   `OTR_MetaBriefImagePromptGen` through `OTR_ImageGenDispatcher` into the ledger,
   then into `_otr_video_engines/render_driver.py`. Make the image phase own every
   required scene/mesh-fodder/radio-face target for the effective engine and fail
   before video if a target is absent or unmaterialized. Preserve the existing
   no-fallback and no-scene-as-mesh-fodder rules. Cover `still_word`, `mesh_stage`,
   `ltx_video`, and `ltx_audio_in` as siblings.
3. **Short-leg provider capacity.** Reproduce the two 45-word Gemma failures with
   their captured prompt/source inputs. Measure the exact prompt plus full output
   reservation at P2/P3. Fix the shared capacity/structured-call contract, not by
   raising a target, clipping prose, or weakening structural validation. Preserve
   the truthful terminal disposition and keep the fix independent of the P0
   fact-span repair.
4. **SciFi News Pro markup repair.** Reproduce the two exact `BAD_LINE_SHAPE`
   rows and audit the four-attempt ladder’s prompt, repair boundary, and route
   ownership. The repair must convert eligible stage-direction-shaped lines into
   valid spoken text or return a truthful terminal failure; it must not silently
   route to a legacy fallback or mutate unrelated lines.
5. **Freeze residual safety.** Reproduce `shot_004_b2` from the WAN prompt’s ledger
   and identify why the safety repair left an empty voiced row after cleanup. Keep
   BUG-LOCAL-276 fail-closed; repair the producer/repair ownership upstream and
   prove CastLock receives `frozen_clean` or `frozen_with_warns` only after the
   spoken ledger is actually nonempty.
6. **Qwen readiness (closed by removal).** The optional unprovisioned
   Qwen-Image engine was removed from the registry, smoke, and engine-specific
   tests on 2026-07-23. Historical failures remain in this inventory; no future
   matrix case may select that engine, and no fallback is claimed.

## Verification gate

After any code change: focused regression tests, full Windows suite, Bug Bible
regression, canonical workflow validator/round-trip/link-widget audit, then a
selective fresh headless run. A fix is not closed until the relevant live case
returns `RESULT SUCCESS`, the server logs `obs_publish OK`, and the expected file
exists under the ledger-owned episode root plus `otr/obs`.
