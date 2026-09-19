# Cloud balance preflight (queue time)

`OTR_WorkflowValidator` runs three $0 gates before any node spends:

1. `ensure_prompt_cloud_slugs` -- is every picked model live?
2. `ensure_prompt_cloud_balance` -- can each wallet pay the worst case? (this doc)
3. `ensure_prompt_visual_assets` -- are the local weights on disk?

Module: `nodes/_otr_shared/cloud_balance_preflight.py`. Wired in
`nodes/_otr_workflow_validator.py::_queue_time_readiness_gates`. No widget, no
canonical JSON change; every `workflows/**/*.json` already carries the validator.

## Three wallets, never pooled

| Wallet | Spends when | Remaining query | Unit |
| --- | --- | --- | --- |
| `comfy` | `comfy:slot-*` writer; any `cloud_*` / `sonilo` / `ideo` engine | `GET https://api.comfy.org/customers/balance`, bearer from `cloud_media_backend.resolve_auth` (`OTR_COMFY_API_KEY` or pack key file) | `*_micros` are CENTS: `remaining_usd = effective_balance_micros / 100` (falls back to `amount_micros`). Display credits = USD x 211 (`ComfyUI_frontend` `comfyCredits.ts`). Never `$0.01/credit`. |
| `openrouter` | `openrouter:slot-*` writer | `GET https://openrouter.ai/api/v1/key` with the generation key; `data.limit_remaining` when not null. Uncapped key -> `GET /api/v1/credits` (`total_credits - total_usage`). A 401/403 there is a warn, never a refusal. | USD |
| `google` | `google_api:slot-*` writer; `google_*` engines | **None.** The Gemini API key has no remaining-dollar endpoint. Estimate and log only. No Cloud Billing code. | USD (estimate) |

Live check 2026-09-18: Comfy returned `$77.69` (~16,392 credits); OpenRouter `/key`
was uncapped and `/credits` answered with the generation key (`$52.16`).

## Estimate (deliberately high -- the script does not exist yet)

| Line | Count | Unit price |
| --- | --- | --- |
| paid video engine | `max(OTR_VideoRenderBatch.beats, 40)` clips per distinct engine | adapter `_estimated_usd` on a stub request: 4 s clips for a 1-act, 8 s otherwise (the Veo menu); LTX 2.5 -> `ltx25_estimated_usd` |
| paid stills | `max(OTR_ImageDirector.fresh_cap, 15)` x image roles not dropped by no-still pairing | adapter `_est_usd` (`ideo` -> speed price map) |
| `sonilo` | 4 cues | `estimate_music_usd(OTR_SONILO_MIN_DURATION_S, default 30 s)` |
| `cloud_elevenlabs` | `act_count` (default 3) x 4,000 chars | `estimate_tts_usd` |
| writer | one line per wallet at that backend's per-run token ceiling (`OPENROUTER_MAX_TOKENS_PER_RUN` 300k, `OTR_COMFY_MAX_TOKENS_PER_RUN` 1M, Google 300k) | OpenRouter cache `pricing` (max of prompt/completion) when present, else a $5/Mtok floor |

Then `needed = sum + max(15% of sum, $1.00)`. `replay_from` skips the writer
line and still prices the media. A local-only graph opens no wallet and makes
no request.

## Refuse vs warn

- `remaining < needed` -> **refuse** (`ValueError`, same channel as the slug gate). The
  message names the wallet, needed, remaining, and the top three cost lines.
- Balance GET fails (transport or non-200) on `api.comfy.org` or on OpenRouter `/key`
  -> **refuse**: the host we are about to charge cannot be verified.
- OpenRouter `/credits` unreachable after an uncapped `/key` -> **warn**.
- No headless Comfy credential at queue time (app login only) -> **warn**; the login is
  injected at execution, so the balance is simply unverified.
- Google -> **warn** with the estimate, always.
- A paid engine with no price function: wallet remaining `0` -> refuse; `> 0` -> warn.

## Tests

`tests/test_cloud_balance_preflight.py` -- socket-free, injectable balance getters
and price tables; also pins the validator call order and that every shipping JSON has
exactly one validator.
