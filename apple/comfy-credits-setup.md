# Comfy Credits remote LLM — setup

The **Comfy Credits** lane lets the writer's creative and/or technical slot
run on a frontier model billed to your **ComfyUI account credits** instead of
your own API key. It is the sibling of the [own-key OpenRouter lane](openrouter-setup.md):
ComfyUI's credit-billed text path *is* the OpenRouter partner node, so both
lanes expose the same frontier catalog — they differ only in **who pays**.

- **OpenRouter lane** → billed to your `OPENROUTER_API_KEY`.
- **Comfy Credits lane** → billed to the prepaid credits of the Comfy account whose API key you sign in with.

The lane is **off until you pick it**: leave both selectors on a local model
and nothing changes -- the offline baseline and the byte-identical audio path
are untouched. There is no enable flag (removed 2026-09-19); the pick plus the
queue's Comfy API key is the whole switch.

## Enable it

1. **Sign in to ComfyUI with a Comfy API key.** Create the key at
   <https://platform.comfy.org> on the account whose credits should pay (the
   sign-in dialog's own "Need an API key? Get one here" link goes there).
   Already signed in with Google or email? Sign out, sign in again, choose
   **Comfy API Key** and paste the key. Then `Settings → Credits` to top up
   (prepaid -- no surprise charges). A key works on any host, `localhost` or
   not. **That sign-in is the only credential the pack reads** -- the same
   `api_key_comfy_org` hidden input ComfyUI's own partner nodes use. The pack
   keeps no key file and reads no environment variable on the server
   (rip 2026-09-19: three credential sources in two resolution orders was a
   defect, not a convenience). It does **not** request the logged-in session
   token, because the Comfy Registry security scan flags any third-party pack
   that declares that hidden input (2026-09-02). A plain email / Google
   sign-in without an API key therefore does not enable this lane.

   **Why a Google / email sign-in cannot be accepted (re-verified
   2026-10-01).** A Google-signed-in Comfy Desktop user with credits ran
   `otr_cloud_low_1act.json` on 2026-09-30 and was refused; this is what was
   checked before deciding the refusal, not the pack, had to change:
   * *The registry evidence.* `GET https://api.comfy.org/nodes/comfyui-old-time-radio/versions?include_status_reason=true`
     still shows `2.0.0-alpha.13` to `.15` with two `pylint-scanner`
     findings of severity `critical`, type `prohibited-string`, text
     "Prohibited string detected" naming the session-bearer type string, on
     the writer's `hidden` dict. The `api_key_comfy_org` line beside it was
     not flagged, and every version since alpha.30 (which declares only the
     key) has passed the automated scan, apart from unrelated findings.
   * *How ComfyUI hands the session over.* ComfyUI's `execution.py`
     (`get_input_data`) gives the signed-in session to a node only when its
     hidden inputs declare it: a V1 node by the type string, a V3 node by
     `io.Hidden.auth_token_comfy_org`, which a V3 schema with
     `is_api_node=True` adds automatically. No core helper makes the
     authenticated call for a custom node, and Comfy's own partner nodes
     (`comfy_api_nodes/util/_helpers.py`) read it from their own hidden
     inputs. OTR's in-process partner calls supply those inputs themselves,
     so they would still need the session from OTR code. The app always
     sends both `extra_data` fields; a Google / email sign-in fills only the
     session one.
   * *What was rejected.*
     * Building the type string at runtime: that is deliberate evasion of a
       registry security rule, and the registry also runs admin code reviews.
     * The V3 enum or `is_api_node`: no flagged string appears, but it takes
       exactly what the critical prohibits. Whether Comfy-Org allows it for
       third-party packs is unresolved; asking on the registry-backend
       tracker would settle it.
     * Reading the server's `extra_data` directly: the same evasion.
   * *The one sanctioned route,* recorded for later: let Comfy's own partner
     nodes make the call, on the canvas or through node expansion, so the
     session never reaches OTR code. The writer's sequential, data-dependent
     calls and the per-beat media calls make that a rewrite of every cloud
     lane, not a fix.
   * *What changed instead.* Every cloud lane now refuses with one shared
     text (`NO_CREDENTIAL_HINT` in `nodes/_otr_shared/cloud_media_backend.py`,
     which the Comfy Credits writer reuses). It explains why the sign-in is
     not enough and gives the two steps above, plus the comfy-cli
     `COMFY_API_KEY` route. The queue-time balance warning says the same in
     one line. The text is static, so no credential can reach an error
     report.

   **Headless (no app sign-in):** put the key in `OTR_COMFY_API_KEY` in the
   environment of the machine that *submits* the prompt and submit through
   `scripts/otr_api.py`. The submitter sends it as
   `extra_data.api_key_comfy_org` and ComfyUI hands it to the workflow's
   **0 - Comfy Credential** node exactly as the app's sign-in would; that node
   passes it to the rest and never lets it reach an error report. The ComfyUI server itself never reads
   that variable.

   **Headless with comfy-cli:** `comfy run --workflow workflows/otr_cloud_low_1act.json`
   against your local ComfyUI does the same thing when the key is in
   `COMFY_API_KEY` (or passed as `--api-key`): comfy-cli puts it in
   `extra_data.api_key_comfy_org` and converts the saved workflow to API format
   itself (checked against comfy-cli source, 2026-10-01). **Do not also be
   signed in with `comfy cloud login`:** a browser session takes precedence and
   comfy-cli then sends the session token instead of the key, which this pack
   does not read -- the run stops with "carries no Comfy API key".
2. On the **1. Story Writer** node, two pickers are always present:
   `comfy_slot_a_model` (creative) and `comfy_slot_b_model` (technical). Pick a
   model in each. Then set `creative_writing_model` to **`comfy:slot-a`** and/or
   `technical_model` to **`comfy:slot-b`** to route that slot through Comfy
   Credits. Leaving the selector on a local model id keeps that slot local.

The pickers lead with a **`(enable Comfy Credits)`** sentinel row -- the
placeholder value older saved graphs carry -- followed by the pinned catalog.

## Recommended defaults

| Slot | Default slug | Why |
|------|--------------|-----|
| creative (`comfy_slot_a_model`) | `anthropic/claude-sonnet-5` | Cheap-cloud story pass; native Sonnet 5 cannot turn reasoning off, so the lane sends `reasoning_effort=low` |
| technical (`comfy_slot_b_model`) | `openai/gpt-5.6-luna` | JSON / bookkeeping with `reasoning_effort=none` on the OpenRouter proxy (Credits widget label `off`) |

Shipping cheap Comfy Cloud graphs pin that pair (1-act / 3-act / 5-act). Deluxe pins `openai/gpt-5.6-sol` on creative and the same Luna on technical. The combo also lists Terra, the `-pro` twins, Grok 4.20, GPT-5.5, and Claude Opus 4.7 so older saved graphs still load. Do not add `~*-latest` aliases — Credits rejects them.

Override per slot without changing the pick via
`OTR_COMFY_SLOT_A_DEFAULT` / `OTR_COMFY_SLOT_B_DEFAULT`. The full pinned catalog
lives in `nodes/_otr_comfy_backend.py` (`COMFY_LLM_MODELS`) — bump it when
ComfyUI's partner catalog changes.

## Knowing which model ran

The resolved slug is surfaced three ways: the widget tooltip names it, a
`[ComfyCredits] … → <slug>` line is logged at resolution, and the resolved
public slug is stamped into run meta (the auth token is never logged or
stamped).

## Cost guards

Prepaid credits cap spend account-side, and before the first credit moves OTR
checks your Comfy balance against a deliberately high estimate of the episode
and refuses to start if it cannot cover it. There are no token caps once a run
starts — no per-call or per-episode ceiling and no output cap — because a
mid-run cap throws away everything the run had already paid for. Each call logs
the tokens the provider actually billed, with a running total for the episode.

A failed call **aborts the run** with a clear error — there is no mid-episode
fall-back to a local model and no silent remote→remote swap.

## First-run endpoint check (operator)

The Comfy proxy surface is env-overridable so you can repoint it without a code
change. The defaults were verified 2026-06-01 against the bundled partner-node
source (`comfy_api_nodes/nodes_openrouter.py` + `util/_helpers.py`):

- `OTR_COMFY_API_BASE` (default `https://api.comfy.org`)
- `OTR_COMFY_CHAT_PATH` (default `/proxy/openrouter/api/v1/chat/completions`)

If the first live run fails with a clear "confirm OTR_COMFY_API_BASE /
OTR_COMFY_CHAT_PATH" error, point these at the live Comfy proxy and re-run. The
lane is isolated behind the provider seam, so a mismatch degrades to that error
— it never crashes the writer.
