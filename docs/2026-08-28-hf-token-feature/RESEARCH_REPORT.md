# A Hugging Face token story for ComfyUI

**Research report -- 2026-08-28**
**Scope:** ComfyUI Core, ComfyUI frontend/Desktop, ComfyUI-Manager, comfy-cli,
Comfy Cloud/API nodes, Hugging Face's current client behavior, representative
custom-node patterns, and Old Time Radio (OTR).

## Executive verdict

The operator's instinct is right, with one important qualification: ComfyUI does
not currently ship a **dedicated, secure local Hugging Face credential service**,
but current ComfyUI Desktop does ship a generic per-install environment-variable
editor that can technically inject `HF_TOKEN`. Desktop itself warns that those
values are stored unencrypted and must not contain API keys or tokens, so it is
not the built-in secret feature users are asking for. Comfy Cloud has a managed
Secrets product and Core has special Comfy Account credentials for paid API
nodes, but neither exposes a Hugging Face token to an ordinary local custom node.
Two open, conflicting Core proposals are already working on authenticated model
downloads, so upstream participation should concentrate there rather than open a
duplicate issue or add an OTR-owned token store.

For node packs today, the honest best practice is to use
`huggingface_hub` and defer to its native credential resolution: a cached
`hf auth login` for an interactive workstation, or `HF_TOKEN` supplied by the
operator's secret manager/environment for a headless deployment. Never put a
token in a node, widget, workflow, pack config, sample, log, exception, or support
bundle. OTR is close to that posture but does **not** currently honor a cached Hub
login: its gated-model preflight checks only process environment/HKCU, and OTR
also relocates `HF_HOME`, which relocates the default token file. That is a real
root-cause gap and should be fixed before documenting `hf auth login` as the
preferred path.

## State of the world

| Surface | What exists on 2026-08-28 | Where the credential lives | Can OTR use it? |
|---|---|---|---|
| ComfyUI Core, local | No merged HF credential setting or authenticated model-download service. Two active proposals exist: [Core #14586](https://github.com/Comfy-Org/ComfyUI/pull/14586) and [Core #14657](https://github.com/Comfy-Org/ComfyUI/pull/14657). Both are open and conflicting at this snapshot. | Proposal-dependent; not a released contract. | Not today. |
| ComfyUI Desktop 1.0.46 | A generic Environment Variables launch setting can inject `HF_TOKEN`, but it is not HF-specific and the UI explicitly warns against storing tokens there. | Plain JSON in Desktop's per-install `installations.json`; visually masked fields are not encrypted. | Technically yes after restart, but it is not a recommended secret store. |
| Desktop built-in missing-model downloader | Gated HF downloads can reuse an Electron browser session after the user signs in and accepts terms. | Browser/session cookies; no provider token is exported. | No. It is scoped to Desktop's built-in download flow, not Python custom nodes. |
| Comfy Cloud Secrets | A first-party API Keys & Secrets UI includes Hugging Face/Civitai providers. | Cloud backend; reads return metadata, not plaintext. | No on local/Desktop/portable, and no sanctioned plaintext-read API exists for a custom node. |
| Comfy API-node auth | Comfy Account token/API key is carried to paid Partner/API nodes through hidden execution inputs and stripped from history. | Comfy Account/session state; an entered API key is persisted in frontend `localStorage`. | No. It authenticates to Comfy's proxy, not Hugging Face. |
| ComfyUI-Manager | No `HF_TOKEN`, `HF_API_TOKEN`, token field, or token-injection setting exists in current Manager. It supports `HF_ENDPOINT` for mirrors. | None. | No. |
| comfy-cli | `HF_API_TOKEN`, config key `hf_api_token`, and `--set-hf-api-token` exist for `comfy model download`. | Environment or plaintext comfy-cli `config.ini`. | No. comfy-cli does not inject the ComfyUI server process. |
| Hugging Face Hub client | `HF_TOKEN`, legacy `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`, and the token cached by `hf auth login`. | Operator environment/secret manager or Hub's per-user token cache. | Yes, once OTR stops rejecting cached-login-only authentication. |

## 1. First-party ComfyUI mechanisms

### Core: not merged, but two live proposals already occupy the design space

Current Core contains no released HF credential feature flag or model-downloader
auth service; the current [feature-flag module](https://github.com/Comfy-Org/ComfyUI/blob/0a33ed6c28f926d14536235771c222f9e6d1026b/comfy_api/feature_flags.py)
has no such facility. The important upstream activity is:

- [ComfyUI #14586](https://github.com/Comfy-Org/ComfyUI/pull/14586),
  **server-side model downloads + Hugging Face OAuth**, with companion frontend
  [#13062](https://github.com/Comfy-Org/ComfyUI_frontend/pull/13062). Both were
  open/conflicting at the snapshot, and the frontend change is inert without its
  backend. The design depends on a Comfy-Org-owned HF OAuth client.
- [ComfyUI #14657](https://github.com/Comfy-Org/ComfyUI/pull/14657), a broader
  **model downloader manager**, also open/conflicting. Its current PR head already
  resolves provider environment variables before OAuth
  ([resolver](https://github.com/Comfy-Org/ComfyUI/blob/a7b63915dcf1a259cb8ffb3370b078990dbe9e91/app/model_downloader/auth/resolver.py)),
  while its local token store explicitly characterizes its machine-bound XOR as
  obfuscation rather than confidentiality
  ([token store](https://github.com/Comfy-Org/ComfyUI/blob/a7b63915dcf1a259cb8ffb3370b078990dbe9e91/app/model_downloader/auth/token_store.py)).

This is prior art to revive and harden, not a blank space. Because both proposals
are unmerged and conflicting, their interfaces are evidence of direction, not an
API OTR can depend on.

### Desktop: a generic launcher setting, not a credential vault

The released Desktop source defines an
[Environment Variables field](https://github.com/Comfy-Org/Comfy-Desktop/blob/539df073165d63f56db95ebd96d462b8dde24108/src/main/sources/common/launchSettingsFields.ts#L329-L335).
At launch, Desktop merges the inherited `process.env` and then the per-install
user environment into the ComfyUI process
([launch path](https://github.com/Comfy-Org/Comfy-Desktop/blob/539df073165d63f56db95ebd96d462b8dde24108/src/main/lib/ipc/shared.ts#L985-L1005)).
That corrects an OTR README claim: Desktop does inherit the environment visible
to the Desktop process. What it cannot see is a user-scope variable added after
the already-running Desktop process inherited its parent environment, until the
app is restarted; the Windows HKCU read remains a useful compatibility bridge.

This field is not safe to present as the HF answer. Install settings are serialized
to [`installations.json`](https://github.com/Comfy-Org/Comfy-Desktop/blob/539df073165d63f56db95ebd96d462b8dde24108/src/main/installations.ts#L117)
as ordinary JSON
([write path](https://github.com/Comfy-Org/Comfy-Desktop/blob/539df073165d63f56db95ebd96d462b8dde24108/src/main/installations.ts#L222-L223)).
The current UI warning says values are stored unencrypted and tells users not to
add API keys or tokens
([English locale](https://github.com/Comfy-Org/Comfy-Desktop/blob/539df073165d63f56db95ebd96d462b8dde24108/locales/en.json#L1379)).
Sensitive-looking names are merely displayed as password fields
([masking heuristic](https://github.com/Comfy-Org/Comfy-Desktop/blob/539df073165d63f56db95ebd96d462b8dde24108/src/renderer/src/views/comfyUISettings/EnvVarsField.vue#L8-L10)).
Desktop [PR #590](https://github.com/Comfy-Org/Comfy-Desktop/pull/590) briefly
explored Electron `safeStorage`, but the released source and warning above are
the authoritative current behavior.

Desktop has a second, narrower mechanism: merged Desktop
[#1275](https://github.com/Comfy-Org/Comfy-Desktop/pull/1275) and frontend
[#13742](https://github.com/Comfy-Org/ComfyUI_frontend/pull/13742) let the built-in
missing-model flow open a trusted Hugging Face page in the same Electron session,
then retry after sign-in/terms acceptance. That cookie-based flow deliberately
adds no OAuth token store and gives no credential to an OTR Python process.

### Cloud Secrets: real managed secrets, deliberately not a local node API

Comfy Cloud exposes **API Keys & Secrets** with Hugging Face and Civitai providers,
but the official documentation limits the feature to eligible Comfy Cloud plans
and explicitly says it is unavailable in Desktop and portable
([Cloud model-import docs](https://github.com/Comfy-Org/docs/blob/47f2fbf2b118d055ea4701ae0cb73a624bc3d022/cloud/import-models.mdx#L67-L115),
[scope limitation](https://github.com/Comfy-Org/docs/blob/47f2fbf2b118d055ea4701ae0cb73a624bc3d022/cloud/import-models.mdx#L141-L142)).
The frontend sends a value only on create/update
([Secrets CRUD](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/platform/secrets/api/secretsApi.ts#L42-L83));
read responses contain metadata rather than the plaintext value
([response type](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/packages/ingest-types/src/types.gen.ts#L889-L921),
[write type](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/packages/ingest-types/src/types.gen.ts#L3439-L3459)).
That is a sound secret boundary, but it means there is no sanctioned way for an
ordinary local custom node to ask for the stored HF plaintext.

### API-node credentials: Comfy Account auth, not provider-key storage

Comfy's paid API-node system is a different credential plane. The frontend obtains
the active Comfy Account auth before queueing
([app handoff](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/scripts/app.ts#L1653-L1672))
and inserts `auth_token_comfy_org` or `api_key_comfy_org` into prompt `extra_data`
([request body](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/scripts/api.ts#L1049-L1085)).
Core removes those values from ordinary prompt metadata, holds them for execution,
and strips them from history/job responses
([server queue path](https://github.com/Comfy-Org/ComfyUI/blob/0a33ed6c28f926d14536235771c222f9e6d1026b/server.py#L1072-L1133),
[execution cleanup](https://github.com/Comfy-Org/ComfyUI/blob/0a33ed6c28f926d14536235771c222f9e6d1026b/main.py#L380-L403)).
V3 schemas marked `is_api_node=True` get corresponding hidden inputs
([schema injection](https://github.com/Comfy-Org/ComfyUI/blob/0a33ed6c28f926d14536235771c222f9e6d1026b/comfy_api/latest/_io.py#L1738-L1754)),
and first-party helpers attach them to `api.comfy.org`
([proxy helper](https://github.com/Comfy-Org/ComfyUI/blob/0a33ed6c28f926d14536235771c222f9e6d1026b/comfy_api_nodes/util/_helpers.py#L36-L75)).
The official contract describes a Comfy API key, not arbitrary provider secrets
([API-key integration docs](https://github.com/Comfy-Org/docs/blob/47f2fbf2b118d055ea4701ae0cb73a624bc3d022/development/comfyui-server/api-key-integration.mdx)).

An API key entered manually is password-masked but persisted in frontend
`localStorage` under `comfy_api_key`
([masked form](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/components/dialog/content/signin/ApiKeyForm.vue#L31-L56),
[store definition](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/stores/apiKeyAuthStore.ts#L15-L20),
[persistence](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/stores/apiKeyAuthStore.ts#L69-L104)).
None of this is a supported HF-token read/store service. OTR should not label
itself an API node merely to receive or repurpose a user's Comfy Account auth.

### Manager and comfy-cli: two products that were easy to conflate

The problem statement's claim that Manager injects an HF token does not survive
inspection of current Manager. There is no HF token key or flag. Manager documents
only `HF_ENDPOINT` for selecting a mirror
([Manager README](https://github.com/Comfy-Org/ComfyUI-Manager/blob/f39cbd56fecae0b27a446c0cd450cd591f3a8bea/README.md#L326-L346)).
Its downloader uses `HfApi().repo_info`, which may see ambient Hub credentials,
but retrieves the actual bytes with a bare `requests.get` and no bearer header
([download path](https://github.com/Comfy-Org/ComfyUI-Manager/blob/f39cbd56fecae0b27a446c0cd450cd591f3a8bea/glob/manager_downloader.py#L132-L155)).
That is not a gated-download credential mechanism.

The separate official **comfy-cli** does have one. Its namespace is
`HF_API_TOKEN` / `hf_api_token`, not the Hub-standard `HF_TOKEN`
([constants](https://github.com/Comfy-Org/comfy-cli/blob/929198f62d7641b2175f701a304bd562673a73c1/comfy_cli/constants.py#L52-L59)),
and the current flag is `--set-hf-api-token`
([model command](https://github.com/Comfy-Org/comfy-cli/blob/929198f62d7641b2175f701a304bd562673a73c1/comfy_cli/command/models/models.py#L367-L374)).
Its precedence is persisted command value, environment, then config
([config resolution](https://github.com/Comfy-Org/comfy-cli/blob/929198f62d7641b2175f701a304bd562673a73c1/comfy_cli/config_manager.py#L56-L74)).
It is scoped to `comfy model download`; it neither launches nor injects a running
ComfyUI server. Public history includes the introduction in
[#193](https://github.com/Comfy-Org/comfy-cli/pull/193), a request to adopt
standard `HF_TOKEN` in [#280](https://github.com/Comfy-Org/comfy-cli/issues/280),
and layered resolution in [#296](https://github.com/Comfy-Org/comfy-cli/pull/296).

The problem statement's ComfyDeploy observation is accurate but orthogonal:
ComfyDeploy documents a Hugging Face token in its hosted model-download settings
([ComfyDeploy download docs](https://docs.comfydeploy.com/docs/models/download)).
That credential belongs to a separate deployment service and private model
volume; it is not evidence of a local ComfyUI Core or custom-node credential API.

No merged public RFC or registry-backend credential contract was found in the
searched Comfy-Org snapshots. That is a bounded absence claim, not proof that
private work does not exist; the open Core/frontend PRs above are the concrete
public venue.

## 2. Hugging Face's native contract

As of `huggingface_hub` 1.29.0, `get_token()` recognizes the standard environment
variables and token file, with `HF_TOKEN` taking precedence over the cached value
([resolver source](https://github.com/huggingface/huggingface_hub/blob/4237d95c603db491cb1070898c74c97e4d7c2582/src/huggingface_hub/utils/_auth.py#L58-L83),
[environment branch](https://github.com/huggingface/huggingface_hub/blob/4237d95c603db491cb1070898c74c97e4d7c2582/src/huggingface_hub/utils/_auth.py#L145-L154)).
The documented modern login is `hf auth login`; it validates and caches the token
for Hub-aware libraries
([quick start](https://huggingface.co/docs/huggingface_hub/quick-start)).
`HF_TOKEN_PATH` controls the token-file location, while
`HUGGING_FACE_HUB_TOKEN` remains a deprecated compatibility alias
([environment-variable reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)).

When an SDK call uses `token=None`, the resolved credential is sent implicitly;
`token=True` requires an available credential, and `token=False` explicitly
disables authentication
([header construction](https://github.com/huggingface/huggingface_hub/blob/4237d95c603db491cb1070898c74c97e4d7c2582/src/huggingface_hub/utils/_headers.py#L125-L154)).
This is why a node pack does not need a token field. It should call the official
client and let the client resolve the operator's credential.

The cached login is a standard **file-backed credential**, not an OS-vault claim.
The Hub client creates the token directory/file with restrictive `0700`/`0600`
permissions on POSIX and documents the Windows permission limitation in code
([token-file write](https://github.com/huggingface/huggingface_hub/blob/4237d95c603db491cb1070898c74c97e4d7c2582/src/huggingface_hub/utils/_auth.py#L33-L48)).
Its advantage for a node pack is one provider-owned resolver and one per-user
credential location, rather than a copy in every workflow or pack; deployments
that require stronger custody should inject `HF_TOKEN` from their secret manager.

Authentication and authorization remain separate. A user must request/accept a
gated repository's terms in the browser
([gated-model documentation](https://huggingface.co/docs/hub/models-gated)).
Hugging Face recommends read or fine-grained tokens for downloads, separate tokens
per app/usage, managed storage rather than hardcoding, and rotation after exposure
([token security guidance](https://huggingface.co/docs/hub/main/security-tokens)).

## 3. What reputable custom-node patterns actually do

The sample is intentionally pattern-based rather than a popularity league table.
It covers current, inspectable implementations that exercise materially different
credential paths.

| Pattern and example | User friction | Security judgment |
|---|---|---|
| **Native Hub resolution, no pack UI.** `kijai/ComfyUI-HunyuanVideoWrapper` calls `snapshot_download()` without a token field ([example call](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper/blob/fcbd6729a9b0b8ff6037c598bbada4a6bdc6d967/nodes.py#L691-L724)). | One machine login or deployment secret works across Hub-aware tools. | **Best current pack pattern.** No second store and no workflow secret. |
| **Central first-party write-only secret UI.** Comfy Cloud demonstrates metadata-only reads; Core PRs explore local equivalents. | Friendly once configured. | Potentially strong if plaintext is never readable and OS-backed protection is used; not available to local packs today. |
| **Ordinary Comfy setting / generic Desktop environment field.** `jnxmx/ComfyUI_HuggingFace_Downloader` adds a normal setting and prefers it over `HF_TOKEN` ([setting](https://github.com/jnxmx/ComfyUI_HuggingFace_Downloader/blob/2bba5db6a52479e8ad465dbade19dd0da0784bd3/js/settings.js#L3-L13), [resolution](https://github.com/jnxmx/ComfyUI_HuggingFace_Downloader/blob/2bba5db6a52479e8ad465dbade19dd0da0784bd3/downloader.py#L150-L186)). | Convenient in-app entry. | Plaintext application state; visually hiding a value is not encryption. Creates another place to rotate and protect. |
| **Pack-local config file.** `tori29umai0123/ComfyUI-Model_Downloader` uses a local INI and manual bearer header ([config/download code](https://github.com/tori29umai0123/ComfyUI-Model_Downloader/blob/964ac384ca7d4f4cf4dbfdd681d963753b377eef/model_downloader.py#L20-L62)). | Pack-specific setup and troubleshooting. | Weaker: another plaintext store and custom HTTP/auth behavior. Gitignore reduces accidental commit risk but is not secret protection. |
| **Guarded pack settings endpoint.** `ideogram-oss/ComfyUI-Ideogram4` returns boolean configured-state rather than the value and sets POSIX mode `0600` ([routes/storage](https://github.com/ideogram-oss/ComfyUI-Ideogram4/blob/c05545d71e61b7ce47534a972eaeefd958a3719f/nodes.py#L136-L211)). | Good in-pack UX. | Better than a readable setting, but still a second plaintext store and POSIX permissions do not supply equivalent Windows vault protection. |
| **Token node/widget.** `ciri/comfyui-model-downloader` accepts `hf_token` as a node input ([node](https://github.com/ciri/comfyui-model-downloader/blob/c48061e234cb5bf0e17d82a88e546c049f8fa441/nodes/hf/hf_download.py#L5-L50)) and its example workflow carries the widget value ([workflow](https://github.com/ciri/comfyui-model-downloader/blob/c48061e234cb5bf0e17d82a88e546c049f8fa441/examples/workflows/hf-demo.json#L19-L25)). | Obvious, per-workflow setup. | **Worst pattern.** Workflow serialization turns a credential into shareable graph data. |

The verdict is therefore simple: **Hub-native cached login or a managed
`HF_TOKEN`, with no node field and no pack store.** Cached login has the least
ongoing workstation friction; environment/secret-manager injection is the right
headless pattern. A future centralized Comfy service can improve the UI, but a
pack-specific credential layer cannot.

## 4. Concrete security hazards and checklist

ComfyUI serializes normal widget values into the graph
([node serialization](https://github.com/Comfy-Org/ComfyUI_frontend/blob/512f72d5f7eee46ceb3c94526a01eb5979ab7df9/src/lib/litegraph/src/LGraphNode.ts#L1195-L1240)).
A password-style input only changes its appearance; it does not make the workflow
a secret store. Core execution failures can also include a node's current input
values
([execution errors](https://github.com/Comfy-Org/ComfyUI/blob/0a33ed6c28f926d14536235771c222f9e6d1026b/execution.py#L628-L712)).
The token-widget example above proves that this is not hypothetical ecosystem
behavior.

For OTR and any upstream downloader:

- [ ] Do not define a token node input, widget, workflow property, hidden prompt
      input, or sample value.
- [ ] Do not persist provider plaintext in `comfy.settings.json`, a pack INI/JSON,
      Desktop's current generic environment field, workflow backups, queue/history,
      or support bundles.
- [ ] Do not log token contents, bearer headers, request objects, subprocess
      argument/kwargs files, raw environment dumps, or unsanitized exceptions.
- [ ] Use `huggingface_hub` rather than hand-rolling requests, redirects, and auth.
      The Hub client itself redacts authorization in debug cURL output
      ([HTTP debug source](https://github.com/huggingface/huggingface_hub/blob/4237d95c603db491cb1070898c74c97e4d7c2582/src/huggingface_hub/utils/_http.py#L1066-L1110)).
- [ ] If upstream code ever performs manual redirects, send a bearer credential
      only to the exact intended HTTPS provider host and strip it before any
      cross-host hop.
- [ ] Ask for a read/fine-grained token only; never require write scope for model
      downloads. Rotate a token after any suspected workflow/log exposure.
- [ ] Report authentication failure and unaccepted repository terms as distinct
      states; never imply that pasting a token grants the model license.
- [ ] If a UI-managed store is built, expose set/delete/status, not plaintext
      readback, and use OS-backed secret protection where the platform provides it.

## 5. OTR audit and recommendation

### The current gap

OTR's intended non-store design is sound, but its implementation does not yet
honor the full standard it claims to honor:

1. The shared resolver checks only `os.environ["HF_TOKEN"]`, then Windows
   `HKCU\Environment\HF_TOKEN`
   ([shared resolver](../../nodes/_otr_shared/hf_token.py#L81-L114)). A second
   implementation repeats the same two-source logic
   ([duplicate resolver](../../nodes/_otr_hf_auth.py#L25-L47)). Neither calls
   Hugging Face `get_token()` or reads its configured token cache.
2. The gated catalog preflight raises before `snapshot_download()` whenever that
   two-source resolver returns `None`
   ([catalog preflight/download](../../nodes/_otr_model_catalog.py#L1811-L1895)).
   The later SDK call uses the resolver result as `token=`; `None` would normally
   allow the Hub SDK to resolve its cache implicitly, but the gated branch never
   reaches that call. A perfectly valid `hf auth login` token therefore never gets
   a chance to work for the gated curated model that needs it.
3. The engine-profile check itself reads only `os.getenv("HF_TOKEN")`
   ([profile check](../../nodes/_otr_engine_profiles.py#L422-L433)). In a normal
   package load, root `__init__.py` has already called the shared helper and can
   copy HKCU into that process environment
   ([startup bake-in](../../__init__.py#L77-L83)), so production indirectly sees
   HKCU. It still cannot see a cached Hub login, direct-import/test paths do not
   receive the bake-in, and credential policy is duplicated at the call site.
4. Prestartup sets `HF_HOME` under ComfyUI's model tree
   ([prestartup](../../prestartup_script.py#L56-L64)), while the runtime helper
   may set `HF_HOME`/`HF_HUB_CACHE` to a shared Windows location
   ([HF environment helper](../../nodes/_otr_hf_env.py#L85-L130)). Because the
   Hub token defaults under `HF_HOME`, this can make a normal token written by
   `hf auth login` outside ComfyUI invisible even after the resolver is corrected.
   This setting is also a live cache-layout contract: the loader, path resolver,
   and Bark adapter still derive locations from `$HF_HOME/hub`
   ([loader](../../nodes/_otr_model_loader.py#L1609-L1622),
   [paths](../../nodes/_otr_paths.py#L574-L613),
   [Bark](../../nodes/_otr_bark_lib.py#L152-L162)). It cannot safely be removed
   from prestartup in isolation.

### Recommended change

**Keep startup bake-in pure, and add one execution-time Hub-aware resolver.** The
root-cause implementation should:

1. Keep the existing import-time `ensure_hf_token()` limited to pure-stdlib
   process-environment/HKCU inspection and compatibility export. Do not import the
   Hub client, perform network-capable auth work, or mutate its credential files
   during node registration.
2. Add one separate resolver invoked only on the gated download/generate path.
   It should call the pure startup helper, then `huggingface_hub.get_token()` so
   `HF_TOKEN_PATH`, the cached `hf auth login` credential, supported aliases, and
   standard environment precedence work on every platform. This timing matters:
   current Hub tokens can refresh during resolution and an explicitly configured
   OIDC exchange can raise rather than silently fall back
   ([Hub resolver behavior](https://github.com/huggingface/huggingface_hub/blob/4237d95c603db491cb1070898c74c97e4d7c2582/src/huggingface_hub/utils/_auth.py#L58-L84)).
3. Remove the duplicate policy in `_otr_hf_auth.py` (or make it a compatibility
   import of the new runtime resolver) and route the catalog and engine-profile
   checks through that runtime path. Missing credentials must remain valid for
   public repositories; opted-in OIDC/auth failures should become actionable,
   redacted errors rather than being swallowed as "no token."
4. Preserve the existing cache layout in the immediate credential fix. Before
   OTR overrides `HF_HOME`, if the user has not set `HF_TOKEN_PATH`, capture and
   export the Hub token path derived from the **pre-override** native home. Do not
   merely stop setting `HF_HOME` or change only prestartup to `HF_HUB_CACHE`:
   current consumers would split scanner, downloader, and loader caches. A future
   cache/credential decoupling is worthwhile only as a repo-wide migration to one
   canonical hub-cache resolver, updating every `$HF_HOME/hub` consumer and
   proving existing snapshots remain discoverable without redownload.
5. Continue logging only credential source/presence, never value, prefix, header,
   or exception data that might contain it. Update gated-model errors to describe
   cached login, environment injection, and repository approval as separate steps.
6. Add regression coverage for: import/registration performs no Hub network/auth
   work; environment precedence; HKCU only on Windows; cached-login-only success
   on Windows/macOS/Linux; custom and pre-override `HF_TOKEN_PATH`; existing cache
   discovery; public no-token success; gated no-token failure; OIDC failure
   redaction; and no token in logs/errors.

Do not remove the HKCU fallback merely because Desktop now merges `process.env`.
It still closes the stale-parent-environment case without introducing storage.
Do remove the README's absolute statement that Desktop does not inherit user
environment variables.

### Exact README wording for the gated tier

The following should replace the current token-placement paragraphs **in the same
change as the resolver/cache fix above**. Publishing it before that fix would
promise cached-login behavior OTR does not yet provide.

````markdown
### Hugging Face access -- only for a gated model

Most users need no Hugging Face account or token: the canonical workflow and every
default model are ungated. If you select a gated model, first sign in on its
Hugging Face repository page and accept the publisher's terms. Authentication
cannot replace that approval.

Preferred setup:

```text
hf auth login
```

Run that once under the same OS account that launches ComfyUI, then restart
ComfyUI. OTR uses Hugging Face's standard credential resolution, so the active
cached login works across Windows, macOS, and Linux.

For headless servers, containers, or a managed secret service, set `HF_TOKEN`
before launching ComfyUI. Use a read or fine-grained token limited to the required
repositories. On Windows, OTR also checks the user-scope
`HKCU\Environment\HF_TOKEN` value when it is absent from the current process;
that is a compatibility bridge, not an OTR credential store.

Never put a token in a node, workflow, prompt, OTR config file, or bug report. If
the download still returns `401`/`403`, confirm both that the account accepted the
exact repository's terms and that `hf auth whoami` sees the intended account.
````

The existing gated-model table and repo-specific links should remain immediately
above this replacement text.

## 6. Upstream proposal

Do **not** open a duplicate generic "add HF token setting" issue. Comment with the
grounded requirements below on [Core #14657](https://github.com/Comfy-Org/ComfyUI/pull/14657)
and [Core #14586](https://github.com/Comfy-Org/ComfyUI/pull/14586), cross-linking
the companion frontend [#13062](https://github.com/Comfy-Org/ComfyUI_frontend/pull/13062).
If maintainers decide the provider-credential contract needs a durable RFC, that
discussion should produce it; OTR should not pre-empt the two active implementations.

The proposal, in five sentences:

1. ComfyUI Core and the frontend should own authenticated model download, while custom nodes continue using provider SDKs rather than private credential stores.
2. On local installs, the downloader should first honor Hugging Face's native resolver -- `HF_TOKEN`, the legacy alias, `HF_TOKEN_PATH`, and the token written by `hf auth login` -- with standard environment precedence.
3. If Comfy adds OAuth or a credential UI, its API should expose only write, delete, and status operations, keep plaintext unreadable to extensions, use OS-backed protection where available, and attach bearer credentials only to the exact HTTPS provider host across redirects.
4. The UI must treat repository license acceptance as a separate state and send the user to the exact repository terms page when approval is missing.
5. Cloud Secrets and Comfy Account API-node credentials should remain separate products and must not be repurposed as a plaintext token feed for third-party local nodes.

## Evidence limits and snapshot

This report used public source, documentation, issues, and PRs available on
2026-08-28. Key source snapshots were: ComfyUI
`0a33ed6c28f926d14536235771c222f9e6d1026b`, frontend
`512f72d5f7eee46ceb3c94526a01eb5979ab7df9`, Desktop/v1.0.46
`539df073165d63f56db95ebd96d462b8dde24108`, Manager
`f39cbd56fecae0b27a446c0cd450cd591f3a8bea`, comfy-cli
`929198f62d7641b2175f701a304bd562673a73c1`, Comfy docs
`47f2fbf2b118d055ea4701ae0cb73a624bc3d022`, and
`huggingface_hub` v1.29.0
`4237d95c603db491cb1070898c74c97e4d7c2582`. Open-PR status and implementation
details can change after this date. Absence claims are limited to the public
repositories, histories, code searches, and documentation inspected; they do not
claim knowledge of private roadmaps.
