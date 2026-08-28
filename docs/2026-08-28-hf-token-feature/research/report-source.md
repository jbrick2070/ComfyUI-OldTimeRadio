# Research source notes: Hugging Face credentials in ComfyUI

**Internal research record -- 2026-08-28**

## Research question

Determine whether ComfyUI has or is developing a first-party Hugging Face
credential mechanism; distinguish Core, Desktop, Cloud Secrets, API-node auth,
Manager, and comfy-cli; establish current Hugging Face client behavior and
representative ecosystem patterns; audit OTR's implementation; recommend the
least-fragmented current posture and an upstream route.

## Method and source priority

1. Read the supplied problem statement and repository operating rules.
2. Read OTR's token/cache resolvers, gated-model preflight, engine profiles, and
   README.
3. Inspect pinned current snapshots of ComfyUI, frontend, Desktop, Manager,
   comfy-cli, Comfy docs, public RFCs, and `huggingface_hub`.
4. Search current code plus public issue/PR history for HF credential, gated
   download, model downloader, OAuth, secret, API key, and environment-variable
   terms.
5. Inspect representative custom nodes for materially different storage and
   transport patterns.
6. Prefer released source and official documentation over PR descriptions;
   treat open PRs as direction, not current API.
7. Record contradictions explicitly and bound negative findings to searched
   public artifacts.

## Pinned source snapshots

| Repository | Commit / release | Snapshot use |
|---|---|---|
| Comfy-Org/ComfyUI | `0a33ed6c28f926d14536235771c222f9e6d1026b` | Current Core/API-node behavior |
| Comfy-Org/ComfyUI_frontend | `512f72d5f7eee46ceb3c94526a01eb5979ab7df9` | Secrets UI, prompt auth, serialization |
| Comfy-Org/Comfy-Desktop | `539df073165d63f56db95ebd96d462b8dde24108` / v1.0.46 | Released launcher/env persistence |
| Comfy-Org/ComfyUI-Manager | `f39cbd56fecae0b27a446c0cd450cd591f3a8bea` | Manager downloader/config |
| Comfy-Org/comfy-cli | `929198f62d7641b2175f701a304bd562673a73c1` | CLI credential namespace and scope |
| Comfy-Org/rfcs | `5d1bab72d0a2fd54a266dc00eb7a6323d3aad41a` | Public RFC search |
| Comfy-Org/docs | `47f2fbf2b118d055ea4701ae0cb73a624bc3d022` | Official Cloud/API integration scope |
| huggingface/huggingface_hub | `4237d95c603db491cb1070898c74c97e4d7c2582` / v1.29.0 | Stable token-resolution contract |

Open-PR head used for ComfyUI #14657 auth internals:
`a7b63915dcf1a259cb8ffb3370b078990dbe9e91`.

## Discovery queries and code probes

Representative terms: `HF_TOKEN`, `HF_API_TOKEN`, `hf_api_token`,
`HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`, `HuggingFace`, `Hugging Face`,
`gated`, `model downloader`, `OAuth`, `Secrets`, `api_key_comfy_org`,
`auth_token_comfy_org`, `is_api_node`, `Environment Variables`, `safeStorage`,
`HF_ENDPOINT`, `snapshot_download`, `Authorization`, `serialize_widgets`, and
`installations.json`.

Public history searched across ComfyUI, frontend, Desktop, Manager, comfy-cli,
registry/RFC-related public repositories, and documentation. Current-source
searches were repeated after history searches so abandoned experiments would not
be mistaken for released behavior.

## Findings by lane

### First-party local surfaces

- Core has no merged HF credential setting/service at the pinned head.
- Core #14586 plus frontend #13062 propose server-side download and HF OAuth;
  both are open/conflicting and require coordinated backend/frontend delivery.
- Core #14657 proposes a broader downloader manager, is open/conflicting, and at
  its inspected head resolves environment credentials before OAuth. Its local
  token store calls its XOR construction obfuscation rather than confidentiality.
- Desktop v1.0.46 has a generic per-install environment-variable editor. It merges
  inherited `process.env` and user configuration into the ComfyUI child process.
  Values persist as ordinary JSON in `installations.json`; the current UI warns
  that they are unencrypted and should not contain secrets. A password input is
  visual masking only.
- Desktop's gated built-in downloader uses same-session browser cookies after the
  user signs in/accepts terms. It does not export provider auth to custom nodes.
- Desktop PR #590 is historical evidence of a `safeStorage` experiment, not
  current behavior; current source and warning control the conclusion.

### Cloud and API nodes

- Cloud Secrets is a first-party provider-secret product, but official docs scope
  it to eligible Cloud plans and explicitly exclude Desktop/portable.
- Secret create/update accepts a value; list/get responses contain metadata, not
  plaintext. No sanctioned local custom-node plaintext read path was found.
- API-node hidden inputs contain Comfy Account token/API key. Core keeps these out
  of ordinary prompt history and first-party helpers send them to `api.comfy.org`.
- This is proxy/account authentication, not an arbitrary provider-key store. It is
  not a legitimate HF token source for OTR.

### Manager and CLI contradiction

- The problem statement's assertion that Manager injects HF credentials is false
  for the pinned current Manager. Only `HF_ENDPOINT` is documented.
- `HfApi().repo_info` may inherit ambient Hub auth, but Manager's byte download is
  a bare `requests.get`; this does not make Manager a gated download solution.
- The separate comfy-cli has `HF_API_TOKEN`, `hf_api_token`, and
  `--set-hf-api-token`, with persisted/config/environment resolution scoped to
  `comfy model download`. It does not inject a running server.
- ComfyDeploy does document an HF token setting, but it belongs to its hosted
  model-download/private-volume product, not a local ComfyUI credential contract.

### Hugging Face contract

- Modern official command: `hf auth login`; old `huggingface-cli login` wording
  should not be used for new documentation.
- Stable v1.29.0 resolves `HF_TOKEN`, deprecated legacy alias, configured token
  path/cache, and supported OIDC/host-specific sources. Environment token wins
  over cached login.
- `token=None` permits implicit standard resolution, `token=True` requires one,
  and `token=False` disables authentication.
- Token and gated-repository approval are separate requirements.
- Official guidance favors read/fine-grained, per-app/usage credentials and
  managed storage; leaked tokens should be rotated.
- Cached login is file-backed, not an OS vault. Hub applies `0700`/`0600` on
  POSIX; Windows relies on its filesystem/profile protections.

### Ecosystem patterns

- Hub-native/no field: kijai HunyuanVideo wrapper.
- General Comfy setting: jnxmx downloader; ordinary setting state, plus a
  subprocess path that serializes kwargs to temporary JSON and may expose raw
  errors.
- Pack INI/manual bearer: tori downloader.
- Node/widget: ciri downloader; example workflow demonstrates serializable token
  location.
- Guarded pack route: Ideogram4; boolean-only read and POSIX `0600`, but still a
  second plaintext store and no equivalent Windows vault guarantee.
- Security ranking for the report: native Hub resolution first; centralized
  first-party write-only protected storage next if released; generic plaintext
  settings; pack-local plaintext config; token node/widget last.

### OTR audit

- `nodes/_otr_shared/hf_token.py`: process `HF_TOKEN` -> Windows HKCU -> None;
  cached result; exports the HKCU value to standard/legacy env names; never calls
  Hub `get_token()`.
- `nodes/_otr_hf_auth.py`: duplicate env/HKCU implementation, no Hub cache.
- `_otr_model_catalog.py`: gated preflight rejects on this limited resolver before
  Hub download. Its later `token=None` would normally allow the SDK's implicit
  cache resolution, but the gated branch never reaches it.
- `_otr_engine_profiles.py`: direct `os.getenv("HF_TOKEN")`. Normal package startup
  has already baked HKCU into the process, so it indirectly sees HKCU there; it
  still misses cached Hub login and direct-import/test paths.
- `prestartup_script.py` and `_otr_hf_env.py` set/derive `HF_HOME` for model cache.
  Hub's default token file also lives under `HF_HOME`, so OTR can hide the token
  written to the normal per-user cache by an external `hf auth login`.
- Several loaders/path helpers still derive `$HF_HOME/hub`; changing only
  prestartup to set `HF_HUB_CACHE` would split cache discovery and risk repeated
  downloads.
- Root `__init__.py` calls the pure shared resolver during package import.
  `huggingface_hub.get_token()` can refresh browser OAuth/file state or raise for
  an opted-in OIDC exchange, so Hub-aware resolution belongs on the gated
  execution path, not node registration.
- README's claim that Desktop does not inherit user-scope variables is too
  absolute. Current Desktop passes its inherited environment. HKCU still helps
  when the Desktop process predates the environment change.

## Contradictions resolved

| Initial/implied claim | Resolution |
|---|---|
| No first-party Comfy mechanism exists | No dedicated released local secure store exists, but Desktop has a generic plaintext environment editor, Desktop has a cookie-only built-in flow, Cloud has managed Secrets, and two Core proposals are active. |
| Manager can inject an HF token | False in current Manager; likely confusion with comfy-cli or ambient Hub behavior. |
| comfy-cli has `--hf-token` | Current flag is `--set-hf-api-token`; namespace is `HF_API_TOKEN` / `hf_api_token`. |
| OTR already honors standard Hub machinery | False for cached login: preflight only sees environment/HKCU, and OTR's `HF_HOME` relocation changes the default token path. |
| Desktop never inherits user-scope environment | False as an absolute claim; it passes inherited `process.env`, but an already-running parent has stale environment state. |
| Password-looking input makes a token safe | False; normal widget/settings serialization and plaintext persistence remain. |

## Synthesis rationale

The recommendation avoids both fragmentation and a false promise. OTR should not
build storage or UI. It should retain the pure import-time Windows bridge, add
Hub-aware resolution only on gated execution, and preserve the native token path
before its existing cache-root override. A later cache decoupling must be a
repo-wide canonical resolver migration, not an isolated environment edit. The
upstream ask should join the two active model-downloader PRs and insist on
native-provider fallback, write-only status APIs, protected storage, redirect
scoping, and explicit license state. Cloud Secrets/API-node auth remain
deliberately separate because exposing their plaintext would break their
security/product boundaries.

## Evidence limits

- PR state and code can change after the snapshot.
- No-public-artifact findings are bounded to repositories, history, and terms
  searched; they do not assert that private work is absent.
- Representative custom nodes demonstrate patterns and hazards, not ecosystem
  prevalence statistics.
- No live credential was created or transmitted during this research.

## Independent spot check

A separate read-only code review of the polished report identified and corrected
three over-simplifications before delivery: normal startup indirectly gives the
engine-profile check the HKCU value; multiple OTR consumers still depend on
`$HF_HOME/hub`, making a prestartup-only cache change unsafe; and Hub-aware token
resolution must remain off the import/registration path because refresh/OIDC can
perform stateful or network-capable work. It also tightened the frontend prompt
transport and password-mask citations. The reviewer re-read the revised section
and reported the must-fix findings resolved.
