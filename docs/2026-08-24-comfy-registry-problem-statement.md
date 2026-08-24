# Comfy Registry: problem statement for `comfyui-old-time-radio`

**Written 2026-08-24. Every claim below is a recorded request/response or a local
run, not an inference.** Publisher `fluxus`, node id `comfyui-old-time-radio`.

---

## 1. The one-sentence problem

Two separate things were being called "my nodes don't register," and only one of
them is real: **`2.0.0-alpha.7` has been sitting at `NodeVersionStatusPending`
since 2026-08-24T05:48:29Z, so `latest_version` still resolves to `alpha.6`** —
and separately, **node extraction has never succeeded for this pack** -- the
registry's import-based extractor (see 3b) has recorded zero nodes for every
version, while healthy packs show thousands.

## 2. What is PROVEN WORKING (so it stops being re-investigated)

| Claim | Verdict | Evidence |
|---|---|---|
| The pack is registered and active | **WORKING** | `GET /nodes/comfyui-old-time-radio` -> `status: NodeStatusActive`, 13 downloads |
| The alpha.7 upload completed | **WORKING** | `GET .../versions/2.0.0-alpha.7` -> HTTP 200, record exists, `createdAt 2026-08-24T05:48:29.409619Z` |
| The artifact is served | **WORKING** | `GET https://cdn.comfy.org/fluxus/comfyui-old-time-radio/2.0.0-alpha.7/node.zip` -> **HTTP 200, 5,540,961 bytes** |
| The artifact is complete | **WORKING** | 799 files; root `__init__.py`; `pyproject.toml` reading `version = "2.0.0-alpha.7"`; 293 files under `nodes/`; `node_list.json` present with 25 entries |
| Dependencies recorded | **WORKING** | `dependencies: list[12]`, matching `requirements.txt` exactly. (The `alpha.3` failure mode — a `dynamic` field publishing as `[]` — is not present.) |
| **The shipped code registers its nodes** | **WORKING** | Extracted the real published `node.zip` and loaded its `__init__.py`: `[OldTimeRadio] OK - All 25 nodes loaded successfully`, **25/25 registered, 0 failures** |

**So the package is not broken.** Anyone who installs it gets working nodes.

## 3. What is NOT WORKING, split into its two real causes

### 3a. `alpha.7` is stuck in `Pending` (the only genuinely open item)

`status: NodeVersionStatusPending`. While a version is pending,
`latest_version` reports the previous version (`alpha.6`).

Known from prior investigation of `Comfy-Org/registry-backend`: promotion
Pending -> Active is done **only** by Comfy-Org's own Cloud Scheduler cron
hitting an internal `/security-scan` endpoint, which considers only versions
older than 30 minutes. **There is no publisher self-service path to Active.**

**Waiting has already been tried on a previous version and the state did not
change on its own within the observed window.** That is why this is being raised
rather than waited out again.

### 3b. Node extraction has never succeeded for this pack (CORRECTED 2026-08-24)

The earlier draft of this section claimed the node panel was empty for every
pack; that was a wrong-endpoint artifact. The truth, verified against
`Comfy-Org/registry-backend` source and live API reads:

* `GET /comfy-nodes?node_id=<id>` works: comfyui-impact-pack 7,921 entries,
  comfyui-kjnodes 4,206, rgthree-comfy 1,124 -- **comfyui-old-time-radio 0**,
  on alpha.6 and alpha.7 alike.
* Extraction is import-based: `node-pack-extract` boots a CPU ComfyUI with the
  published zip installed and reads `/object_info`, filtering on
  `python_module == "custom_nodes.comfyui-old-time-radio"`. Dynamic
  registration is fine (rgthree extracts 1,124).
* **Our pack loads clean under a faithful local reproduction of those
  conditions** -- the published alpha.7 zip in a folder named
  `comfyui-old-time-radio`, hyphenated module name, prestartup first, no env,
  CPU: 25/25 nodes, zero failures.
* Versions carry `comfy_node_extract_status` (default `pending`). Extraction
  fires only via `POST /comfy-nodes/backfill` (auth-gated, default
  `max_node=10` per sweep, pending-only). **A version marked `failed` is never
  selected again.** The field is not publicly readable.

So the empty panel means one of exactly two things, and only Comfy-Org can say
which: our versions are still `pending` (waiting on a 10-per-sweep global
backfill), or an extraction ran, failed inside their container, and is now
terminally parked.

## 4. The ask, stated precisely

1. **What is `comfy_node_extract_status` for `comfyui-old-time-radio`'s
   versions?** If any are `failed`: what did the node-pack-extract Cloud Build
   log record, and can they be re-queued (the backfill selects `pending` only,
   so `failed` appears terminal)? If `pending`: roughly when does the backfill
   sweep reach newly published packs? Our pack loads 25/25 nodes under a
   faithful local reproduction of the extractor's conditions, so we expect a
   run to succeed.
2. **Can `2.0.0-alpha.7` be promoted from `Pending` to `Active`,** or can you
   say what is blocking its security scan? The same path produced `Active` for
   alpha.4/5/6.

## 5. What is explicitly NOT being asked

* Not asking for help with installation — the pack installs and registers
  25/25 nodes from the published artifact.
* Not asking about dependencies — 12 recorded correctly.
* Not requesting a version deletion. (Noted for our own records: deleting a
  *version* is a soft delete that permanently burns the version string, whereas
  deleting the *node* is a hard delete that frees them. Neither is wanted here.)

## 6. Reproduction commands

```bash
curl -s https://api.comfy.org/nodes/comfyui-old-time-radio
curl -s https://api.comfy.org/nodes/comfyui-old-time-radio/versions
curl -s https://api.comfy.org/nodes/comfyui-old-time-radio/versions/2.0.0-alpha.7
curl -sI https://cdn.comfy.org/fluxus/comfyui-old-time-radio/2.0.0-alpha.7/node.zip
curl -s "https://api.comfy.org/comfy-nodes?node_id=comfyui-old-time-radio"   # ours: total=0
curl -s "https://api.comfy.org/comfy-nodes?node_id=rgthree-comfy" | head -c 200  # control: populated
```

---

## Internal note — do not send this section

**The local-side work is finished and should not be repeated.** `node_list.json`
now ships (generated from the loader's declaration table, pinned by
`tests/test_node_list_manifest.py` with a vacuity floor). The idea of replacing
dynamic registration with a literal static `NODE_CLASS_MAPPINGS` **cannot**
achieve registry node visibility — extraction is import-based (a real ComfyUI
boot reading `/object_info`), and dynamic packs like rgthree extract fine — so
it must not be attempted as a fix for this.

The only remaining publisher-side lever on any `supported_*` field is
`requires-comfyui` in `[tool.comfy]` (as `ComfyUI-AnimateDiff-Evolved` declares),
which populates `supported_comfyui_version`. It affects compatibility filtering,
**not** node listing, and it costs a version bump — so spend it alongside a
change worth publishing rather than on its own.
