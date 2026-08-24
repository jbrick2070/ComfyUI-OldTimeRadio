# Comfy Registry: problem statement for `comfyui-old-time-radio`

**Written 2026-08-24. Every claim below is a recorded request/response or a local
run, not an inference.** Publisher `fluxus`, node id `comfyui-old-time-radio`.

---

## 1. The one-sentence problem

Two separate things were being called "my nodes don't register," and only one of
them is real: **`2.0.0-alpha.7` has been sitting at `NodeVersionStatusPending`
since 2026-08-24T05:48:29Z, so `latest_version` still resolves to `alpha.6`** —
and separately, **the registry has no per-node listing for ANY pack**, so an
empty node panel is not evidence of anything wrong with this pack.

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

### 3b. The per-node panel is empty for EVERY pack, not just this one

This is a separate issue and is almost certainly not a bug in this pack:

* A version record's **complete** key set is `changelog, createdAt,
  dependencies, deprecated, downloadUrl, id, node_id, status,
  supported_accelerators, supported_comfyui_frontend_version,
  supported_comfyui_version, supported_os, tags, tags_admin, version`.
  **There is no field anywhere for node classes.**
* `GET /nodes/<id>/comfy-nodes` -> **404** for `comfyui-old-time-radio`,
  `comfyui-kjnodes`, `rgthree-comfy`, `comfyui-dramabox`.
* `GET /nodes/<id>/nodes` -> **404** as well.
* **Control comparison:** `comfyui-kjnodes` 1.5.0 — one of the most-installed
  packs in the ecosystem — returns `supported_os []`,
  `supported_comfyui_version ""`, `supported_accelerators []`, `tags []`:
  identical in shape to ours, with the same three `[tool.comfy]` keys
  (`PublisherId`, `DisplayName`, `Icon`).

**Conclusion:** node listing is fed by a separate extraction service that does
not appear to populate for most or all packs. Nothing a publisher ships can
change it.

## 4. The ask, stated precisely

1. **Can `2.0.0-alpha.7` be promoted from `Pending` to `Active`,** or can you say
   what is blocking its security scan? The artifact is complete and served, and
   the same publishing path produced `Active` for `alpha.4`, `alpha.5` and
   `alpha.6`.
2. **Is the per-node listing (`/comfy-nodes`) expected to work at all?** If it is
   deprecated or not populated for most packs, saying so publicly would stop
   publishers debugging their own packaging over it. If it *is* meant to work,
   what does a pack have to ship for the extractor to find its nodes?

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
curl -s -o /dev/null -w "%{http_code}\n" https://api.comfy.org/nodes/comfyui-kjnodes/comfy-nodes
```

---

## Internal note — do not send this section

**The local-side work is finished and should not be repeated.** `node_list.json`
now ships (generated from the loader's declaration table, pinned by
`tests/test_node_list_manifest.py` with a vacuity floor). The idea of replacing
dynamic registration with a literal static `NODE_CLASS_MAPPINGS` **cannot**
achieve registry node visibility — there is no schema field for the result to
land in — so it must not be attempted as a fix for this.

The only remaining publisher-side lever on any `supported_*` field is
`requires-comfyui` in `[tool.comfy]` (as `ComfyUI-AnimateDiff-Evolved` declares),
which populates `supported_comfyui_version`. It affects compatibility filtering,
**not** node listing, and it costs a version bump — so spend it alongside a
change worth publishing rather than on its own.
