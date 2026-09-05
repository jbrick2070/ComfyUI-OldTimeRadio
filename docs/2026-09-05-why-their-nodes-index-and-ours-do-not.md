# Approved packs' nodes are in the registry's node index. Ours are not. Why?

**The operator's hypothesis, and it is a good one:** the approved packs share some
"identical factor" we lack, and that factor is the missing link to approval. Below is
what was MEASURED today against the live API. Every number is a real request, not a
recollection. **Refute anything you can check and find false.**

## 1. THE BINARY DIFFERENCE

`GET https://api.comfy.org/comfy-nodes/<ComfyUINodeName>/node` is the endpoint the
**ComfyUI frontend itself calls** to resolve an unknown node to the pack that provides it
(`inferPackFromNodeName` in `useConflictDetection`; it is the fallback whenever a
workflow node carries no `properties.cnr_id`).

| node name | pack | result |
|---|---|---|
| `UnetLoaderGGUF` | ComfyUI-GGUF | **200** |
| `DualCLIPLoaderGGUF` | ComfyUI-GGUF | **200** |
| `AIO_Preprocessor` | comfyui_controlnet_aux | **200** |
| `CannyEdgePreprocessor` | comfyui_controlnet_aux | **200** |
| `ADE_AnimateDiffLoaderWithContext` | AnimateDiff-Evolved | **200** |
| `OTR_CastLock` | ours | **404** |
| `OTR_WorkflowValidator` | ours | **404** |

404 body: `{"error":"","message":"No node found containing the specified ComfyUI node name"}`

**Their nodes are indexed. Ours are not.** That is the cleanest structural difference
found between us and every approved pack.

**A correction I OWE, because I got this wrong in the first draft of this brief and
Cursor caught it:** I wrote that CLAUDE.md "cites `/nodes/<id>/comfy-nodes`'s 404 as
evidence about our pack". It does not. CLAUDE.md:488-495 says the opposite and says it
correctly -- that endpoint 404s for every pack sampled, the panel is fed by a separate
extraction service, and *"do not diagnose from the registry page"*. My accusation was a
misreading of our own notes. What IS true and worth adding: there are FOUR distinct URLs
here and they must not be conflated --
`GET /nodes/<id>/comfy-nodes` (pack panel, 404s widely),
`GET /nodes/<id>/versions/<ver>/comfy-nodes` (the versioned extract receipt, and the only
object this repo has ever used as proof extract ran),
`GET /comfy-nodes?node_id=<id>` (pack-scoped index), and
`GET /comfy-nodes/<ComfyUINodeName>/node` (class-name lookup, section 1 above).


## 1b. THE CARD'S "N Nodes" AND THE NODE INDEX ARE DIFFERENT PIPELINES

Measured off the registry front page plus the API, so do not conflate them:
`ComfyUI-KJNodes` shows **no node count on its card** yet `ImageResizeKJ` and
`ColorMatch` both resolve **200** on `/comfy-nodes/<name>/node`. So a pack can be
indexed while its card shows nothing. Whatever populates the card is a third
service again. **Our pack is absent from BOTH**, but they are separate failures
and evidence about one is not evidence about the other.

## 2. THE OBVIOUS EXPLANATION IS ALREADY REFUTED

Comfy-Org's node-pack-extract boots a headless CPU ComfyUI in a Linux container, installs
the pack's requirements under `set -e`, and reads `/object_info`. Our standing theory has
been that OUR dependency list breaks that container (kokoro pulls torch; a multi-GB
download blows the 600 s extract timeout). **Declared dependency weight does not
correlate with being indexed:**

| pack | deps | indexed? |
|---|---|---|
| rgthree-comfy | 0 | yes |
| ComfyUI-GGUF | 3 (gguf, sentencepiece, protobuf) | yes |
| ComfyUI-Crystools | 6 (incl. **torch**) | yes |
| was-node-suite-comfyui | 20 (incl. **two `git+https://` URLs**) | Active |
| comfyui_controlnet_aux | 24 (incl. **torch, torchvision, opencv-python, scikit-image**) | yes |
| ours | 19 | **no** |

`comfyui_controlnet_aux` installs torch AND torchvision AND opencv in that same container
and its nodes are indexed. So "our deps are too heavy" is not sufficient on its own.

## 3. THE FORK WE CANNOT SETTLE FROM OUTSIDE -- THIS IS THE QUESTION

**Is node extraction a PREREQUISITE for promotion, or a CONSEQUENCE of it?**

Every pack whose nodes are indexed is `Active`. We are `Flagged` and have never been
promoted since the ban. Two readings fit every observation we have:

* **(A) Extraction is upstream.** The extract container fails on our pack, so no nodes
  are indexed, and something in the promotion path needs them. Then fixing the extract is
  a real path to approval and we should chase it hard.
* **(B) Extraction is downstream.** Extraction only runs for versions that already
  reached Active. Then our empty index is a SYMPTOM of being Flagged, chasing it is
  wasted effort, and it tells us nothing.

**Settle this from `Comfy-Org/registry-backend` and the extract service if you can read
them.** Name the file and the ordering. If you cannot read them, say so and propose the
cheapest experiment that distinguishes (A) from (B) -- for example, whether any pack
anywhere is `Flagged` or `Pending` AND has indexed nodes, or `Active` with none. A single
counter-example settles it.

## 4. A SECOND MEASURED ODDITY: our node id is the only re-worded one

| pack | registry id | repo basename | relationship |
|---|---|---|---|
| ComfyUI-Crystools | `ComfyUI-Crystools` | ComfyUI-Crystools | exact |
| comfyui_controlnet_aux | `comfyui_controlnet_aux` | comfyui_controlnet_aux | exact |
| ComfyUI-GGUF | `ComfyUI-GGUF` | ComfyUI-GGUF | exact |
| rgthree-comfy | `rgthree-comfy` | rgthree-comfy | exact |
| was-node-suite-comfyui | `was-node-suite-comfyui` | was-node-suite-comfyui | exact |
| comfyui-videohelpersuite | `comfyui-videohelpersuite` | ComfyUI-VideoHelperSuite | case-fold |
| comfyui-easy-use | `comfyui-easy-use` | ComfyUI-Easy-Use | case-fold |
| **ours** | **`comfyui-old-time-radio`** | **ComfyUI-OldTimeRadio** | **re-worded** |

Everyone else either preserves the repo string exactly or lowercases it. A pure
case-fold of our repo is `comfyui-oldtimeradio`; we published `comfyui-old-time-radio`,
inserting word hyphens nobody else inserts.

Three more measurements about ids:
* **Lookups are case-insensitive.** `nodes/comfyui-gguf` and `nodes/comfyui-crystools`
  return **302** redirects to the canonical mixed-case id. So case alone never forks a
  listing -- but our extra hyphens make ours a genuinely different string.
* **`nodes/ComfyUI-OldTimeRadio` is 404 -- that id is FREE.**
* **The id does not follow `[project] name` after the first publish.** ComfyUI-GGUF's
  `pyproject.toml` declares `name = "comfyui-gguf"` today, yet its registry id is
  `ComfyUI-GGUF` and there is no second listing. Crystools declares
  `name = "ComfyUI-Crystools"` and gets that id. **Question for you: does editing
  `[project] name` to a genuinely different string RENAME the node, CREATE a second
  listing, or get ignored?** This decides whether "make the id match the repo" is a
  one-line change or a new pack. Do not guess -- find it in the backend or say it is
  unknown.

## 5. WHAT WE WANT FROM YOU

1. **Settle section 3.** Prerequisite or consequence? Cite the code or name the
   experiment.
2. **If extraction is a prerequisite: what, specifically, fails for us?** We ship 19
   deps; `controlnet_aux` ships 24 including torch and is fine. Look at our
   `pyproject.toml`, `requirements.txt`, `__init__.py` and `prestartup_script.py` for
   what would break a headless CPU Linux container under `set -e` where
   `controlnet_aux`'s would not. Candidates worth checking rather than assuming:
   `prestartup_script.py` running at import and doing network or filesystem work; a
   dependency with no Linux wheel; a marker that resolves wrongly on Linux; an
   `__init__.py` import that needs a GPU or a model file.
3. **Rule on the id.** Given sections 3 and 4, is renaming worth doing AT ALL, and if so
   is it a rename or a new listing? Both of you previously CUT "new node id" as a
   ban-escape. This is a different motive -- convention consistency -- so re-rule it on
   that basis, and say plainly if the answer is still no.
4. **Anything a byte-level structural diff would show.** If you think the difference is
   in `pyproject.toml` shape, `__init__.py` export shape, node-class registration shape,
   or widget/`INPUT_TYPES` shape rather than dependencies, say which and how to test it.

**Ground every claim. A refutation with a source beats agreement.**
