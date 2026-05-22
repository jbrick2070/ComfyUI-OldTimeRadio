# Two-Model Selector — Scoping Document

**Date:** 2026-05-13
**Branch target (when implemented):** `v2.0-alpha`
**Status:** SCOPING ONLY — no code changes, no file deletions.
**Author:** read-only analysis pass for Jeffrey Brick.

---

## 0. Goal (Jeffrey, 2026-05-13)

Single source of truth for every model pick in the workflow:

- **Slot 1 — Creative model.** Drives narrative LLM passes (outline, cast, dialogue, polish).
- **Slot 2 — Technical model.** Drives structured LLM passes (JSON validators, freeze-cascade verdicts, critic, normalizers, format rescue).

Both live on the Story Writer (`OTR_LedgerScriptWriter`). Every other node that currently picks an LLM (Freeze Cascade, LFC Phase 4/5/6, the visual path's selector) is rewired to pull from those two slots instead of exposing its own `model_id` widget.

Non-LLM model picks (TTS / SFX / music / video) also centralize per Jeffrey's "Everything" answer. The doc proposes one workable path and flags it as the open decision.

UX:
- Models not present on disk render **red** in the dropdown.
- Selecting a red model triggers a download attempt.
- Loaded / available models render normally.

Security:
- Treated as "sensible defaults" since Jeffrey doesn't have a hard requirement: path-traversal rejection on model_id strings, allow-list enforcement, fail-loud when downstream gets no model, refuse `.bin` / `.pkl` auto-load.

---

## 1. Current state — model_id surface (HEAD `f11fee1`)

### 1a. LLM-consuming nodes that currently expose `model_id`

| # | File                                            | Line  | Widget surface             | Notes                                                    |
|--:|-------------------------------------------------|------:|----------------------------|----------------------------------------------------------|
| 1 | `nodes/OTR_LedgerScriptWriter.py`               | 1163  | combo `_MODEL_CHOICES`     | The keeper. Will gain a second slot.                     |
| 2 | `nodes/OTR_LedgerFreezeCascade.py`              |  148  | STRING (free-form)         | Marked for removal in Jeffrey's annotated screenshot.    |
| 3 | `nodes/OTR_LFCPhase4Scene.py`                   |   83  | STRING (free-form)         | Standalone runner — same removal pattern.                |
| 4 | `nodes/OTR_LFCPhase5Voice.py`                   |   64  | STRING (free-form)         | Standalone runner — same removal pattern.                |
| 5 | `nodes/OTR_LFCPhase6Arc.py`                     |   63  | STRING (free-form)         | Standalone runner — same removal pattern.                |
| 6 | `nodes/vram_context_test.py`                    |  138  | combo `_LLM_MODEL_CHOICES` | Test bench — defer or carve-out (see §10).               |
| 7 | `visual/llm_selector.py` (`OTR_VisualLLMSelector`) | n/a | combo `_LLM_MODEL_CHOICES` | Already a single-source pattern for the visual path. Either retired or rewired to consume from the writer. |

### 1b. Non-LLM model picks that currently expose `model_id`

| # | File                                  | Line  | Widget               | Class of model      |
|--:|---------------------------------------|------:|----------------------|---------------------|
| 8 | `nodes/musicgen_theme.py`             |  393  | STRING               | MusicGen (audio)    |
| 9 | `nodes/batch_audiogen_generator.py`   |  260  | combo (2 entries)    | AudioGen / SFX      |

### 1c. Adjacent model picks (different naming, same problem class)

| # | File                                | Pattern                | Class of model               |
|--:|-------------------------------------|------------------------|------------------------------|
|10 | `nodes/batch_ltx_render.py`         | `model_name`           | LTX video                    |
|11 | `nodes/batch_humo_render.py`        | (HuMo model path)      | HuMo video character clips   |
|12 | `nodes/_voice_backends/bark.py`     | `bark_model` resolver  | Bark TTS                     |
|13 | `nodes/kokoro_announcer.py`         | (Kokoro voice id)      | Kokoro announcer             |
|14 | `nodes/_otr_voice_resolver.py`      | (voice backend select) | TTS routing                  |

(Lines elided; the §1c set is in scope only if the "Everything" answer extends past LLMs — see §10 open decision.)

### 1d. Existing infrastructure worth keeping (do NOT delete)

- `nodes/_otr_model_loader.py` — facade with `load_llm() / make_generate_fn() / make_polish_generate_fn() / unload_llm()`. Loader is already centralized; the missing piece is the **selector**, not the loader. The selector work plugs into this loader unchanged.
- `nodes/_otr_model_loader.py::MODEL_CONTEXT_CAPS` — per-model context windows. Reused by Slot 1 and Slot 2 both.
- `tests/test_two_llm_split.py` — proves a prior `cleanup_model_id` widget existed on the writer and was wired through structured paths exactly as Slot 2 needs. The widget was deleted by the S15.5+ writer slim-down (`OTR_LedgerScriptWriter.py:2475` shows `cleanup_model_id` in a legacy-strip loop). The resolver semantics in the test (`_resolve_cleanup_model_id`) are the contract Slot 2 should re-adopt verbatim.
- `visual/llm_selector.py` — proves the "broadcast one model_id over a STRING socket to N consumers" pattern works in ComfyUI. Pattern is reused by Slot 1 and Slot 2.

**Read of the build history:** a partial version of this feature shipped, then was rolled back during cleanup. We are finishing the rollback **and** reinstating the design properly, this time centralized on the writer and removed from everywhere else.

---

## 2. Target state

### 2a. Story Writer widget order (post-change)

```
episode_title
target_words
num_characters
seed
model_creative           <-- NEW: Slot 1 (was: model_id)
model_technical          <-- NEW: Slot 2 (replaces the prior cleanup_model_id)
custom_premise
include_act_breaks
act_count
style
style_custom
creativity
optimization_profile
perfect_run_spacesaver
min_p
repetition_penalty
max_new_tokens_cap
enable_polish_pass
```

Widget order is load-bearing — saved workflow JSONs bind by index. The two new widgets replace `model_id` at its current position (1163) and add `model_technical` immediately after, so the rest of the layout shifts down by exactly one slot. Existing workflow JSONs need a one-time re-write (§6).

### 2b. Story Writer output sockets (new)

```
script_text       (existing)
script_json       (existing)
news_used         (existing)
estimated_minutes (existing)
model_creative    NEW   STRING — broadcast to downstream consumers
model_technical   NEW   STRING — broadcast to downstream consumers
```

Why broadcast as outputs rather than read from a global: ComfyUI nodes are functionally pure given their inputs, and a wired STRING socket is the idiomatic way to make selection explicit + visible on the canvas. Matches `OTR_VisualLLMSelector`'s pattern.

### 2c. Downstream consumer change pattern

For every node currently exposing `model_id`:

1. Remove the widget from `INPUT_TYPES["optional"]`.
2. Add a `STRING` socket (no widget) named `model_creative` OR `model_technical` depending on the node's role.
3. Remove the `model_id` default from the `run()` signature; pull from the socket instead.
4. If the socket is unwired, **fail loud** with `MissingModelInputError` (new error class). No silent default.

Routing decisions (from Jeffrey's "Creative = everything narrative; Technical = everything structured" answer):

| Node                                        | Slot         | Reason                                       |
|---------------------------------------------|--------------|----------------------------------------------|
| `OTR_LedgerScriptWriter` outline pass       | creative     | narrative                                    |
| `OTR_LedgerScriptWriter` cast pass          | creative     | narrative                                    |
| `OTR_LedgerScriptWriter` dialogue composer  | creative     | narrative                                    |
| `OTR_LedgerScriptWriter` polish pass        | creative     | narrative                                    |
| `OTR_LedgerScriptWriter` WORD_EXTEND rescue | technical    | structured                                   |
| `OTR_LedgerScriptWriter` FORMAT_NORM        | technical    | structured                                   |
| `OTR_LedgerScriptWriter` Grammarian         | technical    | structured                                   |
| `OTR_LedgerScriptWriter` LLM_RESCUE         | technical    | structured                                   |
| `OTR_LedgerScriptWriter` announcer intro/outro | creative  | narrative framing pass (BUG-LOCAL-255, 2026-05-22) |
| `OTR_LedgerFreezeCascade` (Phase 1/2/9)     | technical    | reviewer verdicts                            |
| `OTR_LFCPhase4Scene`                        | creative     | narrative coherence                          |
| `OTR_LFCPhase5Voice`                        | technical    | per-line targeted rewrites                   |
| `OTR_LFCPhase6Arc`                          | technical    | editor-note scaffold (structured output)     |
| `OTR_VisualLLMSelector` consumers           | creative     | visual prompt cleanup is prose               |

The two existing LFC LLM helpers (`_otr_lfc_llm_helpers.py`) take `generate_fn` as an argument, so the routing decision lives entirely at the **node** layer. No helper-layer change needed.

---

## 3. Dropdown source — scanned + curated (Jeffrey's "Both" answer)

### 3a. New module: `nodes/_otr_model_catalog.py`

```python
# Pseudocode — final shape lands at implementation time.

CURATED_LLM_MODELS = [
    # Pinned recommendations, ordered for the dropdown.
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen2.5-14B-Instruct",
    "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
    "inflatebot/MN-12B-Mag-Mell-R1",
]

HIDDEN_LLM_MODELS = set()  # explicit hide-list, e.g. broken weights

def scan_local_llm_cache() -> list[ScanResult]:
    """Walk HF_HOME / models--*/snapshots/*. Return repo_id + on_disk bool."""
    ...

def build_dropdown_choices() -> list[DropdownEntry]:
    """
    Merge curated + scanned. Each entry carries:
        repo_id:   str   (canonical HF id, what goes to the loader)
        label:     str   (display string; appends [NOT DOWNLOADED] suffix when on_disk=False)
        on_disk:   bool  (drives the red-state UX in §4)
        curated:   bool  (pinned entries first; scanned-only entries after)
    """
    ...
```

### 3b. Selector behavior

- **At node-init time:** call `build_dropdown_choices()` once. ComfyUI calls `INPUT_TYPES()` per node placement, so the cost is amortized.
- **Order:** curated first (Jeffrey's pins), scanned-only entries after, alphabetized within each group.
- **Drift detection:** if a curated entry is not on disk, mark it `on_disk=False` rather than hiding it (UX matches Jeffrey's "red dropdown" ask).
- **Allow-list enforcement:** if a saved workflow JSON binds `model_creative` to a string that is neither in `CURATED_LLM_MODELS` nor in the local scan, the writer falls back to the curated default and logs `[Selector] Unknown model_id %r, falling back to %r` — same shape as `VisualLLMSelector` today.

---

## 4. Red-state UX + auto-download (Jeffrey's UX ask)

### 4a. Dropdown rendering

ComfyUI's stock combo widget does not natively support per-entry color. Two paths:

**Path A — label suffix (minimum-effort, zero JS).**
- Render not-on-disk entries as `"<repo_id>   [NOT DOWNLOADED]"`.
- Loader strips the suffix before HF lookup (the same `.split(" ", 1)[0]` pattern already in `VisualLLMSelector::select`).
- No JS, no Comfy front-end coupling. Works in ComfyUI Desktop today.

**Path B — JS widget extension (matches the screenshot intent).**
- Add `web/js/otr_model_dropdown_color.js`.
- Patch the combo widget to color not-on-disk entries red and show a download icon.
- Higher effort, depends on Comfy's widget API stability across releases.

**Recommendation:** ship Path A first (one PR). Path B becomes a follow-up sprint behind a `feedback_lean_docs_fix_code`-style gate once the rest works end-to-end.

### 4b. Auto-download trigger

On `load_llm(model_id)`:

1. Check local HF cache (the existing `huggingface_hub.snapshot_download` resolver path is already idempotent for present caches).
2. If snapshot is missing, call `snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer.*"])`.
3. Refuse `*.bin` / `*.pkl` per §5 security rule.
4. Emit a single "Downloading <repo_id> — first run only" log line + a Comfy queue progress message so Jeffrey sees something happening.
5. After download, retry the local-cache path. Fail loud if the second attempt still fails.

**Important:** auto-download must respect the `HF_TOKEN` env var (set in `HKCU\Environment` per CLAUDE.md). The loader reads it via `winreg` for gated repos; ungated repos work without it.

**Important #2:** auto-download is opt-in via a config flag on the catalog module, defaulting to ON. A future Jeffrey-on-a-plane-with-no-network scenario needs the off switch.

---

## 5. Security

Five rules, all enforced inside `nodes/_otr_model_catalog.py::validate_model_id()`:

1. **Path-traversal rejection.** A `model_id` may contain `[A-Za-z0-9_\-./]` only. Reject `..`, leading `/`, drive letters, `\`. Repo IDs are always `org/name` on HF; anything else is a workflow tampering signal.
2. **Allow-list enforcement.** Effective `model_id` after suffix-stripping must be in `CURATED_LLM_MODELS` OR present in the local-cache scan. Anything else falls back to the curated default with a WARNING log line. No exception raised — this is the back-compat path for older workflow JSONs.
3. **Format gating on download.** `snapshot_download(allow_patterns=...)` whitelists `*.safetensors` / `*.gguf` / `*.json` / `*.txt` / `tokenizer.*`. Refuses to pull `*.bin` / `*.pkl` / `*.pickle` automatically. Manual override only via a documented one-off script — not via the dropdown UX.
4. **Fail-loud on missing socket.** New `nodes/_otr_model_inputs.py::MissingModelInputError`. Raised by any consumer node that runs without `model_creative` / `model_technical` wired. No silent default. Matches existing `feedback_wire_it_or_dont_ship_it`.
5. **No remote-code-execution loaders.** `transformers.AutoModel.from_pretrained(..., trust_remote_code=False)` everywhere. There is no current site with `trust_remote_code=True` in the repo; this section is preventative and the lint to enforce it lands in `tests/test_workflow_json_guardrails.py`.

---

## 6. Workflow JSON re-wiring

`workflows/otr_scifi_16gb_full.json` is the only canonical workflow per `feedback_minimum_json_files`. Changes required:

1. Story Writer node `widgets_values`:
   - At the index currently holding `"mistralai/Mistral-Nemo-Instruct-2407"` (model_id), keep `"mistralai/Mistral-Nemo-Instruct-2407"` (now `model_creative`).
   - Insert immediately after: `"mistralai/Mistral-Nemo-Instruct-2407"` (the technical default — same model in single-LLM mode).
   - Every subsequent widget index shifts +1.

2. Story Writer outputs:
   - Add two new output sockets at the end: `model_creative` (STRING) and `model_technical` (STRING).

3. Every consumer node:
   - Remove `model_id` from `widgets_values`.
   - Add a new input link from the writer's `model_creative` or `model_technical` output.

4. `OTR_VisualLLMSelector` instance, if present:
   - Either remove the node entirely (preferred — kills a redundant pick site) and wire visual consumers to the writer's `model_creative`, OR
   - Keep as a `model_creative` passthrough for UI clarity, accepting the duplication.

5. Run `tests/test_workflow_json_guardrails.py` after the re-write. A new test (§9) pins the writer's widget order and the absence of `model_id` widgets everywhere downstream.

**Script:** `scripts/migrate_workflow_two_model_selector.py` (new) — read the JSON, rewrite widget arrays + add link entries, emit the new JSON beside the old. Old JSON kept under `legacy_archive/` is rejected by `feedback_no_legacy_back_compat`, so the migration is destructive: the new JSON overwrites in place, the migration script is a one-shot, git history is the rollback path.

---

## 7. Non-LLM model picks — the "Everything" gap

Jeffrey's answer was "Everything — TTS / audio / video model picks also centralize." But the original ask names only **two** dropdowns. Reconciling:

### 7a. Three workable shapes

**Shape A — Writer carries N slots.**
Story Writer gains: `model_creative`, `model_technical`, `model_tts`, `model_sfx`, `model_music`, `model_video_image`, `model_video_motion`, `model_upscale`. Seven dropdowns. Simple to understand; widget panel gets crowded.

**Shape B — Dedicated `OTR_ModelHub` node.**
New node sits at the top of the graph carrying every model pick. Writer reads `model_creative` + `model_technical` from it via wires. Music / SFX / video / TTS read their own slots from it. Cleaner separation; introduces a new node type, slightly more graph clutter.

**Shape C — Two slots on writer, defaults elsewhere.**
Writer carries only the two LLM slots. TTS/SFX/music/video keep their current widgets but get **defaults locked** behind a "MAINTAINER" config layer — i.e. the dropdowns hide unless a config flag is set. Honors the "ONLY place users pick" intent without forcing graph re-architecture.

**Recommendation: Shape B.** It scales (when HuMo 2.0 ships, add one slot; no writer churn) and matches the existing `OTR_VisualLLMSelector` precedent — a one-purpose pick node already proved usable.

### 7b. If Shape B is picked, the hub node looks like:

```
OTR_ModelHub
    Outputs (all STRING):
        model_creative
        model_technical
        model_tts            (e.g. "kokoro" or "bark")
        model_sfx            (e.g. "facebook/audiogen-medium")
        model_music          (e.g. "facebook/musicgen-small")
        model_video_image    (e.g. FLUX checkpoint id)
        model_video_motion   (e.g. HuMo / LTX checkpoint id)
        model_upscale        (e.g. RealESRGAN id)
```

Each output carries the dropdown logic from §3 scoped to its own catalog. Writer reads `model_creative` + `model_technical` from the hub; every other consumer reads its own slot. Hub is the **only** node in the graph exposing a model dropdown.

---

## 8. Per-file change manifest

For implementation only — no edits today. Counts are estimates.

### 8a. New files

| Path                                          | LOC est. | Purpose                                       |
|-----------------------------------------------|---------:|-----------------------------------------------|
| `nodes/_otr_model_catalog.py`                 |     150  | Scan + curated catalog + validator            |
| `nodes/_otr_model_inputs.py`                  |      40  | `MissingModelInputError` + socket helpers     |
| `nodes/OTR_ModelHub.py` (if Shape B)          |     180  | The single pick node                          |
| `scripts/migrate_workflow_two_model_selector.py` |   80  | One-shot JSON re-writer                       |
| `tests/test_model_catalog_scan.py`            |     120  | Catalog unit tests                            |
| `tests/test_two_model_selector_wiring.py`     |     180  | End-to-end: writer broadcasts, consumers read |
| `web/js/otr_model_dropdown_color.js` (Path B) |     120  | Red-state rendering (deferred)                |

### 8b. Modified files

| Path                                  | Change                                                            |
|---------------------------------------|-------------------------------------------------------------------|
| `nodes/OTR_LedgerScriptWriter.py`     | Add `model_creative` + `model_technical` widgets; add output sockets; route Slot 2 to structured paths per the §2c table; remove old `_MODEL_CHOICES` literal in favor of catalog import. |
| `nodes/OTR_LedgerFreezeCascade.py`    | Delete `model_id` widget (lines 148-155 + `run()` signature param); add `model_technical` socket input.                                |
| `nodes/OTR_LFCPhase4Scene.py`         | Same delete + add (creative socket).                              |
| `nodes/OTR_LFCPhase5Voice.py`         | Same delete + add (technical socket).                             |
| `nodes/OTR_LFCPhase6Arc.py`           | Same delete + add (technical socket).                             |
| `nodes/musicgen_theme.py`             | Delete `model_id` widget (line 393); add `model_music` socket (if Shape B).      |
| `nodes/batch_audiogen_generator.py`   | Delete `model_id` widget (line 260); add `model_sfx` socket (if Shape B).        |
| `visual/llm_selector.py`              | Either delete file (preferred) OR convert to a passthrough.       |
| `nodes/__init__.py`                   | Register `OTR_ModelHub` (if Shape B); deregister `OTR_VisualLLMSelector` if deleted. |
| `nodes/vram_context_test.py`          | Test-bench — defer or carve-out per §10.                          |
| `workflows/otr_scifi_16gb_full.json`  | Widget array re-write + link table update + new node entry (if Shape B). |
| `tests/test_workflow_json_guardrails.py` | New assertions: no `model_id` widget anywhere except the hub / writer; writer widget order pinned. |

### 8c. Deleted symbols (per `feedback_no_legacy_back_compat`)

- `OTR_LedgerScriptWriter._MODEL_CHOICES` — replaced by catalog import. Delete the literal.
- `OTR_LedgerScriptWriter::DEFAULT_MODEL_ID` — replaced by catalog default. Delete.
- `OTR_LedgerFreezeCascade::DEFAULT_MODEL_ID` — same. Delete.
- `OTR_LFCPhase4Scene::DEFAULT_MODEL_ID` — same. Delete.
- `OTR_LFCPhase5Voice::DEFAULT_MODEL_ID` — same. Delete.
- `OTR_LFCPhase6Arc::DEFAULT_MODEL_ID` — same. Delete.
- `visual/llm_selector.py::_LLM_MODEL_CHOICES` — replaced by catalog. Delete file if going Shape B.

---

## 9. Test plan

### 9a. New tests

1. `tests/test_model_catalog_scan.py`
   - Catalog returns curated entries even when local cache is empty.
   - Catalog returns scanned-only entries when curated list is empty.
   - Validator rejects `..`, absolute paths, drive letters, backslashes.
   - Validator strips UI suffix (`[ALPHA]`, `(EXPERIMENTAL)`) before allow-list check.
   - Allow-list bypass falls back to curated default + WARN log.

2. `tests/test_two_model_selector_wiring.py`
   - Writer with `model_creative != model_technical` routes structured paths to technical (mirrors `test_two_llm_split.py`).
   - Writer with `model_creative == model_technical` behaves identically to single-LLM mode (back-compat check).
   - Missing creative socket on consumer → `MissingModelInputError`.
   - Missing technical socket on consumer → `MissingModelInputError`.

3. `tests/test_workflow_json_guardrails.py` (extend existing)
   - Pin Story Writer widget order.
   - Assert no node in the workflow JSON carries `widgets_values` containing a model_id string other than the writer / hub.
   - Assert the writer's two new outputs are link sources for the right consumers (creative → Phase 4, technical → cascade + Phase 5 + Phase 6).

### 9b. Existing tests likely to break

- `tests/test_two_llm_split.py` — design twin of the new selector. Either kept as-is (the resolver semantics still apply at the writer's Slot 1 / Slot 2 boundary) or absorbed into `test_two_model_selector_wiring.py`. Decision at implementation time.
- `tests/test_workflow_json_guardrails.py::TestWriterStyleSentinelDefault` — pins widget index for `style`. Index shifts by +1; the test needs the new index.
- `tests/test_lfc_*` — anything that calls a phase node with `model_id="..."` positionally needs to switch to the socket-pulled path.

### 9c. Bug Bible regression — mandatory per CLAUDE.md

Run after every commit in the implementation sprint:

```
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v
pytest tests/test_dropdown_guardrails.py -v
pytest tests/test_core.py -v
pytest tests/v2/test_audio_byte_identical.py -v
```

C7 byte-identity (audio) must hold across the change — if Slot 1 defaults to the same model the writer used before, audio output should be byte-identical to baseline. If the audio path drifts, **revert immediately** per Prime Directive 1.

---

## 10. Open decisions for Jeffrey

These are real forks in the design — pick before implementation starts.

1. **Shape A vs B vs C for non-LLM models (§7).**
   Recommendation: **B** (OTR_ModelHub node). Cleanest scaling, matches existing precedent.

2. **Path A vs B for the red-state UX (§4a).**
   Recommendation: **A first** (`[NOT DOWNLOADED]` label suffix), Path B JS extension as follow-up sprint.

3. **Auto-download default ON or OFF?**
   Recommendation: **ON** with a `OTR_MODEL_HUB_AUTO_DOWNLOAD=0` env-var off switch for the offline case.

4. **`vram_context_test.py` (test bench) — touch or skip?**
   Recommendation: **skip in the first PR**. It's a test bench; carve-out documented in the PR description. Loop it back in once the main change is green.

5. **Keep `OTR_VisualLLMSelector` as a thin passthrough, or delete it?**
   Recommendation: **delete it** per `feedback_no_legacy_back_compat`. Visual consumers wire to the writer's `model_creative` directly (or to OTR_ModelHub if Shape B wins).

6. **Slot 2 default — same as Slot 1 (single-LLM mode) or a different small model?**
   Recommendation: **same as Slot 1 by default** (`mistralai/Mistral-Nemo-Instruct-2407`). Preserves audio C7 baseline. Users opt into the split by changing Slot 2.

---

## 11. Round-robin trigger points (per CLAUDE.md)

Skip the round-robin for the mechanical pieces (widget moves, JSON re-write, test scaffolding — single-engineer execution).

Trigger the round-robin for:

- **The non-LLM Shape A/B/C decision (§7).** Architectural; affects every downstream node type.
- **The VRAM behavior of two models being requested back-to-back in one run.** `_otr_model_loader.py` caches one model at a time. If Slot 1 = Mistral-Nemo (24 GB on disk, ~12 GB VRAM) and Slot 2 = a smaller technical model, the loader must `_flush_vram_keep_llm()` between phases per Prime Directive 2 — not `force_vram_offload()`. Worth a ChatGPT + Gemini sanity check on the swap pattern before wiring.
- **The auto-download UX from a Windows-only HF path.** First-time downloads through `huggingface_hub` on Windows have edge cases (long paths, junction-resolution under `HF_HOME`). Round-robin worth running before shipping.

---

## 12. Rollout phases (suggested)

| Phase | Scope                                                                                       | Gate                                                  |
|------:|---------------------------------------------------------------------------------------------|-------------------------------------------------------|
|     0 | Decide §10 questions 1-6.                                                                   | Jeffrey sign-off.                                     |
|     1 | Land `_otr_model_catalog.py` + tests. No widget changes yet. Catalog importable + scanned.  | `test_model_catalog_scan.py` green.                   |
|     2 | Add Slot 1 + Slot 2 widgets to writer. Old `model_id` widget deleted (no transition shim).  | Bug Bible regression green + audio C7 byte-identical. |
|     3 | Rewire every consumer (Freeze Cascade + LFC Phase 4/5/6 + visual selector).                 | `test_two_model_selector_wiring.py` green.            |
|     4 | OTR_ModelHub (if Shape B) + non-LLM consumer rewires.                                       | Workflow loads in ComfyUI Desktop with zero warnings. |
|     5 | Path B JS extension for red-state UX (optional follow-up sprint).                           | Visual confirmation in ComfyUI Desktop.               |

Phases 2 + 3 are the load-bearing work. Phases 1 + 4 + 5 are scaffolding around them.

---

## 13. Estimate

- Phase 0: 1 conversation with Jeffrey.
- Phase 1: ~half a day. Pure module + tests.
- Phase 2: ~half a day. Writer widget surgery + audio C7 re-run.
- Phase 3: ~1 day. Touch 5 nodes, run regression after each.
- Phase 4: ~1 day (Shape B). Otherwise 0.5 day.
- Phase 5: separate sprint, ~half a day.

Total in-sprint: **2.5–3 working days**, deferred Phase 5 follow-up.

---

## 14. References

- Annotated screenshot from Jeffrey, 2026-05-13.
- `nodes/OTR_LedgerScriptWriter.py` (the keeper).
- `nodes/OTR_LedgerFreezeCascade.py:148` (removal target).
- `nodes/_otr_model_loader.py` (unchanged loader facade).
- `tests/test_two_llm_split.py` (the prior partial implementation).
- `visual/llm_selector.py` (the existing single-source precedent).
- `CLAUDE.md` Prime Directives 1, 2, 3 (audio C7, VRAM ceiling, wire-it-or-don't-ship-it).
- Memory: `feedback_no_legacy_back_compat`, `feedback_wire_it_or_dont_ship_it`, `feedback_minimum_json_files`, `feedback_lean_docs_fix_code`.

---

**End of scoping document. No code or workflow JSON modified.**
