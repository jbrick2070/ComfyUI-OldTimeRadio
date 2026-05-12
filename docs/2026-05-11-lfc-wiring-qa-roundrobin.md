# LFC Cascade Wiring — Round-Robin QA

**Repo:** `ComfyUI-OldTimeRadio`
**Branch:** `v2.0-alpha`
**HEAD:** `c1be3f0`
**Workflow file:** `workflows/otr_scifi_16gb_full.json`
**Scope of this doc:** just the wiring around the new
`OTR_LedgerFreezeCascade` node (id 62). Not the LLM prompts. Not
the phase internals. Wiring only.

Paste this whole document into ChatGPT + Gemini and ask for
validity feedback against the questions in §5. The full
workflow JSON is 47kB and most of it is unchanged from before the
LFC sprint — feeding the whole file just burns reviewer tokens
on cast pools and HuMo loader nodes.

---

## 1. What changed in this sprint

Node id **62** (`OTR_LedgerFreezeCascade`) was originally
`OTR_LedgerScriptReviewer` (a 3-pass cast-gated reviewer that
returned `reviewer_verdict`). Across the LFC sprint it:

- Renamed: type + S&R + title.
- Gained 8 new widgets (Phase 3 polish on/off, polish-announcer
  on/off, Phases 4 / 4.5 / 5 / 6 enable toggles, Phases 7 / 8
  enable toggles, VRAM ceiling).
- Renamed output socket 4: `reviewer_verdict` → `freeze_verdict`.
- Got rewired into the `news_used` chain so the writer's
  `news_used` flows THROUGH the cascade to SignalLostVideo
  instead of around it.

Old name still resolves via NODE_CLASS_MAPPINGS rename alias in
`__init__.py`, so any workflow JSON that still says
`OTR_LedgerScriptReviewer` keeps loading.

---

## 2. Cascade node (id 62) JSON slice

```json
{
  "id": 62,
  "type": "OTR_LedgerFreezeCascade",
  "title": "1b. Ledger Freeze Cascade (Phase 0..10)",
  "properties": {
    "Node name for S&R": "OTR_LedgerFreezeCascade"
  },
  "widgets_values": [
    "mistralai/Mistral-Nemo-Instruct-2407",
    false,    // enable_phase_3_polish
    false,    // polish_announcer_beats
    false,    // enable_phase_4_scene_coherence
    false,    // enable_phase_4_5_smart_suggestion
    false,    // enable_phase_5_voice_drift
    false,    // enable_phase_6_episode_arc
    true,     // enable_phase_7_audio_readiness
    true,     // enable_phase_8_video_readiness
    14.0      // vram_ceiling_gb
  ],
  "inputs": [
    { "name": "script_text",       "type": "STRING", "link": 106 },
    { "name": "script_json",       "type": "STRING", "link": 107 },
    { "name": "news_used",         "type": "STRING", "link": 108 },
    { "name": "estimated_minutes", "type": "INT",    "link": 109 }
  ],
  "outputs": [
    { "name": "script_text",       "type": "STRING", "links": [1],                 "slot_index": 0 },
    { "name": "script_json",       "type": "STRING", "links": [2, 12, 16, 19, 24], "slot_index": 1 },
    { "name": "news_used",         "type": "STRING", "links": [110],               "slot_index": 2 },
    { "name": "estimated_minutes", "type": "INT",    "links": [],                  "slot_index": 3 },
    { "name": "freeze_verdict",    "type": "STRING", "links": [],                  "slot_index": 4 }
  ]
}
```

### Cascade node INPUT_TYPES contract (Python side)

The cascade's `INPUT_TYPES()` declares the widgets in this iteration
order (Python `dict` insertion order is the canonical contract):

```
0  model_id                          STRING   default "mistralai/Mistral-Nemo-Instruct-2407"
1  enable_phase_3_polish             BOOLEAN  default False
2  polish_announcer_beats            BOOLEAN  default False
3  enable_phase_4_scene_coherence    BOOLEAN  default False
4  enable_phase_4_5_smart_suggestion BOOLEAN  default False
5  enable_phase_5_voice_drift        BOOLEAN  default False
6  enable_phase_6_episode_arc        BOOLEAN  default False
7  enable_phase_7_audio_readiness    BOOLEAN  default True
8  enable_phase_8_video_readiness    BOOLEAN  default True
9  vram_ceiling_gb                   FLOAT    default 14.0
```

`widgets_values` in the JSON is positional. Slot 0..9 must agree
with INPUT_TYPES iteration order above.

---

## 3. `news_used` link chain (W2 fix in commit 12.1)

Pre-fix the writer's `news_used` had two wires:
- one to SignalLostVideo (id 12), bypassing the cascade
- one to the cascade (id 62), but the cascade's `news_used`
  OUTPUT had `links=[]` — the cascade was a dead end on this
  field

Post-fix it's a single chain through the cascade:

```
writer(1).news_used
    └── link 108 ──> cascade(62).news_used [input slot 2]
                     │
                     │ (cascade.news_used is a passthrough
                     │  output -- the cascade does not mutate it)
                     │
                     └── link 110 ──> SignalLostVideo(12).news_used [input slot 2]
```

JSON state at HEAD `c1be3f0`:

```
last_link_id           = 110
writer.news_used.links = [108]
cascade.news_used.links = [110]
SignalLostVideo.news_used.input.link = 110
```

The data flowing through is byte-identical pre/post fix; this is
purely a topology change so the LFC node honors its
"passthrough outputs exist" contract.

---

## 4. Widget-to-orchestrator threading (Python side)

Each widget on the cascade node maps 1:1 to a kwarg on
`run_freeze_cascade` (Python). The node's `run()` method is the
glue:

```
Widget                              -> run_freeze_cascade kwarg
model_id                            -> (drives make_generate_fn + make_polish_generate_fn)
enable_phase_3_polish               -> enable_phase_3_polish
polish_announcer_beats              -> polish_announcer_beats
enable_phase_4_scene_coherence      -> enable_phase_4_scene_coherence
enable_phase_4_5_smart_suggestion   -> enable_phase_4_5_smart_suggestion
enable_phase_5_voice_drift          -> enable_phase_5_voice_drift
enable_phase_6_episode_arc          -> enable_phase_6_episode_arc
enable_phase_7_audio_readiness      -> enable_phase_7_audio_readiness
enable_phase_8_video_readiness      -> enable_phase_8_video_readiness
vram_ceiling_gb                     -> vram_ceiling_gb
```

Each of the seven `enable_phase_*` flags gates ONE specific cascade
phase. The corresponding phase-record (`meta.cleanup_passes`) is
always stamped — disabling a phase records `{"skipped": True}` on
`meta.<phase>_record` AND still appends an entry to
`meta.cleanup_passes` so the soak telemetry stays contiguous.

---

## 5. Wiring validity questions (for the round-robin)

Ask ChatGPT + Gemini to answer each of these in turn. Disagreements
between them are the actually-useful signal.

### Q1 — Positional widget order

Does the JSON `widgets_values` array agree with the Python
`INPUT_TYPES` iteration order in §2? Mismatch would mean every
slider sets the wrong widget at workflow load — usually a silent
bug because the types happen to line up (BOOL vs BOOL vs BOOL).

### Q2 — Output socket 4 rename

Was every reference to the OLD output name `reviewer_verdict`
removed? The fifth output is now `freeze_verdict`. Downstream
graphs that consumed slot 4 by **index** still work; graphs that
consumed it by **name** would break. The workflow shows
`freeze_verdict` with `links=[]` — nobody consumes it today. Is
that the right contract, or should we route it to a preview
node?

### Q3 — news_used link chain integrity

Verify the link chain in §3:
- Writer's `news_used` output should have EXACTLY `[108]`.
- Cascade's `news_used` output should have EXACTLY `[110]`.
- SignalLostVideo's `news_used` input link should be `110`.
- `last_link_id` should be ≥ `110`.

Any deviation = silent wire dropped on workflow load.

### Q4 — Defaults vs ADR intent

The ADR says:
- Phase 4 / 4.5 / 5 / 6 (heavy LLM phases) default OFF until soak.
- Phase 7 / 8 (deterministic, cheap) default ON.
- Phase 3 polish + polish_announcer_beats default OFF.
- VRAM ceiling default 14.0 GB.

Do the JSON `widgets_values` defaults in §2 match? On any disagreement
flag it -- defaults shipped wrong is a "the cascade is doing more
than the operator thinks" surprise.

### Q5 — Legacy back-compat surface

`nodes/OTR_LedgerScriptReviewer.py` is now a shim that re-exports
`OTR_LedgerFreezeCascade`. `__init__.py` has a `_RENAME_ALIASES`
entry mapping `"OTR_LedgerScriptReviewer"` to
`"OTR_LedgerFreezeCascade"`. A workflow JSON that still says
`"type": "OTR_LedgerScriptReviewer"` should load. Is this back-compat
surface sound, or is there a path where a legacy JSON ends up with
the wrong class?

### Q6 — Cascade as the only LFC node

Should the cascade be a single ComfyUI node (current state) or should
each phase be its own node so the operator can wire them
individually? Current state is one node + 7 BOOLEAN toggles. Trade-off:
single node = simple wiring, less granular control; per-phase nodes
= more graph clutter, finer control over phase ordering. Which is
the more correct design for the 5080 / single-operator workflow?

### Q7 — `freeze_verdict` orphan output

Slot 4 (`freeze_verdict`) currently has `links=[]`. Operator has to
inspect the ledger JSON to see what the cascade decided. Should this
get routed to a preview node (`ShowText|pysssss` is the de-facto
ComfyUI standard for text preview but it's an extension dep)? Or is
the operator's path to inspect `meta.freeze_verdict` in the saved
ledger acceptable?

---

## 6. Acceptance criteria for "wiring is clean"

Round-robin should agree on these. If both reviewers can answer
YES to all six, the wiring is shippable.

1. `widgets_values` array length == 10 (model_id + 8 BOOL + 1 FLOAT).
2. `widgets_values` positional defaults exactly match the
   INPUT_TYPES defaults in §2.
3. `news_used` link chain shape matches §3 exactly.
4. `last_link_id` ≥ `110` (the highest link id in the array).
5. The cascade node's title and S&R property both reference the new
   class name `OTR_LedgerFreezeCascade`.
6. Cascade output slot 4 is named `freeze_verdict` (not the old
   `reviewer_verdict`).

---

## 7. What to send the round-robin

Just this document. The workflow JSON is also in the repo if a
reviewer pushes for a full file, but for the questions in §5 the
slices in §2 + §3 are the only relevant cuts. Sending the whole
JSON usually produces shallower feedback ("looks fine") because
the reviewer's attention dilutes across 1500 lines of unrelated
graph state.

If a reviewer wants more context on what the cascade actually
does internally, point them at the ADR (`docs/2026-05-11-multi-turn-polish-adr.md`)
and the QA handoff (`docs/2026-05-11-multi-turn-polish-qa-handoff.md`).

---

**End of wiring QA.**
