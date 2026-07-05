# Prompt Surgery Checklist

Status: scaffold checklist. This is not production code.

Goal: clone/reuse good story machinery, then surgically replace only the source-
or story-model-specific prompt language.

Important correction: this is not only visual cleanup. Story LLM prompts and
Python routing also carry sci-fi/news assumptions. The target is to repurpose
the same ledger slots with different prompt profiles and source-specific logic,
not merely move old phrases into a new file.

Legend:

- `SHARED`: keep common radio-drama craft.
- `PROFILE`: replace with `StoryPromptProfile` field.
- `STORY_MODEL`: source-scoped story model guardrail.
- `VISUAL_POLICY`: defer to visual-style policy stage.
- `DEFERRED/DEAD`: do not touch until proven active.

## Ledger/Story Prompt Sites

| File | Action | Notes |
|---|---|---|
| `nodes/news_interpreter.py` | PROFILE | Science/news interpreter becomes one source brain; archive source brain should not call it. |
| `nodes/_otr_outline.py` | PROFILE + STORY_MODEL | Add defaulted `OutlineRequest` fields and `outline_system_prompt` override path. |
| `nodes/_otr_pitch_room.py` | PROFILE | System prompt and story-form language must be profile-aware. |
| `nodes/_otr_story_select.py` | PROFILE | System prompt must not hardcode science-fiction for media archive. |
| `nodes/_otr_dramatic_state_llm.py` | PROFILE | Replace `NEWS KEY TERMS`, `NEWS PREMISE`, and "news event" language. |
| `nodes/_otr_line_composer.py` | PROFILE | Grounding instruction and coda mode become source-aware; coda returns `LineResult`. |
| `nodes/_otr_casting.py` | SHARED + PROFILE | Cast craft stays shared; source/casting brief comes from active packet. |
| `nodes/OTR_LedgerScriptWriter.py` | PROFILE + TRANSPLANT | Title/coda/source resolution path changes only in production transplant. |
| `nodes/_otr_style_picker.py` | PROFILE + STORY_MODEL | Add override kwargs; never run sci-fi inventor prompt for media archive. |

## Visual Prompt Sites

Visual/video prompt extraction is separate from the first source/story
transplant.

| File | Action | Notes |
|---|---|---|
| `nodes/_otr_story_brief_helpers.py` | VISUAL_POLICY | Shared finish seam and forbidden-term scrub owner. |
| `nodes/otr_meta_brief_image_prompt.py` | VISUAL_POLICY | Direct `IMAGE_GRADE_TAIL` appends are V3 visual-stage work. |
| `nodes/otr_shot_lock.py` | VISUAL_POLICY | Stamp/read `meta.visual_style` during visual stage. |
| `nodes/_otr_video_engines/render_driver.py` | VISUAL_POLICY | Deep fallback/motion prompt extraction is staged and risky. |

## Media Archive Forbidden Drift

Do not let media archive prompts become sci-fi anthology prompts.

Forbidden drift examples should be enforced by tests/metadata, not necessarily
rendered verbatim into live prompts:

- Star-Trek-style mission plot
- Amazing-Stories-style twist anthology
- spaceship rescue
- laboratory containment breach
- generic science experiment emergency
