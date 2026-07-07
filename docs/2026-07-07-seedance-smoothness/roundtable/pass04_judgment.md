# Pass 04 Judgment - Final

## Verdict

Implemented.

The final plan converged on a Seedance-only prompt conditioner, not a workflow
JSON change and not a global style-pack edit.

## What Changed

- `nodes/_otr_video_engines/eng_cloud_video.py`
  - Added `_condition_seedance_prompt()`.
  - Added Seedance smooth-motion constants and regex softeners.
  - Applied conditioning only inside
    `CloudSeedance2Engine._partner_inputs()`.
  - Preserved the installed Partner Node payload shape.
  - Preserved Seedance duration clamping to `4..15s`.
  - Added bounded structured logging for prompt hashes/excerpts and requested
    duration.

- `tests/test_cloud_video_adapters.py`
  - Added tests for risky `music_open` softening.
  - Added idempotence coverage.
  - Added empty-prompt helper guard coverage.
  - Strengthened Seedance request-shape assertions.
  - Added under-minimum duration coverage.
  - Added coverage proving Wan/Kling/Pixverse prompts are unchanged.

## JSON Decision

No workflow JSON edit was needed.

The real workflow already feeds `text_prompt`, `init_image`, and `audio_ref`
into the Seedance engine path. This change alters only the adapter's internal
prompt text before it calls the Partner Node.

No visual style JSON edit was made either. `sci_fi_radio.json` contains rough
opener verbs, but those are deliberately part of the LTX motion register. The
Seedance-specific conditioner keeps LTX behavior intact while stabilizing
Seedance.

## Duration Decision

For cloud providers with a minimum duration, render the provider minimum and
trim to the audio beat.

For Seedance specifically, beats shorter than 4 seconds still request a 4 second
provider clip. `OTR_SilentComposite` keeps the head frames needed for the beat.
The smooth-motion clause makes those head frames useful by asking motion to
begin immediately.

## Verification

- `tests/test_cloud_video_adapters.py`: `19 passed`
- Full repo suite: `6643 passed, 34 skipped, 2 xfailed`
- Bug Bible: `16 passed, 7 skipped, 3 xfailed`
- `py_compile` on touched Python files: passed
- `git diff --check` on touched Python files: passed

## Roundtable Spend

- Pass 01: about `$0.1132`
- Pass 02: about `$0.1296`
- Pass 03: about `$0.1491`
- Pass 04: about `$0.1285`
- Total: about `$0.5204`
