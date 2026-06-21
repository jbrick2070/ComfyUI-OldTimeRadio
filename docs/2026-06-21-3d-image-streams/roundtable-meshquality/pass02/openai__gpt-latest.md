<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The proposed behavior is not code-ready against the grounding: current render_clip still always projects the still, the workflow/wiring claim is unverified, and several Blender-stage edge cases need explicit fixes/tests before build.

MUST-FIX BEFORE BUILD:
1. [Chunk B / eng_mesh_stage.render_clip] Current grounded code always passes `portrait=still` to `build_blender_cmd(...)`, so the proposed default `surface="gradient"` / “do NOT pass the fodder photo” is not what will run. Concrete fix: change `render_clip` to compute:
   - default: `surface="gradient"`, `portrait=""`;
   - if `os.environ.get("OTR_MESH_PROJECT_PORTRAIT") == "1"`: `surface="portrait"`, `portrait=still`;
   then pass both to `build_blender_cmd`. Add a unit test that default render command contains `--surface gradient` and no `--portrait`, and opt-in contains `--surface portrait --portrait <still>`.

2. [Chunk B / build_blender_cmd + parse_stage_args] The grounding `build_blender_cmd` has no `surface` parameter and `parse_stage_args` has no `--surface`; any implementation that appends `--surface` before updating the parser will make Blender fail with an argparse error. Concrete fix: add `surface=""` keyword to `build_blender_cmd`, append `["--surface", surface]` only when nonempty, and add `p.add_argument("--surface", choices=("flat", "gradient", "portrait"), default="flat")` in `parse_stage_args`. Add tests for omitted surface byte-identical legacy command and parsed choices.

3. [Q1 / wiring] The “NO workflow JSON change” claim is not confirmable from the grounding because `workflows/otr_scifi_16gb_full.json`, `OT