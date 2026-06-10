<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The plan lacks concrete definitions for the profile format, generator mechanism, VRAM enforcement, and model manifestâwithout these, it cannot be built.

MUST-FIX BEFORE BUILD:
1. [Strawman shape / Open questions] The profile format and its mapping to widget values and `OTR_ENABLE_*` flags are undefined. The generator cannot translate a profile into a valid tier JSON without a spec. **Fix:** Define a profile schema (e.g., JSON with `hardware_tier`, `enabled_engines`, per-role engine overrides, etc.) and specify exactly how each field sets env flags and patches Director widgets.
2. [Refined recommendation] The generatorâs method for modifying the master JSON is unspecified. `patch_widget_by_name` requires live `/object_info` schemas from a running ComfyUIâa heavy, undocumented dependency for a build step. **Fix:** Either document that the generator needs a ComfyUI server, or redesign to apply the profile at runtime (load master + profile â patch in-process) and drop the pre-generation step. If generation is kept, provide an offline schema snapshot or a nameâbased patcher that does not require a live server.
3. [The hard parts / VRAM safety] No mechanism ties engines to VRAM requirements or enforces the â¤14.5â¯GB budget per tier. The 8â¯GB tier must autoâexclude heavy engines, but how? **Fix:** Add a `vram_estimate_mb` attribute to engine adapters (or a separate capability table) and have the profile validator reject combinations that exceed the tierâs ceiling.
4. [The drift bug, mechanically] The plan states headless scripts will use generated tier JSONs, but does not detail the required changes to `queue_smoke.py` and soak runners. **Fix:** Specify that headless scripts will load the appropriate tier JSON (e.g., `otr_8gb_lite.json`) instead of the master, and remove hardâcoded widget patches. If they still use the master, they must read the profile and apply patches consistently.
5. [Modelâasset acquisition] The perâprofile model manifest is mentioned but not defined (format, location, generation). The installer cannot be built. **Fix:** Define a manifest format (e.g., JSON mapping `engine_id` â list of model files/URLs) and a process to generate it from engine metadata, or cut the manifest and have the installer query engine requirements dynamically.

SHOULD-FIX:
1. [Strawman shape / Reconciliation] Ambiguity: will endâusers load the master graph with env flags, or preâgenerated tier JSONs? The plan says âsame graph every time; only the profile changesâ but also âship preâdialed files.â Clarify the userâfacing artifact and how the wizard selects it.
2. [Refined recommendation] The generator introduces a build step that must be run after every master change. Without CI enforcement, drift will recur. Add a CI check that validates generated tier JSONs match the master + profile.
3. [The hard parts / Combinatorial validity] The plan mentions testing via representative profiles but does not list them concretely. Provide a fixed set of profiles (e.g., 16GBâfull: all engines; 8GBâlite: exclude 14B, etc.) with expected engine sets.
4. [Grounded crossâplatform audit] The suggestion to âstandardize the inâprocess engines on sdpa / xformersâ is a change that affects the graph and profile. Specify how this standardization is reflected in the profile (e.g., a flag for attention backend) and whether it requires graph changes.

OPTIONAL / NICE-TO-HAVE:
- Provide a JSON Schema for the profile file.
- Add a `--profile` flag to headless scripts to select the tier JSON.
- Autoâdetection of hardware in the wizard could be a separate module.

CUT THESE (overâengineering):
- The perâprofile model manifest can be replaced by dynamic resolution: the installer reads the profile, queries the engine registry for required models, and downloads them. This avoids maintaining a separate manifest artifact. Safe to cut if the engine registry can expose model requirements (not yet shown in grounding, but can be added).