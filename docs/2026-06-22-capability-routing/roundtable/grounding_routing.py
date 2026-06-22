# GROUNDING -- verbatim code for the capability-routing design. Verify against THIS.

# ===== nodes/_otr_shared/role_compat.py =====

ROLE_AVAILABLE_INPUTS = {
    # announcer_visual: supplies all four (still + prompt + audio slice + base clip)
    "announcer_visual": frozenset({"text_prompt", "init_image", "audio_ref", "base_clip_ref"}),
    "music_visual":     frozenset({"text_prompt", "init_image", "audio_ref", "base_clip_ref"}),
    "scene_broll":      frozenset({"text_prompt", "init_image", "audio_ref", "base_clip_ref"}),
    "character_video":  frozenset({"text_prompt", "init_image", "base_clip_ref"}),
    "background_abstract": frozenset({"text_prompt"}),
}

def engine_fits_role(descriptor, role: str) -> bool:
    """True iff descriptor can serve role (fail-closed).
    Requires (1) role in the engine's `roles` AND (2) every token in the engine's
    `required_inputs` is available in the role."""
    available = role_available_inputs(role)
    if not isinstance(descriptor, dict):
        return False
    roles = descriptor.get("roles")
    required = descriptor.get("required_inputs")
    if roles is None or required is None:
        return False
    if role not in tuple(roles):          # <-- THE WHITELIST GATE (the gap)
        return False
    required_set = set(required)
    if not required_set <= INPUT_TOKENS:
        return False
    return required_set <= available       # <-- the capability match

# ===== engine declarations (each in nodes/_otr_video_engines/eng_*.py) =====
# wan_i2v:       required_inputs = ("init_image",)              default_roles = ()   render_aspect="wide"
# ltx_video:     required_inputs = ("text_prompt",)            default_roles = ()   render_aspect="wide"
# character_3d:  required_inputs = ("audio_ref", "init_image") default_roles = ()
# visualizer:    required_inputs = ("audio_ref",)              default_roles = ()
# ltx_av_music:  audio-conditioned (audio_ref); the music/announcer DEFAULT
# humo / humo_1.7B: audio_driven_face (audio_ref + init_image)
# cheap_families: station_card default_roles=("announcer_visual",), broll=("scene_broll",), etc.
#
# OPEN: where does the director descriptor's `roles` come from? otr_video_director.py builds
# descriptors with "required_inputs": tuple(getattr(eng, "required_inputs", ())). The `roles`
# key source (default_roles? a separate `roles` attr? capability-derived?) is the thing to confirm --
# it is what excludes wan_i2v from announcer_visual despite the input match passing.

# ===== render-side SECOND gate: nodes/_otr_video_engines/render_driver.py =====
def _present_request_tokens(request):
    present = set()
    if request.get("text_prompt"): present.add("text_prompt")
    if "init_image" in (request.get("asset_refs") or {}): present.add("init_image")
    if request.get("audio_ref") is not None: present.add("audio_ref")
    if request.get("base_clip_ref"): present.add("base_clip_ref")
    return present

def _assert_family_inputs_satisfiable(engine_name, request):
    # uses FAMILY_REQUIRED_INPUTS (by FAMILY) vs _present_request_tokens; raises FamilyInputGap LOUD.
    ...

# LIVE WALL: OTR_VideoDirector: engine 'wan_i2v' does not fit any role for slot
# 'announcer_video_model' ('announcer_visual',) -- input match PASSES (announcer supplies init_image)
# but the `roles` whitelist excludes wan_i2v. ltx_video (text_prompt) fits everywhere.
