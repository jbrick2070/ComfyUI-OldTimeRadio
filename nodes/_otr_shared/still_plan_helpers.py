"""Per-model STILL-PLAN schema and pure helpers (S1 of the still-plans build).

Sole authority for the closed :class:`StillPlanRow` schema and the ONE pure
helper every reader must go through, ``resolve_row_aspect``. Also provides
schema validation utilities the POST-REGISTRATION audit runs against every
adapter's ``still_plan`` class attribute.

**Nothing in this module reads the plan for production**; adapters declare a
``still_plan`` next to ``family`` / ``render_aspect`` / ``accepts_still`` and
the S1 audit test proves the invariant
``registry.CAPABILITIES.keys() == all_engine_names() == valid-plan owners``.
S2 wires the plan into the seven consumers atomically.

Spec of record:
``docs/2026-07-25-still-plans-locked-build-spec.md`` (locked @ ``84328aa1``),
section 5 (schema) and section 6 (declaration and audit).

Two doctrine points from the spec that shape this module:

- **A missing plan is UNKNOWN and fails closed. An explicit ``()`` means
  "needs no images".** The audit MUST treat "no ``still_plan`` attribute at
  all" as an error and "``still_plan = ()``" as a valid declaration.
- **Validation is a POST-REGISTRATION AUDIT, never decorator-time.**
  ``nodes/_otr_video_engines/__init__.py`` wraps every adapter import in
  ``try: ... except Exception: pass``, so a ``ValueError`` raised in a class
  body or the ``@register`` decorator would SILENTLY DELETE the engine from
  the menu -- a plan typo would present as "that model disappeared". Therefore
  this module NEVER validates at construction time; :class:`StillPlanRow` is
  a plain :func:`collections.namedtuple` and validation is triggered by the
  audit outside the guarded imports.

stdlib-only, cold-import clean (V-12): importing this module pulls in NOTHING
external (no torch / no registry / no ComfyUI). UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

from collections import namedtuple


# ---------------------------------------------------------------------------
# CLOSED ENUMS -- exact tokens per spec section 5. Adding a fifth token to
# any of these is an operator decision, never a coder's judgment call.
# ---------------------------------------------------------------------------

#: The kind of still the row describes. ``portrait`` is the one kind whose
#: prompt SUBJECT comes from cast (``prompt_field_source == None``,
#: ``visual_style == None`` per spec section 4); every other kind carries an
#: authored ``framing_geometry`` at layer 2.
KIND_PORTRAIT = "portrait"
KIND_SCENE_OPEN = "scene_open"
KIND_SCENE_BEAT = "scene_beat"
KIND_SCENE_CHARACTER = "scene_character"
KIND_MESH_FODDER = "mesh_fodder"
KIND_SCENE_BACKGROUND_PLATE = "scene_background_plate"

VALID_KINDS = frozenset((
    KIND_PORTRAIT,
    KIND_SCENE_OPEN,
    KIND_SCENE_BEAT,
    KIND_SCENE_CHARACTER,
    KIND_MESH_FODDER,
    KIND_SCENE_BACKGROUND_PLATE,
))

#: How the row multiplies over an episode. ``per_beat`` covers "per episode"
#: too -- the enumerator upstream filters ``scene_open`` to the opening beat
#: (the spec does not add a ``per_episode`` token because the enumerator's
#: opening-beat filter is already the ONE place that decision is made).
CARDINALITY_PER_BEAT = "per_beat"
CARDINALITY_PER_SUBJECT = "per_subject"
CARDINALITY_PER_RECURRING_SUBJECT = "per_recurring_subject"
CARDINALITY_PER_BOOKEND_ROLE = "per_bookend_role"

VALID_CARDINALITIES = frozenset((
    CARDINALITY_PER_BEAT,
    CARDINALITY_PER_SUBJECT,
    CARDINALITY_PER_RECURRING_SUBJECT,
    CARDINALITY_PER_BOOKEND_ROLE,
))

#: What validator slot the row satisfies. ``scene_background_plate`` carries
#: ``target_class == "scene"`` per spec section 5: the render_driver matches
#: ``kind.startswith("scene_")`` then applies plate-over-scene precedence, so
#: a cutover that filters only the three cinematic scene kinds would break
#: every mesh beat.
TARGET_CLASS_SCENE = "scene"
TARGET_CLASS_PORTRAIT = "portrait"
TARGET_CLASS_MESH = "mesh"

VALID_TARGET_CLASSES = frozenset((
    TARGET_CLASS_SCENE,
    TARGET_CLASS_PORTRAIT,
    TARGET_CLASS_MESH,
))

#: Aspect on the still. ``inherit_engine`` is NEVER compared directly; every
#: reader goes through :func:`resolve_row_aspect` (spec section 5).
ASPECT_WIDE = "wide"
ASPECT_PORTRAIT = "portrait"
ASPECT_INHERIT_ENGINE = "inherit_engine"

VALID_ASPECTS = frozenset((
    ASPECT_WIDE,
    ASPECT_PORTRAIT,
    ASPECT_INHERIT_ENGINE,
))

#: The closed activation enum (never a bool, never an expression per spec
#: section 5). ``when_engine_talking`` evaluates the engine's own
#: ``wants_talking_prompt()`` hook; ``when_ltx_i2v_enabled`` evaluates the
#: FROZEN ``routing_state.enable_ltx_i2v`` (S0b's captured field). Adding a
#: fifth token is an operator decision.
REQUIRED_ALWAYS = "always"
REQUIRED_NEVER = "never"
REQUIRED_WHEN_ENGINE_TALKING = "when_engine_talking"
REQUIRED_WHEN_LTX_I2V_ENABLED = "when_ltx_i2v_enabled"

VALID_REQUIRED = frozenset((
    REQUIRED_ALWAYS,
    REQUIRED_NEVER,
    REQUIRED_WHEN_ENGINE_TALKING,
    REQUIRED_WHEN_LTX_I2V_ENABLED,
))

#: Style-tail policy consulted by the visual-style authority at prompt time.
#: The plan NAMES the policy, never the tokens (spec section 4).
STYLE_TAIL_FULL = "full"
STYLE_TAIL_MINIMAL_CLEAN = "minimal_clean"

VALID_STYLE_TAIL_POLICIES = frozenset((
    STYLE_TAIL_FULL,
    STYLE_TAIL_MINIMAL_CLEAN,
))


# ---------------------------------------------------------------------------
# The row itself. A plain namedtuple so a class-body construction cannot
# raise -- the audit is what fails closed on invalid data (spec section 6:
# "Validation is a POST-REGISTRATION AUDIT, never decorator-time").
# ---------------------------------------------------------------------------
StillPlanRow = namedtuple(
    "StillPlanRow",
    ("kind", "cardinality", "target_class", "aspect", "required",
     "framing_geometry", "style_tail_policy"),
)


# ---------------------------------------------------------------------------
# The ONE aspect resolver. Every reader of a plan row that needs a concrete
# ``wide`` / ``portrait`` answer MUST go through this helper. ``engine_facts``
# is the ``{"engine_id", "family", "provider_side"}`` descriptor S0b will
# introduce; only ``family`` and the engine's shipped ``render_aspect`` are
# read here. Pure over inputs.
# ---------------------------------------------------------------------------
def resolve_row_aspect(row, engine_facts):
    """Return ``"wide"`` or ``"portrait"`` for one plan row, resolving
    ``aspect="inherit_engine"`` against the engine's shipped ``render_aspect``.

    ``engine_facts`` is a plain dict-or-namedtuple with at least
    ``engine_render_aspect`` (or the engine's ``render_aspect`` attribute --
    accepted BOTH ways so an S1 caller building facts by hand doesn't have to
    guess S0b's key names). Unknown / missing engine aspect resolves to
    ``portrait`` (the safe legacy look at HEAD -- matches
    ``OTR_VideoDirector._role_aspects``).

    Direct ``inherit_engine`` comparisons are forbidden per spec section 5:
    an audit that filters on the string ``"inherit_engine"`` in a consumer
    is a bug this helper exists to prevent.
    """
    aspect = str(getattr(row, "aspect", None) or "")
    if aspect == ASPECT_WIDE:
        return ASPECT_WIDE
    if aspect == ASPECT_PORTRAIT:
        return ASPECT_PORTRAIT
    if aspect == ASPECT_INHERIT_ENGINE:
        engine_aspect = None
        if isinstance(engine_facts, dict):
            engine_aspect = (engine_facts.get("engine_render_aspect")
                             or engine_facts.get("render_aspect"))
        else:
            engine_aspect = (getattr(engine_facts, "engine_render_aspect", None)
                             or getattr(engine_facts, "render_aspect", None))
        if engine_aspect == ASPECT_WIDE:
            return ASPECT_WIDE
        return ASPECT_PORTRAIT
    # An unknown aspect on a row that reached us here is a defect the audit
    # should have caught; degrade to the safe legacy look rather than raise
    # inside a hot render path.
    return ASPECT_PORTRAIT


# ---------------------------------------------------------------------------
# The POST-REGISTRATION audit surface. Pure functions the S1 test file
# imports; no consumer wires these into production paths at S1.
# ---------------------------------------------------------------------------
def validate_still_plan_row(row):
    """Raise ``ValueError`` with a specific field name if ``row`` is not a
    valid :class:`StillPlanRow` per spec section 5. Never mutates."""
    if not isinstance(row, tuple):
        raise ValueError(
            "still_plan row must be a tuple / StillPlanRow, got %s"
            % type(row).__name__)
    if len(row) != len(StillPlanRow._fields):
        raise ValueError(
            "still_plan row must have %d fields (%s), got %d"
            % (len(StillPlanRow._fields), StillPlanRow._fields, len(row)))
    for field_name, valid_set in (
            ("kind", VALID_KINDS),
            ("cardinality", VALID_CARDINALITIES),
            ("target_class", VALID_TARGET_CLASSES),
            ("aspect", VALID_ASPECTS),
            ("required", VALID_REQUIRED),
            ("style_tail_policy", VALID_STYLE_TAIL_POLICIES)):
        value = getattr(row, field_name, None) if hasattr(row, field_name) \
            else row[StillPlanRow._fields.index(field_name)]
        if value not in valid_set:
            raise ValueError(
                "still_plan row field %r has invalid value %r; allowed: %s"
                % (field_name, value, sorted(valid_set)))
    # framing_geometry: authored TEXT; may be empty but must be a str.
    framing = row[StillPlanRow._fields.index("framing_geometry")] \
        if not hasattr(row, "framing_geometry") else row.framing_geometry
    if not isinstance(framing, str):
        raise ValueError(
            "still_plan row field 'framing_geometry' must be a str, got %s"
            % type(framing).__name__)


def validate_still_plan(plan):
    """Raise ``ValueError`` if ``plan`` is not a valid
    ``tuple[StillPlanRow, ...]``. An empty tuple ``()`` is VALID and means
    "engine needs no images" (spec section 6). Never mutates."""
    if not isinstance(plan, tuple):
        raise ValueError(
            "still_plan must be a tuple (of StillPlanRow), got %s"
            % type(plan).__name__)
    for i, row in enumerate(plan):
        try:
            validate_still_plan_row(row)
        except ValueError as exc:
            raise ValueError(
                "still_plan row %d: %s" % (i, exc)) from exc


_MISSING_STILL_PLAN = object()


def engine_has_still_plan(engine):
    """``True`` iff ``engine`` declares a ``still_plan`` attribute (including
    an empty tuple ``()``). This is the audit's UNKNOWN vs "needs no images"
    boundary: missing attribute -> ``False``; explicit ``()`` -> ``True``."""
    plan = getattr(engine, "still_plan", _MISSING_STILL_PLAN)
    return plan is not _MISSING_STILL_PLAN and isinstance(plan, tuple)


def engine_has_valid_still_plan(engine):
    """``True`` iff ``engine`` declares a ``still_plan`` AND that plan
    validates. Wraps :func:`validate_still_plan` in a bool -- the audit test
    uses this for the SET membership answer and calls
    :func:`validate_still_plan` separately for the specific failure message
    on a diff."""
    if not engine_has_still_plan(engine):
        return False
    try:
        validate_still_plan(getattr(engine, "still_plan"))
    except ValueError:
        return False
    return True


__all__ = [
    "StillPlanRow",
    "KIND_PORTRAIT",
    "KIND_SCENE_OPEN",
    "KIND_SCENE_BEAT",
    "KIND_SCENE_CHARACTER",
    "KIND_MESH_FODDER",
    "KIND_SCENE_BACKGROUND_PLATE",
    "VALID_KINDS",
    "CARDINALITY_PER_BEAT",
    "CARDINALITY_PER_SUBJECT",
    "CARDINALITY_PER_RECURRING_SUBJECT",
    "CARDINALITY_PER_BOOKEND_ROLE",
    "VALID_CARDINALITIES",
    "TARGET_CLASS_SCENE",
    "TARGET_CLASS_PORTRAIT",
    "TARGET_CLASS_MESH",
    "VALID_TARGET_CLASSES",
    "ASPECT_WIDE",
    "ASPECT_PORTRAIT",
    "ASPECT_INHERIT_ENGINE",
    "VALID_ASPECTS",
    "REQUIRED_ALWAYS",
    "REQUIRED_NEVER",
    "REQUIRED_WHEN_ENGINE_TALKING",
    "REQUIRED_WHEN_LTX_I2V_ENABLED",
    "VALID_REQUIRED",
    "STYLE_TAIL_FULL",
    "STYLE_TAIL_MINIMAL_CLEAN",
    "VALID_STYLE_TAIL_POLICIES",
    "resolve_row_aspect",
    "validate_still_plan_row",
    "validate_still_plan",
    "engine_has_still_plan",
    "engine_has_valid_still_plan",
]
