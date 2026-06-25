# Verbatim grounding excerpt -- nodes/_otr_radio_editor.py

These are exact copies of the relevant blocks (line numbers as of HEAD
dc438761). The full file is ~1600 lines; only the schema, the action set, and
the structured_call invocation are reproduced. The companion grounding file is
the REAL `nodes/_otr_structured_call.py` (whole).

## Schema (the proven-failure schema is NESTED) -- lines 253-308

```python
class BeatEdit(BaseModel):
    """One edit instruction against a fixed beat index."""

    beat_index: int = Field(..., ge=0, description="0-based index into the voiced-beat list")
    action: str = Field(
        ...,
        description="one of the two-tier action set",
    )
    new_line: Optional[str] = Field(
        default=None,
        description="replacement prose for SHORTEN/RECOMPOSE/SPLIT/MERGE",
    )
    merge_with_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="MERGE_SHORT_LINES only: adjacent same-speaker beat to fold in",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_field_aliases(cls, data):
        """BUG-LOCAL-303: LLMs (claude-opus included) routinely emit the
        shortened field name ``index`` instead of the schema's ``beat_index``
        (and ``merge_with`` for ``merge_with_index``). Accept those as aliases
        so the length / micro-repair pass validates on attempt 1 instead of
        burning 2-3 credit-billed structured-call retries on a pure field-name
        mismatch. Best-effort: only a plain dict is remapped, and an explicit
        ``beat_index`` always wins over ``index``."""
        if isinstance(data, dict):
            if "beat_index" not in data and "index" in data:
                data = {**data, "beat_index": data["index"]}
            if "merge_with_index" not in data and "merge_with" in data:
                data = {**data, "merge_with_index": data["merge_with"]}
        return data

    def is_tier1(self) -> bool:
        return self.action in TIER1_ACTIONS

    def is_tier2(self) -> bool:
        return self.action in TIER2_ACTIONS


class RadioEditPlan(BaseModel):
    """The full edit list for one episode + the editor's projected word
    total. ``post_validate_plan`` is the deterministic gate over an
    instance (passed to ``structured_call`` as ``post_validator``)."""

    edits: List[BeatEdit] = Field(default_factory=list)
    projected_word_total: int = Field(..., ge=0)
```

## Action set (the `action` value space; `lever:'S...'` is consistent with these) -- lines 146-168

```python
TIER1_ACTIONS: frozenset = frozenset(
    {
        "KEEP",
        "SHORTEN_LINE",
        "CLEAN_PUNCTUATION",
        # ... (Tier-1 render-safe edits)
    }
)

TIER2_ACTIONS: frozenset = frozenset(
    {"CUT_LINE", "REMOVE_REDUNDANT_BEAT", "SPLIT_LINE", "MERGE_SHORT_LINES"}
)

ALL_ACTIONS: Tuple[str, ...] = (
    "KEEP",
    "SHORTEN_LINE",
    "CLEAN_PUNCTUATION",
    # ... full union of Tier-1 + Tier-2
)
```

Guard1 (in `post_validate_plan` / `make_post_validator`) rejects an out-of-set
action LOUDLY: `Guard1: unknown action {edit.action!r} on beat_index ...;
allowed actions are {ALL_ACTIONS}`. So even a mis-aliased `lever` value that is
not a real action fails closed at the post_validator, never silently applies.

## The structured_call invocation (schema=RadioEditPlan; nested edits) -- lines 1421-1431

```python
    plan: RadioEditPlan = structured_call_fn(
        prompt=prompt,
        schema=RadioEditPlan,
        slot_fn=slot_fn,
        base_temperature=base_temperature,
        structural_retry_temperature=structural_retry_temperature,
        repair_prompt_factory=_default_repair_prompt,
        post_validator=post_validator,
        max_new_tokens=max_new_tokens,
        helper_name=f"radio_editor[{editor_model}]",
    )
```

NOTE: there are TWO entrypoints that both call `structured_call(schema=RadioEditPlan, ...)`:
`run_radio_editor` (helper_name `radio_editor[...]`) and `normalize_length`
(helper_name `normalize_length[...]`). The PROVEN failure was
`normalize_length[openrouter:slot-a]`. Both share the SAME RadioEditPlan ->
List[BeatEdit] schema, so the nested-alias annotation point (BeatEdit) is the
same for both. `repair_prompt_factory=_default_repair_prompt` is the existing
per-call-site factory (relevant to pass04 C4: the schema-aware repair is wired
at THIS call site, not in the core).
