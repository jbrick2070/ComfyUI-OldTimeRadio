"""Pure helper module for dynamic dialogue prompt policies."""

from typing import Any, Dict, Iterable, Union


_COCKNEY_ORTHOGRAPHY_RULE = (
    "\n\nConvey the Cockney accent through phrasing, idiom, cadence, and rhythm. "
    "Use standard English spelling in every spoken line; do not encode pronunciation "
    "with phonetic misspellings."
)


def roster_has_lemmy(roster: Iterable[Union[str, Dict[str, Any]]]) -> bool:
    """Return True if LEMMY is present in roster names or cast dicts."""
    if not roster:
        return False
    for item in roster:
        if isinstance(item, str):
            if item.strip().upper() == "LEMMY":
                return True
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip().upper()
            char_id = str(item.get("char_id") or "").strip().upper()
            if name == "LEMMY" or char_id == "LEMMY":
                return True
    return False


def append_dialogue_policy(
    system_prompt: str, roster: Iterable[Union[str, Dict[str, Any]]]
) -> str:
    """Append roster-specific policy rules to system prompt after router resolution."""
    system = system_prompt or ""
    if roster_has_lemmy(roster):
        system = system + _COCKNEY_ORTHOGRAPHY_RULE
    return system
