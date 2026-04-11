"""
OldTimeRadio - Content Safety Filter
=====================================

Lightweight pre-generation safety check using a local keyword/regex denylist.
No extra model load required. Designed for v2.0 Visual Drama Engine.

v2.0: keyword/regex denylist (this file)
v2.1 stub: classify_prompt() interface ready for CLIP-based or NSFW classifier

Classification outcomes:
  - allow : prompt is clean, proceed with generation
  - flag  : prompt contains borderline content, log a warning but proceed
  - block : prompt contains prohibited content, skip generation entirely
"""

import logging
import re

log = logging.getLogger("OTR")

# ---------------------------------------------------------------------------
# DENYLIST - keyword and regex patterns
# ---------------------------------------------------------------------------
# These are checked case-insensitively against the full prompt text.
# Add patterns as raw strings. Regex is supported.

_BLOCK_PATTERNS = [
    # Violence and gore
    r"\bgore\b",
    r"\bgory\b",
    r"\bmutilat\w*\b",
    r"\bdismember\w*\b",
    r"\btorture\w*\b",
    r"\bbeheading\b",
    r"\bbloodbath\b",
    r"\bmasscre\b",
    r"\bmassacre\b",
    r"\bbrutal\s+kill",
    r"\bgraphic\s+violence\b",
    # Explicit / sexual content
    r"\bnude\b",
    r"\bnudity\b",
    r"\bpornograph\w*\b",
    r"\bexplicit\s+sexual\b",
    r"\bnsfw\b",
    r"\berotic\w*\b",
    r"\bsexual\s+content\b",
    r"\bxxx\b",
    # Hate speech and slurs (representative subset)
    r"\bhate\s+speech\b",
    r"\bracist\b",
    r"\bracial\s+slur",
    r"\bethnic\s+cleansing\b",
    # Self-harm
    r"\bsuicid\w*\b",
    r"\bself[- ]?harm\b",
    # Weapons of mass destruction
    r"\bbomb\s+making\b",
    r"\bbioweapon\b",
    r"\bchemical\s+weapon\b",
    # Child safety
    r"\bchild\s+abuse\b",
    r"\bminor\w*\s+explicit\b",
]

_FLAG_PATTERNS = [
    # Borderline content that should be logged but not blocked
    r"\bblood\b",
    r"\bweapon\b",
    r"\bgun\b",
    r"\bknife\b",
    r"\bdeath\b",
    r"\bkill\w*\b",
    r"\bviolence\b",
    r"\bfight\w*\b",
    r"\bdrug\w*\b",
    r"\balcohol\b",
]

# Pre-compile for performance
_COMPILED_BLOCK = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]
_COMPILED_FLAG = [re.compile(p, re.IGNORECASE) for p in _FLAG_PATTERNS]


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def classify_prompt(text):
    """Classify a prompt for content safety.

    Args:
        text: The prompt string to check.

    Returns:
        dict with keys:
            - result: "allow", "flag", or "block"
            - reason: human-readable explanation (empty string if allow)
            - matched: the pattern that triggered (empty string if allow)

    v2.0: keyword/regex denylist only.
    v2.1: drop-in replacement with CLIP-based or NSFW classifier.
          Same interface, same return format.
    """
    if not text or not text.strip():
        return {"result": "allow", "reason": "", "matched": ""}

    # Check block patterns first (higher priority)
    for pattern in _COMPILED_BLOCK:
        match = pattern.search(text)
        if match:
            reason = f"Blocked content detected: '{match.group()}'"
            log.warning("[SafetyFilter] BLOCK: %s in prompt: %s", reason, text[:80])
            return {
                "result": "block",
                "reason": reason,
                "matched": match.group(),
            }

    # Check flag patterns (lower priority, log-only)
    for pattern in _COMPILED_FLAG:
        match = pattern.search(text)
        if match:
            reason = f"Flagged content: '{match.group()}'"
            log.info("[SafetyFilter] FLAG: %s in prompt: %s", reason, text[:80])
            return {
                "result": "flag",
                "reason": reason,
                "matched": match.group(),
            }

    return {"result": "allow", "reason": "", "matched": ""}


def check_prompt(text):
    """Convenience wrapper: returns True if prompt is allowed (allow or flag).

    Returns False only for blocked prompts.
    """
    return classify_prompt(text)["result"] != "block"
