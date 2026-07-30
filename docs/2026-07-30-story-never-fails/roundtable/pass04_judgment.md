# Round 4 Judgment - Final Convergence

## Decision

CONVERGED. BUILD.

OpenAI and Gemini returned yes-with-fixes. Their remaining must-fix items ask
for exact constants and algorithms that Revision 3 summarized too compactly;
the grounded implementation defines them below. DeepSeek again spent its
output allowance without returning usable content and contributes no finding.

## Exact resolutions

- RSS normalized `full_text`: at most 2,097,152 characters.
- RSS normalized seven-field A0 serialization: at most 8,519,680 UTF-8 bytes
  (`4 * 2 MiB + 128 KiB`).
- Operator-pinned A0: unchanged 48,000 UTF-8 byte cap.
- Source transport: unchanged HTTPS-only, redirect/address/media-type checked,
  2 MiB decoded-response seam.
- RSS content index: zero-based raw list index; non-list collection is absent.
- Selected body: longest already-clean text by Python character count; ties
  RSS, URL article, summary.
- Body hash: lowercase SHA-256 of exact selected body UTF-8 bytes.
- 800-char preview: two `" ... "` separators; split the remaining 790 chars as
  head 263, centered middle 263, tail 264.
- Maximum quote: `MAX_QUOTE_CHARS == 240`; overlap is 239.
- Window offset: exact complete-A0 coordinate of the first window character.
  Next start is `end - 239`; a sentence rewind that cannot advance uses the
  hard end.
- Every local P0 artifact carries the complete normalized A0 digest. It
  validates locally without coordinate relocation, rebases both bounds of only
  `full_text` spans, then validates against complete A0 without relocation.
- No global validator state is installed, so the panel's proposed `finally`
  cleanup has no corresponding runtime object and is rejected as inapplicable.
- Merge identity, window/local-ID namespace, stable traversal, even-sampling
  formula, caps, contiguous IDs, parent remap, and final full-A0 validation are
  explicit in the final plan and executable tests.
- Retry rejection is whitespace-collapsed, ASCII-bounded to 600 characters.
  UUID is 32 hex characters and instruction text is fixed. P0 reserves 1,024
  tokens for the complete mapping.
- "First prompt identity" means cycle one for a given invocation contains no
  `writer_retry`; it does not claim the new windowed P0 prompt is byte-equal to
  the obsolete pre-window production prompt.
- No retry layer catches `BaseException`.

## Final boundary

Canonicalization, final graph/safety validation, cleanup defense, assembly,
delivery/authorship stamping, freeze, save, reopen, and proof remain outside
every candidate retry. There is no new must-fix architecture issue.
