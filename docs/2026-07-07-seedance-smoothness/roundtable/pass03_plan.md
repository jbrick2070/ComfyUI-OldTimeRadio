# Pass 03 Plan - Final Candidate For Convergence

## Build

Implement a Seedance-only prompt conditioner in
`nodes/_otr_video_engines/eng_cloud_video.py`.

Do not edit `workflows/otr_scifi_16gb_full.json` and do not edit visual style
JSON in this pass.

## Helper Contract

```python
def _condition_seedance_prompt(prompt: str) -> tuple[str, dict]:
    ...
```

Rules:

1. The caller must pass a non-empty prompt from `_text_prompt_input(request)`.
2. First check for this stable marker:

   ```text
   Gentle parallax only; all motion gradual and physically continuous.
   ```

3. If the marker exists, return the prompt unchanged with `changed=False`.
4. Otherwise apply softeners to the source prompt only.
5. Append the smooth-motion clause.

Metadata schema:

- `changed: bool`
- `original_sha8: str`
- `conditioned_sha8: str`
- `original_excerpt: str`
- `conditioned_excerpt: str`
- `softeners_applied: list[str]`

Hash:

```python
hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
```

Excerpt:

```python
re.sub(r"\s+", " ", text).strip()[:160]
```

## Softener Order

Use compiled case-insensitive regexes in this exact order:

1. `dynamic dolly push` -> `slow controlled dolly push`
2. `handheld dolly` -> `stabilized dolly`
3. `whip[- ]pans?` -> `slowly sweeps`
4. `white[- ]hot` -> `bright warm glow`
5. `rapid zooms?` -> `slow controlled push`
6. `aggressively` -> `subtly`
7. standalone `handheld` -> `stabilized`

Normalize replacements to the exact lowercase replacement phrases.

## Smooth-Motion Clause

```text
One continuous uncut shot. Smooth stabilized camera on a slow dolly with gentle ease-in and ease-out. Motion begins immediately in the first frame and remains gentle and continuous throughout. Preserve the reference-image composition and framing. No whip pans, handheld shake, sudden reframing, jump cuts, or rapid zooms. Gentle parallax only; all motion gradual and physically continuous.
```

The clause intentionally does not say `16:9`; the adapter may send
`ratio=adaptive`.

## Adapter Integration

In `CloudSeedance2Engine._partner_inputs()`:

1. `prompt = self._text_prompt_input(request)`
2. `conditioned_prompt, meta = _condition_seedance_prompt(prompt)`
3. Preserve the current model/resolution/ratio/duration/reference image/audio
   behavior.
4. Use `conditioned_prompt` for `model["prompt"]`.
5. Keep the top-level request shape unchanged:
   `{"model": ..., "seed": ..., "watermark": False}`.
6. Log one structured, sanitized line with the metadata and requested duration.

Do not mutate the request object.

## Duration Policy

Keep the current provider-minimum behavior:

- Seedance valid duration is `4..15s`.
- If the beat is shorter than 4s, request 4s.
- `OTR_SilentComposite` trims the kept head frames to the audio-derived beat.
- This is correct for cloud APIs that cannot render below their minimum.
- The immediate-motion clause makes the head-trim useful.

## Tests

Add/adjust tests in `tests/test_cloud_video_adapters.py`.

Required tests:

- grounded `music_open` text is softened;
- second and third helper calls are byte-identical;
- metadata hashes/excerpts/softener names are stable;
- Seedance partner input shape is unchanged and uses the conditioned prompt;
- short beat below 4s requests `duration == 4`;
- Wan/Kling/Pixverse prompts are not affected.

## Manual QA

After tests are green, run one small Seedance A/B on a representative opener.

Use 2-3 samples per variant because Seedance seed is documented as
non-deterministic. If `Seedance 2.0 Fast` still stutters, compare
`OTR_CLOUD_SEEDANCE_MODEL="Seedance 2.0"` for the same still/audio/prompt.
