# Google TTS BYO API Plan

Status: planning handoff, not implementation.

Goal: add `google_tts` as a direct Google/Gemini API voice engine for OTR users
who want a BYO Google API-key path. This is separate from Comfy Cloud / Comfy
Credits and must not route through Partner nodes.

## Product Decision

OTR should keep the existing Comfy Cloud voice path as ElevenLabs through Comfy
Credits:

- Current engine id: `elevenlabs`
- Current voice bank: `elevenlabs_cloud`
- Current Partner row: `cloud_elevenlabs_tts`

`google_tts` is a new direct-Google lane:

- Direct Gemini API / Google API key.
- No Comfy Credits.
- No Partner-node bridge.
- No local fallback to Bark, Kokoro, IndexTTS, or anything else.
- No fallback to ElevenLabs, Comfy Cloud, Partner nodes, or any non-Google
  provider.
- A bounded retry to another allowlisted Google/Gemini TTS model is allowed
  inside the `google_tts` adapter only, if explicitly configured and logged as a
  Google TTS retry.
- No automatic fallback into `google_tts` from a local engine failure.
- If `google_tts` is explicitly selected and anything is wrong, raise before or
  at invoke with a clear error.
- Experimental until voice stability is proven across a cast.

Highly theoretical future roadmap only:

- `elevenlabs_byo` could later be a direct ElevenLabs API-key adapter.
- Do not build that as part of the Google TTS sprint.

## Official Google Surface Checked 2026-07-08

Primary sources:

- Gemini API TTS docs: https://ai.google.dev/gemini-api/docs/speech-generation
- Gemini Developer API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Google Cloud Text-to-Speech pricing: https://cloud.google.com/text-to-speech/pricing

Grounded facts:

- Gemini API TTS is Preview.
- The current docs recommend the Interactions API for latest features.
- TTS accepts text-only input and outputs audio-only.
- Single-speaker and multispeaker TTS are supported.
- Multispeaker is limited to up to 2 configured speakers in the documented
  request shape.
- Official TTS request shape:
  - Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/interactions`
  - Header: `x-goog-api-key: $GEMINI_API_KEY`
  - Body includes:
    - `model`
    - `input`
    - `response_format: {"type": "audio"}`
    - `generation_config.speech_config`
- Official Python example uses `client.interactions.create(...)`, not
  `client.models.generate_content(...)`.
- Official output convenience property is `interaction.output_audio.data`,
  base64-encoded PCM.
- Official example writes 24 kHz, 16-bit, mono WAV.
- Current documented voice options include 30 prebuilt names such as `Kore`,
  `Puck`, `Aoede`, `Algenib`, `Charon`, `Fenrir`, and `Enceladus`.
- Supported models listed in the TTS docs:
  - `gemini-3.1-flash-tts-preview`
  - `gemini-2.5-flash-preview-tts`
  - `gemini-2.5-pro-preview-tts`

Allowed same-provider retry order, if enabled:

1. Configured/default model.
2. `gemini-2.5-flash-preview-tts`.
3. `gemini-3.1-flash-tts-preview`.

Do not retry Pro by default because it is higher cost. Do not retry outside
Google/Gemini TTS.

Pricing distinction:

- Gemini Developer API pricing lists `gemini-2.5-flash-preview-tts` as free of
  charge on the Free Tier, with paid Standard pricing at $0.50 / 1M text input
  tokens and $10.00 / 1M audio output tokens.
- Gemini Developer API pricing lists `gemini-2.5-pro-preview-tts` paid Standard
  at $1.00 / 1M text input tokens and $20.00 / 1M audio output tokens.
- Google Cloud Text-to-Speech pricing separately lists Gemini-TTS with no free
  usage limit, including Gemini 3.1 Flash TTS Preview at $1.00 / 1M text input
  tokens and $20.00 / 1M audio output tokens.
- Google Cloud says audio tokens correspond to 25 tokens per second of audio.

Approximate episode math:

- 20 minutes = 1200 seconds.
- 1200 seconds * 25 audio tokens/sec = 30,000 audio tokens.
- At $20 / 1M audio tokens, output audio is about $0.60, plus small text input
  cost.
- At $10 / 1M audio tokens, output audio is about $0.30, plus small text input
  cost.

Do not advertise "free" as guaranteed production cost. Say:

> Gemini Developer API may have a free tier for some preview TTS models, but
> Google Cloud TTS pricing lists no free usage limit for Gemini-TTS. OTR should
> support BYO key and show cost assumptions by selected surface/model.

## First Implementation Scope

Build only the per-line single-speaker adapter first.

Engine ids and profile names:

- Engine: `google_tts`
- Voice bank: `google_tts`
- Character profile: `char_google_tts_v1`
- Announcer profile: `announcer_google_tts_v1`

Default model:

- Use `gemini-2.5-flash-preview-tts` for the low-cost first profile.
- Allow opt-in env override to `gemini-3.1-flash-tts-preview`.
- Do not default to Pro.

Initial voices:

- `Algenib` male/gravelly, authority or captain.
- `Puck` male/upbeat, younger or energetic.
- `Kore` female/firm, scientist or grounded narrator.
- `Aoede` female/breezy, narrator or exposition.
- Add more only after the first four prove stable.

Voice-quality gate:

- This engine is not useful unless it can produce a strong announcer voice and
  gender-plausible character casting.
- Prefer a British-sounding announcer style when the selected Google voice/prompt
  can support it.
- Announcer gender should be mixed across episodes: choose male-coded vs
  female-coded announcer candidates roughly 50/50 when both are available,
  deterministically from the episode seed.
- Do not let character casting reuse the announcer voice unless explicitly
  forced by the operator. The default should preserve announcer separation.
- Character voice assignment must respect cast gender where available, then
  timbre/age hints, and should fail loud if the bank cannot satisfy the selected
  voice-bank policy without reuse.
- If the first Google voice bank cannot deliver a credible announcer plus
  distinct gender-aware character voices, keep `google_tts` experimental and do
  not include it in any "google all" preset.

Do not implement multispeaker batching in the first pass. OTR already has a
per-line voice pipeline and cast locking; preserve that path first.

## Required Code Touchpoints

No code in this thread. The build window should update these files:

- `nodes/_otr_audio_engines/eng_google_tts.py`
- `nodes/_otr_audio_engines/__init__.py`
- `nodes/_otr_audio_engines/registry.py`
- `nodes/_otr_engine_profiles.py`
- `nodes/cast_lock.py`
- `config/audio_engine_profiles.yaml`
- `config/voice_reference_bank.json`
- focused tests under `tests/`

Import placement:

- Add `eng_google_tts` in the cloud/direct-API audio engine group in
  `nodes/_otr_audio_engines/__init__.py`.
- Label it as direct Google API, not Comfy Partner cloud.

Workflow JSON:

- If adding `google_tts` to an existing dropdown changes node widget choices or
  defaults, update `workflows/otr_canonical.json` in the same code change.
- Widgets are positional; only append optional widgets. Do not insert mid-vector.

## Adapter Contract

`GoogleTTSVoice` should be a normal no-argument registered audio adapter.

Required metadata:

- `name = "google_tts"`
- `roles = ("char_voice", "announcer_voice")`
- `default_roles = ()`
- `commercial_clean = True`
- `interface = "per_line"`
- `sample_rate = 24000`
- `native = False`
- `requires_voice_ref = False`
- `voice_ref_field = "provider_voice_id"`
- `voice_ref_kind = "provider_voice_id"`
- `missing_ref_fallback = None`

Fail-loud guards:

- Blank text raises.
- Missing provider voice id raises.
- Missing Google API key raises before request.
- Unsupported model id raises before request.
- Unsupported voice name raises before request.
- Malformed or missing audio output raises with a clear Google TTS message.
- No local fallback is allowed.
- No alternate cloud/provider fallback is allowed.
- No silent model fallback is allowed. If the selected Google model is
  unsupported, fail before request. If a supported Google model is temporarily
  unavailable, the adapter may retry another allowlisted Google/Gemini TTS model
  only when configured to do so; otherwise fail loud.

Auth resolution:

- Accept `OTR_GOOGLE_API_KEY`, `GEMINI_API_KEY`, then `GOOGLE_API_KEY`.
- Pass the resolved key to the Google client or REST request.
- Redact all key values from exceptions and logs.
- Wrap the network call and response parsing in an explicit sanitizer. If
  `urllib.error.HTTPError`, `URLError`, JSON parse, or malformed-response errors
  are raised, re-raise an OTR/Google TTS error whose message replaces the actual
  API key with `<REDACTED>`.

Preferred transport:

- First implementation may use direct REST with stdlib `urllib` to avoid a new
  hard startup dependency.
- If using `google-genai`, import it lazily inside adapter load/invoke paths.
- Do not import `google`, `numpy`, or `torch` at module scope.

REST request shape:

This is the current official REST shape from the Interactions TTS docs. Do not
replace it with older protobuf-style `voice_config.prebuilt_voice_config`
snippets unless a live probe proves the official REST example has changed.

```json
{
  "model": "gemini-2.5-flash-preview-tts",
  "input": "Say with quiet dread: The signal is still coming from inside the wall.",
  "response_format": {
    "type": "audio"
  },
  "generation_config": {
    "speech_config": [
      {
        "voice": "Kore"
      }
    ]
  }
}
```

REST response handling:

- Prefer `output_audio.data` if present.
- Also tolerate `outputAudio.data` in raw JSON responses, because REST casing can
  drift across generated surfaces.
- Base64-decode before interpreting PCM.
- Treat the decoded bytes as signed 16-bit little-endian PCM unless a live probe
  proves the response carries an explicit format marker that says otherwise.
- Convert to float32 range [-1, 1].
- Return OTR audio dict:

```python
{"waveform": tensor_shape_1_C_T, "sample_rate": 24000}
```

Implementation notes:

- Copy the NumPy array before `torch.from_numpy(...)` so PyTorch does not wrap a
  read-only buffer.
- Shape must be `[1, 1, T]`.
- Let the existing downstream path resample to 48 kHz where needed.

## Delivery Vector Mapping

OTR's delivery vector keys are defined in `nodes/_otr_delivery_vector.py`:

- `happy`
- `angry`
- `sad`
- `afraid`
- `disgusted`
- `melancholic`
- `surprised`
- `calm`

Do not look for non-existent keys such as `emotion`, `pacing`, or `intensity`.

First-pass mapping:

```python
{
    "angry": "with controlled fury, clipped and tense",
    "afraid": "with audible fear, slightly breathless",
    "happy": "warmly and brightly",
    "sad": "quietly, with restrained grief",
    "disgusted": "with barely concealed contempt",
    "melancholic": "wistful and reflective",
    "surprised": "with a startled intake, then urgency",
    "calm": ""
}
```

Use the dominant non-calm emotion above a threshold, then build a natural
language prefix. Example:

First threshold:

```python
_EMOTION_PREFIX_THRESHOLD = 0.15
```

```text
Say with audible fear, slightly breathless: I can still hear it tapping.
```

Audio tags:

- Gemini docs say tags such as `[whispers]`, `[laughs]`, `[cough]`, `[sighs]`,
  and `[gasp]` can influence delivery.
- Do not blindly replace every parenthesis with brackets.
- Add a small allowlist translator for known stage directions:
  - `(sighs)` -> `[sighs]`
  - `(whispers)` -> `[whispers]`
  - `(gasps)` -> `[gasp]`
  - `(laughs)` -> `[laughs]`
- Leave unknown parentheticals as normal text or strip them according to the
  existing voice-text cleanup path.
- Implement this in `GoogleTTSVoice.prepare_text(...)`, not after the default
  neutral prep. The shared `nodes/_otr_script_prep.py` cleanup strips both
  parentheticals and bracket tags, so the adapter must preserve allowlisted
  directions before calling the neutral cleaner, then restore them as Gemini
  bracket tags.

## Profile Contract

Current `EngineProfile.runtime` permits:

- `in_graph`
- `oop_venv`
- `cloud`

Current validation requires `runtime: cloud` profiles to have a Partner row.
`google_tts` is direct Google API, not Partner. The root fix is to add a new
runtime in the same commit as the adapter:

```yaml
runtime: direct_api
partner_row: ""
provider_id: google
auth_required: true
billing_category: tts
canonicalizer: audio
error_policy: fail_loud
```

Do not overload `cloud` for direct Google APIs and do not label direct Google
REST as `oop_venv`. Update `nodes/_otr_engine_profiles.py` in the same commit:

- Add `"direct_api"` to `_VALID_RUNTIMES`.
- Require `runtime: direct_api` profiles to use `error_policy: fail_loud`.
- Require `runtime: direct_api` profiles to have `auth_required: true`.
- Require `runtime: direct_api` profiles to keep `partner_row: ""`.
- Keep the existing `runtime: cloud` Partner-row validation unchanged for Comfy
  Partner nodes.

Profile skeleton:

```yaml
- profile_id: char_google_tts_v1
  role: char_voice
  engine: google_tts
  commercial_clean: true
  model_path: ""
  model_sha256: ""
  default_params: {model: gemini-2.5-flash-preview-tts}
  allowed_voice_banks: [google_tts]
  engine_impl_version: "1"
  sample_rate: 24000
  requires_hf_token: false
  rank: 50
  is_default: false
  runtime: direct_api
  needs_ref_clip: false
  caps: {emotion_vector: true, duration_control: false}
  license_state: clean
  warn_text: "Google Gemini TTS is preview; voice stability may vary."
  partner_row: ""
  provider_id: google
  auth_required: true
  billing_category: tts
  canonicalizer: audio
  error_policy: fail_loud
```

Mirror this for `announcer_google_tts_v1`.

Also update `_LEGACY_FIRST_ENGINES` in `nodes/_otr_engine_profiles.py` so Comfy
dropdowns can include `google_tts` without doing YAML IO at class-definition
time:

- Append `"google_tts"` to `char_voice`.
- Append `"google_tts"` to `announcer_voice`.

Also update `EngineProfileResolver.rank_chain(...)`:

- The automatic rank/fallback chain currently excludes only `runtime == "cloud"`.
- Exclude `runtime in ("cloud", "direct_api")`.
- Direct Google API profiles must be reachable only by explicit selection, never
  by an automatic fallback/default chain. This preserves the no-fallback,
  no-silent-spend, and no-wrong-shaped-fallback contract.

## Voice Bank Contract

Add `google_tts` to `nodes/cast_lock.py` voice bank choices:

- Append `"google_tts"` to `_VOICE_BANKS`.
- Append `"google_tts"` to `_CHAR_VOICE_ENGINES`.
- Append `"google_tts"` to `_ANNOUNCER_VOICE_ENGINES`.

Append only. These are widget choices feeding saved workflows.

Add `config/voice_reference_bank.json` entries shaped like existing cloud
ElevenLabs rows, but with Google voice names:

```json
{
  "voice_ref_id": "gt_kore",
  "engine": "google_tts",
  "gender": "female",
  "timbre": ["firm", "clear"],
  "roles": ["char_voice", "announcer_voice"],
  "age_band": "adult",
  "ref_path": "cloud:google_tts:Kore",
  "ref_sha256": "cloud",
  "commercial_clean": true,
  "provider_voice_id": "Kore"
}
```

Minimum first bank:

- `gt_algenib`
- `gt_puck`
- `gt_kore`
- `gt_aoede`

Announcer/cast assignment rules:

- Mark at least one male-coded and one female-coded preferred announcer
  candidate in the Google bank, with a British delivery prompt hint where
  appropriate.
- Keep the selected announcer candidate out of the default character assignment
  pool unless `allow_voice_reuse` or an explicit operator override says
  otherwise.
- The first pass should include at least two male-coded and two female-coded
  character-capable voices so CastLock can make basic gender-aware assignments.
- The announcer should be selectable independently from character voices.

## Regression Tests

Add focused tests before any live smoke:

- `google_tts` registers and appears in char/announcer engine choices.
- `google_tts` appears in `_LEGACY_FIRST_ENGINES` for char and announcer voice.
- `google_tts` direct-api profiles do not appear in
  `EngineProfileResolver.rank_chain(...)` for char or announcer voice.
- `google_tts` appears in CastLock voice bank and voice-engine choices.
- `google_tts` CastLock assignment keeps announcer and character voices distinct
  by default.
- `google_tts` voice-bank tests cover gender-aware character assignment and
  deterministic 50/50 male/female announcer selection across different episode
  seeds when both announcer genders are available.
- `google_tts` has a `CAPABILITIES` row with `cpu_ok: True`,
  `requires_sidecar: False`, and no local model requirements.
- `runtime: direct_api` profile validation accepts `partner_row: ""`,
  `auth_required: true`, and `error_policy: fail_loud`.
- `runtime: cloud` still rejects a blank Partner row.
- Importing the module does not import `google`, `numpy`, or `torch` at module
  scope.
- Missing API key fails before request.
- API key is redacted from raised errors and logs.
- Unsupported model fails before request.
- Unsupported voice fails before request.
- Blank text fails before request.
- Missing `provider_voice_id` fails before request.
- Adapter sends the Interactions request shape with:
  - `response_format.type == "audio"`
  - `generation_config.speech_config[0].voice == provider_voice_id`
- Adapter tests assert it does not emit protobuf-style
  `voice_config.prebuilt_voice_config` for the Interactions REST path.
- Base64 PCM output decodes to `[1, 1, T]` PyTorch tensor at 24000 Hz.
- Response parsing accepts both `output_audio.data` and `outputAudio.data`.
- Read-only NumPy buffer bug is covered by asserting the tensor is writable or
  by exercising a downstream operation that would fail on read-only storage.
- The adapter calls `.copy()` on the `np.frombuffer(...)` result before
  `torch.from_numpy(...)`.
- Delivery vector uses the real 8 emotion keys.
- Delivery-vector prefixing uses a named threshold constant; first value:
  `0.15`.
- Stage-direction tag allowlist maps known tags only.
- `google_tts` has no fallback call path to Bark, Kokoro, IndexTTS, or
  ElevenLabs.
- `google_tts` has no fallback call path to Comfy Cloud, Partner nodes, or any
  non-Google provider.
- Same-provider retry tests cover the allowlisted Google/Gemini TTS retry path:
  disabled by default unless configured, never includes Pro by default, logs the
  model switch, and fails loud when all configured Google models fail.
- Explicit `google_tts` failures raise; tests must not accept synthesized
  substitute audio from any other engine.
- Existing hardcoded profile/dropdown tests are updated intentionally:
  `tests/test_engine_profiles.py` profile ids and
  `tests/test_announcer_voice.py` / character voice dropdown expectations must
  include the appended Google entries without changing index 0 defaults.
- Existing `elevenlabs_cloud` profile and tests remain unchanged.
- `audio_engine_profiles.yaml` loads cleanly.
- `voice_reference_bank.json` schema checks pass.

## Live Probe Gate

Do not run a full episode first.

First live probe should:

1. Use a one-line prompt.
2. Use `Kore`.
3. Write the WAV directly to `otr/episodes/google_tts_probe/`.
4. Confirm file exists.
5. Confirm sample rate, channel count, duration, and non-silence.
6. Confirm no local voice model loaded.

Second live probe should:

1. Render the same line three times with the same voice.
2. Check that voice identity is subjectively stable enough for preview use.
3. If unstable, keep `google_tts` marked experimental and do not include it in
   any `google_all` profile.

Third live probe should:

1. Render one male-coded and one female-coded announcer line with the
   British-leaning announcer setup.
2. Render at least one male-coded and one female-coded character line.
3. Confirm the announcers do not sound like the character voices.
4. Confirm the character voices fit the cast gender/timbre well enough for
   preview use.

## Build-Window Instruction

When coding starts in the separate window, start from this doc and the official
Google links above. Do not use stale SDK snippets from older Gemini TTS docs.
The first target is a mocked, test-green `google_tts` adapter. Live API probing
comes only after the adapter contract and profile/wiring tests pass.
