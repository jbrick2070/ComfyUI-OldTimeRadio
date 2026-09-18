# Multilingual episodes

One dropdown changes the language of the whole episode. On
**OTR_LedgerScriptWriter**, set `episode_language` to one of:

- `Off`
- `English`
- `Spanish`
- `Portuguese`
- `Italian`
- `French`
- `Hindi`
- `Japanese`
- `Mandarin`

The folder is named `apple/` for historical reasons. This feature is not
Apple-specific.

---

## What the one switch changes

The writer resolves `episode_language` once, before loading a language model,
and records the language on the production ledger. Every later stage reads that
ledger stamp rather than reading the dropdown again.

The selected row controls:

- the instruction that asks the writer to author the story directly in the
  chosen language;
- the title instruction;
- the station greeting, sign-off and announcer framing;
- the display name for the reserved announcer in captions;
- caption wrapping and script-aware fonts;
- the audience-facing headings in the credits;
- the Kokoro language code and the voices eligible for casting;
- source-bank combinations that must refuse for fidelity.

`SIGNAL LOST` remains the station name in every language. Machine receipts,
engine ids and diagnostic text also remain English. Visual prompts stay English
in this first version because they are instructions to image and video models,
not audience dialogue.

The internal cast identity remains `ANNOUNCER`. Only its painted name changes
to `LOCUTOR`, `ANNUNCIATORE`, `播音员` and the other language-specific labels.
Changing the identity key would break cast, voice, caption and credit joins.

---

## Captions: native, but not translated

There is no second caption-language switch and no translation pass.

The caption burner displays the exact spoken line stored in the ledger:

- a native-language script line produces a native-language caption;
- an English line produces an English caption;
- if a writer mixes languages, the finished episode and its captions remain
  mixed.

This is why an English line performed by a Spanish Kokoro voice can sound like
"Spanglish" while the caption is still English. The voice changes pronunciation;
it does not translate text.

The selected language still controls the caption surface around that text:
the announcer label, line wrapping and font. Latin scripts use word wrapping,
Hindi wraps by Unicode grapheme so a Devanagari conjunct is not split, and
Japanese/Mandarin wrap by CJK character. Credits follow the same script-aware
font policy.

---

## Voices and Python versions

Kokoro is the day-one multilingual voice engine. For every non-English episode,
set both Cast Lock engine widgets and both voice-node engine widgets to
`kokoro`. A profile that changes characters to IndexTTS2 must be changed back
before the run. Cast Lock refuses Bark, IndexTTS2 and every other engine on a
non-English row instead of borrowing an English voice.

The admitted Kokoro pools are:

- English (`b` announcer, `a` and `b` character pool): 28 voices
- Spanish (`e`): `ef_dora`, `em_alex`, `em_santa`
- Portuguese (`p`): `pf_dora`, `pm_alex`, `pm_santa`
- Italian (`i`): `if_sara`, `im_nicola`
- French (`f`): `ff_siwis`
- Hindi (`h`): `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`
- Japanese (`j`): `jf_alpha`, `jf_gongitsune`, `jf_nezumi`,
  `jf_tebukuro`, `jm_kumo`
- Mandarin (`z`): `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`,
  `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

French and Italian are intentionally thin pools. The caster may reuse a voice
inside that language. It never fills a thin pool with an English voice.

Python 3.10 through 3.12 use Kokoro's torch backend and are the multilingual
path. Python 3.13 uses `kokoro-onnx`, which this pack treats as English-only.
Python 3.14 has no packaged Kokoro path. Japanese and Mandarin additionally
require the matching `misaki[ja]` or `misaki[zh]` readiness extra; those extras
are checked only when that language is selected and are not an English-install
tax.

Install one with ComfyUI's own Python before selecting that row:

```text
<ComfyUI Python> -m pip install "misaki[ja]"
<ComfyUI Python> -m pip install "misaki[zh]"
```

Use only the line for the language you need.

---

## Source banks and music

Every source bank works on every language row.

- `original`, `my_story`, `media_archive` and `scifi_news_pro` are AUTHORED in
  the language: the writer is told to write natively and never hands the model
  an English draft. A My Story prompt typed in English still yields a native
  episode; SciFi News Pro keeps its news source as published and writes the new
  story natively.
- `shakespeare` performs its passage TRANSLATED: the selected passage is
  translated once, in order, speakers and cut unchanged, and then performed
  verbatim exactly as the English lane performs Folger's text. The ledger's
  `verbatim_passage.translation` receipt carries both hashes.
- `public_domain` adapts natively through the same writer seams.
- The announcer's spoken credit line (the source acknowledgement, and the My
  Story attribution) is authored per row in `config/episode_languages.json`
  under `spoken`, so it is never a translation either.

Language does not choose the music. The source bank does. If eight language
tests all pin `original`, all eight correctly ask for salsa conjunto. To hear
different scores, vary the source bank: `media_archive` uses jazz,
`scifi_news_pro` uses Detroit techno, `original` uses salsa, and `my_story`
uses its own `music_style` or the house orchestra.

---

## `Off`, English and old workflows

`Off` is not a registry language. It preserves the old unstamped ledger path,
which downstream stages read as English.

`English` explicitly stamps `en` and a language-row receipt. A workflow saved
before the widget existed, or a caller that omits the value, also resolves to
English. An unknown non-empty value fails before generation; it never silently
falls back.

A replay keeps the frozen ledger's language. Selecting a different language
for that replay refuses. If the same language row changed since the episode was
frozen, the replay records row drift and uses the current row.

---

## Add your own language

The dropdown is registry-driven. A real admitted row appears without a writer
code change and without editing a workflow JSON. The work is in two data files,
the language-specific dependencies, tests and one live proof.

### 1. Confirm that Kokoro can actually speak it

This registry mirrors `kokoro` 0.9.4 `LANG_CODES`. A new row needs a real
Kokoro language code and real voice files. Adding a label for a language Kokoro
does not serve creates a menu entry that cannot speak; that is not support.

If a different TTS engine is required, add and qualify that engine first. The
day-one registry admits Kokoro only.

### 2. Add a row to `config/episode_languages.json`

Copy the closest existing row. Every row needs:

- unique `iso`, `label`, `sort_order` and Kokoro `lang_code`;
- `admitted`, `native_header` and `row_revision`;
- `authoring`: `spoken_name`, `writer_instruction`, `visual_prompt_iso`,
  `title_instruction`;
- `spoken`: every station/announcer string carried by the English row --
  the eight chrome strings plus the ten spoken credit sentences (the
  provenance coda templates and the two My Story attribution templates,
  with their `{work_title}` / `{author}` / `{name}` placeholders);
- `credits`: every heading and label carried by the English row;
- `captions`: `font_policy`, `wrap_policy`, `cps_policy`;
- `engines.kokoro`: `lang_code` and a non-empty `voices` list;
- `admission`: `source_bank_exclusions`, `readiness_extras`,
  `min_voice_count`.

Use `visual_prompt_iso: "en"` for the current feature. Leave
`source_bank_exclusions` empty: every bank is eligible on every row (the
verbatim lane translates its passage). The list exists only for a bank that
genuinely cannot carry a language yet.

Choose the caption policy by script:

- spaced Latin script: `latin_arial`, `word_split`, `latin_17`;
- Devanagari-like grapheme script: a script-capable font policy,
  `unicode_grapheme`, and a soft CPS policy;
- CJK: `cjk`, `cjk_chars`, `cjk_soft`.

If a new script needs another font family or wrapping rule, that is a code
change: add the policy to both captions and credits and test the pixels it
paints.

Increment `row_revision` whenever an admitted row's audience text, voice list,
font policy or admission contract changes. Replays use that revision and the
row hash as their drift receipt.

### 3. Add every Kokoro voice to `config/voice_reference_bank.json`

Copy a Kokoro voice entry and give it:

- the real `voice_ref_id`, `ref_path`, gender and descriptive fields;
- `engine: "kokoro"`;
- both `announcer_voice` and `char_voice` roles when the voice can serve both;
- `languages: ["<your iso>"]`;
- at least one usable announcer candidate for the row.

`languages` is the authority. An absent or empty list means English. The code
never guesses a language from `ef_`, `jf_` or any other voice-id prefix.

The startup prefetch reads the admitted registry rows, so a correctly listed
voice joins the startup fetch automatically. It must still exist in Kokoro's
real model repository.

### 4. Make admission fail before generation

Keep `min_voice_count` no larger than the row's real roster. The readiness
token format implemented today is `misaki[<adapter>]`; the gate imports
`misaki.<adapter>` completely so a missing transitive G2P package also refuses.
Do not put an arbitrary package name in `readiness_extras` without adding its
probe to `_otr_episode_languages.py`. The row must refuse if a required adapter
cannot import, if no eligible voice remains, if its Kokoro code duplicates
another row, or if any required spoken/credit/caption string is absent.

Do not add `Off` as a row. Do not infer language from a voice name. Do not let a
missing row, voice or dependency degrade to English.

### 5. Run the focused contract

At minimum, run:

- `tests/test_episode_languages.py`
- `tests/test_episode_language_writer.py`
- `tests/test_episode_language_voice.py`
- `tests/test_episode_language_painted_show.py`
- `tests/test_kokoro_covers_every_bank.py`
- `tests/test_kokoro_voice_prefetch.py`
- `tests/test_announcer_voice.py`

Then run the full project regression and the shared Bug Bible regression. A
unit test that selects the new row proves the row, not the full path.

### 6. Publish one canonical episode

The qualifying proof is a one-act run through
`workflows/otr_canonical.json`, with:

- the new language stamped on the final ledger;
- only voices declared for that language;
- native glyphs surviving captions and credits;
- `obs_publish OK`;
- a non-empty MP4 in `otr/obs/`.

Inspect the finished audio and captions. A successful render with English
writer leakage proves routing and media plumbing, but it does not prove fully
native authorship.

---

For general voice setup see [VOICES.md](VOICES.md). For source-bank behaviour
see [BANKS.md](BANKS.md), and for extension gates see
[EXTENDING.md](EXTENDING.md) and [PREFLIGHT.md](PREFLIGHT.md).
