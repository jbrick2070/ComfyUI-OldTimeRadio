# Research prompt: find (or plan) a "Kokoro killer" text-to-speech model

Paste everything below the line into Gemini (Deep Research mode if available). Written
2026-10-21 for the Old-Time Radio (OTR) ComfyUI node pack.

---

You are a research analyst. I need a sourced, current survey of open text-to-speech (TTS) models,
and a judgement on whether one already beats my current engine -- or what it would take to build
one. Search the web thoroughly (model cards on Hugging Face, GitHub repos, papers on arXiv, release
notes, benchmark leaderboards, and practitioner reports). Prefer releases from the last 12 months.
Cite a link for every factual claim, and mark anything you could not verify as "unverified".
Do not guess numbers.

## What I make

Old-Time Radio is a ComfyUI custom node pack that writes, casts, voices, scores and films whole
radio plays on the user's own machine. Every character needs a distinct, consistent voice, in the
episode's language, across a 5-15 minute play. It must run on three kinds of machine: Windows with
an NVIDIA GPU (8-16 GB), Apple Silicon Macs (M-series, 16 GB unified memory), and CPU-only boxes.

## What I use now (the baseline to beat)

**Kokoro-82M** (Apache-2.0), run through ONNX (`kokoro-onnx`):
- Built-in voices, no reference audio needed, about 50 voice packs.
- About 82M parameters, StyleTTS 2-style.
- Real-time factor about 0.12 on CPU, measured on my box. It runs anywhere.
- Eight languages in practice: English (US and UK), Spanish, French, Hindi, Italian, Japanese,
  Portuguese and Mandarin.
- Weaknesses:
  - A limited range of voices and accents.
  - Uneven quality outside English.
  - Little control of emotion or delivery.
  - Few distinct older or younger voices for character casting.

Also tried: **Bark** (wide expressive range, but slow, erratic and drifting), and reference-clip
cloners (IndexTTS2, Chatterbox, and similar), which I want to avoid as the default because they need
a reference audio file per voice.

## The "killer" I am looking for -- hard requirements

1. **Built-in voices.** It ships many named voices or speaker embeddings. No reference audio file
   is needed to get a voice. Prompted voice design from a text description ("a gravelly old sea
   captain") counts as built-in if it is deterministic per seed.
2. **Open weights, auto-downloaded, never gated -- and still multilingual.**
   - **Self-fetching:** my pack downloads every model on first use, with no user step. So the
     weights for ALL its languages and voices must be fetchable by a script from a public repository
     with no login, no token, no "accept the terms" click, and no gated Hugging Face repo.
   - **Gated means disqualified:** a model whose multilingual weights sit behind a gate, a
     sign-up or a manual download fails this rule, even if its English model is open.
   - **What to report:** say exactly where each language's weights live, their total download size,
     and whether a pinned, hash-checked fetch is possible.
   - **Licence:** it must allow redistribution in an open-source node pack. Say the exact licence,
     and whether commercial use is allowed.
3. **Fast on all three targets.** Report real-time factor or seconds per second of audio on:
   - an NVIDIA consumer GPU;
   - an Apple Silicon Mac (MPS, MLX, Core ML or ANE);
   - a CPU.
   Say whether ONNX, MLX or GGUF builds exist, and who maintains them.
4. **Multilingual, with accents.** Which languages, how good each one is (native-speaker or MOS
   evidence), and whether one voice can speak several languages or accents.
5. **Small enough:** under about 1B parameters preferred, and say the VRAM or RAM it needs.
6. **Stable over long text:** no drift, mumbling or hallucinated words over a page of dialogue.

Nice to have:
- emotion or style control;
- SSML or prosody control;
- age and gender range for casting;
- streaming;
- word timestamps (for captions);
- an existing ComfyUI node.

## Deliverables

1. **A comparison table** of every serious candidate, including Kokoro as the baseline row:
   - name, release date, parameters, licence;
   - built-in voices (count; yes or no);
   - languages, and accent support;
   - speed on GPU, Mac and CPU (with sources);
   - ONNX, MLX or GGUF availability;
   - quality evidence (benchmarks, MOS, arena ranks);
   - long-text stability;
   - ComfyUI support;
   - known problems.
2. **The top three picks for my case**, with the reason for each, and what each would lose against
   Kokoro.
3. **A verdict:** does a Kokoro killer exist today under these rules? Yes, no, or "close", and
   exactly what is missing.
4. **How we would train our own -- required, whatever the verdict.** Write a full plan.

   **What we have:**
   - an RTX 5080 Laptop (16 GB) and an RTX 4060 Laptop (8 GB) on Windows;
   - Apple Silicon Macs for testing;
   - a RunPod account for renting A100, H100 or L40S class GPUs by the hour;
   - one developer plus AI coding assistants;
   - no budget for licensed voice actors yet.

   **What the plan must cover:**
   - **a. Starting point.** Compare three routes:
     - **Train from scratch:** a small StyleTTS 2-style model, the way Kokoro was built.
     - **Fine-tune or extend an open model:** say whether Kokoro's own training code and recipe are
       public; if not, say which open base is the best substitute.
     - **Distil from a larger open model:** for each candidate, say whether its licence allows using
       its outputs as training data.

     For each route give the expected quality, the cost, the risk, and the time to a first usable
     voice.
   - **b. Data -- the hard part.**
     - Which openly licensed, multi-speaker, multilingual speech corpora permit training a model
       AND shipping its built-in voices in an open-source pack.
     - For each: hours, speakers, languages and accents, recording quality, transcript quality,
       and the exact licence.
     - Flag any that forbid commercial use, voice redistribution or derivative voices.
     - How many hours per language and per voice a small model needs.
     - Whether synthetic data is a legitimate option, and under what licence terms.
     - How to build our own voices legally: volunteers, consent forms, how much audio per speaker,
       and recording guidelines.
   - **c. Pipeline.**
     - Cleaning, segmentation, forced alignment and phonemisation, naming the tools (for example the
       grapheme-to-phoneme frontends used per language).
     - Speaker and accent labels.
     - Train, validation and test splits.
   - **d. Compute and cost.**
     - GPU-hours and dollars on rented cards for a first model and for a polished one, citing what
       Kokoro and comparable small TTS models are reported to have cost.
     - What can run on our own 16 GB card versus what needs rented A100s or H100s.
     - Wall-clock timeline.
   - **e. Speed targets and export.**
     - How to keep it fast: size, vocoder choice, quantisation.
     - Export to ONNX (CPU and NVIDIA), MLX or Core ML (Mac), and GGUF if relevant.
     - The real-time factor to aim for on each target.
   - **f. Evaluation.**
     - Blind A/B listening tests against Kokoro per language and accent.
     - Automatic metrics: word error rate via an ASR model, speaker similarity, and an MOS
       predictor.
     - Long-text stability tests.
     - The bar it must clear to replace Kokoro as the default.
   - **g. Milestones.** A staged roadmap:
     - one language with a few voices;
     - then eight languages;
     - then accents and emotion.

     Include a go/no-go check at each stage.
   - **h. Legal and ethical risks.**
     - Voice likeness and consent.
     - Dataset licence traps.
     - Model licence compatibility with an MIT node pack.
     - What to put in the model card.
5. **Sources:** every link, grouped by candidate and by plan section.

Be blunt. If a model's speed claims come only from its own authors, say so. If a "multilingual"
claim means accented English, say so.
