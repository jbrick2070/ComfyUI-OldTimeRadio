# Final pick -- Windows HF_HOME, lowest-friction fix

**F, hardened.** Do not keep the 47703d7a decline. Do not ask the user
to enable Long Paths. Do not patch huggingface_hub. Do not rename Lumina.

A ComfyUI Desktop user whose models-adjacent `HF_HOME` cannot fit under
MAX_PATH must still press Run and get an episode. The cache they write
must be one cache, short enough to materialise, chosen before
`huggingface_hub` is imported.

Decided 2026-09-23. r1 panel: Codex gpt-6-astra, Antigravity Gemini 3.8
Flash (High), Claude sonnet. Cursor driver; Cursor CLI excluded.

Re-grounded 2026-09-25 at `d167f0ab`: `prestartup_script.py` still
carries the decline-to-pin branch (the `if "HF_HOME" not in environ:`
block that logs "NOT pinning HF_HOME"), so this row is still open.

## Decision

When `HF_HOME` is not already in the process environment, prestartup
**always assigns** a chosen root (one assignment, after the choice):

1. Windows only: `HKCU\Environment\HF_HOME` (Desktop often has this and
   not the process var). If it is set and `len(value) <= room`, use it.
   If it is set and too long, log and continue. Never keep a too-long pin.
2. Also honor an already-set `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` the
   same way: a short one wins; a too-long one is the bug, not a setting
   to preserve.
3. The models-adjacent root (`<comfy>/models/huggingface`), if we are
   not on Windows, or it fits `room`, or `LongPathsEnabled=1`.
4. Else if `C:\ComfyUI-Models` exists, the candidate
   `C:\ComfyUI-Models\huggingface` fits `room`, and a non-raising write
   probe succeeds: use it. Creating the `huggingface` subdir under an
   existing models root is the DEPENDENCIES.md convention, not a split.
5. Else the huggingface_hub-shaped user cache, via `os.path` only:
   `os.path.join(os.getenv("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "huggingface")`,
   if it fits `room` and the write probe succeeds.
6. If nothing qualifies: leave `HF_HOME` unset, log every candidate with
   its length and why it was refused. Do not assign a too-long path.

`room` remains `259 - longest_tail - 1`. The 162 tail is the visual
`_SOURCES` bound, not a writer/TTS claim.

## Same-change companions (not a second row)

- `_otr_hf_env._default_hf_home()`: on Windows, return
  `C:\ComfyUI-Models\huggingface` only when `C:\ComfyUI-Models` exists;
  otherwise the user cache above. Duplicate the check -- prestartup must
  not import `nodes/`.
- Rewrite the prestartup comment and warning. There is no "do not pin"
  branch and no "enable Long Paths or set HF_HOME yourself".
- `apple/DEPENDENCIES.md`: HF_HOME is `<models_root>/huggingface` when
  that root fits; on Windows, when it cannot, the short fallback above.
  State the exception so a later reader does not "fix" it back.
- Tests: extract `_choose_hf_home(...)` (or equivalent) so the decision
  is callable without importing the rest of prestartup. One assignment
  site. Behavioral cases for registry / too-long adjacent / existing
  `C:\ComfyUI-Models` / neither. Keep the `_SOURCES` re-derivation of 162.
  `len(r"C:\ComfyUI-Models\huggingface")` is 29, not 27.

## Blast radius (unchanged in intent)

- Roots that already fit: byte-identical pin.
- `LongPathsEnabled=1` (this 5080): unchanged.
- Victim class (Desktop root too long, `C:\ComfyUI-Models` present):
  cache joins the existing models tree.
- Stranger Desktop, no `C:\ComfyUI-Models`, username 8+: explicit user
  cache. No UAC. Press Run.

## Rejected

| option | why |
| --- | --- |
| A (do not pin) | `_hf_fetch` never calls `ensure_hf_home` and never passes `cache_dir`; Lumina follows the import-time constant. Dual cache. |
| B (Long Paths) | admin + reboot |
| C (`\\?\` in our fetch) | writer/TTS/transformers still use pointer_path |
| D (shorter Lumina) | next long repo repeats the cliff |
| E (always `C:\ComfyUI-Models`) | mkdir `C:\` is UAC for a stranger |
| Cut F step 4's models-root join | splits the 4060: Lumina in `~/.cache`, the other weights in `C:\ComfyUI-Models` |

## Proof after the code lands

Not a push gate. A LongPaths-off Desktop boot with no launcher
`HF_HOME`, username 8+, Lumina fetch. The live decline is that commit.
