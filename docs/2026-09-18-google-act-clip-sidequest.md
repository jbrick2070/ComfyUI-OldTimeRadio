# Sidequest -- Google 1-act / 3-act clip length (2026-09-18)

Operator notes (decoded from the other window):

- 100 for 1-act, 200 for 3-act (I dunno)
- 120 for 1-act?
- You could turn off the switch. It is a knob. Off defaults to joining
  the engine menu.

## Lock

These are **Veo clip frames at the 25 fps canvas**, not word counts
and not beats. Word targets were ripped. Cloud 1-act / 3-act profiles
already share `beats: 40`.

Veo only posts 4 / 6 / 8 seconds (`discrete_frames` 100 / 150 / 200).
`eng_google_veo_video._duration_s` snaps:

- target <= 5s -> 4s (100 frames)
- target <= 7s -> 6s (150 frames)
- else -> 8s (200 frames)

So:

| Graph | Pin | Posted | Why |
| --- | --- | --- | --- |
| Google 1-act | 100 frames | 4s | cheap menu slot |
| Google 3-act | 200 frames | 8s | long menu slot |
| 120 frames | do not use | same 4s as 100 | 120/25 = 4.8s snaps down |

`VideoDirector.max_render_frames` is the knob. `0` means unpinned:
planning joins the engine menu. Leave it 0 unless the graph pins
100 (1-act) or 200 (3-act).

Draft lane presets (not shipping graphs, not in SHIPPING_SET):

- `config/profiles/google_veo_low_1act.json` -- 2 chars, `max_render_frames` 100
- `config/profiles/google_veo_low_3act.json` -- 3 chars, `max_render_frames` 200

Writer pins `google_api:slot-a/b` to `gemini-flash-latest` /
`gemini-flash-lite-latest`. Apply on a keyless box will COMBO-refuse
those writer handles until emission adds a Google admit path. Leave
them draft.
