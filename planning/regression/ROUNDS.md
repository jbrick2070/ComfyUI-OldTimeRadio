# Regression rounds: render time against episode length

Each shipped graph renders a one-act episode whose LENGTH varies run to run: the writer
writes a fresh script every time, and a longer play films more beats. A render time on its
own therefore says little; the number to compare between rounds and machines is **render
seconds per second of finished episode** (s/s), read from the published file in `otr/obs/`.

Operator, 2026-10-23: "just look at the duration of the finished files ... and add them to
the matrix". Durations are read with ffprobe from the published `_final.mp4`; render time is
the harness's RESULT line (submit to published).

## 4060 (RTX 4060 Laptop, 8 GB), the seven `otr_8gb_*` graphs, 1 act

| Graph | 2026-10-20/21 at the older main | | | 2026-10-23 at 6d7b6948 | | |
|---|---:|---:|---:|---:|---:|---:|
| | render s | episode s | s/s | render s | episode s | s/s |
| `otr_8gb_low` | 514 | 78.6 | 6.5 | 969 | 76.1 | 12.7 |
| `otr_8gb_still` | 931 | 69.7 | 13.4 | 617 | 81.7 | 7.6 |
| `otr_8gb_animatediff` | 2,825 | 89.7 | 31.5 | 1,891 | 76.0 | 24.9 |
| `otr_8gb_video` | 624 | 80.2 | 7.8 | 618 | 78.2 | 7.9 |
| `otr_8gb_ltx25_foley` | 3,611 | 75.0 | 48.1 | 3,374 | 77.3 | 43.6 |
| `otr_8gb_ltx25_mime` | 3,031 | 78.4 | 38.7 | 5,538 | 112.4 | 49.3 |
| `otr_8gb_ltx25_audio_in` | 3,504 | 67.4 | 52.0 | 10,054 | 149.5 | 67.3 |

All fourteen legs passed. Reading: the 2026-10-23 mime and audio-in renders that looked
1.8x and 2.9x slower were mostly LONGER EPISODES (112 s and 150 s against 78 s and 67 s;
both drew the Sci-Fi News bank). Per second of episode they are about 27-29% slower, which is
the part still worth a look. `otr_8gb_low` is the first leg after the server boots, so its
time carries the cold start; compare it across rounds with that in mind.

## 5080 (RTX 5080 Laptop, 16 GB), the `otr_16gb_*` graphs, 1 act, 2026-10-23 at 2def3e12

| Graph | render s | episode s | s/s |
|---|---:|---:|---:|
| `otr_16gb_low` | 353 | 119.6 | 3.0 |
| `otr_16gb_still` | 1,543 | 181.7 | 8.5 |
| `otr_16gb_video` | 8,849 | 173.2 | 51.1 |
| `otr_16gb_foley` | 12,090 | 186.2 | 64.9 |

`otr_16gb_foley` finished after the harness was stopped; its time is submit (12:27:13) to
published (15:48:43), read from ComfyUI's history and the file. Paused by the operator after
it; not run this round: `otr_16gb_foley_open`,
`otr_16gb_mime`, `otr_16gb_animatediff`, `otr_16gb_foley_flux3_ref`,
`otr_16gb_foley_open_flux3_ref`.
