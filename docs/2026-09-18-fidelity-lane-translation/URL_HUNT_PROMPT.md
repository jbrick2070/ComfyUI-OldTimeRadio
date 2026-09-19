# Prompt: find the missing public-domain Shakespeare translation URLs

Paste this whole file into ChatGPT (with browsing) or a cursor agent.

---

I need URLs for specific public-domain Shakespeare translation SCENES. A
previous pass failed because it listed plausible URLs without opening them, so
the rule here is: **if you did not open the page and see the scene's dialogue,
mark it NOT FOUND.** A URL that 404s or lands on an index costs me more than a
blank.

## The legal rule (non-negotiable)

A translation qualifies only if it clears **both**:
- **US:** first published **before 1931**, and
- **life+70:** translator died **before 1956**.

One test is not enough. Do not offer a translation that is public domain in
only one jurisdiction, and do not rely on a noncommercial or fair-use argument.
State the translator's death year and the edition's first-publication year for
every row.

## What I already have (do NOT re-research these)

| Lang | Translator | Status |
|---|---|---|
| fr | François-Victor Hugo (d.1873), 1865-72 | have the play page, need the SCENE |
| fr | François Guizot (d.1874) | have the Gutenberg book, need the scene range |
| it | Carlo Rusconi (d.1889), 1858-59 | only have the collected-works INDEX |
| es | Menéndez y Pelayo (d.1912) | have the play pages |
| es | Moratín (d.1828) Hamlet | have the Gutenberg book |
| pt | Luís I (d.1889) Hamlet | have the Gutenberg book |
| hi | Lala Sitaram (d.1937), 1917 | have the archive.org scan |
| ja | Tsubouchi (d.1935), 沙翁傑作集 1921-23 | have Macbeth/Midsummer/Twelfth Night scans |

## WHAT I NEED — seven rows

**1. Japanese — 沙翁傑作集 No. 7, The Tempest (テムペスト / 颶風), 1921.**
The Wikimedia Commons / NDL scan URL. I guessed `NDL979376` and it 404'd.
The sibling files that DO work look like:
`https://commons.wikimedia.org/wiki/File:NDL979379_沙翁傑作集_第10編_(マクベス)_part1.pdf`
Give the real NDL id and file name for volume 7. While you are there, confirm
the volume number and year for **Hamlet, King Lear, Much Ado, As You Like It
and The Comedy of Errors** in the same pre-1931 series — I need to know which
of those exist before 1931 at all.

**2. Italian — Rusconi, Macbeth Act 1 Scene 3.**
I have `https://it.wikisource.org/wiki/Teatro_completo_di_Shakspeare`, which is
the collected-works index with no act headings. I need the page for Macbeth
itself (Italian: *Macbetto* or *Macbeth*), ideally the subpage carrying Act I.

**3. French — Hugo, Hamlet Act 1 Scene 1** and **4. French — Hugo, King Lear
Act 1 Scene 1.** The Macbeth page I have is
`https://fr.wikisource.org/wiki/Macbeth_(trad._Hugo)`. Give the equivalent for
*Hamlet* and *Le Roi Lear* in the same Hugo translation, and say whether the
play is one page or split into act subpages (I need whichever page actually
contains the dialogue).

**5. Spanish — As You Like It Act 3 Scene 2**, translator **Guillermo
Macpherson** (d.1898, published 1873-1897, verse, translated direct from
English). Reported to be at cervantesvirtual.com under "Biblioteca de
Traducciones Españolas". I need a working text URL **and** a plain statement of
that site's reuse terms (its *Marco legal* page) — if the terms forbid reuse,
say so and I will drop it rather than argue fair use.

**6. Italian — Maffei or Carcano verse alternatives.** Andrea Maffei (believed
d.1885 — confirm) did *Macbeth* 1863 and *La tempesta* 1869; Giulio Carcano
(d.1884) did the complete plays in verse 1875-82 but I have found no digital
text. If a digital text exists for either, give the URL; if not, say "no
digital text located" and stop.

**7. Mandarin — anything that clears BOTH tests.** This is the one I expect to
come back empty, and an honest empty is the right answer. Zhu Shenghao (d.1944)
was published from 1947, so he fails the US test; Tian Han (d.1968) published
1922-24, so he fails life+70. If a *third* pre-1931 Chinese translator exists
whose translation was published before 1931 and who died before 1956, name them
with evidence. Otherwise confirm there is none.

## Output format

One row per item:

```
lang | play | scene | translator (born-died) | first published | URL |
opened? yes/no | what is actually on that page (index / whole play / single
scene / page scan) | speaker-label style, quoted from the text | licence or
reuse terms
```

Then a short list of anything you marked NOT FOUND, and why.

## Two traps that already cost me time

- **A work ID is not a file.** Aozora work IDs and archive.org identifiers
  exist for texts that have never been transcribed. Open it.
- **"A transcribir" / 作業中 / "not proofread"** on a page means the scan is
  uploaded and the text does not exist yet. That is a NOT FOUND, not a source.
- Strip any `?utm_source=` parameter before giving me a URL. Its presence tells
  me the row was never opened.
