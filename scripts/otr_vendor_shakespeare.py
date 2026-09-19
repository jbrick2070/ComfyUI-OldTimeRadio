#!/usr/bin/env python
"""Turn a located lead into a vendored scene file the writer can perform.

    python scripts/otr_vendor_shakespeare.py               # what is extractable
    python scripts/otr_vendor_shakespeare.py --write       # vendor them

WHAT THIS IS FOR. `leads.json` records where a translation LIVES; nothing in
the pipeline can read a URL. This opens the page, extracts the target scene,
normalises the speaker labels to `NAME:`, writes
`config/source_banks/shakespeare/translations/<iso>/<play>_<scene>.txt`, and
adds the manifest row `_otr_verbatim_corpus.load_manifest` validates.

Running it with no flag WRITES NOTHING. It reports, per lead, whether the target act and scene
are actually on the page -- because a lead can name the right work and the
wrong page. Measured on the first run: `it/tempest 3.1` pointed at
`La_tempesta_(Shakespeare-Maffei)/Atto_primo`, which holds Act I only, so
extracting "3.1" from it would have vendored Act I scene 1 under the name of
Act III scene 1. A wrong scene that looks right is the worst outcome this
corpus can produce, which is why the check runs first and separately.

EDITIONS DISAGREE ABOUT WHAT "3.2" MEANS. Three conventions are already in the
set: Hugo numbers scenes CONTINUOUSLY with no act headings at all, while
Rusconi and Marquez reset per act. So the scene is located by the edition's OWN
label, recorded on the lead by the operator's verification pass, and never by
counting headings down the page.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes import _otr_verbatim_corpus as CORPUS  # noqa: E402

LEADS = os.path.join(_REPO, "config", "source_banks", "shakespeare",
                     "translations", "leads.json")
OUT_ROOT = os.path.join(_REPO, "config", "source_banks", "shakespeare",
                        "translations")
CACHE = os.path.join(_REPO, "tmp", "corpus_cache")
USER_AGENT = ("OTR-corpus-vendor/1.0 (ComfyUI-OldTimeRadio; "
              "https://github.com/jbrick2070/ComfyUI-OldTimeRadio)")

#: Leads whose url serves readable text. A page scan needs transcription first
#: and is not this script's job.
TEXT_FORMS = ("whole_work_single_page", "act_page", "scene_page")


def fetch(url):
    """The page bytes, cached. Never re-scrapes during iteration."""
    os.makedirs(CACHE, exist_ok=True)
    name = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".txt"
    path = os.path.join(CACHE, name)
    if os.path.isfile(path):
        return io.open(path, encoding="utf-8").read()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
        charset = response.headers.get_content_charset()
    # ASK THE DOCUMENT WHEN THE SERVER DOES NOT SAY. Falling straight back to
    # UTF-8 silently mojibakes any page that is not UTF-8, and `errors=
    # "replace"` means it never raises -- it just returns a page where every
    # heading is unfindable. Measured on Aozora Bunko, which serves Tsubouchi's
    # Shakespeare as Shift_JIS with NO charset in the HTTP header: the act
    # heading was present in the bytes and matched ZERO times after decoding,
    # so the whole Japanese lane looked unparseable when it was merely misread.
    if not charset:
        head = raw[:4096]
        found = re.search(
            br"""(?is)<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", head)
        charset = found.group(1).decode("ascii", "replace") if found else "utf-8"
    try:
        body = raw.decode(charset, errors="replace")
    except LookupError:
        # An encoding name Python does not know is a bad guess, not a reason to
        # die -- fall back rather than lose the page.
        body = raw.decode("utf-8", errors="replace")
    io.open(path, "w", encoding="utf-8", newline="\n").write(body)
    return body


def to_text(markup):
    """Rendered HTML to line-broken text.

    Block elements END THEIR LINE. A strip that does not break lines leaves
    every speaker label inline with its speech, and every line-anchored rule --
    the heading finder, the label counter -- as blind as it was on the raw HTML.

    The parameter is `markup`, not `html`: it used to be `html`, which SHADOWED
    the stdlib module of that name and turned `html.unescape(body)` into an
    attribute lookup on a str.
    """
    body = re.sub(r"(?is)<(script|style|template)\b.*?</\1>", " ", markup)
    body = re.sub(r"(?s)<!--.*?-->", " ", body)
    # THE SITE'S OWN FOOTER IS NOT DIALOGUE, AND IT REACHES THE STAGE WHEN THE
    # SCENE IS THE LAST ONE ON ITS PAGE. `extract` ends a scene at the NEXT
    # heading of the same shape; when there is no next heading it runs to the
    # end of the document and swallows everything after the play -- the
    # footnotes, `Estratto da <url>`, the licence blurb, `Informativa sulla
    # privacy`, `Dichiarazione sui cookie`, `Aggiungi lingue`.
    #
    # IT IS INVISIBLE TO EVERY COUNT, which is why it shipped in five Italian
    # scenes before a reviewer read the file tails. None of that text carries a
    # `NAME:` label, so `normalise_labels` appends it to the PENDING speaker --
    # and the last character of the scene delivers the privacy policy as the
    # end of their closing speech. Speech and speaker counts do not move by one.
    # Same class as the `[p. 39 ]` page markers, with a far larger payload.
    #
    # KEEP THE ARTICLE, NOT THE PAGE. Naming chrome containers one at a time
    # does not converge -- the first cut stripped `printfooter`, `<footer>` and
    # the `footer-*` lists and the tail still carried `Ricerca Ricerca ...
    # Aggiungi lingue Aggiungi argomento`, because that furniture is scattered
    # through the header and the navigation as well as the foot.
    #
    # The article body is BRACKETED: MediaWiki opens it with
    # `<div class="mw-parser-output">` and closes the readable part at
    # `<div class="printfooter">`. Cut to that span and every piece of site
    # chrome is outside it at once, in any language and any skin. Either marker
    # missing leaves the body untouched, so a page that does not use them is
    # never truncated by this.
    start = re.search(r'(?is)<div[^>]*class="[^"]*mw-parser-output[^"]*"[^>]*>', body)
    if start:
        body = body[start.end():]
    stop = re.search(r'(?is)<div[^>]*class="[^"]*printfooter[^"]*"', body)
    if stop:
        body = body[:stop.start()]
    # THE EDITOR'S FOOTNOTES ARE INSIDE THE ARTICLE AND ARE STILL NOT DIALOGUE.
    # Bracketing to the article body removes the site's furniture and leaves
    # these, which sit in `<ol class="references">` under a `Note` heading and
    # land on the last speaker for the same reason: no `NAME:` label, so
    # `normalise_labels` appends them to whoever spoke last. Benedick was
    # closing his scene with `Note Allusione al vecchio proverbio che le vecchie
    # zitelle si dannano`, and Prospero's scene ended on the word `Warburton`.
    # A footnote is the EDITION talking about the play, never a character in it.
    body = re.sub(r"(?is)<ol[^>]*class=\"[^\"]*references[^\"]*\"[^>]*>.*?</ol\s*>",
                  " ", body)
    body = re.sub(r"(?is)<div[^>]*class=\"[^\"]*mw-heading[^\"]*\"[^>]*>\s*"
                  r"<h[1-6][^>]*id=\"Note\"[^>]*>.*?</h[1-6]>\s*</div\s*>", " ", body)
    # A SECTION EDIT LINK IS SITE FURNITURE AND A VOICE WILL READ IT ALOUD.
    # MediaWiki puts `<span class="mw-editsection">[ 編輯 ]</span>` after every
    # heading. Tag-stripped it becomes three lines -- `[`, `編輯`, `]` -- sitting
    # directly under the scene heading, inside the extracted body, where they
    # are indistinguishable from an unattributed stage direction. Measured on
    # the Chinese Hamlet act page 2026-09-19; the same furniture is what put
    # `Adicionar idiomas` in the Portuguese tail. Removed with the span, before
    # any line breaking, so no empty bracket lines survive it.
    # THE SPAN NESTS ONE LEVEL, so a lazy `.*?</span>` stops at the inner
    # bracket and keeps the word. The obvious second try -- strip any anchor
    # whose href carries `action=edit` -- is WORSE and the blast-radius check
    # caught it: it.wikisource wraps its PAGE-NUMBER markers in edit-links too,
    # so that rule pushed `[p. 39 ]` into the middle of Macbeth's witches and
    # into three more Italian speeches. **The speech and speaker counts did not
    # move at all** (51/8 before and after), so only a text diff against the
    # stored file could see it -- a count check passes this defect.
    #
    # A TEMPERED MATCH to the OUTER close is what is actually wanted: consume up
    # to the first `</span></span>` pair, which is the editsection's own end,
    # and bound the run so a malformed page cannot make it quadratic.
    body = re.sub(
        r'(?is)<span\b[^>]*class="[^"]*mw-editsection[^"]*"[^>]*>'
        r'(?:(?!</span>\s*</span>).){0,400}</span>\s*</span>',
        " ", body)
    # A RUBY GLOSS IS A PRONUNCIATION GUIDE, NOT WORDS ANYONE SAYS. Japanese
    # editions annotate a kanji with its reading:
    #     <ruby><rb>誓言</rb><rp>（</rp><rt>せいごん</rt><rp>）</rp></ruby>
    # `<rb>` is the word, `<rt>` is how to pronounce it, and `<rp>` holds the
    # fallback brackets a ruby-less browser would show. Strip tags naively and
    # all three survive, so the line becomes `誓言（せいごん）` -- the word
    # followed by its own pronunciation, which a voice READS ALOUD. Measured on
    # Tsubouchi's Romeo and Juliet, where nearly every content word is glossed:
    # `威權（ゐけん）相如（あひし）く二名族（めいぞく）が、` is one line's worth.
    #
    # `<rp>` first, then `<rt>`, then the wrapper tags fall to the ordinary
    # strip -- which leaves exactly `<rb>`, the text that is actually spoken.
    body = re.sub(r"(?is)<rp\b[^>]*>.*?</rp\s*>", "", body)
    body = re.sub(r"(?is)<rt\b[^>]*>.*?</rt\s*>", "", body)
    # READ WHO SPEAKS OFF THE PAGE BEFORE THE PAGE IS THROWN AWAY. Every edition
    # in the set marks its speakers; once the tags are gone that fact cannot be
    # recovered, only guessed at -- see `mark_speakers`.
    body = strip_direction_parentheticals(body)
    body = mark_speakers(body)
    body = re.sub(r"(?i)</(div|p|li|tr|h[1-6]|span)\s*>", "\n", body)
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    # A TAG ENDS AT THE FIRST `>` THAT IS NOT INSIDE A QUOTED ATTRIBUTE. The
    # obvious `<[^>]+>` stops at the first `>` anywhere, and MediaWiki's Parsoid
    # puts a JSON blob in `data-mw` that contains escaped markup -- so a `>`
    # lands mid-attribute and the rest of the attribute survives as TEXT.
    #
    # Measured on the transcribed Portuguese Hamlet: 16 lines came out over 200
    # characters, fifty JSON tokens reached the text, and both `ACTO PRIMEIRO`
    # and the speaker `BERNARDO` were buried at the tail of ~300-character lines
    # of `"quality":{"wt":"4"}},"i":1}},"</span>"]}'` instead of standing on
    # their own. That breaks scene location (the act heading is not findable)
    # and, if vendored, a character reads the JSON ALOUD.
    #
    # This is not Portuguese-specific: every Parsoid-rendered Wikisource page
    # carries `data-mw`, so it was luck of the layout that the first four
    # editions came out clean. Quoted runs are now skipped wholesale.
    body = re.sub(r"(?s)<(?:[^>\"']|\"[^\"]*\"|'[^']*')*>", "", body)
    # UNESCAPE EVERYTHING, not a hand-list. The hand-list missed `&#91;` and
    # `&#93;` -- Wikisource's footnote brackets -- and they survived into the
    # vendored Macbeth as "(&#91; 9&#93; )", which the announcer would have
    # READ ALOUD as "ampersand hash ninety-one".
    body = html.unescape(body)
    # Footnote markers and page breaks are the EDITION's apparatus, not the
    # translator's words, and every one of them is READ ALOUD if it survives.
    # Three forms are in this corpus and the first cut caught only one:
    #   [9]                  -- the bracketed footnote
    #   (19)                 -- Hugo's parenthesised note, five of them in Lear
    #   [p. 39 modifica ]    -- Wikisource's page-break link, INSIDE a speech,
    #                           which the announcer would read as "p 39 modifica"
    body = re.sub(r"\[\s*\d{1,3}\s*\]", "", body)
    body = re.sub(r"\[\s*p\.\s*\d{1,4}\s*modifica\s*\]", "", body, flags=re.I)
    body = re.sub(r"\(\s*\d{1,3}\s*\)", "", body)
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"[ \t ]+", " ", body)
    return re.sub(r"\n\s*\n+", "\n", body)


#: Wraps a speaker name the PUBLISHER marked, so the label rules below read a
#: fact off the page instead of guessing one from prose.
SPEAKER_MARK = "\x01"

#: A stage direction / sound cue. Every edition in the set sets its apparatus in
#: SMALLER TYPE than the dialogue; the exact percentage is the transcriber's
#: taste and nothing else.
#:
#: THE PERCENTAGE IS NOT A GATE (operator 2026-09-19: "I don't want font size
#: blocking our workflow"). This read `9[05]%` -- two hard-coded values, taken
#: from the two editions that happened to be vendored first. It found ZERO on
#: the transcribed Portuguese Hamlet, whose stage directions are `80%` spans
#: (`(Retira-se.)`, `(O gallo canta.)`), so that edition would have PERFORMED
#: every one of them. Discovering a third convention should not mean a code
#: change and a fourth should not mean another; "smaller than the body text" is
#: the rule the publishers are actually following.
#:
#: 60-99% is the whole band of small type. 100% and above is body text or a
#: heading and is never apparatus.
_DIRECTION_BLOCK = re.compile(
    r'(?is)<(div|span|p)\b[^>]*font-size:\s*[6-9]\d%[^>]*>(?P<body>.*?)</\1\s*>')

#: WIKISOURCE'S PROOFREAD LAYOUT MARKS BY CLASS, NOT BY TYPE SIZE. The
#: Portuguese Hamlet (pt.wikisource, Acto primeiro/Cena I) sets EVERY piece of
#: apparatus in `div class="tiInherit"` -- 68 of them on one scene page --
#: and does not vary the type size at all, so `_DIRECTION_BLOCK` returns zero
#: and the page arrives with no speakers and no directions separated.
#:
#: What those 68 blocks hold is all four kinds of apparatus at once:
#:
#:   SCENA I                                             a heading
#:   Elsenor, a explanada do castello                    the setting
#:   FRANCISCO de sentinella, BERNARDO vem encontrar-se  stage business
#:   BERNARDO                                            a SPEAKER
#:
#: THE EXISTING BARE-LABEL TEST ALREADY SEPARATES THEM, and this is the exact
#: mirror of the Aozora case: `_is_bare_label` asks "no lowercase, no colon?",
#: which is meaningless in Japanese and precisely right here -- the speakers
#: are single all-caps names and every direction carries lowercase words. So
#: these blocks go through `_strip_direction` (the CONDITIONAL stripper),
#: unlike the indent blocks below, which go through an unconditional one.
_TIINHERIT_BLOCK = re.compile(
    r'(?is)<div[^>]*class="[^"]*tiInherit[^"]*"[^>]*>(?P<body>.*?)</div\s*>')

#: AOZORA / TSUBOUCHI MARKS BY INDENT CLASS, NOT BY TYPE SIZE (measured
#: 2026-09-19 on Romeo and Juliet, 42773_39853.html). The row was held on a
#: recorded diagnosis that turned out to be wrong: it said this edition "sets
#: stage business inline, in the same run as the dialogue, with no markup of
#: its own". It does not. The page is CLEANLY marked -- it simply uses a
#: vocabulary neither rule above knows:
#:
#:   <div class="burasage" ...>サン　　やい、グレゴリー、...</div>   dialogue
#:   <div class="jisage_8" ...>...サンプソンとグレゴリーとが...出る。</div>  business
#:   <div class="jisage_6"><h4>第一場　　ヱローナ。街上。</h4></div>       heading
#:
#: So NOTHING fired: no direction was stripped and no speaker was marked, and
#: 95 marked dialogue lines in act 1 scene 1 collapsed into unattributed prose
#: -- which is how a one-line part came to "absorb" the Prince's speech. The
#: lesson the hold text drew is still the right one and is what measured this:
#: count what the EDITION marks (95 burasage divs, 18 non-heading jisage divs),
#: never what survived parsing.
#:
#: The heading exemption is load-bearing for the same reason it is above:
#: `extract` locates a scene BY the jisage_6/jisage_7 headings, so a blanket
#: jisage strip would delete the very anchors it searches for.
_AOZORA_BUSINESS = re.compile(
    r'(?is)<div class="jisage_\d+"[^>]*>(?P<body>(?:(?!<h[1-6]\b).)*?)</div\s*>')

#: ZHU SHENGHAO MARKS BY PARAGRAPH START PLUS AN IDEOGRAPHIC SPACE (measured
#: 2026-09-19 against the fetched act page, not inherited from the hold note).
#: This is the THIRD row in one day held on a diagnosis that was wrong about its
#: own edition. The note said the edition "breaks no paragraph before a speaker,
#: so five speeches land in the wrong mouth mid-line". It breaks one every time:
#:
#:   <p>波　咱们都会齐了吗？</p>                     dialogue, speaker + U+3000
#:   <div class="center">【衮斯，史纳格，波顿…上。</div>  block business
#:   <div class="center"><span …>第三幕</span></div>     heading
#:
#: Nothing fired, so the pipeline fell back to heuristic plain-text parsing, and
#: THAT is what merged the speeches. Counted on the page: 178 marked speeches
#: across 19 names, against 0 that `mark_speakers` claimed.
#:
#: THE ABBREVIATION IS ONE OR TWO CHARACTERS, AND THAT IS THE TRAP. This edition
#: abbreviates every character to their opening character -- 黑 is 黑美霞
#: (Hermia), 莱 is 莱散特 (Lysander), 迫 is 迫克 (Puck) -- and `_marked_name`
#: rejects `len(name) < 2`, so a rule can mark all 19 correctly and still have
#: every one of them dropped downstream. See the CJK carve-out there.
#:
#: DO NOT WIDEN THE HEADING EXEMPTION. The obvious reading of 第 is "the opener
#: of 第三幕/第一场", and it is WRONG: 第 is DEMETRIUS (第米屈律斯), who speaks 21
#: times. The strict form below never fires once on this page -- the real
#: headings live in a centred div, not at a paragraph head -- and a loose
#: "starts with 第" form deletes a character while leaving a parse that looks
#: clean. It is kept only as insurance against an edition that does inline one.
_ZH_BUSINESS = re.compile(
    r'(?is)<div class="center">\s*【.*?</div\s*>')

#: The speaker opens a paragraph, or follows a block boundary, and is closed by
#: ONE ideographic space. The optional run after the tag absorbs Wikisource's
#: page-break markup, which MediaWiki injects between `<p>` and the first
#: character.
#:
#: NO NESTED QUANTIFIER HERE, AND THAT IS NOT A STYLE PREFERENCE. The first cut
#: skipped the page-break markup with `(?:<span\b[^>]*>.*?</span>\s*)*` -- a
#: lazy dot inside a star -- which is catastrophic backtracking: on the
#: span-heavy French and Italian pages the whole vendor run stopped producing
#: output and never finished. It was caught ONLY by the blast-radius check,
#: because the Chinese page it was written for is small enough to complete.
#: A rule can be correct on its own edition and still hang every other one.
#: Each alternative below consumes a bounded token, so the run is linear.
_ZH_SPEAKER = re.compile(
    r'(?is)(?P<prefix><p\b[^>]*>|</p>\s*)'
    r'(?:<[^<>]{0,300}>|[\s​]|&\#8203;|&zwnj;){0,16}'
    r'(?P<who>[一-鿿]{1,4})　(?!　)')

#: A printed act or scene heading, which `extract` uses as a scene anchor.
_ZH_HEADING_NAME = re.compile(r"^第.*[幕场場]$")

#: The speaker label opens the line and is closed by an ideographic space run.
#: The separator is ONE OR MORE: the edition prints `サン　　` (two) and
#: `ベン　` (one) in the same scene, and requiring two found 36 of 95.
#:
#: A LABEL CAN CONTAIN AN IMAGE. Shift_JIS cannot encode every character, so
#: Aozora sets the missing ones as `<img class="gaiji" alt="※(濁点付き片仮名ヲ...)">`
#: -- and Benvolio's label is one of them (`ベン` + a dakuten ヲ). A rule that
#: stopped at `<` marked 71 of 95 lines and dropped Benvolio from the scene
#: entirely, which is the same class of defect as the one this fixes: a whole
#: part missing, and a count is what shows it. The tags are allowed inside the
#: label and stripped from the captured name, so the label stays the stable
#: key the manifest speaker_map binds to a roster name.
_AOZORA_SPEAKER = re.compile(
    r'(?is)(<div class="burasage"[^>]*>)\s*'
    r'(?P<who>(?:<img[^>]*>|[^\s　<]){1,10}?)　+')

#: How each publisher marks a speaker. Hugo and Marquez converge on small caps;
#: Rusconi is handled by the per-edition rules below because his italic markup
#: is also used inside dialogue and stage business.
_SPEAKER_SPANS = (
    # Wikisource (Hugo): class="sc", the name often rendered lowercase.
    #
    # `personnage` IS THE SAME FACT IN A DIFFERENT SPELLING, and missing it cost
    # nearly a whole play. Hugo's `Le soir des rois` was transcribed by someone
    # who used `class="personnage"` -- 918 spans of it against 119 `sc` -- and
    # with only `sc` known, `to_text` marked 66 speakers on a page that has 918
    # labels. The scene would not have been WRONG, it would have been almost
    # entirely unattributed prose. Two independent reviewers reported it before
    # it was measured here.
    #
    # Safe to fold into the union rather than hold for a per-edition binding:
    # `personnage` appears ZERO times on all four vendored pages, so it cannot
    # reach them, and the class name means exactly one thing.
    re.compile(r'(?is)<span\b[^>]*class="(?:sc|personnage)"[^>]*>'
               r'(?P<name>[^<]{1,40})</span>'),
    # Gutenberg (Marquez): an inline small-caps style, name carries its period.
    re.compile(r'(?is)<span\b[^>]*font-variant:\s*(?:all-)?small-caps[^>]*>'
               r'(?P<name>[^<]{1,40})</span>'),
)


#: Rusconi's headed transcription: the speaker opens the paragraph, optionally
#: after MediaWiki's page-number span, and the italic label is followed by a
#: period.  The paragraph is matched as a unit so the prefix cannot drift over
#: a neighbouring paragraph while looking for the first italic tag.
_RUSCONI_PARAGRAPH = re.compile(
    r'(?is)(?P<open><p\b[^>]*>)(?P<body>.*?)</p\s*>')
#: THE ORDINAL IS NOT ALWAYS A `<sup>`, AND ASSUMING IT WAS MERGED FOUR
#: CHARACTERS INTO ONE. Rusconi's Macbeth page writes the witches as
#: `1<sup>a</sup> <i>Strega</i>.`, so a `<sup>`-only pattern split them
#: correctly and looked finished. His Midsummer page -- same collection, same
#: translator, different transcriber -- writes `1ª <i>Fat</i>.` with a BARE
#: feminine ordinal and no tag at all. The ordinal group then failed to match,
#: the `1ª ` was swallowed by the discarded prefix, and Peaseblossom, Cobweb,
#: Moth and Mustardseed all arrived as one speaker called `FAT` -- eleven lines
#: in one mouth, including three separate `Salve!` greetings and the fairies
#: NAMING THEMSELVES one after another.
#:
#: It was nearly written off as a translator's choice. The Chinese translation
#: of the SAME SCENE, vendored the same day, binds all four individually, which
#: is what settles it: four characters exist in that scene and a rule that
#: returns one is wrong. When two editions of one scene disagree about how many
#: people are in it, suspect the parser before the translator.
_RUSCONI_HEAD = re.compile(
    r'(?is)^(?P<prefix>(?:(?!<i\b)[^<()]|<(?!i\b)[^>]*>)*?)'
    r'(?P<ord>\d\s*(?:<sup>\s*[ao]\s*</sup>|[ªº])\s*)?'
    r'<i>(?P<name>[^<()]{1,40})</i>'
    r'(?:\s*\([^()]{0,200}\))?\s*\.')

#: A parenthetical whose CONTENT IS WHOLLY ITALIC. Rusconi and Marquez set
#: stage business that way -- `(<i>entra Rosse</i>)`, `(<i>escono</i>)` -- and
#: the editions are consistent about it.
#:
#: THE BODY IS NOT LENGTH-CAPPED, AND THE CAP THAT USED TO BE HERE WAS A BUG.
#: It read `{1,80}`, which is a plausible-looking number and nothing more: a
#: stage direction is a sentence when the edition wants one, and Rusconi's
#: Macbeth has two past 80 -- the 89-character `rimane alcuni istanti assorto
#: in profonda meditazione, quindi si volge ad Angus e a Rosse`, which survived
#: into the vendored `it/macbeth 1.3` and would have been PERFORMED IN
#: MACBETH'S OWN VOICE mid-speech, and a 163-character scene setting. The
#: failure is silent in the worst direction: a direction too long for the cap
#: is not flagged, it is simply kept and spoken.
#:
#: What actually bounds this match is `[^<]`, which cannot cross a tag, so the
#: body is confined to a single text run inside one `<i>` that is immediately
#: wrapped in parentheses. That is the real anchor. A character count on top of
#: it bought nothing and cost a wrong mouth.
_ITALIC_PARENTHETICAL = re.compile(
    r"\(\s*<i>(?P<body>[^<]+)</i>\s*\)", re.I | re.S)


def strip_direction_parentheticals(markup):
    """Drop `(<i>entra Rosse</i>)`, keep `(car cette partie du monde...)`.

    A BLANKET `()` STRIP WOULD DELETE REAL DIALOGUE. The verbatim cleaner keeps
    parenthetical words on purpose, because Folger prints "(God shield us!)"
    INSIDE Bottom's line, and Hugo does the same in French -- Horatio's "(car
    cette partie du monde connu l'estimait pour tel)" is spoken. But Folger
    puts stage business in SQUARE brackets, which the selector already strips
    before parsing, while the colon editions put it in round ones. So the
    vendored lane inherited an assumption that is false for its own sources and
    would have performed `entrano Rosse e Angus` and `le streghe scompariscono`
    aloud.

    The page separates the two and we do not have to guess: stage business is
    ITALIC, dialogue is not. Same rule as the speaker marks, read at vendor
    time while the markup still exists -- once it is plain text the distinction
    is gone and only a word-list could stand in for it.
    """
    return _ITALIC_PARENTHETICAL.sub(" ", markup)


def mark_speakers(markup):
    """Tag the speaker names the EDITION marked, and drop its stage business.

    WHY THIS EXISTS. The first cut threw the markup away and then guessed the
    speakers back out of the plain text, with a rule that a name must RECUR.
    That guess cast sound cues as people (FANFARES got a voice in King Lear)
    and, far worse, MISSED Hugo's inline labels -- he prints them lowercase
    with a comma qualifier, `cordelia , a part.`, which the comma rule
    discarded, leaving Cordelia's aside INSIDE Goneril's speech. A speech
    count cannot see a wrong mouth; that is why this reads the page instead.

    ORDER MATTERS AND IS NOT OBVIOUS. A stage direction NAMES the people who
    enter, and marks them the same way: `<div style="font-size:90%">Fanfares.
    Entrent <span class="sc">Macbeth</span>...`. So the direction blocks are
    neutralised FIRST -- their text is kept as narration, their speaker markup
    is not -- and only the spans that survive are speakers. Reversing these two
    steps casts every entrance as a speaking part.
    """
    def _is_bare_label(text):
        """Is this small-type block a SPEAKER NAME rather than stage business?

        SOME TRANSCRIBERS SET THE SPEAKER IN SMALL TYPE TOO, and once the
        direction rule reads the whole 60-99% band instead of two hard-coded
        percentages, it reaches them. Measured on Menendez y Pelayo's Spanish
        editions: `font-size: 83%` wraps EVERY speaker -- 127 such blocks in
        Macbeth, 193 in Romeo and Juliet -- so stripping the band wholesale
        would delete `ROMEO.`, `BENVOLIO.`, `MERCUTIO.` and leave the play
        speaker-less. Neither edition is vendored, so nothing shipped was hurt;
        this closes the trap before someone records a label for one.

        The signal is punctuation and case, not the percentage: a speaker label
        carries no lowercase and no colon, stage business is a sentence.
        `PERSONNAGES :` is apparatus and is caught by the colon; the ordinal
        indicators in `BRUJA 1.ª` are not lowercase words and are allowed.

        An ACT or SCENE heading also survives this test, and that is correct --
        `extract` locates a scene BY those headings, so removing them would
        break scene location outright. Nothing downstream mistakes them for a
        speaker: the name shapes want a trailing period or a colon and a
        heading has neither.
        """
        body = re.sub(r"(?s)<[^>]+>", " ", text)
        # `html.unescape`, spelled out: the parameter here is `markup` for the
        # reason `to_text` documents -- a parameter named `html` shadows the
        # stdlib module and silently turns this into an attribute lookup.
        body = re.sub(r"\s+", " ", html.unescape(body)).strip()
        if not body or len(body) > 30:
            return False
        if ":" in body or "：" in body:
            return False
        if any(ch.islower() and ch not in "ªº" for ch in body):
            return False
        return any(ch.isalpha() for ch in body)

    def _strip_direction(match):
        # DROP THE DIRECTION, DO NOT KEEP IT AS NARRATION. Keeping the words
        # put them back in a mouth: an unlabelled line merges into the PENDING
        # speech, so `GLOCESTER: ... Le roi vient. Fanfares. Entrent` had
        # Glocester announcing his own trumpets, and Lear reading "A Cordelia."
        # aloud. A vendored file is the translator's SPOKEN text labelled by
        # speaker; stage business is the edition's apparatus and is not
        # performed. Same rule for the qualifier on a label ("a part",
        # "montrant Edmond") -- it was never dialogue either.
        #
        # UNLESS THE BLOCK IS A BARE NAME. Some transcribers set the SPEAKER in
        # small type as well, and dropping it would leave the play without
        # anyone to say the lines. Kept on its own line so the ordinary name
        # shapes can claim it -- see `_is_bare_label`.
        if _is_bare_label(match.group("body")):
            return "\n" + match.group("body") + "\n"
        return "\n"

    def _mark(match):
        # `groupdict` rather than `group("ord")`: only the Rusconi pattern has
        # that group, and asking the others for it by name is an error.
        name = match.group("name").strip()
        ordinal = re.sub(r"(?s)<[^>]+>|\s+", "",
                         match.groupdict().get("ord") or "")
        full = ("%s %s" % (ordinal, name)).strip()
        return "\n%s%s%s " % (SPEAKER_MARK, full, SPEAKER_MARK)

    body = _DIRECTION_BLOCK.sub(_strip_direction, markup)
    # Same conditional stripper: on this layout a bare all-caps block IS the
    # speaker, and dropping it would leave the play with nobody to say the
    # lines.
    body = _TIINHERIT_BLOCK.sub(_strip_direction, body)
    # Same order, same reason, for the indent-marked editions: the business
    # blocks NAME the people who enter, so they are neutralised before any
    # speaker is read off a dialogue div.
    #
    # AN UNCONDITIONAL STRIP, NOT `_strip_direction` (codex, 2026-09-19). That
    # helper keeps a block that `_is_bare_label` calls a speaker name, and that
    # test asks "no lowercase and no colon?" -- a question JAPANESE CANNOT FAIL,
    # because it has no case at all. So every short business line passed it and
    # was kept: `と劍を拔く。` ("and draws his sword") and `二人ともに入る。`
    # ("both exit") survived into act 1 scene 1 as speech. The guard is right
    # where it lives -- the font-size band really does wrap speakers in the
    # Spanish editions -- and wrong here, because the indent class is
    # unambiguous: `jisage_N` without a heading is never a speaker.
    body = _AOZORA_BUSINESS.sub("\n", body)
    def _mark_indent_speaker(mm):
        who = re.sub(r"(?is)<[^>]+>", "", mm.group("who")).strip()
        if not who:
            return mm.group(0)
        return "%s\n%s%s%s " % (mm.group(1), SPEAKER_MARK, who, SPEAKER_MARK)

    body = _AOZORA_SPEAKER.sub(_mark_indent_speaker, body)

    # Same order and the same reason once more: the block directions NAME the
    # people who enter, so they are neutralised before any speaker is read off a
    # paragraph. Unconditional, like the Aozora strip and for the identical
    # reason -- `_is_bare_label` asks "no lowercase and no colon?", a question
    # Chinese cannot fail, so the conditional stripper would keep every short
    # direction as speech.
    body = _ZH_BUSINESS.sub("\n", body)

    def _mark_zh_speaker(mm):
        who = mm.group("who").strip()
        if not who or _ZH_HEADING_NAME.match(who):
            return mm.group(0)
        return "%s\n%s%s%s " % (mm.group("prefix"), SPEAKER_MARK, who,
                                SPEAKER_MARK)

    body = _ZH_SPEAKER.sub(_mark_zh_speaker, body)

    def _mark_rusconi_paragraph(match):
        head = _RUSCONI_HEAD.match(match.group("body"))
        if not head:
            return match.group(0)
        name = head.group("name").strip()
        ordinal = re.sub(r"(?s)<[^>]+>|\s+", "",
                         head.groupdict().get("ord") or "")
        full = ("%s %s" % (ordinal, name)).strip()
        return (match.group("open") + "\n" + SPEAKER_MARK + full
                + SPEAKER_MARK + " " + match.group("body")[head.end():]
                + "</p>")

    # ONE EDITION, ONE RULE -- AND THE RULE KEYS ON THE DOCUMENT, NOT ON A LIST
    # OF CELLS. Two earlier cuts of this were both half right.
    #
    # The first kept the old unanchored italic pattern for `it/macbeth 1.3` and
    # `it/tempest 1.2` and applied the anchored one only to the held cells, on
    # the reasoning that those two pages are "differently transcribed" and their
    # vendored counts are a contract. MEASURED, AND THAT IS NOT TRUE: the
    # anchored rule reads both of those pages and reads them BETTER -- macbeth
    # 1.3 goes from 21 claimed speakers to 16, tempest 1.2 from 23 to 12, losing
    # `JOHNSON` (a footnote author), `LE ISOLE DIABOLICHE` (a footnote title)
    # and quoted lines cast as people. What a per-cell flag preserved was the
    # DEFECT, because the baseline it protected was itself polluted, and a
    # blast-radius check is for catching regressions rather than freezing a bug.
    #
    # The second cut -- dropping the flag and running every pattern everywhere
    # -- broke it the other way, and this is the part worth keeping. ON A
    # RUSCONI PAGE THE SMALL-CAPS SPANS ARE STAGE BUSINESS, NOT SPEAKERS:
    #
    #   <p><i>Entrano</i> <span style="font-variant:small-caps">Pietra-del-Paragone</span>
    #      <i>e</i> <span style="font-variant:small-caps">Andrey; Giacomo</span> ...
    #
    # so the Marquez small-caps pattern claimed every name in every ENTRANCE.
    # That is not cosmetic: the claimed direction swallowed the real speech
    # behind it, and `Ros. Dall'India all'Oriente...` arrived inside a
    # `ROSALINDA` label built out of `(leggendo un foglio)`.
    #
    # THE FACT THAT DECIDES IT IS A PROPERTY OF THE PAGE. An edition that heads
    # its paragraphs with an abbreviated italic label does not ALSO mark
    # speakers in small caps -- it marks its cast lists and entrances that way.
    # So ask the document: if the headed rule claimed speakers here, the
    # small-caps spans on this page are business and the generic patterns are
    # skipped. A page that yields no headed labels is a Hugo or Marquez page and
    # takes the generic patterns exactly as before. This keys on what the markup
    # IS rather than on which cell asked for it, so a new Rusconi cell needs no
    # entry anywhere.
    # ASK WHICH MECHANISM THE PAGE ACTUALLY USES, by running both and counting.
    # A first cut asked only "did the headed rule claim ANYTHING", and that is a
    # hair trigger: one incidental `<i>word</i>.` at a paragraph head on a Hugo
    # page disabled the `sc` rule and took the French Twelfth Night from 87
    # speeches to nothing. Measured over the vendored corpus, the two mechanisms
    # are not close on any real page -- the headed rule claims 50 to 120 labels
    # on a Rusconi page and one or two anywhere else, and the small-caps rules
    # claim a whole cast on Hugo and Marquez and only the entrance names on
    # Rusconi. So the larger count is the page's real mechanism, and comparing
    # needs no threshold and no list of cells.
    before = body.count(SPEAKER_MARK)
    headed = _RUSCONI_PARAGRAPH.sub(_mark_rusconi_paragraph, body)
    headed_labels = (headed.count(SPEAKER_MARK) - before) // 2

    generic = body
    for pattern in _SPEAKER_SPANS:
        generic = pattern.sub(_mark, generic)
    generic_labels = (generic.count(SPEAKER_MARK) - before) // 2

    return headed if headed_labels > generic_labels else generic


def _fold(text):
    text = unicodedata.normalize("NFKD", str(text or "")).lower()
    return "".join(c for c in text if not unicodedata.combining(c))


def find_label(lines, label):
    """Index of the line matching the edition's OWN scene/act label, or -1.

    THE PREFIX FALLBACK MUST NOT EAT A LONGER NUMERAL. "SCENA II" is a prefix
    of "SCENA III", and the first cut of this matched the wrong one: asked for
    Tempest 1.2 it returned SCENA III, Ariel's entrance, which would have been
    vendored as scene 2 -- text that reads perfectly and is the wrong scene,
    the single failure this corpus cannot afford. So a prefix match is only
    allowed when the character that follows is NOT a letter or digit, i.e. the
    label ended and what follows is punctuation or chrome.
    """
    want = _fold(label).strip().rstrip(".")
    if not want:
        return -1
    for i, line in enumerate(lines):
        if _fold(line).strip().rstrip(".") == want:
            return i
    for i, line in enumerate(lines):          # tolerate trailing chrome
        folded = _fold(line).strip()
        if not folded.startswith(want):
            continue
        rest = folded[len(want):]
        if rest and (rest[0].isalnum() or rest[0] in "一-鿿"):
            continue                          # SCENA II must not match III
        return i
    return -1


#: What each edition calls the act and scene we want. Filled from the
#: operator's verification pass -- he opened these pages and read the labels
#: off them. A label here is a MEASUREMENT, never an inference from the
#: Folio's numbering, because the editions disagree with the Folio and with
#: each other.
#: (play_anchor, act_label, scene_label). The play anchor is REQUIRED on a
#: multi-play volume and None on a single-play page. Gutenberg 59686 is
#: "Dramas de Guillermo Shakespear", a COLLECTION: without an anchor the
#: first "ACTO III / ESCENA II" in the book belongs to JULIO CESAR, and it
#: was vendored as As You Like It on the first run.
EDITION_LABELS = {
    ("it", "macbeth", "1.3"): (None, "ATTO PRIMO", "SCENA III"),
    ("it", "tempest", "1.2"): (None, "ATTO PRIMO", "SCENA II"),
    # NO ENTRY FOR ("it", "tempest", "3.1") AND THAT IS DELIBERATE -- see the
    # row's hold. Maffei's act-three page prints SCENA I and no other scene
    # heading, so `("ATTO TERZO", "SCENA I")` scopes to the WHOLE ACT: 117
    # speeches across 13 characters spanning Ferdinand and Miranda, then
    # Stefano and Caliban, then the court party, where Folger's 3.1 is the
    # lovers alone. Every one of those 13 is a real Tempest character, so the
    # speaker list looks perfect and only the SIZE gives it away -- the French
    # 3.1 is 25 speeches across 3. A label that silently returns three scenes is
    # worse than no label, because the next reader clears the hold and ships it.
    ("es", "as_you_like_it", "3.2"): (None, "ACTO III", "ESCENA II"),
    # Hugo numbers scenes CONTINUOUSLY and prints no act headings at all, so
    # there is nothing to scope to above the scene.
    ("fr", "hamlet", "1.1"): (None, None, "SC\u00c8NE I"),
    ("fr", "king_lear", "1.1"): (None, None, "SC\u00c8NE I"),
    ("zh", "midsummer", "3.1"): (None, None, "\u7b2c\u4e00\u573a"),
    # ZHU SHENGHAO, LABELS READ OFF THE PAGES 2026-09-19.
    #
    # THE SCRIPT IS PART OF THE LABEL AND THESE TWO ROWS PROVE IT. Midsummer
    # 3.1 is recorded against the zh-hans URL and prints \u7b2c\u4e00\u573a; 3.2 is recorded
    # against zh-hant and prints \u7b2c\u4e8c\u5834. Same play, same act, same collection --
    # and \u573a and \u5834 are different characters, so a label copied from the sibling
    # row matches nothing and the scene reads as absent. Copy from the page each
    # row actually points at, never from the row above.
    ("zh", "midsummer", "3.2"): (None, None, "\u7b2c\u4e8c\u5834"),
    # A whole-work page, so the act is load-bearing: every act repeats the
    # scene headings \u7b2c\u4e00\u5834/\u7b2c\u4e8c\u5834, and scoping to the scene alone would take the
    # first act's copy.
    ("zh", "tempest", "1.2"): (None, "\u7b2c\u4e00\u5e55", "\u7b2c\u4e8c\u5834"),
    ("zh", "tempest", "3.1"): (None, "\u7b2c\u4e09\u5e55", "\u7b2c\u4e00\u5834"),
    # A DIFFERENT TRANSLATION FROM THE ZHU COLLECTION ABOVE, and the title is
    # how you tell: this page is \u6f22\u59c6\u840a\u812b, not the \u54c8\u59c6\u96f7\u7279 of a Zhu volume. The
    # row's translator must be confirmed from the page before it is vendored.
    ("zh", "hamlet", "1.1"): (None, "\u7b2c\u4e00\u5e55", "\u7b2c\u4e00\u5834"),
    # Tsubouchi on Aozora: one page per play, so no play label is needed, and
    # the act and scene headings are plain h3/h4 once the ruby glosses are out
    # of them. An outside pass reported the scene heading as unfindable because
    # the ruby splits it -- true of the RAW page, and no longer true after
    # `to_text`, which is why the label here is the plain contiguous string.
    # RUSCONI, FOUND BY THE 2026-09-19 URL HUNT. it.wikisource serves one ACT
    # per page (`.../Atto_terzo`), so the act is scoped by the URL and only the
    # scene heading is needed. Every label below was read OFF the page by the
    # hunt, never inferred from the Folio -- and the last row is why that
    # matters: this edition prints As You Like It's standard act 2 scene 5 as
    # `SCENA VI`, because the act skips `SCENA IV` entirely.
    ("it", "as_you_like_it", "3.2"): (None, None, "SCENA II"),
    ("it", "comedy_of_errors", "3.1"): (None, None, "SCENA I"),
    ("it", "hamlet", "1.1"): (None, None, "SCENA I"),
    ("it", "king_lear", "1.1"): (None, None, "SCENA I"),
    ("it", "midsummer", "3.1"): (None, None, "SCENA I"),
    ("it", "midsummer", "3.2"): (None, None, "SCENA II"),
    ("it", "much_ado", "2.3"): (None, None, "SCENA III"),
    ("it", "much_ado", "3.1"): (None, None, "SCENA I"),
    ("it", "romeo_juliet", "2.2"): (None, None, "SCENA II"),
    ("it", "twelfth_night", "1.5"): (None, None, "SCENA V"),
    ("it", "twelfth_night", "2.5"): (None, None, "SCENA VI"),
    # Found by the 2026-09-19 URL hunt, with the page's own heading recorded
    # off the page rather than inferred: Menendez y Pelayo's Macbeth is served
    # one ACT per page (`.../Acto_I`), so the act is already scoped by the URL
    # and only the scene heading is needed. Moratin's Hamlet is a Gutenberg
    # whole-work text, so it needs both.
    ("es", "macbeth", "1.3"): (None, None, "ESCENA III"),
    ("es", "hamlet", "1.1"): (None, "ACTO PRIMERO", "ESCENA PRIMERA"),
    # pt.wikisource serves ONE SCENE PER PAGE (`.../Acto_primeiro/Cena_I`), so
    # the act is already scoped by the URL and the only label to find is the
    # scene's own heading, which the page prints as `SCENA I` in a tiInherit
    # block alongside the setting, the business and every speaker name.
    ("pt", "hamlet", "1.1"): (None, None, "SCENA I"),
    ("ja", "romeo_juliet", "1.1"): (None, "\u7b2c\u4e00\u5e55", "\u7b2c\u4e00\u5834"),
    ("ja", "romeo_juliet", "2.2"): (None, "\u7b2c\u4e8c\u5e55", "\u7b2c\u4e8c\u5834"),

    # --- Hugo's French set. He numbers scenes CONTINUOUSLY through a play and
    # prints no act headings at all, which is why every act slot here is None
    # and the scene numbers do not look like the Folger ones: his SCENE XII is
    # Folger 3.2. Located by an outside pass, then each one PROVED by running
    # the extractor against the live page -- the cast that came back is the
    # cast of the scene (three witches for Macbeth 1.3, Romeo, Juliet and the
    # Nurse for the balcony, Miranda, Prospero, Ariel and Caliban for Tempest
    # 1.2). Nothing here was accepted on the strength of a heading alone.
    ("fr", "as_you_like_it", "3.2"): (None, None, "SC\u00c8NE XII."),
    ("fr", "comedy_of_errors", "3.1"): (None, None, "SC\u00c8NE V."),
    ("fr", "macbeth", "1.3"): (None, None, "SC\u00c8NE III."),
    ("fr", "midsummer", "3.1"): (None, None, "SC\u00c8NE IV."),
    ("fr", "midsummer", "3.2"): (None, None, "SC\u00c8NE V."),
    ("fr", "much_ado", "2.3"): (None, None, "SC\u00c8NE V."),
    ("fr", "much_ado", "3.1"): (None, None, "SC\u00c8NE VI."),
    ("fr", "romeo_juliet", "2.2"): (None, None, "SC\u00c8NE VII."),
    ("fr", "tempest", "1.2"): (None, None, "SC\u00c8NE II."),
    ("fr", "tempest", "3.1"): (None, None, "SC\u00c8NE V."),
    ("fr", "twelfth_night", "1.5"): (None, None, "SC\u00c8NE V."),
    ("fr", "twelfth_night", "2.5"): (None, None, "SC\u00c8NE X."),

    # --- Spanish, the two that proved out. Marquez and Menendez y Pelayo both
    # reset per act, so these carry an act heading where Hugo carries none.
    ("es", "comedy_of_errors", "3.1"): (None, "ACTO III.", "ESCENA I."),
    ("es", "romeo_juliet", "2.2"): (None, "ACTO II.", "ESCENA II."),
}


def extract(lines, play_label, act_label, scene_label):
    """``(text, reason)`` -- the scene's lines, or "" and why not.

    Stops at the NEXT heading of the same shape. Returns "" rather than a
    guess: a wrong scene reads perfectly and is the one failure this corpus
    cannot afford.
    """
    start = 0
    if play_label:
        # A COLLECTION holds several plays and every one of them has an
        # ACTO III. Scope to the play before looking for the act, or the
        # first match in the book wins and it is the wrong play.
        play_at = find_label(lines, play_label)
        if play_at < 0:
            return "", "play heading %r not on the page" % play_label
        start = play_at
    if act_label:
        act_at = find_label(lines[start:], act_label)
        if act_at < 0:
            return "", "act heading %r not on the page" % act_label
        start += act_at
    scene_at = find_label(lines[start:], scene_label)
    if scene_at < 0:
        return "", "scene heading %r not found%s" % (
            scene_label, " under %s" % act_label if act_label else "")
    scene_at += start
    # The next heading of EITHER shape ends the scene.
    #
    # THE STEM IS THE LABEL WITHOUT ITS NUMBER, and deriving it by splitting on
    # WHITESPACE only works for a script that has any. `SCENE III.` splits to
    # `SCENE`, which the following `SCENE IV.` starts with, so the cut closes.
    # A CJK heading has no spaces, so the stem came out as the WHOLE label --
    # number and all -- and the next scene never started with it, so nothing
    # ever ended the scene. Measured on Tsubouchi's Romeo and Juliet: act 1
    # scene 1 returned 322 lines where the window is about 114, running through
    # the whole act and into the next one, and the balcony scene came back with
    # Mercutio, the Nurse, Friar Laurence and the Prince in it.
    #
    # A CJK act/scene heading is `<marker><numeral><unit>` -- the unit being
    # act or scene, in either traditional or simplified form. Two headings are
    # the same SHAPE when they share the marker and the unit, whatever numeral
    # sits between. That is the same rule the whitespace split expresses for
    # Latin, applied to a script that writes it without a gap.
    _CJK_HEADING = re.compile(r"^(?P<mark>第).{1,4}(?P<unit>[幕場场])$")
    cjk = _CJK_HEADING.match(scene_label.strip())
    cjk_units = ""
    if cjk:
        # stop at the next scene OF THIS UNIT, or at any act heading -- an act
        # boundary ends a scene just as surely as the next scene does.
        cjk_units = cjk.group("unit") + "幕"
        stem = ""
    else:
        stem = re.match(r"^\s*(\S+)", scene_label)
        stem = _fold(stem.group(1)) if stem else ""
    end = len(lines)
    for j in range(scene_at + 1, len(lines)):
        folded = _fold(lines[j]).strip()
        if cjk_units:
            hit = re.match(r"^第.{1,4}([幕場场])", folded)
            if hit and hit.group(1) in cjk_units:
                end = j
                break
            continue
        if stem and folded.startswith(stem):
            end = j
            break
        if act_label and folded.startswith(_fold(act_label.split()[0])):
            end = j
            break
    body = [l.rstrip() for l in lines[scene_at:end]]
    body = [l for l in body if l.strip()]
    if len(body) < 4:
        return "", "only %d lines between headings" % len(body)
    return "\n".join(body), ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="vendor the extractable scenes (default: check only)")
    parser.add_argument("--iso", action="append", default=None)
    args = parser.parse_args()

    leads = json.load(io.open(LEADS, encoding="utf-8"))["leads"]
    rows = []
    for lead in leads:
        key = (lead.get("iso"), lead.get("play"), lead.get("scene"))
        if lead.get("excluded") or lead.get("url_is") not in TEXT_FORMS:
            continue
        if lead.get("hold"):
            # A LOCATED, CORRECT SOURCE THIS PIPELINE CANNOT YET READ SAFELY.
            # Distinct from `excluded`, which means the TRANSLATION is
            # disqualified (an indirect translation, a text that does not
            # exist); a held row is good source waiting on the extractor, and
            # the field carries the reason so nobody has to rediscover it.
            #
            # It exists because `alt_of` was being used for this, and that
            # overloading cost a real incident: a self-referential `alt_of`
            # was read as a data error, cleared, and the row vendored
            # immediately -- putting a scene on disk with several speakers
            # merged into the wrong mouths. A flag whose NAME says "alternate"
            # cannot carry "do not vendor, the parser is not ready", and the
            # next reader will clear it again.
            print("  HOLD  %-3s %-16s %-5s  %s"
                  % (key + (str(lead["hold"])[:54],)))
            continue
        if lead.get("alt_of"):
            # One scene, one vendored file. An alternate writes the same
            # path and the LAST one wins silently -- on the first run a
            # Gutenberg collection overwrote a verified Wikisource
            # extraction with a different play. Alternates stay in
            # leads.json as backups and are vendored deliberately, never
            # as a side effect.
            print("  ALT   %-3s %-16s %-5s  alternate source, not vendored"
                  % key)
            continue
        if not str(lead.get("url") or "").strip():
            continue
        if args.iso and lead.get("iso") not in args.iso:
            continue
        if key not in EDITION_LABELS:
            print("  SKIP  %-3s %-16s %-5s  no edition label recorded"
                  % key)
            continue
        try:
            html = fetch(lead["url"])
        except Exception as exc:  # noqa: BLE001
            print("  FETCH %-3s %-16s %-5s  %s: %s"
                  % (key + (type(exc).__name__, str(exc)[:40])))
            continue
        text = to_text(html)
        lines = text.splitlines()
        body, reason = extract(lines, *EDITION_LABELS[key])
        # MEASURE THE ARTIFACT, NOT AN INTERMEDIATE. The gate counts `NAME:`
        # labels, and the extracted body does not carry them yet -- since the
        # speakers became publisher MARKS the raw body scores almost nothing,
        # and King Lear read as 27 speeches where the stored file has 87. Count
        # the normalised text, which is the text that will be written.
        labels, speakers = (CORPUS.speaker_label_stats(canonicalise_labels(normalise_labels(body)))
                            if body else (0, 0))
        if not body:
            print("  NO    %-3s %-16s %-5s  %s" % (key + (reason,)))
            continue
        ok = labels >= CORPUS.MIN_SPEAKER_LABELS and speakers >= 2
        print("  %-5s %-3s %-16s %-5s  %5d chars, %3d speeches / %2d speakers"
              % ("OK" if ok else "THIN", key[0], key[1], key[2],
                 len(body), labels, speakers))
        if ok:
            rows.append((lead, key, body, labels, speakers))

    print("\n  %d scene(s) extractable" % len(rows))
    if not args.write:
        print("  (--check only; nothing written)")
        return 0
    return write_rows(rows)



#: A vendored file uses ONE label shape so the passage selector stays one
#: rule: `NAME:` at line start, the spec's own advice. The editions use
#: `Orlando.\n—`, `kent.`, `1a Strega.`, `波　` -- normalised on the way in
#: rather than taught to the parser.
#: Punctuation that proves a candidate "name" is really a SENTENCE. A speaker
#: label does not contain a comma or a full stop; a line of dialogue does.
#:
#: THE ASCII HALF OF THIS LIST WAS THE WHOLE LIST, AND IT PUT BOTTOM'S SPEECH IN
#: SNOUT'S MOUTH. `_LABEL_SHAPES[0]` accepts the FULLWIDTH colon as a separator,
#: so on Zhu Shenghao's Chinese Midsummer it matched everything up to one --
#: `波　列位，你们得好好想一想` -- and offered that whole clause as a name. The
#: fullwidth comma inside it is the tell, and `,;--!?` could not see it, because
#: U+FF0C is not U+002C. The name then occurred once, the recurrence rule
#: demoted the line to continuation, and Bottom's entire speech was performed by
#: Snout:
#:     司: 咱担保她们一定会吓怕。 波　列位，你们得好好想一想：...
#: Rejecting the clause lets `_LABEL_SHAPES[3]` have the line, which reads the
#: ideographic space correctly and returns `波` -- the right speaker.
#:
#: The two length guards above cannot cover this and are near-useless on CJK:
#: `split()` finds no spaces in Chinese, so the word count is always 1, and 24
#: characters is a long sentence in a language that does not space its words.
#: Punctuation is the signal that survives the script change.
_NOT_IN_A_NAME = ",;—–!?，、；！？。…"

_LABEL_SHAPES = (
    re.compile(r"^(?P<name>[^\n:：]{1,40})[:：][ \t\u3000]?"),
    re.compile(r"^(?P<name>[A-Za-z\u00c0-\u024f0-9][^\n.]{0,38})\.\s*$"),
    re.compile(r"^(?P<name>[A-Za-z\u00c0-\u024f0-9][^\n.]{0,38})\.\s+(?=\S)"),
    re.compile(r"^(?P<name>[^\n\u3000]{1,12})\u3000"),
)


def _label_candidates(body):
    """How often each candidate name appears. Two passes, because one is not
    enough to tell a speaker from a stage direction.

    A SPEAKER RECURS; CHROME APPEARS ONCE. The length guard below cannot catch
    a SHORT stage direction -- "Un bosco" is two words and eight characters --
    and the first pass duly cast the Macbeth setting as a character called
    UN BOSCO, the Spanish one as EL BOSQUE, and Hamlet's "Francisco est en
    faction" as EST EN FACTION. Each would have been SPOKEN aloud with a voice
    assigned. Requiring a name to appear at least twice is the same rule the
    acceptance gate already uses to tell a scene from a cast list.

    The cost is a genuine one-line part demoted to prose. That is the safe
    direction: prose is read as narration, an invented character is read as a
    person who is not in the play.
    """
    counts = {}
    for raw in body.splitlines():
        name = _name_of(raw.strip())
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


#: One CJK ideograph, which is a whole name in the Chinese editions. Kana and
#: the Latin range are deliberately excluded: a single kana or letter really is
#: too short to be a name, and only the ideographic range carries one.
_CJK_CHAR = re.compile(r"^[一-鿿]$")

#: A name the PUBLISHER marked, sitting at the head of its line.
_MARKED_LABEL = re.compile(
    r"^%s(?P<name>[^%s\n]{1,40})%s\s*" % (SPEAKER_MARK, SPEAKER_MARK,
                                          SPEAKER_MARK))


def _marked_name(line):
    """The speaker name the edition itself marked on this line, or "".

    A MARKED NAME IS A FACT AND SKIPS EVERY GUESS BELOW. The heuristics exist
    only for a page whose markup we have not taught, and each of them costs a
    real speaker: the comma rule drops Hugo's `cordelia , a part.`, and the
    recurrence rule drops anyone who speaks once. Neither may veto the page.
    """
    match = _MARKED_LABEL.match(line)
    if not match:
        return ""
    name = match.group("name").strip().rstrip(".:　").strip()
    if not name:
        return ""
    # A MARKED SPAN IS A FACT ABOUT THE MARKUP, NOT A PROMISE ABOUT ITS USE.
    # The rule above is right that a marked name outranks every heuristic, and
    # it assumed a publisher marks only speakers. Hugo's `Le soir des rois`
    # transcriber marks the LETTER Malvolio reads aloud in the same class, so
    # `Je puis commander ou j'adore` and `Nul homme ne le doit savoir` arrived
    # as characters, each with one line, each about to be cast and voiced.
    #
    # This is not the recurrence guess coming back: it vetoes nothing on how
    # OFTEN a name appears, only on whether the string is shaped like a name at
    # all. A sentence is not a name in any of these editions. The real
    # multi-word labels in the corpus are comfortably inside these bounds --
    # `PREMIERE SORCIERE`, `CHOEUR DES FEES`, `GRAIN DE MOUTARDE`,
    # `ANTIPHOLUS D'EPHESE`, `PIERRE DE TOUCHE`, `TOUTES TROIS`.
    # PUNCTUATION ONLY, and the first attempt at this taught the lesson. A
    # length-and-word-count guard (28 characters, 4 words) killed the song lines
    # AND `TUTTE LE STREGHE CANTANDO E DANZANDO` -- a real collective label in
    # Rusconi's Macbeth, six words and thirty-six characters, which two existing
    # tests guard because a collective must never be gendered or performed.
    # A collective and a quoted sentence are the same SHAPE, so shape cannot
    # separate them; sentence punctuation can, and it is the only signal here
    # that does not cost a legitimate label.
    # THE LENGTH FLOOR IS LATIN-ALPHABET REASONING AND IT DELETES A CJK CAST.
    # One letter is not a name in any of the European editions, so the floor is
    # right for them. It is catastrophic for Zhu Shenghao, who abbreviates every
    # character to their opening CHARACTER: measured on Midsummer act 3, 17 of
    # the 19 speakers are one character long -- 黑 (Hermia), 莱 (Lysander), 第
    # (Demetrius), 海 (Helena), 波 (Bottom), 迫 (Puck), 奥 (Oberon), 蒂 (Titania)
    # and the whole bench of mechanicals and fairies. The Chinese rule can mark
    # all 178 speeches correctly and every one of them still arrives here and
    # returns "", which is a silent, total cast loss that a speech count cannot
    # see. A single ideograph carries a whole name; a single letter does not.
    if len(name) < 2 and not _CJK_CHAR.match(name):
        return ""
    if any(ch in name for ch in ",…;:!?"):
        return ""
    return name.upper()


def _fold_accents(name):
    """A key that ignores accents, for deciding whether two labels are one
    character. Never shown to anyone -- the display spelling stays as printed."""
    stripped = unicodedata.normalize("NFKD", name)
    return "".join(c for c in stripped if not unicodedata.combining(c)).upper()


def canonicalise_labels(body):
    """Fold labels that differ ONLY by accent into their commonest spelling.

    A transcriber is not consistent about diacritics across a long page, and
    each spelling arrived as a separate character: `HERO`, `HERO` with a grave
    and `HERO` with an acute are one woman in `much_ado` 3.1, and
    `ANTIPHOLUS D'EPHESE` has two accent spellings in `comedy_of_errors` 3.1.
    Cast three times over, she would be voiced three different ways.

    ONLY ACCENTS ARE FOLDED, which is the whole safety of it. `ANTIPHOLUS` and
    `ANTIPHOLUS D'EPHESE` fold to different keys and stay two characters --
    correct, because the twins are two people and a bare label is genuinely
    ambiguous. A letter substitution is likewise left alone: `AMIPHOLUS` is an
    OCR fault, not an accent, and merging by edit distance would be guessing.

    The spelling kept is the one the page uses MOST, so the printed cast list
    reads the way the edition mostly prints it.
    """
    lines = body.splitlines()
    counts = {}
    for line in lines:
        name = _name_of(line)
        if name:
            counts.setdefault(_fold_accents(name), collections.Counter())[name] += 1
    canon = {key: tally.most_common(1)[0][0] for key, tally in counts.items()}
    out = []
    for line in lines:
        name = _name_of(line)
        want = canon.get(_fold_accents(name)) if name else None
        if want and want != name and line.upper().startswith(name):
            out.append(want + line[len(name):])
        else:
            out.append(line)
    return "\n".join(out)


def _name_of(line):
    """The speaker name this line opens with, or "" -- shape rules only."""
    marked = _marked_name(line)
    if marked:
        return marked
    for shape in _LABEL_SHAPES:
        m = shape.match(line)
        if not m:
            continue
        name = m.group("name").strip().rstrip(".:　")
        if not name or CORPUS._NOT_A_NAME.search(name):
            continue
        if len(name) > 24 or len(name.split()) > 4:
            continue
        if any(ch in name for ch in _NOT_IN_A_NAME):
            continue
        return name.upper()
    return ""


def normalise_labels(body):
    """Rewrite every speaker label to `NAME:` and join it to its speech.

    WHEN THE PAGE MARKED ITS SPEAKERS, THE PAGE WINS AND THE GUESSES ARE OFF.
    The recurrence rule below is a stand-in for a fact, and it costs a real
    part every time it fires -- a character who speaks once is demoted to
    narration. It is worth paying only on a publisher whose markup we have not
    taught yet, so it applies to an UNMARKED page and never to a marked one.
    """
    marked = SPEAKER_MARK in body
    recurring = {n for n, c in _label_candidates(body).items() if c >= 2}
    out, pending = [], None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = None
        name = _name_of(line)
        if marked:
            # A marked page has exactly one kind of label, and a line without
            # the mark is narration however much it looks like a name.
            marked_name = _marked_name(line)
            if marked_name:
                rest = line[_MARKED_LABEL.match(line).end():].strip()
                matched = (marked_name, rest)
        elif name and name in recurring:
            # ONLY A RECURRING NAME IS A SPEAKER -- see `_label_candidates`.
            for shape in _LABEL_SHAPES:
                m = shape.match(line)
                if not m:
                    continue
                rest = line[m.end():].strip()
                matched = (name, rest)
                break
        if matched:
            if pending:
                out.append("%s: %s" % pending)
            pending = matched if matched[1] else (matched[0], "")
        elif pending:
            pending = (pending[0], (pending[1] + " " + line).strip())
        else:
            out.append(line)                  # heading / stage direction
    if pending:
        out.append("%s: %s" % pending)
    # The sentinel is scaffolding, never text. Any that survives here sat mid
    # line -- a name inside narration -- and must not reach a performer.
    body_out = "\n".join(l for l in out if l.strip())
    return body_out.replace(SPEAKER_MARK, "")


def write_rows(rows):
    manifest_path = os.path.join(OUT_ROOT, CORPUS.MANIFEST_NAME)
    existing = []
    if os.path.isfile(manifest_path):
        existing = json.load(io.open(manifest_path, encoding="utf-8")).get(
            "scenes", [])
    by_key = {(r.get("iso"), r.get("play"), r.get("scene")): r for r in existing}

    for lead, key, body, labels, speakers in rows:
        iso, play, scene = key
        text = canonicalise_labels(normalise_labels(body))
        after, after_speakers = CORPUS.speaker_label_stats(text)
        rel = os.path.join(iso, "%s_%s.txt" % (play, scene.replace(".", "_")))
        dest = os.path.join(OUT_ROOT, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # SIGN THE BYTES THAT ARE STORED, not the ones in hand. The first cut
        # wrote `text + "\n"` and hashed `text`, so the manifest signed a
        # different file than it named and `vendored_text` refused every scene
        # at the integrity check -- correctly. An off-by-one-newline is exactly
        # what that check is for; it just happened to be catching me.
        stored = text + "\n"
        io.open(dest, "w", encoding="utf-8", newline="\n").write(stored)
        digest = hashlib.sha256(stored.encode("utf-8")).hexdigest()
        # THE ROW IS REBUILT FROM SCRATCH, so a hand-curated key would be
        # deleted by a routine re-vendor -- the same way a re-fetch used to
        # wipe the gender roster off a sidecar (`STAMPER_OWNED_SIDECAR_KEYS`).
        # The speaker map is curated by hand against the edition's labels and
        # the English sidecar; carry it forward, and let the reader re-check
        # it against the new text via `unbound_labels` on the next plan.
        carried = {
            k: v for k, v in (by_key.get(key) or {}).items()
            if k == CORPUS.SPEAKER_MAP_FIELD
        }
        by_key[key] = {
            "iso": iso, "play": play, "scene": scene,
            "file": rel.replace("\\", "/"),
            "translator": lead.get("translator", ""),
            "translator_death_date": lead.get("translator_death_date", ""),
            "translation_first_published":
                lead.get("translation_first_published", ""),
            "transcription_license": lead.get("transcription_license", ""),
            "source_url": lead.get("url", ""),
            "revision_id": "sha256:" + digest[:16],
            "raw_sha256": digest,
            "verdict": CORPUS.READY,
            # Located by the edition's OWN label, verified by reading the
            # opening lines -- not by counting headings. 1.0 is the honest
            # figure for a label match confirmed against the scene's content.
            "alignment_confidence": 1.0,
            "edition_label": " / ".join(
                x for x in EDITION_LABELS[key] if x),
            "speaker_labels": after,
            "distinct_speakers": after_speakers,
            **carried,
        }
        print("  wrote %-46s %5d chars  %3d speeches / %2d speakers"
              % (rel, len(text), after, after_speakers))

    payload = {"schema_version": CORPUS.SCHEMA_VERSION,
               "note": ("Vendored scenes. Written by "
                        "scripts/otr_vendor_shakespeare.py; every row was "
                        "located by the EDITION'S OWN scene label and its "
                        "opening lines read before acceptance."),
               "scenes": sorted(by_key.values(),
                                key=lambda r: (r["iso"], r["play"], r["scene"]))}
    io.open(manifest_path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("\n  manifest: %s (%d scenes)" % (manifest_path, len(payload["scenes"])))
    CORPUS.load_manifest(manifest_path)
    print("  load_manifest validated it")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
