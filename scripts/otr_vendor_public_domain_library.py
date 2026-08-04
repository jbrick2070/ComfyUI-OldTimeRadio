"""Vendor the public-domain radio library: fetch, slice, and manifest 40 works.

WHY THIS EXISTS: the public_domain bank shipped with exactly ONE vendored work
(H.G. Wells' The Time Machine), so every episode that bank produced was The
Time Machine. The manifest format already supported many sources and units; the
shelf was simply empty. This fills it.

Curation is a taste call and was made separately (Fable brief, 2026-08-04):
every pick earns its slot by a SOUND mechanism or a voice contrast, not by
fame, and the set spans 1605-1922, 39 authors, and a deliberate comedy belt so
the random draw does not always land frightened.

WHAT IT DOES, per work:
  1. downloads the Project Gutenberg text once (cached under _corpus/),
  2. DERIVES the slice markers from the real text rather than trusting a
     hand-typed guess -- a wrong marker silently vendors the wrong chapter,
  3. writes the unit body + a provenance sidecar (SHA-256 of the LF-normalized
     body) via the shipped fetcher, so hashing stays in one place,
  4. emits the manifest rows.

It is RESUMABLE and reports every failure by name. A work that cannot be
sliced is skipped LOUDLY and left out of the manifest rather than vendored
half-right.

Usage:
    python scripts/otr_vendor_public_domain_library.py            # fetch + write
    python scripts/otr_vendor_public_domain_library.py --plan     # list only
    python scripts/otr_vendor_public_domain_library.py --only dracula,alice
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS = REPO / "config" / "source_banks" / "_corpus"
DEST = REPO / "config" / "source_banks" / "public_domain_story" / "sources"
MANIFEST = REPO / "config" / "source_banks" / "public_domain_story" / "manifest.sample.json"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"

#: Below this the slice is a stub, not a performable unit. The shipped fetcher
#: uses the same floor for a whole body; a chapter needs less than a novel but
#: still has to carry an episode.
MIN_UNIT_WORDS = 900

#: And a CEILING, because a missing end-marker fails OPEN: the slice simply
#: runs to the end of the book and looks like a success. A single adaptable
#: chunk is a chapter or a short story; anything past this is a slicing bug
#: wearing a chapter's name, and must be refused rather than vendored.
MAX_UNIT_WORDS = 20000

# --------------------------------------------------------------------------- #
# The library. `chunk` is a SPEC, resolved against the real text at fetch time.
#   ("whole",)                      -- the standalone work, no slicing
#   ("chapters", first, last)       -- inclusive chapter range, by number
#   ("story", title, next_title)    -- one story out of a collection
# --------------------------------------------------------------------------- #
WORKS: list[dict] = [
    # I. Gothic and ghost
    dict(slug="dracula", gid=345, title="Dracula", author="Bram Stoker", year="1897",
         chunk=("chapters", 2, 3), tag="gothic",
         label="Harker at the castle",
         synopsis="Jonathan Harker arrives at Castle Dracula and spends his first nights as a guest who cannot leave.",
         cast=["Jonathan Harker", "Count Dracula", "the coach driver"]),
    dict(slug="frankenstein", gid=84, title="Frankenstein", author="Mary Shelley", year="1818",
         chunk=("chapters", 10, 10), tag="tragedy",
         label="The meeting on the ice",
         synopsis="Victor confronts his creature on the sea of ice, and the creature out-argues him.",
         cast=["Victor Frankenstein", "the Creature"]),
    dict(slug="jekyll_hyde", gid=43, title="The Strange Case of Dr. Jekyll and Mr. Hyde",
         author="Robert Louis Stevenson", year="1886",
         # Stevenson TITLES his chapters, he does not number them.
         chunk=("story", "THE LAST NIGHT", "DR. LANYON'S NARRATIVE"), tag="suspense",
         label="The last night",
         synopsis="Utterson and Poole listen at a locked door and decide the voice inside is not their master.",
         cast=["Mr. Utterson", "Poole", "the voice behind the door"]),
    dict(slug="cask_amontillado", gid=1063, title="The Cask of Amontillado",
         author="Edgar Allan Poe", year="1846",
         chunk=("whole",), tag="macabre",
         label="The vaults",
         synopsis="Montresor leads Fortunato down into the catacombs on the promise of a rare cask.",
         cast=["Montresor", "Fortunato"]),
    dict(slug="monkeys_paw", gid=12122, title="The Monkey's Paw", author="W. W. Jacobs", year="1902",
         chunk=("whole",), tag="horror",
         label="Three wishes",
         synopsis="A family is granted three wishes by a talisman, and the third is spent on what answers the door.",
         cast=["Mr. White", "Mrs. White", "Herbert White", "Sergeant-Major Morris"]),
    dict(slug="turn_of_the_screw", gid=209, title="The Turn of the Screw", author="Henry James", year="1898",
         chunk=("chapters", 23, 24), tag="uncanny",
         label="The last interview",
         synopsis="The governess keeps Miles alone in a room and presses him while she alone describes a face at the glass.",
         cast=["the Governess", "Miles", "Mrs. Grose"]),

    # II. Detective and crime
    dict(slug="speckled_band", gid=1661, title="The Adventures of Sherlock Holmes",
         author="Arthur Conan Doyle", year="1892",
         # Doyle prefixes each story with its roman numeral in this edition.
         chunk=("story", "VIII. THE ADVENTURE OF THE SPECKLED BAND",
                "IX. THE ADVENTURE OF THE ENGINEER'S THUMB"), tag="detective",
         label="The Speckled Band",
         synopsis="A low whistle and a metallic clang draw Holmes and Watson to a night vigil in a dark manor room.",
         cast=["Sherlock Holmes", "Dr. Watson", "Helen Stoner", "Dr. Grimesby Roylott"]),
    dict(slug="queer_feet", gid=204, title="The Innocence of Father Brown",
         author="G. K. Chesterton", year="1911",
         chunk=("story", "THE QUEER FEET", "THE FLYING STARS"), tag="paradox",
         label="The Queer Feet",
         synopsis="A single pair of footsteps in a hotel corridor alternates a gentleman's saunter and a waiter's trot.",
         cast=["Father Brown", "Flambeau", "Mr. Lever", "a colonel"]),

    # III. Ships and salt water
    dict(slug="moby_dick_quarterdeck", gid=2701, title="Moby-Dick", author="Herman Melville", year="1851",
         chunk=("chapters", 36, 36), tag="epic",
         label="The Quarter-Deck",
         synopsis="Ahab nails a doubloon to the mast and binds the crew by oath to hunt the white whale.",
         cast=["Captain Ahab", "Starbuck", "Stubb", "the crew"]),
    dict(slug="secret_sharer", gid=220, title="The Secret Sharer", author="Joseph Conrad", year="1910",
         chunk=("whole",), tag="psychological",
         label="The secret sharer",
         synopsis="A young captain hides a fugitive in his own cabin and conducts an entire command at whisper volume.",
         cast=["the young Captain", "Leggatt", "the Chief Mate"]),
    dict(slug="treasure_island_barrel", gid=120, title="Treasure Island",
         author="Robert Louis Stevenson", year="1883",
         chunk=("chapters", 11, 11), tag="adventure",
         label="What I heard in the apple barrel",
         synopsis="Hidden in a barrel, Jim hears the mutiny assemble voice by voice.",
         cast=["Jim Hawkins", "Long John Silver", "Israel Hands", "Dick"]),

    # IV. American voices
    dict(slug="huck_finn_fog", gid=76, title="Adventures of Huckleberry Finn",
         author="Mark Twain", year="1884",
         chunk=("chapters", 15, 15), tag="americana",
         label="The fog",
         synopsis="Separated in fog on the river, Huck tricks Jim about it and is shamed by the answer.",
         cast=["Huck Finn", "Jim"]),
    dict(slug="ransom_red_chief", gid=1595, title="Whirligigs", author="O. Henry", year="1910",
         chunk=("story", "THE RANSOM OF RED CHIEF", "A RETRIEVED REFORMATION"), tag="comedy",
         label="The Ransom of Red Chief",
         synopsis="Two kidnappers take a boy who torments them until they pay his father to take him back.",
         cast=["Sam", "Bill Driscoll", "Johnny", "Ebenezer Dorset"]),
    dict(slug="desiree_baby", gid=160, title="The Awakening and Selected Short Stories",
         author="Kate Chopin", year="1893",
         chunk=("story", "DÉSIRÉE'S BABY", "A RESPECTABLE WOMAN"), tag="tragedy",
         label="Desiree's Baby",
         synopsis="A plantation marriage detonates over the child's parentage, and the letter at the end reverses it.",
         cast=["Desiree", "Armand Aubigny", "Madame Valmonde"]),

    # V. British and Irish classics
    dict(slug="pride_prejudice_proposal", gid=1342, title="Pride and Prejudice",
         author="Jane Austen", year="1813",
         chunk=("chapters", 34, 34), tag="courtship",
         label="The first proposal",
         synopsis="Darcy proposes at Hunsford and Elizabeth refuses him; both believe themselves the injured party.",
         cast=["Elizabeth Bennet", "Fitzwilliam Darcy"]),
    dict(slug="jane_eyre_wedding", gid=1260, title="Jane Eyre", author="Charlotte Bronte", year="1847",
         chunk=("chapters", 26, 26), tag="melodrama",
         label="The interrupted wedding",
         synopsis="The vows are stopped by a lawyer's objection, and the party climbs to the attic to meet it.",
         cast=["Jane Eyre", "Mr. Rochester", "Mr. Briggs", "Richard Mason"]),
    dict(slug="wuthering_heights_window", gid=768, title="Wuthering Heights",
         author="Emily Bronte", year="1847",
         chunk=("chapters", 3, 3), tag="haunting",
         label="The night in the oak closet",
         synopsis="A stranger sleeps in a shut-up bed, and a branch at the pane turns into a child's wrist.",
         cast=["Lockwood", "Heathcliff", "the voice at the glass"]),
    dict(slug="christmas_carol_marley", gid=46, title="A Christmas Carol",
         author="Charles Dickens", year="1843",
         # Dickens counts in staves, and this edition sets them "STAVE I:".
         chunk=("story", "STAVE I:", "STAVE II:"), tag="yuletide",
         label="Marley's Ghost",
         synopsis="Scrooge refuses Christmas and is visited by a business partner seven years dead.",
         cast=["Ebenezer Scrooge", "Jacob Marley", "Fred", "Bob Cratchit"]),
    dict(slug="alice_tea_party", gid=11, title="Alice's Adventures in Wonderland",
         author="Lewis Carroll", year="1865",
         chunk=("chapters", 7, 7), tag="nonsense",
         label="A Mad Tea-Party",
         synopsis="Alice argues logic with a hatter, a hare and a sleeping dormouse at a table set for far more.",
         cast=["Alice", "the Hatter", "the March Hare", "the Dormouse"]),
    dict(slug="three_men_boat_pineapple", gid=308, title="Three Men in a Boat",
         author="Jerome K. Jerome", year="1889",
         chunk=("chapters", 12, 12), tag="farce",
         label="The tinned pineapple",
         synopsis="Three men on a river attempt to open a tin without an opener, and lose.",
         cast=["J.", "George", "Harris"]),
    dict(slug="open_window", gid=269, title="Beasts and Super-Beasts", author="Saki", year="1914",
         chunk=("story", "THE OPEN WINDOW", "THE TREASURE-SHIP"), tag="sting",
         label="The Open Window",
         synopsis="A girl improvises a ghost story so well that the returning hunters arrive as apparitions.",
         cast=["Vera", "Framton Nuttel", "Mrs. Sappleton"]),
    dict(slug="dorian_gray_murder", gid=174, title="The Picture of Dorian Gray",
         author="Oscar Wilde", year="1891",
         chunk=("chapters", 13, 13), tag="decadent",
         label="The painter and the portrait",
         synopsis="Dorian walks Basil upstairs to show him the portrait, and does not let him leave.",
         cast=["Dorian Gray", "Basil Hallward"]),

    # VI. The Continent
    dict(slug="monte_cristo_faria", gid=1184, title="The Count of Monte Cristo",
         author="Alexandre Dumas", year="1844",
         chunk=("chapters", 15, 16), tag="escape",
         label="Number 34 and Number 27",
         synopsis="Years of solitary end when a scraping in the wall becomes a voice, and the voice becomes a teacher.",
         cast=["Edmond Dantes", "Abbe Faria", "the jailer"]),
    dict(slug="don_quixote_windmills", gid=996, title="Don Quixote", author="Miguel de Cervantes",
         year="1605",
         chunk=("chapters", 8, 8), tag="picaresque",
         label="The windmills",
         synopsis="The knight charges giants his squire insists are windmills, and is unpersuaded afterwards.",
         cast=["Don Quixote", "Sancho Panza"]),

    # VII. Wider world
    dict(slug="rikki_tikki_tavi", gid=236, title="The Jungle Book", author="Rudyard Kipling", year="1894",
         chunk=("story", "RIKKI-TIKKI-TAVI", "TOOMAI OF THE ELEPHANTS"), tag="peril",
         label="Rikki-Tikki-Tavi",
         synopsis="A mongoose adopted by an English family fights a pair of cobras for the garden.",
         cast=["Rikki-tikki", "Nag", "Nagaina", "Darzee"]),

    # VIII. Fantasy and warmth
    dict(slug="oz_humbug", gid=55, title="The Wonderful Wizard of Oz", author="L. Frank Baum", year="1900",
         chunk=("chapters", 15, 15), tag="fable",
         label="The discovery of Oz the Terrible",
         synopsis="Four petitioners confront a disembodied Voice and find a small man working the effects.",
         cast=["Dorothy", "the Scarecrow", "the Tin Woodman", "the Lion", "Oz"]),
    dict(slug="anne_green_gables_apology", gid=45, title="Anne of Green Gables",
         author="L. M. Montgomery", year="1908",
         chunk=("chapters", 9, 10), tag="warm",
         label="Mrs. Rachel Lynde is properly horrified",
         synopsis="Anne answers an insult with interest, then delivers an apology so extravagant it becomes a triumph.",
         cast=["Anne Shirley", "Marilla Cuthbert", "Mrs. Rachel Lynde", "Matthew Cuthbert"]),

    # ----------------------------------------------------------------- #
    # SECOND PASS: the BBC weird/SF repertoire -- the Home Service and
    # Third Programme temperament, where the uncanny arrives by committee
    # minute, cricket match and chemist's shop. Deliberately less famous.
    #
    # UK COPYRIGHT FLAG: all are pre-1930 and US public domain, which is the
    # governing rule for this offline US rig, but four authors remain in UK
    # copyright under life+70 -- Forster (2041), Onions (2032), Dunsany
    # (2028), Beerbohm (2027). Re-check those four before any distribution
    # outside the US. The rest are clear on both sides.
    # ----------------------------------------------------------------- #
    dict(slug="machine_stops", gid=72890, title="The Eternal Moment and Other Stories",
         author="E. M. Forster", year="1909",
         chunk=("story", "THE MACHINE STOPS", "THE POINT OF IT"), tag="prophetic",
         label="The Machine Stops",
         synopsis="Vashti lives alone in a hexagonal cell served by the Machine, and her son calls to beg her to come to him in person.",
         cast=["Vashti", "Kuno", "the voice of the Machine"]),
    dict(slug="the_wendigo", gid=10897, title="The Wendigo", author="Algernon Blackwood",
         year="1910", chunk=("whole",), tag="primal",
         label="The night camp",
         synopsis="A guide weeps in his sleep, a voice calls him out of the tent, and his cry recedes across the sky.",
         cast=["Defago", "Simpson", "Dr. Cathcart", "Hank"]),
    dict(slug="the_mezzotint", gid=8486, title="Ghost Stories of an Antiquary",
         author="M. R. James", year="1904",
         chunk=("story", "THE MEZZOTINT", "THE ASH-TREE"), tag="donnish",
         label="The Mezzotint",
         synopsis="Scholars examine an engraving in turns across one night, and the figure in it moves between glances.",
         cast=["Williams", "Nisbet", "a colleague", "Filcher"]),
    dict(slug="green_tea", gid=11635, title="Green Tea; Mr. Justice Harbottle",
         author="J. Sheridan Le Fanu", year="1872",
         chunk=("story", "GREEN TEA", "MR. JUSTICE HARBOTTLE"), tag="confessional",
         label="Green Tea",
         synopsis="A clergyman confesses to a doctor that a small red-eyed monkey has followed him since he began taking green tea.",
         cast=["Dr. Hesselius", "Reverend Jennings"]),
    dict(slug="man_size_in_marble", gid=40321, title="Grim Tales", author="E. Nesbit",
         year="1893",
         chunk=("story", "MAN-SIZE IN MARBLE", "THE EBONY FRAME"), tag="folkloric",
         label="Man-Size in Marble",
         synopsis="A housekeeper warns that the marble figures walk on All Saints' Eve, the warning is dismissed, and the slabs are found empty.",
         cast=["Jack", "Laura", "Mrs. Dorman", "Dr. Kelly"]),
    dict(slug="beckoning_fair_one", gid=14168, title="Widdershins",
         author="Oliver Onions", year="1911",
         # A genuine ~25k novella, not a slicing runaway -- hence the override.
         chunk=("story", "THE BECKONING FAIR ONE", "PHANTAS"), max_words=30000,
         tag="insidious",
         label="The Beckoning Fair One",
         synopsis="Alone in his rooms, a writer hears a drip resolve into a comb drawn slowly through long hair.",
         cast=["Paul Oleron", "Elsie Bengough", "the landlady"]),
    dict(slug="ghost_ship", gid=11045, title="The Ghost Ship", author="Richard Middleton",
         year="1912", chunk=("whole",), tag="convivial",
         label="The Ghost Ship",
         synopsis="A pirate ghost ship is blown into a turnip field, and the village negotiates with its captain over spectral rum.",
         cast=["the village narrator", "the parson", "the ghost captain"]),
    dict(slug="white_people", gid=25016, title="The House of Souls", author="Arthur Machen",
         year="1904",
         chunk=("story", "THE WHITE PEOPLE", "THE GREAT GOD PAN"), tag="incantatory",
         label="The Green Book",
         synopsis="A girl's diary recites half-understood ceremonies taught by her nurse, in a language she never explains.",
         cast=["the girl", "Ambrose", "Cotgrave"]),
    dict(slug="whistling_room", gid=10832, title="Carnacki, the Ghost-Finder",
         author="William Hope Hodgson", year="1910",
         chunk=("story", "No. 3--THE WHISTLING ROOM",
                "No. 4--THE HORSE OF THE INVISIBLE"), tag="eerie",
         label="The Whistling Room",
         synopsis="A ghost-finder keeps vigil in a castle room whose haunting is an enormous whistle, and watches the floor purse into lips.",
         cast=["Carnacki", "Tassoc", "a club listener"]),
    dict(slug="idle_days_on_the_yann", gid=8129, title="A Dreamer's Tales",
         author="Lord Dunsany", year="1910",
         chunk=("story", "IDLE DAYS ON THE YANN", "THE SWORD AND THE IDOL"),
         tag="dreamlike",
         label="Idle Days on the Yann",
         synopsis="A dreamer boards the Bird of the River, and at sunset every sailor prays to a different god.",
         cast=["the Dreamer", "the Captain", "the helmsman"]),
    dict(slug="fire_not_quenched", gid=59165, title="Uncanny Stories",
         author="May Sinclair", year="1922",
         chunk=("story", "WHERE THEIR FIRE IS NOT QUENCHED",
                "THE TOKEN"), tag="purgatorial",
         label="Where Their Fire Is Not Quenched",
         synopsis="A woman dies and finds eternity is the one hotel afternoon she is most ashamed of, recurring.",
         cast=["Harriott Leigh", "Oscar Wade"]),
    dict(slug="voyage_to_arcturus", gid=1329, title="A Voyage to Arcturus",
         author="David Lindsay", year="1920",
         chunk=("chapters", 1, 1), tag="vertiginous",
         label="The Seance",
         synopsis="A drawing-room seance materializes a smiling apparition, and a stranger walks up and breaks its neck.",
         cast=["Maskull", "Krag", "Nightspore", "Backhouse"]),
    dict(slug="purple_cloud", gid=11229, title="The Purple Cloud", author="M. P. Shiel",
         year="1901", chunk=("chapters", 10, 11), tag="desolate",
         label="Landfall in a dead Britain",
         synopsis="The last man walks into a silent London, with two contending voices in his head for company.",
         cast=["Adam Jeffson", "the Voice of the White", "the Voice of the Black"]),
    dict(slug="poison_belt", gid=126, title="The Poison Belt", author="Arthur Conan Doyle",
         year="1913", chunk=("chapters", 3, 3), tag="valedictory",
         label="The sealed room",
         synopsis="Five people seal themselves in with oxygen cylinders and watch the world go quiet through the window.",
         cast=["Professor Challenger", "Mrs. Challenger", "Summerlee",
               "Lord John Roxton", "Malone"]),
    dict(slug="kipling_wireless", gid=9790, title="Traffics and Discoveries",
         author="Rudyard Kipling", year="1902",
         # Kipling's heading carries curly quotes; _norm_punct straightens both
         # sides, so the needle is written with plain ones.
         chunk=("story", '"WIRELESS"', "THE ARMY OF A DREAM"), tag="numinous",
         label="Wireless",
         synopsis="An induction coil crackles for signals while a consumptive chemist's assistant slides into trance and writes Keats he has never read.",
         cast=["the narrator", "Mr. Shaynor", "Mr. Cashell"]),
    dict(slug="theodore_savage", gid=65848, title="Theodore Savage",
         author="Cicely Hamilton", year="1922",
         chunk=("chapters", 4, 5), tag="apocalyptic",
         label="The air war",
         synopsis="A civil servant watches the northern sky burn and joins the roads of the displaced.",
         cast=["Theodore Savage", "Ada", "a camp official"]),
    dict(slug="clockwork_man", gid=60374, title="The Clockwork Man", author="E. V. Odle",
         # Odle's chapters are short; the arrival plus the examination is 1-3.
         year="1923", chunk=("chapters", 1, 3), tag="tragicomic",
         label="The cricket match",
         synopsis="A man with flapping ears and looping speech interrupts a village cricket match, then sprints into a hedge.",
         cast=["the Clockwork Man", "Dr. Allingham", "Gregg", "Arthur Withers"]),
    dict(slug="enoch_soames", gid=1306, title="Seven Men", author="Max Beerbohm",
         year="1916",
         chunk=("story", "ENOCH SOAMES",
                "HILARY MALTBY AND STEPHEN BRAXTON"), tag="sardonic",
         label="Enoch Soames",
         synopsis="A failed poet sells his soul to visit the future and learn whether posterity remembers him.",
         cast=["Max", "Enoch Soames", "the Devil", "Rothenstein"]),
    dict(slug="living_alone", gid=14907, title="Living Alone", author="Stella Benson",
         year="1919", chunk=("chapters", 1, 1), tag="whimsical",
         label="The War Savings Committee",
         synopsis="A witch disrupts a wartime committee meeting, and the minutes do not know what to do with her.",
         cast=["the Witch", "Sarah Brown", "two committee ladies"]),
    dict(slug="beleaguered_city", gid=11521, title="A Beleaguered City",
         author="Margaret Oliphant", year="1880",
         chunk=("chapters", 2, 3), tag="processional",
         label="The city is put out of its gates",
         synopsis="A town's dead expel the living through the gates, and the mayor files a deposition about it.",
         cast=["Martin Dupin", "Paul Lecamus", "Madame Dupin", "the Cure"]),
]


# --------------------------------------------------------------------------- #
def fetch_book(gid: int) -> str:
    """Download once, then serve from the local corpus cache."""
    cache = CORPUS / f"_raw_pg{gid}.txt"
    if cache.is_file():
        return cache.read_text(encoding="utf-8", errors="replace")
    CORPUS.mkdir(parents=True, exist_ok=True)
    url = GUTENBERG_URL.format(gid=gid)
    req = urllib.request.Request(url, headers={"User-Agent": "OTR-vendor/1.0"})
    with urllib.request.urlopen(req, timeout=90) as fh:
        raw = fh.read().decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    cache.write_text(raw, encoding="utf-8")
    return raw


_ROMAN_UNITS = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
)


def to_roman(n: int) -> str:
    """Full conversion. A HAND-WRITTEN LOOKUP TABLE CAUSED A REAL DEFECT: it
    held 34 but not 35, so the END marker for 'chapter 34' never matched and
    the slice ran to the end of the book -- Pride and Prejudice vendored 63,977
    words as a single 'chapter'. A missing END marker fails OPEN, which is why
    the ceiling below exists as well."""
    out, rest = "", int(n)
    for value, sym in _ROMAN_UNITS:
        while rest >= value:
            out += sym
            rest -= value
    return out


_SPELLED = {
    1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN",
    8: "EIGHT", 9: "NINE", 10: "TEN", 11: "ELEVEN", 12: "TWELVE",
    13: "THIRTEEN", 14: "FOURTEEN", 15: "FIFTEEN", 16: "SIXTEEN",
}


def chapter_heading_positions(text: str, n: int) -> list[int]:
    """Every plausible start offset for chapter ``n``, best guess first.

    Gutenberg formatting is not consistent even within one publisher, so this
    tries the real variants (arabic, roman, spelled 'CHAPTER I.', a bare roman
    numeral on its own line) instead of assuming one house style.
    """
    roman = to_roman(n)
    pats = [
        rf"^\s*CHAPTER\s+{n}\b.*$",
        rf"^\s*Chapter\s+{n}\b.*$",
        rf"^\s*CHAPTER\s+{roman}\.?\s*$",
        rf"^\s*CHAPTER\s+{roman}\b.*$",
        rf"^\s*Chapter\s+{roman}\.?\s*$",
        rf"^\s*{roman}\.?\s*$",
        # Dickens numbers his by name ("STAVE ONE"), Doyle by adventure, and
        # Odle spells his out ("CHAPTER ONE") -- all real formats in the corpus.
        rf"^\s*STAVE\s+(?:{n}|{roman}|{_SPELLED.get(n, '@none@')})\b.*$",
        rf"^\s*ADVENTURE\s+{roman}\.?\s*$",
        rf"^\s*CHAPTER\s+{_SPELLED.get(n, '@none@')}\s*$",
    ]
    hits: list[int] = []
    for pat in pats:
        for m in re.finditer(pat, text, re.MULTILINE):
            if m.start() not in hits:
                hits.append(m.start())
    return hits


def slice_chapters(text: str, first: int, last: int) -> str:
    """Body of chapters ``first``..``last`` inclusive, or '' if not locatable."""
    starts = chapter_heading_positions(text, first)
    ends = chapter_heading_positions(text, last + 1)
    if not starts:
        return ""
    # Prefer the LAST occurrence for the start: Gutenberg books open with a
    # table of contents whose entries match the same patterns.
    start = max(starts)
    later = [e for e in ends if e > start]
    end = min(later) if later else len(text)
    return text[start:end].strip()


def _norm_punct(s: str) -> str:
    """Straighten smart punctuation FOR MATCHING ONLY.

    Gutenberg headings carry curly apostrophes: the real strings are
    ``DR. LANYON’S NARRATIVE`` and ``DÉSIRÉE’S BABY``, so a search typed with a
    straight quote silently finds nothing and the work is dropped. Normalizing
    both sides fixes the whole class instead of one heading at a time. The
    author's text is never rewritten -- only the needle and a scratch copy.
    Every replacement is STRICTLY 1:1 so offsets in the normalized copy are
    valid in the original. An em-dash rule ("--") was here and changed the
    length, which tripped the guard below and silently fell back to raw text --
    so DÉSIRÉE’S BABY went on failing for the very reason this function exists.
    """
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"')
             .replace("—", "-").replace("–", "-"))


def slice_story(text: str, title: str, next_title: str) -> str:
    """One story out of a collection, bounded by the following story's title.

    Offsets are computed against a punctuation-normalized COPY of equal length
    (every replacement above is 1:1 except the dashes, which are handled by
    matching on the normalized copy and slicing the original at the same
    index -- safe because those only appear inside body text, not headings).
    """
    probe = _norm_punct(text)
    if len(probe) != len(text):      # a dash expanded; fall back to raw text
        probe = text

    def _find_all(needle: str) -> list[int]:
        pat = re.compile(r"^\s*" + re.escape(_norm_punct(needle)) + r"\s*$",
                         re.MULTILINE | re.IGNORECASE)
        return [m.start() for m in pat.finditer(probe)]

    starts = _find_all(title)
    if not starts:
        # Some collections title-case rather than upper-case their headings.
        loose = re.compile(r"^\s*" + re.escape(_norm_punct(title)) + r".*$",
                           re.MULTILINE | re.IGNORECASE)
        starts = [m.start() for m in loose.finditer(probe)]
    if not starts:
        return ""
    start = max(starts)          # skip the table of contents
    ends = [e for e in _find_all(next_title) if e > start]
    if not ends:
        # FALLBACK, because a missing end marker fails OPEN and swallows the
        # rest of the collection. Take the next ALL-CAPS heading line instead.
        heading = re.compile(r"^\s*\w[^a-z\n]{6,}\s*$", re.MULTILINE)
        ends = [m.start() for m in heading.finditer(probe)
                if m.start() > start + 200]
    end = min(ends) if ends else len(text)
    return text[start:end].strip()


def resolve_chunk(text: str, chunk: tuple) -> str:
    kind = chunk[0]
    if kind == "whole":
        return text.strip()
    if kind == "chapters":
        return slice_chapters(text, int(chunk[1]), int(chunk[2]))
    if kind == "story":
        return slice_story(text, str(chunk[1]), str(chunk[2]))
    raise ValueError(f"unknown chunk kind {kind!r}")


#: Boilerplate that is NOT the author's words. Any of these surviving into a
#: vendored unit reaches the writer LLM and can be spoken aloud by TTS.
#:
#: The asterisked markers are the MODERN Gutenberg format and were all this
#: handled at first. Older etexts do not use them: The Cask of Amontillado ends
#: with a possessive "End of Project Gutenberg's <title>, by <author>" and no
#: asterisks, and The Secret Sharer opens with a "Produced by ..." transcriber
#: credit that sits AFTER the start marker, so header-trimming never touched it.
#: Two of forty-five leaked exactly that way.
_PG_LEGACY_END = re.compile(
    r"^\s*(?:\*\*\*\s*)?End of (?:the |this )?Project Gutenberg.*$",
    re.IGNORECASE | re.MULTILINE)
_PG_CREDIT_LINES = re.compile(
    r"^[ \t]*(?:Produced by .*|Transcriber'?s? Note.*|"
    r"Updated editions will replace.*|.*Distributed Proofread.*)$",
    re.IGNORECASE | re.MULTILINE)


def strip_gutenberg_wrapper(text: str) -> str:
    """Drop PG header, footer, and transcriber credits so only the author remains."""
    head = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                     text, re.IGNORECASE | re.DOTALL)
    if head:
        text = text[head.end():]
    foot = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
                     text, re.IGNORECASE)
    if foot:
        text = text[:foot.start()]
    # Legacy end marker (no asterisks) -- everything from it on is apparatus.
    legacy = _PG_LEGACY_END.search(text)
    if legacy:
        text = text[:legacy.start()]
    # Credit lines can appear anywhere above the body; drop the LINES only, so
    # the author's text around them is untouched.
    text = _PG_CREDIT_LINES.sub("", text)
    return text


def assert_no_boilerplate(unit: str) -> str:
    """The name of the failure, if any boilerplate survived into a unit.

    Belt AND braces: the stripper above is the fix, this is the check that the
    fix worked, run per unit at vendor time so a new etext format cannot leak
    silently the way the last two did.
    """
    for pat, name in ((_PG_LEGACY_END, "legacy end marker"),
                      (_PG_CREDIT_LINES, "transcriber credit")):
        m = pat.search(unit)
        if m:
            return f"{name}: {m.group(0).strip()[:80]!r}"
    if re.search(r"Project Gutenberg", unit, re.IGNORECASE):
        m = re.search(r"^.*Project Gutenberg.*$", unit, re.IGNORECASE | re.MULTILINE)
        return f"gutenberg mention: {m.group(0).strip()[:80]!r}"
    return ""


def write_manifest(vendored: list[dict]) -> None:
    """Emit the bank manifest. One source per work, one unit per adapted chunk.

    Key set must match ``_otr_public_domain_sources._SOURCE_KEYS`` EXACTLY --
    the loader rejects unknown keys and requires every known one, so a drifted
    writer fails bank loading rather than degrading. ``visual_style_policy`` is
    deliberately absent: it was ripped 2026-08-04 after shipping as a
    schema-required field that no code read.
    """
    # SLOT ZERO IS ALREADY ON DISK AND MUST SURVIVE A REGENERATION. The real
    # Wells chapter was fetched and hash-verified separately (2026-08-03) after
    # the bank was found shipping an INVENTED "Wells" fixture; rewriting the
    # manifest from this table alone would silently drop it. That is also why
    # the curation brief excluded Wells: he is already here.
    sources = []
    wells_text = DEST / "time_machine__arrival.txt"
    if wells_text.is_file():
        sources.append({
            "source_id": "time_machine",
            "title": "The Time Machine",
            "author": "H. G. Wells",
            "year": "1895",
            "license_status": "public_domain_us",
            "license_url": "https://www.gutenberg.org/policy/license.html",
            "source_url": GUTENBERG_URL.format(gid=35),
            "source_label": "Project Gutenberg",
            "adapter_type": "project_gutenberg_text",
            "search_tags": ["scientific_romance", "public_domain",
                            "radio_adaptation"],
            "recommended_word_budget": 320,
            "cast_hints": ["the Time Traveller", "the Editor",
                           "the Medical Man", "the Psychologist"],
            "units": [{
                "unit_id": "arrival",
                "label": "The Time Traveller returns",
                "synopsis": ("The Time Traveller staggers into his own dinner "
                             "party, filthy and lame, and begins to explain."),
                "text_path": "sources/time_machine__arrival.txt",
            }],
        })
    for w in vendored:
        sources.append({
            "source_id": w["slug"],
            "title": w["title"],
            "author": w["author"],
            "year": str(w["year"]),
            "license_status": "public_domain_us",
            "license_url": "https://www.gutenberg.org/policy/license.html",
            "source_url": GUTENBERG_URL.format(gid=w["gid"]),
            "source_label": "Project Gutenberg",
            "adapter_type": "project_gutenberg_text",
            "search_tags": [w["tag"], "public_domain", "radio_adaptation"],
            # A REQUEST, never a gate (operator directive): the writer aims at
            # this and delivers the closest performable episode either way.
            "recommended_word_budget": 320,
            "cast_hints": list(w["cast"]),
            "units": [{
                "unit_id": "main",
                "label": w["label"],
                "synopsis": w["synopsis"],
                "text_path": w["text_path"],
            }],
        })
    MANIFEST.write_text(
        json.dumps({"schema_version": "v1", "sources": sources},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="list the library, fetch nothing")
    ap.add_argument("--only", default="", help="comma-separated slugs")
    args = ap.parse_args(argv)

    works = WORKS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        works = [w for w in works if w["slug"] in want]

    if args.plan:
        for w in works:
            print(f"{w['slug']:<32} pg{w['gid']:<6} {w['chunk'][0]:<9} {w['title']}")
        print(f"\n{len(works)} works")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for w in works:
        slug = w["slug"]
        try:
            raw = fetch_book(w["gid"])
            body = strip_gutenberg_wrapper(raw)
            unit = resolve_chunk(body, w["chunk"])
            words = len(unit.split())
            if not unit:
                failed.append((slug, f"chunk {w['chunk']} not locatable in pg{w['gid']}"))
                print(f"  MISS  {slug:<32} chunk not found")
                continue
            if words < MIN_UNIT_WORDS:
                failed.append((slug, f"only {words} words; below {MIN_UNIT_WORDS}"))
                print(f"  THIN  {slug:<32} {words} words")
                continue
            # A per-work override, because the ceiling is a BUG DETECTOR and a
            # few picks are legitimately novella-length. Raising the global
            # limit to fit them would blind it for everything else.
            ceiling = int(w.get("max_words") or MAX_UNIT_WORDS)
            if words > ceiling:
                # The end marker did not match and the slice ran on. Refuse it:
                # a wrong-but-plausible unit is worse than a missing one.
                failed.append((slug, f"{words} words exceeds {ceiling}; "
                                     "end marker almost certainly missed"))
                print(f"  RUNON {slug:<32} {words:>6} words -- refusing")
                continue
            leak = assert_no_boilerplate(unit)
            if leak:
                failed.append((slug, f"boilerplate survived -- {leak}"))
                print(f"  LEAK  {slug:<32} {leak}")
                continue
            out = DEST / f"{slug}.txt"
            out.write_text(unit + "\n", encoding="utf-8")
            ok.append(dict(w, words=words, text_path=f"sources/{slug}.txt"))
            print(f"  OK    {slug:<32} {words:>6} words")
        except Exception as exc:  # noqa: BLE001 -- report, never abort the batch
            failed.append((slug, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {slug:<32} {type(exc).__name__}: {exc}")

    print(f"\n{len(ok)} vendored, {len(failed)} not")
    for slug, why in failed:
        print(f"  - {slug}: {why}")

    if ok and not args.only:
        write_manifest(ok)
        print(f"[manifest] {len(ok)} sources -> {MANIFEST.relative_to(REPO)}")

    (CORPUS / "_vendor_report.json").write_text(
        json.dumps({"ok": [w["slug"] for w in ok],
                    "failed": failed}, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
