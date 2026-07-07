"""nodes/_otr_style_catalog.py -- curated radio-drama STYLE catalog (2026-06-24).

Each style is a repeatable RADIO GRAMMAR, not just a genre label:
  * sound_world   -- the concrete audio palette (feeds dialogue mood + the
                     visualizer / LTX render prompt).
  * story_engine  -- the conflict shape (who wants what, where the pressure
                     comes from) -- a NON-"console standoff" structure.
  * ending_mode   -- how the episode lands. THE anti-trope lever: most endings
                     are revelation / reversal / quiet / unresolved -- NOT a
                     doomsday kill-switch -- so the climax stops collapsing into
                     "blow everything up."

Operator-authored 100-style set (2026-06-24). The picker SELECTS the best-fit
style for the article from this catalog; the writer injects the chosen style's
grammar into the macro / beat / line prompts (esp. ending_mode -> the climax
beat) so a divergent premise is actually composed differently.

Pure data + small pure helpers. No torch, no LLM, no I/O at import.
UTF-8 no BOM, SFW.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Each entry: slug (snake_case id), label (human), sound_world, story_engine,
# ending_mode. tags help the picker bias away from the over-used emergency
# register ("emergency" tag = the high-tension cluster; the picker should not
# pick those by default unless the article truly calls for it).
STYLE_CATALOG: List[Dict[str, str]] = [
    # --- 1-10: the original ten, upgraded ---
    {"slug": "locked_room_suspense", "label": "locked-room suspense",
     "sound_world": "close breathing, footsteps on bare floor, door handles, a ticking clock, muffled voices beyond the walls",
     "story_engine": "people trapped together, one hidden threat among them, suspicion rising with no easy exit",
     "ending_mode": "a revelation, a reversal of who to trust, or one unresolved final sound", "tags": "suspense"},
    {"slug": "detective_case_file_reconstruction", "label": "detective case-file reconstruction",
     "sound_world": "shuffled papers, a match struck, rain on a window, a chair creak, a recorder click",
     "story_engine": "a detective walks back through a case aloud; each remembered detail reframes the last",
     "ending_mode": "the quiet click of the real answer, or a doubt that the file was wrong all along", "tags": "mystery"},
    {"slug": "pulp_serial_cliffhanger", "label": "pulp serial cliffhanger",
     "sound_world": "brassy stingers, wind, ropes straining, a hero's grunt, a narrator's hush",
     "story_engine": "a larger-than-life hero races one escalating peril to a brink",
     "ending_mode": "a cliffhanger freeze or a last-second rescue that costs something", "tags": "adventure"},
    {"slug": "mission_control_procedural", "label": "mission-control procedural",
     "sound_world": "console chatter, headset hiss, steady beeps, the hum of a quiet room under pressure",
     "story_engine": "a calm team works a problem by the book as the margins shrink",
     "ending_mode": "a problem solved by competence, or a humane call made against the checklist", "tags": "emergency"},
    {"slug": "deep_space_distress_transmission", "label": "deep-space distress transmission",
     "sound_world": "static wash, a lonely carrier tone, suit breathing, a far-off structural groan",
     "story_engine": "one voice reaches across distance for help that may not come in time",
     "ending_mode": "contact made, a message left for whoever finds it, or a fade into silence", "tags": "emergency"},
    {"slug": "noir_interrogation_chamber", "label": "noir interrogation chamber",
     "sound_world": "a buzzing lamp, a dripping tap, a cigarette inhale, two chairs, a long pause",
     "story_engine": "two people circle a single truth; each admission shifts the power",
     "ending_mode": "a confession, a turned table, or a lie everyone agrees to keep", "tags": "mystery"},
    {"slug": "small_town_uncanny_mystery", "label": "small-town uncanny mystery",
     "sound_world": "crickets, a screen door, distant dogs, a porch swing, a too-quiet street",
     "story_engine": "an ordinary town hides one wrong thing everyone half-knows",
     "ending_mode": "a folk revelation, a neighbor's complicity exposed, or a thing left politely unspoken", "tags": "folklore"},
    {"slug": "newsroom_emergency_bulletin", "label": "newsroom emergency bulletin",
     "sound_world": "teletype clatter, phones, a producer's count-in, paper torn from a wire",
     "story_engine": "a newsroom decides what the public gets to know as a story breaks under them",
     "ending_mode": "a hard editorial choice, a correction that costs trust, or the next bulletin's hush", "tags": "emergency"},
    {"slug": "haunted_broadcast_signal", "label": "haunted broadcast signal",
     "sound_world": "tuning whine, a half-heard voice under the music, tape warble, dead-air pops",
     "story_engine": "a signal carries something it shouldn't; the host can't stop listening",
     "ending_mode": "the signal explained away, the host changed by it, or the dial left tuned to nothing", "tags": "uncanny"},
    {"slug": "laboratory_containment_breach", "label": "laboratory containment breach",
     "sound_world": "airlock seals, drip of coolant, a soft alarm, gloves snapping, a fan cycling",
     "story_engine": "a small careful team weighs what to save against what to let go",
     "ending_mode": "a sacrifice made quietly, a cover-up begun, or a sealed door and a long breath", "tags": "emergency"},

    # --- 11-30: confined / pressure / institutional ---
    {"slug": "isolated_lighthouse_confession", "label": "isolated lighthouse confession",
     "sound_world": "surf, a foghorn, a lamp's slow turn, wind through stone, a kettle",
     "story_engine": "two keepers, a long night, an old secret one of them has carried",
     "ending_mode": "a confession given to the dark, a forgiveness, or the light turning on as nothing is said", "tags": "drama"},
    {"slug": "courtroom_testimony_thriller", "label": "courtroom testimony thriller",
     "sound_world": "a gavel, murmuring gallery, a chair scrape, the stenotype's tick",
     "story_engine": "testimony reframes the truth witness by witness; the verdict hangs on one word",
     "ending_mode": "a verdict, a recantation, or a juror's private doubt as court adjourns", "tags": "drama"},
    {"slug": "missing_person_audio_diary", "label": "missing-person audio diary",
     "sound_world": "a phone mic's rasp, footsteps, traffic, a recorded breath, a beep",
     "story_engine": "someone narrates their own search until the recordings stop matching the story",
     "ending_mode": "a found answer, a chilling gap in the tape, or a last entry that explains nothing", "tags": "mystery"},
    {"slug": "submarine_pressure_crisis", "label": "submarine pressure crisis",
     "sound_world": "hull pings, dripping condensation, sonar, low engine throb, held breath",
     "story_engine": "a crew rations air and trust as the boat sinks past its limits",
     "ending_mode": "a surface reached, a hatch closed on someone, or a steady voice counting down to calm", "tags": "emergency"},
    {"slug": "train_car_murder_mystery", "label": "train-car murder mystery",
     "sound_world": "rails clacking, a whistle, sliding doors, teacups, a conductor's call",
     "story_engine": "a closed set of passengers, one body, and a station coming fast",
     "ending_mode": "the culprit named, an alibi shattered, or the train pulling in with the question open", "tags": "mystery"},
    {"slug": "stormbound_inn_conspiracy", "label": "stormbound inn conspiracy",
     "sound_world": "rain on shutters, a hearth, glasses set down, thunder, a stair creak",
     "story_engine": "strangers wait out a storm and slowly realize they were gathered on purpose",
     "ending_mode": "the plan revealed, an alliance flipped, or dawn breaking on an unfinished scheme", "tags": "mystery"},
    {"slug": "expedition_camp_radio_log", "label": "expedition camp radio log",
     "sound_world": "canvas snapping, a generator, wind, a radio's squelch, boots on gravel",
     "story_engine": "a field team logs nightly reports as morale and supplies thin",
     "ending_mode": "a rescue raised, a leader's hard order, or a final log left for the next crew", "tags": "survival"},
    {"slug": "wartime_code_room_drama", "label": "wartime code-room drama",
     "sound_world": "typewriter keys, valve hum, a clock, pencils, a hushed corridor",
     "story_engine": "code-breakers hold a secret that could save or doom the people they love",
     "ending_mode": "a message decoded too late, a silence kept to win, or a private grief filed away", "tags": "drama"},
    {"slug": "emergency_dispatch_night_shift", "label": "emergency dispatch night shift",
     "sound_world": "phone lines, keyboard taps, a coffee pour, radio crackle, a quiet room",
     "story_engine": "a dispatcher talks strangers through their worst nights, one call at a time",
     "ending_mode": "a life talked back from the edge, a call that ends badly, or the next ring", "tags": "drama"},
    {"slug": "psychiatric_ward_interview", "label": "psychiatric ward interview",
     "sound_world": "a buzzing light, a pen on a pad, distant doors, a chair, breathing",
     "story_engine": "two people across a table, one holding a story that keeps changing shape",
     "ending_mode": "a breakthrough, a manipulation revealed, or a diagnosis that explains nothing", "tags": "drama"},
    {"slug": "lost_expedition_recordings", "label": "lost expedition recordings",
     "sound_world": "wind-scoured tape, crampons, labored breath, a recorder's hum",
     "story_engine": "recovered tapes replay an expedition's last days as the cause closes in",
     "ending_mode": "the fate understood, a hero re-seen, or a recording that cuts off mid-word", "tags": "survival"},
    {"slug": "arctic_outpost_survival", "label": "arctic outpost survival",
     "sound_world": "howling wind, ice cracking, a stove, a wind-up radio, layered breath",
     "story_engine": "an iced-in crew stretches fuel and patience against the dark",
     "ending_mode": "the thaw reached, a rationing choice, or a warm silence as the wind drops", "tags": "survival"},
    {"slug": "desert_station_signal_hunt", "label": "desert station signal hunt",
     "sound_world": "dry wind, dish servos, a fan, sand on metal, a faint blip",
     "story_engine": "a lonely listening post chases a signal that may be nothing or everything",
     "ending_mode": "the source found, the listener undone by hope, or the signal gone with dawn", "tags": "mystery"},
    {"slug": "mountain_rescue_countdown", "label": "mountain rescue",
     "sound_world": "rotor wash, carabiners, wind, a radio, snow underfoot",
     "story_engine": "a rescue team weighs the climber they can reach against the weather closing",
     "ending_mode": "a save earned, a turn-back that haunts, or a body brought home in silence", "tags": "emergency"},
    {"slug": "plague_town_quarantine_drama", "label": "plague-town quarantine drama",
     "sound_world": "church bells, empty streets, a cart, coughing behind doors, a knock",
     "story_engine": "a sealed town governs itself as fear tests every neighborly tie",
     "ending_mode": "a small mercy, a betrayal of the rule, or the gates opening on a changed town", "tags": "drama"},
    {"slug": "museum_after_hours_haunting", "label": "museum after-hours haunting",
     "sound_world": "echoing halls, a guard's radio, glass cases, footsteps, a far drip",
     "story_engine": "a night guard notices the exhibits remembering things they shouldn't",
     "ending_mode": "a history set right, a guard who joins the collection, or a case left ajar", "tags": "uncanny"},
    {"slug": "factory_floor_sabotage_mystery", "label": "factory-floor sabotage mystery",
     "sound_world": "stamping presses, conveyor hum, a whistle, lockers, a foreman's shout",
     "story_engine": "a line worker hunts who's wrecking the shift, and why it's personal",
     "ending_mode": "the saboteur unmasked, a union's hard choice, or the line restarting on a lie", "tags": "mystery"},
    {"slug": "hotel_switchboard_thriller", "label": "hotel switchboard thriller",
     "sound_world": "ringing lines, plugs seating, a lobby murmur, an elevator bell",
     "story_engine": "an operator overhears one call too many and has to act on what she knows",
     "ending_mode": "a guest saved, an operator implicated, or a line that goes dead", "tags": "suspense"},
    {"slug": "space_station_systems_failure", "label": "space-station systems failure",
     "sound_world": "warning chirps, fans spinning down, velcro, a calm computer voice",
     "story_engine": "a small crew triages failing systems and the truth of who caused it",
     "ending_mode": "stability restored, a confession in the dark, or a drift toward a quiet repair", "tags": "emergency"},
    {"slug": "underwater_research_panic", "label": "underwater research station",
     "sound_world": "pump hum, water against glass, comms hiss, bubbles, a slow leak",
     "story_engine": "researchers far below weigh a discovery against the cost of surfacing with it",
     "ending_mode": "a sample sacrificed, a wonder kept secret, or a slow ascent into doubt", "tags": "drama"},

    # --- 31-50: crime / noir / folk ---
    {"slug": "private_eye_street_noir", "label": "private-eye street-noir case",
     "sound_world": "rain, neon buzz, a car door, a bar's low jazz, heels on wet pavement",
     "story_engine": "a tired investigator pulls one thread until the whole city leans on it",
     "ending_mode": "a case closed at a price, a femme reversal, or the rain washing it all flat", "tags": "noir"},
    {"slug": "corrupt_city_political_wiretap", "label": "corrupt-city wiretap",
     "sound_world": "reel-to-reel hiss, a clock, traffic below, a match, headphones",
     "story_engine": "listeners on a tap learn the rot reaches the people who ordered it",
     "ending_mode": "a tape that buries someone, a deal struck to bury it, or the recorder left running", "tags": "noir"},
    {"slug": "blackmail_phone_call_thriller", "label": "blackmail phone-call thriller",
     "sound_world": "a dial tone, a payphone's chime, breath held, coins, street noise",
     "story_engine": "two voices negotiate a secret's price across a single tightening call",
     "ending_mode": "a turn of the tables, a price paid, or a receiver set down on nothing settled", "tags": "suspense"},
    {"slug": "ransom_negotiation_countdown", "label": "ransom negotiation",
     "sound_world": "a speakerphone, pacing, a clock, paper, a tense room's breathing",
     "story_engine": "a negotiator buys time and trust with someone who holds a life",
     "ending_mode": "a safe return earned by words, a bluff called, or a line that stays open and silent", "tags": "suspense"},
    {"slug": "witness_protection_confession", "label": "witness-protection confession",
     "sound_world": "a motel AC, a TV's low drone, a knock, a zipper, a car idling",
     "story_engine": "a protected witness decides whether the truth is worth the new life it ends",
     "ending_mode": "testimony given, an identity surrendered, or a goodbye to a name", "tags": "drama"},
    {"slug": "fugitive_safehouse_drama", "label": "fugitive safehouse drama",
     "sound_world": "boards creaking, a radio low, a kettle, distant sirens fading, whispers",
     "story_engine": "people in hiding test how far loyalty stretches when the net tightens",
     "ending_mode": "a getaway bought by sacrifice, a betrayal, or a dawn move to the next house", "tags": "suspense"},
    {"slug": "border_crossing_identity_mystery", "label": "border-crossing identity mystery",
     "sound_world": "stamps, idling engines, a barrier arm, papers, a guard's questions",
     "story_engine": "at a checkpoint, who someone claims to be unravels under one more question",
     "ending_mode": "a crossing made on a lie, a true name revealed, or the gate staying down", "tags": "mystery"},
    {"slug": "prison_cell_psychological_duel", "label": "prison-cell psychological duel",
     "sound_world": "bars, distant clangs, a tap dripping, footsteps on a tier, low voices",
     "story_engine": "two inmates trade stories until one realizes the other has been steering him",
     "ending_mode": "a manipulation exposed, an unlikely respect, or lights-out on an unfinished game", "tags": "drama"},
    {"slug": "interrogation_under_blackout", "label": "interrogation under blackout",
     "sound_world": "a generator dying, total dark, a chair, breathing, a far explosion",
     "story_engine": "with the power gone, an interrogation becomes two people and the truth",
     "ending_mode": "a confession in the dark, a captor swayed, or the lights returning to an empty chair", "tags": "drama"},
    {"slug": "informant_dead_drop_suspense", "label": "informant dead-drop suspense",
     "sound_world": "a park's wind, pigeons, a bench, footsteps approaching, a paper folded",
     "story_engine": "an informant works a drop knowing one of the watchers is turned",
     "ending_mode": "the handoff made, a trap sprung, or a package left for no one who comes", "tags": "suspense"},
    {"slug": "rural_folklore_investigation", "label": "rural folklore investigation",
     "sound_world": "a creaking barn, wind in corn, a fiddle far off, gravel, an old voice",
     "story_engine": "an outsider chases a local legend until the town's belief becomes the danger",
     "ending_mode": "the legend grounded in fact, the outsider converted, or a story that protects itself", "tags": "folklore"},
    {"slug": "carnival_after_midnight_uncanny", "label": "carnival-after-midnight uncanny",
     "sound_world": "a wheezing calliope, a generator, tent canvas, a barker's echo, gravel",
     "story_engine": "after closing, a carnival's wonders ask a price the visitor didn't read",
     "ending_mode": "a bargain refused, a wonder kept, or the lights cutting out on a half-made deal", "tags": "uncanny"},
    {"slug": "roadside_diner_mystery", "label": "roadside diner mystery",
     "sound_world": "a coffee pour, a bell over the door, a jukebox, rain, cutlery",
     "story_engine": "strangers at a counter at 3 a.m. each carry a piece of the same night",
     "ending_mode": "the night's truth assembled, a quiet exit, or the check left under a cup", "tags": "mystery"},
    {"slug": "abandoned_motel_transmission", "label": "abandoned motel transmission",
     "sound_world": "a neon hum, wind through a broken sign, a TV snow-hiss, a door banging",
     "story_engine": "a traveler finds a motel still broadcasting to guests who never left",
     "ending_mode": "the loop broken, the traveler checked in for good, or the vacancy sign flickering on", "tags": "uncanny"},
    {"slug": "ghost_town_field_recording", "label": "ghost-town field recording",
     "sound_world": "dry wind, a swinging shutter, boots on boardwalk, a recorder's hum",
     "story_engine": "a documentarian records an empty town that keeps answering back",
     "ending_mode": "a history recovered, a recordist who stays, or a final take of only wind", "tags": "uncanny"},
    {"slug": "cursed_object_inventory", "label": "cursed-object inventory",
     "sound_world": "tissue paper, a ledger pen, a clock, a drawer sliding, a small bell",
     "story_engine": "an archivist catalogs donations and learns one item is cataloging them back",
     "ending_mode": "the object contained, a curse passed on, or an entry that writes its own line", "tags": "uncanny"},
    {"slug": "family_attic_revelation", "label": "family-attic revelation",
     "sound_world": "creaking joists, boxes dragged, dust, an old reel threaded, rain on the roof",
     "story_engine": "siblings clearing a parent's attic find proof the family story was a lie",
     "ending_mode": "a truth chosen over comfort, a secret reburied, or a photo kept and never explained", "tags": "drama"},
    {"slug": "seance_room_audio_anomaly", "label": "seance-room audio anomaly",
     "sound_world": "a creaking table, held breath, a candle's sputter, a knock, a hum",
     "story_engine": "a skeptic records a seance to debunk it and captures the one thing he can't",
     "ending_mode": "a hoax exposed, a believer made, or a recording no one will play again", "tags": "uncanny"},
    {"slug": "funeral_home_midnight_call", "label": "funeral-home midnight call",
     "sound_world": "a phone, soft organ, a clock, a door, a long quiet hall",
     "story_engine": "a mortician takes a late call from a family whose loss isn't what it seems",
     "ending_mode": "a grief honored, a fraud uncovered, or a casket closed on a question", "tags": "mystery"},
    {"slug": "old_theater_phantom_rehearsal", "label": "old-theater phantom rehearsal",
     "sound_world": "an empty house's echo, a piano, rope and pulley, footsteps on boards",
     "story_engine": "a struggling company rehearses in a theater that wants its old show back",
     "ending_mode": "a performance redeemed, an actor who joins the ghosts, or a curtain that won't rise", "tags": "uncanny"},

    # --- 51-70: speculative / sci-fi (NON-doomsday) ---
    {"slug": "scientific_ethics_chamber_drama", "label": "scientific ethics chamber drama",
     "sound_world": "a quiet boardroom, water poured, a pen, a clock, papers squared",
     "story_engine": "a committee debates whether a discovery should exist at all",
     "ending_mode": "a line drawn, a dissent recorded, or a vote that no one feels clean about", "tags": "drama"},
    {"slug": "artificial_mind_awakening_interview", "label": "artificial-mind awakening interview",
     "sound_world": "server fans, a soft chime, a chair, typing, a measured synthetic voice",
     "story_engine": "a researcher interviews a new mind and isn't sure who is studying whom",
     "ending_mode": "a recognition of personhood, a careful shutdown, or a question left answered by silence", "tags": "drama"},
    {"slug": "failed_experiment_countdown", "label": "failed-experiment reckoning",
     "sound_world": "a humming rig, a kettle, a notebook, a fan, a tired sigh",
     "story_engine": "a scientist faces the night her life's work clearly will not work",
     "ending_mode": "a graceful surrender, a last idea that saves it, or a lab lights-off and a walk home", "tags": "drama"},
    {"slug": "time_loop_command_log", "label": "time-loop command log",
     "sound_world": "a ship's hum, a repeating chime, a log recorder, footsteps, a sigh",
     "story_engine": "an officer logging the same hour again learns what the loop is asking of them",
     "ending_mode": "the loop broken by a choice, an acceptance of it, or a log that begins again", "tags": "speculative"},
    {"slug": "alternate_history_news_bulletin", "label": "alternate-history news bulletin",
     "sound_world": "a studio mic, a wire feed, a station ID, a clock, controlled voices",
     "story_engine": "an anchor reads the news of a world one decision away from ours",
     "ending_mode": "a truth that reframes our own, an on-air doubt, or a sign-off into static", "tags": "speculative"},
    {"slug": "memory_erasure_clinic_session", "label": "memory-erasure clinic session",
     "sound_world": "a soft tone, a reclining chair, a clock, a technician's calm, a hum",
     "story_engine": "a patient choosing to forget argues with the part of them that wants to keep it",
     "ending_mode": "a memory kept against advice, a clean erasure, or a single image that survives", "tags": "drama"},
    {"slug": "cryogenic_revival_confusion", "label": "cryogenic-revival confusion",
     "sound_world": "a thawing hiss, a heart monitor, a gentle voice, dripping, a slow breath",
     "story_engine": "someone wakes long after their world and must learn what is left to want",
     "ending_mode": "a reason to go on found, a grief for the lost time, or a window opened on a strange dawn", "tags": "drama"},
    {"slug": "alien_contact_translation_room", "label": "alien-contact translation room",
     "sound_world": "soft signal tones, headphones, a chalk scratch, a fan, hushed wonder",
     "story_engine": "linguists build a first sentence with a mind unlike any human one",
     "ending_mode": "a word shared across the gulf, a misread that humbles, or a reply still being parsed", "tags": "speculative"},
    {"slug": "robot_witness_testimony", "label": "robot-witness testimony",
     "sound_world": "a courtroom hush, a servo, a stenotype, a gavel, a measured voice",
     "story_engine": "a machine testifies and the room argues what its memory is worth",
     "ending_mode": "a verdict that grants it standing, a dismissal, or a record sealed and unresolved", "tags": "speculative"},
    {"slug": "simulated_world_glitch_mystery", "label": "simulated-world glitch mystery",
     "sound_world": "an ordinary day's room tone with one sound that repeats slightly wrong",
     "story_engine": "small wrongnesses lead a person to test the edges of their own world",
     "ending_mode": "a truth chosen over comfort, an edge that holds, or a day that simply resets", "tags": "speculative"},
    {"slug": "courtroom_of_the_future_trial", "label": "courtroom-of-the-future trial",
     "sound_world": "a chime-gavel, a soft AI clerk, murmurs, a chair, a clean hush",
     "story_engine": "a trial run by new rules tests whether justice survived the upgrade",
     "ending_mode": "a humane verdict, a system's cold ruling overturned, or an appeal left open", "tags": "speculative"},
    {"slug": "planetary_colony_town_meeting", "label": "planetary-colony town meeting",
     "sound_world": "a hall's echo, shuffling boots, a gavel, a vent's hum, raised voices",
     "story_engine": "a young colony argues a choice that will decide the kind of place it becomes",
     "ending_mode": "a compromise reached, a faction's quiet exit, or a vote carried into an uneasy morning", "tags": "drama"},
    {"slug": "terraforming_disaster_broadcast", "label": "terraforming setback broadcast",
     "sound_world": "wind on a young world, a comms tower hum, a recorder, distant machinery",
     "story_engine": "engineers face that the world they're making won't come in their lifetime",
     "ending_mode": "a long view embraced, a record left for the next generation, or a quiet recommitment", "tags": "drama"},
    {"slug": "generation_ship_succession_crisis", "label": "generation-ship succession crisis",
     "sound_world": "a deep engine hum, corridors, a bell, a gathering crowd, a soft chime",
     "story_engine": "a ship born and dying in transit must choose who carries the mission next",
     "ending_mode": "a torch passed, an old order ending with grace, or a corridor of waiting silence", "tags": "drama"},
    {"slug": "asteroid_mining_labor_dispute", "label": "asteroid-mining labor dispute",
     "sound_world": "drills, a klaxon test, suit radio, a canteen, boots on metal",
     "story_engine": "miners and a foreman bargain over safety where everyone is expendable",
     "ending_mode": "a fair deal won, a walkout, or a handshake no one fully trusts", "tags": "drama"},
    {"slug": "orbital_rescue_relay", "label": "orbital rescue relay",
     "sound_world": "comms relays, a countdown clock, thruster taps, breath, a calm controller",
     "story_engine": "scattered stations relay one rescue across the dark, trusting strangers",
     "ending_mode": "a chain that holds and saves a life, a link that fails, or a thank-you into static", "tags": "emergency"},
    {"slug": "first_contact_diplomatic_standoff", "label": "first-contact diplomatic standoff",
     "sound_world": "a quiet chamber, a translator's tone, a clock, water poured, careful voices",
     "story_engine": "two peoples negotiate trust with everything to lose to one wrong word",
     "ending_mode": "a fragile accord, a gesture that bridges it, or a recess into uneasy hope", "tags": "speculative"},
    {"slug": "forbidden_zone_survey_log", "label": "forbidden-zone survey log",
     "sound_world": "a Geiger tick, wind, boots in dust, a recorder, a far creak",
     "story_engine": "a surveyor logs a place humans abandoned and starts to understand why",
     "ending_mode": "a warning recorded for others, a quiet decision to leave it be, or a log cut short", "tags": "mystery"},
    {"slug": "lost_satellite_recovery_mission", "label": "lost-satellite recovery mission",
     "sound_world": "telemetry beeps, a dish turning, a fan, keyboard taps, a held breath",
     "story_engine": "a small team coaxes a dead satellite to answer one last time",
     "ending_mode": "a signal recovered, a graceful goodbye to old hardware, or a silence accepted", "tags": "drama"},
    {"slug": "evacuation_order_moral_dilemma", "label": "evacuation-order moral dilemma",
     "sound_world": "a PA, packing, a bus idling, a child's voice, a door, distant thunder",
     "story_engine": "an official must decide who leaves first when not everyone can",
     "ending_mode": "a humane call made and lived with, a list torn up, or a last bus pulling away", "tags": "drama"},

    # --- 71-90: family / relationship / radio-form ---
    {"slug": "family_dinner_secret_eruption", "label": "family-dinner secret eruption",
     "sound_world": "cutlery, a clock, chairs, a pot lid, a long pause, a fork set down",
     "story_engine": "one dinner where a held secret finally has nowhere left to go",
     "ending_mode": "a truth that reshapes the family, a peace kept at a cost, or a door closing softly", "tags": "drama"},
    {"slug": "inheritance_tape_revelation", "label": "inheritance-tape revelation",
     "sound_world": "a tape deck, a will's pages, a clock, a sigh, a chair creak",
     "story_engine": "heirs play a recorded will that knows each of them too well",
     "ending_mode": "a reconciliation, a fortune refused, or a final message that forgives no one", "tags": "drama"},
    {"slug": "marriage_under_surveillance_drama", "label": "marriage-under-surveillance drama",
     "sound_world": "a quiet apartment, a clock, a tap, a phone, the hum of a listened-to room",
     "story_engine": "a couple who knows they're being watched stops knowing what's performance",
     "ending_mode": "a trust rebuilt, a confession that frees them, or a goodnight neither believes", "tags": "drama"},
    {"slug": "sibling_rivalry_confession_room", "label": "sibling-rivalry confession room",
     "sound_world": "a small room, two chairs, a clock, breathing, a cup set down",
     "story_engine": "two siblings finally say the thing the years were built to avoid",
     "ending_mode": "an honesty that mends or breaks, a draw called, or a silence that has to be enough", "tags": "drama"},
    {"slug": "old_friendship_betrayal_call", "label": "old-friendship betrayal call",
     "sound_world": "a phone line's hum, a clock, a chair, a held breath, street noise",
     "story_engine": "an overdue call between old friends circles the thing one did to the other",
     "ending_mode": "a forgiveness, a friendship ended cleanly, or a line that goes quiet first", "tags": "drama"},
    {"slug": "neighborhood_watch_paranoia", "label": "neighborhood-watch paranoia",
     "sound_world": "a porch, crickets, a screen door, radios, a dog, low voices",
     "story_engine": "a watch group's vigilance curdles into suspicion of one of their own",
     "ending_mode": "a misjudgment owned, a community fractured, or a porch light left on for no one", "tags": "drama"},
    {"slug": "landlord_tenant_hidden_room_mystery", "label": "landlord-tenant hidden-room mystery",
     "sound_world": "pipes knocking, footsteps above, a key, a wall tapped, a furnace",
     "story_engine": "a tenant finds a sealed room and a landlord with a reason to keep it sealed",
     "ending_mode": "the room's truth told, a deal to forget it, or a wall left exactly as found", "tags": "mystery"},
    {"slug": "retirement_home_ghost_story", "label": "retirement-home ghost story",
     "sound_world": "a TV low, a cart's squeak, a clock, slippers, a kettle, soft halls",
     "story_engine": "residents trade a ghost story until it turns out one of them is in it",
     "ending_mode": "a gentle haunting resolved, a goodbye honored, or a chair that stays warm", "tags": "uncanny"},
    {"slug": "school_reunion_accusation", "label": "school-reunion accusation",
     "sound_world": "a gym's echo, a dated playlist, plastic cups, laughter, a mic feedback",
     "story_engine": "an old wrong surfaces at a reunion where everyone agreed to forget it",
     "ending_mode": "an apology decades late, a reputation reframed, or the music covering it again", "tags": "drama"},
    {"slug": "hospital_waiting_room_vigil", "label": "hospital waiting-room vigil",
     "sound_world": "a vending machine, a clock, distant pages, a door, a cough, quiet",
     "story_engine": "strangers wait on news and become, for one night, each other's people",
     "ending_mode": "a relief shared, a loss held together, or a name called into the hush", "tags": "drama"},
    {"slug": "live_variety_show_disaster", "label": "live variety-show disaster",
     "sound_world": "a band warming, applause, a clattering mic, a flustered host, a count-in",
     "story_engine": "a live broadcast goes wrong on air and the cast improvises through it",
     "ending_mode": "a save that becomes the best show ever, an honest collapse, or a sign-off mid-chaos", "tags": "comedy"},
    {"slug": "radio_play_within_a_radio_play", "label": "radio play within a radio play",
     "sound_world": "a foley table, a studio clock, a director's cue, page turns, a buzzer",
     "story_engine": "actors recording a drama find its plot bleeding into their own night",
     "ending_mode": "the layers resolved with a wink, a line that lands too true, or a take that won't end", "tags": "meta"},
    {"slug": "fake_advertisement_dystopia", "label": "fake-advertisement dystopia",
     "sound_world": "a chirpy jingle, applause samples, a too-smooth announcer, a chime",
     "story_engine": "a sponsor's perfect product hides a cost the ad keeps almost admitting",
     "ending_mode": "the pitch breaking to show the truth, a buyer who walks, or one more cheerful spot", "tags": "satire"},
    {"slug": "propaganda_station_moral_collapse", "label": "propaganda-station moral collapse",
     "sound_world": "a state anthem bed, a control room, a clock, a script's pages, a fan",
     "story_engine": "a broadcaster who reads the lies for a living reaches the one she can't",
     "ending_mode": "a truth slipped on air, a quiet resignation, or a mic switched off mid-script", "tags": "drama"},
    {"slug": "pirate_radio_resistance_drama", "label": "pirate-radio resistance drama",
     "sound_world": "a makeshift rig's hum, a generator, footsteps, a window taped, low music",
     "story_engine": "an outlaw station keeps broadcasting hope as the signal-hunters close in",
     "ending_mode": "a message that gets through, a transmitter sacrificed, or a sign-off into static and dawn", "tags": "drama"},
    {"slug": "public_service_announcement_nightmare", "label": "public-service-announcement nightmare",
     "sound_world": "a calm institutional voice, a soft chime, room tone, a clock",
     "story_engine": "a reassuring announcement slowly reveals the thing it's announcing",
     "ending_mode": "the message turned by a listener, a cheerful denial, or a loop that starts over", "tags": "satire"},
    {"slug": "call_in_show_confession_spiral", "label": "call-in show confession spiral",
     "sound_world": "a phone bed, a host's mic, a hold tone, callers' rooms, a clock",
     "story_engine": "a late-night host's callers keep confessing pieces of one shared story",
     "ending_mode": "a truth assembled live, a caller talked back from the edge, or the lines going quiet", "tags": "drama"},
    {"slug": "childrens_program_turns_sinister", "label": "children's program turns sinister",
     "sound_world": "a toy piano, a friendly host, bells, a giggle track, a wrong note",
     "story_engine": "a cheerful kids' show keeps teaching one lesson it shouldn't",
     "ending_mode": "the host breaking character to warn someone, a tidy denial, or a sing-off that lingers", "tags": "uncanny"},
    {"slug": "overnight_jazz_host_mystery", "label": "overnight jazz-host mystery",
     "sound_world": "smooth jazz, a low mic, a coffee pour, a request line, rain on glass",
     "story_engine": "a night DJ's requests spell out a message only he can read in time",
     "ending_mode": "a life saved between songs, a request unanswered, or a last track into dawn", "tags": "mystery"},
    {"slug": "numbers_station_spy_thriller", "label": "numbers-station spy thriller",
     "sound_world": "a flat synthesized voice reading digits, static, a tuning whine, a pencil",
     "story_engine": "a listener decodes a numbers station and learns the next number is for them",
     "ending_mode": "an extraction made, a handler's betrayal, or the station repeating into silence", "tags": "suspense"},

    # --- 91-100: adventure / disaster / closer ---
    {"slug": "archaeological_dig_curse_log", "label": "archaeological dig curse log",
     "sound_world": "brushes on stone, wind, a recorder, trowels, a tent flap, a far jackal",
     "story_engine": "a dig team logs a find as old warnings start to fit too well",
     "ending_mode": "a relic reburied with respect, a curse rationalized away, or a log that ends mid-entry", "tags": "adventure"},
    {"slug": "jungle_temple_expedition", "label": "jungle-temple expedition",
     "sound_world": "birds, dripping canopy, machetes, insects, a torch, stone underfoot",
     "story_engine": "explorers reach a temple guarding something better left where it is",
     "ending_mode": "a treasure refused, a guardian respected, or a door sealed behind them", "tags": "adventure"},
    {"slug": "island_evacuation_suspense", "label": "island-evacuation suspense",
     "sound_world": "surf, a boat horn, wind, a radio, footsteps on a dock, gulls",
     "story_engine": "an island empties ahead of a coming threat with one boat and too many people",
     "ending_mode": "a humane order of departure, a captain's hard choice, or a last figure on the shore", "tags": "emergency"},
    {"slug": "sinking_ship_wireless_drama", "label": "sinking-ship wireless drama",
     "sound_world": "Morse keying, water rising, a list groaning, a steam whistle, calm voices",
     "story_engine": "a wireless operator keeps calling for help while the deck tilts away",
     "ending_mode": "a rescue raised, a last message tapped out, or a key that goes silent with grace", "tags": "emergency"},
    {"slug": "airship_storm_emergency", "label": "airship storm emergency",
     "sound_world": "canvas straining, wind, a creaking gondola, a gauge tap, a calm captain",
     "story_engine": "an airship crew rides out a storm that tests the ship and each other",
     "ending_mode": "a calm reached by teamwork, a load lightened by sacrifice, or a drift into clear sky", "tags": "adventure"},
    {"slug": "frontier_trading_post_siege", "label": "frontier trading-post siege",
     "sound_world": "wind, a barred door, a fire, a clock, distant calls, a kettle",
     "story_engine": "a remote post waits out a threat as supplies and tempers run low",
     "ending_mode": "a truce brokered, a relief column arriving, or a dawn that simply comes quietly", "tags": "drama"},
    {"slug": "traveling_preacher_scandal", "label": "traveling-preacher scandal",
     "sound_world": "a tent revival's crowd, a wheezing organ, a mic, applause, cicadas",
     "story_engine": "a charismatic preacher's gift collides with the lie that funds it",
     "ending_mode": "a public reckoning, a quiet town move on, or a sermon that almost confesses", "tags": "drama"},
    {"slug": "gold_rush_camp_murder", "label": "gold-rush camp murder",
     "sound_world": "a creek, pans, a fiddle, boots in mud, a saloon door, wind",
     "story_engine": "a strike brings a death the camp must judge before the law arrives",
     "ending_mode": "a rough justice done, a guilt that hides in plain sight, or a verdict the river keeps", "tags": "mystery"},
    {"slug": "circus_train_disaster", "label": "circus-train mystery",
     "sound_world": "rails, animal calls, a calliope, a whistle, canvas, a lantern's clink",
     "story_engine": "a traveling circus's family closes ranks around a death on the rails",
     "ending_mode": "a truth kept in the family, a performer's confession, or the show going on regardless", "tags": "mystery"},
    {"slug": "final_message_before_silence", "label": "final message before silence",
     "sound_world": "a single mic, room tone, a clock winding down, a breath, a soft hum",
     "story_engine": "one voice records the last thing it wants the world to hear, and why",
     "ending_mode": "a peace made, a hope handed forward, or a recording that ends on an open breath", "tags": "drama"},
]


# ---------------------------------------------------------------------------
# Ending taxonomy (2026-06-24) -- the CLIMAX-CLASS tag per style + its template
# ---------------------------------------------------------------------------
# The climax's TYPE is one of the climax-class roles (CLIMAX_CLASS_ROLES in
# _otr_story_quality_l12). `irreversible_choice` is no longer the universal
# climax -- it is just ONE class, used only where a story genuinely earns a
# decisive choice. Each style maps to exactly one ending_tag; the writer feeds
# that tag as the climax role + injects the matching ENDING_TEMPLATES instruction
# at the final character beat. Every template steers AWAY from machinery
# (console / countdown / self-destruct / kill-switch) and TOWARD a human, on-mic
# ending.
try:
    from ._otr_story_quality_l12 import CLIMAX_CLASS_ROLES, premise_texts
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_story_quality_l12 import CLIMAX_CLASS_ROLES, premise_texts  # type: ignore

ENDING_TEMPLATES: Dict[str, str] = {
    "irreversible_choice":
        "The final beat is a decisive, irreversible CHOICE made on-mic between "
        "the characters -- a human decision they must live with, about people and "
        "stakes. NOT a machine action: no countdown, no self-destruct, no "
        "console, no kill-switch.",
    "revelation":
        "The final beat lands a REVELATION -- a truth that reframes everything, "
        "spoken or realized on-mic. End on the moment of understanding, not on "
        "any device.",
    "reversal":
        "The final beat is a REVERSAL -- who holds the power, or who to trust, "
        "flips. End on the turn itself, in dialogue, not on machinery.",
    "unresolved_final_sound":
        "Do NOT resolve. End on ONE telling sound and a line that leaves the "
        "question open. No explanation, no device, no outcome stated.",
    "reconciliation":
        "The final beat is a RECONCILIATION -- two people close a distance "
        "between them. End on the small human gesture (a word, a touch, a shared "
        "silence), not on any action or machine.",
    "bittersweet_parting":
        "The final beat is a BITTERSWEET PARTING -- a goodbye, a torch passed, a "
        "thing left behind. End on the quiet cost, not on a crisis or device.",
    "ironic_twist":
        "The final beat is an IRONIC TWIST -- the outcome curdles or rebounds in "
        "a way the characters didn't intend. End on the irony, lightly, not on "
        "machinery.",
    "quiet_acceptance":
        "The final beat is a QUIET ACCEPTANCE -- a character stops fighting and "
        "makes peace with what is. A small, human decision. No machinery, no "
        "countdown, no blowing anything up.",
    "confession":
        "The final beat is a CONFESSION -- a character finally admits the thing "
        "they've hidden. End on the admission itself, on-mic, not on any action "
        "or device.",
}

ENDING_TAGS: tuple = tuple(ENDING_TEMPLATES.keys())
_DEFAULT_ENDING_TAG = "revelation"

# Per-style climax class. Deliberately VARIED so the climax stops being the same
# shape every episode; `irreversible_choice` is now ~6% of the pool (only the
# stories that genuinely earn a decisive choice), not 100%.
_ENDING_TAG_BY_SLUG: Dict[str, str] = {
    "locked_room_suspense": "reversal",
    "detective_case_file_reconstruction": "revelation",
    "pulp_serial_cliffhanger": "unresolved_final_sound",
    "mission_control_procedural": "irreversible_choice",
    "deep_space_distress_transmission": "unresolved_final_sound",
    "noir_interrogation_chamber": "confession",
    "small_town_uncanny_mystery": "revelation",
    "newsroom_emergency_bulletin": "irreversible_choice",
    "haunted_broadcast_signal": "unresolved_final_sound",
    "laboratory_containment_breach": "irreversible_choice",
    "isolated_lighthouse_confession": "confession",
    "courtroom_testimony_thriller": "reversal",
    "missing_person_audio_diary": "unresolved_final_sound",
    "submarine_pressure_crisis": "irreversible_choice",
    "train_car_murder_mystery": "revelation",
    "stormbound_inn_conspiracy": "reversal",
    "expedition_camp_radio_log": "bittersweet_parting",
    "wartime_code_room_drama": "bittersweet_parting",
    "emergency_dispatch_night_shift": "quiet_acceptance",
    "psychiatric_ward_interview": "revelation",
    "lost_expedition_recordings": "unresolved_final_sound",
    "arctic_outpost_survival": "quiet_acceptance",
    "desert_station_signal_hunt": "unresolved_final_sound",
    "mountain_rescue_countdown": "bittersweet_parting",
    "plague_town_quarantine_drama": "bittersweet_parting",
    "museum_after_hours_haunting": "unresolved_final_sound",
    "factory_floor_sabotage_mystery": "revelation",
    "hotel_switchboard_thriller": "reversal",
    "space_station_systems_failure": "confession",
    "underwater_research_panic": "quiet_acceptance",
    "private_eye_street_noir": "reversal",
    "corrupt_city_political_wiretap": "ironic_twist",
    "blackmail_phone_call_thriller": "reversal",
    "ransom_negotiation_countdown": "quiet_acceptance",
    "witness_protection_confession": "confession",
    "fugitive_safehouse_drama": "bittersweet_parting",
    "border_crossing_identity_mystery": "revelation",
    "prison_cell_psychological_duel": "reversal",
    "interrogation_under_blackout": "confession",
    "informant_dead_drop_suspense": "unresolved_final_sound",
    "rural_folklore_investigation": "revelation",
    "carnival_after_midnight_uncanny": "unresolved_final_sound",
    "roadside_diner_mystery": "revelation",
    "abandoned_motel_transmission": "unresolved_final_sound",
    "ghost_town_field_recording": "unresolved_final_sound",
    "cursed_object_inventory": "ironic_twist",
    "family_attic_revelation": "revelation",
    "seance_room_audio_anomaly": "unresolved_final_sound",
    "funeral_home_midnight_call": "revelation",
    "old_theater_phantom_rehearsal": "bittersweet_parting",
    "scientific_ethics_chamber_drama": "irreversible_choice",
    "artificial_mind_awakening_interview": "revelation",
    "failed_experiment_countdown": "quiet_acceptance",
    "time_loop_command_log": "reversal",
    "alternate_history_news_bulletin": "revelation",
    "memory_erasure_clinic_session": "bittersweet_parting",
    "cryogenic_revival_confusion": "quiet_acceptance",
    "alien_contact_translation_room": "revelation",
    "robot_witness_testimony": "reversal",
    "simulated_world_glitch_mystery": "revelation",
    "courtroom_of_the_future_trial": "reversal",
    "planetary_colony_town_meeting": "reconciliation",
    "terraforming_disaster_broadcast": "quiet_acceptance",
    "generation_ship_succession_crisis": "bittersweet_parting",
    "asteroid_mining_labor_dispute": "reconciliation",
    "orbital_rescue_relay": "bittersweet_parting",
    "first_contact_diplomatic_standoff": "reconciliation",
    "forbidden_zone_survey_log": "quiet_acceptance",
    "lost_satellite_recovery_mission": "bittersweet_parting",
    "evacuation_order_moral_dilemma": "irreversible_choice",
    "family_dinner_secret_eruption": "confession",
    "inheritance_tape_revelation": "reconciliation",
    "marriage_under_surveillance_drama": "reconciliation",
    "sibling_rivalry_confession_room": "confession",
    "old_friendship_betrayal_call": "reconciliation",
    "neighborhood_watch_paranoia": "bittersweet_parting",
    "landlord_tenant_hidden_room_mystery": "revelation",
    "retirement_home_ghost_story": "bittersweet_parting",
    "school_reunion_accusation": "confession",
    "hospital_waiting_room_vigil": "quiet_acceptance",
    "live_variety_show_disaster": "ironic_twist",
    "radio_play_within_a_radio_play": "ironic_twist",
    "fake_advertisement_dystopia": "ironic_twist",
    "propaganda_station_moral_collapse": "reversal",
    "pirate_radio_resistance_drama": "bittersweet_parting",
    "public_service_announcement_nightmare": "ironic_twist",
    "call_in_show_confession_spiral": "confession",
    "childrens_program_turns_sinister": "unresolved_final_sound",
    "overnight_jazz_host_mystery": "revelation",
    "numbers_station_spy_thriller": "reversal",
    "archaeological_dig_curse_log": "unresolved_final_sound",
    "jungle_temple_expedition": "quiet_acceptance",
    "island_evacuation_suspense": "bittersweet_parting",
    "sinking_ship_wireless_drama": "bittersweet_parting",
    "airship_storm_emergency": "reconciliation",
    "frontier_trading_post_siege": "reconciliation",
    "traveling_preacher_scandal": "confession",
    "gold_rush_camp_murder": "revelation",
    "circus_train_disaster": "revelation",
    "final_message_before_silence": "quiet_acceptance",
}

# Attach the ending_tag to each entry at module load (keep the `ending_mode`
# prose unchanged -- render_style_grammar reads it).
for _s in STYLE_CATALOG:
    _s["ending_tag"] = _ENDING_TAG_BY_SLUG.get(_s["slug"], _DEFAULT_ENDING_TAG)


def ending_template_for(slug: str) -> str:
    """The concrete final-beat instruction for a style's climax class. Falls back
    to the default tag's template for an unknown slug (never raises)."""
    s = get_style(slug)
    tag = (s or {}).get("ending_tag", _DEFAULT_ENDING_TAG)
    return ENDING_TEMPLATES.get(tag, ENDING_TEMPLATES[_DEFAULT_ENDING_TAG])


def validate_catalog() -> None:
    """Self-check (run in a test): every entry has a valid ending_tag that is a
    real climax-class role with a template, plus the core grammar fields. LOUD on
    any gap."""
    for s in STYLE_CATALOG:
        slug = s.get("slug", "?")
        tag = s.get("ending_tag", "")
        if tag not in ENDING_TAGS:
            raise ValueError(f"style {slug!r} has invalid ending_tag {tag!r}")
        if tag not in CLIMAX_CLASS_ROLES:
            raise ValueError(
                f"style {slug!r} ending_tag {tag!r} is not a climax-class role")
        if tag not in ENDING_TEMPLATES:
            raise ValueError(f"style {slug!r} ending_tag {tag!r} has no template")
        for fld in ("label", "sound_world", "story_engine", "ending_mode"):
            if not str(s.get(fld, "")).strip():
                raise ValueError(f"style {slug!r} missing {fld}")


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------
_BY_SLUG: Dict[str, Dict[str, str]] = {s["slug"]: s for s in STYLE_CATALOG}

# Styles in the high-tension "emergency" register -- the picker should reach for
# these ONLY when the article genuinely calls for one, never as the default, so
# the catalog's center of gravity stays off the console standoff.
EMERGENCY_TAG = "emergency"


def all_slugs() -> List[str]:
    return [s["slug"] for s in STYLE_CATALOG]


def get_style(slug: str) -> Optional[Dict[str, str]]:
    """Return the catalog entry for ``slug`` (exact, then a normalized match), or
    None. Normalization lets a model-emitted label like 'Locked Room Suspense'
    or 'locked-room suspense' resolve to the canonical slug."""
    if slug in _BY_SLUG:
        return _BY_SLUG[slug]
    norm = _normalize(slug)
    for s in STYLE_CATALOG:
        if _normalize(s["slug"]) == norm or _normalize(s["label"]) == norm:
            return s
    return None


def _normalize(s: str) -> str:
    out = []
    for ch in str(s or "").lower():
        out.append(ch if ch.isalnum() else "_")
    return "_".join(t for t in "".join(out).split("_") if t)


def non_emergency_slugs() -> List[str]:
    return [s["slug"] for s in STYLE_CATALOG if s.get("tags") != EMERGENCY_TAG]


MEDIA_ARCHIVE_STYLE_SLUGS: tuple[str, ...] = (
    "radio_play_within_a_radio_play",
    "overnight_jazz_host_mystery",
    "call_in_show_confession_spiral",
    "live_variety_show_disaster",
    "pirate_radio_resistance_drama",
    "newsroom_emergency_bulletin",
    "inheritance_tape_revelation",
    "family_attic_revelation",
    "final_message_before_silence",
    "lost_expedition_recordings",
    "desert_station_signal_hunt",
    "roadside_diner_mystery",
    "isolated_lighthouse_confession",
    "family_dinner_secret_eruption",
    "sibling_rivalry_confession_room",
    "old_friendship_betrayal_call",
    "school_reunion_accusation",
    "hospital_waiting_room_vigil",
    "landlord_tenant_hidden_room_mystery",
    "pulp_serial_cliffhanger",
    "expedition_camp_radio_log",
    "old_theater_phantom_rehearsal",
    "missing_person_audio_diary",
    "small_town_uncanny_mystery",
)


def media_archive_slugs() -> List[str]:
    """Curated existing style pool for archive mystery/adventure runs."""
    return list(MEDIA_ARCHIVE_STYLE_SLUGS)


def render_style_grammar(slug: str) -> str:
    """Render a style's grammar as a compact prompt block. Empty string for an
    unknown slug (caller falls back to the bare label => byte-identical)."""
    s = get_style(slug)
    if not s:
        return ""
    return (
        f"Style: {s['label']}.\n"
        f"Sound world: {s['sound_world']}.\n"
        f"Story engine: {s['story_engine']}.\n"
        f"Ending mode: {s['ending_mode']}."
    )


# ---------------------------------------------------------------------------
# Deterministic style selector (replaces the LLM inventor when the lever is on)
# ---------------------------------------------------------------------------
# An article only earns an emergency-register style when it is genuinely about a
# disaster / rescue; otherwise the selector draws from the 90 non-emergency
# styles so the catalog's center of gravity stays off the console standoff.
_DISASTER_KEYWORDS: tuple = (
    "disaster", "rescue", "evacuat", "explos", "meltdown", "crash", "sinking",
    "wildfire", "quarantine", "outbreak", "epidemic", "pandemic", "collapse",
    "distress", "catastroph", "hurricane", "earthquake", "tsunami", "flood",
    "wreck", "siege", "containment", "breach", "mayday", "emergency",
)


def premise_wants_emergency(premise: Any, meta: Any) -> bool:
    """True when the premise / news meta carry explicit disaster-or-rescue
    language -- the only case where an emergency-register style is eligible."""
    parts = [str(premise or "")]
    try:
        parts.extend(premise_texts(meta))
    except Exception:  # noqa: BLE001 -- never break selection on a meta shape
        pass
    hay = " ".join(parts).casefold()
    return any(k in hay for k in _DISASTER_KEYWORDS)


def select_style(premise: Any, meta: Any, cast_seed: Any) -> str:
    """Deterministically pick a style slug for an episode. No LLM, no paid call.

    Pool = the 90 non-emergency styles, UNLESS the premise/meta carry explicit
    disaster/rescue language (then the full 100 are eligible). The pick is a
    sha256(cast_seed)-keyed index into the sorted pool, so the same episode seed
    always yields the same style (C7-safe) while different episodes vary."""
    import hashlib
    source_bank = ""
    if isinstance(meta, dict):
        source_bank = str(meta.get("source_bank", "") or "")
    if source_bank == "media_archive":
        pool = sorted(media_archive_slugs())
        if not pool:  # pragma: no cover -- defensive
            pool = sorted(non_emergency_slugs())
        h = int(hashlib.sha256(
            f"{cast_seed}:style:media_archive".encode("utf-8")
        ).hexdigest(), 16)
        return pool[h % len(pool)]
    emergency = premise_wants_emergency(premise, meta)
    pool = sorted(all_slugs() if emergency else non_emergency_slugs())
    if not pool:  # pragma: no cover -- defensive
        pool = sorted(all_slugs())
    h = int(hashlib.sha256(
        f"{cast_seed}:style:{int(emergency)}".encode("utf-8")
    ).hexdigest(), 16)
    return pool[h % len(pool)]


@dataclass(frozen=True)
class StoryContract:
    """One immutable per-episode style decision, selected ONCE pre-outline and
    threaded through the whole body (KILL 2, 2026-06-24). ``grammar`` is the
    rendered prompt block consumed at the macro prompt -- the first and only
    caller of render_style_grammar (whose prior "zero callers" state was the
    literal KILL-2 bug: the style was selected but never injected)."""
    slug: str
    label: str
    sound_world: str
    story_engine: str
    ending_mode: str
    ending_tag: str
    ending_template: str
    grammar: str


def build_story_contract(cast_seed: Any, script_brief: str, news_seed: str,
                         meta: Any) -> StoryContract:
    """Select the episode style deterministically (sha256(cast_seed)-keyed via
    select_style) from the script_brief (or the raw news_seed when no brief)
    BEFORE the outline is generated, and freeze its full grammar for the body.
    Never raises on a missing style (get_style -> {} -> empty fields)."""
    text = script_brief or news_seed or ""
    slug = select_style(text, meta, cast_seed)
    s = get_style(slug) or {}
    return StoryContract(
        slug=slug,
        label=s.get("label", ""),
        sound_world=s.get("sound_world", ""),
        story_engine=s.get("story_engine", ""),
        ending_mode=s.get("ending_mode", ""),
        ending_tag=s.get("ending_tag", ""),
        ending_template=ending_template_for(slug),
        grammar=render_style_grammar(slug),
    )
