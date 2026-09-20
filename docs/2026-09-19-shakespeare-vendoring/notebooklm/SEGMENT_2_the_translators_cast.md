# The translator's cast is not Shakespeare's

## The one thing to take away (3-5 sentences)

A cast list in a historical translation is an editorial act, not a photocopy of Shakespeare's people. The books measured here seat a carpenter he left silent, print the play-within-the-play's roles as separate personages, give two dukes one shared line, and abbreviate names so that three letters can mean two different men in one bound volume. A **fold** is a written pairing -- this printed label, in this scene, is that English roster name. An **unbound** label ships raw, costing a voice and never a line, because a wrong fold is worse than no fold. Rename the people and the cell is an adaptation; carry the translator's own form (`FERNANDO`) and it can be bound without rewriting him. The centre is the line Shakespeare gives to several people at once: every edition marks that unison differently, and those marks decide who a listener thinks is in the room.

## What we found (concrete cases, each with its page, label and count)

**The joiner who was not supposed to speak.** Folger's English *Midsummer* 3.1 has no SNUG among its speakers [midsummer__act3_scene1.provenance.json:16-29]. Macpherson's 1897 PERSONAJES page still lists `AJUSTADO , ebanista` [p.381; GROK_FOLD_TABLES_measured.md:92-103], and in the rehearsal he speaks once: `AJUS. Es imposible representar un muro.` [p.416; n=1]. Folger gives that wall-problem to Snout [midsummer__act3_scene1.txt:91-92]. Rusconi's 1838 Italian seats the joiner the same way: `SNUG: È impossibile portare un muro sulla scena.` [it/midsummer_3_1.txt:26]. Two translators put a speech in a mouth the English list leaves empty; the Spanish measurement leaves `AJUS` unbound rather than invent a Snug [GROK_FOLD_TABLES_measured.md:103, 208-210].

**Roles listed as people.** The same PERSONAJES page lists the mechanicals by trade (`BORRAS, tejedor`, `FLAUTA , remienda -fuelles`) and then, separately, `PÍRAMO, TISBE , MURO` as `Personajes del entremés` [p.381; GROK_FOLD_TABLES_measured.md:94-95, 106-108]. In 3.1 those role-names speak: `TISBE` three times (including `TisbE`), `Pla` once, `Pir` / `PIR` twice [GROK_FOLD_TABLES_measured.md:106-108]. One line on p.417 begins `TISBE. ;On blanco lirio , Píramo radiante !`. Folger keeps the actor -- `BOTTOM, [as Pyramus]`, `FLUTE, [as Thisbe]` [midsummer__act3_scene1.txt:118, 137] -- while Rusconi prints `PIR:` and `TIS:` as Macpherson does [it/midsummer_3_1.txt:31, 35]. Folding `TISBE` to Flute would steal the actor's voice; both Thisbe forms stay unbound together [GROK_FOLD_TABLES_measured.md:216-219].

**Two lords, one line.** Folger's Lear 1.1 prints `ALBANY/CORNWALL  Dear sir, forbear.` [king_lear__act1_scene1.txt:257] and then omits both names from the scene's speaker list [king_lear__act1_scene1.provenance.json:16-26]. Macpherson's PERSONAJES still seats `DUQUE DE CORNU ALLES` and `DUQUE DE ALBANIA` [p.243; GROK_FOLD_TABLES_measured.md:70-72]. On p.249 the cue is `ALB . Y CORN.` and the line is `Deteneos, señor.` [n=1; GROK_FOLD_TABLES_measured.md:87]. Hugo and Rusconi print the same joint -- `ALBANY ET CORNOUAILLES: Cher sire, arrêtez .` / `ALB. E CORN: Amato sire, fermatevi.` -- and bind it to BOTH [fr/king_lear_1_1.txt:49; it/king_lear_1_1.txt:52; manifest rows fr/king_lear 1.1 and it/king_lear 1.1]. The Spanish row ships **unbound**: the joint is real, and neither duke is on the English 1.1 list [GROK_FOLD_TABLES_measured.md:87, 253-254]. Printing the pair is a pattern; inventing a collective owner is not.

**Three letters, two men, one book.** Jaime Clark's 1873 volume binds *La tempestad* and *La noche de Reyes* under one file hash [GROK_FOLD_TABLES_measured.md:41-44, 64; leads.json, Clark 1873, 204pp]. PERSONAJES p.15 lists `SEBASTIAN , su hermano` -- Alonso's brother -- and `FERNANDO, hijo del rey de Nápoles.` On p.18 the storm cues `SEB. ¡Malhaya tu lengua`. On p.122, after `ACTO II`, the same three letters open Viola's brother: `SEB. No quisiera` [GROK_FOLD_TABLES_measured.md:41-44, 155-157]. A Spanish-wide `SEB -> Sebastian` table would be right twice and still wrong. The key is `(sha256, scene)`, never the volume and never the language [GROK_FOLD_TABLES_measured.md:41-46, 226-228].

**The photograph that moved a fold.** Macpherson 3.2 p.424 is the page that settled `Per`. The text layer reads `Per.` over `Dame á Lisandro , si de mí te dueles ; Entrégamelo ya, Demetrio amigo.` The first table treated that as Demetrius [GROK_FOLD_TABLES_measured.md:126]. The crop says the ink is `HER.` -- Hermia asking Demetrius for Lysander -- and the Demetrius fold would have put her speech in his mouth [GROK_FOLD_TABLES_measured.md:8-10]. One three-letter form is not identity. The rows the photograph would not settle (`Hek`, `DER`) were demoted to unbound on the same asymmetry [GROK_FOLD_TABLES_measured.md:11-14, 28-29].

**A name carried as written, then folded.** Ramos prints Ferdinand as `FERNANDO` in the 1914 *Tempestade* -- 3.1 opens `FERNANDO: Ha exercicios que são penosos` [pt/tempest_3_1.txt:1]. The stem test refuses it: eight letters, three shared with FERDINAND [scripts/otr_vendor_scan.py:94-100, 624-630]. A dedicated alias, scoped to one scene and never parked among place-names, was allowed [scripts/otr_vendor_scan.py:86-100]. Both Portuguese Tempest rows record `folds`: `FERNANDO` -> `FERDINAND` [manifest rows pt/tempest 1.2 and 3.1]. Clark's PERSONAJES makes the same naming choice in Spanish. That is a translated name, not a new person.

**A renamed cast is an adaptation.** Nine Hindi cells (Sitaram) were closed on 2026-09-19 because a version that renames the people is not a translation [docs/OTR_STANDING_RULINGS.md:77-81; docs/GO_FORWARD_PLAN.md:129-130]. Two carry measured names: the 1917 As You Like It OCR reads `रसलीना`, `सुशीला`, `खुशीला`, `लीलाधर` [leads.json, hi/as_you_like_it 3.2]; Comedy of Errors locates `अन्तपाल` and `डमरू` as renamed Antipholus and Dromio [leads.json, hi/comedy_of_errors 3.1]. The other seven closed on the same rule, not for lack of a scan [docs/OTR_STANDING_RULINGS.md:80-81]. Castilho's 1874 Portuguese *Midsummer* was refused for a different reason: he worked from a French intermediary, which costs those two cells [leads.json, pt/midsummer 3.1 and 3.2, `excluded`; docs/OTR_STANDING_RULINGS.md:77-79].

**The in-unison dilemma -- four shapes, one problem.** Shakespeare sometimes gives a line to several people at once. The editions do not print that the same way.

| Shape | Edition | What is printed | What was decided |
|---|---|---|---|
| Collective English name | Folger *Macbeth* 1.3 | `ALL, [dancing in a circle]` [macbeth__act1_scene3.txt:56] | Invent one owner called ALL. |
| Collective in the target language | Ramos, 1912, Portuguese *Macbeth* 1.3 | `TODAS TRES: Somos irmãs as tres feiticeiras` [pt/macbeth_1_3.txt:12] | Name the three as a group; the resolver maps `TODAS` through FUNCTION_NAMES to ALL [scripts/otr_vendor_scan.py:74, 645-653; manifest row pt/macbeth 1.3]. |
| Two names joined | Macpherson, 1897, Spanish *Lear* 1.1 p.249 | `ALB . Y CORN. Deteneos, señor.` [GROK_FOLD_TABLES_measured.md:87] | Print both dukes; do not pick one. Ships unbound. |
| Chorus inside the song | Ramos, 1914, Portuguese *Tempest* 1.2 | Inside Ariel's `Desembarca` speech: `Côro (dispersamente) ¡ Escuta, escuta !` [pt/tempest_1_2.txt:99] | The burden is not a person. Folger had already set it as `[Burden dispersedly, within:] Bow-wow.` [tempest__act1_scene2.txt:613], a direction, not a cue. |

The witches are the same problem from the other side. Ramos writes `1.ª FEITICEIRA: Onde foste, minha irmã ?` [pt/macbeth_1_3.txt:1] -- number, then the function word. Menéndez y Pelayo's 1881 Spanish writes `BRUJA 1.ª: ¿Qué has hecho, hermana?` [es/macbeth_1_3.txt:3; manifest row es/macbeth 1.3] -- function word, then the number. The resolver accepts either order because the two languages disagree about which comes first [scripts/otr_vendor_scan.py:25-29, 645-646]. A character who is a number in one language is a word in another; both still have to land on FIRST WITCH.

## Why it matters for a performance (what a listener would hear go wrong)

A radio episode assigns one voice per speaking name. Fold `TISBE` to Flute and the bellows-mender recites Thisbe as himself -- or two voices appear for one woman if `Pla` is folded and `TISBE` is not [GROK_FOLD_TABLES_measured.md:216-219]. Fold `Per` to Demetrius and Hermia asks for Lysander in his voice [GROK_FOLD_TABLES_measured.md:8-10]. Fold `SEB` by language and Viola's brother answers the boatswain. Leave `AJUS` unbound and the joiner still says his wall line; the roll just has no pre-cast voice, which is the cheap error [PROMPT7_grok_a_wrong_fold_is_worse_than_no_fold.md:12-19; docs/GO_FORWARD_PLAN.md:255-261]. `TODAS TRES` bound to ALL is the one unison a voice can own today; `ALB . Y CORN` is the same wish without an owner [manifest row pt/macbeth 1.3; GROK_FOLD_TABLES_measured.md:87].

## For the hosts: three hooks (one line each) and two open questions

1. In one 1873 Spanish book, `SEB.` is the King of Naples' brother on page 18 and Viola's twin on page 122.
2. Nine Hindi "translations" were closed because the people on the page -- `रसलीना`, `अन्तपाल`, `डमरू` -- were no longer Shakespeare's people.
3. The text layer on Macpherson p.424 read Hermia's cue as `Per.`; the photograph showed her asking Demetrius for Lysander.

Open questions: when a line is written for several mouths at once, should a performance split it, pick one voice, or keep the translator's raw label? And if Macpherson and Rusconi both give Snug a speech Folger withholds, whose 3.1 is the scene a listener is hearing?

## Claims register

| oddity or claim | edition (translator, year) | evidence | status |
|---|---|---|---|
| Snug (`AJUSTADO, ebanista`) speaks once in 3.1; Folger seats no SNUG speaker | Macpherson, *Sueño en noche de verbena*, 1897, p.381 and p.416 (`AJUS.`, n=1) | PERSONAJES + `AJUS. Es imposible representar un muro.`; Folger speakers list [midsummer__act3_scene1.provenance.json:16-29]; fold table [GROK_FOLD_TABLES_measured.md:103] | MEASURED |
| Same wall speech given to Snug in Italian | Rusconi, 1838, *Sogno* 3.1 | `SNUG: È impossibile portare un muro sulla scena.` [it/midsummer_3_1.txt:26] | MEASURED |
| Folger gives that wall speech to Snout | Folger *Midsummer* 3.1 | [midsummer__act3_scene1.txt:91-92] | MEASURED |
| `TISBE` / `PÍRAMO` listed as entremés personages apart from Flauta and Borras | Macpherson 1897, p.381; cues in 3.1 (TISBE 3, Pla 1, Pir/PIR 2) | PERSONAJES block + fold table [GROK_FOLD_TABLES_measured.md:94-95, 106-108, 216-219]; p.417 `TISBE.` line | MEASURED |
| Rusconi also cues the interlude by role | Rusconi 1838, 3.1 | `PIR:` / `TIS:` [it/midsummer_3_1.txt:31, 35] | MEASURED |
| Folger keeps the actor (`BOTTOM, [as Pyramus]`) | Folger *Midsummer* 3.1 | [midsummer__act3_scene1.txt:118, 137] | MEASURED |
| `ALB . Y CORN.` / `Deteneos, señor.` is a real joint; ships unbound | Macpherson 1897, *Lear* 1.1 p.249 (n=1) | [GROK_FOLD_TABLES_measured.md:87, 253-254]; Lear 1.1 sidecar omits both dukes [king_lear__act1_scene1.provenance.json:16-26] | MEASURED |
| Same joint line bound to BOTH in French and Italian | Hugo 1865-72; Rusconi 1838 | [fr/king_lear_1_1.txt:49] + manifest `ALBANY ET CORNOUAILLES`; [it/king_lear_1_1.txt:52] + manifest `ALB. E CORN` | MEASURED |
| Folger prints `ALBANY/CORNWALL` for the same speech | Folger *Lear* 1.1 | [king_lear__act1_scene1.txt:257] | MEASURED |
| `SEB.` is two Sebastians in one Clark hash | Clark, *La tempestad / La noche de Reyes*, 1873, pp.18 and 122 | [GROK_FOLD_TABLES_measured.md:41-46]; p.18 `SEB. ¡Malhaya tu lengua`; p.122 `SEB. No quisiera` | MEASURED |
| `Per` on p.424 is Hermia, not Demetrius | Macpherson 1897, *Midsummer* 3.2 p.424 (n=1) | Image-verdict banner [GROK_FOLD_TABLES_measured.md:8-10]; speech `Dame á Lisandro` | MEASURED |
| Unbound costs a voice, never the line; a wrong fold costs a wrong mouth | standing rule, 2026-09-19 | [PROMPT7_grok_a_wrong_fold_is_worse_than_no_fold.md:12-19]; [docs/OTR_STANDING_RULINGS.md:43-52]; [docs/GO_FORWARD_PLAN.md:255-261] | RULING |
| `FERNANDO` carried as printed and folded to FERDINAND | Ramos, *A Tempestade*, 1914, scenes 1.2 and 3.1 | [pt/tempest_1_2.txt:101]; [pt/tempest_3_1.txt:1]; manifest `folds`; [scripts/otr_vendor_scan.py:94-100] | MEASURED |
| Renamed cast is an adaptation; nine Hindi cells closed | Lala Sitaram, various dates 1915-1926 | [docs/OTR_STANDING_RULINGS.md:77-81]; [docs/GO_FORWARD_PLAN.md:129-130]; measured names [leads.json hi/as_you_like_it 3.2; hi/comedy_of_errors 3.1] | RULING (closure); MEASURED (two name lists) |
| Castilho refused as a French intermediary | Castilho, *Sonho*, 1874 (two Midsummer cells) | [leads.json pt/midsummer 3.1 and 3.2 `excluded`]; [docs/OTR_STANDING_RULINGS.md:77-79] | RULING |
| `TODAS TRES` maps to ALL | Ramos, *Macbeth*, 1912, 1.3 | [pt/macbeth_1_3.txt:12]; FUNCTION_NAMES `TODAS` [scripts/otr_vendor_scan.py:74]; manifest row pt/macbeth 1.3 | MEASURED |
| Folger marks the witches' charm as ALL | Folger *Macbeth* 1.3 | [macbeth__act1_scene3.txt:56] | MEASURED |
| `Côro (dispersamente)` sits inside Ariel's `Desembarca` speech | Ramos, *Tempestade*, 1914, 1.2 | [pt/tempest_1_2.txt:99]; Folger burden [tempest__act1_scene2.txt:613] | MEASURED |
| Witch labels reverse ordinal and function across languages | Ramos 1912 (`1.ª FEITICEIRA`); Menéndez y Pelayo 1881 (`BRUJA 1.ª`) | [pt/macbeth_1_3.txt:1]; [es/macbeth_1_3.txt:3]; [scripts/otr_vendor_scan.py:25-29, 645-646] | MEASURED |
| Expected unbound set after the image pass (Lear joint + Midsummer roles + Snug) | Macpherson 1897, the three Spanish cells | Lear 1.1: `ALB . Y CORN`, `REQ`; Midsummer 3.1: `AJUS`, `TISBE`, `TisbE`, `Pla`, `Pir`, `PIR`; Midsummer 3.2: `Hek`, `DER` [GROK_FOLD_TABLES_measured.md:27-29] | MEASURED |
| A translator's extra speech is not a defect of the source | operator, 2026-09-19 | [docs/GO_FORWARD_PLAN.md:293-300] | RULING |
| `SEB` applied by volume alone would still "occur" in the wrong scene | Clark 1873 | Write-time assertion cannot catch it [GROK_FOLD_TABLES_measured.md:226-228] | INFERRED (from the measured collision) |

SOURCES READ: 30
