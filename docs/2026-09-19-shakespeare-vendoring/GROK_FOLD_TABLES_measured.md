# Fold tables for the eight Spanish scan cells -- MEASURED (Grok, 2026-09-19)

Read-only measurement, coordinate reader (`0b35b419`), the eight official
page windows. Spoken names are the edition's own PERSONAJES; roster targets
are the English sidecar `name` fields exactly. `*` marks a one-occurrence
OCR form. Scratch was `%TEMP%\otr_p6_fold\`; the repo was untouched.

**THE KEY IS `(sha256, scene_stem)`, NEVER THE VOLUME AND NEVER THE
LANGUAGE.** The form that decides it is `SEB`: the Clark Tempest / Noche de
Reyes volume (`b79223db...`) prints `SEB.` for Alonso's brother from p.18
and for Viola's brother from p.122 -- one hash, two Sebastians. And `BUF`
is Lear's Fool in Macpherson (p.266) and Feste in Clark: a global Spanish
`BUF -> FOOL` is exactly the table the operator forbade.

**Two label corrections found on the way.** `es/midsummer 3.1` returns no
scene for `ESCENA PRIMERA .`; the page prints `ESCENA PRIMERA` with no
period. `es/twelfth_night 1.5` cannot be extracted at all on the FLAT path
with `ESCENA V .`; the coordinate reader extracts it (14,869 chars). The
registry must carry the exact working label per cell.

**Reader vs flattened inventories.** Tempest 3.1 unchanged. Lear lost one
bare `LEAR`. Midsummer 3.2 dropped `HER`/`PUCH` and turned `Den .` into a
false joint. Much Ado 2.3 split two `CLAUD.` and one `Leo.` into `Y ...`
false joints (dialogue starting with "Y", not joints).

## Edition keys

| volume | sha256 | pages |
|---|---|---|
| Macpherson Tomo I | `5c4847276297a7ffcd752535afa7b7b670f714598472eb621f3baf0e7d77aa68` | 472 |
| Clark Tempest / Noche de Reyes | `b79223db195e6b0c75bfed0b14c8d4120171ab0f99d8731e14f269db76f129d3` | 204 |
| Clark Otelo / Mucho ruido | `51adcd8f5ee0af634e6ae914b6c04356c2bd8f67320ace98e6606c435d0f44c3` | 294 |

## 1. es/king_lear 1.1 -- Macpherson, pages 244-255

PERSONAJES p.243: LEAR, REY DE FRANCIA, DUQUE DE BORGOÑA, DUQUE DE CORNU
ALLES, DUQUE DE ALBANIA, CONDE DE KENT, CONDE DE GLOSTER, EDMUNDO,
GONERILA, REGANIA, CORDELIA, BUFÓN. Scene roster has no Albany, no
Cornwall, no Fool.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `LEAR .` / `LEAR.` | 17+6 | LEAR | LEAR | PERSONAJES; opens the love-test |
| `CORD.` / `CORD .`* | 11+1 | CORDELIA | CORDELIA | PERSONAJES; "Nada ." after Lear's "¿Qué? ¿nada?" |
| `KENT.` / `Kent.` | 9+2 | KENT | KENT | PERSONAJES |
| `LENT.`* | 1 | KENT | KENT | After `EL REY LEAR .`; "No. Dad la muerte" is Kent's physician speech |
| `GLÓS.` / `Glós.` | 4+4 | GLOSTER | GLOUCESTER | PERSONAJES "CONDE DE GLOSTER" |
| `Edm.` / `Edm .` / `EDM .`* | 1+1+1 | EDMUNDO | EDMUND | PERSONAJES; answers Kent |
| `Gon.` / `Gon .`* / `GoN.`* / `Gov.`* | 4+1+1+1 | GONERILA | GONERIL | `Gov.` is "despidiendo al Rey de Francia" -- Goneril's close |
| `REG .` / `Reg.` / `Reg .`* | 2+2+1 | REGANIA | REGAN | PERSONAJES |
| `REQ .`* | 1 | REGANIA | REGAN | "Pensaremos en ello" then `GoN.` -- Folger Regan/Goneril close |
| `R. DE F.`* / `R.DEF`* / `R.DEF.` | 1+1+2 | FRANCIA | FRANCE | Title-initial of "REY DE FRANCIA"; "Dulce Cordelia" |
| `Borg .`* / `Borg.`* / `BORG.`* / `Dorg .`* | 1+1+1+1 | BORGOÑA | BURGUNDY | dower dialogue with Lear |
| `ALB . Y CORN.`* | 1 | ALB . Y CORN | **UNBOUND** | Real joint; "Deteneos, señor." Neither name is on the 1.1 sidecar |

Do not load: `REGANIA`* (entry direction), `FRANCIA`* (entry with
Burgundy), `Duquesa de Borgoña`* (Lear's address, not a cue).

## 2. es/midsummer 3.1 -- Macpherson, pages 414-421 (label `ESCENA PRIMERA`, no period)

PERSONAJES p.381 lists the mechanicals by trade, then separately "PÍRAMO,
TISBE, MURO ... Personajes del entremés".

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `BORR.` / `BORR .` / `BORA.`* / `Boar.`* | 14+3+1+1 | BORRAS | BOTTOM | "BORRAS, tejedor"; calls himself "Borras el tejedor" |
| `MEM .` / `MEM.` / `Mem.` / `Mem .` | 5+2+4+2 | MEMBRILLO | QUINCE | "MEMBRILLO, carpintero"; "Pedro Membrillo" in Bottom's mouth |
| `Hoc.` | 4 | HOCICO | SNOUT | "HOCICO, calderero"; lion/wall fears |
| `Han .`* / `Ham .`* | 1+1 | HAMBRON | STARVELING | "HAMBRÓN, sastre" |
| `AJUS.`* | 1 | AJUSTADO | **UNBOUND** | "AJUSTADO, ebanista". Folger 3.1 has no SNUG speaker; translator seated him |
| `Puck.`* / `Puck .`* / `Puch.`* | 1+1+1 | PUCK | ROBIN | "PUCK, ó el buen Robin" |
| `Tit.` / `Tit .`* / `Tr .`* | 2+1+1 | TITANIA | TITANIA | `Tr . (Despertandose.)` is her waking speech |
| `TISBE.` / `TisbE.`* | 2+1 | TISBE | **UNBOUND** | PERSONAJES lists TISBE as entremés, separate from FLAUTA |
| `Pla .`* | 1 | TISBE | **UNBOUND** | Same rehearsal beat; role, not actor |
| `Pir .`* / `PIR .`* | 1+1 | PIRAMO | **UNBOUND** | PÍRAMO listed next to TISBE, separate from BORRAS |
| `Col.` / `Chi.`* | 2+1 | CHICHARILLO | PEASEBLOSSOM | Titania summons Chicharillo first |
| `Tel.` / `Tel .`* / `TEL.`* | 1+1+1 | TELARAÑA | COBWEB | self-names "Telaraña" |
| `POL .`* / `POL.`* | 1+1 | POLILLA | MOTE | third fairy in the four-reply block |
| `Mos.` / `Mos .` / `MOSTAZA.`* | 2+1+1 | MOSTAZA | MUSTARDSEED | fourth fairy |
| `Todos.`* | 1 | TODOS | ALL | "¿Qué hacemos?" after the four fairies |

`Tel. Y yo .` and `Mos . Y yo .` are not joints -- they are fairy "and I."

## 3. es/midsummer 3.2 -- Macpherson, pages 421-440

PERSONAJES: LISANDRO, DEMETRIO, HERMIA, ELENA, OBERÓN, PUCK.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `Lis.` / `Lis .` / `Lrs .`* / `Los.`* | 11+9+1+1 | LISANDRO | LYSANDER | `Lrs . Sepárate, enanilla` is Lysander to Hermia |
| `ELEN.` / `ELEN .` / `Elen.` / `Blen .`* | 7+7+2+1 | ELEN | HELENA | `Blen . Nunca dos burladores` is her speech |
| `HER .` / `HER.` / `Her .` / `Her.`* / `HEK .`* / `Hek.`* / `Hør.`* / `UER .`* | 7+4+2+1+1+1+1+1 | HERMIA | HERMIA | `HEK . Perro cruel` after Demetrius's hounds |
| `Dem.` / `Dem .` / `DEM .` / `DEM.`* / `DEN .` / `Der .`* / `DER .`* / `Per.`* | 7+6+3+1+2+1+1+1 | DEMETRIO | DEMETRIUS | `Den . Y si pudiera` is Demetrius, not a joint |
| `OBER.` / `OBER .` | 7+2 | OBERON | OBERON | PERSONAJES |
| `Puck .` / `Puck.` / `PUCK .`* / `PUCK.`* / `Prck .`* / `Pock .`* | 8+3+1+1+1+1 | PUCK | ROBIN | `PUCK . Rey de las sombras` is Robin to Oberon |

## 4. es/tempest 3.1 -- Clark, pages 58-62

PERSONAJES p.15: FERNANDO, MIRANDA, PRÓSPERO. No count moved vs flattened.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `FER .` / `Fer .` | 6+4 | FERNANDO | FERDINAND | log-bearing opener |
| `MIR .` / `Mir .` / `Mır .`* | 5+4+1 | MIRANDA | MIRANDA | `Mır` is U+0131 in `Mir` |
| `Min .`* | 1 | MIRANDA | MIRANDA | between two `FER`; "Mi indignidad: hacer oferta no oso" is hers; no other roster name can take it. THE WEAKEST ROW IN THIS FILE: PERSONAJES + alternation + text, not a photograph |
| `Prós.` | 3 | PROSPERO | PROSPERO | asides, then the close about supper |

## 5. es/twelfth_night 1.5 -- Clark, pages 110-122 (COORDINATE READER ONLY; flat cannot extract it)

PERSONAJES p.99: "Feste, bufon", "Don TOBIAS Regueldo", OLIVIA, VIOLA,
MARÍA, MALVOLIO.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `Oliv .` / `Oliv.` / `OLIV .` / `OLIV.` / `Oliy .` / `Oliy.`* | 19+14+5+3+6+1 | OLIVIA | OLIVIA | PERSONAJES |
| `Viol.` / `Viol .` / `VIOL.` / `VIOL .`* | 10+6+4+1 | VIOLA | VIOLA | PERSONAJES |
| `MAL.` / `Mal.`* / `Mal .`* | 7+1+1 | MALVOLIO | MALVOLIO | after "Sale MalvoLIO" |
| `Mar.` / `Mar .` / `MAR.` | 6+3+3 | MARIA | MARIA | opens with the Fool |
| `BUF.` / `BUF .` / `Buf.` / `Buf .`* / `BuF.` / `Bur.` | 6+2+6+1+3+4 | BUF | FOOL | one clown only; `Bur.` answers Maria and later "cuculus non" -- Fool speeches, not a second man. THIS IS "BUF+BUR IS ONE FOOL" |
| `D . Tob.` / `D . Tob .`* / `D . TOB .`* | 2+1+1 | TOB | TOBY | "Don TOBIAS"; eructation |

Do not load: `Salen Maria y el BUFON`* (entry direction). `SEB` on page
122 is OUTSIDE this extract (2.1, after `ACTO II`) though inside the page
window.

## 6. es/twelfth_night 2.5 -- Clark, pages 136-144

Roster has no Fool. No count moved.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `D . TOB.` / `D . TOB .` / `D . Tob.` / `D . Tob .` / `D , TOB.`* / `D. TOB.`* | 11+5+5+5+1+1 | TOB | TOBY | box-tree comments |
| `D . AND.` / `D . AND .` / `D .AND .`* / `D . And.`* | 7+3+1+1 | AND | ANDREW | "Don ANDRÉS DE SecoROSTRO" |
| `MAL.` / `MAL .` / `MAL`* / `MALV .` / `Malv.`* / `MAJ .`* | 13+4+1+2+1+1 | MALVOLIO | MALVOLIO | `MAJ . Luego sigue una I` is the letter M,O,A,I -- only Malvolio reads it |
| `FAB.` / `FAB .` / `Fab.`* | 12+4+1 | FABIO | FABIAN | "FABIO" |
| `MAR .`* / `Mar.`* / `Max .`* | 1+1+1 | MARIA | MARIA | `Max . ... ¿le hace efecto ?` is Maria's "does it work upon him?" |

`Tor` / `Toe` / `Tok` / `ToB` are Toby OCR elsewhere in this volume (pp.
129, 132, 158, 162, 164) and NOT in this window. Do not load them here.

## 7. es/much_ado 2.3 -- Clark Otelo volume, pages 220-230

PERSONAJES p.189: Don PEDRO, CLAUDIO, Benito, LEONATO, BALTASAR, "Un
paje", BEATRIZ.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `D . Ped .` / `D . Ped.` / `D . PED.` / `D . PED .` / `D . ÞED .`* / `D . Pes .`* | 11+6+3+3+1+1 | PED | PRINCE | "Don PEDRO". `ÞED` is thorn-for-P; `Pes` is "Es un Héctor" -- both Prince lines |
| `CLAUD.` / `CLAUD .` | 11+6 | CLAUDIO | CLAUDIO | `CLAUD. Y es` / `Y le` are Claudio starting with "Y", not joints |
| `Ben.` / `BEN .` / `Ben .`* | 8+2+1 | BENITO | BENEDICK | "Benito" |
| `Leo .` / `LEO.` / `LEO .` / `Leo.` / `Leonato .`* | 4+4+3+2+1 | LEONATO | LEONATO | `Leo. Y esto` is Leonato, not a joint |
| `Balt.` / `BAL.`* / `BALT.`* | 4+1+1 | BALTASAR | BALTHASAR | the song |
| `PAJE.`* / `PAJE .`* | 1+1 | PAJE | BOY | "Un paje"; "¿Señor ?" after Benedick's "¡Rapaz!" |
| `Bea.`* / `Bea .`* / `BEA .`* | 1+1+1 | BEATRIZ | BEATRICE | after "Sale BEATRIZ" |

`Pep` / `Per` / `Pen` / `Dep` are Pedro OCR elsewhere in this volume, not
in 2.3.

## 8. es/much_ado 3.1 -- Clark Otelo volume, pages 230-234

No count moved. `D . Pep . Sácala` is on page 234 AFTER `ESCENA II` (3.2)
and is not in this extract.

| printed form | n | spoken | roster | evidence |
|---|---:|---|---|---|
| `HERO.` / `HERO .` / `Hero.`* / `Hero .`* | 7+4+1+1 | HERO | HERO | opens the gulling |
| `URS.` / `URs.` / `URS .`* | 7+2+1 | URSULA | URSULA | "ÚRSULA". `URS. Y si` is Ursula, not a joint |
| `MARG .`* | 1 | MARGARITA | MARGARET | "La haré bajar" then exit |
| `BEA .`* | 1 | BEATRIZ | BEATRICE | "(Se adelanta .)" after Hero and Ursula leave |

## The rule for translator's part vs OCR damage (measured against PERSONAJES)

* Listed as its own character AND a translation of a sidecar name -> fold,
  bind (`MEMBRILLO -> QUINCE`, `PAJE -> BOY`, `BUF -> FOOL`).
* Listed as its own character AND a role, or a part the sidecar does not
  seat in this scene -> keep, UNBOUND (`TISBE`, `PIRAMO`, `AJUSTADO`,
  `ALB . Y CORN`).
* NOT listed, one-off, sits on a Folger speech already owned by a listed
  sibling -> OCR, fold (`LENT -> KENT`, `Min -> MIRANDA`, `MAJ -> MALVOLIO`,
  `REQ -> REGANIA`).
* "Same three letters" is never the evidence.

Both Thisbe forms (`TISBE`, `Pla`) stay UNBOUND together: `TISBE -> FLUTE`
would steal Flute's actor voice and merge the role the translator printed
separately; `Pla -> FLUTE` with `TISBE` unbound would invent a second voice
for one woman.

## The write-time assertion

Refuse unless every loaded printed form occurs in THAT extract, under
THAT sha256, at count >= 1, each row carrying a witness prefix that still
matches a line. Catches `Tor`/`Toe`/`Tok` copied into 2.5, `D . Pep` from
3.2 copied into 3.1, and the bare `LEAR` the coordinate reader dropped. It
CANNOT catch `SEB` applied to the wrong scene in the same book (the form
still occurs) -- which is why the scene key is not optional.

## Voice cost, folded vs roster

| cell | roster | voices after fold | without fold | ghosts closed by fold? |
|---|---:|---:|---:|---|
| king_lear 1.1 | 9 | 10 (9 + joint UNBOUND) | ~16 | yes; leftover is Albany+Cornwall |
| midsummer 3.1 | 12 | 14 (11 bound + TISBE + PIRAMO + AJUSTADO) | ~20 | yes; the extra two are roles + Snug |
| midsummer 3.2 | 6 | 6 | ~16 | yes -- the OCR-ghost cell |
| tempest 3.1 | 3 | 3 | 4 | yes |
| twelfth_night 1.5 | 6 | 6 | 7 (`BUF`+`BUR`) | yes |
| twelfth_night 2.5 | 5 | 5 | 7 | yes |
| much_ado 2.3 | 7 | 7 | 10 | yes |
| much_ado 3.1 | 4 | 4 | 4 | already clean |

## Refuted (by the measurer, of its own brief)

* Official midsummer 3.1 label `ESCENA PRIMERA .` returns a scene -- it
  does not; the heading is `ESCENA PRIMERA`.
* Flat and coordinate inventories match on TN 1.5 -- flat cannot extract
  `ESCENA V .`; the 117-mark table is a coordinate-reader result.
* `Tor`/`Toe`/`Tok` sit in the 2.5 window -- they do not.
* `Pep`/`Per` sit in the 2.3 window -- only `Pes` and `ÞED` do.
* `SEB` "later in the same Clark book" is the same man -- it is not.
* A sha256-only edition table is enough -- one hash, two Sebastians.
* `ALB . Y CORN` can bind through a place-name fold -- absent from the
  sidecar; unbound is the honest row.
* Page images were not re-opened for every one-off; `Min` is the weakest
  row and is disclosed as such.
