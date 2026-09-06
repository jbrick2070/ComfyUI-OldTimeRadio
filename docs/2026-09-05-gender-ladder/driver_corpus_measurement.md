# Driver measurement -- the alias corpus (266 real roster rows)

Run by the driver against every `*provenance*.json` under `config/source_banks/`.
Not a fixture: these are the shipped rows.

## 43 rows carry a title in their OWN name

`Mrs. Rachel Lynde`, `Dr. Lira Kell`, `Don Quixote`, `Count Dracula`,
`Madame Valmonde`, `Lord Ronald`, `Dr. Hesselius`, `Mr./Mrs. Otis` ...

The proposed rule ALLOWS an honorific-stripped match against these, because the
row itself carries the rank. **No regression on this set.**

## 44 rows expose a GIVEN-NAME alias -- the collision class

`Anne Shirley`->`anne`, `Ebenezer Scrooge`->`ebenezer`, `Sancho Panza`->`sancho`,
`Dorian Gray`->`dorian`, `Bob Cratchit`->`bob`, `Armand Aubigny`->`armand` ...

## The existing abstention ALREADY covers same-surname collisions

Two live pairs prove it, and they are why the surname half of the fork is not
urgent:

| rows | shared alias | genders | result |
|---|---|---|---|
| `Marilla Cuthbert` / `Matthew Cuthbert` | `cuthbert` | female / male | `ambiguous_join` -> abstain |
| `Mr. Otis` / `Mrs. Otis` | `otis` | male / female | `ambiguous_join` -> abstain |

`_verdict_from` (`nodes/_otr_roster_gender.py:343`) returns
`("unknown","ambiguous_join")` when matched rows disagree. **Working as designed.**

**The Fitzwilliam case escapes it for one reason only: the colliding character has
NO ROW.** There is nothing to disagree with. Any fix must therefore work when the
other person is ABSENT from the roster -- a rule that relies on detecting a
collision between two present rows cannot fix this.

## The cost of the driver's proposed rule, stated honestly

Refusing a given-name alias match when the slot carried a rank the row lacks also
refuses a LEGITIMATE slot like `MISS ANNE` -> `Anne Shirley` (row carries no
title). That is a FALSE NEGATIVE: the join abstains and the 40/40/20 roll decides.

**That trade is the right one for this module.** Its stated philosophy is
"abstain honestly" and "declining is an answer"; a false negative costs a roll,
while the false positive costs a confident WRONG CITATION attached to permanent
provenance. Ranked by damage, abstention wins.
