# MiniMax H3 -- authorization attestation (S9 evidence-manifest artifact)

**TEMPLATE / DRAFT FOR THE OPERATOR.** Fill the bracketed fields, delete this
banner and the notes at the bottom, then file the result as
`docs/H3_LICENSE_ATTESTATION.md` and reference it from the S9 evidence
manifest. It replaces filing the raw correspondence.

Why this shape: the grant is CONDITIONED on the commitments in the operator's
own request email, and that email is not in either repo. What the build needs
is not the correspondence itself but a durable, checkable statement of the
terms the code must honor. Raw emails carry personal data (addresses, message
ids, a third party's contact details) that does not belong in a repository
that may be published; an attestation carries the obligations without them.

---

## Attestation

I, [OPERATOR NAME], operator of this installation, attest to the following
regarding the use of MiniMax H3 model weights in this project.

**1. Authorization received.** MiniMax granted written authorization to use
MiniMax H3 and MiniMax H3 Works, by email from an official MiniMax address,
dated [DATE]. The grant names the licensee as [LICENSEE NAME AS WRITTEN IN
THE GRANT]. [If the grant contains a clerical error in the licensee name,
state it plainly here and whether a corrected copy was requested.]

**2. Conditions I committed to.** The authorization is conditioned on the
representations in my request. Those commitments were, in substance:
- [COMMITMENT 1 -- e.g. use confined to local, offline generation on hardware
  I own and operate]
- [COMMITMENT 2 -- e.g. no hosted service, no public inference endpoint, no
  redistribution of weights in any form, quantized or otherwise]
- [COMMITMENT 3 -- e.g. non-commercial use / the exact commercial scope you
  described]
- [COMMITMENT 4 -- any attribution, territory, or content commitments made]
Add or remove lines so this list matches what you actually wrote. Where you
are unsure of the exact wording, say "in substance" as above rather than
paraphrasing a stronger or weaker promise than you made.

**3. Operating constraints this places on the build.** For as long as H3 is
enabled in this project:
- H3 inference runs only on the operator's own hardware, offline.
- H3 weights are never redistributed, republished, or bundled into any
  release artifact of this project.
- No H3-backed hosted service or shared endpoint is exposed.
- [Any commercial-use limitation, stated exactly.]
- If any commitment above ceases to hold, H3 lanes are disabled until the
  authorization is re-confirmed in writing.

**4. Where the primary record lives.** The original authorization and the
request that conditions it are retained by the operator outside this
repository at [LOCATION -- e.g. "operator's mail archive"], and are available
on request for audit. A copy of the grant is archived at
`vram-recipe-lab/docs/H3_LICENSE_GRANT.md`.

**5. Scope of this attestation.** This document records terms, not legal
advice, and does not expand the grant. If the authorization's actual text
conflicts with anything here, the authorization governs.

Signed: [OPERATOR NAME] -- [DATE]

---

## Notes before you file it (delete this section)

- Verify section 2 against the email you actually sent; the grant rests on it,
  so an inaccurate summary is worse than a vague one.
- Do not paste message ids, email addresses, or the counterparty's personal
  details. The point of the attestation is that it can live in a public repo.
- If you are unsure whether your commitments allow the commercial scope you
  intend, that is a question for MiniMax (or a lawyer), not for this file --
  record what you promised, not what you hope it permits.
- Two known artifacts of the grant worth noting in section 1 if still true:
  the licensee name was recorded with a transposed letter, and the grant did
  not restate the geographic terms of the underlying community license.
