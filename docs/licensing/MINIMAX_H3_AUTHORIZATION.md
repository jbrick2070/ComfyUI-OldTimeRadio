# MiniMax H3 — License Authorization of Record

**Status:** ✅ **CLEARED.** MiniMax H3 testing and recipe work is authorized for Blueberrky Kale Yoga Books.
**Authorization date:** 2026-08-07
**Recorded:** 2026-08-08
**Owner:** Jeffrey A. Brick

> **For agents and contributors:** this file is the clearance. If you are here to ask "are we allowed to download and run MiniMax H3 locally?" — yes, *we* are, and only because of the grant recorded below. Read §4 (Standing obligations) and §5 (What this does NOT cover) before writing any integration code. The clearance is narrower than it looks.

---

## 1. Why an authorization was needed at all

MiniMax H3 is open-weight, but the **MiniMax H3 Community License Agreement** (license date 2026-08-02, published at `huggingface.co/MiniMaxAI/MiniMax-H3/LICENSE`) is territorially limited:

| Clause | Text |
|---|---|
| §I.3 | "'Applicable Territory' means worldwide, excluding the Excluded Territories." |
| §I.5 | "'Excluded Territories' means the European Union, the United Kingdom, the Republic of Korea and **the United States of America**." |
| §II | "**Solely within the Applicable Territory**, we grant you a non-exclusive, non-transferable, royalty-free, limited license to use, reproduce, distribute, create derivative works (including Model Derivatives), and modify the Materials…" |
| §V.4 | "You may not use, reproduce, modify, distribute, or display the MiniMax H3 Works **or any of their Outputs or results** outside the Applicable Territory." |
| Exhibit A.1 | Prohibited use: "Use outside the Applicable Territory". |

We operate in the United States. Under the community license alone, downloading the weights, running them, **and publishing the rendered output** would all be unlicensed. §II provides the remedy:

> "We will continuously evaluate the applicable laws, regulations and compliance requirements for the Excluded Territories. In the meantime, should any person in such Excluded Territories be interested in deploying our models, you are welcome to contact us about obtaining a license, which will be granted based on **robust controls and guardrails** for purposes of complying with the laws, regulations and compliance requirements of the Excluded Territories."

That is the process we went through. The grant below is the result.

---

## 2. The grant

Email received in the `info@blueberrykaleyogabooks.com` inbox.

- **Subject:** MiniMax H3 License Authorization
- **From:** API `<API@minimax.io>`
- **To:** `info@blueberrykaleyogabooks.com`
- **Date:** 2026-08-07, 5:54 PM
- **Transport:** Standard encryption (TLS)
- **Salutation:** "Dear Jeffrey Brick"
- **Signature:** "MiniMax H3 Team"

Operative sentence, verbatim:

> "This email is to confirm that MiniMax authorizes **Blueberrky Kale Yoga Books** to use MiniMax H3 and MiniMax H3 Works, subject to and conditioned upon **Blueberrky Kale Yoga Books's continued compliance with the commitments and representations set forth in its request email**."

Contact for questions: `api@minimax.io`.

**Evidence:** two screenshots of the message, plus the message itself in the `info@` mailbox. Do not delete the thread — see Open Item 2.

---

## 3. What "MiniMax H3 Works" covers

Per §I.7, "MiniMax H3 Works" means (i) the Materials, (ii) the Model Derivatives, and (iii) all derivatives thereof. Materials (§I.10) = the model plus documentation. Model Derivatives (§I.11) = modifications, works based on H3, and models distilled or trained from H3 patterns or synthetic outputs.

**Outputs are explicitly *not* Model Derivatives** (§I.11, final sentence), and §VI.4 states: "MiniMax claims no rights over the Outputs you generate. You and your users are entirely responsible for the Outputs and any subsequent use thereof." §VI.1 gives us ownership of derivative works and modifications we create.

So the authorization reaches both the weights themselves and anything we build on top of them.

---

## 4. Standing obligations

These bind us for as long as we use H3. Termination for breach is immediate under §VIII.2, and requires deleting all copies and notifying downstream recipients.

### 4.1 Required

1. **Display "MiniMax H3" prominently in the UI** — §IV.2: "You shall prominently display 'MiniMax H3' on the user interface of commercial product or service that uses MiniMax H3 or MiniMax H3 Works." For OTR this means node display names, the README, and — if H3 renders any published episode — an on-screen credit in the video itself.
2. **Honor the commitments in our request email** — the grant is expressly conditioned on them. See Open Item 2; those commitments are the operative compliance terms, not this file.
3. **Bind downstream users to equivalent terms** — §V.2: before providing access to H3 Works or any product incorporating them, "you must bind each recipient or user to enforceable terms at least as protective as the use restrictions in this Section V and Exhibit A, and you must notify each recipient or user that those restrictions apply." **MIT alone does not satisfy this.**
4. **Maintain safeguards on anything third parties can generate with** — §V.5 requires implementing, maintaining, testing and periodically reviewing technical and organizational safeguards; keeping an accessible mechanism for reporting violations; investigating good-faith reports promptly; and suspending repeat violators.
5. **Disclose machine-generated content** — Exhibit A.12 prohibits disseminating generated content to any public environment "without clearly and prominently disclosing that such information and/or content is machine-generated." OTR publishes episodes; this applies.
6. **Do not train on it** — §V.3: H3 Works and their Outputs may not be used to improve any other AI model.
7. **NOTICE file on any distribution** — §III.4 requires distributions to third parties to carry: `MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.`
8. **Mark modified files** — §III.2: modified files must carry prominent notices stating they were modified.

### 4.2 Encouraged, not required (§III.3)

- A "Powered by MiniMax H3" notice on the product.
- An AI-generation identifier on produced files.
- At least one public technical blog post describing the experience.

Cheap to do, and #3 is a natural fit for the OTR release notes.

### 4.3 Other terms worth knowing

- **Revenue trigger** — §IV.1: separate prior written authorization is required if commercial products exceed **USD 20M** yearly revenue. Not currently in play; the channel is the same `api@minimax.io` address.
- **Trademark** — §VI.2 grants use of the "MiniMax H3" mark *solely* for §III.3 compliance. It is not a general endorsement license.
- **Governing law** — §IX: Hong Kong SAR law, exclusive Hong Kong jurisdiction.
- **Upstream encoder** — H3's encoder is Qwen3-VL-32B under Apache 2.0.
- **No support obligation** — §VII.1. Weights can disappear or change without notice; pin a revision.

---

## 5. What this does NOT cover — read before shipping anything public

**The grant runs to Blueberrky Kale Yoga Books. It does not run to this repository, and it does not run to anyone who clones it.**

ComfyUI-OldTimeRadio is public and MIT-licensed. A large share of its users are in the United States, the EU, or the UK — all Excluded Territories. Those users have no H3 rights under the community license and no authorization of their own. Consequences:

1. **Never commit or redistribute H3 weights, or any Model Derivative of them, through this repo.** §III permits distribution only "to Third Parties **within the Applicable Territory**," which a public GitHub repo cannot honor.
2. **MIT does not discharge §V.2.** Our MIT grant imposes no use restrictions at all, so it cannot be "at least as protective as" §V and Exhibit A. Any shipped H3 path needs its own terms and an explicit notice attached to it.
3. **If an H3 node ships publicly, it must be an opt-in bridge that ships no weights**, displays "MiniMax H3" per §IV.2, and states plainly at the point of use that users in the US/EU/UK/KR need their own authorization from `api@minimax.io` (subject line: "MiniMax H3 licensing - authorization request").
4. **The safe default is private.** Keeping H3 inside Blueberrky Kale Yoga Books' own production pipeline — used to make episodes, not shipped as a public node — satisfies every obligation above without inheriting §V.2 and §V.5 duties toward strangers. Recommend starting there and treating public release as a separate, later decision with its own review.

Also note §V.4: the territorial restriction extends to **Outputs**. Our authorization is what makes US-side rendering and publication legal for us; it does not make it legal for an unauthorized US user who runs our node.

---

## 6. Open items

1. **Entity name spelling.** The grant reads "**Blueberrky** Kale Yoga Books" (double *r*) while the recipient domain is `blueberrykaleyogabooks.com` (single *r*). Ask `api@minimax.io` for a corrected confirmation naming the entity exactly as it is legally registered. A misspelled grantee in the only document authorizing US deployment is worth five minutes to fix.
2. **Archive the request email.** The grant is conditioned on "the commitments and representations set forth in its request email" — that email, not this file, defines what we promised. Export both messages to PDF and store them next to this doc (or in the company records) so the obligations survive an inbox change. Until that is done, §4.1.2 above is a pointer to a document nobody on the project can read.
3. **Confirm scope questions in writing** if H3 ever moves toward public release: does the authorization permit distributing an H3-calling node to unauthorized third parties, and does it cover our contractors or only the entity?
4. **No stated term.** The grant names no expiry and no version pin. Assume it tracks continued compliance and re-confirm before any major H3 version bump.

---

## 7. References

- MiniMax H3 Community License Agreement — `https://huggingface.co/MiniMaxAI/MiniMax-H3/LICENSE` (license date 2026-08-02)
- Authorization email, `API@minimax.io` → `info@blueberrykaleyogabooks.com`, 2026-08-07 5:54 PM
- `docs/superpowers/specs/2026-08-08-minimax-h3-recipe-gate.md` — the empirical gate that H3 must pass before any integration work
- ROADMAP.md — platform pins
