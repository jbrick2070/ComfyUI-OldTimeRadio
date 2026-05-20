# Round-robin question -- OTR LLM prompt tier strategy

You are reviewing a design brief from a one-person OTR (old-time-radio) creative-engineering team. The brief is at the end of this message. Read it once. Then output ONLY the structure below. No preamble, no "I'll review the brief now," no restating the question, no closing summary, no "great brief" remarks.

## Required output (strict format)

### 1. Decision
ONE letter: A, B, C, D, E, or F. No explanation.

### 2. Defense (max 3 bullets, one sentence each)
- ...
- ...
- ...

### 3. Riskiest assumption you are making
One sentence.

### 4. Falsification datum
The single measurement that would flip your answer. Format: "If [test] shows [threshold], switch to [option]."

### 5. Agree or disagree with operator hypothesis (Option F)
ONE word: AGREE | DISAGREE | CONDITIONAL. Then ONE sentence.

### 6. Sprint plan (imperative bullets, 5-7 steps)
Numbered. Each step starts with a verb. Each step is one sentence. No nesting.

### 7. Talkie-as-research-lane reframe (brief Q1)
ONE word: VALID | INVALID | INSUFFICIENT_INFO. Then ONE sentence.

### 8. "Small LLMs prefer lean prompts" at OTR catalog spread (brief Q3)
ONE word: GENERALIZES | DOES_NOT_GENERALIZE | DEPENDS. Then ONE sentence with the specific citation, paper, or empirical datum you are relying on.

### 9. Seventh option I have not considered (brief Q4)
Either ONE sentence describing a 7th option, or "NONE."

### 10. Highest-leverage single move
If Jeffrey could only do ONE thing this week, what is it? ONE sentence.

## Hard constraints on your output

- No bullet longer than one sentence
- No "great question," "thoughtful brief," "as you noted," "let me think through this"
- No restating the options or recapping what the brief said
- ASCII only, no em-dashes, use `--` for dashes
- If you genuinely cannot answer a section, write "INSUFFICIENT INFO" -- do not pad
- Total output target: under 400 words
- Output starts with the heading `### 1. Decision` -- nothing before it

---

# BRIEF (read once, then output the structure above)

[paste the contents of docs/2026-05-16-prompt-tier-strategy.md here]
