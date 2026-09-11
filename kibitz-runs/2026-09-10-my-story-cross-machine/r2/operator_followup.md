# Operator steering after R2 dispatch

The operator proposes making the first LLM pass decide how to organize a large
input, summarize it into workable chunks, and use additional passes if needed.
Root accepts this direction for review: model chooses organization, actual
tokenizer/provider capacity measures fit. Original manual source remains intact;
working briefs cannot silently change people, relationships, events or selected
acts. Do not call this permission for arbitrary truncation or content rejection.

This modifies the R2 premise of always injecting the entire raw source into
every pass. That is valid when it fits, but cannot handle a source larger than
one actual context window. R3 must resolve progressive source reading/briefing,
source-linked coverage and the existing repair/slot owners, without infinite
summarization loops or claiming a model can inspect text it never received.
The queued A2 real native-HF capacity correction may therefore be a prerequisite
of adaptive input handling rather than later work. No architecture code has
been implemented against the superseded unconditional-full-source assumption.

This file was written after dispatch. Current R2 reviewers were not given it;
do not claim their pending reviews cover this user steering. Ground their output
against input.md, then explicitly submit this delta in the next review input.
