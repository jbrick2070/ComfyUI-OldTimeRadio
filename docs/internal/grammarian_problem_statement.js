const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
        PageBreak } = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function hdr(text) {
  return new TableCell({
    borders,
    width: { size: 2340, type: WidthType.DXA },
    shading: { fill: "1A1A2E", type: ShadingType.CLEAR },
    margins: cellMargins,
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", font: "Consolas", size: 20 })] })]
  });
}
function val(text, w) {
  return new TableCell({
    borders,
    width: { size: w || 7020, type: WidthType.DXA },
    margins: cellMargins,
    children: [new Paragraph({ children: [new TextRun({ text, font: "Consolas", size: 20 })] })]
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1A1A2E" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2D4A7A" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "444444" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // ===== TITLE =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [
        new TextRun("OTR v2.0 GRAMMARIAN PASS"),
      ]}),
      new Paragraph({ children: [
        new TextRun({ text: "Problem Statement + Architecture Reference", italics: true, color: "666666" }),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun({ text: "Document for round-robin iteration. Jeffrey reviews, asks for outputs, refines.", size: 20, color: "888888" }),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun({ text: "Last updated: 2026-04-13 | BUG-LOCAL-023 fix session", size: 20, color: "888888" }),
      ]}),

      // ===== WHAT IS IT =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. What Is the Grammarian?")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun("A lightweight LLM copy-edit pass that runs "),
        new TextRun({ text: "after", bold: true }),
        new TextRun(" NAME_CLEANUP and "),
        new TextRun({ text: "before", bold: true }),
        new TextRun(" the JSON parser. It is a "),
        new TextRun({ text: "second-round failsafe", bold: true }),
        new TextRun(", not a rewrite. The grammarian does not make the script sound literary or academic. It catches the kind of rough edges that trip up the downstream BatchBark TTS and parser: run-on sentences that confuse prosody, garbled punctuation from high-temperature generation, and logic contradictions that break immersion."),
      ]}),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "Design intent: ", bold: true }),
        new TextRun("If you read the script before and after the grammarian, it should feel like the same script with fewer stumbles. NOT like an English professor rewrote it. The voice, slang, rhythm, and character quirks must survive untouched."),
      ]}),

      // ===== PIPELINE POSITION =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Pipeline Position")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun("Post-generation pipeline order in story_orchestrator.py:"),
      ]}),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1200, 2400, 5760],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, width: { size: 1200, type: WidthType.DXA }, shading: { fill: "1A1A2E", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "Step", bold: true, color: "FFFFFF", size: 20 })] })] }),
            new TableCell({ borders, width: { size: 2400, type: WidthType.DXA }, shading: { fill: "1A1A2E", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "Pass", bold: true, color: "FFFFFF", size: 20 })] })] }),
            new TableCell({ borders, width: { size: 5760, type: WidthType.DXA }, shading: { fill: "1A1A2E", type: ShadingType.CLEAR }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: "Purpose", bold: true, color: "FFFFFF", size: 20 })] })] }),
          ]}),
          ...[ ["3a.1", "BOLD_NORM", "Strip bold/markdown artifacts"],
               ["3a.2", "WORD_EXTEND", "Extend if <70% target word count"],
               ["3a.3", "ANNOUNCER", "Inject announcer bookends"],
               ["3a.4", "FORMAT_NORM", "Normalize CHARACTER: dialogue format"],
               ["3b", "NAME_CLEANUP", "Python fuzzy-match fix (no LLM)"],
               ["3c", "GRAMMARIAN", "Light grammar/logic polish (LLM, temp 0.3)"],
               ["4", "PARSE", "Convert to structured JSON for BatchBark"],
          ].map(([step, pass_, purpose]) => new TableRow({ children: [
            new TableCell({ borders, width: { size: 1200, type: WidthType.DXA }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: step, font: "Consolas", size: 20 })] })] }),
            new TableCell({ borders, width: { size: 2400, type: WidthType.DXA }, margins: cellMargins,
              shading: step === "3c" ? { fill: "FFF3CD", type: ShadingType.CLEAR } : undefined,
              children: [new Paragraph({ children: [new TextRun({ text: pass_, font: "Consolas", size: 20, bold: step === "3c" })] })] }),
            new TableCell({ borders, width: { size: 5760, type: WidthType.DXA }, margins: cellMargins,
              children: [new Paragraph({ children: [new TextRun({ text: purpose, size: 20 })] })] }),
          ]})),
        ]
      }),

      // ===== MODEL CONSTRAINTS =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Model Constraints")] }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3120, 6240],
        rows: [
          new TableRow({ children: [hdr("Parameter"), new TableCell({ borders, width: { size: 6240, type: WidthType.DXA }, shading: { fill: "1A1A2E", type: ShadingType.CLEAR }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "Value", bold: true, color: "FFFFFF", font: "Consolas", size: 20 })] })] })] }),
          new TableRow({ children: [hdr("Model"), val("google/gemma-4-E4B-it (quantized)", 6240)] }),
          new TableRow({ children: [hdr("Temperature"), val("0.3 (structural / safe)", 6240)] }),
          new TableRow({ children: [hdr("Top-p"), val("0.92 (inherited default)", 6240)] }),
          new TableRow({ children: [hdr("Context cap"), val("6144 tokens (prompt guard truncates)", 6240)] }),
          new TableRow({ children: [hdr("Token budget (single)"), val("min(2048, max(256, len(script) // 3))", 6240)] }),
          new TableRow({ children: [hdr("Token budget (chunk)"), val("min(1024, max(128, len(chunk) // 3))", 6240)] }),
          new TableRow({ children: [hdr("Timeout (single)"), val("150 seconds", 6240)] }),
          new TableRow({ children: [hdr("Timeout (chunk)"), val("90 seconds per chunk", 6240)] }),
          new TableRow({ children: [hdr("Chunk threshold"), val("50+ dialogue lines triggers chunking", 6240)] }),
          new TableRow({ children: [hdr("Chunk split"), val("By === SCENE N === markers; fallback: 40 raw lines", 6240)] }),
          new TableRow({ children: [hdr("Generation speed"), val("~15-16 tok/s on RTX 5080 (Blackwell)", 6240)] }),
          new TableRow({ children: [hdr("VRAM at grammarian"), val("~7.7 GB (model loaded, post-script-gen)", 6240)] }),
        ]
      }),

      // ===== CURRENT PROMPT =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Current Prompt (Verbatim)")] }),
      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Single-pass prompt (scripts under 50 lines)")] }),
      new Paragraph({
        spacing: { after: 200 },
        shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
        children: [new TextRun({ text: `You are a radio drama copy editor. Your ONLY job is to polish
the script below for grammar, punctuation, and natural spoken flow.

RULES:
1. Fix grammar, spelling, and punctuation errors in dialogue lines.
2. Smooth awkward phrasing so every line sounds natural when read aloud.
3. Fix logic errors (character referenced before introduction, contradictions).
4. Break run-on sentences into punchy radio-paced delivery.
5. Keep all character names EXACTLY as they are. Do not rename anyone.
6. Keep ALL [SFX:], [ENV:], [VOICE:], and scene markers EXACTLY as they are.
7. Do NOT add new dialogue lines, scenes, or content.
8. Do NOT remove any dialogue lines - every character line must survive.
9. Do NOT add commentary, notes, or explanations.
10. Output ONLY the polished script. Nothing else.

SCRIPT TO POLISH:
{script_text}`, font: "Consolas", size: 18 })]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Chunked prompt (scripts over 50 lines)")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun("Identical rules, but header says \"script segment\" and input is one scene at a time."),
      ]}),

      // ===== SAFETY RAILS =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. Safety Rails")] }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3120, 6240],
        rows: [
          new TableRow({ children: [hdr("Rail"), new TableCell({ borders, width: { size: 6240, type: WidthType.DXA }, shading: { fill: "1A1A2E", type: ShadingType.CLEAR }, margins: cellMargins, children: [new Paragraph({ children: [new TextRun({ text: "Behavior", bold: true, color: "FFFFFF", font: "Consolas", size: 20 })] })] })] }),
          new TableRow({ children: [hdr("Output too short"), val("Reject if output < 50% of input chars. Keep original.", 6240)] }),
          new TableRow({ children: [hdr("Dialogue line loss"), val("Reject if post-polish lines < 80% of pre-polish lines. Keep original.", 6240)] }),
          new TableRow({ children: [hdr("Timeout"), val("Single: 150s. Per-chunk: 90s. Failed = keep original.", 6240)] }),
          new TableRow({ children: [hdr("Skip threshold"), val("Scripts under 500 chars skip entirely.", 6240)] }),
          new TableRow({ children: [hdr("Chunk failure"), val("Each chunk independent. Failed chunk keeps original text. Others still polish.", 6240)] }),
          new TableRow({ children: [hdr("Reassembly check"), val("Final dialogue count must hold to 80% of pre-total. If not, revert ALL to original.", 6240)] }),
        ]
      }),

      // ===== THE PROBLEM =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. The Problem Statement")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "Current concern: ", bold: true }),
        new TextRun("The grammarian prompt says \"polish\" and \"smooth awkward phrasing\" which risks making dialogue sound overly formal. A cosmic horror episode where a character says \"Ain't no way that thing's natural\" should NOT become \"There is no way that entity is natural.\" The grammarian should fix the typo in \"natrual\" but leave the voice alone."),
      ]}),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "What the grammarian should actually do:", bold: true }),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("1. Fix obvious typos and misspellings that would sound wrong in TTS (\"teh\" not \"the\")"),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("2. Fix broken punctuation that confuses the parser (missing colons after character names, unclosed parentheses)"),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("3. Fix logical contradictions (character in two scenes at once, referencing something that hasn't happened)"),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("4. Break extreme run-on sentences (100+ words with no period) that cause TTS to lose breath"),
      ]}),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun("5. Ensure every line has a valid CHARACTER: prefix so the parser can extract it"),
      ]}),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "What the grammarian must NOT do:", bold: true }),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("1. Rewrite casual dialogue into formal English"),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("2. Remove slang, contractions, or character voice quirks"),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("3. Add fancy vocabulary or literary flourishes"),
      ]}),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun("4. Change sentence structure for \"elegance\""),
      ]}),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun("5. Touch anything that already parses correctly and sounds fine spoken aloud"),
      ]}),

      // ===== CHUNKING ARCHITECTURE =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("7. Chunking Architecture")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "Why chunking: ", bold: true }),
        new TextRun("BUG-LOCAL-023. A 67-line space opera script needed ~136s at 15 tok/s with a 2048 token budget. The old 75s timeout killed it. Rather than just raising the timeout (which wastes VRAM time on failures), we split by scene markers and polish each scene independently."),
      ]}),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "Flow:\n", bold: true }),
        new TextRun("Script has >50 dialogue lines? Split by === SCENE N === markers. No markers? Fall back to 40-line raw chunks. Each chunk gets its own grammarian prompt, its own 90s timeout, its own safety checks. Failed chunks keep original text. Reassemble. Final dialogue count check. Done."),
      ]}),

      // ===== ITERATION SECTION =====
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("8. Round Robin: Prompt Iteration Space")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "This section is for Jeffrey to iterate on. Each version below is a candidate prompt. Ask for a new output and it gets added here.", italics: true, color: "666666" }),
      ]}),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Version A: Current (copy-editor framing)")] }),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun({ text: "Tone: ", bold: true }), new TextRun("Professional copy editor. Risk of over-polishing."),
      ]}),
      new Paragraph({ spacing: { after: 200 }, shading: { fill: "F5F5F5", type: ShadingType.CLEAR }, children: [
        new TextRun({ text: "\"You are a radio drama copy editor. Your ONLY job is to polish the script below for grammar, punctuation, and natural spoken flow.\"", font: "Consolas", size: 18 }),
      ]}),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Version B: Parser failsafe framing")] }),
      new Paragraph({ spacing: { after: 100 }, children: [
        new TextRun({ text: "Tone: ", bold: true }), new TextRun("Technical QA pass. Minimal touch. Parser-first."),
      ]}),
      new Paragraph({ spacing: { after: 200 }, shading: { fill: "F5F5F5", type: ShadingType.CLEAR }, children: [
        new TextRun({ text: `You are a QA checker for a radio drama text-to-speech pipeline. The script below will be fed to a parser that extracts CHARACTER: dialogue lines, then to a TTS engine that reads them aloud.

Your job is to make MINIMAL fixes so the script parses cleanly and sounds correct when spoken. Do NOT rewrite. Do NOT improve style. Do NOT add vocabulary. Leave the voice and personality of every character exactly as written.

FIX ONLY:
1. Typos and misspellings that would sound wrong in TTS.
2. Missing or broken CHARACTER: prefixes that would fail the parser.
3. Unclosed parentheses, missing periods, or garbled punctuation.
4. Contradictions (character in two places, referencing future events).
5. Extreme run-on sentences (80+ words no period) that break TTS breath.

DO NOT TOUCH:
- Slang, contractions, sentence fragments, or character quirks.
- [SFX:], [ENV:], [VOICE:], or === SCENE === markers.
- Anything that already works for the parser and sounds fine aloud.

Output ONLY the fixed script. No commentary.`, font: "Consolas", size: 18 }),
      ]}),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Version C: (Your next iteration goes here)")] }),
      new Paragraph({ spacing: { after: 200 }, children: [
        new TextRun({ text: "Ask Jeffrey for feedback on A vs B, then generate C.", italics: true, color: "999999" }),
      ]}),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "/sessions/nifty-quirky-franklin/mnt/ComfyUI-OldTimeRadio/docs/grammarian_problem_statement.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Written to: " + outPath);
});
