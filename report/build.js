// Build the CircuitVision report (.docx) from content.js.
// usage: node build.js <out.docx> [pages.json]
const fs = require("fs");
const path = require("path");
const d = require("docx");
const C = require("./content.js");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Footer, AlignmentType,
  WidthType, ShadingType, BorderStyle, PageNumber, NumberFormat, TabStopType, LevelFormat,
  SimpleField, HeadingLevel, VerticalAlign, PageBreak,
} = d;

const OUT = process.argv[2];
const PAGES = process.argv[3] && fs.existsSync(process.argv[3]) ? JSON.parse(fs.readFileSync(process.argv[3])) : {};
const FIG = path.join(__dirname, "fig");
const TW = 8666;                      // text width in twips (A4 - 1.25in - 1in)
const FONT = "Times New Roman";
const MONO = "Courier New";

// ---------- pre-pass: figure/table numbers and citation order ----------
const figNum = {}, tabNum = {}, citeNum = {};
const allBlocks = [...C.chapters, ...C.appendices].flatMap(([, blocks]) => blocks);
let nf = 0, nt = 0, nc = 0;
for (const b of allBlocks) {
  if (b[0] === "fig") figNum[b[3]] = ++nf;
  if (b[0] === "table") tabNum[b[1]] = ++nt;
}
function scanCites(s) {
  for (const m of String(s).matchAll(/\{cite:([^}]+)\}/g))
    for (const k of m[1].split(",")) {
      if (!C.refs[k]) throw new Error("unknown ref " + k);
      if (!citeNum[k]) citeNum[k] = ++nc;
    }
}
for (const b of allBlocks) scanCites(JSON.stringify(b));
for (const k of Object.keys(C.refs)) if (!citeNum[k]) throw new Error("uncited ref " + k);

function subst(s) {
  return s
    .replace(/\{fig:(\w+)\}/g, (_, k) => { if (!figNum[k]) throw new Error("fig " + k); return `Figure ${figNum[k]}`; })
    .replace(/\{tab:(\w+)\}/g, (_, k) => { if (!tabNum[k]) throw new Error("tab " + k); return `Table ${tabNum[k]}`; })
    .replace(/\{cite:([^}]+)\}/g, (_, ks) => ks.split(",").map(k => `[${citeNum[k]}]`).join(", "));
}

// ---------- inline markup -> runs ----------
function runs(text, base = {}) {
  const out = [];
  const parts = subst(text).split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[\[[^\]]+\]\])/);
  for (const p of parts) {
    if (!p) continue;
    if (p.startsWith("**")) out.push(new TextRun({ ...base, text: p.slice(2, -2), bold: true }));
    else if (p.startsWith("[[")) out.push(new TextRun({ ...base, text: p.slice(2, -2), bold: true, highlight: "yellow" }));
    else if (p.startsWith("`")) out.push(new TextRun({ ...base, text: p.slice(1, -1), font: MONO, size: (base.size || 24) - 2 }));
    else if (p.startsWith("*") && p.length > 2) out.push(new TextRun({ ...base, text: p.slice(1, -1), italics: true }));
    else out.push(new TextRun({ ...base, text: p }));
  }
  return out;
}

const body = (text, opts = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 360, after: 120 }, ...opts });

// ---------- lists ----------
const numberingConfig = [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] }];
let olCount = 0;
function list(items, ordered) {
  let ref = "bullets";
  if (ordered) {
    ref = `ol${++olCount}`;
    numberingConfig.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] });
  }
  return items.map(t => new Paragraph({ children: runs(t), numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, after: 60 } }));
}

// ---------- tables ----------
const border = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, header) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: header ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(text, { size: 21, bold: header || undefined }),
      spacing: { line: 252, after: 0 } })],
  });
}
function caption(kind, n, text) {
  return new Paragraph({
    style: "Caption", alignment: AlignmentType.CENTER, keepNext: kind === "Table",
    spacing: { before: kind === "Table" ? 200 : 80, after: kind === "Table" ? 80 : 240 },
    children: [new TextRun({ text: `${kind} `, bold: true }),
      new SimpleField(`SEQ ${kind} \\* ARABIC`, String(n)),
      new TextRun({ text: ": ", bold: true }), ...runs(text)],
  });
}
function table(key, cap, headers, rows, widths) {
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== TW) throw new Error(`table ${key} widths sum ${sum} != ${TW}`);
  return [caption("Table", tabNum[key], cap), new Table({
    width: { size: TW, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }))],
  }), new Paragraph({ spacing: { after: 120 }, children: [] })];
}

// ---------- figures ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function figure(file, widthPx, key, cap) {
  const f = path.join(FIG, file);
  const [w, h] = pngSize(f);
  const width = Math.min(widthPx, 575);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f),
      transformation: { width, height: Math.round(width * h / w) } })] }),
    caption("Figure", figNum[key], cap)];
}

function code(text) {
  const lines = text.split("\n");
  return lines.map((l, i) => new Paragraph({
    children: [new TextRun({ text: l.replace(/ /g, " ") || " ", font: MONO, size: 16 })],
    shading: { fill: "F3F4F6", type: ShadingType.CLEAR, color: "auto" },
    spacing: { line: 240, before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0 },
    indent: { left: 120, right: 120 }, keepLines: true, keepNext: i < lines.length - 1,
  }));
}

function blocks(list) {
  const out = [];
  for (const b of list) {
    switch (b[0]) {
      case "h2": out.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b[1])] })); break;
      case "h3": out.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(b[1])] })); break;
      case "p": out.push(body(b[1])); break;
      case "ul": out.push(...list_(b[1], false)); break;
      case "ol": out.push(...list_(b[1], true)); break;
      case "table": out.push(...table(b[1], b[2], b[3], b[4], b[5])); break;
      case "fig": out.push(...figure(b[1], b[2], b[3], b[4])); break;
      case "code": out.push(...code(b[1])); break;
      default: throw new Error("block " + b[0]);
    }
  }
  return out;
}
const list_ = list;

const h1 = (text, opts = {}) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
  children: [new TextRun(text)], ...opts });
const centeredTitle = text => new Paragraph({ style: "PrelimTitle", pageBreakBefore: true, children: [new TextRun(text)] });

// ---------- TOC / LOF / LOT as real Word fields with cached entries ----------
const tocEntries = [];   // {level, text}
const lofEntries = [], lotEntries = [];
function fieldBlock(id, entries, style1, style2) {
  if (!entries.length) return [];
  return entries.map((e, i) => {
    const kids = [];
    if (i === 0) kids.push(new TextRun({ text: `§FB${id}§` }));
    kids.push(new TextRun({ text: e.text }));
    kids.push(new TextRun({ children: ["\t", PAGES[e.key] || "00"] }));
    if (i === entries.length - 1) kids.push(new TextRun({ text: `§FE${id}§` }));
    return new Paragraph({ style: e.level === 2 ? style2 : style1, children: kids,
      tabStops: [{ type: TabStopType.RIGHT, position: TW, leader: "dot" }] });
  });
}

// ---------- assemble ----------
const M = C.meta;
const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
    children: [new TextRun({ text: M.institute, bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 900, after: 300 },
    children: [new TextRun({ text: "MINI PROJECT REPORT", bold: true, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240, line: 360 },
    children: [new TextRun({ text: "“" + M.title + "”", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480, after: 60 },
    children: [new TextRun({ text: "Submitted in partial fulfilment of the requirements for the subject", italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
    children: [new TextRun({ text: M.subject, bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Submitted by", bold: true })] }),
  new Table({ width: { size: 5600, type: WidthType.DXA }, columnWidths: [3400, 2200], alignment: AlignmentType.CENTER,
    rows: [["Name", "Roll Number"], ...M.students].map((r, i) => new TableRow({ children: r.map((c, j) =>
      new TableCell({ width: { size: [3400, 2200][j], type: WidthType.DXA }, borders,
        shading: i === 0 ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: c, bold: i === 0 })] })] })) })) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 480 }, children: [new TextRun(M.programme)] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "Project Guide", bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun(M.guide)] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: M.institute, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun(M.year)] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Date of submission: " }),
    new TextRun({ text: M.date, highlight: "yellow" })] }),
];

const sigLine = (name, role) => [
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 1000, after: 0 }, children: [new TextRun("______________________")] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 0 }, children: [new TextRun({ text: name, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun(role)] }),
];
const datePlace = [
  new Paragraph({ spacing: { before: 600, after: 0 }, children: [new TextRun("Date: ____________________")] }),
  new Paragraph({ children: [new TextRun("Place: Mumbai")] }),
];

function prelimHeading(text, key) {
  tocEntries.push({ level: 1, text, key });
  return h1(text, { alignment: AlignmentType.CENTER });
}

const prelim = [];
prelim.push(prelimHeading("Certificate", "Certificate"), ...blocks(C.prelim.certificate), ...sigLine(M.guide, "Project Guide"), ...datePlace);
prelim.push(prelimHeading("Declaration", "Declaration"), ...blocks(C.prelim.declaration));
prelim.push(new Paragraph({ spacing: { before: 600 }, children: [] }));
prelim.push(new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: [3600, 2200, 2866],
  rows: [["Name", "Roll Number", "Signature"], ...M.students.map(s => [...s, ""])].map((r, i) => new TableRow({
    height: { value: i ? 700 : 400, rule: "atLeast" },
    children: r.map((c, j) => cell(c, [3600, 2200, 2866][j], i === 0)) })) }));
prelim.push(...datePlace);
prelim.push(prelimHeading("Acknowledgement", "Acknowledgement"), ...blocks(C.prelim.ack));
prelim.push(new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 400 }, children: [new TextRun({ text: "Pranshu Kumar (231090054)" })] }));
prelim.push(new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Pawan Shine (231090063)" })] }));
prelim.push(prelimHeading("Abstract", "Abstract"), ...blocks(C.prelim.abstract));
prelim.push(new Paragraph({ spacing: { before: 200 }, alignment: AlignmentType.JUSTIFIED, children: [
  new TextRun({ text: "Keywords: ", bold: true }),
  new TextRun({ text: "hand-drawn circuit recognition, YOLOv8, image processing, wire tracing, OCR, SPICE netlist, context-free grammar, Earley parser, hidden Markov model, ngspice", italics: true })] }));
prelim.push(prelimHeading("List of Abbreviations", "List of Abbreviations"));
prelim.push(new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: [1800, 6866],
  rows: C.abbrev.map(([a, b]) => new TableRow({ children: [cell(`**${a}**`, 1800, false), cell(b, 6866, false)] })) }));

tocEntries.push({ level: 1, text: "List of Figures", key: "List of Figures" });
tocEntries.push({ level: 1, text: "List of Tables", key: "List of Tables" });

// main chapters, appendices and references
const main = [];
for (const [title, bl] of C.chapters) {
  tocEntries.push({ level: 1, text: title, key: title });
  main.push(h1(title));
  for (const b of bl) if (b[0] === "h2") tocEntries.push({ level: 2, text: b[1], key: b[1] });
  main.push(...blocks(bl));
}
tocEntries.push({ level: 1, text: "References", key: "References" });
main.push(h1("References"));
const refOrder = Object.keys(citeNum).sort((a, b) => citeNum[a] - citeNum[b]);
for (const k of refOrder)
  main.push(new Paragraph({ children: [new TextRun(`[${citeNum[k]}]\t`), ...runs(C.refs[k])],
    tabStops: [{ type: TabStopType.LEFT, position: 600 }], indent: { left: 600, hanging: 600 },
    alignment: AlignmentType.LEFT, spacing: { line: 300, after: 100 } }));
for (const [title, bl] of C.appendices) {
  tocEntries.push({ level: 1, text: title, key: title });
  main.push(h1(title));
  main.push(...blocks(bl));
}
for (const b of allBlocks) {
  if (b[0] === "fig") lofEntries.push({ level: 1, text: `Figure ${figNum[b[3]]}: ${subst(b[4])}`, key: `Figure ${figNum[b[3]]}:` });
  if (b[0] === "table") lotEntries.push({ level: 1, text: `Table ${tabNum[b[1]]}: ${subst(b[2])}`, key: `Table ${tabNum[b[1]]}:` });
}

// TOC / LOF / LOT pages (placed after the abbreviations); field instructions are inserted by finish.py
const finalLists = [centeredTitle("Table of Contents"), ...fieldBlock(1, tocEntries, "TOC1", "TOC2"),
  h1("List of Figures", { alignment: AlignmentType.CENTER }), ...fieldBlock(2, lofEntries, "TableofFigures", "TableofFigures"),
  h1("List of Tables", { alignment: AlignmentType.CENTER }), ...fieldBlock(3, lotEntries, "TableofFigures", "TableofFigures")];

const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 22 })] })] });
const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1800, footer: 600 } };

const doc = new Document({
  creator: "Pranshu Kumar, Pawan Shine", title: "CircuitVision Mini Project Report",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true }, paragraph: { spacing: { before: 0, after: 360 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
      { id: "PrelimTitle", name: "Prelim Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 32, bold: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 360 } } },
      { id: "Caption", name: "caption", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 22 } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 100, after: 40 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 23 },
        paragraph: { indent: { left: 360 }, spacing: { after: 30 } } },
      { id: "TableofFigures", name: "table of figures", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 23 },
        paragraph: { indent: { left: 0 }, spacing: { after: 60 } } },
    ],
  },
  numbering: { config: numberingConfig },
  sections: [
    { properties: { page }, children: cover },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: footer }, children: [...prelim, ...finalLists] },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      footers: { default: footer }, children: main },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  fs.writeFileSync(OUT + ".entries.json", JSON.stringify({ toc: tocEntries, lof: lofEntries, lot: lotEntries }, null, 1));
  console.log("wrote", OUT, "figures", nf, "tables", nt, "refs", nc);
});
