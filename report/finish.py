"""Post-process the docx (real TOC/LOF/LOT fields, proper tabs) and compute page numbers from its PDF render.

  python finish.py fields <in.docx> <out.docx>
  python finish.py pages  <rendered.pdf> <entries.json> <pages.json>
"""
import json
import re
import subprocess
import sys
import zipfile

INSTR = {"1": 'TOC \\o "1-2" \\h \\z \\u', "2": 'TOC \\h \\z \\c "Figure"', "3": 'TOC \\h \\z \\c "Table"'}


def fields(src, dst):
    zin = zipfile.ZipFile(src)
    zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            x = data.decode("utf8")
            for k, instr in INSTR.items():
                rep = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
                       f'<w:r><w:instrText xml:space="preserve"> {instr} </w:instrText></w:r>'
                       '<w:r><w:fldChar w:fldCharType="separate"/></w:r>')
                x, n1 = re.subn(rf'<w:r><w:t xml:space="preserve">§FB{k}§</w:t></w:r>', lambda m: rep, x)
                x, n2 = re.subn(rf'<w:r><w:t xml:space="preserve">§FE{k}§</w:t></w:r>',
                                '<w:r><w:fldChar w:fldCharType="end"/></w:r>', x)
                assert n1 == 1 and n2 == 1, (k, n1, n2)
            assert "§" not in x
            pat = re.compile(r'<w:t xml:space="preserve">([^<]*)\t([^<]*)</w:t>')
            while pat.search(x):
                x = pat.sub(r'<w:t xml:space="preserve">\1</w:t><w:tab/><w:t xml:space="preserve">\2</w:t>', x)
            x = x.replace('<w:t xml:space="preserve"></w:t>', "")
            x = re.sub(r'<w:highlightCs [^>]*/>', "", x)     # docx-js emits it; not in the schema
            data = x.encode("utf8")
        zout.writestr(item, data)
    zout.close()


def page_texts(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    out = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), pdf, "-"], capture_output=True, text=True).stdout
        out.append([l.strip() for l in t.splitlines() if l.strip()])
    return out


def pages(pdf, entries_json, out_json):
    texts = page_texts(pdf)
    e = json.load(open(entries_json))
    res, missing = {}, []
    body = next(i for i, lines in enumerate(texts) if "Chapter 1: Introduction" in lines)

    def roman(n):
        out = ""
        for v, s in ((40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
            while n >= v:
                out += s; n -= v
        return out
    # page 1 = cover (unnumbered), pages 2.. = i, ii, ...; body starts again at 1
    labels = [None] + [roman(i) for i in range(1, body)] + [str(i - body + 1) for i in range(body, len(texts))]
    # also allow headings that wrap onto two lines
    texts = [lines + [a + " " + b for a, b in zip(lines, lines[1:])] for lines in texts]
    for kind in ("toc", "lof", "lot"):
        for ent in e[kind]:
            key = ent["key"]
            hit = None
            start = 0 if kind == "toc" else body
            for i in range(start, len(texts)):
                lines = texts[i]
                ok = key in lines if kind == "toc" else any(l.startswith(key + " ") or l == key for l in lines)
                if ok and labels[i]:
                    hit = labels[i]
                    break
            if hit is None:
                missing.append(key)
            res[key] = hit or "00"
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"pages: {len(texts)}; resolved {len(res) - len(missing)}/{len(res)}; missing: {missing}")


if __name__ == "__main__":
    {"fields": fields, "pages": pages}[sys.argv[1]](*sys.argv[2:])
