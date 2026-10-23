"""Evaluate text -> netlist on the frozen handwritten set (nlp/data/handwritten_eval.jsonl).

  python -m circuitvision.nlp.evaluate --out results/nlp_eval

The 30 sentences were written and annotated (tokens, tags, source, circuit
tree) before any change to the NLP code they evaluate. They are independent of
the synthetic training corpus (nlp/corpus.py) and must not be tuned against.

Metrics
  tagging    tagger run on the GOLD tokenization: token accuracy, per-tag
             precision / recall / F1, macro F1
  circuit    full system from the raw sentence: exact match of source value
             and circuit tree (parallel branches unordered, series ordered)
  component  multiset of (class, value) leaves incl. the source: P / R / F1
  netlist    share of generated netlists passing grammar + semantic checks
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ..grammar import validate
from .hmm import tag_report
from .lexicon import TAGS, tokenize
from .parser import parse
from .text2netlist import load_tagger
from .tree import canonical, to_netlist

EVAL = Path(__file__).with_name("data") / "handwritten_eval.jsonl"
LEAF = __import__("re").compile(r"(resistor|capacitor|inductor|diode|voltage_source):([^,()]+)")


def leaves(tree_str: str, src: str | None) -> Counter:
    c = Counter(LEAF.findall(tree_str))
    if src:
        c[("voltage_source", src)] += 1
    return c


def load(path=EVAL):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def evaluate(items, tagger=None) -> dict:
    tagger = tagger or load_tagger()
    gold_tags = [it["tags"] for it in items]
    pred_tags = [tagger.tag(it["tokens"]) for it in items]
    acc, per = tag_report(gold_tags, pred_tags)
    rows, comp = [], Counter()
    for it, pt in zip(items, pred_tags):
        row = {"id": it["id"], "text": it["text"], "gold_tree": it["tree"], "gold_source": it["source"],
               "tag_errors": [f"{w}:{g}->{p}" for w, g, p in zip(it["tokens"], it["tags"], pt) if g != p]}
        try:
            words = tokenize(it["text"])
            src, tree = parse(words, tagger.tag(words))
            row["tree"] = canonical(tree)
            row["source"] = src[2] if src else None
            row["netlist_valid"] = not validate(to_netlist(tree, src))
        except Exception as e:                       # report, never crash the evaluation
            row.update(tree=None, source=None, netlist_valid=False, error=f"{type(e).__name__}: {e}")
        row["exact"] = row["tree"] == it["tree"] and row["source"] == it["source"]
        g = leaves(it["tree"], it["source"])
        p = leaves(row["tree"] or "", row["source"])
        comp["tp"] += sum((g & p).values()); comp["fp"] += sum((p - g).values()); comp["fn"] += sum((g - p).values())
        if not row["exact"]:
            row["error_type"] = ("exception" if "error" in row else
                                 "tokenization" if tokenize(it["text"]) != it["tokens"] else
                                 "tagging" if row["tag_errors"] else
                                 "source" if row["tree"] == it["tree"] else
                                 "components/values" if g != p else "structure")
        rows.append(row)
    P = comp["tp"] / max(1, comp["tp"] + comp["fp"])
    R = comp["tp"] / max(1, comp["tp"] + comp["fn"])
    tags_present = [t for t in TAGS if t in per]
    return {
        "label": "handwritten evaluation set (independent of synthetic training data)",
        "sentences": len(items),
        "tagging": {"token_accuracy": round(acc, 4),
                    "macro_f1": round(sum(per[t][2] for t in tags_present) / len(tags_present), 4),
                    "per_tag": {t: {"precision": round(per[t][0], 3), "recall": round(per[t][1], 3),
                                    "f1": round(per[t][2], 3)} for t in tags_present}},
        "circuit_exact_match": round(sum(r["exact"] for r in rows) / len(rows), 4),
        "component": {"precision": round(P, 4), "recall": round(R, 4),
                      "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0},
        "netlist_valid": round(sum(r["netlist_valid"] for r in rows) / len(rows), 4),
        "error_types": dict(Counter(r.get("error_type") for r in rows if not r["exact"])),
        "sentences_detail": rows,
    }


def to_markdown(r: dict) -> str:
    t = r["tagging"]
    L = [f"# Text -> netlist: {r['label']}\n",
         f"- Sentences: {r['sentences']}",
         f"- Token accuracy (gold tokenization): {t['token_accuracy']:.3f}; macro F1 {t['macro_f1']:.3f}",
         f"- Exact circuit match: {r['circuit_exact_match']:.3f}",
         f"- Component P / R / F1: {r['component']['precision']:.3f} / {r['component']['recall']:.3f} / {r['component']['f1']:.3f}",
         f"- Valid netlists: {r['netlist_valid']:.3f}",
         f"- Error types: {r['error_types'] or 'none'}\n",
         "| tag | P | R | F1 |", "|---|---|---|---|"]
    L += [f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |" for k, v in t["per_tag"].items()]
    L += ["\n## Failures\n", "| # | sentence | expected | got | type |", "|---|---|---|---|---|"]
    for s in r["sentences_detail"]:
        if not s["exact"]:
            L.append(f"| {s['id']} | {s['text']} | {s['gold_source'] or '-'}; {s['gold_tree']} | "
                     f"{s['source'] or '-'}; {s['tree']} | {s.get('error_type')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="write <out>.json and <out>.md")
    a = ap.parse_args()
    r = evaluate(load())
    md = to_markdown(r)
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.with_suffix(".json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        a.out.with_suffix(".md").write_text(md)


if __name__ == "__main__":
    main()
