"""Build the report's results tables from experiment artifacts only.

  python -m circuitvision.report_tables --exp experiments/baseline_v1 --out results/baseline_v1_tables.md

Every number is read from a file and every table names its source, so the
report stays traceable. Missing artifacts are reported as NOT AVAILABLE, never
estimated. Real-data tables come from --exp; synthetic and NLP tables from
results/ in the repository.
"""
import argparse
import json
from pathlib import Path

NA = "NOT AVAILABLE (artifact missing: experiment not run yet)"
ROOT = Path(__file__).resolve().parents[2]


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def real_tables(exp: Path, label: str) -> list[str]:
    L = [f"## {label}\n"]
    conv = _load(exp / "conversion_stats.json")
    L.append(f"### Dataset split\n\nSource: `{exp.name}/conversion_stats.json`\n")
    if conv:
        L += ["| split | images | objects |", "|---|---|---|"]
        for s, c in conv["splits"].items():
            L.append(f"| {s} | {c.get('images', 0)} | {c.get('objects', 0)} |")
        L.append(f"\nValidation: {', '.join(conv['val_drafters'])}; test: {', '.join(conv['test_drafters'])}. "
                 f"Conversion problems: {conv['problems'] or 'none'}.\n")
    else:
        L.append(NA + "\n")

    m = _load(exp / "test" / "metrics.json")
    L.append(f"### Detection on the test drafter\n\nSource: `{exp.name}/test/metrics.json`\n")
    if m:
        o = m["overall"]
        L += [f"{m['test_images']} test images, {m['gt_objects']} ground-truth objects.\n",
              "| class | P | R | AP@0.5 | AP@0.5:0.95 |", "|---|---|---|---|---|",
              f"| **all** | {_f(o['precision'])} | {_f(o['recall'])} | {_f(o['map50'])} | {_f(o['map50_95'])} |"]
        for c, v in m["per_class"].items():
            L.append(f"| {c} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['ap50'])} | {_f(v['ap50_95'])} |")
        errs = {k: v for k, v in m.items() if k.startswith("error_counts")}
        for k, v in errs.items():
            L.append(f"\nError counts ({k.replace('error_counts_at_', '')}): {v}\n")
    else:
        L.append(NA + "\n")

    d = _load(exp / "downstream" / "summary.json")
    L.append(f"### Downstream pipeline\n\nSource: `{exp.name}/downstream/summary.json`\n")
    if d:
        if "in_scope_images" not in d:
            L.append("Summary is from an older evaluate.py format; re-run `python -m circuitvision.evaluate`.\n")
            return L
        L.append(f"Images: {d['images']} (in-scope only: {d['in_scope_images']}). "
                 f"Rotation convention: {d.get('rotation_convention')}.\n")
        L.append(f"- Orientation: {d.get('orientation')}")
        L.append(f"- OCR: {d.get('ocr')}\n")
        L += ["| mode | subset | images | terminals ok (PROXY) | valid netlist | simulates | "
              "cross-photo consistency (PROXY) |", "|---|---|---|---|---|---|---|"]
        for mode in ("gt_boxes", "detector"):
            for sub in ("all", "in_scope_only"):
                s = d.get(mode, {}).get(sub)
                if s:
                    L.append(f"| {mode} | {sub} | {s['images']} | {_f(s['topology_terminals_ok_PROXY'])} | "
                             f"{_f(s['netlist_valid'])} | {_f(s['simulates'])} | "
                             f"{_f(s['consistency_PROXY']['consistent'])} ({s['consistency_PROXY']['pairs']} pairs) |")
        if "detector" in d:
            L.append(f"\nDetector vs GT-box netlists (isomorphic graphs): "
                     f"{_f(d['detector']['all'].get('isomorphic_to_gt_boxes'))}.\n")
    else:
        L.append(NA + "\n")
    o = _load(exp / "orientation.json")
    if o:
        L.append(f"### Orientation classifier\n\nSource: `{exp.name}/orientation.json`\n")
        L.append(f"Convention: {o['convention']}\n")
        L += ["| class | train crops | test crops | test accuracy (real crops) |", "|---|---|---|---|"]
        for c, v in o["classes"].items():
            L.append(f"| {c} | {v.get('train_crops')} | {v.get('test_crops', '-')} | {_f(v.get('test_accuracy_real'))} |")
    return L


def synthetic_tables(results: Path) -> list[str]:
    L = ["\n## SYNTHETIC / CONTROLLED\n"]
    t = _load(results / "synthetic_topology.json")
    L.append("### Wire tracing on synthetic photos with known graphs\n\nSource: `results/synthetic_topology.json` "
             "(exact terminal-net partition; 5 circuits x 3 seeds per condition)\n")
    if t:
        L += ["| condition | exact-topology rate |", "|---|---|"]
        for cond, row in t["results"].items():
            L.append(f"| {cond} | {sum(row.values()) / len(row):.2f} |")
        L.append(f"| **overall** | **{t['overall']:.3f}** |")
    else:
        L.append(NA + "\n")
    L.append("\n### Text -> netlist (handwritten sentences, independent of the synthetic training corpus)\n")
    L += ["| set | sentences | token acc. | macro F1 | exact circuit | component F1 | valid netlist | source |",
          "|---|---|---|---|---|---|---|---|"]
    for name, label in (("nlp_eval_baseline", "30-sentence set, baseline (before NLP changes)"),
                        ("nlp_eval", "30-sentence set, after changes (optimistic: same error categories)"),
                        ("nlp_eval_fresh", "fresh 20-sentence set, evaluated once")):
        r = _load(results / f"{name}.json")
        if r:
            L.append(f"| {label} | {r['sentences']} | {_f(r['tagging']['token_accuracy'])} | "
                     f"{_f(r['tagging']['macro_f1'])} | {_f(r['circuit_exact_match'])} | "
                     f"{_f(r['component']['f1'])} | {_f(r['netlist_valid'])} | `results/{name}.json` |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path, help="real-data experiment folder (e.g. experiments/baseline_v1)")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tables.md")
    ap.add_argument("--label", default="REAL CGHD (baseline_v1)", help="heading for the --exp tables")
    a = ap.parse_args()
    L = ["# CircuitVision results tables\n",
         "Generated by `python -m circuitvision.report_tables` from artifacts only.\n"]
    L += real_tables(a.exp, a.label) if a.exp else [f"## {a.label}\n", NA + "\n"]
    L += synthetic_tables(a.results)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
