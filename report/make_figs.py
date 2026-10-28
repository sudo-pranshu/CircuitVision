"""Figures for the CircuitVision report, generated from the project code and result files.

  python report/make_figs.py report/fig <synthetic-run-dir> [figure functions...]

fig_app.png (Gradio screenshot) and the console figures were captured from real runs and are
committed as images; fig_detector needs a synthetic-trained model that is not in Git.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from circuitvision import preprocess, synth, netlist, topology          # noqa: E402
from circuitvision.pipeline import run, overlay                        # noqa: E402
from circuitvision.orientation import rotated_samples, positive_side   # noqa: E402

OUT = Path(sys.argv[1])            # e.g. report/fig
OUT.mkdir(parents=True, exist_ok=True)
E2E = Path(sys.argv[2])            # synthetic run folder (synth_best.pt, cghd2/); only fig_detector needs it
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444"})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, BOXFILL = "#1f2937", "#6b7280", "#eef3fb"


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------- block diagrams ----------
def box(ax, x, y, w, h, text, fill=BOXFILL, edge=BLUE, bold=False, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1, color=MUTED))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    w, h = 1.45, 0.72
    img = [("Phone\nphoto", 0.1), ("Preprocess\n(binarize)", 1.75), ("YOLOv8s\ndetection", 3.4),
           ("Orientation\n(HOG+SVM)", 5.05), ("Wire tracing\n(topology)", 6.7), ("OCR +\nvalues", 8.35)]
    ax.text(0.1, 5.75, "Image path (computer vision + image processing)", fontsize=9, color=MUTED, style="italic")
    for t, x in img:
        box(ax, x, 4.8, w, h, t)
    for (_, x1), (_, x2) in zip(img, img[1:]):
        arrow(ax, x1 + w, 4.8 + h / 2, x2, 4.8 + h / 2)
    txt = [("English\nsentence", 0.1), ("Tokenize", 1.75), ("HMM tagger\n(Viterbi)", 3.4), ("Chunk + S/P\nparser", 5.05),
           ("Circuit\ntree", 6.7)]
    ax.text(0.1, 3.55, "Text path (NLP)", fontsize=9, color=MUTED, style="italic")
    for t, x in txt:
        box(ax, x, 2.6, w, h, t, fill="#eefaf5", edge=AQUA)
    for (_, x1), (_, x2) in zip(txt, txt[1:]):
        arrow(ax, x1 + w, 2.6 + h / 2, x2, 2.6 + h / 2)
    back = [("SPICE\nnetlist", 1.75), ("CFG + Earley\nvalidation", 3.4), ("ngspice\n(.op)", 5.05),
            ("Compare with\nreference", 6.7), ("Overlay /\nweb demo", 8.35)]
    ax.text(0.1, 1.35, "Shared back end", fontsize=9, color=MUTED, style="italic")
    for t, x in back:
        box(ax, x, 0.4, w, h, t, fill="#fdf1ea", edge=ORANGE)
    for (_, x1), (_, x2) in zip(back, back[1:]):
        arrow(ax, x1 + w, 0.4 + h / 2, x2, 0.4 + h / 2)
    # elbow routes into the netlist box, no crossings: OCR lane low, tree lane high
    ax.plot([9.07, 9.07, 2.85], [4.8, 1.75, 1.75], color=MUTED, lw=1)
    arrow(ax, 2.85, 1.75, 2.85, 0.4 + h)
    ax.plot([7.42, 7.42, 2.1], [2.6, 2.15, 2.15], color=MUTED, lw=1)
    arrow(ax, 2.1, 2.15, 2.1, 0.4 + h)
    save(fig, "fig_architecture.png")


def fig_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    steps = [("Download\n+ MD5", BOXFILL, BLUE), ("Dataset\naudit", BOXFILL, BLUE),
             ("GATE 1\nrotation", "#fff4e0", "#c98500"), ("YOLO\nlabels", BOXFILL, BLUE),
             ("GATE 2\noverlays", "#fff4e0", "#c98500"), ("Train\nYOLOv8s", BOXFILL, BLUE),
             ("Test\ndrafter 12", BOXFILL, BLUE), ("Down-\nstream", BOXFILL, BLUE),
             ("Evidence\nzip", BOXFILL, BLUE)]
    w, gap = 0.98, 0.1
    for i, (t, f, e) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        box(ax, x, 1.0, w, 1.2, t, fill=f, edge=e, fs=7.5, bold=t.startswith("GATE"))
        if i:
            arrow(ax, x - gap, 1.6, x, 1.6)
    ax.text(5, 0.45, "Gates are assert cells: execution stops until a human records approval and notes in gates/gate*.json.",
            ha="center", fontsize=8, color=MUTED)
    save(fig, "fig_protocol.png")


# ---------- image processing ----------
def fig_preprocess():
    c = synth.bridge()
    img, dets = synth.render(c, scale=3.0, grid=True, shadow=True, noise=8.0, blur=3, jpeg=80, seed=1)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    k = max(15, (min(gray.shape[:2]) // 15) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dark = (255.0 * cv2.subtract(paper, gray) / np.maximum(paper, 1)).astype(np.uint8)
    mask = preprocess.binarize(img)
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2))
    for a, (im, t) in zip(axs, [(rgb(img), "(a) input: grid paper,\nshadow, noise, JPEG"),
                                (paper, "(b) paper estimate\n(grey closing)"),
                                (255 - dark, "(c) contrast relative\nto paper"),
                                (255 - mask, "(d) binary ink mask\n(Otsu, floor 12)")]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t, fontsize=8); a.axis("off")
    save(fig, "fig_preprocess.png")


def fig_tracing():
    c = synth.bridge()
    img, dets = synth.render(c, scale=3.0, grid=True, seed=2)
    info = {}
    comps, labels = run(img, dets, info)
    vis = overlay(img, comps, labels)
    spice = netlist.to_spice(comps, title="bridge (synthetic)", notes=[f"ground: {info['ground']}"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    axs[0].imshow(rgb(vis)); axs[0].axis("off")
    axs[0].set_title("Symbols masked, wire regions labelled\n(one colour per net)", fontsize=8.5)
    axs[1].axis("off")
    axs[1].text(0.0, 0.95, spice.replace("* ground: implicit: first source, second terminal", "* ground: implicit (first source,\n*         second terminal)"), family="Liberation Mono", fontsize=10.5, va="top", color=INK)
    axs[1].set_title("Generated netlist", fontsize=8.5)
    save(fig, "fig_tracing.png")
    return spice


def fig_conditions():
    names = ["clean", "grid_paper", "shadow", "pencil_on_grid", "pen_gaps", "wobbly_lines"]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.3))
    for a, n in zip(axs.ravel(), names):
        img, _ = synth.render(synth.crossover(), seed=0, **synth.CONDITIONS[n])
        a.imshow(rgb(img)); a.set_title(n.replace("_", " "), fontsize=8.5); a.axis("off")
    save(fig, "fig_conditions.png")


def fig_orientation():
    c = synth.parallel_rc()
    img, dets = synth.render(c, scale=3.0, seed=0)
    d = next(d for d in dets if d.cls == "diode")
    x1, y1, x2, y2 = d.box
    crop = img[max(0, y1 - 6):y2 + 6, max(0, x1 - 6):x2 + 6]
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 1.9))
    for a, (im, r) in zip(axs, rotated_samples(crop, 0)):
        a.imshow(rgb(im)); a.axis("off")
        a.set_title(f"{r}°: + side {positive_side('diode', r)}", fontsize=10)
    save(fig, "fig_orientation.png")


def fig_detector():
    from circuitvision.detect import Detector
    det = Detector(str(E2E / "synth_best.pt"))
    xml = E2E / "cghd2/drafter_12/annotations/C1205_D1_P1.xml"
    imgp = next((E2E / "cghd2/drafter_12/images").glob("C1205_D1_P1.*"))
    img = cv2.imread(str(imgp))
    from circuitvision.pipeline import detections_from_xml
    gt = detections_from_xml(xml)
    pr = det(img)
    vis_gt, vis_pr = img.copy(), img.copy()
    for d in gt:
        cv2.rectangle(vis_gt, d.box[:2], d.box[2:], (60, 160, 30), 3)
    for d in pr:
        col = (214, 120, 42) if d.cls != "crossover" else (52, 104, 235)
        cv2.rectangle(vis_pr, d.box[:2], d.box[2:], col, 3)
    found = {d.cls for d in pr}
    for d in gt:
        if d.cls == "crossover" and "crossover" not in found:
            cx, cy = (d.box[0] + d.box[2]) // 2, (d.box[1] + d.box[3]) // 2
            cv2.circle(vis_pr, (cx, cy), 55, (40, 40, 220), 4)
            cv2.putText(vis_pr, "missed crossover", (cx + 60, cy + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 40, 220), 3)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    axs[0].imshow(rgb(vis_gt)); axs[0].axis("off"); axs[0].set_title("(a) ground-truth boxes", fontsize=8.5)
    axs[1].imshow(rgb(vis_pr)); axs[1].axis("off"); axs[1].set_title("(b) detector boxes (conf ≥ 0.35)", fontsize=8.5)
    save(fig, "fig_detector_failure.png")
    return [(d.cls, d.box, round(d.conf, 2)) for d in pr], [(d.cls, d.box) for d in gt]


# ---------- charts ----------
def fig_topology_chart():
    r = json.load(open(REPO / "results/synthetic_topology.json"))["results"]
    conds = list(r)
    vals = [float(np.mean(list(r[c].values()))) for c in conds]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    y = np.arange(len(conds))[::-1]
    ax.barh(y, vals, height=0.6, color=BLUE)
    ax.set_yticks(y, [c.replace("_", " ") for c in conds], fontsize=10.5)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Exact-topology rate (5 circuits × 3 seeds per condition)")
    for yy, v in zip(y, vals):
        if v < 1:
            ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(1, color="#ccc", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
    save(fig, "fig_topology_chart.png")


def fig_nlp_chart():
    sets = [("30-sentence, before", "nlp_eval_baseline.json"), ("30-sentence, after (optimistic)", "nlp_eval.json"),
            ("Fresh 20-sentence (once)", "nlp_eval_fresh.json")]
    metrics = [("token_accuracy", "Token acc."), ("macro_f1", "Macro F1"), ("exact_match", "Exact circuit"),
               ("component_f1", "Component F1")]
    paths = {"token_accuracy": ("tagging", "token_accuracy"), "macro_f1": ("tagging", "macro_f1"),
             "exact_match": ("circuit_exact_match",), "component_f1": ("component", "f1")}
    data = []
    for _, f in sets:
        j = json.load(open(REPO / "results" / f))
        row = []
        for key, _ in metrics:
            v = j
            for k in paths[key]:
                v = v[k]
            row.append(v)
        data.append(row)
    data = np.array(data, float)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y = np.arange(len(metrics))[::-1]
    for i, ((name, _), col, mk) in enumerate(zip(sets, (BLUE, ORANGE, AQUA), ("o", "s", "^"))):
        ax.scatter(data[i], y + (1 - i) * 0.18, s=46, color=col, marker=mk, label=name, zorder=3,
                   edgecolors="white", linewidths=0.8)
    ax.set_yticks(y, [m[1] for m in metrics]); ax.set_xlim(0.75, 1.02); ax.set_ylim(-0.6, len(metrics) - 0.4)
    ax.set_xlabel("Score (axis starts at 0.75)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
    save(fig, "fig_nlp_chart.png")
    return data


def console(name, text, title):
    lines = text.rstrip("\n").split("\n")
    fig = plt.figure(figsize=(6.6, 0.2 + 0.17 * (len(lines) + 2)))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.02, 0.98, title, color="#9cdcfe", family="Liberation Mono", fontsize=8, va="top")
    fig.text(0.02, 0.98 - 1.6 / (len(lines) + 2), text, color="#e6e6e6", family="Liberation Mono",
             fontsize=8, va="top", linespacing=1.25)
    fig.savefig(OUT / name, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    only = sys.argv[3:]
    if only:
        for f in only:
            globals()[f]()
        sys.exit()
    fig_architecture(); fig_protocol(); fig_preprocess(); spice = fig_tracing(); fig_conditions(); fig_orientation()
    fig_topology_chart(); nlp = fig_nlp_chart(); pr, gt = fig_detector()
    json.dump({"tracing_netlist": spice, "nlp": nlp.tolist(), "detector_pred": pr, "detector_gt": gt},
              open(OUT / "fig_data.json", "w"), indent=1, default=str)
