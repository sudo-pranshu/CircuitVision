"""Evaluate a trained detector on the held-out test split and collect evidence.

  python -m circuitvision.detector_eval --weights best.pt --data data/yolo/dataset.yaml \
      --exp experiments/baseline_v1

Writes to <exp>/test/:
  metrics.json          overall + per-class P, R, AP50, AP50-95; image and GT counts
  confusion_matrix.csv  raw counts (rows = predicted, cols = true; last = background)
  *.png                 ultralytics plots (confusion matrices, PR curves)
  per_image.csv         TP / FP / FN / confusions per test image at conf 0.25
  qualitative/<kind>/   drawn examples: correct, missed, false_positive,
                        text_confusion, misclassified, crowded
Drawing key: green = ground truth, blue = correct prediction,
red = false positive, orange = missed ground truth.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .classes import CLASSES

TEXT = CLASSES.index("text")


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt_boxes, gt_cls, pr_boxes, pr_cls, pr_conf, thr=0.5):
    """Greedy match by confidence. -> dict with tp/fp/fn indices and confusions."""
    gt_boxes, pr_boxes = np.asarray(gt_boxes, float).reshape(-1, 4), np.asarray(pr_boxes, float).reshape(-1, 4)
    iou = iou_matrix(pr_boxes, gt_boxes)
    used, tp, fp, conf_pairs = set(), [], [], []
    for p in np.argsort(-np.asarray(pr_conf)):
        cands = [g for g in range(len(gt_boxes)) if g not in used and gt_cls[g] == pr_cls[p] and iou[p, g] >= thr]
        if cands:
            g = max(cands, key=lambda g: iou[p, g])
            used.add(g); tp.append((p, g))
            continue
        fp.append(p)
        others = [g for g in range(len(gt_boxes)) if gt_cls[g] != pr_cls[p] and iou[p, g] >= thr]
        if others:
            g = max(others, key=lambda g: iou[p, g])
            conf_pairs.append((int(gt_cls[g]), int(pr_cls[p])))       # (true, predicted)
    fn = [g for g in range(len(gt_boxes)) if g not in used]
    return {"tp": tp, "fp": fp, "fn": fn, "confusions": conf_pairs}


def crowding(gt_boxes) -> float:
    """Mean over boxes of (box size / distance to nearest other box centre)."""
    b = np.asarray(gt_boxes, float).reshape(-1, 4)
    if len(b) < 2:
        return 0.0
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    size = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(len(b)) * 1e9
    return float(np.mean(size / (d.min(1) + 1e-9)))


def read_gt(label_file: Path, w: int, h: int):
    boxes, cls = [], []
    for line in label_file.read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)); cls.append(int(c))
    return np.array(boxes).reshape(-1, 4), np.array(cls, int)


def draw(img, gt_b, gt_c, pr_b, pr_c, m):
    out = img.copy()
    t = max(2, round(max(img.shape[:2]) / 700))
    box = lambda b, col, th: cv2.rectangle(out, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col, th)
    label = lambda b, s, col: cv2.putText(out, s, (int(b[0]), max(12, int(b[1]) - 4)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, col, max(1, t // 2))
    for g in range(len(gt_b)):
        box(gt_b[g], (0, 170, 0), t)
    for p, _ in m["tp"]:
        box(pr_b[p], (255, 90, 0), max(1, t // 2))
    for p in m["fp"]:
        box(pr_b[p], (0, 0, 230), t); label(pr_b[p], f"FP {CLASSES[pr_c[p]]}", (0, 0, 230))
    for g in m["fn"]:
        box(gt_b[g], (0, 140, 255), t + 1); label(gt_b[g], f"MISS {CLASSES[gt_c[g]]}", (0, 140, 255))
    s = 1400 / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else out


def run_val(model, data: Path, out: Path, imgsz: int):
    met = model.val(data=str(data), split="test", imgsz=imgsz, batch=16, plots=True,
                    project=str(out / "runs"), name="val", exist_ok=True)
    per_class = {}
    for i, c in enumerate(met.box.ap_class_index):
        per_class[CLASSES[int(c)]] = {"precision": float(met.box.p[i]), "recall": float(met.box.r[i]),
                                      "ap50": float(met.box.ap50[i]), "ap50_95": float(met.box.ap[i])}
    overall = {"precision": float(met.box.mp), "recall": float(met.box.mr),
               "map50": float(met.box.map50), "map50_95": float(met.box.map)}
    np.savetxt(out / "confusion_matrix.csv", met.confusion_matrix.matrix, fmt="%d", delimiter=",",
               header=",".join(CLASSES + ["background"]))
    for png in Path(met.save_dir).glob("*.png"):
        shutil.copy(png, out / png.name)
    return overall, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--exp", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for qualitative analysis")
    ap.add_argument("--per-kind", type=int, default=6)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    out = a.exp / "test"
    out.mkdir(parents=True, exist_ok=True)
    yolo_root = a.data.parent
    images = sorted((yolo_root / "images" / "test").glob("*"))
    gt_counts = Counter()
    for f in (yolo_root / "labels" / "test").glob("*.txt"):
        gt_counts.update(CLASSES[int(l.split()[0])] for l in f.read_text().split("\n") if l.strip())

    overall, per_class = run_val(model, a.data, out, a.imgsz)

    rows, totals = [], Counter()
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_b, gt_c = read_gt(yolo_root / "labels" / "test" / (img_path.stem + ".txt"), w, h)
        r = model.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        pr_b = r.boxes.xyxy.cpu().numpy(); pr_c = r.boxes.cls.cpu().numpy().astype(int)
        m = match(gt_b, gt_c, pr_b, pr_c, r.boxes.conf.cpu().numpy())
        conf = Counter(m["confusions"])
        text_conf = sum(v for (t, p), v in conf.items() if TEXT in (t, p))
        row = {"image": img_path.name, "gt": len(gt_b), "tp": len(m["tp"]), "fp": len(m["fp"]),
               "fn": len(m["fn"]), "misclassified": sum(conf.values()), "text_confusion": text_conf,
               "crowding": round(crowding(gt_b), 3)}
        totals.update({k: v for k, v in row.items() if k not in ("image", "crowding")})
        totals.update({f"confusion:{CLASSES[t]}->{CLASSES[p]}": v for (t, p), v in conf.items()})
        rows.append((row, img_path, gt_b, gt_c, pr_b, pr_c, m))   # no pixels kept: test photos are large

    with open(out / "per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0][0])); wr.writeheader()
        wr.writerows(r[0] for r in rows)

    kinds = {
        "correct": lambda r: (r["fp"] == 0 and r["fn"] == 0, r["gt"]),
        "missed": lambda r: (r["fn"] > 0, r["fn"]),
        "false_positive": lambda r: (r["fp"] - r["misclassified"] > 0, r["fp"] - r["misclassified"]),
        "text_confusion": lambda r: (r["text_confusion"] > 0, r["text_confusion"]),
        "misclassified": lambda r: (r["misclassified"] - r["text_confusion"] > 0, r["misclassified"]),
        "crowded": lambda r: (r["fp"] + r["fn"] > 0, r["crowding"]),
    }
    for kind, key in kinds.items():
        picked = sorted((x for x in rows if key(x[0])[0]), key=lambda x: -key(x[0])[1])[:a.per_kind]
        (out / "qualitative" / kind).mkdir(parents=True, exist_ok=True)
        for row, img_path, gt_b, gt_c, pr_b, pr_c, m in picked:
            img = cv2.imread(str(img_path))
            cv2.imwrite(str(out / "qualitative" / kind / (Path(row["image"]).stem + ".jpg")),
                        draw(img, gt_b, gt_c, pr_b, pr_c, m), [cv2.IMWRITE_JPEG_QUALITY, 85])

    report = {"weights": a.weights, "split": "test", "imgsz": a.imgsz,
              "test_images": len(images), "gt_objects": sum(gt_counts.values()),
              "gt_per_class": dict(gt_counts), "overall": overall, "per_class": per_class,
              f"error_counts_at_conf_{a.conf}": dict(totals)}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("test_images", "gt_objects", "overall")}, indent=1))
    for c, v in per_class.items():
        print(f"  {c:<15} P {v['precision']:.3f}  R {v['recall']:.3f}  AP50 {v['ap50']:.3f}  AP50-95 {v['ap50_95']:.3f}")


if __name__ == "__main__":
    main()
