"""Train the baseline YOLO component detector with a fixed, recorded config.

  python -m circuitvision.train --data data/yolo/dataset.yaml --exp experiments/baseline_v1
  python -m circuitvision.train --data data/yolo/dataset.yaml --exp /tmp/smoke --quick
  python -m circuitvision.train --data data/yolo/dataset.yaml --exp experiments/baseline_v1 --auto

--auto picks the step from the checkpoint on disk: finished run -> record and
exit, interrupted run -> resume from last.pt, nothing yet -> train from scratch.

Everything that defines the run is fixed here (no "auto" choices), written to
<exp>/train_config.json, and ultralytics' own args.yaml is copied next to it.
Device: CUDA (Colab) > MPS (Apple Silicon) > CPU.
"""
import argparse
import json
import platform
import shutil
from pathlib import Path

BASELINE = {
    "model": "yolov8s.pt",          # COCO-pretrained checkpoint
    "epochs": 100,
    "imgsz": 1024,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 30,
    "seed": 0,
    "deterministic": True,
    # augmentation: symbols are valid under flips; value jitter mimics paper/lighting
    "fliplr": 0.5, "flipud": 0.5, "degrees": 5.0, "scale": 0.5, "translate": 0.1,
    "hsv_h": 0.0, "hsv_s": 0.3, "hsv_v": 0.4, "mosaic": 1.0, "close_mosaic": 10,
}


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def environment() -> dict:
    import torch
    import ultralytics
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "device": pick_device()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def checkpoint_state(exp: Path):
    """-> ("fresh", None) | ("resume", epochs_done) | ("done", None) from <exp>/runs/train/weights/last.pt.

    ultralytics strips the optimizer and sets epoch=-1 when a run ends (all epochs
    or early stopping); an interrupted run's last.pt keeps both.
    """
    last = exp / "runs" / "train" / "weights" / "last.pt"
    if not last.exists():
        return "fresh", None
    import torch
    try:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as e:
        raise SystemExit(f"{last} cannot be read ({e}). It was probably cut off by a disconnect. "
                         f"Rename the runs/train folder to keep it for reference, then train again.")
    epoch = ckpt.get("epoch", -1)
    if epoch == -1 and ckpt.get("optimizer") is None:
        return "done", None
    return "resume", epoch + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--exp", required=True, type=Path, help="experiment folder for records")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 epochs, 10%% data")
    ap.add_argument("--batch", type=int, help="override only if the GPU runs out of memory")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from <exp>/runs/train/weights/last.pt")
    ap.add_argument("--auto", action="store_true",
                    help="resume if interrupted, skip if finished, otherwise train")
    a = ap.parse_args()

    if a.auto:
        state, done = checkpoint_state(a.exp)
        print(f"training state: {state}" + (f" ({done} epochs done)" if done else ""), flush=True)
        if state == "done":
            _finish(a.exp / "runs" / "train", a)
            return
        a.resume = state == "resume"

    from ultralytics import YOLO
    if a.resume:                       # same run, same config: ultralytics restores its args
        last = a.exp / "runs" / "train" / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
        _finish(Path(model.trainer.save_dir), a)
        return

    cfg = dict(BASELINE)
    if a.quick:
        cfg.update(epochs=3, fraction=0.1)
    if a.batch:
        cfg["batch"] = a.batch

    a.exp.mkdir(parents=True, exist_ok=True)
    stats = a.data.parent / "conversion_stats.json"
    record = {"config": dict(cfg), "data": str(a.data.resolve()), "environment": environment(),
              "dataset": json.loads(stats.read_text())["splits"] if stats.exists() else None}
    (a.exp / "train_config.json").write_text(json.dumps(record, indent=1))

    model = YOLO(cfg.pop("model"))
    model.train(data=str(a.data), device=pick_device(),
                project=str(a.exp / "runs"), name="train", exist_ok=True, **cfg)
    _finish(Path(model.trainer.save_dir), a)


def _finish(run: Path, a):
    record_path = a.exp / "train_config.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for f in ("args.yaml", "results.csv", "results.png"):
        if (run / f).exists():
            shutil.copy(run / f, a.exp / f)
    best = run / "weights" / "best.pt"
    record["best_checkpoint"] = str(best)
    record["completed"] = True
    record_path.write_text(json.dumps(record, indent=1))
    print(f"best checkpoint: {best}")
    print(f"next: python -m circuitvision.detector_eval --weights {best} "
          f"--data {a.data} --exp {a.exp}")


if __name__ == "__main__":
    main()
