"""Train the YOLO component detector.

  python -m circuitvision.train --data data/yolo/dataset.yaml            # full run
  python -m circuitvision.train --data data/yolo/dataset.yaml --quick    # smoke test

Device is picked automatically: CUDA (Colab) > MPS (Apple Silicon) > CPU.
"""
import argparse


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="3 epochs, 10%% data")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=3 if a.quick else a.epochs,
        fraction=0.1 if a.quick else 1.0,
        imgsz=a.imgsz,
        batch=a.batch,
        device=pick_device(),
        patience=20,
        # Drawings: flips are safe for symbols; colour jitter mimics paper/lighting.
        fliplr=0.5, flipud=0.5, degrees=5, hsv_v=0.4, mosaic=1.0,
        project="runs", name="detect",
    )
    metrics = model.val(split="test")
    print(f"test mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
