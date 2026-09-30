import json
import os
import subprocess
import sys
from pathlib import Path

from circuitvision.evaluate import norm_text, orientation_stage
from circuitvision.types import Detection

from fake_cghd import make


class StubOrient:
    def __init__(self, rot):
        self.rot = rot

    def annotate(self, img, dets):
        for d in dets:
            d.rotation = self.rot


def test_orientation_stage_gt_boxes():
    gt = [Detection("diode", (0, 0, 10, 10), rotation=90), Detection("voltage_source", (20, 0, 30, 10), rotation=0),
          Detection("resistor", (40, 0, 50, 10))]
    assert orientation_stage(StubOrient(90), None, gt) == (1, 2)


def test_orientation_stage_detector_boxes_needs_iou_match():
    gt = [Detection("diode", (0, 0, 10, 10), rotation=90)]
    hit = [Detection("diode", (0, 0, 10, 10), rotation=90)]
    miss = [Detection("diode", (50, 50, 60, 60), rotation=90)]
    assert orientation_stage(StubOrient(90), None, gt, hit) == (1, 1)
    assert orientation_stage(StubOrient(90), None, gt, miss) == (0, 1)


def test_norm_text():
    assert norm_text("10 kΩ") == norm_text("10k") == "10k"


def test_evaluate_cli_writes_summary(tmp_path):
    root = make(tmp_path / "cghd")
    subprocess.run([sys.executable, "-m", "circuitvision.evaluate", "--cghd", str(root),
                    "--drafters", "12", "--exp", str(tmp_path / "exp")], check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    s = json.loads((tmp_path / "exp" / "downstream" / "summary.json").read_text())
    assert s["images"] == 2 and "gt_boxes" in s and "detector" not in s
