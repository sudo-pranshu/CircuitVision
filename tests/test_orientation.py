import cv2
import numpy as np

from circuitvision import orientation as O
from circuitvision import preprocess, topology
from circuitvision.types import Detection


def diode_crop(rng, size=64):
    """Diode at rotation 0: anode (triangle base) left, cathode bar right."""
    img = np.full((size, size, 3), 255, np.uint8)
    j = lambda: int(rng.integers(-3, 4))
    tri = np.array([[16 + j(), 14 + j()], [16 + j(), 50 + j()], [44 + j(), 32 + j()]])
    cv2.polylines(img, [tri], True, (0, 0, 0), 2)
    cv2.line(img, (46 + j(), 14 + j()), (46 + j(), 50 + j()), (0, 0, 0), 3)
    cv2.line(img, (2, 32), (16, 32), (0, 0, 0), 2)
    cv2.line(img, (46, 32), (62, 32), (0, 0, 0), 2)
    return img


def test_positive_side_rotates_ccw():
    assert O.positive_side("diode", 0) == "left"
    assert O.positive_side("diode", 90) == "bottom"
    assert O.positive_side("diode", 180) == "right"
    assert O.positive_side("diode", 270) == "top"
    assert O.positive_side("resistor", 90) is None
    assert O.positive_side("diode", None) is None


def test_hog_svm_learns_rotation():
    rng = np.random.default_rng(0)
    train = [s for _ in range(30) for s in O.rotated_samples(diode_crop(rng), 0)]
    X = np.stack([O.features(c) for c, _ in train])
    y = np.array([r for _, r in train])
    model = O.OrientationModel({"diode": O.OrientationModel.train_svm(X, y)})
    test = [s for _ in range(10) for s in O.rotated_samples(diode_crop(rng), 0)]
    assert O.accuracy(model, "diode", test) >= 0.95


def test_polarity_sets_terminal_order():
    img = np.full((200, 300, 3), 255, np.uint8)
    D = (120, 80, 180, 120)
    cv2.rectangle(img, (123, 83), (177, 117), (0, 0, 0), 2)
    cv2.line(img, (20, 100), (120, 100), (0, 0, 0), 3)
    cv2.line(img, (180, 100), (280, 100), (0, 0, 0), 3)
    mask = preprocess.binarize(img)
    nets0, _ = topology.extract_nets(mask, [Detection("diode", D, rotation=0)])
    nets180, _ = topology.extract_nets(mask, [Detection("diode", D, rotation=180)])
    assert len(nets0[0]) == 2
    assert nets180[0] == nets0[0][::-1]


def test_convention_from_environment():
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = ("from circuitvision import orientation as O;"
            "print(O.PLUS_AT_0, O.CGHD_CLOCKWISE, O.from_cghd('90'), O.positive_side('diode', 0), O.CONVENTION_SOURCE)")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CIRCUITVISION_")}
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    default = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout
    assert default.startswith("left False 90 left code default (UNVERIFIED)")
    env.update(CIRCUITVISION_PLUS_AT_0="right", CIRCUITVISION_CGHD_CLOCKWISE="1")
    set_ = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout
    assert set_.startswith("right True 270 right environment")
    env["CIRCUITVISION_PLUS_AT_0"] = "up"
    bad = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "must be one of" in bad.stderr
