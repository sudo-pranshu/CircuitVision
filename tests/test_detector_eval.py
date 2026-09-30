import numpy as np

from circuitvision.classes import CLASSES
from circuitvision.detector_eval import crowding, iou_matrix, match

R, C, T = CLASSES.index("resistor"), CLASSES.index("capacitor"), CLASSES.index("text")


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    assert abs(iou_matrix(a, np.array([[0, 0, 10, 10]]))[0, 0] - 1.0) < 1e-6
    assert abs(iou_matrix(a, np.array([[5, 0, 15, 10]]))[0, 0] - 1 / 3) < 1e-6
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_match_tp_fp_fn_and_confusion():
    gt = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
    gt_c = [R, C, T]
    pr = [[0, 0, 10, 10],      # correct resistor
          [40, 0, 50, 10],     # text predicted as resistor -> FP + confusion
          [80, 0, 90, 10]]     # nothing there -> background FP
    m = match(gt, gt_c, pr, [R, R, C], [0.9, 0.8, 0.7])
    assert m["tp"] == [(0, 0)]
    assert sorted(m["fp"]) == [1, 2]
    assert sorted(m["fn"]) == [1, 2]
    assert m["confusions"] == [(T, R)]


def test_duplicate_prediction_is_fp():
    m = match([[0, 0, 10, 10]], [R], [[0, 0, 10, 10], [1, 0, 11, 10]], [R, R], [0.9, 0.95])
    assert len(m["tp"]) == 1 and len(m["fp"]) == 1 and m["fn"] == []


def test_crowding_increases_when_boxes_are_close():
    far = crowding([[0, 0, 10, 10], [100, 0, 110, 10]])
    near = crowding([[0, 0, 10, 10], [12, 0, 22, 10]])
    assert near > far > 0
