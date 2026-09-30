import json

from circuitvision.audit import audit
from circuitvision.cghd_to_yolo import clip_box, convert, parse_voc

from fake_cghd import make

VAL, TEST = {"drafter_11"}, {"drafter_12"}


def test_clip_box():
    assert clip_box(10, 10, 50, 30, 200, 100) == (10, 10, 50, 30)
    assert clip_box(180, 60, 230, 90, 200, 100) == (180, 60, 200, 90)
    assert clip_box(50, 50, 50, 70, 200, 100) is None
    assert clip_box(50, 30, 10, 10, 200, 100) == (10, 10, 50, 30)     # inverted


def test_parse_voc_reads_rotation_and_text(tmp_path):
    root = make(tmp_path / "cghd")
    _, _, objs = parse_voc(next((root / "drafter_1" / "annotations").glob("C0*.xml")))
    by_cls = {o[0]: o for o in objs}
    assert by_cls["diode"][6] == "90"
    assert by_cls["voltage_source"][6] == "0"
    assert by_cls["text"][5] == "10k"
    assert "transistor.bjt" not in by_cls


def test_convert_counts_and_labels(tmp_path):
    root = make(tmp_path / "cghd")
    r = convert(root, tmp_path / "yolo", VAL, TEST)
    assert r["splits"]["train"]["images"] == 2
    assert r["splits"]["test"]["images"] == 2
    assert r["problems"]["missing_image"] == 1
    assert r["problems"]["boxes_dropped"] == 6          # zero-width inductor x 6 images
    assert r["problems"]["boxes_clipped"] == 6
    assert r["ignored_classes"] == {"transistor.bjt": 6}
    # 5 in-scope objects kept per image: resistor, diode, source, text, capacitor
    label = (tmp_path / "yolo" / "labels" / "test" / "drafter_12_C0_D1_P1.txt").read_text().split("\n")
    assert len([l for l in label if l]) == 5
    cls, cx, cy, bw, bh = label[0].split()
    assert (float(cx), float(cy), float(bw), float(bh)) == (0.15, 0.2, 0.2, 0.2)
    assert json.loads((tmp_path / "yolo" / "conversion_stats.json").read_text())["splits"]


def test_audit_report(tmp_path):
    root = make(tmp_path / "cghd")
    r, samples = audit(root, VAL, TEST)
    assert r["drafters"] == ["drafter_1", "drafter_11", "drafter_12"]
    assert r["test_drafters_present"] == ["drafter_12"]
    assert r["per_drafter"]["drafter_1"]["xml_without_image"] == 1
    assert r["rotation"]["diode"] == {"90": 6}
    assert r["boxes"]["out_of_bounds"] == 6 and r["boxes"]["inverted_or_empty"] == 6
    assert r["size_check"] == {"xml_matches_displayed": 6}
    assert samples and "<rotation>" in samples[0]
