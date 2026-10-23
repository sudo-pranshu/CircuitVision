"""Regression tests for the real CGHD annotation format and image frames."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from circuitvision import netlist, preprocess, topology
from circuitvision.audit import audit
from circuitvision.cghd_to_yolo import convert, load_for_annotation, parse_voc
from circuitvision.classes import OTHER, map_cghd
from circuitvision.orientation import from_cghd, is_mirrored
from circuitvision.pipeline import detections_from_xml
from circuitvision.types import Detection

REAL = Path(__file__).parent / "data" / "cghd_real" / "C145_D1_P1.xml"


def test_real_xml_rotation_is_nested_in_bndbox():
    _, _, raw = parse_voc(REAL, raw=True)
    battery = [o for o in raw if o.cls == "voltage.battery"][0]
    assert battery.rotation == "269"            # was None before: parser looked under <object>
    assert from_cghd(battery.rotation) == 270
    assert [o.text for o in raw if o.cls == "text"] == ["L", "S1", "S2"]


def test_real_xml_out_of_scope_kept_as_other_for_masking():
    w, h, scoped = parse_voc(REAL)
    assert (w, h) == (1536, 2049)
    assert {o.cls for o in scoped} == {"voltage_source", "text", "junction"}
    _, _, kept = parse_voc(REAL, keep_other=True)
    assert sum(o.cls == OTHER for o in kept) == 3          # lamp + 2 switches
    dets = detections_from_xml(REAL)
    assert [d.rotation for d in dets if d.cls == "voltage_source"] == [270]


def test_explicit_class_map():
    assert map_cghd("resistor.adjustable") == "resistor"
    assert map_cghd("capacitor.polarized") == "capacitor"
    assert map_cghd("diode.light_emitting") == "diode"
    assert map_cghd("voltage.battery") == "voltage_source"
    for out in ("voltage.ac", "diode.thyrector", "vss", "terminal", "transistor.bjt", "lamp"):
        assert map_cghd(out) is None, out


def test_from_cghd_and_mirror_parsing():
    assert [from_cghd(v) for v in ("2", "8", "89", "181", "269", "359")] == [0, 0, 90, 180, 270, 0]
    assert from_cghd(None) is None and from_cghd("n/a") is None
    assert is_mirrored(None) is None and is_mirrored("1") and is_mirrored("true") and not is_mirrored("0")


def test_mirror_tag_parsed_anywhere(tmp_path):
    x = tmp_path / "m.xml"
    x.write_text("<annotation><size><width>10</width><height>10</height></size>"
                 "<object><name>diode</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax>"
                 "<ymax>5</ymax><rotation>90</rotation><mirror>1</mirror></bndbox></object></annotation>")
    o = parse_voc(x)[2][0]
    assert (o.rotation, o.mirror) == ("90", "1")
    assert detections_from_xml(x)[0].mirror is True


def test_audit_on_real_annotation(tmp_path):
    d = tmp_path / "cghd" / "drafter_13"
    (d / "annotations").mkdir(parents=True)
    (d / "images").mkdir()
    (d / "annotations" / REAL.name).write_text(REAL.read_text())
    cv2.imwrite(str(d / "images" / "C145_D1_P1.jpg"), np.full((2049, 1536, 3), 255, np.uint8))
    r, samples = audit(tmp_path / "cghd", {"drafter_11"}, {"drafter_13"})
    assert r["rotation_present"] == {"voltage.battery": 1}
    assert r["polarized_objects_without_rotation"] == {}
    assert r["bndbox_fields"]["rotation"] == 1
    assert r["rotation_bin"]["voltage.battery"] == {"270": 1}
    assert r["rotation_aspect"]["voltage.battery"]["270"] == {"wide": 1}
    assert r["size_check"] == {"xml_matches_displayed": 1}
    assert samples and "<rotation>269</rotation>" in samples[0]


def _exif_jpeg(path, orientation, w=60, h=40):
    img = np.full((h, w, 3), 255, np.uint8)
    img[:, : w // 3] = 0                                   # dark left third (stored pixels)
    im = Image.fromarray(img)
    ex = im.getexif()
    ex[274] = orientation
    im.save(path, quality=95, exif=ex)
    return path


def test_load_for_annotation_picks_the_annotated_frame(tmp_path):
    p = _exif_jpeg(tmp_path / "a.jpg", 6)                # stored 60x40, displayed 40x60
    img, frame = load_for_annotation(p, 60, 40)
    assert frame == "raw" and img.shape[:2] == (40, 60)
    img, frame = load_for_annotation(p, 40, 60)
    assert frame == "displayed" and img.shape[:2] == (60, 40)
    assert load_for_annotation(_exif_jpeg(tmp_path / "b.jpg", 3), 60, 40)[1] == "ambiguous"
    assert load_for_annotation(_exif_jpeg(tmp_path / "c.jpg", 1), 60, 40)[1] == "upright"


def test_converter_writes_exif_free_image_in_annotation_frame(tmp_path):
    d = tmp_path / "cghd" / "drafter_1"
    (d / "annotations").mkdir(parents=True)
    (d / "images").mkdir()
    _exif_jpeg(d / "images" / "C1_D1_P1.jpg", 6)
    (d / "annotations" / "C1_D1_P1.xml").write_text(
        "<annotation><size><width>60</width><height>40</height></size><object><name>resistor</name>"
        "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>20</xmax><ymax>40</ymax></bndbox></object></annotation>")
    r = convert(tmp_path / "cghd", tmp_path / "yolo", set(), set())
    assert r["problems"]["exif_frame_raw"] == 1
    out = tmp_path / "yolo" / "images" / "train" / "drafter_1_C1_D1_P1.jpg"
    assert Image.open(out).getexif().get(274, 1) == 1
    img = cv2.imread(str(out))
    assert img.shape[:2] == (40, 60) and img[:, :15].mean() < 60   # dark band where the box is


def test_out_of_scope_symbol_does_not_short_wires():
    img = np.full((200, 400, 3), 255, np.uint8)
    lamp = (170, 70, 230, 130)                             # out-of-scope symbol in the middle
    cv2.circle(img, (200, 100), 25, (0, 0, 0), 3)
    cv2.line(img, (60, 100), (175, 100), (0, 0, 0), 3)
    cv2.line(img, (225, 100), (340, 100), (0, 0, 0), 3)
    cv2.rectangle(img, (23, 90), (57, 110), (0, 0, 0), 2)
    cv2.rectangle(img, (343, 90), (377, 110), (0, 0, 0), 2)
    dets = [Detection("resistor", (20, 85, 60, 115)), Detection("resistor", (340, 85, 380, 115)),
            Detection(OTHER, lamp)]
    nets, _ = topology.extract_nets(preprocess.binarize(img), dets)
    assert nets[0][-1] != nets[1][0]                      # lamp ink must not merge the two wires
    assert [c.ref for c in netlist.build(dets, nets)] == ["R1", "R2"]


def test_implicit_ground_rules():
    from circuitvision import netlist as nl
    dets = [Detection("voltage_source", (0, 0, 10, 10)), Detection("resistor", (20, 0, 30, 10))]
    info = {}
    comps = nl.build(dets, {0: [3, 7], 1: [3, 7]}, info)
    assert info["ground"].startswith("implicit: first source") and comps[0].nets == [3, 0]
    info = {}
    nl.build([Detection("resistor", (0, 0, 1, 1)), Detection("resistor", (0, 0, 1, 1))],
             {0: [1, 2], 1: [2, 3]}, info)
    assert info["ground"] == "implicit: most connected net"
