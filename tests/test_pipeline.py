import cv2
import numpy as np

from circuitvision import netlist, preprocess, topology
from circuitvision.types import Detection


def line(img, p, q):
    cv2.line(img, p, q, (0, 0, 0), 3)


def box_symbol(img, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (0, 0, 0), 2)


def voltage_divider():
    img = np.full((420, 500, 3), 255, np.uint8)
    V1, R1, R2, G = (80, 180, 120, 260), (200, 65, 280, 95), (385, 160, 415, 240), (230, 345, 270, 385)
    for b in (V1, R1, R2, G):
        box_symbol(img, b)
    line(img, (100, 180), (100, 80)); line(img, (100, 80), (200, 80))      # V1+ -> R1
    line(img, (280, 80), (400, 80)); line(img, (400, 80), (400, 160))      # R1 -> R2
    line(img, (400, 240), (400, 340)); line(img, (100, 260), (100, 340))   # to bottom rail
    line(img, (100, 340), (400, 340)); line(img, (250, 340), (250, 345))   # rail + gnd
    img = cv2.GaussianBlur(img, (3, 3), 0)
    dets = [Detection("voltage_source", V1), Detection("resistor", R1),
            Detection("resistor", R2), Detection("gnd", G),
            Detection("text", (190, 30, 290, 60), text="10k")]
    return img, dets


def run(img, dets):
    mask = preprocess.remove_small_blobs(preprocess.binarize(img))
    nets, _ = topology.extract_nets(mask, dets)
    comps = netlist.build(dets, nets)
    netlist.attach_values(comps, dets)
    return {c.ref: c for c in comps}


def test_voltage_divider():
    c = run(*voltage_divider())
    top, mid = c["V1"].nets[0], c["R1"].nets[1]
    assert c["V1"].nets == [top, 0]
    assert c["R1"].nets == [top, mid]
    assert c["R2"].nets == [mid, 0]
    assert len({top, mid, 0}) == 3
    assert c["R1"].value == "10k"


def test_crossover_does_not_short():
    img = np.full((400, 400, 3), 255, np.uint8)
    X = (185, 185, 215, 215)
    Ra, Rb = (20, 190, 80, 210), (320, 190, 380, 210)     # horizontal path
    Rc, Rd = (190, 20, 210, 80), (190, 320, 210, 380)     # vertical path
    for b in (X, Ra, Rb, Rc, Rd):
        box_symbol(img, b)
    line(img, (80, 200), (320, 200))
    line(img, (200, 80), (200, 320))
    dets = [Detection("crossover", X), Detection("resistor", Ra), Detection("resistor", Rb),
            Detection("resistor", Rc), Detection("resistor", Rd)]
    c = run(img, dets)
    h = c["R1"].nets[-1]
    v = c["R3"].nets[-1]
    assert c["R2"].nets[0] == h
    assert c["R4"].nets[0] == v
    assert h != v


def test_spice_output():
    img, dets = voltage_divider()
    mask = preprocess.binarize(img)
    nets, _ = topology.extract_nets(mask, dets)
    comps = netlist.build(dets, nets)
    netlist.attach_values(comps, dets)
    out = netlist.to_spice(comps)
    assert "R1" in out and "V1" in out and out.strip().endswith(".end")
    assert "WARNING" not in out
