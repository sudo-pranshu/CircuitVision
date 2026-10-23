"""Topology on synthetic circuits with KNOWN graphs (SYNTHETIC/CONTROLLED)."""
import cv2
import numpy as np
import pytest

from circuitvision import grammar, netlist, preprocess, synth, topology
from circuitvision.netlist import parse_value
from circuitvision.pipeline import run, trace
from circuitvision.types import Component, Detection

# Conditions the method must handle exactly (wobble / 15% box jitter are measured, not asserted)
MUST_PASS = ["clean", "photo_scale_x6", "grid_paper", "shadow", "noise_blur_jpeg",
             "all_combined", "pen_gaps", "pencil_on_grid", "box_jitter_10pct"]


@pytest.mark.parametrize("cond", MUST_PASS)
@pytest.mark.parametrize("make", synth.CIRCUITS, ids=lambda m: m.__name__)
def test_exact_topology(make, cond):
    c = make()
    img, dets = synth.render(c, **synth.CONDITIONS[cond])
    nets = trace(img, dets)
    pred = [nets.get(i, []) for i in range(len(c.parts))]
    assert synth.same_partition(pred, synth.gt_partition(c)), (pred, synth.gt_partition(c))


def test_same_partition():
    assert synth.same_partition([[5, 7], [7, 9]], [(1, 2), (2, 3)])
    assert not synth.same_partition([[5, 7], [5, 9]], [(1, 2), (2, 3)])
    assert not synth.same_partition([[5, 7, 8]], [(1, 2)])


def test_grid_lines_are_not_ink():
    img, _ = synth.render(synth.divider(), scale=3, grid=True)
    mask = preprocess.binarize(img)
    assert mask[5:30, 5:30].max() == 0          # empty grid area: no ink


def test_shadowed_strokes_are_kept():
    img, _ = synth.render(synth.divider(), scale=3, shadow=True)
    mask = preprocess.binarize(img)
    x = int(400 * 3)                             # right-hand wire, deepest shadow
    assert mask[int(300 * 3), x - 10:x + 10].max() == 255


def test_dangling_wire_is_a_floating_node():
    c = synth.divider()
    c.wires[1] = [(270, 80), (400, 80), (400, 140)]        # break: wire stops short of R2
    img, dets = synth.render(c)
    comps, _ = run(img, dets)
    errors = grammar.validate(netlist.to_spice(comps))
    assert any("floating" in e for e in errors), errors


def test_scale_invariance_of_parameters():
    c = synth.bridge()
    small = trace(*synth.render(c, scale=1))
    big = trace(*synth.render(c, scale=8))
    assert synth.same_partition([big[i] for i in range(len(c.parts))],
                                [tuple(small[i]) for i in range(len(c.parts))])


def test_symbol_scale():
    dets = [Detection("resistor", (0, 0, 300, 120)), Detection("text", (0, 0, 10, 10))]
    assert topology.symbol_scale(dets) == pytest.approx(5.0)
    assert topology.symbol_scale([]) == 1.0


def test_value_parsing_rkm_and_units():
    assert parse_value("4k7") == ("4.7k", None)
    assert parse_value("2R2") == ("2.2", None)
    assert parse_value("4,7µF") == ("4.7u", "capacitor")
    assert parse_value("1M5") == ("1.5meg", None)
    assert parse_value("R1") == (None, None)


def test_value_attachment_unit_check_and_one_to_one():
    r = Component("R1", "resistor", (100, 100, 160, 120))
    cap = Component("C1", "capacitor", (300, 100, 320, 160))
    labels = [Detection("text", (110, 70, 150, 90), text="10uF"),   # nearest to R1 but a capacitance
              Detection("text", (290, 170, 330, 190), text="10uF"),
              Detection("text", (100, 125, 160, 140), text="4k7")]
    netlist.attach_values([r, cap], labels)
    assert (r.value, cap.value) == ("4.7k", "10u")
