"""The demo app's reports: they must surface what was assumed or uncertain."""
from circuitvision.app import image_report, text_report
from circuitvision.types import Component, Detection


def test_text_report_lists_assumptions_and_simulates():
    spice, rep = text_report("a resistor in series with a capacitor")
    assert "R1" in spice and "Assumed, not read from your text" in rep and "No voltage source" in rep
    spice, rep = text_report("a 9V battery driving a 1k resistor in series with a 2k resistor")
    assert "Assumed" not in rep and "Validation: OK" in rep


def test_text_report_explains_empty_input():
    assert text_report("make me a circuit") == ("", "No circuit components were found in that description. "
                                                    "Name at least one part.")


def test_image_report_flags_dropped_parts_defaults_and_low_confidence():
    dets = [Detection("voltage_source", (0, 0, 10, 10), conf=0.31), Detection("resistor", (20, 0, 30, 10), conf=0.9)]
    comps = [Component("V1", "voltage_source", (0, 0, 10, 10), [1, 2, 3]),
             Component("R1", "resistor", (20, 0, 30, 10), [1, 0])]
    spice = "* t\nR1 1 0 1k\n* WARNING V1: found 3 terminals [1, 2, 3], skipped\n.op\n.end\n"
    rep = image_report(dets, comps, spice, {"ground": "implicit: most connected net"})
    assert "uncertain" in rep and "V1 was left out: 3 wire ends" in rep
    assert "R1: no value label was read" in rep and "ground was chosen automatically" in rep
    assert "Validation: FAILED" in rep and "Simulation: skipped" in rep


def test_image_report_flags_one_symbol_with_two_labels():
    dets = [Detection("capacitor", (0, 0, 100, 40), conf=0.75), Detection("voltage_source", (1, 0, 100, 41), conf=0.61)]
    comps = [Component("C1", "capacitor", (0, 0, 100, 40), [1, 0], "1u"),
             Component("V1", "voltage_source", (1, 0, 100, 41), [1, 0], "DC 9")]
    rep = image_report(dets, comps, "* t\nV1 1 0 DC 9\nC1 1 0 1u\n.op\n.end\n", {"ground": "symbol"})
    assert "detected as both capacitor (0.75) and voltage_source (0.61)" in rep
