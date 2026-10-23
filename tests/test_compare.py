import pytest

from circuitvision import simulate
from circuitvision.compare import compare, elements, isomorphic, spice_float

REF = "* r\nV1 1 0 DC 5\nR1 1 2 10k\nR2 2 0 1k\n.op\n.end\n"


def test_spice_float():
    assert spice_float("4.7k") == 4700 and spice_float("DC 5") == 5 and spice_float("1meg") == 1e6
    assert spice_float("DMOD") == "DMOD" and spice_float(None) is None


def test_isomorphic_ignores_net_names_and_line_order():
    pred = "* p\nR2 9 4 1k\nV1 7 4 5\nR1 7 9 10k\n.end\n"
    assert isomorphic(pred, REF) and isomorphic(pred, REF, values=False)


def test_values_and_wiring_differences():
    assert not isomorphic(REF.replace("R2 2 0 1k", "R2 2 0 2k"), REF)
    assert isomorphic(REF.replace("R2 2 0 1k", "R2 2 0 2k"), REF, values=False)
    assert not isomorphic(REF.replace("R2 2 0", "R2 1 0"), REF, values=False)   # parallel, not series
    assert not isomorphic(REF.replace("V1 1 0", "V1 0 1"), REF)                  # source polarity


def test_elements_parsing():
    assert elements(REF)[1] == ("R", "1", "2", 10000.0)


@pytest.mark.skipif(not simulate.available(), reason="ngspice not installed")
def test_op_match_is_reference_node_independent():
    pred = "* p\nV1 5 3 5\nR1 5 8 10k\nR2 8 3 1k\nR9 3 0 1k\n.op\n.end\n"   # extra element
    assert compare(pred, REF)["op_match"] is False
    shifted = "* p\nV1 0 3 5\nR1 0 8 10k\nR2 8 3 1k\n.op\n.end\n"           # ground moved
    r = compare(shifted, REF)
    assert r["topology_match"] and r["exact_match"] and r["op_match"]
