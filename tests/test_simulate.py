import pytest

from circuitvision import simulate

pytestmark = pytest.mark.skipif(not simulate.available(), reason="ngspice not installed")


def test_divider_operating_point():
    op = simulate.operating_point("* d\nV1 1 0 DC 5\nR1 1 2 10k\nR2 2 0 1k\n.op\n.end\n")
    assert op["v(1)"] == pytest.approx(5.0)
    assert op["v(2)"] == pytest.approx(5 * 1 / 11, rel=1e-3)


def test_broken_netlist_raises():
    with pytest.raises(RuntimeError):
        simulate.operating_point("* bad\nR1 1 2\n.end\n")
