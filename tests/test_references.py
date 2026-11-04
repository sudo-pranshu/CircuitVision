"""Hand-written reference netlists (references/drafter_12, task T-3) must be valid."""
import re
from pathlib import Path

import pytest

from circuitvision.grammar import validate

REFS = Path(__file__).resolve().parent.parent / "references"
FILES = sorted(REFS.glob("drafter_*/*.cir"))
# removing an out-of-scope part leaves its terminals unconnected; nothing else is allowed
REDUCED_OK = re.compile(r"^(node \S+ is floating \(1 connection\)|no ground node \(0\))$")


def reference_errors(text: str) -> list[str]:
    errs = validate(text)
    if any(l.startswith("* omitted:") for l in text.splitlines()):
        errs = [e for e in errs if not REDUCED_OK.match(e)]
    return errs


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_reference_netlist(path):
    assert re.fullmatch(r"C\d+_D\d+\.cir", path.name), "name must be <circuit>_<drawing>.cir"
    text = path.read_text()
    assert text.startswith(f"* {path.stem}\n"), "first line must be the title '* <circuit>_<drawing>'"
    assert reference_errors(text) == []


def test_template_follows_the_rules():
    text = (REFS / "TEMPLATE.cir.example").read_text()
    assert reference_errors(text) == []
    assert validate(text), "the template omits a switch, so plain validation should flag it"
