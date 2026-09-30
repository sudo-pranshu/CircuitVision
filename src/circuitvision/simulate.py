"""Run a netlist through ngspice (batch mode) and return the DC operating point."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

LINE = re.compile(r"^\s*(v\(\w+\)|\w+#branch)\s*=\s*(\S+)", re.I)
CONTROL = ".control\nop\nprint all\n.endc\n"


def available() -> bool:
    return shutil.which("ngspice") is not None


def operating_point(netlist: str, timeout: int = 20) -> dict[str, float]:
    """-> {"v(1)": 5.0, "v1#branch": -4.5e-4, ...}. Raises RuntimeError on failure."""
    if not available():
        raise RuntimeError("ngspice not found (macOS: brew install ngspice)")
    body = "\n".join(l for l in netlist.splitlines() if l.strip().lower() not in (".op", ".end"))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "c.cir"
        path.write_text(body + "\n" + CONTROL + ".end\n")
        res = subprocess.run(["ngspice", "-b", str(path)], capture_output=True,
                             text=True, timeout=timeout)
    out = {m.group(1).lower(): float(m.group(2))
           for m in map(LINE.match, res.stdout.splitlines()) if m}
    if not out:
        err = (res.stderr or res.stdout).strip().splitlines()[-3:]
        raise RuntimeError("simulation failed: " + " | ".join(err))
    return out
