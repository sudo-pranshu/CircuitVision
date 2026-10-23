"""Class set for the mini-project scope, and the explicit mapping from CGHD labels.

CGHD has ~59 labels (see classes.json in the dataset). The mapping is explicit,
not prefix-based, so every in-scope label is a deliberate decision:

  resistor        resistor, resistor.adjustable, resistor.photo
                  (all drawn as a two-terminal resistor body)
  capacitor       capacitor.unpolarized, capacitor.polarized
  inductor        inductor
  voltage_source  voltage.dc, voltage.battery
                  (voltage.ac is excluded: it would be simulated as DC)
  diode           diode, diode.light_emitting, diode.zener
                  (zener breakdown is not modelled; simulated as a plain diode)
  gnd, junction, crossover, text

Every other label (transistors, ICs, switches, lamps, voltage.ac,
diode.thyrector, vss, terminal, ...) is OUT OF SCOPE: it is not a YOLO class,
but its ink is still masked during wire tracing (class "other") so it cannot
short two wires together.
"""

# Canonical classes used by the detector (order = YOLO class id)
CLASSES = [
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "diode",
    "gnd",
    "junction",
    "crossover",
    "text",
]

OTHER = "other"   # out-of-scope symbol: masked in topology, never a netlist element

CGHD_MAP = {
    "resistor": "resistor",
    "resistor.adjustable": "resistor",
    "resistor.photo": "resistor",
    "capacitor.unpolarized": "capacitor",
    "capacitor.polarized": "capacitor",
    "inductor": "inductor",
    "voltage.dc": "voltage_source",
    "voltage.battery": "voltage_source",
    "diode": "diode",
    "diode.light_emitting": "diode",
    "diode.zener": "diode",
    "gnd": "gnd",
    "junction": "junction",
    "crossover": "crossover",
    "text": "text",
}

# Classes that are wiring structure, not netlist elements
STRUCTURAL = {"junction", "crossover", "text", "gnd"}

# SPICE prefix and default value per element class
SPICE = {
    "resistor": ("R", "1k"),
    "capacitor": ("C", "1u"),
    "inductor": ("L", "1m"),
    "voltage_source": ("V", "DC 5"),
    "diode": ("D", "DMOD"),
}


def map_cghd(label: str) -> str | None:
    """CGHD label -> canonical class, or None if out of scope."""
    return CGHD_MAP.get(label.strip().lower())
