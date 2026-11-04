"""Demo UI: a photo of a hand-drawn circuit, or a sentence, in; a SPICE netlist out.

  python -m circuitvision.app --weights runs/baseline_v1/best.pt [--orient models] [--no-ocr] [--lan] [--share]

Without --weights only the text tab works. Every report says what was assumed or
uncertain, so a guess is never presented as the user's circuit.
"""
import argparse
import re

import cv2

from . import grammar, netlist, simulate
from .classes import SPICE
from .pipeline import overlay, run

LOW_CONF = 0.5
SAME_SYMBOL_IOU = 0.7
IMAGE_HELP = ("Photograph or scan a circuit drawn in dark pen on plain or grid paper, filling most of the frame. "
              "Supported parts: resistor, capacitor, inductor, DC source/battery, diode/LED, ground, junction dots and "
              "crossovers. Other symbols (switches, AC sources, transistors, logic gates) are ignored, so a circuit "
              "that contains them cannot be completed.")
TEXT_HELP = ('Describe a series/parallel circuit, e.g. "a 9V battery driving a 1k resistor in series with two 2k '
             'resistors in parallel". Bridges and circuits that need named nodes are not supported.')


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0, w) * max(0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _explain(errors: list[str]) -> list[str]:
    hints = {"no ground node": "no part was connected to a reference node, usually because no wires were traced",
             "is floating": "a wire end connects to only one part: a wire gap, or a part whose terminals were not found",
             "is shorted": "both ends of a part landed on the same wire",
             "syntax error": "the netlist text is malformed"}
    return [e + next((f"  ({h})" for k, h in hints.items() if k in e), "") for e in errors]


def simulate_report(spice: str, errors: list[str]) -> list[str]:
    if errors:
        return ["Simulation: skipped, because the netlist is not valid."]
    if not simulate.available():
        return ["Simulation: ngspice is not installed."]
    try:
        op = simulate.operating_point(spice)
    except RuntimeError as e:
        return [f"Simulation failed: {e}"]
    return ["DC operating point (ngspice):"] + [f"  {k} = {v:.4g}" for k, v in op.items()]


def image_report(dets, comps, spice: str, info: dict) -> str:
    parts = [d for d in dets if d.cls not in ("junction", "crossover", "text")]
    lines = [f"Detected {len(parts)} parts, {sum(d.cls == 'junction' for d in dets)} junctions, "
             f"{sum(d.cls == 'crossover' for d in dets)} crossovers, {sum(d.cls == 'text' for d in dets)} labels."]
    lines += [f"  {d.cls:<15} confidence {d.conf:.2f}" + ("  <- uncertain" if d.conf < LOW_CONF else "")
              for d in sorted(parts, key=lambda d: d.conf)]
    warn = []
    for line in spice.splitlines():
        m = re.match(r"\* WARNING (\w+): found (\d+) terminals", line)
        if m:
            warn.append(f"{m[1]} was left out: {m[2]} wire ends were found at it instead of 2.")
    for c in comps:
        if len(c.nets) == 2 and c.value is None and c.cls != "diode":
            warn.append(f"{c.ref}: no value label was read; the default {SPICE[c.cls][1]} was used.")
    if info.get("ground", "").startswith("implicit"):
        warn.append(f"No ground symbol: ground was chosen automatically ({info['ground']}).")
    for i, a in enumerate(parts):              # one symbol, two labels: e.g. a battery also read as a capacitor
        for b in parts[i + 1:]:
            if a.cls != b.cls and _iou(a.box, b.box) > SAME_SYMBOL_IOU:
                warn.append(f"One symbol was detected as both {a.cls} ({a.conf:.2f}) and {b.cls} ({b.conf:.2f}); "
                            "one of them is wrong, so the netlist has an extra part.")
    if any(d.conf < LOW_CONF for d in parts):
        warn.append(f"Some parts were detected with confidence below {LOW_CONF}; check them on the overlay.")
    errors = grammar.validate(spice)
    lines += ["", "Validation: " + ("OK" if not errors else "FAILED")] + [f"  {e}" for e in _explain(errors)]
    if warn:
        lines += ["", "Check before trusting this netlist:"] + [f"  - {w}" for w in warn]
    lines += [""] + simulate_report(spice, errors)
    return "\n".join(lines)


def text_report(text: str):
    from .nlp.text2netlist import text_to_netlist_with_notes
    if not text or not text.strip():
        return "", "Type a circuit description."
    try:
        spice, _, errors, notes = text_to_netlist_with_notes(text)
    except ValueError:
        return "", "No circuit components were found in that description. Name at least one part."
    lines = ["Validation: " + ("OK" if not errors else "FAILED")] + [f"  {e}" for e in _explain(errors)]
    if notes:
        lines += ["", "Assumed, not read from your text:"] + [f"  - {n}" for n in notes]
    lines += [""] + simulate_report(spice, errors)
    return spice, "\n".join(lines)


def build_ui(weights: str | None = None, use_ocr: bool = True, orient: str | None = None):
    import gradio as gr

    detector = orienter = None
    if weights:
        from .detect import Detector
        detector = Detector(weights)
    if orient:
        from .orientation import OrientationModel
        orienter = OrientationModel.load(orient)

    def process(rgb):
        if detector is None:
            return None, "", "The image tab needs trained weights: start the app with --weights best.pt."
        if rgb is None:
            return None, "", "Upload a photo of a hand-drawn circuit."
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dets = detector(img)
        if use_ocr:
            from .ocr import read_values
            read_values(img, dets)
        if orienter:
            orienter.annotate(img, dets)
        info = {}
        comps, labels = run(img, dets, info)
        spice = netlist.to_spice(comps, notes=[f"ground: {info['ground']}"])
        vis = cv2.cvtColor(overlay(img, comps, labels), cv2.COLOR_BGR2RGB)
        return vis, spice, image_report(dets, comps, spice, info)

    with gr.Blocks(title="CircuitVision") as ui:
        gr.Markdown("# CircuitVision\nHand-drawn circuit or plain-English description → SPICE netlist → "
                    "validation → DC simulation. Results can be wrong: read the checks under each netlist.")
        with gr.Tab("Photo"):
            gr.Markdown(IMAGE_HELP)
            with gr.Row():
                img_in = gr.Image(label="Hand-drawn circuit")
                img_out = gr.Image(label="Detected parts and traced nets")
            img_btn = gr.Button("Convert", variant="primary")
            img_net = gr.Code(label="SPICE netlist (copy button at top right)")
            img_rep = gr.Textbox(label="Detections, checks and simulation", lines=14)
            img_btn.click(process, img_in, [img_out, img_net, img_rep])
        with gr.Tab("Text"):
            gr.Markdown(TEXT_HELP)
            txt_in = gr.Textbox(label="Circuit description", lines=2)
            txt_btn = gr.Button("Convert", variant="primary")
            txt_net = gr.Code(label="SPICE netlist (copy button at top right)")
            txt_rep = gr.Textbox(label="Checks and simulation", lines=10)
            txt_btn.click(text_report, txt_in, [txt_net, txt_rep])
    ui.process, ui.text_report = process, text_report       # handlers, for headless tests
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="trained YOLO weights; without them only the text tab works")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--orient", help="folder with orientation models")
    ap.add_argument("--share", action="store_true", help="public link for demos")
    ap.add_argument("--lan", action="store_true",
                    help="also serve on this computer's Wi-Fi address, so a phone on the same network can open it")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if a.lan:
        print(f"\n  On your phone (same Wi-Fi), open:  http://{lan_ip()}:{a.port}\n"
              "  In the Photo tab, tap the image box and choose Take Photo.\n", flush=True)
    build_ui(a.weights, not a.no_ocr, a.orient).launch(
        share=a.share, server_port=a.port, server_name="0.0.0.0" if a.lan else None)


def lan_ip() -> str:
    """This computer's address on the local network (no packets are sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    main()
