"""Demo UI: upload a photo of a hand-drawn circuit, get the netlist back.

  python -m circuitvision.app --weights runs/detect/weights/best.pt
"""
import argparse

import cv2

from . import grammar, netlist, simulate
from .pipeline import overlay, run


def build_ui(weights: str, use_ocr: bool = True):
    import gradio as gr
    from .detect import Detector

    detector = Detector(weights)

    def process(rgb):
        if rgb is None:
            return None, "", "Upload an image."
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dets = detector(img)
        if use_ocr:
            from .ocr import read_values
            read_values(img, dets)
        comps, labels = run(img, dets)
        spice = netlist.to_spice(comps)
        errors = grammar.validate(spice)
        report = ["Validation: " + ("OK" if not errors else "; ".join(errors))]
        if not errors and simulate.available():
            try:
                op = simulate.operating_point(spice)
                report += [f"{k} = {v:.4g}" for k, v in op.items()]
            except RuntimeError as e:
                report.append(str(e))
        vis = cv2.cvtColor(overlay(img, comps, labels), cv2.COLOR_BGR2RGB)
        return vis, spice, "\n".join(report)

    return gr.Interface(
        fn=process,
        inputs=gr.Image(label="Hand-drawn circuit"),
        outputs=[gr.Image(label="Detected components and nets"),
                 gr.Code(label="SPICE netlist"),
                 gr.Textbox(label="Validation and DC operating point", lines=8)],
        title="CircuitVision",
        description="Hand-drawn circuit → SPICE netlist → simulation",
        flagging_mode="never",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--share", action="store_true", help="public link for demos")
    a = ap.parse_args()
    build_ui(a.weights, not a.no_ocr).launch(share=a.share)


if __name__ == "__main__":
    main()
