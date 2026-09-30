from dataclasses import dataclass, field


@dataclass
class Detection:
    cls: str                      # canonical class, e.g. "resistor"
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    conf: float = 1.0
    text: str | None = None       # for "text" detections (OCR result)
    rotation: int | None = None   # 0/90/180/270, for polarized parts


@dataclass
class Component:
    ref: str                      # e.g. "R1"
    cls: str
    box: tuple[int, int, int, int]
    nets: list[int] = field(default_factory=list)
    value: str | None = None
