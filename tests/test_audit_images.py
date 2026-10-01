"""The fast header path must report exactly what two full cv2 decodes report."""
import cv2
import numpy as np
from PIL import Image

from circuitvision.audit import all_image_sizes, image_sizes, image_sizes_full


def _jpeg(path, orientation, w=120, h=80):
    img = np.full((h, w, 3), 230, np.uint8)
    cv2.line(img, (5, 5), (w - 5, h // 2), (0, 0, 0), 3)
    im = Image.fromarray(img)
    ex = im.getexif()
    ex[274] = orientation
    im.save(path, quality=90, exif=ex)
    return path


def test_header_matches_full_decode_for_all_exif_orientations(tmp_path):
    for o in range(1, 9):
        p = _jpeg(tmp_path / f"o{o}.jpg", o)
        assert image_sizes(p) == image_sizes_full(p), f"orientation {o}"
    raw, shown = image_sizes(tmp_path / "o6.jpg")
    assert raw == (120, 80) and shown == (80, 120)


def test_png_uses_full_decode(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.zeros((30, 50, 3), np.uint8))
    assert image_sizes(p) == image_sizes_full(p) == ((50, 30), (50, 30))


def test_unreadable_and_truncated_agree(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert image_sizes(bad) == image_sizes_full(bad) == (None, None)
    full = _jpeg(tmp_path / "full.jpg", 6, 400, 300).read_bytes()
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(full[: len(full) // 2])
    assert image_sizes(cut) == image_sizes_full(cut)


def test_parallel_matches_serial_and_reports_progress(tmp_path, capsys):
    paths = [_jpeg(tmp_path / f"{i}.jpg", 1 + i % 8) for i in range(25)]
    serial = all_image_sizes(paths, workers=1, every=10)
    parallel = all_image_sizes(paths, workers=2, every=10)
    assert serial == parallel and len(serial) == 25
    out = capsys.readouterr().out
    assert "Audited 10/25 images" in out and "Audited 25/25 images" in out
