"""Fail loudly unless exactly one complete OpenCV distribution is installed.

  python scripts/check_opencv.py
"""
import importlib.metadata as md
import sys

NAMES = ["opencv-python", "opencv-python-headless", "opencv-contrib-python",
         "opencv-contrib-python-headless"]

present = {}
for n in NAMES:
    try:
        present[n] = md.version(n)
    except md.PackageNotFoundError:
        pass
print("opencv distributions:", present)
try:
    import cv2
    cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    cv2.ml.SVM_create()
except (ImportError, AttributeError) as e:
    sys.exit(f"OpenCV is broken ({e}). Run: bash scripts/pin_opencv.sh")
print("cv2", cv2.__version__, "from", cv2.__file__)
if len(present) != 1:
    sys.exit(f"expected exactly one OpenCV distribution, found {present}. Run: bash scripts/pin_opencv.sh")
print("OpenCV OK: HOGDescriptor and ml.SVM available")
