#!/usr/bin/env bash
# Leave exactly one OpenCV distribution installed, at a pinned version.
#
# Why: opencv-python, opencv-python-headless and opencv-contrib-python(-headless)
# all install into the same site-packages/cv2/ folder. Colab ships several of them,
# and ultralytics / easyocr each request a different one. When pip later removes
# or replaces one, it deletes files the others still rely on, leaving cv2/ hollow:
# `import cv2` succeeds but attributes such as cv2.HOGDescriptor are missing.
#
# Run this AFTER every other pip install. It must be the last package step.
set -euo pipefail
CV_PKG="${CV_PKG:-opencv-python-headless}"
CV_VERSION="${CV_VERSION:-4.10.0.84}"

python -m pip uninstall -y -q opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless || true
# Remove leftover cv2/ folders wherever Python still finds them (hollow namespace
# packages survive the uninstall). Only paths that end in /cv2 are touched.
python - <<'PY' | while read -r d; do [ "${d##*/}" = "cv2" ] && rm -rf "$d"; done
import importlib.util
spec = importlib.util.find_spec("cv2")
for p in (spec.submodule_search_locations or []) if spec else []:
    print(p)
PY
python -m pip install -q --no-deps --force-reinstall "${CV_PKG}==${CV_VERSION}"

python "$(dirname "$0")/check_opencv.py"
