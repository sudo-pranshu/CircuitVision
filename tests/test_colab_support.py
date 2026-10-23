"""Restart support for the Colab run: training state detection and Gate 2 sampling."""
from pathlib import Path

import pytest

from circuitvision.visualize import pick_stratified


def _labels(tmp_path, split, stem, classes):
    d = tmp_path / "labels" / split
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{stem}.txt"
    f.write_text("".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in classes))
    return f


def test_gate2_sample_covers_splits_classes_and_exif(tmp_path):
    files = [_labels(tmp_path, "train", f"t{i}", [0, 1]) for i in range(40)]
    files += [_labels(tmp_path, "val", f"v{i}", [0]) for i in range(5)]
    files += [_labels(tmp_path, "test", f"x{i}", [1]) for i in range(5)]
    files.append(_labels(tmp_path, "train", "rare", [7]))
    exif = {"train/t3", "val/v2"}
    got = pick_stratified(sorted(files), 12, priority=exif)
    keys = {f"{f.parent.name}/{f.stem}" for f in got}
    assert len(got) == 12 and len(keys) == 12
    assert exif <= keys                                        # EXIF re-encodes are always shown
    assert all(sum(f.parent.name == s for f in got) >= 2 for s in ("train", "val", "test"))
    assert "train/rare" in keys                                # every class appears


def test_training_state(tmp_path):
    torch = pytest.importorskip("torch")
    from circuitvision.train import checkpoint_state
    w = tmp_path / "runs" / "train" / "weights"
    assert checkpoint_state(tmp_path) == ("fresh", None)
    w.mkdir(parents=True)
    torch.save({"epoch": 41, "optimizer": {"state": {}}}, w / "last.pt")   # interrupted
    assert checkpoint_state(tmp_path) == ("resume", 42)
    torch.save({"epoch": -1, "optimizer": None}, w / "last.pt")            # finished or early-stopped
    assert checkpoint_state(tmp_path) == ("done", None)
    (w / "last.pt").write_bytes(b"truncated")
    with pytest.raises(SystemExit, match="cannot be read"):
        checkpoint_state(tmp_path)
