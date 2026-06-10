"""Shared simulation scene constants for teleop recording and rollout."""

from __future__ import annotations

import numpy as np


BOX_SPECS = (
    ("red", "red_box", [1.0, 0.0, 0.0, 1.0], (30, 30, 230)),
    ("orange", "orange_box", [1.0, 0.45, 0.0, 1.0], (0, 150, 255)),
    ("black", "black_box", [0.02, 0.02, 0.02, 1.0], (20, 20, 20)),
)

PLATE_SPECS = (
    ("blue", "blue_plate", [0.05, 0.18, 1.0, 1.0], (255, 90, 20)),
    ("green", "green_plate", [0.1, 0.7, 0.2, 1.0], (45, 180, 45)),
    ("yellow", "yellow_plate", [1.0, 0.85, 0.05, 1.0], (20, 220, 245)),
)

RANDOM_WORKSPACE = dict(x=(0.195, 0.773), y=(-2.059, -0.891))
FIXED_RED_BOX_XY = np.array([0.457, -1.612], dtype=np.float64)
FIXED_BLUE_PLATE_XY = np.array([0.577, -1.612], dtype=np.float64)

