"""Shared simulation scene constants for teleop recording and rollout."""

from __future__ import annotations

import re

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

SUPPORTED_EVAL_BOX_COLORS = ("red",)
SUPPORTED_PLATE_COLORS = tuple(spec[0] for spec in PLATE_SPECS)


def build_task_prompt(box_color: str, plate_color: str) -> str:
    return f"put {box_color} box to {plate_color} plate"


def parse_task_prompt(prompt: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*put\s+(?P<box>\w+)\s+box\s+to\s+(?P<plate>\w+)\s+plate\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            "Expected prompt format: 'put <box_color> box to <plate_color> plate'"
        )
    return match.group("box").lower(), match.group("plate").lower()


def resolve_task_prompt_and_target(
    prompt: str | None,
    target_box: str,
    target_plate: str,
    explicit_target_box: bool,
    explicit_target_plate: bool,
) -> tuple[str, str, str]:
    target_box = target_box.lower()
    target_plate = target_plate.lower()
    if target_box not in SUPPORTED_EVAL_BOX_COLORS:
        raise ValueError(
            f"Unsupported target box color '{target_box}'. "
            f"Supported colors: {', '.join(SUPPORTED_EVAL_BOX_COLORS)}"
        )
    if target_plate not in SUPPORTED_PLATE_COLORS:
        raise ValueError(
            f"Unsupported target plate color '{target_plate}'. "
            f"Supported colors: {', '.join(SUPPORTED_PLATE_COLORS)}"
        )

    if prompt is None:
        return build_task_prompt(target_box, target_plate), target_box, target_plate

    prompt_box, prompt_plate = parse_task_prompt(prompt)
    if prompt_box not in SUPPORTED_EVAL_BOX_COLORS:
        raise ValueError(
            f"Unsupported prompt target box color '{prompt_box}'. "
            f"Supported colors: {', '.join(SUPPORTED_EVAL_BOX_COLORS)}"
        )
    if prompt_plate not in SUPPORTED_PLATE_COLORS:
        raise ValueError(
            f"Unsupported prompt target plate color '{prompt_plate}'. "
            f"Supported colors: {', '.join(SUPPORTED_PLATE_COLORS)}"
        )
    if explicit_target_box and prompt_box != target_box:
        raise ValueError(
            f"--prompt box color '{prompt_box}' conflicts with --target-box '{target_box}'"
        )
    if explicit_target_plate and prompt_plate != target_plate:
        raise ValueError(
            f"--prompt plate color '{prompt_plate}' conflicts with --target-plate '{target_plate}'"
        )
    return prompt, prompt_box, prompt_plate
