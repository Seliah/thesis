"""Module for saving constant values.

These values should be available at the start of the program and never change.
See :module:`state` for runtime variables.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

from scipy.sparse import lil_array

from user_secrets import DATABASE_PATH

IS_SERVICE = os.getenv("RUN_AS_SERVICE", None) == "True"
"""Flag that state whether this program is run as a systemd service unit."""

TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo
"""Timezone that the program runs in."""

PATH_MOTIONS = DATABASE_PATH / "motions.npy"

GRID_SIZE = (9, 16)
"""How many rows and columns should exist to define the cells."""
CELLS = GRID_SIZE[0] * GRID_SIZE[1]
INTERVAL = 1

MotionData = Dict[str, Dict[str, lil_array]]
