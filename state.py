from __future__ import annotations

import os
from time import tzname

from pytz import timezone
from scipy.sparse import lil_array

from user_secrets import DATABASE_PATH

IS_SERVICE = os.getenv("RUN_AS_SERVICE", None) == "True"
"""Flag that state whether this program is run as a systemd service unit."""

TIMEZONE = timezone(tzname[0])
"""Timezone that the program runs in."""

PATH_MOTIONS = DATABASE_PATH / "motions.npy"

GRID_SIZE = (9, 16)
"""How many rows and columns should exist to define the cells."""
CELLS = GRID_SIZE[0] * GRID_SIZE[1]
INTERVAL = 1

terminating = False
"""Flag to quit long running processes."""

motions: dict[str, dict[str, lil_array]]
