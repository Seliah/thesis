"""Module for saving runtime values.

These values are meant to change throughout the lifetime of the program.
See :module:`definitions` for constant variables.
"""

from __future__ import annotations

from asyncio import Event
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.definitions import MotionData

terminating = Event()
"""Flag to quit long running processes."""

motions: MotionData = {}
