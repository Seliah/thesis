"""Module for saving runtime values.

These values are meant to change throughout the lifetime of the program.
See :module:`definitions` for constant variables.
"""

from __future__ import annotations

from asyncio import Event
from logging import DEBUG, basicConfig, getLogger
from typing import TYPE_CHECKING

from analysis.read import load_motions

if TYPE_CHECKING:
    from analysis.definitions import MotionData

basicConfig(level=DEBUG)
_logger = getLogger()
_logger.debug("Logging set up.")
_logger.debug("Loading state.")

terminating = Event()
"""Flag to quit long running processes."""

motions: MotionData = load_motions()
