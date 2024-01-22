"""Module for saving runtime values.

These values are meant to change throughout the lifetime of the program.
See :module:`definitions` for constant variables.
"""

from __future__ import annotations

from multiprocessing import Manager

from analysis.definitions import MotionData
from analysis.vision.read import load_motions

manager = Manager()
termination_event = manager.Event()

terminating = False
"""Flag to quit long running processes."""

motions: MotionData = load_motions()
