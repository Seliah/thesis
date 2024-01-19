"""Module for saving runtime values."""

from __future__ import annotations

from multiprocessing import Manager

from definitions import MotionData

from analysis.vision.read import load_motions

manager = Manager()
termination_event = manager.Event()

terminating = False
"""Flag to quit long running processes."""

motions: MotionData = load_motions()
