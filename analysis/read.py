"""Module for reading application variables from disk."""
from logging import getLogger

from numpy import load
from rich.console import Console

from analysis import definitions

_logger = getLogger(__name__)
console = Console()


def load_motions() -> definitions.MotionData:
    """Load the motion store from disk."""
    try:
        _logger.debug(f"Loading motion data from {definitions.PATH_MOTIONS}.")
        with definitions.PATH_MOTIONS.open("rb") as f:
            return load(f, allow_pickle=True).item()
    except FileNotFoundError:
        _logger.info("No saved motion data was found, starting fresh.")
        return {}
