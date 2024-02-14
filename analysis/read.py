"""Module for reading application variables from disk."""

from numpy import load
from rich.console import Console

from analysis import definitions
from analysis.app_logging import logger

console = Console()


def load_motions() -> definitions.MotionData:
    """Load the motion store from disk."""
    try:
        logger.debug(f"Loading motion data from {definitions.PATH_MOTIONS}.")
        with definitions.PATH_MOTIONS.open("rb") as f:
            return load(f, allow_pickle=True).item()
    except FileNotFoundError:
        logger.info("No saved motion data was found, starting fresh.")
        return {}
