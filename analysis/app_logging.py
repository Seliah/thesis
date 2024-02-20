"""Module to provide a logger instance to use globally."""
from logging import WARN, basicConfig, getLogger

from rich.logging import RichHandler

getLogger("httpx").setLevel(WARN)
getLogger("httpcore").setLevel(WARN)
getLogger("zeep").setLevel(WARN)

FORMAT = "%(message)s"
basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = getLogger()
