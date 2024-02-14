from logging import basicConfig, getLogger

from rich.logging import RichHandler

FORMAT = "%(message)s"
basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()],
)

logger = getLogger()
