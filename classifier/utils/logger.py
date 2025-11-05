from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

# Create Rich Console instance
console = Console(stderr=True, force_terminal=True)

# Remove default loguru handler
logger.remove()

# Add Rich handler
logger.add(
    RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=True,
        markup=True,
    ),
    format="{message}",
    level="INFO",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

__all__ = ["logger", "console"]
