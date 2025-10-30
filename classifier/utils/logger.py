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

# Optional: Add file logging handler
# logger.add(
#     "logs/{time:YYYY-MM-DD}.log",
#     rotation="00:00",
#     retention="30 days",
#     level="DEBUG",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
#     encoding="utf-8",
# )

__all__ = ["logger", "console"]
