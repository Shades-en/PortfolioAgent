"""
Reusable logging utility with colored output and OpenTelemetry trace_id injection.

Recommended usage:
    from ai_server.utils.logger import setup_logging
    import logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    logger.info("Loaded module")

Environment variables:
    LOG_LEVEL: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
    NO_COLOR: if truthy (1/true/yes/on), disables colored output
    LOG_TIME_FORMAT: strftime format for timestamps. Default: %H:%M:%S
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union
import hashlib
from ai_server.utils.general import _env_flag

try:
    from opentelemetry import trace
except Exception:  # pragma: no cover - otel optional at import time
    trace = None  # type: ignore


# ANSI color codes
RESET = "\x1b[0m"
BOLD = "\x1b[1m"

COLORS = {
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "bright_black": "\x1b[90m",
    "bright_red": "\x1b[91m",
    "bright_green": "\x1b[92m",
    "bright_yellow": "\x1b[93m",
    "bright_blue": "\x1b[94m",
    "bright_magenta": "\x1b[95m",
    "bright_cyan": "\x1b[96m",
    "bright_white": "\x1b[97m",
}


# Dynamic per-source coloring: choose a color deterministically from a palette
_COLOR_PALETTE = [
    "bright_blue",
    "bright_cyan",
    "bright_magenta",
    "bright_green",
    "yellow",
    "bright_yellow",
    "bright_white",
    "cyan",
    "magenta",
    "blue",
    "bright_red",
    "bright_black",
    "green",
]


def _color_for_source(name: str) -> str:
    if not name:
        return COLORS["white"]
    # Use a stable hash (md5 first byte) to pick a palette index
    h = hashlib.md5(name.encode("utf-8")).digest()[0]
    key = _COLOR_PALETTE[h % len(_COLOR_PALETTE)]
    return COLORS.get(key, COLORS["white"])


# Per-level colors
LEVEL_COLORS = {
    logging.DEBUG: COLORS["bright_black"],
    logging.INFO: COLORS["green"],
    logging.WARNING: COLORS["yellow"],
    logging.ERROR: COLORS["red"],
    logging.CRITICAL: COLORS["bright_red"],
}


def _supports_color() -> bool:
    # Only disable color when NO_COLOR is an explicit truthy value
    if _env_flag("NO_COLOR", default=False):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

class OTelColorFormatter(logging.Formatter):
    """Formatter that adds colors and OTel trace_id if available."""

    def __init__(self, use_colors: bool = True, timefmt: Optional[str] = None) -> None:
        super().__init__()
        self.use_colors = use_colors
        self.timefmt = timefmt or os.getenv("LOG_TIME_FORMAT", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        ts = datetime.fromtimestamp(record.created).strftime(self.timefmt)

        # Level name with color
        levelname = record.levelname
        if self.use_colors:
            level_color = LEVEL_COLORS.get(record.levelno, COLORS["white"])
            levelname_fmt = f"{level_color}{BOLD}{levelname:8s}{RESET}"
        else:
            levelname_fmt = f"{levelname:8s}"

        # Source label and color (prefer explicit record.source; fallback to logger name)
        source_name = getattr(record, "source", None) or record.name or "default"
        if self.use_colors:
            src_color = _color_for_source(source_name)
            source_fmt = f"{src_color}{source_name}{RESET}"
        else:
            source_fmt = source_name

        # OTel trace id if available
        trace_id_str = ""
        if trace is not None:
            try:
                span = trace.get_current_span()
                ctx = span.get_span_context() if span is not None else None
                if ctx and ctx.is_valid:
                    trace_id = ctx.trace_id  # int
                    trace_id_str = f" trace_id={trace_id:032x}"
            except Exception:
                trace_id_str = ""

        message = record.getMessage()

        return f"{ts} | {levelname_fmt} | {source_fmt} | {message}{trace_id_str}"


def _coerce_level(level: Optional[Union[str, int]]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)


def setup_logging(level: Optional[Union[str, int]] = None) -> logging.Logger:
    """Configure the root logger with colored formatter for modules using logging.getLogger(__name__)."""
    root = logging.getLogger()
    root.setLevel(_coerce_level(level))
    formatter = OTelColorFormatter(use_colors=_supports_color())
    found_stream = False
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setLevel(root.level)
            h.setFormatter(formatter)
            found_stream = True
    if not found_stream:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(root.level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return root


__all__ = [
    "OTelColorFormatter",
    "setup_logging",
]
