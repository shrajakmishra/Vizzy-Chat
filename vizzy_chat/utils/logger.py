"""
Vizzy Chat — Structured Logging
────────────────────────────────
Application-wide logger using Python's standard logging module.
Outputs to console and a rotating log file.
Zero external dependencies.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_LEVEL

# ── Ensure log directory exists ────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Formatter ──────────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

# ── Root logger setup ─────────────────────────────────────────
_root = logging.getLogger("vizzy")
_root.setLevel(logging.DEBUG)

# Prevent duplicate handlers on Streamlit rerun
if not _root.handlers:
    # Console handler
    _console = logging.StreamHandler(sys.stderr)
    _console.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    _console.setFormatter(_formatter)
    _root.addHandler(_console)

    # File handler (rotating, 10 MB max, keep 3 backups)
    _file_handler = RotatingFileHandler(
        LOG_DIR / "vizzy_chat.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_formatter)
    _root.addHandler(_file_handler)


def get_logger(module_name: str = __name__) -> logging.Logger:
    """Return a child logger under the 'vizzy' namespace."""
    return logging.getLogger(f"vizzy.{module_name}")
