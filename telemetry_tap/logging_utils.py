from __future__ import annotations

import logging

TRACE_LEVEL = 5


def _trace(self: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


def configure_logging(level: int) -> None:
    logging.addLevelName(TRACE_LEVEL, "TRACE")
    setattr(logging.Logger, "trace", _trace)
    handlers: list[logging.Handler] = []
    try:
        from colorlog import ColoredFormatter  # type: ignore

        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)s %(name)s %(message)s",
            log_colors={
                "TRACE": "cyan",
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handlers.append(handler)
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handlers.append(handler)
    logging.basicConfig(level=level, handlers=handlers)


def resolve_log_level(verbosity: int, fallback: str) -> int:
    if verbosity >= 2:
        return TRACE_LEVEL
    if verbosity == 1:
        return logging.DEBUG
    return logging._nameToLevel.get(fallback.upper(), logging.INFO)
