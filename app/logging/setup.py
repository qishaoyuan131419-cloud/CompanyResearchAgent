import logging
import sys
from datetime import UTC, datetime
from typing import Any

from pythonjsonlogger.json import JsonFormatter

_STANDARD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class ServiceJsonFormatter(JsonFormatter):
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.fromtimestamp(record.created, UTC).isoformat(
            timespec="milliseconds"
        )
        log_record["level"] = record.levelname
        log_record["logger"] = record.name


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ServiceJsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


class StructuredLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(event, extra=self._safe_extra(event, fields))

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(event, extra=self._safe_extra(event, fields))

    @staticmethod
    def _safe_extra(event: str, fields: dict[str, Any]) -> dict[str, Any]:
        safe = {key: value for key, value in fields.items() if key not in _STANDARD_FIELDS}
        safe["event"] = event
        return safe
