"""
log.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

A module for setting up logging.
"""
import logging
import os
from typing import Any
from typing import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codem.preprocessing import CodemParameters
    from vcd.preprocessing import VCDParameters

try:
    import websocket
    from pythonjsonlogger import jsonlogger
except ImportError:
    pass
else:

    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(
            self,
            log_record: Dict[str, Any],
            record: logging.LogRecord,
            message_dict: Dict[str, Any],
        ) -> None:
            super().add_fields(log_record, record, message_dict)
            if log_record.get("level"):
                log_record["level"] = log_record["level"].upper()
            else:
                log_record["level"] = record.levelname

            if log_record.get("type") is None:
                log_record["type"] = "log_message"
            return None

    class WebSocketHandler(logging.Handler):
        def __init__(self, level: str, websocket: "websocket.WebSocket") -> None:
            super().__init__(level)
            self.ws = websocket
            # TODO: check if websocket is already connected?

        def emit(self, record: logging.LogRecord) -> None:
            msg = self.format(record)
            _ = self.ws.send(msg)
            return None

        def close(self) -> None:
            self.ws.close()
            return super().close()


class Log:
    def __init__(self, config: Dict[str, Any]):
        """
        Creates logging formatting and structure

        Parameters
        ----------
        config:
            Dictionary representing the runtime config
        """

        self.logger = logging.getLogger("codem")
        self.logger.setLevel(logging.DEBUG)

        # disable loggers
        logging.getLogger("matplotlib.font_manager").disabled = True

        # File Handler for Logging
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler = logging.FileHandler(
            os.path.join(config.get("OUTPUT_DIR", "."), "log.txt")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        self.logger.addHandler(file_handler)
        self.relay = None

        # Supplemental Handler
        if config["LOG_TYPE"] == "rich":
            from rich.logging import RichHandler

            log_handler = RichHandler()
        elif config["LOG_TYPE"] == "websocket":
            formatter = CustomJsonFormatter()
            self.relay = websocket.WebSocket()
            url = f'ws://{config["WEBSOCKET_URL"]}/websocket'
            try:
                self.relay.connect(url)
            except ConnectionRefusedError as err:
                raise ConnectionRefusedError(f"Connection Refused to {url}")
            log_handler = WebSocketHandler("DEBUG", websocket=self.relay)
            log_handler.setFormatter(formatter)
        else:
            log_handler = logging.StreamHandler()
        log_handler.setLevel("DEBUG")
        self.logger.addHandler(log_handler)

    def __del__(self) -> None:
        if isinstance(self.logger, WebSocketHandler):
            self.logger.close()
