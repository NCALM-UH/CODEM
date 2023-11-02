import json
from contextlib import ContextDecorator
from typing import Any
from typing import Dict


try:
    import websocket
except ImportError:
    pass
else:

    class WebSocketProgress(ContextDecorator):
        def __init__(self, url: str) -> None:
            super().__init__()
            self.ws = websocket.WebSocket()
            self.tasks: Dict[str, int] = {}
            self.current: Dict[str, int] = {}
            self.url = url

        def __enter__(self) -> Any:
            url = f'ws://{self.url}/websocket'
            try:
                self.ws.connect(url)
            except ConnectionRefusedError as err:
                raise ConnectionRefusedError(f"Connection Refused to {url}")
            return self

        def __exit__(self, *args: Any, **kwargs: Any) -> None:
            self.ws.close()
            return None

        def advance(self, name: str, value: int) -> None:
            self.current[name] += value
            new_value = self.current[name]
            self.ws.send(json.dumps({"advance": new_value, "type": "progress"}))
            return None

        def add_task(self, title: str, total: int) -> str:
            self.tasks[title] = total
            self.current[title] = 0
            return title
