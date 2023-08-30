from typing import Any


class DummyConsole:
    """
    Class that acts as a drop in replacement for rich's Console, in case
    rich is not present on the system, or we are not wanting rich output
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.level = float("-inf")

    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args)
