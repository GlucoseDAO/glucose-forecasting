"""Plain console output safe for Windows terminals (no Rich markup, no Unicode icons)."""
from __future__ import annotations

from rich.console import Console
from rich.markup import escape

# highlight=False avoids syntax coloring; escape() disables [bracket] markup.
_CONSOLE = Console(highlight=False)


def echo_plain(message: str) -> None:
    """Print one line; square brackets and backslashes are not interpreted as Rich styles."""
    _CONSOLE.print(escape(message))
