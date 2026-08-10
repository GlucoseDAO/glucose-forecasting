"""Plain console output safe for Windows terminals (no Rich markup, no Unicode icons)."""
from __future__ import annotations

from rich.console import Console
from rich.markup import escape

from common.console import safe_echo

# highlight=False avoids syntax coloring; escape() disables [bracket] markup.
_CONSOLE = Console(highlight=False)


def echo_plain(message: str) -> None:
    """Print one line; square brackets and backslashes are not interpreted as Rich styles."""
    try:
        _CONSOLE.print(escape(message))
    except UnicodeEncodeError:
        safe_echo(message)
