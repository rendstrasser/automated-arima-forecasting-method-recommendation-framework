"""General utility functions."""
import sys


def debugger_is_active() -> bool:
    """Return if the debugger is currently active.

    Returns
    -------
    debuggger_is_active: bool
        True, if the debugger is active.
    """

    gettrace = getattr(sys, "gettrace", lambda: None)
    return gettrace() is not None
