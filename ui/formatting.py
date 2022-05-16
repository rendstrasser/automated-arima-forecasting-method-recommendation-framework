"""Module for formatting utilities."""


def format_period_in_s(period_in_s: float):
    """Formats a period in milliseconds or seconds.

    Parameters
    ----------
    period_in_s: float
        Period in seconds.

    Returns
    -------
    formatted_period: str
        Formatted string of period.
    """
    period_in_m = period_in_s / 60

    if period_in_m >= 1:
        rest_in_s = period_in_s % 60
        return f"{period_in_m:.0f} minutes and {rest_in_s:.2f} seconds"

    if period_in_s >= 1:
        return f"{period_in_s:.4f} seconds"

    return f"{period_in_s * 1000:.0f} milliseconds"
