"""Default settings for variants of SARIMAX sample generators.

SARIMAX defaults are as close as possible to the defaults in the library pmdarima [1],
which is a python port for the auto.arima function in R.
Using the maximum defaults is also partially suggested by [2].

References
----------
[1] http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html#pmdarima.arima.auto_arima  # noqa
[2] https://www.sciencedirect.com/science/article/pii/0167923695000313
"""
from math import sqrt

DEFAULT_SARIMAX_MIN_p = 1
DEFAULT_SARIMAX_MAX_p = 4
DEFAULT_SARIMAX_MIN_d = 1
DEFAULT_SARIMAX_MAX_d = 2
DEFAULT_SARIMAX_MIN_q = 1
DEFAULT_SARIMAX_MAX_q = 4
DEFAULT_SARIMAX_MIN_P = 1
DEFAULT_SARIMAX_MAX_P = 2
DEFAULT_SARIMAX_MIN_D = 0
DEFAULT_SARIMAX_MAX_D = 1
DEFAULT_SARIMAX_MIN_Q = 1
DEFAULT_SARIMAX_MAX_Q = 2
# DEFAULT_SARIMAX_M_CANDIDATES = [7, 14, 30, 31]
DEFAULT_SARIMAX_M_CANDIDATES = [7, 14]
DEFAULT_SARIMAX_MIN_m = 7
DEFAULT_SARIMAX_MAX_m = 31

DEFAULT_MIN_VARIANCE = 0.1
DEFAULT_MAX_VARIANCE = 5
DEFAULT_MIN_SD = sqrt(DEFAULT_MIN_VARIANCE)
DEFAULT_MAX_SD = sqrt(DEFAULT_MAX_VARIANCE)
