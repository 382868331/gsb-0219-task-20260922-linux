"""delayed_ope — off-policy evaluation with delayed reward settlement.

Public surface (see module README for the full contract):

* :class:`LogRecord`         -- one logged decision/feedback tuple;
* :class:`EstimatorResult`   -- IPS / SNIPS / DR estimates plus ESS;
* :class:`MaturityReport`    -- which records were mature at cutoff;
* :func:`evaluate_policy`    -- the single entry point;
* exception types in :mod:`delayed_ope.errors`.
"""

from .core import (
    DEFAULT_MAX_ACTIONS,
    DEFAULT_MAX_RECORDS,
    EstimatorResult,
    LogRecord,
    MaturityReport,
    evaluate_policy,
)
from .errors import (
    LogValidationError,
    NoMatureRecordsError,
    NumericalOverflowError,
    OPEError,
    PolicyValidationError,
    QTableValidationError,
    ZeroSupportError,
)

__all__ = [
    "LogRecord",
    "EstimatorResult",
    "MaturityReport",
    "evaluate_policy",
    "DEFAULT_MAX_RECORDS",
    "DEFAULT_MAX_ACTIONS",
    "OPEError",
    "LogValidationError",
    "PolicyValidationError",
    "ZeroSupportError",
    "QTableValidationError",
    "NumericalOverflowError",
    "NoMatureRecordsError",
]

__version__ = "1.0.0"
