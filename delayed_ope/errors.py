"""Exception hierarchy for delayed-feedback off-policy evaluation.

All *contract* violations are value errors carrying a location (field,
record index, or context/action pair) so that callers can pinpoint the
offending input.  Non-finite computed results are deliberately *not*
``ValueError``: the inputs were valid, but floating-point arithmetic
overflowed, which is a separate numerical failure.
"""

__all__ = [
    "OPEError",
    "LogValidationError",
    "PolicyValidationError",
    "ZeroSupportError",
    "QTableValidationError",
    "NumericalOverflowError",
    "NoMatureRecordsError",
]


class OPEError(ValueError):
    """Base class for contract violations in this package."""


class LogValidationError(OPEError):
    """A log record or evaluation-cutoff argument violates the contract."""


class PolicyValidationError(OPEError):
    """The target-policy probability table is malformed or incomplete."""


class ZeroSupportError(PolicyValidationError):
    """Target policy puts positive probability on an unsupported action.

    Raised when, for a context present in the log, pi(a|x) > 0 while
    action ``a`` was never observed under that context (the empirical
    support of the logging policy is empty for that pair).  The IPS
    importance ratio would be undefined (division by zero), so the
    evaluation is refused instead of silently clipping or ignoring it.
    """


class QTableValidationError(OPEError):
    """The supplied q_hat table is malformed or misses required entries."""


class NumericalOverflowError(ArithmeticError):
    """An intermediate weight or aggregate overflowed to infinity.

    Inputs were all finite, but the computed arithmetic was not; the
    finite-result contract forbids reporting such values.
    """


class NoMatureRecordsError(OPEError):
    """No record had its reward observed by the cutoff.

    This is not a malformed-input error: the log may be perfectly
    valid, but at the requested cutoff every record is still pending.
    Delayed-but-unobserved rewards are never imputed as zero, so no
    estimate exists.  The (empty) maturity accounting is attached.
    """

    def __init__(self, message: str, *, total: int, immature: int, cutoff: float):
        super().__init__(message)
        self.total = total
        self.immature = immature
        self.cutoff = cutoff
