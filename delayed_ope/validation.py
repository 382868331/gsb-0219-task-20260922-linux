"""Shared numeric and type validation helpers.

Contract-wide rules implemented here:

* every numeric input must be a finite real (``NaN``/``+inf``/``-inf``
  are rejected);
* integer-typed parameters must not be ``bool`` (``True`` is an
  ``int`` in Python, but accepting it would silently mean 1);
* values of ``typing`` typing constructs are never validated here —
  only runtime values.
"""

import math
from typing import Any

from .errors import LogValidationError

__all__ = [
    "ensure_finite",
    "ensure_int",
    "ensure_hashable_key",
    "ensure_probability",
    "reject_bool",
]


def reject_bool(value: Any, name: str) -> None:
    """Reject booleans where an int-like parameter is expected."""
    if isinstance(value, bool):
        raise LogValidationError(
            f"{name} must be int, got bool ({value!r})"
        )


def ensure_finite(value: Any, name: str, exc_type: type = LogValidationError) -> float:
    """Return ``float(value)`` after rejecting non-real/non-finite input.

    ``int`` (excluding ``bool``) and ``float`` are accepted.  Strings
    are not coerced — a numeric field containing a string is a
    location-visible error, not an implicit parse.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise exc_type(
            f"{name} must be a real number (int/float, not bool), "
            f"got {type(value).__name__} ({value!r})"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise exc_type(f"{name} must be finite, got {value!r}")
    return fv


def ensure_probability(
    value: Any,
    name: str,
    *,
    zero_allowed: bool = True,
    one_allowed: bool = True,
    exc_type: type = LogValidationError,
) -> float:
    """Validate a probability: finite real in the requested sub-range."""
    p = ensure_finite(value, name, exc_type=exc_type)
    if p < 0.0 or p > 1.0:
        raise exc_type(f"{name} must lie in [0, 1], got {value!r}")
    if not zero_allowed and p == 0.0:
        raise exc_type(f"{name} must be strictly positive, got 0.0")
    if not one_allowed and p == 1.0:
        raise exc_type(f"{name} must be strictly below 1.0, got 1.0")
    return p


def ensure_int(value: Any, name: str, *, min_value: int | None = None) -> int:
    """Validate an integer parameter (booleans rejected)."""
    reject_bool(value, name)
    if not isinstance(value, int):
        raise LogValidationError(
            f"{name} must be int, got {type(value).__name__} ({value!r})"
        )
    if min_value is not None and value < min_value:
        raise LogValidationError(f"{name} must be >= {min_value}, got {value}")
    return value


def ensure_hashable_key(value: Any, name: str) -> object:
    """Validate a context/action label used as a dict key.

    Hashable non-None values of any type are allowed (they only ever
    serve as dict keys); ``None`` is rejected because it almost always
    marks a missing field.
    """
    if value is None:
        raise LogValidationError(f"{name} must not be None")
    try:
        hash(value)
    except TypeError as e:
        raise LogValidationError(f"{name} must be hashable, got {value!r}") from e
    return value
