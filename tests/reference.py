"""Independent exact reference for the off-policy estimators.

This module deliberately does **not** import :mod:`delayed_ope`.  Tests
use it to derive expected values from first principles with
:class:`fractions.Fraction`, so the expected numbers can never simply
mirror a bug in the code under test.

A "spec" record here is a plain tuple::

    (context, action, reward, logged_propensity, decision_time, observed_time)

with reward/propensity as :class:`~fractions.Fraction` and times as
integers.  The test layer converts the exact rationals to ``float``
only when feeding the core package.
"""

from fractions import Fraction
from typing import Mapping, Sequence

Spec = tuple[object, object, Fraction, Fraction, int, int]


def mature_specs(specs: Sequence[Spec], cutoff: int) -> list[Spec]:
    """Maturity rule, independently: observed_time <= cutoff."""
    return [s for s in specs if s[5] <= cutoff]


def _weight(
    spec: Spec,
    pi: Mapping[object, Mapping[object, Fraction]],
    clip: Fraction | None,
) -> Fraction:
    x, a, _r, mu, _td, _to = spec
    w = pi[x][a] / mu
    if clip is not None and w > clip:
        w = clip
    return w


def _baseline(
    x: object,
    pi: Mapping[object, Mapping[object, Fraction]],
    q: Mapping[object, Mapping[object, Fraction]],
) -> Fraction:
    return sum(
        p * q[x][a] for a, p in pi[x].items() if p > 0
    )


def estimate(
    specs: Sequence[Spec],
    pi: Mapping[object, Mapping[object, Fraction]],
    cutoff: int,
    *,
    q: Mapping[object, Mapping[object, Fraction]] | None = None,
    clip: Fraction | None = None,
) -> dict[str, Fraction | None]:
    """Exact rational IPS / SNIPS / DR over the mature subset.

    Keys: ``ips``, ``snips`` (None if total weight is zero),
    ``dr`` (None if no q table), ``ess`` (None if total weight zero),
    ``sum_weight``, ``n_mature``.
    """
    mature = mature_specs(specs, cutoff)
    n = len(mature)
    sum_wr = Fraction(0)
    sum_w = Fraction(0)
    sum_ww = Fraction(0)
    dr_sum = Fraction(0)
    for s in mature:
        x, a, r, _mu, _td, _to = s
        w = _weight(s, pi, clip)
        sum_wr += w * r
        sum_w += w
        sum_ww += w * w
        if q is not None:
            # When pi(a|x) == 0 the correction term is exactly zero and
            # q_hat is not contractually required for that action.
            if pi[x][a] > 0:
                correction = w * (r - q[x][a])
            else:
                correction = Fraction(0)
            dr_sum += _baseline(x, pi, q) + correction
    return {
        "n_mature": n,
        "ips": sum_wr / n if n else None,
        "snips": (sum_wr / sum_w) if sum_w > 0 else None,
        "dr": (dr_sum / n) if (q is not None and n) else None,
        "ess": (sum_w * sum_w / sum_ww) if sum_w > 0 else None,
        "sum_weight": sum_w,
    }


def true_value(
    contexts: Sequence[object],
    pi: Mapping[object, Mapping[object, Fraction]],
    reward_table: Mapping[object, Mapping[object, Fraction]],
) -> Fraction:
    """V(pi) = mean over contexts of sum_a pi(a|x) r(x,a).

    Contexts are weighted uniformly, matching the structured logs the
    tests build (equal numbers of draws per context).
    """
    total = Fraction(0)
    for x in contexts:
        total += sum(pi[x][a] * reward_table[x][a] for a in pi[x])
    return total / len(contexts)


def policy_to_float(
    pi: Mapping[object, Mapping[object, Fraction]],
) -> dict[object, dict[object, float]]:
    """Exact rational policy -> float policy for the float-based core."""
    return {x: {a: float(p) for a, p in row.items()} for x, row in pi.items()}


def specs_to_records(specs: Sequence[Spec]):
    """Build core LogRecord inputs from exact specs (floats for nums).

    Imported lazily inside the function body so this reference module
    itself never depends on the package under test.
    """
    from delayed_ope import LogRecord

    return [
        LogRecord(
            context=x,
            action=a,
            reward=float(r),
            logged_propensity=float(mu),
            decision_time=td,
            observed_time=to,
        )
        for (x, a, r, mu, td, to) in specs
    ]
