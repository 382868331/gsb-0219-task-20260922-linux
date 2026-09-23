"""Off-policy evaluation with delayed, asynchronous reward settlement.

Implements three estimators of the value
``V(pi) = E_{x~D}[ sum_a pi(a|x) * r(x,a) ]``

from a bandit feedback log with delayed rewards:

* **IPS**   -- inverse propensity scoring,           :math:`\\pi/\\mu \\cdot r`;
* **SNIPS** -- self-normalised IPS,                  weighted mean of rewards;
* **DR**    -- doubly robust with a supplied q_hat, :math:`m(x) + \\pi/\\mu(r-q)`.

Maturity rule (the central contract of this task)
-------------------------------------------------
A record is *mature* at the evaluation cutoff ``T`` iff its reward was
actually observed by then, i.e. ``observed_time <= T`` (records also
satisfy ``observed_time >= decision_time``, enforced on input).
Immature records (decision made but settlement pending) are
**excluded** from every estimate.  They are never treated as reward 0.
Callers receive the exact maturity counts and indices.

Estimators here are *sample-average* estimators over mature records
(``1/n_m * sum_i ...``), as standard in bandit OPE; when the target
policy mixes over actions that the log never explored for a context,
exact evaluation is impossible and the call is refused
(``ZeroSupportError``), not approximated.

Scope/limits (also documented in README): finite tabular setting only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .errors import (
    LogValidationError,
    NoMatureRecordsError,
    NumericalOverflowError,
    PolicyValidationError,
    QTableValidationError,
    ZeroSupportError,
)
from .validation import (
    ensure_finite,
    ensure_hashable_key,
    ensure_int,
    ensure_probability,
)

__all__ = [
    "LogRecord",
    "EstimatorResult",
    "MaturityReport",
    "evaluate_policy",
    "DEFAULT_MAX_RECORDS",
    "DEFAULT_MAX_ACTIONS",
]

# Interface limits: declared as input boundaries in README.  The
# implementation is O(n) in memory and time; the caps guard against
# accidental denial-of-service in batch use, they are not algorithmic
# requirements.
DEFAULT_MAX_RECORDS = 1_000_000
DEFAULT_MAX_ACTIONS = 10_000


@dataclass(frozen=True)
class LogRecord:
    """One logged bandit decision with delayed reward feedback.

    Attributes:
        context: immutable/hashable context label ``x``.
        action:  immutable/hashable played action ``a``.
        reward:  observed finite real reward (valid only if mature at
                 the cutoff; the value is nevertheless required to be
                 present and finite in every record — see README).
        logged_propensity: probability mu(a|x) under which ``a`` was
                 sampled; strictly positive.
        decision_time: time the action was chosen, ``t_d``.
        observed_time: time the reward was observed, ``t_o >= t_d``.
    """

    context: object
    action: object
    reward: float
    logged_propensity: float
    decision_time: float
    observed_time: float

    def __post_init__(self) -> None:
        # Build/validate through one helper so the error always names
        # both the field and the record (records are often anonymous
        # dicts from JSON logs, so no global index here).
        validate_record(self)


def validate_record(rec: LogRecord) -> None:
    """Validate a single :class:`LogRecord`, raising locatable errors."""
    where = f"record(context={rec.context!r}, action={rec.action!r})"
    ensure_hashable_key(rec.context, f"{where}.context")
    ensure_hashable_key(rec.action, f"{where}.action")
    ensure_finite(rec.reward, f"{where}.reward")
    p = ensure_finite(
        rec.logged_propensity, f"{where}.logged_propensity"
    )
    if p <= 0.0:
        raise LogValidationError(
            f"{where}.logged_propensity must be strictly positive (got {p!r}); "
            "actions with zero logging probability cannot appear in the log"
        )
    if p > 1.0:
        raise LogValidationError(
            f"{where}.logged_propensity must be <= 1.0, got {p!r}"
        )
    td = ensure_finite(rec.decision_time, f"{where}.decision_time")
    to = ensure_finite(rec.observed_time, f"{where}.observed_time")
    if to < td:
        raise LogValidationError(
            f"{where}.observed_time ({to!r}) precedes decision_time ({td!r})"
        )


@dataclass(frozen=True)
class MaturityReport:
    """Exactly which records fed the estimators and which did not.

    Attributes:
        cutoff: the evaluation cutoff time ``T``.
        total: number of validated input records.
        mature: records with ``observed_time <= T`` (used by estimates).
        immature: records with ``observed_time > T`` (excluded, never
            treated as zero reward).
        mature_indices: positions of mature records in the input order.
        immature_indices: positions of immature records in input order.
    """

    cutoff: float
    total: int
    mature: int
    immature: int
    mature_indices: tuple[int, ...] = field(default_factory=tuple)
    immature_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EstimatorResult:
    """Point estimates and diagnostics.

    The three estimates are sample-average estimates over the mature
    records:

    ``ips  = mean_i  w_i r_i``
    ``snips= (sum w r)/(sum w)`` (``None`` if no positive weight)
    ``dr   = mean_i [ m(x_i) + w_i (r_i - q_hat(x_i,a_i)) ]``
           (``None`` if q_hat not supplied)

    Unclipped values are always reported alongside clipped ones so a
    caller can see the effect of clipping rather than trust it blindly.
    """

    maturity: MaturityReport
    n_mature: int
    ips: float
    snips: float
    dr: float | None
    clip: float | None
    ips_unclipped: float
    snips_unclipped: float
    dr_unclipped: float | None
    ess_unclipped: float | None
    ess_clipped: float | None
    max_weight: float

    @property
    def ess(self) -> float | None:
        """Effective sample size under the (clipped, if any) weights."""
        return self.ess_clipped if self.clip is not None else self.ess_unclipped


def _validate_target_policy(
    policy: Mapping[object, Mapping[object, float]],
    contexts: Sequence[object],
    logged_actions: Mapping[object, set[object]],
) -> dict[object, dict[object, float]]:
    """Validate the pi table; return it as a plain nested dict.

    Checks per context present in the log: mapping exists, probabilities
    are finite and in [0,1], rows sum to 1 (within a documented strict
    tolerance), and at least one action has positive probability.

    Every action observed under a context must be listed in that
    context's row explicitly (probability 0 is allowed): a missing
    entry is ambiguous and refused rather than silently read as 0.
    """
    if not isinstance(policy, Mapping):
        raise PolicyValidationError(
            "target_policy must be a Mapping context -> {action: probability}, "
            f"got {type(policy).__name__}"
        )
    result: dict[object, dict[object, float]] = {}
    # Distinct contexts, validated in the order they first appear so
    # error messages are reproducible.
    seen: set[object] = set()
    ordered_contexts: list[object] = []
    for x in contexts:
        if x not in seen:
            seen.add(x)
            ordered_contexts.append(x)

    for x in ordered_contexts:
        if x not in policy:
            raise PolicyValidationError(
                f"target_policy has no row for context {x!r} that occurs in the log"
            )
        row = policy[x]
        if not isinstance(row, Mapping) or len(row) == 0:
            raise PolicyValidationError(
                f"target_policy[{x!r}] must be a non-empty Mapping action->probability"
            )
        if len(row) > DEFAULT_MAX_ACTIONS:
            raise PolicyValidationError(
                f"target_policy[{x!r}] lists {len(row)} actions, "
                f"above the limit {DEFAULT_MAX_ACTIONS}"
            )
        probs: dict[object, float] = {}
        total = 0.0
        for a, p in row.items():
            ensure_hashable_key(a, f"target_policy[{x!r}] action key")
            if a in probs:
                # Mapping already guarantees uniqueness, but keep the
                # guard explicit for non-dict Mapping implementations.
                raise PolicyValidationError(
                    f"target_policy[{x!r}] lists action {a!r} more than once"
                )
            pv = ensure_probability(
                p,
                f"target_policy[{x!r}][{a!r}]",
                exc_type=PolicyValidationError,
            )
            probs[a] = pv
            total += pv
        if not math.isfinite(total):
            raise PolicyValidationError(
                f"target_policy[{x!r}] probability sum is not finite"
            )
        if total <= 0.0:
            raise PolicyValidationError(
                f"target_policy[{x!r}] must put positive probability somewhere"
            )
        # Strict normalisation tolerance; rows must genuinely sum to 1.
        if abs(total - 1.0) > 1e-12:
            raise PolicyValidationError(
                f"target_policy[{x!r}] probabilities sum to {total!r}, "
                "expected 1.0 within 1e-12"
            )
        for a_logged in logged_actions.get(x, ()):
            if a_logged not in probs:
                raise PolicyValidationError(
                    f"target_policy[{x!r}] does not list logged action "
                    f"{a_logged!r}; add it explicitly (probability 0 is "
                    "allowed) instead of relying on an implicit default"
                )
        result[x] = probs
    return result


def _validate_qhat(
    q_hat: Mapping[object, Mapping[object, float]],
    pi_rows: Mapping[object, Mapping[object, float]],
) -> dict[object, dict[object, float]]:
    """Validate q_hat: it must give a finite value for every (x,a) the
    target policy assigns positive probability (extra entries allowed)."""
    if not isinstance(q_hat, Mapping):
        raise QTableValidationError(
            "q_hat must be a Mapping context -> {action: expected reward}, "
            f"got {type(q_hat).__name__}"
        )
    out: dict[object, dict[object, float]] = {}
    for x, row in pi_rows.items():
        if x not in q_hat:
            raise QTableValidationError(
                f"q_hat has no row for context {x!r} required by target policy"
            )
        qrow = q_hat[x]
        if not isinstance(qrow, Mapping):
            raise QTableValidationError(
                f"q_hat[{x!r}] must be a Mapping action -> finite real"
            )
        out_row: dict[object, float] = {}
        for a, p in row.items():
            if p > 0.0:
                if a not in qrow:
                    raise QTableValidationError(
                        f"q_hat[{x!r}] is missing action {a!r} "
                        "(required: pi(a|x) > 0)"
                    )
                out_row[a] = ensure_finite(
                    qrow[a],
                    f"q_hat[{x!r}][{a!r}]",
                    exc_type=QTableValidationError,
                )
        out[x] = out_row
    return out


def _clip_weight(w: float, clip: float | None) -> float:
    if clip is None:
        return w
    return w if w < clip else clip


def _ess_from_totals(sum_w: float, sum_ww: float) -> float | None:
    """Effective sample size (Kish): ``(sum w)^2 / sum w^2``.

    ``None`` if no weight is positive.  Overflow raises
    ``NumericalOverflowError`` rather than returning infinity/NaN.
    """
    if not math.isfinite(sum_w) or not math.isfinite(sum_ww):
        raise NumericalOverflowError(
            "weight aggregate overflowed while computing effective sample size"
        )
    if sum_w <= 0.0:
        return None
    ess = sum_w * sum_w / sum_ww
    if not math.isfinite(ess):
        raise NumericalOverflowError("effective sample size overflowed")
    return ess


def evaluate_policy(
    records: Sequence[LogRecord],
    target_policy: Mapping[object, Mapping[object, float]],
    cutoff: float,
    *,
    q_hat: Mapping[object, Mapping[object, float]] | None = None,
    clip: float | None = None,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> EstimatorResult:
    """Run IPS / SNIPS / DR on the mature subset of ``records``.

    Args:
        records: validated bandit log, in any order.
        target_policy: ``{context: {action: pi(a|x)}}``.  Every context
            occurring in the log needs a complete row summing to 1.
        cutoff: evaluation time ``T``.  Only records with
            ``observed_time <= T`` are mature.
        q_hat: optional expected-reward table for the DR estimate.
            Must cover every positive-probability target action.
        clip: optional explicit weight clipping threshold ``M > 0``.
            Weights are clipped to ``min(pi(a|x)/mu, M)`` for the
            clipped columns; unclipped columns are always returned.
        max_records: interface cap on log size (bool rejected).

    Returns:
        EstimatorResult with both (un)clipped estimates, ESS and the
        maturity accounting.

    Raises:
        LogValidationError: malformed record / cutoff / clip.
        PolicyValidationError: malformed target policy.
        ZeroSupportError: pi(a|x) > 0 for an (x,a) pair never observed
            in the log (zero empirical logging support).
        QTableValidationError: malformed/incomplete q_hat.
        NumericalOverflowError: finite inputs produced an infinite
            aggregate.
    """
    # ---- argument-level validation (fail before touching any state) ----
    if isinstance(records, bool) or not isinstance(records, Sequence):
        raise LogValidationError(
            "records must be a Sequence of LogRecord, got "
            f"{type(records).__name__}"
        )
    ensure_int(max_records, "max_records", min_value=1)
    if len(records) > max_records:
        raise LogValidationError(
            f"record count {len(records)} exceeds max_records {max_records}"
        )
    if len(records) == 0:
        raise LogValidationError("records must contain at least one entry")
    cutoff_f = ensure_finite(cutoff, "cutoff")
    clip_f: float | None = None
    if clip is not None:
        clip_f = ensure_finite(clip, "clip")
        if clip_f <= 0.0:
            raise LogValidationError(f"clip must be > 0, got {clip!r}")

    # Records validate themselves on construction, but re-validate
    # defensively for hand-built duck-typed sequences and to attach the
    # record index to the message.
    contexts: list[object] = []
    logged_actions: dict[object, set[object]] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, LogRecord):
            raise LogValidationError(
                f"records[{i}] must be LogRecord, got {type(rec).__name__}"
            )
        try:
            validate_record(rec)
        except LogValidationError as e:
            raise LogValidationError(f"records[{i}]: {e}") from None
        contexts.append(rec.context)
        logged_actions.setdefault(rec.context, set()).add(rec.action)

    pi_rows = _validate_target_policy(target_policy, contexts, logged_actions)
    q_rows = _validate_qhat(q_hat, pi_rows) if q_hat is not None else None

    # ---- maturity split (explicit accounting, never fill zero) ----
    mature_idx: list[int] = []
    immature_idx: list[int] = []
    for i, rec in enumerate(records):
        if rec.observed_time <= cutoff_f:
            mature_idx.append(i)
        else:
            immature_idx.append(i)
    report = MaturityReport(
        cutoff=cutoff_f,
        total=len(records),
        mature=len(mature_idx),
        immature=len(immature_idx),
        mature_indices=tuple(mature_idx),
        immature_indices=tuple(immature_idx),
    )

    # ---- structural support refusal (cutoff-independent) ----
    # Empirical support: (x,a) pairs actually present in the log.
    # The contractual refusal triggers exactly on:
    #   pi(a|x) > 0 and pair (x,a) never observed at all.
    # Checked before the empty-maturity guard so the error reflects
    # the policy/log incompatibility itself, not the chosen cutoff.
    observed_pairs: set[tuple[object, object]] = set()
    for rec in records:
        observed_pairs.add((rec.context, rec.action))

    for x, row in pi_rows.items():
        for a, p in row.items():
            if p > 0.0 and (x, a) not in observed_pairs:
                raise ZeroSupportError(
                    f"target policy assigns pi({a!r}|{x!r}) = {p!r} > 0 but "
                    "context/action pair has zero logging support "
                    "(never observed in the log); IPS ratio is undefined"
                )

    if not mature_idx:
        # Fully immature cutoff on a structurally compatible log:
        # nothing may be estimated.  Returning a number would disguise
        # missing feedback as evidence.
        raise NoMatureRecordsError(
            "no mature records at cutoff "
            f"{cutoff_f!r} ({len(immature_idx)}/{len(records)} immature); "
            "delayed unobserved rewards are not treated as zero",
            total=len(records),
            immature=len(immature_idx),
            cutoff=cutoff_f,
        )

    n = len(mature_idx)

    # DR baseline m(x) = sum_a pi(a|x) q_hat(x,a), precomputed once
    # per context rather than re-summed for every mature record.
    m_by_context: dict[object, float] = {}
    if q_rows is not None:
        for x, row in pi_rows.items():
            m_by_context[x] = sum(
                p2 * q_rows[x][a2] for a2, p2 in row.items() if p2 > 0.0
            )

    # Running totals (no per-record weight retention: O(1) extra memory
    # beyond the caller's log and the index accounting).
    sum_ww_u = 0.0
    sum_ww_c = 0.0
    sum_wr_u = 0.0
    sum_w_u = 0.0
    sum_wr_c = 0.0
    sum_w_c = 0.0
    dr_sum_u = 0.0
    dr_sum_c = 0.0
    max_w = 0.0

    for i in mature_idx:
        rec = records[i]
        pi_a = pi_rows[rec.context][rec.action]
        mu = rec.logged_propensity
        w = pi_a / mu
        if not math.isfinite(w):
            raise NumericalOverflowError(
                f"importance weight overflowed for records[{i}] "
                f"(pi={pi_a!r}, mu={mu!r})"
            )
        wc = _clip_weight(w, clip_f)
        if w > max_w:
            max_w = w
        r = rec.reward
        sum_wr_u += w * r
        sum_w_u += w
        sum_ww_u += w * w
        sum_wr_c += wc * r
        sum_w_c += wc
        sum_ww_c += wc * wc
        if q_rows is not None:
            mx = m_by_context[rec.context]
            if pi_a > 0.0:
                # q_hat for positive-probability actions was required
                # and validated; the correction is w*(r - q(x,a)).
                qx = q_rows[rec.context][rec.action]
                correction_u = w * (r - qx)
                correction_c = wc * (r - qx)
            else:
                # pi(a|x) == 0: weight is zero and q_hat for this
                # action is not required; the correction is exactly 0.
                correction_u = 0.0
                correction_c = 0.0
            dr_sum_u += mx + correction_u
            dr_sum_c += mx + correction_c

    for name, val in (
        ("sum(weight*reward) unclipped", sum_wr_u),
        ("sum(weight) unclipped", sum_w_u),
        ("sum(weight^2) unclipped", sum_ww_u),
        ("sum(weight*reward) clipped", sum_wr_c),
        ("sum(weight) clipped", sum_w_c),
        ("sum(weight^2) clipped", sum_ww_c),
        ("DR unclipped", dr_sum_u),
        ("DR clipped", dr_sum_c),
    ):
        if not math.isfinite(val):
            raise NumericalOverflowError(f"{name} overflowed")

    if n == 0:
        # Unreachable: guarded above by NoMatureRecordsError.  Kept as
        # a defensive assertion since all later code divides by n.
        raise AssertionError("internal: mature set empty but no error raised")

    ips_u = sum_wr_u / n
    ips_c = sum_wr_c / n
    snips_u = sum_wr_u / sum_w_u if sum_w_u > 0.0 else None
    snips_c = sum_wr_c / sum_w_c if sum_w_c > 0.0 else None
    dr_u = dr_sum_u / n if q_rows is not None else None
    dr_c = dr_sum_c / n if q_rows is not None else None

    for nm, v in (("IPS unclipped", ips_u), ("IPS clipped", ips_c)):
        if not math.isfinite(v):
            raise NumericalOverflowError(f"{nm} overflowed")
    for nm, v in (("SNIPS unclipped", snips_u), ("SNIPS clipped", snips_c)):
        if v is not None and not math.isfinite(v):
            raise NumericalOverflowError(f"{nm} overflowed")
    for nm, v in (("DR unclipped", dr_u), ("DR clipped", dr_c)):
        if v is not None and not math.isfinite(v):
            raise NumericalOverflowError(f"{nm} overflowed")

    ess_u = _ess_from_totals(sum_w_u, sum_ww_u)
    ess_c = _ess_from_totals(sum_w_c, sum_ww_c) if clip_f is not None else None

    return EstimatorResult(
        maturity=report,
        n_mature=n,
        ips=ips_c if clip_f is not None else ips_u,
        snips=snips_c if clip_f is not None else snips_u,
        dr=dr_c if clip_f is not None else dr_u,
        clip=clip_f,
        ips_unclipped=ips_u,
        snips_unclipped=snips_u,
        dr_unclipped=dr_u,
        ess_unclipped=ess_u,
        ess_clipped=ess_c,
        max_weight=max_w,
    )
