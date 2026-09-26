"""delayed_ope: 含延迟结算反馈的离线策略评估（纯标准库）。

公开接口见 :func:`evaluate` 与 :class:`LogEvent`、:class:`EvaluationResult`。
误差口径、输入上限与设计取舍见仓库 README。
"""

from .core import (
    Estimate,
    EvaluationResult,
    InvalidLogError,
    InvalidPolicyError,
    InvalidQHatError,
    InfeasibleProbabilityIntervalsError,
    LogEvent,
    NoMatureSamplesError,
    OPEError,
    RobustConstraintCheck,
    RobustIPSBound,
    RobustIPSResult,
    RobustLogEvent,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
    robust_ips,
)

__all__ = [
    "LogEvent",
    "Estimate",
    "EvaluationResult",
    "OPEError",
    "InvalidLogError",
    "InvalidPolicyError",
    "InvalidQHatError",
    "InfeasibleProbabilityIntervalsError",
    "ZeroSupportError",
    "WeightOverflowError",
    "NoMatureSamplesError",
    "RobustConstraintCheck",
    "RobustIPSBound",
    "RobustIPSResult",
    "RobustLogEvent",
    "evaluate",
    "robust_ips",
]
