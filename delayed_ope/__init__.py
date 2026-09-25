"""delayed_ope: 含延迟结算反馈的离线策略评估（纯标准库）。

公开接口见 :func:`evaluate`、:class:`IncrementalEvaluation` 与
:class:`LogEvent`、:class:`EvaluationResult`。
误差口径、输入上限与设计取舍见仓库 README。
"""

from .core import (
    Estimate,
    EvaluationResult,
    IncrementalEvaluation,
    InvalidLogError,
    InvalidPolicyError,
    InvalidQHatError,
    LogEvent,
    NoMatureSamplesError,
    CutoffOutOfOrderError,
    OPEError,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
)

__all__ = [
    "LogEvent",
    "Estimate",
    "EvaluationResult",
    "IncrementalEvaluation",
    "CutoffOutOfOrderError",
    "OPEError",
    "InvalidLogError",
    "InvalidPolicyError",
    "InvalidQHatError",
    "ZeroSupportError",
    "WeightOverflowError",
    "NoMatureSamplesError",
    "evaluate",
]
