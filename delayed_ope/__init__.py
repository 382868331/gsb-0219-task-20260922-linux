"""delayed_ope: 含延迟结算反馈的离线策略评估（纯标准库）。

公开接口：evaluate（批量入口）与 TimelineEvaluator（按观察时间推进的增量
入口），以及 LogEvent、EvaluationResult。误差口径、输入上限与设计取舍见
仓库 README。
"""

from .core import (
    CutoffRegressionError,
    DuplicateEventIdError,
    Estimate,
    EvaluationResult,
    InvalidLogError,
    InvalidPolicyError,
    InvalidQHatError,
    LogEvent,
    NoMatureSamplesError,
    OPEError,
    TimelineEvaluator,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
)

__all__ = [
    "LogEvent",
    "Estimate",
    "EvaluationResult",
    "TimelineEvaluator",
    "OPEError",
    "InvalidLogError",
    "InvalidPolicyError",
    "InvalidQHatError",
    "ZeroSupportError",
    "WeightOverflowError",
    "NoMatureSamplesError",
    "DuplicateEventIdError",
    "CutoffRegressionError",
    "evaluate",
]
