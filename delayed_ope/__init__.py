"""delayed_ope: 含延迟结算反馈的离线策略评估（纯标准库）。

公开接口见 :func:`evaluate` / :func:`evaluate_trajectories` 与
:class:`LogEvent`、:class:`EvaluationResult`、:class:`Trajectory`、
:class:`TrajectoryEvaluationResult`。误差口径、输入上限与设计取舍见仓库
README。
"""

from .core import (
    Estimate,
    EvaluationResult,
    InvalidLogError,
    InvalidPolicyError,
    InvalidQHatError,
    InvalidTrajectoryError,
    LogEvent,
    NoMatureSamplesError,
    OPEError,
    Step,
    Trajectory,
    TrajectoryEstimate,
    TrajectoryEvaluationResult,
    WeightDiagnostics,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
    evaluate_trajectories,
)

__all__ = [
    "LogEvent",
    "Estimate",
    "EvaluationResult",
    "Step",
    "Trajectory",
    "WeightDiagnostics",
    "TrajectoryEstimate",
    "TrajectoryEvaluationResult",
    "OPEError",
    "InvalidLogError",
    "InvalidTrajectoryError",
    "InvalidPolicyError",
    "InvalidQHatError",
    "ZeroSupportError",
    "WeightOverflowError",
    "NoMatureSamplesError",
    "evaluate",
    "evaluate_trajectories",
]
