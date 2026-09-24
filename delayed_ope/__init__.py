"""delayed_ope: 含延迟结算反馈的离线策略评估（纯标准库）。

公开接口：
* 单决策延迟结算口径：:func:`evaluate` 与 :class:`LogEvent`、
  :class:`EvaluationResult`；
* 多步完整轨迹（最多 5 步）：:func:`evaluate_trajectories` 与
  :class:`TrajectoryStep`、:class:`Trajectory`、:class:`TrajectoryEvalResult`。

误差口径、输入上限与设计取舍见仓库 README。
"""

from .core import (
    Estimate,
    EvaluationResult,
    InvalidLogError,
    InvalidPolicyError,
    InvalidQHatError,
    LogEvent,
    NoMatureSamplesError,
    OPEError,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
)
from .trajectory import (
    MAX_TRAJECTORIES,
    MAX_TRAJECTORY_STEPS,
    InvalidTrajectoryError,
    Trajectory,
    TrajectoryEvalResult,
    TrajectoryStep,
    WeightDiagnostics,
    evaluate_trajectories,
)

__all__ = [
    "LogEvent",
    "Estimate",
    "EvaluationResult",
    "OPEError",
    "InvalidLogError",
    "InvalidPolicyError",
    "InvalidQHatError",
    "ZeroSupportError",
    "WeightOverflowError",
    "NoMatureSamplesError",
    "evaluate",
    "TrajectoryStep",
    "Trajectory",
    "WeightDiagnostics",
    "TrajectoryEvalResult",
    "InvalidTrajectoryError",
    "MAX_TRAJECTORY_STEPS",
    "MAX_TRAJECTORIES",
    "evaluate_trajectories",
]
