"""多步完整轨迹（最多 5 步）的轨迹级重要性采样与逐决策估计。

口径定义（与 README 一致）
-------------------------
一条**完整轨迹**是 1..``MAX_TRAJECTORY_STEPS``（=5）个 :class:`TrajectoryStep`
的序列，每步含 context、action、已结算 reward 与行为（记录）策略概率
``behavior_prob = mu(a_t | x_t)``。回报为逐步 reward 的**无折扣和**
``G = sum_t r_t``。逐步累乘重要性权重

    w_t   = pi(a_t | x_t) / mu(a_t | x_t)        （单步权重）
    rho_t = prod_{s<=t} w_s                       （到第 t 步的累乘权重）

n 条轨迹上的两个估计量（T_i 为第 i 条轨迹长度）：

    TIS  = (1/n) * sum_i rho_{i,T_i} * G_i                （轨迹重要性采样）
    PDIS = (1/n) * sum_i sum_{t<=T_i} rho_{i,t} * r_{i,t} （逐决策估计）

在绝对连续（pi << mu）下二者都是 V(pi) = E_pi[sum_t r_t] 的无偏估计；
测试用小 MDP 枚举真值核对。某步目标概率为 0 时该轨迹累乘权重归零，其后的
单步权重不再计算（轨迹在目标策略下概率为 0，贡献恒 0，避免 0*inf）。

零行为概率拒绝（严格）：任一步 pi(a_t|x_t) > 0 而 behavior_prob == 0，权重
无定义，整体抛 :class:`ZeroSupportError`（对全部轨迹的全部步校验）。pi == 0
且 mu == 0 时权重取 0，合法。

有效权重诊断（:class:`WeightDiagnostics`）：末步累乘权重的 ESS、最大末步
权重、零权重条数，以及每个决策步 t 的 ESS（在长度 >= t 的轨迹上、对 rho_t
计算），用于暴露随步数累乘的权重退化。

本模块不实现延迟奖励、删失修正与自适应截断（本轮题面明确不要求）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .core import (
    MAX_WEIGHT,
    InvalidLogError,
    WeightOverflowError,
    ZeroSupportError,
    _check_policy_rows,
    _finite_float,
    _finite_prob,
)

__all__ = [
    "TrajectoryStep",
    "Trajectory",
    "WeightDiagnostics",
    "TrajectoryEvalResult",
    "evaluate_trajectories",
    "InvalidTrajectoryError",
    "MAX_TRAJECTORY_STEPS",
    "MAX_TRAJECTORIES",
]

# ---- 接口边界（题面上限；短演示不必穷举） -------------------------------
MAX_TRAJECTORY_STEPS = 5
MAX_TRAJECTORIES = 1_000_000


class InvalidTrajectoryError(InvalidLogError):
    """轨迹结构或取值非法（步数越界、NaN/Infinity、字段类型错等）。

    继承 :class:`InvalidLogError`，调用方按日志非法统一捕获亦可。
    """


@dataclass(frozen=True)
class TrajectoryStep:
    """轨迹中的一步决策记录。

    属性
    ----
    context:
        该步上下文标识（hashable）。
    action:
        该步实际记录的动作标识（hashable）。
    reward:
        该步已结算的有限实数回报。
    behavior_prob:
        行为（记录）策略在该步 (context, action) 上的概率，区间 [0, 1]。
    """

    context: object
    action: object
    reward: float
    behavior_prob: float


@dataclass(frozen=True)
class Trajectory:
    """一条完整轨迹：1..``MAX_TRAJECTORY_STEPS`` 个 :class:`TrajectoryStep`。

    ``steps`` 建议传 tuple；任何 Sequence 均可，本库只读不写。
    """

    steps: tuple


@dataclass(frozen=True)
class WeightDiagnostics:
    """有效权重诊断。

    ``ess`` 为各轨迹末步累乘权重 rho_{T_i} 的有效样本量；``step_ess[t]``
    为第 t+1 个决策步的累乘权重在长度 >= t+1 的轨迹上的 ESS（下标 0 起，
    对应第 1 步）。``max_weight``/``n_zero_weight`` 针对末步累乘权重。
    """

    ess: float
    max_weight: float
    n_zero_weight: int
    step_ess: tuple


@dataclass(frozen=True)
class TrajectoryEvalResult:
    """轨迹评估结果。``tis``/``pdis`` 定义见模块 docstring。"""

    n_trajectories: int
    horizon: int
    tis: float
    pdis: float
    trajectory_weights: tuple
    diagnostics: WeightDiagnostics

    def summary(self) -> str:
        d = self.diagnostics
        step_str = ", ".join(
            f"t{t + 1}:{v:.6g}" for t, v in enumerate(d.step_ess))
        return "\n".join([
            f"n_trajectories={self.n_trajectories}, horizon={self.horizon}",
            f"TIS={self.tis:.12g}, PDIS={self.pdis:.12g}",
            f"ESS(final)={d.ess:.6g}/{self.n_trajectories}, "
            f"max_weight={d.max_weight:.6g}, "
            f"n_zero_weight={d.n_zero_weight}",
            f"step ESS: {step_str}",
        ])


def _validate_trajectories(trajectories) -> None:
    if not isinstance(trajectories, Sequence) or isinstance(
            trajectories, (str, bytes)):
        raise InvalidTrajectoryError(
            "trajectories 必须是 Trajectory 的序列（Sequence）")
    if len(trajectories) == 0:
        raise InvalidTrajectoryError("至少需要一条轨迹，得到空序列")
    if len(trajectories) > MAX_TRAJECTORIES:
        raise InvalidTrajectoryError(
            f"轨迹数 {len(trajectories)} 超过接口上限 {MAX_TRAJECTORIES}")
    for i, tr in enumerate(trajectories):
        where = f"trajectories[{i}]"
        if not isinstance(tr, Trajectory):
            raise InvalidTrajectoryError(
                f"{where} 必须是 Trajectory，得到 {type(tr).__name__}")
        steps = tr.steps
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise InvalidTrajectoryError(
                f"{where}.steps 必须是 TrajectoryStep 的序列")
        if not 1 <= len(steps) <= MAX_TRAJECTORY_STEPS:
            raise InvalidTrajectoryError(
                f"{where}.steps 长度 {len(steps)} 超出 "
                f"[1, {MAX_TRAJECTORY_STEPS}]（完整轨迹最多 "
                f"{MAX_TRAJECTORY_STEPS} 步）")
        for j, st in enumerate(steps):
            sw = f"{where}.steps[{j}]"
            if not isinstance(st, TrajectoryStep):
                raise InvalidTrajectoryError(
                    f"{sw} 必须是 TrajectoryStep，得到 "
                    f"{type(st).__name__}")
            _finite_float(st.reward, f"{sw}.reward", InvalidTrajectoryError)
            _finite_prob(st.behavior_prob, f"{sw}.behavior_prob",
                         InvalidTrajectoryError)
            try:
                hash(st.context)
                hash(st.action)
            except TypeError as exc:
                raise InvalidTrajectoryError(
                    f"{sw}: context/action 必须可哈希") from exc


def _ess(weights) -> float:
    sum_w = math.fsum(weights)
    sum_w2 = math.fsum(w * w for w in weights)
    return (sum_w * sum_w) / sum_w2 if sum_w2 > 0.0 else 0.0


def evaluate_trajectories(
    trajectories: Sequence[Trajectory],
    target_policy: Mapping[object, Mapping[object, float]],
) -> TrajectoryEvalResult:
    """多步完整轨迹的 TIS / PDIS 评估（无副作用；校验失败不返回任何结果）。

    参数
    ----
    trajectories:
        :class:`Trajectory` 序列，非空、长度 <= ``MAX_TRAJECTORIES``；每条
        轨迹 1..``MAX_TRAJECTORY_STEPS`` 步，各轨迹长度可不同。
    target_policy:
        目标策略概率表 ``context -> (action -> prob)``，口径与
        :func:`delayed_ope.evaluate` 相同：日志中出现的每个 context 必须有
        归一概率行，每个出现过的 (context, action) 必须显式列值。

    返回
    ----
    :class:`TrajectoryEvalResult`：``tis``（轨迹重要性采样）、``pdis``
    （逐决策估计）、各轨迹末步累乘权重与 :class:`WeightDiagnostics`。

    错误语义
    --------
    * 轨迹结构/取值非法（空轨迹、超过 5 步、NaN/Infinity、bool 数值、
      概率越界、不可哈希标识） -> :class:`InvalidTrajectoryError`；
    * 策略表非法 -> :class:`InvalidPolicyError`；
    * 任一步 pi>0 而 behavior_prob==0 -> :class:`ZeroSupportError`
      （对全部轨迹的全部步校验）；
    * 单步或累乘权重超过 ``MAX_WEIGHT`` -> :class:`WeightOverflowError`。
    """
    _validate_trajectories(trajectories)

    contexts = set()
    pairs = set()
    for tr in trajectories:
        for st in tr.steps:
            contexts.add(st.context)
            pairs.add((st.context, st.action))
    _check_policy_rows(target_policy, contexts, pairs)

    # 零支持是结构性不相容，对全部轨迹的全部步校验。
    for i, tr in enumerate(trajectories):
        for j, st in enumerate(tr.steps):
            p_tgt = float(target_policy[st.context][st.action])
            p_beh = float(st.behavior_prob)
            if p_tgt > 0.0 and p_beh == 0.0:
                raise ZeroSupportError(
                    f"零支持拒绝（trajectories[{i}].steps[{j}]）：context="
                    f"{st.context!r}, action={st.action!r}，目标概率="
                    f"{p_tgt:g} 而行为概率={p_beh:g}；重要性权重无定义")

    n = len(trajectories)
    horizon = max(len(tr.steps) for tr in trajectories)

    final_weights: list[float] = []
    tis_terms: list[float] = []
    pdis_terms: list[float] = []
    step_weights: list[list[float]] = [[] for _ in range(horizon)]

    for i, tr in enumerate(trajectories):
        rewards = [float(st.reward) for st in tr.steps]
        rhos: list[float] = []
        rho = 1.0
        for j, st in enumerate(tr.steps):
            p_tgt = float(target_policy[st.context][st.action])
            if rho == 0.0 or p_tgt == 0.0:
                # 轨迹在目标策略下概率已为 0：贡献恒 0，不再计算后续
                # 单步权重（避免 0 * 溢出权重产生 NaN）。
                rho = 0.0
            else:
                w = p_tgt / float(st.behavior_prob)
                if not math.isfinite(w) or w > MAX_WEIGHT:
                    raise WeightOverflowError(
                        f"权重溢出（trajectories[{i}].steps[{j}]）："
                        f"context={st.context!r}, action={st.action!r}，"
                        f"w={p_tgt:g}/{float(st.behavior_prob):g} 超过上限 "
                        f"{MAX_WEIGHT:g}")
                rho *= w
                if not math.isfinite(rho) or rho > MAX_WEIGHT:
                    raise WeightOverflowError(
                        f"累乘权重溢出（trajectories[{i}] 第 {j + 1} 步）："
                        f"rho={rho!r} 超过上限 {MAX_WEIGHT:g}")
            rhos.append(rho)
            step_weights[j].append(rho)
        final_weights.append(rhos[-1])
        tis_terms.append(rhos[-1] * math.fsum(rewards))
        pdis_terms.append(math.fsum(r * rv for r, rv in zip(rhos, rewards)))

    tis = math.fsum(tis_terms) / n
    pdis = math.fsum(pdis_terms) / n

    diagnostics = WeightDiagnostics(
        ess=_ess(final_weights),
        max_weight=max(final_weights),
        n_zero_weight=sum(1 for w in final_weights if w == 0.0),
        step_ess=tuple(_ess(ws) for ws in step_weights),
    )
    return TrajectoryEvalResult(
        n_trajectories=n,
        horizon=horizon,
        tis=tis,
        pdis=pdis,
        trajectory_weights=tuple(final_weights),
        diagnostics=diagnostics,
    )
