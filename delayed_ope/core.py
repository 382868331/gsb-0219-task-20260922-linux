"""核心算法：延迟结算反馈口径下的 IPS / SNIPS / DR 离线策略评估。

口径定义（与 README 一致）
-------------------------
每条日志事件 e 含 decision_time t_d 与 observed_time t_o（t_o >= t_d）。
固定评估截止时刻 T，事件为**成熟样本**当且仅当 ``t_o <= T``。
未成熟事件（t_o > T，即回报在截止时刻尚未观察到）只被排除在求和之外，绝不
按 reward = 0 计入。成熟样本上定义重要性权重

    w_i = pi_target(a_i | x_i) / pi_logged(a_i | x_i)

若存在事件使目标概率为正而记录策略支持为零（logged_propensity = 0），权重
无定义，评估整体拒绝（:class:`ZeroSupportError`）。该校验与截止时刻无关：
零支持是目标策略与日志的不相容，对所有事件执行。

估计量（n_m 为成熟样本数，r_i 为该事件已结算的真实回报）：

    IPS   = (1/n_m) * sum_i w_i r_i
    SNIPS = sum_i w_i r_i / sum_i w_i
    DR    = (1/n_m) * sum_i [ m_pi(x_i) + w_i (r_i - q_hat(x_i, a_i)) ]

其中 ``m_pi(x) = sum_a pi_target(a|x) * q_hat(x, a)`` 为 q_hat 在目标策略下
的边际价值。DR 的恒等式（对任意 q_hat 成立）：

    E_{x, a~mu}[ m_pi(x) + (pi/mu)(r - q_hat(x,a)) ]
      = sum_a pi(a|x) E[r|x,a] = V(pi)

q_hat 完美时校正项恒为 0，DR 即退化为确定性的 m_pi 平均。SNIPS 在
``sum_i w_i = 0``（目标策略对所有被记录动作赋零概率）时无定义，对应字段为
``None``；这不是异常，ESS 同时为 0，调用方可显式识别退化情形。

截断为可选的显式后处理：``w_clipped = min(w, M)``，对 IPS/SNIPS/DR 中的权重
统一生效。未截断组始终报告，截断组另报。有效样本量

    ESS = (sum_i w_i)^2 / sum_i w_i^2

对所用权重集合分别计算；正权重时 1 <= ESS <= n_m，全零权重时 ESS = 0。

稳健 IPS（:func:`robust_ips`）将每个 context 的行为概率视为一组共享变量：
``lower_a <= p_a <= upper_a`` 且 ``Σ p_a = 1``。同一 context 的多条日志不分别
选择倾向概率；在每 context 至多 3 个动作的闭区间单纯形上枚举顶点、边/面驻点
和内点驻点，求未截断 IPS 的最小值/最大值。区间不可满足时先拒绝，不做投影。

本模块只实现本题契约的有限数学子集：实数标量、有限时间、离散上下文/动作。
"""

from __future__ import annotations

import math
from itertools import combinations, product
from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "LogEvent",
    "RobustLogEvent",
    "Estimate",
    "EvaluationResult",
    "RobustIPSBound",
    "RobustConstraintCheck",
    "RobustIPSResult",
    "evaluate",
    "robust_ips",
    "OPEError",
    "InvalidLogError",
    "InvalidPolicyError",
    "InvalidQHatError",
    "ZeroSupportError",
    "WeightOverflowError",
    "NoMatureSamplesError",
]

# ---- 接口边界（题面上限；短演示不必穷举） -------------------------------
MAX_EVENTS = 1_000_000
MAX_ACTIONS_PER_CONTEXT = 100_000
MAX_ROBUST_ACTIONS_PER_CONTEXT = 3
# 允许的最大权重；比值超过此值视为数值溢出而非合法极大权重。
MAX_WEIGHT = 1e300
# 概率归一化容差：每个 context 的动作概率之和必须与 1.0 的偏差在此范围内。
PROB_SUM_TOL = 1e-9


class OPEError(Exception):
    """本库所有错误的基类。"""


class InvalidLogError(OPEError):
    """日志事件结构或取值非法（含 NaN/Infinity、时序矛盾等）。"""


class InvalidPolicyError(OPEError):
    """目标策略概率表非法（缺动作、概率越界、不归一、NaN/Infinity）。"""


class InvalidQHatError(OPEError):
    """q_hat 取值非法（NaN/Infinity），或对所需 (context, action) 缺值。"""


class ZeroSupportError(OPEError):
    """存在事件：目标策略概率为正而记录策略支持为零（logged_propensity=0）。"""


class WeightOverflowError(OPEError):
    """重要性权重超过 ``MAX_WEIGHT``，拒绝以避免静默产生 inf/NaN。"""


class NoMatureSamplesError(OPEError):
    """截止时刻前没有任何成熟样本（全部延迟未观测）。"""


class InfeasibleProbabilityIntervalsError(InvalidPolicyError):
    """行为概率正区间与“同一上下文概率和为 1”不相容。"""


@dataclass(frozen=True)
class LogEvent:
    """一条带延迟结算的决策日志。

    属性
    ----
    context:
        上下文标识（hashable，通常为 int/str/tuple）。
    action:
        实际记录的动作标识（hashable）。
    reward:
        已结算的有限实数回报。未成熟事件的 reward 不参与任何求和，从不改记 0。
    logged_propensity:
        记录策略在 (context, action) 上的概率，区间 [0, 1]。
    decision_time:
        决策时刻（可比较的有限实数）。
    observed_time:
        回报可观察时刻（有限实数，且 ``observed_time >= decision_time``）。
    """

    context: object
    action: object
    reward: float
    logged_propensity: float
    decision_time: float
    observed_time: float


@dataclass(frozen=True)
class RobustLogEvent:
    """一条行为概率以上下文级区间表示的决策日志。

    与 :class:`LogEvent` 不同，本事件不保存单条记录自己的倾向得分；
    行为概率由 ``robust_ips`` 的 ``behavior_intervals`` 按上下文统一给出，
    因而同一上下文的所有动作共享一组求和为 1 的变量。
    """

    context: object
    action: object
    reward: float
    decision_time: float = 0.0
    observed_time: float = 1.0


@dataclass(frozen=True)
class Estimate:
    """一组（截断或未截断）权重下的三个估计量与有效样本量。

    ``snips`` 在权重和为 0 时为 ``None``（0/0 无定义）；``dr`` 在未提供
    q_hat 时为 ``None``。
    """

    ips: float
    snips: float | None
    dr: float | None
    ess: float
    n: int


@dataclass(frozen=True)
class EvaluationResult:
    """评估结果。成熟样本口径在字段中显式给出。"""

    cutoff: float
    n_total: int
    n_mature: int
    n_immature: int
    mature_indices: tuple[int, ...]
    unclipped: Estimate
    clipped: Estimate | None = None
    clip_value: float | None = None
    q_hat_used: bool = False

    def summary(self) -> str:
        head = list(self.mature_indices[:8])
        tail_note = ""
        if len(self.mature_indices) > 8:
            tail_note = f", ...（共 {len(self.mature_indices)} 个，见字段）"
        lines = [
            f"cutoff={self.cutoff:g}",
            f"n_total={self.n_total}, n_mature={self.n_mature}, "
            f"n_immature={self.n_immature}",
            f"mature_indices(head)={head}{tail_note}",
            _fmt_est("unclipped", self.unclipped),
        ]
        if self.clipped is not None:
            lines.append(
                _fmt_est(f"clipped@M={self.clip_value:g}", self.clipped))
        return "\n".join(lines)


def _fmt_est(tag: str, est: Estimate) -> str:
    sn = "undefined" if est.snips is None else f"{est.snips:.12g}"
    dr = "n/a" if est.dr is None else f"{est.dr:.12g}"
    return (
        f"[{tag}] IPS={est.ips:.12g}, SNIPS={sn}, DR={dr}, "
        f"ESS={est.ess:.6g}/{est.n}"
    )


@dataclass(frozen=True)
class RobustConstraintCheck:
    """一个边界概率表在单个上下文上的约束复核结果。"""

    context: object
    feasible_intervals: bool
    lower_sum: float
    upper_sum: float
    probability_sum: float
    max_interval_violation: float


@dataclass(frozen=True)
class RobustIPSBound:
    """稳健 IPS 的一个边界及达到该边界的上下文概率表。"""

    ips: float
    behavior_policy: dict[object, dict[object, float]]
    context_contributions: dict[object, float]
    constraint_checks: tuple[RobustConstraintCheck, ...]


@dataclass(frozen=True)
class RobustIPSResult:
    """稳健 IPS 的最小值/最大值与成熟样本口径。"""

    cutoff: float
    n_total: int
    n_mature: int
    n_immature: int
    mature_indices: tuple[int, ...]
    minimum: RobustIPSBound
    maximum: RobustIPSBound

    def summary(self) -> str:
        return "\n".join((
            f"cutoff={self.cutoff:g}",
            f"n_total={self.n_total}, n_mature={self.n_mature}, "
            f"n_immature={self.n_immature}",
            f"robust IPS range=[{self.minimum.ips:.12g}, "
            f"{self.maximum.ips:.12g}]",
        ))


def _finite_float(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise InvalidLogError(
            f"{what} 必须是有限实数，得到 {type(x).__name__}")
    fx = float(x)
    if not math.isfinite(fx):
        raise InvalidLogError(f"{what} 拒绝 NaN/Infinity，得到 {x!r}")
    return fx


def _finite_prob(x, what: str, exc: type = InvalidLogError) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise exc(f"{what} 概率必须是实数，得到 {type(x).__name__}")
    fx = float(x)
    if not math.isfinite(fx):
        raise exc(f"{what} 概率拒绝 NaN/Infinity，得到 {x!r}")
    if not (0.0 <= fx <= 1.0):
        raise exc(f"{what} 概率超出 [0,1]，得到 {fx!r}")
    return fx


def _validate_events(events: Sequence[LogEvent]) -> None:
    if not isinstance(events, Sequence):
        raise InvalidLogError("events 必须是 LogEvent 的序列（Sequence）")
    if len(events) > MAX_EVENTS:
        raise InvalidLogError(
            f"事件数 {len(events)} 超过接口上限 {MAX_EVENTS}")
    for i, e in enumerate(events):
        if not isinstance(e, LogEvent):
            raise InvalidLogError(
                f"events[{i}] 必须是 LogEvent，得到 {type(e).__name__}")
        where = f"events[{i}]"
        _finite_float(e.reward, f"{where}.reward")
        _finite_prob(e.logged_propensity, f"{where}.logged_propensity")
        td = _finite_float(e.decision_time, f"{where}.decision_time")
        to = _finite_float(e.observed_time, f"{where}.observed_time")
        if to < td:
            raise InvalidLogError(
                f"{where}: observed_time({to:g}) < decision_time({td:g})")
        try:
            hash(e.context)
            hash(e.action)
        except TypeError as exc:
            raise InvalidLogError(
                f"{where}: context/action 必须可哈希") from exc


def _validate_policy(
    policy: Mapping[object, Mapping[object, float]],
    events: Sequence[LogEvent],
) -> None:
    if not isinstance(policy, Mapping):
        raise InvalidPolicyError(
            "target_policy 必须是 Mapping[context, Mapping[action, prob]]")
    contexts_in_log = {e.context for e in events}
    logged_pairs = {(e.context, e.action) for e in events}
    for ctx in contexts_in_log:
        if ctx not in policy:
            raise InvalidPolicyError(
                f"target_policy 缺少 context {ctx!r} 的动作概率行")
        row = policy[ctx]
        if not isinstance(row, Mapping):
            raise InvalidPolicyError(
                f"target_policy[{ctx!r}] 必须是 action->prob 的 Mapping")
        if len(row) > MAX_ACTIONS_PER_CONTEXT:
            raise InvalidPolicyError(
                f"context {ctx!r} 动作数 {len(row)} 超过上限 "
                f"{MAX_ACTIONS_PER_CONTEXT}")
        if len(row) == 0:
            raise InvalidPolicyError(f"context {ctx!r} 的概率行为空")
        total = 0.0
        for a, pr in row.items():
            fpr = _finite_prob(
                pr, f"target_policy[{ctx!r}][{a!r}]", InvalidPolicyError)
            total += fpr
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise InvalidPolicyError(
                f"context {ctx!r} 的概率之和={total!r}，与 1.0 的偏差超过 "
                f"容差 {PROB_SUM_TOL:g}")
    for ctx, act in logged_pairs:
        if act not in policy[ctx]:
            # 不把“漏行动作”静默当作概率 0。
            raise InvalidPolicyError(
                f"target_policy[{ctx!r}] 缺少日志动作 {act!r} 的概率"
                "（不支持将漏写动作隐式视为 0）")


def _make_q_lookup(q_hat: Mapping):
    """返回 q(ctx, action) 查表函数；扁平键优先于嵌套映射。"""
    def lookup(ctx, act):
        if (ctx, act) in q_hat:
            v = q_hat[(ctx, act)]
        elif ctx in q_hat and isinstance(q_hat[ctx], Mapping):
            inner = q_hat[ctx]
            if act not in inner:
                raise InvalidQHatError(
                    f"q_hat[{ctx!r}] 缺少动作 {act!r} 的取值")
            v = inner[act]
        else:
            raise InvalidQHatError(
                f"q_hat 缺少 ({ctx!r}, {act!r}) 的取值")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InvalidQHatError(
                f"q_hat[{ctx!r}][{act!r}] 必须是实数，"
                f"得到 {type(v).__name__}")
        v = float(v)
        if not math.isfinite(v):
            raise InvalidQHatError(
                f"q_hat[{ctx!r}][{act!r}] 拒绝 NaN/Infinity，得到 {v!r}")
        return v

    return lookup


def _validate_cutoff(cutoff: float) -> float:
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise InvalidLogError(
            f"cutoff 必须是有限实数，得到 {type(cutoff).__name__}")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff):
        raise InvalidLogError("cutoff 拒绝 NaN/Infinity")
    return cutoff


def _validate_robust_events(events: Sequence[RobustLogEvent]) -> None:
    if not isinstance(events, Sequence):
        raise InvalidLogError(
            "events 必须是 RobustLogEvent 的序列（Sequence）")
    if len(events) > MAX_EVENTS:
        raise InvalidLogError(
            f"事件数 {len(events)} 超过接口上限 {MAX_EVENTS}")
    for i, event in enumerate(events):
        if not isinstance(event, RobustLogEvent):
            raise InvalidLogError(
                f"events[{i}] 必须是 RobustLogEvent，得到 "
                f"{type(event).__name__}")
        where = f"events[{i}]"
        _finite_float(event.reward, f"{where}.reward")
        td = _finite_float(event.decision_time, f"{where}.decision_time")
        to = _finite_float(event.observed_time, f"{where}.observed_time")
        if to < td:
            raise InvalidLogError(
                f"{where}: observed_time({to:g}) < decision_time({td:g})")
        try:
            hash(event.context)
            hash(event.action)
        except TypeError as exc:
            raise InvalidLogError(
                f"{where}: context/action 必须可哈希") from exc


def _validate_robust_target_policy(
    target_policy: Mapping[object, Mapping[object, float]],
    events: Sequence[RobustLogEvent],
) -> None:
    if not isinstance(target_policy, Mapping):
        raise InvalidPolicyError(
            "target_policy 必须是 Mapping[context, Mapping[action, prob]]")
    contexts = {event.context for event in events}
    for ctx in contexts:
        if ctx not in target_policy:
            raise InvalidPolicyError(
                f"target_policy 缺少 context {ctx!r} 的动作概率行")
        row = target_policy[ctx]
        if not isinstance(row, Mapping) or len(row) == 0:
            raise InvalidPolicyError(
                f"target_policy[{ctx!r}] 必须是非空 action->prob Mapping")
        if len(row) > MAX_ACTIONS_PER_CONTEXT:
            raise InvalidPolicyError(
                f"context {ctx!r} 目标动作数 {len(row)} 超过上限 "
                f"{MAX_ACTIONS_PER_CONTEXT}")
        total = 0.0
        for action, probability in row.items():
            total += _finite_prob(
                probability,
                f"target_policy[{ctx!r}][{action!r}]",
                InvalidPolicyError,
            )
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise InvalidPolicyError(
                f"context {ctx!r} 的目标概率之和={total!r}，与 1.0 的偏差"
                f"超过容差 {PROB_SUM_TOL:g}")
    for i, event in enumerate(events):
        if event.action not in target_policy[event.context]:
            raise InvalidPolicyError(
                f"target_policy[{event.context!r}] 缺少 events[{i}] 的日志"
                f"动作 {event.action!r}（不支持漏写隐式视为 0）")


def _validate_behavior_intervals(
    behavior_intervals: Mapping[object, Mapping[object, tuple[float, float]]],
    events: Sequence[RobustLogEvent],
) -> dict[object, tuple[tuple[object, ...], list[float], list[float]]]:
    if not isinstance(behavior_intervals, Mapping):
        raise InvalidPolicyError(
            "behavior_intervals 必须是 "
            "Mapping[context, Mapping[action, (lower, upper)]]")
    parsed: dict[object, tuple[tuple[object, ...], list[float], list[float]]] = {}
    contexts = {event.context for event in events}
    for ctx in contexts:
        if ctx not in behavior_intervals:
            raise InvalidPolicyError(
                f"behavior_intervals 缺少 context {ctx!r} 的概率区间行")
        row = behavior_intervals[ctx]
        if not isinstance(row, Mapping):
            raise InvalidPolicyError(
                f"behavior_intervals[{ctx!r}] 必须是 action->区间 的 Mapping")
        if len(row) == 0:
            raise InvalidPolicyError(
                f"context {ctx!r} 的行为概率区间为空")
        if len(row) > MAX_ROBUST_ACTIONS_PER_CONTEXT:
            raise InvalidPolicyError(
                f"context {ctx!r} 的动作数 {len(row)} 超过稳健 IPS 上限 "
                f"{MAX_ROBUST_ACTIONS_PER_CONTEXT}")
        actions = tuple(row.keys())
        lows: list[float] = []
        highs: list[float] = []
        for action in actions:
            interval = row[action]
            if not isinstance(interval, tuple | list) or len(interval) != 2:
                raise InvalidPolicyError(
                    f"behavior_intervals[{ctx!r}][{action!r}] 必须是 "
                    "(lower, upper) 二元组")
            lower = _finite_prob(
                interval[0],
                f"behavior_intervals[{ctx!r}][{action!r}].lower",
                InvalidPolicyError,
            )
            upper = _finite_prob(
                interval[1],
                f"behavior_intervals[{ctx!r}][{action!r}].upper",
                InvalidPolicyError,
            )
            if lower <= 0.0:
                raise InvalidPolicyError(
                    f"behavior_intervals[{ctx!r}][{action!r}].lower 必须是"
                    f"正数，得到 {lower:g}")
            if lower > upper:
                raise InvalidPolicyError(
                    f"behavior_intervals[{ctx!r}][{action!r}] 下界 "
                    f"{lower:g} 大于上界 {upper:g}")
            lows.append(lower)
            highs.append(upper)
        lower_sum = math.fsum(lows)
        upper_sum = math.fsum(highs)
        if lower_sum > 1.0 + PROB_SUM_TOL or upper_sum < 1.0 - PROB_SUM_TOL:
            raise InfeasibleProbabilityIntervalsError(
                f"context {ctx!r} 的行为概率区间不可满足：下界和="
                f"{lower_sum:g}，上界和={upper_sum:g}；闭区间单纯形要求 "
                "Σlower <= 1 <= Σupper")
        parsed[ctx] = (actions, lows, highs)
    for i, event in enumerate(events):
        actions = parsed[event.context][0]
        if event.action not in actions:
            raise InvalidPolicyError(
                f"behavior_intervals[{event.context!r}] 缺少 events[{i}] "
                f"动作 {event.action!r} 的正概率区间")
    return parsed


def _candidate_inside(point, lows, highs, tolerance=1e-10):
    if abs(math.fsum(point) - 1.0) > PROB_SUM_TOL:
        return False
    return all(
        value > 0.0
        and value >= low - tolerance
        and value <= high + tolerance
        for value, low, high in zip(point, lows, highs))


def _stationary_point(indices, total, coefficients):
    roots = []
    signs = set()
    for idx in indices:
        coefficient = coefficients[idx]
        if coefficient == 0.0:
            return None
        signs.add(1 if coefficient > 0.0 else -1)
        roots.append(math.sqrt(abs(coefficient)))
    if len(signs) != 1:
        return None
    scale = total / math.fsum(roots)
    return tuple(scale * root for root in roots)


def _optimize_context(actions, lows, highs, coefficients):
    n = len(actions)
    if n == 1:
        if not (lows[0] <= 1.0 <= highs[0]):
            raise InfeasibleProbabilityIntervalsError(
                "单动作概率必须等于 1，但区间不包含 1")
        point = (1.0,)
        value = math.fsum(coefficients) / 1.0
        return value, dict(zip(actions, point))

    candidates = []

    # 顶点：至多一个变量不在自己的端点上。
    for free in range(n):
        for endpoint_bits in product((0, 1), repeat=n - 1):
            point = [0.0] * n
            used = 0.0
            cursor = 0
            for idx in range(n):
                if idx == free:
                    continue
                point[idx] = lows[idx] if endpoint_bits[cursor] == 0 else highs[idx]
                used += point[idx]
                cursor += 1
            point[free] = 1.0 - used
            candidates.append(tuple(point))

    # KKT 面：一组变量位于区间内部，其余变量固定在端点；在面上求驻点。
    for interior_size in range(2, n + 1):
        for interior in combinations(range(n), interior_size):
            fixed = tuple(idx for idx in range(n) if idx not in interior)
            for endpoint_bits in product((0, 1), repeat=len(fixed)):
                point = [0.0] * n
                fixed_sum = 0.0
                for idx, bit in zip(fixed, endpoint_bits):
                    point[idx] = lows[idx] if bit == 0 else highs[idx]
                    fixed_sum += point[idx]
                remaining = 1.0 - fixed_sum
                values = _stationary_point(interior, remaining, coefficients)
                if values is None:
                    continue
                for idx, value in zip(interior, values):
                    point[idx] = value
                candidates.append(tuple(point))

    feasible = [point for point in candidates
                if _candidate_inside(point, lows, highs)]
    if not feasible:
        raise InfeasibleProbabilityIntervalsError(
            "概率区间与归一化约束没有找到可行解")

    def objective(point):
        terms = [coefficient / probability
                 for coefficient, probability in zip(coefficients, point)]
        value = math.fsum(terms)
        if not math.isfinite(value):
            raise WeightOverflowError(
                f"稳健 IPS 目标值非有限：系数={coefficients!r}，概率点={point!r}")
        return value

    best_min = min(feasible, key=objective)
    best_max = max(feasible, key=objective)
    return (
        objective(best_min),
        dict(zip(actions, best_min)),
        objective(best_max),
        dict(zip(actions, best_max)),
    )


def _constraint_check(context, actions, lows, highs, probabilities):
    sum_probability = math.fsum(probabilities[a] for a in actions)
    violation = 0.0
    for action, low, high in zip(actions, lows, highs):
        value = probabilities[action]
        violation = max(violation, low - value, value - high, 0.0)
    lower_sum = math.fsum(lows)
    upper_sum = math.fsum(highs)
    feasible = (
        lower_sum <= 1.0 + PROB_SUM_TOL
        and upper_sum >= 1.0 - PROB_SUM_TOL
        and abs(sum_probability - 1.0) <= PROB_SUM_TOL
        and violation <= PROB_SUM_TOL
    )
    return RobustConstraintCheck(
        context=context,
        feasible_intervals=feasible,
        lower_sum=lower_sum,
        upper_sum=upper_sum,
        probability_sum=sum_probability,
        max_interval_violation=max(violation, 0.0),
    )


def robust_ips(
    events: Sequence[RobustLogEvent],
    target_policy: Mapping[object, Mapping[object, float]],
    behavior_intervals: Mapping[object, Mapping[object, tuple[float, float]]],
    cutoff: float,
) -> RobustIPSResult:
    """计算行为概率为正闭区间时未截断 IPS 的最小值和最大值。

    每个 context 的行为动作共享一个概率向量 ``p_a``，满足
    ``lower_a <= p_a <= upper_a`` 且 ``Σ_a p_a = 1``。同一 context 的多条记录
    不会分别挑选倾向得分；一个边界使用同一张概率表。目标策略的成熟样本
    IPS 为 ``(1/n_m) Σ_i π(a_i|x_i) r_i / p_{a_i}``。

    稳健入口仅支持每个 context 至多 3 个行为动作，且不做权重截断。范围只
    反映传入的倾向区间；不包含奖励、目标策略、未观测过程或其他模型误差。
    """
    cutoff = _validate_cutoff(cutoff)
    _validate_robust_events(events)
    _validate_robust_target_policy(target_policy, events)
    parsed = _validate_behavior_intervals(behavior_intervals, events)

    for ctx in {event.context for event in events}:
        target_actions = set(target_policy[ctx].keys())
        interval_actions = set(parsed[ctx][0])
        if target_actions != interval_actions:
            raise InvalidPolicyError(
                f"context {ctx!r} 的目标动作集合与行为区间动作集合不一致："
                f"target={tuple(target_policy[ctx].keys())!r}, "
                f"intervals={parsed[ctx][0]!r}")

    mature_indices = tuple(
        index for index, event in enumerate(events)
        if event.observed_time <= cutoff)
    if not mature_indices:
        raise NoMatureSamplesError(
            f"cutoff={cutoff:g} 前无成熟样本：{len(events)} 条事件的回报"
            "均尚未观察到；未成熟事件不按 reward=0 计入")

    contexts = tuple(parsed.keys())
    coefficient_terms = {
        ctx: [[] for _ in parsed[ctx][0]] for ctx in contexts
    }
    for index in mature_indices:
        event = events[index]
        actions, _, _ = parsed[event.context]
        action_index = actions.index(event.action)
        target_probability = float(
            target_policy[event.context][event.action])
        if target_probability > 0.0:
            lower = parsed[event.context][1][action_index]
            max_weight = target_probability / lower
            if not math.isfinite(max_weight) or max_weight > MAX_WEIGHT:
                raise WeightOverflowError(
                    f"权重溢出（events[{index}]，成熟样本）：目标概率="
                    f"{target_probability:g}，行为概率下界={lower:g}，"
                    f"最大可能权重超过上限 {MAX_WEIGHT:g}；稳健入口不截断权重")
        coefficient_terms[event.context][action_index].append(
            target_probability * float(event.reward))

    min_policy = {}
    max_policy = {}
    min_contributions = {}
    max_contributions = {}
    min_checks = []
    max_checks = []
    min_total = 0.0
    max_total = 0.0
    n_mature = len(mature_indices)

    for ctx in contexts:
        actions, lows, highs = parsed[ctx]
        coefficients = [math.fsum(terms) for terms in coefficient_terms[ctx]]
        min_value, min_probabilities, max_value, max_probabilities = (
            _optimize_context(actions, lows, highs, coefficients))
        min_policy[ctx] = min_probabilities
        max_policy[ctx] = max_probabilities
        min_contributions[ctx] = min_value / n_mature
        max_contributions[ctx] = max_value / n_mature
        min_total += min_value / n_mature
        max_total += max_value / n_mature
        min_checks.append(
            _constraint_check(ctx, actions, lows, highs, min_probabilities))
        max_checks.append(
            _constraint_check(ctx, actions, lows, highs, max_probabilities))

    minimum = RobustIPSBound(
        ips=min_total,
        behavior_policy=min_policy,
        context_contributions=min_contributions,
        constraint_checks=tuple(min_checks),
    )
    maximum = RobustIPSBound(
        ips=max_total,
        behavior_policy=max_policy,
        context_contributions=max_contributions,
        constraint_checks=tuple(max_checks),
    )
    return RobustIPSResult(
        cutoff=cutoff,
        n_total=len(events),
        n_mature=n_mature,
        n_immature=len(events) - n_mature,
        mature_indices=mature_indices,
        minimum=minimum,
        maximum=maximum,
    )


def _compute_estimates(
    rewards: list[float],
    weights: list[float],
    m_pi: list[float] | None,
    q_logged: list[float] | None,
) -> Estimate:
    n = len(rewards)
    sum_w = math.fsum(weights)
    sum_wr = math.fsum(w * r for w, r in zip(weights, rewards))
    sum_w2 = math.fsum(w * w for w in weights)
    ips = sum_wr / n
    snips = sum_wr / sum_w if sum_w > 0.0 else None
    dr = None
    if m_pi is not None:
        dr = math.fsum(
            m + w * (r - q)
            for m, w, r, q in zip(m_pi, weights, rewards, q_logged)
        ) / n
    ess = (sum_w * sum_w) / sum_w2 if sum_w2 > 0.0 else 0.0
    return Estimate(ips=ips, snips=snips, dr=dr, ess=ess, n=n)


def evaluate(
    events: Sequence[LogEvent],
    target_policy: Mapping[object, Mapping[object, float]],
    cutoff: float,
    q_hat: Mapping | None = None,
    clip_value: float | None = None,
) -> EvaluationResult:
    """按延迟结算口径执行离线策略评估（无副作用；校验失败不返回任何结果）。

    参数
    ----
    events:
        :class:`LogEvent` 序列，长度 <= ``MAX_EVENTS``。
    target_policy:
        目标策略概率表 ``context -> (action -> prob)``。日志中每个 context
        都必须有一行；行概率为有限实数、落在 [0,1]、和与 1 的绝对偏差 <=
        ``PROB_SUM_TOL``；日志中出现过的每个 (context, action) 都必须显式
        列出（漏写不隐式视为 0）。
    cutoff:
        固定评估截止时刻（有限实数，``bool`` 不被接受）。成熟 iff
        ``observed_time <= cutoff``。
    q_hat:
        可选。``(context, action) -> value`` 扁平映射，或
        ``context -> (action -> value)`` 嵌套映射（扁平键优先）。必须为每个
        **成熟 context 概率行内的全部动作**提供有限实数值（DR 的 m_pi 需要
        完整一行）。DR 仅在提供 q_hat 时计算。
    clip_value:
        可选正有限实数 M。给出时额外报告权重 ``min(w, M)`` 的截断组结果；
        未截断组始终报告。

    返回
    ----
    :class:`EvaluationResult`：显式的 ``n_total/n_mature/n_immature`` 与
    ``mature_indices``，以及未截断（和截断）组 IPS/SNIPS/DR/ESS。

    错误语义
    --------
    * 数值 NaN/Infinity、数值参数传 bool、时序矛盾、概率越界/不归一 ->
      ``InvalidLogError`` / ``InvalidPolicyError`` / ``InvalidQHatError``；
    * 全部事件未成熟 -> :class:`NoMatureSamplesError`（不把延迟回报当 0）；
    * 任一事件 p_target>0 而 logged_propensity 为 0 ->
      :class:`ZeroSupportError`（对全部事件校验，不受 cutoff 影响）；
    * 权重超过 ``MAX_WEIGHT`` -> :class:`WeightOverflowError`。
    """
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise InvalidLogError(
            f"cutoff 必须是有限实数，得到 {type(cutoff).__name__}")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff):
        raise InvalidLogError("cutoff 拒绝 NaN/Infinity")

    if clip_value is not None:
        if isinstance(clip_value, bool) or not isinstance(
                clip_value, (int, float)):
            raise InvalidPolicyError(
                "clip_value 必须是正实数或 None，得到 "
                f"{type(clip_value).__name__}")
        clip_value = float(clip_value)
        if not math.isfinite(clip_value) or clip_value <= 0.0:
            raise InvalidPolicyError(
                f"clip_value 必须是正有限实数，得到 {clip_value!r}")

    _validate_events(events)
    _validate_policy(target_policy, events)

    # 零支持是目标策略与日志的结构性不相容，对全部事件校验（与 cutoff 无关）。
    # 权重本身只对成熟样本计算，故溢出等数值检查仅施加于被使用的权重。
    for i, e in enumerate(events):
        p_log = float(e.logged_propensity)
        p_tgt = float(target_policy[e.context][e.action])
        if p_tgt > 0.0 and p_log == 0.0:
            raise ZeroSupportError(
                f"零支持拒绝（events[{i}]）：context={e.context!r}, "
                f"action={e.action!r}，目标概率={p_tgt:g} 而记录策略概率="
                f"{p_log:g}；重要性权重无定义")

    # 成熟口径：observed_time <= cutoff。
    mature_idx = tuple(
        i for i, e in enumerate(events) if e.observed_time <= cutoff)
    if not mature_idx:
        raise NoMatureSamplesError(
            f"cutoff={cutoff:g} 前无成熟样本：{len(events)} 条事件的回报"
            "均尚未观察到；未成熟事件不按 reward=0 计入")

    rewards: list[float] = []
    weights: list[float] = []
    for i in mature_idx:
        e = events[i]
        p_log = float(e.logged_propensity)
        p_tgt = float(target_policy[e.context][e.action])
        if p_tgt <= 0.0:
            w = 0.0
        else:
            w = p_tgt / p_log
            if not math.isfinite(w) or w > MAX_WEIGHT:
                raise WeightOverflowError(
                    f"权重溢出（events[{i}]，成熟样本）：context="
                    f"{e.context!r}, action={e.action!r}，w={p_tgt:g}/"
                    f"{p_log:g} 超过上限 {MAX_WEIGHT:g}")
        rewards.append(float(e.reward))
        weights.append(w)
    m_pi: list[float] | None = None
    q_logged: list[float] | None = None
    if q_hat is not None:
        if not isinstance(q_hat, Mapping):
            raise InvalidQHatError(
                "q_hat 必须是 Mapping（(ctx,action)->v 或 ctx->(action->v)）")
        q_lookup = _make_q_lookup(q_hat)
        m_pi = []
        q_logged = []
        for i in mature_idx:
            e = events[i]
            row = target_policy[e.context]
            # 整个成熟行先取值（含覆盖与有限性校验），再 fsum，保证可定位。
            qs = {a: q_lookup(e.context, a) for a in row}
            m_pi.append(math.fsum(float(row[a]) * qs[a] for a in row))
            q_logged.append(qs[e.action])

    unclipped = _compute_estimates(rewards, weights, m_pi, q_logged)

    clipped_est: Estimate | None = None
    if clip_value is not None:
        cw = [min(w, clip_value) for w in weights]
        clipped_est = _compute_estimates(rewards, cw, m_pi, q_logged)

    return EvaluationResult(
        cutoff=cutoff,
        n_total=len(events),
        n_mature=len(mature_idx),
        n_immature=len(events) - len(mature_idx),
        mature_indices=mature_idx,
        unclipped=unclipped,
        clipped=clipped_est,
        clip_value=clip_value,
        q_hat_used=q_hat is not None,
    )
