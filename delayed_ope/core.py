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

本模块只实现本题契约的有限数学子集：实数标量、有限时间、离散上下文/动作。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = [
    "LogEvent",
    "Estimate",
    "EvaluationResult",
    "evaluate",
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
