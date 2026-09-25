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

按观察时间推进的增量接口
-------------------------
TimelineEvaluator 在创建时一次加载并冻结日志与目标策略（及可选的 q_hat、
clip_value），之后用 advance(cutoff) 单调推进观察时间：只有
observed_time <= cutoff 且此前未纳入的事件组被加入。同一 observed_time
的记录作为整组原子纳入，绝不暴露半组状态。查询 query() 返回的
IPS/SNIPS/DR/ESS 与在同截止时刻直接调用 evaluate 的结果完全相同（同一批
成熟样本、同一事件顺序、同一个 _compute_estimates 与 math.fsum 求和）。
未成熟回报从不按 0 计入；尚无任何成熟样本时查询抛 NoMatureSamplesError，
与批量入口口径一致（分母为零不凭空给 0 估计）。
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


class DuplicateEventIdError(InvalidLogError):
    """增量接口输入日志出现重复（或缺失/不可哈希）的 event_id。"""


class CutoffRegressionError(OPEError):
    """advance 的截止时刻相对已推进时刻倒退（等值重推进是幂等合法操作）。"""


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
    event_id:
        可选事件标识（可哈希）。批量 evaluate 忽略本字段；增量接口
        TimelineEvaluator 要求每条事件给出唯一非 None 的 event_id，
        重复即拒绝（防止同一决策被重复纳入累计量）。
    """

    context: object
    action: object
    reward: float
    logged_propensity: float
    decision_time: float
    observed_time: float
    event_id: object = None


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


def _validate_cutoff(cutoff) -> float:
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
        raise InvalidLogError(
            f"cutoff 必须是有限实数，得到 {type(cutoff).__name__}")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff):
        raise InvalidLogError("cutoff 拒绝 NaN/Infinity")
    return cutoff


def _validate_clip_value(clip_value):
    if clip_value is None:
        return None
    if isinstance(clip_value, bool) or not isinstance(
            clip_value, (int, float)):
        raise InvalidPolicyError(
            "clip_value 必须是正实数或 None，得到 "
            f"{type(clip_value).__name__}")
    clip_value = float(clip_value)
    if not math.isfinite(clip_value) or clip_value <= 0.0:
        raise InvalidPolicyError(
            f"clip_value 必须是正有限实数，得到 {clip_value!r}")
    return clip_value


def _check_zero_support(events, target_policy) -> None:
    # 零支持是目标策略与日志的结构性不相容，对全部事件校验（与 cutoff 无关）。
    for i, e in enumerate(events):
        p_log = float(e.logged_propensity)
        p_tgt = float(target_policy[e.context][e.action])
        if p_tgt > 0.0 and p_log == 0.0:
            raise ZeroSupportError(
                f"零支持拒绝（events[{i}]）：context={e.context!r}, "
                f"action={e.action!r}，目标概率={p_tgt:g} 而记录策略概率="
                f"{p_log:g}；重要性权重无定义")


def _mature_weight(e, target_policy, i: int) -> float:
    # 权重只对成熟样本计算，溢出等数值检查仅施加于被使用的权重。
    p_log = float(e.logged_propensity)
    p_tgt = float(target_policy[e.context][e.action])
    if p_tgt <= 0.0:
        return 0.0
    w = p_tgt / p_log
    if not math.isfinite(w) or w > MAX_WEIGHT:
        raise WeightOverflowError(
            f"权重溢出（events[{i}]，成熟样本）：context="
            f"{e.context!r}, action={e.action!r}，w={p_tgt:g}/"
            f"{p_log:g} 超过上限 {MAX_WEIGHT:g}")
    return w


def _event_dr_terms(e, target_policy, q_lookup):
    # 整个概率行先取值（含覆盖与有限性校验），再 fsum，保证可定位。
    row = target_policy[e.context]
    qs = {a: q_lookup(e.context, a) for a in row}
    m_pi = math.fsum(float(row[a]) * qs[a] for a in row)
    return m_pi, qs[e.action]


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
    cutoff = _validate_cutoff(cutoff)
    clip_value = _validate_clip_value(clip_value)

    _validate_events(events)
    _validate_policy(target_policy, events)
    _check_zero_support(events, target_policy)

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
        weights.append(_mature_weight(e, target_policy, i))
        rewards.append(float(e.reward))
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
            m, q = _event_dr_terms(events[i], target_policy, q_lookup)
            m_pi.append(m)
            q_logged.append(q)

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


class TimelineEvaluator:
    """按观察时间推进的增量 OPE：一次加载冻结，随 cutoff 单调纳入成熟样本。

    与批量 evaluate 的关系
    ----------------------
    创建时完成与 cutoff 无关的全部校验（日志结构、策略表、零支持、clip 配
    置、event_id 唯一性），并冻结输入快照；之后 advance 单调推进观察时间，
    query 返回的 IPS/SNIPS/DR/ESS 与在同一截止时刻直接调用批量 evaluate 的
    结果完全相同（同一成熟样本集、同一原始事件顺序、同一 _compute_estimates
    与 math.fsum 稳定求和）。

    契约
    ----
    * 每条事件必须携带唯一非 None 的可哈希 event_id，重复/缺失/不可哈希
      即在创建时抛 DuplicateEventIdError。
    * 同一 observed_time 的记录作为整组原子纳入；任何中间状态都不会暴露
      半组结果（组内全部贡献先计算完毕才提交，组内出错则整批不生效）。
    * advance(cutoff) 只允许单调不减：倒退抛 CutoffRegressionError；以相同
      cutoff 重复推进是幂等空操作。
    * 未成熟回报从不按 0 计入；尚无任何成熟样本时 query 抛
      NoMatureSamplesError（与批量入口同口径，分母为零不凭空给 0）。
    * 权重截断配置 clip_value 在创建时固定，之后不可更改。
    * 权重溢出（WeightOverflowError）与 q_hat 取值错误（InvalidQHatError）
      与批量入口一样只在样本真正成熟时被触发，即在包含该样本的 advance 调
      用中抛出，且该次 advance 不产生任何部分变更。

    不同截止时刻的查询对应不同的估计对象口径：cutoff=T 的查询是对“T 之前
    已成熟子日志”这一有限总体的策略价值估计，样本组成随 T 改变，估计值随
    之改变是口径使然，不是实现不稳定。
    """

    def __init__(
        self,
        events: Sequence[LogEvent],
        target_policy: Mapping[object, Mapping[object, float]],
        q_hat: Mapping | None = None,
        clip_value: float | None = None,
    ) -> None:
        clip_value = _validate_clip_value(clip_value)
        _validate_events(events)
        _validate_policy(target_policy, events)
        _check_zero_support(events, target_policy)
        if q_hat is not None and not isinstance(q_hat, Mapping):
            raise InvalidQHatError(
                "q_hat 必须是 Mapping（(ctx,action)->v 或 ctx->(action->v)）")

        # 冻结快照：之后的推进与查询不依赖调用方持有的可变容器。
        self._events: tuple[LogEvent, ...] = tuple(events)
        self._policy = {ctx: dict(row) for ctx, row in target_policy.items()}
        self._q_hat = dict(q_hat) if q_hat is not None else None
        self._q_lookup = _make_q_lookup(self._q_hat) if q_hat is not None else None
        self._clip_value = clip_value

        seen: set = set()
        for i, e in enumerate(self._events):
            eid = e.event_id
            if eid is None:
                raise DuplicateEventIdError(
                    f"events[{i}] 缺少 event_id（增量接口要求唯一非 None 标识）")
            try:
                hash(eid)
            except TypeError as exc:
                raise DuplicateEventIdError(
                    f"events[{i}].event_id 必须可哈希，得到 "
                    f"{type(eid).__name__}") from exc
            if eid in seen:
                raise DuplicateEventIdError(
                    f"重复 event_id（events[{i}]）：{eid!r}；同一决策不允许"
                    "被重复纳入累计量")
            seen.add(eid)

        # 按 observed_time 分组（同刻同组），组序列升序，推进指针单调右移。
        groups: dict[float, list[int]] = {}
        for i, e in enumerate(self._events):
            groups.setdefault(float(e.observed_time), []).append(i)
        self._groups: list[tuple[float, tuple[int, ...]]] = sorted(
            ((t, tuple(idx)) for t, idx in groups.items()),
            key=lambda g: g[0])
        self._next_group = 0
        self._cutoff: float | None = None
        # 已成熟事件的贡献，按原始事件下标升序保持（与批量入口的求和顺序一致，
        # 保证 fsum 结果逐位相同）。条目：(index, w, r, m_pi, q_logged)。
        self._entries: list[tuple[int, float, float, float, float]] = []

    @property
    def cutoff(self) -> float | None:
        """当前已推进到的截止时刻；尚未 advance 时为 None。"""
        return self._cutoff

    @property
    def n_mature(self) -> int:
        return len(self._entries)

    def advance(self, cutoff: float) -> None:
        """推进观察时间到 cutoff，整组纳入新成熟样本（幂等、拒绝倒退）。

        失败（权重溢出、q_hat 取值非法等）时不留下任何部分变更：所有新组成
        熟的贡献先完整计算，全部成功后才一次性提交。
        """
        cutoff = _validate_cutoff(cutoff)
        if self._cutoff is not None:
            if cutoff < self._cutoff:
                raise CutoffRegressionError(
                    f"cutoff 倒退拒绝：当前已推进到 {self._cutoff:g}，"
                    f"收到 {cutoff:g}；观察时间只能单调不减")
            if cutoff == self._cutoff:
                return  # 等值重推进：幂等空操作

        new_entries: list[tuple[int, float, float, float, float]] = []
        next_group = self._next_group
        while (next_group < len(self._groups)
               and self._groups[next_group][0] <= cutoff):
            for i in self._groups[next_group][1]:
                e = self._events[i]
                w = _mature_weight(e, self._policy, i)
                if self._q_lookup is not None:
                    m, q = _event_dr_terms(e, self._policy, self._q_lookup)
                else:
                    m, q = 0.0, 0.0
                new_entries.append((i, w, float(e.reward), m, q))
            next_group += 1

        # 全部新组计算成功后才提交：按原始下标归并保持求和顺序。
        new_entries.sort(key=lambda t: t[0])
        merged: list[tuple[int, float, float, float, float]] = []
        it_old, it_new = iter(self._entries), iter(new_entries)
        old = next(it_old, None)
        new = next(it_new, None)
        while old is not None or new is not None:
            if new is None or (old is not None and old[0] < new[0]):
                merged.append(old)
                old = next(it_old, None)
            else:
                merged.append(new)
                new = next(it_new, None)
        self._entries = merged
        self._next_group = next_group
        self._cutoff = cutoff

    def query(self) -> EvaluationResult:
        """返回当前截止时刻的评估结果，与同 cutoff 的批量 evaluate 完全相同。"""
        if not self._entries:
            where = (f"cutoff={self._cutoff:g}" if self._cutoff is not None
                     else "尚未 advance")
            raise NoMatureSamplesError(
                f"{where} 前无成熟样本：{len(self._events)} 条事件的回报"
                "均尚未观察到；未成熟事件不按 reward=0 计入")
        weights = [t[1] for t in self._entries]
        rewards = [t[2] for t in self._entries]
        if self._q_lookup is not None:
            m_pi = [t[3] for t in self._entries]
            q_logged = [t[4] for t in self._entries]
        else:
            m_pi = q_logged = None
        unclipped = _compute_estimates(rewards, weights, m_pi, q_logged)
        clipped_est: Estimate | None = None
        if self._clip_value is not None:
            cw = [min(w, self._clip_value) for w in weights]
            clipped_est = _compute_estimates(rewards, cw, m_pi, q_logged)
        return EvaluationResult(
            cutoff=self._cutoff,
            n_total=len(self._events),
            n_mature=len(self._entries),
            n_immature=len(self._events) - len(self._entries),
            mature_indices=tuple(t[0] for t in self._entries),
            unclipped=unclipped,
            clipped=clipped_est,
            clip_value=self._clip_value,
            q_hat_used=self._q_lookup is not None,
        )
