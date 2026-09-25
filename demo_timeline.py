"""按观察时间推进的增量 OPE 演示（纯标准库、离线、确定性种子）。

运行：python demo_timeline.py（无网络、无 sleep、无预录结果）。

展示：
1. 同一份冻结日志上，随 cutoff 推进，样本逐渐成熟，IPS/SNIPS/DR/ESS 随之
   改变 —— 不同截止时刻对应不同的估计对象口径（该时刻已成熟子日志上的策
   略价值估计），估计改变是口径使然；
2. 每个截止时刻都与批量入口 evaluate 独立重算的结果逐位核对；
3. 真实触发倒退截止与重复 event_id 两类拒绝。
"""

from __future__ import annotations

import random
import time

from delayed_ope import (
    CutoffRegressionError,
    DuplicateEventIdError,
    LogEvent,
    TimelineEvaluator,
    evaluate,
)

CONTEXTS = ("news", "music")
ACTIONS = ("a0", "a1", "a2")
CONTEXT_P = {"news": 0.6, "music": 0.4}
MU = {
    "news":  {"a0": 0.50, "a1": 0.30, "a2": 0.20},
    "music": {"a0": 0.20, "a1": 0.30, "a2": 0.50},
}
PI = {
    "news":  {"a0": 0.25, "a1": 0.50, "a2": 0.25},
    "music": {"a0": 0.10, "a1": 0.40, "a2": 0.50},
}
REWARD = {
    "news":  {"a0": 0.0, "a1": 1.0, "a2": 0.3},
    "music": {"a0": -0.2, "a1": 0.4, "a2": 1.0},
}
Q_HAT = {(x, a): 0.7 * REWARD[x][a] + 0.1 for x in CONTEXTS for a in ACTIONS}

N_EVENTS = 3000
HORIZON = 12.0
CUTOFFS = (3.0, 6.0, 9.0, 12.0, 18.0)
CLIP_M = 2.0


def sample(rng, table):
    u = rng.random()
    acc = 0.0
    last = None
    for k, p in table.items():
        acc += p
        last = k
        if u < acc:
            return k
    return last


def generate_events(seed):
    rng = random.Random(seed)
    events = []
    for i in range(N_EVENTS):
        x = sample(rng, CONTEXT_P)
        a = sample(rng, MU[x])
        td = rng.uniform(0.0, HORIZON)
        delay = rng.expovariate(1.0 / 1.5)
        events.append(LogEvent(
            context=x, action=a, reward=float(REWARD[x][a]),
            logged_propensity=MU[x][a],
            decision_time=td, observed_time=td + delay,
            event_id=f"ev-{i}"))
    return events


def main():
    t0 = time.perf_counter()
    events = generate_events(seed=20260925)

    print("=" * 76)
    print("按观察时间推进的增量 OPE（一次加载冻结，advance 单调推进，同刻整组纳入）")
    print("=" * 76)
    tl = TimelineEvaluator(events, PI, q_hat=Q_HAT, clip_value=CLIP_M)
    print(f"日志 {len(events)} 条已加载并冻结；clip_value={CLIP_M:g} 创建时固定")
    print(f"{'cutoff':>7} {'n_mature':>8} {'IPS':>10} {'SNIPS':>10} "
          f"{'DR':>10} {'ESS':>8}  与批量一致")
    for c in CUTOFFS:
        tl.advance(c)
        got = tl.query()
        ref = evaluate(events, PI, c, q_hat=Q_HAT, clip_value=CLIP_M)
        same = (
            got.unclipped == ref.unclipped and got.clipped == ref.clipped
            and got.mature_indices == ref.mature_indices)
        u = got.unclipped
        print(f"{c:7.1f} {got.n_mature:8d} {u.ips:10.6f} {u.snips:10.6f} "
              f"{u.dr:10.6f} {u.ess:8.1f}  {same}")
        assert same, "增量查询与批量入口不一致"
    print("-" * 76)
    print("说明：每个 cutoff 的估计对象是“该时刻已成熟子日志”上的策略价值；")
    print("样本随时间成熟、组成改变，估计值随之改变属于口径定义，而非不稳定。")
    print("重复推进幂等：再次 advance(18.0) 结果不变 ->", end=" ")
    before = tl.query()
    tl.advance(18.0)
    print(tl.query().unclipped == before.unclipped)

    print()
    print("=" * 76)
    print("实际触发的拒绝边界（真实计算，非预录文本）")
    print("=" * 76)
    try:
        tl.advance(6.0)  # 倒退
    except CutoffRegressionError as exc:
        print(f"[拒绝 1] CutoffRegressionError: {exc}")
    dup = [LogEvent("news", "a0", 1.0, 0.5, 0.0, 1.0, event_id="k"),
           LogEvent("news", "a1", 1.0, 0.3, 0.0, 1.0, event_id="k")]
    try:
        TimelineEvaluator(dup, PI)
    except DuplicateEventIdError as exc:
        print(f"[拒绝 2] DuplicateEventIdError: {exc}")
    print("-" * 76)
    print(f"实际计算耗时 {time.perf_counter() - t0:.3f} 秒（无 sleep）。")


if __name__ == "__main__":
    main()

