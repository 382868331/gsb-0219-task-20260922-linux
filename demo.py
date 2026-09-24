"""固定输入演示：延迟结算口径下的 IPS / SNIPS / DR 对照。

运行：python demo.py（纯标准库，无网络、无 sleep、无预录结果）。

数据由本文件内固定种子的确定性代码生成：已知奖励律 R(x,a)、记录策略 mu、
目标策略 pi 与延迟分布。真实策略价值 V(pi) 由生成律直接代数求和给出，作为
对照（独立于 delayed_ope 的计算）。
"""

from __future__ import annotations

import random
import time

from delayed_ope import (
    LogEvent,
    NoMatureSamplesError,
    Trajectory,
    TrajectoryStep,
    ZeroSupportError,
    evaluate,
    evaluate_trajectories,
)

CONTEXTS = ("news", "music", "shop")
ACTIONS = ("a0", "a1", "a2", "a3")

# 已知生成律（演示“真值”来源；测试不使用本文件）。
CONTEXT_P = {"news": 0.45, "music": 0.35, "shop": 0.20}
MU = {  # 记录策略
    "news":  {"a0": 0.55, "a1": 0.30, "a2": 0.10, "a3": 0.05},
    "music": {"a0": 0.15, "a1": 0.50, "a2": 0.30, "a3": 0.05},
    "shop":  {"a0": 0.10, "a1": 0.15, "a2": 0.55, "a3": 0.20},
}
PI = {  # 目标策略：把质量移向高回报动作
    "news":  {"a0": 0.30, "a1": 0.45, "a2": 0.20, "a3": 0.05},
    "music": {"a0": 0.10, "a1": 0.30, "a2": 0.50, "a3": 0.10},
    "shop":  {"a0": 0.05, "a1": 0.10, "a2": 0.55, "a3": 0.30},
}
REWARD = {
    "news":  {"a0": 0.0, "a1": 1.0, "a2": 0.4, "a3": -0.5},
    "music": {"a0": 0.2, "a1": 0.1, "a2": 1.0, "a3": -0.2},
    "shop":  {"a0": -0.3, "a1": 0.0, "a2": 0.5, "a3": 1.0},
}
# 不完美的 q_hat：向 0.3 收缩 + 固定偏置（刻意不是真值）。
Q_HAT = {
    (x, a): 0.6 * REWARD[x][a] + 0.12
    for x in CONTEXTS for a in ACTIONS
}

N_EVENTS = 6000
CUTOFF = 10.0
CLIP_M = 1.5


def sample_category(rng: random.Random, table: dict) -> str:
    u = rng.random()
    acc = 0.0
    last = None
    for k, p in table.items():
        acc += p
        last = k
        if u < acc:
            return k
    return last  # 浮点边界保护


def generate_events(seed: int) -> list[LogEvent]:
    rng = random.Random(seed)
    events = []
    for i in range(N_EVENTS):
        x = sample_category(rng, CONTEXT_P)
        a = sample_category(rng, MU[x])
        r = REWARD[x][a]
        td = rng.uniform(0.0, CUTOFF)          # 决策时刻
        delay = rng.expovariate(1.0 / 1.2)     # 结算延迟（均值 1.2）
        events.append(LogEvent(
            context=x, action=a, reward=float(r),
            logged_propensity=MU[x][a],
            decision_time=td,
            observed_time=td + delay,
        ))
    return events


def true_value() -> float:
    return sum(
        CONTEXT_P[x] * sum(PI[x][a] * REWARD[x][a] for a in ACTIONS)
        for x in CONTEXTS)


def main() -> None:
    t0 = time.perf_counter()
    events = generate_events(seed=20260923)

    print("=" * 68)
    print("正常结果：延迟结算口径 OPE（固定 cutoff，未成熟样本不计 reward=0）")
    print("=" * 68)
    res = evaluate(events, PI, CUTOFF, q_hat=Q_HAT, clip_value=CLIP_M)
    print(res.summary())
    print(f"q_hat: 不完美（收缩+偏置）；真值 V(pi) = {true_value():.12g}")
    print("-" * 68)
    print("三估计对照（未截断）：")
    u = res.unclipped
    print(f"  IPS   = {u.ips:.12g}")
    print(f"  SNIPS = {u.snips:.12g}")
    print(f"  DR    = {u.dr:.12g}   <- 借 q_hat 降方差")
    c = res.clipped
    print(f"截断 M={CLIP_M:g}：IPS={c.ips:.12g}  SNIPS={c.snips:.12g}  "
          f"DR={c.dr:.12g}")
    print(f"ESS：未截断 {u.ess:.1f}/{u.n}  截断 {c.ess:.1f}/{c.n} "
          "（截断抬高 ESS，但引入偏差，两组都报告）")

    # 同一数据上换用“完美 q_hat=真实奖励律”：校正项恒为 0。
    perfect_q = {(x, a): REWARD[x][a] for x in CONTEXTS for a in ACTIONS}
    rp = evaluate(events, PI, CUTOFF, q_hat=perfect_q)
    print(f"完美 q_hat 时 DR = {rp.unclipped.dr:.12g}（无权重噪声，"
          "仅随已实现 context 分布微动）")

    print()
    print("=" * 68)
    print("实际触发的拒绝边界（真实计算，非预录文本）")
    print("=" * 68)
    # 边界 1：目标有正概率、记录策略支持为零 -> ZeroSupportError。
    bad_policy = {x: dict(PI[x]) for x in CONTEXTS}
    bad_events = list(events) + [LogEvent(
        context="news", action="a3", reward=1.0,
        logged_propensity=0.0,                 # 记录策略在此 (x,a) 零支持
        decision_time=1.0, observed_time=2.0)]
    try:
        evaluate(bad_events, bad_policy, CUTOFF, q_hat=Q_HAT)
    except ZeroSupportError as exc:
        print(f"[拒绝 1] ZeroSupportError: {exc}")

    # 边界 2：cutoff 提前到所有回报都未观察 -> NoMatureSamplesError，
    # 直接证明未成熟事件没有被当作 reward=0 参与估计。
    try:
        evaluate(events, PI, cutoff=-1.0, q_hat=Q_HAT)
    except NoMatureSamplesError as exc:
        print(f"[拒绝 2] NoMatureSamplesError: {exc}")

    print("-" * 68)
    print(f"实际计算耗时 {time.perf_counter() - t0:.3f} 秒（无 sleep）。")


# ---- 多步轨迹演示（<=5 步）：TIS / PDIS 对照小 MDP 枚举真值 --------------
MDP_STATES = ("s0", "s1")
MDP_ACTIONS = ("a0", "a1")
MDP_P0 = {"s0": 0.5, "s1": 0.5}
MDP_MU = {  # 行为策略
    "s0": {"a0": 0.75, "a1": 0.25},
    "s1": {"a0": 0.25, "a1": 0.75},
}
MDP_PI = {  # 目标策略
    "s0": {"a0": 0.25, "a1": 0.75},
    "s1": {"a0": 0.50, "a1": 0.50},
}
MDP_REWARD = {
    "s0": {"a0": 1.0, "a1": -0.5},
    "s1": {"a0": 0.25, "a1": 2.0},
}
MDP_HORIZON = 4
N_TRAJECTORIES = 20000


def mdp_true_value() -> float:
    """按 MDP 分布定义递归枚举的精确 V(pi)（独立于 delayed_ope）。"""
    def v(x, t):
        if t > MDP_HORIZON:
            return 0.0
        nxt = "s1" if x == "s0" else "s0"  # 动作 a1 翻转状态，a0 停留
        return sum(
            MDP_PI[x][a] * (MDP_REWARD[x][a]
                            + v(nxt if a == "a1" else x, t + 1))
            for a in MDP_ACTIONS)

    return sum(MDP_P0[x] * v(x, 1) for x in MDP_STATES)


def generate_trajectories(seed: int) -> list[Trajectory]:
    rng = random.Random(seed)
    trajs = []
    for _ in range(N_TRAJECTORIES):
        x = "s0" if rng.random() < MDP_P0["s0"] else "s1"
        steps = []
        for _ in range(MDP_HORIZON):
            a = "a0" if rng.random() < MDP_MU[x]["a0"] else "a1"
            steps.append(TrajectoryStep(x, a, MDP_REWARD[x][a], MDP_MU[x][a]))
            if a == "a1":
                x = "s1" if x == "s0" else "s0"
        trajs.append(Trajectory(tuple(steps)))
    return trajs


def trajectory_demo() -> None:
    print()
    print("=" * 68)
    print(f"多步完整轨迹（{MDP_HORIZON} 步）：TIS / PDIS 对照枚举真值")
    print("=" * 68)
    trajs = generate_trajectories(seed=20260924)
    res = evaluate_trajectories(trajs, MDP_PI)
    print(res.summary())
    v = mdp_true_value()
    print(f"枚举真值 V(pi) = {v:.12g}")
    print(f"偏差：TIS {res.tis - v:+.4g}，PDIS {res.pdis - v:+.4g}"
          "（随机日志上仅有统计一致性；普查恒等见测试）")
    d = res.diagnostics
    print(f"有效权重诊断：末步 ESS={d.ess:.1f}/{res.n_trajectories}，"
          f"逐步 ESS 随累乘步数下降（权重退化显式可见）")

    # 实际触发的拒绝：某步行为概率为 0 而目标策略概率为正。
    bad = list(trajs) + [Trajectory((
        TrajectoryStep("s0", "a1", 1.0, 0.5),
        TrajectoryStep("s1", "a0", 1.0, 0.0),   # 行为策略在此步零支持
    ))]
    try:
        evaluate_trajectories(bad, MDP_PI)
    except ZeroSupportError as exc:
        print(f"[拒绝 3] ZeroSupportError: {exc}")


if __name__ == "__main__":
    main()
    trajectory_demo()
