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
    InvalidTrajectoryError,
    LogEvent,
    NoMatureSamplesError,
    Step,
    Trajectory,
    WeightOverflowError,
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


# ---- 轨迹演示用的小 MDP（确定性转移/奖励；mu、pi 为状态相关随机策略） ----
T_STATES = (0, 1, 2)
T_ACTIONS = (0, 1)
TRAJ_HORIZON = 5
TRAJ_GAMMA = 0.9
N_TRAJ = 4000

T_TRANS = {  # 确定性转移
    (0, 0): 1, (0, 1): 2,
    (1, 0): 1, (1, 1): 2,
    (2, 0): 1, (2, 1): 0,
}
T_REWARD = {
    (0, 0): 1.0, (0, 1): 0.0,
    (1, 0): 0.0, (1, 1): 2.0,
    (2, 0): -1.0, (2, 1): 3.0,
}
T_MU = {  # 记录策略
    0: {0: 0.55, 1: 0.45},
    1: {0: 0.70, 1: 0.30},
    2: {0: 0.40, 1: 0.60},
}
T_PI = {  # 目标策略：偏向高回报动作
    0: {0: 0.35, 1: 0.65},
    1: {0: 0.20, 1: 0.80},
    2: {0: 0.55, 1: 0.45},
}


def generate_trajectories(seed: int):
    """按 mu 滚动小 MDP 生成固定 horizon 轨迹；真值由反向递推独立给出。"""
    rng = random.Random(seed)
    trajs = []
    for _ in range(N_TRAJ):
        s = 0
        steps = []
        for _t in range(TRAJ_HORIZON):
            a = sample_category(rng, T_MU[s])
            steps.append(Step(
                state=s, action=a, reward=float(T_REWARD[(s, a)]),
                logged_prob=T_MU[s][a], target_prob=T_PI[s][a]))
            s = T_TRANS[(s, a)]
        trajs.append(Trajectory(tuple(steps), TRAJ_GAMMA))
    return trajs, trajectory_true_value()


def trajectory_true_value() -> float:
    """Bellman 反向递推 V_t(s)=Σ_a pi(a|s)[R+γV_{t+1}(T)]（独立真值）。"""
    v = {s: 0.0 for s in T_STATES}
    for _ in range(TRAJ_HORIZON):
        v = {
            s: sum(T_PI[s][a]
                   * (T_REWARD[(s, a)]
                      + TRAJ_GAMMA * v[T_TRANS[(s, a)]])
                   for a in T_ACTIONS)
            for s in T_STATES
        }
    return v[0]


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

    print()
    print("=" * 68)
    print("轨迹 OPE：最多 5 步完整轨迹，逐步累乘权重（固定种子随机采样）")
    print("=" * 68)
    trajs, v_traj_true = generate_trajectories(seed=20260913)
    tres = evaluate_trajectories(trajs)
    print(tres.summary())
    print(f"真值（按生成律反向递推，独立于本库）V_pi(H={TRAJ_HORIZON}, "
          f"gamma={TRAJ_GAMMA:g}) = {v_traj_true:.12g}")
    te = tres.estimate
    print("-" * 68)
    print("本次抽样（固定种子）：r_t 只带累乘到 t 的前缀权重，PD 两项")
    print("的抽样误差均小于整条轨迹权重；方差性质随样本而异，不是逐样本保证。")
    print(f"  |IS    - V| = {abs(te.is_ - v_traj_true):.4g}")
    print(f"  |SNIS  - V| = {abs(te.snis - v_traj_true):.4g}")
    print(f"  |PD-IS - V| = {abs(te.pd_is - v_traj_true):.4g}")
    print(f"  |PD-SNIS-V| = {abs(te.pd_snis - v_traj_true):.4g}")

    print()
    print("=" * 68)
    print("轨迹边界与实际错误（真实计算，非预录文本）")
    print("=" * 68)
    # 合法边界：某步 pi=0 且 mu=0 -> 因子 0；与一条普通轨迹混批，
    # 权重诊断显式暴露 min=0，SNIS 仍由另一条定义。
    edge = [
        Trajectory((Step("s0", 1, 5.0, 0.0, 0.0),
                    Step("s2", 0, 1.0, 0.0, 0.0)), TRAJ_GAMMA),
        Trajectory((Step("s0", 0, 1.0, 0.6, 0.4),
                    Step("s1", 1, 2.0, 0.3, 0.5)), TRAJ_GAMMA),
    ]
    er = evaluate_trajectories(edge)
    print(f"[合法边界] 含 pi=0/mu=0 零因子：终权重 min={er.weights.min:g}, "
          f"max={er.weights.max:.6g}，IS={er.estimate.is_:.6g}")

    # 实际错误 1：任一步 pi>0 而 mu=0 -> ZeroSupportError。
    try:
        evaluate_trajectories([Trajectory((
            Step("s0", 1, 1.0, 0.0, 0.6),
            Step("s2", 0, 1.0, 0.2, 0.3),
        ), TRAJ_GAMMA)])
    except ZeroSupportError as exc:
        print(f"[错误 1] ZeroSupportError: {exc}")

    # 实际错误 2：逐步累乘溢出（1e200 * 1e200 = 1e400）。
    try:
        evaluate_trajectories([Trajectory((
            Step("s0", 0, 1.0, 1e-200, 1.0),
            Step("s1", 0, 1.0, 1e-200, 1.0),
        ), 1.0)])
    except WeightOverflowError as exc:
        print(f"[错误 2] WeightOverflowError: {exc}")

    # 实际错误 3：6 步轨迹超过 5 步上限。
    try:
        evaluate_trajectories([Trajectory(tuple(
            Step(t % 3, 0, 1.0, 0.5, 0.5) for t in range(6)), 1.0)])
    except InvalidTrajectoryError as exc:
        print(f"[错误 3] InvalidTrajectoryError: {exc}")

    print("-" * 68)
    print(f"实际计算耗时 {time.perf_counter() - t0:.3f} 秒（无 sleep）。")


if __name__ == "__main__":
    main()
