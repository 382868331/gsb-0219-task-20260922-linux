"""独立参考：小 MDP 上按定义枚举全部轨迹的精确期望，不导入 delayed_ope。

本模块只按有限 MDP 的分布定义做代数求和，为轨迹测试提供与被测实现相互
独立的基准：

    V(pi) = E_pi[ sum_{t=1..H} r_t ]
          = sum_x0 p0(x0) * v(x0, 1)
    v(x, t) = sum_a pi(a|x) * ( R(x,a) + sum_x' P(x'|x,a) * v(x', t+1) )
    v(x, H+1) = 0

回报为无折扣逐步和，与被测轨迹模块的口径一致。
"""

from __future__ import annotations

# 固定的小 MDP 世界：2 个状态（context）、2 个动作。
STATES = (0, 1)
ACTIONS = (0, 1)


def mdp_value(p0, transitions, rewards, policy, horizon):
    """按定义递归枚举的精确期望 V(pi)（无折扣逐步回报和）。

    参数
    ----
    p0:          {state: prob} 初始状态分布；
    transitions: {state: {action: {next_state: prob}}}；
    rewards:     {state: {action: float}} 确定性逐步回报；
    policy:      {state: {action: prob}} 被评估策略；
    horizon:     步数上限 H（>=1）。
    """
    def v(x, t):
        if t > horizon:
            return 0.0
        total = 0.0
        for a in ACTIONS:
            r = rewards[x][a]
            future = 0.0
            for x2, p in transitions[x][a].items():
                future += p * v(x2, t + 1)
            total += policy[x][a] * (r + future)
        return total

    return sum(p0[x] * v(x, 1) for x in STATES)
