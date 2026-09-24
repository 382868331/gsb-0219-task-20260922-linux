"""独立参考：小 MDP 上按分布定义直接枚举轨迹真值。

本模块**不导入 delayed_ope**。它在一个确定性转移、确定性奖励、2 动作、
3 状态、有限 horizon 的小 MDP 上：

* 按记录策略 mu 枚举全部动作序列并给出整数多重数（mu 概率为十分数，
  深度 L 时每条序列的多重数恰为 ``10**L * P_mu(序列)``），据此构造
  “普查轨迹”——每条不同序列出现的次数严格正比于其 mu 概率；
* 按目标策略 pi 独立枚举，直接求折扣回报的分布均值作为真值；
* 同时给出逐步概率，供被测核心逐步累乘。

普查上轨迹 IS/SNIS 与逐决策 PD-IS/PD-SNIS 都应恒等于真值：权重与计数
的乘积是精确整数比，不依赖被测实现生成期望值。
"""

from __future__ import annotations

from dataclasses import dataclass

# 状态 0=入口，1=左室，2=右室；动作 0/1。转移与奖励均确定。
STATES = (0, 1, 2)
ACTIONS = (0, 1)
START_STATE = 0
BRANCH = 10  # mu/pi 概率均为十分数


def transition(state, action):
    """确定性转移 T(s,a)。"""
    if state == 0:
        return 1 if action == 0 else 2
    if state == 1:
        return 1 if action == 0 else 2
    return 1 if action == 0 else 2


def reward(state, action):
    """确定性奖励 R(s,a)。"""
    table = {
        (0, 0): 1.0, (0, 1): 0.0,
        (1, 0): 0.0, (1, 1): 2.0,
        (2, 0): -1.0, (2, 1): 3.0,
    }
    return table[(state, action)]


@dataclass(frozen=True)
class Leaf:
    states: tuple       # s_0..s_L（s_{t+1}=T(s_t,a_t)）
    actions: tuple      # a_0..a_{L-1}
    mu_mult: int        # 该序列在 mu 下的整数多重数（合计 10**L）
    pi_mult: int        # 该序列在 pi 下的整数多重数


def _enumerate(mu_counts, pi_counts, state, depth, states, actions,
               mu_mult, pi_mult, out):
    if depth == 0:
        out.append(Leaf(tuple(states), tuple(actions), mu_mult, pi_mult))
        return
    for a in ACTIONS:
        cm = mu_counts[state][a]
        cp = pi_counts[state][a]
        nxt = transition(state, a)
        _enumerate(mu_counts, pi_counts, nxt, depth - 1,
                   states + [nxt], actions + [a],
                   mu_mult * cm, pi_mult * cp, out)


def enumerate_leaves(mu_counts, pi_counts, horizon):
    """枚举全部动作序列及其在 mu/pi 下的整数多重数。"""
    out = []
    _enumerate(mu_counts, pi_counts, START_STATE, horizon,
               [START_STATE], [], 1, 1, out)
    return out


def discounted_return(leaf, discount):
    g = 0.0
    gt = 1.0
    for t, a in enumerate(leaf.actions):
        g += gt * reward(leaf.states[t], a)
        gt *= discount
    return g


def true_value_under_pi(pi_counts, horizon, discount):
    """直接在 pi 分布上枚举：V = Σ_seq P_pi(seq) * G(seq)。"""
    # mu 仅用于枚举形状；pi 多重数与 mu 无关，mu 传全 5 即可。
    dummy = {s: {0: 5, 1: 5} for s in STATES}
    leaves = enumerate_leaves(dummy, pi_counts, horizon)
    total = 0.0
    for leaf in leaves:
        prob_pi = leaf.pi_mult / BRANCH ** horizon
        total += prob_pi * discounted_return(leaf, discount)
    return total


def leaf_step_probs(leaf, mu_counts, pi_counts):
    """返回序列每步的 (mu_prob, pi_prob)，供逐步累乘。"""
    return [
        (mu_counts[leaf.states[t]][leaf.actions[t]] / BRANCH,
         pi_counts[leaf.states[t]][leaf.actions[t]] / BRANCH)
        for t in range(len(leaf.actions))
    ]


def policy_value_backup(pi_counts, horizon, discount):
    """有限时域策略评估的反向递推（独立于枚举的第二种真值算法）。

    V_t(s) = Σ_a pi(a|s) [ R(s,a) + γ V_{t+1}(T(s,a)) ]，V_L ≡ 0。
    """
    v = {s: 0.0 for s in STATES}
    for _ in range(horizon):
        nv = {}
        for s in STATES:
            nv[s] = sum(
                pi_counts[s][a] / BRANCH
                * (reward(s, a) + discount * v[transition(s, a)])
                for a in ACTIONS)
        v = nv
    return v[START_STATE]
