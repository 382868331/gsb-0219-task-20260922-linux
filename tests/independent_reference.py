"""独立参考：只在给定的有限上下文-动作分布表上做精确期望代数。

本模块**不导入 delayed_ope**，期望值直接由分布定义计算，用于在测试中给出
与被测实现相互独立的基准：

    V(pi) = sum_x p(x) * sum_a pi(a|x) * R(x, a)

以及在“普查样本”（每个 (x,a) 出现次数严格正比于 p(x)*mu(a|x)）上，
IPS / SNIPS / DR 应当恒等于 V(pi) 的代数恒等式参考值。
"""

from __future__ import annotations

# 固定的小有限世界：2 个上下文，3 个动作。
CONTEXTS = (0, 1)
ACTIONS = (0, 1, 2)
# 每个上下文普查条数（mu 的分母），取 10 使常见概率为整数计数。
CENSUS_PER_CONTEXT = 10


def policy_value(context_p, pi, reward_table):
    """V(pi) = Σ_x p(x) Σ_a pi(a|x) R(x,a)，按定义直接求和。"""
    total = 0.0
    for x in CONTEXTS:
        row = 0.0
        for a in ACTIONS:
            row += pi[x][a] * reward_table[x][a]
        total += context_p[x] * row
    return total
