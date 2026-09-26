"""独立参考：只在给定的有限上下文-动作分布表上做精确期望代数。

本模块**不导入 delayed_ope**，期望值直接由分布定义计算，用于在测试中给出
与被测实现相互独立的基准：

    V(pi) = sum_x p(x) * sum_a pi(a|x) * R(x, a)

以及在“普查样本”（每个 (x,a) 出现次数严格正比于 p(x)*mu(a|x)）上，
IPS / SNIPS / DR 应当恒等于 V(pi) 的代数恒等式参考值。
"""

from __future__ import annotations

import math

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


def two_action_interval_extrema(c0, c1, low0, high0, low1, high1):
    """二维 f=c0/p+c1/(1-p) 的端点和驻点参照。

    本函数只按一元函数定义枚举，不导入 delayed_ope。返回原始（未除以成熟
    样本数）最小值、最大值及对应的 p0。
    """
    p_low = max(low0, 1.0 - high1)
    p_high = min(high0, 1.0 - low1)
    candidates = [p_low, p_high]
    if (c0 > 0.0 and c1 > 0.0) or (c0 < 0.0 and c1 < 0.0):
        candidates.append(
            math.sqrt(abs(c0)) / (math.sqrt(abs(c0)) + math.sqrt(abs(c1))))

    def objective(p):
        return c0 / p + c1 / (1.0 - p)

    feasible = [p for p in candidates if p_low - 1e-12 <= p <= p_high + 1e-12]
    p_min = min(feasible, key=objective)
    p_max = max(feasible, key=objective)
    return objective(p_min), objective(p_max), p_min, p_max
