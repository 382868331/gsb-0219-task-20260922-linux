"""本次迭代演示：共享行为概率变量下的稳健 IPS 最小/最大值。"""

from __future__ import annotations

import time

from delayed_ope import (
    InfeasibleIntervalError,
    RobustObservation,
    robust_ips_range,
)


def fmt_table(table):
    parts = []
    for ctx, row in table.items():
        probs = ", ".join(f"{action!r}: {p:.12g}" for action, p in row.items())
        parts.append(f"{ctx!r}: {{{probs}}}")
    return "{" + "; ".join(parts) + "}"


def main():
    start = time.perf_counter()

    observations = [
        RobustObservation("x", 0, 2.0),
        RobustObservation("x", 0, -2.0),
        RobustObservation("x", 1, 1.0),
        RobustObservation("y", 0, -1.0),
        RobustObservation("y", 1, 2.0),
        RobustObservation("y", 2, 0.5),
    ]
    target_policy = {
        "x": {0: 0.5, 1: 0.5},
        "y": {0: 0.25, 1: 0.5, 2: 0.25},
    }
    propensity_intervals = {
        "x": {0: (0.25, 0.70), 1: (0.30, 0.75)},
        "y": {0: (0.10, 0.55), 1: (0.15, 0.60), 2: (0.10, 0.50)},
    }

    print("=" * 72)
    print("稳健 IPS：行为概率为正闭区间，同一 context 内动作共享一个概率向量")
    print("=" * 72)
    print("观测：")
    for item in observations:
        print(f"  context={item.context!r}, action={item.action!r}, "
              f"reward={item.reward:g}")

    result = robust_ips_range(
        observations, target_policy, propensity_intervals)
    print(f"\n未截断 IPS 范围（n={result.n}）：")
    print(f"  minimum = {result.minimum:.12g}")
    print(f"  maximum = {result.maximum:.12g}")
    print("\n达到最小值的行为概率表：")
    print(" " + fmt_table(result.min_probabilities))
    print("达到最大值的行为概率表：")
    print(" " + fmt_table(result.max_probabilities))

    print("\n约束检查：")
    for check in result.constraint_checks:
        print(f"  context={check.context!r}: feasible={check.feasible}, "
              f"sum(lower)={check.lower_sum:g}, "
              f"sum(upper)={check.upper_sum:g}")

    print("\n说明：context='x' 的动作 0 有 +2/-2 两条反馈，其加权系数抵消；")
    print("同一 context 下多条记录仍复用同一概率表，不能逐条挑选概率。")

    print("\n" + "=" * 72)
    print("边界情况：区间盒与概率单纯形不相交，先拒绝")
    print("=" * 72)
    bad_observations = [RobustObservation("z", 0, 1.0)]
    bad_policy = {"z": {0: 0.5, 1: 0.5}}
    bad_intervals = {"z": {0: (0.80, 0.95), 1: (0.80, 0.95)}}
    try:
        robust_ips_range(bad_observations, bad_policy, bad_intervals)
    except InfeasibleIntervalError as exc:
        print(f"InfeasibleIntervalError: {exc}")

    print(f"\n实际计算耗时 {time.perf_counter() - start:.6f} 秒（无 sleep）。")


if __name__ == "__main__":
    main()
