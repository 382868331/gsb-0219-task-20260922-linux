"""本轮迭代演示：正闭区间行为概率下的稳健 IPS 最小/最大值。

运行：python demo_iteration.py

本脚本只使用标准库和仓库内固定输入；不访问网络、不 sleep、不预录结果。
"""

from __future__ import annotations

import time

from delayed_ope import (
    InfeasibleProbabilityIntervalsError,
    RobustLogEvent,
    robust_ips,
)


TARGET_POLICY = {
    # 二维：正系数情形会在可行域内部出现驻点。
    "two": {"a": 0.5, "b": 0.5},
    # 二维：同一动作 a 的 +10/-10 反馈先在奖励和中抵消。
    "cancel": {"a": 0.4, "b": 0.6},
    # 三维：三个动作共享一个和为 1 的行为概率向量。
    "three": {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0},
}

BEHAVIOR_INTERVALS = {
    "two": {"a": (0.20, 0.80), "b": (0.20, 0.80)},
    "cancel": {"a": (0.25, 0.75), "b": (0.25, 0.75)},
    "three": {
        "a": (0.20, 0.50),
        "b": (0.20, 0.50),
        "c": (0.20, 0.50),
    },
}


def build_events() -> list[RobustLogEvent]:
    return [
        RobustLogEvent("two", "a", 4.0, 0.0, 1.0),
        RobustLogEvent("two", "b", 2.0, 0.0, 1.0),
        RobustLogEvent("cancel", "a", 10.0, 0.0, 1.0),
        RobustLogEvent("cancel", "a", -10.0, 0.0, 1.0),
        RobustLogEvent("cancel", "b", 2.0, 0.0, 1.0),
        RobustLogEvent("three", "a", 3.0, 0.0, 1.0),
        RobustLogEvent("three", "b", 3.0, 0.0, 1.0),
        RobustLogEvent("three", "c", 3.0, 0.0, 1.0),
    ]


def print_bound(name: str, bound) -> None:
    print(f"{name}: IPS={bound.ips:.12g}")
    for context, row in bound.behavior_policy.items():
        formatted = ", ".join(f"{action}={probability:.12g}"
                              for action, probability in row.items())
        print(f"  μ[{context}] = {{{formatted}}}")
    for check in bound.constraint_checks:
        print(
            "  check "
            f"{check.context}: feasible={check.feasible_intervals}, "
            f"sum={check.probability_sum:.12g}, "
            f"Σlower={check.lower_sum:.12g}, "
            f"Σupper={check.upper_sum:.12g}, "
            f"max_violation={check.max_interval_violation:.3g}"
        )


def main() -> None:
    started = time.perf_counter()
    events = build_events()
    result = robust_ips(
        events=events,
        target_policy=TARGET_POLICY,
        behavior_intervals=BEHAVIOR_INTERVALS,
        cutoff=2.0,
    )

    print("=" * 72)
    print("稳健 IPS：行为概率为正闭区间，同一上下文共享归一化变量")
    print("=" * 72)
    print(result.summary())
    print("-" * 72)
    print_bound("最小值边界", result.minimum)
    print("-" * 72)
    print_bound("最大值边界", result.maximum)
    print("-" * 72)
    print("two 上下文系数 c=(2,1)，二维最小值由内部驻点达到；")
    print("cancel 上下文同一动作 a 的加权正负反馈抵消，只剩 b 的正项。")
    print("three 上下文最小值是均匀内点，最大值在区间箱与单纯形的交点。")
    print("本范围只包含给定倾向区间，不包含奖励模型、目标策略或其他误差。")

    bad_intervals = {
        context: dict(row) for context, row in BEHAVIOR_INTERVALS.items()
    }
    bad_intervals["two"] = {"a": (0.70, 0.90), "b": (0.70, 0.90)}
    print("=" * 72)
    print("边界情况：Σlower=1.40 > 1，区间不可满足，先拒绝、不优化")
    print("=" * 72)
    try:
        robust_ips(events, TARGET_POLICY, bad_intervals, cutoff=2.0)
    except InfeasibleProbabilityIntervalsError as exc:
        print(f"InfeasibleProbabilityIntervalsError: {exc}")

    print("-" * 72)
    print(f"实际计算耗时 {time.perf_counter() - started:.6f} 秒（无 sleep）。")


if __name__ == "__main__":
    main()
