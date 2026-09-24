"""轨迹 OPE 必要测试：小 MDP 枚举真值 + 手算例 + 边界与输入校验。

期望值来源（均不调用 delayed_ope 的算法得到）：
* tests.mdp_reference —— 在确定性小 MDP 上按分布定义直接枚举，并用反向
  策略评估递推交叉核对（两条独立算法必须一致）；
* 测试内按逐步累乘定义手算的标量。
被测核心只负责产生待核对的输出。
"""

from __future__ import annotations

import math
import unittest

from delayed_ope import (
    InvalidTrajectoryError,
    Step,
    Trajectory,
    WeightOverflowError,
    ZeroSupportError,
    evaluate_trajectories,
)
from delayed_ope.core import MAX_HORIZON, MAX_WEIGHT

from mdp_reference import (
    ACTIONS,
    BRANCH,
    STATES,
    discounted_return,
    enumerate_leaves,
    leaf_step_probs,
    policy_value_backup,
    reward,
    true_value_under_pi,
)

# mu/pi 均为十分数。配置 A：处处正支持；配置 B：含 pi=0 的孔（仍 pi<<mu）。
MU_A = {
    0: {0: 5, 1: 5},
    1: {0: 8, 1: 2},
    2: {0: 3, 1: 7},
}
PI_A = {
    0: {0: 5, 1: 5},
    1: {0: 2, 1: 8},
    2: {0: 7, 1: 3},
}
MU_B = {
    0: {0: 6, 1: 4},
    1: {0: 8, 1: 2},
    2: {0: 3, 1: 7},
}
PI_B = {
    0: {0: 10, 1: 0},   # pi(a1|s0)=0：含该动作的序列权重为 0 但合法
    1: {0: 10, 1: 0},
    2: {0: 0, 1: 10},
}


def build_census(mu_counts, pi_counts, horizon, discount):
    """构造 mu 普查：每个不同动作序列出现 mu_mult 次（严格正比于 P_mu）。"""
    leaves = enumerate_leaves(mu_counts, pi_counts, horizon)
    trajs = []
    for leaf in leaves:
        if leaf.mu_mult == 0:
            continue  # mu 零支持的序列在普查中不出现
        probs = leaf_step_probs(leaf, mu_counts, pi_counts)
        steps = tuple(
            Step(leaf.states[t], leaf.actions[t],
                 float(reward(leaf.states[t], leaf.actions[t])),
                 mu_p, pi_p)
            for t, (mu_p, pi_p) in enumerate(probs))
        trajs.extend([Trajectory(steps, discount)] * leaf.mu_mult)
    return trajs


class MDPCensusIdentityTests(unittest.TestCase):
    """普查轨迹上 IS/SNIS/PD-IS/PD-SNIS 恒等于枚举真值。"""

    def test_two_independent_truth_algorithms_agree(self):
        # 先核对两条独立参考算法本身一致（枚举 vs 反向递推）。
        for L in range(1, MAX_HORIZON + 1):
            for gamma in (1.0, 0.5, 0.0):
                v1 = true_value_under_pi(PI_A, L, gamma)
                v2 = policy_value_backup(PI_A, L, gamma)
                self.assertLessEqual(
                    abs(v1 - v2), 1e-12 * max(1.0, abs(v1)),
                    f"L={L} gamma={gamma}: 枚举={v1} 递推={v2}")

    def test_census_identity_until_horizon_five(self):
        configs = [(MU_A, PI_A), (MU_B, PI_B)]
        case_no = 0
        for mu_c, pi_c in configs:
            for L in range(1, MAX_HORIZON + 1):
                for gamma in (1.0, 0.5, 0.0):
                    case_no += 1
                    trajs = build_census(mu_c, pi_c, L, gamma)
                    self.assertEqual(len(trajs), BRANCH ** L)
                    res = evaluate_trajectories(trajs)
                    expected = true_value_under_pi(pi_c, L, gamma)
                    # 独立第二真值。
                    expected2 = policy_value_backup(pi_c, L, gamma)
                    self.assertLessEqual(
                        abs(expected - expected2),
                        1e-12 * max(1.0, abs(expected)))
                    e = res.estimate
                    for name, got in (
                        ("IS", e.is_),
                        ("SNIS", e.snis),
                        ("PD-IS", e.pd_is),
                        ("PD-SNIS", e.pd_snis),
                    ):
                        self.assertIsNotNone(
                            got, f"{name} 不应为 None case={case_no}")
                        self.assertTrue(
                            math.isfinite(got),
                            f"{name} 非有限 case={case_no}")
                        self.assertLessEqual(
                            abs(got - expected),
                            1e-9 * max(1.0, abs(expected)),
                            f"{name} case={case_no} L={L} gamma={gamma}: "
                            f"got={got!r} expected={expected!r}")
                    self.assertEqual(res.horizon, L)
                    self.assertEqual(res.weights.n, BRANCH ** L)
        self.assertEqual(case_no, 2 * MAX_HORIZON * 3)

    def test_census_identity_mixed_lengths(self):
        # 变长轨迹：L=1 普查整体复制 3 份、L=2 普查复制 2 份，真值按
        # 轨迹数加权。检验 Gamma(L) 归一的 PD-SNIS 在变长下仍恒等。
        gamma = 1.0
        c1 = build_census(MU_A, PI_A, 1, gamma) * 3
        c2 = build_census(MU_A, PI_A, 2, gamma) * 2
        trajs = c1 + c2
        n1, n2 = len(c1), len(c2)
        v1 = true_value_under_pi(PI_A, 1, gamma)
        v2 = true_value_under_pi(PI_A, 2, gamma)
        expected = (n1 * v1 + n2 * v2) / (n1 + n2)
        res = evaluate_trajectories(trajs)
        self.assertEqual(res.horizon, 2)
        e = res.estimate
        for name, got in (
            ("IS", e.is_), ("SNIS", e.snis),
            ("PD-IS", e.pd_is), ("PD-SNIS", e.pd_snis),
        ):
            self.assertLessEqual(
                abs(got - expected), 1e-9 * max(1.0, abs(expected)),
                f"{name}: got={got!r} expected={expected!r}")


class HandComputedTests(unittest.TestCase):
    """两条 2 步轨迹的逐步手算（gamma=1 与 gamma=0.5 两组）。"""

    def _trajs(self, gamma):
        # A: 因子 1.0 然后 0.5；奖励 1, 2。
        a = Trajectory((
            Step("s0", "x", 1.0, 0.5, 0.5),
            Step("s1", "y", 2.0, 0.8, 0.4),
        ), gamma)
        # B: 因子 1.0 然后 3.0；奖励 0, 4。
        b = Trajectory((
            Step("s0", "z", 0.0, 0.5, 0.5),
            Step("s2", "x", 4.0, 0.2, 0.6),
        ), gamma)
        return [a, b]

    def test_gamma_one(self):
        res = evaluate_trajectories(self._trajs(1.0))
        e, w = res.estimate, res.weights
        # W_A=0.5 G_A=3；W_B=3 G_B=4。
        self.assertAlmostEqual(e.is_, (0.5 * 3 + 3 * 4) / 2, places=12)
        self.assertAlmostEqual(e.snis, 13.5 / 3.5, places=12)
        # PD 分子：A=1*1+0.5*2=2；B=1*0+3*4=12。
        self.assertAlmostEqual(e.pd_is, 14 / 2, places=12)
        # PD 分母（Gamma(2)=2）：A=.75 B=2。
        self.assertAlmostEqual(e.pd_snis, 14 / 2.75, places=12)
        # 终权重诊断。
        self.assertEqual(w.min, 0.5)
        self.assertAlmostEqual(w.max, 3.0, places=12)
        self.assertAlmostEqual(w.mean, 1.75, places=12)
        self.assertAlmostEqual(w.ess, 3.5 ** 2 / (0.25 + 9), places=9)
        self.assertAlmostEqual(w.n, 2)
        # 前缀权重：1, .5, 1, 3。
        self.assertEqual(w.n_decision, 4)
        self.assertAlmostEqual(
            w.decision_ess, 5.5 ** 2 / (1 + 0.25 + 1 + 9), places=9)
        self.assertLessEqual(w.ess, 2.0 + 1e-9)
        self.assertGreaterEqual(w.ess, 1.0 - 1e-9)

    def test_gamma_half(self):
        res = evaluate_trajectories(self._trajs(0.5))
        e = res.estimate
        # G_A=1+.5*2=2；G_B=0+.5*4=2。
        self.assertAlmostEqual(e.is_, (0.5 * 2 + 3 * 2) / 2, places=12)
        self.assertAlmostEqual(e.snis, 7 / 3.5, places=12)
        # PD 分子：A=1+.5*.5*2=1.5；B=0+.5*3*4=6。
        self.assertAlmostEqual(e.pd_is, 7.5 / 2, places=12)
        # Gamma(2)=1.5：A 权重和 1+.5*.5=1.25；B 权重和 1+.5*3=2.5。
        self.assertAlmostEqual(e.pd_snis, 7.5 / (1.25 / 1.5 + 2.5 / 1.5),
                               places=12)

    def test_single_step_matches_plain_weights(self):
        # L=1 时轨迹量与逐决策量必须相同；SNIS=加权均值定义。
        trajs = [
            Trajectory((Step(0, 0, 1.0, 0.5, 0.9),), 1.0),   # W=1.8
            Trajectory((Step(0, 1, 0.0, 0.5, 0.1),), 1.0),   # W=0.2
        ]
        res = evaluate_trajectories(trajs)
        e = res.estimate
        self.assertAlmostEqual(e.is_, e.pd_is, places=15)
        self.assertAlmostEqual(e.snis, e.pd_snis, places=15)
        self.assertAlmostEqual(e.is_, 1.8 / 2, places=12)
        self.assertAlmostEqual(e.snis, 1.8 / 2.0, places=12)
        self.assertEqual(res.weights.n_decision, 2)
        self.assertAlmostEqual(res.weights.ess,
                               res.weights.decision_ess, places=12)


class ZeroSupportAndOverflowTests(unittest.TestCase):
    def test_zero_support_at_early_step_rejected(self):
        trajs = [Trajectory((
            Step(0, 1, 1.0, 0.0, 0.5),       # mu=0, pi>0
            Step(1, 0, 1.0, 0.5, 0.5),
        ), 1.0)]
        with self.assertRaises(ZeroSupportError):
            evaluate_trajectories(trajs)

    def test_zero_support_at_late_step_rejected(self):
        # 即使前面的因子已使累乘为 0（pi=0 合法步），后面的结构性
        # 不相容仍必须拒绝，而不是被 0 掩盖。
        trajs = [Trajectory((
            Step(0, 1, 1.0, 0.5, 0.0),       # pi=0：因子 0，合法
            Step(2, 1, 1.0, 0.0, 0.5),       # mu=0 pi>0：拒绝
        ), 1.0)]
        with self.assertRaises(ZeroSupportError):
            evaluate_trajectories(trajs)

    def test_zero_zero_factor_is_legal_zero_weight(self):
        # pi=0 且 mu=0：因子取 0；全部终权重为 0 时 SNIS=None。
        trajs = [
            Trajectory((Step(0, 1, 5.0, 0.0, 0.0),), 1.0),
            Trajectory((
                Step(0, 1, 2.0, 0.5, 0.0),     # pi=0, mu>0：因子 0
                Step(2, 0, 1.0, 0.0, 0.0),     # 0*0：合法零
            ), 1.0),
        ]
        res = evaluate_trajectories(trajs)
        self.assertEqual(res.estimate.is_, 0.0)
        self.assertEqual(res.estimate.pd_is, 0.0)
        self.assertIsNone(res.estimate.snis)
        self.assertIsNone(res.estimate.pd_snis)
        self.assertEqual(res.weights.ess, 0.0)
        self.assertEqual(res.weights.decision_ess, 0.0)
        self.assertEqual(res.weights.max, 0.0)
        self.assertEqual(res.weights.min, 0.0)
        self.assertEqual(res.weights.mean, 0.0)

    def test_accumulated_overflow_rejected(self):
        # 每步因子 1e200：第二步前缀 1e400 > MAX_WEIGHT。
        trajs = [Trajectory((
            Step(0, 0, 1.0, 1e-200, 1.0),
            Step(1, 0, 1.0, 1e-200, 1.0),
        ), 1.0)]
        with self.assertRaises(WeightOverflowError):
            evaluate_trajectories(trajs)

    def test_prefix_overflow_even_if_final_cancelled(self):
        # 终权重不溢出但前缀溢出也要拒绝（PD 估计使用前缀权重）：
        # 因子 1e200 后再来一个 0（pi=0），终权重=0，但 rho_1=1e200 合法，
        # 再加一步 1e200 -> 前缀 1e400 时必须拒绝。
        trajs = [Trajectory((
            Step(0, 0, 1.0, 1e-200, 1.0),    # rho=1e200
            Step(1, 0, 1.0, 0.5, 0.0),       # rho=0
            Step(2, 0, 1.0, 1e-200, 1.0),    # rho=0，无溢出
        ), 1.0)]
        # 该例不溢出（0 吸收）；改为第三步在归零前溢出：
        trajs2 = [Trajectory((
            Step(0, 0, 1.0, 1e-200, 1.0),    # 1e200
            Step(1, 0, 1.0, 1e-200, 1.0),    # 1e400 -> 溢出
            Step(2, 0, 1.0, 0.5, 0.0),
        ), 1.0)]
        # 第一组合法且终/前缀诊断有限。
        res = evaluate_trajectories(trajs)
        self.assertTrue(math.isfinite(res.estimate.pd_is))
        with self.assertRaises(WeightOverflowError):
            evaluate_trajectories(trajs2)

    def test_boundary_weight_at_max_allowed(self):
        # 单步权重 1/1e-300 ≈ 1e300，合法（常规 binary64 舍入）。
        trajs = [Trajectory((Step(0, 0, 1.0, 1e-300, 1.0),), 1.0)]
        res = evaluate_trajectories(trajs)
        self.assertLessEqual(res.weights.max, MAX_WEIGHT)
        self.assertLessEqual(abs(res.estimate.is_ - 1e300) / 1e300, 1e-14)


class TrajectoryValidationTests(unittest.TestCase):
    GOOD = (Step(0, 0, 1.0, 0.5, 0.5),)

    def test_empty_batch(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([])

    def test_not_a_sequence(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(Trajectory(self.GOOD))

    def test_empty_and_overlong_trajectory(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([Trajectory(())])
        long_steps = (Step(t, 0, 1.0, 0.5, 0.5)
                      for t in range(MAX_HORIZON + 1))
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([Trajectory(tuple(long_steps))])

    def test_wrong_element_types(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([("x", 0, 1.0, 0.5, 0.5)])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory([Step(0, 0, 1.0, 0.5, 0.5)])])  # list 非 tuple
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((("x", 0, 1.0, 0.5, 0.5),))])

    def test_non_finite_and_range(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, float("nan"), 0.5, 0.5),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, 1.0, 1.01, 0.5),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, 1.0, 0.5, -0.2),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, 1.0, 0.5, float("inf")),))])

    def test_bool_rejected_in_numeric_positions(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, True, 0.5, 0.5),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, 1.0, False, 0.5),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, 0, 1.0, 0.5, True),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory(self.GOOD, discount=True)])

    def test_discount_bounds_and_consistency(self):
        for bad in (1.01, -0.1, float("inf"), float("nan")):
            with self.assertRaises(InvalidTrajectoryError):
                evaluate_trajectories([Trajectory(self.GOOD, bad)])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([
                Trajectory(self.GOOD, 1.0),
                Trajectory((Step(0, 0, 1.0, 0.5, 0.5),), 0.9),
            ])

    def test_unhashable_state_action(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step([0], 0, 1.0, 0.5, 0.5),))])
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((Step(0, {0}, 1.0, 0.5, 0.5),))])

    def test_failure_leaves_no_partial_state(self):
        # 纯函数无状态：失败调用不影响后续合法调用，输入也不被修改。
        good = [Trajectory((
            Step(0, 0, 1.0, 0.5, 0.5),
            Step(1, 1, 2.0, 0.5, 0.5),
        ), 1.0)]
        with self.assertRaises(ZeroSupportError):
            evaluate_trajectories(good + [Trajectory((
                Step(0, 1, 1.0, 0.0, 0.5),
            ), 1.0)])
        res = evaluate_trajectories(good)
        self.assertEqual(res.n_trajectories, 1)
        self.assertEqual(len(good[0].steps), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
