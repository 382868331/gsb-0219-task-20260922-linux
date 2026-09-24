"""多步轨迹（<=5 步）TIS / PDIS 的必要测试。

期望值来源（均不调用 delayed_ope 的算法得到）：
* tests.mdp_reference —— 小 MDP 上按分布定义递归枚举的精确 V(pi)；
* 测试内按公式手算的标量求和（权重、ESS、TIS/PDIS 小例）。
被测核心只负责产生待核对的输出。
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from delayed_ope import (
    InvalidPolicyError,
    InvalidTrajectoryError,
    Trajectory,
    TrajectoryStep,
    WeightOverflowError,
    ZeroSupportError,
    evaluate_trajectories,
)
from delayed_ope.trajectory import MAX_TRAJECTORY_STEPS

from mdp_reference import ACTIONS, STATES, mdp_value

# ---- 固定小 MDP（与 mdp_reference 的世界一致） ---------------------------
P0 = {0: 0.5, 1: 0.5}
# 确定性转移：next = (x + a) % 2。
TRANSITIONS = {
    x: {a: {(x + a) % 2: 1.0} for a in ACTIONS} for x in STATES
}
# 行为策略（分母 4，保证普查计数为整数）。
MU = {0: {0: 0.75, 1: 0.25}, 1: {0: 0.25, 1: 0.75}}
# 目标策略（pi << mu 成立：mu 无零）。
PI = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
REWARDS = {0: {0: 1.0, 1: -0.5}, 1: {0: 0.25, 1: 2.0}}

PI2 = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.75, 1: 0.25}}
REWARDS2 = {0: {0: 0.0, 1: 1.5}, 1: {0: -1.0, 1: 0.5}}


def build_mdp_census(mu, horizon):
    """构造普查轨迹集：每条轨迹条数严格正比于 p0(x0) * prod_t mu(a_t|x_t)。

    p0 为 halves、mu 为 quarters，故 BASE = 2 * 4**horizon 时计数恰为整数。
    """
    base = 2 * (4 ** horizon)
    trajs = []
    for x0 in STATES:
        for actions in itertools.product(ACTIONS, repeat=horizon):
            steps = []
            x = x0
            prob = P0[x0]
            for a in actions:
                steps.append(TrajectoryStep(x, a, REWARDS[x][a], mu[x][a]))
                prob *= mu[x][a]
                (x2,) = (s for s, p in TRANSITIONS[x][a].items() if p > 0.0)
                x = x2
            count = base * prob
            self_count = int(round(count))
            assert abs(count - self_count) < 1e-9, "普查计数必须为整数"
            trajs.extend([Trajectory(tuple(steps))] * self_count)
    return trajs


class CensusIdentityTests(unittest.TestCase):
    """普查轨迹上 TIS/PDIS 与小 MDP 枚举真值的恒等对照。"""

    def _check_census(self, pi, horizon, case):
        trajs = build_mdp_census(MU, horizon)
        res = evaluate_trajectories(trajs, pi)
        expected = mdp_value(P0, TRANSITIONS, REWARDS, pi, horizon)
        self.assertEqual(res.n_trajectories, len(trajs))
        self.assertEqual(res.horizon, horizon)
        for name, got in (("TIS", res.tis), ("PDIS", res.pdis)):
            self.assertTrue(math.isfinite(got), f"{name} 非有限 case={case}")
            self.assertLessEqual(
                abs(got - expected),
                1e-11 * max(1.0, abs(expected)),
                f"{name} case={case}: got={got!r} expected={expected!r}")
        return res

    def test_census_identity_max_horizon(self):
        # 上限步数 H=5 的普查恒等（两套目标策略）。
        self._check_census(PI, MAX_TRAJECTORY_STEPS, "H5/pi")
        self._check_census(PI2, MAX_TRAJECTORY_STEPS, "H5/pi2")

    def test_census_identity_short_horizons(self):
        for horizon in (1, 2, 3):
            self._check_census(PI, horizon, f"H{horizon}")
            self._check_census(PI2, horizon, f"H{horizon}/pi2")

    def test_census_identity_alt_rewards(self):
        # 换回报表（全局替换 REWARDS 语义太重，这里直接构造第二套普查）。
        base = 2 * (4 ** 3)
        trajs = []
        for x0 in STATES:
            for actions in itertools.product(ACTIONS, repeat=3):
                steps = []
                x = x0
                prob = P0[x0]
                for a in actions:
                    steps.append(
                        TrajectoryStep(x, a, REWARDS2[x][a], MU[x][a]))
                    prob *= MU[x][a]
                    x = (x + a) % 2
                count = int(round(base * prob))
                trajs.extend([Trajectory(tuple(steps))] * count)
        res = evaluate_trajectories(trajs, PI2)
        expected = mdp_value(P0, TRANSITIONS, REWARDS2, PI2, 3)
        for name, got in (("TIS", res.tis), ("PDIS", res.pdis)):
            self.assertLessEqual(
                abs(got - expected), 1e-11 * max(1.0, abs(expected)),
                f"{name} alt-rewards: got={got!r} expected={expected!r}")


class HandComputedTests(unittest.TestCase):
    """两条轨迹的手工小例：权重、TIS/PDIS、诊断全部按公式手算。"""

    PI_X = {"x": {0: 0.5, 1: 0.5}}

    def _two_trajectories(self):
        t1 = Trajectory((
            TrajectoryStep("x", 0, 1.0, 0.5),    # w=1, rho=1
            TrajectoryStep("x", 1, 2.0, 0.25),   # w=2, rho=2
        ))
        t2 = Trajectory((
            TrajectoryStep("x", 1, 0.0, 0.5),    # w=1, rho=1
        ))
        return [t1, t2]

    def test_hand_weights_and_estimates(self):
        res = evaluate_trajectories(self._two_trajectories(), self.PI_X)
        self.assertEqual(res.n_trajectories, 2)
        self.assertEqual(res.horizon, 2)
        self.assertEqual(res.trajectory_weights, (2.0, 1.0))
        # TIS = (rho2*G1 + rho1*G2)/2 = (2*3 + 1*0)/2 = 3。
        self.assertAlmostEqual(res.tis, 3.0, places=12)
        # PDIS = ((1*1 + 2*2) + (1*0))/2 = 2.5。
        self.assertAlmostEqual(res.pdis, 2.5, places=12)

    def test_hand_diagnostics(self):
        res = evaluate_trajectories(self._two_trajectories(), self.PI_X)
        d = res.diagnostics
        # 末步权重 (2, 1)：ESS = 3^2/(4+1) = 1.8。
        self.assertAlmostEqual(d.ess, 1.8, places=12)
        self.assertEqual(d.max_weight, 2.0)
        self.assertEqual(d.n_zero_weight, 0)
        # 第 1 步权重 (1,1) -> ESS 2；第 2 步仅长轨迹 (2,) -> ESS 1。
        self.assertEqual(len(d.step_ess), 2)
        self.assertAlmostEqual(d.step_ess[0], 2.0, places=12)
        self.assertAlmostEqual(d.step_ess[1], 1.0, places=12)

    def test_variable_lengths_horizon_and_step_ess(self):
        # 长度 1..5 混合；w 全为 1（pi == mu），各步 ESS = 覆盖该步的条数。
        pi = {"x": {0: 1.0}}
        trajs = [
            Trajectory(tuple(
                TrajectoryStep("x", 0, 1.0, 1.0) for _ in range(k)))
            for k in range(1, 6)
        ]
        res = evaluate_trajectories(trajs, pi)
        self.assertEqual(res.horizon, 5)
        # 回报全 1、权重全 1：TIS = mean(长度) = 3；PDIS 同。
        self.assertAlmostEqual(res.tis, 3.0, places=12)
        self.assertAlmostEqual(res.pdis, 3.0, places=12)
        for t, expected_ess in enumerate((5.0, 4.0, 3.0, 2.0, 1.0)):
            self.assertAlmostEqual(
                res.diagnostics.step_ess[t], expected_ess, places=12)


class SupportAndWeightTests(unittest.TestCase):
    PI_X = {"x": {0: 0.5, 1: 0.5}}

    def test_zero_behavior_prob_rejected(self):
        trajs = [Trajectory((TrajectoryStep("x", 0, 1.0, 0.0),))]
        with self.assertRaises(ZeroSupportError):
            evaluate_trajectories(trajs, self.PI_X)

    def test_zero_behavior_prob_rejected_at_later_step(self):
        # 零支持出现在非首步同样拒绝（对全部步校验）。
        trajs = [Trajectory((
            TrajectoryStep("x", 0, 1.0, 0.5),
            TrajectoryStep("x", 1, 1.0, 0.0),
        ))]
        with self.assertRaises(ZeroSupportError):
            evaluate_trajectories(trajs, self.PI_X)

    def test_target_zero_on_zero_behavior_is_weight_zero(self):
        # pi==0 且 mu==0：权重取 0，合法；该轨迹贡献恒 0，诊断显式标出。
        pi = {"x": {0: 1.0, 1: 0.0}}
        trajs = [
            Trajectory((TrajectoryStep("x", 1, 5.0, 0.0),)),
            Trajectory((TrajectoryStep("x", 0, 2.0, 1.0),)),
        ]
        res = evaluate_trajectories(trajs, pi)
        self.assertEqual(res.trajectory_weights, (0.0, 1.0))
        self.assertEqual(res.diagnostics.n_zero_weight, 1)
        # TIS = (0*5 + 1*2)/2 = 1；PDIS 同。
        self.assertAlmostEqual(res.tis, 1.0, places=12)
        self.assertAlmostEqual(res.pdis, 1.0, places=12)

    def test_single_step_overflow_rejected(self):
        pi = {"x": {0: 1.0}}
        trajs = [Trajectory((TrajectoryStep("x", 0, 1.0, 1e-308),))]
        with self.assertRaises(WeightOverflowError):
            evaluate_trajectories(trajs, pi)

    def test_cumulative_overflow_rejected(self):
        # 单步 w=1e70 合法，5 步累乘 1e350 超过 MAX_WEIGHT。
        pi = {"x": {0: 1.0}}
        trajs = [Trajectory(tuple(
            TrajectoryStep("x", 0, 1.0, 1e-70)
            for _ in range(MAX_TRAJECTORY_STEPS)))]
        with self.assertRaises(WeightOverflowError):
            evaluate_trajectories(trajs, pi)

    def test_zero_prefix_short_circuits_later_huge_weight(self):
        # 首步 pi=0 使轨迹权重归零后，后续极端行为概率不再触发溢出：
        # 该轨迹在目标策略下概率为 0，贡献恒 0。
        pi = {"x": {0: 0.0, 1: 1.0}}
        trajs = [
            Trajectory((
                TrajectoryStep("x", 0, 1.0, 0.5),      # pi=0 -> rho=0
                TrajectoryStep("x", 1, 1.0, 1e-308),   # 不再计算 w
            )),
            Trajectory((TrajectoryStep("x", 1, 3.0, 1.0),)),
        ]
        res = evaluate_trajectories(trajs, pi)
        self.assertEqual(res.trajectory_weights, (0.0, 1.0))
        self.assertAlmostEqual(res.tis, 1.5, places=12)


class ValidationTests(unittest.TestCase):
    PI_X = {"x": {0: 0.5, 1: 0.5}}
    GOOD = [Trajectory((TrajectoryStep("x", 0, 1.0, 0.5),))]

    def test_container_and_length(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([], self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories("nope", self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([("x", 0, 1.0, 0.5)], self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([Trajectory(())], self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([Trajectory(tuple(
                TrajectoryStep("x", 0, 1.0, 0.5) for _ in range(6)))],
                self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories([Trajectory((("x", 0, 1.0, 0.5),))],
                                  self.PI_X)

    def test_reject_nan_inf_and_range(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidTrajectoryError):
                evaluate_trajectories(
                    [Trajectory((TrajectoryStep("x", 0, bad, 0.5),))],
                    self.PI_X)
            with self.assertRaises(InvalidTrajectoryError):
                evaluate_trajectories(
                    [Trajectory((TrajectoryStep("x", 0, 1.0, bad),))],
                    self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((TrajectoryStep("x", 0, 1.0, 1.5),))], self.PI_X)

    def test_reject_bool_and_unhashable(self):
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((TrajectoryStep("x", 0, True, 0.5),))], self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((TrajectoryStep("x", 0, 1.0, False),))],
                self.PI_X)
        with self.assertRaises(InvalidTrajectoryError):
            evaluate_trajectories(
                [Trajectory((TrajectoryStep(["x"], 0, 1.0, 0.5),))],
                self.PI_X)

    def test_policy_validation(self):
        with self.assertRaises(InvalidPolicyError):
            evaluate_trajectories(self.GOOD, {"y": {0: 1.0}})
        with self.assertRaises(InvalidPolicyError):
            evaluate_trajectories(self.GOOD, {"x": {1: 1.0}})
        with self.assertRaises(InvalidPolicyError):
            evaluate_trajectories(self.GOOD, {"x": {0: 0.5, 1: 0.4}})
        with self.assertRaises(InvalidPolicyError):
            evaluate_trajectories(self.GOOD, {"x": {0: 1.5, 1: -0.5}})

    def test_policy_not_mutated(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        evaluate_trajectories(self.GOOD, pi)
        self.assertEqual(pi, {"x": {0: 0.5, 1: 0.5}})


class StochasticConsistencyTests(unittest.TestCase):
    """随机（非普查）轨迹上的统计一致性：仅宽松统计带，非边界口径。"""

    def test_pdis_tis_approach_enumerated_value(self):
        rng = random.Random(20260924)
        horizon = 2
        n = 60000
        trajs = []
        for _ in range(n):
            x = 0 if rng.random() < P0[0] else 1
            steps = []
            for _ in range(horizon):
                u = rng.random()
                a = 0 if u < MU[x][0] else 1
                steps.append(TrajectoryStep(x, a, REWARDS[x][a], MU[x][a]))
                x = (x + a) % 2
            trajs.append(Trajectory(tuple(steps)))
        res = evaluate_trajectories(trajs, PI)
        expected = mdp_value(P0, TRANSITIONS, REWARDS, PI, horizon)
        self.assertLessEqual(abs(res.pdis - expected), 0.05)
        self.assertLessEqual(abs(res.tis - expected), 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
