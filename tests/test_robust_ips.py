"""稳健 IPS 范围测试：二维独立枚举/驻点参照、抵消与约束拒绝。"""

from __future__ import annotations

import math
import unittest

from delayed_ope import (
    InfeasibleIntervalError,
    InvalidIntervalError,
    InvalidLogError,
    InvalidPolicyError,
    RobustObservation,
    WeightOverflowError,
    robust_ips_range,
)


def independent_two_action_range(c0, c1, lo0, hi0, lo1, hi1):
    p0_values = [lo0, hi0, 1.0 - lo1, 1.0 - hi1]
    if c0 > 0.0 and c1 > 0.0:
        p0_values.append(math.sqrt(c0) / (math.sqrt(c0) + math.sqrt(c1)))
    elif c0 < 0.0 and c1 < 0.0:
        p0_values.append(math.sqrt(-c0) / (math.sqrt(-c0) + math.sqrt(-c1)))
    values = []
    for p0 in p0_values:
        p1 = 1.0 - p0
        if p0 >= 1e-12 and p1 >= 1e-12:
            if lo0 - 1e-10 <= p0 <= hi0 + 1e-10 and (
                    lo1 - 1e-10 <= p1 <= hi1 + 1e-10):
                values.append(c0 / p0 + c1 / p1)
    return min(values), max(values)


def obs(context, action, reward):
    return RobustObservation(context=context, action=action, reward=reward)


class RobustIPSTests(unittest.TestCase):
    def test_endpoints_match_two_action_reference(self):
        observations = [obs("x", 0, 2.0), obs("x", 1, -2.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        intervals = {"x": {0: (0.2, 0.7), 1: (0.1, 0.6)}}
        result = robust_ips_range(observations, pi, intervals)
        lo, hi = independent_two_action_range(1.0, -1.0, 0.2, 0.7, 0.1, 0.6)
        self.assertAlmostEqual(result.minimum, lo / 2, places=12)
        self.assertAlmostEqual(result.maximum, hi / 2, places=12)
        self.assert_probabilities(result, pi, intervals)

    def test_positive_interior_stationary_point(self):
        observations = [obs("x", 0, 4.0), obs("x", 1, 2.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        intervals = {"x": {0: (0.1, 0.9), 1: (0.1, 0.9)}}
        result = robust_ips_range(observations, pi, intervals)
        lo, hi = independent_two_action_range(2.0, 1.0, 0.1, 0.9, 0.1, 0.9)
        self.assertAlmostEqual(result.minimum, lo / 2, places=12)
        self.assertAlmostEqual(result.maximum, hi / 2, places=12)
        self.assertAlmostEqual(
            result.min_probabilities["x"][0],
            math.sqrt(2.0) / (math.sqrt(2.0) + 1.0), places=10)
        self.assert_probabilities(result, pi, intervals)

    def test_negative_interior_stationary_point(self):
        observations = [obs("x", 0, -4.0), obs("x", 1, -2.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        intervals = {"x": {0: (0.1, 0.9), 1: (0.1, 0.9)}}
        result = robust_ips_range(observations, pi, intervals)
        lo, hi = independent_two_action_range(-2.0, -1.0, 0.1, 0.9, 0.1, 0.9)
        self.assertAlmostEqual(result.minimum, lo / 2, places=12)
        self.assertAlmostEqual(result.maximum, hi / 2, places=12)
        self.assertAlmostEqual(
            result.max_probabilities["x"][0],
            math.sqrt(2.0) / (math.sqrt(2.0) + 1.0), places=10)
        self.assert_probabilities(result, pi, intervals)

    def test_same_action_positive_negative_feedback_cancels(self):
        observations = [obs("x", 0, 2.0), obs("x", 0, -2.0), obs("x", 1, 1.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        intervals = {"x": {0: (0.25, 0.7), 1: (0.3, 0.75)}}
        result = robust_ips_range(observations, pi, intervals)
        self.assertAlmostEqual(result.min_probabilities["x"][1], 0.75)
        self.assertAlmostEqual(result.max_probabilities["x"][1], 0.30)
        self.assertAlmostEqual(result.minimum, (0.5 / 0.75) / 3, places=12)
        self.assertAlmostEqual(result.maximum, (0.5 / 0.30) / 3, places=12)
        self.assert_probabilities(result, pi, intervals)

    def test_same_action_records_share_one_probability(self):
        observations = [obs("x", 0, 2.0), obs("x", 0, 3.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        intervals = {"x": {0: (0.2, 0.8), 1: (0.2, 0.8)}}
        result = robust_ips_range(observations, pi, intervals)
        for boundary, probs, expected in (
                ("min", result.min_probabilities, result.minimum),
                ("max", result.max_probabilities, result.maximum)):
            p0 = probs["x"][0]
            recomputed = (0.5 * 2.0 + 0.5 * 3.0) / p0 / result.n
            self.assertAlmostEqual(recomputed, expected, places=12)

    def test_singleton_intervals_match_point_estimate(self):
        observations = [obs("x", 0, 2.0), obs("x", 1, -1.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        intervals = {"x": {0: (0.4, 0.4), 1: (0.6, 0.6)}}
        result = robust_ips_range(observations, pi, intervals)
        point_ips = ((0.5 / 0.4) * 2.0 + (0.5 / 0.6) * -1.0) / 2
        self.assertAlmostEqual(result.minimum, point_ips, places=12)
        self.assertAlmostEqual(result.maximum, point_ips, places=12)

    def test_three_action_and_independent_contexts(self):
        observations = [
            obs("a", 0, 2.0), obs("a", 1, 1.0), obs("a", 2, 4.0),
            obs("b", 0, -1.0),
        ]
        pi = {"a": {0: 0.25, 1: 0.5, 2: 0.25}, "b": {0: 0.4, 1: 0.6}}
        intervals = {
            "a": {0: (0.05, 0.7), 1: (0.05, 0.7), 2: (0.05, 0.7)},
            "b": {0: (0.2, 0.8), 1: (0.2, 0.8)},
        }
        result = robust_ips_range(observations, pi, intervals)
        self.assertAlmostEqual(
            result.min_probabilities["a"][2] / result.min_probabilities["a"][0],
            math.sqrt(2.0), places=9)
        self.assert_probabilities(result, pi, intervals)
        check = {c.context: c for c in result.constraint_checks}
        self.assertTrue(all(c.feasible for c in result.constraint_checks))
        self.assertAlmostEqual(check["a"].lower_sum, 0.15)
        self.assertAlmostEqual(check["b"].upper_sum, 1.6)

    def test_infeasible_and_invalid_intervals(self):
        observations = [obs("x", 0, 1.0)]
        pi = {"x": {0: 0.5, 1: 0.5}}
        with self.assertRaises(InfeasibleIntervalError):
            robust_ips_range(observations, pi,
                             {"x": {0: (0.8, 0.9), 1: (0.8, 0.95)}})
        with self.assertRaises(InfeasibleIntervalError):
            robust_ips_range(observations, pi,
                             {"x": {0: (0.1, 0.2), 1: (0.1, 0.2)}})
        with self.assertRaises(InfeasibleIntervalError):
            robust_ips_range(observations, pi,
                             {"x": {0: (0.8, 0.2), 1: (0.2, 0.8)}})
        with self.assertRaises(InvalidIntervalError):
            robust_ips_range(observations, pi,
                             {"x": {0: (0.0, 0.2), 1: (0.8, 0.8)}})
        with self.assertRaises(InvalidIntervalError):
            robust_ips_range(observations, pi, {"x": {0: (0.5, 0.5)}})

    def test_validation_and_overflow(self):
        good_obs = [obs("x", 0, 1.0)]
        good_pi = {"x": {0: 0.5, 1: 0.5}}
        good_iv = {"x": {0: (0.2, 0.8), 1: (0.2, 0.8)}}
        with self.assertRaises(InvalidLogError):
            robust_ips_range([], good_pi, good_iv)
        with self.assertRaises(InvalidLogError):
            robust_ips_range([RobustObservation("x", 0, float("nan"))],
                             good_pi, good_iv)
        with self.assertRaises(InvalidPolicyError):
            robust_ips_range(good_obs, {"x": {0: 0.4, 1: 0.5}}, good_iv)
        with self.assertRaises(InvalidPolicyError):
            robust_ips_range(
                good_obs,
                {"x": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}},
                {"x": {0: (0.1, 0.4), 1: (0.1, 0.4),
                        2: (0.1, 0.4), 3: (0.1, 0.4)}})
        with self.assertRaises(InvalidIntervalError):
            robust_ips_range(
                good_obs, {"x": {0: 1.0, 1: 0.0}},
                {"x": {0: (1e-308, 1.0), 1: (0.0, 0.0)}})
        with self.assertRaises(WeightOverflowError):
            robust_ips_range(
                good_obs, {"x": {0: 1.0, 1: 0.0}},
                {"x": {0: (1e-308, 1.0), 1: (1e-300, 1.0)}})

    def assert_probabilities(self, result, pi, intervals):
        for table in (result.min_probabilities, result.max_probabilities):
            for ctx, row in table.items():
                for action, p in row.items():
                    lo, hi = intervals[ctx][action]
                    self.assertGreater(p, 0.0)
                    self.assertGreaterEqual(p, lo - 1e-10)
                    self.assertLessEqual(p, hi + 1e-10)
                    self.assertGreaterEqual(float(pi[ctx][action]), 0.0)
                self.assertAlmostEqual(sum(row.values()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
