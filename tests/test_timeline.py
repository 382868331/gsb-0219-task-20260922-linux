"""TimelineEvaluator（按观察时间推进的增量接口）测试。

参照值来源（均不由被测增量逻辑生成）：
* 批量入口 evaluate 在同一截止时刻的结果 —— 契约要求逐位相同；
* 测试内按公式手算的标量求和（独立参考）。
"""

from __future__ import annotations

import math
import unittest

from delayed_ope import (
    CutoffRegressionError,
    DuplicateEventIdError,
    InvalidLogError,
    InvalidQHatError,
    LogEvent,
    NoMatureSamplesError,
    TimelineEvaluator,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
)

PI = {"x": {0: 0.6, 1: 0.4}, "y": {0: 0.25, 1: 0.75}}
Q = {("x", 0): 0.3, ("x", 1): -0.2, ("y", 0): 1.1, ("y", 1): 0.0}


def make_mixed_log():
    """决策顺序与观察顺序相反 + 同刻成熟组 + 跨 context 的混合日志。

    下标即决策顺序；observed_time 刻意让晚决策的事件先成熟。
    """
    return [
        # idx, ctx, act, reward, p_log, t_d, t_o
        LogEvent("x", 0, 1.0, 0.5, 0.0, 9.0, event_id="e0"),
        LogEvent("x", 1, 2.0, 0.5, 1.0, 2.0, event_id="e1"),
        LogEvent("y", 0, -1.0, 0.5, 2.0, 5.0, event_id="e2"),
        LogEvent("y", 1, 0.5, 0.5, 3.0, 5.0, event_id="e3"),  # 与 e2 同刻
        LogEvent("x", 0, 3.0, 0.5, 4.0, 5.0, event_id="e4"),  # 同刻组第三个
        LogEvent("x", 1, -2.0, 0.5, 5.0, 7.5, event_id="e5"),
    ]


def assert_same_result(tc, got, ref):
    tc.assertEqual(got.cutoff, ref.cutoff)
    tc.assertEqual(
        (got.n_total, got.n_mature, got.n_immature),
        (ref.n_total, ref.n_mature, ref.n_immature))
    tc.assertEqual(got.mature_indices, ref.mature_indices)
    for group in ("unclipped", "clipped"):
        g, r = getattr(got, group), getattr(ref, group)
        tc.assertEqual(g is None, r is None)
        if g is not None:
            for f in ("ips", "snips", "dr", "ess", "n"):
                tc.assertEqual(
                    getattr(g, f), getattr(r, f),
                    f"{group}.{f} 与批量入口不一致")


class BatchEquivalenceTests(unittest.TestCase):
    """每个事件边界独立筛选日志调批量 evaluate 作参照，要求完全相同。"""

    def test_equivalence_at_every_event_boundary(self):
        events = make_mixed_log()
        boundaries = sorted({e.observed_time for e in events})
        cutoffs = [boundaries[0] - 0.5]
        for b in boundaries:
            cutoffs += [b - 1e-9, b, b + 0.25]
        tl = TimelineEvaluator(events, PI, q_hat=Q, clip_value=2.0)
        seen_mature = None
        for c in cutoffs:
            tl.advance(c)
            try:
                ref = evaluate(events, PI, c, q_hat=Q, clip_value=2.0)
            except NoMatureSamplesError:
                with self.assertRaises(NoMatureSamplesError):
                    tl.query()
                continue
            got = tl.query()
            assert_same_result(self, got, ref)
            seen_mature = got.n_mature
        self.assertEqual(seen_mature, len(events))

    def test_equivalence_without_q_and_clip(self):
        events = make_mixed_log()
        tl = TimelineEvaluator(events, PI)
        for c in (2.0, 5.0, 9.0):
            tl.advance(c)
            assert_same_result(self, tl.query(), evaluate(events, PI, c))

    def test_frozen_snapshot_immune_to_caller_mutation(self):
        events = make_mixed_log()
        pi = {x: dict(row) for x, row in PI.items()}
        q = dict(Q)
        tl = TimelineEvaluator(events, pi, q_hat=q)
        tl.advance(5.0)
        before = tl.query()
        pi["x"][0] = 0.1
        pi["x"][1] = 0.9
        q[("x", 0)] = 99.0
        tl.advance(9.0)
        after = tl.query()
        # 冻结后不受调用方改动影响：与用原始配置重算的批量结果一致。
        assert_same_result(self, after, evaluate(events, PI, 9.0, q_hat=Q))
        self.assertEqual(before.n_mature, 4)


class GroupAtomicityTests(unittest.TestCase):
    def test_same_observed_time_enters_as_whole_group(self):
        events = make_mixed_log()
        tl = TimelineEvaluator(events, PI)
        tl.advance(5.0 - 1e-9)
        self.assertEqual(tl.query().mature_indices, (1,))
        tl.advance(5.0)
        # e2/e3/e4 同刻整组出现，不存在只含其中一部分的状态。
        self.assertEqual(tl.query().mature_indices, (1, 2, 3, 4))

    def test_failed_advance_leaves_no_partial_change(self):
        # 后成熟的组内含溢出权重：advance 抛错，此前状态原样保留。
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id="a"),
            LogEvent("x", 0, 1.0, 1e-308, 0.0, 2.0, event_id="b"),
            LogEvent("x", 1, 1.0, 0.5, 0.0, 2.0, event_id="c"),
        ]
        tl = TimelineEvaluator(events, PI)
        tl.advance(1.0)
        ok = tl.query()
        with self.assertRaises(WeightOverflowError):
            tl.advance(2.0)
        self.assertEqual(tl.cutoff, 1.0)
        assert_same_result(self, tl.query(), ok)

    def test_failed_advance_qhat_error_atomic(self):
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id="a"),
            LogEvent("y", 0, 1.0, 0.5, 0.0, 2.0, event_id="b"),
        ]
        bad_q = {("x", 0): 0.3, ("x", 1): -0.2}  # 缺 y 行
        tl = TimelineEvaluator(events, PI, q_hat=bad_q)
        tl.advance(1.0)
        ok = tl.query()
        with self.assertRaises(InvalidQHatError):
            tl.advance(2.0)
        assert_same_result(self, tl.query(), ok)


class MonotonicityTests(unittest.TestCase):
    def test_regression_rejected_and_state_kept(self):
        events = make_mixed_log()
        tl = TimelineEvaluator(events, PI)
        tl.advance(5.0)
        ok = tl.query()
        with self.assertRaises(CutoffRegressionError):
            tl.advance(4.999)
        assert_same_result(self, tl.query(), ok)

    def test_repeated_advance_idempotent(self):
        events = make_mixed_log()
        tl = TimelineEvaluator(events, PI, q_hat=Q)
        tl.advance(5.0)
        first = tl.query()
        tl.advance(5.0)
        tl.advance(5.0)
        assert_same_result(self, tl.query(), first)

    def test_cutoff_validation(self):
        tl = TimelineEvaluator(make_mixed_log(), PI)
        with self.assertRaises(InvalidLogError):
            tl.advance(float("nan"))
        with self.assertRaises(InvalidLogError):
            tl.advance(True)


class EventIdTests(unittest.TestCase):
    def test_duplicate_id_rejected(self):
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id="dup"),
            LogEvent("x", 1, 2.0, 0.5, 0.0, 1.0, event_id="dup"),
        ]
        with self.assertRaises(DuplicateEventIdError):
            TimelineEvaluator(events, PI)

    def test_missing_or_unhashable_id_rejected(self):
        with self.assertRaises(DuplicateEventIdError):
            TimelineEvaluator(
                [LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0)], PI)  # event_id=None
        with self.assertRaises(DuplicateEventIdError):
            TimelineEvaluator(
                [LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id=["x"])], PI)

    def test_duplicate_id_is_invalid_log_subclass(self):
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id=1),
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id=1),
        ]
        with self.assertRaises(InvalidLogError):
            TimelineEvaluator(events, PI)


class DegenerateStateTests(unittest.TestCase):
    def test_query_before_any_mature_raises(self):
        events = make_mixed_log()
        tl = TimelineEvaluator(events, PI)
        with self.assertRaises(NoMatureSamplesError):
            tl.query()
        tl.advance(1.0)  # 全部 observed_time > 1
        with self.assertRaises(NoMatureSamplesError):
            tl.query()

    def test_target_zero_probability_explicit_state(self):
        # 目标对记录动作赋零概率：w=0，IPS=0、SNIPS=None、ESS=0，与批量一致。
        pi = {"x": {0: 1.0, 1: 0.0}}
        events = [
            LogEvent("x", 1, 5.0, 0.0, 0.0, 1.0, event_id="z"),
            LogEvent("x", 1, 7.0, 0.5, 0.0, 2.0, event_id="w"),
        ]
        tl = TimelineEvaluator(events, pi)
        tl.advance(2.0)
        got = tl.query()
        ref = evaluate(events, pi, 2.0)
        assert_same_result(self, got, ref)
        self.assertEqual(got.unclipped.ips, 0.0)
        self.assertIsNone(got.unclipped.snips)
        self.assertEqual(got.unclipped.ess, 0.0)

    def test_zero_support_rejected_at_load_even_if_immature(self):
        events = [LogEvent("x", 1, 1.0, 0.0, 0.0, 99.0, event_id="z")]
        with self.assertRaises(ZeroSupportError):
            TimelineEvaluator(events, PI)


class ClipConfigTests(unittest.TestCase):
    def test_clip_fixed_at_creation_and_matches_batch(self):
        events = make_mixed_log()
        tl = TimelineEvaluator(events, PI, q_hat=Q, clip_value=1.5)
        tl.advance(9.0)
        got = tl.query()
        self.assertEqual(got.clip_value, 1.5)
        assert_same_result(
            self, got, evaluate(events, PI, 9.0, q_hat=Q, clip_value=1.5))
        self.assertIsNotNone(got.clipped)

    def test_no_clip_by_default(self):
        tl = TimelineEvaluator(make_mixed_log(), PI)
        tl.advance(9.0)
        self.assertIsNone(tl.query().clipped)


class IndependentFormulaTests(unittest.TestCase):
    """手算标量参照：不经过批量入口，直接按定义公式核对增量查询。"""

    def test_manual_formula_at_two_cutoffs(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        events = [
            LogEvent("x", 0, 1.0, 0.25, 0.0, 1.0, event_id="p"),  # w=2
            LogEvent("x", 1, 3.0, 0.75, 0.0, 1.0, event_id="q"),  # w=2/3
            LogEvent("x", 0, 2.0, 0.5, 0.0, 5.0, event_id="r"),   # w=1
        ]
        tl = TimelineEvaluator(events, pi)
        tl.advance(1.0)
        got = tl.query()
        # 手算：IPS=(2*1 + (2/3)*3)/2 = 2；SNIPS=4/(8/3)=1.5；
        # ESS=(8/3)^2/(4+4/9)=64/40=1.6。
        self.assertAlmostEqual(got.unclipped.ips, 2.0, places=12)
        self.assertAlmostEqual(got.unclipped.snips, 1.5, places=12)
        self.assertAlmostEqual(got.unclipped.ess, 1.6, places=12)
        tl.advance(5.0)
        got = tl.query()
        # 手算：IPS=(2 + 2 + 2)/3 = 2；SNIPS=6/(11/3)=18/11。
        self.assertAlmostEqual(got.unclipped.ips, 2.0, places=12)
        self.assertAlmostEqual(got.unclipped.snips, 18.0 / 11.0, places=12)
        self.assertEqual(got.n_mature, 3)

    def test_manual_dr_formula(self):
        pi = {"x": {0: 0.75, 1: 0.25}}
        q = {("x", 0): 0.4, ("x", 1): 0.8}
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, event_id="p"),  # w=1.5
            LogEvent("x", 1, 0.0, 0.5, 0.0, 3.0, event_id="q"),  # w=0.5
        ]
        tl = TimelineEvaluator(events, pi, q_hat=q)
        tl.advance(3.0)
        m_pi = 0.75 * 0.4 + 0.25 * 0.8
        manual = ((m_pi + 1.5 * 0.6) + (m_pi + 0.5 * (-0.8))) / 2
        self.assertAlmostEqual(tl.query().unclipped.dr, manual, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
