"""必要测试：枚举有限分布的无偏恒等式 + 各边界与输入校验。

期望值来源（均不调用 delayed_ope 的算法得到）：
* tests.independent_reference —— 按分布定义直接求和；
* 测试内按公式手算的标量求和。
被测核心只负责产生待核对的输出。
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from delayed_ope import (
    CutoffOutOfOrderError,
    IncrementalEvaluation,
    InvalidLogError,
    InvalidPolicyError,
    InvalidQHatError,
    LogEvent,
    NoMatureSamplesError,
    WeightOverflowError,
    ZeroSupportError,
    evaluate,
)
from delayed_ope.core import MAX_EVENTS, MAX_WEIGHT, PROB_SUM_TOL

from independent_reference import (
    ACTIONS,
    CENSUS_PER_CONTEXT,
    CONTEXTS,
    policy_value,
)


def compositions(total, parts):
    """整数 composition：parts 个非负整数之和为 total。"""
    for c in itertools.product(range(total + 1), repeat=parts - 1):
        s = sum(c)
        if s <= total:
            yield (*c, total - s)


# 取一个确定的、覆盖含零/均匀/偏置形状的子集。
def select_distributions():
    allc = list(compositions(CENSUS_PER_CONTEXT, len(ACTIONS)))
    idxs = {0, 1, 3, 7, 12, 21, 34, 65}
    picked = [allc[i] for i in sorted(idxs & set(range(len(allc))))]
    # 保证关键形状在场：全在一个动作、均匀、两个含零孔。
    must = [(10, 0, 0), (4, 3, 3), (0, 10, 0), (0, 5, 5)]
    have = set(picked)
    for m in must:
        if m not in have:
            picked.append(m)
    return picked


DIST_SHAPES = select_distributions()


def counts_to_policy(counts):
    return {a: c / CENSUS_PER_CONTEXT for a, c in zip(ACTIONS, counts)}


REWARD_TABLES = [
    {0: {0: 0.0, 1: 1.0, 2: 0.0}, 1: {0: 1.0, 1: 0.0, 2: 0.0}},
    {0: {0: -1.0, 1: 0.5, 2: 2.0}, 1: {0: 2.0, 1: -2.0, 2: 0.25}},
    {0: {0: 3.0, 1: 3.0, 2: 3.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
]

# (N_0, N_1)，均为 10 的倍数。
CONTEXT_COUNT_PAIRS = [(20, 20), (10, 30), (40, 10)]

# 与 R 不同的 q 表，用于验证“普查数据上 DR 对任意 q 恒等于 V(pi)”。
Q_TABLES = [
    {0: {0: 0.3, 1: -0.2, 2: 1.25}, 1: {0: -1.0, 1: 0.75, 2: 0.0}},
    {0: {0: 0.0, 1: 0.0, 2: 0.0}, 1: {0: 0.0, 1: 0.0, 2: 0.0}},
]


def build_census(mu_counts, pi_counts, context_counts, reward_table,
                 observed=1.0, cutoff=2.0):
    """构造普查日志：(x,a) 的条数严格正比于 N_x * mu(a|x)。"""
    mu = {x: counts_to_policy(mu_counts[x]) for x in CONTEXTS}
    pi = {x: counts_to_policy(pi_counts[x]) for x in CONTEXTS}
    events = []
    for x in CONTEXTS:
        nx = context_counts[x]
        for a in ACTIONS:
            k = nx * mu_counts[x][ACTIONS.index(a)] // CENSUS_PER_CONTEXT
            for _ in range(k):
                events.append(LogEvent(x, a, reward_table[x][a],
                                       mu[x][a], 0.0, observed))
    return events, mu, pi


def table_to_flat_q(q_table):
    return {(x, a): q_table[x][a] for x in CONTEXTS for a in ACTIONS}


class CensusIdentityTests(unittest.TestCase):
    """枚举有限上下文动作分布：无截断 IPS/SNIPS/DR 的普查期望恒等。"""

    def test_unclipped_expectation_enumeration(self):
        case_no = 0
        skipped_support = 0
        for mu_shape in DIST_SHAPES:
            for pi_shape in DIST_SHAPES:
                mu_counts = {0: mu_shape, 1: mu_shape[::-1]}
                pi_counts = {0: pi_shape, 1: pi_shape}
                # 重要性恒等式仅在绝对连续 pi << mu 下成立。不满足时普查中
                # 该 (x,a) 条数为 0，无法以普查核对恒等式，跳过枚举
                # （零支持拒绝由专门用例在“真的出现零倾向事件”时覆盖）。
                support_ok = all(
                    pi_counts[x][j] == 0 or mu_counts[x][j] > 0
                    for x in CONTEXTS for j in range(len(ACTIONS)))
                for n_pair in CONTEXT_COUNT_PAIRS:
                    context_counts = dict(zip(CONTEXTS, n_pair))
                    for r_table in REWARD_TABLES:
                        events, mu, pi = build_census(
                            mu_counts, pi_counts, context_counts, r_table)
                        if not support_ok:
                            skipped_support += 1
                            continue
                        case_no += 1
                        n_total = sum(context_counts.values())
                        ctx_p = {x: context_counts[x] / n_total
                                 for x in CONTEXTS}
                        expected = policy_value(ctx_p, pi, r_table)
                        res = evaluate(events, pi, cutoff=2.0)
                        self.assertEqual(res.n_mature, n_total)
                        self.assertEqual(res.n_immature, 0)
                        for name, got in (
                            ("IPS", res.unclipped.ips),
                            ("SNIPS", res.unclipped.snips),
                        ):
                            self.assertIsNotNone(got)
                            self.assertTrue(
                                math.isfinite(got),
                                f"{name} 非有限值 case={case_no}")
                            self.assertLessEqual(
                                abs(got - expected),
                                1e-11 * max(1.0, abs(expected)),
                                f"{name} case={case_no}: got={got!r} "
                                f"expected={expected!r}")
        self.assertGreater(case_no, 100)
        self.assertGreater(skipped_support, 0)

    def test_dr_identity_for_arbitrary_q_on_census(self):
        # 普查上 DR 对*任意* q_hat 恒等于 V(pi)；再额外验证完美 q。
        case_no = 0
        mu_counts = {0: (5, 3, 2), 1: (2, 2, 6)}
        pi_counts = {0: (2, 5, 3), 1: (6, 2, 2)}
        context_counts = dict(zip(CONTEXTS, (20, 20)))
        ctx_p = {x: 0.5 for x in CONTEXTS}
        for r_table in REWARD_TABLES:
            events, mu, pi = build_census(
                mu_counts, pi_counts, context_counts, r_table)
            expected = policy_value(ctx_p, pi, r_table)
            for q_table in Q_TABLES + [r_table]:
                case_no += 1
                res = evaluate(events, pi, 2.0, q_hat=table_to_flat_q(q_table))
                self.assertLessEqual(
                    abs(res.unclipped.dr - expected),
                    1e-11 * max(1.0, abs(expected)),
                    f"DR case={case_no}: got={res.unclipped.dr!r} "
                    f"expected={expected!r}")
            # 完美 q：每条样本校正项为 0，DR 等于已实现 context 上 m_pi 的
            # 平均（独立手算）。
            res = evaluate(events, pi, 2.0, q_hat=table_to_flat_q(r_table))
            realized = {}
            for e in events:
                realized[e.context] = realized.get(e.context, 0) + 1
            manual = sum(
                cnt / len(events) * sum(pi[x][a] * r_table[x][a]
                                        for a in ACTIONS)
                for x, cnt in realized.items())
            self.assertLessEqual(abs(res.unclipped.dr - manual), 1e-12)

    def test_perfect_q_on_stochastic_log(self):
        # 非普查的随机日志：q_hat 完美时 w(r-q)=0，DR 精确等于已实现
        # context 序列上 m_pi 的平均（无估计噪声），并接近真实 V。
        rng = random.Random(20260923)
        mu = {0: (0.5, 0.3, 0.2), 1: (0.2, 0.2, 0.6)}
        pi = {0: (0.25, 0.55, 0.20), 1: (0.4, 0.1, 0.5)}
        r_table = REWARD_TABLES[1]
        n = 4000
        events = []
        for _ in range(n):
            x = 0 if rng.random() < 0.5 else 1
            u = rng.random()
            acc = 0.0
            a = ACTIONS[-1]
            for k, pr in enumerate(mu[x]):
                acc += pr
                if u < acc:
                    a = ACTIONS[k]
                    break
            events.append(LogEvent(x, a, float(r_table[x][a]),
                                   mu[x][ACTIONS.index(a)], 0.0, 1.0))
        pi_map = {x: dict(zip(ACTIONS, pi[x])) for x in CONTEXTS}
        res = evaluate(events, pi_map, 2.0, q_hat=table_to_flat_q(r_table))
        # 独立手算已实现 m_pi 平均。
        manual = sum(
            sum(pi_map[e.context][a] * r_table[e.context][a]
                for a in ACTIONS)
            for e in events) / n
        self.assertLessEqual(abs(res.unclipped.dr - manual), 1e-12)
        # 辅助统计检查（确定性种子）：真实价值由参考定义给出。
        v_true = policy_value({0: 0.5, 1: 0.5}, pi_map, r_table)
        self.assertLessEqual(abs(res.unclipped.dr - v_true), 0.2)
        # IPS/SNIPS 也应接近 V（仅统计意义，宽松，非边界口径）。
        self.assertLessEqual(abs(res.unclipped.ips - v_true), 0.2)
        self.assertLessEqual(abs(res.unclipped.snips - v_true), 0.2)


class MaturityTests(unittest.TestCase):
    """成熟口径：observed_time <= cutoff；未成熟绝不当 reward 0。"""

    def test_mixed_maturity_excludes_immature(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0),
            LogEvent("x", 1, 0.0, 0.5, 0.0, 3.0),   # 未成熟
            LogEvent("x", 0, 3.0, 0.5, 2.0, 4.0),   # 未成熟
            LogEvent("x", 1, 2.0, 0.5, 2.0, 2.0),   # 成熟
        ]
        res = evaluate(events, pi, cutoff=2.0)
        self.assertEqual((res.n_total, res.n_mature, res.n_immature),
                         (4, 2, 2))
        self.assertEqual(res.mature_indices, (0, 3))
        # 手算（w=1）：IPS=(1+2)/2=1.5；SNIPS=3/2=1.5。
        self.assertAlmostEqual(res.unclipped.ips, 1.5, places=12)
        self.assertAlmostEqual(res.unclipped.snips, 1.5, places=12)

    def test_all_immature_rejected(self):
        pi = {"x": {0: 1.0}}
        events = [LogEvent("x", 0, 99.0, 1.0, 0.0, 5.0)]
        with self.assertRaises(NoMatureSamplesError):
            evaluate(events, pi, cutoff=4.0)
        # 边界：observed_time == cutoff 算成熟。
        res = evaluate(events, pi, cutoff=5.0)
        self.assertEqual(res.n_mature, 1)

    def test_immature_reward_never_counted_as_zero(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        # 若错误地把未成熟当 reward=0，IPS 会变成 1/2 而非 1。
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0),
            LogEvent("x", 1, 7.0, 0.5, 0.0, 9.0),
        ]
        res = evaluate(events, pi, cutoff=1.0)
        self.assertEqual(res.n_mature, 1)
        self.assertAlmostEqual(res.unclipped.ips, 1.0, places=12)


class SupportAndWeightTests(unittest.TestCase):
    def test_zero_support_rejected(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        events = [LogEvent("x", 1, 1.0, 0.0, 0.0, 1.0)]
        with self.assertRaises(ZeroSupportError):
            evaluate(events, pi, cutoff=2.0)

    def test_zero_support_checked_even_when_immature(self):
        # 结构性不相容，不因事件未成熟而放过。
        pi = {"x": {0: 0.5, 1: 0.5}}
        events = [LogEvent("x", 1, 1.0, 0.0, 0.0, 9.0)]
        with self.assertRaises(ZeroSupportError):
            evaluate(events, pi, cutoff=1.0)

    def test_target_zero_on_zero_support_is_finite_weight_zero(self):
        # 目标概率 0 / 记录概率 0：权重定义为 0，合法；SNIPS 因全零权重
        # 无定义（None），IPS=0，ESS=0，口径显式而非静默。
        pi = {"x": {0: 1.0, 1: 0.0}}
        events = [LogEvent("x", 1, 5.0, 0.0, 0.0, 1.0)]
        res = evaluate(events, pi, cutoff=2.0)
        self.assertEqual(res.unclipped.ips, 0.0)
        self.assertIsNone(res.unclipped.snips)
        self.assertEqual(res.unclipped.ess, 0.0)

    def test_extreme_but_finite_weight(self):
        pi = {"x": {0: 1.0}}
        events = [LogEvent("x", 0, 2.0, 1e-20, 0.0, 1.0)]
        res = evaluate(events, pi, 2.0)
        self.assertLessEqual(
            abs(res.unclipped.ips - 2e20) / 2e20, 1e-12)
        # 单一样本主导：ESS 退化为 1。
        self.assertAlmostEqual(res.unclipped.ess, 1.0, places=12)

    def test_overflow_weight_rejected(self):
        pi = {"x": {0: 1.0}}
        events = [LogEvent("x", 0, 1.0, 1e-308, 0.0, 1.0)]
        with self.assertRaises(WeightOverflowError):
            evaluate(events, pi, 2.0)

    def test_explicit_clipping_reports_both_and_ess(self):
        pi = {"x": {0: 0.9, 1: 0.1}}
        events = [
            LogEvent("x", 0, 1.0, 0.1, 0.0, 1.0),   # w=9
            LogEvent("x", 1, 1.0, 0.9, 0.0, 1.0),   # w=1/9
        ]
        res = evaluate(events, pi, 2.0, clip_value=2.0)
        # 未截断手算。
        w1, w2 = 9.0, 0.1 / 0.9
        ips_u = (w1 + w2) / 2
        sn_u = (w1 + w2) / (w1 + w2)
        self.assertAlmostEqual(res.unclipped.ips, ips_u, places=12)
        self.assertAlmostEqual(res.unclipped.snips, sn_u, places=12)
        ess_u = (w1 + w2) ** 2 / (w1 * w1 + w2 * w2)
        self.assertAlmostEqual(res.unclipped.ess, ess_u, places=9)
        self.assertLessEqual(res.unclipped.ess, 2.0 + 1e-9)
        self.assertGreaterEqual(res.unclipped.ess, 1.0 - 1e-9)
        # 截断手算：2 与 1/9。
        c1, c2 = 2.0, w2
        ips_c = (c1 + c2) / 2
        self.assertIsNotNone(res.clipped)
        self.assertAlmostEqual(res.clipped.ips, ips_c, places=12)
        self.assertNotAlmostEqual(res.clipped.ips, res.unclipped.ips)
        self.assertGreater(res.clipped.ess, res.unclipped.ess)

    def test_ess_bounds_on_moderate_weights(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0),
            LogEvent("x", 1, 0.0, 0.5, 0.0, 1.0),
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0),
        ]
        res = evaluate(events, pi, 2.0)
        self.assertAlmostEqual(res.unclipped.ess, 3.0, places=12)


class DRFormulaTests(unittest.TestCase):
    def test_dr_uses_marginal_q_under_target(self):
        # 手工小例：x 上两动作，q 与 r 不同；按定义独立求和核对。
        pi = {"x": {0: 0.75, 1: 0.25}}
        q = {("x", 0): 0.4, ("x", 1): 0.8}
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0),  # w=1.5
            LogEvent("x", 1, 0.0, 0.5, 0.0, 1.0),  # w=0.5
        ]
        res = evaluate(events, pi, 2.0, q_hat=q)
        m_pi = 0.75 * 0.4 + 0.25 * 0.8
        manual = (
            (m_pi + 1.5 * (1.0 - 0.4))
            + (m_pi + 0.5 * (0.0 - 0.8))
        ) / 2
        self.assertAlmostEqual(res.unclipped.dr, manual, places=12)

    def test_nested_q_form_equivalent_to_flat(self):
        pi = {"x": {0: 0.6, 1: 0.4}}
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0),
            LogEvent("x", 1, 2.0, 0.5, 0.0, 1.0),
        ]
        flat = {("x", 0): 0.1, ("x", 1): 0.9}
        nested = {"x": {0: 0.1, 1: 0.9}}
        r1 = evaluate(events, pi, 2.0, q_hat=flat)
        r2 = evaluate(events, pi, 2.0, q_hat=nested)
        self.assertEqual(r1.unclipped.dr, r2.unclipped.dr)


class ValidationTests(unittest.TestCase):
    GOOD_PI = {"x": {0: 0.5, 1: 0.5}}

    def _ev(self, **kw):
        defaults = dict(
            events=[LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0)],
            target_policy=self.GOOD_PI, cutoff=2.0)
        defaults.update(kw)
        return evaluate(**defaults)

    def test_reject_nan_inf(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidLogError):
                evaluate([LogEvent("x", 0, bad, 0.5, 0.0, 1.0)],
                         self.GOOD_PI, 2.0)
            with self.assertRaises(InvalidLogError):
                evaluate([LogEvent("x", 0, 1.0, 0.5, bad, 1.0)],
                         self.GOOD_PI, 2.0)
            with self.assertRaises(InvalidPolicyError):
                self._ev(target_policy={"x": {0: bad, 1: 0.5}})

    def test_reject_bool_for_numeric_params(self):
        with self.assertRaises(InvalidLogError):
            self._ev(cutoff=True)
        with self.assertRaises(InvalidPolicyError):
            self._ev(clip_value=False)
        # 事件内数值位置同样拒绝 bool（bool 不是合法数值）。
        for fld, val in (
            ("reward", True),
            ("logged_propensity", False),
            ("decision_time", True),
            ("observed_time", False),
        ):
            kw = dict(context="x", action=0, reward=1.0,
                      logged_propensity=0.5, decision_time=0.0,
                      observed_time=1.0)
            kw[fld] = val
            with self.assertRaises(InvalidLogError):
                evaluate([LogEvent(**kw)], self.GOOD_PI, 2.0)
        with self.assertRaises(InvalidPolicyError):
            self._ev(target_policy={"x": {0: True, 1: 0.0}})

    def test_propensity_range_and_time_order(self):
        with self.assertRaises(InvalidLogError):
            evaluate([LogEvent("x", 0, 1.0, 1.1, 0.0, 1.0)],
                     self.GOOD_PI, 2.0)
        with self.assertRaises(InvalidLogError):
            evaluate([LogEvent("x", 0, 1.0, -0.1, 0.0, 1.0)],
                     self.GOOD_PI, 2.0)
        with self.assertRaises(InvalidLogError):
            evaluate([LogEvent("x", 0, 1.0, 0.5, 2.0, 1.0)],
                     self.GOOD_PI, 2.0)

    def test_policy_normalization_and_coverage(self):
        with self.assertRaises(InvalidPolicyError):
            self._ev(target_policy={"x": {0: 0.5, 1: 0.4}})
        with self.assertRaises(InvalidPolicyError):
            self._ev(target_policy={"x": {0: 0.5}})
        with self.assertRaises(InvalidPolicyError):
            self._ev(target_policy={"y": {0: 0.5, 1: 0.5}})
        with self.assertRaises(InvalidPolicyError):
            self._ev(target_policy={"x": {0: 1.5, 1: -0.5}})

    def test_bad_clip_value(self):
        with self.assertRaises(InvalidPolicyError):
            self._ev(clip_value=0.0)
        with self.assertRaises(InvalidPolicyError):
            self._ev(clip_value=float("inf"))

    def test_qhat_validation(self):
        with self.assertRaises(InvalidQHatError):
            self._ev(q_hat={("x", 0): float("nan"), ("x", 1): 0.0})
        with self.assertRaises(InvalidQHatError):
            # 成熟行内动作缺值（m_pi 需要完整概率行）。
            self._ev(q_hat={("x", 0): 0.1})
        with self.assertRaises(InvalidQHatError):
            self._ev(q_hat=42)

    def test_bad_event_container_and_type(self):
        with self.assertRaises(InvalidLogError):
            evaluate("not a sequence", self.GOOD_PI, 2.0)
        with self.assertRaises(InvalidLogError):
            evaluate([("x", 0, 1.0, 0.5, 0.0, 1.0)], self.GOOD_PI, 2.0)


class SideEffectTests(unittest.TestCase):
    def test_policy_not_mutated(self):
        pi = {"x": {0: 0.5, 1: 0.5}}
        events = [LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0)]
        evaluate(events, pi, 2.0)
        self.assertEqual(pi, {"x": {0: 0.5, 1: 0.5}})


class IncrementalEvaluationTests(unittest.TestCase):
    def _direct_reference(self, events, pi, cutoff, q_hat=None, clip=None):
        mature = [e for e in events if e.observed_time <= cutoff]
        self.assertTrue(mature)
        weights = []
        clipped = []
        rewards = []
        m_values = []
        q_values = []
        for event in mature:
            weight = pi[event.context][event.action] / event.logged_propensity
            weights.append(weight)
            clipped.append(min(weight, clip) if clip is not None else weight)
            rewards.append(event.reward)
            if q_hat is not None:
                row = pi[event.context]
                m_values.append(math.fsum(
                    row[a] * q_hat[(event.context, a)] for a in row))
                q_values.append(q_hat[(event.context, event.action)])

        def estimate(ws):
            sum_w = math.fsum(ws)
            sum_wr = math.fsum(w * r for w, r in zip(ws, rewards))
            sum_w2 = math.fsum(w * w for w in ws)
            dr = None
            if q_hat is not None:
                dr = math.fsum(
                    m + w * (r - q)
                    for m, w, r, q in zip(
                        m_values, ws, rewards, q_values)) / len(mature)
            return {
                "ips": sum_wr / len(mature),
                "snips": sum_wr / sum_w if sum_w != 0.0 else None,
                "dr": dr,
                "ess": sum_w * sum_w / sum_w2 if sum_w2 != 0.0 else 0.0,
                "n": len(mature),
            }

        return estimate(weights), estimate(clipped) if clip else None

    def _assert_est_equal(self, got, expected):
        self.assertEqual(got.n, expected["n"])
        self.assertEqual(got.snips is None, expected["snips"] is None)
        for key in ("ips", "snips", "dr", "ess"):
            if expected[key] is None:
                self.assertIsNone(getattr(got, key))
            else:
                self.assertTrue(math.isfinite(getattr(got, key)), key)
                self.assertAlmostEqual(
                    getattr(got, key), expected[key], places=10, msg=key)

    def test_boundaries_match_batch_and_direct_formula(self):
        # 决策顺序与观察顺序相反；事件 2/3 同一观察时刻，必须整组出现。
        events = [
            LogEvent("x", 0, 1.0, 0.8, 0.0, 4.0, "e0"),
            LogEvent("x", 1, -2.0, 0.2, 1.0, 3.0, "e1"),
            LogEvent("x", 0, 3.0, 0.4, 2.0, 2.0, "e2"),
            LogEvent("x", 1, 0.5, 0.1, 0.0, 2.0, "e3"),
        ]
        pi = {"x": {0: 0.25, 1: 0.75}}
        q = {("x", 0): 0.3, ("x", 1): -0.4}
        clip = 2.0
        evaluator = IncrementalEvaluation(events, pi, q, clip)

        with self.assertRaises(NoMatureSamplesError):
            evaluator.advance(1.0)
        self.assertEqual(evaluator.cutoff, 1.0)

        res = evaluator.advance(2.0)
        self.assertEqual(res.mature_indices, (2, 3))
        direct_u, direct_c = self._direct_reference(
            events, pi, 2.0, q, clip)
        batch = evaluate(events, pi, 2.0, q, clip)
        self._assert_est_equal(res.unclipped, direct_u)
        self._assert_est_equal(res.clipped, direct_c)
        self.assertEqual(res, batch)

        res = evaluator.advance(3.5)
        self.assertEqual(res.mature_indices, (1, 2, 3))
        direct_u, direct_c = self._direct_reference(
            events, pi, 3.5, q, clip)
        batch = evaluate(events, pi, 3.5, q, clip)
        self._assert_est_equal(res.unclipped, direct_u)
        self._assert_est_equal(res.clipped, direct_c)
        self.assertEqual(res, batch)

        res = evaluator.advance(4.0)
        self.assertEqual(res.mature_indices, (0, 1, 2, 3))
        direct_u, direct_c = self._direct_reference(
            events, pi, 4.0, q, clip)
        batch = evaluate(events, pi, 4.0, q, clip)
        self._assert_est_equal(res.unclipped, direct_u)
        self._assert_est_equal(res.clipped, direct_c)
        self.assertEqual(res, batch)

    def test_same_observation_time_is_atomic_and_repeated_advance_idempotent(self):
        events = [
            LogEvent("x", 0, 1.0, 0.5, 0.0, 2.0, "a"),
            LogEvent("x", 1, 2.0, 0.5, 1.0, 2.0, "b"),
        ]
        pi = {"x": {0: 0.5, 1: 0.5}}
        evaluator = IncrementalEvaluation(events, pi)
        first = evaluator.advance(2.0)
        second = evaluator.advance(2.0)
        self.assertEqual(first, second)
        self.assertEqual(second.mature_indices, (0, 1))

    def test_backward_cutoff_rejected(self):
        events = [LogEvent("x", 0, 1.0, 1.0, 0.0, 1.0, "a")]
        pi = {"x": {0: 1.0}}
        evaluator = IncrementalEvaluation(events, pi)
        evaluator.advance(2.0)
        with self.assertRaises(CutoffOutOfOrderError):
            evaluator.advance(1.9)

    def test_no_mature_samples_keeps_explicit_empty_state(self):
        events = [LogEvent("x", 0, 9.0, 1.0, 0.0, 2.0, "a")]
        pi = {"x": {0: 1.0}}
        evaluator = IncrementalEvaluation(events, pi)
        with self.assertRaises(NoMatureSamplesError):
            evaluator.advance(1.0)
        self.assertEqual(evaluator.cutoff, 1.0)
        self.assertEqual(evaluator.mature_indices, ())
        with self.assertRaises(CutoffOutOfOrderError):
            evaluator.advance(0.5)
        with self.assertRaises(NoMatureSamplesError):
            evaluator.result()
        res = evaluator.advance(2.0)
        self.assertEqual(res.n_mature, 1)

    def test_target_zero_probability_keeps_snips_undefined(self):
        events = [
            LogEvent("x", 1, 5.0, 0.0, 0.0, 1.0, "a"),
            LogEvent("x", 1, -4.0, 0.0, 1.0, 2.0, "b"),
        ]
        pi = {"x": {0: 1.0, 1: 0.0}}
        evaluator = IncrementalEvaluation(events, pi)
        evaluator.advance(1.0)
        res = evaluator.advance(2.0)
        self.assertEqual(res.unclipped.ips, 0.0)
        self.assertIsNone(res.unclipped.snips)
        self.assertEqual(res.unclipped.ess, 0.0)

    def test_duplicate_event_id_rejected(self):
        events = [
            LogEvent("x", 0, 1.0, 1.0, 0.0, 1.0, "same"),
            LogEvent("x", 0, 2.0, 1.0, 1.0, 2.0, "same"),
        ]
        pi = {"x": {0: 1.0}}
        with self.assertRaises(InvalidLogError):
            IncrementalEvaluation(events, pi)
        with self.assertRaises(InvalidLogError):
            evaluate(events, pi, 2.0)

    def test_clip_and_policy_are_fixed_after_creation(self):
        events = [LogEvent("x", 0, 1.0, 0.5, 0.0, 1.0, "a")]
        pi = {"x": {0: 0.5, 1: 0.5}}
        evaluator = IncrementalEvaluation(events, pi, clip_value=3.0)
        pi["x"][0] = 1.0
        res = evaluator.advance(1.0)
        self.assertEqual(res.clip_value, 3.0)
        self.assertAlmostEqual(res.unclipped.ips, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
