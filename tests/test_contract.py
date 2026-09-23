"""Input contract and error-semantics tests."""

import unittest

from delayed_ope import (
    LogRecord,
    LogValidationError,
    NoMatureRecordsError,
    PolicyValidationError,
    QTableValidationError,
    ZeroSupportError,
    evaluate_policy,
)


def _record(**over):
    base = dict(
        context="x",
        action="a",
        reward=1.0,
        logged_propensity=0.5,
        decision_time=0.0,
        observed_time=1.0,
    )
    base.update(over)
    return LogRecord(**base)


POLICY = {"x": {"a": 1.0}}


class RecordValidationTests(unittest.TestCase):
    def test_finite_reward_required(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(LogValidationError) as cm:
                    _record(reward=bad)
                self.assertIn("reward", str(cm.exception))

    def test_finite_propensity_required(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(LogValidationError):
                    _record(logged_propensity=bad)

    def test_propensity_range(self):
        with self.assertRaises(LogValidationError):
            _record(logged_propensity=0.0)
        with self.assertRaises(LogValidationError):
            _record(logged_propensity=-0.1)
        with self.assertRaises(LogValidationError):
            _record(logged_propensity=1.0000001)

    def test_bool_rejected_as_numeric(self):
        # bool is an int subclass; it must never be silently accepted.
        with self.assertRaises(LogValidationError):
            _record(reward=True)
        with self.assertRaises(LogValidationError):
            _record(decision_time=False)

    def test_string_number_rejected(self):
        with self.assertRaises(LogValidationError):
            _record(reward="1.0")
        with self.assertRaises(LogValidationError):
            _record(logged_propensity="0.5")

    def test_none_keys_rejected(self):
        with self.assertRaises(LogValidationError):
            _record(context=None)
        with self.assertRaises(LogValidationError):
            _record(action=None)

    def test_unhashable_context_rejected(self):
        with self.assertRaises(LogValidationError):
            _record(context=["x"])

    def test_observed_before_decision_rejected(self):
        with self.assertRaises(LogValidationError) as cm:
            _record(decision_time=5.0, observed_time=4.0)
        self.assertIn("observed_time", str(cm.exception))

    def test_observed_equal_decision_allowed(self):
        rec = _record(decision_time=3.0, observed_time=3.0)
        self.assertEqual(rec.observed_time, 3.0)

    def test_int_numerics_accepted(self):
        rec = _record(reward=2, logged_propensity=1, decision_time=0, observed_time=1)
        self.assertEqual(rec.reward, 2)


class EvaluateArgumentTests(unittest.TestCase):
    def setUp(self):
        self.recs = [_record()]

    def test_cutoff_must_be_finite(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(LogValidationError):
                    evaluate_policy(self.recs, POLICY, cutoff=bad)

    def test_cutoff_bool_rejected(self):
        with self.assertRaises(LogValidationError):
            evaluate_policy(self.recs, POLICY, cutoff=True)

    def test_clip_must_be_positive_finite(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(LogValidationError):
                    evaluate_policy(self.recs, POLICY, cutoff=2.0, clip=bad)

    def test_clip_bool_rejected(self):
        with self.assertRaises(LogValidationError):
            evaluate_policy(self.recs, POLICY, cutoff=2.0, clip=True)

    def test_max_records_bool_rejected(self):
        with self.assertRaises(LogValidationError):
            evaluate_policy(self.recs, POLICY, cutoff=2.0, max_records=True)

    def test_max_records_must_be_positive_int(self):
        for bad in (0, -1, 1.5, "10"):
            with self.subTest(bad=bad):
                with self.assertRaises(LogValidationError):
                    evaluate_policy(self.recs, POLICY, cutoff=2.0, max_records=bad)

    def test_record_count_cap_enforced(self):
        with self.assertRaises(LogValidationError) as cm:
            evaluate_policy(self.recs, POLICY, cutoff=2.0, max_records=0)
        self.assertIn("max_records", str(cm.exception))

    def test_records_must_be_sequence_of_logrecord(self):
        with self.assertRaises(LogValidationError):
            evaluate_policy(("not records",), POLICY, cutoff=2.0)
        with self.assertRaises(LogValidationError):
            evaluate_policy([("x", "a", 1.0, 0.5, 0.0, 1.0)], POLICY, cutoff=2.0)

    def test_empty_log_rejected(self):
        with self.assertRaises(LogValidationError):
            evaluate_policy([], POLICY, cutoff=2.0)

    def test_error_locates_record_index(self):
        # LogRecord validates at construction, so bypass the frozen
        # dataclass to corrupt record 1 post hoc; the batch entry point
        # must re-validate and prefix the error with the index.
        bad = _record()
        object.__setattr__(bad, "reward", float("nan"))
        recs = [_record(), bad]
        with self.assertRaises(LogValidationError) as cm:
            evaluate_policy(recs, POLICY, cutoff=2.0)
        self.assertIn("records[1]", str(cm.exception))
        self.assertIn("reward", str(cm.exception))


class PolicyValidationTests(unittest.TestCase):
    def setUp(self):
        self.recs = [_record()]

    def test_missing_context_row(self):
        with self.assertRaises(PolicyValidationError) as cm:
            evaluate_policy(self.recs, {"y": {"a": 1.0}}, cutoff=2.0)
        self.assertIn("x", str(cm.exception))

    def test_row_must_be_mapping(self):
        with self.assertRaises(PolicyValidationError):
            evaluate_policy(self.recs, {"x": [("a", 1.0)]}, cutoff=2.0)

    def test_probabilities_sum_to_one(self):
        with self.assertRaises(PolicyValidationError):
            evaluate_policy(self.recs, {"x": {"a": 0.8, "b": 0.1}}, cutoff=2.0)

    def test_probability_out_of_range(self):
        with self.assertRaises(PolicyValidationError):
            evaluate_policy(self.recs, {"x": {"a": 1.2, "b": -0.2}}, cutoff=2.0)

    def test_nan_probability_rejected(self):
        with self.assertRaises(PolicyValidationError):
            evaluate_policy(self.recs, {"x": {"a": float("nan")}}, cutoff=2.0)

    def test_all_zero_row_rejected(self):
        # Action a is observed, but a row assigning it probability 0
        # sums to 0 and must be refused (no target distribution).
        with self.assertRaises(PolicyValidationError):
            evaluate_policy(self.recs, {"x": {"a": 0.0}}, cutoff=2.0)

    def test_normalisation_tolerance_boundary(self):
        # 1e-12 deviation is the documented edge of the tolerance band.
        with self.assertRaises(PolicyValidationError):
            evaluate_policy(
                self.recs, {"x": {"a": 1.0 + 2e-12}}, cutoff=2.0
            )

    def test_logged_action_missing_from_policy_row_rejected(self):
        # b was logged under x but the target row omits b entirely.
        # Implicit reading as pi(b|x)=0 would be ambiguous; require an
        # explicit zero entry instead of raising a raw KeyError later.
        recs = [
            _record(action="a"),
            _record(action="b", logged_propensity=0.5,
                    decision_time=1.0, observed_time=2.0),
        ]
        with self.assertRaises(PolicyValidationError) as cm:
            evaluate_policy(recs, {"x": {"a": 1.0}}, cutoff=2.0)
        self.assertIn("b", str(cm.exception))

    def test_logged_action_explicit_zero_is_accepted(self):
        recs = [
            _record(action="a"),
            _record(action="b", logged_propensity=0.5,
                    decision_time=1.0, observed_time=2.0),
        ]
        res = evaluate_policy(recs, {"x": {"a": 1.0, "b": 0.0}}, cutoff=2.0)
        self.assertEqual(res.n_mature, 2)


class ZeroSupportTests(unittest.TestCase):
    def test_positive_target_prob_but_never_logged(self):
        # Logging only ever played 'a' under x; target wants some 'b'.
        recs = [
            _record(action="a"),
            _record(action="a", decision_time=1.0, observed_time=2.0),
        ]
        pi = {"x": {"a": 0.7, "b": 0.3}}
        with self.assertRaises(ZeroSupportError) as cm:
            evaluate_policy(recs, pi, cutoff=10.0)
        msg = str(cm.exception)
        self.assertIn("b", msg)
        self.assertIn("zero logging support", msg)

    def test_zero_target_prob_on_unlogged_action_is_fine(self):
        recs = [_record(action="a")]
        pi = {"x": {"a": 1.0, "b": 0.0}}
        res = evaluate_policy(recs, pi, cutoff=10.0)
        self.assertEqual(res.ips, 2.0)

    def test_support_refusal_fires_with_mature_data(self):
        # The only observed action under x is 'a'; the target mixing in
        # 'b' must be refused even though a mature record exists.
        recs = [_record(action="a")]
        with self.assertRaises(ZeroSupportError):
            evaluate_policy(recs, {"x": {"a": 0.5, "b": 0.5}}, cutoff=10.0)

    def test_support_refusal_checks_full_log_pairs(self):
        # 'b' appears in the log, but under a different context y; the
        # pair (x, b) still has zero support.
        recs = [
            _record(context="x", action="a"),
            _record(context="y", action="b"),
        ]
        pi = {"x": {"a": 0.5, "b": 0.5}, "y": {"b": 1.0}}
        with self.assertRaises(ZeroSupportError) as cm:
            evaluate_policy(recs, pi, cutoff=10.0)
        self.assertIn("x", str(cm.exception))

    def test_support_refusal_precedes_empty_maturity(self):
        # Structural refusal must win even when the cutoff makes every
        # record immature: support compatibility is cutoff-independent.
        recs = [
            _record(action="a", decision_time=0.0, observed_time=10.0),
        ]
        with self.assertRaises(ZeroSupportError):
            evaluate_policy(
                recs, {"x": {"a": 0.5, "b": 0.5}}, cutoff=1.0
            )


class QHatValidationTests(unittest.TestCase):
    def setUp(self):
        self.recs = [_record()]

    def test_qhat_must_be_mapping(self):
        with self.assertRaises(QTableValidationError):
            evaluate_policy(self.recs, POLICY, cutoff=2.0, q_hat=42)

    def test_qhat_missing_context(self):
        with self.assertRaises(QTableValidationError) as cm:
            evaluate_policy(self.recs, POLICY, cutoff=2.0, q_hat={"y": {"a": 0.0}})
        self.assertIn("x", str(cm.exception))

    def test_qhat_missing_positive_prob_action(self):
        recs = [_record(action="a"), _record(action="b", logged_propensity=0.5)]
        pi = {"x": {"a": 0.5, "b": 0.5}}
        with self.assertRaises(QTableValidationError) as cm:
            evaluate_policy(recs, pi, cutoff=2.0, q_hat={"x": {"a": 1.0}})
        self.assertIn("b", str(cm.exception))

    def test_qhat_nonfinite_rejected(self):
        with self.assertRaises(QTableValidationError):
            evaluate_policy(
                self.recs, POLICY, cutoff=2.0, q_hat={"x": {"a": float("inf")}}
            )

    def test_qhat_extra_entries_allowed(self):
        res = evaluate_policy(
            self.recs,
            POLICY,
            cutoff=2.0,
            q_hat={"x": {"a": 1.0, "ghost": 9.0}},
        )
        self.assertIsNotNone(res.dr)


class ImmaturityTests(unittest.TestCase):
    def test_all_immature_raises_with_accounting(self):
        recs = [
            _record(decision_time=0.0, observed_time=10.0),
            _record(decision_time=1.0, observed_time=11.0),
        ]
        with self.assertRaises(NoMatureRecordsError) as cm:
            evaluate_policy(recs, POLICY, cutoff=5.0)
        self.assertEqual(cm.exception.total, 2)
        self.assertEqual(cm.exception.immature, 2)
        self.assertEqual(cm.exception.cutoff, 5.0)

    def test_boundary_observed_equal_cutoff_is_mature(self):
        recs = [_record(observed_time=5.0)]
        res = evaluate_policy(recs, POLICY, cutoff=5.0)
        self.assertEqual(res.n_mature, 1)

    def test_immature_records_are_not_zero_filled(self):
        # One mature (r=1) and one delayed record with arbitrary reward.
        # IPS must use only the mature record; the immature value must
        # not enter as zero (which would halve the estimate).
        recs = [
            _record(reward=1.0, observed_time=1.0),
            _record(reward=100.0, decision_time=1.0, observed_time=9.0),
        ]
        res = evaluate_policy(recs, POLICY, cutoff=5.0)
        self.assertEqual(res.n_mature, 1)
        self.assertEqual(res.ips, 2.0)
        self.assertEqual(res.maturity.immature_indices, (1,))


if __name__ == "__main__":
    unittest.main()
