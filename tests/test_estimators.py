"""Estimator correctness tests against an independent Fraction reference.

The reference in :mod:`tests.reference` recomputes IPS/SNIPS/DR/ESS with
exact rationals and never imports the package under test for expected
values; float conversion happens only at the core boundary.
"""

import random
import unittest
from fractions import Fraction
from itertools import product

from delayed_ope import (
    LogRecord,
    NoMatureRecordsError,
    NumericalOverflowError,
    ZeroSupportError,
    evaluate_policy,
)

from tests.reference import (
    estimate,
    policy_to_float,
    specs_to_records,
    true_value,
)


F = Fraction  # two-arg construction F(1, 2) is used throughout


def assertCloseFraction(testcase, actual, expected, *, tol=1e-12, label=""):
    """Compare a float result with an exact Fraction.

    Tolerance is an absolute float error of 1e-12 (scaled by the
    magnitude of the exact value for large magnitudes, but never
    looser than 1e-9 relative); this is a tight arithmetic bound, not
    a slack that could hide an estimator bug.
    """
    if expected is None:
        testcase.assertIsNone(actual, f"{label}: expected None")
        return
    ef = float(expected)
    testcase.assertIsNotNone(actual, f"{label}: got None, expected {ef}")
    abs_err = abs(actual - ef)
    scale = max(1.0, abs(ef))
    testcase.assertLessEqual(
        abs_err,
        tol * scale,
        f"{label}: actual={actual!r} exact={ef!r} abs_err={abs_err!r}",
    )


class ReferenceParityTests(unittest.TestCase):
    """Fixed datasets: core output must match exact rational results."""

    def setUp(self):
        # Two contexts, two actions, mixed maturity, varied propensities.
        self.specs = [
            # x
            ("x", "a", F(1), F(1, 2), 0, 1),
            ("x", "b", F(2), F(1, 4), 0, 8),   # immature at cutoff 5
            ("x", "a", F(0), F(1, 2), 1, 3),
            ("x", "b", F(-1), F(3, 4), 2, 4),
            # y
            ("y", "a", F(3), F(1, 3), 0, 2),
            ("y", "b", F(1), F(2, 3), 1, 9),   # immature at cutoff 5
        ]
        self.pi = {
            "x": {"a": F(1, 4), "b": F(3, 4)},
            "y": {"a": F(2, 3), "b": F(1, 3)},
        }
        self.q = {
            "x": {"a": F(1, 2), "b": F(-1, 2)},
            "y": {"a": F(4), "b": F(0)},
        }
        self.cutoff = 5

    def _run(self, *, clip=None, with_q=True):
        recs = specs_to_records(self.specs)
        return evaluate_policy(
            recs,
            policy_to_float(self.pi),
            self.cutoff,
            q_hat=policy_to_float(self.q) if with_q else None,
            clip=float(clip) if clip is not None else None,
        )

    def _expected(self, *, clip=None, with_q=True):
        return estimate(
            self.specs,
            self.pi,
            self.cutoff,
            q=self.q if with_q else None,
            clip=clip,
        )

    def test_unclipped_matches_reference(self):
        res = self._run()
        exp = self._expected()
        self.assertEqual(res.n_mature, exp["n_mature"])
        assertCloseFraction(self, res.ips, exp["ips"], label="ips")
        assertCloseFraction(self, res.snips, exp["snips"], label="snips")
        assertCloseFraction(self, res.dr, exp["dr"], label="dr")
        assertCloseFraction(self, res.ess_unclipped, exp["ess"], label="ess")

    def test_clipped_matches_reference_and_unclipped_retained(self):
        clip = F(1, 1)
        res = self._run(clip=clip)
        exp_c = self._expected(clip=clip)
        exp_u = self._expected()
        # clipped columns
        assertCloseFraction(self, res.ips, exp_c["ips"], label="ips-c")
        assertCloseFraction(self, res.snips, exp_c["snips"], label="snips-c")
        assertCloseFraction(self, res.dr, exp_c["dr"], label="dr-c")
        assertCloseFraction(self, res.ess_clipped, exp_c["ess"], label="ess-c")
        # unclipped columns still present and match the no-clip reference
        assertCloseFraction(self, res.ips_unclipped, exp_u["ips"], label="ips-u")
        assertCloseFraction(self, res.snips_unclipped, exp_u["snips"], label="snips-u")
        assertCloseFraction(self, res.dr_unclipped, exp_u["dr"], label="dr-u")
        assertCloseFraction(self, res.ess_unclipped, exp_u["ess"], label="ess-u")
        # clipping actually changed something in this dataset
        self.assertNotAlmostEqual(
            float(exp_c["ips"]), float(exp_u["ips"]), places=15
        )

    def test_without_qhat_dr_is_none(self):
        res = self._run(with_q=False)
        self.assertIsNone(res.dr)
        self.assertIsNone(res.dr_unclipped)

    def test_maturity_accounting(self):
        res = self._run()
        self.assertEqual(res.maturity.total, 6)
        self.assertEqual(res.maturity.mature, 4)
        self.assertEqual(res.maturity.immature, 2)
        self.assertEqual(res.maturity.mature_indices, (0, 2, 3, 4))
        self.assertEqual(res.maturity.immature_indices, (1, 5))

    def test_clip_threshold_equal_to_weight_is_stable(self):
        # Clip exactly at a realized weight: min(w, M) == w, results
        # must equal the unclipped reference.
        res = self._run(clip=F(2))
        exp = self._expected(clip=F(2))
        assertCloseFraction(self, res.ips, exp["ips"], label="ips@2")
        assertCloseFraction(self, res.snips, exp["snips"], label="snips@2")


class EnumeratedUnbiasednessTests(unittest.TestCase):
    """Exhaustively enumerate a finite logging distribution.

    Setup (all rewards observed, no delay):
        contexts x0,x1 drawn uniformly;
        under each context the logger plays a0/a1 with fixed mu;
        reward depends deterministically on (x, a).

    For the complete log containing every (x,a) stratum with the
    relative frequencies the logger induces, IPS/SNIPS/DR sample
    averages reduce algebraically to constants we can compute exactly.
    IPS on the *fully enumerated* joint equals V(pi) because each
    stratum appears proportionally to mu(a|x) * P(x): the weight
    cancels the sampling frequency.  We verify that exhaustively over
    a grid of (mu, pi, rewards, clip), using the Fraction reference
    plus an independent closed-form V(pi).
    """

    def test_unclipped_ips_equals_target_value_on_full_enumeration(self):
        # Grid kept small: exact enumeration, no sampling noise.
        mus = [F(1, 2), F(1, 3), F(2, 5)]
        pis = [F(1, 2), F(1, 4), F(3, 5)]
        reward_choices = [F(0), F(1), F(-1), F(2, 3)]
        count = 0
        for mu_a, pi_a in product(mus, pis):
            # deterministically pick a reward table from the grid
            for r00 in reward_choices:
                reward_table = {
                    "x": {"a": r00, "b": F(1) - r00},
                    "y": {"a": r00 + F(1, 3), "b": -r00},
                }
                mu = {
                    "x": {"a": mu_a, "b": 1 - mu_a},
                    "y": {"a": mu_a, "b": 1 - mu_a},
                }
                pi = {
                    "x": {"a": pi_a, "b": 1 - pi_a},
                    "y": {"a": pi_a, "b": 1 - pi_a},
                }
                specs = self._enumerate(mu, reward_table)
                pi_f = policy_to_float(pi)
                recs = specs_to_records(specs)
                res = evaluate_policy(recs, pi_f, cutoff=100)
                v = true_value(["x", "y"], pi, reward_table)
                assertCloseFraction(self, res.ips, v, tol=1e-11, label="ips=V")
                # SNIPS equals V(pi) here too: stratum counts are
                # exactly proportional to mu, hence sum w / n = 1 and
                # the self-normalised ratio coincides with IPS.
                assertCloseFraction(self, res.snips, v, tol=1e-11, label="snips=V")
                count += 1
        self.assertGreater(count, 20)

    @staticmethod
    def _enumerate(mu, reward_table):
        """Build the fully enumerated log.

        Each context gets ``LCM`` decision slots; action a appears
        mu(a|x)*LCM times and b the rest, so the realized empirical
        logging distribution equals mu exactly.  Rewards are the
        deterministic r(x,a); all observed by time 1.
        """
        specs = []
        # mu denominators in the grid are 2,3,5 -> 30 slots per context.
        slots = 30
        for x in ("x", "y"):
            n_a = int(mu[x]["a"] * slots)
            n_b = slots - n_a
            for _ in range(n_a):
                specs.append((x, "a", reward_table[x]["a"], mu[x]["a"], 0, 1))
            for _ in range(n_b):
                specs.append((x, "b", reward_table[x]["b"], mu[x]["b"], 0, 1))
        return specs


class PerfectQHatTests(unittest.TestCase):
    """With q_hat(x,a) = true r(x,a), DR collapses to V(pi) exactly."""

    def test_perfect_q_removes_variance_term(self):
        reward_table = {
            "x": {"a": F(3, 2), "b": F(-2)},
            "y": {"a": F(0), "b": F(4, 7)},
        }
        mu = {"x": {"a": F(1, 3), "b": F(2, 3)},
              "y": {"a": F(3, 4), "b": F(1, 4)}}
        pi = {"x": {"a": F(2, 5), "b": F(3, 5)},
              "y": {"a": F(1, 6), "b": F(5, 6)}}
        specs = []
        for x in ("x", "y"):
            for a in ("a", "b"):
                for _ in range(6):
                    specs.append((x, a, reward_table[x][a], mu[x][a], 0, 1))
        res = evaluate_policy(
            specs_to_records(specs),
            policy_to_float(pi),
            cutoff=10,
            q_hat=policy_to_float(reward_table),
        )
        v = true_value(["x", "y"], pi, reward_table)
        assertCloseFraction(self, res.dr, v, label="dr-perfect")
        # Per-record correction term is zero: DR also equals the mean
        # of m(x) over records; check it does NOT depend on weights by
        # comparing to a re-weighted clipping run.
        res_clip = evaluate_policy(
            specs_to_records(specs),
            policy_to_float(pi),
            cutoff=10,
            q_hat=policy_to_float(reward_table),
            clip=0.01,
        )
        assertCloseFraction(self, res_clip.dr, v, label="dr-perfect-clip")
        assertCloseFraction(self, res.dr, res_clip.dr, label="dr-inv-weight")


class ExtremeWeightTests(unittest.TestCase):
    def test_extreme_but_finite_weights_match_reference(self):
        # mu tiny but finite; weights large (5e8), still float-finite.
        specs = [
            ("x", "a", F(1), F(1, 1), 0, 1),
            ("x", "b", F(1), Fraction(1, 1_000_000_000), 0, 1),
        ]
        pi = {"x": {"a": F(1, 2), "b": F(1, 2)}}
        res = evaluate_policy(specs_to_records(specs), policy_to_float(pi), 10)
        exp = estimate(specs, pi, 10)
        assertCloseFraction(self, res.ips, exp["ips"], tol=1e-9, label="ips-extreme")
        assertCloseFraction(self, res.ess_unclipped, exp["ess"], tol=1e-9, label="ess-extreme")
        # One dominant weight: ESS close to 1, far below n=2.
        self.assertLess(res.ess_unclipped, 1.2)
        self.assertGreaterEqual(res.max_weight, 5e8 * 0.999)

    def test_clip_tames_extreme_weight(self):
        specs = [
            ("x", "a", F(1), F(1, 1), 0, 1),
            ("x", "b", F(1), F(1, 1_000_000), 0, 1),
        ]
        pi = {"x": {"a": F(1, 2), "b": F(1, 2)}}
        res = evaluate_policy(
            specs_to_records(specs), policy_to_float(pi), 10, clip=10.0
        )
        exp_c = estimate(specs, pi, 10, clip=F(10))
        assertCloseFraction(self, res.ips, exp_c["ips"], label="ips-clip")
        assertCloseFraction(self, res.ess_clipped, exp_c["ess"], label="ess-clip")
        # clipped ESS must exceed unclipped ESS
        self.assertGreater(res.ess_clipped, res.ess_unclipped)

    def test_numerical_overflow_is_a_distinct_error(self):
        # mu at the smallest positive normal float: pi/mu overflows.
        specs = [
            ("x", "a", F(1), F(1), 0, 1),
            ("x", "b", F(1), F(1), 0, 1),
        ]
        recs = specs_to_records(specs)
        # hand-lower the second propensity to the float minimum
        recs[1] = LogRecord(
            context="x", action="b", reward=1.0,
            logged_propensity=5e-324,
            decision_time=0, observed_time=1,
        )
        pi = {"x": {"a": 0.5, "b": 0.5}}
        with self.assertRaises(NumericalOverflowError):
            evaluate_policy(recs, pi, cutoff=10)


class SNIPSEdgeTests(unittest.TestCase):
    def test_zero_target_prob_on_sampled_action(self):
        # Logged 'b' but target never plays b: weight 0, contributes
        # nothing; SNIPS must ignore it rather than divide wrongly.
        specs = [
            ("x", "a", F(2), F(1, 2), 0, 1),
            ("x", "b", F(9), F(1, 2), 0, 1),
        ]
        pi = {"x": {"a": F(1), "b": F(0)}}
        res = evaluate_policy(specs_to_records(specs), policy_to_float(pi), 10)
        exp = estimate(specs, pi, 10)
        assertCloseFraction(self, res.ips, exp["ips"], label="ips")
        assertCloseFraction(self, res.snips, exp["snips"], label="snips")
        self.assertEqual(res.snips, 2.0)

    def test_dr_when_logged_action_has_zero_target_prob(self):
        # b is logged (so support exists for required actions) but
        # pi(b|x)=0.  q_hat may legitimately omit b; DR must not
        # KeyError and the zero-weight record contributes only m(x).
        specs = [
            ("x", "a", F(2), F(1, 2), 0, 1),
            ("x", "b", F(9), F(1, 2), 0, 1),
        ]
        pi = {"x": {"a": F(1), "b": F(0)}}
        q = {"x": {"a": F(1, 2)}}  # deliberately no entry for b
        res = evaluate_policy(
            specs_to_records(specs), policy_to_float(pi), 10,
            q_hat=policy_to_float(q),
        )
        exp = estimate(specs, pi, 10, q=q)
        assertCloseFraction(self, res.dr, exp["dr"], label="dr-zero-prob")
        # Exact value: both records mature; m(x)=0.5; record a adds
        # w*(2-0.5)=2*1.5=3 -> term 3.5; record b adds 0 -> 0.5;
        # mean = (3.5 + 0.5)/2 = 2.
        assertCloseFraction(self, res.dr, F(2), label="dr-exact")

    def test_all_mature_weights_zero_snips_is_none(self):
        # Mature record played only actions with pi(a|x)=0; the
        # positive-probability action's record is still immature.
        # IPS = 0 and SNIPS is None (undefined), never a divide-by-zero.
        specs = [
            ("x", "a", F(3), F(1, 2), 0, 9),   # immature at cutoff 5
            ("x", "b", F(9), F(1, 2), 0, 1),   # mature, pi(b|x)=0
        ]
        pi = {"x": {"a": F(1), "b": F(0)}}
        q = {"x": {"a": F(1, 2)}}
        res = evaluate_policy(
            specs_to_records(specs), policy_to_float(pi), 5,
            q_hat=policy_to_float(q),
        )
        exp = estimate(specs, pi, 5, q=q)
        self.assertEqual(res.n_mature, 1)
        assertCloseFraction(self, res.ips, exp["ips"], label="ips-zero")
        self.assertIsNone(res.snips)
        self.assertIsNone(exp["snips"])
        # DR = m(x) + 0 = 0.5.
        assertCloseFraction(self, res.dr, F(1, 2), label="dr-m-only")


class RandomizedParityTests(unittest.TestCase):
    """Supplementary random cross-checks against the Fraction oracle.

    Expected values always come from the independent exact reference;
    randomization only chooses the data, so this broadens coverage
    without making the core its own oracle.
    """

    def test_many_random_logs(self):
        rng = random.Random(424242)
        contexts = ("c0", "c1")
        actions = ("k0", "k1", "k2")
        n_trials = 40
        for trial in range(n_trials):
            # Random logging policy as exact fractions (denom 20).
            mu: dict = {}
            pi: dict = {}
            for x in contexts:
                mu[x] = _random_dist(rng, actions, 20)
                pi[x] = _random_dist(rng, actions, 10)
            # Random q table (small rationals).
            q = {
                x: {a: Fraction(rng.randint(-3, 3), 2) for a in actions}
                for x in contexts
            }
            specs = []
            # Coverage floor: one record per (context, action) the
            # logger supports, so support refusal only occurs in the
            # genuinely impossible case pi(a|x)>0 while mu(a|x)=0.
            for x in contexts:
                for a in actions:
                    if mu[x][a] > 0:
                        specs.append((
                            x, a,
                            Fraction(rng.randint(-4, 4), rng.randint(1, 4)),
                            mu[x][a], 0, rng.randint(1, 8),
                        ))
            n = rng.randint(1, 40)
            for _ in range(n):
                x = contexts[rng.randrange(len(contexts))]
                a = _sample_fraction(rng, mu[x])
                r = Fraction(rng.randint(-4, 4), rng.randint(1, 4))
                td = rng.randint(0, 9)
                to = td + rng.randint(0, 6)
                specs.append((x, a, r, mu[x][a], td, to))
            cutoff = rng.randint(2, 12)
            clip = None if rng.random() < 0.5 else Fraction(rng.randint(1, 4))
            with_q = rng.random() < 0.8
            try:
                res = evaluate_policy(
                    specs_to_records(specs),
                    policy_to_float(pi),
                    cutoff,
                    q_hat=policy_to_float(q) if with_q else None,
                    clip=float(clip) if clip is not None else None,
                )
            except ZeroSupportError:
                # Reference oracle only applies when support suffices;
                # skip structurally-incompatible draws (tested directly
                # elsewhere).
                continue
            except NoMatureRecordsError:
                # The oracle must agree that nothing was mature.
                self.assertEqual(
                    estimate(specs, pi, cutoff, q=q if with_q else None)["n_mature"],
                    0,
                )
                continue
            exp = estimate(
                specs, pi, cutoff,
                q=q if with_q else None, clip=clip,
            )
            assertCloseFraction(
                self, res.ips, exp["ips"], tol=1e-9,
                label=f"ips trial{trial}",
            )
            assertCloseFraction(
                self, res.snips, exp["snips"], tol=1e-9,
                label=f"snips trial{trial}",
            )
            if with_q:
                assertCloseFraction(
                    self, res.dr, exp["dr"], tol=1e-9,
                    label=f"dr trial{trial}",
                )
            assertCloseFraction(
                self,
                res.ess_clipped if clip is not None else res.ess_unclipped,
                exp["ess"],
                tol=1e-9,
                label=f"ess trial{trial}",
            )


def _random_dist(rng: random.Random, keys, denom: int) -> dict:
    """Random distribution as exact Fractions summing to 1."""
    cuts = sorted(rng.sample(range(1, denom), len(keys) - 1))
    bounds = [0, *cuts, denom]
    return {
        k: Fraction(bounds[i + 1] - bounds[i], denom)
        for i, k in enumerate(keys)
    }


def _sample_fraction(rng: random.Random, dist: dict):
    u = rng.randrange(20) / 20.0
    cum = 0.0
    for k, p in dist.items():
        cum += float(p)
        if u < cum:
            return k
    return next(reversed(dist))


if __name__ == "__main__":
    unittest.main()
