"""Fixed-input demonstration of delayed-feedback off-policy evaluation.

Run with::

    python demo.py

Everything here is deterministic (a seeded PRNG over a fixed, tabular
toy recommendation problem).  The script:

1. generates a small structured bandit log with delayed rewards;
2. evaluates a target policy at a fixed cutoff and prints IPS, SNIPS
   and DR (clipped and unclipped) plus ESS and maturity accounting;
3. compares against the ground-truth policy value computed directly
   from the known reward table (not via the package under test);
4. triggers two real contract boundaries:
   - an early cutoff with no mature records -> NoMatureRecordsError
     (delayed unobserved rewards are NOT imputed as zero);
   - a target policy reaching for an unsupported action
     -> ZeroSupportError.

Expected runtime: well under a second on a normal laptop.
"""

import random

from delayed_ope import (
    LogRecord,
    NoMatureRecordsError,
    ZeroSupportError,
    evaluate_policy,
)

# ---------------------------------------------------------------------------
# Fixed toy problem: two user contexts, three content actions.
# ---------------------------------------------------------------------------
CONTEXTS = ("news_user", "music_user")
ACTIONS = ("a_politics", "a_sports", "a_music")

# Deterministic true reward table r(x, a).
REWARD_TABLE = {
    "news_user": {"a_politics": 1.0, "a_sports": 0.4, "a_music": -0.5},
    "music_user": {"a_politics": -0.6, "a_sports": 0.2, "a_music": 1.2},
}

# Behaviour/logging policy mu(a|x): explores, but sparsely on the
# cross-interest actions (0.05), which produces large pi/mu weights
# wherever the target reaches onto them.
LOGGING_POLICY = {
    "news_user": {"a_politics": 0.60, "a_sports": 0.35, "a_music": 0.05},
    "music_user": {"a_politics": 0.05, "a_sports": 0.35, "a_music": 0.60},
}

# Target policy pi(a|x): puts real mass on the rarely-logged actions,
# so importance weights reach 4.0 and clipping visibly binds.
TARGET_POLICY = {
    "news_user": {"a_politics": 0.60, "a_sports": 0.20, "a_music": 0.20},
    "music_user": {"a_politics": 0.15, "a_sports": 0.25, "a_music": 0.60},
}

# Deliberately biased q_hat: true reward + 0.3 constant bias.
# DR removes that bias through the importance-weighted correction.
Q_HAT = {
    x: {a: REWARD_TABLE[x][a] + 0.3 for a in ACTIONS} for x in CONTEXTS
}

N_RECORDS = 2400
CLIP = 2.0  # binds on the 4.0 / 3.0 cross-interest weights
CUTOFF = 18.0  # decision times run over [0, 24); some rewards settle after it


def _sample_action(rng: random.Random, probs: dict[str, float]) -> str:
    u = rng.random()
    cum = 0.0
    for a in ACTIONS:
        cum += probs[a]
        if u < cum:
            return a
    return ACTIONS[-1]


def generate_log() -> list[LogRecord]:
    """Deterministically build the delayed-feedback log."""
    rng = random.Random(20260923)
    records: list[LogRecord] = []
    for i in range(N_RECORDS):
        x = CONTEXTS[i % 2]  # exactly balanced contexts
        a = _sample_action(rng, LOGGING_POLICY[x])
        r = REWARD_TABLE[x][a]  # deterministic tabular reward
        decision_time = i * 0.01  # 0.00 .. 23.99
        # Settlement delay: 70% settle in 1 time unit, 30% take 10.
        delay = 1.0 if rng.random() < 0.7 else 10.0
        records.append(
            LogRecord(
                context=x,
                action=a,
                reward=r,
                logged_propensity=LOGGING_POLICY[x][a],
                decision_time=decision_time,
                observed_time=decision_time + delay,
            )
        )
    return records


def ground_truth_value() -> float:
    """V(pi) computed directly from the known tables (independent)."""
    total = 0.0
    for x in CONTEXTS:
        total += sum(TARGET_POLICY[x][a] * REWARD_TABLE[x][a] for a in ACTIONS)
    return total / len(CONTEXTS)


def print_result(res, truth: float) -> None:
    m = res.maturity
    print(f"  cutoff T = {m.cutoff:g}")
    print(
        f"  maturity: {m.mature} mature / {m.immature} immature "
        f"of {m.total} total (immature excluded, never reward 0)"
    )
    print(f"  max unclipped weight pi/mu : {res.max_weight:.4f}")
    print(f"  ESS unclipped / clipped     : "
          f"{res.ess_unclipped:.2f} / {res.ess_clipped:.2f}")
    print("  estimates (unclipped | clipped at M=%g):" % res.clip)
    print(f"    IPS   : {res.ips_unclipped: .5f} | {res.ips: .5f}")
    print(f"    SNIPS : {res.snips_unclipped: .5f} | {res.snips: .5f}")
    print(f"    DR    : {res.dr_unclipped: .5f} | {res.dr: .5f}")
    print(f"  ground truth V(pi) (direct table calc): {truth: .5f}")


def main() -> None:
    print("=" * 70)
    print("Delayed-feedback off-policy evaluation demo")
    print("=" * 70)

    records = generate_log()
    truth = ground_truth_value()

    print("\n[1] NORMAL RESULT: IPS / SNIPS / DR at cutoff", CUTOFF)
    res = evaluate_policy(
        records,
        TARGET_POLICY,
        cutoff=CUTOFF,
        q_hat=Q_HAT,
        clip=CLIP,
    )
    print_result(res, truth)

    print("\n[2a] REJECTION BOUNDARY: cutoff before any reward settles")
    early_cutoff = 0.5  # first decisions settle at t=1, so zero mature
    try:
        evaluate_policy(records, TARGET_POLICY, cutoff=early_cutoff)
    except NoMatureRecordsError as e:
        print(f"  raised {type(e).__name__}: {e}")
        print("  -> no number reported; pending rewards are not set to 0.")

    print("\n[2b] REJECTION BOUNDARY: target wants an unsupported action")
    # Point the target at an action the logger genuinely never offered
    # under that context; the probabilities still sum to 1.
    unsupported = dict(TARGET_POLICY)
    unsupported["news_user"] = {
        "a_politics": 0.7, "a_sports": 0.15, "a_music": 0.05,
        "a_video": 0.10,
    }
    try:
        evaluate_policy(records, unsupported, cutoff=CUTOFF)
    except ZeroSupportError as e:
        print(f"  raised {type(e).__name__}: {e}")

    print("\nDemo finished.")


if __name__ == "__main__":
    main()
