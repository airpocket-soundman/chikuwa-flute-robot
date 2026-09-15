import numpy as np

from flute_rl import MLP, FluteEnv, rollout
from flute_rl.feedback import FeedbackPolicy, FeedbackResidualPolicy


def test_feedback_improves_first_take():
    env = FluteEnv(progress=0.8, feedback=True)
    seeds = range(1_000_000, 1_000_012)
    blind = [rollout(env, FeedbackPolicy(fb_gain=0.0), seed=s)["mean_abs_cents"] for s in seeds]
    heard = [rollout(env, FeedbackPolicy(fb_gain=0.05), seed=s)["mean_abs_cents"] for s in seeds]
    assert np.nanmean(heard) < 0.7 * np.nanmean(blind)


def test_zero_network_matches_classic_feedback():
    env = FluteEnv(progress=0.8, takes=2, feedback=True)
    net = MLP(FeedbackResidualPolicy.feature_dim(), 3)
    a = rollout(env, FeedbackPolicy(), seed=7)
    b = rollout(env, FeedbackResidualPolicy(FeedbackPolicy(), net), seed=7)
    assert np.allclose(a["actions"], b["actions"])
