import pytest

pytest.importorskip("torch")

from envs.structure_hold.baseline import CascadedPd  # noqa: E402
from envs.structure_hold.config import StructureHoldConfig  # noqa: E402
from envs.structure_hold.env import StructureHoldEnv  # noqa: E402
from eval.settle_split import split  # noqa: E402


def test_split_separates_the_settle_from_the_steady_hold():
    cfg = StructureHoldConfig.from_dict({"episode_length_s": 7.0,
                                         "disturbance": {"mean_speed": 0.0}})
    env = StructureHoldEnv(cfg, 8, seed=1)
    r = split(env, CascadedPd(env))
    assert r["alive"] == 1.0
    # The PD starts ~0.3 m off and is in by 2 s, so the steady hold is tighter than the episode.
    assert r["e_0.5s"] > r["e_2s"]
    assert r["p95_steady"] < r["p95_all"]
    assert "e_6s" in r and abs(r["offset_x"]) < 0.1
