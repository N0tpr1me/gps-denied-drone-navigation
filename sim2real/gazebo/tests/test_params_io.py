import numpy as np
import pytest

from envs.disturbance import DisturbanceConfig, disturbance_force, sample_params
from sim2real.gazebo.params_io import HEADER, read_params, write_params


def test_round_trip_is_exact(tmp_path):
    cfg = DisturbanceConfig(num_gusts=3, k_lin=(0.1, 0.3))
    p = sample_params(cfg, 4, seed=7, wall_point=(0.5, 0.0, 0.0), wall_normal=(-1.0, 0.0, 0.0))
    path = tmp_path / "params.txt"
    write_params(path, p, env=2)
    q = read_params(path)
    assert q.num_envs == 1
    for name in p.__dataclass_fields__:
        np.testing.assert_array_equal(np.asarray(getattr(q, name))[0],
                                      np.asarray(getattr(p, name))[2], err_msg=name)
    t, pos, vel = 3.3, np.array([[-1.0, 0.2, 2.0]]), np.array([[0.5, -0.1, 0.0]])
    np.testing.assert_array_equal(
        disturbance_force(q, t, pos, vel), disturbance_force(p, t, np.tile(pos, (4, 1)),
                                                             np.tile(vel, (4, 1)))[2:3]
    )


def test_free_air_and_no_gusts(tmp_path):
    p = sample_params(DisturbanceConfig(num_gusts=0), 1, seed=0)
    write_params(tmp_path / "p.txt", p)
    q = read_params(tmp_path / "p.txt")
    assert q.gust_start.shape == (1, 0) and q.gust_wind.shape == (1, 0, 3)
    assert not q.has_wall[0]


def test_header_and_shape_errors(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("something else\n")
    with pytest.raises(ValueError, match="header"):
        read_params(bad)
    p = sample_params(DisturbanceConfig(), 1, seed=0)
    write_params(bad, p)
    text = bad.read_text().replace("k_quad 1 ", "k_quad 2 ")
    bad.write_text(text)
    with pytest.raises(ValueError, match="k_quad"):
        read_params(bad)
    with pytest.raises(IndexError):
        write_params(bad, p, env=1)


def test_file_is_the_documented_format(tmp_path):
    p = sample_params(DisturbanceConfig(num_gusts=1, turb_components=4), 1, seed=1)
    write_params(tmp_path / "p.txt", p)
    lines = (tmp_path / "p.txt").read_text().splitlines()
    assert lines[0] == HEADER
    fields = {line.split()[0]: int(line.split()[1]) for line in lines[1:]}
    assert fields["mean_wind"] == 3 and fields["gust_wind"] == 3 and fields["turb_amp"] == 12
