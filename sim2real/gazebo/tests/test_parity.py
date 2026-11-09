"""The C++ model in plugin/Disturbance.hh must equal envs/disturbance/core.py.

Needs a C++ compiler; skipped (with the reason printed by ``pytest -rs``) where there is
none or it cannot build, for example a Mac without the Xcode licence accepted.
"""

import pytest

from sim2real.gazebo.parity import (
    PARITY_TOLERANCE,
    build_parity_binary,
    find_compiler,
    run_parity,
)


@pytest.fixture(scope="module")
def parity_binary(tmp_path_factory):
    compiler = find_compiler()
    if compiler is None:
        pytest.skip("no C++ compiler on PATH")
    try:
        return build_parity_binary(tmp_path_factory.mktemp("parity"), compiler)
    except RuntimeError as e:
        pytest.skip(f"C++ compiler unusable: {str(e).splitlines()[-1][:120]}")


@pytest.mark.parametrize("surface2", [False, True], ids=["wall", "wall_and_deck"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpp_model_matches_numpy(parity_binary, tmp_path, seed, surface2):
    report = run_parity(parity_binary, seed=seed, samples=512, work_dir=tmp_path,
                        surface2=surface2)
    assert report["ok"], report
    assert max(report["max_error"].values()) <= PARITY_TOLERANCE
