"""bench_test.py's expected policy follows the module's generated header."""

import pytest

pytest.importorskip("pymavlink")

from sim2real.px4 import bench_test  # noqa: E402


def test_default_policy_is_the_compiled_in_one():
    text = bench_test.HEADER.read_text()
    assert f'NAME = "{bench_test.RELEASED_POLICY}"' in text


def test_header_without_a_name_is_an_error(tmp_path):
    bad = tmp_path / "PolicyWeights.hpp"
    bad.write_text("// nothing here\n")
    with pytest.raises(ValueError):
        bench_test.compiled_policy(bad)
