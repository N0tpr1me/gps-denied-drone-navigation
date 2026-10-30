import numpy as np

from sim2real.px4.rig_logging_check import ARRAY_ID, KEYS, analyse, decode_chars, key_name


def chars(names):
    rows = np.zeros((len(names), 10), dtype=np.uint8)
    for i, n in enumerate(names):
        rows[i, :len(n)] = list(n.encode())
    return {f"key[{j}]": rows[:, j] for j in range(10)}


def test_decode_chars_joins_pyulog_columns():
    assert decode_chars(chars(["s_s1", "s_gust"]), "key") == ["s_s1", "s_gust"]


def test_analyse_counts_streams_and_judges_mocap_on_rate():
    sent = {key_name(p, k): 10 for p in ("burst", "spaced") for k in KEYS}
    sent.update(array=10, mocap=100)
    # Burst: only the last key of each tick survives; spaced: all of them.
    names = ["b_gust"] * 10 + [key_name("spaced", k) for k in KEYS] * 10
    topics = {
        "debug_key_value": {"timestamp": np.arange(len(names)), **chars(names)},
        "debug_array": {"timestamp": np.arange(10), "id": np.full(10, ARRAY_ID)},
        # 31 samples over 1 s: about the logger's 30 ms cap, from 100 sent.
        "vehicle_mocap_odometry": {"timestamp": np.linspace(0, 1e6, 31).astype(np.int64)},
    }
    r = analyse(topics, sent)
    assert r["logged"]["b_s1"] == 0 and r["logged"]["b_gust"] == 10
    assert r["findings"] == {"burst_keys_complete": False}
    assert r["mocap_logged_hz"] == 30.0
    assert all(r["checks"].values())


def test_analyse_fails_without_the_topics():
    sent = {key_name(p, k): 10 for p in ("burst", "spaced") for k in KEYS}
    sent.update(array=10, mocap=100)
    r = analyse({}, sent)
    assert not r["checks"]["debug_key_value_logged"] and not r["checks"]["mocap_rate_ok"]
