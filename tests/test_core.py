import pytest

from elastihash import ElasticHashMap


def make_table(capacity=2000, delta=1 / 16, seed=0):
    return ElasticHashMap(capacity=capacity, delta=delta, seed=seed)


def test_basic_insert_and_get():
    table = make_table()
    table.insert("a", 1)
    table.insert("b", 2)
    assert table.get("a") == 1
    assert table.get("b") == 2
    assert table.get("missing") is None
    assert table.get("missing", "default") == "default"


def test_dunder_protocol():
    table = make_table()
    table["x"] = 42
    assert table["x"] == 42
    assert "x" in table
    assert "y" not in table
    with pytest.raises(KeyError):
        _ = table["y"]


def test_len_and_load_factor():
    table = make_table(capacity=1000)
    assert len(table) == 0
    for i in range(100):
        table[f"key-{i}"] = i
    assert len(table) == 100
    assert table.load_factor == pytest.approx(0.1)


def test_all_inserted_keys_are_retrievable():
    n = 4000
    delta = 1 / 16
    table = make_table(capacity=n, delta=delta, seed=42)
    num_keys = int(n * (1 - delta))
    keys = [f"key-{i}" for i in range(num_keys)]

    for i, k in enumerate(keys):
        table.insert(k, i)

    for i, k in enumerate(keys):
        assert table.get(k) == i


def test_iteration_yields_all_keys():
    table = make_table(capacity=1000)
    inserted = {f"key-{i}" for i in range(50)}
    for k in inserted:
        table[k] = 1
    assert set(iter(table)) == inserted


def test_raises_when_full():
    table = ElasticHashMap(capacity=10, delta=0.5, seed=0)
    inserted = 0
    try:
        for i in range(10):
            table.insert(f"k{i}", i)
            inserted += 1
    except RuntimeError:
        pass
    assert inserted <= 10
    with pytest.raises(RuntimeError):
        for i in range(10, 30):
            table.insert(f"k{i}", i)


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        ElasticHashMap(capacity=0)
    with pytest.raises(ValueError):
        ElasticHashMap(capacity=100, delta=0)
    with pytest.raises(ValueError):
        ElasticHashMap(capacity=100, delta=1)


def test_stats_snapshot():
    table = make_table(capacity=1000)
    for i in range(50):
        table[f"key-{i}"] = i
    stats = table.stats()
    assert stats.size == 50
    assert stats.capacity == 1000
    assert stats.load_factor == pytest.approx(0.05)
    assert stats.num_levels == len(stats.level_sizes)
