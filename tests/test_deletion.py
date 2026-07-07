"""
Tests for ElasticHashMap's deletion API: delete(), pop(), discard(), and
__delitem__ (`del table[key]`).

The most important test here is `test_delete_does_not_break_other_keys`,
which exercises exactly the scenario tombstoning exists to protect
against: deleting key A must not cause a false "not found" for a
different key B that happened to be placed further down a probe sequence
because A was in the way at the time.
"""

import random

import pytest

from elastihash import ElasticHashMap


def test_delete_removes_key():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    table["a"] = 1
    table["b"] = 2

    removed_value = table.delete("a")

    assert removed_value == 1
    assert "a" not in table
    assert table.get("a") is None
    assert table["b"] == 2  # unaffected
    assert len(table) == 1


def test_delete_missing_key_raises_keyerror():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    with pytest.raises(KeyError):
        table.delete("nope")


def test_delitem_syntax():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    table["a"] = 1
    del table["a"]
    assert "a" not in table
    with pytest.raises(KeyError):
        del table["a"]  # already gone


def test_pop_returns_value_and_removes():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    table["a"] = 1
    assert table.pop("a") == 1
    assert "a" not in table


def test_pop_missing_key_raises_without_default():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    with pytest.raises(KeyError):
        table.pop("nope")


def test_pop_missing_key_returns_default():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    assert table.pop("nope", "fallback") == "fallback"
    assert table.pop("nope", None) is None


def test_discard_never_raises():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    table["a"] = 1
    assert table.discard("a") is True
    assert table.discard("a") is False  # already gone, no error
    assert table.discard("never-existed") is False


def test_deleted_slot_is_reused_by_a_later_insert():
    """After deletion, the freed capacity should be usable again (not a
    permanent leak), at least in terms of overall size accounting."""
    n = 2000
    table = ElasticHashMap(capacity=n, delta=1 / 4, seed=1)
    for i in range(1000):
        table[f"key-{i}"] = i

    size_before = len(table)
    for i in range(0, 1000, 2):  # delete every other key
        table.discard(f"key-{i}")
    assert len(table) == size_before - 500

    # re-insert new keys; total size accounting must stay correct
    for i in range(500):
        table[f"new-key-{i}"] = i
    assert len(table) == 500 + 500  # 500 survivors + 500 new keys

    # surviving original keys must still be correct
    for i in range(1, 1000, 2):
        assert table[f"key-{i}"] == i
    for i in range(500):
        assert table[f"new-key-{i}"] == i


def test_iteration_excludes_deleted_keys():
    table = ElasticHashMap(capacity=1000, delta=1 / 16, seed=1)
    for i in range(20):
        table[f"key-{i}"] = i
    for i in range(0, 20, 2):
        table.discard(f"key-{i}")

    remaining = set(iter(table))
    expected = {f"key-{i}" for i in range(1, 20, 2)}
    assert remaining == expected


def test_delete_does_not_break_other_keys():
    """
    This is the critical correctness test for tombstoning.

    We insert a large batch of keys (so the table has genuine probe-
    sequence collisions -- i.e. some keys are placed several probes deep
    because earlier slots in their sequence were already taken), delete a
    large random subset, and then verify every *surviving* key is still
    found correctly.

    If deletion cleared slots back to plain "empty" instead of using a
    tombstone, this test would intermittently fail: a survivor that was
    forced past a slot occupied by a since-deleted key would wrongly
    appear "not found," because the search would stop at the now-empty
    slot instead of continuing past it.
    """
    n = 6000
    delta = 1 / 8
    table = ElasticHashMap(capacity=n, delta=delta, seed=123)

    rng = random.Random(123)
    num_keys = int(n * (1 - delta) * 0.8)  # leave headroom
    keys = [f"item-{i}" for i in range(num_keys)]
    for i, k in enumerate(keys):
        table[k] = i

    # delete roughly half of them, in a random order
    to_delete = set(rng.sample(keys, num_keys // 2))
    for k in to_delete:
        table.discard(k)

    survivors = [k for k in keys if k not in to_delete]
    for k in survivors:
        i = keys.index(k)
        assert table.get(k) == i, f"survivor {k!r} was lost after deletions"

    for k in to_delete:
        assert k not in table

    assert len(table) == len(survivors)
