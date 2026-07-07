"""
Large-scale validation tests for ElasticHashMap.

These tests generate arrays of random (key, value) entries -- exactly what
you'd do to validate this against your own data -- and check two things:

  1. Correctness: every inserted entry must be retrievable afterward, with
     the exact value it was inserted with.
  2. A soft regression check on worst-case probe complexity, so a future
     change that accidentally reintroduces pathological behavior (e.g. the
     "expensive case" cascading, see README > Limitations) gets caught.

The million-entry stress test is marked `@pytest.mark.slow` and skipped by
default -- see conftest.py. Run it explicitly with:

    pytest tests/test_large_scale.py -m slow --run-slow -v -s
"""

import math
import random
import time

import pytest

from elastihash import ElasticHashMap


def generate_random_entries(n: int, seed: int = 42) -> list[tuple[int, int]]:
    """Returns a list of (value, key) pairs with random 63-bit integer keys.

    This is the same shape of data you'd use to validate against your own
    array of entries: swap this out for your real keys/values and the
    tests below still apply unchanged.
    """
    rng = random.Random(seed)
    keys = [rng.getrandbits(63) for _ in range(n)]
    return list(enumerate(keys))


def capacity_for(n_items: int, delta: float) -> int:
    """Smallest sensible capacity for n_items at a given load factor."""
    return int(n_items / (1 - delta)) + 10


# ----------------------------------------------------------------------
# 1. Correctness at increasing (but CI-friendly) scales
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n_items", [1_000, 20_000, 100_000])
def test_random_entries_round_trip(n_items):
    """Every inserted (key, value) pair must be retrievable afterward."""
    delta = 1 / 16
    capacity = capacity_for(n_items, delta)
    table = ElasticHashMap(capacity=capacity, delta=delta, seed=1)
    entries = generate_random_entries(n_items, seed=1)

    for value, key in entries:
        table.insert(key, value)

    assert len(table) == n_items
    assert table.load_factor == pytest.approx(n_items / capacity)

    for value, key in entries:
        assert table.get(key) == value
        assert key in table

    # a key that was never inserted must not be found
    never_inserted_key = -1
    assert table.get(never_inserted_key, "missing") == "missing"
    assert never_inserted_key not in table


@pytest.mark.parametrize("delta", [1 / 8, 1 / 16, 1 / 64])
def test_round_trip_across_load_factors(delta):
    """Correctness must hold near the low, middle, and high end of the
    load factors this structure is meant to support."""
    n_items = 20_000
    capacity = capacity_for(n_items, delta)
    table = ElasticHashMap(capacity=capacity, delta=delta, seed=3)
    entries = generate_random_entries(n_items, seed=3)

    for value, key in entries:
        table.insert(key, value)
    for value, key in entries:
        assert table.get(key) == value


# ----------------------------------------------------------------------
# 2. Soft regression check on worst-case probe complexity
# ----------------------------------------------------------------------
@pytest.mark.parametrize("delta", [1 / 16, 1 / 64])
def test_worst_case_probe_count_stays_bounded(delta):
    """
    The paper's guarantee is O(log(1/delta)) worst-case expected probes.
    This implementation's constants are looser than the paper's exact
    construction (see README > Limitations), so this is a *generous* sanity
    bound, not a tight one -- its purpose is to catch a regression that
    reintroduces unbounded scans, not to verify the paper's tight constant.
    """
    n_items = 20_000
    capacity = capacity_for(n_items, delta)
    table = ElasticHashMap(capacity=capacity, delta=delta, seed=7)
    entries = generate_random_entries(n_items, seed=7)

    for value, key in entries:
        table.insert(key, value)

    max_probes = 0
    for value, key in entries:
        before = table.stats().total_search_probes
        assert table.get(key) == value
        max_probes = max(max_probes, table.stats().total_search_probes - before)

    generous_bound = 50 * math.log2(1 / delta)
    assert max_probes < generous_bound, (
        f"worst-case search probes ({max_probes}) exceeded the generous "
        f"regression bound ({generous_bound:.0f}) for delta={delta}. "
        f"This may indicate a real performance regression."
    )


# ----------------------------------------------------------------------
# 3. Million-entry stress test (opt-in -- see conftest.py)
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_million_entry_stress():
    """
    Insert one million random entries and verify every single one round-trips
    correctly. This is the test to point at your own million-entry array to
    validate the structure yourself.

    Skipped by default. Run explicitly with:

        pytest tests/test_large_scale.py -m slow --run-slow -v -s

    Expect this to take on the order of tens of seconds on a typical laptop.
    """
    n_items = 1_000_000
    delta = 1 / 16
    capacity = capacity_for(n_items, delta)

    table = ElasticHashMap(capacity=capacity, delta=delta, seed=1)
    entries = generate_random_entries(n_items, seed=1)

    t0 = time.perf_counter()
    for value, key in entries:
        table.insert(key, value)
    insert_time = time.perf_counter() - t0

    assert len(table) == n_items
    assert table.load_factor == pytest.approx(n_items / capacity)

    t0 = time.perf_counter()
    mismatches = [key for value, key in entries if table.get(key) != value]
    search_time = time.perf_counter() - t0

    assert not mismatches, f"{len(mismatches)} keys failed to round-trip correctly"

    print(
        f"\n[test_million_entry_stress] n={n_items:,}  "
        f"insert={insert_time:.2f}s ({n_items/insert_time:,.0f} ops/s)  "
        f"search={search_time:.2f}s ({n_items/search_time:,.0f} ops/s)"
    )
