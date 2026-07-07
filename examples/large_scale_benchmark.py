"""
Large-scale benchmark: generates an array of ~1,000,000 random entries and
compares insert/lookup performance across:

  1. ElasticHashMap        -- this package
  2. UniformProbingHashMap -- classical greedy baseline (same probe model)
  3. dict                  -- Python's built-in hash table (wall-clock only;
                               it rehashes on resize, so it's not a like-for-
                               like probe-complexity comparison, just a
                               real-world reference point)

Run with:  python examples/large_scale_benchmark.py [N_ITEMS]
"""

import math
import random
import sys
import time

from elastihash import ElasticHashMap


class _Entry:
    __slots__ = ("key", "value")

    def __init__(self, key, value):
        self.key = key
        self.value = value


class UniformProbingHashMap:
    """Classical greedy baseline: first-fit over a random probe sequence."""

    def __init__(self, capacity, seed=None):
        self.n = capacity
        self.arr = [None] * capacity
        self.size = 0
        self.salt = random.Random(seed).getrandbits(64)

    def _probe(self, key, j):
        size = self.n
        base = hash((key, self.salt, "base")) % size
        if size <= 1:
            return base
        step = (hash((key, self.salt, "step")) % (size - 1)) + 1
        while math.gcd(step, size) != 1:
            step += 1
            if step >= size:
                step = 1
                break
        return (base + j * step) % size

    def insert(self, key, value):
        for j in range(self.n):
            slot = self._probe(key, j)
            if self.arr[slot] is None:
                self.arr[slot] = _Entry(key, value)
                self.size += 1
                return
        raise RuntimeError("table full")

    def get(self, key, default=None):
        for j in range(self.n):
            slot = self._probe(key, j)
            cell = self.arr[slot]
            if cell is None:
                return default
            if cell.key == key:
                return cell.value
        return default


def generate_random_entries(n, seed=42):
    """An array of n (key, value) pairs with random 63-bit integer keys."""
    rng = random.Random(seed)
    keys = [rng.getrandbits(63) for _ in range(n)]
    return list(enumerate(keys))  # [(value, key), ...] -- value = original index


def bench(name, insert_fn, get_fn, entries):
    t0 = time.perf_counter()
    for value, key in entries:
        insert_fn(key, value)
    t_insert = time.perf_counter() - t0

    t0 = time.perf_counter()
    ok = True
    for value, key in entries:
        if get_fn(key) != value:
            ok = False
    t_search = time.perf_counter() - t0

    print(
        f"  {name:22s}: correct={ok!s:5s}  "
        f"insert={t_insert:7.2f}s ({len(entries)/t_insert:,.0f} ops/s)  "
        f"search={t_search:7.2f}s ({len(entries)/t_search:,.0f} ops/s)"
    )


if __name__ == "__main__":
    n_items = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    delta = 1 / 16
    capacity = int(n_items / (1 - delta)) + 10

    print(f"Generating {n_items:,} random entries...")
    entries = generate_random_entries(n_items)

    print(f"\n=== {n_items:,} entries, capacity={capacity:,}, delta={delta:.4f} ===\n")

    elastic = ElasticHashMap(capacity=capacity, delta=delta, seed=1)
    bench("ElasticHashMap", elastic.insert, elastic.get, entries)

    uniform = UniformProbingHashMap(capacity=capacity, seed=1)
    bench("UniformProbing", uniform.insert, uniform.get, entries)

    d = {}
    bench("dict (builtin)", lambda k, v: d.__setitem__(k, v), d.get, entries)

    print(
        "\nNote: dict's numbers reflect CPython's real, highly-optimized C\n"
        "implementation (which is also allowed to rehash/reorder on resize).\n"
        "It's included as a real-world reference point, not a fair\n"
        "probe-complexity comparison -- see the README for why."
    )
