"""
Benchmarks ElasticHashMap against a classical greedy uniform-probing hash
table, across increasing load factors, counting actual probes (the cost
metric the paper analyzes).

The result to look for: ElasticHashMap's *worst-case* search cost should
grow much more slowly than uniform probing's as the table fills up
(O(log 1/delta) vs O(1/delta)).

Run with:  python examples/benchmark.py
"""

import math
import random

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
        self.total_insert_probes = 0
        self.total_search_probes = 0

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
            self.total_insert_probes += 1
            slot = self._probe(key, j)
            if self.arr[slot] is None:
                self.arr[slot] = _Entry(key, value)
                self.size += 1
                return
        raise RuntimeError("table full")

    def get(self, key, default=None):
        probes = 0
        for j in range(self.n):
            probes += 1
            slot = self._probe(key, j)
            cell = self.arr[slot]
            if cell is None:
                break
            if cell.key == key:
                self.total_search_probes += probes
                return cell.value
        self.total_search_probes += probes
        return default


def bench_uniform(name, table, keys):
    for i, k in enumerate(keys):
        table.insert(k, i)

    ok = True
    max_probes = 0
    for i, k in enumerate(keys):
        before = table.total_search_probes
        if table.get(k) != i:
            ok = False
        max_probes = max(max_probes, table.total_search_probes - before)

    avg_search = table.total_search_probes / len(keys)
    avg_insert = table.total_insert_probes / len(keys)
    print(
        f"  {name:22s}: correct={ok!s:5s}  avg_insert={avg_insert:7.2f}  "
        f"avg_search={avg_search:7.2f}  max_search={max_probes:6d}"
    )


def bench_elastic(name, table, keys):
    for i, k in enumerate(keys):
        table.insert(k, i)

    ok = True
    max_probes = 0
    for i, k in enumerate(keys):
        before = table.stats().total_search_probes
        if table.get(k) != i:
            ok = False
        max_probes = max(max_probes, table.stats().total_search_probes - before)

    s = table.stats()
    avg_search = s.total_search_probes / len(keys)
    avg_insert = s.total_insert_probes / len(keys)
    print(
        f"  {name:22s}: correct={ok!s:5s}  avg_insert={avg_insert:7.2f}  "
        f"avg_search={avg_search:7.2f}  max_search={max_probes:6d}"
    )


if __name__ == "__main__":
    N = 40_000
    for delta in (1 / 16, 1 / 64, 1 / 256):
        num_keys = int(N * (1 - delta))
        keys = [f"key-{i}" for i in range(num_keys)]
        print(f"=== delta={delta:.5f}  (load factor {num_keys/N:.4f}, n={N}) ===")

        bench_uniform("UniformProbing (greedy)", UniformProbingHashMap(N, seed=1), keys)
        bench_elastic(
            "ElasticHashMap (non-greedy)",
            ElasticHashMap(N, delta=delta, probe_patience=2.0, seed=1),
            keys,
        )
        print()
