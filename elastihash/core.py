"""
elastihash.core
~~~~~~~~~~~~~~~

An implementation of *elastic hashing*: an open-addressed hash table that
achieves O(1) amortized and O(log(1/delta)) worst-case expected probe
complexity **without ever reordering elements** after they're inserted.

This is based on Theorem 1 of:

    Martín Farach-Colton, Andrew Krapivin, William Kuszmaul.
    "Optimal Bounds for Open Addressing Without Reordering."
    arXiv:2501.02305, 2025.

See the project README for a full explanation and citation details.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Iterator, Optional, TypeVar

__all__ = ["ElasticHashMap", "ElasticHashMapStats"]

KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT")

_MISSING = object()

# Sentinel marking a slot that *was* occupied but has since been deleted.
# Distinct from `None` (a slot that has never been used). This distinction
# is what makes deletion possible without breaking search correctness --
# see ElasticHashMap's docstring, "On deletion", for why.
_TOMBSTONE = object()


@dataclass
class ElasticHashMapStats:
    """Point-in-time diagnostics for an :class:`ElasticHashMap` instance."""

    size: int
    capacity: int
    load_factor: float
    num_levels: int
    level_sizes: list[int]
    current_level: int
    expensive_case_hits: int
    total_insert_probes: int
    total_search_probes: int

    @property
    def avg_insert_probes(self) -> float:
        return self.total_insert_probes / self.size if self.size else 0.0


class _Entry(Generic[KT, VT]):
    __slots__ = ("key", "value")

    def __init__(self, key: KT, value: VT) -> None:
        self.key = key
        self.value = value


class ElasticHashMap(Generic[KT, VT]):
    """A fixed-capacity open-addressed hash map using elastic hashing.

    Elastic hashing splits its backing array into geometrically shrinking
    sub-arrays and, on insertion, deliberately probes deeper into a
    nearly-full sub-array before falling back to the next, emptier one.
    That non-greedy behavior is what allows the table to keep worst-case
    expected search cost at ``O(log(1/delta))`` even at very high load
    factors -- something a classical greedy scheme (e.g. plain uniform
    probing) provably cannot do.

    Once a key is placed, it is **never moved** -- there is no rehashing
    and no resizing, the capacity is fixed up front. Deletion *is*
    supported (see "On deletion" below), implemented via tombstoning
    rather than by clearing slots outright.

    On deletion
    -----------
    Deleting a key doesn't clear its slot back to "empty" -- it marks it
    as a *tombstone* instead. This distinction matters: the algorithm's
    search correctness relies on the fact that hitting a truly-empty slot
    means "no key was ever placed here or later in this probe sequence."
    If deletion just cleared the slot, a later search for a *different*
    key that had been forced past this slot (because it was occupied at
    the time) would incorrectly stop early and report a false miss.
    Tombstoned slots are reused by future insertions, so deleted space
    isn't wasted at the slot level.

    **Important caveat.** The O(1) amortized / O(log(1/delta)) worst-case
    guarantees this structure is built around are proven in the paper
    for a purely **insertion-only** sequence. The paper explicitly notes
    that once deletions enter the picture, even classical schemes like
    linear or uniform probing "have resisted analysis," and that the best
    known amortized bound for insert+delete workloads is
    ``delta^-Omega(1)`` -- i.e. *not* the elegant bounds this structure
    otherwise gets you. In short: deletion here is implemented for
    correctness and everyday usability, not because the paper's
    performance guarantees are known to survive it. If your workload is
    insert-heavy with occasional deletes, this is a very reasonable
    trade-off; if it's delete-heavy, don't expect the same asymptotic
    behavior the benchmarks show for insert-only workloads.

    Parameters
    ----------
    capacity:
        Total number of slots in the underlying array (``n``).
    delta:
        Target maximum load factor is ``1 - delta``. Must be in
        ``(0, 1)``. Smaller values allow a fuller table at the cost of
        more probing per operation.
    probe_patience:
        Constant multiplier on the per-level probe budget
        (``f(eps) = probe_patience * min(log^2(1/eps), log(1/delta))``
        from the paper). Higher values trade a bit more insertion-time
        work for a lower chance of falling into the costly rebalancing
        path. ``2.0`` is a good default; see the README benchmarks.
    seed:
        Optional seed for the internal hash salts, for reproducibility.

    Examples
    --------
    >>> table = ElasticHashMap(capacity=1000, delta=0.1)
    >>> table["hello"] = "world"
    >>> table["hello"]
    'world'
    >>> "missing" in table
    False
    >>> del table["hello"]
    >>> "hello" in table
    False
    """

    def __init__(
        self,
        capacity: int,
        delta: float = 1 / 16,
        probe_patience: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not (0 < delta < 1):
            raise ValueError("delta must be in the open interval (0, 1)")

        self._n = capacity
        self._delta = delta
        self._c = probe_patience
        self._rng = random.Random(seed)

        self._levels: list[list[Optional[_Entry[KT, VT]]]] = []
        self._level_salt: list[int] = []
        remaining = capacity
        min_last_level = max(4, int(delta * capacity))
        while remaining > min_last_level:
            size = max(1, remaining // 2)
            self._levels.append([None] * size)
            self._level_salt.append(self._rng.getrandbits(64))
            remaining -= size
        if remaining > 0:
            self._levels.append([None] * remaining)
            self._level_salt.append(self._rng.getrandbits(64))

        self._num_levels = len(self._levels)
        self._free_count = [len(level) for level in self._levels]
        self._current_level = 0
        self._size = 0

        # Instrumentation -- not required for correctness, useful for
        # understanding/benchmarking probe behavior.
        self._total_insert_probes = 0
        self._total_search_probes = 0
        self._expensive_case_hits = 0

    # ------------------------------------------------------------------
    # Dunder / Mapping-like protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def __setitem__(self, key: KT, value: VT) -> None:
        self.insert(key, value)

    def __getitem__(self, key: KT) -> VT:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value  # type: ignore[return-value]

    def __contains__(self, key: object) -> bool:
        return self._locate(key) is not None

    def __iter__(self) -> Iterator[KT]:
        for level in self._levels:
            for entry in level:
                if entry is not None and entry is not _TOMBSTONE:
                    yield entry.key

    def __delitem__(self, key: KT) -> None:
        self.delete(key)

    def __repr__(self) -> str:
        return (
            f"ElasticHashMap(size={self._size}, capacity={self._n}, "
            f"load_factor={self.load_factor:.4f})"
        )

    @property
    def capacity(self) -> int:
        return self._n

    @property
    def load_factor(self) -> float:
        return self._size / self._n

    # ------------------------------------------------------------------
    # Probe sequence: double hashing, guaranteed to cover every slot in
    # a level as j ranges over [0, len(level)) (step chosen coprime with
    # the level size).
    # ------------------------------------------------------------------
    def _probe(self, key: KT, level: int, j: int) -> int:
        arr = self._levels[level]
        size = len(arr)
        salt = self._level_salt[level]
        base = hash((key, salt, "base")) % size
        if size <= 1:
            return base
        step = (hash((key, salt, "step")) % (size - 1)) + 1
        while math.gcd(step, size) != 1:
            step += 1
            if step >= size:
                step = 1
                break
        return (base + j * step) % size

    def _free_fraction(self, level: int) -> float:
        size = len(self._levels[level])
        return self._free_count[level] / size if size else 0.0

    def _probe_budget(self, eps: float) -> int:
        """f(eps) = c * min(log^2(1/eps), log(1/delta)), per the paper."""
        eps = max(eps, 1e-9)
        log_inv_eps = math.log2(1 / eps)
        log_inv_delta = math.log2(1 / self._delta)
        budget = self._c * min(log_inv_eps**2, log_inv_delta)
        return max(1, math.ceil(budget))

    def _place_first_free(
        self, key: KT, value: VT, level: int, max_probes: int
    ) -> bool:
        arr = self._levels[level]
        for j in range(max_probes):
            self._total_insert_probes += 1
            slot = self._probe(key, level, j)
            cell = arr[slot]
            if cell is None or cell is _TOMBSTONE:
                arr[slot] = _Entry(key, value)
                self._free_count[level] -= 1
                return True
        return False

    def _fallback_scan(self, key: KT, value: VT) -> bool:
        """Correctness safety net: a full linear scan across every level.
        Guards against edge cases in the approximate level-sizing; should
        essentially never trigger in practice."""
        for level in range(self._num_levels):
            arr = self._levels[level]
            for slot in range(len(arr)):
                cell = arr[slot]
                if cell is None or cell is _TOMBSTONE:
                    arr[slot] = _Entry(key, value)
                    self._free_count[level] -= 1
                    return True
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, key: KT, value: VT) -> None:
        """Insert ``key`` with ``value``. Raises ``RuntimeError`` if the
        table is at capacity."""
        if self._size >= self._n:
            raise RuntimeError("ElasticHashMap is at capacity")

        i = self._current_level
        while True:
            if i >= self._num_levels:
                if self._fallback_scan(key, value):
                    self._size += 1
                    return
                raise RuntimeError("ElasticHashMap is at capacity")

            is_last_level = i == self._num_levels - 1
            eps1 = self._free_fraction(i)

            if is_last_level:
                if self._place_first_free(key, value, i, len(self._levels[i])):
                    self._size += 1
                    return
                if self._fallback_scan(key, value):
                    self._size += 1
                    return
                raise RuntimeError("ElasticHashMap is at capacity")

            eps2 = self._free_fraction(i + 1)

            # Case 2: level i is essentially exhausted -> permanently advance.
            if eps1 <= self._delta / 2:
                self._current_level = i + 1
                i += 1
                continue

            # Case 3 ("expensive case"): level i+1 already quite full ->
            # force more probing into level i, capped to avoid unbounded
            # worst-case scans (the paper's batch scheduling keeps this
            # essentially never triggered; our simplified per-key
            # scheduling makes it more frequent, so we bound its cost).
            if eps2 <= 0.25:
                self._expensive_case_hits += 1
                budget = min(
                    len(self._levels[i]), 32 * self._probe_budget(self._delta / 2)
                )
                if self._place_first_free(key, value, i, budget):
                    self._size += 1
                    return
                self._current_level = i + 1
                i += 1
                continue

            # Case 1: budgeted probing into level i, else fall through.
            budget = self._probe_budget(eps1)
            if self._place_first_free(key, value, i, budget):
                self._size += 1
                return
            if self._place_first_free(key, value, i + 1, len(self._levels[i + 1])):
                self._size += 1
                return
            self._current_level = i + 1
            i += 1

    def _locate(self, key: KT) -> Optional[tuple[int, int]]:
        """Find the (level, slot) holding `key`, or None if absent.

        Search interleaves probes across levels by an estimate of the
        merged probe index ``phi(i, j) = O(i * j^2)`` from the paper's
        Lemma 1, rather than fully draining one level before trying the
        next -- this is what keeps search cost low even for keys that
        landed deep during insertion.

        A level is abandoned for good the moment a truly-*empty* slot
        (``None``, i.e. never used) is found in it -- that's correct
        because insertion always uses the first available slot within
        whatever budget it had, so nothing can be placed beyond a slot
        that's never been touched. Tombstoned slots (previously
        occupied, since deleted) are different: they must NOT stop the
        search, since some other key may well have been forced past a
        slot that was occupied at the time, even though it's since been
        vacated.
        """
        probes = 0
        active = [True] * self._num_levels

        def cost(level: int, j: int) -> int:
            return (level + 1) * (j + 1) ** 2

        heap = [(cost(level, 0), level, 0) for level in range(self._num_levels)]
        heapq.heapify(heap)

        while heap:
            _, level, j = heapq.heappop(heap)
            if not active[level]:
                continue
            arr = self._levels[level]
            if j >= len(arr):
                active[level] = False
                continue
            probes += 1
            slot = self._probe(key, level, j)
            cell = arr[slot]
            if cell is None:
                active[level] = False
                continue
            if cell is not _TOMBSTONE and cell.key == key:
                self._total_search_probes += probes
                return level, slot
            heapq.heappush(heap, (cost(level, j + 1), level, j + 1))

        self._total_search_probes += probes
        return None

    def get(self, key: KT, default: Any = None) -> Any:
        """Look up ``key``, returning ``default`` if it's not present."""
        found = self._locate(key)
        if found is None:
            return default
        level, slot = found
        return self._levels[level][slot].value  # type: ignore[union-attr]

    def delete(self, key: KT) -> VT:
        """Delete ``key`` and return its value. Raises ``KeyError`` if
        ``key`` is not present.

        Implemented via tombstoning: the slot is marked deleted (not
        cleared to empty) so future searches for *other* keys remain
        correct. See the class docstring's "On deletion" section for why,
        and for an important caveat about the guarantees this structure
        aims for once deletions are involved.
        """
        found = self._locate(key)
        if found is None:
            raise KeyError(key)
        level, slot = found
        value = self._levels[level][slot].value  # type: ignore[union-attr]
        self._levels[level][slot] = _TOMBSTONE
        self._free_count[level] += 1
        self._size -= 1
        return value

    def pop(self, key: KT, default: Any = _MISSING) -> Any:
        """Delete ``key`` and return its value, like ``dict.pop``.

        If ``key`` is absent, returns ``default`` if given, else raises
        ``KeyError``.
        """
        try:
            return self.delete(key)
        except KeyError:
            if default is _MISSING:
                raise
            return default

    def discard(self, key: KT) -> bool:
        """Delete ``key`` if present. Returns whether anything was
        removed -- never raises, unlike :meth:`delete`."""
        try:
            self.delete(key)
            return True
        except KeyError:
            return False

    def stats(self) -> ElasticHashMapStats:
        """Return a snapshot of internal diagnostics, useful for
        benchmarking and understanding probe behavior."""
        return ElasticHashMapStats(
            size=self._size,
            capacity=self._n,
            load_factor=self.load_factor,
            num_levels=self._num_levels,
            level_sizes=[len(level) for level in self._levels],
            current_level=self._current_level,
            expensive_case_hits=self._expensive_case_hits,
            total_insert_probes=self._total_insert_probes,
            total_search_probes=self._total_search_probes,
        )
