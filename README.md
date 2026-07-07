# ElastiHash

**A hash table that probes deep so your queries don't have to.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Tests](https://github.com/yourusername/elastihash/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/elastihash/actions/workflows/tests.yml)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2501.02305-b31b1b.svg)](https://arxiv.org/abs/2501.02305)

ElastiHash is a Python implementation of **elastic hashing** — an
open-addressed hash table that achieves near-optimal search performance
**without ever moving a key once it's been inserted**. No rehashing on
resize, no reordering, no tombstones drifting around. Just a fixed array
and a cleverer probing strategy.

It's built on a genuinely new result in data structures theory, published
in early 2025, that overturned assumptions which had stood since the
1970s and 80s. See [Credits & Citation](#credits--citation) below.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Deletion](#deletion)
- [Testing & Validation](#testing--validation)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Limitations](#limitations)
- [Credits & Citation](#credits--citation)
- [License](#license)

---

## Why this exists

Every open-addressed hash table has to answer one question: when a slot
is taken, where does the key go next? The textbook answer — **uniform
probing**, i.e. try slots in random order and take the first free one —
has been known since 1985 to be *provably optimal* among **greedy**
strategies (Yao, *"Uniform Hashing is Optimal"*). For decades, that
seemed to settle the matter: if you're not allowed to reorder elements
after inserting them, you're stuck with:

- `Θ(log(1/δ))` amortized expected probes per lookup, and
- `Θ(1/δ)` worst-case expected probes per lookup

as the table approaches `1 - δ` full.

In January 2025, Farach-Colton, Krapivin, and Kuszmaul showed this wasn't
the end of the story. By having insertions behave **non-greedily** —
deliberately probing past free slots in an almost-full region before
settling elsewhere — you can drive worst-case expected probe complexity
down to `O(log(1/δ))`, and amortized down to `O(1)`. Both are provably
optimal. They also disproved a second, related conjecture of Yao's about
*greedy* schemes specifically. See the paper for the full story — it's a
clean, readable result even if you're not a specialist.

ElastiHash implements the first of those two schemes: **elastic hashing**
(the paper's Theorem 1).

## Installation

```bash
pip install elastihash
```

Or, from source:

```bash
git clone https://github.com/sid0818/ElasticHashMap.git
cd elastihash
pip install -e .
```

## Quickstart

```python
from elastihash import ElasticHashMap

# Capacity is fixed up front -- elastic hashing never resizes or rehashes.
table = ElasticHashMap(capacity=10_000, delta=1 / 16)

table["apple"] = "a red or green fruit"
print(table["apple"])          # "a red or green fruit"
print("banana" in table)       # False
print(table.get("banana", "not found"))

print(len(table), "/", table.capacity)
```

## How it works

The backing array is split into geometrically shrinking sub-arrays
`A₁, A₂, A₃, …` (each about half the size of the last). Each key gets its
own independent probe sequence *within* each sub-array.

On insertion, instead of greedily taking the first free slot it finds,
the algorithm:

1. If the current sub-array `Aᵢ` still has meaningfully more than a `δ/2`
   fraction free, it probes up to `f(ε) = O(min(log²(1/ε), log(1/δ)))`
   slots deep into `Aᵢ` looking for a free one — deliberately going past
   slots it doesn't end up using.
2. If that budget is exhausted, it falls through to `Aᵢ₊₁`, which — being
   emptier — is likely to place the key quickly.
3. Once `Aᵢ` is essentially full, all future insertions skip it entirely
   and move on to `Aᵢ₊₁`.

The insight: this "wastes" probes at *insertion* time, but that cost
never shows up at *search* time, because search interleaves its checks
across all sub-arrays (weighted by an estimate of `i · j²`) rather than
exhaustively draining one sub-array before trying the next. Most of the
probes an insertion burns become irrelevant to how expensive it later is
to *find* that key.

## Testing & Validation

```bash
pip install -e ".[dev]"
```

**Fast suite** (runs in a few seconds, this is what CI runs on every push):

```bash
pytest tests/ -v
```

This covers correctness round-trips at several scales (1K–100K entries),
across several load factors, plus a soft regression check on worst-case
probe count.

**Validate it yourself against a large random array** — the same thing
you'd do with your own million-entry dataset — is a first-class,
opt-in test rather than a one-off script:

```bash
pytest tests/test_large_scale.py -m slow --run-slow -v -s
```

This generates 1,000,000 random `(key, value)` pairs, inserts all of
them, then asserts every single one round-trips to the exact value it
was inserted with — printing throughput as it goes:

```
[test_million_entry_stress] n=1,000,000  insert=7.21s (138,726 ops/s)  search=13.12s (76,210 ops/s)
```

It's skipped by default (see `tests/conftest.py`) so the everyday suite
stays fast; `--run-slow` opts back in. Swap `n_items` or `delta` in
`tests/test_large_scale.py` to validate at whatever scale or load factor
matters to you.

**Comparative benchmarks** against a classical uniform-probing baseline
and Python's built-in `dict` (not asserted as pass/fail, just numbers to
look at):

```bash
python examples/benchmark.py                    # probe-count comparison, delta sweep
python examples/large_scale_benchmark.py 1000000 # wall-clock comparison at scale
```

## Deletion

```python
del table["apple"]              # raises KeyError if absent
table.delete("apple")           # same thing, returns the removed value
table.pop("apple", "default")   # dict-style pop
table.discard("apple")          # removes if present, never raises
```

Deleting a key doesn't clear its slot back to "empty" — it's marked with
a **tombstone** instead. This matters: the algorithm's search correctness
depends on the fact that hitting a truly-empty slot means *"no key was
ever placed here, or later in this probe sequence."* If deletion cleared
the slot outright, a later search for some *other* key that had been
pushed past this slot (because it was occupied at the time) could
incorrectly stop early and report a false miss. Tombstoned slots are
reused by future insertions, so deleted space isn't wasted.

> **A caveat worth knowing:** the `O(1)` amortized / `O(log 1/δ)`
> worst-case guarantees this whole structure is built around are proven
> in the paper for a purely **insertion-only** sequence. The paper is
> explicit that once deletions enter the picture, even classical schemes
> have "resisted analysis," and the best known bound for mixed
> insert/delete workloads is `δ^-Ω(1)` — nowhere near as good. Deletion
> here is implemented for correctness and everyday usability, not
> because these performance guarantees are known to survive it. Fine for
> insert-heavy workloads with occasional deletes; don't expect the same
> asymptotics shown in the benchmarks below if your workload is
> delete-heavy.

## Benchmarks

Comparing against a classical greedy uniform-probing table
(`examples/benchmark.py`), counting actual probes at increasing load
factors (`n = 40,000`):

| Load factor | Structure | avg insert | avg search | **max search** |
|---|---|---:|---:|---:|
| 93.75% (δ=1/16)  | Uniform probing (greedy) | 2.96 | 2.96 | **94** |
| | **ElasticHashMap** | 3.52 | 5.94 | **56** |
| 98.44% (δ=1/64)  | Uniform probing (greedy) | 4.21 | 4.21 | **288** |
| | **ElasticHashMap** | 4.77 | 9.18 | **107** |
| 99.61% (δ=1/256) | Uniform probing (greedy) | 5.56 | 5.56 | **797** |
| | **ElasticHashMap** | 6.04 | 12.59 | **223** |

**The takeaway:** as the table gets fuller, ElasticHashMap's *worst-case*
search cost grows far more slowly than uniform probing's — roughly
matching the theory's `O(log 1/δ)` vs `O(1/δ)` gap. At 99.6% full, the
worst-case lookup is **~3.6× faster** in the worst observed case.

Run it yourself: `python examples/benchmark.py`

## API Reference

```python
ElasticHashMap(capacity, delta=1/16, probe_patience=2.0, seed=None)
```

| Method | Description |
|---|---|
| `table[key] = value` / `table.insert(key, value)` | Insert a key. Raises `RuntimeError` if at capacity. |
| `table[key]` | Get a value, raising `KeyError` if absent. |
| `table.get(key, default=None)` | Get a value, or `default` if absent. |
| `del table[key]` / `table.delete(key)` | Delete a key, raising `KeyError` if absent. Returns the removed value. |
| `table.pop(key, default=_MISSING)` | Delete and return a value, dict-style. Raises `KeyError` if absent and no default given. |
| `table.discard(key)` | Delete a key if present; never raises. Returns whether anything was removed. |
| `key in table` | Membership check. |
| `len(table)` | Number of keys currently stored. |
| `table.load_factor` | Current fraction of slots filled. |
| `iter(table)` | Iterate over all stored keys. |
| `table.stats()` | Diagnostics: level sizes, probe counters, etc. |

Full docstrings are in [`elastihash/core.py`](elastihash/core.py).

## Limitations

This is a faithful, tested reference implementation — not a
constant-factor-optimized reproduction of the paper's exact construction.
Specifically:

- The paper's precise **batch scheduling** (which provably makes the
  costly "expensive case" essentially never trigger) is approximated here
  with a simpler per-key rule, tuned empirically via `probe_patience`
  rather than derived analytically. This means average-case search cost
  in practice is higher than the paper's proven `O(1)` bound — worst-case
  behavior is where this implementation faithfully reproduces the
  paper's advantage.
- Capacity is fixed at construction time; there's no dynamic resizing.
- This targets clarity and correctness over raw throughput. It's a great
  fit for learning, experimentation, and benchmarking — not (yet) a
  drop-in replacement for `dict` in performance-critical code.

Contributions tightening these up to more closely match the paper's exact
guarantees are very welcome.

## Credits & Citation

This project is a Python implementation of an algorithm devised entirely
by others. All credit for the ideas, proofs, and results belongs to the
paper's authors:

> **Martín Farach-Colton** (New York University)
> **Andrew Krapivin** (University of Cambridge)
> **William Kuszmaul** (Carnegie Mellon University)
>
> *"Optimal Bounds for Open Addressing Without Reordering."*
> arXiv:2501.02305 [cs.DS], 2025.
> https://arxiv.org/abs/2501.02305

If you use ElastiHash in academic work, please cite the original paper,
not this repository:

```bibtex
@article{farach-colton2025optimal,
  title   = {Optimal Bounds for Open Addressing Without Reordering},
  author  = {Farach-Colton, Mart{\'\i}n and Krapivin, Andrew and Kuszmaul, William},
  journal = {arXiv preprint arXiv:2501.02305},
  year    = {2025}
}
```

This implementation has no affiliation with the authors or their
institutions and has not been reviewed by them. Any bugs, simplifications,
or deviations from the paper's exact construction (see
[Limitations](#limitations)) are solely the responsibility of this
repository's contributors.

## License

MIT — see [LICENSE](LICENSE). The license applies to this implementation
only; it makes no claim over the underlying algorithm or the paper it's
based on.
