"""
ElastiHash
~~~~~~~~~~

A Python implementation of *elastic hashing*, the open-addressed hash
table from Farach-Colton, Krapivin & Kuszmaul's "Optimal Bounds for Open
Addressing Without Reordering" (arXiv:2501.02305).

    >>> from elastihash import ElasticHashMap
    >>> table = ElasticHashMap(capacity=1000, delta=0.1)
    >>> table["hello"] = "world"
    >>> table["hello"]
    'world'
"""

from .core import ElasticHashMap, ElasticHashMapStats

__all__ = ["ElasticHashMap", "ElasticHashMapStats"]
__version__ = "0.1.0"
