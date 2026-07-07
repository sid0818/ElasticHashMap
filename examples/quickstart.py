"""A 60-second tour of ElastiHash."""

from elastihash import ElasticHashMap

# Capacity is fixed up front -- elastic hashing never resizes or rehashes.
table = ElasticHashMap(capacity=10_000, delta=1 / 16, seed=42)

table["apple"] = "a red or green fruit"
table["banana"] = "a yellow fruit"
table.insert("cherry", "a small red fruit")  # same thing as __setitem__

print(table["apple"])
print("banana" in table)
print(table.get("durian", "not found"))
print(len(table), "/", table.capacity, "slots used")

# Deletion
del table["cherry"]
print("cherry" in table)                 # False
print(table.pop("banana"))                # "a yellow fruit", and removes it
print(table.discard("does-not-exist"))    # False, never raises
print(len(table), "/", table.capacity, "slots used")

# Peek at internal diagnostics
stats = table.stats()
print(f"load factor: {stats.load_factor:.4f}")
print(f"levels: {stats.level_sizes}")
