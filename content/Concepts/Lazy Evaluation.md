---
type: concept
aliases: [lazy evaluator, lazily evaluated, transformations and actions]
course: [EPRM]
tags: [distributed-computing, programming, mlops]
status: complete
---

# Lazy Evaluation

## Definition

> [!definition] Lazy evaluation
> **Lazy evaluation** delays computing an expression until its result is actually needed. In [[Apache Spark]], **transformations** (`select`, `filter`, `withColumn`, `join`, `groupBy`) only add steps to a plan, and nothing runs until an **action** (`head`, `collect`, `count`, `show`, `write`) asks for a result.

## Intuition

> [!intuition] Hand over the whole shopping list
> If you send someone to the shop item by item, they walk back and forth for each one. Hand them the whole list and they can plan one route. Spark waits for the whole job so that it can optimise the plan before running any of it: filter early, read only the columns that are used, and push conditions down into the file reader.

## Formulation

No formulas. An example in PySpark:

```python
data = spark.read.parquet("data/trx")
doubled = data.select((F.col("amount") * 2).alias("double_amount"))
filtered = doubled.filter(F.col("amount") > 100)
filtered.explain("formatted")   # shows the plan, computes nothing
filtered.head(5)                # action: the job runs now
```

The physical plan, read from the bottom up, scans only the `amount` column with the filter pushed into the Parquet scan, applies the filter, and only then doubles the surviving values, although the code doubled first.

## Key Properties / Variants

- **Optimisations it enables:** reordering operations (filter before projection), column pruning (read only used columns), predicate pushdown (filters applied while reading, using file statistics to skip data).
- **Errors appear late.** A runtime error in a transformation (a failing function, bad data) surfaces only when an action runs, so the failing line may not be the line with the bug. A misspelt column name is still caught at once, when the step is checked against the schema.
- **Repeated actions recompute.** Each action runs the plan again unless the intermediate result is cached or saved.
- `explain()` prints the plan and returns `None`.
- The same idea exists outside Spark, for example Python generators and Polars' lazy API.

## Connections

- Used by: [[Apache Spark]]
- Pays off most with columnar files: [[Apache Parquet]]

## Appears In

- [[EPRM-L02 - Distributed Data Processing]]
