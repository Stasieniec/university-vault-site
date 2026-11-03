---
type: concept
aliases: [Spark, PySpark, SparkSQL, Spark SQL]
course: [EPRM]
tags: [distributed-computing, data-engineering, mlops]
status: complete
---

# Apache Spark

## Definition

> [!definition] Apache Spark
> **Apache Spark** is an open-source engine for distributed data processing. A program written against its DataFrame API (**PySpark** in Python) or in SQL (**SparkSQL**) is turned by a **driver** into a plan of **tasks**, one per data **partition**, which **executors** on worker nodes run in parallel. Spark handles chunking the data, scheduling, moving data between machines and recovering from lost workers.

## Intuition

> [!intuition] Pandas-style code, many machines
> Computing a mean on one machine is one line of pandas. On a hundred machines you would have to chunk the data, send each chunk somewhere, track which nodes are alive, combine partial results and redo work when a node dies. Spark does all of that behind an API that reads like SQL in Python, so you write `data.groupby("category").agg(F.mean("amount"))` and Spark works out how to run it across the cluster.

It is also overkill for data that fits on one machine: it adds a cluster, lazy plans and new failure modes. Use it when data size and latency requirements demand it.

## Formulation

How a computation such as a mean is split: each partition $k$ computes a partial result locally (here its sum $S_k$ and count $n_k$), and only those are combined:

$$\bar{x} = \frac{\sum_{k} S_k}{\sum_{k} n_k}$$

Averaging the partitions' means instead is only correct when all partitions have the same size.

## Key Properties / Variants

- **Architecture.** The **driver** runs your program, builds the plan and schedules tasks. A **cluster manager** (Spark standalone, YARN or Kubernetes) allocates resources. **Executors** on worker nodes run tasks, cache data and exchange data during shuffles. `master("local[2]")` runs all of it on one machine with two cores, so at most two tasks run at a time.
- **Partitions and tasks.** A partition is one slice of the data, and roughly one task processes one partition. When reading files Spark packs small files together and splits big ones (at 128 MB by default).
- **Narrow and wide operations.** Narrow ones (arithmetic on columns, filters, sampling) work on each partition independently. Wide ones (group by, join, sort) need data from many partitions together and cause a **shuffle**: data is written out, sent over the network to the executor that owns its key, and read back. Shuffles usually dominate a job's cost. Aggregations first reduce each partition to partial results per key, which shrinks the shuffle.
- **Lazy evaluation.** Transformations only build a plan; actions (`head`, `collect`, `count`, `write`) run it. This lets Spark reorder filters, read only the needed columns and push filters into the file scan. See [[Lazy Evaluation]].
- **Library stack.** Spark Core and RDDs (resilient distributed datasets, the low-level partitioned collection) at the bottom; Spark SQL and DataFrames, the pandas API on Spark, Structured Streaming and MLlib on top; a Spark Connect client for remote use.
- **Fault tolerance.** A lost partition is recomputed from the plan that produced it. This does not survive the whole application crashing, which is why long pipelines save intermediate results.

Core PySpark patterns:

```python
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.master("local[2]").getOrCreate()
data = (
    spark.read.parquet("data/trx")
    .select(["chck_date_day", "category", "amount"])
    .filter(F.col("amount") > 100)
)
stats = data.groupby("category").agg(F.mean("amount").alias("mean"))
data.explain("formatted")     # show the physical plan, run nothing
stats.show()                  # action: runs the job
```

## Connections

- Usual storage format: [[Apache Parquet]] (columnar, so Spark reads only needed columns and skips row groups)
- Serves the analytics side of [[OLTP vs OLAP]]
- Execution model: [[Lazy Evaluation]]
- Part of the data side of [[MLOps]] pipelines

## Appears In

- [[EPRM-L02 - Distributed Data Processing]]
