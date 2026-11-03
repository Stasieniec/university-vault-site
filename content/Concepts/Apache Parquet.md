---
type: concept
aliases: [Parquet, Parquet file format]
course: [EPRM]
tags: [data-engineering, file-formats, mlops]
status: complete
---

# Apache Parquet

## Definition

> [!definition] Apache Parquet
> **Apache Parquet** is an open-source **columnar** file format for analytical data. A file is split into **row groups** (blocks of rows); within each row group every column is stored as its own **column chunk**, made of **pages** that are encoded and compressed. A **footer** at the end of the file holds the schema and metadata for every row group and column chunk.

## Intuition

> [!intuition] Shelve by topic when people ask by topic
> If people ask for single records by ID, you keep each record together (a row layout, like a CSV or a backend database). If they ask for "the ages of all users", you want all ages in one place. Analytics and ML training read a few columns of many rows, so Parquet keeps each column together: a query reads only the columns it uses, and similar values sitting side by side compress well.

## Formulation

No formulas. The file layout:

```
"PAR1"                              4-byte magic number
Row group 0
  Column chunk a: page 0, page 1, ...   (page = header, repetition levels,
  Column chunk b: ...                     definition levels, values)
Row group 1 ...
Footer: version, schema, key/value pairs;
        per row group and column: type, path, encodings, codec,
        number of values, page offsets, compressed/uncompressed size
footer length (4 bytes), "PAR1"
```

A reader reads the end of the file first to find the footer, then seeks directly to the column chunks it needs.

## Key Properties / Variants

- **Column pruning.** Only the columns a query uses are read from disk.
- **Compression.** Values of one column share a type and often repeat, so encodings such as dictionary encoding (a small table of distinct values plus integer codes) and run-length encoding (value, run length) work well, followed by a codec such as Snappy.
- **Statistics and skipping.** Column chunk metadata includes statistics such as minimum and maximum values, so a filter like `amount > 100` can skip row groups that cannot match. Spark shows this as `PushedFilters` in a physical plan.
- **Schema and types** are stored in the file, unlike CSV.
- **Repetition and definition levels** encode nested fields and nulls.
- **Partitioned datasets.** A dataset is usually a folder of Parquet files split by a column's value, such as `data/trx/chck_date_day=2019-01-01/part-00000-....snappy.parquet`. A filter on the partition column skips whole folders, which also makes reading only new days cheap for incremental pipelines.

## Connections

- The storage side of the analytics workload in [[OLTP vs OLAP]]
- Read and written by [[Apache Spark]], whose [[Lazy Evaluation]] lets it push filters and column selection into the Parquet scan

## Appears In

- [[EPRM-L02 - Distributed Data Processing]]
