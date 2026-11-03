---
type: concept
aliases: [OLTP, OLAP, online transaction processing, online analytical processing, online analytics processing]
course: [EPRM]
tags: [data-engineering, databases, mlops]
status: complete
---

# OLTP vs OLAP

## Definition

> [!definition] OLTP and OLAP
> **OLTP (online transaction processing)** is the workload of an application's backend: reading and writing a small number of records per query, fetched by key, with low latency. **OLAP (online analytical processing)** is the workload of analytics: aggregating over large numbers of records that were loaded in bulk. Storage optimised for one is poorly suited to the other.

## Intuition

> [!intuition] One user's profile versus everyone's average
> Showing a user their profile needs one row with many fields: a **wide** result (many columns, few rows), and it must be fast because a person is waiting. Computing the average ride duration per month needs one or two fields from millions of rows: a **tall** result (many rows, few columns), and nobody minds if it takes a minute. Running the second kind of query on a database built for the first causes latency spikes for the app's users.

## Formulation

No formulas. The comparison, from Kleppmann's *Designing Data-Intensive Applications* as used in EPRM:

| Property | OLTP | OLAP |
|---|---|---|
| Main read pattern | Few records per query, fetched by key | Aggregate over many records |
| Main write pattern | Low-latency writes from user input | Bulk import, ETL |
| Primarily used by | End users, via apps | Analytics |
| Data sizes | GBs to TBs | TBs to PBs |

## Key Properties / Variants

- **Storage layout follows the workload.** Row layouts keep each record together, which suits OLTP. Columnar layouts keep each column together, which suits OLAP, because a query reads only the columns it needs and columns compress well. See [[Apache Parquet]].
- **Separate the two.** ML pipelines and analysts copy data out of the backend's OLTP store, through ETL (extract, transform, load), into analytical storage, and run heavy queries there.
- **Writes.** OLTP writes are small, frequent and atomic; OLAP data arrives in bulk loads.

## Connections

- Columnar format for the OLAP side: [[Apache Parquet]]
- Engine for large OLAP-style processing: [[Apache Spark]]
- The data engineering side of [[MLOps]]

## Appears In

- [[EPRM-L02 - Distributed Data Processing]]
- [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]] (the backend database problem in the ride-hailing example)
