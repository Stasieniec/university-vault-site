---
type: concept
aliases: [Airflow, Airflow DAG, Airflow Dag, workflow orchestration, batch workflow]
course: [EPRM]
tags: [mlops, deployment, tools]
status: complete
---

# Apache Airflow

## Definition

> [!definition] Apache Airflow
> **Apache Airflow** is an open-source orchestrator for batch workflows. A workflow is a **DAG** (directed acyclic graph) of **tasks**: Airflow runs each task when the tasks it depends on have finished, on a schedule or when triggered, and records every run.

A **task** is the unit of execution (one script, one container run, one shell command). An **operator** is the template a task is made from: `BashOperator` runs a shell command, `DockerOperator` starts a container. A **Dag run** is one execution of the whole DAG.

## Intuition

> [!intuition] A recipe with arrows
> "When the trigger fires, run A; when A finishes, run B and C; when both finish, run D; if anything fails, do this." Drawn out, that is a graph with arrows from each task to the tasks that wait for it. It must have no cycles, because a task that waits on itself, directly or through others, could never start.

## Defining a DAG

```python
import pendulum
from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    dag_id="nightly_scoring",
    schedule="30 2 * * *",                       # cron: every day at 02:30
    start_date=pendulum.datetime(2026, 11, 1, tz="Europe/Amsterdam"),
    catchup=False,                               # do not replay missed days
    default_args={"retries": 1},
):
    a = BashOperator(task_id="a", bash_command="echo a")
    b = BashOperator(task_id="b", bash_command="echo b")
    c = BashOperator(task_id="c", bash_command="echo c")
    d = BashOperator(task_id="d", bash_command="echo d")
    a >> [b, c] >> d                             # b and c run in parallel
```

- `x >> y` means y runs after x; a list fans out or fans in.
- By default a task runs only when **all** its upstream tasks succeeded (trigger rule `all_success`).
- `schedule` takes a cron string (`minute hour day-of-month month day-of-week`), a preset such as `"@daily"`, or `None` for manual runs only.
- Cron times are read in the DAG's time zone, which is UTC unless `start_date` carries another one.
- In Airflow 3, missed runs between `start_date` and now are not replayed by default (`catchup=False`), and new DAGs start **paused** until someone enables them.

## Key Properties / Variants

- **Triggers:** a schedule, a file arriving, or a change in a database (change data capture), plus manual and API triggers.
- **Failure handling:** retries per task, and callbacks such as `on_failure_callback` for an action when the run fails.
- **Name clash:** "DAG" here is the workflow, which Airflow's own documentation now writes "Dag" to separate it from the data structure.
- Spark builds a DAG of operations too, inside one job; Airflow's DAG sits a level above, ordering whole jobs.

## Connections

- Runs the batch pattern in [[ML Deployment Patterns]]
- Tasks are often [[Docker]] containers
- Splitting a job into tasks with saved outputs is the recovery design in [[EPRM-L02 - Distributed Data Processing]]
- Part of: [[MLOps]]

## Appears In

- [[EPRM-L04 - Deployment Patterns]]
