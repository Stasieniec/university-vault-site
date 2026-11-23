---
type: concept
aliases: [Deployment Patterns, deployment patterns, model serving, batch deployment, API deployment, streaming deployment, edge deployment, batch prediction, online prediction]
course: [EPRM]
tags: [mlops, deployment]
status: complete
---

# ML Deployment Patterns

## Definition

> [!definition] Four ways to serve or deploy an ML/AI system
> - **Batch:** a predefined workflow that starts on a trigger, runs and exits.
> - **API:** a service that is always up and answers requests from other systems.
> - **Streaming:** a system that listens to one or more message streams (a message bus) and acts on each message.
> - **Edge:** the model is deployed directly on the device that uses it, such as a phone.

Like most MLOps practice, these patterns come from backend engineering. Batch and API are the most common.

## Intuition

> [!intuition] Who starts the work, and when?
> Batch computes predictions ahead of time for everything that exists when it runs, so it is cheap and simple but stale for anything new. An API computes a prediction when someone asks, so it is fresh but has to be up and fast all the time. Streaming reacts to events as they arrive without anyone asking. Edge removes the network entirely by moving the model to the user's device.

## Key Properties / Variants

| | Batch | API | Streaming | Edge |
|---|---|---|---|---|
| Starts on | trigger (schedule, file, CDC) | a client request | a message on a stream | the app on the device |
| Runs | then exits | always up | always listening | on the device |
| Freshness | as of the last run | per request | per event | per request, offline |
| Example | nightly scores for all users | `/predict` endpoint | fraud check per payment event | text recognition on a phone |

- **Batch** suits large volumes where predictions can wait. A new input gets no prediction until the next run. See [[Apache Airflow]] for running batch workflows as DAGs.
- **API** suits on-demand predictions for other systems. See [[HTTP API]].
- **Edge** (client-side deployment in Kästner's terms) works offline, keeps data on the device and has no network latency, but depends on the device's hardware, makes updates harder and exposes the model to whoever has the device.
- Kästner's own list of serving designs is slightly different: model as a **library**, model as a **service**, **batch processing**, and **cached (precomputed) predictions**, plus hybrids.

## Connections

- Part of: [[MLOps]] (the release and deploy steps)
- Implemented with: [[Apache Airflow]] (batch), [[HTTP API]] (API), [[Docker]] (packaging either one)
- Risk in every pattern: [[Training-Serving Skew]]

## Appears In

- [[EPRM-L04 - Deployment Patterns]]
- [[EPRM-Book Ch10 - Deploying a Model]]
