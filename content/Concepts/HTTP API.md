---
type: concept
aliases: [API, APIs, REST API, web API, API endpoint, endpoint, HTTP methods, HTTP status codes, status code]
course: [EPRM]
tags: [software-engineering, deployment]
status: complete
---

# HTTP API

## Definition

> [!definition] API, as the EPRM course puts it
> An API is a **service that is always up and ready to receive requests**. On a request, with or without input, it runs a specific process (an **endpoint**) and returns a result, with or without data. HTTP APIs are the most common kind.

## Intuition

> [!intuition] A counter with labelled windows
> Each endpoint is a window with a sign (`/predict`, `/health`). You walk up with a verb (GET: give me something; POST: here is something, process it), maybe a form (the body), and you get back a code saying how it went plus whatever you asked for.

## Request and response

| Client sends | Server returns |
|---|---|
| **Method:** GET, POST, PUT, DELETE | **Status code:** 200, 404, 500, ... |
| **URL / endpoint:** `/predict`, `/health` | **Response data:** JSON, HTML |
| **Params** (optional): `?key=value` in the URL | |
| **Body / payload** (optional): data, often JSON | |

**Methods.** GET fetches and carries its inputs in the URL; POST sends a body for the server to process; PUT creates or replaces a resource at a URL; DELETE removes one.

**Status codes** by first digit: 2xx success (200 OK), 4xx the client's fault (400 bad request, 404 not found, 405 method not allowed), 5xx the server's fault (500 internal server error, 503 unavailable).

## A model behind an API (Flask)

```python
from flask import Flask, request
import pickle

app = Flask(__name__)
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)                 # loaded once, at start-up

@app.get("/health")
def health():
    return {"status": "ok"}                # a dict becomes JSON, status 200

@app.post("/predict")
def predict():
    x = request.get_json()["features"]
    return {"prediction": model.predict([x]).tolist()}
```

Run with `flask --app app run --host 0.0.0.0 --port 5001`. The built-in server is for development; production uses a WSGI server such as gunicorn.

## Key Properties / Variants

- A route answers only GET unless `methods=[...]` (or `@app.post`) says otherwise; another method gets 405.
- A `/health` endpoint lets Docker Compose health checks and orchestrators see whether the service is up.
- Inference is usually stateless, so many API instances can run side by side behind a load balancer (Kästner).

## Connections

- The API pattern in [[ML Deployment Patterns]]
- Packaged with [[Docker]]; the model often comes from [[MLflow]]
- Must encode features exactly as training did: [[Training-Serving Skew]]

## Appears In

- [[EPRM-L04 - Deployment Patterns]]
- [[EPRM-Book Ch10 - Deploying a Model]]
