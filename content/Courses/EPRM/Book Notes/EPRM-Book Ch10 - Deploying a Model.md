---
type: book-chapter
course: 5204EPRM6Y
book: "Machine Learning in Production: From Models to Products"
chapter: 10
sections: ["Model Inference Function", "Feature Encoding", "Model Serving Infrastructure", "Deployment Architecture Trade-offs", "Model Inference in a System", "Documenting Model-Inference Interfaces"]
topics:
  - "[[ML Deployment Patterns]]"
  - "[[HTTP API]]"
  - "[[Training-Serving Skew]]"
  - "[[Feature Store]]"
status: draft
---

# EPRM-Book Ch10: Deploying a Model

> [!abstract] In short
> Using a model for predictions (inference) is technically simple: a side-effect-free function from a feature vector to a prediction, easy to wrap in a REST API, put in a container and scale. The hard parts are around it. Feature encoding must be identical in training and inference, or **training-serving skew** silently breaks predictions. And **where** the model runs (on a server, on the user's device, or split between them) trades off latency, offline use, privacy, cost, hardware limits and how easily the model can be updated and experimented with. Christian Kästner's advice is to decide this like a software architect: gather the requirements, measure the constraints, compare designs.
>
> Week 4 reading, alongside [[EPRM-L04 - Deployment Patterns]].

The chapter's running example: a big tech company builds an **augmented-reality translation app** for smart glasses that translates text in the wearer's view, such as street signs for a tourist. It needs OCR (optical character recognition: finding and reading text in an image), translation, and image stabilisation (keeping overlaid text in place as the head moves). The glasses have little compute and battery but talk to a smartphone over Bluetooth, which can talk to cloud servers over Wi-Fi or mobile data. Each model could run on any of the three.

## Inference is a simple function

At its core, a model's prediction interface is a **side-effect-free function**: a feature vector in, a prediction out, computed with the parameters learned in training (for most models, a long sequence of matrix multiplications, additions and if-statements). Models are stored in a serialised file and loaded into memory for inference.

```python
model = ...  # learn a model, or load a serialised one
def infer(feature1, feature2):
    x = np.array([feature1, feature2]).reshape(1, -1)   # one row, two features
    return model.predict(x)
```

Because inference stores nothing between calls (it is **stateless**, or pure), separate requests can be answered by separate processes with no coordination: the problem is embarrassingly parallel. Any context a prediction needs is passed in with each request.

## Feature encoding: inside or outside the service

Almost every system needs a **feature encoding** step that turns the original input (a JPEG, a GPS coordinate, a sentence, a database row) into the vector the model expects. It can be as simple as scaling an age into [-1, 1] or as involved as database queries or another model (a saliency model, which finds the important region of an image, to crop it first).

The encoding can live **inside the inference service** (the client sends raw inputs) or be **the client's job** (the client sends feature vectors), or be its own service. The choice depends on how expensive encoding is, how often it changes, whether feature vectors are stored, and how many models share it. For OCR, encoding inside the service is convenient and lets the model change without touching clients; sending a compact feature vector instead of a raw image may save network traffic. The same choice applies to **postprocessing**, such as mapping an LLM's free-text answer ("positive", "1", a sentence starting with "positive") to one sentiment label.

### The same encoding for training and inference

> [!definition] Training-serving skew
> Inconsistency between the feature encoding used in training and the one used at inference. Runtime data then does not match the training data, and predictions are wrong.

It ranges from one feature scaled differently to completely scrambled inputs (colour channels in a different order). The standard defence: write the encoding as **one function used by both training and inference**, as simple as a Python function both import, and **version it together with the model**. See [[Training-Serving Skew]].

### Feature stores

Large organisations want two more things: **reuse** of feature code across projects (several models may need "likely local language from GPS coordinates"), and **precomputing or caching** expensive features that many models need. **Feature stores** do both: they catalogue reusable feature-engineering code and compute, precompute and cache feature vectors. Examples: Feast, Tecton, AWS SageMaker Feature Store. See [[Feature Store]].

## Four ways to serve a model

Kästner lists at least four common designs, plus hybrids:

- **Model inference as a library.** The model is embedded in the application and called as a local function. It ships with the binaries or is downloaded. Examples: Tesseract for OCR, TensorFlow Lite for mobile apps, TensorFlow.js in the browser.
- **Model inference as a service.** Likely the most common: a dedicated server answers network requests over a REST API, possibly on special hardware, scaled by running many instances, usually in containers.
- **Batch processing.** If the model is smaller than the data, move the model to the data and run inference in bulk where the data is stored, instead of sending all the data to a service.
- **Cached predictions.** If predictions are expensive but there are few distinct inputs, precompute or cache them and reduce inference to a database lookup (translations of common words).

The lecture's version of this list (batch, API, streaming, edge) is in [[ML Deployment Patterns]].

## A service in the cloud

A **microservice** is an independently deployable process offering a small set of functions to other components over the network, usually through REST-style HTTP requests. It can be built by a separate team with its own technology stack. An inference service for one model fits this exactly: one function, predict. It scales by running many instances behind a **load balancer** (which spreads requests over the instances), added or removed as demand changes. **Serverless** computing goes further: the cloud provider scales the function automatically and charges per request instead of per running machine.

A custom service needs little code. A Python program loads the model **once, when the process starts**, and uses Flask to answer HTTP requests, running inference per request:

```python
from flask import Flask, request, jsonify
app = Flask(__name__)
translation_model = ...  # load model

@app.route('/translate', methods=['POST'])
def pred():
    text_to_translate = request.json.get('text')
    target_lang = request.json.get('target')
    output_array = translation_model(text_to_translate, target_lang)
    return jsonify({"response": decode(output_array)})
```

It is then packaged in a container, with pinned versions and a production server (gunicorn) as the command:

```dockerfile
FROM python:3.8-buster
RUN pip install uwsgi==2.0.20
RUN pip install numpy==1.22.0
RUN pip install tensorflow==2.7.0
RUN pip install flask==2.0.2
RUN pip install gunicorn==20.1.0
COPY models/model.pf /model/
COPY ./serve.py /app/main.py
WORKDIR ./app
EXPOSE 4040
CMD ["gunicorn", "-b 0.0.0.0:4040", "main:app"]
```

Since every inference service has the same ingredients (an inference function that loads the model, encodes features and predicts; code that accepts REST requests; a container specification; optionally monitoring, autoscaling and A/B testing), much of it is automated by tools such as BentoML, Cortex, TFX serving and Seldon Core, and by cloud serverless platforms.

## Where should the model run?

Kästner simplifies the many possible topologies to two places: **servers** controlled by the application provider (own or cloud), and the **user-facing device** (desktop, phone, kiosk, smart-home device, the glasses). Computation can sit entirely on the device, entirely on servers (the device only collects input and shows output), or be split.

**Server-side deployment.**
- Dedicated resources, scaled independently; the only option for large foundation models and models whose memory, energy or GPU needs exceed the device.
- Operating costs land on the provider.
- Needs a network connection: no offline use, and network latency and bandwidth add up, especially with large inputs like images, audio or video. A drone's obstacle avoidance cannot stream video to a server and wait.
- The model can be deployed with the business logic or as a separate service, which can be scaled on its own hardware.

**Client-side deployment.**
- Uses the device's hardware, so no server costs; works offline; no network latency; user data can stay on the device, which is usually seen as protecting privacy.
- The device must be able to run the model: phones often lack GPUs and memory, so inference is slower, and on many different devices latency and user experience vary. Predictable latency needs managed hardware (kiosks, trains) or a narrow set of supported devices.
- Models make app downloads and updates larger, and inference costs battery.
- Anyone with the device can extract the model: a risk for intellectual property and for **adversarial attacks**, since crafting malicious inputs is easier with the model in hand.

**Hybrid deployment** balances these: use the server model only on a fast connection or low battery, or ship a small table of **precomputed predictions** to the client and ask the server only for inputs the table lacks, which also keeps the model itself off the device.

## Updates and experiments depend on where the model runs

**Server-side**, a new model version can replace the old one behind the same endpoint (if inputs and outputs stay compatible) without changing clients, and every user gets the same version. That flexibility is also what makes **experimenting in production** (different models for different users) and **telemetry** (watching request rates and the distribution of inputs and outputs) easy.

**Client-side**, updates mean moving possibly large models to devices with unreliable or no connections. Three typical designs:

- **Offline, never updated:** embedded in cheap hardware such as toys; no feedback from production either.
- **Application updates:** the model ships inside the app and is updated with it, so only tested combinations of app and model reach users.
- **Separate model updates:** models download on their own (one translation model per language), avoiding full app downloads, at the risk of combining app and model versions that were never tested together, especially if feature-encoding code and model drift apart.

Users resist frequent updates (download size, metered connections, forced restarts). Facebook, for example, updated its mobile apps only every two weeks while updating server-side models and code far more often. Experiments are hard client-side: rolling back a failed experiment is slow, feature flags to switch between embedded models make downloads bigger, and telemetry may have to be buffered until the device is online, so it arrives later and less complete.

Requirements can collide: very frequent model updates rule out client-side deployment, while required offline operation rules out guaranteed timely updates.

## Architectural reasoning: the translation glasses

Kästner's process: list the quality requirements, measure the constraints, then compare designs.

**Requirements:** accurate translations; low latency, especially for moving displayed text with the head, to avoid cybersickness (motion sickness from technology, when perception and motion disagree); acceptable offline or on slow connections; no constant streaming of video to the cloud, for privacy; careful use of battery and the glasses' and phones' limited, varied hardware. Cloud cost, frequent model updates and telemetry are low priorities.

**Constraints to measure:** what the hardware can run and at what energy cost; acceptable latency (literature: 200 ms is noticeable as a pause in speech, 20 ms as video delay, about 5 ms is cited as the virtual reality cybersickness threshold); network latency (Bluetooth round trips of 40 to 200 ms); bandwidth (Bluetooth up to 3 Mbit/s against 4 to 10 Mbit/s for low to medium quality video).

**Resulting design to explore:** full server-side OCR breaks the privacy and offline goals, and the weak priority on updates and telemetry makes client-side options plausible. So a hybrid: image stabilisation on the glasses to move text with the view at very low latency, so OCR need not run on every frame; translation only when new text appears, with local caching, and a delay of a second or two acceptable; translation on a server for quality and context (what other users saw at the same sign), plus a smaller on-device model for offline use. Where OCR runs depends on model size, hardware, energy and accuracy, possibly with the glasses cropping text regions and sending only those. The final decision needs experiments.

## Models inside a larger system

**Separate models from business logic.** Systems usually separate concerns, as in a three-tier architecture (presentation, logic, data). ML models should likewise be clearly separated from the business logic that uses their predictions (an inventory system's ordering rules, a train controller's hard-coded safeguards), and the training pipeline and telemetry infrastructure from the non-ML parts. Separation lets each evolve independently and makes it easier to plan for model mistakes.

**Compose several models** in common patterns:

- **Ensembles:** several models trained for the same problem predict in parallel, and the results are combined (average, majority or most conservative), as random forests average trees. A **metamodel** learns how to combine them instead (model stacking), for example which OCR model to trust in which light.
- **Sequential decomposition:** each model handles one part of the task (OCR, then translation), often interleaved with hand-written rules; LLM pipelines that first extract quotes and then answer work the same way.
- **Cascade or two-phase prediction:** a small, fast model handles most inputs and decides whether a larger, slower one is needed, cutting inference cost. Voice assistants detect the wake word on the device and send only the following audio to large cloud models.

## Documenting the inference interface

The model's interface is often the boundary between the team that builds the model and the team that uses it, so it needs documentation beyond the usual technical description (endpoint, input and output formats). Hard to capture but important:

- **intended use cases** (a translation model meant for typical text may fail on slang and should not be trusted with medical diagnoses);
- the **target distribution**, which inputs the model supports (horizontal text only? emoji?), roughly the model's preconditions;
- **expected accuracy**, with how it was evaluated, possibly broken down by input groups;
- **latency, throughput and availability**, for a service as a service level agreement;
- other qualities (explainability, robustness, calibration) and considerations of fairness, safety, security and privacy.

Proposed templates: **Model Cards** (one to two pages: intended use, training and evaluation data, demographic factors, results broken down by demographics, ethics, limitations) and **FactSheets**. In practice, Kästner finds, model documentation is poor and model cards are rarely adopted beyond a dozen flagship examples, and he counts documenting models among the significant open practical challenges.

## What to be able to do

- Explain why inference is easy to scale, and where feature encoding can live.
- Define training-serving skew and the shared-function defence; say what a feature store adds.
- List Kästner's four serving designs and when batch or cached predictions beat a service.
- Compare server-side, client-side and hybrid deployment on latency, offline use, privacy, cost, hardware and updates.
- Describe the three client-side update strategies and why experimentation is easier server-side.
- Name the three model composition patterns and what a model card documents.

## Flashcards

> [!card]- Why is model inference easy to parallelise?
> It is stateless: each request is independent, so separate processes need no coordination.

> [!card]- What are the two places feature encoding can live, relative to an inference service?
> Inside the service, which receives raw inputs, or with the client, which sends feature vectors.

> [!card]- Give one reason to put feature encoding inside the inference service.
> The model can change without changing clients.

> [!card]- What is Kästner's standard defence against training-serving skew?
> One encoding function used by training and inference, versioned with the model.

> [!card]- What two problems do feature stores solve?
> Reusing and discovering feature code, and precomputing or caching expensive features.

> [!card]- What are Kästner's four common designs for serving a model?
> As a library, as a service, batch processing, cached predictions.

> [!card]- When does batch processing beat an inference service, per Kästner?
> When the model is smaller than the data: move the model to the data.

> [!card]- When are cached predictions worth it?
> When predictions are expensive but there are few distinct inputs.

> [!card]- What is a microservice?
> An independently deployable process offering a few functions over the network, usually via REST.

> [!card]- How does serverless deployment differ in scaling and cost?
> The cloud scales it automatically and charges per request.

> [!card]- When is server-side deployment the only option?
> When the model is too large for the device, such as foundation models.

> [!card]- What are the main drawbacks of server-side model deployment?
> No offline use, plus network latency and bandwidth costs.

> [!card]- Name three benefits of client-side model deployment.
> No server costs, works offline, data stays on the device.

> [!card]- Why does client-side deployment raise security concerns?
> Users can extract the model, enabling IP theft and adversarial attacks.

> [!card]- How can precomputed predictions help a hybrid deployment?
> The client looks up common inputs locally and asks the server only for the rest.

> [!card]- Why are experiments and rollbacks easier with server-side models?
> The model behind the endpoint can be swapped without changing clients.

> [!card]- What are the three client-side model update strategies?
> Never updated offline, updated with the app, separate model updates.

> [!card]- What is the risk of updating models separately from the app?
> App and model versions that were never tested together get combined.

> [!card]- What is a cascade (two-phase) prediction?
> A small fast model decides whether a larger slower model is needed.

> [!card]- What is the difference between an ensemble and a metamodel?
> An ensemble combines predictions by a fixed rule; a metamodel learns how to combine them.

> [!card]- Why separate ML models from business logic?
> So each can evolve independently and model mistakes can be planned for.

> [!card]- What does a model card document?
> Intended use, data, evaluation results by group, ethical considerations, limitations.

## Links

- **Course:** [[EPRM - Overview|Course overview]]
- **Lecture:** [[EPRM-L04 - Deployment Patterns]] (batch and API deployment, Airflow, Flask, feature state)
- **Earlier reading by the same author:** [[EPRM-Book Ch22 - Technical Debt]]
- **Concepts:** [[ML Deployment Patterns]] · [[HTTP API]] · [[Training-Serving Skew]] · [[Feature Store]] · [[Docker]] · [[MLOps]]
- **Source:** Christian Kästner, *Machine Learning in Production: From Models to Products* (MIT Press), Chapter 10 "Deploying a Model", online edition, https://mlip-cmu.github.io/book/10-deploying-a-model.html (CC BY-NC-ND 4.0). The Flask and Dockerfile examples are the chapter's.
- **Further (cited in the chapter):** Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019); Arnold et al., "FactSheets" (IBM Journal of Research and Development, 2019); Lakshmanan, Robinson and Munn, *Machine Learning Design Patterns* (O'Reilly, 2020); Yokoyama, "Machine Learning System Architectural Pattern for Improving Operational Stability" (ICSA-C 2019).
