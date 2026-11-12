---
type: concept
aliases: [Dockerfile, Docker Compose, docker compose, docker-compose]
course: [EPRM]
tags: [software-engineering, mlops, deployment, tools]
status: complete
---

# Docker

## Definition

> [!definition] Docker
> **Docker** is the most widely used platform for [[Containerization|containers]]: a **Dockerfile** describes how to build an image, the `docker` command-line tool builds, runs and inspects images and containers, and **Docker Compose** defines and starts several containers together.

## Intuition

> [!intuition] A recipe, a package, a running copy
> The Dockerfile is the recipe, the image is the sealed package it produces, and a container is one running copy of that package. You build once and run the same image on your laptop, in CI and in production.

## The Dockerfile

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/main_training.py"]
```

- `FROM` picks the base image; pin its tag for reproducibility.
- `WORKDIR` sets the directory later paths are relative to.
- `COPY` brings files from the build context into the image; `RUN` executes a command at build time.
- `ENV` sets environment variables; `CMD` is the default command when a container starts.
- Each instruction is a cached **layer**, and a change rebuilds that layer and all later ones. So dependencies are installed before the frequently changing code is copied.
- A `.dockerignore` file keeps data, virtual environments and `.git` out of the build context.

## Everyday commands

```bash
docker build -t name:tag -f path/to/Dockerfile .   # build an image from context "."
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" -e KEY=value name:tag
docker ps -a                                        # containers, including exited ones
docker logs <container>                             # stdout and stderr
docker exec -it <container> bash                    # shell in a running container
docker run -it --entrypoint /bin/bash name:tag      # shell instead of the app
docker image inspect name:tag                       # command, env, working directory
```

`-p host:container` publishes a port, `-v host:container` mounts a folder, `-e` sets a variable, `--rm` deletes the container when it exits.

## Docker Compose

A `docker-compose.yml` lists **services**, each an image or a build, with its environment, ports and dependencies. `docker compose up` starts them on a shared network, where each service is reachable under its **service name** as host name; `docker compose down -v` removes the containers and their volumes. `depends_on` with `condition: service_healthy` waits for a dependency's health check, since a started database is not yet ready for connections. Compose is light orchestration for one machine; Kubernetes orchestrates across many.

## Key Properties / Variants

- **A container lives as long as its main process.** If the script ends or crashes, the container stops.
- **`localhost` inside a container is the container.** Servers must listen on `0.0.0.0` to be reachable through a published port; other containers are reached by service name, the host by `host.docker.internal` on Docker Desktop.
- **Platform.** `docker build --platform linux/amd64` builds for a server's architecture from an Apple Silicon machine.

## Connections

- Implements: [[Containerization]]
- Used for: [[Integration Testing]] (dependencies as Compose services), running an [[MLflow]] tracking server
- Part of: [[MLOps]]

## Appears In

- [[EPRM-L03 - Containerization and Software Engineering Processes]]
