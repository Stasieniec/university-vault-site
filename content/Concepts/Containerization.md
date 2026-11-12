---
type: concept
aliases: [containers, container, Docker container, containerisation]
course: [EPRM]
tags: [software-engineering, mlops, deployment]
status: complete
---

# Containerization

## Definition

> [!definition] Container
> A **container** is a packaged application (code, interpreter, libraries, configuration) running as an isolated process on a host machine. It uses the host's resources and operating system kernel, behaves like an isolated machine inside the host, and runs the same way on every host. **Containerization** is deploying software as containers.

## Intuition

> [!intuition] Ship the box instead of the laptop
> A shipping container separates the product from the transport: the carrier moves the box without knowing what is inside. Software has the same need. Machines differ in CPU architecture, operating system, OS configuration and package versions, so code that works on one machine breaks on another ("it works on my machine"); even `print('hello', flush=True)` fails on Python 2.7. Instead of shipping your laptop, you ship a container holding exactly the environment the code was tested in.

## How it works

- An **image** is the package: a read-only, layered file system snapshot (base OS files, runtime, libraries, code) plus metadata such as the default command. A **container** is a running instance of an image with a thin writable layer; what it writes is lost when it is removed, unless it goes to a mounted volume.
- Images are built in **layers**, one per build instruction, cached and shared between images.
- Isolation comes from Linux kernel features: **namespaces** limit what a process can see (processes, network, file system, hostname), **control groups (cgroups)** limit what it can use (CPU, memory).
- Images are pushed to a **registry** and run by an **orchestrator** such as Kubernetes, which schedules them on a pool of machines.

## Containers versus virtual machines

| | Virtual machine | Container |
|---|---|---|
| **Packs** | A whole computer, with a guest OS | The app and its dependencies |
| **Runs on** | A hypervisor | The host OS kernel, via a container engine |
| **Weight** | Heavy, slow to start | Light, starts in seconds |
| **Isolation** | Strong: separate kernels | Weaker: shared kernel |

## Key Properties / Variants

- **Same everywhere, within limits.** Containers share the host kernel, so Linux images need a Linux kernel (Docker Desktop on macOS and Windows runs a Linux VM underneath), and an image is built for one CPU architecture (`arm64` on Apple Silicon, usually `amd64` on servers).
- **Isolation is the default and can be opened on purpose:** mounted folders, published ports and privileged mode give a container access to the host.
- **One container, one application instance.** A system of several services becomes several containers, which Docker Compose or Kubernetes wire together.
- **Reproducibility needs pinned versions.** A floating base image tag such as `python:3-slim` resolves to newer Python over time.

## Connections

- The standard tool: [[Docker]]
- Enables: [[Integration Testing]] with real dependencies in their own containers
- Deployed across: [[Development, Acceptance and Production Environments]]
- Part of: [[MLOps]] (build, release and deploy stages)

## Appears In

- [[EPRM-L03 - Containerization and Software Engineering Processes]]
