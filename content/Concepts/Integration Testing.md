---
type: concept
aliases: [integration test, integration tests, testing pyramid, end-to-end test, E2E test]
course: [EPRM]
tags: [software-engineering, testing, mlops]
status: complete
---

# Integration Testing

## Definition

> [!definition] Integration test
> An **integration test** checks your code's integration and interaction with other systems: a database, an API, a tracking server, cloud storage, or other components of your own system, run for real instead of mocked.

## Intuition

> [!intuition] The mock always agrees with you
> A [[Unit Testing|unit test]] replaces the database with a mock that returns whatever rows you tell it to. That is what makes it fast and isolated, and also why it can never tell you whether your SQL is right: a typo in a column name passes. An integration test sends the real query to a real database and checks what comes back.

## The testing pyramid

```text
            /\
           /  \        UI / end-to-end tests   (few)
          /----\
         /      \      integration tests
        /--------\
       /          \    unit tests              (many)
      /------------\
```

- **Unit tests** test your own code, with dependencies mocked.
- **Integration tests** test its interaction with other systems.
- **End-to-end (E2E) tests** simulate the user's interaction with the whole system, from input to output.

All must be automated. Moving up, tests get harder to implement and slower, and a failure is harder to trace to its root cause, so there are many unit tests, fewer integration tests and few end-to-end tests.

## Automating integration tests with containers

The FIRST rules (fast, isolated, repeatable, self-validating, thorough) rule out a shared development database: others change its data, and a test machine may not be able to reach it. Instead each dependency runs in its own [[Containerization|container]], started from an existing image and set up exactly as the test expects. [[Docker|Docker Compose]] starts the dependency and a test container together:

```yaml
services:
  db:
    image: postgres:16
    environment: {POSTGRES_USER: test, POSTGRES_PASSWORD: test, POSTGRES_DB: trx}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test -d trx"]
      interval: 2s
      retries: 15
  tests:
    build: .                    # compose file at the repo root here
    volumes: [".:/app"]         # mount the repo so tests/ is in the container
    environment:
      DATABASE_URL: postgresql://test:test@db:5432/trx
    depends_on:
      db: {condition: service_healthy}
    command: ["python", "-m", "pytest", "tests/integration"]
```

- Connect by **service name** (`db`), since `localhost` in the test container is the test container.
- **Wait for readiness** with a health check; a started database is not yet accepting connections.
- **Create known data** in a fixture and remove volumes afterwards (`docker compose down -v`), so every run starts from the same state.
- `docker compose up --abort-on-container-exit --exit-code-from tests` returns the tests' exit code, so CI fails when they fail.

Alternative: the `testcontainers` library starts throwaway containers from inside a pytest fixture.

## Key Properties / Variants

- Integration tests live in their own folder (`tests/integration/`), apart from unit tests, because they need running services and are slower.
- Yesilbek's view: with AI-assisted engineering, testing, especially UI and end-to-end tests, is much more important now.

## Connections

- Builds on: [[Unit Testing]]
- Uses: [[Containerization]], [[Docker]]
- Runs before deployment to: [[Development, Acceptance and Production Environments|acceptance and production]]
- Part of: [[MLOps]] (the test stage)

## Appears In

- [[EPRM-L03 - Containerization and Software Engineering Processes]]
