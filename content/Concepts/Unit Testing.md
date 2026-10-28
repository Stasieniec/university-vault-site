---
type: concept
aliases: [unit test, unit tests, FIRST principles]
course: [EPRM]
tags: [software-engineering, testing, mlops]
status: complete
---

# Unit Testing

## Definition

> [!definition] Unit test
> A **unit test** checks one **unit**, an atomic element of code such as a function or a class, in isolation: it calls the unit with known inputs and automatically compares what happens with what should happen.

## Intuition

> [!intuition] How do you know your code works, even without coffee?
> Reading code carefully catches some bugs; a test catches them every time the code changes, automatically. Small tests on small units also tell you exactly where something broke. Many ML bugs produce no error at all: a wrong formula still returns numbers, and the model trains happily on wrong features. Only a test that knows the expected answer notices.

There are two kinds of unit to test:
- A unit that **performs an action** (for example, a function that shuts the machine down). The question is: is the action correct?
- A unit that **returns an output for given parameters** (for example, `sum(a, b)`). The question is: is the output correct for these parameters?

For the second kind, test both the **happy flow** (normal inputs) and the **edge cases** (extreme or invalid inputs):

```python
sum(1, 2) == 3                     # happy flow
sum(-1, -99999998) == -99999999    # edge case: large negative numbers
sum("a", 2)                        # edge case: should raise an error
```

## Anatomy of a unit test

A unit test runs in three phases:

1. **Setup**: prepare the **inputs** and the **mocks** (stand-ins for dependencies).
2. **Collect**: **call** the unit with its parameters and collect the **output**.
3. **Compare**: compare the output with the **expectations**.

Background beyond the course material: the same structure is widely known as Arrange, Act, Assert.

**Dependencies.** A unit often uses other code.
- An **internal dependency** is another unit inside your own system.
- An **external dependency** is something outside it: a database, an API, the file system, the operating system.

A unit test should not depend on the outside world, so external dependencies are **mocked** (replaced by a simulated version that returns fixed, known answers). The test then checks only the unit's own logic, and it runs fast and gives the same result every time.

> [!example] The z-score bug, caught by a test
> ```python
> def get_z_score(data):
>     return (data["amount"] - data["amount"].mean()) / data["amount"].std()
>
> def test_get_z_score():
>     data = pd.DataFrame({"amount": [2.0, 4.0, 6.0]})   # setup: mean 4, std 2
>     result = get_z_score(data)                          # collect
>     assert list(result) == [-1.0, 0.0, 1.0]             # compare
> ```
> The original version without brackets computed $x - \mu/\sigma$ and returns `[0.0, 2.0, 4.0]` here, so the test fails and exposes the bug. The choice of input matters: with `[1.0, 2.0, 3.0]` the standard deviation is 1, both versions return `[-1.0, 0.0, 1.0]`, and the bug slips through.

> [!example] Mocking an external dependency
> ```python
> from unittest.mock import Mock
>
> def fraud_rate(client):
>     rows = client.fetch_transactions()          # external dependency (an API)
>     return sum(r["fraud"] for r in rows) / len(rows)
>
> def test_fraud_rate():
>     client = Mock()                              # setup: simulated API
>     client.fetch_transactions.return_value = [{"fraud": 1}, {"fraud": 0}]
>     assert fraud_rate(client) == 0.5             # collect and compare
> ```

## Key Properties / Variants

Good unit tests follow **FIRST**:
- **Fast**: they run in milliseconds, so you run them constantly.
- **Isolated / Independent**: no test depends on another test or on shared state; any order works.
- **Repeatable**: the same result in every environment and on every run (no reliance on the network, the clock or randomness without a seed).
- **Self-validating**: the test passes or fails by itself; nobody has to inspect output by eye.
- **Thorough**: cover the happy flow and the edge cases.

Background beyond the course material: in Martin's *Clean Code* the T stands for **Timely** (write the tests just before the production code that makes them pass); the EPRM version uses Thorough.

- **Testability depends on design.** A 30-line function that reads a file, computes features and encodes categories cannot be tested piece by piece. Small functions that do one thing, with I/O separated from transformation, can be tested with a five-row table. See [[Clean Code]].
- **Unit versus integration tests.** Unit tests check units in isolation with mocked dependencies; integration tests check that real components work together. A well-organised repository keeps them in separate folders.

## Connections

- Enabled by: [[Clean Code]] (small, single-purpose, side-effect-free functions)
- Part of: [[MLOps]] (the test stage of the DevOps loop; tests run before release to acceptance and production)

## Appears In

- [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]]
