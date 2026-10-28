---
type: concept
aliases: [clean coding]
course: [EPRM]
tags: [software-engineering, mlops]
status: complete
---

# Clean Code

## Definition

> [!definition] Clean code
> **Clean code** is code written to be **read, understood and changed** by other people, with intention-revealing names, small single-purpose functions, comments only where they explain why, and no more complexity than today's problem needs. The rules taught in EPRM come from Robert C. Martin's *Clean Code* (2008).

Bjarne Stroustrup (creator of C++), quoted by Martin: "I like my code to be elegant and efficient. The logic should be straightforward to make it hard for bugs to hide, the dependencies minimal to ease maintenance [...]. Clean code does one thing well."

## Intuition

> [!intuition] Is it enough for code to run?
> No. Code that runs can still hide bugs, resist testing and cost every future reader time. Production ML code is read, debugged and extended long after it was written, often by someone else. Clean code makes it obvious what each piece does, so bugs have fewer places to hide and changes are cheap.

> [!example] A bug that runs without complaint
> A feature function computed a z-score as
> `x["amount"] - x["amount"].mean() / x["amount"].std()`.
> Division binds tighter than subtraction, so this is $x - \mu/\sigma$ instead of $(x - \mu)/\sigma$. No exception, no crash: the pipeline completes and silently feeds wrong features to the model. Buried as a lambda inside a 30-line function, it could not be tested on its own. Extracting it into a small, well-named function `get_z_score` made a direct test possible, and the test exposed the bug (see [[Unit Testing]]).

## Key Properties / Variants

**Naming.** "There are only two hard things in Computer Science: cache invalidation and naming things" (Phil Karlton).
- Use **intention-revealing names**: `feature_engineering` says what the function does; `model_preparation` could mean anything.
- **Avoid mental mapping**: the reader should not have to translate `cat` into "category" or remember what `df2` holds. Avoid names that differ by a single character (`encoder`, `encoded`).
- Replace informal names (`data_crunching`) with standard domain terms (`data_preprocessing`).

**Functions.**
- **Keep them small.**
- **Do one thing.** A function that both reads a file and cleans the data has two reasons to change; split I/O from transformation so the transformation can be tested on an in-memory table.
- **Have no side effects**: a function should not secretly change state it was not asked to change.
- **Do not repeat yourself (DRY)**: the same line copy-pasted for several columns becomes a loop over a list of columns, which also prevents copy-paste bugs.
- Avoid **flag arguments** (`run(cross_validate=True)`): a boolean that switches behaviour means the function does two things.
- Remove **shallow wrappers** that only rename a library call (`read_data(path)` that just calls `pd.read_csv(path)`); a wrapper earns its place when it adds something, such as validation or retries.

**Comments.**
- Comments do not make up for bad code.
- Explain **why**, not what. `# CSV encodes strings with quotes, e.g. "'abc'" -> "abc"` explains a data anomaly; `# remove the ' characters` only repeats the code.
- Warn of consequences.
- Do not use a comment when a well-named function or variable can say the same thing.
- No commented-out code: version control remembers old code.

**Keep it simple.**
- Write for today, not for "what we may need in the future".
- Simple is best; simple is elegant.
- Do not try to flex your programming or engineering skills.
- Simplicity is backed by Brooks's "No Silver Bullet" (1987). Background beyond the course material: Brooks separates the essential complexity of a problem from the accidental complexity we add while solving it, and only the accidental part can be removed. The everyday version of "write for today" is often called YAGNI ("you aren't gonna need it").

**Other practices** shown in the refactoring example: replace magic numbers (`123.45`) with named configuration values; keep scripts' logic inside a `main()` function instead of the global scope; understand the algorithms you call (a random forest with `max_depth=1` is an ensemble of single-split stumps that cannot model interactions between features).

**Repository structure.** A clean project separates source code by stage (configs, data processing, feature engineering, modelling, utilities), keeps unit and integration tests in their own folder, and has infrastructure files (Dockerfile, deployment and monitoring configs), a README and a requirements file.

## Connections

- Makes possible: [[Unit Testing]] (small, single-purpose, side-effect-free functions are testable)
- Part of: [[MLOps]] (code that runs in production must be maintainable)

## Appears In

- [[EPRM-L01 - Data Roles, ML Lifecycle, Clean Code and Unit Testing]]
