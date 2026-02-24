---
type: coding-assignment
course: IR
week: 1
language: python
concepts:
  - "[[Information Retrieval]]"
status: complete
---

# IR-A00: Assignment 0 — Warmup

## Overview

Familiarization with GitHub Classroom workflow. Implement a TREC run file parser.

**Task:** Implement `load_run_file` function in `modules/dataset.py` to parse TREC-format run files.

**TREC run file format:** Each line: `queryID Q0 docID rank score runID`

**Key details:**
- Individual assignment, PASS/FAIL (must pass 100% of tests)
- Python 3.11, uses pytest for testing
- Submit via git push to main branch

## Takeaway

The TREC run file format is the standard way to represent ranked retrieval results in IR evaluation. Understanding this format is essential for all subsequent assignments.
