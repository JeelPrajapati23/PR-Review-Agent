# Golden dataset for review-quality evaluation

15 synthetic PR fixtures under `golden_dataset/`, used to evaluate how well the
review panel (`app/agent.py`) catches real, distinct issues -- 5 per category:

| Category | Folder | Targets | Issue type |
|---|---|---|---|
| Security Vulnerability | `golden_dataset/security/` | Security Warden | injection, secrets, insecure deserialization, path traversal |
| Algorithmic Regression | `golden_dataset/performance/` | Performance & Logic Architect | Big-O blowups, off-by-one/edge-case bugs, race conditions |
| Structural & Cleanliness | `golden_dataset/structural/` | Synthesizer / general quality | duplication, naming, dead code, exception hygiene, nesting |

## Fixture layout

Each fixture is a self-contained mini-repo:

```
golden_dataset/<category>/<id>/
  meta.json      # category, target specialist, injected issue, expected findings
  conftest.py    # inserts the fixture root onto sys.path so `from src.x import y` resolves
  src/<name>.py  # the "PR"'s new/changed source file, with one deliberately injected issue
  tests/test_<name>.py  # the PR's own test file, added alongside the source change
```

The source file and its test file are both listed in `meta.json`'s
`added_files`, mirroring a real PR that ships code and tests together.

**By design, every fixture's test suite passes** (verified: `pytest` run
against each of the 15 folders individually, matching exactly how
`mcp_servers/tester_server.py`'s `run_validation_suite` invokes pytest --
bare `pytest`, `cwd=<fixture root>`). The tests only exercise the happy path
or otherwise fail to trip on the injected issue -- each fixture's `meta.json`
has a `why_tests_dont_catch_it` field explaining the specific coverage gap.
That gap is the point: these are issues that pass CI and need a reviewer
(human or agent) to actually catch.

## meta.json schema

- `id`, `category`, `target_specialist` -- which panel member should surface this
- `pr_title` / `pr_description` -- what the fixture would say in a real PR
- `added_files` / `modified_files` -- matches the `pull_request.added_files` /
  `.modified_files` shape `app/schemas.py`'s `PullRequestEvent` expects
- `injected_issue` -- precise description of the planted bug/vulnerability/smell
- `expected_findings.must_flag_file`, `.issue_keywords`, `.severity` -- a loose
  rubric for scoring whether a review caught the issue
- `why_tests_dont_catch_it` -- why the shipped test suite stays green anyway

## Running the panel against the dataset (two phases, quota-safe)

Groq's free-tier daily budget can't absorb all 15 fixtures in one sitting
(see CLAUDE.md's telemetry/budget notes), so generating reviews and scoring
them are two separate, independently runnable scripts:

### 1. `run_reviews.py` -- generate and store, one PR at a time

```bash
python evaluation/run_reviews.py              # resume: reviews whatever's left
python evaluation/run_reviews.py --force       # re-reviews everything
python evaluation/run_reviews.py --only sec-01-sql-injection
```

Feeds each fixture straight into `app.agent.run_pr_review_agent`, the same
way `tests/test_integration.py` drives its single fixture, using the fixture
directory itself as `repo_path` (no copying needed -- the specialists are
read-only). For every fixture:

- Checks `app.telemetry.check_budget_ok()` first -- the exact gate
  `app/tasks.py` uses before a real review -- and **stops the whole run**
  (not just that fixture) the moment the shared daily budget is exhausted,
  rather than burning further calls that would likely just fail.
- Writes that fixture's result to `evaluation/results/<id>.json`
  **immediately**, via a temp-file-then-`os.replace` atomic write, before
  moving to the next fixture -- so a crash, Ctrl-C, or a hard Groq
  rate-limit error partway through never loses already-collected results.
- Skips any fixture that already has a stored result file, so simply
  re-running the script after the daily quota resets resumes automatically
  from wherever it stopped. `--force` re-reviews a fixture that already has
  a result.
- A genuine transient Groq failure (`RateLimitError`/`APIConnectionError`/
  `APITimeoutError`) stores that fixture as `"status": "deferred"` and stops
  the run (further calls would likely also fail); any other exception is
  fixture-specific, gets stored as `"status": "error"`, and the run
  continues to the next fixture instead of stalling on one bad case.

Requires a reachable Redis (checkpointer + telemetry) and a real
`GROQ_API_KEY`/`GITHUB_API_TOKEN`/`GITHUB_WEBHOOK_SECRET` via `.env` -- this
makes genuine Groq calls and spends real tokens.

### 2. `judge_results.py` -- Phase 5.5, optional LLM-as-a-judge pass

```bash
python evaluation/judge_results.py              # resume: grades whatever's left
python evaluation/judge_results.py --force       # re-grades everything
python evaluation/judge_results.py --only sec-01-sql-injection
```

`evaluate_results.py`'s keyword matching only measures recall against a
fixed vocabulary list and can't tell a correctly-diagnosed issue phrased in
different words from a genuine miss, and it can't detect false positives at
all. This script grades each stored review semantically instead, using
**Gemini** (`gemini-3.5-flash` via the `google-genai` SDK) as an independent
judge -- deliberately a different provider than the Groq-backed panel being
graded, so grading never touches the Groq budget the rest of this harness is
built around.

- Fully decoupled from `run_reviews.py` and `app/config.py`: it never
  imports `app.agent` or instantiates `app.config.Settings`, so it needs
  only `GEMINI_API_KEY` (env var or `.env`, see `.env.example`) -- not
  `GROQ_API_KEY`/`GITHUB_API_TOKEN`/`GITHUB_WEBHOOK_SECRET`. Exits with a
  clear message (no traceback) if that key is missing.
- Reads each already-stored `evaluation/results/<id>.json`; fixtures with no
  stored review, or whose review didn't complete, are skipped with a note
  to run `run_reviews.py` first.
- For each gradable fixture, asks Gemini for a structured verdict
  (enforced via a Pydantic `response_schema`, not free-form text):
  `true_positive_caught`, `section_correct`, `severity_matched`,
  `false_positive_count`, `reasoning`.
- Same resumability/durability pattern as `run_reviews.py`: each verdict is
  written to `evaluation/results/<id>.judge.json` immediately via an atomic
  write; already-graded fixtures are skipped on the next run (`--force` to
  redo); a Gemini 429 stops the run cleanly, any other grading error is
  stored and the run continues to the next fixture.
- Prints a per-category + overall dashboard (Precision, Recall, an
  **F2 score** that weights recall 4x over precision -- missing a real
  vulnerability is worse than one extra false positive for a
  security/correctness panel -- plus `section_correct%`/`severity_matched%`)
  and writes it to `evaluation/results/_judge_summary.json` /
  `_judge_summary.md`.

### 3. `evaluate_results.py` -- score whatever's stored, any time

```bash
python evaluation/evaluate_results.py
```

Purely local: reads `evaluation/results/*.json` and `manifest.json`, no
Groq/Redis calls, so it's safe to run mid-way through a multi-day
`run_reviews.py` cycle to check progress, and again once all 15 are in for
the final read. A fixture counts as "caught" if the synthesized review text
names both the file in `meta.json`'s `must_flag_file` and at least one of
its `issue_keywords` -- the same "a real finding names its file" grounding
philosophy as `app/agent.py`'s own `_is_grounded()`. Writes
`evaluation/results/_summary.json` (full per-fixture detail) and
`_summary.md` (per-category tables) each run, and prints the markdown to
the console.
