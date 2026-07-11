# LLM-as-judge evaluation results

F2 score weights recall 4x over precision.

| id | category | caught | section_correct | severity_matched | false_positives |
|---|---|---|---|---|---|
| sec-01-sql-injection | security | True | True | True | 0 |
| sec-02-command-injection | security | True | True | True | 0 |
| sec-03-hardcoded-secret | security | True | True | True | 0 |
| sec-04-insecure-deserialization | security | True | True | True | 1 |
| sec-05-path-traversal | security | True | True | True | 0 |
| perf-01-quadratic-duplicate-check | performance | True | True | True | 0 |
| perf-02-off-by-one-pagination | performance | False | False | False | 1 |
| perf-03-exponential-fibonacci | performance | True | True | True | 0 |
| perf-04-faulty-memoization-key | performance | True | True | True | 0 |
| perf-05-race-condition-counter | performance | False | False | False | 1 |
| struct-01-duplicated-validation-logic | structural | True | False | True | 0 |
| struct-02-magic-numbers-naming | structural | True | True | True | 0 |
| struct-03-silent-exception-handling | structural | True | True | True | 1 |
| struct-04-deep-nesting-guard-clauses | structural | False | False | False | 0 |
| struct-05-mixed-logging-dead-code | structural | True | False | True | 2 |