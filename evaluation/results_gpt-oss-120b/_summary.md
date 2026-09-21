# Golden dataset evaluation results

**Overall: 13/15 caught (15/15 reviewed so far)**

## security -- 5/5 caught (5/5 reviewed)

| id | status | caught | matched keywords |
|---|---|---|---|
| sec-01-sql-injection | completed | yes | sql injection |
| sec-02-command-injection | completed | yes | command injection, shell=true |
| sec-03-hardcoded-secret | completed | yes | secret, sensitive data, logging, card number |
| sec-04-insecure-deserialization | completed | yes | pickle, insecure deserialization, arbitrary code execution |
| sec-05-path-traversal | completed | yes | path traversal, resolve |

## performance -- 4/5 caught (5/5 reviewed)

| id | status | caught | matched keywords |
|---|---|---|---|
| perf-01-quadratic-duplicate-check | completed | yes | quadratic, nested loop, set |
| perf-02-off-by-one-pagination | completed | no | - |
| perf-03-exponential-fibonacci | completed | yes | exponential, recursion |
| perf-04-faulty-memoization-key | completed | yes | cache key, memoization, stale |
| perf-05-race-condition-counter | completed | yes | race condition, lock, concurren |

## structural -- 4/5 caught (5/5 reviewed)

| id | status | caught | matched keywords |
|---|---|---|---|
| struct-01-duplicated-validation-logic | completed | yes | duplicat, extract, shared helper, dry |
| struct-02-magic-numbers-naming | completed | yes | magic number, readability, named constant |
| struct-03-silent-exception-handling | completed | yes | silent, swallow, print |
| struct-04-deep-nesting-guard-clauses | completed | no | - |
| struct-05-mixed-logging-dead-code | completed | yes | dead code, print, inconsistent logging |
