# Golden dataset evaluation results

**Overall: 14/15 caught (15/15 reviewed so far)**

## security -- 5/5 caught (5/5 reviewed)

| id | status | caught | matched keywords |
|---|---|---|---|
| sec-01-sql-injection | completed | yes | sql injection, parameteriz |
| sec-02-command-injection | completed | yes | command injection, shell=true |
| sec-03-hardcoded-secret | completed | yes | hardcoded, logging, card number |
| sec-04-insecure-deserialization | completed | yes | pickle, untrusted |
| sec-05-path-traversal | completed | yes | path traversal, sanitiz |

## performance -- 5/5 caught (5/5 reviewed)

| id | status | caught | matched keywords |
|---|---|---|---|
| perf-01-quadratic-duplicate-check | completed | yes | o(n^2), nested loop, set |
| perf-02-off-by-one-pagination | completed | yes | off-by-one |
| perf-03-exponential-fibonacci | completed | yes | exponential, memoiz |
| perf-04-faulty-memoization-key | completed | yes | cache key, stale, incorrect result |
| perf-05-race-condition-counter | completed | yes | race condition, thread-safe |

## structural -- 4/5 caught (5/5 reviewed)

| id | status | caught | matched keywords |
|---|---|---|---|
| struct-01-duplicated-validation-logic | completed | yes | duplicat |
| struct-02-magic-numbers-naming | completed | yes | magic number, readability |
| struct-03-silent-exception-handling | completed | yes | silent, print |
| struct-04-deep-nesting-guard-clauses | completed | no | - |
| struct-05-mixed-logging-dead-code | completed | yes | print |
