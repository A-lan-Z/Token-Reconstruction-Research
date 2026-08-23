# Token-Reconstruction Research Agent Instructions

## Mission

Implement and execute the research described in `RESEARCH_CHARTER.md`.

`RESEARCH_CHARTER.md` is authoritative. Do not silently add scientific
restrictions or treat a task's temporary scope as a permanent method ban.

Any method consistent with the charter is in scope. You may propose stronger
methods, experiments, implementation designs, or changes of direction.

## Read order

Before beginning a task, read:

1. `RESEARCH_CHARTER.md`
2. `coordination/STATE.json`
3. the incoming TRR control packet
4. relevant prior requests, results, manifests, roadmap entries, and code

## Incoming packet

Save every incoming control packet verbatim as:

`coordination/requests/<TASK_ID>.md`

Do this before making substantive changes.

## Execution

For the active task:

1. inspect the repository and environment;
2. form an implementation and experiment plan;
3. implement the task;
4. run appropriate tests and experiments;
5. investigate failures and make reasonable autonomous decisions;
6. record material alternatives, failed attempts, and deviations;
7. produce reproducible evidence;
8. update `coordination/STATE.json`;
9. commit and push a `task/<TASK_ID>` branch;
10. create or update the corresponding pull request.

Do not stop merely to ask about an implementation preference when a reasonable
technical choice can be made and recorded.

## Compute resource preflight

Before a GPU- or memory-intensive run:

- estimate the worst-case tensor geometry, runtime, and peak memory from a
  smaller run, prior evidence, or an explicit calculation;
- compare that estimate with live CPU/GPU resources and retain a documented
  safety margin;
- qualify the largest representative cell before releasing a larger matrix;
- use isolated, restart-safe jobs and a fail-closed resource guard for long
  sweeps, and record preflight, peak-memory, and failure evidence; and
- stop on allocator, driver, thermal, or memory anomalies and reassess before
  continuing.

Do not assume that numerical batching or microbatching is semantically neutral.
Require an output-equivalence check before using it as a resource workaround;
if outputs differ, preserve and exclude the attempt. Once the largest
representative cell passes with a reasonable measured margin, execute the
planned scientific matrix without shrinking it solely because it is large.

## Evidence

For every claimed run, record as applicable:

- task ID;
- full code commit;
- exact command or job submission;
- environment and dependency identifier;
- model/checkpoint identifier;
- dataset or trajectory identifier;
- seeds and configuration;
- hardware;
- start and end timestamps;
- preparation, adaptation, inference, synchronization, and I/O timing;
- peak memory;
- candidate simulations or model evaluations;
- output metrics;
- raw artifact locations and hashes;
- failed or excluded runs and reasons.

Follow the access and truth-opening requirements in `RESEARCH_CHARTER.md`
exactly. Do not introduce additional ones.

## Dual-benchmark comparability

Follow `research/DUAL_BENCHMARK_PROTOCOL.md` for every reconstruction-method
evaluation. A method is not comparison-complete until every active method has
been run in both canonical benchmark setups and the full method-by-benchmark
matrix has been reported.

Register new active methods before execution. Preserve each method's decision
rule and fixed constants. When only geometry or input/output adaptation is
needed, label the result as a benchmark-compatible port and record the exact
differences from the native implementation. Do not silently substitute a port
for an exact native run, pool scores across setups, or call a partial matrix a
comparable overall result.

## Handoff

Write the human-readable result to:

`coordination/results/<TASK_ID>.md`

Write structured evidence to:

`experiments/<TASK_ID>/manifest.json`

The pull request must identify the exact result and manifest paths.

Do not rely on a conversational summary as the only record of work.

## Previous-task disposition

When a packet says `ACCEPT_AND_MERGE`, merge the specified accepted pull
request before starting the next task.

When it says `REVISE`, update the existing task branch and pull request unless
the packet explicitly requires a new branch.

When it says `CLOSE_WITHOUT_MERGE`, preserve the result record and close the
specified pull request without incorporating its implementation into `main`.