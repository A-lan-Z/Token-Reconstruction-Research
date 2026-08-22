# Token-Reconstruction Research Goal and Access Model

## Goal

Given an intermediate activation emitted at a declared boundary inside a transformer, reconstruct the discrete input-token sequence that produced it. The research seeks methods that improve reconstruction quality, end-to-end runtime, memory use, implementation complexity, or a practically meaningful combination of those properties.

Let `x = (x_0, ..., x_{n-1})` be the unknown token sequence and let

```text
H_c = P_{theta,k}(x)
```

be the activation observed after the unavailable prefix `P` at cut depth `k`. A reconstruction method receives only the information permitted below and emits `x_hat`. The central scientific question is how accurately and efficiently `x_hat` can approximate `x` under that access model.

The goal does not assume a particular inverse model, search procedure, training method, decoder schedule, auxiliary dataset, or online-update mechanism.

## Permitted information

For the current record, reconstruction may use:

- `H_c`, the activation tensor at the declared cut;
- the matching attention mask and position IDs;
- tensor shapes, dtypes, ordering, record identity, cut depth, and other stage-local metadata that does not encode source truth;
- parameters and computation legitimately available at or after the declared boundary;
- public architecture, tokenizer, checkpoint metadata, public checkpoints, and public auxiliary data;
- learned approximations built without prohibited truth access;
- the single declared BOS token identity (`128000` for the reference Llama tokenizer);
- tokens already reconstructed earlier in the same record; and
- persistent state committed after earlier records were completely reconstructed.

## Unavailable information

Before the reconstruction for a record is frozen, the method may not use:

- the source plaintext or its true token IDs, apart from the declared BOS token;
- any additional known source prefix, calibration strip, or hidden-token bridge;
- target labels, token-level correctness, decoded source output, scalar loss derived from source truth, or a signal revealing whether the true token is present among candidates;
- the live parameters or callable computation of the unavailable prefix;
- the complete target model as a queryable oracle; or
- evaluation truth, directly or indirectly, to choose candidates, update state, tune thresholds, route work, stop search, select a model, alter timing, or revise the submitted output.

Public knowledge about a model family or separately obtained public checkpoint is permitted; access to the particular unavailable live prefix that generated `H_c` is not.

## Causal state

For record `t`, every decision must be a function only of its permitted observation, the declared BOS token, its already reconstructed prefix, public resources, and state derived from completed records `< t`. Information learned from record `t` may first affect record `t + 1`.

Within a record, a recovered prefix may be carried forward to reconstruct later positions. It may not be replaced or corrected using subsequently opened truth.

## Evaluation separation

For each evaluated record, candidate generation, adaptation, routing, stopping, timing, and the final reconstructed sequence must be immutable before source truth is opened. Truth is then used only to calculate declared evaluation metrics.

Reported improvements must include every required reconstruction cost and compare like-for-like quality, runtime, memory, and complexity. The intended outcome is a reproducible improvement on a declared baseline or a clearly established Pareto improvement—not a method selected using the evaluated answers.
