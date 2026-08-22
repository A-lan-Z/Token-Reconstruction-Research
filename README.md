# Token Reconstruction Research

This repository is a clean starting point for research on recovering discrete input tokens from an observed intermediate activation at a declared cut in a transformer.

The canonical problem statement is [`RESEARCH_GOAL_AND_ACCESS_MODEL.md`](RESEARCH_GOAL_AND_ACCESS_MODEL.md).

## Research objective

For an unknown token sequence `x`, let the unavailable prefix of a target model produce the boundary activation

```text
H_c = P_{theta,k}(x)
```

at cut depth `k`. The research goal is to construct a reconstruction procedure that maps the permitted observation and public information to a token estimate `x_hat`, while improving the quality/runtime/memory/complexity trade-off over declared baselines.

The repository deliberately does not prescribe a reconstruction architecture, optimization method, training curriculum, dataset, or hypothesis. Those are research choices to be established by evidence.

## Access model

At reconstruction time, a method may use:

- the activation tensor at the declared cut;
- its attention mask, position IDs, tensor geometry, ordering, and stage-local metadata;
- the local stage parameters available at or after that boundary;
- public model architecture, tokenizer, checkpoint metadata, and public auxiliary data;
- a learned approximation of the unavailable prefix, provided it was learned without prohibited per-record truth access; and
- causal state derived only from records that were completed earlier.

It may not use:

- the source plaintext or its true token IDs, except the declared BOS token;
- a known chat prefix, calibration strip, or any other hidden-token bridge supplied for the current record;
- labels, token-level loss, decoded source output, or any candidate-presence oracle for the current record;
- parameters or a callable implementation of the unavailable live prefix;
- the complete target model as an oracle; or
- evaluation truth before all candidates, adaptation, routing, stopping, timing, and output decisions for that record are immutable.

For the reference Llama tokenizer, the declared BOS token ID is `128000`.

### Causality and evaluation

A decision for record `t` may depend on its permitted observation, the BOS token, tokens reconstructed earlier within that record, public resources, and state committed after completed records `< t`. An update learned from record `s` may first affect record `s + 1`.

Ground truth is opened only after the reconstruction decision and all measured execution decisions have been frozen. It is used for scoring, never for selecting or revising the submitted reconstruction.

## Repository contents

- `src/token_reconstruction/` defines the neutral access contract, safe observation serialization, and a configurable contiguous public-prefix executor.
- `reference/strict_bos/` preserves a strict-BOS historical comparator so new work can measure against it. It is a frozen comparator, not a recommended design or a constraint on new research.
- `resources/` records only public resource identities and provenance.
- `tests/` checks the access boundary and create-only observation handling.

## Deliberate exclusions

This seed contains no target activations, source truth, selected evaluation examples, model weights, datasets, trained lenses or adapters, result files, previous conclusions, experiment plans, search spaces, or research handoffs. Public weights and datasets should be fetched from their authoritative sources and pinned when an experiment actually needs them. Keeping those items out prevents old choices or evidence from silently steering a fresh investigation.

## Quick checks

```bash
python -m pip install -e '.[dev]'
pytest
python reference/strict_bos/preflight_wavefront.py --output /tmp/wavefront-smoke.json
```
