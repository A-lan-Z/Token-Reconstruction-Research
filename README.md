# Token Reconstruction Research

This repository began as a neutral starting point for token-reconstruction
research and now also contains the administrative records used by the
repository-backed TRR review relay.

## Authority

[`RESEARCH_CHARTER.md`](RESEARCH_CHARTER.md) is the sole authoritative research
definition. This README is non-authoritative repository orientation only. It
does not add to, replace, or independently interpret the charter.

The research summary below is non-authoritative. Consult
`RESEARCH_CHARTER.md` directly for the research definition.

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

## Original seed and current state

The original neutral seed contained no target activations, source truth,
selected evaluation examples, model weights, datasets, trained lenses or
adapters, prior scientific findings, or selected research direction. The
current tree additionally contains administrative TRR-0000 requests, results,
delivery evidence, and deterministic bootstrap artifacts.

Those relay records validate orchestration and reproducibility; they are not
scientific findings and do not select a reconstruction method or research
direction. Evaluation traces and truth remain outside this repository.

## Quick checks

```bash
python -m pip install -e '.[dev]'
pytest
python reference/strict_bos/preflight_wavefront.py --output /tmp/wavefront-smoke.json
```
