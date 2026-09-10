# Joint-activation proposal prototype

This package is a completed **CPU synthetic mechanism study**, not a Llama experiment.
The proposed joint-window search improves a narrow control, but loses to a wider
ordinary sequential control in accuracy and work. Read RESEARCH_REPORT.md first.

## Files

- `RESEARCH_REPORT.md`: mathematics, executed results, cost control, sources and limits.
- `AGENT4_NATIVE_GATE.md`: optional bounded native proposal gate; not a full rescue task.
- `prototype.py`: cached random causal forward model and discrete window search.
- `PLAN.md`: initial pre-run plan.
- `AMENDMENT.md`: declared implementation/policy change before second cohort.
- `COST_CONTROL.md`: pre-run declaration for the retrospective wider sequential control.
- `results_v1.json`, `results_v2.json`, `results_cost_control.json`: distinct result stages.
- `proposal_probe.json`: retrospective same-state proposal-only ablation.
- `test_prototype.py` and `test_log.txt`: 8 passing local synthetic tests.
- `historical_v1/`: exact executable snapshot of the first stage.

Tested on Python 3.13.5, PyTorch 2.10.0+cpu, float64, one CPU thread. No pretrained
weights or external dataset is needed. The code uses only PyTorch and the standard
library. No package installation/network request is performed by these scripts.
Check `results_v1.json` for the exact originally recorded Python version.

## Re-run

Run in a copy of this directory to avoid overwriting retained evidence:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m unittest discover -s . -p 'test_*.py'
python second_cohort.py
python cost_control.py
python proposal_probe.py
```

First-stage replay:

```sh
cd historical_v1
python run_experiments.py
```

CPU runtimes will vary. Exact stored predictions and operation counts are the primary
replay targets within the declared numeric implementation. This package does not claim
cross-version bitwise reproducibility or equivalent Llama/GPU timing.

The test log retains a harmless PyTorch scalar-conversion warning in the diagnostic
formatter. It is not an unrecorded test failure or an optimizer warning.
