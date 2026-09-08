# TRR-0010 public-development learning curves

This figure is a diagnostics-only view of the four registered fit arms. It plots domain-balanced validation token accuracy at the registered checkpoints `[0, 1000, 2000, 4000, 8000, 12000, 13000]` and marks each arm's registered selected checkpoint. The step-0 line is labelled as the unchanged shared-start reference; the current directional arm's selected checkpoint is step 0.

The plot is labelled **Public development / checkpoint selection**. It uses only the public diagnostics table below; it performs no fitting, model loading, truth access, label access, source selection, or final-quality reanalysis.

- Input: `../combined_fit_diagnostics_cost_table_v3.json`
- Input SHA256: `f0831f79c4a6e737ebbbe6ad9f837fd61967421b421683ddcb8b6db4b01db8c0`
- Plot source: `plot_learning_curve.py`
- Plot source SHA256: `b98a9332a42e3c5c80ac5239b85d05136ae1a39f4d28be225422abd49c6184a2`
- SVG: `learning_curve_public_development.svg` (SHA256 `8578e0a1a12391c045d4d14ae3977fd78c87a4a9482d93a35e4e937967f413db`)
- PNG: `learning_curve_public_development.png` (SHA256 `5312260993c5b823666b2deb777c30a8fb19e1686ad432c4fca9b09d05db5351`)
- Structured receipt: `figure_receipt.json`

Selected checkpoints are current fixed 8000, current directional 0, expanded fixed 13000, and expanded directional 12000. The input table records `truth_opened=false` and `final_quality_available=false`, so the figure does not support a final truth-quality claim.

Reproduce with:

```bash
PYTHONPATH=. python3 experiments/TRR-0010/report/learning_curve_public_development_v1/plot_learning_curve.py \
  --input experiments/TRR-0010/report/combined_fit_diagnostics_cost_table_v3.json \
  --output-svg experiments/TRR-0010/report/learning_curve_public_development_v1/learning_curve_public_development.svg \
  --output-png experiments/TRR-0010/report/learning_curve_public_development_v1/learning_curve_public_development.png \
  --receipt experiments/TRR-0010/report/learning_curve_public_development_v1/figure_receipt.json
```
