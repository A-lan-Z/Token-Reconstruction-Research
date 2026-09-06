# TRR-0006 precision preflight handoff

This is a provisional, pre-truth, CPU-only adequacy analysis. It did not run a model, open new truth, fit/select a method, change global `coordination/STATE.json`, or register a final sample size.

The published TRR-0005 scorer binds the exact endpoint as `U_CP(g;n, alpha) - L_CP(h;n, alpha)` with `alpha = 0.05/32 = 0.0015625`; the practical exact margin is 5 pp. The token margin is 0.5 pp with the registered source-bootstrap upper tail `0.05/16 = 0.003125`.

Published causal-versus-trained-diagonal Finance counts are P0 `g/h = 5/4` and synthetic-LoRA `5/6` out of 128; Pile is `1/1` for both targets. At n=1024, the Finance P0 plug-in count `(40,32)` gives `gain U=6.03768 pp`, `loss L=1.75195 pp`, and `B=4.28573 pp`. This plug-in is descriptive, not a decision probability.

Exact multinomial planning probabilities for `B <= 5 pp`:

| true `(p_gain,p_loss)` | n=768 | n=1024 | n=1536 | n=2048 |
|---|---:|---:|---:|---:|
| Finance P0 observed `(5/128,4/128)` | 0.571 | 0.805 | 0.977 | 0.998 |
| Finance LoRA observed `(5/128,6/128)` | 0.892 | 0.981 | 1.000 | 1.000 |
| higher discordance `(0.08,0.05)`, net +3 pp | 0.006 | 0.011 | 0.030 | 0.062 |
| higher discordance `(0.10,0.05)`, net +5 pp | 0.000028 | 0.000029 | 0.000029 | 0.000029 |
| higher discordance `(0.10,0.10)`, net 0 pp | 0.173 | 0.327 | 0.642 | 0.853 |

The n=1024 plan therefore has an explicit approximately 80% per-target exclusion probability for the observed Finance P0 rates, with higher probability for LoRA and effectively 100% for observed Pile rates. This supports a provisional adequacy recommendation only under those observed-rate scenarios. It is not uniformly adequate: a true net +2 pp case `(0.06,0.04)` gives exclusion probability about 0.139 at n=1024, while a true net +5 pp case `(0.10,0.05)` is almost never excluded (as required if it is a real useful effect). The point estimate reaches +5 pp about 0.489 in the latter case; this is descriptive resolution, not positive-evidence power.

Target dependence is retained. Using the published per-record 3×3 event patterns, Finance target-joint exclusion at n=1024 is about 0.799 in a 20,000-draw paired-category sensitivity simulation (95% MC half-width 0.0056), while the no-independence Frechet range from exact marginal probabilities is `[0.786, 0.805]`. Pile is effectively 1.0. Do not report a product across P0 and LoRA as a guaranteed joint power claim; if all four cells are combined, state any cross-domain independence assumption or use a global Frechet bound.

Artifacts:

- `sample_size_sensitivity.py` SHA-256 `412a002530929dfe107e68943a82360a91cc8038e746846a69761efff650c3a4`
- `sample_size_sensitivity.json` SHA-256 `e17c13cc79cc16e28cc1f66328651be558c09b80f4c13c8621dfa5b56e4f7034`
- `sample_size_sensitivity.md` SHA-256 `ee75c2073963ec252e92a132309b6854ae3a16ffac81a11c9f2332bd68d66db8`
- Source TRR-0005 result SHA-256 `b3366bc65a8cff7b44920a3f778c33f2229f1d780cb256fb2731aa450f830389`; decision plan SHA-256 `d4ae0da99e5c0a00e454c16b539441e193cb76b9e52673c8e674bfdd2df1245`.

The remaining packet tail must resolve final selection/registration handling before this becomes a registered adequacy decision.
