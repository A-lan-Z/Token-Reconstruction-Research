# TRR-0006 precision preflight (provisional)

Status: CPU-only, read-only planning; no new truth/model/fits and no registration.

Exact rule: B = U_CP(g;n,0.0015625) - L_CP(h;n,0.0015625); exclusion event B <= 5 pp.
Token rule: registered source-bootstrap upper tail alpha = 0.05/16 = 0.003125; margin = 0.5 pp.

## Exact endpoint sensitivity at n=1024 per target

| Scenario (pg, ph) | P(exclude 5pp) | P(point net >=5pp) | plug-in net upper pp |
|---|---:|---:|---:|
| null_no_discordance (0.0000, 0.0000) | 1.000000 | 0.000000 | 0.629 |
| pile_like_observed (0.0078, 0.0078) | 1.000000 | 0.000000 | 1.779 |
| finance_p0_observed (0.0391, 0.0312) | 0.804839 | 0.000000 | 4.286 |
| finance_lora_observed (0.0391, 0.0469) | 0.981436 | 0.000000 | 3.076 |
| moderate_net_plus_2_low (0.0400, 0.0200) | 0.408131 | 0.000054 | 5.242 |
| moderate_net_plus_2 (0.0600, 0.0400) | 0.138984 | 0.001150 | 6.044 |
| useful_net_plus_5_low_harm (0.0800, 0.0300) | 0.000030 | 0.486587 | 9.143 |
| useful_net_plus_5_balanced (0.1000, 0.0500) | 0.000029 | 0.489235 | 9.824 |
| balanced_zero_net_high_discordance (0.0500, 0.0500) | 0.810003 | 0.000000 | 4.125 |

## Observed-rate target dependence

| Domain | P0 exclusion | LoRA exclusion | target-joint Frechet range | 9-category paired-shape MC |
|---|---:|---:|---:|---:|
| pile | 1.000000 | 1.000000 | [1.000000, 1.000000] | 1.000000 ± 0.000000 |
| finance | 0.804839 | 0.981436 | [0.786274, 0.804839] | 0.798650 ± 0.005558 |

The TRR-0005-shaped Finance joint estimate is close to the no-independence Frechet interval; it is a scenario, not a guarantee for the new panel. Pile is effectively certain under its very low observed discordance rates. Across all four cells, a target-independent product would be only a labeled assumption; without independence, use the corresponding Frechet bounds.

## Provisional adequacy recommendation

The 1024-per-domain plan is adequate for the registered exact-margin exclusion if rates remain near the observed Pile/Finance discordance rates: per-target exclusion probabilities are effectively 1.000 (Pile), 0.805 (Finance P0), and 0.981 (Finance LoRA), with a dependence-aware Finance target-joint range of about 0.785–0.805. It is not guaranteed to resolve a small positive exact effect: at true net +2 pp, exclusion is about 0.14 for a (6%,4%) rate pair; at true net +5 pp, exclusion is approximately 3e-5 while the point estimate reaches +5 pp only about 0.49 of the time. Keep the recommendation provisional until the remaining packet tail provides the final registration/selection instruction.

Artifact: `/tmp/trr0006_precision/precision_preflight.json`; elapsed CPU time 6.455s; simulation draws per domain 20000.
