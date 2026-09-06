# TRR-0006 bounded exact sample-size sensitivity

Pre-truth CPU-only planning under the registered exact bound `U_CP(g)-L_CP(h)`, with one-sided tail alpha 0.05/32 = 0.0015625 and exclusion margin 5 pp.

| Scenario | n | plug-in B upper pp | P(exclude 5 pp) | P(point net >= 5 pp) |
|---|---:|---:|---:|---:|
| finance_p0_observed (0.039,0.031) | 128 | 11.471 | 0.007951 | 0.033779 |
| finance_p0_observed (0.039,0.031) | 768 | 4.850 | 0.571146 | 0.000007 |
| finance_p0_observed (0.039,0.031) | 1024 | 4.286 | 0.804839 | 0.000000 |
| finance_p0_observed (0.039,0.031) | 1536 | 3.625 | 0.976877 | 0.000000 |
| finance_p0_observed (0.039,0.031) | 2048 | 3.235 | 0.998393 | 0.000000 |
| finance_lora_observed (0.039,0.047) | 128 | 10.893 | 0.016862 | 0.011585 |
| finance_lora_observed (0.039,0.047) | 768 | 3.696 | 0.892141 | 0.000000 |
| finance_lora_observed (0.039,0.047) | 1024 | 3.076 | 0.981436 | 0.000000 |
| finance_lora_observed (0.039,0.047) | 1536 | 2.349 | 0.999780 | 0.000000 |
| finance_lora_observed (0.039,0.047) | 2048 | 1.920 | 0.999999 | 0.000000 |
| higher_discordance_net_plus_3 (0.080,0.050) | 128 | 16.377 | 0.000207 | 0.254032 |
| higher_discordance_net_plus_3 (0.080,0.050) | 768 | 8.303 | 0.005804 | 0.060554 |
| higher_discordance_net_plus_3 (0.080,0.050) | 1024 | 7.626 | 0.011075 | 0.035683 |
| higher_discordance_net_plus_3 (0.080,0.050) | 1536 | 6.742 | 0.029857 | 0.015615 |
| higher_discordance_net_plus_3 (0.080,0.050) | 2048 | 6.261 | 0.061627 | 0.005917 |
| higher_discordance_net_plus_5 (0.100,0.050) | 128 | 19.424 | 0.000020 | 0.487801 |
| higher_discordance_net_plus_5 (0.100,0.050) | 768 | 10.684 | 0.000028 | 0.495067 |
| higher_discordance_net_plus_5 (0.100,0.050) | 1024 | 9.824 | 0.000029 | 0.489235 |
| higher_discordance_net_plus_5 (0.100,0.050) | 1536 | 8.969 | 0.000029 | 0.507122 |
| higher_discordance_net_plus_5 (0.100,0.050) | 2048 | 8.444 | 0.000029 | 0.496983 |
| higher_discordance_net_plus_5b (0.120,0.070) | 128 | 20.257 | 0.000021 | 0.490210 |
| higher_discordance_net_plus_5b (0.120,0.070) | 768 | 11.223 | 0.000030 | 0.496028 |
| higher_discordance_net_plus_5b (0.120,0.070) | 1024 | 10.409 | 0.000031 | 0.490801 |
| higher_discordance_net_plus_5b (0.120,0.070) | 1536 | 9.368 | 0.000032 | 0.506599 |
| higher_discordance_net_plus_5b (0.120,0.070) | 2048 | 8.849 | 0.000033 | 0.497569 |
| higher_discordance_zero_net (0.100,0.100) | 128 | 16.481 | 0.002101 | 0.098890 |
| higher_discordance_zero_net (0.100,0.100) | 768 | 6.532 | 0.172873 | 0.000951 |
| higher_discordance_zero_net (0.100,0.100) | 1024 | 5.627 | 0.326589 | 0.000161 |
| higher_discordance_zero_net (0.100,0.100) | 1536 | 4.593 | 0.641732 | 0.000006 |
| higher_discordance_zero_net (0.100,0.100) | 2048 | 3.968 | 0.852741 | 0.000000 |

The observed Finance P0 rates yield about 0.805 exclusion probability at n=1024 (0.571 at n=768; 0.977 at n=1536), while the observed LoRA rates yield about 0.981 at n=1024 (0.892 at n=768; 0.9998 at n=1536). Higher-discordance positive-net scenarios are much less likely to be excluded: at true net +3 pp (8%,5%), the n=1024 exclusion probability is about 0.011; at true net +5 pp (10%,5%), it is about 0.00003. These are sensitivity probabilities, not a registration decision.

Artifact: `/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0006/experiments/TRR-0006/precision/sample_size_sensitivity.json`; elapsed CPU time 2.744s.
