# Development7: constrain soft mixtures without proposing candidates
Dev6_r1 complete48cells, repeatability passed under explicit deterministic execution. Adaptive256 best objective recovers matched natural254/254 and stress78/78, shifted natural251/254 and stress78/78. Per-position best reaches252/254 shifted; baseline is perfect on all8. End-to-end inference~1.67s for128positions and~1.18s for40, still too slow.

Hypothesis: mixtures can fit activation while their argmax token is wrong; encouraging a discrete solution may improve quality/convergence. This is an unproven hypothesis. No vocabulary pruning or separate token verifier is added.

Eight frozen exploratory configurations,256steps each, adaptive per-position LR from dev6_r1, constantbaseLR.6/decay.98 and momentreset32:
- control: unchanged soft objective.
- gini003,gini01,gini03: add lambda*(1-sum(p^2)) averaged acrosspositions, lambda=.003/.01/.03 fromiteration32.
- hardhalf64: soft embedding moves linearly to50% hard argmax embedding overiterations64..96.
- hardfull64: moves to100% hard argmax over64..96.
- hardfull128: moves to100% over128..160.
- hardfull0: hard argmax embedding throughout.
For hardening the backward derivative remains that of the complete soft vocabulary mixture. No top-K operation, no rejected token pool, and one whole-sequence prefix forward/backward per update. This is direct online optimization of one evolving sequence. It neither trains prefix weights nor fits an offline inverse.

Record final/snapshot argmax, best combined-objective iterate, best observed-error iterate and per-position best-error diagnostic. Best-state selection is eligible only after the final objective/hardening regime is reached. All selection signals depend only on observations, never labels. Record confidence/purity/error traces to test the mixture hypothesis after freeze.

64cells: same8 previously opened development observations. Sources/flags/data hashes fixed, complete freeze before current truth opening. Deterministic flags and native length geometry unchanged. Test straight-through forward endpoints and declared backward surrogate independently. Qualify each config on public128-token input, repeat outputs and loss exactly3times, save before gate. Control outputs must reproduce all outputs of dev6_r1 adaptive256 on8inputs; any difference is preserved and excluded pending diagnosis.

Resource:prior deterministic largest128<=3.622GiB reserved; added full probability-square/autograd scratch estimated <=0.5GiB pergraph pool; total process guard <=6GiB with at most2lengths. Admission >=9000MiB, free>=2GiB, hostavailable>=8GiB,RSS<=10GiB,temp<80C. Qualify largest beforematrix. Expected<5minutes; timeout1200. No canonical replacement claim or new active method.
