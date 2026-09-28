# TRR-0014 — Prefix-native candidate generation

Objective: find a promising materially different reconstruction mechanism with no separately trained token predictor, learned inverse, external LM, or independently fitted component. All model-dependent computation must derive from the supplied/recovered prefix itself.

This is exploratory mechanism research, not a replacement claim or completed canonical comparison. The current prefix-only solvers and stopped variants remain preserved. No task is merged.

## Candidate mechanisms and rationale
1. **Forward-response lookup with context correction.** Cache each vocabulary token's cut activation under the single allowed BOS context using the supplied prefix. Rank observed activations in that response space, rather than raw embedding space. Correct a query by the difference between the current-context response and BOS-context response of an actually checked candidate. Verification uses the same current prefix. The cache has no fitted parameters or auxiliary-data supervision; it is a disposable prefix-derived acceleration structure, invalidated by any prefix weight/version change. Charge its complete construction, storage and rebuild cost. Success would replace repeated optimization with lookup and a few actual forwards.
2. If that fails, inspect analytic removal of context contributions within the prefix and other prefix-native structural proposals. Do not treat past numerical-solver failures as a ban, but require a material mechanism difference.
3. Do not use other studies' fitted A1/B1 except the explicit historical comparator.

## Initial probe
Use only the already-opened rescue development observations (3 clips, 16 positions), as retrospective mechanism development. Freeze predictions before the separate scorer loads labels. No current-record source labels enter the predictor. Query three fixed lookup variants (cosine, Euclidean, context-corrected Euclidean), then inspect why they succeed/fail. Parameter choices and negative variants remain recorded. No hidden holdout or P03 access.

Forward cache chunks are a declared numerical execution contract, not assumed equivalent. Qualify multiple batch sizes, singleton/full paths, and candidate ranking on fixed public fixtures; preserve differences. Use one frozen native candidate helper for verification and the same baseline implementation/geometry. The cache is only a proposal; numerical equality is not claimed without measurement.

## Preflight
Known geometry: vocabulary 128256, width 2048, cut 4. A FP32 vocabulary response table occupies 1,050,673,152 bytes; BF16 table half that. Prefix + raw embeddings ~1.01 GB BF16. At 256 candidates and 127 cached preceding tokens, four-layer KV expansion is ~256 MiB; current MLP intermediates ~16 MiB per live layer. Expected peak under 4.5 GiB with no backward graph; hard cap 6 GiB, >=2 GiB free GPU, >=8 GiB host available, <80 C. Largest representative used geometry must qualify before full generation. Estimate full table runtime from first 1024 entries and record live hardware. Isolated create-only jobs, watchdog and failed runs retained.

## What earns further work
Require a strong proposal-recall improvement over raw-embedding lookup and/or the tested discrete solver, with a credible measured quality/cost path against A1+A2 after including cache rebuilds. A promising development finding must survive a frozen own-history test on new inputs, and a paired mismatched-prefix diagnostic. Do not call matching public weights recovered weights. Canonical dual benchmarks remain required before any overall replacement claim.

## Literature
- Nikolaou et al., Language Models are Injective and Hence Invertible, ICLR 2026, https://arxiv.org/html/2510.15511v4 . The worst-case verifier bound is T times vocabulary size; this does not promise cheap bounded proposals.
- Słowikowski and Majewski, Recovering Input Text from Hidden States, https://arxiv.org/html/2607.00852v1 . Continuous search with broad final candidate windows remains costly; its GPT-2 results do not establish this project's prefix-only quality/cost.
- Elhage et al., A Mathematical Framework for Transformer Circuits, https://transformer-circuits.pub/2021/framework/index.html . Residual additions motivate explicitly separating token and context effects; proposed method remains a hypothesis requiring native tests.
