# Causal consistency correction in intrinsic response coordinates

Parallel context stripping with only top1 guesses failed to close natural recall (2004/2032 matched,2002/2032 shifted), although it covered all stress tokens. These are proposal-only negatives.

Test context stripping after actual native verification using its reconstructed history. Initial128 candidates concatenate64 embedding-metric and64 MLP-dictionary-metric proposals. After selecting best native cosine, query q=H-P(best;own_history)+D_best. If the intrinsic dictionary's corrected top1 is the same token, stop. Otherwise verify its top32, retain best native cosine across actual checks, and repeat at most twice. This is a structural fixed-point check, not a calibrated correctness threshold or guarantee. Max192 logical checks per token. Padded candidates=-1 and scores=-2 are serialization only. No fitted predictor or backward step.

All48 opened R1 records, all outputs frozen before retrospective score; record rounds, native simulation count and synchronized wall compute. This is the only new decision rule. Original evidence unchanged. Largest128positions/K128 initial vs prior qualified128/K256; prefix+twoFP32 dictionaries+oneBF16 intrinsic table~3.7GB,6GB guard. Qualify synthetic128positions before matrix. Watchdog180s. Prefix construction batch256 retained from previous probe; no batching change.
