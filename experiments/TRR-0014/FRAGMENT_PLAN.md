# Token-boundary decomposition of prefix-native candidates

Post-score inspection of all42 omissions from the embedding/MLP union found word-continuation tokens: ing, ed, ised, ron, ling, aining, etc. Own-history examples show the hidden state ranks whole words or semantic neighbors (surprised for ised after surpr; complaining for aining after compl; gloves for ves after glo). This is opened-development diagnosis, not a fresh discovery claim.

New mechanism: use the existing public tokenizer to expand whole-token proposals into their proper character suffixes. Tokenize each suffix and add its first piece. No language model, word list, frequency fit, training corpus, learned parameter, or label-derived exception. This can recover a token piece when the prefix geometry points toward a containing word.

Freeze two development bases: embedding-metric top64; and its union with MLP-intrinsic top64. Preserve all base IDs, then round-robin each candidate's suffix list (longest first), deduplicate deterministically, cap512 and pad by repeating first candidate. Report fixed256 and512 recall; no threshold sweep or correctness routing. Non-decodable byte fragments containing Unicode replacement are skipped. Actual token values are always valid tokenizer IDs, and native A2 will determine context compatibility.

This probe uses only already-frozen no-truth base proposals and tokenizer operations. All96 outputs freeze before any labels are read. No GPU and no reconstruction/time claim. If promising, the selected new mechanism needs its own native A2 qualification, complete causal run, and independent fresh confirmation. Full tokenizer expansion cache is prefix-independent metadata; prefix metric still invalidates on every weight change.
