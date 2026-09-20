# Development34 public common-token stress extension

The original48-case public study passed but never rejected a trial. All24paired
activation errors were identical between fixed and backtracked updates. It
therefore did not exercise backtracking on the actual prefix. The toy CPU checks
did exercise60rejected trials, but are not a substitute for actual-prefix cases.

Freeze16public strings in the stress driver: ordinary words, programming words,
digits, whitespace, braces, a parenthesis and a code fence. Tokenize each with the
pinned public tokenizer and take its first token after the declared BOS. These
strings are chosen directly as public fixtures, without using hidden target IDs.
Keep duplicate first-token identities if any; report them rather than silently
changing the context set.

Generate observations for each two-position sequence separately under the public
checkpoint in FP32 and BF16. The reconstruction map stays the same frozen FP32
current-token function and uses only the BOS history. The dtype variation is an
explicit public observation condition, not an equivalent execution claim.
Release the temporary BF16 model after target generation before loading FP32.
Account for both preparations and retain their fixture hashes.

The matrix is16contexts x2observation dtypes x2rules x8/32updates =128cases,
three exact repetitions each. Use the existing unchanged fixed and backtracked
trajectory functions. Independently qualify all32initial probability gradients
against HuggingFace autograd at fixed BOS history. Save all final logits, traces,
direct FP64 KL values, commit arrays and numerical references. Require exact
repeatability, the descent rule's Armijo/monotonicity invariants, unchanged BOS
history and resource margins.

After the complete matrix, summarize recovery of these explicitly public token
identities as a numerical sanity check. Do not call it benchmark accuracy or use
identities to route or stop an in-progress trajectory. Hidden reconstruction
outputs and labels are not loaded by this experiment.

Preflight reuses the prior3.127GiB measured peak with additional preparation
margin. The matrix has no attention context larger than one past position; its
maximum32updates and four trials are already qualified. Retain the external
idle-GPU guard and internal8GiB reserved /3GiB GPU free /8GiB host available
bounds. Preserve and exclude failures before any repair.

This extends numerical coverage of the already frozen rule; it does not register
a new active canonical method or make a replacement claim.
