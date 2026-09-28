# Why the mixed shortlist can help

The expensive A2 step is unchanged: test256 token candidates with the prefix, choose the closest activation, and commit that token. The new work changes only which256 tokens get tested.

The two existing lookup tables already score the vocabulary. One contains transformed input embeddings; the other contains transformed vectors after each token embedding has passed through the prefix MLP residuals without attention. Both come from the same supplied prefix, without training a predictor. Their cosine scores provide cheap clues, not proof of a token's identity.

Keep the existing top64 from each table. Look a little farther down those same tables (top128 each) to obtain more words whose tokenizer suffixes may contain the missing token. This adds small top-k and tokenizer-list operations, not transformer simulations. Each distinct word votes once for every suffix it supplies. Reserve64 suffix slots for repeated suggestions. Fill the remaining slots with suffixes whose own vectors score highest in either existing lookup table. For example, a useful rare ending may receive only one vote; its own strong lookup match can now beat many irrelevant one-vote endings. All base candidates remain protected and the total remains256.

This has a real cost: more CPU suffix processing, scalar score transfers and top-k work. The two full-vocabulary matrix multiplications and the number of A2 simulations remain unchanged. The lookup tables still need rebuilding when prefix weights change; tokenizer suffix metadata does not. No corpus-frequency table, fitted A1, gradient training or per-answer correctness signal is added.

The successful development policy was chosen using previously opened records. That is algorithm development, not a fresh result. Its constants and implementation were then frozen before the new R4 panel. The actual evaluation still supplies the public prefix as the approximation; it does not demonstrate performance throughout a prefix-recovery trajectory.
