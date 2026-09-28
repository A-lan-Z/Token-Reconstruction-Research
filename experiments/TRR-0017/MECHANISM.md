# Mechanisms in plain terms

## The original reconstruction loop

At each missing token, a proposer supplies256 possibilities. A2 runs the four recovered/supplied prefix layers on each possibility, using the tokens already reconstructed as context. It keeps the possibility whose predicted hidden vector most closely matches the observed vector. A good proposer saves work by limiting the possibilities, but all256 still pass through the prefix.

## The direct-inversion experiments

Instead of testing a list, start with one rough vector for every token position. Run the prefix on that whole estimate and measure how far its output is from the observed output. Use derivatives to move the estimated vectors in a helpful direction. After a fixed number of updates, match the estimated vectors to the vocabulary.

The difficulty is that vectors can move to places where no real token exists. The model's output can become a much closer match while the final token choices stay wrong. Solving continuous equations is not the same as finding a sequence of valid token IDs.

The discrete version therefore makes every forward pass use real token embeddings. Its adjustable vectors guide which complete sequence is used on the next pass. A further change lets each position's update focus on that position's own output, instead of changing earlier tokens to compensate for mistakes later in the sequence. These methods have no candidate shortlist or independently trained decoder, but still need repeated prefix passes and full-vocabulary matching.

## The exact cache improvement

Every candidate has the same already-reconstructed history. The old helper first copies that history, expands all four layers' caches into256 branches, and then appends the candidate's own keys and values.

The replacement keeps a read-only reference to the committed history. When a layer needs its attention input, it directly builds the final tensor from that reference and the candidate's new keys and values. It avoids the intermediate replicated cache and keeping all four candidate caches alive together. The final attention tensor is still expanded to the same shape as before; its values and arithmetic are unchanged.

This removes redundant copying, not candidate evaluation. The same256 possibilities still go through all four layers. It needs no new learned component or preparation table of its own, and can be used with the existing no-A1 proposer. Prefix-derived proposal-table construction remains a separate cost.

For length128, K256, eight KV heads and head width64, one layer's BF16 keys plus values occupy64MiB when expanded across candidates. Holding four such layers costs256MiB; the adapter only materializes the current layer. These are tensor-size calculations, not a claim of a256MiB decrease in the whole process's measured peak.
