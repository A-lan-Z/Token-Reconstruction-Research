# Development follow-up3: discrete parallel refinement

Dev2 completed40 frozen cells. All five variants retained the same token identities on the8 development observations although continuous losses improved, consistent with off-vocabulary fitting. This motivates a different optimization domain.

At each iteration project the entire evolving sequence against all128256 embedding rows; forward uses these actual embeddings with a straight-through gradient to the continuous proxy. No K-sized shortlist, independent A1 or candidate verifier. Retain the complete sequence with the lowest observed loss, not per-token best guesses from incompatible contexts. Adam only, fixed iteration schedule. Initial state is prefix-weight full-vocabulary top1 as dev2.

Six preregistered variants: raw/full/64/lr0.003; raw/full/128/lr0.01; whitened/full/64/lr0.003; cosine/full/64/lr0.003; raw/diagonal/64/lr0.003; cosine/diagonal/64/lr0.003. The diagonal variant detaches other positions' key/value gradients while retaining own-query and same-position key/value gradients. It uses eager attention and is a separate numerical approximation, not an equivalent kernel substitution.

Same8 development records, all48 outputs freeze before scoring. Length128 synthetic qualification before matrix. Dev2 peak reserved5.495GB, below6GiB guard; eager attention adds <20MiB. Full-vocabulary projection at each step costs128x128256~66MB temporary. Expected<3min; watchdog900s. If the best candidate-free method remains poor, complete its canonical cells for a quantitative negative result and separately investigate numerical-equivalent A2 execution savings.
