# What changes inside A2

A2 answers a forward question: if the next token were this candidate, what activation would the current prefix produce? It compares the simulated activation with the observed one. The existing decoder repeats this for256 candidates at each position and commits one winner before advancing.

The first optimization records the GPU operations for one complete A2 step. Python submits that recorded step as one graph instead of dispatching its many operations individually. We make a separate recorded step for each context length, so a40-token input keeps its native40-token geometry. A128-token input runs the same causal sequence up to128. Each record starts with BOS. The candidate batch, scores and separate batch1winning-token commit remain the same. The earlier shared-cache adapter also removes redundant intermediate cache copies. Exact equality is checked against native A2.

The second optimization changes how attention reads the recovered history. All256candidate branches have the same past; only the proposed next token differs. Instead of making256physical copies of the past keys and values, a small GPU kernel reads the one shared history plus the current candidate's own key/value. It still calculates every candidate's attention and still runs the remaining prefix operations. Its FP32 accumulation differs from the native SDPA kernel, so its accuracy requires a separate experiment.

Neither optimization selects a smaller shortlist. Neither is a learned inverse or an offline-trained token predictor. The replay program contains operations and memory addresses, not remembered answers. It reads new activations and candidate IDs on every run. When existing weight tensors change in place, the program reads their new values; the qualification explicitly tests a public-weight change and restoration. Replacing the tensor storage or changing geometry requires a new capture. The no-A1 proposer's derived lookup tables retain their own rebuild requirement.

These approaches pursue the user's option of making A2 cheaper. They do not yet deliver reconstruction without candidate proposal or A2 forward calculations. Direct candidate-free inverse attempts in TRR-0017 remained inaccurate and expensive; they do not establish impossibility.

The present benchmark uses the supplied public prefix. Weight-refresh execution checks are not an actual recovered-prefix training trajectory, and target-LoRA robustness is not prefix recovery. No qualified recovered-prefix trajectory was located in the existing result records.
