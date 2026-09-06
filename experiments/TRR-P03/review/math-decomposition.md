# TRR-P03 compact-scoring algebra review

This is a predeclared Stage 2 implementation note. It is not a result and is
not permission to run decomposition before the Stage 1 gate.

Let the frozen affine lens be `g(h)=Wh+a`, and let `b_v` be the public
boundary prototype for vocabulary ID `v`. The uncompressed projected readout
uses

```text
q       = normalize_fp32(g(h))
z_v     = normalize_fp32(g(b_v))
score_v = q dot z_v.
```

The affine bias is retained in both transforms. Candidate normalization is
part of the cosine rule and is not dropped because the query norm is shared.

For each predeclared rank `r` in `(128, 256)`, decompose the frozen float32
candidate matrix

```text
C_proj[v,:] = z_v
C_proj approximately U_r Sigma_r V_r^T.
```

Persist `A_r=U_r Sigma_r`, `B_r=V_r`, and row norms of `A_r`. Since the
columns of `B_r` are orthonormal, the compact score is

```text
score_r(v) = dot(q B_r, A_r[v,:] / ||A_r[v,:]||).
```

A zero factor row is invalid and fails closed. Apply the same float32
construction to `C_a1[v,:]=normalize_fp32(E_v)` for the same-rank compact A1
control. Use randomized range finding with oversampling 16, two power
iterations, fixed seed 8675309, QR, and a small SVD. Freeze ranks, seed,
precision, row normalization, score kernel, and standalone ascending-ID tie
rule before opening `p03-s2` truth. Sign ambiguity is harmless when both
factors are transformed together; a canonical sign may be recorded for byte
reproducibility.

The footprint audit reports full candidate bytes, factors, retained
lens/query assets, construction workspace, deployed readout bytes, model
memory, score blocks, and I/O. The practical targets are candidate-factor
storage at most 25% of the full table, resident readout storage at most 35%,
and at least 20% lower full scoring time on the same query set, hardware, and
timing boundary.

Quality retention on `p03-s2` allows at most 1.0 percentage point token loss,
one exact-record loss, and a paired record-cluster lower CI bound of at least
-1.0 point versus the uncompressed projected parent. A rank that misses
accuracy is rejected without a new rank or rescue factorization. A rank that
meets quality and storage but misses runtime is reported as footprint-only.
