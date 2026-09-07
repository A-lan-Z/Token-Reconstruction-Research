# TRR-P08 setup inventory

The P08 worktree is isolated from the published P07 parent at commit cffb10c14d3b74f4bd0227804e3ac380aa475686. This setup binds published metadata only. It has not selected fresh records, loaded model or observation tensors, started fitting, opened truth, or inspected TRR-0009 science.

The reusable public H128 recipe is bound to the published coverage_mix_v1 fit descriptor: 1,200 public fit records, 48 public validation records, 127 scored post-BOS positions, hidden size 2,048, vocabulary size 128,256, and the shared normalized F32 readout table. The P06 learned public affine descriptor is recorded separately from the source-design identity/zero default; its tensor and trajectory were not opened here. P06 past-only and positionwise states for seeds 6106/6107 are retained as context. The P06 full-record states are explicitly excluded by the packet.

The proposed comparison has four arms: positionwise and past-only, each with joint and affine-then-complete fitting. All arms must use the same public fit and validation roles, H128 geometry, readout vocabulary, and total exposure. The staged affine phase must count toward that budget. Phase lengths and any optional anchor remain pending design freeze.

A modest fresh panel is proposed as 64 paired source records per Finance and Pile domain under public_base and public_lora_2601, but no records are selected. Selection requires a frozen source universe, stable ordering, and the root-owned approved hash-only ledgers. P06/P07 and the published TRR-0006 subset are recorded as hash-only exclusions. The exact approved TRR-0008 opaque export was verified by path, byte count, SHA, and schema; no other TRR-0008 or any TRR-0009 file was read. P03 remains sealed. Universal disjointness is not claimed.

The inherited P06 largest-cell guard is documented in preflight.json for planning only. A complete-mask qualifier and root resource authorization are required before any P08 job.
