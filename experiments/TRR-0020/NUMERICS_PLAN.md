# Repeatability diagnosis
Dev6 stopped at its repeatability gate after8cells. No development truth is opened for this incomplete attempt.
Reproduce constant256 on one public synthetic128-token fixture,seed200026,three times per process. Compare default execution with deterministic PyTorch algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8, using byte-identical public observation tensors. Save every output and loss trace before comparing. Different arithmetic mode is a proposed numerical-method revision, not a silently equivalent resource workaround.
Same geometry and<=6GiBprocess,>=2GiBfree guard; each process expected<30s,timeout180s. If deterministic mode is stable, preregister a fresh output matrix rather than combining it with the partial default-mode cells.
