# Mechanism and literature notes

Primary source checked19Sep2026:
[Language Models are Injective and Hence Invertible, Nikolaou et al., ICLR2026](https://arxiv.org/html/2510.15511v4).
The theorem is about separation of discrete token sequences under its stated idealized assumptions. Its constructive recovery argument still enumerates vocabulary tokens in the worst case (at mostT|V| checks), using gradients to guide that enumeration. It does not supply a cheap algebraic inverse or prove that unconstrained continuous embedding optimization returns the correct vocabulary row. Public implementation: https://github.com/giorgosnikolaou/SIPIT . We implemented the present experiments independently against this project's native prefix, not by claiming to reproduce SIPIT.

Project evidence motivating this task: scripts/agent4/solver.py recomputes an entire growing sequence and verifies individual proposed tokens. scripts/trr0014/reverse_probe.py used6 damped or10 Anderson iterations of residual subtraction and failed token recovery. These were not efficient whole-sequence constrained inverses.

Our hypotheses: whole-sequence optimization shares dense operations across positions; a diagonal causal gradient prevents one position from being moved merely to compensate for later-position errors; straight-through projection keeps forward estimates on actual token vectors. None is an exact inverse guarantee. Low continuous activation loss need not imply correct tokens. BF16 rounding and prefix-target mismatch further separate the benchmark from the ideal theorem.

A2 cache optimization is independent: its values and final candidate geometry stay unchanged. It removes redundant cache copying, not transformer arithmetic, the candidate list, or the prefix-derived lookup setup.

