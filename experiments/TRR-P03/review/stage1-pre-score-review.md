# TRR-P03 Stage 1 pre-score review

**PASS.** Both full target arms completed the pre-score protocol and are eligible for truth scoring under the strict joint receipt. This review was completed from receipt metadata only; no truth or prediction row/tensor content was opened.

The two generation bundles contain the ordered 24-record panel with scored lengths 16, 39, 64, and 128 (six records each; 1,482 scored tokens). Bundle A is the matched public evaluator arm; bundle B is the shifted evaluator arm. Both are truth free and use source commit 6edb276a3a536988a1d2cc9f3aa4c29e90e1a6b1, seed 20260906, CPU, deterministic Torch settings, and batch size 4.

Both reconstruction roots froze all four required methods and the four exact anchors (p03-s1-r0007, p03-s1-r0009, p03-s1-r0011, p03-s1-r0012) before truth. Both use the pinned public base Llama checkpoint, canonical query/prototype chunks 256/8192, float32 scores, and the same plan hash f9c4d4bfa0fd8649891154326b32f5ad1e29d53d33af606fd4d53534d6debcb5. The two source manifests and shared table assets agree. The three standalone static readouts use descending score with ascending-token-ID ties; native A1+A2 uses the published torch.topk proposal order followed by first-argmax. Their freeze receipts are bound to the two roots by the strict validator.

All four watchdogs passed with wrapper and child exit code 0, no termination reason, an 8 GiB RSS cap, a 10 GiB minimum host-available-memory threshold, and a 900 second timeout. The maximum reconstruction internal RSS was 7,818,420 KiB (arm A), leaving about 0.544 GiB below the cap; the other arm was 7,817,944 KiB. The strict receipt 7e4d848fd33fb4218c0ffbfc1c41a5c0806f746d5107f5c02b430118e45b534d reports status VALIDATED and validation STAGE1_JOINT_VALIDATION_PASS with truth_opened false.

The scoring command was therefore eligible to open truth only after the two roots and strict receipt were frozen. This file records the pre-score disposition; the numeric Stage 1 decision is in stage1-gate-review.md and gate.json.
