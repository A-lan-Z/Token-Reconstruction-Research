TRR-P01 final evaluator preparation

Status: PUBLIC_ARMS_FROZEN_BEFORE_SCORING
Execution: CPU only; no CUDA allocation; no reconstruction or truth scoring has run.
Implementation commit: 6e05feeade57593cdadea2d4db4ce40085a51f59

The fresh paired public arms are under public/arm-000 and public/arm-001.
Both sanitized configs declare the exact eight arms:
- boundary.cosine
- boundary.l2
- raw_embedding.cosine
- raw_embedding.l2
- reference_corrected.cosine
- reference_corrected.l2
- historical_a1.cosine
- historical_a1_a2_port.cosine

Geometry is [16,40,2048] BF16 with identical opaque record order and mask/position digests. The evaluator condition map and private truth remain under evaluator_private and are not part of either public arm.

The exact command is in command.txt and evaluator output is in evaluator-final-20260905.log. Reconstruction remains pending the coordinated uncontended CPU interval.

External provenance sidecar: source_certification.json
- Certified commit: 6e05feeade57593cdadea2d4db4ce40085a51f59
- Sidecar SHA256: 82c939a4443b226f970e67b31b35dfcfb9e930a9ffa408d9883e6dddf31e4d77
- Five evaluator-imported task source files matched git-show bytes at that commit and were CLEAN in the working tree.
- Sidecar binds command.txt, evaluator log, native evaluator evidence, and all six public-arm artifacts. Native evaluator evidence was not rewritten and private truth was not reopened.
