# Public resources

`public_resources.json` records the identity and immutable revision of the public architecture/tokenizer checkpoint used by the reference comparator. It does not prescribe a dataset or a training source for new methods.

This repository intentionally does not contain:

- target boundary activations or their plaintext/token truth;
- hand-selected calibration or evaluation examples;
- model weights already available from their authoritative registry;
- trained lenses, adapters, rerankers, or other learned reconstruction state; or
- prior outputs, scores, rankings, or conclusions.

When a new experiment introduces a public dataset or another public checkpoint, add its exact revision, fingerprint, license, and role to the manifest as part of that experiment. Keep evaluation traces and truth in access-controlled storage outside Git.
