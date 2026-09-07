# TRR-P09: frozen fixed-control fits and public validation handoff

TRR-P09 now contains the completed Agent2 frozen-control handoff. Both fixed-control arms ran under the signed Stage3 contract for 13,000 optimizer updates and passed clean external watchdogs and immutable CPU audits.

## Results

- **B0:** selected step 8,000; public-validation Finance `0.9837906004`, Pile `0.9585999016`, equal-domain mean `0.9711952510`
- **B1:** selected step 13,000; public-validation Finance `0.9938484252`, Pile `0.9950172244`, equal-domain mean `0.9944328248`
- **Selected validation difference:** B1 minus B0 `0.0232375738` (`2.323757` percentage points)
- **Compute:** 13,000 updates and 6,656,000 position draws per arm; GPU released after clean completion
- **Denominators:** B0 1,200 records / 124,371 post-BOS positions; B1 12,000 records / 1,243,710 post-BOS positions

All metrics above are public validation only. Target observation, final panel, and evaluation truth remain unopened. No new scoring or evaluation was performed in this handoff.

## Evidence

- Result: `coordination/results/TRR-P09.md`
- Structured manifest: `experiments/TRR-P09/manifest.json`
- Parallel state: `coordination/parallel/TRR-P09.json`
- B0 audit: `experiments/TRR-P09/results/current-fixed-r3/audit.json` (PASS 5/5)
- B1 audit: `experiments/TRR-P09/results/expanded-fixed-r3/audit.json` (PASS 93/93)
- Validation-only plot: `experiments/TRR-P09/review/fixed-control-validation-curves-r1.png`

This PR remains draft and unmerged. Earlier failed attempts and their receipts remain preserved as excluded provenance.

B0 audit source provenance limitation: the original audit helper/command was not retained; its immutable audit and raw evidence are preserved, with an explicit provenance sidecar. B1 includes the audit source and exact invocation.
