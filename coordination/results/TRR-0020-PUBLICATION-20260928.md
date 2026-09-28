# TRR-0020 GitHub publication — 28 September 2026

All nine outstanding research branch heads have been pushed and read back from GitHub. Eight draft pull requests were created and the existing TRR-P12 draft was updated. Original scientific history and archived evidence are preserved. No pull request was merged.

## Authorization and scope

The owner's instruction, saved verbatim in `coordination/requests/TRR-0020-PUBLISH-20260928.md`, authorizes this publication: “Push them, do things appropriately”.

The historical local-only and automatic-review-block statements in earlier results describe the status at the time those records were written. This explicit 28 September authorization supersedes that publication status. Earlier scientific records remain unchanged.

Research remains `STOPPED_BY_OWNER_AFTER_COMPLETED_BENCHMARK`; the scientific objective is not achieved. No experiments or parameter searches were resumed. Bennett64 is a whole-vocabulary method without a candidate proposer or separate verifier, but the completed canonical comparison remains less accurate and slower than both ordinary and optimized A1+A2.

## Published research heads

| Branch | Preserved research commit | Draft PR |
| --- | --- | --- |
| task/TRR-0014 | `ae8288faeb45334c4aa8a9a0f8a14a1def1f8d80` | [#32](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/32) |
| task/TRR-0017 | `02a2ead7348cd68027024fe1464a148e5ca4da65` | [#33](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/33) |
| task/TRR-0018 | `7a65a0adefd0f1597a251c2ab7f54f4fb43f7c06` | [#34](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/34) |
| task/TRR-0019 | `1f70d412f414cff635ee3b52309ec67ee2461195` | [#35](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/35) |
| task/TRR-0020 | `a47ee5bef4fadc1e40b11bdcfea901a956e224ce` | [#39](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/39) |
| task/TRR-P12 | `d4d3d9bcc068536ea6883ac329b666347eba64a1` | [#26](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/26) |
| task/agent4-holistic-reassessment | `00c79da2856f0c787bf28a96ebc47be63083549b` | [#37](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/37) |
| task/agent4-joint-native-gate | `6ce8a8be603fa805fc3edfe9618324fa5ccb50de` | [#38](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/38) |
| task/agent4-prefix-only-rescue | `ce04a8dfb11452407df94468964723c6294b8933` | [#36](https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/36) |

These commit IDs freeze the original scientific heads. A subsequent commit on `task/TRR-0020` contains only this publication record, its audit evidence, the authorization, and the state update. Draft PR bases follow the recorded branch ancestry; the TRR-0018 sibling result and TRR-P12 result retain their own branches and handoffs.

Main was verified unchanged at `84479014a6c07fa6f17d53df3f50eb32e3ef140c`.

## Evidence and validation

- The final Bennett64 archive's two parts and all 2,565 logical artifacts (2,284 unique objects) passed size and SHA-256 verification. The reconstructed 85,838,799-byte archive hashes to `89f2aa6da8f2f4383b54477de2a9be5ba5494ac911d1b294a1c11e3b5a76889c`.
- All nine branches contain their task result and structured manifest. TRR-P12's immutable plan remains in `experiments/TRR-P12/manifest.json`; its final execution evidence is `experiments/TRR-P12/final-evidence.json`.
- The object audit covers 4,750 unpublished objects, including 3,080 blobs totaling 7,900,115,390 uncompressed bytes. No blob exceeds GitHub's 100 MiB limit. The largest is 67,308,523 bytes.
- A limited scan of 2,762 new text blobs found no high-confidence credential/private-key matches. This is not a comprehensive security audit.
- Historical whitespace diagnostics are retained in `branch_audit.json`; frozen scientific files and verbatim requests were not rewritten.
- Publication validation did not rerun scientific experiments. Original run commands, configurations, hardware, timing, failures, exclusions, and scientific limitations remain in the task manifests and results.
- The two pre-existing untracked animation request/result documents remain local and untouched. External checkpoints and datasets retain their original locations and access requirements.

## Transfer and reproducibility

Fifteen successful, sequential, non-forced pushes published the original histories. Bennett64 required seven ancestry-preserving batches because its committed experiment archives exceed GitHub's per-push size limit. Windows Git 2.48.1 was used after the WSL network transport stalled. Every accepted push has a start/end timestamp, exact commit, exit code, and log in `experiments/TRR-0020/publication-20260928/push_receipts.json`.

Command pattern, run against the original repository with Windows Git:

```text
git -c safe.directory=<repository> -c pack.threads=2 -c pack.windowMemory=128m -C <repository> push --porcelain --progress origin <recorded-commit>:refs/heads/<recorded-branch>
```

`push_plan.json` supplies all recorded commits and branch names. `remote_before.json` and `remote_after.json` preserve the branch snapshots around the research uploads. `pull_requests.json` records verified review links and original research heads. `validation.json`, `branch_audit.json`, and `object_audit.json` capture the publication checks.

Structured publication evidence: `experiments/TRR-0020/publication-20260928/manifest.json`.

Verified at 2026-09-28T08:03:23.010719Z.
