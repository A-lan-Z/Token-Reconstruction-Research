# TRR-0003 footing: historical A1 provenance and timing audit

**Status:** read-only audit for the retrospective TRR-0003 panel, 2026-09-05.
This fragment records provenance limits and timing paths for the report. It does
not alter the historical A1 state or reopen any evaluation truth.

## Historical A1 provenance

The state used by the TRR-0003 historical comparator is:

- Artifact: `outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt`
- Size: 16,787,653 bytes
- SHA-256: `33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742`
- Role: the retained public-Alpaca-fitted affine A1 proposer; TRR-0003 did not
  refit it.

Safe archive inspection of this exact artifact exposed only `hidden=0` and
`corpus=alpaca` in its saved metadata. It did not expose the auxiliary row
selection, number of optimization steps, optimizer settings, or a measured
fit duration. Consequently, the recipe below is **documented historical recipe
provenance, not verified exact fit provenance for this current A1 artifact**.

The recipe is supported by the historical implementation and protocol records:

- Implementation: `/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/attack.py`
  and its `inv_common.py` helpers.
- Historical description: `/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/docs/INPUT_TOKEN_RECOVERY_L4_20260730.md`.
- Explicit protocol: `/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/research/adaptive_a1_a2_strict_bos_20260817_goal_01a00b08/round006_e2_prepare_protocol_v2.json`.
- Closest machine-readable fit manifest:
  `/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/out/lens_alpaca_rawbase_cut4_20260811.manifest.json`,
  SHA-256 `eda833bdd7219e6a4947ca3944dcf586513e8e2a5eb70934e99cf9f4ddddc8df`.
  That manifest describes a different raw-base lens (state SHA
  `de29c21227e6b7b5acd4353fc19d04b792cdb4060e2b56b5ee57409484cec309`), so it
  must not be presented as the exact training record for the retained A1.

The documented recipe has public Alpaca supervision (52,002 rows), 1,200
sampled auxiliary sequences (sample seed 7, maximum length 192, minimum length
32), and 125,571 collected token positions. The affine input lens starts with
identity `W`, zero `b`, and scalar logit parameter `s=3`; it normalizes the
public embedding table and trains with full-vocabulary cross-entropy using
AdamW (optimizer seed 0, learning rate `1e-3`, weight decay 0), batch size 512,
3,000 steps, and a cosine annealing schedule. The historical report describes
the fit cost only qualitatively as “~2 minutes on public data”; no exact
machine-readable duration for the retained A1 was found. Reusing this state
therefore has an unavailable historical preparation cost, rather than zero
preparation cost.

## Missing merged timing paths

`experiments/TRR-0003/evidence/common_score_v2.json` contains 52
`timing_path` entries, one for each method × condition × style cell. The
referenced paths all have the form:

```
outputs/TRR-0003/common_matrix_v2/{finance|pile}/{public_base|public_lora_2601}/{method}.evidence.json
```

The 52 entries were checked after scoring; none of those files exists in the
merged output directory. The prediction tensors and score metrics remain the
frozen outputs, but these path strings must not be treated as existing timing
artifacts.

The original timing evidence is retained at these locations:

| Source | Evidence location | Use in report |
| --- | --- | --- |
| Historical A1, frozen A1+A2, and direct inverse comparator cells | `outputs/TRR-0003/footing/comparator_matrix_v2/{finance|pile}/{public_base|public_lora_2601}/{direct_inverse\|frozen_a1_a2_k256\|historical_alpaca_a1}.evidence.json` | Comparator inference and candidate-simulation timings (12 files) |
| Track A raw checkpoint diagnostic cells | `outputs/TRR-0003/track_a_diagnostics/{finance_public_base\|finance_public_lora_2601\|pile_public_base\|pile_public_lora_2601}/.../evidence.json` | Structural-inversion source timing; aliases are serialization exports |
| Track B standalone decoder panel | `outputs/TRR-0003/track_b/panel_selected_v1/prediction_evidence.json` | Per-method first/warm inference summaries and load phases |
| Track B public preparation and fitting | `outputs/TRR-0003/track_b/public_fit_v2/prepare_evidence.json`, `outputs/TRR-0003/track_b/main_fit_v1/fit_evidence.json`, and `outputs/TRR-0003/track_b/extended_fit_1800_v1/fit_evidence.json` | Preparation and training costs, separate from steady-state inference |

For a complete handoff, the report or manifest should reference these original
evidence files (or explicitly mark merged timing as unavailable) rather than
claiming that the 52 placeholder targets exist.
