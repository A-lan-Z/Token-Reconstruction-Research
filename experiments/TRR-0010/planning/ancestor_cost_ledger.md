# TRR-0010 ancestor cost ledger

Generated 2026-09-07T09:52:38.278664Z from the published TRR-0009 tree at commit
4602f98eeb03ac121d8bbc230d7e2b219551b914. The source manifest is
experiments/TRR-0009/manifest.json (SHA-256 00d2e9bcb75962072dabdac50ad38b32e9ad638fae77c9da75faec1760048e64). The ledger reads only the TRR-0009
manifest/evidence and its declared TRR-0005/TRR-0007 ancestor manifests/reports.
It does not open private truth, select data, run a model, or rerun a fit.

## Accounting boundary

The **historical published campaign-effort sum** is **2,098.154381422 seconds
(34 min 58.154381422 s)**:

| campaign | scope | wall seconds | evidence |
|---|---|---:|---|
| TRR-0005 | published eight-fit campaign | 801.700000000 | experiments/TRR-0005/manifest.json ef670253241886e7d224b1c57fbdcaa9bb6974253fd24975e098f4255fe0c7d0; coordination/results/TRR-0005.md 67c82d2b9b51cc32d405b0eacf751f40ffd17cae221a873d0cb53591e82dc30f |
| TRR-0007 | whole fit executions, including shared setup and diagnostics | 526.384407000 | experiments/TRR-0007/manifest.json e0e0bca00746825edffde0960e793a3877d74098a3d901e0c327925449ea4a7b; coordination/results/TRR-0007.md 5ded66b734f5e7997798ea93656bd0e9b098ee557f4b9fd0dff1cd046f656341 |
| TRR-0009 | continuation training whole run | 770.069974422 | experiments/TRR-0009/training/run_v1/run_receipt.json 6ad636b789fc6b39b432a1ce67c1f2347d627f7655fafa55244404d295da18d2 |

This is a sum of distinct published campaign scopes, not one elapsed wall clock
and not the required preparation cost for a single selected decoder. It includes
the TRR-0005 eight-fit history, all four TRR-0007 arms, and the TRR-0009
adaptable comparator plus unchanged-anchor reload. The TRR-0007 four-arm fit
sum (481.94894 s) and TRR-0009 arm sum (758.014065665 s) are alternate detail
and are not added to their respective whole-run figures.

## Selected shared-start lineage

The exact known full-arm fitting lower bound along the selected TRR-0009
lineage is **478.962289827 seconds**:

| lineage component | selected step | full-arm fit wall seconds | role |
|---|---:|---:|---|
| TRR-0007 current-enriched residual MLP-512 | 2100 | 123.139566510 | published shared starting state |
| TRR-0009 continued fixed readout | 400 | 355.822723317 | selected fixed continuation |

This lower bound excludes shared preparation, validation/checkpoint-selection
decomposition, initializer-construction time, and unknown earlier lineage costs.
The TRR-0009 adaptable arm is a separate comparator and is deliberately not
included.

The TRR-0007 residual initializer is traceable from the published run evidence
and source bindings: **TRR-0005 neutral identity-affine W/b/s=3; deterministic
diagonal QKV seed 4005; residual up zero**. The published builder calls
build_residual_mlp512, which builds the diagonal affine base through
build_current_positionwise and build_decoder with
DIAGONAL_ATTENTION_METHOD. There is no separately fitted affine initializer;
its separate construction wall is UNKNOWN, so no invented initializer fit cost
is added. The initializer evidence is:

- experiments/TRR-0007/method_freeze.json, SHA-256 301248af5cbdf95388ec1d39b42b00529227e4e3aa72613646dcd15a1f0dcfe9;
- experiments/TRR-0007/enriched_fit_v1/run_evidence.json, SHA-256 80959b3bac25d76158a0fcaf9381a71915c3d5bc19f4f63faa15665f721d5cc6;
- src/token_reconstruction/trr0007_positionwise.py, SHA-256 89fcd036a57407c8c49294aa5fd15ccef46f02b6fabcc35435d097f1213de485;
- src/token_reconstruction/trr0005_joint_decoder.py, SHA-256 9cab3cf5ad664c37d91bcd90fe1852df71da25800ddf37291a57000908d3d38f.

## Shared start and selected checkpoints

TRR-0009 starts from the published TRR-0007 current-enriched residual state:

- selected step **2100**, state bytes **29,390,628**, SHA-256
  2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8;
- state path experiments/TRR-0007/enriched_fit_v1/current_enriched/trr0007_residual_mlp512/selected.safetensors;
- its own full-arm fit wall was **123.139566510 s**, while the complete
  TRR-0007 four-arm campaign used the 526.384407 s whole-run scope above.

TRR-0009 selected its two continued arms at step **400** under a 3,000-step
schedule with validation every 100 steps and the earliest-maximum validation
metric:

| state | selected step | file bytes | file SHA-256 | arm wall seconds |
|---|---:|---:|---|---:|
| continued adaptable readout | 400 | 29,870,540 | 93661674c5e92144b84779737dee46a604afeb1ce1553966e464f0c84b2133c3 | 393.802617079 |
| continued fixed readout | 400 | 29,390,492 | 5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14 | 355.822723317 |
| unchanged anchor | 0 | 29,390,628 | 2a44a91b01ca1fdc9615eab804872c50c606199704eaf51fa114cc0d7959ddc8 | 8.388725269 |

A selected step identifies the exported checkpoint; it is not the cost of the
full schedule. Checkpoint-selection/validation wall time is **UNKNOWN as a
separate component** and is included in the recorded whole-run scopes.

## Shared preparation and capture

| task | shared activity | seconds | accounting |
|---|---|---:|---|
| TRR-0005 | full corpus preparation (18.158178837 s inner; 20.013262749 s receipt) | 20.013262749 | historical preparation, separate from its eight-fit total |
| TRR-0007 | campaign preparation total | 101.452513133 | accepted 23.666454504 + prior diagnostic 19.876863287 + superseded/repeated 57.909195342; components partition this total |
| TRR-0007 | improved activation capture | 8.224661700 | shared public-bank activation capture |
| TRR-0009 | continuation adapter/model preparation | 9.148822392 | charged once, not once per arm |
| TRR-0009 | fresh public observation capture | 16.958597586 | one capture shared by the method matrix |

TRR-0009 also reports prediction 145.870592506 s, precision timing
235.260320174 s, challenge 3.878089745 s, and largest-cell qualification
10.092721462 s. These are evaluation/lifecycle receipts and are not added to
the historical campaign sum or selected-lineage lower bound. The normalized
shared embedding table is 1,050,673,488 bytes (SHA-256
ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1); its
original construction time is not established here.

## Unknowns that must remain visible

- The historical A1 lens's original fit/preparation cost is explicitly
  unavailable in TRR-0005.
- TRR-0007's bounded A1+A2 diagnostic has fit_wall_seconds:null.
- The retained TRR-6/TRR-5 reference was not refit in TRR-0009, and its earlier
  fitting provenance is not declared by this lineage.
- There is no separately fitted affine initializer for the selected residual;
  neutral initializer construction time is UNKNOWN_SEPARATE.
- Foundation-model pretraining and any ancestors before the declared TRR-0005
  and TRR-0007 evidence are outside this ledger.
- Checkpoint-selection/validation decomposition and disk-cold startup/E
  construction time are unmeasured.

Evidence includes the TRR-0009 manifest 00d2e9bcb75962072dabdac50ad38b32e9ad638fae77c9da75faec1760048e64, training receipt
6ad636b789fc6b39b432a1ce67c1f2347d627f7655fafa55244404d295da18d2, qualification fed367ae4ce26aa0a2ee3d932522f481f02232ba1214474d7a50790a2391134e, challenge c8bbfcd1e94f750de7ef6601cff42d081eeafc3e1092663814a9ecb9133616b8,
capture 2a8ec28780de7b85ee2139d5e062aaf8cecd095fffa24bd1fc2aa9767b1c27b8, prediction baf0de29de9f4ce3697232c2a42addf27cf4fd03c753c52dc9de07fa5a68d844, timing 8f8e7a549cb9a58efad312bc4428b37856850ded275ee8bd49e487cea7711475,
and cost summary 15bb335ca8ed02a7196675f740946cc1a129bc3bb50e09bb94a910feb0abf765. Ancestor bindings are the TRR-0007
manifest e0e0bca00746825edffde0960e793a3877d74098a3d901e0c327925449ea4a7b/report 5ded66b734f5e7997798ea93656bd0e9b098ee557f4b9fd0dff1cd046f656341 and TRR-0005 manifest
ef670253241886e7d224b1c57fbdcaa9bb6974253fd24975e098f4255fe0c7d0/report 67c82d2b9b51cc32d405b0eacf751f40ffd17cae221a873d0cb53591e82dc30f.
