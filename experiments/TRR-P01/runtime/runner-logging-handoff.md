TRR-P01 runner logging handoff

Observed timeout
- arm-000 bounded command at implementation commit 6e05feeade57593cdadea2d4db4ce40085a51f59 returned timeout exit 124.
- Intended timeout deadline was 2026-09-05T08:46:50Z (210 seconds after 08:43:20Z start); exit was collected at 2026-09-05T08:48:28.985094Z.
- Process verification found no live PID. Output contains only preflight.json; no predictions, truth, or arm-001. Phase reached is UNKNOWN because that committed runner had no phase telemetry. The outer log contains a model-loader progress line, which is insufficient to attribute the timeout to model loading or static lookup.

Pending source diff (root review/commit required before a fresh run)
- scripts/trr_p01/reconstruct.py only.
- Adds create-only phase_progress.jsonl with fsync after every JSONL event.
- Logs start/end for preflight_resource_guard, prototype-table load, public model/prefix load, aggregate static lookup, each boundary/raw-embedding metric arm, reference correction aggregate and each metric, and historical fixed-K256 aggregate. Compute elapsed values are captured before end-event fsync; start events are flushed before compute timers begin.
- Successful evidence/finish receipts bind the stable phase-progress file hash. Numerical code, metric order, chunk sizes, and batching are unchanged.
- CPU-only py_compile passed; no heavy computation was run after timeout.

Geometry diagnosis
- Final static phase: observations[:,1:,:] gives 16*39=624 query rows for each of four full-vocabulary arms: boundary cosine/L2 and raw-embedding cosine/L2, with query chunks 256 and prototype chunks 8192.
- Qualification: only 8 public rows at one position, two PrototypeTable metrics (8 rows x cosine/L2); it does not exercise raw-embedding lookup or 39-position static routing.
- Therefore the qualifier does not predict full end-to-end runtime; no phase attribution or extrapolated timeout cause is claimed.

Fresh-run preparation
- Runner logging is committed as e43a595d0f4300d5db8f93c86881b455dfa30ea4.
- Local pinned public checkpoint verified at /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6.
- model.safetensors: 2,471,645,608 bytes; SHA256 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f. config.json SHA256 2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb; tokenizer_config.json SHA256 9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e. Metadata matches revision 9213176726f574b556790deb65791e0c5aa438b6, LlamaForCausalLM, hidden 2048, vocab 128256, BF16, BOS 128000.
- Exact future commands are in reconstruct-final-r2-arm-000-command.sh and reconstruct-final-r2-arm-001-command.sh. They set local-only model path, CPU, correction, fixed historical lens, implementation commit e43a595, and a 1,100-second fail-closed timeout; both runs are serial and create-only.
- Preparation/config records: reconstruct-final-r2/preparation.json and reconstruct-final-r2-arm-{000,001}-reservation.json. No command has been launched; status remains pending explicit CPU release.
