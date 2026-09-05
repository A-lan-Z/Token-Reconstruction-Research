#!/usr/bin/env bash
set -o pipefail
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:scripts/trr_p01 /usr/bin/python3 experiments/TRR-P01/runtime/build_joint_freeze_sidecar.py --sidecar experiments/TRR-P01/runtime/joint-freeze-sidecar.json --validation experiments/TRR-P01/runtime/joint-freeze-validation.json --runtime-root experiments/TRR-P01/runtime
