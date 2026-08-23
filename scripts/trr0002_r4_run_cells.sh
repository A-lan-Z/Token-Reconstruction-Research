#!/usr/bin/env bash
# Execute each R4 target/policy cell in a fresh process to bound CUDA memory.
set -euo pipefail

trr4_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$trr4_root"

trr4_plan="experiments/TRR-0002/historical-input-target-bridge/preregistration.json"
trr4_input="outputs/TRR-0002/historical-input-target-bridge/reconstructor_input"
trr4_parts="outputs/TRR-0002/historical-input-target-bridge/parts"
trr4_combined="outputs/TRR-0002/historical-input-target-bridge/predictions"
trr4_model="/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
trr4_lens="/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/out/lens_alpaca.pt"

trr4_conditions=(
  public_base_target_cut4
  finance_generation300_target_cut4
  vikhr_heavy_target_cut4
)
trr4_historical_policies=(
  a1a2_589f6e179eb4626877c2
  a1a2_c316cdf581012bd81cfa
  a1a2_422b282c012ff665ee2e
  a1a2_43ea0bb737bc075531ca
  a1a2_cb89b524f27c2d5e25eb
  a1a2_cce5e6b5435e9b1bee34
  a1a2_13f73c306bf8946e9a28
  a1a2_91503c1f37fac38c4e20
  a1a2_ae35177bb01fa67279c3
  a1a2_a49923b51936a41a41fb
  a1a2_d1700b9c0f2b1b32ec13
  a1a2_6de800ba92c3d0ec0808
)
trr4_checkpoint_policies=(
  a1a2_589f6e179eb4626877c2
  a1a2_43ea0bb737bc075531ca
  a1a2_13f73c306bf8946e9a28
)

export PYTHONPATH=src:scripts
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_cell() {
  local trr4_proposer=$1
  local trr4_condition=$2
  local trr4_policy=$3
  local trr4_output="$trr4_parts/$trr4_proposer/$trr4_condition/$trr4_policy"
  local trr4_time="$trr4_parts/$trr4_proposer/$trr4_condition/${trr4_policy}.time.txt"
  if [[ -d "$trr4_output" ]]; then
    if [[ -s "$trr4_output/evidence.json" && -s "$trr4_output/predictions.safetensors" && -s "$trr4_time" ]]; then
      echo "R4 cell already complete; preserving and skipping: $trr4_output"
      return 0
    fi
    echo "R4 cell output is incomplete and will not be overwritten: $trr4_output" >&2
    exit 1
  fi
  if [[ -e "$trr4_output" || -L "$trr4_output" ]]; then
    echo "R4 cell output path is not a directory: $trr4_output" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$trr4_output")"
  python3 scripts/trr_compute_preflight.py \
    --minimum-free-gib 10 \
    --probe-mib 600 \
    > "${trr4_time%.time.txt}.preflight.json"
  trr4_args=(
    scripts/trr0002_r4_historical_target_bridge.py predict
    --repository-root .
    --plan "$trr4_plan"
    --input-root "$trr4_input"
    --model-path "$trr4_model"
    --proposer "$trr4_proposer"
    --condition-id "$trr4_condition"
    --policy-id "$trr4_policy"
    --output-directory "$trr4_output"
  )
  if [[ "$trr4_proposer" == historical_alpaca_affine_a1 ]]; then
    trr4_args+=(--lens-path "$trr4_lens")
  fi
  /usr/bin/time -v -o "$trr4_time" python3 "${trr4_args[@]}"
  sleep 2
}

for trr4_condition in "${trr4_conditions[@]}"; do
  for trr4_policy in "${trr4_historical_policies[@]}"; do
    run_cell historical_alpaca_affine_a1 "$trr4_condition" "$trr4_policy"
  done
done
for trr4_condition in "${trr4_conditions[@]}"; do
  for trr4_policy in "${trr4_checkpoint_policies[@]}"; do
    run_cell checkpoint_identity_a1 "$trr4_condition" "$trr4_policy"
  done
done

trr4_historical_parts=()
for trr4_condition in "${trr4_conditions[@]}"; do
  for trr4_policy in "${trr4_historical_policies[@]}"; do
    trr4_historical_parts+=(
      --part-directory
      "$trr4_parts/historical_alpaca_affine_a1/$trr4_condition/$trr4_policy"
    )
  done
done
python3 scripts/trr0002_r4_historical_target_bridge.py combine \
  --repository-root . \
  --plan "$trr4_plan" \
  --input-root "$trr4_input" \
  "${trr4_historical_parts[@]}" \
  --output-directory "$trr4_combined/historical-alpaca-a1"

trr4_checkpoint_parts=()
for trr4_condition in "${trr4_conditions[@]}"; do
  for trr4_policy in "${trr4_checkpoint_policies[@]}"; do
    trr4_checkpoint_parts+=(
      --part-directory
      "$trr4_parts/checkpoint_identity_a1/$trr4_condition/$trr4_policy"
    )
  done
done
python3 scripts/trr0002_r4_historical_target_bridge.py combine \
  --repository-root . \
  --plan "$trr4_plan" \
  --input-root "$trr4_input" \
  "${trr4_checkpoint_parts[@]}" \
  --output-directory "$trr4_combined/checkpoint-identity-a1"
