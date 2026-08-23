#!/usr/bin/env bash
# Run one R3 proposer against sanitized observations in a fail-closed namespace.
set -euo pipefail

usage() {
  echo "usage: $0 PROPOSER INPUT_ROOT CODE_ROOT BASE_MODEL_SNAPSHOT SITE_PACKAGES OUTPUT_ROOT [CONTROL_ASSET_ROOT]" >&2
  exit 2
}

if [[ $# -lt 6 || $# -gt 8 ]]; then
  usage
fi

trr3_proposer=$1
trr3_input_root=$2
trr3_code_root=$3
trr3_model_root=$4
trr3_site_packages=$5
trr3_output_root=$6
trr3_asset_root=""

case "$trr3_proposer" in
  checkpoint_identity)
    if [[ "${TRR3_ISOLATION_INTERNAL:-0}" != 1 && $# -ne 6 ]]; then
      usage
    fi
    ;;
  alpaca_affine_control)
    if [[ "${TRR3_ISOLATION_INTERNAL:-0}" != 1 && $# -ne 7 ]]; then
      usage
    fi
    trr3_asset_root=$7
    ;;
  *)
    echo "unknown R3 proposer: $trr3_proposer" >&2
    exit 2
    ;;
esac

resolve_directory() {
  local trr3_candidate
  trr3_candidate=$(readlink -f -- "$1")
  if [[ ! -d "$trr3_candidate" || -L "$trr3_candidate" ]]; then
    echo "required regular directory is absent: $1" >&2
    exit 1
  fi
  printf '%s\n' "$trr3_candidate"
}

trr3_input_root=$(resolve_directory "$trr3_input_root")
trr3_code_root=$(resolve_directory "$trr3_code_root")
trr3_model_root=$(resolve_directory "$trr3_model_root")
trr3_site_packages=$(resolve_directory "$trr3_site_packages")
if [[ -n "$trr3_asset_root" ]]; then
  trr3_asset_root=$(resolve_directory "$trr3_asset_root")
  if [[ ! -f "$trr3_asset_root/lens_alpaca.pt" || -L "$trr3_asset_root/lens_alpaca.pt" ]]; then
    echo "control asset root lacks lens_alpaca.pt" >&2
    exit 1
  fi
fi

if [[ "${TRR3_ISOLATION_INTERNAL:-0}" != 1 ]]; then
  if [[ -e "$trr3_output_root" || -L "$trr3_output_root" ]]; then
    echo "R3 prediction output is create-only: $trr3_output_root" >&2
    exit 1
  fi
  mkdir -p -- "$trr3_output_root"
  trr3_output_root=$(resolve_directory "$trr3_output_root")
  trr3_stage=$(mktemp -d "/tmp/trr0002-owner-r3.XXXXXX")
  chmod 700 "$trr3_stage"
  export TRR3_ISOLATION_INTERNAL=1
  if [[ -n "$trr3_asset_root" ]]; then
    exec /usr/bin/unshare --user --map-root-user --mount --net --pid --fork -- \
      "$0" "$trr3_proposer" "$trr3_input_root" "$trr3_code_root" \
      "$trr3_model_root" "$trr3_site_packages" "$trr3_output_root" \
      "$trr3_asset_root" "$trr3_stage"
  fi
  exec /usr/bin/unshare --user --map-root-user --mount --net --pid --fork -- \
    "$0" "$trr3_proposer" "$trr3_input_root" "$trr3_code_root" \
    "$trr3_model_root" "$trr3_site_packages" "$trr3_output_root" "$trr3_stage"
fi

if [[ "$trr3_proposer" == "checkpoint_identity" ]]; then
  if [[ $# -ne 7 ]]; then
    usage
  fi
  trr3_stage=$7
else
  if [[ $# -ne 8 ]]; then
    usage
  fi
  trr3_stage=$8
fi
case "$trr3_stage" in
  /tmp/trr0002-owner-r3.*) ;;
  *)
    echo "unexpected R3 isolation staging path" >&2
    exit 1
    ;;
esac
if [[ ! -d "$trr3_stage" || -L "$trr3_stage" ]]; then
  echo "R3 isolation staging root is invalid" >&2
  exit 1
fi

/usr/bin/mount --make-rprivate /
mkdir -p \
  "$trr3_stage/code" \
  "$trr3_stage/dev/shm" \
  "$trr3_stage/etc" \
  "$trr3_stage/input" \
  "$trr3_stage/model-repo" \
  "$trr3_stage/output" \
  "$trr3_stage/proc" \
  "$trr3_stage/site-packages" \
  "$trr3_stage/tmp/hf" \
  "$trr3_stage/usr" \
  "$trr3_stage/.old_root"
if [[ -n "$trr3_asset_root" ]]; then
  mkdir -p "$trr3_stage/assets"
fi

/usr/bin/mount --bind "$trr3_stage" "$trr3_stage"

bind_read_only() {
  local -a trr3_mount_targets
  local trr3_source=$1
  local trr3_destination=$2
  local trr3_position
  /usr/bin/mount --rbind "$trr3_source" "$trr3_destination"
  /usr/bin/mount --make-rslave "$trr3_destination"
  mapfile -t trr3_mount_targets < <(/usr/bin/findmnt -Rrn -o TARGET "$trr3_destination")
  for ((trr3_position=${#trr3_mount_targets[@]} - 1; trr3_position >= 0; trr3_position--)); do
    /usr/bin/mount -o remount,bind,ro "${trr3_mount_targets[$trr3_position]}"
  done
}

/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr3_stage/tmp"
/usr/bin/mkdir -p "$trr3_stage/tmp/hf"
/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr3_stage/dev/shm"
bind_read_only /usr "$trr3_stage/usr"
bind_read_only /etc "$trr3_stage/etc"
bind_read_only "$trr3_code_root" "$trr3_stage/code"
bind_read_only "$trr3_input_root" "$trr3_stage/input"
bind_read_only "$trr3_model_root" "$trr3_stage/model-repo"
bind_read_only "$trr3_site_packages" "$trr3_stage/site-packages"
if [[ -n "$trr3_asset_root" ]]; then
  bind_read_only "$trr3_asset_root" "$trr3_stage/assets"
fi
/usr/bin/mount --bind "$trr3_output_root" "$trr3_stage/output"

for trr3_device in null zero random urandom dxg; do
  if [[ ! -e "/dev/$trr3_device" ]]; then
    echo "required device is absent: /dev/$trr3_device" >&2
    exit 1
  fi
  : > "$trr3_stage/dev/$trr3_device"
  /usr/bin/mount --bind "/dev/$trr3_device" "$trr3_stage/dev/$trr3_device"
  /usr/bin/mount -o remount,bind,ro "$trr3_stage/dev/$trr3_device"
done
ln -s /proc/self/fd "$trr3_stage/dev/fd"
ln -s /proc/self/fd/0 "$trr3_stage/dev/stdin"
ln -s /proc/self/fd/1 "$trr3_stage/dev/stdout"
ln -s /proc/self/fd/2 "$trr3_stage/dev/stderr"
/usr/bin/mount -t proc -o nosuid,nodev,noexec proc "$trr3_stage/proc"

ln -s usr/bin "$trr3_stage/bin"
ln -s usr/sbin "$trr3_stage/sbin"
ln -s usr/lib "$trr3_stage/lib"
if [[ -d /usr/lib64 ]]; then
  ln -s usr/lib64 "$trr3_stage/lib64"
fi

/usr/sbin/pivot_root "$trr3_stage" "$trr3_stage/.old_root"
cd /
/usr/bin/umount -l /.old_root
/usr/bin/rmdir /.old_root
/usr/bin/mount -o remount,bind,ro /

trr3_env=(
  PATH=/usr/bin:/usr/sbin
  PYTHONPATH=/code/src:/code:/site-packages
  HF_HOME=/tmp/hf
  HF_HUB_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false
  LD_LIBRARY_PATH=/site-packages/torch/lib:/usr/lib/wsl/lib
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONNOUSERSITE=1
  LANG=C.UTF-8
)

/usr/bin/env -i "${trr3_env[@]}" /usr/bin/python3 -s \
  /code/scripts/trr0002_r3_access_probe.py \
  --proposer "$trr3_proposer" \
  --output /output/access_manifest.json

trr3_predict_args=(
  /code/scripts/trr0002_r3_strict_surrogate.py predict
  --config /input/config.json
  --input-root /input
  --model-path /model-repo
  --proposer "$trr3_proposer"
  --output-directory /output
  --access-manifest /output/access_manifest.json
)
if [[ "$trr3_proposer" == "alpaca_affine_control" ]]; then
  trr3_predict_args+=(--lens-path /assets/lens_alpaca.pt)
fi

exec /usr/bin/env -i "${trr3_env[@]}" /usr/bin/python3 -s "${trr3_predict_args[@]}"
