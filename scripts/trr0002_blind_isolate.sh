#!/usr/bin/env bash
# Launch the frozen TRR-0002 method inside a fail-closed native namespace.
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_ROOT CODE_ROOT MODEL_REPO SITE_PACKAGES OUTPUT_ROOT" >&2
  exit 2
}

if [[ $# -ne 5 && $# -ne 6 ]]; then
  usage
fi

trr2_input_root=$1
trr2_code_root=$2
trr2_model_repo=$3
trr2_site_packages=$4
trr2_output_root=$5

resolve_directory() {
  local trr2_candidate
  trr2_candidate=$(readlink -f -- "$1")
  if [[ ! -d "$trr2_candidate" || -L "$trr2_candidate" ]]; then
    echo "required regular directory is absent: $1" >&2
    exit 1
  fi
  printf '%s\n' "$trr2_candidate"
}

trr2_input_root=$(resolve_directory "$trr2_input_root")
trr2_code_root=$(resolve_directory "$trr2_code_root")
trr2_model_repo=$(resolve_directory "$trr2_model_repo")
trr2_site_packages=$(resolve_directory "$trr2_site_packages")

if [[ "${TRR2_ISOLATION_INTERNAL:-0}" != 1 ]]; then
  if [[ -e "$trr2_output_root" || -L "$trr2_output_root" ]]; then
    echo "blind method output is create-only: $trr2_output_root" >&2
    exit 1
  fi
  mkdir -p -- "$trr2_output_root"
  trr2_output_root=$(resolve_directory "$trr2_output_root")
  trr2_stage=$(mktemp -d "/tmp/trr0002-calibrated.XXXXXX")
  chmod 700 "$trr2_stage"
  export TRR2_ISOLATION_INTERNAL=1
  exec /usr/bin/unshare --user --map-root-user --mount --net --pid --fork -- \
    "$0" "$trr2_input_root" "$trr2_code_root" "$trr2_model_repo" \
    "$trr2_site_packages" "$trr2_output_root" "$trr2_stage"
fi

if [[ $# -ne 6 ]]; then
  usage
fi
trr2_stage=$6
case "$trr2_stage" in
  /tmp/trr0002-calibrated.*) ;;
  *)
    echo "unexpected TRR-0002 isolation staging path" >&2
    exit 1
    ;;
esac
if [[ ! -d "$trr2_stage" || -L "$trr2_stage" ]]; then
  echo "TRR-0002 isolation staging root is invalid" >&2
  exit 1
fi

/usr/bin/mount --make-rprivate /
mkdir -p \
  "$trr2_stage/code" \
  "$trr2_stage/dev/shm" \
  "$trr2_stage/etc" \
  "$trr2_stage/input" \
  "$trr2_stage/model-repo" \
  "$trr2_stage/output" \
  "$trr2_stage/proc" \
  "$trr2_stage/site-packages" \
  "$trr2_stage/tmp/hf" \
  "$trr2_stage/usr" \
  "$trr2_stage/.old_root"

/usr/bin/mount --bind "$trr2_stage" "$trr2_stage"

bind_read_only() {
  local -a trr2_mount_targets
  local trr2_source=$1
  local trr2_destination=$2
  local trr2_position
  /usr/bin/mount --rbind "$trr2_source" "$trr2_destination"
  /usr/bin/mount --make-rslave "$trr2_destination"
  mapfile -t trr2_mount_targets < <(/usr/bin/findmnt -Rrn -o TARGET "$trr2_destination")
  for ((trr2_position=${#trr2_mount_targets[@]} - 1; trr2_position >= 0; trr2_position--)); do
    /usr/bin/mount -o remount,bind,ro "${trr2_mount_targets[$trr2_position]}"
  done
}

/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr2_stage/tmp"
/usr/bin/mkdir -p "$trr2_stage/tmp/hf"
/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr2_stage/dev/shm"
bind_read_only /usr "$trr2_stage/usr"
bind_read_only /etc "$trr2_stage/etc"
bind_read_only "$trr2_code_root" "$trr2_stage/code"
bind_read_only "$trr2_input_root" "$trr2_stage/input"
bind_read_only "$trr2_model_repo" "$trr2_stage/model-repo"
bind_read_only "$trr2_site_packages" "$trr2_stage/site-packages"
/usr/bin/mount --bind "$trr2_output_root" "$trr2_stage/output"

for trr2_device in null zero random urandom dxg; do
  if [[ ! -e "/dev/$trr2_device" ]]; then
    echo "required device is absent: /dev/$trr2_device" >&2
    exit 1
  fi
  : > "$trr2_stage/dev/$trr2_device"
  /usr/bin/mount --bind "/dev/$trr2_device" "$trr2_stage/dev/$trr2_device"
  /usr/bin/mount -o remount,bind,ro "$trr2_stage/dev/$trr2_device"
done
ln -s /proc/self/fd "$trr2_stage/dev/fd"
ln -s /proc/self/fd/0 "$trr2_stage/dev/stdin"
ln -s /proc/self/fd/1 "$trr2_stage/dev/stdout"
ln -s /proc/self/fd/2 "$trr2_stage/dev/stderr"
/usr/bin/mount -t proc -o nosuid,nodev,noexec proc "$trr2_stage/proc"

ln -s usr/bin "$trr2_stage/bin"
ln -s usr/sbin "$trr2_stage/sbin"
ln -s usr/lib "$trr2_stage/lib"
if [[ -d /usr/lib64 ]]; then
  ln -s usr/lib64 "$trr2_stage/lib64"
fi

/usr/sbin/pivot_root "$trr2_stage" "$trr2_stage/.old_root"
cd /
/usr/bin/umount -l /.old_root
/usr/bin/rmdir /.old_root
/usr/bin/mount -o remount,bind,ro /

trr2_env=(
  PATH=/usr/bin:/usr/sbin
  PYTHONPATH=/code/src:/code:/site-packages
  HF_HOME=/tmp/hf
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false
  LD_LIBRARY_PATH=/site-packages/torch/lib:/usr/lib/wsl/lib
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONNOUSERSITE=1
  LANG=C.UTF-8
)

/usr/bin/env -i "${trr2_env[@]}" /usr/bin/python3 -s \
  /code/scripts/trr0002_blind_access_probe.py \
  --output /output/access_manifest.json

exec /usr/bin/env -i "${trr2_env[@]}" /usr/bin/python3 -s \
  /code/scripts/trr0002_blind_reconstruct.py \
  --config /input/sanitized_config.json \
  --input-root /input \
  --model-path "/model-repo/snapshots/9213176726f574b556790deb65791e0c5aa438b6" \
  --reference-source /code/reference/strict_bos/round001_teacher.py \
  --output-directory /output \
  --access-manifest /output/access_manifest.json
