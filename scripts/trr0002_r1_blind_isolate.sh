#!/usr/bin/env bash
# Launch the frozen owner-R1 winner inside a fail-closed native namespace.
set -euo pipefail

usage() {
  echo "usage: $0 INPUT_ROOT CODE_ROOT MODEL_REPO SITE_PACKAGES OUTPUT_ROOT" >&2
  exit 2
}

if [[ $# -ne 5 && $# -ne 6 ]]; then
  usage
fi

trr2r1_input_root=$1
trr2r1_code_root=$2
trr2r1_model_repo=$3
trr2r1_site_packages=$4
trr2r1_output_root=$5

resolve_directory() {
  local trr2r1_candidate
  trr2r1_candidate=$(readlink -f -- "$1")
  if [[ ! -d "$trr2r1_candidate" || -L "$trr2r1_candidate" ]]; then
    echo "required regular directory is absent: $1" >&2
    exit 1
  fi
  printf '%s\n' "$trr2r1_candidate"
}

trr2r1_input_root=$(resolve_directory "$trr2r1_input_root")
trr2r1_code_root=$(resolve_directory "$trr2r1_code_root")
trr2r1_model_repo=$(resolve_directory "$trr2r1_model_repo")
trr2r1_site_packages=$(resolve_directory "$trr2r1_site_packages")

if [[ "${TRR2R1_ISOLATION_INTERNAL:-0}" != 1 ]]; then
  if [[ -e "$trr2r1_output_root" || -L "$trr2r1_output_root" ]]; then
    echo "blind method output is create-only: $trr2r1_output_root" >&2
    exit 1
  fi
  mkdir -p -- "$trr2r1_output_root"
  trr2r1_output_root=$(resolve_directory "$trr2r1_output_root")
  trr2r1_stage=$(mktemp -d "/tmp/trr0002-owner-r1.XXXXXX")
  chmod 700 "$trr2r1_stage"
  export TRR2R1_ISOLATION_INTERNAL=1
  exec /usr/bin/unshare --user --map-root-user --mount --net --pid --fork -- \
    "$0" "$trr2r1_input_root" "$trr2r1_code_root" "$trr2r1_model_repo" \
    "$trr2r1_site_packages" "$trr2r1_output_root" "$trr2r1_stage"
fi

if [[ $# -ne 6 ]]; then
  usage
fi
trr2r1_stage=$6
case "$trr2r1_stage" in
  /tmp/trr0002-owner-r1.*) ;;
  *)
    echo "unexpected owner-R1 isolation staging path" >&2
    exit 1
    ;;
esac
if [[ ! -d "$trr2r1_stage" || -L "$trr2r1_stage" ]]; then
  echo "owner-R1 isolation staging root is invalid" >&2
  exit 1
fi

/usr/bin/mount --make-rprivate /
mkdir -p \
  "$trr2r1_stage/code" \
  "$trr2r1_stage/dev/shm" \
  "$trr2r1_stage/etc" \
  "$trr2r1_stage/input" \
  "$trr2r1_stage/model-repo" \
  "$trr2r1_stage/output" \
  "$trr2r1_stage/proc" \
  "$trr2r1_stage/site-packages" \
  "$trr2r1_stage/tmp/hf" \
  "$trr2r1_stage/usr" \
  "$trr2r1_stage/.old_root"

/usr/bin/mount --bind "$trr2r1_stage" "$trr2r1_stage"

bind_read_only() {
  local -a trr2r1_mount_targets
  local trr2r1_source=$1
  local trr2r1_destination=$2
  local trr2r1_position
  /usr/bin/mount --rbind "$trr2r1_source" "$trr2r1_destination"
  /usr/bin/mount --make-rslave "$trr2r1_destination"
  mapfile -t trr2r1_mount_targets < <(/usr/bin/findmnt -Rrn -o TARGET "$trr2r1_destination")
  for ((trr2r1_position=${#trr2r1_mount_targets[@]} - 1; trr2r1_position >= 0; trr2r1_position--)); do
    /usr/bin/mount -o remount,bind,ro "${trr2r1_mount_targets[$trr2r1_position]}"
  done
}

/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr2r1_stage/tmp"
/usr/bin/mkdir -p "$trr2r1_stage/tmp/hf"
/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr2r1_stage/dev/shm"
bind_read_only /usr "$trr2r1_stage/usr"
bind_read_only /etc "$trr2r1_stage/etc"
bind_read_only "$trr2r1_code_root" "$trr2r1_stage/code"
bind_read_only "$trr2r1_input_root" "$trr2r1_stage/input"
bind_read_only "$trr2r1_model_repo" "$trr2r1_stage/model-repo"
bind_read_only "$trr2r1_site_packages" "$trr2r1_stage/site-packages"
/usr/bin/mount --bind "$trr2r1_output_root" "$trr2r1_stage/output"

for trr2r1_device in null zero random urandom dxg; do
  if [[ ! -e "/dev/$trr2r1_device" ]]; then
    echo "required device is absent: /dev/$trr2r1_device" >&2
    exit 1
  fi
  : > "$trr2r1_stage/dev/$trr2r1_device"
  /usr/bin/mount --bind "/dev/$trr2r1_device" "$trr2r1_stage/dev/$trr2r1_device"
  /usr/bin/mount -o remount,bind,ro "$trr2r1_stage/dev/$trr2r1_device"
done
ln -s /proc/self/fd "$trr2r1_stage/dev/fd"
ln -s /proc/self/fd/0 "$trr2r1_stage/dev/stdin"
ln -s /proc/self/fd/1 "$trr2r1_stage/dev/stdout"
ln -s /proc/self/fd/2 "$trr2r1_stage/dev/stderr"
/usr/bin/mount -t proc -o nosuid,nodev,noexec proc "$trr2r1_stage/proc"

ln -s usr/bin "$trr2r1_stage/bin"
ln -s usr/sbin "$trr2r1_stage/sbin"
ln -s usr/lib "$trr2r1_stage/lib"
if [[ -d /usr/lib64 ]]; then
  ln -s usr/lib64 "$trr2r1_stage/lib64"
fi

/usr/sbin/pivot_root "$trr2r1_stage" "$trr2r1_stage/.old_root"
cd /
/usr/bin/umount -l /.old_root
/usr/bin/rmdir /.old_root
/usr/bin/mount -o remount,bind,ro /

trr2r1_env=(
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

/usr/bin/env -i "${trr2r1_env[@]}" /usr/bin/python3 -s \
  /code/scripts/trr0002_r1_blind_access_probe.py \
  --output /output/access_manifest.json

exec /usr/bin/env -i "${trr2r1_env[@]}" /usr/bin/python3 -s \
  /code/scripts/trr0002_r1_blind_reconstruct.py \
  --config /input/sanitized_config.json \
  --input-root /input \
  --model-path "/model-repo/snapshots/9213176726f574b556790deb65791e0c5aa438b6" \
  --reference-source /code/reference/strict_bos/round001_teacher.py \
  --output-directory /output \
  --access-manifest /output/access_manifest.json
