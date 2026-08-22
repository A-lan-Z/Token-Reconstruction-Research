#!/usr/bin/env bash
# Launch one TRR-0001-R1 method in a fail-closed native namespace boundary.
set -euo pipefail

usage() {
  echo "usage: $0 METHOD INPUT_ROOT CODE_ROOT MODEL_REPO SITE_PACKAGES OUTPUT_ROOT" >&2
  exit 2
}

if [[ $# -ne 6 && $# -ne 7 ]]; then
  usage
fi

trr_r1_method=$1
trr_r1_input_root=$2
trr_r1_code_root=$3
trr_r1_model_repo=$4
trr_r1_site_packages=$5
trr_r1_output_root=$6

case "$trr_r1_method" in
  direct_inverse|causal_public_surrogate_search) ;;
  *) usage ;;
esac

resolve_directory() {
  local trr_r1_candidate
  trr_r1_candidate=$(readlink -f -- "$1")
  if [[ ! -d "$trr_r1_candidate" || -L "$trr_r1_candidate" ]]; then
    echo "required regular directory is absent: $1" >&2
    exit 1
  fi
  printf '%s\n' "$trr_r1_candidate"
}

trr_r1_input_root=$(resolve_directory "$trr_r1_input_root")
trr_r1_code_root=$(resolve_directory "$trr_r1_code_root")
trr_r1_model_repo=$(resolve_directory "$trr_r1_model_repo")
trr_r1_site_packages=$(resolve_directory "$trr_r1_site_packages")

if [[ "${TRR_R1_ISOLATION_INTERNAL:-0}" != 1 ]]; then
  if [[ -e "$trr_r1_output_root" || -L "$trr_r1_output_root" ]]; then
    echo "method output is create-only: $trr_r1_output_root" >&2
    exit 1
  fi
  mkdir -p -- "$trr_r1_output_root"
  trr_r1_output_root=$(resolve_directory "$trr_r1_output_root")
  trr_r1_stage=$(mktemp -d "/tmp/trr0001-r1-${trr_r1_method}.XXXXXX")
  chmod 700 "$trr_r1_stage"
  export TRR_R1_ISOLATION_INTERNAL=1
  exec /usr/bin/unshare --user --map-root-user --mount --net --pid --fork --     "$0" "$trr_r1_method" "$trr_r1_input_root" "$trr_r1_code_root"     "$trr_r1_model_repo" "$trr_r1_site_packages" "$trr_r1_output_root"     "$trr_r1_stage"
fi

if [[ $# -ne 7 ]]; then
  usage
fi
trr_r1_stage=$7
case "$trr_r1_stage" in
  /tmp/trr0001-r1-direct_inverse.*|/tmp/trr0001-r1-causal_public_surrogate_search.*) ;;
  *)
    echo "unexpected isolation staging path" >&2
    exit 1
    ;;
esac
if [[ ! -d "$trr_r1_stage" || -L "$trr_r1_stage" ]]; then
  echo "isolation staging root is invalid" >&2
  exit 1
fi

/usr/bin/mount --make-rprivate /
mkdir -p   "$trr_r1_stage/code"   "$trr_r1_stage/dev/shm"   "$trr_r1_stage/etc"   "$trr_r1_stage/input"   "$trr_r1_stage/model-repo"   "$trr_r1_stage/output"   "$trr_r1_stage/proc"   "$trr_r1_stage/site-packages"   "$trr_r1_stage/tmp/home"   "$trr_r1_stage/tmp/hf"   "$trr_r1_stage/usr"   "$trr_r1_stage/.old_root"

/usr/bin/mount --bind "$trr_r1_stage" "$trr_r1_stage"

bind_read_only() {
  local trr_r1_source=$1
  local trr_r1_destination=$2
  /usr/bin/mount --rbind "$trr_r1_source" "$trr_r1_destination"
  /usr/bin/mount --make-rslave "$trr_r1_destination"
  /usr/bin/mount -o remount,bind,ro "$trr_r1_destination"
}

/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr_r1_stage/tmp"
/usr/bin/mkdir -p "$trr_r1_stage/tmp/home" "$trr_r1_stage/tmp/hf"
/usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$trr_r1_stage/dev/shm"
bind_read_only /usr "$trr_r1_stage/usr"
bind_read_only /etc "$trr_r1_stage/etc"
bind_read_only "$trr_r1_code_root" "$trr_r1_stage/code"
bind_read_only "$trr_r1_input_root" "$trr_r1_stage/input"
bind_read_only "$trr_r1_model_repo" "$trr_r1_stage/model-repo"
bind_read_only "$trr_r1_site_packages" "$trr_r1_stage/site-packages"
/usr/bin/mount --bind "$trr_r1_output_root" "$trr_r1_stage/output"

for trr_r1_device in null zero random urandom dxg; do
  if [[ ! -e "/dev/$trr_r1_device" ]]; then
    echo "required device is absent: /dev/$trr_r1_device" >&2
    exit 1
  fi
  : > "$trr_r1_stage/dev/$trr_r1_device"
  /usr/bin/mount --bind "/dev/$trr_r1_device" "$trr_r1_stage/dev/$trr_r1_device"
  /usr/bin/mount -o remount,bind,ro "$trr_r1_stage/dev/$trr_r1_device"
done
ln -s /proc/self/fd "$trr_r1_stage/dev/fd"
ln -s /proc/self/fd/0 "$trr_r1_stage/dev/stdin"
ln -s /proc/self/fd/1 "$trr_r1_stage/dev/stdout"
ln -s /proc/self/fd/2 "$trr_r1_stage/dev/stderr"
/usr/bin/mount -t proc -o nosuid,nodev,noexec proc "$trr_r1_stage/proc"

ln -s usr/bin "$trr_r1_stage/bin"
ln -s usr/sbin "$trr_r1_stage/sbin"
ln -s usr/lib "$trr_r1_stage/lib"
if [[ -d /usr/lib64 ]]; then
  ln -s usr/lib64 "$trr_r1_stage/lib64"
fi

/usr/sbin/pivot_root "$trr_r1_stage" "$trr_r1_stage/.old_root"
cd /
/usr/bin/umount -l /.old_root
/usr/bin/rmdir /.old_root
/usr/bin/mount -o remount,bind,ro /

trr_r1_env=(
  HOME=/tmp/home
  PATH=/usr/bin:/usr/sbin
  PYTHONPATH=/code/src:/site-packages
  HF_HOME=/tmp/hf
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false
  LD_LIBRARY_PATH=/site-packages/torch/lib:/usr/lib/wsl/lib
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONNOUSERSITE=1
  LANG=C.UTF-8
)

/usr/bin/env -i "${trr_r1_env[@]}" /usr/bin/python3 -s   /code/scripts/trr0001_r1_access_probe.py   --method "$trr_r1_method"   --output /output/access_manifest.json

exec /usr/bin/env -i "${trr_r1_env[@]}" /usr/bin/python3 -s   /code/scripts/trr0001_r1_reconstruct.py   --config /input/sanitized_config.json   --input-root /input   --model-path "/model-repo/snapshots/9213176726f574b556790deb65791e0c5aa438b6"   --output-directory /output   --method "$trr_r1_method"   --access-manifest /output/access_manifest.json
