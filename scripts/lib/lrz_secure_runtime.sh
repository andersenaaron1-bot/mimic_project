#!/usr/bin/env bash
# shellcheck shell=bash

if [[ -n "${_LRZ_SECURE_RUNTIME_SH:-}" ]]; then
  return 0
fi
_LRZ_SECURE_RUNTIME_SH=1

lrz_log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

lrz_die() {
  lrz_log "ERROR: $*"
  exit 1
}

lrz_have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

lrz_require_cmd() {
  local cmd="$1"
  lrz_have_cmd "$cmd" || lrz_die "Required command not found: $cmd"
}

lrz_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

lrz_is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

lrz_setup_job_hardening() {
  umask "${JOB_UMASK:-077}"
  ulimit -c 0 >/dev/null 2>&1 || true
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}

lrz_make_runtime_root() {
  local base="${RUNTIME_BASE_DIR:-${SLURM_TMPDIR:-/tmp}}"
  mkdir -p "$base"
  local root
  root="$(mktemp -d "${base%/}/ehr-runtime-${SLURM_JOB_ID:-$$}-XXXX")"
  chmod 700 "$root" || true
  printf '%s\n' "$root"
}

lrz_pick_container_image() {
  local local_image="${1:-}"
  local remote_image="${2:-}"

  if [[ -n "$local_image" && -f "$local_image" ]]; then
    printf '%s\n' "$local_image"
    return 0
  fi

  if [[ -n "$remote_image" ]]; then
    printf '%s\n' "$remote_image"
    return 0
  fi

  lrz_die "No container image available. Set IMAGE/IMAGE_LOCAL or IMAGE_REMOTE."
}

lrz_crypto_backend() {
  local requested="${EHR_ENCRYPTION_BACKEND:-auto}"
  case "${requested,,}" in
    auto)
      if lrz_have_cmd age; then
        printf 'age\n'
      else
        printf 'openssl\n'
      fi
      ;;
    age|openssl) printf '%s\n' "${requested,,}" ;;
    *) lrz_die "Unsupported EHR_ENCRYPTION_BACKEND: $requested (use auto|age|openssl)" ;;
  esac
}

lrz_encrypt_dir_to_archive() {
  local src_dir="$1"
  local out_archive="$2"
  local backend
  backend="$(lrz_crypto_backend)"

  [[ -d "$src_dir" ]] || lrz_die "Directory to encrypt does not exist: $src_dir"
  [[ -n "$out_archive" ]] || lrz_die "Missing output archive path for encryption"

  lrz_require_cmd tar
  lrz_require_cmd zstd

  mkdir -p "$(dirname "$out_archive")"

  if [[ "$backend" == "age" ]]; then
    lrz_require_cmd age

    local -a recipient_args=()
    local line trimmed

    if [[ -n "${EHR_AGE_RECIPIENT:-}" ]]; then
      recipient_args+=("-r" "$EHR_AGE_RECIPIENT")
    fi

    if [[ -n "${EHR_AGE_RECIPIENTS_FILE:-}" ]]; then
      [[ -f "$EHR_AGE_RECIPIENTS_FILE" ]] || lrz_die "Recipient file not found: $EHR_AGE_RECIPIENTS_FILE"
      while IFS= read -r line; do
        trimmed="$(lrz_trim "${line%%#*}")"
        [[ -z "$trimmed" ]] && continue
        recipient_args+=("-r" "$trimmed")
      done <"$EHR_AGE_RECIPIENTS_FILE"
    fi

    [[ ${#recipient_args[@]} -gt 0 ]] || lrz_die "No AGE recipients configured. Set EHR_AGE_RECIPIENT or EHR_AGE_RECIPIENTS_FILE."

    tar -C "$src_dir" -cf - . \
      | zstd -T0 -q \
      | age "${recipient_args[@]}" -o "$out_archive"
    return 0
  fi

  lrz_require_cmd openssl
  [[ -n "${EHR_OPENSSL_PASSPHRASE_FILE:-}" ]] || lrz_die "Set EHR_OPENSSL_PASSPHRASE_FILE for openssl encryption."
  [[ -f "$EHR_OPENSSL_PASSPHRASE_FILE" ]] || lrz_die "OpenSSL passphrase file not found: $EHR_OPENSSL_PASSPHRASE_FILE"

  tar -C "$src_dir" -cf - . \
    | zstd -T0 -q \
    | openssl enc -aes-256-cbc -pbkdf2 -salt -pass "file:$EHR_OPENSSL_PASSPHRASE_FILE" -out "$out_archive"
}

lrz_decrypt_archive_to_dir() {
  local in_archive="$1"
  local out_dir="$2"
  local backend
  backend="$(lrz_crypto_backend)"

  [[ -f "$in_archive" ]] || lrz_die "Encrypted archive not found: $in_archive"
  lrz_require_cmd tar
  lrz_require_cmd zstd

  mkdir -p "$out_dir"

  if [[ "$backend" == "age" ]]; then
    lrz_require_cmd age
    [[ -n "${EHR_AGE_IDENTITY_FILE:-}" ]] || lrz_die "Set EHR_AGE_IDENTITY_FILE to decrypt AGE archives."
    [[ -f "$EHR_AGE_IDENTITY_FILE" ]] || lrz_die "AGE identity file not found: $EHR_AGE_IDENTITY_FILE"

    age --decrypt -i "$EHR_AGE_IDENTITY_FILE" "$in_archive" \
      | zstd -d -T0 -q \
      | tar -C "$out_dir" -xf -
    return 0
  fi

  lrz_require_cmd openssl
  [[ -n "${EHR_OPENSSL_PASSPHRASE_FILE:-}" ]] || lrz_die "Set EHR_OPENSSL_PASSPHRASE_FILE for openssl decryption."
  [[ -f "$EHR_OPENSSL_PASSPHRASE_FILE" ]] || lrz_die "OpenSSL passphrase file not found: $EHR_OPENSSL_PASSPHRASE_FILE"

  openssl enc -d -aes-256-cbc -pbkdf2 -pass "file:$EHR_OPENSSL_PASSPHRASE_FILE" -in "$in_archive" \
    | zstd -d -T0 -q \
    | tar -C "$out_dir" -xf -
}

lrz_stage_input_dir() {
  local plain_dir="$1"
  local encrypted_archive="$2"
  local runtime_root="$3"
  local label="$4"

  if [[ -n "$encrypted_archive" ]]; then
    local staged_dir="${runtime_root%/}/${label}"
    mkdir -p "$staged_dir"
    lrz_log "Decrypting $encrypted_archive into $staged_dir"
    lrz_decrypt_archive_to_dir "$encrypted_archive" "$staged_dir"
    printf '%s\n' "$staged_dir"
    return 0
  fi

  [[ -d "$plain_dir" ]] || lrz_die "Input directory not found: $plain_dir"
  printf '%s\n' "$plain_dir"
}

lrz_sync_dir() {
  local src_dir="$1"
  local dst_dir="$2"

  [[ -d "$src_dir" ]] || lrz_die "Source directory missing for sync: $src_dir"
  [[ -n "$dst_dir" && "$dst_dir" != "/" ]] || lrz_die "Refusing unsafe sync destination: $dst_dir"

  mkdir -p "$dst_dir"
  if lrz_have_cmd rsync; then
    rsync -a --delete "$src_dir"/ "$dst_dir"/
  else
    find "$dst_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    cp -a "$src_dir"/. "$dst_dir"/
  fi
}

lrz_sha256_file() {
  local target="$1"
  [[ -f "$target" ]] || return 1
  if lrz_have_cmd sha256sum; then
    sha256sum "$target"
    return 0
  fi
  if lrz_have_cmd shasum; then
    shasum -a 256 "$target"
    return 0
  fi
  return 1
}
