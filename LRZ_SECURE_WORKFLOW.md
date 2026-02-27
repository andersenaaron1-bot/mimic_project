# LRZ Secure Container + ETL/Training Workflow

This repo now includes a secure-by-default Slurm workflow for LRZ AI Systems with:

- Pyxis/Enroot-first container execution.
- Optional encryption at rest for ETL/training inputs and outputs.
- Local scratch staging (`$SLURM_TMPDIR`) when encryption is enabled.
- Reusable `.sqsh` images to avoid repeated `docker://` pulls.

## 1. Compliance Gate (Art. 9 GDPR data)

Before processing MIMIC data, obtain the required LRZ special agreement for Art. 9 GDPR workloads and use the secured infrastructure path approved by LRZ.

## 2. Build Container Images for Pyxis

Prefer local `.sqsh` images on DSS:

```bash
scripts/build_enroot_image.sh --target train --output /dss/<proj>/containers/ehr-train.sqsh
scripts/build_enroot_image.sh --target etl   --output /dss/<proj>/containers/meds-etl.sqsh
scripts/build_enroot_image.sh --target pipeline-cpu --output /dss/<proj>/containers/ehr-pipeline-cpu.sqsh
scripts/build_enroot_image.sh --target train-overlay --output /dss/<proj>/containers/ehr-train-overlay.sqsh
```

Each build also emits `<image>.sha256`.

## 3. AGE Key Setup (for encrypted archives)

Create a keypair once:

```bash
age-keygen -o ~/.config/ehr/age-key.txt
grep '^# public key:' ~/.config/ehr/age-key.txt
```

Set runtime env:

```bash
export EHR_AGE_IDENTITY_FILE=~/.config/ehr/age-key.txt
export EHR_AGE_RECIPIENT='age1...your-public-recipient...'
```

You can also use `EHR_AGE_RECIPIENTS_FILE` with one recipient per line.

If `age` is unavailable on the host, scripts fall back to OpenSSL automatically. For that path set:

```bash
export EHR_ENCRYPTION_BACKEND=openssl
export EHR_OPENSSL_PASSPHRASE_FILE=~/.config/ehr/openssl-passphrase.txt
```

## 4. ETL Job Modes

Script: `scripts/sbatch_meds_etl.sh`

Plain mode (fastest, no extra encryption step):

```bash
RAW_MIMIC_DIR=/dss/<proj>/mimic-iv-2.2 \
OUT_DIR=/dss/<proj>/meds_mimiciv_2.2 \
sbatch scripts/sbatch_meds_etl.sh
```

Encrypted ingress/egress:

```bash
RAW_MIMIC_ENC_ARCHIVE=/dss/<proj>/secure/raw_mimic.tar.zst.age \
OUT_ENC_ARCHIVE=/dss/<proj>/secure/meds_out.tar.zst.age \
KEEP_PLAINTEXT_OUT=0 \
sbatch scripts/sbatch_meds_etl.sh
```

## 5. Training Job Modes

Script: `scripts/sbatch_single.sh`

Plain mode:

```bash
DATA_DIR=/dss/<proj>/datasets \
OUT_DIR=/dss/<proj>/outputs/run_001 \
sbatch scripts/sbatch_single.sh
```

Encrypted dataset + encrypted checkpoints:

```bash
DATA_ENC_ARCHIVE=/dss/<proj>/secure/datasets.tar.zst.age \
OUT_ENC_ARCHIVE=/dss/<proj>/secure/train_out_001.tar.zst.age \
KEEP_PLAINTEXT_OUT=0 \
sbatch scripts/sbatch_single.sh
```

## 6. Important Runtime Variables

- `STAGE_TO_LOCAL=auto|true|false`: `auto` stages when encrypted input archives are used.
- `KEEP_PLAINTEXT_OUT=0|1`: keep or remove plaintext output directory.
- `IMAGE_LOCAL`: preferred local `.sqsh`.
- `IMAGE_REMOTE`: remote fallback image URI.
- `REPO_READONLY=0|1` (training/dev): mount repo read-only when possible.
- `EHR_ENCRYPTION_BACKEND=auto|age|openssl`: select crypto backend (`auto` prefers `age`).

## 7. Notes on Performance and Flexibility

- Plain mode preserves original direct-I/O behavior.
- Encryption mode only adds decrypt/encrypt passes at boundaries.
- `.sqsh` reuse avoids repeated image pulls and improves startup latency.
- The same scripts support both encrypted and plaintext workflows via env vars.

## 8. CPU Pipeline Image (meds_reader + timeline/tokenization prep)

Use `containers/Dockerfile.pipeline_cpu` when you want a stable post-ETL image for:
- `meds_reader_convert`
- timeline build (`src/ehr_hier/data/compile_dataset.py`)
- tokenization/CVAE prep scripts in `scripts/`

Preflight example:

```bash
IMAGE=/dss/<proj>/containers/ehr-pipeline-cpu.sqsh
RUN_DIR=/dss/<proj>/runs/pipeline_check_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

srun -A default -p lrz-cpu --qos=cpu --cpus-per-task=1 --mem=4G --time=00:10:00 \
  --container-image="$IMAGE" \
  --container-mounts="$PWD:/workspace/ehr-hier,$RUN_DIR:/run" \
  bash -lc 'set -euo pipefail; cd /workspace/ehr-hier; command -v meds_reader_convert; python -c "import meds_reader, torch, pyarrow, polars; print(\"pipeline-cpu preflight OK\")"'
```

## 9. Remote GPU Image Build (GitHub Actions)

To avoid local CUDA-heavy Docker builds, use the workflow:
- `.github/workflows/build-train-overlay.yml`
- Dockerfile: `containers/Dockerfile.train_overlay`
- Deps: `requirements.train_overlay.txt`

Trigger from GitHub Actions (`workflow_dispatch`) with:
- `tag` (for example `train-overlay-latest`)
- `ngc_tag` (for example `24.10-py3`)

The workflow pushes:
- `ghcr.io/<owner>/ehr-train:<tag>`
- optional `ghcr.io/<owner>/ehr-train:sha-<commit>`

Use directly on LRZ:

```bash
IMAGE_REMOTE='docker://ghcr.io#<owner>/ehr-train:train-overlay-latest'
```

Or import to local `.sqsh` (when `enroot` is available):

```bash
scripts/build_enroot_image.sh --source-image "$IMAGE_REMOTE" --output /dss/<proj>/containers/ehr-train-overlay.sqsh
```

This keeps the image stable while your code remains flexible via repo mounts:

```bash
--container-mounts "$PWD:/workspace/ehr-hier,..."
```
