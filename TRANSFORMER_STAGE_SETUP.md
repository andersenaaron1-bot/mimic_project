# Transformer Stage Setup (Branch + Image on LRZ)

This file defines a stable workflow for the transformer stage to avoid:
- remote/branch confusion (`origin` vs `dev`)
- repeated heavy image imports on LRZ
- drifting image tags between runs

## 1) Branch policy

Use one active branch for this stage and track `origin` only:
- base branch: `lrz-etl-bootstrap`
- transformer work branch: `transformer-v1`
- upstream/tracking remote: `origin`
- optional mirror push to `dev` only when needed for collaboration

### Local commands (Windows/Git Bash or Linux shell)

```bash
git fetch origin --prune
git fetch dev --prune
git checkout lrz-etl-bootstrap
git pull --ff-only origin lrz-etl-bootstrap
git checkout -B transformer-v1
git push -u origin transformer-v1
```

Optional mirror (do not set tracking to `dev`):

```bash
git push dev transformer-v1:transformer-v1
```

Sanity check:

```bash
git branch -vv
git rev-parse --abbrev-ref --symbolic-full-name @{u}
```

Expected upstream: `origin/transformer-v1`

## 2) Image policy (transformer training)

Use a pinned train overlay image and import it to a local `.sqsh` once.

Recommended remote image tag:
- `docker://ghcr.io#andersenaaron1-bot/ehr-train:transformer-v1`

Recommended local LRZ image path:
- `/dss/.../containers/ehr-train-transformer-v1.sqsh`

### 2.1 Build/push image to GHCR

Use GitHub Actions workflow:
- `.github/workflows/build-train-overlay.yml`
- inputs:
  - `tag=transformer-v1`
  - `ngc_tag=24.10-py3`
  - `push_sha_tag=true`

If `gh` CLI is configured:

```bash
gh workflow run build-train-overlay.yml -f tag=transformer-v1 -f ngc_tag=24.10-py3 -f push_sha_tag=true
```

### 2.2 Import once to LRZ local `.sqsh`

Run on LRZ host (not inside a pyxis container):

```bash
export REPO=$HOME/mimic_project
export DSS_HOST=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2
export IMAGE_REMOTE='docker://ghcr.io#andersenaaron1-bot/ehr-train:transformer-v1'
export IMAGE_LOCAL="$DSS_HOST/containers/ehr-train-transformer-v1.sqsh"
mkdir -p "$DSS_HOST/containers"
```

```bash
srun -p lrz-cpu --qos=cpu --cpus-per-task=4 --mem=16G --time=01:00:00 bash -lc "set -euo pipefail; cd \"$REPO\"; scripts/build_enroot_image.sh --source-image \"$IMAGE_REMOTE\" --output \"$IMAGE_LOCAL\""
```

Verify:

```bash
ls -lh "$IMAGE_LOCAL" "$IMAGE_LOCAL.sha256"
```

## 3) LRZ run env for transformer stage

Set these in each new LRZ shell:

```bash
export REPO=$HOME/mimic_project
export DSS_HOST=/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2
export DEPS_HOST=$DSS_HOST/containers/runtime_pydeps
export IMAGE_LOCAL=$DSS_HOST/containers/ehr-train-transformer-v1.sqsh
export IMAGE_REMOTE='docker://ghcr.io#andersenaaron1-bot/ehr-train:transformer-v1'
export IMAGE="${IMAGE_LOCAL}"
```

Fallback if local image is missing:

```bash
test -f "$IMAGE_LOCAL" || export IMAGE="$IMAGE_REMOTE"
echo "$IMAGE"
```

## 4) Pull branch on LRZ

```bash
cd "$REPO"
git fetch origin --prune
git checkout transformer-v1
git pull --ff-only origin transformer-v1
```

