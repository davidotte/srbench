#!/bin/bash -x
#
# SLURM template for building a method's Apptainer image from alg.def. Set the partition below
# and adapt CPU/memory to your cluster. Requires unprivileged (--fakeroot) apptainer builds.
#
# Usage (from the srbench directory):
#   sbatch scripts/build_sif.sh rasr
#   sbatch scripts/build_sif.sh snip images/snip.sif
#
# Environment variables:
#   APPTAINER_TMPDIR - build scratch directory (default: <srbench>/tmp)
#
#SBATCH -p <cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=sif_build
#SBATCH -D .
#SBATCH --time=04:00:00
#SBATCH --output=output/sif-build-out.%j
#SBATCH --error=error/sif-build-err.%j

set -euo pipefail

APPTAINER_CMD=$(command -v apptainer 2>/dev/null || command -v singularity 2>/dev/null) \
    || { echo "ERROR: Neither apptainer nor singularity found in PATH"; exit 1; }

ALGORITHM="${1:?Usage: sbatch scripts/build_sif.sh <algorithm> [out.sif]}"
SRBENCH_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set, run via sbatch from the srbench directory}"
DEF_FILE="${SRBENCH_ROOT}/alg.def"
OUT_SIF="$(realpath -m "${2:-${SRBENCH_ROOT}/images/${ALGORITHM}.sif}")"

if [ ! -d "${SRBENCH_ROOT}/algorithms/${ALGORITHM}" ]; then
    echo "ERROR: algorithms/${ALGORITHM} not found in ${SRBENCH_ROOT}"
    exit 1
fi

echo "INFO: building ${ALGORITHM} from ${DEF_FILE} -> ${OUT_SIF}"
mkdir -p "$(dirname "${OUT_SIF}")"

# mksquashfs can segfault when the node's /tmp is too small for the image
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SRBENCH_ROOT}/tmp}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
mkdir -p "${APPTAINER_TMPDIR}"

# alg.def resolves its paths relative to srbench's parent directory
cd "$(dirname "${SRBENCH_ROOT}")"

srun -n 1 -c "${SLURM_CPUS_PER_TASK}" \
    $APPTAINER_CMD build --fakeroot --build-arg "ALGORITHM=${ALGORITHM}" "${OUT_SIF}" "${DEF_FILE}"

echo "INFO: done -> ${OUT_SIF}"
