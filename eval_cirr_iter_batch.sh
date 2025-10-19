#!/usr/bin/env bash

# 示例（单行，无可选参数）：
# bash ./eval_cirr_iter_batch.sh <experiment_dir> <iteration_dir> <output_file>
# 例：
# bash ./eval_cirr_iter_batch.sh ./experiments/IterativeCIRR_qwen2_5vl_7b_20251012_004205_copy_gruopsamplerfix_copy_triplet_loss_i0.8_t0.2_margin0.1 training_iter_1 ./results/qwen2_5vl/eval_results_iter1_bs9_maxp384_384_prompt_triplet_loss_i0.8_t0.2_margin0.1.json

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: eval_cirr_iter_batch.sh <experiment_dir> <iteration_dir> <output_file> [options] [-- <extra_eval_args>]

Required arguments:
  experiment_dir   Path to the experiment root, e.g. ./experiments/IterativeCIRR_qwen2_5vl_7b_20251012_004205_copy_use_only_new_sample_test
  iteration_dir    Iteration subdirectory name, e.g. training_iter_1
  output_file      JSON file where eval_cirr.py will append results

Options:
  --start STEP     First checkpoint step to evaluate (default: 500)
  --end STEP       Last checkpoint step to evaluate (default: 5000)
  --step STEP      Step interval between checkpoints (default: 500)
  -h, --help       Show this help message

Extra arguments after '--' are forwarded to eval_cirr.sh unchanged.
The script simply loops checkpoints and reuses the same output file.
EOF
}

if [[ $# -lt 3 ]]; then
    usage
    exit 1
fi

EXPERIMENT_DIR=$1
ITERATION_DIR=$2
OUTPUT_FILE=$3
shift 3

START_STEP=500
END_STEP=5000
STEP_SIZE=500
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)
            START_STEP=$2
            shift 2
            ;;
        --end)
            END_STEP=$2
            shift 2
            ;;
        --step)
            STEP_SIZE=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_cirr.sh"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    echo "Could not locate eval_cirr.sh at ${EVAL_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
    echo "Experiment directory does not exist: ${EXPERIMENT_DIR}" >&2
    exit 1
fi

ITERATION_PATH="${EXPERIMENT_DIR%/}/${ITERATION_DIR}"

if [[ ! -d "${ITERATION_PATH}" ]]; then
    echo "Iteration directory does not exist: ${ITERATION_PATH}" >&2
    exit 1
fi

printf "Running CIRR evaluation for checkpoints in %s/%s\n" "${EXPERIMENT_DIR}" "${ITERATION_DIR}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

for (( step=START_STEP; step<=END_STEP; step+=STEP_SIZE )); do
    CHECKPOINT_TAG=$(printf "checkpoint-%d" "${step}")
    CHECKPOINT_PATH="${ITERATION_PATH}/${CHECKPOINT_TAG}"

    if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
        printf "[WARN] Skip missing checkpoint: %s\n" "${CHECKPOINT_PATH}"
        continue
    fi

    printf "[INFO] Evaluating %s -> %s\n" "${CHECKPOINT_PATH}" "${OUTPUT_FILE}"

    CMD=(env SHLVL=0 bash "${EVAL_SCRIPT}" --model_path "${CHECKPOINT_PATH}" --output_file "${OUTPUT_FILE}")
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    "${CMD[@]}"
done

printf "Sequential evaluation complete. Results stored in %s\n" "${OUTPUT_FILE}"
