#!/usr/bin/env bash
# Reproducible no-Docker Ray Serve benchmark runner.
# Designed for GPU VMs where Docker is unavailable or undesirable, for example Runpod.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_URL="${REPO_URL:-https://github.com/adapstory/gliner-guard-serve.git}"
BRANCH="${BRANCH:-feat/ray-serve-uni-bi}"
WORKTREE_DIR="${WORKTREE_DIR:-/root/gliner-guard-serve-pr-local}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

DURATION="${DURATION:-15m}"
USERS="${USERS:-100}"
SPAWN_RATE="${SPAWN_RATE:-1}"
WARMUP_REQS="${WARMUP_REQS:-50}"
DATASET="${DATASET:-prompts}"
REPEATS="${REPEATS:-1}"
CURATE_RUN="${CURATE_RUN:-${REPEATS}}"
BENCH_MATRIX_MODE="${BENCH_MATRIX_MODE:-default}"
BENCH_PROTOCOLS="${BENCH_PROTOCOLS:-rest grpc}"
SWEEP_CONFIGS="${SWEEP_CONFIGS:-B1:8:0.01 B2:8:0.05 B3:16:0.01 B4:16:0.05 B5:32:0.05 B6:32:0.10 B7:64:0.05 B8:64:0.10}"
UPDATE_README="${UPDATE_README:-}"

MODELS="${MODELS:-uni bi}"
DTYPES="${DTYPES:-bf16 fp16}"

REST_DYNBATCH_ID="${REST_DYNBATCH_ID:-B4}"
REST_DYNBATCH_BATCH_SIZE="${REST_DYNBATCH_BATCH_SIZE:-16}"
REST_DYNBATCH_TIMEOUT="${REST_DYNBATCH_TIMEOUT:-0.05}"
GRPC_DYNBATCH_ID="${GRPC_DYNBATCH_ID:-B16}"
GRPC_DYNBATCH_BATCH_SIZE="${GRPC_DYNBATCH_BATCH_SIZE:-16}"
GRPC_DYNBATCH_TIMEOUT="${GRPC_DYNBATCH_TIMEOUT:-0.05}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
EXPECT_GPU="${EXPECT_GPU:-1}"

RAW_RESULTS_DIR_REL="${RAW_RESULTS_DIR_REL:-}"
LOG_DIR_REL="${LOG_DIR_REL:-}"
SUMMARY_CSV_REL="${SUMMARY_CSV_REL:-}"

case "${BENCH_MATRIX_MODE}" in
    default|sweep) ;;
    *)
        echo "Unsupported BENCH_MATRIX_MODE=${BENCH_MATRIX_MODE}. Use default or sweep." >&2
        exit 1
        ;;
esac

if [ -z "${RAW_RESULTS_DIR_REL}" ]; then
    if [ "${BENCH_MATRIX_MODE}" = "sweep" ]; then
        RAW_RESULTS_DIR_REL="artifacts/raw-results/sweep"
    else
        RAW_RESULTS_DIR_REL="artifacts/raw-results"
    fi
fi

if [ -z "${LOG_DIR_REL}" ]; then
    if [ "${BENCH_MATRIX_MODE}" = "sweep" ]; then
        LOG_DIR_REL="artifacts/logs/sweep"
    else
        LOG_DIR_REL="artifacts/logs"
    fi
fi

if [ -z "${SUMMARY_CSV_REL}" ]; then
    if [ "${BENCH_MATRIX_MODE}" = "sweep" ]; then
        SUMMARY_CSV_REL="artifacts/raw-results/sweep-summary.csv"
    else
        SUMMARY_CSV_REL="artifacts/raw-results/benchmark-summary.csv"
    fi
fi

if [ -z "${UPDATE_README}" ]; then
    if [ "${BENCH_MATRIX_MODE}" = "sweep" ]; then
        UPDATE_README="0"
    else
        UPDATE_README="1"
    fi
fi

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${RAY_TMPDIR}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

normalize_dtype_tag() {
    case "${1,,}" in
        bf16|bfloat16) echo "bf16" ;;
        fp16|float16) echo "fp16" ;;
        *)
            echo "Unsupported TORCH_DTYPE=$1. Use bf16 or fp16." >&2
            exit 1
            ;;
    esac
}

model_id_for() {
    case "$1" in
        uni) echo "hivetrace/gliner-guard-uniencoder" ;;
        bi) echo "hivetrace/gliner-guard-biencoder" ;;
        *)
            echo "Unsupported model short name: $1" >&2
            exit 1
            ;;
    esac
}

curated_model_name_for() {
    case "$1" in
        uni) echo "gliner-guard-uni" ;;
        bi) echo "gliner-guard-bi" ;;
        *)
            echo "Unsupported model short name: $1" >&2
            exit 1
            ;;
    esac
}

duration_to_seconds() {
    case "$1" in
        *h) echo $(( ${1%h} * 3600 )) ;;
        *m) echo $(( ${1%m} * 60 )) ;;
        *s) echo "${1%s}" ;;
        *) echo "$1" ;;
    esac
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

prepare_worktree() {
    if [ ! -d "${WORKTREE_DIR}/.git" ]; then
        log "Cloning ${REPO_URL} into ${WORKTREE_DIR}"
        git clone "${REPO_URL}" "${WORKTREE_DIR}"
    fi

    if [ -n "$(git -C "${WORKTREE_DIR}" status --porcelain --untracked-files=no)" ]; then
        echo "Dirty worktree detected in ${WORKTREE_DIR}. Refusing to continue." >&2
        exit 1
    fi

    git -C "${WORKTREE_DIR}" fetch origin
    if git -C "${WORKTREE_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
        git -C "${WORKTREE_DIR}" checkout "${BRANCH}"
    else
        git -C "${WORKTREE_DIR}" checkout -b "${BRANCH}" "origin/${BRANCH}"
    fi
    git -C "${WORKTREE_DIR}" pull --ff-only origin "${BRANCH}"
}

sync_envs() {
    log "Syncing ray-serve environment"
    (
        cd "${WORKTREE_DIR}/ray-serve"
        uv python install "${PYTHON_VERSION}"
        uv sync --python "${PYTHON_VERSION}"
        uv run python -m grpc_tools.protoc \
            -I=proto \
            --python_out=. \
            --grpc_python_out=. \
            proto/gliner_guard.proto
    )

    log "Syncing test-script environment"
    (
        cd "${WORKTREE_DIR}/test-script"
        uv sync --python "${PYTHON_VERSION}"
    )
}

raw_results_dir() {
    printf '%s/%s' "${WORKTREE_DIR}" "${RAW_RESULTS_DIR_REL}"
}

log_dir() {
    printf '%s/%s' "${WORKTREE_DIR}" "${LOG_DIR_REL}"
}

summary_csv_path() {
    printf '%s/%s' "${WORKTREE_DIR}" "${SUMMARY_CSV_REL}"
}

extract_aggregated_stats() {
    local stats_file="$1"
    python3 - "$stats_file" <<'PY'
import csv
import sys

path = sys.argv[1]

with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))

row = next((candidate for candidate in rows if candidate.get("Name") == "Aggregated"), None)
if row is None:
    raise SystemExit(1)

print(
    "|".join(
        [
            row.get("Request Count", ""),
            row.get("Failure Count", ""),
            row.get("Requests/s", ""),
            row.get("50%", ""),
            row.get("95%", ""),
            row.get("99%", ""),
        ]
    )
)
PY
}

ensure_summary_header() {
    local summary_file
    summary_file="$(summary_csv_path)"
    mkdir -p "$(dirname "${summary_file}")"

    if [ ! -f "${summary_file}" ]; then
        printf '%s\n' \
            'protocol,model,dtype,config_id,batch_size,batch_timeout,run,dataset,users,spawn_rate,duration,request_count,failure_count,rps,p50_ms,p95_ms,p99_ms,stats_csv,html_report,gpu_csv,log_file' \
            > "${summary_file}"
    fi
}

append_summary_row() {
    local protocol="$1"
    local model_short="$2"
    local dtype_tag="$3"
    local config_id="$4"
    local batch_size="$5"
    local batch_timeout="$6"
    local run_num="$7"
    local prefix="$8"
    local stats_file
    local html_file
    local gpu_file
    local log_file
    local summary_file
    local stats_line
    local request_count
    local failure_count
    local rps
    local p50
    local p95
    local p99

    stats_file="$(raw_results_dir)/${prefix}_stats.csv"
    html_file="$(raw_results_dir)/${prefix}.html"
    gpu_file="$(raw_results_dir)/gpu-${prefix}.csv"
    log_file="$(log_dir)/${prefix}.log"

    if [ ! -f "${stats_file}" ]; then
        log "Missing stats file for summary append: ${stats_file}"
        return 1
    fi

    stats_line="$(extract_aggregated_stats "${stats_file}")" || {
        log "Failed to parse aggregated stats from ${stats_file}"
        return 1
    }
    IFS='|' read -r request_count failure_count rps p50 p95 p99 <<< "${stats_line}"

    ensure_summary_header
    summary_file="$(summary_csv_path)"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${protocol}" \
        "${model_short}" \
        "${dtype_tag}" \
        "${config_id}" \
        "${batch_size}" \
        "${batch_timeout}" \
        "${run_num}" \
        "${DATASET}" \
        "${USERS}" \
        "${SPAWN_RATE}" \
        "${DURATION}" \
        "${request_count}" \
        "${failure_count}" \
        "${rps}" \
        "${p50}" \
        "${p95}" \
        "${p99}" \
        "${stats_file}" \
        "${html_file}" \
        "${gpu_file}" \
        "${log_file}" \
        >> "${summary_file}"
}

stop_ray_stack() {
    local pid_file="${1:-}"
    if [ -n "${pid_file}" ] && [ -f "${pid_file}" ]; then
        local pid
        pid="$(cat "${pid_file}")"
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
        rm -f "${pid_file}"
    fi

    if [ -x "${WORKTREE_DIR}/ray-serve/.venv/bin/ray" ]; then
        "${WORKTREE_DIR}/ray-serve/.venv/bin/ray" stop --force >/dev/null 2>&1 || true
    fi
    pkill -f "${WORKTREE_DIR}/ray-serve/serve_app.py" 2>/dev/null || true
    pkill -f "${WORKTREE_DIR}/ray-serve/serve_app_grpc.py" 2>/dev/null || true
}

wait_ready_rest() {
    local attempts="${1:-180}"
    for _ in $(seq 1 "${attempts}"); do
        if curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d '{"text":"healthcheck"}'; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_ready_grpc() {
    local attempts="${1:-120}"
    for _ in $(seq 1 "${attempts}"); do
        if (
            cd "${WORKTREE_DIR}/test-script"
            GLINER_HOST=localhost:9000 .venv/bin/python - <<'PY'
import grpc
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "..", "ray-serve"))
import gliner_guard_pb2
import gliner_guard_pb2_grpc

channel = grpc.insecure_channel("localhost:9000")
stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(channel)
stub.Predict(gliner_guard_pb2.PredictRequest(text="ping"), timeout=5)
print("ok")
PY
        ) 2>/dev/null | grep -q '^ok$'; then
            return 0
        fi
        sleep 2
    done
    return 1
}

warmup_rest() {
    for i in $(seq 1 "${WARMUP_REQS}"); do
        curl -sf -o /dev/null http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"warmup request ${i}\"}" &
    done
    wait
}

warmup_grpc() {
    (
        cd "${WORKTREE_DIR}/test-script"
        GLINER_HOST=localhost:9000 .venv/bin/python - <<PY
import grpc
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "..", "ray-serve"))
import gliner_guard_pb2
import gliner_guard_pb2_grpc

channel = grpc.insecure_channel("localhost:9000")
stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(channel)
for i in range(${WARMUP_REQS}):
    stub.Predict(gliner_guard_pb2.PredictRequest(text=f"warmup {i}"), timeout=30)
PY
    )
}

gpu_memory_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk 'NR==1 {print int($1)}'
}

assert_gpu_in_use() {
    if [ "${EXPECT_GPU}" != "1" ]; then
        return 0
    fi

    local attempts="${1:-15}"
    local used_mb

    for _ in $(seq 1 "${attempts}"); do
        used_mb="$(gpu_memory_used_mb || echo 0)"
        if [ "${used_mb}" -gt 0 ]; then
            log "GPU check passed: ${used_mb} MiB in use"
            return 0
        fi
        sleep 2
    done

    log "GPU check failed: nvidia-smi still reports 0 MiB in use"
    return 1
}

start_server() {
    local app_script="$1"
    local model_id="$2"
    local dtype_tag="$3"
    local batch_size="$4"
    local batch_timeout="$5"
    local log_path="$6"
    local pid_file="$7"

    mkdir -p "$(dirname "${log_path}")"
    stop_ray_stack "${pid_file}"

    (
        cd "${WORKTREE_DIR}/ray-serve"
        export MODEL_ID="${model_id}"
        export TORCH_DTYPE="${dtype_tag}"
        export MAX_BATCH_SIZE="${batch_size}"
        export BATCH_WAIT_TIMEOUT="${batch_timeout}"
        export UV_CACHE_DIR="${UV_CACHE_DIR}"
        export HF_HOME="${HF_HOME}"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export RAY_TMPDIR="${RAY_TMPDIR}"
        export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE}"
        nohup .venv/bin/python "${app_script}" > "${log_path}" 2>&1 &
        echo $! > "${pid_file}"
    )
}

summarize_stats() {
    local stats_file="$1"
    local stats_line
    local request_count
    local failure_count
    local rps
    local p50
    local p95
    local p99

    if [ ! -f "${stats_file}" ]; then
        log "Missing stats file: ${stats_file}"
        return 1
    fi

    stats_line="$(extract_aggregated_stats "${stats_file}")" || {
        log "Failed to parse aggregated stats from ${stats_file}"
        return 1
    }
    IFS='|' read -r request_count failure_count rps p50 p95 p99 <<< "${stats_line}"
    printf 'RPS=%.1f P50=%sms P95=%sms P99=%sms Failures=%s\n' \
        "${rps}" "${p50}" "${p95}" "${p99}" "${failure_count}"
}

run_rest_bench() {
    local prefix="$1"
    local model_short="$2"
    local model_id="$3"
    local dtype_tag="$4"
    local batch_size="$5"
    local batch_timeout="$6"
    local config_id="$7"
    local run_num="$8"
    local app_script="${9:-serve_app.py}"
    local duration_secs
    local raw_dir
    local logs_dir
    local log_path
    local pid_file
    local gpu_pid

    raw_dir="$(raw_results_dir)"
    logs_dir="$(log_dir)"
    mkdir -p "${raw_dir}" "${logs_dir}"
    duration_secs="$(duration_to_seconds "${DURATION}")"
    log_path="${logs_dir}/${prefix}.log"
    pid_file="${logs_dir}/${prefix}.pid"

    log "Starting REST server for ${prefix}"
    start_server "${app_script}" "${model_id}" "${dtype_tag}" "${batch_size}" "${batch_timeout}" "${log_path}" "${pid_file}"
    if ! wait_ready_rest; then
        log "REST server failed to become ready, see ${log_path}"
        stop_ray_stack "${pid_file}"
        return 1
    fi

    log "Warmup REST ${prefix}"
    warmup_rest
    assert_gpu_in_use || {
        stop_ray_stack "${pid_file}"
        return 1
    }

    bash "${WORKTREE_DIR}/scripts/collect_gpu_metrics.sh" \
        "${raw_dir}/gpu-${prefix}.csv" \
        "${duration_secs}" &
    gpu_pid=$!

    (
        cd "${WORKTREE_DIR}/test-script"
        DATASET="${DATASET}" GLINER_HOST=http://localhost:8000 \
            .venv/bin/locust -f test-gliner.py \
            --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
            --csv="${raw_dir}/${prefix}" \
            --html="${raw_dir}/${prefix}.html"
    )

    wait "${gpu_pid}" 2>/dev/null || true
    stop_ray_stack "${pid_file}"
    summarize_stats "${raw_dir}/${prefix}_stats.csv"
    append_summary_row "rest" "${model_short}" "${dtype_tag}" "${config_id}" "${batch_size}" "${batch_timeout}" "${run_num}" "${prefix}"
}

run_grpc_bench() {
    local prefix="$1"
    local model_short="$2"
    local model_id="$3"
    local dtype_tag="$4"
    local batch_size="$5"
    local batch_timeout="$6"
    local config_id="$7"
    local run_num="$8"
    local duration_secs
    local raw_dir
    local logs_dir
    local log_path
    local pid_file
    local gpu_pid

    raw_dir="$(raw_results_dir)"
    logs_dir="$(log_dir)"
    mkdir -p "${raw_dir}" "${logs_dir}"
    duration_secs="$(duration_to_seconds "${DURATION}")"
    log_path="${logs_dir}/${prefix}.log"
    pid_file="${logs_dir}/${prefix}.pid"

    log "Starting gRPC server for ${prefix}"
    start_server "serve_app_grpc.py" "${model_id}" "${dtype_tag}" "${batch_size}" "${batch_timeout}" "${log_path}" "${pid_file}"
    if ! wait_ready_rest; then
        log "REST health check failed for gRPC app, see ${log_path}"
        stop_ray_stack "${pid_file}"
        return 1
    fi
    if ! wait_ready_grpc; then
        log "gRPC endpoint failed to become ready, see ${log_path}"
        stop_ray_stack "${pid_file}"
        return 1
    fi

    log "Warmup gRPC ${prefix}"
    warmup_grpc
    assert_gpu_in_use || {
        stop_ray_stack "${pid_file}"
        return 1
    }

    bash "${WORKTREE_DIR}/scripts/collect_gpu_metrics.sh" \
        "${raw_dir}/gpu-${prefix}.csv" \
        "${duration_secs}" &
    gpu_pid=$!

    (
        cd "${WORKTREE_DIR}/test-script"
        DATASET="${DATASET}" GLINER_HOST=localhost:9000 \
            .venv/bin/locust -f test-gliner-grpc.py \
            --headless -u "${USERS}" -r "${SPAWN_RATE}" --run-time "${DURATION}" \
            --csv="${raw_dir}/${prefix}" \
            --html="${raw_dir}/${prefix}.html"
    )

    wait "${gpu_pid}" 2>/dev/null || true
    stop_ray_stack "${pid_file}"
    summarize_stats "${raw_dir}/${prefix}_stats.csv"
    append_summary_row "grpc" "${model_short}" "${dtype_tag}" "${config_id}" "${batch_size}" "${batch_timeout}" "${run_num}" "${prefix}"
}

curate_selected_run() {
    local model_short="$1"
    local dtype_tag="$2"
    local curated_model

    curated_model="$(curated_model_name_for "${model_short}")"
    (
        cd "${WORKTREE_DIR}"
        python3 scripts/curate_ray_results.py \
            --source-dir "${RAW_RESULTS_DIR_REL}" \
            --model "${curated_model}" \
            --dtype "${dtype_tag}" \
            --rest-nobatch "ray-rest-${dtype_tag}-nobatch-${model_short}-${DATASET}-run${CURATE_RUN}" \
            --rest-dynbatch "ray-rest-${dtype_tag}-${REST_DYNBATCH_ID}-${model_short}-${DATASET}-run${CURATE_RUN}" \
            --grpc-dynbatch "ray-grpc-${dtype_tag}-${GRPC_DYNBATCH_ID}-${model_short}-${DATASET}-run${CURATE_RUN}"
    )
}

update_readme_table() {
    if [ "${UPDATE_README}" != "1" ]; then
        return
    fi
    (
        cd "${WORKTREE_DIR}"
        make bench-readme
    )
}

main() {
    local model_short
    local dtype
    local dtype_tag
    local model_id
    local run
    local protocol
    local config_spec
    local config_id
    local batch_size
    local batch_timeout
    local -a model_list
    local -a dtype_list
    local -a protocol_list
    local -a sweep_config_list

    export PATH="$HOME/.local/bin:$PATH"
    export UV_CACHE_DIR HF_HOME RAY_TMPDIR PYTHONDONTWRITEBYTECODE

    ensure_uv
    prepare_worktree
    sync_envs

    read -r -a model_list <<< "${MODELS}"
    read -r -a dtype_list <<< "${DTYPES}"
    read -r -a protocol_list <<< "${BENCH_PROTOCOLS}"
    read -r -a sweep_config_list <<< "${SWEEP_CONFIGS}"

    mkdir -p "$(raw_results_dir)" "$(log_dir)"
    ensure_summary_header

    log "Starting no-Docker benchmark matrix"
    log "Matrix mode: ${BENCH_MATRIX_MODE}"
    log "Worktree: ${WORKTREE_DIR}"
    log "Models: ${MODELS}"
    log "DTypes: ${DTYPES}"
    log "Repeats: ${REPEATS}"
    log "Dataset: ${DATASET}"
    log "Summary CSV: $(summary_csv_path)"
    if [ "${BENCH_MATRIX_MODE}" = "sweep" ]; then
        log "Protocols: ${BENCH_PROTOCOLS}"
        log "Sweep configs: ${SWEEP_CONFIGS}"
    fi

    for model_short in "${model_list[@]}"; do
        model_id="$(model_id_for "${model_short}")"
        for dtype in "${dtype_list[@]}"; do
            dtype_tag="$(normalize_dtype_tag "${dtype}")"
            if [ "${BENCH_MATRIX_MODE}" = "sweep" ]; then
                for config_spec in "${sweep_config_list[@]}"; do
                    IFS=':' read -r config_id batch_size batch_timeout <<< "${config_spec}"
                    if [ -z "${config_id}" ] || [ -z "${batch_size}" ] || [ -z "${batch_timeout}" ]; then
                        echo "Malformed SWEEP_CONFIGS entry: ${config_spec}" >&2
                        exit 1
                    fi

                    for run in $(seq 1 "${REPEATS}"); do
                        for protocol in "${protocol_list[@]}"; do
                            case "${protocol}" in
                                rest)
                                    run_rest_bench \
                                        "ray-rest-${dtype_tag}-${config_id}-${model_short}-${DATASET}-run${run}" \
                                        "${model_short}" "${model_id}" "${dtype_tag}" "${batch_size}" "${batch_timeout}" "${config_id}" "${run}"
                                    ;;
                                grpc)
                                    run_grpc_bench \
                                        "ray-grpc-${dtype_tag}-${config_id}-${model_short}-${DATASET}-run${run}" \
                                        "${model_short}" "${model_id}" "${dtype_tag}" "${batch_size}" "${batch_timeout}" "${config_id}" "${run}"
                                    ;;
                                *)
                                    echo "Unsupported BENCH_PROTOCOLS entry: ${protocol}" >&2
                                    exit 1
                                    ;;
                            esac
                        done
                    done
                done
            else
                for run in $(seq 1 "${REPEATS}"); do
                    run_rest_bench \
                        "ray-rest-${dtype_tag}-nobatch-${model_short}-${DATASET}-run${run}" \
                        "${model_short}" "${model_id}" "${dtype_tag}" 0 "${REST_DYNBATCH_TIMEOUT}" "nobatch" "${run}"

                    run_rest_bench \
                        "ray-rest-${dtype_tag}-${REST_DYNBATCH_ID}-${model_short}-${DATASET}-run${run}" \
                        "${model_short}" "${model_id}" "${dtype_tag}" "${REST_DYNBATCH_BATCH_SIZE}" "${REST_DYNBATCH_TIMEOUT}" "${REST_DYNBATCH_ID}" "${run}"

                    run_grpc_bench \
                        "ray-grpc-${dtype_tag}-${GRPC_DYNBATCH_ID}-${model_short}-${DATASET}-run${run}" \
                        "${model_short}" "${model_id}" "${dtype_tag}" "${GRPC_DYNBATCH_BATCH_SIZE}" "${GRPC_DYNBATCH_TIMEOUT}" "${GRPC_DYNBATCH_ID}" "${run}"
                done

                curate_selected_run "${model_short}" "${dtype_tag}"
            fi
        done
    done

    update_readme_table
    stop_ray_stack

    log "Benchmarks complete."
    log "Raw artifacts: $(raw_results_dir)"
    log "Summary CSV: $(summary_csv_path)"
    if [ "${BENCH_MATRIX_MODE}" = "default" ]; then
        log "Curated results: ${WORKTREE_DIR}/results/ray-serve"
    fi
}

main "$@"
