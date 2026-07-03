"""Submit the three Statepassing text pretraining experiments to Slurm."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
from pathlib import Path
import re
import shlex
import subprocess

from absl import app, flags

FLAGS = flags.FLAGS

_SOURCE_ROOT = Path("/fast/project/HFMI_SynergyUnit/p-doom_shared/salan")
_DATASET_ROOT = _SOURCE_ROOT / "datasets" / "fineweb_edu_dedup_30b_8kto32k"
_INDEX_ROOT = _SOURCE_ROOT / "pretrain_indexes" / "fineweb_edu_dedup_30b_8kto32k_4096_eos248046"
_SAVE_ROOT = _SOURCE_ROOT / "runs" / "statepassing_pretrain"
_LOG_DIR = _SOURCE_ROOT / "logs" / "statepassing_pretrain"
_JAX_CACHE_ROOT = _SOURCE_ROOT / "jax_cache" / "statepassing_pretrain"
_DEFAULT_PRETRAIN_HIDDEN_SIZE = 768
_WIKI_PROGRESS_PATH = Path(
    "/fast/home/salan.isaqzoi/gh_projects/salanobp/codex-setup/Wiki/documentation/"
    "statepassing_pretrain_experiment_progress.md"
)


class PretrainMode(enum.StrEnum):
    IID_BASELINE = "iid_baseline"
    STATEPASSING_NO_BPTT = "statepassing_no_bptt"
    STATEPASSING_BPTT = "statepassing_bptt"


flags.DEFINE_string("submit_run_id", None, "Run id used for jobs/checkpoints/logs.")
flags.DEFINE_string("submit_dataset_root", str(_DATASET_ROOT), "Doc-chain dataset root.")
flags.DEFINE_string("submit_source_root", str(_SOURCE_ROOT), "Shared source root for staging.")
flags.DEFINE_string("submit_index_root", str(_INDEX_ROOT), "Reusable pretraining index root.")
flags.DEFINE_string("submit_save_root", str(_SAVE_ROOT), "Checkpoint root.")
flags.DEFINE_string("submit_log_dir", str(_LOG_DIR), "Slurm log root.")
flags.DEFINE_string("submit_jax_cache_root", str(_JAX_CACHE_ROOT), "JAX compilation cache root.")
flags.DEFINE_string(
    "submit_wiki_path", str(_WIKI_PROGRESS_PATH), "Wiki progress file for monitor updates."
)
flags.DEFINE_string("submit_partition", "standard", "Slurm partition.")
flags.DEFINE_string("submit_qos", None, "Optional Slurm QoS.")
flags.DEFINE_string("submit_time", "24:00:00", "Training job time limit.")
flags.DEFINE_string("submit_index_time", "12:00:00", "Index-build job time limit.")
flags.DEFINE_integer("submit_nodes_per_run", 1, "Nodes per training run.")
flags.DEFINE_integer("submit_gpus_per_node", 8, "GPUs/tasks per node for each training run.")
flags.DEFINE_multi_string(
    "submit_mode_shape",
    [],
    "Optional per-mode shape as mode:gpus_per_node:batch_size:grad_accum_steps. "
    "Repeat exactly once per mode to use different GPU counts. Batch size and "
    "grad accumulation must still be identical across modes.",
)
flags.DEFINE_integer("submit_cpus_per_task", 12, "CPUs per training task.")
flags.DEFINE_bool(
    "submit_single_process_per_run",
    False,
    "Run one JAX process per Slurm job with all allocated GPUs visible locally.",
)
flags.DEFINE_integer("submit_index_cpus", 32, "CPUs for the index-build job.")
flags.DEFINE_integer("submit_seq_len", 2048, "Segment length for index building and training.")
flags.DEFINE_integer("submit_num_segments", 2, "Fixed chunks per statepassing window.")
flags.DEFINE_integer("submit_batch_size", 128, "Global number of submit_seq_len-token segments.")
flags.DEFINE_integer("submit_grad_accum_steps", 4, "Gradient accumulation steps.")
flags.DEFINE_integer("submit_max_tokens", 15_000_000_000, "Total token budget.")
flags.DEFINE_integer("submit_warmup_tokens", 150_000_000, "Warmup token budget.")
flags.DEFINE_float("submit_learning_rate", 3e-4, "Peak learning rate.")
flags.DEFINE_float("submit_weight_decay", 0.1, "AdamW weight decay.")
flags.DEFINE_float("submit_adam_beta1", 0.9, "AdamW beta1.")
flags.DEFINE_float("submit_adam_beta2", 0.95, "AdamW beta2.")
flags.DEFINE_float("submit_adam_eps", 1e-8, "AdamW epsilon.")
flags.DEFINE_float("submit_max_grad_norm", 1.0, "Gradient clipping norm.")
flags.DEFINE_integer("submit_seed", 0, "RNG seed shared by all modes.")
flags.DEFINE_integer("submit_save_every", 1000, "Checkpoint every N optimizer steps.")
flags.DEFINE_integer("submit_log_every", 10, "Log every N optimizer steps.")
flags.DEFINE_integer("submit_val_every", 500, "Validate every N optimizer steps.")
flags.DEFINE_integer("submit_val_steps", 10, "Validation batches.")
flags.DEFINE_integer("submit_bptt_chunks", None, "BPTT span in chunks for statepassing.")
flags.DEFINE_bool("submit_pass_gdn_state", True, "Pass GDN recurrent state between chunks.")
flags.DEFINE_integer("submit_gdn_layer_limit", None, "Only pass state for first N GDN layers.")
flags.DEFINE_bool("submit_pass_rope_positions", False, "Pass chunk-aware RoPE position ids.")
flags.DEFINE_bool("submit_pass_conv_state", False, "Pass 1D conv state between chunks.")
flags.DEFINE_integer("submit_records_per_shard", 100_000, "Index records per shard.")
flags.DEFINE_integer("submit_eos_check_records", 1000, "Records sampled for EOS sanity.")
flags.DEFINE_float("submit_min_eos_fraction", 0.95, "Required dominant EOS fraction.")
flags.DEFINE_bool("submit_build_indexes", True, "Submit an index-build job if indexes are absent.")
flags.DEFINE_bool("submit_overwrite_indexes", False, "Overwrite index directories during build.")
flags.DEFINE_bool(
    "submit_stage_to_scratch", True, "Stage dataset and indexes to node-local scratch."
)
flags.DEFINE_bool("submit_run_pallas_tests", True, "Run Pallas kernel tests inside GPU jobs first.")
flags.DEFINE_string("submit_wandb_entity", "pdoom", "W&B entity; empty disables this flag.")
flags.DEFINE_string("submit_wandb_project", "omegalax", "W&B project; empty disables W&B.")
flags.DEFINE_string("submit_wandb_group", None, "W&B group; defaults to run id.")
flags.DEFINE_string("submit_wandb_tags", "statepassing,pretrain", "Comma-separated W&B tags.")
flags.DEFINE_multi_string(
    "submit_wandb_resume_id",
    [],
    "Optional per-mode W&B resume id as mode:run_id. Repeat once per resumed mode.",
)
flags.DEFINE_string("submit_wandb_resume", None, "Optional W&B resume policy.")
flags.DEFINE_bool("submit_dry_run", False, "Write scripts and print sbatch commands only.")
flags.DEFINE_bool("submit_monitor", True, "Submit a lightweight monitor job.")
flags.DEFINE_bool(
    "submit_sequential_train_jobs",
    False,
    "If true, chain train jobs with afterok dependencies in mode order.",
)
flags.DEFINE_integer("submit_monitor_poll_seconds", 1200, "Monitor poll interval.")
flags.DEFINE_string("submit_monitor_time", "24:00:00", "Monitor job time limit.")


def _quote(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def _flag_value(name: str):
    try:
        return getattr(FLAGS, name)
    except flags.UnparsedFlagAccessError:
        return FLAGS[name].value


def _run_id() -> str:
    return FLAGS.submit_run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _all_modes() -> list[PretrainMode]:
    return [
        PretrainMode.IID_BASELINE,
        PretrainMode.STATEPASSING_NO_BPTT,
        PretrainMode.STATEPASSING_BPTT,
    ]


@dataclasses.dataclass(frozen=True)
class RunSpec:
    mode: PretrainMode
    nodes: int
    gpus_per_node: int
    batch_size: int
    grad_accum_steps: int

    @property
    def total_tasks(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def optimizer_segments(self) -> int:
        return self.batch_size * self.grad_accum_steps


def indexes_ready(index_root: str | Path) -> bool:
    root = Path(index_root).expanduser().resolve()
    for split in ("train", "val"):
        if not (root / split / "metadata.json").exists():
            return False
    return True


def any_index_metadata_exists(index_root: str | Path) -> bool:
    root = Path(index_root).expanduser().resolve()
    for split in ("train", "val"):
        if (root / split / "metadata.json").exists():
            return True
    return False


def validate_submit_shape(
    *,
    batch_size: int,
    nodes: int,
    gpus_per_node: int,
    num_segments: int = 2,
    single_process_per_run: bool = False,
) -> int:
    total_tasks = int(nodes) * int(gpus_per_node)
    if total_tasks <= 0:
        raise ValueError("Total training tasks must be positive.")
    if not single_process_per_run and _DEFAULT_PRETRAIN_HIDDEN_SIZE % total_tasks != 0:
        raise ValueError(
            f"fsdp_size={total_tasks} must divide hidden_size={_DEFAULT_PRETRAIN_HIDDEN_SIZE}."
        )
    divisor = int(num_segments) * total_tasks
    if batch_size % divisor != 0:
        raise ValueError(
            f"Statepassing batch_size={batch_size} must be divisible by "
            f"num_segments * total_tasks={divisor}."
        )
    return total_tasks


def parse_run_specs(
    raw_shapes: list[str],
    *,
    nodes_per_run: int,
    default_gpus_per_node: int,
    default_batch_size: int,
    default_grad_accum_steps: int,
    single_process_per_run: bool = False,
) -> list[RunSpec]:
    if not raw_shapes:
        specs = [
            RunSpec(
                mode=mode,
                nodes=nodes_per_run,
                gpus_per_node=default_gpus_per_node,
                batch_size=default_batch_size,
                grad_accum_steps=default_grad_accum_steps,
            )
            for mode in _all_modes()
        ]
    else:
        specs_by_mode: dict[PretrainMode, RunSpec] = {}
        for raw in raw_shapes:
            parts = raw.split(":")
            if len(parts) != 4:
                raise ValueError(
                    "--submit_mode_shape must be mode:gpus_per_node:batch_size:grad_accum_steps"
                )
            mode = PretrainMode(parts[0])
            if mode in specs_by_mode:
                raise ValueError(f"Duplicate shape for mode {mode.value}")
            specs_by_mode[mode] = RunSpec(
                mode=mode,
                nodes=nodes_per_run,
                gpus_per_node=int(parts[1]),
                batch_size=int(parts[2]),
                grad_accum_steps=int(parts[3]),
            )

        missing = [mode.value for mode in _all_modes() if mode not in specs_by_mode]
        if missing:
            raise ValueError(f"Missing --submit_mode_shape entries for modes: {missing}")

        specs = [specs_by_mode[mode] for mode in _all_modes()]
    batch_sizes = {spec.batch_size for spec in specs}
    grad_accum_steps = {spec.grad_accum_steps for spec in specs}
    if len(batch_sizes) != 1 or len(grad_accum_steps) != 1:
        raise ValueError(
            "All modes must use the same batch_size and grad_accum_steps; got "
            f"batch_sizes={sorted(batch_sizes)} "
            f"grad_accum_steps={sorted(grad_accum_steps)}"
        )
    for spec in specs:
        validate_submit_shape(
            batch_size=spec.batch_size,
            nodes=spec.nodes,
            gpus_per_node=spec.gpus_per_node,
            num_segments=_flag_value("submit_num_segments"),
            single_process_per_run=single_process_per_run,
        )
    return specs


def parse_wandb_resume_ids(raw_ids: list[str]) -> dict[PretrainMode, str]:
    ids_by_mode: dict[PretrainMode, str] = {}
    for raw in raw_ids:
        parts = raw.split(":", 1)
        if len(parts) != 2:
            raise ValueError("--submit_wandb_resume_id must be mode:run_id")
        mode = PretrainMode(parts[0])
        if mode in ids_by_mode:
            raise ValueError(f"Duplicate W&B resume id for mode {mode.value}")
        ids_by_mode[mode] = parts[1]
    return ids_by_mode


def _path_under(path: Path, root: Path) -> str:
    return str(path.expanduser().resolve().relative_to(root.expanduser().resolve()))


def _sbatch_header(
    *,
    job_name: str,
    partition: str,
    qos: str | None,
    time_limit: str,
    log_dir: Path,
    nodes: int,
    ntasks_per_node: int | None = None,
    cpus_per_task: int,
    gres_gpu: int | None = None,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --output={log_dir}/%x_%j.log",
        f"#SBATCH --error={log_dir}/%x_%j.log",
    ]
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if ntasks_per_node is None:
        lines.append("#SBATCH --ntasks=1")
    else:
        lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    if gres_gpu is not None:
        lines.append(f"#SBATCH --gres=gpu:{gres_gpu}")
    return "\n".join(lines)


def render_index_sbatch(
    *,
    repo_root: str | Path,
    dataset_root: str | Path,
    index_root: str | Path,
    log_dir: str | Path,
    run_id: str,
    partition: str,
    qos: str | None,
    time_limit: str,
    cpus_per_task: int,
    records_per_shard: int,
    eos_check_records: int,
    min_eos_fraction: float,
    overwrite: bool,
) -> str:
    splits = "--split=train --split=val"
    overwrite_flag = " --overwrite" if overwrite else ""
    return f"""{
        _sbatch_header(
            job_name=f"sp_index_{run_id}",
            partition=partition,
            qos=qos,
            time_limit=time_limit,
            log_dir=Path(log_dir),
            nodes=1,
            cpus_per_task=cpus_per_task,
        )
    }

set -euo pipefail
cd {_quote(repo_root)}
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

cat "$0"
uv run python scripts/build_pretrain_indexes.py \\
  --root={_quote(dataset_root)} \\
  --out_dir={_quote(index_root)} \\
  {splits} \\
  --chunk_length={_flag_value("submit_seq_len")} \\
  --num_segments={_flag_value("submit_num_segments")} \\
  --eos_id=248046 \\
  --records_per_shard={records_per_shard} \\
  --eos_check_records={eos_check_records} \\
  --min_eos_fraction={min_eos_fraction}{overwrite_flag}
"""


def _train_flags(
    *,
    mode: PretrainMode,
    save_root: Path,
    jax_cache_root: Path,
    run_id: str,
    total_tasks: int,
    batch_size: int,
    grad_accum_steps: int,
    single_process_per_run: bool,
    wandb_resume_id: str | None = None,
) -> list[str]:
    fsdp_size = 1 if single_process_per_run else total_tasks
    dp_size = total_tasks if single_process_per_run else 1
    flags_out = [
        f"--pretrain_mode={mode.value}",
        "--train_index_path=${TRAIN_INDEX_ROOT}/train",
        "--val_index_path=${TRAIN_INDEX_ROOT}/val",
        f"--seq_len={_flag_value('submit_seq_len')}",
        f"--batch_size={batch_size}",
        f"--max_tokens={_flag_value('submit_max_tokens')}",
        f"--warmup_tokens={_flag_value('submit_warmup_tokens')}",
        f"--learning_rate={_flag_value('submit_learning_rate')}",
        f"--weight_decay={_flag_value('submit_weight_decay')}",
        f"--adam_beta1={_flag_value('submit_adam_beta1')}",
        f"--adam_beta2={_flag_value('submit_adam_beta2')}",
        f"--adam_eps={_flag_value('submit_adam_eps')}",
        "--lr_schedule=cosine",
        "--lr_end_factor=0.1",
        f"--max_grad_norm={_flag_value('submit_max_grad_norm')}",
        f"--grad_accum_steps={grad_accum_steps}",
        f"--seed={_flag_value('submit_seed')}",
        "--tp_size=1",
        f"--fsdp_size={fsdp_size}",
        f"--dp_size={dp_size}",
        f"--save_dir={save_root / run_id / mode.value}",
        f"--jax_cache_dir={jax_cache_root / run_id / mode.value}",
        f"--save_every={_flag_value('submit_save_every')}",
        f"--log_every={_flag_value('submit_log_every')}",
        f"--val_every={_flag_value('submit_val_every')}",
        f"--val_steps={_flag_value('submit_val_steps')}",
        "--peak_tflops=h100_sxm",
        "--resume=if_present",
        "--text_attn_backend=mosaic_gpu",
        f"--pass_gdn_state={_flag_value('submit_pass_gdn_state')}",
        f"--pass_rope_positions={_flag_value('submit_pass_rope_positions')}",
        f"--pass_conv_state={_flag_value('submit_pass_conv_state')}",
    ]
    bptt_chunks = _flag_value("submit_bptt_chunks")
    if bptt_chunks is not None:
        flags_out.append(f"--bptt_chunks={bptt_chunks}")
    gdn_layer_limit = _flag_value("submit_gdn_layer_limit")
    if gdn_layer_limit is not None:
        flags_out.append(f"--gdn_layer_limit={gdn_layer_limit}")
    if single_process_per_run:
        flags_out.extend(["--iterator_fsdp_size=1", "--iterator_dp_size=1"])
    wandb_project = _flag_value("submit_wandb_project")
    if wandb_project:
        group = _flag_value("submit_wandb_group") or run_id
        wandb_entity = _flag_value("submit_wandb_entity")
        if wandb_entity:
            flags_out.append(f"--wandb_entity={wandb_entity}")
        flags_out.extend(
            [
                f"--wandb_project={wandb_project}",
                f"--wandb_group={group}",
                f"--wandb_name={run_id}-{mode.value}",
                f"--wandb_tags={_flag_value('submit_wandb_tags')}",
            ]
        )
        if wandb_resume_id:
            flags_out.append(f"--wandb_id={wandb_resume_id}")
        wandb_resume = _flag_value("submit_wandb_resume")
        if wandb_resume:
            flags_out.append(f"--wandb_resume={wandb_resume}")
    return flags_out


def _train_flags_text(flags_in: list[str]) -> str:
    return " \\\n  ".join(flag.replace("=", '="', 1) + '"' for flag in flags_in)


def _wandb_env_block(log_dir: Path, mode: PretrainMode) -> str:
    if not _flag_value("submit_wandb_project"):
        return ""
    return f"""export WANDB_MODE=online
export WANDB_DIR={_quote(log_dir / "wandb" / mode.value)}
export WANDB_CACHE_DIR={_quote(log_dir / "wandb_cache")}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
"""


def _local_device_ids(num_devices: int) -> str:
    return ",".join(str(idx) for idx in range(num_devices))


def render_train_sbatch(
    *,
    repo_root: str | Path,
    source_root: str | Path,
    dataset_root: str | Path,
    index_root: str | Path,
    save_root: str | Path,
    log_dir: str | Path,
    jax_cache_root: str | Path,
    run_id: str,
    mode: PretrainMode,
    partition: str,
    qos: str | None,
    time_limit: str,
    nodes: int,
    gpus_per_node: int,
    batch_size: int,
    grad_accum_steps: int,
    cpus_per_task: int,
    stage_to_scratch: bool,
    run_pallas_tests: bool,
    single_process_per_run: bool,
    wandb_resume_id: str | None = None,
) -> str:
    source_root = Path(source_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    index_root = Path(index_root).expanduser().resolve()
    dataset_rel = _path_under(dataset_root, source_root)
    index_rel = _path_under(index_root, source_root)
    total_tasks = nodes * gpus_per_node
    launch_tasks = 1 if single_process_per_run else total_tasks
    launch_ntasks_per_node = 1 if single_process_per_run else gpus_per_node
    jax_local_device_ids = _local_device_ids(gpus_per_node) if single_process_per_run else "0"
    step_gpu_args = (
        f" --gres=gpu:{gpus_per_node}"
        if single_process_per_run
        else " --gpus-per-task=1 --gpu-bind=single:1"
    )
    flags_text = _train_flags_text(
        _train_flags(
            mode=mode,
            save_root=Path(save_root),
            jax_cache_root=Path(jax_cache_root),
            run_id=run_id,
            wandb_resume_id=wandb_resume_id,
            total_tasks=total_tasks,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            single_process_per_run=single_process_per_run,
        )
    )
    wandb_env_block = _wandb_env_block(Path(log_dir), mode)
    stage_block = ""
    if stage_to_scratch:
        stage_block = f"""
export OMEGALAX_PRETRAIN_SOURCE_ROOT={_quote(source_root)}
export OMEGALAX_PRETRAIN_LOCAL_ROOT="${{SLURM_TMPDIR:-/scratch/$USER/$SLURM_JOB_ID}}/salan"
export LOCAL_DATASET_ROOT="${{OMEGALAX_PRETRAIN_LOCAL_ROOT}}/{dataset_rel}"
export LOCAL_INDEX_ROOT="${{OMEGALAX_PRETRAIN_LOCAL_ROOT}}/{index_rel}"
srun --ntasks="${{SLURM_NNODES}}" --ntasks-per-node=1 bash -lc 'set -euo pipefail
mkdir -p "$(dirname "$LOCAL_DATASET_ROOT")" "$(dirname "$LOCAL_INDEX_ROOT")"
rsync -a "$DATASET_ROOT/" "$LOCAL_DATASET_ROOT/"
rsync -a "$INDEX_ROOT/" "$LOCAL_INDEX_ROOT/"
'
export TRAIN_INDEX_ROOT="${{LOCAL_INDEX_ROOT}}"
"""
    pallas_block = ""
    if run_pallas_tests:
        pallas_block = """
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 bash -lc 'set -euo pipefail
export JAX_PLATFORMS=cuda
export JAX_LOCAL_DEVICE_IDS=0
export OMEGALAX_DELTANET_KERNEL=pallas
export WANDB_MODE=disabled
uv run python -m pytest tests/test_gated_delta_rule_pallas.py tests/test_gated_delta_rule_pallas_bwd.py -q
'
"""
    return f"""{
        _sbatch_header(
            job_name=f"sp_{mode.value}_{run_id}",
            partition=partition,
            qos=qos,
            time_limit=time_limit,
            log_dir=Path(log_dir),
            nodes=nodes,
            ntasks_per_node=launch_ntasks_per_node,
            cpus_per_task=cpus_per_task,
            gres_gpu=gpus_per_node,
        )
    }

set -euo pipefail
cd {_quote(repo_root)}
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_LOCAL_DEVICE_IDS={jax_local_device_ids}
export OMEGALAX_DELTANET_KERNEL=pallas
export HF_HOME={_quote(source_root / "huggingface")}
export HF_HUB_OFFLINE="${{HF_HUB_OFFLINE:-1}}"
export LD_LIBRARY_PATH="${{CUDA_HOME:-/usr/local/cuda}}/extras/CUPTI/lib64:${{LD_LIBRARY_PATH:-}}"
export DATASET_ROOT={_quote(dataset_root)}
export INDEX_ROOT={_quote(index_root)}
export TRAIN_INDEX_ROOT="${{INDEX_ROOT}}"
{wandb_env_block}

cat "$0"
nvidia-smi
{stage_block}
{pallas_block}
srun --ntasks={launch_tasks} --ntasks-per-node={launch_ntasks_per_node}{
        step_gpu_args
    } bash -lc 'set -euo pipefail
uv run python scripts/train_text_pretrain.py \\
  {flags_text}
'
"""


def render_monitor_sbatch(
    *,
    repo_root: str | Path,
    log_dir: str | Path,
    run_id: str,
    partition: str,
    qos: str | None,
    time_limit: str,
    job_ids: list[str],
    wiki_path: str | Path,
    poll_seconds: int,
) -> str:
    return f"""{
        _sbatch_header(
            job_name=f"sp_monitor_{run_id}",
            partition=partition,
            qos=qos,
            time_limit=time_limit,
            log_dir=Path(log_dir),
            nodes=1,
            cpus_per_task=1,
        )
    }

set -euo pipefail
cd {_quote(repo_root)}
source .venv/bin/activate
export PYTHONUNBUFFERED=1

cat "$0"
uv run python scripts/monitor_text_pretrain_slurm.py \\
  --monitor_job_ids={_quote(",".join(job_ids))} \\
  --monitor_log_dir={_quote(log_dir)} \\
  --monitor_wiki_path={_quote(wiki_path)} \\
  --monitor_poll_seconds={poll_seconds}
"""


def _submit(script_path: Path, *, dependency: str | None, dry_run: bool) -> str | None:
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script_path))
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return None
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(proc.stdout.strip(), flush=True)
    match = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from: {proc.stdout!r}")
    return match.group(1)


def main(_) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_id = _run_id()
    log_dir = Path(FLAGS.submit_log_dir).expanduser().resolve() / run_id
    job_dir = log_dir / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    run_specs = parse_run_specs(
        FLAGS.submit_mode_shape,
        nodes_per_run=FLAGS.submit_nodes_per_run,
        default_gpus_per_node=FLAGS.submit_gpus_per_node,
        default_batch_size=FLAGS.submit_batch_size,
        default_grad_accum_steps=FLAGS.submit_grad_accum_steps,
        single_process_per_run=FLAGS.submit_single_process_per_run,
    )
    wandb_resume_ids = parse_wandb_resume_ids(FLAGS.submit_wandb_resume_id)

    index_root = Path(FLAGS.submit_index_root).expanduser().resolve()
    index_job_id = None
    if FLAGS.submit_build_indexes and not indexes_ready(index_root):
        if any_index_metadata_exists(index_root) and not FLAGS.submit_overwrite_indexes:
            raise ValueError(
                f"Partial indexes found under {index_root}; rerun with "
                "--submit_overwrite_indexes or choose a fresh --submit_index_root."
            )
        index_script = job_dir / "build_pretrain_indexes.sbatch"
        index_script.write_text(
            render_index_sbatch(
                repo_root=repo_root,
                dataset_root=FLAGS.submit_dataset_root,
                index_root=index_root,
                log_dir=log_dir,
                run_id=run_id,
                partition=FLAGS.submit_partition,
                qos=FLAGS.submit_qos,
                time_limit=FLAGS.submit_index_time,
                cpus_per_task=FLAGS.submit_index_cpus,
                records_per_shard=FLAGS.submit_records_per_shard,
                eos_check_records=FLAGS.submit_eos_check_records,
                min_eos_fraction=FLAGS.submit_min_eos_fraction,
                overwrite=FLAGS.submit_overwrite_indexes,
            )
        )
        index_job_id = _submit(index_script, dependency=None, dry_run=FLAGS.submit_dry_run)
    elif not indexes_ready(index_root):
        raise ValueError(f"Indexes are not ready and --submit_build_indexes=false: {index_root}")
    else:
        print(f"Indexes already present: {index_root}", flush=True)

    dependency = f"afterok:{index_job_id}" if index_job_id else None
    train_job_ids = []
    for spec in run_specs:
        train_script = job_dir / f"train_{spec.mode.value}.sbatch"
        train_script.write_text(
            render_train_sbatch(
                repo_root=repo_root,
                source_root=FLAGS.submit_source_root,
                dataset_root=FLAGS.submit_dataset_root,
                index_root=index_root,
                save_root=FLAGS.submit_save_root,
                log_dir=log_dir,
                jax_cache_root=FLAGS.submit_jax_cache_root,
                run_id=run_id,
                mode=spec.mode,
                partition=FLAGS.submit_partition,
                qos=FLAGS.submit_qos,
                time_limit=FLAGS.submit_time,
                nodes=spec.nodes,
                gpus_per_node=spec.gpus_per_node,
                batch_size=spec.batch_size,
                grad_accum_steps=spec.grad_accum_steps,
                cpus_per_task=FLAGS.submit_cpus_per_task,
                stage_to_scratch=FLAGS.submit_stage_to_scratch,
                run_pallas_tests=FLAGS.submit_run_pallas_tests,
                single_process_per_run=FLAGS.submit_single_process_per_run,
                wandb_resume_id=wandb_resume_ids.get(spec.mode),
            )
        )
        train_job_id = _submit(train_script, dependency=dependency, dry_run=FLAGS.submit_dry_run)
        if train_job_id is not None:
            train_job_ids.append(train_job_id)
            if FLAGS.submit_sequential_train_jobs:
                dependency = f"afterok:{train_job_id}"

    if FLAGS.submit_monitor and train_job_ids:
        monitor_script = job_dir / "monitor.sbatch"
        monitor_script.write_text(
            render_monitor_sbatch(
                repo_root=repo_root,
                log_dir=log_dir,
                run_id=run_id,
                partition=FLAGS.submit_partition,
                qos=FLAGS.submit_qos,
                time_limit=FLAGS.submit_monitor_time,
                job_ids=[job_id for job_id in ([index_job_id] if index_job_id else [])]
                + train_job_ids,
                wiki_path=FLAGS.submit_wiki_path,
                poll_seconds=FLAGS.submit_monitor_poll_seconds,
            )
        )
        _submit(monitor_script, dependency=None, dry_run=FLAGS.submit_dry_run)

    print(f"run_id={run_id}", flush=True)
    print(f"slurm_logs={log_dir}", flush=True)
    print(f"save_root={Path(FLAGS.submit_save_root).expanduser().resolve() / run_id}", flush=True)


if __name__ == "__main__":
    app.run(main)
