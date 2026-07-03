"""Launch the three comparable text pretraining experiments."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from absl import app, flags

from omegalax.trainers.pretrain import PretrainMode

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "index_root", None, "Root produced by build_pretrain_indexes.py.", required=True
)
flags.DEFINE_string("save_root", None, "Root directory for per-mode checkpoints.", required=True)
flags.DEFINE_multi_enum(
    "experiment_mode",
    [mode.value for mode in PretrainMode],
    [mode.value for mode in PretrainMode],
    "Experiment modes to launch.",
)
flags.DEFINE_string("wandb_name_prefix", None, "Optional W&B run-name prefix.")
flags.DEFINE_integer("tp_size", 1, "Tensor parallelism size forwarded to train jobs.")
flags.DEFINE_integer("fsdp_size", 1, "FSDP size forwarded to train jobs.")
flags.DEFINE_integer("dp_size", 1, "Data parallelism size forwarded to train jobs.")
flags.DEFINE_multi_string(
    "extra_arg",
    [],
    "Extra argument forwarded to every train_text_pretrain.py invocation. "
    "Repeat for each flag, e.g. --extra_arg=--batch_size=8.",
)
flags.DEFINE_bool("parallel", True, "Run jobs in parallel. If false, run sequentially.")
flags.DEFINE_bool("dry_run", False, "Print commands without launching them.")


def build_commands(
    *,
    index_root: str | Path,
    save_root: str | Path,
    modes: list[str],
    wandb_name_prefix: str | None = None,
    tp_size: int | None = 1,
    fsdp_size: int | None = 1,
    dp_size: int | None = 1,
    extra_args: list[str] | None = None,
) -> list[list[str]]:
    index_root = Path(index_root).expanduser().resolve()
    save_root = Path(save_root).expanduser().resolve()
    extra_args = list(extra_args or [])
    script = Path(__file__).with_name("train_text_pretrain.py")
    commands = []
    for raw_mode in modes:
        mode = PretrainMode(raw_mode)
        cmd = [
            sys.executable,
            str(script),
            f"--pretrain_mode={mode.value}",
            f"--train_index_path={index_root / 'train'}",
            f"--val_index_path={index_root / 'val'}",
            f"--save_dir={save_root / mode.value}",
        ]
        if tp_size is not None:
            cmd.append(f"--tp_size={tp_size}")
        if fsdp_size is not None:
            cmd.append(f"--fsdp_size={fsdp_size}")
        if dp_size is not None:
            cmd.append(f"--dp_size={dp_size}")
        if wandb_name_prefix:
            cmd.append(f"--wandb_name={wandb_name_prefix}-{mode.value}")
        cmd.extend(extra_args)
        commands.append(cmd)
    return commands


def main(_) -> None:
    commands = build_commands(
        index_root=FLAGS.index_root,
        save_root=FLAGS.save_root,
        modes=FLAGS.experiment_mode,
        wandb_name_prefix=FLAGS.wandb_name_prefix,
        tp_size=FLAGS.tp_size,
        fsdp_size=FLAGS.fsdp_size,
        dp_size=FLAGS.dp_size,
        extra_args=FLAGS.extra_arg,
    )
    for cmd in commands:
        print(" ".join(cmd), flush=True)
    if FLAGS.dry_run:
        return

    if FLAGS.parallel:
        procs = [subprocess.Popen(cmd) for cmd in commands]
        codes = [proc.wait() for proc in procs]
    else:
        codes = [subprocess.run(cmd, check=False).returncode for cmd in commands]
    failures = [code for code in codes if code != 0]
    if failures:
        raise SystemExit(max(failures))


if __name__ == "__main__":
    app.run(main)
