"""Execute the two-device state-usage evaluation regression in a fresh process."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> None:
    import jax

    assert jax.device_count() == 2, jax.devices()

    from omegalax.evals.runner import run_evals
    from tests.test_eval_runner import _build_manifest, _save_checkpoint, _spec

    root = Path(sys.argv[1])
    checkpoint_root = root / "checkpoint"
    _save_checkpoint(checkpoint_root, step=1, dp_size=2)
    manifest_path, _ = _build_manifest(root / "dataset", documents_per_length=2)

    run_evals(
        (_spec(checkpoint_root, c_train=2),),
        mode="subset",
        manifest_path=manifest_path,
        experiments=("gdn",),
        plot_types=("heatmap",),
        document_cap=2,
        tp_size=1,
        fsdp_size=1,
        dp_size=2,
        batch_size=2,
    )

    run_dir = checkpoint_root / "evals" / "state_usage_v2" / "step_1"
    assert json.loads((run_dir / "status.json").read_text())["complete"] is True
    for condition in ("true_gdn", "zero_gdn", "shuffled_gdn"):
        assert tuple((run_dir / "raw" / "all" / "gdn").glob(f"{condition}_shard_*.parquet"))
    assert (run_dir / "summary" / "all" / "metrics.parquet").is_file()
    (root / "passed").write_text("ok\n")


if __name__ == "__main__":
    main()
