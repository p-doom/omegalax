"""Migrate one legacy MultiSteps(k=1) checkpoint into the direct optimizer schema."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omegalax.trainers.checkpoint_migration import migrate_multisteps_k1_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--destination_root", required=True)
    parser.add_argument("--checkpoint_step", required=True, type=int)
    args = parser.parse_args()
    migrated_step = migrate_multisteps_k1_checkpoint(
        args.source_root,
        args.destination_root,
        args.checkpoint_step,
    )
    print(f"Migrated checkpoint written to {migrated_step}")


if __name__ == "__main__":
    main()
