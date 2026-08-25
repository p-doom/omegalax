"""Shared checkpoint helpers for train-state and Grain iterators."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import stat
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

import grain
import orbax.checkpoint as ocp

# NOTE: plain `TypeAlias` (not the PEP 695 `type X = ...` statement) so the module
# imports on Python 3.11, matching `requires-python = ">=3.11"`.
GrainIterator: TypeAlias = grain.DataLoaderIterator | grain.DatasetIterator

CHECKPOINT_MANIFEST_FILENAME = "checkpoint.manifest.json"
_CHECKPOINT_MANIFEST_SCHEMA = "omegalax.sft-checkpoint"
_CHECKPOINT_MANIFEST_VERSION = 1
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STEP_RE = re.compile(r"[0-9]{6,}\Z")


@dataclasses.dataclass(frozen=True)
class CheckpointIdentities:
    model_sha256: str
    dataset_sha256: str
    source_sha256: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _validate_sha256(getattr(self, field.name), field.name)


@dataclasses.dataclass(frozen=True)
class ValidationReceipt:
    step: int
    batches: int
    loss_sum_hex: str
    supervised_tokens: int
    dataset_sha256: str

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("Validation receipt step must be a positive integer.")
        if type(self.batches) is not int or self.batches <= 0:
            raise ValueError("Validation receipt batches must be a positive integer.")
        if type(self.supervised_tokens) is not int or self.supervised_tokens <= 0:
            raise ValueError("Validation receipt supervised_tokens must be a positive integer.")
        if type(self.loss_sum_hex) is not str:
            raise TypeError("Validation receipt loss_sum_hex must be a string.")
        try:
            loss_sum = float.fromhex(self.loss_sum_hex)
        except ValueError as error:
            raise ValueError(
                "Validation receipt loss_sum_hex is not a hexadecimal float."
            ) from error
        if not math.isfinite(loss_sum) or loss_sum < 0.0 or loss_sum.hex() != self.loss_sum_hex:
            raise ValueError(
                "Validation receipt loss_sum_hex must be the canonical hexadecimal form of a "
                "finite non-negative float."
            )
        _validate_sha256(self.dataset_sha256, "validation.dataset_sha256")


@dataclasses.dataclass(frozen=True)
class VerifiedCheckpoint:
    path: Path
    manifest: dict[str, object]
    sha256: str
    step: int
    identities: CheckpointIdentities


def _validate_sha256(value: object, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint manifest key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid checkpoint manifest JSON constant {value!r}.")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_regular_file(parent_fd: int, name: str, *, max_bytes: int | None = None) -> bytes:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(
            f"Checkpoint file must be a readable no-follow regular file: {name}."
        ) from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint entry is not a regular file: {name}.")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError(f"Checkpoint file exceeds {max_bytes} bytes: {name}.")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Checkpoint file changed while reading: {name}.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(f"Checkpoint file changed while reading: {name}.")
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Checkpoint file changed while reading: {name}.")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _hash_regular_file(
    parent_fd: int,
    name: str,
    *,
    make_durable: bool,
) -> tuple[os.stat_result, str]:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(
            f"Checkpoint file must be a readable no-follow regular file: {name}."
        ) from error
    try:
        if make_durable:
            os.fchmod(fd, 0o440)
            os.fsync(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Checkpoint entry is not a regular file: {name}.")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError(f"Checkpoint file changed while hashing: {name}.")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError(f"Checkpoint file changed while hashing: {name}.")
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Checkpoint file changed while hashing: {name}.")
        return before, digest.hexdigest()
    finally:
        os.close(fd)


def _inventory_directory(
    directory_fd: int,
    relative: tuple[str, ...] = (),
    *,
    make_durable: bool,
) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    with os.scandir(directory_fd) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        if not relative and name == CHECKPOINT_MANIFEST_FILENAME:
            continue
        path = "/".join((*relative, name))
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISREG(before.st_mode):
            file_stat, digest = _hash_regular_file(
                directory_fd,
                name,
                make_durable=make_durable,
            )
            after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_mode,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(f"Checkpoint file changed while inventorying: {path}.")
            files.append(
                {
                    "path": path,
                    "size": file_stat.st_size,
                    "sha256": digest,
                }
            )
            continue
        if stat.S_ISDIR(before.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                child_stat = os.fstat(child_fd)
                if (before.st_dev, before.st_ino) != (child_stat.st_dev, child_stat.st_ino):
                    raise ValueError(f"Checkpoint directory changed while opening: {path}.")
                files.extend(
                    _inventory_directory(
                        child_fd,
                        (*relative, name),
                        make_durable=make_durable,
                    )
                )
                if make_durable:
                    os.fchmod(child_fd, 0o550)
                    os.fsync(child_fd)
            finally:
                os.close(child_fd)
            continue
        raise ValueError(f"Checkpoint contains a non-regular entry: {path}.")
    return files


def _manifest_step(path: Path) -> int:
    if _STEP_RE.fullmatch(path.name) is None:
        raise ValueError(
            f"Checkpoint path must name one zero-padded numeric step directory, got {path}."
        )
    step = int(path.name)
    if step <= 0 or str(step).zfill(6) != path.name:
        raise ValueError(f"Checkpoint step directory has a non-canonical name: {path.name!r}.")
    return step


def _parse_manifest(raw: bytes) -> dict[str, object]:
    try:
        manifest = json.loads(
            raw,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint manifest is not canonical JSON.") from error
    if type(manifest) is not dict or raw != _canonical_json(manifest):
        raise ValueError("Checkpoint manifest is not canonical JSON.")
    return manifest


def _validate_manifest(manifest: dict[str, object], step: int) -> CheckpointIdentities:
    if set(manifest) != {
        "schema",
        "version",
        "step",
        "phase",
        "identities",
        "state",
        "validation",
        "files",
    }:
        raise ValueError("Checkpoint manifest has an incompatible top-level schema.")
    if manifest["schema"] != _CHECKPOINT_MANIFEST_SCHEMA:
        raise ValueError("Checkpoint manifest schema is incompatible.")
    if type(manifest["version"]) is not int or manifest["version"] != _CHECKPOINT_MANIFEST_VERSION:
        raise ValueError("Checkpoint manifest version is incompatible.")
    if type(manifest["step"]) is not int or manifest["step"] != step:
        raise ValueError("Checkpoint manifest step does not match its directory.")

    phase = manifest["phase"]
    if type(phase) is not dict or set(phase) != {"schedule_horizon", "invocation_end_step"}:
        raise ValueError("Checkpoint manifest phase schema is invalid.")
    horizon = phase["schedule_horizon"]
    end = phase["invocation_end_step"]
    if type(horizon) is not int or type(end) is not int or not 0 < step <= end <= horizon:
        raise ValueError("Checkpoint manifest phase bounds are invalid.")

    identity_values = manifest["identities"]
    identity_names = {field.name for field in dataclasses.fields(CheckpointIdentities)}
    if type(identity_values) is not dict or set(identity_values) != identity_names:
        raise ValueError("Checkpoint manifest identities schema is invalid.")
    identities = CheckpointIdentities(**identity_values)

    state = manifest["state"]
    if type(state) is not dict or set(state) != {
        "optimizer_generation",
        "input_logical_shards",
        "rng_sha256",
        "optimizer_fatal_status",
    }:
        raise ValueError("Checkpoint manifest state schema is invalid.")
    if type(state["optimizer_generation"]) is not int or state["optimizer_generation"] != step:
        raise ValueError("Checkpoint manifest optimizer generation does not match its step.")
    if type(state["input_logical_shards"]) is not int or state["input_logical_shards"] != 8:
        raise ValueError("Checkpoint manifest must bind exactly eight logical input shards.")
    _validate_sha256(state["rng_sha256"], "state.rng_sha256")
    if state["optimizer_fatal_status"] != "healthy":
        raise ValueError("Checkpoint manifest optimizer status must be healthy.")

    validation = manifest["validation"]
    if type(validation) is not dict:
        raise ValueError("Checkpoint manifest validation schema is invalid.")
    receipt = ValidationReceipt(**validation)
    if receipt.step != step:
        raise ValueError("Checkpoint validation receipt step does not match its checkpoint.")

    files = manifest["files"]
    if type(files) is not list or not files:
        raise ValueError("Checkpoint manifest files must be a non-empty list.")
    prior_path = None
    for item in files:
        if type(item) is not dict or set(item) != {"path", "size", "sha256"}:
            raise ValueError("Checkpoint manifest file entry schema is invalid.")
        relative_path = item["path"]
        if (
            type(relative_path) is not str
            or not relative_path
            or relative_path.startswith("/")
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
            or relative_path == CHECKPOINT_MANIFEST_FILENAME
            or (prior_path is not None and relative_path <= prior_path)
        ):
            raise ValueError("Checkpoint manifest file paths must be unique canonical POSIX paths.")
        prior_path = relative_path
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError(f"Checkpoint manifest file size is invalid: {relative_path}.")
        _validate_sha256(item["sha256"], f"files[{relative_path}].sha256")
    return identities


def verify_checkpoint(path: str | Path) -> VerifiedCheckpoint:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        raise ValueError("Checkpoint path must be absolute.")
    step = _manifest_step(checkpoint_path)
    parent_fd = os.open(
        checkpoint_path.parent,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        directory_fd = os.open(
            checkpoint_path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        manifest_raw = _read_regular_file(
            directory_fd,
            CHECKPOINT_MANIFEST_FILENAME,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        manifest = _parse_manifest(manifest_raw)
        identities = _validate_manifest(manifest, step)
        files = _inventory_directory(directory_fd, make_durable=False)
        if files != manifest["files"]:
            raise ValueError("Checkpoint payload does not match its exhaustive file inventory.")
        manifest_again = _read_regular_file(
            directory_fd,
            CHECKPOINT_MANIFEST_FILENAME,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        if manifest_again != manifest_raw:
            raise ValueError("Checkpoint manifest changed while verifying.")
        return VerifiedCheckpoint(
            path=checkpoint_path,
            manifest=manifest,
            sha256=hashlib.sha256(manifest_raw).hexdigest(),
            step=step,
            identities=identities,
        )
    finally:
        os.close(directory_fd)
        os.close(parent_fd)


def publish_checkpoint(
    staging_step_path: str | Path,
    final_step_path: str | Path,
    *,
    phase: dict[str, int],
    identities: CheckpointIdentities,
    rng_sha256: str,
    validation: ValidationReceipt,
) -> VerifiedCheckpoint:
    staging_path = Path(staging_step_path)
    final_path = Path(final_step_path)
    if not staging_path.is_absolute() or not final_path.is_absolute():
        raise ValueError("Checkpoint publication paths must be absolute.")
    step = _manifest_step(final_path)
    if staging_path.parent.parent != final_path.parent or not staging_path.parent.name.startswith(
        f".pending-{final_path.name}-"
    ):
        raise ValueError("Checkpoint staging directory is not the expected hidden sibling.")
    if validation.step != step:
        raise ValueError("Checkpoint validation receipt step does not match the publication step.")
    _validate_sha256(rng_sha256, "state.rng_sha256")

    root_fd = os.open(
        final_path.parent,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    staging_root_fd = os.open(
        staging_path.parent,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        try:
            os.stat(final_path.name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Checkpoint step already exists: {final_path}.")
        staging_fd = os.open(
            staging_path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=staging_root_fd,
        )
        try:
            try:
                os.stat(CHECKPOINT_MANIFEST_FILENAME, dir_fd=staging_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("Checkpoint staging directory already has a manifest.")
            files = _inventory_directory(staging_fd, make_durable=True)
            manifest = {
                "schema": _CHECKPOINT_MANIFEST_SCHEMA,
                "version": _CHECKPOINT_MANIFEST_VERSION,
                "step": step,
                "phase": dict(phase),
                "identities": dataclasses.asdict(identities),
                "state": {
                    "optimizer_generation": step,
                    "input_logical_shards": 8,
                    "rng_sha256": rng_sha256,
                    "optimizer_fatal_status": "healthy",
                },
                "validation": dataclasses.asdict(validation),
                "files": files,
            }
            _validate_manifest(manifest, step)
            manifest_raw = _canonical_json(manifest)
            temporary_name = f".{CHECKPOINT_MANIFEST_FILENAME}.{os.getpid()}.tmp"
            manifest_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o440,
                dir_fd=staging_fd,
            )
            try:
                view = memoryview(manifest_raw)
                while view:
                    written = os.write(manifest_fd, view)
                    if written <= 0:
                        raise OSError("Checkpoint manifest write made no progress.")
                    view = view[written:]
                os.fsync(manifest_fd)
            finally:
                os.close(manifest_fd)
            os.rename(
                temporary_name,
                CHECKPOINT_MANIFEST_FILENAME,
                src_dir_fd=staging_fd,
                dst_dir_fd=staging_fd,
            )
            os.fchmod(staging_fd, 0o750)
            os.fsync(staging_fd)
            os.rename(
                staging_path.name,
                final_path.name,
                src_dir_fd=staging_root_fd,
                dst_dir_fd=root_fd,
            )
            os.fchmod(staging_fd, 0o550)
            os.fsync(staging_fd)
            os.fsync(root_fd)
        finally:
            os.close(staging_fd)
    finally:
        os.close(staging_root_fd)
        os.close(root_fd)
    return verify_checkpoint(final_path)


class ResumeMode(StrEnum):
    """Trainer resume policy.

    NEVER      — always start fresh; do not consult any existing checkpoints.
    IF_PRESENT — resume from the latest checkpoint at ``save_dir`` if one exists,
                 otherwise start fresh. Right mode for SLURM time-limit recovery,
                 where the same recipe may be submitted with no checkpoint yet
                 (first run) or with checkpoints from a previous timed-out attempt.
    REQUIRED   — resume; error if no usable checkpoint is found. Right mode for
                 explicit "this must be a continuation" workflows.
    """

    NEVER = "never"
    IF_PRESENT = "if_present"
    REQUIRED = "required"


def register_grain_iterator_handler(
    handler_registry: ocp.handlers.DefaultCheckpointHandlerRegistry,
) -> None:
    handler_registry.add(
        "input_iter",
        grain.checkpoint.CheckpointSave,
        cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
    )
    handler_registry.add(
        "input_iter",
        grain.checkpoint.CheckpointRestore,
        cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
    )


def make_grain_save_args(train_state: Any, input_iter: GrainIterator) -> ocp.args.Composite:
    items: dict[str, Any] = {
        "train_state": ocp.args.PyTreeSave(train_state),
        "input_iter": grain.checkpoint.CheckpointSave(input_iter),
    }
    return ocp.args.Composite(**items)


def make_grain_restore_args(
    abstract_train_state: Any, input_iter: GrainIterator
) -> ocp.args.Composite:
    items: dict[str, Any] = {
        "train_state": ocp.args.PyTreeRestore(abstract_train_state),
        "input_iter": grain.checkpoint.CheckpointRestore(input_iter),
    }
    return ocp.args.Composite(**items)


def restored_input_iter(restored: Any) -> GrainIterator:
    return restored["input_iter"]
