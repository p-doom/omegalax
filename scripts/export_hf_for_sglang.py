"""Export an OmegaLAX Qwen3.5-4B text checkpoint as a complete HF snapshot for SGLang.

Language-model tensors come from the OmegaLAX checkpoint (params-only restore, fp32 master
weights) through OmegaLAX's own HF key mapping, an untied lm_head.weight (Qwen3.5-9B) as well;
every other file and tensor (vision tower, MTP head, config, tokenizer, chat template) is copied
unchanged from the base HF snapshot. Each
exported tensor is cast to the dtype the base snapshot stores for the same key, and key set and
shapes must match the base snapshot exactly.

  --checkpoint <root>/<step>   export trained weights
  --checkpoint base            export the base snapshot itself (mapping round-trip check)
"""

import os
import socket
import sys

assert os.environ.get("SLURM_JOB_ID"), "run inside a Slurm allocation"
assert os.environ.get("SLURMD_NODENAME", "").split(".")[0] == socket.gethostname().split(".")[0]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

SOURCE = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/salan/runs/rollouts_all_20261004/analysis/"
    "distillation_design_20261023/comparison_20261025/tasks/jobs/production_fast/source"
)
BASE = Path(
    "/fast/project/HFMI_SynergyUnit/p-doom_shared/huggingface/hub/models--Qwen--Qwen3.5-4B/"
    "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
sys.path.insert(0, str(SOURCE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flax import nnx  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import omegalax  # noqa: E402
from omegalax.models.params_utils import (  # noqa: E402
    build_inverse_mapping,
    flatten_pure_state,
    inverse_transform,
)
from omegalax.models.qwen3_5.loader import _get_non_expert_mapping, create_qwen3_5_from_safetensors  # noqa: E402
from omegalax.text import api as text_api  # noqa: E402
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params  # noqa: E402

assert Path(omegalax.__file__).resolve().parents[1] == SOURCE, omegalax.__file__
LM = "model.language_model."
# Same special cases as omegalax.models.qwen3_5.params.export_qwen3_5_to_safetensors (dense text).
SPECIAL = (
    (
        re.compile(r"text\.layers\.([0-9]+)\.linear_attn\.norm\.weight"),
        LM + "layers.{}.linear_attn.norm.weight",
        lambda a: a,
    ),
    (
        re.compile(r"text\.layers\.([0-9]+)\.linear_attn\.conv_weight"),
        LM + "layers.{}.linear_attn.conv1d.weight",
        lambda a: a[:, None, :],
    ),
    (
        re.compile(r"text\.layers\.([0-9]+)\.linear_attn\.dt_bias"),
        LM + "layers.{}.linear_attn.dt_bias",
        lambda a: a,
    ),
    (
        re.compile(r"text\.layers\.([0-9]+)\.linear_attn\.A_log"),
        LM + "layers.{}.linear_attn.A_log",
        lambda a: a,
    ),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_text_model(checkpoint):
    if checkpoint == "base":
        model, _ = create_qwen3_5_from_safetensors(
            str(BASE), tp_size=1, fsdp_size=1, dp_size=1, text_only=True
        )
        return model, {"source": str(BASE)}
    resolved = resolve_checkpoint(checkpoint)
    cfg = text_api.resolve_config(str(resolved.config_path))
    model, cfg = text_api.init_model(cfg, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1)
    model = restore_model_params(model, resolved)
    return model, {
        "source": str(resolved.step_path),
        "step": resolved.step,
        "config_sha256": sha256(resolved.config_path),
        "checkpoint_metadata_sha256": sha256(resolved.step_path / "_CHECKPOINT_METADATA"),
        "param_dtypes": sorted(
            {str(x.dtype) for x in jax.tree.leaves(nnx.state(model, nnx.Param))}
        ),
    }


def to_hf_tensors(model):
    flat = flatten_pure_state(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    inverse = build_inverse_mapping(_get_non_expert_mapping())
    out, unmatched = {}, []
    for key, value in flat.items():
        arr = np.asarray(jax.device_get(value))
        for regex, template, transform in SPECIAL:
            if m := regex.fullmatch(key):
                out[template.format(*m.groups())] = transform(arr)
                break
        else:
            for regex, template, transform in inverse:
                if m := re.fullmatch(regex, key):
                    out[template.format(*m.groups())] = inverse_transform(
                        arr, transform.value if hasattr(transform, "value") else transform
                    )
                    break
            else:
                unmatched.append(key)
    assert not unmatched, f"unmapped parameters: {unmatched}"
    return out


def to_torch(arr, dtype):
    if arr.dtype == jnp.bfloat16:
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), f"refusing to overwrite {args.out}"

    model, provenance = load_text_model(args.checkpoint)
    exported = to_hf_tensors(model)
    del model

    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    base_lm, rest = {}, {}
    for key, shard in index["weight_map"].items():
        with safe_open(str(BASE / shard), framework="pt") as f:
            # An untied base (Qwen3.5-9B) stores the trained LM head as lm_head.weight; a tied one
            # (Qwen3.5-4B) has neither the key nor an lm_head parameter in the OmegaLAX model.
            (base_lm if key.startswith(LM) or key == "lm_head.weight" else rest)[key] = (
                f.get_tensor(key)
            )
    assert set(exported) == set(base_lm), set(exported) ^ set(base_lm)

    lm, diffs = {}, {}
    for key, base_tensor in base_lm.items():
        tensor = to_torch(exported[key], base_tensor.dtype)
        assert tuple(tensor.shape) == tuple(base_tensor.shape), (
            key,
            tensor.shape,
            base_tensor.shape,
        )
        lm[key] = tensor
        diffs[key] = float((tensor.float() - base_tensor.float()).abs().max())

    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.mkdir(parents=True)
    save_file(lm, str(tmp / "model-language.safetensors"), metadata={"format": "pt"})
    save_file(rest, str(tmp / "model-other.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model-language.safetensors" for k in lm} | {
        k: "model-other.safetensors" for k in rest
    }
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": index.get("metadata", {}), "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
        )
    )
    for path in BASE.iterdir():
        if not path.name.startswith("model.safetensors") and not path.name.startswith("model-"):
            shutil.copy2(path, tmp / path.name)
    changed = sorted(k for k, d in diffs.items() if d > 0)
    summary = {
        "provenance": provenance,
        "base_snapshot": str(BASE),
        "omegalax_source": str(SOURCE),
        "num_language_tensors": len(lm),
        "num_copied_tensors": len(rest),
        "language_tensors_changed_vs_base": len(changed),
        "max_abs_diff_vs_base": max(diffs.values()),
        "dtypes": sorted({str(t.dtype) for t in lm.values()}),
    }
    (tmp / "export_provenance.json").write_text(json.dumps(summary, indent=2))
    tmp.rename(args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
